import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa

from hirag_prod._llm import ChatCompletion, EmbeddingService
from hirag_prod._utils import _limited_gather  # Concurrency Rate Limiting Tool
from hirag_prod.chunk import BaseChunk, FixTokenChunk
from hirag_prod.entity import BaseEntity, VanillaEntity
from hirag_prod.loader import load_document
from hirag_prod.storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
    NetworkXGDB,
    RetrievalStrategyProvider,
)

# Log Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("HiRAG").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("HiRAG")


@dataclass
class DocumentProcessingState:
    """Track processing state for a document"""

    document_id: str
    uri: str
    total_chunks: int
    processed_chunk_ids: Set[str] = field(default_factory=set)
    extracted_entity_chunk_ids: Set[str] = field(default_factory=set)
    processed_entity_ids: Set[str] = field(default_factory=set)
    processed_relation_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    @property
    def chunks_complete(self) -> bool:
        return len(self.processed_chunk_ids) == self.total_chunks

    @property
    def entities_extracted(self) -> bool:
        return len(self.extracted_entity_chunk_ids) == self.total_chunks

    def get_pending_chunks(self, all_chunks) -> List:
        """Get chunks that haven't been processed yet"""
        return [
            chunk for chunk in all_chunks if chunk.id not in self.processed_chunk_ids
        ]

    def get_pending_entity_chunks(self, all_chunks) -> List:
        """Get chunks that haven't had entity extraction yet"""
        return [
            chunk
            for chunk in all_chunks
            if chunk.id not in self.extracted_entity_chunk_ids
        ]


class ResumeManager:
    """Manages resume functionality for document processing"""

    def __init__(self, chunks_table, entities_table, gdb):
        self.chunks_table = chunks_table
        self.entities_table = entities_table
        self.gdb = gdb

    async def get_document_state_by_uri(
        self, uri: str, chunks: List
    ) -> DocumentProcessingState:
        """Analyze current processing state for a document based on URI"""
        if not chunks:
            raise ValueError("No chunks provided")

        # Get document info from first chunk (for fallback)
        first_chunk = chunks[0]
        total_chunks = len(chunks)

        # Create chunk ID mapping for matching
        new_chunk_ids = {chunk.id for chunk in chunks}

        logger.info(f"🔍 Checking resume state for document: {uri}")

        # Try to find existing chunks by URI first
        existing_chunks_by_uri = []
        existing_document_id = None

        try:
            existing_chunks_by_uri = (
                await self.chunks_table.query().where(f"uri == '{uri}'").to_list()
            )

            if existing_chunks_by_uri:
                # Get the document_id from existing data
                existing_document_id = existing_chunks_by_uri[0]["document_id"]
                logger.info(
                    f"📋 Found existing document with ID: {existing_document_id}"
                )
                logger.info(
                    f"📋 Found {len(existing_chunks_by_uri)} existing chunks by URI"
                )
            else:
                logger.info(f"📋 No existing chunks found for URI: {uri}")

        except Exception as e:
            logger.warning(f"Error querying chunks by URI: {e}")

        # Use existing document_id if found, otherwise use new document_id
        document_id = existing_document_id or first_chunk.metadata.document_id

        state = DocumentProcessingState(
            document_id=document_id, uri=uri, total_chunks=total_chunks
        )

        # Check which chunks are already processed by matching chunk IDs
        if existing_chunks_by_uri:
            existing_chunk_ids = {
                chunk["document_key"] for chunk in existing_chunks_by_uri
            }
            # Find intersection between new chunks and existing chunks
            state.processed_chunk_ids = new_chunk_ids.intersection(existing_chunk_ids)
            logger.info(
                f"📊 Matched {len(state.processed_chunk_ids)}/{total_chunks} chunks by ID"
            )

            # If no ID matches but we have chunks with same URI, there might be content changes
            if len(state.processed_chunk_ids) == 0 and len(existing_chunks_by_uri) > 0:
                logger.warning(
                    f"⚠️ Found {len(existing_chunks_by_uri)} chunks with same URI but different IDs"
                )
                logger.warning("⚠️ This might indicate document content has changed")
        else:
            logger.info("📊 No existing chunks found, full processing needed")

        # Check extracted entities
        try:
            if existing_document_id:
                existing_entities = (
                    await self.entities_table.query()
                    .where(f"'{existing_document_id}' in chunk_ids")
                    .to_list()
                )

                # Track which chunks have had entities extracted
                for entity in existing_entities:
                    chunk_ids = entity.get("chunk_ids", [])
                    if isinstance(chunk_ids, list):
                        # Only count chunks that are in our current chunk set
                        matching_chunks = new_chunk_ids.intersection(set(chunk_ids))
                        state.extracted_entity_chunk_ids.update(matching_chunks)
                    state.processed_entity_ids.add(entity["document_key"])

                logger.info(
                    f"🏷️ Found {len(state.processed_entity_ids)} entities, "
                    f"covering {len(state.extracted_entity_chunk_ids)}/{total_chunks} chunks"
                )
            else:
                logger.info("🏷️ No existing entities found")
        except Exception as e:
            logger.warning(f"Error checking existing entities: {e}")

        # Check relations in graph database
        try:
            if hasattr(self.gdb, "get_document_relations") and existing_document_id:
                existing_relations = await self.gdb.get_document_relations(
                    existing_document_id
                )
                state.processed_relation_pairs = set(existing_relations)
                logger.info(
                    f"🔗 Found {len(state.processed_relation_pairs)} relations already processed"
                )
            else:
                logger.info("🔗 No existing relations found")
        except Exception as e:
            logger.warning(f"Error checking existing relations: {e}")

        return state

    async def get_document_state(self, chunks: List) -> DocumentProcessingState:
        """Legacy method for backward compatibility - redirects to URI-based method"""
        if not chunks:
            raise ValueError("No chunks provided")

        uri = chunks[0].metadata.uri
        logger.info(
            "🔄 Using legacy get_document_state - redirecting to URI-based method"
        )
        return await self.get_document_state_by_uri(uri, chunks)

    def log_resume_plan(self, state: DocumentProcessingState, chunks: List):
        """Log what will be processed in this run"""
        pending_chunks = len(state.get_pending_chunks(chunks))
        pending_entity_chunks = len(state.get_pending_entity_chunks(chunks))

        logger.info("📋 RESUME PLAN:")
        logger.info(f"  📄 Document: {state.uri}")
        logger.info(f"  🆔 Document ID: {state.document_id}")
        logger.info(
            f"  ✅ Chunks: {len(state.processed_chunk_ids)}/{state.total_chunks} complete"
        )
        logger.info(
            f"  📝 Entity extraction: {len(state.extracted_entity_chunk_ids)}/{state.total_chunks} complete"
        )
        logger.info(f"  🔗 Entities: {len(state.processed_entity_ids)} processed")
        logger.info(f"  🌐 Relations: {len(state.processed_relation_pairs)} processed")

        if pending_chunks == 0 and pending_entity_chunks == 0:
            logger.info("  🎉 Document processing appears complete!")
        else:
            logger.info(
                f"  ⏳ Will process {pending_chunks} chunks and extract entities from {pending_entity_chunks} chunks"
            )

    async def check_document_changes(self, uri: str, chunks: List) -> Dict[str, Any]:
        """Check if document content has changed compared to stored version"""
        try:
            existing_chunks = (
                await self.chunks_table.query().where(f"uri == '{uri}'").to_list()
            )

            if not existing_chunks:
                return {"changed": False, "reason": "no_existing_data"}

            new_chunk_ids = {chunk.id for chunk in chunks}
            existing_chunk_ids = {chunk["document_key"] for chunk in existing_chunks}

            if new_chunk_ids == existing_chunk_ids:
                return {"changed": False, "reason": "identical_chunks"}
            else:
                return {
                    "changed": True,
                    "reason": "different_chunks",
                    "new_chunks": len(new_chunk_ids),
                    "existing_chunks": len(existing_chunk_ids),
                    "common_chunks": len(
                        new_chunk_ids.intersection(existing_chunk_ids)
                    ),
                }
        except Exception as e:
            logger.warning(f"Error checking document changes: {e}")
            return {"changed": True, "reason": "check_failed", "error": str(e)}


@dataclass
class HiRAG:
    # LLM
    chat_service: ChatCompletion = field(default_factory=ChatCompletion)
    embedding_service: EmbeddingService = field(default_factory=EmbeddingService)

    # Chunk documents
    chunker: BaseChunk = field(
        default_factory=lambda: FixTokenChunk(chunk_size=1200, chunk_overlap=200)
    )

    # Entity extraction
    entity_extractor: BaseEntity = field(default=None)

    # Storage
    vdb: BaseVDB = field(default=None)
    gdb: BaseGDB = field(default=None)

    # Concurrency Parameters
    chunk_upsert_concurrency: int = 16
    entity_upsert_concurrency: int = 16
    relation_upsert_concurrency: int = 16

    # Embedding batch processing parameters
    embedding_batch_size: int = 1000  # Maximum texts to embed in single API call

    async def initialize_tables(self):
        # Initialize the chunks table
        try:
            self.chunks_table = await self.vdb.db.open_table("chunks")
        except Exception as e:
            if str(e) == "Table 'chunks' was not found":
                self.chunks_table = await self.vdb.db.create_table(
                    "chunks",
                    schema=pa.schema(
                        [
                            pa.field("text", pa.string()),
                            pa.field("document_key", pa.string()),
                            pa.field("type", pa.string()),
                            pa.field("filename", pa.string()),
                            pa.field("page_number", pa.int32()),
                            pa.field("uri", pa.string()),
                            pa.field("private", pa.bool_()),
                            pa.field(
                                "chunk_idx", pa.int32()
                            ),  # The index of the chunk in the document
                            pa.field(
                                "document_id", pa.string()
                            ),  # The id of the document that the chunk is from
                            pa.field("vector", pa.list_(pa.float32(), 1536)),
                            pa.field("uploaded_at", pa.timestamp("ms")),
                        ]
                    ),
                )
            else:
                raise e
        try:
            self.entities_table = await self.vdb.db.open_table("entities")
        except Exception as e:
            if str(e) == "Table 'entities' was not found":
                self.entities_table = await self.vdb.db.create_table(
                    "entities",
                    schema=pa.schema(
                        [
                            pa.field("text", pa.string()),
                            pa.field("document_key", pa.string()),
                            pa.field("vector", pa.list_(pa.float32(), 1536)),
                            pa.field("entity_type", pa.string()),
                            pa.field("description", pa.list_(pa.string())),
                            pa.field("chunk_ids", pa.list_(pa.string())),
                        ]
                    ),
                )
            else:
                raise e

    @classmethod
    async def create(cls, **kwargs):
        # LLM
        chat_service = ChatCompletion()
        # Use embedding_batch_size from kwargs if provided, otherwise use default
        embedding_batch_size = kwargs.get("embedding_batch_size", 1000)
        embedding_service = EmbeddingService(default_batch_size=embedding_batch_size)

        if kwargs.get("vdb") is None:
            lancedb = await LanceDB.create(
                embedding_func=embedding_service.create_embeddings,
                db_url="kb_test/hirag.db",
                strategy_provider=RetrievalStrategyProvider(),
            )
            kwargs["vdb"] = lancedb
        if kwargs.get("gdb") is None:
            gdb = NetworkXGDB.create(
                path="kb_test/hirag.gpickle",
                llm_func=chat_service.complete,
            )
            kwargs["gdb"] = gdb

        if kwargs.get("entity_extractor") is None:
            entity_extractor = VanillaEntity.create(
                extract_func=chat_service.complete,
                llm_model_name="gpt-4o-mini",
            )
            kwargs["entity_extractor"] = entity_extractor

        # Set the services in kwargs
        kwargs["chat_service"] = chat_service
        kwargs["embedding_service"] = embedding_service

        instance = cls(**kwargs)
        await instance.initialize_tables()
        return instance

    async def _process_document_with_resume(
        self, chunks, state: DocumentProcessingState, with_graph: bool = True
    ):
        """
        Enhanced document processing with smart resume capabilities and optimized batch operations.
        Now uses unified knowledge graph extraction to avoid redundant processing.
        """
        total_start = time.perf_counter()

        # Step 1: Process pending chunks with BATCH optimization
        pending_chunks = state.get_pending_chunks(chunks)
        if pending_chunks:
            start_upsert_chunks = time.perf_counter()
            logger.info(
                f"📤 Upserting {len(pending_chunks)} pending chunks using BATCH operation..."
            )

            # OPTIMIZATION: Use batch upsert instead of individual upserts
            if len(pending_chunks) > 1:
                # Prepare batch data
                texts_to_embed = []
                properties_list = []

                for chunk in pending_chunks:
                    logger.debug(f"[batch_upsert] Preparing chunk id={chunk.id}")
                    texts_to_embed.append(chunk.page_content)
                    properties_list.append(
                        {
                            "document_key": chunk.id,
                            "text": chunk.page_content,
                            **chunk.metadata.__dict__,
                        }
                    )

                # Single batch operation instead of many individual ones
                await self.vdb.upsert_texts(
                    texts_to_embed=texts_to_embed,
                    properties_list=properties_list,
                    table=self.chunks_table,
                    mode="append",
                )
            else:
                # Single chunk - use individual method
                chunk = pending_chunks[0]
                await self.vdb.upsert_text(
                    text_to_embed=chunk.page_content,
                    properties={
                        "document_key": chunk.id,
                        "text": chunk.page_content,
                        **chunk.metadata.__dict__,
                    },
                    table=self.chunks_table,
                    mode="append",
                )

            upsert_chunks_time = time.perf_counter() - start_upsert_chunks
            logger.info(f"✅ Chunk batch upsert completed in {upsert_chunks_time:.3f}s")
        else:
            logger.info("⏭️ All chunks already processed, skipping chunk upsert")

        if not with_graph:
            total_time = time.perf_counter() - total_start
            logger.info(f"📊 Total processing time: {total_time:.3f}s")
            return

        # Step 2: Unified knowledge graph extraction for pending chunks
        pending_entity_chunks = state.get_pending_entity_chunks(chunks)
        entities = []
        relations = []

        if pending_entity_chunks:
            start_kg_extraction = time.perf_counter()
            logger.info(
                f"🔍 Extracting knowledge graph from {len(pending_entity_chunks)} pending chunks..."
            )

            # Extract both entities and relations in one unified operation
            entities, relations = await self.entity_extractor.extract_knowledge_graph(
                pending_entity_chunks, use_cache=False
            )
            
            kg_extraction_time = time.perf_counter() - start_kg_extraction
            logger.info(
                f"✅ Knowledge graph extraction completed in {kg_extraction_time:.3f}s"
            )
        else:
            logger.info("⏭️ All chunks already have knowledge graph extracted")

        # Step 3: Process new entities with BATCH optimization
        if entities:
            start_upsert_entities = time.perf_counter()

            # Filter out existing entities
            new_entities = [
                ent for ent in entities if ent.id not in state.processed_entity_ids
            ]

            if new_entities:
                logger.info(
                    f"📤 Upserting {len(new_entities)} new entities using BATCH operation..."
                )

                # OPTIMIZATION: Use batch upsert for entities
                if len(new_entities) > 1:
                    # Prepare batch data for entities
                    texts_to_embed = []
                    properties_list = []

                    for ent in new_entities:
                        description_text = (
                            " | ".join(ent.metadata.description)
                            if ent.metadata.description
                            else ""
                        )
                        texts_to_embed.append(description_text)
                        properties_list.append(
                            {
                                "document_key": ent.id,
                                "text": ent.page_content,
                                **ent.metadata.__dict__,
                            }
                        )

                    # Single batch operation for all entities
                    await self.vdb.upsert_texts(
                        texts_to_embed=texts_to_embed,
                        properties_list=properties_list,
                        table=self.entities_table,
                        mode="append",
                    )
                else:
                    # Single entity - use individual method
                    ent = new_entities[0]
                    description_text = (
                        " | ".join(ent.metadata.description)
                        if ent.metadata.description
                        else ""
                    )
                    await self.vdb.upsert_text(
                        text_to_embed=description_text,
                        properties={
                            "document_key": ent.id,
                            "text": ent.page_content,
                            **ent.metadata.__dict__,
                        },
                        table=self.entities_table,
                        mode="append",
                    )

                upsert_entities_time = time.perf_counter() - start_upsert_entities
                logger.info(
                    f"✅ Entity batch upsert completed in {upsert_entities_time:.3f}s"
                )
            else:
                logger.info("⏭️ No new entities to upsert")

            # Upsert entities to graph
            start_upsert_graph = time.perf_counter()
            if new_entities:
                await self.gdb.upsert_nodes(
                    new_entities, concurrency=self.entity_upsert_concurrency
                )
            upsert_graph_time = time.perf_counter() - start_upsert_graph
            logger.info(f"✅ Graph nodes upsert completed in {upsert_graph_time:.3f}s")
        else:
            logger.info("⏭️ No new entities to process")

        # Step 4: Process relations extracted in the unified approach
        if relations:
            start_upsert_relations = time.perf_counter()
            logger.info(f"📤 Upserting {len(relations)} relations...")

            relation_coros = []
            for rel in relations:
                # Simple duplicate check based on source-target pair
                rel_pair = (rel.source, rel.target)
                if rel_pair not in state.processed_relation_pairs:
                    relation_coros.append(self.gdb.upsert_relation(rel))

            if relation_coros:
                await _limited_gather(
                    relation_coros, self.relation_upsert_concurrency
                )
                upsert_relations_time = time.perf_counter() - start_upsert_relations
                logger.info(
                    f"✅ Relation upsert completed in {upsert_relations_time:.3f}s"
                )
            else:
                logger.info("⏭️ All relations already exist")
        else:
            logger.info("ℹ️ No relations to process")

        total_time = time.perf_counter() - total_start
        logger.info(f"📊 Total processing time: {total_time:.3f}s")

    async def _process_document(self, chunks, with_graph: bool = True):
        """
        Legacy wrapper for backward compatibility.
        Calls the new resume-enabled processing method.
        """
        logger.info(
            "🔄 Using legacy _process_document method - upgrading to resume-enabled version"
        )

        # Create a minimal state for non-resume processing
        if not chunks:
            return

        document_uri = chunks[0].metadata.uri
        resume_manager = ResumeManager(self.chunks_table, self.entities_table, self.gdb)

        # Use the new URI-based method
        state = await resume_manager.get_document_state_by_uri(document_uri, chunks)
        resume_manager.log_resume_plan(state, chunks)

        await self._process_document_with_resume(chunks, state, with_graph)

    async def insert_to_kb(
        self,
        document_path: str,
        content_type: str,
        with_graph: bool = True,
        document_meta: Optional[dict] = None,
        loader_configs: Optional[dict] = None,
    ):
        """
        Enhanced insert_to_kb with smart resume functionality.

        This method now supports:
        - Resuming from any stage of processing (chunks, entities, relations)
        - Skipping already processed components
        - Robust error handling and state tracking
        - Detection of document content changes
        - Unified knowledge graph extraction to avoid redundant processing
        """
        logger.info(f"🚀 Starting document processing: {document_path}")
        start_total = time.perf_counter()

        # Clear entity extractor cache to ensure fresh processing for this document
        if hasattr(self.entity_extractor, 'clear_cache'):
            self.entity_extractor.clear_cache()

        # Load and chunk the document
        documents = await asyncio.to_thread(
            load_document,
            document_path,
            content_type,
            document_meta,
            loader_configs,
            loader_type="langchain",
        )
        logger.info(f"📄 Loaded {len(documents)} documents")

        # Chunk all documents
        all_chunks = [chunk for doc in documents for chunk in self.chunker.chunk(doc)]
        logger.info(f"✂️ Created {len(all_chunks)} chunks")

        if not all_chunks:
            logger.warning("⚠️ No chunks created from document")
            return

        # Get document URI for resume functionality
        document_uri = all_chunks[0].metadata.uri

        # Initialize resume manager and analyze current state
        resume_manager = ResumeManager(self.chunks_table, self.entities_table, self.gdb)

        # Check for document changes
        change_info = await resume_manager.check_document_changes(
            document_uri, all_chunks
        )
        if change_info.get("changed"):
            logger.info(f"📝 Document changes detected: {change_info.get('reason')}")
            if change_info.get("reason") == "different_chunks":
                logger.info(f"  📊 New chunks: {change_info.get('new_chunks')}")
                logger.info(
                    f"  📊 Existing chunks: {change_info.get('existing_chunks')}"
                )
                logger.info(f"  📊 Common chunks: {change_info.get('common_chunks')}")

        # Get processing state based on URI
        state = await resume_manager.get_document_state_by_uri(document_uri, all_chunks)

        # Log resume plan
        resume_manager.log_resume_plan(state, all_chunks)

        # Check if document is already fully processed
        if (
            state.chunks_complete
            and state.entities_extracted
            and not state.get_pending_chunks(all_chunks)
            and not state.get_pending_entity_chunks(all_chunks)
        ):
            logger.info("🎉 Document already fully processed!")

            # Verify data integrity
            logger.info("🔍 Verifying data integrity...")
            if len(state.processed_chunk_ids) == len(all_chunks):
                logger.info("✅ All chunks verified in database")
            else:
                logger.warning(
                    f"⚠️ Chunk count mismatch: {len(state.processed_chunk_ids)} in DB vs {len(all_chunks)} generated"
                )

            return

        # Process document with resume capabilities
        await self._process_document_with_resume(all_chunks, state, with_graph)

        # Dump graph state
        if with_graph:
            await self.gdb.dump()

        total = time.perf_counter() - start_total
        logger.info(f"🏁 Total pipeline time: {total:.3f}s")

    async def query_chunks(
        self, query: str, topk: int = 10, topn: int = 5
    ) -> list[dict[str, Any]]:
        chunks = await self.vdb.query(
            query=query,
            table=self.chunks_table,
            topk=topk,  # before reranking
            topn=topn,  # after reranking
            columns_to_select=[
                "text",
                "uri",
                "filename",
                "private",
                "uploaded_at",
                "document_key",
            ],
        )
        return chunks

    async def query_entities(
        self, query: str, topk: int = 10, topn: int = 5
    ) -> list[dict[str, Any]]:
        entities = await self.vdb.query(
            query=query,
            table=self.entities_table,
            topk=topk,  # before reranking
            topn=topn,  # after reranking
            columns_to_select=["text", "document_key", "entity_type", "description"],
        )
        return entities

    async def query_relations(
        self, query: str, topk: int = 10, topn: int = 5
    ) -> tuple[list[str], list[str]]:
        # search the entities
        recall_entities = await self.query_entities(query, topk, topn)
        recall_entities = [entity["document_key"] for entity in recall_entities]
        # search the relations
        recall_neighbors = []
        recall_edges = []
        for entity in recall_entities:
            neighbors, edges = await self.gdb.query_one_hop(entity)
            recall_neighbors.extend(neighbors)
            recall_edges.extend(edges)
        return recall_neighbors, recall_edges

    async def query_all(self, query: str) -> dict[str, list[dict]]:
        # search chunks
        recall_chunks = await self.query_chunks(query, topk=10, topn=5)
        # search entities
        recall_entities = await self.query_entities(query, topk=10, topn=5)
        # search relations
        recall_neighbors, recall_edges = await self.query_relations(
            query, topk=10, topn=5
        )
        # merge the results
        # TODO: the recall results are not returned in the same format
        return {
            "chunks": recall_chunks,
            "entities": recall_entities,
            "neighbors": recall_neighbors,
            "relations": recall_edges,
        }

    async def clean_up(self):
        await self.gdb.clean_up()
        await self.vdb.clean_up()

    # Document status management utilities
    async def get_document_status(
        self, document_id: str = None, uri: str = None
    ) -> Dict[str, Any]:
        """
        Get processing status for a specific document.

        Args:
            document_id: The document ID to check
            uri: The document URI to check (alternative to document_id)

        Returns:
            Dictionary with processing status information
        """
        if not document_id and not uri:
            raise ValueError("Either document_id or uri must be provided")

        # Find chunks for this document
        if document_id:
            query_condition = f"document_id == '{document_id}'"
        else:
            query_condition = f"uri == '{uri}'"

        try:
            chunks_data = (
                await self.chunks_table.query().where(query_condition).to_list()
            )
            if not chunks_data:
                return {
                    "status": "not_found",
                    "message": "No chunks found for this document",
                }

            # Extract document info
            first_chunk = chunks_data[0]
            doc_id = first_chunk["document_id"]
            doc_uri = first_chunk["uri"]
            total_chunks = len(chunks_data)

            # Check entities
            entities_data = (
                await self.entities_table.query()
                .where(f"'{doc_id}' in chunk_ids")
                .to_list()
            )

            covered_chunks = set()
            for entity in entities_data:
                chunk_ids = entity.get("chunk_ids", [])
                if isinstance(chunk_ids, list):
                    covered_chunks.update(chunk_ids)

            return {
                "status": "found",
                "document_id": doc_id,
                "uri": doc_uri,
                "chunks": {
                    "total": total_chunks,
                    "processed": total_chunks,  # All chunks in DB are processed
                    "complete": True,
                },
                "entities": {
                    "total_entities": len(entities_data),
                    "chunks_with_entities": len(covered_chunks),
                    "entity_extraction_complete": len(covered_chunks) == total_chunks,
                },
                "last_updated": first_chunk.get("uploaded_at"),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking document status: {e}",
            }

    async def list_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all documents in the knowledge base with their processing status.

        Args:
            limit: Maximum number of documents to return

        Returns:
            List of document status dictionaries
        """
        try:
            # Get unique documents from chunks table
            chunks_data = await self.chunks_table.query().limit(limit * 10).to_list()

            # Group by document_id
            docs_by_id = defaultdict(list)
            for chunk in chunks_data:
                docs_by_id[chunk["document_id"]].append(chunk)

            results = []
            for doc_id, chunks in list(docs_by_id.items())[:limit]:
                status = await self.get_document_status(document_id=doc_id)
                if status.get("status") == "found":
                    results.append(status)

            return results

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    async def remove_document(
        self, document_id: str = None, uri: str = None, force: bool = False
    ) -> Dict[str, Any]:
        """
        Remove a document and all its associated data from the knowledge base.

        Args:
            document_id: The document ID to remove
            uri: The document URI to remove (alternative to document_id)
            force: If True, ignore errors and continue removal

        Returns:
            Dictionary with removal results
        """
        if not document_id and not uri:
            raise ValueError("Either document_id or uri must be provided")

        results = {
            "document_id": document_id,
            "uri": uri,
            "chunks_removed": 0,
            "entities_removed": 0,
            "relations_removed": 0,
            "errors": [],
        }

        try:
            # Get document info first
            status = await self.get_document_status(document_id=document_id, uri=uri)
            if status.get("status") != "found":
                return {"status": "not_found", "message": "Document not found"}

            doc_id = status["document_id"]
            results["document_id"] = doc_id
            results["uri"] = status["uri"]

            # Remove chunks
            try:
                # Note: LanceDB doesn't have direct delete by condition,
                # so we'd need to implement this based on the specific storage backend
                logger.warning(
                    "Chunk removal not implemented - requires backend-specific implementation"
                )
                results["errors"].append("Chunk removal not implemented")
            except Exception as e:
                results["errors"].append(f"Error removing chunks: {e}")
                if not force:
                    raise

            # Remove entities
            try:
                logger.warning(
                    "Entity removal not implemented - requires backend-specific implementation"
                )
                results["errors"].append("Entity removal not implemented")
            except Exception as e:
                results["errors"].append(f"Error removing entities: {e}")
                if not force:
                    raise

            # Remove relations from graph
            try:
                if hasattr(self.gdb, "remove_document_relations"):
                    relations_removed = await self.gdb.remove_document_relations(doc_id)
                    results["relations_removed"] = relations_removed
                else:
                    logger.warning("Graph relation removal not implemented")
                    results["errors"].append("Graph relation removal not implemented")
            except Exception as e:
                results["errors"].append(f"Error removing relations: {e}")
                if not force:
                    raise

            return results

        except Exception as e:
            results["errors"].append(f"General error: {e}")
            return results

    async def repair_document(
        self, document_id: str = None, uri: str = None
    ) -> Dict[str, Any]:
        """
        Attempt to repair a partially processed document by re-running failed stages.

        Args:
            document_id: The document ID to repair
            uri: The document URI to repair

        Returns:
            Dictionary with repair results
        """
        if not document_id and not uri:
            raise ValueError("Either document_id or uri must be provided")

        try:
            # Get current status
            status = await self.get_document_status(document_id=document_id, uri=uri)
            if status.get("status") != "found":
                return {"status": "not_found", "message": "Document not found"}

            doc_id = status["document_id"]
            doc_uri = status["uri"]

            # Check if repair is needed
            needs_repair = (
                not status["chunks"]["complete"]
                or not status["entities"]["entity_extraction_complete"]
            )

            if not needs_repair:
                return {
                    "status": "no_repair_needed",
                    "message": "Document appears to be fully processed",
                }

            logger.info(f"🔧 Attempting to repair document: {doc_uri}")

            # For repair, we need to reload the document and re-process
            # This is a simplified approach - in production you might want to store
            # the original document path for easier repair
            return {
                "status": "repair_needed",
                "message": "Document repair requires re-processing with original document path",
                "suggestion": "Use insert_to_kb with the original document path - the resume functionality will handle partial processing",
            }

        except Exception as e:
            return {"status": "error", "message": f"Error during repair: {e}"}

    # Document status and debugging utilities
    async def debug_document_resume_state(
        self,
        document_path: str,
        content_type: str,
        document_meta: Optional[dict] = None,
        loader_configs: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Debug utility to check the resume state of a document without processing it.
        Useful for troubleshooting resume issues.
        """
        logger.info(f"🔍 Debugging resume state for: {document_path}")

        try:
            # Load and chunk the document (same as insert_to_kb)
            documents = await asyncio.to_thread(
                load_document,
                document_path,
                content_type,
                document_meta,
                loader_configs,
                loader_type="langchain",
            )

            all_chunks = [
                chunk for doc in documents for chunk in self.chunker.chunk(doc)
            ]

            if not all_chunks:
                return {"error": "No chunks created from document"}

            document_uri = all_chunks[0].metadata.uri
            resume_manager = ResumeManager(
                self.chunks_table, self.entities_table, self.gdb
            )

            # Get detailed state information
            state = await resume_manager.get_document_state_by_uri(
                document_uri, all_chunks
            )
            change_info = await resume_manager.check_document_changes(
                document_uri, all_chunks
            )

            # Create detailed debug info
            debug_info = {
                "document_uri": document_uri,
                "document_id": state.document_id,
                "total_chunks_generated": len(all_chunks),
                "chunks_in_db": len(state.processed_chunk_ids),
                "entities_extracted_chunks": len(state.extracted_entity_chunk_ids),
                "total_entities": len(state.processed_entity_ids),
                "total_relations": len(state.processed_relation_pairs),
                "chunks_complete": state.chunks_complete,
                "entities_extracted": state.entities_extracted,
                "change_detection": change_info,
                "pending_chunks": len(state.get_pending_chunks(all_chunks)),
                "pending_entity_chunks": len(
                    state.get_pending_entity_chunks(all_chunks)
                ),
                "chunk_ids_sample": {
                    "generated_chunk_ids": [chunk.id for chunk in all_chunks[:5]],
                    "processed_chunk_ids": list(state.processed_chunk_ids)[:5],
                    "extracted_entity_chunk_ids": list(
                        state.extracted_entity_chunk_ids
                    )[:5],
                },
            }

            return debug_info

        except Exception as e:
            logger.error(f"Error during resume state debugging: {e}")
            return {"error": str(e)}
