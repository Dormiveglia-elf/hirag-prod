import logging
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional

from hirag_prod._utils import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _limited_gather,
    compute_mdhash_id,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from hirag_prod.prompt import PROMPTS
from hirag_prod.schema import Chunk, Entity, Relation
from hirag_prod.summarization import BaseSummarizer, TrancatedAggregateSummarizer

from .base import BaseEntity


@dataclass
class VanillaEntity(BaseEntity):
    # === Core Components ===
    llm_model_name: str = field(default="gpt-4o-mini")
    extract_func: Callable
    entity_description_summarizer: BaseSummarizer = field(default=None)
    
    # === Unified Knowledge Graph Extraction Parameters ===
    kg_extract_prompt: str = field(
        default_factory=lambda: PROMPTS["KG_EXTRACTION"]
    )
    kg_extract_continue_prompt: str = field(
        default_factory=lambda: PROMPTS["KG_EXTRACTION_CONTINUE"]
    )
    kg_extract_max_gleaning: int = field(default_factory=lambda: 1)
    entity_extract_termination_prompt: str = field(
        default_factory=lambda: PROMPTS["entity_if_loop_extraction"]
    )
    relation_extract_termination_prompt: str = field(
        default_factory=lambda: PROMPTS["relation_if_loop_extraction"]
    )
    kg_extract_context: dict = field(
        default_factory=lambda: {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "entity_types": ",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
            "language": "English",
        }
    )

    # === Cache for Extracted Knowledge Graph ===
    _cached_kg_results: Dict[str, Tuple[List[Entity], List[Relation]]] = field(default_factory=dict)
    _cache_enabled: bool = field(default=True)

    # === Legacy Support ===
    entity_extract_prompt: str = field(
        default_factory=lambda: PROMPTS["hi_entity_extraction"]
    )
    relation_extract_prompt: str = field(
        default_factory=lambda: PROMPTS["hi_relation_extraction"]
    )
    continue_prompt: str = field(
        default_factory=lambda: PROMPTS["entity_continue_extraction"]
    )
    entity_extract_max_gleaning: int = field(default_factory=lambda: 1)
    relation_extract_max_gleaning: int = field(default_factory=lambda: 1)
    entity_extract_context: dict = field(
        default_factory=lambda: {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "entity_types": ",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        }
    )
    relation_extract_context: dict = field(
        default_factory=lambda: {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        }
    )

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def __post_init__(self):
        if self.entity_description_summarizer is None:
            self.entity_description_summarizer = TrancatedAggregateSummarizer(
                llm_model_name=self.llm_model_name,
                extract_func=self.extract_func,
            )

    def _get_chunks_cache_key(self, chunks: List[Chunk]) -> str:
        """Generate a cache key for the given chunks."""
        return "|".join(sorted([chunk.id for chunk in chunks]))

    async def extract_knowledge_graph(self, chunks: List[Chunk], use_cache: bool = True) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract both entities and relations in a unified approach using KG_EXTRACTION prompt.
        This is more efficient than separate entity and relation extraction.
        
        Args:
            chunks: List of text chunks to extract knowledge from
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple of (entities, relations) extracted from the chunks
        """
        # Check cache first if enabled
        if use_cache and self._cache_enabled:
            cache_key = self._get_chunks_cache_key(chunks)
            if cache_key in self._cached_kg_results:
                logging.info(f"[KG-Cache] Using cached results for {len(chunks)} chunks")
                return self._cached_kg_results[cache_key]

        async def _process_single_chunk_kg(chunk: Chunk) -> Tuple[List[Entity], List[Relation]]:
            """Process a single chunk to extract both entities and relations."""
            try:
                chunk_key = chunk.id
                content = chunk.page_content
                start_time = time.time()
                
                # 1. Initial combined knowledge graph extraction
                kg_extraction_prompt = self.kg_extract_prompt.format(
                    **self.kg_extract_context, input_text=content
                )
                kg_string_result = await self.extract_func(
                    model=self.llm_model_name,
                    prompt=kg_extraction_prompt,
                )

                content_history = pack_user_ass_to_openai_messages(
                    kg_extraction_prompt, kg_string_result
                )

                # 2. Continue extraction for missed entities/relations
                for glean_idx in range(self.kg_extract_max_gleaning):
                    continue_prompt = self.kg_extract_continue_prompt.format(
                        **self.kg_extract_context
                    )
                    glean_result = await self.extract_func(
                        model=self.llm_model_name,
                        prompt=continue_prompt,
                        history_messages=content_history,
                    )

                    content_history += pack_user_ass_to_openai_messages(
                        continue_prompt, glean_result
                    )
                    kg_string_result += glean_result
                    
                    if glean_idx == self.kg_extract_max_gleaning - 1:
                        break

                    # Check if we should continue extraction
                    termination_check = await self._check_extraction_completion(
                        content_history, kg_string_result
                    )
                    if not termination_check:
                        break

                # 3. Parse the combined result into entities and relations
                entities, relations = await self._parse_kg_result(kg_string_result, chunk_key)
                
                elapsed = time.time() - start_time
                logging.info(
                    f"[KG] Finished extracting knowledge graph for chunk {chunk_key}, "
                    f"entities: {len(entities)}, relations: {len(relations)}, "
                    f"took {elapsed:.2f} seconds"
                )
                return entities, relations
                
            except Exception as e:
                logging.exception(f"[KG] Knowledge graph extraction failed for chunk {chunk.id}")
                warnings.warn(f"Knowledge graph extraction failed for chunk {chunk.id}: {e}")
                return [], []

        # Process all chunks concurrently
        kg_extraction_concurrency: int = 4
        extraction_coros = [_process_single_chunk_kg(chunk) for chunk in chunks]
        results_list = await _limited_gather(extraction_coros, kg_extraction_concurrency)
        
        # Separate and flatten entities and relations
        all_entities = []
        all_relations = []
        for entities, relations in results_list:
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        # Merge duplicate entities
        merged_entities = await self._merge_duplicate_entities(all_entities)
        
        # Filter relations to only include those with valid entities
        valid_relations = self._filter_valid_relations(all_relations, merged_entities)
        
        final_result = (merged_entities, valid_relations)
        
        # Cache the result if enabled
        if use_cache and self._cache_enabled:
            cache_key = self._get_chunks_cache_key(chunks)
            self._cached_kg_results[cache_key] = final_result
            logging.info(f"[KG-Cache] Cached results for {len(chunks)} chunks")
        
        return final_result

    async def entity(self, chunks: List[Chunk]) -> List[Entity]:
        """
        Extract entities efficiently using cached knowledge graph results.
        
        Args:
            chunks: List of text chunks to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities, _ = await self.extract_knowledge_graph(chunks, use_cache=True)
        return entities

    async def relation(self, chunks: List[Chunk], entities: Optional[List[Entity]] = None) -> List[Relation]:
        """
        Extract relations efficiently using cached knowledge graph results.
        
        Args:
            chunks: List of text chunks to extract relations from
            entities: Optional list of entities for validation (kept for backward compatibility)
            
        Returns:
            List of extracted relations
        """
        # Use cached results from previous entity extraction
        cached_entities, cached_relations = await self.extract_knowledge_graph(chunks, use_cache=True)
        
        # If entities are provided, validate relations against them
        if entities:
            entity_ids = {entity.id for entity in entities}
            valid_relations = [
                relation for relation in cached_relations
                if relation.source in entity_ids and relation.target in entity_ids
            ]
            return valid_relations
        
        return cached_relations

    def clear_cache(self):
        """Clear the knowledge graph extraction cache."""
        self._cached_kg_results.clear()
        logging.info("[KG-Cache] Cleared knowledge graph extraction cache")

    def disable_cache(self):
        """Disable caching for knowledge graph extraction."""
        self._cache_enabled = False
        self.clear_cache()
        logging.info("[KG-Cache] Disabled knowledge graph extraction cache")

    def enable_cache(self):
        """Enable caching for knowledge graph extraction."""
        self._cache_enabled = True
        logging.info("[KG-Cache] Enabled knowledge graph extraction cache")

    async def _check_extraction_completion(self, content_history: list, current_result: str) -> bool:
        """
        Check if we should continue extraction by asking the model if anything was missed.
        
        Args:
            content_history: Conversation history
            current_result: Current extraction result
            
        Returns:
            True if we should continue extraction, False otherwise
        """
        try:
            # Simple heuristic: if the current result is very short, likely more to extract
            if len(current_result.strip()) < 50:
                return True
                
            # Ask model if entities are missing
            entity_check = await self.extract_func(
                model=self.llm_model_name,
                prompt=self.entity_extract_termination_prompt,
                history_messages=content_history,
            )
            entity_needs_continue = entity_check.strip().lower().startswith("yes")
            
            # Ask model if relations are missing
            relation_check = await self.extract_func(
                model=self.llm_model_name,
                prompt=self.relation_extract_termination_prompt,
                history_messages=content_history,
            )
            relation_needs_continue = relation_check.strip().lower().startswith("yes")
            
            return entity_needs_continue or relation_needs_continue
            
        except Exception as e:
            logging.warning(f"[KG] Error checking extraction completion: {e}")
            return False

    async def _parse_kg_result(self, kg_result: str, chunk_key: str) -> Tuple[List[Entity], List[Relation]]:
        """
        Parse the combined knowledge graph extraction result into separate entities and relations.
        
        Args:
            kg_result: Raw result from KG extraction
            chunk_key: ID of the source chunk
            
        Returns:
            Tuple of (entities, relations)
        """
        try:
            # Split the result into individual records
            records = split_string_by_multi_markers(
                kg_result,
                [
                    self.kg_extract_context["record_delimiter"],
                    self.kg_extract_context["completion_delimiter"],
                ],
            )
            
            entities = []
            relations = []
            entity_name_to_obj = {}  # Track entities for relation building
            
            for record in records:
                # Extract content within parentheses
                match = re.search(r"\((.*?)\)", record)
                if not match:
                    continue
                    
                record_content = match.group(1)
                record_attributes = split_string_by_multi_markers(
                    record_content, [self.kg_extract_context["tuple_delimiter"]]
                )
                
                if len(record_attributes) < 2:
                    continue
                    
                record_type = record_attributes[0].strip().strip('"')
                
                if record_type == "entity":
                    entity_data = await _handle_single_entity_extraction(
                        record_attributes, chunk_key
                    )
                    if entity_data:
                        entity = Entity(
                            id=compute_mdhash_id(entity_data["entity_name"], prefix="ent-"),
                            page_content=entity_data["entity_name"],
                            metadata={
                                "entity_type": entity_data["entity_type"],
                                "description": [entity_data["description"]],
                                "chunk_ids": [chunk_key],
                            },
                        )
                        entities.append(entity)
                        entity_name_to_obj[entity_data["entity_name"]] = entity
                        
                elif record_type == "relationship":
                    relation_data = await _handle_single_relationship_extraction(
                        record_attributes, chunk_key
                    )
                    if relation_data:
                        # Verify that both source and target entities exist
                        src_name = relation_data["src_id"]
                        tgt_name = relation_data["tgt_id"]
                        
                        # For now, create placeholder relations - will be validated later
                        relation = {
                            "source_name": src_name,
                            "target_name": tgt_name,
                            "description": relation_data["description"],
                            "weight": relation_data["weight"],
                            "chunk_id": chunk_key,
                        }
                        relations.append(relation)
            
            return entities, relations
            
        except Exception as e:
            logging.exception(f"[KG] Error parsing knowledge graph result for chunk {chunk_key}")
            return [], []

    async def _merge_duplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merge entities with the same name across different chunks.
        
        Args:
            entities: List of entities to merge
            
        Returns:
            List of merged entities
        """
        if not entities:
            return []
            
        # Group entities by name
        entities_by_name = defaultdict(list)
        for entity in entities:
            entities_by_name[entity.page_content].append(entity)
        
        async def _merge_entity_group(entity_name: str, entity_group: List[Entity]) -> Entity:
            """Merge a group of entities with the same name."""
            if len(entity_group) == 1:
                return entity_group[0]
                
            try:
                # Collect all descriptions and chunk_ids
                all_descriptions = []
                all_chunk_ids = []
                entity_types = []
                
                for entity in entity_group:
                    all_descriptions.extend(entity.metadata.description)
                    all_chunk_ids.extend(entity.metadata.chunk_ids)
                    entity_types.append(entity.metadata.entity_type)
                
                # Choose the most common entity type
                entity_type_counter = Counter(entity_types)
                final_entity_type = entity_type_counter.most_common(1)[0][0]
                
                # Create merged entity
                merged_entity = Entity(
                    id=compute_mdhash_id(entity_name, prefix="ent-"),
                    page_content=entity_name,
                    metadata={
                        "entity_type": final_entity_type,
                        "description": all_descriptions,
                        "chunk_ids": list(set(all_chunk_ids)),  # Remove duplicates
                    },
                )
                
                logging.info(
                    f"[KG-Merge] Merged entity '{entity_name}': "
                    f"type={final_entity_type}, chunks={len(set(all_chunk_ids))}, "
                    f"descriptions={len(all_descriptions)}"
                )
                return merged_entity
                
            except Exception as e:
                logging.exception(f"[KG-Merge] Error merging entity '{entity_name}': {e}")
                return entity_group[0]  # Return first entity as fallback
        
        # Merge entities concurrently
        merge_coros = [
            _merge_entity_group(name, group) 
            for name, group in entities_by_name.items()
        ]
        
        entity_merge_concurrency: int = 4
        merged_entities = await _limited_gather(merge_coros, entity_merge_concurrency)
        
        return [entity for entity in merged_entities if entity is not None]

    def _filter_valid_relations(self, relations: List[dict], entities: List[Entity]) -> List[Relation]:
        """
        Filter relations to only include those where both source and target entities exist.
        
        Args:
            relations: List of relation dictionaries from parsing
            entities: List of valid entities
            
        Returns:
            List of valid Relation objects
        """
        # Create mapping from entity names to entity objects
        entity_name_to_obj = {entity.page_content: entity for entity in entities}
        
        valid_relations = []
        for relation_data in relations:
            src_name = relation_data["source_name"]
            tgt_name = relation_data["target_name"]
            
            if src_name in entity_name_to_obj and tgt_name in entity_name_to_obj:
                source_entity = entity_name_to_obj[src_name]
                target_entity = entity_name_to_obj[tgt_name]
                
                relation = Relation(
                    source=source_entity.id,
                    target=target_entity.id,
                    properties={
                        "description": relation_data["description"],
                        "weight": relation_data["weight"],
                        "chunk_id": relation_data["chunk_id"],
                    },
                )
                valid_relations.append(relation)
            else:
                logging.warning(
                    f"[KG] Skipping relation with missing entities: "
                    f"{src_name} -> {tgt_name}"
                )
        
        return valid_relations

    # === Legacy Methods (for backward compatibility) ===
    
    async def legacy_entity_extraction(self, chunks: List[Chunk]) -> List[Entity]:
        """
        Legacy method for separate entity extraction using hi_entity_extraction prompt.
        Kept for backward compatibility and comparison purposes.
        
        Args:
            chunks: List of text chunks to extract entities from
            
        Returns:
            List of extracted entities using the legacy approach
        """
        async def _process_single_content_entity(chunk: Chunk) -> List[Entity]:
            try:
                chunk_key = chunk.id
                content = chunk.page_content
                start_time = time.time()
                
                # 1. Initial extraction using legacy prompt
                entity_extraction_prompt = self.entity_extract_prompt.format(
                    **self.entity_extract_context, input_text=content
                )
                entity_string_result = await self.extract_func(
                    model=self.llm_model_name,
                    prompt=entity_extraction_prompt,
                )

                content_history = pack_user_ass_to_openai_messages(
                    entity_extraction_prompt, entity_string_result
                )

                # 2. Continue to extract entities for higher quality extraction
                for glean_idx in range(self.entity_extract_max_gleaning):
                    glean_result = await self.extract_func(
                        model=self.llm_model_name,
                        prompt=self.continue_prompt,
                        history_messages=content_history,
                    )

                    content_history += pack_user_ass_to_openai_messages(
                        self.continue_prompt, glean_result
                    )
                    entity_string_result += glean_result
                    
                    if glean_idx == self.entity_extract_max_gleaning - 1:
                        break

                    entity_extraction_termination_str: str = (
                        await self.extract_func(
                            model=self.llm_model_name,
                            prompt=self.entity_extract_termination_prompt,
                            history_messages=content_history,
                        )
                    )
                    entity_extraction_termination_str = (
                        entity_extraction_termination_str.strip()
                        .strip('"')
                        .strip("'")
                        .lower()
                    )
                    if entity_extraction_termination_str != "yes":
                        break

                # 3. Parse entities from the result
                records: List[str] = split_string_by_multi_markers(
                    entity_string_result,
                    [
                        self.entity_extract_context["record_delimiter"],
                        self.entity_extract_context["completion_delimiter"],
                    ],
                )

                # 4. Extract entity objects using regex
                entities = []
                for record in records:
                    record = re.search(r"\((.*?)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(
                        record, [self.entity_extract_context["tuple_delimiter"]]
                    )
                    entity = await _handle_single_entity_extraction(
                        record_attributes, chunk_key
                    )
                    if entity is not None:
                        entities.append(
                            Entity(
                                id=compute_mdhash_id(
                                    entity["entity_name"], prefix="ent-"
                                ),
                                page_content=entity["entity_name"],
                                metadata={
                                    "entity_type": entity["entity_type"],
                                    "description": [entity["description"]],
                                    "chunk_ids": [chunk_key],
                                },
                            )
                        )
                
                elapsed = time.time() - start_time
                logging.info(
                    f"[Legacy-Entity] Finished extracting entities for chunk {chunk_key}, "
                    f"took {elapsed:.2f} seconds"
                )
                return entities
                
            except Exception as e:
                logging.exception(f"[Legacy-Entity] Extraction failed for chunk {chunk.id}")
                warnings.warn(f"Legacy entity extraction failed for chunk {chunk.id}: {e}")
                return []

        # Process chunks concurrently
        entity_extraction_concurrency: int = 4
        extraction_coros = [_process_single_content_entity(chunk) for chunk in chunks]
        entities_list = await _limited_gather(extraction_coros, entity_extraction_concurrency)
        
        # Flatten and merge entities
        entities = [
            entity
            for entity_list in entities_list
            if entity_list
            for entity in entity_list
        ]
        
        return await self._merge_duplicate_entities(entities)

    async def legacy_relation_extraction(
        self, chunks: List[Chunk], entities: List[Entity]
    ) -> List[Relation]:
        """
        Legacy method for separate relation extraction using hi_relation_extraction prompt.
        Kept for backward compatibility and comparison purposes.
        
        Args:
            chunks: List of text chunks to extract relations from
            entities: List of entities to build relations between
            
        Returns:
            List of extracted relations using the legacy approach
        """
        async def _process_single_content_relation(
            chunk: Chunk, entities_dict: Dict[str, Entity]
        ) -> List[Relation]:
            try:
                chunk_key = chunk.id
                content = chunk.page_content
                start_time = time.time()
                
                # 1. Initial extraction using legacy prompt
                relation_extract_prompt = self.relation_extract_prompt.format(
                    **self.relation_extract_context,
                    entities=[e.page_content for e in entities_dict.values()],
                    input_text=content,
                )
                relation_string_result = await self.extract_func(
                    model=self.llm_model_name,
                    prompt=relation_extract_prompt,
                )

                content_history = pack_user_ass_to_openai_messages(
                    relation_extract_prompt, relation_string_result
                )

                # 2. Continue to extract relations for higher quality extraction
                for glean_idx in range(self.relation_extract_max_gleaning):
                    glean_result = await self.extract_func(
                        model=self.llm_model_name,
                        prompt=self.continue_prompt,
                        history_messages=content_history,
                    )

                    content_history += pack_user_ass_to_openai_messages(
                        self.continue_prompt, glean_result
                    )
                    relation_string_result += glean_result
                    
                    if glean_idx == self.relation_extract_max_gleaning - 1:
                        break

                    relation_extraction_termination_str: str = (
                        await self.extract_func(
                            model=self.llm_model_name,
                            prompt=self.relation_extract_termination_prompt,
                            history_messages=content_history,
                        )
                    )
                    relation_extraction_termination_str = (
                        relation_extraction_termination_str.strip()
                        .strip('"')
                        .strip("'")
                        .lower()
                    )
                    if relation_extraction_termination_str != "yes":
                        break

                # 3. Parse relations from the result
                records = split_string_by_multi_markers(
                    relation_string_result,
                    [
                        self.relation_extract_context["record_delimiter"],
                        self.relation_extract_context["completion_delimiter"],
                    ],
                )

                # 4. Extract relation objects using regex
                relations = []
                for record in records:
                    record = re.search(r"\((.*)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(
                        record, [self.relation_extract_context["tuple_delimiter"]]
                    )
                    relation = await _handle_single_relationship_extraction(
                        record_attributes, chunk_key
                    )
                    if relation is not None:
                        try:
                            source = entities_dict[relation["src_id"]]
                        except KeyError:
                            logging.warning(
                                f"[Legacy-Relation] Source entity {relation['src_id']} "
                                f"not found in entities_dict, skipping relation {relation}"
                            )
                            continue
                        try:
                            target = entities_dict[relation["tgt_id"]]
                        except KeyError:
                            logging.warning(
                                f"[Legacy-Relation] Target entity {relation['tgt_id']} "
                                f"not found in entities_dict, skipping relation {relation}"
                            )
                            continue
                        relation = Relation(
                            source=source.id,
                            target=target.id,
                            properties={
                                "description": relation["description"],
                                "weight": relation["weight"],
                                "chunk_id": chunk.id,
                            },
                        )
                        relations.append(relation)
                
                elapsed = time.time() - start_time
                logging.info(
                    f"[Legacy-Relation] Finished extracting relations for chunk {chunk_key}, "
                    f"took {elapsed:.2f} seconds"
                )
                return relations
                
            except Exception as e:
                logging.exception(f"[Legacy-Relation] Extraction failed for chunk {chunk.id}")
                warnings.warn(f"Legacy relation extraction failed for chunk {chunk.id}: {e}")
                return []

        # Process chunks concurrently
        relation_extraction_concurrency: int = 5
        relation_coros = [
            _process_single_content_relation(
                chunk,
                {
                    e.page_content: e
                    for e in entities
                    if chunk.id in e.metadata.chunk_ids
                },
            )
            for chunk in chunks
        ]

        relations_list = await _limited_gather(
            relation_coros, relation_extraction_concurrency
        )

        # Flatten the list of relation lists into a single list of relations
        relations = [
            relation
            for relation_list in relations_list
            if relation_list
            for relation in relation_list
        ]
        
        return relations

    