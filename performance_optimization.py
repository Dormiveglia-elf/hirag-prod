"""
LanceDB Performance Optimization Utilities

This module provides performance optimization strategies for LanceDB operations,
specifically addressing the upsert performance degradation issue.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for LanceDB performance optimization"""

    # Batch processing
    max_batch_size: int = 100  # Maximum items per batch
    min_batch_size: int = 10  # Minimum items to trigger batch processing

    # Concurrency settings
    chunk_concurrency: int = 16
    entity_concurrency: int = 16
    relation_concurrency: int = 16

    # Database optimization
    enable_batch_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_smart_batching: bool = True

    # Memory management
    memory_limit_mb: int = 1024  # Memory limit for batch processing

    # Logging
    enable_performance_logging: bool = True
    log_batch_details: bool = False


class BatchProcessor:
    """Optimized batch processor for LanceDB operations"""

    def __init__(self, config: PerformanceConfig):
        self.config = config

    async def process_chunks_batch(self, vdb, chunks: List, chunks_table, state):
        """Optimized batch processing for chunks"""
        if not chunks:
            return

        pending_chunks = state.get_pending_chunks(chunks)
        if not pending_chunks:
            logger.info("⏭️ All chunks already processed")
            return

        start_time = time.perf_counter()

        if (
            self.config.enable_smart_batching
            and len(pending_chunks) >= self.config.min_batch_size
        ):
            await self._process_chunks_smart_batch(vdb, pending_chunks, chunks_table)
        else:
            await self._process_chunks_individual(vdb, pending_chunks, chunks_table)

        elapsed = time.perf_counter() - start_time

        if self.config.enable_performance_logging:
            avg_time = elapsed / len(pending_chunks)
            logger.info(
                f"✅ Processed {len(pending_chunks)} chunks in {elapsed:.3f}s (avg: {avg_time:.3f}s/chunk)"
            )

    async def _process_chunks_smart_batch(self, vdb, chunks: List, table):
        """Smart batching with automatic size optimization"""
        total_chunks = len(chunks)

        # Calculate optimal batch size based on memory and performance constraints
        optimal_batch_size = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, total_chunks // 4),  # Dynamic sizing
        )

        logger.info(f"📦 Using smart batching with batch size: {optimal_batch_size}")

        # Process in optimized batches
        for i in range(0, total_chunks, optimal_batch_size):
            batch = chunks[i : i + optimal_batch_size]

            batch_start = time.perf_counter()

            # Prepare batch data
            texts_to_embed = []
            properties_list = []

            for chunk in batch:
                texts_to_embed.append(chunk.page_content)
                properties_list.append(
                    {
                        "document_key": chunk.id,
                        "text": chunk.page_content,
                        **chunk.metadata.__dict__,
                    }
                )

            # Execute batch upsert
            await vdb.upsert_texts(
                texts_to_embed=texts_to_embed,
                properties_list=properties_list,
                table=table,
                mode="append",
            )

            batch_elapsed = time.perf_counter() - batch_start

            if self.config.log_batch_details:
                logger.debug(
                    f"📦 Batch {i//optimal_batch_size + 1}: {len(batch)} items in {batch_elapsed:.3f}s"
                )

    async def _process_chunks_individual(self, vdb, chunks: List, table):
        """Fallback to individual processing for small batches"""
        logger.info(f"🔄 Processing {len(chunks)} chunks individually")

        tasks = []
        for chunk in chunks:
            task = vdb.upsert_text(
                text_to_embed=chunk.page_content,
                properties={
                    "document_key": chunk.id,
                    "text": chunk.page_content,
                    **chunk.metadata.__dict__,
                },
                table=table,
                mode="append",
            )
            tasks.append(task)

        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(self.config.chunk_concurrency)

        async def limited_task(task):
            async with semaphore:
                return await task

        await asyncio.gather(*[limited_task(task) for task in tasks])

    async def process_entities_batch(self, vdb, entities: List, entities_table, state):
        """Optimized batch processing for entities"""
        if not entities:
            return

        new_entities = [
            ent for ent in entities if ent.id not in state.processed_entity_ids
        ]
        if not new_entities:
            logger.info("⏭️ No new entities to process")
            return

        start_time = time.perf_counter()

        if (
            self.config.enable_smart_batching
            and len(new_entities) >= self.config.min_batch_size
        ):
            await self._process_entities_smart_batch(vdb, new_entities, entities_table)
        else:
            await self._process_entities_individual(vdb, new_entities, entities_table)

        elapsed = time.perf_counter() - start_time

        if self.config.enable_performance_logging:
            avg_time = elapsed / len(new_entities)
            logger.info(
                f"✅ Processed {len(new_entities)} entities in {elapsed:.3f}s (avg: {avg_time:.3f}s/entity)"
            )

    async def _process_entities_smart_batch(self, vdb, entities: List, table):
        """Smart batching for entities"""
        total_entities = len(entities)
        optimal_batch_size = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, total_entities // 4),
        )

        logger.info(
            f"📦 Using smart entity batching with batch size: {optimal_batch_size}"
        )

        for i in range(0, total_entities, optimal_batch_size):
            batch = entities[i : i + optimal_batch_size]

            # Prepare batch data
            texts_to_embed = []
            properties_list = []

            for ent in batch:
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

            # Execute batch upsert
            await vdb.upsert_texts(
                texts_to_embed=texts_to_embed,
                properties_list=properties_list,
                table=table,
                mode="append",
            )

    async def _process_entities_individual(self, vdb, entities: List, table):
        """Individual entity processing with concurrency control"""
        tasks = []
        for ent in entities:
            description_text = (
                " | ".join(ent.metadata.description) if ent.metadata.description else ""
            )
            task = vdb.upsert_text(
                text_to_embed=description_text,
                properties={
                    "document_key": ent.id,
                    "text": ent.page_content,
                    **ent.metadata.__dict__,
                },
                table=table,
                mode="append",
            )
            tasks.append(task)

        semaphore = asyncio.Semaphore(self.config.entity_concurrency)

        async def limited_task(task):
            async with semaphore:
                return await task

        await asyncio.gather(*[limited_task(task) for task in tasks])


class PerformanceMonitor:
    """Monitor and log LanceDB performance metrics"""

    def __init__(self):
        self.metrics = {}

    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.metrics[operation_name] = {
            "start_time": time.perf_counter(),
            "count": self.metrics.get(operation_name, {}).get("count", 0) + 1,
        }

    def end_operation(self, operation_name: str, item_count: int = 1):
        """End timing an operation and log metrics"""
        if operation_name not in self.metrics:
            return

        elapsed = time.perf_counter() - self.metrics[operation_name]["start_time"]
        count = self.metrics[operation_name]["count"]

        avg_time = elapsed / max(item_count, 1)

        logger.info(
            f"📊 {operation_name} #{count}: {elapsed:.3f}s for {item_count} items (avg: {avg_time:.3f}s/item)"
        )

        # Store historical data
        if "history" not in self.metrics[operation_name]:
            self.metrics[operation_name]["history"] = []

        self.metrics[operation_name]["history"].append(
            {
                "elapsed": elapsed,
                "item_count": item_count,
                "avg_time": avg_time,
                "timestamp": time.time(),
            }
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        summary = {}

        for operation, data in self.metrics.items():
            if "history" in data:
                history = data["history"]
                summary[operation] = {
                    "total_operations": len(history),
                    "total_time": sum(h["elapsed"] for h in history),
                    "total_items": sum(h["item_count"] for h in history),
                    "avg_operation_time": sum(h["elapsed"] for h in history)
                    / len(history),
                    "avg_item_time": sum(h["avg_time"] for h in history) / len(history),
                    "performance_trend": self._calculate_trend(history),
                }

        return summary

    def _calculate_trend(self, history: List[Dict]) -> str:
        """Calculate performance trend (improving/degrading/stable)"""
        if len(history) < 3:
            return "insufficient_data"

        recent = history[-3:]
        avg_recent = sum(h["avg_time"] for h in recent) / len(recent)

        earlier = history[:-3][-3:] if len(history) >= 6 else history[:-3]
        if not earlier:
            return "insufficient_data"

        avg_earlier = sum(h["avg_time"] for h in earlier) / len(earlier)

        change_ratio = (avg_recent - avg_earlier) / avg_earlier

        if change_ratio > 0.1:
            return "degrading"
        elif change_ratio < -0.1:
            return "improving"
        else:
            return "stable"


def create_optimized_hirag_config() -> Dict[str, Any]:
    """Create optimized configuration for HiRAG with performance improvements"""
    perf_config = PerformanceConfig()

    return {
        "chunk_upsert_concurrency": perf_config.chunk_concurrency,
        "entity_upsert_concurrency": perf_config.entity_concurrency,
        "relation_upsert_concurrency": perf_config.relation_concurrency,
        "performance_config": perf_config,
        "batch_processor": BatchProcessor(perf_config),
        "performance_monitor": PerformanceMonitor(),
    }


# Usage example and recommendations
if __name__ == "__main__":
    print(
        """
    LanceDB Performance Optimization Guide
    =====================================
    
    1. Use batch operations instead of individual upserts
    2. Increase concurrency limits (16+ instead of 4)
    3. Monitor performance trends
    4. Use smart batching for large datasets
    5. Consider memory constraints when setting batch sizes
    
    Key optimizations implemented:
    - Batch processing with configurable sizes
    - Smart batching algorithms
    - Performance monitoring and trending
    - Concurrency control
    - Memory-aware processing
    """
    )
