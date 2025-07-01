# Knowledge Graph Extraction Optimization

## Problem Fixed

**Issue**: Redundant knowledge graph extraction causing ~2x processing time
- The system was extracting entities and relations twice for the same chunks
- First call: `entity_extractor.entity()` - extracted both entities and relations
- Second call: `entity_extractor.relation()` - re-extracted the same data

## Solution Implemented

### 1. Enhanced VanillaEntity Class (`src/hirag_prod/entity/vanilla.py`)

**Key Improvements**:
- **Unified Extraction**: Single `extract_knowledge_graph()` method for both entities and relations
- **Smart Caching**: Results cached to avoid re-computation within same processing session
- **Optimized Methods**: `entity()` and `relation()` now use cached results efficiently

**New Features**:
```python
# Cache management
_cached_kg_results: Dict[str, Tuple[List[Entity], List[Relation]]]
_cache_enabled: bool = True

# Efficient extraction
async def extract_knowledge_graph(chunks, use_cache=True) -> Tuple[entities, relations]
async def entity(chunks) -> List[Entity]  # Uses cache
async def relation(chunks, entities=None) -> List[Relation]  # Uses cache

# Cache control
def clear_cache()
def disable_cache()
def enable_cache()
```

### 2. Optimized HiRAG Processing (`src/hirag_prod/hirag.py`)

**Process Flow Before**:
```
Step 2: Extract entities (entities + relations extracted)
Step 4: Extract relations (entities + relations extracted AGAIN)
```

**Process Flow After**:
```
Step 2: Unified knowledge graph extraction (entities + relations extracted ONCE)
Step 3: Process entities
Step 4: Process relations (from cached results)
```

**Key Changes**:
- Replaced separate `entity()` and `relation()` calls with single `extract_knowledge_graph()` call
- Added cache clearing at document start to ensure fresh processing
- Improved logging to reflect unified approach

## Performance Benefits

### Time Savings
- **~50% reduction** in knowledge graph extraction time
- **Eliminated redundant LLM calls** for the same chunks
- **Faster processing** for large documents

### Resource Efficiency
- **Reduced API costs** (fewer LLM requests)
- **Lower memory usage** (cached results shared between steps)
- **Better concurrency** (single extraction phase)

### Improved Reliability
- **Consistent results** between entity and relation extraction
- **Better error handling** with unified processing
- **Cleaner resume functionality** with proper cache management

## Implementation Details

### Cache Strategy
- **Session-scoped**: Cache cleared between documents
- **Chunk-based keys**: `"|".join(sorted([chunk.id for chunk in chunks]))`
- **Memory efficient**: Only stores results for current processing session

### Backward Compatibility
- **Legacy methods preserved**: Old `legacy_entity_extraction()` and `legacy_relation_extraction()` kept
- **API unchanged**: Public interface remains the same
- **Optional entities parameter**: `relation()` method still accepts entities for validation

### Error Handling
- **Graceful fallbacks**: Cache misses handled transparently
- **Exception safety**: Processing continues if cache operations fail
- **Logging improvements**: Clear indication of cache usage and optimization benefits

## Usage Example

```python
# The API remains the same, but now optimized internally
entities = await entity_extractor.entity(chunks)  # Extracts and caches
relations = await entity_extractor.relation(chunks)  # Uses cached results

# Or use the unified method directly
entities, relations = await entity_extractor.extract_knowledge_graph(chunks)
```

## Verification

To verify the fix works:
1. Check logs for single knowledge graph extraction per chunk
2. Look for `[KG-Cache]` log entries indicating cache usage
3. Monitor total processing time reduction
4. Confirm no duplicate entity/relation extraction messages

The system now processes documents efficiently with a single pass for knowledge graph extraction, eliminating the previous redundant processing issue. 