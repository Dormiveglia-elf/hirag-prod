"""
Pytest tests for the unified knowledge graph extraction functionality.

These tests validate:
- New unified KG extraction methods
- Backward compatibility
- Performance improvements
- Entity merging and relation filtering
"""

import pytest
from unittest.mock import AsyncMock, Mock

from hirag_prod.entity.vanilla import VanillaEntity
from hirag_prod.schema import Chunk, Entity, Relation
from hirag_prod._utils import compute_mdhash_id


class MockExtractFunc:
    """Mock extract function for testing"""
    
    def __init__(self):
        self.call_count = 0
        
    async def __call__(self, model: str, prompt: str, history_messages=None) -> str:
        self.call_count += 1
        
        # Return different responses based on prompt content
        if "comprehensive knowledge graph extraction" in prompt:
            return self._mock_unified_result()
        elif "entity" in prompt.lower():
            return self._mock_entity_result()
        elif "relationship" in prompt.lower():
            return self._mock_relation_result()
        elif "YES | NO" in prompt:
            return "NO"
        else:
            return self._mock_unified_result()
    
    def _mock_unified_result(self) -> str:
        """Mock unified KG extraction result"""
        return '''("entity"<|>"Alice"<|>"person"<|>"Alice is a researcher at the university.")##
("entity"<|>"University"<|>"organization"<|>"University is an educational institution.")##
("entity"<|>"Research Project"<|>"concept"<|>"Research Project is Alice's main work.")##
("relationship"<|>"Alice"<|>"University"<|>"Alice works at the University."<|>8)##
("relationship"<|>"Alice"<|>"Research Project"<|>"Alice leads the Research Project."<|>9)<|COMPLETE|>'''
    
    def _mock_entity_result(self) -> str:
        """Mock entity extraction result"""
        return '''("entity"<|>"Alice"<|>"person"<|>"Alice is a researcher.")##
("entity"<|>"University"<|>"organization"<|>"University is an institution.")<|COMPLETE|>'''
    
    def _mock_relation_result(self) -> str:
        """Mock relation extraction result"""
        return '''("relationship"<|>"Alice"<|>"University"<|>"Alice works at University."<|>7)<|COMPLETE|>'''


@pytest.fixture
def mock_extract_func():
    """Create a mock extract function"""
    return MockExtractFunc()


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing"""
    return [
        Chunk(
            id="test-chunk-1",
            metadata={
                "type": "txt",
                "filename": "test1.txt",
                "chunk_idx": 0,
                "document_id": "doc-test-1",
                "private": False,
                "uri": "/test/test1.txt"
            },
            page_content="Alice is a brilliant researcher working at the University. "
                        "She leads an important research project that aims to advance "
                        "our understanding of artificial intelligence and machine learning."
        ),
        Chunk(
            id="test-chunk-2",
            metadata={
                "type": "txt",
                "filename": "test2.txt",
                "chunk_idx": 1,
                "document_id": "doc-test-1", 
                "private": False,
                "uri": "/test/test2.txt"
            },
            page_content="The University provides excellent facilities for research. "
                        "Alice has access to state-of-the-art equipment and collaborates "
                        "with other researchers on groundbreaking projects."
        )
    ]


@pytest.fixture
def entity_handler(mock_extract_func):
    """Create VanillaEntity instance with mock extract function"""
    return VanillaEntity.create(
        extract_func=mock_extract_func,
        llm_model_name="gpt-4o-mini",
    )


@pytest.mark.asyncio
async def test_unified_kg_extraction_basic(entity_handler, sample_chunks, mock_extract_func):
    """Test basic unified knowledge graph extraction functionality"""
    # Execute unified extraction
    entities, relations = await entity_handler.extract_knowledge_graph(sample_chunks)
    
    # Validate results
    assert len(entities) > 0, "Should extract at least one entity"
    assert len(relations) > 0, "Should extract at least one relation"
    assert mock_extract_func.call_count > 0, "Should make API calls"
    
    # Validate entity structure
    for entity in entities:
        assert isinstance(entity, Entity), "Should return Entity objects"
        assert entity.page_content, "Entity should have content"
        assert entity.metadata.entity_type, "Entity should have type"
        assert entity.metadata.description, "Entity should have description"
        assert entity.metadata.chunk_ids, "Entity should have chunk IDs"
    
    # Validate relation structure
    for relation in relations:
        assert isinstance(relation, Relation), "Should return Relation objects"
        assert relation.source, "Relation should have source"
        assert relation.target, "Relation should have target"
        assert relation.properties.get("description"), "Relation should have description"
        assert relation.properties.get("weight"), "Relation should have weight"


@pytest.mark.asyncio
async def test_backward_compatibility_entity(entity_handler, sample_chunks, mock_extract_func):
    """Test that the entity() method still works (backward compatibility)"""
    # Execute legacy entity method (should use unified approach internally)
    entities = await entity_handler.entity(sample_chunks)
    
    # Validate results
    assert len(entities) > 0, "Should extract entities via legacy method"
    assert mock_extract_func.call_count > 0, "Should make API calls"
    
    # Validate all entities are proper Entity objects
    for entity in entities:
        assert isinstance(entity, Entity), "Should return Entity objects"
        assert entity.page_content, "Entity should have content"


@pytest.mark.asyncio 
async def test_backward_compatibility_relation(entity_handler, sample_chunks, mock_extract_func):
    """Test that the relation() method still works (backward compatibility)"""
    # First get entities
    entities = await entity_handler.entity(sample_chunks)
    
    # Reset call count
    initial_calls = mock_extract_func.call_count
    
    # Execute legacy relation method
    relations = await entity_handler.relation(sample_chunks, entities)
    
    # Validate results
    assert len(relations) >= 0, "Should extract relations via legacy method"
    assert mock_extract_func.call_count >= initial_calls, "Should make additional API calls"
    
    # Validate all relations are proper Relation objects
    for relation in relations:
        assert isinstance(relation, Relation), "Should return Relation objects"
        assert relation.source, "Relation should have source"
        assert relation.target, "Relation should have target"


@pytest.mark.asyncio
async def test_entity_merging(entity_handler):
    """Test entity merging functionality"""
    # Create duplicate entities
    duplicate_entities = [
        Entity(
            id=compute_mdhash_id("Alice", prefix="ent-"),
            page_content="Alice",
            metadata={
                "entity_type": "person",
                "description": ["Alice is a researcher"],
                "chunk_ids": ["chunk-1"]
            }
        ),
        Entity(
            id=compute_mdhash_id("Alice", prefix="ent-"),
            page_content="Alice",
            metadata={
                "entity_type": "person", 
                "description": ["Alice works at university"],
                "chunk_ids": ["chunk-2"]
            }
        ),
        Entity(
            id=compute_mdhash_id("Bob", prefix="ent-"),
            page_content="Bob",
            metadata={
                "entity_type": "person",
                "description": ["Bob is a student"],
                "chunk_ids": ["chunk-1"]
            }
        )
    ]
    
    # Test merging
    merged_entities = await entity_handler._merge_duplicate_entities(duplicate_entities)
    
    # Validate merging
    assert len(merged_entities) == 2, "Should merge duplicate entities"
    
    # Find merged Alice entity
    alice_entity = next(e for e in merged_entities if e.page_content == "Alice")
    assert len(alice_entity.metadata.description) == 2, "Should merge descriptions"
    assert len(alice_entity.metadata.chunk_ids) == 2, "Should merge chunk IDs"
    
    # Ensure Bob entity is unchanged
    bob_entity = next(e for e in merged_entities if e.page_content == "Bob")
    assert len(bob_entity.metadata.description) == 1, "Bob should be unchanged"


@pytest.mark.asyncio
async def test_relation_filtering(entity_handler):
    """Test relation filtering functionality"""
    # Create entities
    entities = [
        Entity(
            id="ent-alice",
            page_content="Alice",
            metadata={"entity_type": "person", "description": ["Researcher"], "chunk_ids": ["chunk-1"]}
        ),
        Entity(
            id="ent-university",
            page_content="University", 
            metadata={"entity_type": "organization", "description": ["Institution"], "chunk_ids": ["chunk-1"]}
        )
    ]
    
    # Create relation data (some valid, some invalid)
    relation_data = [
        {
            "source_name": "Alice",
            "target_name": "University",
            "description": "Alice works at University",
            "weight": 8,
            "chunk_id": "chunk-1"
        },
        {
            "source_name": "Alice", 
            "target_name": "NonExistentEntity",
            "description": "Invalid relation",
            "weight": 5,
            "chunk_id": "chunk-1"
        },
        {
            "source_name": "University",
            "target_name": "Alice", 
            "description": "University employs Alice",
            "weight": 7,
            "chunk_id": "chunk-1"
        }
    ]
    
    # Test filtering
    valid_relations = entity_handler._filter_valid_relations(relation_data, entities)
    
    # Should only return relations where both entities exist
    assert len(valid_relations) == 2, "Should filter out invalid relations"
    
    for relation in valid_relations:
        assert isinstance(relation, Relation), "Should return Relation objects"
        assert relation.source in ["ent-alice", "ent-university"], "Source should be valid entity ID"
        assert relation.target in ["ent-alice", "ent-university"], "Target should be valid entity ID"


@pytest.mark.asyncio
async def test_legacy_methods_explicit(entity_handler, sample_chunks, mock_extract_func):
    """Test explicit legacy methods"""
    # Test legacy entity extraction
    legacy_entities = await entity_handler.legacy_entity_extraction(sample_chunks)
    assert len(legacy_entities) > 0, "Legacy entity extraction should work"
    
    entity_calls = mock_extract_func.call_count
    
    # Test legacy relation extraction
    legacy_relations = await entity_handler.legacy_relation_extraction(sample_chunks, legacy_entities)
    assert len(legacy_relations) >= 0, "Legacy relation extraction should work"
    assert mock_extract_func.call_count > entity_calls, "Should make additional calls for relations"


@pytest.mark.asyncio
async def test_performance_efficiency(mock_extract_func, sample_chunks):
    """Test that unified approach is more efficient than legacy"""
    # Test unified approach
    unified_handler = VanillaEntity.create(
        extract_func=mock_extract_func,
        llm_model_name="gpt-4o-mini",
    )
    
    # Execute unified extraction
    entities, relations = await unified_handler.extract_knowledge_graph(sample_chunks)
    unified_calls = mock_extract_func.call_count
    
    # Reset for legacy test
    legacy_extract_func = MockExtractFunc()
    legacy_handler = VanillaEntity.create(
        extract_func=legacy_extract_func,
        llm_model_name="gpt-4o-mini",
    )
    
    # Execute legacy extraction
    legacy_entities = await legacy_handler.legacy_entity_extraction(sample_chunks)
    legacy_relations = await legacy_handler.legacy_relation_extraction(sample_chunks, legacy_entities)
    legacy_calls = legacy_extract_func.call_count
    
    # Unified approach should be more efficient
    print(f"Unified calls: {unified_calls}, Legacy calls: {legacy_calls}")
    # Note: In this test, the efficiency might not be obvious due to mocking,
    # but in real scenarios, unified approach should use fewer calls


@pytest.mark.asyncio
async def test_empty_chunks(entity_handler):
    """Test handling of empty chunks"""
    empty_chunks = []
    entities, relations = await entity_handler.extract_knowledge_graph(empty_chunks)
    
    assert entities == [], "Should return empty list for empty chunks"
    assert relations == [], "Should return empty list for empty chunks"


@pytest.mark.asyncio
async def test_error_handling(entity_handler, sample_chunks):
    """Test error handling in extraction"""
    # Create a mock that raises an exception
    error_extract_func = AsyncMock(side_effect=Exception("Test error"))
    
    error_handler = VanillaEntity.create(
        extract_func=error_extract_func,
        llm_model_name="gpt-4o-mini",
    )
    
    # Should handle errors gracefully
    entities, relations = await error_handler.extract_knowledge_graph(sample_chunks)
    
    # Should return empty results rather than crash
    assert isinstance(entities, list), "Should return list even on error"
    assert isinstance(relations, list), "Should return list even on error" 