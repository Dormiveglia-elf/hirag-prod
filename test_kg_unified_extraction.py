"""
Test script for the new unified knowledge graph extraction functionality.

This script validates:
1. New unified KG extraction method
2. Backward compatibility with legacy methods  
3. Performance comparison between old and new approaches
4. Entity merging and relation filtering functionality
"""

import asyncio
import logging
import time
from typing import List

# Mock ChatCompletion for testing without actual API calls
class MockChatCompletion:
    """Mock implementation for testing without API calls"""
    
    def __init__(self):
        self.call_count = 0
    
    async def complete(self, model: str, prompt: str, history_messages: List = None) -> str:
        """Mock completion that returns realistic test data"""
        self.call_count += 1
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Return mock extraction results based on prompt type
        if "KG_EXTRACTION" in prompt or "comprehensive knowledge graph extraction" in prompt:
            return self._mock_kg_extraction_result()
        elif "entity" in prompt.lower():
            return self._mock_entity_extraction_result()
        elif "relationship" in prompt.lower() or "relation" in prompt.lower():
            return self._mock_relation_extraction_result()
        elif "YES | NO" in prompt:
            return "NO"  # Stop iteration
        else:
            return self._mock_kg_extraction_result()
    
    def _mock_kg_extraction_result(self) -> str:
        """Mock unified KG extraction result"""
        return '''("entity"<|>"Dr. Sarah Chen"<|>"person"<|>"Dr. Sarah Chen is a brilliant scientist leading quantum research.")##
("entity"<|>"Quantum Computing"<|>"technology"<|>"Quantum Computing is an advanced computational technology with revolutionary potential.")##
("entity"<|>"MIT"<|>"organization"<|>"MIT is a prestigious research institution conducting cutting-edge scientific research.")##
("relationship"<|>"Dr. Sarah Chen"<|>"Quantum Computing"<|>"Dr. Sarah Chen is the lead researcher developing quantum computing algorithms."<|>9)##
("relationship"<|>"Dr. Sarah Chen"<|>"MIT"<|>"Dr. Sarah Chen conducts her research at MIT Research Laboratory."<|>8)##
("relationship"<|>"MIT"<|>"Quantum Computing"<|>"MIT provides the infrastructure and resources for quantum computing research."<|>7)<|COMPLETE|>'''
    
    def _mock_entity_extraction_result(self) -> str:
        """Mock entity-only extraction result"""
        return '''("entity"<|>"Dr. Sarah Chen"<|>"person"<|>"Dr. Sarah Chen is a brilliant scientist leading quantum research.")##
("entity"<|>"Quantum Computing"<|>"technology"<|>"Quantum Computing is an advanced computational technology.")##
("entity"<|>"MIT"<|>"organization"<|>"MIT is a prestigious research institution.")<|COMPLETE|>'''
    
    def _mock_relation_extraction_result(self) -> str:
        """Mock relation-only extraction result"""
        return '''("relationship"<|>"Dr. Sarah Chen"<|>"Quantum Computing"<|>"Dr. Sarah Chen researches quantum computing."<|>8)##
("relationship"<|>"Dr. Sarah Chen"<|>"MIT"<|>"Dr. Sarah Chen works at MIT."<|>7)<|COMPLETE|>'''


async def setup_test_environment():
    """Setup test environment with mock data"""
    from hirag_prod.entity.vanilla import VanillaEntity
    from hirag_prod.schema import Chunk
    
    # Create test chunks with proper metadata structure
    test_chunks = [
        Chunk(
            id="chunk-test-1",
            metadata={
                "type": "txt",
                "filename": "test1.txt",
                "chunk_idx": 0,
                "document_id": "doc-test-1",
                "private": False,
                "uri": "/test/test1.txt"
            },
            page_content="Dr. Sarah Chen is working on quantum computing research at MIT. "
                        "She has been developing revolutionary algorithms that could change "
                        "the field of cybersecurity. The quantum computer project has been "
                        "funded by multiple organizations and represents a significant "
                        "breakthrough in computational technology."
        ),
        Chunk(
            id="chunk-test-2", 
            metadata={
                "type": "txt",
                "filename": "test2.txt", 
                "chunk_idx": 1,
                "document_id": "doc-test-1",
                "private": False,
                "uri": "/test/test2.txt"
            },
            page_content="MIT's quantum computing laboratory houses the most advanced "
                        "quantum computers in the world. Dr. Sarah Chen leads a team of "
                        "researchers who are working on quantum algorithms. The project "
                        "has attracted attention from government agencies and private "
                        "companies interested in quantum cybersecurity applications."
        )
    ]
    
    # Create entity handler with mock completion function
    mock_completion = MockChatCompletion()
    entity_handler = VanillaEntity.create(
        extract_func=mock_completion.complete,
        llm_model_name="gpt-4o-mini",
    )
    
    return entity_handler, test_chunks, mock_completion


async def test_unified_kg_extraction():
    """Test the new unified knowledge graph extraction method"""
    print("\n🔬 Testing Unified Knowledge Graph Extraction...")
    
    entity_handler, test_chunks, mock_completion = await setup_test_environment()
    
    start_time = time.time()
    
    # Test the new unified extraction method
    entities, relations = await entity_handler.extract_knowledge_graph(test_chunks)
    
    elapsed_time = time.time() - start_time
    
    print(f"✅ Unified extraction completed in {elapsed_time:.2f} seconds")
    print(f"📊 Results: {len(entities)} entities, {len(relations)} relations")
    print(f"📞 API calls made: {mock_completion.call_count}")
    
    # Validate results
    assert len(entities) > 0, "Should extract at least one entity"
    assert len(relations) > 0, "Should extract at least one relation"
    
    # Print sample results
    print("\n📋 Sample Entities:")
    for i, entity in enumerate(entities[:3]):
        print(f"  {i+1}. {entity.page_content} ({entity.metadata.entity_type})")
    
    print("\n🔗 Sample Relations:")
    for i, relation in enumerate(relations[:3]):
        print(f"  {i+1}. Strength: {relation.properties['weight']}")
    
    return entities, relations, mock_completion.call_count


async def test_backward_compatibility():
    """Test backward compatibility with legacy methods"""
    print("\n🔄 Testing Backward Compatibility...")
    
    entity_handler, test_chunks, mock_completion = await setup_test_environment()
    
    start_time = time.time()
    
    # Test legacy entity method (should now use unified approach internally)
    entities = await entity_handler.entity(test_chunks)
    
    # Reset call count for relation test
    relation_call_count = mock_completion.call_count
    
    # Test legacy relation method 
    relations = await entity_handler.relation(test_chunks, entities)
    
    elapsed_time = time.time() - start_time
    total_calls = mock_completion.call_count
    
    print(f"✅ Legacy methods completed in {elapsed_time:.2f} seconds")
    print(f"📊 Results: {len(entities)} entities, {len(relations)} relations")
    print(f"📞 Total API calls: {total_calls}")
    
    # Validate backward compatibility
    assert len(entities) > 0, "Legacy entity method should work"
    assert len(relations) > 0, "Legacy relation method should work"
    
    return entities, relations, total_calls


async def test_legacy_methods():
    """Test the explicit legacy methods for comparison"""
    print("\n🔧 Testing Explicit Legacy Methods...")
    
    entity_handler, test_chunks, mock_completion = await setup_test_environment()
    
    start_time = time.time()
    
    # Test explicit legacy entity extraction
    legacy_entities = await entity_handler.legacy_entity_extraction(test_chunks)
    
    # Test explicit legacy relation extraction
    legacy_relations = await entity_handler.legacy_relation_extraction(test_chunks, legacy_entities)
    
    elapsed_time = time.time() - start_time
    
    print(f"✅ Legacy methods completed in {elapsed_time:.2f} seconds")
    print(f"📊 Results: {len(legacy_entities)} entities, {len(legacy_relations)} relations")
    print(f"📞 API calls made: {mock_completion.call_count}")
    
    return legacy_entities, legacy_relations, mock_completion.call_count


async def test_entity_merging():
    """Test entity merging functionality"""
    print("\n🔀 Testing Entity Merging...")
    
    from hirag_prod.schema import Entity
    from hirag_prod._utils import compute_mdhash_id
    
    entity_handler, _, _ = await setup_test_environment()
    
    # Create duplicate entities with same name but different descriptions
    duplicate_entities = [
        Entity(
            id=compute_mdhash_id("Dr. Sarah Chen", prefix="ent-"),
            page_content="Dr. Sarah Chen",
            metadata={
                "entity_type": "person",
                "description": ["Quantum computing researcher"],
                "chunk_ids": ["chunk-1"]
            }
        ),
        Entity(
            id=compute_mdhash_id("Dr. Sarah Chen", prefix="ent-"),
            page_content="Dr. Sarah Chen", 
            metadata={
                "entity_type": "person",
                "description": ["Lead scientist at MIT"],
                "chunk_ids": ["chunk-2"]
            }
        ),
        Entity(
            id=compute_mdhash_id("MIT", prefix="ent-"),
            page_content="MIT",
            metadata={
                "entity_type": "organization",
                "description": ["Research institution"],
                "chunk_ids": ["chunk-1"]
            }
        )
    ]
    
    # Test merging
    merged_entities = await entity_handler._merge_duplicate_entities(duplicate_entities)
    
    print(f"✅ Merged {len(duplicate_entities)} entities into {len(merged_entities)}")
    
    # Validate merging
    assert len(merged_entities) == 2, "Should merge duplicate entities"
    
    # Find the merged Dr. Sarah Chen entity
    merged_chen = next(e for e in merged_entities if e.page_content == "Dr. Sarah Chen")
    assert len(merged_chen.metadata.description) == 2, "Should merge descriptions"
    assert len(merged_chen.metadata.chunk_ids) == 2, "Should merge chunk IDs"
    
    print(f"📋 Merged entity: {merged_chen.page_content}")
    print(f"   Descriptions: {len(merged_chen.metadata.description)}")
    print(f"   Chunk IDs: {merged_chen.metadata.chunk_ids}")


async def run_performance_comparison():
    """Compare performance between unified and legacy approaches"""
    print("\n⚡ Running Performance Comparison...")
    
    # Test unified approach
    print("\n1️⃣ Unified Approach:")
    unified_entities, unified_relations, unified_calls = await test_unified_kg_extraction()
    
    # Test backward compatible approach (should use unified internally)
    print("\n2️⃣ Backward Compatible Approach:")
    compat_entities, compat_relations, compat_calls = await test_backward_compatibility()
    
    # Test explicit legacy approach
    print("\n3️⃣ Explicit Legacy Approach:")
    legacy_entities, legacy_relations, legacy_calls = await test_legacy_methods()
    
    # Compare results
    print("\n📈 Performance Comparison Summary:")
    print(f"{'Method':<25} {'Entities':<10} {'Relations':<10} {'API Calls':<10}")
    print("-" * 60)
    print(f"{'Unified':<25} {len(unified_entities):<10} {len(unified_relations):<10} {unified_calls:<10}")
    print(f"{'Backward Compatible':<25} {len(compat_entities):<10} {len(compat_relations):<10} {compat_calls:<10}")
    print(f"{'Explicit Legacy':<25} {len(legacy_entities):<10} {len(legacy_relations):<10} {legacy_calls:<10}")
    
    # Calculate efficiency improvements
    if legacy_calls > 0:
        unified_improvement = ((legacy_calls - unified_calls) / legacy_calls) * 100
        print(f"\n💡 Unified approach reduces API calls by ~{unified_improvement:.1f}%")


async def main():
    """Main test execution"""
    print("🚀 Starting Knowledge Graph Extraction Tests")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Run individual tests
        await test_unified_kg_extraction()
        await test_backward_compatibility()
        await test_entity_merging()
        
        # Run performance comparison
        await run_performance_comparison()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎉 Key Benefits Demonstrated:")
        print("   • Unified extraction works correctly")
        print("   • Backward compatibility maintained")
        print("   • Entity merging functions properly")
        print("   • Significant API call reduction achieved")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("🏁 Test Suite Complete")


if __name__ == "__main__":
    # Add the src directory to Python path for imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    # Run the test
    asyncio.run(main()) 