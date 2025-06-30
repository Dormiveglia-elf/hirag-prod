#!/usr/bin/env python3
"""
Test script for HiRAG resume functionality

This script demonstrates how the new resume functionality works:
1. Load a document for the first time (full processing)
2. Simulate process kill by stopping
3. Load the same document again (should resume from existing data)
4. Show debugging information about resume state
"""

import asyncio
import logging
import os
import sys

# Add the src directory to path so we can import hirag_prod
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hirag_prod.hirag import HiRAG

# Configure logging to see resume information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


async def test_resume_functionality():
    """Test the resume functionality with a sample document"""

    print("🧪 Testing HiRAG Resume Functionality")
    print("=" * 50)

    # Create HiRAG instance
    hirag = await HiRAG.create()

    # Test document path (you'll need to provide an actual document)
    test_document = "path/to/your/test/document.pdf"  # Change this to actual document

    if not os.path.exists(test_document):
        print(f"❌ Test document not found: {test_document}")
        print("Please update the test_document path to point to an actual document")
        return

    try:
        print(f"\n📄 Processing document: {test_document}")

        # First run - should do full processing
        print("\n🔄 FIRST RUN (Full Processing)")
        print("-" * 30)
        await hirag.insert_to_kb(
            document_path=test_document,
            content_type="pdf",  # Adjust based on your document type
            with_graph=True,
        )

        print("\n🔍 DEBUGGING RESUME STATE AFTER FIRST RUN")
        print("-" * 45)
        debug_info = await hirag.debug_document_resume_state(
            document_path=test_document, content_type="pdf"
        )

        print(f"Document URI: {debug_info.get('document_uri')}")
        print(f"Document ID: {debug_info.get('document_id')}")
        print(f"Total chunks generated: {debug_info.get('total_chunks_generated')}")
        print(f"Chunks in DB: {debug_info.get('chunks_in_db')}")
        print(f"Entities extracted: {debug_info.get('total_entities')}")
        print(f"Relations found: {debug_info.get('total_relations')}")
        print(
            f"Processing complete: {debug_info.get('chunks_complete') and debug_info.get('entities_extracted')}"
        )

        print("\n🔄 SECOND RUN (Should Resume)")
        print("-" * 30)
        print(
            "This simulates restarting the process and running the same document again..."
        )

        # Second run - should resume and skip already processed data
        await hirag.insert_to_kb(
            document_path=test_document, content_type="pdf", with_graph=True
        )

        print("\n🔍 DEBUGGING RESUME STATE AFTER SECOND RUN")
        print("-" * 46)
        debug_info2 = await hirag.debug_document_resume_state(
            document_path=test_document, content_type="pdf"
        )

        print(f"Chunks still pending: {debug_info2.get('pending_chunks')}")
        print(f"Entity extraction pending: {debug_info2.get('pending_entity_chunks')}")
        print(f"Change detection: {debug_info2.get('change_detection')}")

        # Compare chunk IDs to verify they match
        gen_chunks_1 = set(
            debug_info.get("chunk_ids_sample", {}).get("generated_chunk_ids", [])
        )
        gen_chunks_2 = set(
            debug_info2.get("chunk_ids_sample", {}).get("generated_chunk_ids", [])
        )

        if gen_chunks_1 == gen_chunks_2:
            print("✅ Chunk IDs are consistent between runs")
        else:
            print("❌ Chunk IDs differ between runs - this may indicate a problem")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await hirag.clean_up()
        print("\n🧪 Testing completed")


async def test_with_sample_text():
    """Test resume functionality with a simple text document"""

    print("\n📝 Testing with sample text document")
    print("=" * 40)

    # Create a simple test file
    test_content = """
    This is a sample document for testing HiRAG resume functionality.
    
    The document contains multiple paragraphs to create several chunks.
    Each chunk will be processed separately and stored in the database.
    
    When the process is restarted, HiRAG should detect existing chunks
    and only process the parts that haven't been completed yet.
    
    This helps save time and resources when processing large documents
    or when the process is interrupted for any reason.
    
    The resume functionality is essential for production deployments
    where reliability and efficiency are important.
    """

    test_file = "test_sample.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    try:
        hirag = await HiRAG.create()

        print(f"\n🔄 FIRST RUN - Processing {test_file}")
        await hirag.insert_to_kb(
            document_path=test_file, content_type="txt", with_graph=True
        )

        print(f"\n🔄 SECOND RUN - Should resume from existing data")
        await hirag.insert_to_kb(
            document_path=test_file, content_type="txt", with_graph=True
        )

        # Show debug info
        debug_info = await hirag.debug_document_resume_state(
            document_path=test_file, content_type="txt"
        )

        print(f"\n📊 Final State:")
        print(
            f"  Chunks: {debug_info.get('chunks_in_db')}/{debug_info.get('total_chunks_generated')}"
        )
        print(f"  Entities: {debug_info.get('total_entities')}")
        print(f"  Pending chunks: {debug_info.get('pending_chunks')}")
        print(f"  Pending entity extraction: {debug_info.get('pending_entity_chunks')}")

        await hirag.clean_up()

    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    print("HiRAG Resume Functionality Test")
    print("Choose test mode:")
    print("1. Test with your own document (update test_document path)")
    print("2. Test with generated sample text")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(test_resume_functionality())
    elif choice == "2":
        asyncio.run(test_with_sample_text())
    else:
        print("Invalid choice. Running sample text test...")
        asyncio.run(test_with_sample_text())
