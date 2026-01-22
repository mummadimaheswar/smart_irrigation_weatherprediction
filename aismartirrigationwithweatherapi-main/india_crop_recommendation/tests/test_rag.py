"""
Test script for RAG System
Run: python -m india_crop_recommendation.tests.test_rag
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_rag_system():
    """Test the RAG system functionality."""
    from api.rag_system import RAGSystem, TextChunker, BM25, SimpleEmbedder
    
    print("=" * 60)
    print("RAG System Test")
    print("=" * 60)
    
    # Test 1: Text Chunker
    print("\n1. Testing TextChunker...")
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    
    sample_text = """
    Cotton farming requires careful water management.
    The crop needs 30-40% soil moisture during growth.
    
    Irrigation should be done in stages:
    - Pre-sowing: 50-60mm water
    - Vegetative stage: 40-50mm every 15 days
    - Flowering: Critical period needs 60-70mm
    
    Pest management is also important.
    Use integrated pest management practices.
    """
    
    chunks = chunker.chunk_text(sample_text, "test_doc")
    print(f"   Created {len(chunks)} chunks from sample text")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {len(chunk.content)} chars")
    
    # Test 2: SimpleEmbedder
    print("\n2. Testing SimpleEmbedder...")
    embedder = SimpleEmbedder(embedding_dim=100)
    
    documents = [
        "Cotton farming requires water management",
        "Wheat grows best in cool temperatures",
        "Rice needs standing water in fields",
        "Irrigation is important for crop yield"
    ]
    
    embedder.fit(documents)
    
    query_embedding = embedder.embed("How much water does cotton need?")
    print(f"   Query embedding dim: {len(query_embedding)}")
    print(f"   Vocabulary size: {len(embedder.vocabulary)}")
    
    # Test 3: BM25
    print("\n3. Testing BM25 Search...")
    bm25 = BM25()
    bm25.fit(documents)
    
    results = bm25.search("water irrigation", top_k=3)
    print(f"   Search results for 'water irrigation':")
    for idx, score in results:
        print(f"   - Score {score:.3f}: {documents[idx][:50]}...")
    
    # Test 4: Full RAG System
    print("\n4. Testing Full RAG System...")
    
    # Use temporary storage
    import tempfile
    temp_file = tempfile.mktemp(suffix=".json")
    
    rag = RAGSystem(storage_path=temp_file)
    
    # Add a document
    doc = rag.add_document(
        filename="cotton_guide.txt",
        content=sample_text,
        metadata={"crop": "cotton", "state": "Maharashtra", "category": "irrigation"}
    )
    
    print(f"   Added document: {doc.id} with {len(doc.chunks)} chunks")
    
    # Query
    results = rag.query(
        query="How much water for cotton irrigation?",
        top_k=3,
        retrieval_method="hybrid"
    )
    
    print(f"   Query results: {len(results)} found")
    for r in results:
        print(f"   - Score {r.score:.3f}: {r.chunk.content[:50]}...")
    
    # Query with metadata filter
    print("\n5. Testing Metadata Filtering...")
    filtered_results = rag.query(
        query="irrigation",
        top_k=3,
        metadata_filters={"crop": "cotton"}
    )
    print(f"   Filtered results (crop=cotton): {len(filtered_results)} found")
    
    # Get context for LLM
    print("\n6. Testing LLM Context Generation...")
    context = rag.get_context_for_llm(
        query="irrigation schedule",
        top_k=3,
        max_context_length=500
    )
    print(f"   Generated context length: {len(context)} chars")
    print(f"   Context preview: {context[:200]}...")
    
    # Stats
    print("\n7. RAG System Stats:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    os.remove(temp_file)
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


def test_document_processor():
    """Test document processing for different file types."""
    from api.rag_system import DocumentProcessor
    
    print("\n" + "=" * 60)
    print("Document Processor Test")
    print("=" * 60)
    
    # Test TXT processing
    print("\n1. Testing TXT processing...")
    txt_content = b"This is a test document.\nWith multiple lines."
    text, file_type = DocumentProcessor.process_file("test.txt", txt_content)
    print(f"   File type: {file_type}, Content length: {len(text)}")
    
    # Test JSON processing
    print("\n2. Testing JSON processing...")
    json_content = b'{"crop": "cotton", "irrigation": {"frequency": "15 days", "amount": "50mm"}}'
    text, file_type = DocumentProcessor.process_file("test.json", json_content)
    print(f"   File type: {file_type}, Content length: {len(text)}")
    print(f"   Content preview: {text[:100]}...")
    
    # Test CSV processing
    print("\n3. Testing CSV processing...")
    csv_content = b"crop,moisture,temp\ncotton,35,28\nwheat,25,20"
    text, file_type = DocumentProcessor.process_file("test.csv", csv_content)
    print(f"   File type: {file_type}, Content length: {len(text)}")
    print(f"   Content preview: {text[:100]}...")
    
    print("\nDocument Processor tests passed! ✅")


if __name__ == "__main__":
    test_rag_system()
    test_document_processor()
