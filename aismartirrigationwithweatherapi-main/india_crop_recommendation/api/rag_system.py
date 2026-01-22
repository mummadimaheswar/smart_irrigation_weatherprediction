"""
RAG (Retrieval-Augmented Generation) System for Agricultural Knowledge
Combines semantic search, keyword search (BM25), and metadata filtering

Features:
- Document upload and processing (PDF, TXT, CSV, JSON)
- Text chunking with overlap
- Semantic search using sentence embeddings
- BM25 keyword search
- Metadata filtering (date, source, category, state, crop)
- Hybrid retrieval combining all methods
- LLM response generation with context

Usage:
    from api.rag_system import RAGSystem, DocumentUploader
    
    rag = RAGSystem()
    rag.add_document("agricultural_guide.pdf", metadata={"category": "irrigation"})
    results = rag.query("How to manage soil moisture in cotton fields?")
"""

import os
import re
import json
import uuid
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import Counter
import math

import numpy as np

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char
        }


@dataclass
class Document:
    """Represents an uploaded document."""
    id: str
    filename: str
    content: str
    file_type: str
    upload_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "upload_time": self.upload_time.isoformat(),
            "metadata": self.metadata,
            "num_chunks": len(self.chunks),
            "content_length": len(self.content)
        }


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    chunk: DocumentChunk
    score: float
    retrieval_method: str  # "semantic", "keyword", "hybrid"
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "content": self.chunk.content,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "metadata": self.chunk.metadata
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    """Chunk text into overlapping segments."""
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split by paragraphs first, then by sentences
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_id = f"{document_id}_chunk_{chunk_index}"
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    ))
                    chunk_index += 1
                    
                    # Handle overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk += para + "\n\n"
        
        # Add last chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunks.append(DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        return text
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING SYSTEM (Simple TF-IDF based for no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleEmbedder:
    """
    Simple embedding system using TF-IDF.
    For production, replace with sentence-transformers or OpenAI embeddings.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.is_fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'of',
                     'at', 'by', 'with', 'about', 'to', 'from', 'in', 'on', 'it',
                     'its', 'this', 'that', 'these', 'those'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def fit(self, documents: List[str]):
        """Build vocabulary and IDF from documents."""
        # Build vocabulary
        all_tokens = set()
        doc_freq = Counter()
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            all_tokens.update(tokens)
            for token in tokens:
                doc_freq[token] += 1
        
        # Create vocabulary (limit to top terms by frequency)
        sorted_tokens = sorted(all_tokens, key=lambda x: doc_freq[x], reverse=True)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted_tokens[:self.embedding_dim])}
        
        # Calculate IDF
        num_docs = len(documents)
        for token, freq in doc_freq.items():
            if token in self.vocabulary:
                self.idf[token] = math.log((num_docs + 1) / (freq + 1)) + 1
        
        self.is_fitted = True
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.is_fitted:
            # Return random-ish embedding based on text hash
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val % (2**32))
            return list(np.random.randn(self.embedding_dim).astype(float))
        
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        
        embedding = np.zeros(self.embedding_dim)
        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tfidf = (1 + math.log(count)) * self.idf.get(token, 1.0)
                embedding[idx] = tfidf
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return list(embedding.astype(float))
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 KEYWORD SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class BM25:
    """BM25 keyword search implementation."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.idf: Dict[str, float] = {}
        self.doc_freqs: Dict[str, int] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 2]
    
    def fit(self, documents: List[str]):
        """Fit BM25 on documents."""
        self.corpus = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # Calculate IDF
        n_docs = len(self.corpus)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query against a document."""
        query_terms = self._tokenize(query)
        doc = self.corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        doc_term_freqs = Counter(doc)
        
        for term in query_terms:
            if term not in doc_term_freqs:
                continue
            
            tf = doc_term_freqs[term]
            idf = self.idf.get(term, 0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for top-k documents matching the query."""
        scores = [(idx, self.score(query, idx)) for idx in range(len(self.corpus))]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataFilter:
    """Filter chunks based on metadata criteria."""
    
    @staticmethod
    def filter_chunks(
        chunks: List[DocumentChunk],
        filters: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Filter chunks based on metadata.
        
        Supported filters:
        - state: str or List[str]
        - crop: str or List[str]
        - category: str or List[str]
        - date_from: str (YYYY-MM-DD)
        - date_to: str (YYYY-MM-DD)
        - source: str
        - file_type: str
        """
        if not filters:
            return chunks
        
        filtered = []
        for chunk in chunks:
            if MetadataFilter._matches_filters(chunk.metadata, filters):
                filtered.append(chunk)
        
        return filtered
    
    @staticmethod
    def _matches_filters(metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key == "date_from":
                doc_date = metadata.get("date") or metadata.get("upload_date")
                if doc_date and doc_date < value:
                    return False
            elif key == "date_to":
                doc_date = metadata.get("date") or metadata.get("upload_date")
                if doc_date and doc_date > value:
                    return False
            elif key in metadata:
                meta_value = metadata[key]
                if isinstance(value, list):
                    if isinstance(meta_value, list):
                        if not any(v in meta_value for v in value):
                            return False
                    elif meta_value not in value:
                        return False
                else:
                    if isinstance(meta_value, list):
                        if value not in meta_value:
                            return False
                    elif meta_value != value:
                        return False
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOADER & PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """Process and extract text from various document formats."""
    
    SUPPORTED_FORMATS = {'.txt', '.json', '.csv', '.md', '.pdf'}
    
    @staticmethod
    def process_file(file_path: str, file_content: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Process a file and return its text content and file type.
        
        Returns:
            Tuple of (text_content, file_type)
        """
        path = Path(file_path)
        file_type = path.suffix.lower()
        
        if file_type not in DocumentProcessor.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_type}")
        
        if file_content is None:
            with open(file_path, 'rb') as f:
                file_content = f.read()
        
        if file_type == '.txt' or file_type == '.md':
            return file_content.decode('utf-8', errors='ignore'), file_type
        
        elif file_type == '.json':
            data = json.loads(file_content.decode('utf-8'))
            return DocumentProcessor._json_to_text(data), file_type
        
        elif file_type == '.csv':
            return DocumentProcessor._csv_to_text(file_content.decode('utf-8', errors='ignore')), file_type
        
        elif file_type == '.pdf':
            return DocumentProcessor._pdf_to_text(file_content), file_type
        
        else:
            return file_content.decode('utf-8', errors='ignore'), file_type
    
    @staticmethod
    def _json_to_text(data: Any, prefix: str = "") -> str:
        """Convert JSON data to readable text."""
        lines = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(DocumentProcessor._json_to_text(value, prefix + "  "))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}Item {i+1}:")
                    lines.append(DocumentProcessor._json_to_text(item, prefix + "  "))
                else:
                    lines.append(f"{prefix}- {item}")
        
        else:
            lines.append(f"{prefix}{data}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _csv_to_text(content: str) -> str:
        """Convert CSV to readable text."""
        lines = content.strip().split('\n')
        if not lines:
            return ""
        
        # Parse header
        header = lines[0].split(',')
        text_lines = [f"Headers: {', '.join(header)}", ""]
        
        # Parse rows
        for i, line in enumerate(lines[1:], 1):
            values = line.split(',')
            row_text = f"Row {i}: "
            row_parts = []
            for h, v in zip(header, values):
                row_parts.append(f"{h.strip()}={v.strip()}")
            row_text += ", ".join(row_parts)
            text_lines.append(row_text)
        
        return "\n".join(text_lines)
    
    @staticmethod
    def _pdf_to_text(content: bytes) -> str:
        """Extract text from PDF (simple implementation)."""
        try:
            # Try using PyPDF2 if available
            try:
                from PyPDF2 import PdfReader
                from io import BytesIO
                
                reader = PdfReader(BytesIO(content))
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
            except ImportError:
                pass
            
            # Fallback: try pdfplumber
            try:
                import pdfplumber
                from io import BytesIO
                
                with pdfplumber.open(BytesIO(content)) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                return "\n\n".join(text_parts)
            except ImportError:
                pass
            
            # Final fallback: basic text extraction
            text = content.decode('utf-8', errors='ignore')
            # Remove binary garbage
            text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
            return text
            
        except Exception as e:
            log.warning(f"PDF extraction error: {e}")
            return f"[PDF content - extraction failed: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE (In-Memory)
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """Simple in-memory vector store."""
    
    def __init__(self):
        self.chunks: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.documents: Dict[str, Document] = {}
    
    def add_chunk(self, chunk: DocumentChunk):
        """Add a chunk to the store."""
        self.chunks[chunk.id] = chunk
        if chunk.embedding:
            self.embeddings[chunk.id] = np.array(chunk.embedding)
    
    def add_document(self, document: Document):
        """Add a document and its chunks to the store."""
        self.documents[document.id] = document
        for chunk in document.chunks:
            self.add_chunk(chunk)
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks."""
        return list(self.chunks.values())
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        for chunk in doc.chunks:
            if chunk.id in self.chunks:
                del self.chunks[chunk.id]
            if chunk.id in self.embeddings:
                del self.embeddings[chunk.id]
        
        del self.documents[doc_id]
        return True
    
    def semantic_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar chunks using cosine similarity."""
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        results = []
        for chunk_id, embedding in self.embeddings.items():
            embed_norm = np.linalg.norm(embedding)
            if embed_norm == 0:
                continue
            
            similarity = np.dot(query_vec, embedding) / (query_norm * embed_norm)
            if similarity >= threshold:
                results.append((chunk_id, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def save(self, filepath: str):
        """Save vector store to file."""
        data = {
            "documents": {
                doc_id: {
                    **doc.to_dict(),
                    "content": doc.content,
                    "chunks": [c.to_dict() for c in doc.chunks]
                }
                for doc_id, doc in self.documents.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, filepath: str):
        """Load vector store from file."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for doc_id, doc_data in data.get("documents", {}).items():
            chunks = []
            for chunk_data in doc_data.get("chunks", []):
                chunk = DocumentChunk(
                    id=chunk_data["id"],
                    document_id=chunk_data["document_id"],
                    content=chunk_data["content"],
                    embedding=chunk_data.get("embedding"),
                    metadata=chunk_data.get("metadata", {}),
                    chunk_index=chunk_data.get("chunk_index", 0),
                    start_char=chunk_data.get("start_char", 0),
                    end_char=chunk_data.get("end_char", 0)
                )
                chunks.append(chunk)
            
            doc = Document(
                id=doc_id,
                filename=doc_data["filename"],
                content=doc_data.get("content", ""),
                file_type=doc_data["file_type"],
                upload_time=datetime.fromisoformat(doc_data["upload_time"]),
                metadata=doc_data.get("metadata", {}),
                chunks=chunks
            )
            self.add_document(doc)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class RAGSystem:
    """
    Complete RAG System with hybrid retrieval.
    
    Combines:
    - Semantic search (embeddings)
    - Keyword search (BM25)
    - Metadata filtering
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_dim: int = 384,
        storage_path: Optional[str] = None
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = SimpleEmbedder(embedding_dim=embedding_dim)
        self.bm25 = BM25()
        self.vector_store = VectorStore()
        self.storage_path = storage_path or "rag_data.json"
        
        # Load existing data
        if os.path.exists(self.storage_path):
            self.vector_store.load(self.storage_path)
            self._rebuild_indices()
    
    def _rebuild_indices(self):
        """Rebuild BM25 and embedding indices from stored chunks."""
        chunks = self.vector_store.get_all_chunks()
        if chunks:
            contents = [c.content for c in chunks]
            self.bm25.fit(contents)
            self.embedder.fit(contents)
            
            # Regenerate embeddings if missing
            for chunk in chunks:
                if not chunk.embedding:
                    chunk.embedding = self.embedder.embed(chunk.content)
                    self.vector_store.embeddings[chunk.id] = np.array(chunk.embedding)
    
    def add_document(
        self,
        filename: str,
        content: Optional[Union[str, bytes]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Add a document to the RAG system.
        
        Args:
            filename: Name or path of the file
            content: Raw content (str or bytes). If None, reads from filename path.
            metadata: Additional metadata (state, crop, category, etc.)
        
        Returns:
            Document object
        """
        # Generate document ID
        doc_id = str(uuid.uuid4())[:8]
        
        # Process file content
        if content is None:
            text_content, file_type = DocumentProcessor.process_file(filename)
        elif isinstance(content, bytes):
            text_content, file_type = DocumentProcessor.process_file(
                filename, file_content=content
            )
        else:
            text_content = content
            file_type = Path(filename).suffix.lower() or ".txt"
        
        # Create document
        doc_metadata = metadata or {}
        doc_metadata["upload_date"] = datetime.now().isoformat()[:10]
        
        document = Document(
            id=doc_id,
            filename=filename,
            content=text_content,
            file_type=file_type,
            upload_time=datetime.now(),
            metadata=doc_metadata
        )
        
        # Chunk document
        chunks = self.chunker.chunk_text(text_content, doc_id)
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata = {**doc_metadata, "filename": filename}
        
        document.chunks = chunks
        
        # Generate embeddings
        all_chunks = self.vector_store.get_all_chunks() + chunks
        all_contents = [c.content for c in all_chunks]
        
        # Refit embedder with new content
        self.embedder.fit(all_contents)
        
        # Generate embeddings for new chunks
        for chunk in chunks:
            chunk.embedding = self.embedder.embed(chunk.content)
        
        # Refit BM25
        self.bm25.fit(all_contents)
        
        # Add to vector store
        self.vector_store.add_document(document)
        
        # Save
        self.vector_store.save(self.storage_path)
        
        log.info(f"Added document: {filename} ({len(chunks)} chunks)")
        return document
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the system."""
        result = self.vector_store.delete_document(doc_id)
        if result:
            self._rebuild_indices()
            self.vector_store.save(self.storage_path)
        return result
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        retrieval_method: str = "hybrid",
        metadata_filters: Optional[Dict[str, Any]] = None,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Query the RAG system.
        
        Args:
            query: Search query
            top_k: Number of results to return
            retrieval_method: "semantic", "keyword", or "hybrid"
            metadata_filters: Filters to apply (state, crop, category, date_from, date_to)
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        
        Returns:
            List of RetrievalResult objects
        """
        chunks = self.vector_store.get_all_chunks()
        
        if not chunks:
            return []
        
        # Apply metadata filters first
        if metadata_filters:
            chunks = MetadataFilter.filter_chunks(chunks, metadata_filters)
            if not chunks:
                return []
        
        # Create chunk ID to index mapping for filtered chunks
        chunk_id_to_idx = {c.id: i for i, c in enumerate(chunks)}
        chunk_contents = [c.content for c in chunks]
        
        results = []
        
        if retrieval_method == "semantic" or retrieval_method == "hybrid":
            # Semantic search
            query_embedding = self.embedder.embed(query)
            semantic_results = self.vector_store.semantic_search(
                query_embedding, 
                top_k=top_k * 2
            )
            
            # Filter to only include chunks that passed metadata filter
            for chunk_id, score in semantic_results:
                if chunk_id in chunk_id_to_idx:
                    results.append({
                        "chunk_id": chunk_id,
                        "semantic_score": score,
                        "keyword_score": 0.0
                    })
        
        if retrieval_method == "keyword" or retrieval_method == "hybrid":
            # Keyword search (BM25)
            # Refit BM25 on filtered chunks
            temp_bm25 = BM25()
            temp_bm25.fit(chunk_contents)
            keyword_results = temp_bm25.search(query, top_k=top_k * 2)
            
            for idx, score in keyword_results:
                chunk = chunks[idx]
                
                # Find or create result entry
                existing = next(
                    (r for r in results if r["chunk_id"] == chunk.id), 
                    None
                )
                
                if existing:
                    existing["keyword_score"] = score
                else:
                    results.append({
                        "chunk_id": chunk.id,
                        "semantic_score": 0.0,
                        "keyword_score": score
                    })
        
        # Normalize and combine scores
        if results:
            max_semantic = max(r["semantic_score"] for r in results) or 1.0
            max_keyword = max(r["keyword_score"] for r in results) or 1.0
            
            for r in results:
                norm_semantic = r["semantic_score"] / max_semantic
                norm_keyword = r["keyword_score"] / max_keyword
                
                if retrieval_method == "hybrid":
                    r["final_score"] = (
                        semantic_weight * norm_semantic + 
                        keyword_weight * norm_keyword
                    )
                elif retrieval_method == "semantic":
                    r["final_score"] = norm_semantic
                else:
                    r["final_score"] = norm_keyword
        
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Convert to RetrievalResult objects
        final_results = []
        for r in results[:top_k]:
            chunk = self.vector_store.get_chunk(r["chunk_id"])
            if chunk:
                final_results.append(RetrievalResult(
                    chunk=chunk,
                    score=r["final_score"],
                    retrieval_method=retrieval_method
                ))
        
        return final_results
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        max_context_length: int = 4000
    ) -> str:
        """
        Get formatted context for LLM prompt.
        
        Returns a formatted string with retrieved context chunks.
        """
        results = self.query(
            query=query,
            top_k=top_k,
            retrieval_method="hybrid",
            metadata_filters=metadata_filters
        )
        
        if not results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            chunk_text = f"[Source {i+1}: {result.chunk.metadata.get('filename', 'unknown')}]\n{result.chunk.content}\n"
            
            if total_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n---\n".join(context_parts)
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the system."""
        return [doc.to_dict() for doc in self.vector_store.get_all_documents()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        docs = self.vector_store.get_all_documents()
        chunks = self.vector_store.get_all_chunks()
        
        return {
            "total_documents": len(docs),
            "total_chunks": len(chunks),
            "storage_path": self.storage_path,
            "embedding_dim": self.embedder.embedding_dim,
            "vocabulary_size": len(self.embedder.vocabulary),
            "file_types": list(set(d.file_type for d in docs))
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_rag_instance: Optional[RAGSystem] = None

def get_rag_system(storage_path: Optional[str] = None) -> RAGSystem:
    """Get or create the RAG system singleton."""
    global _rag_instance
    
    if _rag_instance is None:
        default_path = Path(__file__).parent.parent / "rag_data" / "rag_store.json"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        _rag_instance = RAGSystem(storage_path=storage_path or str(default_path))
    
    return _rag_instance


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test the RAG system
    rag = RAGSystem(storage_path="test_rag.json")
    
    # Add sample documents
    sample_doc = """
    # Cotton Farming Guide for Maharashtra
    
    ## Soil Requirements
    Cotton grows best in black cotton soil (regur) with good drainage.
    Ideal soil moisture should be between 30-40% during the growing season.
    
    ## Irrigation Schedule
    - Pre-sowing: 50-60mm
    - Vegetative stage: 40-50mm every 15 days
    - Flowering: 60-70mm every 10 days
    - Boll development: 50-60mm every 12 days
    
    ## Best Planting Time
    Kharif season (June-July) is ideal for cotton planting in Maharashtra.
    
    ## Pest Management
    Common pests include bollworm and whitefly. Use integrated pest management.
    """
    
    doc = rag.add_document(
        "cotton_guide.md",
        content=sample_doc,
        metadata={"crop": "cotton", "state": "Maharashtra", "category": "farming_guide"}
    )
    
    print(f"Added document: {doc.id} with {len(doc.chunks)} chunks")
    
    # Query
    results = rag.query(
        "How much water does cotton need?",
        top_k=3,
        retrieval_method="hybrid"
    )
    
    print(f"\nQuery results ({len(results)} found):")
    for r in results:
        print(f"  Score: {r.score:.3f} - {r.chunk.content[:100]}...")
    
    # With metadata filter
    filtered_results = rag.query(
        "irrigation schedule",
        metadata_filters={"crop": "cotton"},
        top_k=2
    )
    
    print(f"\nFiltered results ({len(filtered_results)} found):")
    for r in filtered_results:
        print(f"  Score: {r.score:.3f} - {r.chunk.content[:100]}...")
    
    print(f"\nStats: {rag.get_stats()}")
