"""
PDF Document Loader and Text Chunker.

This module handles:
1. Loading PDF documents from a directory
2. Extracting text content
3. Splitting text into chunks for embedding
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import fitz  # PyMuPDF

from .config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    metadata: Dict[str, str]
    doc_id: str


@dataclass
class PDFDocument:
    """Represents a loaded PDF document."""
    filename: str
    filepath: str
    full_text: str
    pages: List[str]
    metadata: Dict[str, str]


class PDFLoader:
    """
    Loads PDF documents from a directory.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_pdf(self, pdf_path: str) -> PDFDocument:
        """
        Load a single PDF file and extract text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFDocument with extracted text
        """
        path = Path(pdf_path)
        pages = []
        
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                pages.append(page_text)
        
        full_text = "\n\n".join(pages)
        
        # Extract metadata
        metadata = self._extract_metadata(full_text, path.name)
        
        return PDFDocument(
            filename=path.name,
            filepath=str(path.absolute()),
            full_text=full_text,
            pages=pages,
            metadata=metadata
        )
    
    def load_all_pdfs(self) -> List[PDFDocument]:
        """
        Load all PDF files from the data directory.
        
        Returns:
            List of PDFDocument objects
        """
        documents = []
        
        for pdf_file in self.data_dir.glob("*.pdf"):
            try:
                doc = self.load_pdf(str(pdf_file))
                documents.append(doc)
                print(f"Loaded: {pdf_file.name}")
            except Exception as e:
                print(f"Error loading {pdf_file.name}: {e}")
        
        return documents
    
    def _extract_metadata(self, text: str, filename: str) -> Dict[str, str]:
        """Extract metadata from document text using pattern matching."""
        metadata = {
            "filename": filename,
            "source": filename
        }
        
        # Detect authority and AD ID using regex patterns
        ad_id, authority = self._detect_ad_info(text, filename)
        
        if ad_id:
            metadata["ad_id"] = ad_id
        if authority:
            metadata["authority"] = authority
        
        return metadata
    
    def _detect_ad_info(self, text: str, filename: str) -> tuple:
        """
        Detect AD ID and authority using regex patterns.
        
        Supports:
        - FAA ADs: AD YYYY-NN-NN, US-YYYY-NN-NN
        - EASA ADs: AD YYYY-NNNN, YYYY-NNNNRx
        - Other authorities can be added
        
        Returns:
            Tuple of (ad_id, authority)
        """
        combined = f"{filename} {text[:5000]}"  # Check filename and first part of text
        
        # Pattern definitions for different authorities
        patterns = [
            # FAA AD patterns
            {
                "authority": "FAA",
                "patterns": [
                    r'(?:AD\s*)?US-(\d{4})-(\d{2})-(\d{2})',  # US-2025-23-53
                    r'FAA\s*AD\s*(\d{4})-(\d{2})-(\d{2})',    # FAA AD 2025-23-53
                    r'(?:AD\s+)?(\d{4})-(\d{2})-(\d{2})(?:\s|$)',  # 2025-23-53
                ],
                "id_format": "FAA-{0}-{1}-{2}"
            },
            # EASA AD patterns
            {
                "authority": "EASA",
                "patterns": [
                    r'EASA\s*AD\s*(\d{4})-(\d{4})(?:R(\d+))?',  # EASA AD 2025-0254R1
                    r'(?:AD\s+)?(\d{4})-(\d{4})(?:R(\d+))?',     # 2025-0254R1
                ],
                "id_format": "EASA-{0}-{1}{rev}"
            },
        ]
        
        # Try to detect authority first from explicit mentions
        authority = None
        if re.search(r'\bFAA\b|Federal Aviation Administration', combined, re.IGNORECASE):
            authority = "FAA"
        elif re.search(r'\bEASA\b|European Union Aviation Safety Agency', combined, re.IGNORECASE):
            authority = "EASA"
        
        # Try each pattern
        for pattern_def in patterns:
            for pattern in pattern_def["patterns"]:
                match = re.search(pattern, combined, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    # Format AD ID
                    if pattern_def["authority"] == "FAA":
                        ad_id = f"FAA-{groups[0]}-{groups[1]}-{groups[2]}"
                    elif pattern_def["authority"] == "EASA":
                        revision = f"R{groups[2]}" if len(groups) > 2 and groups[2] else ""
                        ad_id = f"EASA-{groups[0]}-{groups[1]}{revision}"
                    
                    # Use detected authority or pattern default
                    final_authority = authority or pattern_def["authority"]
                    
                    return ad_id, final_authority
        
        # If no pattern matched, try to extract any AD-like identifier
        generic_match = re.search(r'(?:AD|Directive)\s*[:#]?\s*([A-Z0-9-]+)', combined, re.IGNORECASE)
        if generic_match:
            return generic_match.group(1), authority
        
        return None, authority


class TextChunker:
    """
    Splits document text into chunks for embedding.
    
    Uses a simple recursive splitting strategy that respects 
    sentence and paragraph boundaries.
    """
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE, 
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _split_text(self, text: str) -> List[str]:
        final_chunks = []
        
        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # Last resort: character-by-character split
                chunks = [text[i:i + self.chunk_size] 
                         for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
                final_chunks.extend(chunks)
                break
            
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                
                for part in parts:
                    # Check if adding this part exceeds chunk size
                    test_chunk = current_chunk + separator + part if current_chunk else part
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if it's not empty
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        
                        # If part itself is too big, recursively split it
                        if len(part) > self.chunk_size:
                            sub_chunks = self._split_text(part)
                            final_chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = part
                
                # Don't forget the last chunk
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                
                break
        
        # Remove empty chunks
        return [c for c in final_chunks if c.strip()]
    
    def chunk_document(self, pdf_doc: PDFDocument) -> List[Document]:
        """
        Split a PDF document into chunks.
        
        Args:
            pdf_doc: PDFDocument to chunk
            
        Returns:
            List of Document chunks with metadata
        """
        # Split text into chunks
        text_chunks = self._split_text(pdf_doc.full_text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(text_chunks):
            doc = Document(
                content=chunk,
                metadata={
                    **pdf_doc.metadata,
                    "chunk_id": str(i),
                    "total_chunks": str(len(text_chunks))
                },
                doc_id=f"{pdf_doc.metadata.get('ad_id', pdf_doc.filename)}_{i}"
            )
            documents.append(doc)
        
        return documents
    
    def chunk_all_documents(self, pdf_docs: List[PDFDocument]) -> List[Document]:
        """
        Chunk all PDF documents.
        
        Args:
            pdf_docs: List of PDFDocument objects
            
        Returns:
            List of all Document chunks
        """
        all_chunks = []
        
        for pdf_doc in pdf_docs:
            chunks = self.chunk_document(pdf_doc)
            all_chunks.extend(chunks)
            print(f"  → {pdf_doc.filename}: {len(chunks)} chunks")
        
        return all_chunks


def load_and_chunk_pdfs(data_dir: str = "./data") -> List[Document]:
    """
    Convenience function to load and chunk all PDFs.
    
    Args:
        data_dir: Directory containing PDF files
        
    Returns:
        List of Document chunks ready for embedding
    """
    print("Loading PDFs...")
    loader = PDFLoader(data_dir)
    pdf_docs = loader.load_all_pdfs()
    
    print(f"\nChunking {len(pdf_docs)} documents...")
    chunker = TextChunker()
    chunks = chunker.chunk_all_documents(pdf_docs)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks
