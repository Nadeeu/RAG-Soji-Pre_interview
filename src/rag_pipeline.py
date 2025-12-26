"""
Main RAG Pipeline for AD Extraction.

This module orchestrates the complete pipeline:
1. Load PDFs from data directory
2. Chunk documents
3. Generate embeddings
4. Store in ChromaDB
5. Extract rules using LLM
6. Evaluate aircraft
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .document_loader import PDFLoader, TextChunker, Document, load_and_chunk_pdfs
from .embeddings import EmbeddingService, get_embedding_service
from .vector_store import VectorStore, get_vector_store
from .llm_extractor import LLMExtractor, get_llm_extractor
from .models import ADExtractedOutput, ApplicabilityRulesOutput
from .aircraft_evaluator import AircraftEvaluator, EvaluationResult, evaluate_aircraft


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    data_dir: str = "./data"
    persist_embeddings: bool = True
    force_reload: bool = False


class RAGPipeline:
    """
    Complete RAG pipeline for AD extraction and evaluation.
    
    Workflow:
    1. Load PDF documents
    2. Chunk text for embedding
    3. Generate embeddings and store in ChromaDB
    4. Extract rules using LLM with RAG
    5. Evaluate aircraft configurations
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.vector_store = get_vector_store()
        self.llm_extractor = get_llm_extractor()
        
        # Cache for extracted rules
        self.extracted_rules: Dict[str, ADExtractedOutput] = {}
    
    def ingest_documents(self, force_reload: bool = False) -> int:
        """
        Load, chunk, embed and store documents.
        
        Args:
            force_reload: If True, clear existing data and reload all
            
        Returns:
            Number of documents in vector store
        """
        if force_reload:
            print("Clearing existing data...")
            self.vector_store.clear_collection()
        
        # Load and chunk PDFs
        print("\n" + "="*50)
        print("STEP 1: Loading and Chunking PDFs")
        print("="*50)
        
        chunks = load_and_chunk_pdfs(self.config.data_dir)
        
        # Store in vector database (will skip existing documents)
        print("\n" + "="*50)
        print("STEP 2: Generating Embeddings and Storing")
        print("="*50)
        
        added_count = self.vector_store.add_documents(chunks, skip_existing=not force_reload)
        
        final_stats = self.vector_store.get_collection_stats()
        print(f"Total documents in vector store: {final_stats['document_count']}")
        
        return final_stats['document_count']
    
    def update_documents(self, pdf_path: str) -> int:
        """
        Update documents from a specific PDF.
        
        Args:
            pdf_path: Path to PDF to update
            
        Returns:
            Number of chunks updated
        """
        print(f"Updating documents from: {pdf_path}")
        
        # Load and chunk the specific PDF
        loader = PDFLoader(str(Path(pdf_path).parent))
        pdf_doc = loader.load_pdf(pdf_path)
        
        chunker = TextChunker()
        chunks = chunker.chunk_document(pdf_doc)
        
        # Get AD ID from metadata
        ad_id = pdf_doc.metadata.get('ad_id')
        if ad_id:
            # Delete existing documents for this AD
            self.vector_store.delete_by_ad_id(ad_id)
        
        # Add new chunks
        self.vector_store.add_documents(chunks)
        
        # Clear cached rules for this AD
        if ad_id and ad_id in self.extracted_rules:
            del self.extracted_rules[ad_id]
        
        return len(chunks)
    
    def extract_rules(self, ad_id: str, use_cache: bool = True) -> ADExtractedOutput:
        """
        Extract applicability rules for an AD.
        
        Args:
            ad_id: AD identifier
            use_cache: Whether to use cached results
            
        Returns:
            Extracted AD rules
        """
        # Check cache
        if use_cache and ad_id in self.extracted_rules:
            return self.extracted_rules[ad_id]
        
        print(f"\nExtracting rules for: {ad_id}")
        
        # Extract using RAG
        rules = self.llm_extractor.extract_with_rag(
            vector_store=self.vector_store,
            ad_id=ad_id,
            query="applicability aircraft models serial numbers modifications exclusions service bulletins"
        )
        
        # Cache results
        self.extracted_rules[ad_id] = rules
        
        return rules
    
    def extract_all_rules(self) -> Dict[str, ADExtractedOutput]:
        """
        Extract rules for all ADs in the vector store.
        
        Returns:
            Dictionary mapping AD ID to extracted rules
        """
        print("\n" + "="*50)
        print("STEP 3: Extracting Rules using LLM")
        print("="*50)
        
        # Get all unique AD IDs
        all_docs = self.vector_store.get_all_documents()
        ad_ids = set()
        for doc in all_docs:
            ad_id = doc['metadata'].get('ad_id')
            if ad_id:
                ad_ids.add(ad_id)
        
        # Extract rules for each AD
        for ad_id in ad_ids:
            self.extract_rules(ad_id)
            print(f"  {ad_id}: {len(self.extracted_rules[ad_id].applicability_rules.aircraft_models)} models")
        
        return self.extracted_rules
    
    def evaluate_aircraft(
        self,
        model: str,
        msn: int,
        modifications: Optional[List[str]] = None,
        ad_id: Optional[str] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate an aircraft against ADs.
        
        Args:
            model: Aircraft model
            msn: MSN
            modifications: Applied modifications
            ad_id: Specific AD to evaluate against (None for all)
            
        Returns:
            List of evaluation results
        """
        modifications = modifications or []
        results = []
        
        # Get rules to evaluate against
        if ad_id:
            rules_to_check = {ad_id: self.extract_rules(ad_id)}
        else:
            rules_to_check = self.extracted_rules
        
        for rule_ad_id, rules in rules_to_check.items():
            result = evaluate_aircraft(rules, model, msn, modifications)
            results.append(result)
        
        return results
    
    def export_rules(self, output_path: str = "output/extracted_rules.json") -> str:
        """
        Export extracted rules to JSON file.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        export_data = {
            "airworthiness_directives": [
                rules.model_dump() for rules in self.extracted_rules.values()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nRules exported to: {output_path}")
        return output_path
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        vs_stats = self.vector_store.get_collection_stats()
        return {
            "vector_store": vs_stats,
            "extracted_ads": list(self.extracted_rules.keys()),
            "total_rules": len(self.extracted_rules)
        }
    
    def ask(self, question: str, ad_id: Optional[str] = None) -> str:
        """
        Ask a question about the AD documents.
        
        Args:
            question: User's question in natural language
            ad_id: Optional AD ID to filter context
            
        Returns:
            Answer from LLM based on document context
        """
        return self.llm_extractor.ask_question(
            vector_store=self.vector_store,
            question=question,
            ad_id=ad_id
        )
    
    def interactive_qa(self):
        """
        Start interactive Q&A session.
        
        Commands:
        - Type your question to get an answer
        - 'exit' or 'quit' to end session
        - 'stats' to show pipeline statistics
        - 'ads' to list available ADs
        """
        print("\n" + "="*60)
        print("Interactive Q&A Mode")
        print("="*60)
        print("Ask anything about processed AD documents.")
        print("Type 'exit' to quit, 'stats' for statistics, 'ads' for AD list.")
        print("-"*60)
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("Thank you! Goodbye.")
                    break
                
                if question.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"\nStatistics:")
                    print(f"   - Documents: {stats['vector_store']['document_count']}")
                    print(f"   - ADs: {', '.join(stats['extracted_ads']) or 'None yet'}")
                    continue
                
                if question.lower() == 'ads':
                    all_docs = self.vector_store.get_all_documents()
                    ad_ids = set()
                    for doc in all_docs:
                        if 'ad_id' in doc.get('metadata', {}):
                            ad_ids.add(doc['metadata']['ad_id'])
                    print(f"\nAvailable ADs: {', '.join(ad_ids) or 'None yet'}")
                    continue
                
                print("\nSearching for answer...")
                answer = self.ask(question)
                print(f"\nAnswer:\n{answer}")
                
            except KeyboardInterrupt:
                print("\n\nSession ended.")
                break
            except Exception as e:
                print(f"\nError: {e}")


def create_pipeline(data_dir: str = "./data") -> RAGPipeline:
    """Factory function to create pipeline."""
    config = PipelineConfig(data_dir=data_dir)
    return RAGPipeline(config)
