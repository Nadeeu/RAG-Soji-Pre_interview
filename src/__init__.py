# AD Extraction Pipeline with RAG
__version__ = "2.0.0"

from .models import (
    ADExtractedOutput, 
    ApplicabilityRulesOutput,
    AircraftConfiguration,
    EvaluationResult
)
from .document_loader import PDFLoader, TextChunker
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .llm_extractor import LLMExtractor
from .aircraft_evaluator import AircraftEvaluator
from .rag_pipeline import RAGPipeline, PipelineConfig

__all__ = [
    # Models
    "ADExtractedOutput",
    "ApplicabilityRulesOutput",
    "AircraftConfiguration",
    "EvaluationResult",
    # Components
    "PDFLoader",
    "TextChunker",
    "EmbeddingService",
    "VectorStore",
    "LLMExtractor",
    "AircraftEvaluator",
    # Pipeline
    "RAGPipeline",
    "PipelineConfig",
]
