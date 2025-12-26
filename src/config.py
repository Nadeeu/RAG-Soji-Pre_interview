"""
Configuration settings for the AD Extraction Pipeline.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NVIDIA API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Validate API key
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found. Please set it in .env file.")

# Embedding Model
EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"  # NVIDIA NeMo Retriever embedding model

# LLM Model for extraction
LLM_MODEL = "meta/llama-3.1-70b-instruct"  # Using Meta's Llama model available on NVIDIA API

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "airworthiness_directives"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
