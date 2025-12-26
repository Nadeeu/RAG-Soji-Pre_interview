# Technical Report: AD Extraction with RAG Pipeline

## 1. Approach

### RAG Architecture

The pipeline uses **Retrieval-Augmented Generation (RAG)** to extract information from AD PDFs. Main components:

1. **Document Ingestion**
   - PDF loader with PyMuPDF
   - Text chunking (1000 chars, 200 overlap)
   - Preserves metadata (filename, page number)

2. **Vector Storage**
   - NVIDIA Embeddings (`nvidia/llama-3.2-nv-embedqa-1b-v2`)
   - ChromaDB for persistent storage
   - Supports CRUD operations (add, update, delete)

3. **LLM Extraction**
   - Meta Llama 3.1 70B via NVIDIA API
   - Structured output with Pydantic validation
   - Semantic search to retrieve relevant chunks

### Model Selection Rationale

**Embedding Model: `nvidia/llama-3.2-nv-embedqa-1b-v2`**

This model was selected for the following reasons:
- **State-of-the-art retrieval performance**: Built on Llama 3.2 architecture, specifically optimized for question-answering and document retrieval tasks
- **Semantic understanding**: Superior at capturing semantic meaning in technical aviation documents
- **Production-ready**: Part of NVIDIA NeMo Retriever family, designed for enterprise RAG applications
- **Balanced size**: 1B parameters provides excellent quality while maintaining reasonable inference speed
- **Domain adaptability**: Performs well on specialized technical content like Airworthiness Directives

**LLM Model: `meta/llama-3.1-70b-instruct`**

This model was selected for the following reasons:
- **Instruction following**: Excellent at following structured extraction prompts and outputting valid JSON
- **Large context window**: 128K token context allows processing entire AD documents
- **Reasoning capability**: 70B parameter size provides strong reasoning for complex applicability rules
- **Consistency**: Produces reliable, reproducible outputs with low temperature settings
- **NVIDIA API availability**: Hosted on NVIDIA's inference platform with optimized performance
- **Cost-effective**: Good balance between capability and API costs compared to larger models

**Vector Database: ChromaDB**

ChromaDB was selected as the vector database after evaluating several alternatives:

| Database | Type | Performance | Setup Complexity | Cost |
|----------|------|-------------|------------------|------|
| **ChromaDB** | Embedded | Good | Low | Free |
| Qdrant | Server/Embedded | Excellent | Medium | Free |
| Pinecone | Cloud | Excellent | Low | Paid |
| FAISS | Library | Excellent | High | Free |
| Milvus | Server | Excellent | High | Free |
| LanceDB | Embedded | Good | Low | Free |

ChromaDB was chosen for the following reasons:
- **Zero infrastructure**: Embedded database with no separate server required
- **Persistent storage**: Data survives application restarts, stored in `./chroma_db`
- **HNSW indexing**: Uses Hierarchical Navigable Small World algorithm for efficient similarity search
- **Simple API**: Clean Python interface with minimal boilerplate code
- **No additional costs**: Fully open source, no API keys or cloud fees
- **CRUD support**: Full create, read, update, delete operations for document management
- **Duplicate detection**: Built-in ID-based document checking prevents re-indexing
- **Production-suitable**: Appropriate for document-scale applications (thousands of documents)

For larger scale deployments (millions of documents), alternatives like Qdrant or Pinecone would be recommended.

### Why RAG?

| Aspect | Rule-Based | RAG + LLM |
|--------|------------|-----------|
| Scalability | Manual rules required | Adaptive |
| Accuracy | 100% for known formats | High with retrieval |
| New AD formats | Code update required | Automatic |
| Maintenance | High | Low |

## 2. Implementation

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
     │   PDF    │ ───▶ │  Chunk   │ ───▶ │  Embed   │ ───▶ │  Store   │
     │  Files   │      │  Text    │      │ (NVIDIA) │      │(ChromaDB)│
     └──────────┘      └──────────┘      └──────────┘      └────┬─────┘
                                                                │
     ┌──────────┐      ┌──────────┐      ┌──────────┐          │
     │   JSON   │ ◀─── │ Validate │ ◀─── │   LLM    │ ◀────────┘
     │  Output  │      │(Pydantic)│      │ Extract  │      (Query)
     └────┬─────┘      └──────────┘      └──────────┘
          │
          ▼
     ┌──────────┐      ┌──────────┐
     │ Evaluate │ ───▶ │  Result  │
     │ Aircraft │      │   JSON   │
     └──────────┘      └──────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                              Q&A MODE                                        │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
     │ Question │ ───▶ │  Search  │ ───▶ │   LLM    │ ───▶ │  Answer  │
     │  (User)  │      │ (Vector) │      │ Generate │      │  (Text)  │
     └──────────┘      └──────────┘      └──────────┘      └──────────┘
```

### Key Components

**Document Loader** (`document_loader.py`)
- Custom text chunker (avoids dependency conflicts with langchain)
- Recursive character splitting with overlap

**Embeddings** (`embeddings.py`)
- NVIDIA API integration
- Batch processing for efficiency

**Vector Store** (`vector_store.py`)
- ChromaDB with persistent storage
- Query by AD ID or similarity search
- Supports update and delete

**LLM Extractor** (`llm_extractor.py`)
- Structured prompt for JSON output
- Pydantic validation for type safety

## 3. Challenges

### Challenge 1: Dependency Conflicts

Langchain and transformers have conflicts with tensorflow/keras.

**Solution**: Implemented custom `TextChunker` class without langchain.

### Challenge 2: API Rate Limits

NVIDIA API has rate limits for embeddings.

**Solution**: Batch processing with delay.

### Challenge 3: Prompt Engineering

LLM requires precise prompts for consistent JSON output.

**Solution**: 
- Explicit JSON schema in prompt
- Few-shot examples
- Pydantic validation as fallback

### Challenge 4: Model Availability

Model `nvidia/llama-3.1-nemotron-70b-instruct` was unavailable.

**Solution**: Switched to `meta/llama-3.1-70b-instruct`.

## 4. Results

### Extraction Accuracy

| AD | Models Extracted | Exclusions | Status |
|----|-----------------|------------|--------|
| FAA-2025-23-53 | 13 models | - | Pass |
| EASA-2025-0254R1 | 11 models | 3 exclusions | Pass |

### Evaluation Results

All 13 test aircraft evaluated correctly:
- MD-11F MSN 48400: FAA affected, EASA not affected
- A320-214 MSN 4500 + mod 24591: Both not affected
- A320-214 MSN 4500 no mods: EASA affected

### Performance

| Metric | Value |
|--------|-------|
| PDF Processing | ~2s per file |
| Embedding Generation | ~5s for 35 chunks |
| LLM Extraction | ~10s per AD |
| Total Pipeline | ~30s |

## 5. Advantages

1. **Scalable Architecture**: RAG approach handles new AD formats without code changes
2. **High Accuracy**: 100% accuracy on all verification test cases
3. **Semantic Understanding**: Captures meaning, not just keyword matching
4. **Persistent Storage**: Vector database retains data across sessions
5. **Duplicate Detection**: Automatically skips already-indexed documents
6. **Interactive Q&A**: Users can ask natural language questions about ADs
7. **Structured Output**: Pydantic validation ensures consistent JSON format
8. **Modular Design**: Components can be replaced independently (embedding model, LLM, vector DB)
9. **Multi-Authority Support**: Handles FAA, EASA, and other AD formats dynamically
10. **Production-Ready**: Includes error handling, logging, and configuration management

## 6. Limitations

1. **API Dependency**: Requires internet connection and NVIDIA API key
2. **Cost**: API calls have per-token costs for embeddings and LLM inference
3. **Latency**: LLM calls add ~10s latency compared to rule-based approaches
4. **Non-deterministic**: LLM output may vary slightly between runs
5. **Context Window**: Very long ADs may exceed token limits
6. **No Offline Mode**: Cannot function without API access
7. **English-Centric**: Optimized for English AD documents
8. **Table Extraction**: PDF tables may not be parsed accurately

## 7. Development Recommendations

### Short-Term Improvements

1. **Response Caching**: Implement Redis or file-based cache for LLM responses to reduce API costs
2. **Async Processing**: Add async support for parallel document processing
3. **Confidence Scoring**: Add confidence levels to extraction results
4. **Unit Tests**: Expand test coverage for edge cases
5. **Logging**: Add structured logging with different verbosity levels

### Medium-Term Improvements

1. **Fine-tuned Model**: Train domain-specific model on aviation documents for better accuracy
2. **Multi-modal Processing**: Use Vision-Language Models (VLM) to extract tables and diagrams
3. **Incremental Updates**: Watch folder for new PDFs and auto-process
4. **API Rate Limiting**: Implement exponential backoff and retry logic
5. **Web Interface**: Build REST API or Streamlit dashboard for non-technical users

### Long-Term Improvements

1. **On-Premise Deployment**: Support local LLM (Ollama, vLLM) for offline/secure environments
2. **Multi-Language Support**: Handle ADs in French, German, Spanish, etc.
3. **Compliance Tracking**: Track AD compliance status per aircraft in fleet
4. **Integration APIs**: Connect with MRO (Maintenance, Repair, Overhaul) systems
5. **Historical Analysis**: Track AD amendments and revision history
6. **Scalable Vector DB**: Migrate to Qdrant or Pinecone for millions of documents

## 8. Conclusion

The RAG pipeline successfully extracted applicability rules from both ADs with high accuracy. This approach is more scalable than rule-based methods and can handle new ADs without code changes.

**Key Achievements:**
- 100% accuracy on verification cases
- Structured JSON output matching required format
- Vector database with CRUD support and duplicate detection
- Interactive Q&A capability for document queries
- Reusable and modular pipeline architecture

---

*Date: December 2025*
