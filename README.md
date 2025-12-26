# Airworthiness Directive Extraction Pipeline

A RAG (Retrieval-Augmented Generation) pipeline for extracting applicability rules from Airworthiness Directive (AD) PDF documents and evaluating aircraft configurations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │   PDF    │───▶│  Chunk   │───▶│ Embedding│───▶│ ChromaDB │       │
│  │  Loader  │    │  (1000)  │    │  NVIDIA  │    │  Vector  │       │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │
│                                                        │             │
│                                                        ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐       │
│  │  Output  │◀───│   LLM    │◀───│  Semantic Search Query   │       │
│  │  (JSON)  │    │ Extractor│    │                          │       │
│  └──────────┘    └──────────┘    └──────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology |
|----------|-----------|
| PDF Extraction | PyMuPDF (fitz) |
| Embeddings | NVIDIA API (`nvidia/nv-embedqa-e5-v5`) |
| Vector Database | ChromaDB |
| LLM | Meta Llama 3.1 70B (via NVIDIA API) |
| Data Validation | Pydantic |

## Project Structure

```
├── data/                       # AD PDF files (place new AD PDFs here)
├── output/                     # Output JSON files
├── src/
│   ├── config.py              # API configuration and settings
│   ├── document_loader.py     # PDF loading and text chunking
│   ├── embeddings.py          # NVIDIA embedding service
│   ├── vector_store.py        # ChromaDB wrapper (CRUD)
│   ├── llm_extractor.py       # LLM-based rule extraction
│   ├── aircraft_evaluator.py  # Aircraft evaluation
│   ├── rag_pipeline.py        # Orchestration pipeline
│   └── models.py              # Pydantic data models
├── tests/                      # Unit tests
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── .env.example               # Template environment variables
└── README.md
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ad-extraction-pipeline

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your NVIDIA API key
```

## Configuration

1. Copy `.env.example` to `.env`
2. Fill in `NVIDIA_API_KEY` with your API key

```bash
# .env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## Usage

### Running the Pipeline

```bash
python main.py
```

The pipeline will:
1. Load PDFs from `./data`
2. Chunk documents (1000 characters, overlap 200)
3. Generate embeddings with NVIDIA API
4. Store in ChromaDB
5. Extract rules with LLM
6. Evaluate test aircraft

### Q&A Mode

Use `chat.py` to ask questions about AD documents:

```bash
# Interactive mode
python chat.py

# Single question
python chat.py --ask "what aircraft models are in the FAA AD?"

# Ingest documents first, then Q&A
python chat.py --ingest
```

Example questions:
- "What is the subject of EASA AD 2025-0254?"
- "What aircraft models are affected by the FAA AD?"
- "What modifications can exclude an aircraft?"
- "What is the effective date of this AD?"

### Output Files

Output is saved to the `./output` folder:
- `output/extracted_rules.json` - Extracted rules
- `output/evaluation_results.json` - Aircraft evaluation results

### Format Output JSON

```json
{
  "ad_id": "FAA-2025-23-53",
  "applicability_rules": {
    "aircraft_models": ["MD-11", "MD-11F"],
    "msn_constraints": null,
    "excluded_if_modifications": ["SB-XXX"],
    "required_modifications": []
  }
}
```

### Programmatic Usage

```python
from src.rag_pipeline import RAGPipeline, PipelineConfig

# Initialize
config = PipelineConfig(data_dir="data")
pipeline = RAGPipeline(config)

# Ingest documents
pipeline.ingest_documents()

# Extract rules
rules = pipeline.extract_all_rules()

# Evaluate aircraft
results = pipeline.evaluate_aircraft("A320-214", 5234, [])
```

## Evaluation Results

### Aircraft Evaluation Summary

| Aircraft | MSN | Modifications | FAA AD | EASA AD |
|----------|-----|---------------|--------|---------|
| MD-11 | 48123 | - | Affected | Not affected |
| DC-10-30F | 47890 | - | Affected | Not affected |
| A320-214 | 5234 | - | Not affected | Affected |
| A320-232 | 6789 | mod 24591 | Not affected | Not affected |
| A321-111 | 8123 | - | Not affected | Affected |
| MD-11F | 48400 | - | Affected | Not affected |

### Verification (All Pass)

| Test Case | FAA Expected | FAA Actual | EASA Expected | EASA Actual |
|-----------|--------------|------------|---------------|--------------|
| MD-11F MSN 48400 | Affected | Affected | Not affected | Not affected |
| A320-214 MSN 4500 + mod 24591 | Not affected | Not affected | Not affected | Not affected |
| A320-214 MSN 4500 (no mods) | Not affected | Not affected | Affected | Affected |

## Supported AD Formats

The pipeline automatically detects ADs from various authorities:

| Authority | Pattern | Example |
|-----------|---------|--------|
| FAA | `AD YYYY-NN-NN`, `US-YYYY-NN-NN` | FAA-2025-23-53 |
| EASA | `AD YYYY-NNNNRx` | EASA-2025-0254R1 |

To add a new authority, edit `_detect_ad_info()` in [document_loader.py](src/document_loader.py).

## Production Deployment

### Adding New ADs

1. Place AD PDF in `./data` folder
2. Run `python main.py`
3. The pipeline will automatically:
   - Detect AD ID and authority
   - Extract rules with LLM
   - Generate JSON output

### Integration with Other Systems

```python
from src.rag_pipeline import RAGPipeline, PipelineConfig

# Initialize
config = PipelineConfig(data_dir="path/to/your/ad/files")
pipeline = RAGPipeline(config)

# Ingest new documents
pipeline.ingest_documents()

# Extract rules for all ADs
rules = pipeline.extract_all_rules()

# Evaluate specific aircraft
results = pipeline.evaluate_aircraft(
    model="A320-214",
    msn=5234,
    modifications=["mod 24591"]
)

# Check if affected
for result in results:
    print(f"{result.ad_id}: {'Affected' if result.is_affected else 'Not affected'}")
```

### Customization

| Setting | File | Description |
|---------|------|-------------|
| Chunk size | `src/config.py` | Chunk size for embedding |
| LLM Model | `src/config.py` | LLM model used |
| Embedding Model | `src/config.py` | Embedding model |
| AD Patterns | `src/document_loader.py` | Regex patterns for AD detection |

## Dependencies

```
pydantic>=2.0.0
PyMuPDF>=1.24.0
chromadb>=0.4.0
openai>=1.0.0
python-dotenv>=1.0.0
rich>=13.0.0
pytest>=8.0.0
```

## License

MIT License
