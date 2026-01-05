# SemEval2026-AER
Course project for "Large Language Models for Software Engineering 25/26" @ PoliTo. Exploring SemEval 2026 Task 12: Abductive Event Reasoning (AER) to investigate real-world causal inference capabilities in LLM.

## Project Structure

```text
AER-Project/
│
├── data/               # Dataset files (SemEval 2026)
├── paper/              # Project report with ACL template
├── results/            # Model training results
├── src/
│   ├── approaches.py   # Reasoning logic (Baseline, CoT, etc.)
│   ├── dataloader.py   # Data preprocessing and loading
│   ├── evaluator.py    # Evaluate model performance
│   ├── retriever.py    # Retrieves and ranks documents (BM25 + Semantic)
│   └── llm.py          # LLM API Wrapper (DeepSeek)
└── run.py              # Main entry point for experiments
```

## Installation

### Basic Dependencies
```bash
pip install rank-bm25 numpy python-dotenv openai
```

### For Semantic Retrieval
```bash
pip install sentence-transformers
# Optional: for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage (BM25 Retrieval)
```bash
python run.py --docs_path data/dev/docs.json --questions_path data/dev/questions.jsonl
```

### Semantic Retrieval
```bash
# Use semantic retrieval with default model (all-MiniLM-L6-v2)
python run.py --retrieval_method semantic

# Use a more accurate model (slower but better)
python run.py --retrieval_method semantic --semantic_model all-mpnet-base-v2

# Use GPU acceleration (if available)
python run.py --retrieval_method semantic --use_gpu

# Use title_snippet instead of full content (faster, less accurate)
python run.py --retrieval_method semantic --use_full_content=False
```

### Command Line Arguments

**Retrieval Options:**
- `--retrieval_method`: Choose `"bm25"` (default) or `"semantic"`
- `--top_k`: Number of documents to retrieve (default: 10)
- `--no_retrieval`: Disable retrieval, use all documents
- `--semantic_model`: Model name for semantic retrieval (default: `all-MiniLM-L6-v2`)
- `--use_full_content`: Use full document content for semantic retrieval (default: True)
- `--use_gpu`: Use GPU for semantic retrieval (if available)

**Available Semantic Models:**
- `all-MiniLM-L6-v2`: Fast, 384-dim (recommended for speed)
- `all-mpnet-base-v2`: Slower but more accurate, 768-dim (recommended for accuracy)
- `all-MiniLM-L12-v2`: Balanced, 384-dim

## Retrieval Methods

### BM25 (Keyword-based)
- Fast and lightweight
- Based on exact keyword matching
- Good for precise term matching
- Cannot handle synonyms or semantic similarity

### Semantic Retrieval (Vector-based)
- Understands semantic similarity
- Handles synonyms and different expressions
- Better for understanding context
- Requires more computational resources
- Uses sentence transformers to encode text into vectors
