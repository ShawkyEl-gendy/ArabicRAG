# ArabicRAG

![Arabic RAG Architecture](./our%20rag.drawio.png)

ArabicRAG is an Arabic Retrieval-Augmented Generation (RAG) project designed to build and evaluate an Arabic QA pipeline using hybrid dense and sparse retrieval, reranking, NER, context construction, and LLM generation.

## Project Description

This repository implements a multilingual Arabic RAG system that:

- loads and preprocesses Arabic QA datasets,
- chunks document contexts,
- generates dense embeddings and sparse lexical representations,
- indexes chunks into ChromaDB,
- retrieves relevant chunks using dense, sparse, weighted hybrid, and RRF fusion,
- reranks candidates with a cross-encoder,
- add NER with context,
- builds context for generation,
- generates Arabic answers with Ollama LLMs.

It is intended for experimenting with Arabic retrieval and generation workflows using real QA corpora.

## Features

- Dataset loading and preprocessing for Arabic QA corpora.
- Multiple chunking strategies:
  - recursive
  - fixed
  - semantic
- Dense retrieval with `intfloat/multilingual-e5-large` and ChromaDB.
- Sparse retrieval with BGE-M3 lexical vectors via `FlagEmbedding`.
- Hybrid retrieval:
  - weighted fusion
  - reciprocal rank fusion (RRF)
- Cross-encoder reranking using `BAAI/bge-reranker-v2-m3`.
- NER extraction using `hatmimoha/arabic-ner`.
- Enhanced RAG answer generation using local LLM models.
- Evaluation of retrieval with `ranx` metrics.

## Repository Structure

- config.py — central pipeline configuration.
- chunking.py — chunk creation utilities and multiple chunking strategies.
- dense_retrieval.py — dense embedding generation and ChromaDB indexing.
- sparse_retrieval.py — sparse index creation and retrieval with BGE-M3.
- evaluation.py — retrieval evaluation, hybrid retrieval builders, reranking, and metrics.
- ner.py — Arabic NER extraction and preparation.
- enhanced_rag.py — enhanced RAG generation pipeline.
- naive_rag.py — niave RAG generation pipeline.
- prompts.py — Arabic prompt templates for RAG and enhanced RAG.
- preprocess_arabic_data_class.py — Arabic text cleaning and normalization utilities.
- requirements.txt — Python dependency list.
- loading-datasets.ipynb — dataset loading, cleaning, and exploration notebook.

## Technologies Used

- Python 3
- pandas
- numpy
- matplotlib
- sentence-transformers
- transformers
- chromadb
- langchain
- ollama
- FlagEmbedding
- ranx
- ragas
- bert-score
- nltk

## Models Used

- Embedding model: `intfloat/multilingual-e5-large`
- Semantic chunking model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- NER model: `hatmimoha/arabic-ner`
- Sparse retrieval model: `BAAI/bge-m3`
- Cross-encoder reranker: `BAAI/bge-reranker-v2-m3`
- LLMs used in enhanced_rag.py:
  - `aya:8b`
  - `llama3:8b`
  - `qwen2.5:3b`
  - `command-r7b-arabic`

## Installation

```bash
cd src
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Important configuration is stored in config.py:

- `DATASET_PATH` — path to the dataset JSONL file
- `CHUNK_STRATEGY` — chunking strategy
- `CHUNK_SIZE` — maximum chunk length
- `CHUNK_OVERLAP` — overlap between chunks
- `MIN_CHUNK_SIZE` — minimum chunk length
- `SIMILARITY_THRESHOLD` — threshold for semantic chunking and ground truth selection
- `EMBEDDING_MODEL` — dense embedding model
- `DB_PATH` — Chroma DB persistence path
- `COLLECTION_NAME` — Chroma collection name
- `USE_NER` — whether to merge NER metadata into chunk text
- `MODEL_NAME` — default generation model

## Datasets

The repository includes a notebook for loading and cleaning Arabic QA datasets:

- MRC dataset (`MRC Dataset/train.json`, `MRC Dataset/val.json`, `MRC Dataset/test.json`)
- ARCD (Arabic Reading Comprehension Dataset) — [hsseinmz/arcd](https://huggingface.co/datasets/hsseinmz/arcd) (`arcd.csv` / parquet source)
- ArQuAD — [RashaMObeidat/ArQuAD](https://github.com/RashaMObeidat/ArQuAD?utm_source=chatgpt.com) (`ArQuAD/ArQuAD-train.csv`, `ArQuAD/ArQuAD-test.csv`, `ArQuAD/ArQuAD-dev.csv`)
- ArabicaQA — [abdoelsayed/ArabicaQA](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/tree/main/MRC?utm_source=chatgpt.com)

The expected dataset format for pipeline scripts is a `jsonl` or DataFrame with at least:

- `question`
- `context`
- `answer`

For chunking and evaluation, the dataset is also expected to provide identifiers such as:

- `document_id`
- `query_id`

The loading-datasets.ipynb notebook shows how to standardize raw datasets into that format and save files like:

- `mrc-processed.jsonl`
- `arcd-processed.jsonl`
- `ArQuAD-processed.jsonl`

## Project Workflow

1. Data loading
   - `prepare_data.load_dataset()` loads the configured dataset.
2. Chunking
   - `create_chunk_dataframe()` slices each document context into chunks.
   - Strategies include recursive, fixed, semantic, agentic, and document-level.
3. Embedding generation
   - dense_retrieval.py creates dense embeddings with SentenceTransformer.
4. Vector database creation
   - Dense embeddings are persisted into ChromaDB collections.
5. Sparse index creation
   - sparse_retrieval.py generates BGE-M3 lexical vectors for chunks and queries.
6. Retrieval
   - evaluation.py implements dense retrieval, sparse retrieval, weighted fusion, and RRF.
7. Reranking
   - Cross-encoder reranking is implemented with `FlagReranker`.
8. NER
    - enrich each retrieved chunk with NER.
9. Context construction
   - enhanced_rag.py builds a context string from top reranked chunks.
10. LLM generation
    - enhanced_rag.py sends the assembled context and question to Ollama LLMs with enhanced prompt templates.

## How to Run

### Loading Dataset

Open the notebook:

```bash
jupyter notebook loading-datasets.ipynb
```
### Dataset preparation

```bash
python prepare_data.py
```

### Dense vector indexing

```bash
python dense_retrieval.py
```

### Sparse index creation

```bash
python sparse_retrieval.py
```

### Retrieval evaluation

```bash
python evaluation.py
```


### Naive RAG generation

```bash
python naive_reag.py
```

### Enhanced RAG generation

```bash
python enhanced_rag.py
```

## Configuration Parameters

Key configurable variables:

- `CHUNK_STRATEGY`: `fixed`, `recursive`, `semantic`
- `CHUNK_SIZE`: chunk maximum size
- `CHUNK_OVERLAP`: overlap between chunks
- `MIN_CHUNK_SIZE`: minimum chunk length
- `SIMILARITY_THRESHOLD`: similarity cutoff
- `EMBEDDING_MODEL`: dense encoder model
- `DB_PATH`: path for Chroma persistence
- `COLLECTION_NAME`: Chroma collection name
- `USE_NER`: whether to include NER metadata in text
- `MODEL_NAME`: Ollama model for generation

## Evaluation

Evaluation is handled by evaluation.py using `ranx` metrics:

- `recall@5`
- `recall@10`
- `mrr`
- `ndcg@10`

The script builds ground truth relevance labels using a hybrid strategy that combines embedding similarity and token overlap between answers and chunks.

## Output

Generated outputs include:

- Indexed ChromaDB collection at `./db/MRC`
- Processed dataset files in JSONL format
- CSV results from generation scripts under `results/MRC/enhanced_rag/`
- Evaluation metrics printed to console

## Requirements

Dependencies are listed in requirements.txt, including:

- numpy
- pandas
- scipy
- matplotlib
- tqdm
- transformers
- sentence-transformers
- chromadb
- langchain
- langchain-core
- langchain-community
- langchain-text-splitters
- ollama
- openai
- FlagEmbedding
- ranx
- ragas
- nltk
- emoji

## Future Improvements

- Evaluate additional closed-source LLMs for performance comparison.
- Measure the computational complexity and runtime of the RAG pipeline.
- Incorporate a GraphRAG framework to enhance retrieval and multi-hop reasoning.

## License

No license file is included in the repository.

## Citation

No formal citation is provided. If you use this code, please cite the repository as `ArabicRAG`.
