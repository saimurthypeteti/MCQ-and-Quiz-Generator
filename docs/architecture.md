# Architecture

## Components

- `app.py` and `pages/`: Streamlit navigation and nine pages.
- `src/streamlit_ui.py`: UI workflows and local API client.
- `backend/app.py`: local HTTP API, document extraction, rule/model question generation, validation, quiz scoring, and SQLite persistence. It also serves the legacy browser interface.
- `src/rag_index.py`: overlapping chunking, persistent SQLite chunk mappings, per-document FAISS vector indexes, and keyword fallback.
- `ml/scripts/`: dataset normalization and reproducible held-out model comparison.

Streamlit starts the loopback API inside its process if one is not already available. The API can also be run on its own using `python backend/app.py`. SQLite contains extracted source text, section metadata, quiz records, attempts, dataset records, document chunks and vector-index metadata. FAISS files are stored under `data/vector_indexes/`.

## Trust boundaries

Uploads are limited by extension and size, filenames are reduced to basenames, and extracted text is treated as quoted source data rather than model instructions. The server binds to loopback by default and has no authentication; keep it local. Models and API secrets are not bundled or stored in SQLite. There is no external paid API integration.

## Model behavior

`GENERATOR=rules` is the default and creates a transparent cloze-style baseline. `GENERATOR=model` activates the configured local Transformers checkpoint and returns an explicit error if it cannot load or produce a schema-valid question. Embedding retrieval is independent of text generation: a semantic FAISS index does not make the baseline itself an LLM. Keyword-only fallback and model/index status are surfaced to the user.
