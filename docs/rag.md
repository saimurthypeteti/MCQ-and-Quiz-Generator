# Retrieval-Augmented Generation

## Pipeline

When a source document is uploaded, the backend extracts text and page/section metadata, splits it into overlapping chunks, assigns each chunk a UUID, and writes the chunk text and source mapping to SQLite. If Sentence Transformers, the selected local checkpoint, and FAISS are available, normalized embeddings are added to a persistent per-document `IndexFlatIP` index under `data/vector_indexes/`. The SQLite chunk order is the FAISS vector-ID mapping. Otherwise the index status is stored as `keyword_only` and token-overlap retrieval remains available.

When a topic is provided, only chunks from the selected document are ranked. Semantic mode embeds the topic and searches that document's FAISS index; fallback mode ranks by token overlap. `RAG_MIN_SCORE` controls the minimum returned score (default `0.05`). If no chunk passes the threshold, generation returns an error instead of creating unrelated topic questions. The selected chunk sentences are passed to the selected generator. Every returned question keeps its support excerpt, source section, chunk ID, topic, and retrieval score where available.

## Enable local semantic search

Install the optional packages:

```powershell
python -m pip install -r requirements-rag.txt
```

Configure the model before starting Streamlit. For a model that is already cached locally:

```powershell
$env:EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
$env:MODEL_LOCAL_FILES_ONLY = "true"
python -m streamlit run app.py
```

To allow Transformers to download the checkpoint on first use, set `MODEL_LOCAL_FILES_ONLY=false` and run with network access. The model weights are not bundled. Changing the embedding model requires **Rebuild all document indexes** from the RAG & Model Evaluation page. Indexes are local, ignored by Git, and never combine chunks across documents.

## Generation model

The default generator is a deterministic rule baseline, so it remains available on installations without a text-generation checkpoint. The baseline makes cloze-style questions from retrieved source sentences; it does not itself generate pedagogically rich questions. For local Hugging Face generation, install `requirements-ml.txt`, set `GENERATOR=model`, set `MODEL_NAME` to a compatible text-to-text checkpoint, and check that model's RAM/VRAM needs. The prompt treats source text as untrusted data and requires JSON. If the model cannot load or return valid options, the API reports an error rather than silently substituting baseline questions.

## Limits

Retrieval scores rank chunks; they are not calibrated confidence values. Semantic similarity does not verify factual correctness. Answer support currently uses a conservative string/source screen, not an entailment model. Instructor review remains required. Retrieval Precision@K and Recall@K are unavailable until query-to-relevant-passage labels are supplied.
