# Quill Studio — Automatic MCQ & Quiz Generator (P16)

A local-first instructor tool for turning course material into editable, source-linked quizzes. The project includes a Streamlit interface, a local HTTP API, document extraction, persistent per-document chunk indexes, topic-focused retrieval, a rule-based baseline, optional local Hugging Face generation, quiz review/taking/scoring, and dataset evaluation tools.

## Runtime status and implementation choices

The local API uses Python's standard-library HTTP server and SQLite. The Streamlit UI talks to that API and starts it locally when needed. `requirements.txt` contains the UI and document readers; `requirements-rag.txt` adds Sentence Transformers and FAISS; `requirements-ml.txt` adds the optional text-generation model stack. No paid API is required. Model packages and weights are not bundled.

The default generator is a transparent rule baseline: it makes a fill-in-the-source-term question from a sentence and retains the exact supporting excerpt. With semantic dependencies and the local embedding checkpoint available, document chunks are indexed in FAISS and topic-matched before question generation. If those components are unavailable, the app records and displays a keyword-only fallback. Distractors can be weak, so every generated question needs instructor review. Topic-only generation without source material is not enabled.

## Run locally

Use Python 3.10 or newer.

Install the UI and document reader dependencies, then start Streamlit:

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

Or use a full Python path if `py` is unavailable:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Streamlit URL printed in the terminal (normally http://localhost:8501). The app starts the API on `127.0.0.1:8000` in the same process when it is not already running. Existing vanilla-browser API UI remains available at http://127.0.0.1:8000. Data is stored in `data/mcq.sqlite3`; FAISS indexes are stored under `data/vector_indexes/`. Set `MCQ_DATA_DIR`, `PORT`, `HOST`, `MAX_UPLOAD_MB`, and `LOG_LEVEL` to configure storage and the local API. Keep the server bound to loopback unless you add authentication and deployment hardening.

For PDF and DOCX uploads, the base install includes the document extractors. For true semantic topic retrieval, install the embedding/vector packages:

```powershell
python -m pip install -r requirements-rag.txt
```

DOCX extraction uses `python-docx`; PDF extraction uses PyMuPDF. Scanned PDFs are detected as text-sparse and need OCR; OCR is not currently included. Download/cache `sentence-transformers/all-MiniLM-L6-v2`, then set `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` and choose whether model files must be local with `MODEL_LOCAL_FILES_ONLY`. Configure these before starting the app. Rebuild indexes from the RAG evaluation page after changing embedding models. If FAISS or model weights are unavailable, the app explicitly uses keyword ranking instead.

For the optional Hugging Face question-generation model, install `requirements-ml.txt`, set `GENERATOR=model`, and configure `MODEL_NAME` as described in [docs/evaluation.md](docs/evaluation.md). Weight downloads can be large and depend on available RAM/VRAM.

## Main workflow

1. Open **Upload Documents**, upload PDF/DOCX/TXT or paste content, then preview the extracted text.
2. Open **Generate Quiz**, select that document, enter a specific topic and optional learning objectives, and choose question settings.
3. The app retrieves top-ranked document chunks before generating. Open **Question Review** to edit and publish a draft.
4. Use **Take Quiz** to complete a published quiz; scoring is computed by the API and stored in SQLite. **Results** shows explanations and source excerpts.
5. **RAG & Model Evaluation** shows index status, can rebuild document indexes, and imports licensed JSON/JSONL/CSV MCQ data. Evaluation scripts compare the rule baseline with an optional local model on the same held-out records.

Useful API routes: `GET /api/health`, `GET /api/dashboard`, `POST /api/documents`, `POST /api/documents/text`, `GET /api/documents`, `POST /api/generate`, `POST /api/validation`, `GET/PUT /api/quizzes/{id}`, `POST /api/quizzes/{id}/submit`, `GET /api/metrics`, `GET /api/dataset`, `POST /api/dataset/import`, `GET /api/dataset/export?split=test`.

## Tests

Run the available unit tests and syntax check:

```powershell
python -m unittest discover -s backend/tests -v
python -m py_compile app.py backend/app.py src/rag_index.py src/streamlit_ui.py ml/scripts/prepare_dataset.py ml/scripts/evaluate.py
```

Tests use a temporary SQLite directory and synthetic fixture text. That fixture is only for software testing, not a research dataset.

For the research comparison, prepare or import a dataset you are licensed to use; the app does not bundle fabricated or third-party benchmark records. See [docs/dataset.md](docs/dataset.md) for source notes and [docs/evaluation.md](docs/evaluation.md) for the reproducible held-out protocol. No benchmark result is claimed until real records and the selected model checkpoints are supplied and evaluated.

## Model configuration

No model weights are bundled or downloaded. The default is `GENERATOR=rules`. To opt into the local Hugging Face adapter, install `requirements-ml.txt`, set `GENERATOR=model`, and set `MODEL_NAME` to a locally available text-to-text checkpoint that can follow the JSON prompt. `MODEL_LOCAL_FILES_ONLY=true` prevents downloads; set it false only when you explicitly want Transformers to retrieve weights. The first generation request loads the model. Missing packages/weights or unusable output produce an explicit error; there is no silent fallback. Model outputs are marked for instructor review and the current answer support check only checks verbatim occurrence. See `docs/evaluation.md` for the boundary and reproducibility plan.

## Privacy and limitations

Uploaded content is stored locally as extracted text in SQLite. There is no user authentication or multi-user tenancy, and generated records have no deletion UI yet; use this as a single-user local development application. Upload size is capped (15 MB by default), filenames are reduced to a basename, and files are not written to a shared upload directory. Do not expose the server publicly. The application does not include OCR, fine-tuning, automatic answer entailment, a calibrated difficulty model, a timer, or a real annotated evaluation dataset. Automatic dashboard values are read from the database, not hard-coded.

## Repository map

`app.py` Streamlit navigation · `pages/` nine UI pages · `src/rag_index.py` persistent FAISS/SQLite chunk indexing and retrieval · `src/streamlit_ui.py` page flows · `backend/app.py` API, extraction, generation, quizzes and persistence · `frontend/` legacy browser UI · `ml/scripts/` dataset cleaning and held-out comparison · `docs/` guides · `backend/tests/` tests.
