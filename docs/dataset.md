# Dataset format and sourcing

No educational question-generation dataset was present in the repository, and none is bundled. Do not treat the synthetic passages in software tests as research data.

## Accepted record shape

JSON Lines or JSON array records should contain `passage_id`, `passage`, `question`, `options` (array of strings), and `correct_answer`. Optional fields: `explanation`, `difficulty`, `source_id`, `source_page`, `source_section`, and `annotation_source` (for example `human-written` or `student-created`). Keep provenance, license, and source URL in a dataset-level manifest. The preparation script can read CSV, JSON arrays, or JSONL and adapts common SciQ export fields.

## Public dataset candidates to assess

- **SQuAD 2.0**: reading-comprehension questions and contexts; the dataset card identifies CC BY-SA 4.0. It is not an MCQ dataset and includes unanswerable questions; mapping it to distractor options requires a separately documented transformation. Official project: https://rajpurkar.github.io/SQuAD-explorer/ and [dataset card](https://huggingface.co/datasets/rajpurkar/squad_v2).
- **RACE**: English exam passages with four-option questions, commonly used for reading-comprehension research. The dataset's license is unresolved here; treat permissions as unresolved and do not download/redistribute until terms for your use are confirmed. [CMU dataset download page](http://www.cs.cmu.edu/~glai1/data/race/) and [official baselines repository](https://github.com/qizhex/RACE_AR_baselines).
- **SciQ**: science exam questions with distractors and supporting passages. Its dataset card states CC BY-NC 3.0, which restricts commercial use. Check source terms and preserve attribution and the original split. [Dataset card](https://huggingface.co/datasets/allenai/sciq).

These are candidate sources, not installed or evaluated data. Check the current license and terms at acquisition time. The project does not download any dataset automatically. To acquire SQuAD 2.0, use the official [SQuAD download page](https://rajpurkar.github.io/SQuAD-explorer/) or its v2.0 JSON files, then transform context/question/answer fields; SQuAD does not supply MCQ distractors. For SciQ, use the [dataset card](https://huggingface.co/datasets/allenai/sciq) instructions (or `datasets.load_dataset("allenai/sciq")` after installing Hugging Face Datasets), and preserve its license and original split. RACE files are linked from the CMU dataset page; terms are unresolved here, so confirm permission before acquiring or using them. Convert downloaded records to this project's schema before running preparation.

## Prepare a dataset

```powershell
python ml/scripts/prepare_dataset.py --input path/to/authorized-data.jsonl --output-dir data/prepared --seed 16
```

The script normalizes whitespace, drops incomplete and exact-duplicate records, and assigns all records sharing a `passage_id` to one deterministic train/validation/test split to prevent passage leakage. It exports JSONL splits and a manifest. For small datasets, passage-grouping can leave a split empty; inspect the manifest rather than assuming a valid benchmark.

For an authorized SciQ JSON/JSONL export, include license and source metadata in the manifest:

```powershell
python ml/scripts/prepare_dataset.py --input path/to/sciq.json --output-dir data/prepared --seed 16 --dataset-name SciQ --license "CC BY-NC 3.0" --source-url https://huggingface.co/datasets/allenai/sciq
```

The preparation script adapts SciQ's `support`, `correct_answer`, and three `distractor*` fields into the project schema. Use SciQ only where its noncommercial terms fit the project. The Evaluation page accepts canonical JSON/JSONL and asks for license metadata before saving records; it does not infer license permission.
