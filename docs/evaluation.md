# Dataset and model evaluation

## Dataset protocol

The Evaluation page imports JSON arrays or JSONL records with `passage_id`, `passage`, `question`, `options`, and `correct_answer`. It also preserves `explanation`, `difficulty`, source references and `annotation_source` fields for human-written or student-created evaluation items. The license and dataset name are mandatory import metadata. The application does not fetch data or infer permission. Imported records can be exported by split with `/api/dataset/export?split=test`.

No external research dataset is bundled. Use the guidance in [dataset.md](dataset.md) to source an appropriate dataset and respect its license. Do not report the software test fixtures as research data.

## Reproducible baseline/model comparison

`ml/scripts/evaluate.py` evaluates the **same prepared test JSONL** with both methods:

1. Template/rule baseline: blanks a source term and builds options from passage terms plus generic distractors.
2. Pretrained local text-to-text model: prompted to produce an MCQ JSON object. It is not fine-tuned by this project.

The evaluator exports per-example predictions as CSV and a summary JSON. It reports structural question validity; answer-in-passage answerability as a lexical estimate; question-text cosine similarity to the human-written reference; within-method semantic duplicate rate at cosine 0.90; token F1 as a lexical diagnostic; and generation latency. Similarity metrics are populated only if Sentence Transformers and the selected embedding checkpoint load successfully. Similarity is not factual verification. The answerability screen is not entailment, and every generated answer still merits human review.

Prepare and split records by passage group, then run both methods on the exact same `test.jsonl`:

```powershell
python ml/scripts/prepare_dataset.py --input path/to/authorized-data.jsonl --output-dir data/prepared --seed 16 --dataset-name DatasetName --license "LICENSE"
python ml/scripts/evaluate.py --test data/prepared/test.jsonl --output data/evaluation --model PATH_TO_LOCAL_T5_OR_INSTRUCTION_MODEL --similarity-model PATH_TO_LOCAL_SENTENCE_TRANSFORMER
```

By default model loading is local-only. The command records unavailable models or embeddings explicitly; do not treat missing results as zero or as a successful benchmark. Record the dataset manifest, checkpoint revision, tokenizer, prompt, package versions, hardware and generation settings alongside results. No benchmark result exists until this command runs with real authorized data and actual local checkpoints.

## Workspace screening metrics

`GET /api/metrics` calculates structural validity, source excerpt presence, verbatim answer support and duplicate flags over saved quizzes. These workspace screens are separate from dataset comparison; they do not calculate embedding similarity. Human acceptance rate requires recorded instructor decisions and is currently unavailable.
