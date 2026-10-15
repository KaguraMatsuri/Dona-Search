# v4 training and evaluation

These are source-checkout tools. No trained adapter is bundled or accepted merely because training or a numeric preflight finishes. The runtime accepts user-selected models and weights; model quality must be evaluated separately.

## Environment and base

```bash
.venv/bin/python -m pip install -e '.[dev,training,eval]'
```

The current training tool uses an 8-bit MLX base with group size 64 and bound LoRA adapters. It runs under a separate 24 GiB supervisor budget. Run preflight before training. These commands perform real model work and should be scheduled with enough free memory.

## Human labels

The v4 dataset contains `query` and `organizing` records. Query labels must be reviewed and contain at most two generated queries. The original query is added by the pipeline.

Organizing records contain the original candidates, human relevance `judgments` and `selections` indexed by input candidate. Each selection is a segment ID or null. A relevance score alone is not a segment label. `review_status` must be `confirmed` with a nonempty reviewer before building.

```bash
.venv/bin/python scripts/prepare_training_data.py review --schema-version 4 --records records.jsonl --output review-batches
```

Ranking records become pending organizing records with unset selections. Exported `candidate_segments` show IDs, exact text and source offsets. Query records with more than two variants are returned for review. No selection labels or human confirmations are invented. Review the output before building.

```bash
.venv/bin/python scripts/prepare_training_data.py build --schema-version 4 --records reviewed.jsonl --model /path/to/prepared-base --output v4-data
```

The v4 manifest uses dataset schema 2 and `framework_schema_version: 4`. It binds files by hash and checks split separation, reviewed labels, production prompts, tokenizer bounds and exact targets. All present tasks require nonempty train/valid/test splits. Optional organizing `web_test` rows require human confirmation. A small valid dataset is not sufficient for release acceptance.

## Preflight and training

```bash
.venv/bin/python scripts/train_models.py preflight --task both --base /path/to/prepared-base --output preflight-v4
.venv/bin/python scripts/train_models.py train --task both --base /path/to/prepared-base --data v4-data --preflight-root preflight-v4 --output adapters-v4
```

`both` uses the dataset's tasks; v4 preflight defaults to query plus organizing. `--task organizing` runs only the second task. Legacy schema-1 datasets retain query/ranking tasks, and `--task ranking` remains available. Use a separate output directory for each run.

Preflight adapters remain probe artifacts. Checks include finite updates, frozen base preservation, exact adapter reload and the actual inference protocol. Failed or mismatched preflight gates do not permit training. Adapters remain unaccepted after training.

## Evaluation

```bash
.venv/bin/python scripts/eval_models.py --data v4-data --bf16 /path/to/bf16-base --base /path/to/prepared-base --adapters adapters-v4 --output evaluation-v4
```

Schema-2 v4 data selects the organizer evaluator, comparing bf16, quantized and adapted variants on the same candidates. It records protocol failures, rank quality, exact source matching and agreement with human selections. Failed cases remain in the denominator.

Query predictions require human review bound to the exact prediction hash. Supply reviewed records through `--query-reviews`, using `--summarize-only` to update acceptance without loading models.

The existing gate requires 100 public and 10 web cases for each of Chinese and English, all adapted top-1 eligible cases correct, no ranking degradation against bf16, every adapted evidence selection matching the reviewed labels, and 50 reviewed query cases per language. Passing applies only to that reviewed fixture; it is not proof of universal 100% factual correctness.

Old schema-1 ranking reports identify framework schema 3 and do not establish v4 evidence quality. Source receipts and license fields record provenance; they do not replace review of source terms or human annotations.
