#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dona_search.core import CORE_CHECKPOINT, CoreModel
from scripts.core_data import read_jsonl
from scripts.core_eval import metrics, score_examples, write_report


def main():
    core = CoreModel(CORE_CHECKPOINT)
    try:
        predictions = score_examples(core, read_jsonl("data/core/test.jsonl"))
        report = metrics(predictions, core.threshold)
        write_report("checkpoints/core_training/runtime_evaluation.json", report)
        print(json.dumps({"event": "evaluation_complete", **report}), flush=True)
    finally:
        core.close()


if __name__ == "__main__":
    main()
