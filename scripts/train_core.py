import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from dona_search.core import (
    BASE_MODEL,
    BASE_REVISION,
    CORE_CHECKPOINT,
    MAX_LENGTH,
    QUERY_LENGTH,
    CoreModel,
    file_hash,
)
from scripts.core_data import SEED, read_jsonl
from scripts.core_eval import calibrate, metrics, score_examples, write_report

DATA = Path("data/core")
WORK = Path("checkpoints/core_training")
EPOCHS, MICRO_BATCH, ACCUMULATION, LR = 3, 2, 8, 2e-5


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def scoring_core(model, tokenizer, batch_size=8):
    core = CoreModel.__new__(CoreModel)
    core.model, core.tokenizer, core.device = model, tokenizer, "mps"
    core.batch_size, core.threshold = batch_size, 0.5
    return core


def load_model(path, dtype=torch.float32, revision=None):
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    return AutoModelForSequenceClassification.from_pretrained(
        path, revision=revision, dtype=dtype,
        device_map={"": "mps"}, attn_implementation="sdpa",
    )


def encoded_training(rows, tokenizer):
    dataset = Dataset.from_list(rows)
    def encode(batch):
        queries = [tokenizer.decode(tokenizer(q, add_special_tokens=False)["input_ids"][:QUERY_LENGTH], skip_special_tokens=True) for q in batch["query"]]
        result = tokenizer(queries, batch["text"], truncation="only_second", max_length=MAX_LENGTH)
        result["labels"] = [float(label) for label in batch["label"]]
        return result
    return dataset.map(encode, batched=True, remove_columns=dataset.column_names)


def save_training(model, tokenizer, optimizer, scheduler, epoch, step, history):
    pointer = WORK / "latest.json"
    previous = json.loads(pointer.read_text())["checkpoint"] if pointer.exists() else None
    latest = WORK / f"step-{epoch:02d}-{step:05d}"
    latest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(latest, max_shard_size="4GB")
    tokenizer.save_pretrained(latest)
    state = {
        "epoch": epoch, "micro_step": step, "history": history,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
        "data_manifest_sha256": file_hash(DATA / "data_manifest.json"),
    }
    temporary = latest / "trainer_state.tmp"
    torch.save(state, temporary)
    temporary.replace(latest / "trainer_state.pt")
    temporary_pointer = WORK / "latest.tmp"
    temporary_pointer.write_text(json.dumps({"checkpoint": latest.name}))
    temporary_pointer.replace(pointer)
    # Only retire the previous checkpoint created by this training loop.
    if previous and previous != latest.name and Path(previous).name == previous and previous.startswith("step-"):
        shutil.rmtree(WORK / previous)


def train(resume=False):
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps_required")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads((DATA / "data_manifest.json").read_text())
    for name, digest in data_manifest["files"].items():
        if file_hash(DATA / name) != digest:
            raise ValueError("training_data_hash_mismatch")
    rows = read_jsonl(DATA / "train.jsonl")
    if len(rows) != 20000:
        raise ValueError("training_pair_count_invalid")
    validation = read_jsonl(DATA / "validation.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    dataset = encoded_training(rows, tokenizer)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64)
    latest = WORK / json.loads((WORK / "latest.json").read_text())["checkpoint"] if resume else None
    model = load_model(latest if resume else BASE_MODEL, revision=None if resume else BASE_REVISION)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, foreach=False)
    micro_steps = math.ceil(len(dataset) / MICRO_BATCH)
    total_steps = math.ceil(micro_steps / ACCUMULATION) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    history, start_epoch, start_micro = [], 0, 0
    if resume:
        state = torch.load(latest / "trainer_state.pt", map_location="cpu", weights_only=True)
        if state["data_manifest_sha256"] != file_hash(DATA / "data_manifest.json"):
            raise ValueError("resume_data_mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, start_micro = state["history"], state["epoch"], state["micro_step"]
        torch.set_rng_state(state["torch_rng"])
        torch.mps.set_rng_state(state["mps_rng"])
        del state
    best = max((h["selection_score"] for h in history), default=-1.0)
    started = time.perf_counter()
    core = None
    for epoch in range(start_epoch, EPOCHS):
        loader = DataLoader(
            dataset, batch_size=MICRO_BATCH, shuffle=True, collate_fn=collator,
            generator=torch.Generator().manual_seed(SEED + epoch), num_workers=0,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rolling_loss = 0.0
        rolling_count = 0
        for step, batch in enumerate(loader):
            if epoch == start_epoch and step < start_micro:
                continue
            labels = batch.pop("labels").to("mps", dtype=torch.float32)
            inputs = {key: value.to("mps") for key, value in batch.items()}
            logits = model(**inputs).logits.squeeze(-1).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss).item():
                raise ValueError("training_nonfinite_loss")
            accumulation = min(ACCUMULATION, len(loader) - (step // ACCUMULATION) * ACCUMULATION)
            (loss / accumulation).backward()
            rolling_loss += loss.detach().item()
            rolling_count += 1
            del inputs, labels, logits, loss, batch
            if (step + 1) % ACCUMULATION == 0 or step + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm).item():
                    raise ValueError("training_nonfinite_gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.mps.empty_cache()
                if (step + 1) % 80 == 0:
                    emit("training_progress", epoch=epoch + 1, micro_step=step + 1,
                         total_micro_steps=len(loader), mean_loss=rolling_loss / rolling_count,
                         elapsed_seconds=time.perf_counter() - started,
                         current_allocated_bytes=torch.mps.current_allocated_memory(),
                         driver_allocated_bytes=torch.mps.driver_allocated_memory())
                    rolling_loss = 0.0
                    rolling_count = 0
                if (epoch == 0 and step + 1 == ACCUMULATION) or (step + 1) % 2000 == 0:
                    save_training(model, tokenizer, optimizer, scheduler, epoch, step + 1, history)
        model.eval()
        core = scoring_core(model, tokenizer)
        predictions = score_examples(core, validation)
        calibration = calibrate(predictions)
        selection = calibration["correct_evidence_coverage"] + calibration["ranking_window_ndcg_at_10"]
        entry = {"epoch": epoch + 1, "selection_score": selection, "validation": calibration}
        history.append(entry)
        emit("epoch_complete", **entry)
        if selection > best:
            best = selection
            best_path = WORK / "best"
            model.save_pretrained(best_path, max_shard_size="4GB")
            tokenizer.save_pretrained(best_path)
        save_training(model, tokenizer, optimizer, scheduler, epoch + 1, 0, history)
        write_report(WORK / "training_report.json", {"history": history, "data_manifest": data_manifest})
        # Stop once validation reaches the requested operating point, after a complete epoch.
        if calibration["accepted"]:
            break
    del core, model, optimizer, scheduler
    torch.mps.synchronize()
    torch.mps.empty_cache()
    finalize(tokenizer, data_manifest, history)


def finalize(tokenizer, data_manifest, history):
    model = load_model(WORK / "best", torch.float16)
    model.eval()
    model.requires_grad_(False)
    core = scoring_core(model, tokenizer)
    validation_predictions = score_examples(core, read_jsonl(DATA / "validation.jsonl"))
    calibration = calibrate(validation_predictions)
    # The threshold is locked before test data is read or scored.
    threshold = calibration["threshold"]
    test_predictions = score_examples(core, read_jsonl(DATA / "test.jsonl"))
    test_metrics = metrics(test_predictions, threshold)
    CORE_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CORE_CHECKPOINT, max_shard_size="4GB")
    tokenizer.save_pretrained(CORE_CHECKPOINT)
    (CORE_CHECKPOINT / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2))
    files = {p.name: file_hash(p) for p in CORE_CHECKPOINT.iterdir() if p.is_file() and p.name != "core_manifest.json"}
    manifest = {
        "schema_version": 1, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "max_length": MAX_LENGTH, "query_length": QUERY_LENGTH, "stride": 64,
        "threshold": threshold, "files": files,
        "training": {"seed": SEED, "pairs": 20000, "epochs_completed": len(history),
                     "micro_batch": MICRO_BATCH, "gradient_accumulation": ACCUMULATION,
                     "learning_rate": LR, "precision": "float32", "warmup_ratio": 0.05},
        "deployment_precision": "float16", "validation": calibration,
        "test": test_metrics, "accepted": calibration["accepted"] and test_metrics["accepted"],
    }
    (CORE_CHECKPOINT / "core_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_report(WORK / "trained_predictions.json", {"validation": validation_predictions, "test": test_predictions})
    core.close()
    del core, model
    torch.mps.empty_cache()
    # Development-only baseline. It is never a runtime replacement for the owned core.
    baseline = load_model(BASE_MODEL, torch.float16, BASE_REVISION)
    baseline.eval()
    baseline.requires_grad_(False)
    core = scoring_core(baseline, tokenizer)
    baseline_validation_predictions = score_examples(core, read_jsonl(DATA / "validation.jsonl"))
    baseline_calibration = calibrate(baseline_validation_predictions)
    baseline_test_predictions = score_examples(core, read_jsonl(DATA / "test.jsonl"))
    baseline_test = metrics(baseline_test_predictions, baseline_calibration["threshold"])
    write_report(WORK / "baseline_predictions.json", {"validation": baseline_validation_predictions, "test": baseline_test_predictions})
    report = {"trained": {"validation": calibration, "test": test_metrics},
              "baseline": {"validation": baseline_calibration, "test": baseline_test},
              "history": history, "accepted": manifest["accepted"]}
    write_report(WORK / "evaluation_report.json", report)
    emit("training_complete", checkpoint=str(CORE_CHECKPOINT), **report)
    core.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)


if __name__ == "__main__":
    main()
