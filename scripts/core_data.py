import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from dona_search.core import BASE_MODEL, BASE_REVISION, candidate_spans, file_hash

SEED = 42
SOURCES = {
    "qasper": {
        "repo": "allenai/qasper",
        "revision": "06806e4608976fc2fac0a090ac425d5b2b29caf4",
        "source_revision": "fdc9d8214fbab5dd782958601db4d678e6934a54",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": "Dasigi et al., A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers, 2021",
        "splits": ["train", "validation", "test"],
    },
    "squad_v2": {
        "repo": "rajpurkar/squad_v2",
        "revision": "4866f961e7af023d9a9e7c9e951f34fd7f152818",
        "source_revision": "3ffb306f725f7d2ce8394bc1873b24868140c412",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "citation": "Rajpurkar et al., Know What You Don't Know: Unanswerable Questions for SQuAD, 2018",
        "splits": ["train", "validation"],
    },
}


def context_id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def qasper_examples(rows, rejected):
    for row in rows:
        paragraphs = [p for section in row["full_text"]["paragraphs"] for p in section if p.strip()]
        # The title/abstract are raw source text, and offsets reference this exact assembly.
        text = "\n\n".join([row["title"], row["abstract"], *paragraphs])
        positions = defaultdict(list)
        offset = 0
        for paragraph in [row["title"], row["abstract"], *paragraphs]:
            positions[paragraph].append((offset, offset + len(paragraph)))
            offset += len(paragraph) + 2
        qas = row["qas"]
        for i, query in enumerate(qas["question"]):
            annotations = qas["answers"][i]["answer"]
            if not annotations:
                rejected["qasper_missing_annotation"] += 1
                continue
            statuses = {a["unanswerable"] for a in annotations}
            if len(statuses) != 1:
                rejected["qasper_answerability_conflict"] += 1
                continue
            answerable = not statuses.pop()
            evidence_sets = []
            invalid = False
            for annotation in annotations:
                evidence = annotation["evidence"]
                if any("FLOAT SELECTED" in e for e in evidence):
                    rejected["qasper_nontext_evidence"] += 1
                    invalid = True
                    break
                located = set()
                for evidence_text in evidence:
                    if evidence_text not in positions or len(positions[evidence_text]) != 1:
                        rejected["qasper_evidence_location_mismatch"] += 1
                        invalid = True
                        break
                    located.add(positions[evidence_text][0])
                evidence_sets.append(located)
            if invalid:
                continue
            if answerable:
                if not evidence_sets or any(not s for s in evidence_sets):
                    rejected["qasper_missing_text_evidence"] += 1
                    continue
                if any(s != evidence_sets[0] for s in evidence_sets[1:]):
                    rejected["qasper_evidence_conflict"] += 1
                    continue
                gold = sorted(evidence_sets[0])
            else:
                # Unanswerable annotators can highlight where they looked; those are not positives.
                gold = []
            yield {
                "id": "qasper:" + qas["question_id"][i],
                "source": "qasper", "doc_id": "qasper:" + row["id"],
                "query": query, "title": row["title"], "text": text,
                "answerable": answerable, "gold_spans": gold,
            }


def squad_examples(rows, rejected):
    for row in rows:
        text = row["context"]
        gold = []
        answers = row["answers"]
        if len(answers["text"]) != len(answers["answer_start"]):
            rejected["squad_annotation_shape"] += 1
            continue
        invalid = False
        for answer, start in zip(answers["text"], answers["answer_start"]):
            end = start + len(answer)
            if not answer or start < 0 or text[start:end] != answer:
                rejected["squad_answer_location_mismatch"] += 1
                invalid = True
                break
            gold.append((start, end))
        if invalid:
            continue
        yield {
            "id": "squad_v2:" + row["id"], "source": "squad_v2",
            "doc_id": "squad_v2:" + context_id(text),
            "query": row["question"], "title": row["title"], "text": text,
            "answerable": bool(gold), "gold_spans": sorted(set(gold)),
        }


def split_examples(qasper, squad, rejected):
    splits = {key: list(qasper_examples(qasper[key], rejected)) for key in ("train", "validation", "test")}
    test_squad = list(squad_examples(squad["validation"], rejected))
    test_ids = {row["doc_id"] for row in test_squad}
    splits["test"].extend(test_squad)
    for row in squad_examples(squad["train"], rejected):
        if row["doc_id"] in test_ids:
            rejected["squad_context_overlap_with_test"] += 1
            continue
        # Whole contexts stay together. This never partitions questions from one context.
        bucket = int(hashlib.sha256((str(SEED) + row["doc_id"]).encode()).hexdigest()[:8], 16) % 10
        splits["validation" if bucket == 0 else "train"].append(row)
    ids = {key: {row["doc_id"] for row in rows} for key, rows in splits.items()}
    if any(ids[a] & ids[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("document_split_overlap")
    return splits


def select_evaluation(rows, per_class=100):
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for row in rows:
        by_label[(row["source"], row["answerable"])].append(row)
    selected = []
    for answerable in (False, True):
        pools = [by_label[(source, answerable)] for source in SOURCES]
        for pool in pools:
            rng.shuffle(pool)
        group = []
        while len(group) < per_class and any(pools):
            for pool in pools:
                if pool and len(group) < per_class:
                    group.append(pool.pop())
        if len(group) != per_class:
            raise ValueError("insufficient_evaluation_examples")
        selected.extend(group)
    rng.shuffle(selected)
    return selected


def training_pairs(rows, tokenizer, count=20000):
    """Use only gold-supported windows and explicit unanswerable negatives."""
    rng = random.Random(SEED)
    rows = list(rows)
    rng.shuffle(rows)
    pools = defaultdict(list)
    per_bucket = count // 4
    # Each source contributes where it has unique annotations; no oversampling duplicates.
    for source in SOURCES:
        for row in (r for r in rows if r["source"] == source):
            label = int(row["answerable"])
            task = ("ranking", "evidence")[int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % 2]
            key = (task, label)
            if len(pools[key]) >= per_bucket:
                continue
            spans = candidate_spans(tokenizer, row["query"], row["text"], paragraphs=task == "evidence")
            if label:
                spans = [span for span in spans if any(span[0] <= a and span[1] >= b for a, b in row["gold_spans"])]
            rng.shuffle(spans)
            # Limit per-question dominance while allowing all annotated paper evidence.
            for start, end in spans[:4]:
                if len(pools[key]) < per_bucket:
                    pools[key].append({
                        "query": row["query"], "text": row["text"][start:end],
                        "label": label, "task": task, "source": source,
                        "doc_id": row["doc_id"], "query_id": row["id"],
                        "start": start, "end": end,
                    })
    if any(len(pools[(task, label)]) != per_bucket for task in ("ranking", "evidence") for label in (0, 1)):
        raise ValueError("insufficient_training_pairs")
    result = [row for pool in pools.values() for row in pool]
    rng.shuffle(result)
    return result


def build(output=Path("data/core")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loaded, source_files = {}, {}
    for name, source in SOURCES.items():
        paths = {
            split: hf_hub_download(
                source["repo"], f"{name}/{split}/0000.parquet",
                repo_type="dataset", revision=source["revision"],
            ) for split in source["splits"]
        }
        loaded[name] = load_dataset("parquet", data_files=paths)
        source_files[name] = {split: file_hash(Path(path)) for split, path in paths.items()}
    rejected = Counter()
    splits = split_examples(loaded["qasper"], loaded["squad_v2"], rejected)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    training = training_pairs(splits["train"], tokenizer)
    write_jsonl(output / "train.jsonl", training)
    for split in ("validation", "test"):
        write_jsonl(output / f"{split}.jsonl", select_evaluation(splits[split]))
    manifest = {
        "schema_version": 1, "seed": SEED, "sources": SOURCES,
        "source_files_sha256": source_files,
        "split_documents": {k: len({r["doc_id"] for r in v}) for k, v in splits.items()},
        "split_queries": {k: len(v) for k, v in splits.items()},
        "training_pairs": len(training),
        "training_distribution": dict(Counter(f'{r["source"]}:{r["task"]}:{r["label"]}' for r in training)),
        "rejected": dict(rejected),
        "files": {name: file_hash(output / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl")},
    }
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "data_ready", **manifest}), flush=True)


if __name__ == "__main__":
    build()
