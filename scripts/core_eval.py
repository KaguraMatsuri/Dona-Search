import json
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

from dona_search.core import candidate_spans
from dona_search.extract.sentence import select_evidence
from dona_search.interfaces import Document


def score_examples(core, rows):
    predictions = []
    for i, row in enumerate(rows):
        spans = candidate_spans(core.tokenizer, row["query"], row["text"])
        scores = core.score_pairs([(row["query"], row["text"][a:b]) for a, b in spans])
        labels = [int(any(a <= c and b >= d for c, d in row["gold_spans"])) for a, b in spans]
        best = max(range(len(scores)), key=scores.__getitem__, default=None)
        windows = candidate_spans(core.tokenizer, row["query"], row["text"], paragraphs=False)
        window_scores = scores if windows == spans else core.score_pairs([(row["query"], row["text"][a:b]) for a, b in windows])
        window_labels = [int(any(a <= c and b >= d for c, d in row["gold_spans"])) for a, b in windows]
        ndcg = float(ndcg_score([window_labels], [window_scores], k=10)) if len(window_scores) > 1 and any(window_labels) else float(bool(window_labels and window_labels[0]))
        ranked = sorted(range(len(window_scores)), key=window_scores.__getitem__, reverse=True)[:10]
        first = next((rank for rank, index in enumerate(ranked, 1) if window_labels[index]), None)
        doc = Document(row["doc_id"], row["title"], row["text"], "")
        evidences = select_evidence(doc, spans, scores, 0.0)
        ev = evidences[0] if evidences else None
        prediction = {
            "id": row["id"], "source": row["source"], "answerable": row["answerable"],
            "score": scores[best] if best is not None else 0.0,
            "correct": bool(best is not None and labels[best]),
            "span": list(spans[best]) if best is not None else None,
            "ndcg_at_10": ndcg, "mrr_at_10": 1 / first if first else 0.0,
            "candidate_count": len(spans),
            "source_match": bool(ev is None or (0 <= ev.start < ev.end <= len(doc.text) and ev.text == doc.text[ev.start:ev.end])),
        }
        predictions.append(prediction)
        core.cleanup()
        if (i + 1) % 20 == 0:
            print(json.dumps({"event": "evaluation_progress", "completed": i + 1, "total": len(rows)}), flush=True)
    return predictions


def metrics(predictions, threshold):
    answered = [p for p in predictions if p["answerable"]]
    unanswered = [p for p in predictions if not p["answerable"]]
    returned = [p for p in predictions if p["span"] is not None and p["score"] >= threshold]
    covered = [p for p in returned if p["answerable"]]
    false_returns = [p for p in returned if not p["answerable"]]
    coverage = len(covered) / len(answered) if answered else 0.0
    correct_coverage = sum(p["correct"] for p in covered) / len(answered) if answered else 0.0
    evidence_precision = sum(p["correct"] for p in returned) / len(returned) if returned else 0.0
    false_rate = len(false_returns) / len(unanswered) if unanswered else 1.0
    source_match_rate = sum(p["source_match"] for p in returned) / len(returned) if returned else 1.0
    return {
        "threshold": float(threshold), "answerable_count": len(answered),
        "unanswerable_count": len(unanswered), "summary_count": len(returned),
        "answerable_coverage": coverage, "unanswerable_false_summary_rate": false_rate,
        "unanswerable_false_summary_count": len(false_returns),
        "correct_evidence_coverage": correct_coverage,
        "evidence_precision": evidence_precision,
        "source_match_rate": source_match_rate, "generated_summary_characters": 0,
        "ranking_window_ndcg_at_10": float(np.mean([p["ndcg_at_10"] for p in answered])) if answered else 0.0,
        "ranking_window_mrr_at_10": float(np.mean([p["mrr_at_10"] for p in answered])) if answered else 0.0,
        "accepted": bool(answered and unanswered and correct_coverage >= 0.5
                         and evidence_precision == 1.0 and false_rate <= 0.01
                         and source_match_rate == 1.0),
    }


def calibrate(predictions):
    scores = [p["score"] for p in predictions]
    thresholds = sorted({0.0, 1.0, *scores, *(float(np.nextafter(s, np.inf)) for s in scores if s < 1)})
    candidates = [metrics(predictions, threshold) for threshold in thresholds if 0 <= threshold <= 1]
    feasible = [m for m in candidates if m["unanswerable_false_summary_rate"] <= 0.01]
    accepted = [m for m in feasible if m["accepted"]]
    pool = accepted or feasible or candidates
    return max(pool, key=lambda m: (m["correct_evidence_coverage"], m["evidence_precision"], m["answerable_coverage"], m["threshold"]))


def write_report(path, report):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2))
