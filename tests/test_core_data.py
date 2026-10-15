from collections import Counter

import pytest

from scripts.core_data import qasper_examples, squad_examples
from scripts.core_eval import calibrate, metrics


def test_squad_uses_answer_position_instead_of_repeated_string():
    rows = [{"id": "1", "title": "", "context": "same answer. same answer.", "question": "Which one?",
             "answers": {"text": ["same answer"], "answer_start": [13]}}]
    result = next(squad_examples(rows, Counter()))
    assert result["gold_spans"] == [(13, 24)]
    rows[0]["answers"] = {"text": [], "answer_start": []}
    result = next(squad_examples(rows, Counter()))
    assert not result["answerable"] and result["gold_spans"] == []


def test_invalid_answer_position_is_excluded():
    rejected = Counter()
    rows = [{"id": "1", "title": "", "context": "actual text", "question": "Which one?",
             "answers": {"text": ["absent"], "answer_start": [0]}}]
    assert list(squad_examples(rows, rejected)) == []
    assert rejected["squad_answer_location_mismatch"] == 1


def test_qasper_does_not_train_on_free_form_answers():
    row = {"id": "paper", "title": "Title", "abstract": "Abstract", "full_text": {"paragraphs": [["Raw evidence."]]},
           "qas": {"question": ["What evidence?"], "question_id": ["q1"],
                   "answers": [{"answer": [{"unanswerable": False, "evidence": ["Raw evidence."],
                                            "free_form_answer": "Made-up rewritten target."}]}]}}
    result = next(qasper_examples([row], Counter()))
    start, end = result["gold_spans"][0]
    assert result["text"][start:end] == "Raw evidence."
    assert "Made-up" not in str(result)


def test_qasper_conflicting_annotations_and_figures_are_excluded():
    first = {"unanswerable": False, "evidence": ["Raw evidence."]}
    second = {"unanswerable": True, "evidence": []}
    row = {"id": "paper", "title": "Title", "abstract": "Abstract", "full_text": {"paragraphs": [["Raw evidence."]]},
           "qas": {"question": ["What evidence?"], "question_id": ["q1"],
                   "answers": [{"answer": [first, second]}]}}
    rejected = Counter()
    assert list(qasper_examples([row], rejected)) == []
    assert rejected["qasper_answerability_conflict"] == 1
    row["qas"]["answers"][0]["answer"] = [{"unanswerable": False, "evidence": ["FLOAT SELECTED: table"]}]
    assert list(qasper_examples([row], rejected)) == []
    assert rejected["qasper_nontext_evidence"] == 1


def test_calibration_respects_false_return_limit_and_reports_failure():
    def prediction(answerable, score):
        return {"answerable": answerable, "score": score, "correct": answerable,
                "span": [0, 10], "source_match": True, "ndcg_at_10": 1, "mrr_at_10": 1}
    rows = [prediction(False, .8)] * 100 + [prediction(True, .9)] * 60 + [prediction(True, .7)] * 40
    operating_point = calibrate(rows)
    assert operating_point["accepted"]
    assert operating_point["answerable_coverage"] == .6
    assert operating_point["unanswerable_false_summary_rate"] == 0
    failed = metrics(rows, .95)
    assert not failed["accepted"] and failed["answerable_coverage"] == 0


def test_calibration_rejects_coverage_without_enough_correct_evidence():
    def prediction(answerable, score, correct):
        return {"answerable": answerable, "score": score, "correct": correct,
                "span": [0, 10], "source_match": True, "ndcg_at_10": 1, "mrr_at_10": 1}
    rows = ([prediction(False, .8, False)] * 100
            + [prediction(True, .9, True)] * 45
            + [prediction(True, .85, False)] * 10
            + [prediction(True, .7, False)] * 45)
    result = calibrate(rows)
    assert not result["accepted"]
    assert result["correct_evidence_coverage"] == .45


@pytest.mark.parametrize("correct_count, accepted", [(0, False), (49, False), (50, False), (99, False), (100, True)])
def test_acceptance_requires_perfect_evidence_precision(correct_count, accepted):
    def prediction(answerable, correct):
        return {"answerable": answerable, "score": .9 if answerable else .1,
                "correct": correct, "span": [0, 10], "source_match": True,
                "ndcg_at_10": int(correct), "mrr_at_10": int(correct)}
    rows = ([prediction(True, True)] * correct_count
            + [prediction(True, False)] * (100 - correct_count)
            + [prediction(False, False)] * 100)
    result = metrics(rows, .5)
    assert result["answerable_coverage"] == 1.0
    assert result["source_match_rate"] == 1.0
    assert result["unanswerable_false_summary_rate"] == 0
    assert result["correct_evidence_coverage"] == correct_count / 100
    assert result["evidence_precision"] == correct_count / 100
    assert result["accepted"] is accepted


@pytest.mark.parametrize("returned_count, accepted", [(49, False), (50, True)])
def test_perfect_precision_still_requires_minimum_coverage(returned_count, accepted):
    def prediction(answerable, score):
        return {"answerable": answerable, "score": score, "correct": answerable,
                "span": [0, 10], "source_match": True, "ndcg_at_10": 1, "mrr_at_10": 1}
    rows = ([prediction(True, .9)] * returned_count
            + [prediction(True, .1)] * (100 - returned_count)
            + [prediction(False, .1)] * 100)
    result = metrics(rows, .5)
    assert result["evidence_precision"] == 1.0
    assert result["correct_evidence_coverage"] == returned_count / 100
    assert result["accepted"] is accepted


def test_calibration_prefers_perfect_precision_over_higher_coverage():
    def prediction(answerable, score, correct):
        return {"answerable": answerable, "score": score, "correct": correct,
                "span": [0, 10], "source_match": True, "ndcg_at_10": int(correct),
                "mrr_at_10": int(correct)}
    rows = ([prediction(True, .95, True)] * 60
            + [prediction(True, .8, True)] * 39
            + [prediction(True, .85, False)]
            + [prediction(False, .1, False)] * 100)
    result = calibrate(rows)
    assert result["accepted"]
    assert result["correct_evidence_coverage"] == .6
    assert result["evidence_precision"] == 1.0


def test_calibration_finds_valid_correct_evidence_operating_point():
    def prediction(answerable, score, correct):
        return {"answerable": answerable, "score": score, "correct": correct,
                "span": [0, 10], "source_match": True, "ndcg_at_10": int(correct),
                "mrr_at_10": int(correct)}
    rows = ([prediction(False, .8, False)] * 100
            + [prediction(True, .9, True)] * 45
            + [prediction(True, .85, True)] * 5
            + [prediction(True, .7, True)] * 50)
    result = calibrate(rows)
    assert result["accepted"]
    assert result["correct_evidence_coverage"] == .5
    assert result["evidence_precision"] == 1.0
    assert result["unanswerable_false_summary_rate"] == 0
