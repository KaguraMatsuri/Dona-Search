import json

import pytest

from dona_search.core import CoreModel, candidate_spans, paragraph_spans
from dona_search.extract.sentence import SentenceExtractor, select_evidence
from dona_search.interfaces import Document, SearchResult


def test_whitespace_unicode_and_final_unpunctuated_paragraph(tokenizer):
    text = "  Café keeps π unchanged.\n\n\tFinal paragraph without punctuation  "
    spans = candidate_spans(tokenizer, "What stays unchanged?", text)
    assert spans == paragraph_spans(text)
    doc = Document("fixture:one", "", text, "")
    evidence = select_evidence(doc, spans, [0.1, 0.9], 0.5)[0]
    assert evidence.text == "Final paragraph without punctuation"
    assert evidence.text == text[evidence.start:evidence.end]


def test_late_evidence_and_source_order(tokenizer):
    paragraphs = [f"Irrelevant paragraph {i}." for i in range(25)]
    paragraphs.extend(["The weights remain resident.", "The caches are cleared."])
    text = "\n\n".join(paragraphs)
    class Core:
        threshold = 0.5
        def score_pairs(self, pairs):
            return [0.9 if "remain resident" in t else 0.8 if "cleared" in t else 0.1 for _, t in pairs]
    core = Core()
    core.tokenizer = tokenizer
    evidence = SentenceExtractor(core).extract("What remains resident?", Document("fixture:one", "", text, ""), 2)
    assert len(evidence) == 2
    assert [e.start for e in evidence] == sorted(e.start for e in evidence)
    assert all(e.text == text[e.start:e.end] for e in evidence)
    assert SentenceExtractor(core).extract("", Document("", "", "", "")) == []


def test_overflow_windows_preserve_offsets_and_cover_tail(tokenizer):
    text = "Repeated source phrase, " * 500 + "Tail evidence stays verbatim."
    spans = candidate_spans(tokenizer, "Find tail evidence", text)
    assert len(spans) > 1
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(0 <= a < b <= len(text) for a, b in spans)
    assert all(len(tokenizer("Find tail evidence", text[a:b])["input_ids"]) <= 512 for a, b in spans)


def test_no_evidence_has_no_fabricated_summary():
    result = SearchResult(1, "Original engine snippet", "fixture:one", None)
    assert result.format() == "[1] Original engine snippet\n    fixture:one\n"


def test_unicode_subword_boundaries_fit_scoring_limit(tokenizer):
    text = "English source mentions café,日本語,αβγ and naïve words. " * 180
    spans = candidate_spans(tokenizer, "Which languages?", text)
    assert spans[-1][1] == len(text.rstrip())
    assert all(len(tokenizer("Which languages?", text[a:b])["input_ids"]) <= 512 for a, b in spans)


def test_missing_checkpoint_fails_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr("dona_search.core.AutoModelForSequenceClassification.from_pretrained", lambda *a, **k: pytest.fail("unexpected model load"))
    with pytest.raises(FileNotFoundError):
        CoreModel(tmp_path)


def test_manifest_rejects_invalid_threshold(tmp_path):
    (tmp_path / "core_manifest.json").write_text(json.dumps({"threshold": float("nan")}))
    with pytest.raises(ValueError, match="core_manifest_invalid"):
        CoreModel.validate_checkpoint(tmp_path)
