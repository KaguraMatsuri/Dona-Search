from types import SimpleNamespace

import pytest
import torch

from dona_search.core import CoreModel
from dona_search.rerank.cross_encoder import CrossEncoderReranker


def test_microbatch_bounds_and_score_order(tokenizer):
    batches = []
    class Model:
        def __call__(self, **encoded):
            assert not torch.is_grad_enabled()
            count, length = encoded["input_ids"].shape
            assert length <= 512
            offset = sum(batches)
            batches.append(count)
            return SimpleNamespace(logits=torch.arange(offset, offset + count).float().unsqueeze(-1))
    core = CoreModel.__new__(CoreModel)
    core.model, core.tokenizer, core.device, core.batch_size = Model(), tokenizer, "cpu", 8
    scores = core.score_pairs([("Find evidence", f"Candidate {i}.") for i in range(17)])
    assert batches == [8, 8, 1]
    assert scores == torch.sigmoid(torch.arange(17).float()).tolist()


def test_ranking_batches_documents_together_and_keeps_input_order(tokenizer):
    calls = []
    class Core:
        def score_pairs(self, pairs):
            calls.append(pairs)
            return [.9 if "match" in text else .1 for _, text in pairs]
    core = Core()
    core.tokenizer = tokenizer
    reranker = CrossEncoderReranker(core)
    scores = reranker.score_batch("Find match", [{"title": "other", "snippet": "one"},
                                                {"title": "match", "snippet": "two"}])
    assert scores == [.1, .9] and len(calls) == 1 and len(calls[0]) == 2
    assert reranker.score_batch("", []) == []


def test_nonfinite_logits_are_rejected(tokenizer):
    class Model:
        def __call__(self, **encoded):
            return SimpleNamespace(logits=torch.tensor([[float("inf")]]))
    core = CoreModel.__new__(CoreModel)
    core.model, core.tokenizer, core.device, core.batch_size = Model(), tokenizer, "cpu", 8
    with pytest.raises(ValueError, match="core_nonfinite_logits"):
        core.score_pairs([("Find evidence", "Some source text")])
