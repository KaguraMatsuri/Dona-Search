from ..core import CoreModel, candidate_spans
from ..registry import register


@register("reranker", "cross_encoder")
class CrossEncoderReranker:
    def __init__(self, core: CoreModel):
        self.core = core

    def score_batch(self, query: str, items: list[dict]) -> list[float]:
        """items: list of {title, snippet} or {title, text}"""
        pairs, owners = [], []
        for i, item in enumerate(items):
            text = item.get("title", "") + "\n" + (item.get("text") or item.get("snippet", ""))
            for start, end in candidate_spans(self.core.tokenizer, query, text, paragraphs=False):
                pairs.append((query, text[start:end]))
                owners.append(i)
        scores = [0.0] * len(items)
        for owner, score in zip(owners, self.core.score_pairs(pairs)):
            scores[owner] = max(scores[owner], score)
        return scores
