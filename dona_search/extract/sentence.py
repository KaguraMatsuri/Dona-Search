from ..core import CoreModel, candidate_spans
from ..interfaces import Document, Evidence
from ..registry import register


def select_evidence(doc, spans, scores, threshold, top_k=1):
    if top_k < 1:
        return []
    ranked = sorted(zip(spans, scores), key=lambda item: item[1], reverse=True)
    results = []
    for (start, end), score in ranked:
        if score < threshold:
            continue
        if any(start < evidence.end and end > evidence.start for evidence in results):
            continue
        results.append(Evidence(doc.text[start:end], start, end, int(score * 100), score))
        if len(results) == top_k:
            break
    return sorted(results, key=lambda evidence: evidence.start)

@register("extractor", "sentence")
class SentenceExtractor:
    def __init__(self, core: CoreModel):
        self.core = core

    def extract(self, query: str, doc: Document, top_k: int = 1) -> list[Evidence]:
        if top_k < 1:
            return []
        spans = candidate_spans(self.core.tokenizer, query, doc.text)
        scores = self.core.score_pairs([(query, doc.text[start:end]) for start, end in spans])
        return select_evidence(doc, spans, scores, self.core.threshold, top_k)
