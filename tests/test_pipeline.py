import yaml

from dona_search.interfaces import Document, Evidence, ProcessedQuery, SearchHit
from dona_search.pipeline import SearchPipeline
from dona_search.registry import register


def fixture_pipeline(tmp_path, monkeypatch, tokenizer):
    fetches = []
    class Query:
        def process(self, query):
            return ProcessedQuery(query, [query], {})
    class Core:
        threshold = .5
        def score_pairs(self, pairs):
            return [.9 if "wanted" in text else .1 for _, text in pairs]
    core = Core()
    core.tokenizer = tokenizer
    class Runtime:
        def __init__(self, build_query):
            self.query_processor, self.core = build_query(), core
        def call(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)
        def close(self):
            pass
    @register("query", "fixture")
    class FixtureQuery(Query):
        pass
    @register("search", "fixture_multi_name")
    class Provider:
        def search(self, query, k):
            return [SearchHit(1, "one", "fixture:one", "wanted snippet", "fixture"),
                    SearchHit(2, "two", "fixture:two", "unrelated snippet", "fixture")]
    @register("fetcher", "fixture")
    class Fetcher:
        def fetch(self, url):
            fetches.append(url)
            return Document(url, "", "Unrelated.\n\n" + "x" * 2100 + "\n\nLate wanted evidence." if url.endswith("one") else "No matching evidence", "")
    monkeypatch.setattr("dona_search.pipeline.ModelRuntime", Runtime)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"query_processor": {"type": "query_fixture"},
                                     "search": {"providers": [{"type": "search_fixture_multi_name"}] * 2},
                                     "fetcher": {"type": "fetcher_fixture"}}))
    return SearchPipeline(config), fetches


def test_dedup_two_stage_ranking_and_verbatim_empty_summary(tmp_path, monkeypatch, tokenizer):
    pipeline, fetches = fixture_pipeline(tmp_path, monkeypatch, tokenizer)
    results = pipeline.search("Find wanted evidence", top_k=2)
    assert fetches == ["fixture:one", "fixture:two"]
    assert pipeline.reranker.core is pipeline.extractor.core is pipeline.runtime.core
    assert results[0].summary.text == "Late wanted evidence."
    assert results[0].summary.start > 2000
    assert results[1].summary is None
    assert "..." not in results[1].format()


def test_invalid_model_evidence_is_rejected(tmp_path, monkeypatch, tokenizer):
    pipeline, _ = fixture_pipeline(tmp_path, monkeypatch, tokenizer)
    monkeypatch.setattr(pipeline.extractor, "extract", lambda *a, **k: [Evidence("rewritten text", 0, 10, 99, .99)])
    assert all(result.summary is None for result in pipeline.search("Find wanted evidence"))


def test_source_mismatch_keeps_other_results(tmp_path, monkeypatch, tokenizer):
    pipeline, _ = fixture_pipeline(tmp_path, monkeypatch, tokenizer)
    monkeypatch.setattr(pipeline.fetcher, "fetch", lambda url: Document("fixture:other", "", "wanted", ""))
    results = pipeline.search("Find wanted evidence")
    assert [result.url for result in results] == ["fixture:one", "fixture:two"]
    assert all(result.summary is None for result in results)
    assert pipeline.search("Find wanted evidence", top_k=0) == []


def test_fetch_failure_keeps_link_with_empty_summary(tmp_path, monkeypatch, tokenizer):
    pipeline, _ = fixture_pipeline(tmp_path, monkeypatch, tokenizer)
    monkeypatch.setattr(pipeline.fetcher, "fetch", lambda url: None)
    results = pipeline.search("Find wanted evidence")
    assert len(results) == 2 and all(result.summary is None for result in results)
