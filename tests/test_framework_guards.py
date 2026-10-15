import sys
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import yaml

from dona_search import config as config_module
from dona_search.config import (
    ModelSelection,
    PipelineConfigV4,
    read_config,
    validate_config,
)
from dona_search.console import choose_models, save_selection
from dona_search.interfaces import OrganizationDecision, SearchHit
from dona_search.memory import GIB, check_model_memory
from dona_search.organizer import MLXOrganizer, selected_evidence
from dona_search.query.processor import MLXQueryProcessor
from dona_search.runtime import ModelRuntime
from tests.test_console_setup import input_console
from tests.test_model_setup import bundle
from tests.test_pipeline_v4 import components


def test_config_priority_explicit_saved_and_builtin(tmp_path, monkeypatch):
    saved = tmp_path / 'saved' / 'config.yaml'
    monkeypatch.setattr(config_module, 'user_config_path', lambda: saved)
    assert read_config(None)[0]['schema_version'] == 4
    config = PipelineConfigV4(max_queries=1, browser={'kind': 'firefox'})
    save_selection(config, saved)
    assert read_config(None)[0]['max_queries'] == 1
    explicit = tmp_path / 'explicit.yaml'
    explicit.write_text('schema_version: 4\nmax_queries: 2\n')
    assert read_config(explicit)[0]['max_queries'] == 2
    with pytest.raises(FileNotFoundError):
        read_config(tmp_path / 'missing.yaml')
    saved.write_text('not: [valid')
    with pytest.raises(yaml.YAMLError):
        read_config(None)


def test_implicit_v3_upgrade_keeps_relative_sources_and_explicit_compatibility(tmp_path, monkeypatch):
    saved = tmp_path / 'config.yaml'
    saved.write_text('schema_version: 3\nquery_model: {model_path: weights}\nbrowser: {kind: firefox}\n')
    monkeypatch.setattr(config_module, 'user_config_path', lambda: saved)
    raw, base = read_config(None)
    selected = validate_config(raw, base, require_model_paths=False)
    assert selected.schema_version == 4 and selected.query_model.model_path == tmp_path / 'weights'
    assert selected.browser.kind == 'firefox'
    assert read_config(saved)[0]['schema_version'] == 3


@pytest.mark.parametrize('limit', [1, 2, 3])
def test_v4_keeps_original_query_and_respects_total_query_limit(limit):
    from dona_search.interfaces import QueryPlan
    from dona_search.pipeline import SearchPipeline
    planner, organizer, browser = components([])

    class Rewriter(planner):
        def plan(self, raw): return QueryPlan(raw, ['rewritten one', 'rewritten two', 'rewritten three'][:limit])

    with SearchPipeline(PipelineConfigV4(max_queries=limit), build_query=Rewriter,
                        build_organizer=organizer, browser_factory=browser) as pipeline:
        report = pipeline.search_report(' 原始实体 3.46 -排除 ')
    assert report.search_queries == [' 原始实体 3.46 -排除 ', 'rewritten one', 'rewritten two'][:limit]
    assert len(report.tasks) == 3 * limit


@pytest.mark.parametrize('capacity', [2, 3, 4])
def test_batch_merge_preserves_every_result_order_and_verbatim_evidence(capacity):
    hits = [SearchHit(i + 1, str(i), f'https://source.test/{i}', f'原始内容 {i}。', 'test') for i in range(11)]
    scores = {hit.url: (i * 7) % 5 for i, hit in enumerate(hits)}
    organizer = MLXOrganizer.__new__(MLXOrganizer)
    organizer._fits = lambda query, batch: len(batch) <= capacity
    calls = []

    def run(query, batch):
        calls.append(len(batch))
        return OrganizationDecision(sorted(range(len(batch)), key=lambda i: -scores[batch[i].url]),
                                    [0 if scores[hit.url] else None for hit in batch])

    organizer._run = run
    decision = organizer.organize('query', hits)
    assert decision.order == sorted(range(len(hits)), key=lambda i: -scores[hits[i].url])
    assert len(calls) > 1 and max(calls) <= capacity
    for index, hit in enumerate(hits):
        evidence = selected_evidence(hit, decision.selections[index])
        assert (evidence.text if evidence else None) == (hit.snippet if scores[hit.url] else None)


def test_batch_does_not_silently_drop_an_oversized_single_result():
    organizer = MLXOrganizer.__new__(MLXOrganizer)
    organizer._fits = lambda *args: False
    organizer._run = lambda *args: pytest.fail('generation_started')
    with pytest.raises(ValueError, match='organizing_item_context_limit'):
        organizer.organize('query', [SearchHit(1, 't', 'https://a.test/', 'long', 'test')])


def test_memory_preflight_counts_both_resident_models_and_rejects_before_conversion(tmp_path, monkeypatch):
    from dona_search import memory
    from dona_search.config import ArtifactSource
    from dona_search.models import ModelPreparer
    source = bundle(tmp_path / 'weights')
    monkeypatch.setattr(memory, 'recommended_budget', lambda: 24 * GIB)
    result = check_model_memory([(source, 2048), (source, 8192)])
    assert result['estimated_bytes'] >= 2 * GIB + 2 * (source / 'model.safetensors').stat().st_size
    assert check_model_memory([(source, 2048), (source, 8192)], extra_bytes=GIB)['estimated_bytes'] == result['estimated_bytes'] + GIB
    with pytest.raises(ValueError, match='memory_budget_exceeded'):
        check_model_memory([(source, 8192)], GIB)
    selection = ModelSelection(model=str(source))
    config = PipelineConfigV4(query_model=selection, ranking=selection.model_copy(deep=True), memory_budget_bytes=GIB)
    with pytest.raises(ValueError, match='memory_budget_exceeded'):
        ModelPreparer(download=True, cache_dir=tmp_path / 'cache',
            converter=lambda *args, **kwargs: pytest.fail('conversion_started')).prepare(config)
    assert not (tmp_path / 'cache').exists()
    custom = config.model_copy(deep=True)
    custom.query_model.weights = ArtifactSource(path=source)
    with pytest.raises(ValueError, match='memory_budget_exceeded'):
        ModelPreparer().prepare(custom, required_models=('query_model',))


def test_stream_cancellation_closes_the_official_generator(monkeypatch):
    backend = MLXQueryProcessor.__new__(MLXQueryProcessor)
    backend.model, backend.tokenizer, backend.cancel_event = object(), object(), Event()
    closed = []

    def stream(*args, **kwargs):
        try:
            yield SimpleNamespace(text='first')
            backend.cancel_event.set()
            yield SimpleNamespace(text='second')
            pytest.fail('continued_after_cancel')
        finally:
            closed.append(True)

    monkeypatch.setitem(sys.modules, 'mlx_lm', SimpleNamespace(stream_generate=stream))
    monkeypatch.setitem(sys.modules, 'mlx_lm.sample_utils', SimpleNamespace(make_sampler=lambda **kwargs: None))
    with pytest.raises(CancelledError):
        backend.generate('prompt', 10)
    assert closed == [True]


def test_runtime_close_signals_inflight_model_before_releasing_it():
    entered, closed = Event(), Event()

    class Planner:
        def set_cancel_event(self, event): self.cancel_event = event
        def warmup(self): pass
        def cleanup(self): pass
        def close(self): closed.set()
        def plan(self):
            entered.set()
            assert self.cancel_event.wait(2)
            assert not closed.is_set()
            raise CancelledError('inference_cancelled')

    runtime = ModelRuntime(Planner, query_only=True)
    with ThreadPoolExecutor(max_workers=1) as caller:
        running = caller.submit(runtime.call, runtime.query_processor.plan)
        assert entered.wait(2)
        runtime.close()
        with pytest.raises(CancelledError):
            running.result(2)
    assert closed.is_set()


def test_invalid_setup_input_retries_only_that_field(tmp_path, monkeypatch):
    weights, adapter = tmp_path / 'weights', tmp_path / 'adapter'
    weights.mkdir()
    adapter.mkdir()
    values = ['', '2', str(tmp_path / 'missing'), str(weights), '2', str(adapter), 'wrong', 'a' * 64,
              '', '', '', '']
    console = input_console(monkeypatch, values)
    original = PipelineConfigV4(browser={'kind': 'firefox'})
    selected = choose_models(original, console)
    assert selected.browser.kind == 'firefox' and selected.ranking == original.ranking
    assert selected.query_model.weights.path == weights and selected.query_model.adapter.path == adapter
    assert selected.query_model.adapter_base_fingerprint == 'a' * 64
    assert console.file.getvalue().count('输入无效，请重新输入当前项。') == 2
