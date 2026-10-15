import json
from contextlib import closing

import pytest

from dona_search.adapters import file_digest, model_fingerprint, validate_adapter
from dona_search.config import validate_config


def make_files(tmp_path):
    base, adapter = tmp_path / 'base', tmp_path / 'adapter'
    base.mkdir()
    adapter.mkdir()
    (base / 'config.json').write_text('{"quantization":{"bits":8,"group_size":64}}')
    (base / 'model.safetensors').write_bytes(b'fixture')
    for name in ['adapter_config.json', 'adapters.safetensors']:
        (adapter / name).write_bytes(b'fixture')
    manifest = {'schema_version': 1, 'task': 'query', 'base_fingerprint': model_fingerprint(base),
                'files': {n: file_digest(adapter / n) for n in ['adapter_config.json', 'adapters.safetensors']}}
    (adapter / 'dona_adapter_manifest.json').write_text(json.dumps(manifest))
    return base, adapter


def test_adapter_rejects_wrong_task_changed_base_and_changed_weights(tmp_path):
    base, adapter = make_files(tmp_path)
    validate_adapter(base, adapter, 'query')
    with pytest.raises(ValueError, match='task_mismatch'):
        validate_adapter(base, adapter, 'ranking')
    (adapter / 'adapters.safetensors').write_bytes(b'different')
    with pytest.raises(ValueError, match='hash_mismatch'):
        validate_adapter(base, adapter, 'query')
    (base / 'config.json').write_text('{"quantization":{"bits":4}}')
    with pytest.raises(ValueError, match='base_mismatch'):
        validate_adapter(base, adapter, 'query')


def test_config_resolves_each_adapter_and_ignores_unused_model(tmp_path):
    (tmp_path / 'rank').mkdir()
    config = validate_config({'schema_version': 3, 'query_model': {'adapter_path': 'missing'},
                              'ranking': {'adapter_path': 'rank'}}, tmp_path, required_models=('ranking',))
    assert config.ranking.adapter_path == tmp_path / 'rank'
    with pytest.raises(ValueError, match='adapter_path_missing'):
        validate_config(config.model_dump(), tmp_path)


def test_query_only_runtime_does_not_load_legacy_core(monkeypatch):
    import dona_search.runtime as module
    events = []
    class Planner:
        def plan(self, query): return query
        def warmup(self): events.append('warmup')
        def cleanup(self): events.append('cleanup')
        def close(self): events.append('close')
    monkeypatch.setattr(module, 'CoreModel', lambda *a, **k: pytest.fail('legacy_core_loaded'))
    with closing(module.ModelRuntime(build_query=Planner, query_only=True)) as runtime:
        assert runtime.call(runtime.query_processor.plan, 'query') == 'query'
        assert runtime.core is None and runtime.ranker is None
    assert events.count('warmup') == 1 and events.count('close') == 1
