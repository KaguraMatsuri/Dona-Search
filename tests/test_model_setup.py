import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from dona_search.config import (
    ArtifactSource,
    ModelSelection,
    PipelineConfigV4,
    validate_config,
)
from dona_search.models import ModelPreparer, progress_tqdm, validate_bundle


def bundle(path, *, quantized=False, hidden=4):
    path.mkdir(parents=True)
    config = {'model_type': 'qwen3', 'hidden_size': hidden, 'vocab_size': 4}
    if quantized:
        config['quantization'] = {'bits': 8, 'group_size': 64}
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'tokenizer.json').write_text('{}')
    save_file({'weight': np.zeros((4, hidden), np.float32)}, str(path / 'model.safetensors'))
    return path


def test_shared_default_download_and_conversion_are_reused_across_roles_and_launches(tmp_path):
    source = bundle(tmp_path / ('a' * 40))
    calls, conversions, updates = [], [], []
    def downloader(repo, **kwargs):
        calls.append(kwargs['local_files_only'])
        if kwargs['local_files_only']: raise FileNotFoundError('not_cached')
        assert kwargs['max_workers'] == 1
        return str(source)
    def converter(path, target, **kwargs):
        conversions.append(kwargs)
        shutil.copytree(path, target)
        config = json.loads((Path(target) / 'config.json').read_text())
        config['quantization'] = {'bits': 8, 'group_size': 64}
        (Path(target) / 'config.json').write_text(json.dumps(config))
    preparer = ModelPreparer(download=True, cache_dir=tmp_path / 'cache', downloader=downloader,
                             converter=converter, progress=lambda *args: updates.append(args))
    resolved, pinned = preparer.prepare(PipelineConfigV4())
    assert calls == [True, False] and len(conversions) == 1
    assert conversions[0]['q_bits'] == 8 and conversions[0]['trust_remote_code'] is False
    assert resolved.query_model.model_path == resolved.ranking.model_path
    assert resolved.query_model is not resolved.ranking
    assert pinned.query_model.weights is None and pinned.query_model.revision == 'a' * 40
    assert ('ranking', 'cached', 1, 1) in updates
    cached = ModelPreparer(cache_dir=tmp_path / 'cache', downloader=lambda *args, **kwargs: str(source),
                           converter=lambda *args, **kwargs: pytest.fail('second_conversion'))
    again, _ = cached.prepare(pinned)
    assert again.ranking.model_path == resolved.ranking.model_path


def test_custom_complete_bundle_keeps_its_precision_and_original_files(tmp_path):
    source = bundle(tmp_path / 'custom', quantized=True)
    config_file = source / 'config.json'
    config = json.loads(config_file.read_text())
    config['quantization']['bits'] = 4
    config_file.write_text(json.dumps(config))
    original = config_file.read_bytes()
    selection = ModelSelection(model=str(source), weights=ArtifactSource(path=source))
    config = PipelineConfigV4(query_model=selection, ranking=selection.model_copy(deep=True))
    resolved, pinned = ModelPreparer(downloader=lambda *args, **kwargs: pytest.fail('network'),
        converter=lambda *args, **kwargs: pytest.fail('conversion')).prepare(config)
    assert resolved.ranking.model_path == source
    assert pinned.ranking.weights.path == source and config_file.read_bytes() == original


def test_different_model_size_cannot_use_same_family_weights(tmp_path):
    model, wrong = bundle(tmp_path / 'base'), bundle(tmp_path / 'wrong', hidden=8)
    config = PipelineConfigV4(query_model=ModelSelection(model=str(model), weights=ArtifactSource(path=wrong)))
    with pytest.raises(ValueError, match='architecture_mismatch'):
        ModelPreparer().prepare(config, required_models=('query_model',))


def test_sdk_preparation_never_downloads_or_converts_implicitly(tmp_path):
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs['local_files_only'])
        raise FileNotFoundError('not_cached')
    with pytest.raises(ValueError, match='preparation_required'):
        ModelPreparer(downloader=download).prepare(PipelineConfigV4())
    assert calls == [True]


def test_missing_shard_and_misbound_index_are_rejected(tmp_path):
    path = bundle(tmp_path / 'base')
    index = path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'weight': 'missing.safetensors'}}))
    with pytest.raises(ValueError, match='weights_missing'): validate_bundle(path)
    index.write_text(json.dumps({'weight_map': {'invented': 'model.safetensors'}}))
    with pytest.raises(ValueError, match='index_mismatch'): validate_bundle(path)
    index.write_text(json.dumps({'weight_map': {'weight': '../other.safetensors'}}))
    with pytest.raises(ValueError, match='index_invalid'): validate_bundle(path)


def test_v4_relative_paths_and_search_only_missing_sources(tmp_path):
    raw = {'schema_version': 4, 'query_model': {'weights': {'path': 'missing'}}}
    config = validate_config(raw, tmp_path, require_model_paths=False)
    assert config.query_model.model_path == tmp_path / 'missing'
    with pytest.raises(ValueError, match='artifact_path_missing'):
        validate_config(raw, tmp_path)


def test_local_model_identity_is_resolved_relative_to_config_directory(tmp_path):
    source = bundle(tmp_path / 'my-model')
    raw = {'schema_version': 4, 'query_model': {'model': './my-model', 'weights': {'path': 'my-model'}}}
    config = validate_config(raw, tmp_path)
    assert config.query_model.model == str(source)
    resolved, _ = ModelPreparer().prepare(config, required_models=('query_model',))
    assert resolved.query_model.model_path == source


def test_external_adapter_needs_verifiable_base_binding_and_keeps_dona_checks(tmp_path):
    from dona_search.adapters import model_fingerprint, validate_adapter
    base = bundle(tmp_path / 'base')
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}')
    save_file({'layer.lora_a': np.zeros((4, 1), np.float32)}, str(adapter / 'adapters.safetensors'))
    with pytest.raises(ValueError, match='binding_required'):
        validate_adapter(base, adapter, 'query', allow_external=True)
    validate_adapter(base, adapter, 'query', allow_external=True, base_fingerprint=model_fingerprint(base))
    with pytest.raises(ValueError, match='base_mismatch'):
        validate_adapter(base, adapter, 'query', allow_external=True, base_fingerprint='0' * 64)
    (adapter / 'dona_adapter_manifest.json').write_text('{"task":"ranking","schema_version":1}')
    with pytest.raises(ValueError, match='task_mismatch'):
        validate_adapter(base, adapter, 'query', allow_external=True, base_fingerprint=model_fingerprint(base))


@pytest.mark.parametrize('role', ['planner', 'organizer'])
@pytest.mark.parametrize('source', ['remote_weights', 'remote_adapter'])
def test_native_loaders_cannot_ignore_unprepared_custom_sources(role, source, tmp_path, monkeypatch):
    from dona_search.organizer import MLXOrganizer
    from dona_search.query.planner import MLXQueryPlanner
    config = ModelSelection(weights=ArtifactSource(repo='custom/weights')) if source == 'remote_weights' else ModelSelection(
        weights=ArtifactSource(path=tmp_path), adapter=ArtifactSource(repo='custom/adapter'))
    def forbidden(*args, **kwargs): pytest.fail('unselected_model_loaded')
    monkeypatch.setattr('dona_search.query.planner.MLXQueryProcessor', forbidden)
    monkeypatch.setattr('dona_search.organizer.MLXQueryProcessor', forbidden)
    with pytest.raises(ValueError, match='model_preparation_required'):
        (MLXQueryPlanner if role == 'planner' else MLXOrganizer)(config)


@pytest.mark.parametrize('source', [{}, {'repo': 'repo', 'path': '/tmp'}, {'path': '/tmp', 'revision': 'main'}])
def test_sources_are_unambiguous(source):
    with pytest.raises(ValueError): ArtifactSource.model_validate(source)


def test_file_progress_uses_real_counts_without_aggregate_byte_percentages():
    updates = []
    bar = progress_tqdm(lambda *args: updates.append(args))
    with bar(total=3, unit='it') as files:
        files.update(1)
        files.update(2)
    assert ('download', 1, 3) in updates and updates[-1] == ('download', 3, 3)
    assert list(bar(iter([1, 2, 3]), total=3)) == [1, 2, 3]
    assert updates[-1] == ('download', 3, 3)
    previous = list(updates)
    with bar(total=0, unit='B') as bytes_bar:
        bytes_bar.total = 100
        bytes_bar.update(50)
    assert updates == previous


@pytest.mark.parametrize('download', [False, True])
def test_partial_metadata_cache_is_recovered_only_when_download_is_allowed(tmp_path, download):
    weights = bundle(tmp_path / 'custom', quantized=True)
    metadata = tmp_path / ('a' * 40)
    metadata.mkdir()
    (metadata / 'tokenizer.json').write_text('{}')
    calls = []

    def downloader(repo, **kwargs):
        calls.append(kwargs['local_files_only'])
        assert kwargs['allow_patterns'] == ['config.json']
        if not kwargs['local_files_only']:
            shutil.copy2(weights / 'config.json', metadata / 'config.json')
        return str(metadata)

    config = PipelineConfigV4(query_model=ModelSelection(
        model='custom/base', weights=ArtifactSource(path=weights)))
    preparer = ModelPreparer(download=download, downloader=downloader)
    if download:
        resolved, pinned = preparer.prepare(config, required_models=('query_model',))
        assert resolved.query_model.model_path == weights
        assert pinned.query_model.revision == 'a' * 40
        assert calls == [True, False]
    else:
        with pytest.raises(ValueError, match='model_preparation_required'):
            preparer.prepare(config, required_models=('query_model',))
        assert calls == [True]


def test_downloaded_metadata_must_be_valid_before_being_cached(tmp_path):
    metadata = tmp_path / ('a' * 40)
    metadata.mkdir()
    preparer = ModelPreparer(download=True, downloader=lambda *args, **kwargs: str(metadata))
    with pytest.raises(FileNotFoundError):
        preparer.artifact(ArtifactSource(repo='custom/base'), 'query_model', metadata_only=True)
    assert not preparer._artifacts
