"""Prepare complete MLX bundles without modifying custom model sources."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .config import ArtifactSource, ModelSelection

MODEL_FILES = ['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken']


def require_prepared_sources(config):
    if isinstance(config, ModelSelection) and (config.model_path is None
            or (config.adapter is not None and config.adapter_path is None)):
        raise ValueError('model_preparation_required')


def model_source(selection):
    local = Path(selection.model).expanduser()
    if local.is_dir() or selection.model.startswith(('/', '.', '~')):
        return ArtifactSource(path=local)
    return ArtifactSource(repo=selection.model, revision=selection.revision)


def model_cache():
    return Path.home() / 'Library' / 'Caches' / 'dona-search' / 'models'


def bundle_config(path):
    config = json.loads((Path(path) / 'config.json').read_text())
    if not isinstance(config, dict) or not isinstance(config.get('model_type'), str):
        raise TypeError('model_config_invalid')
    return config


def validate_bundle(path):
    from safetensors import safe_open
    path = Path(path)
    config = bundle_config(path)
    if not (any((path / name).is_file() for name in ('tokenizer.json', 'tokenizer.model'))
            or ((path / 'vocab.json').is_file() and (path / 'merges.txt').is_file())):
        raise ValueError('model_tokenizer_missing')
    index_file = path / 'model.safetensors.index.json'
    weights = None
    if index_file.is_file():
        weights = json.loads(index_file.read_text()).get('weight_map')
        if not isinstance(weights, dict) or not weights:
            raise ValueError('model_weight_index_invalid')
        names = set(weights.values())
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError('model_weight_index_invalid')
        files = [path / name for name in sorted(names)]
    else:
        files = sorted(path.glob('*.safetensors'))
    if not files or any(not file.is_file() for file in files):
        raise ValueError('model_weights_missing')
    for file in files:
        with safe_open(str(file), framework='np') as tensors:
            keys = set(tensors.keys())
            if not keys or (weights is not None and any(
                    key not in keys for key, name in weights.items() if name == file.name)):
                raise ValueError('model_weight_index_mismatch')
    return config


def architecture(config):
    text = config.get('text_config', config)
    kind = text.get('model_type', config['model_type']).removesuffix('_text')
    return {'model_type': kind, **{key: text[key] for key in (
        'hidden_size', 'num_hidden_layers', 'num_attention_heads',
        'num_key_value_heads', 'intermediate_size', 'vocab_size',
    ) if key in text}}


def progress_tqdm(callback):
    from huggingface_hub.utils import tqdm

    class DownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            self._unit = kwargs.get('unit', 'it')
            kwargs['disable'] = True
            super().__init__(*args, **kwargs)
            self.refresh()

        def update(self, n=1):
            self.n += n or 0
            self.refresh()

        def refresh(self, *args, **kwargs):
            # File totals are fixed; HF's aggregate byte totals grow during discovery.
            if self._unit != 'B' and hasattr(self, 'n') and hasattr(self, 'total'):
                callback('download', self.n, self.total)

        def __iter__(self):
            try:
                for item in self.iterable:
                    yield item
                    self.update(1)
            finally:
                self.close()

    return DownloadProgress


class ModelPreparer:
    def __init__(self, *, download=False, cache_dir=None, progress=None, downloader=None, converter=None):
        self.download = download
        self.cache_dir = Path(cache_dir) if cache_dir is not None else model_cache()
        self.progress = progress or (lambda role, phase, completed, total: None)
        self.downloader = downloader
        self.converter = converter
        self._artifacts = {}
        self._converted = {}

    def artifact(self, source, role, *, adapter=False, metadata_only=False):
        if source.path is not None:
            path = source.path.expanduser().resolve()
            if not path.is_dir():
                raise ValueError('artifact_path_missing')
            self.progress(role, 'cached', 1, 1)
            return path, source.model_copy(update={'path': path})
        key = (source.repo, source.revision, adapter, metadata_only)
        if key in self._artifacts:
            self.progress(role, 'cached', 1, 1)
            return self._artifacts[key]
        if self.downloader is None:
            from huggingface_hub import snapshot_download
            self.downloader = snapshot_download
        options = {'revision': source.revision or 'main',
                   'allow_patterns': ['config.json'] if metadata_only else MODEL_FILES}
        try:
            path = Path(self.downloader(source.repo, local_files_only=True, **options))
            if metadata_only:
                bundle_config(path)
            elif not adapter:
                validate_bundle(path)
            elif adapter and not all((path / name).is_file() for name in ('adapter_config.json', 'adapters.safetensors')):
                raise ValueError('adapter_files_missing')
            self.progress(role, 'cached', 1, 1)
        except (OSError, ValueError):
            if not self.download:
                raise ValueError('model_preparation_required') from None
            self.progress(role, 'download', 0, None)
            path = Path(self.downloader(source.repo, local_files_only=False, max_workers=1,
                tqdm_class=progress_tqdm(lambda phase, n, total: self.progress(role, phase, n, total)), **options))
            if metadata_only:
                bundle_config(path)
        # Hugging Face snapshots are stored under an immutable resolved commit id.
        revision = path.name if len(path.name) == 40 and all(c in '0123456789abcdef' for c in path.name) else source.revision
        if revision is None:
            raise ValueError('artifact_revision_unresolved')
        result = (path, source.model_copy(update={'revision': revision}))
        self._artifacts[key] = result
        return result

    def quantized(self, source, path, role, budget=None):
        from .adapters import model_fingerprint
        key = (source.repo, source.revision) if source.repo is not None else (str(path), model_fingerprint(path))
        if key in self._converted:
            self.progress(role, 'cached', 1, 1)
            return self._converted[key]
        identity = hashlib.sha256(json.dumps([*key, 8, 64]).encode()).hexdigest()[:24]
        destination = self.cache_dir / identity
        candidates = [destination]
        if source.repo is not None:
            candidates.append(Path(__file__).resolve().parents[1] / 'models' / (source.repo.rsplit('/', 1)[-1] + '-8bit'))
        for candidate in candidates:
            manifest_file = candidate / 'dona_base_manifest.json'
            if not manifest_file.is_file():
                continue
            manifest = json.loads(manifest_file.read_text())
            if manifest.get('bits') != 8 or manifest.get('group_size') != 64:
                raise ValueError('converted_model_settings_mismatch')
            if manifest.get('source_fingerprint') != model_fingerprint(path):
                if candidate == destination:
                    raise ValueError('converted_model_source_mismatch')
                continue
            if manifest.get('fingerprint') != model_fingerprint(candidate):
                raise ValueError('converted_model_hash_mismatch')
            config = validate_bundle(candidate)
            quant = config.get('quantization', config.get('quantization_config', {}))
            if quant.get('bits') != 8 or quant.get('group_size') != 64:
                raise ValueError('converted_model_settings_mismatch')
            self._converted[key] = candidate
            self.progress(role, 'cached', 1, 1)
            return candidate
        if not self.download:
            raise ValueError('model_preparation_required')
        if destination.exists():
            raise ValueError('converted_model_cache_incomplete')
        self.progress(role, 'convert', 0, None)
        from .memory import check_model_memory
        check_model_memory([(path, 0)], budget, conversion=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.prepare-', dir=self.cache_dir))
        try:
            output = staging / 'model'
            if self.converter is None:
                self.converter = convert_default_weights
            self.converter(str(path), str(output), quantize=True, q_bits=8, q_group_size=64,
                           q_mode='affine', trust_remote_code=False)
            converted_config = validate_bundle(output)
            quant = converted_config.get('quantization', converted_config.get('quantization_config', {}))
            if quant.get('bits') != 8 or quant.get('group_size') != 64:
                raise ValueError('converted_model_settings_mismatch')
            manifest = {'schema_version': 1, 'source_fingerprint': model_fingerprint(path),
                        'fingerprint': model_fingerprint(output), 'bits': 8, 'group_size': 64}
            (output / 'dona_base_manifest.json').write_text(json.dumps(manifest))
            output.rename(destination)
        finally:
            shutil.rmtree(staging)
        self._converted[key] = destination
        return destination

    def prepare(self, config, required_models=('query_model', 'ranking')):
        required_models = tuple(required_models)
        resolved, pinned = config.model_copy(deep=True), config.model_copy(deep=True)
        for role in required_models:
            selected = getattr(pinned, role)
            self.progress(role, 'verify', 0, None)
            source = selected.weights
            if source is None:
                source = model_source(selected)
            path, actual = self.artifact(source, role)
            self.progress(role, 'verify', 0, None)
            model_config = validate_bundle(path)
            if selected.weights is not None:
                expected_source = model_source(selected)
                same_repo = (source.repo is not None and expected_source.repo == source.repo
                             and expected_source.revision == actual.revision)
                same_path = (source.path is not None and expected_source.path is not None
                             and expected_source.path.resolve() == source.path.resolve())
                if same_repo or same_path:
                    expected = model_config
                else:
                    expected_path, expected_pin = self.artifact(expected_source, role, metadata_only=True)
                    expected = bundle_config(expected_path)
                    if expected_pin.repo is not None:
                        selected.revision = expected_pin.revision
                if architecture(expected) != architecture(model_config):
                    raise ValueError('model_weights_architecture_mismatch')
                selected.weights = actual
            else:
                if actual.repo is not None:
                    selected.revision = actual.revision
                quant = model_config.get('quantization', model_config.get('quantization_config', {}))
                if quant and quant.get('bits') != 8:
                    raise ValueError('default_weights_require_eight_bit')
                if not quant:
                    path = self.quantized(actual, path, role, config.memory_budget_bytes)
            loaded = selected.model_copy(deep=True)
            loaded.weights = ArtifactSource(path=path)
            if selected.adapter is not None:
                adapter_path, pinned_adapter = self.artifact(selected.adapter, role, adapter=True)
                from .adapters import validate_adapter
                validate_adapter(path, adapter_path, 'query' if role == 'query_model' else 'organizing',
                                 allow_external=True, base_fingerprint=selected.adapter_base_fingerprint)
                selected.adapter = pinned_adapter
                loaded.adapter = ArtifactSource(path=adapter_path)
            setattr(resolved, role, loaded)
            self.progress(role, 'prepared', 1, 1)
        if required_models:
            from .memory import check_model_memory, weight_bytes
            adapters = sum(weight_bytes(getattr(resolved, role).adapter_path) for role in required_models
                           if getattr(resolved, role).adapter_path is not None)
            check_model_memory([(getattr(resolved, role).model_path, getattr(resolved, role).max_prompt_tokens)
                                for role in required_models], config.memory_budget_bytes, extra_bytes=adapters)
        return resolved, pinned


def convert_default_weights(*args, **kwargs):
    import gc

    import mlx.core as mx
    from mlx_lm.convert import convert

    try:
        convert(*args, **kwargs)
    finally:
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
