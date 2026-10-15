"""Provenance checks shared by training and resident model loading."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=64)
def _cached_digest(path, size, modified):
    return file_digest(path)


def model_fingerprint(path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {
        '.json', '.safetensors', '.model', '.txt', '.jinja', '.tiktoken',
    } and p.name != 'dona_base_manifest.json')
    if not (path / 'config.json').is_file() or not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('model_files_missing')
    hashes = {}
    for p in files:
        stat = p.stat()
        hashes[p.name] = _cached_digest(str(p.resolve()), stat.st_size, stat.st_mtime_ns)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def validate_adapter(base_path, adapter_path, task, *, allow_external=False, base_fingerprint=None):
    path = Path(adapter_path)
    if allow_external and not (path / 'dona_adapter_manifest.json').is_file():
        config = json.loads((path / 'adapter_config.json').read_text())
        if not isinstance(config, dict) or not (path / 'adapters.safetensors').is_file():
            raise ValueError('adapter_files_missing')
        binding = base_fingerprint or config.get('base_fingerprint')
        if binding is not None:
            if binding != model_fingerprint(base_path):
                raise ValueError('adapter_base_mismatch')
        else:
            source = config.get('model')
            if not isinstance(source, str) or not Path(source).expanduser().is_dir():
                raise ValueError('adapter_base_binding_required')
            if Path(source).expanduser().resolve() != Path(base_path).resolve():
                raise ValueError('adapter_base_mismatch')
        if config.get('task', task) != task:
            raise ValueError('adapter_task_mismatch')
        # MLX-LM's official loader checks parameter names and tensor shapes.
        return config
    manifest = json.loads((path / 'dona_adapter_manifest.json').read_text())
    if manifest.get('task') != task or manifest.get('schema_version') != 1:
        raise ValueError('adapter_task_mismatch')
    if manifest.get('base_fingerprint') != model_fingerprint(base_path):
        raise ValueError('adapter_base_mismatch')
    for name in ('adapter_config.json', 'adapters.safetensors'):
        if manifest.get('files', {}).get(name) != file_digest(path / name):
            raise ValueError('adapter_hash_mismatch')
    return manifest
