"""Conservative startup estimates; these are not an allocator hard limit."""
import json
from pathlib import Path

GIB = 1024 ** 3


def recommended_budget():
    import mlx.core as mx
    value = mx.device_info().get('max_recommended_working_set_size')
    if not isinstance(value, int) or value <= 0:
        raise ValueError('memory_budget_unavailable')
    return value


def weight_bytes(path):
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        names = set(json.loads(index.read_text())['weight_map'].values())
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError('model_weight_index_invalid')
        weights = [path / name for name in names]
    else:
        weights = list(path.glob('*.safetensors'))
    if not weights:
        raise ValueError('model_weights_missing')
    return sum(file.stat().st_size for file in weights)


def check_model_memory(models, budget=None, *, conversion=False, extra_bytes=0):
    budget = recommended_budget() if budget is None else budget
    resident, workspace = extra_bytes, 0
    for path, prompt_tokens in models:
        size = weight_bytes(path)
        resident += size
        config = json.loads((Path(path) / 'config.json').read_text())
        text = config.get('text_config', config)
        heads = text.get('num_attention_heads', 0)
        hidden = text.get('hidden_size', 0)
        dimension = text.get('head_dim', hidden // heads if heads else 0)
        kv = 4 * text.get('num_hidden_layers', 0) * text.get('num_key_value_heads', heads) * dimension * (prompt_tokens + 1024)
        prefill = min(prompt_tokens, 2048)
        attention = 4 * heads * prefill * prompt_tokens
        projection = 12 * prefill * max(hidden, text.get('intermediate_size', hidden))
        workspace = max(workspace, kv + attention + projection)
    estimated = resident + workspace + 2 * GIB
    if conversion:
        estimated += resident  # Keep room for conversion outputs and temporary tensors.
    if estimated > budget:
        raise ValueError('memory_budget_exceeded')
    return {'estimated_bytes': estimated, 'budget_bytes': budget}
