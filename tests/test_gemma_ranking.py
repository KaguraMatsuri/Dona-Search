import gc
from weakref import ref

import pytest

from dona_search.rerank.gemma import GemmaReranker, permutation_processor, validate_order


@pytest.mark.parametrize('count', [1, 11, 40])
def test_constrained_output_blocks_content_and_duplicate_ids(count):
    import json
    mx = pytest.importorskip('mlx.core')
    class Tokenizer:
        eos_token_id = 127
        def encode(self, text, **kwargs): return [ord(c) for c in text]
    process = permutation_processor(count, Tokenizer())
    expected = '[' + ','.join(str(i) for i in reversed(range(count))) + ']'
    tokens = mx.array([126])
    emitted = []
    for char in expected:
        logits = mx.zeros((1, 128))
        logits[0, 126] = 1000
        logits[0, ord(char)] = 100
        token = int(mx.argmax(process(tokens, logits), axis=-1).item())
        assert token == ord(char)
        emitted.append(chr(token))
        tokens = mx.concatenate([tokens, mx.array([token])])
    assert int(mx.argmax(process(tokens, mx.zeros((1, 128))), axis=-1).item()) == 127
    assert validate_order(json.loads(''.join(emitted)), count) == list(reversed(range(count)))


def test_close_collects_model_cycles_before_clearing_cache():
    events = []
    class Cycle:
        def __init__(self): self.cycle = self
    class Backend:
        def synchronize(self): events.append('synchronize')
        def clear_cache(self): events.append('clear_cache')
    ranker = GemmaReranker.__new__(GemmaReranker)
    ranker.mx = Backend()
    ranker.model, ranker.tokenizer = Cycle(), Cycle()
    model, tokenizer = ref(ranker.model), ref(ranker.tokenizer)
    enabled = gc.isenabled()
    gc.disable()
    try:
        ranker.close()
        assert model() is None and tokenizer() is None
        assert events == ['synchronize', 'clear_cache']
    finally:
        if enabled: gc.enable()
