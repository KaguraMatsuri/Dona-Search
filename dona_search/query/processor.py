import gc
import json
from concurrent.futures import CancelledError
from threading import Event

from ..interfaces import ProcessedQuery
from ..registry import register


@register("query", "mlx")
class MLXQueryProcessor:
    def __init__(self, model: str = "google/gemma-4-E4B-it", *, adapter_path=None):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.cancel_event = Event()
        mx.set_cache_limit(0)
        options = {'trust_remote_code': False, 'tokenizer_config': {'trust_remote_code': False}}
        if adapter_path is not None:
            options['adapter_path'] = str(adapter_path)
        self.model, self.tokenizer = load(model, lazy=False, **options)
        self.model.eval()
        self.model.freeze()
        mx.eval(self.model.parameters())

    def generate(self, prompt, max_tokens):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if self.cancel_event.is_set():
            raise CancelledError('inference_cancelled')
        stream = stream_generate(self.model, self.tokenizer, prompt=prompt,
                                 max_tokens=max_tokens, sampler=make_sampler(temp=0))
        parts = []
        try:
            for response in stream:
                if self.cancel_event.is_set():
                    raise CancelledError('inference_cancelled')
                parts.append(response.text)
        finally:
            stream.close()
        return ''.join(parts)

    def process(self, query: str) -> ProcessedQuery:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        instruction = (
            "Extract search keywords faithfully from the user query. Return one JSON "
            "object only, without Markdown or code fences. It must contain exactly "
            "keywords (a nonempty array of strings) and filters (an object with time "
            "and type). time must be recent only when the user explicitly requests "
            "recent information, otherwise any. type must be article only when the "
            "user explicitly requests articles, otherwise any. Preserve named entities, "
            "exact identifiers, versions and dates in keywords. Preserve explicit "
            "exclusions as negative keywords with a leading hyphen, for example "
            "-Django. Do not invent extra constraints.\n\nQuery: " + query
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(self.tokenizer.encode(prompt)) > 2048:
            raise ValueError("query_token_limit")
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=256,
            sampler=make_sampler(temp=0),
            verbose=False,
        )
        try:
            data = json.loads(response)
            if not isinstance(data, dict) or set(data) != {"keywords", "filters"}:
                raise ValueError
            keywords, filters = data["keywords"], data["filters"]
            if (
                not isinstance(keywords, list)
                or not keywords
                or any(not isinstance(k, str) or not k.strip() for k in keywords)
                or not isinstance(filters, dict)
                or set(filters) != {"time", "type"}
                or filters["time"] not in {"recent", "any"}
                or filters["type"] not in {"article", "any"}
            ):
                raise ValueError
            return ProcessedQuery(query, keywords, filters)
        except (ValueError, TypeError):
            return ProcessedQuery(query, [query], {})

    def warmup(self) -> None:
        self.process("Find information about SQLite transactions")

    def cleanup(self) -> None:
        self.mx.synchronize()
        self.mx.clear_cache()

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        self.cleanup()
