import json

import pytest

from dona_search.query.processor import MLXQueryProcessor


@pytest.mark.parametrize("response", ["not JSON", "[]", '{"keywords":[],"filters":{}}',
                                     '{"keywords":["Apollo"],"filters":{"time":"2015","type":"any"}}',
                                     '{"keywords":["Python"],"filters":{"time":"any","type":"any"},"negative_keywords":["Django"]}'])
def test_invalid_output_preserves_original_query(monkeypatch, response):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "fixture prompt"
        def encode(self, text):
            return [1]
    processor = MLXQueryProcessor.__new__(MLXQueryProcessor)
    processor.model, processor.tokenizer = object(), Tokenizer()
    monkeypatch.setattr("mlx_lm.generate", lambda *args, **kwargs: response)
    query = "Find 2015 Apollo articles excluding Django"
    result = processor.process(query)
    assert result.raw == query and result.keywords == [query] and result.filters == {}


def test_valid_query_fields(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "fixture prompt"
        def encode(self, text):
            return [1]
    processor = MLXQueryProcessor.__new__(MLXQueryProcessor)
    processor.model, processor.tokenizer = object(), Tokenizer()
    output = {"keywords": ["Apollo", "2015"], "filters": {"time": "any", "type": "article"}}
    monkeypatch.setattr("mlx_lm.generate", lambda *args, **kwargs: json.dumps(output))
    result = processor.process("Find 2015 Apollo articles")
    assert result.keywords == output["keywords"] and result.filters == output["filters"]
