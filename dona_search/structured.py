"""Strict model protocol parsing, with an optional single Markdown wrapper."""
import json
import re


def load_model_json(response):
    if not isinstance(response, str):
        raise TypeError('invalid_model_json')
    value = response.strip()
    if value.startswith('```'):
        match = re.fullmatch(r'```json[ \t]*\r?\n(.*?)\r?\n```', value, re.DOTALL)
        if match is None:
            raise ValueError('invalid_model_json_wrapper')
        value = match.group(1)

    def object_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('duplicate_model_json_key')
            result[key] = item
        return result

    def invalid_constant(value):
        raise ValueError('invalid_model_json_constant')

    return json.loads(value, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
