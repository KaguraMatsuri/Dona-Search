import json
from itertools import permutations, product

import pytest

from dona_search.interfaces import OrganizationDecision, SearchHit
from dona_search.organizer import (
    organization_messages,
    parse_organization,
    selected_evidence,
    snippet_segments,
    validate_decision,
)
from dona_search.query.planner import parse_plan
from dona_search.structured import load_model_json


def hits():
    return [SearchHit(1, '标题 A', 'https://a.test/', '无关第一句。\nSQLite 原始说明。', 'a'),
            SearchHit(2, '标题 B', 'https://b.test/', '另一个网站的原文。', 'b')]


@pytest.mark.parametrize('wrapper', ['{}', '```json\n{}\n```', '  ```json\r\n{}\r\n```  '])
def test_single_json_wrapper_for_both_roles(wrapper):
    assert parse_plan(wrapper.format(json.dumps({'queries': ['SQLite 文档']})), 'raw').queries == ['SQLite 文档']
    result = parse_organization(wrapper.format('{"order": [1, 0], "selections": [2, null]}'), hits())
    assert result.order == [1, 0] and result.selections == [2, None]


def test_valid_query_literals_may_contain_backticks_inside_json_strings():
    query = '如何输出 ```json 原文'
    response = '```json\n' + json.dumps({'queries': [query]}, ensure_ascii=False) + '\n```'
    assert parse_plan(response, query).queries == [query]


@pytest.mark.parametrize('response', [
    'explanation\n{"queries":["a"]}', '```json\n{"queries":["a"]}\n```\nextra',
    '```json\n{"queries":["a"]}\n```\n```json\n{}\n```',
    '```python\n{"queries":["a"]}\n```', '```json {"queries":["a"]}```',
    '{"queries":["a"], "queries":["b"]}', '{"queries":[NaN]}',
])
def test_wrapper_does_not_rescue_invalid_outputs(response):
    with pytest.raises(ValueError):
        parse_plan(response, 'raw')


@pytest.mark.parametrize('order,selections', [
    ([0, 0], [None, None]), ([0, 2], [None, None]), ([True, 0], [None, None]),
    ([0, 1], []), ([0, 1], [True, None]), ([0, 1], [-1, None]),
    ([0, 1], [None, 2]), ([0, 1], ['invented text', None]),
])
def test_invalid_orders_and_source_indices_are_rejected(order, selections):
    with pytest.raises(ValueError):
        validate_decision(OrganizationDecision(order, selections), hits())


def test_indexed_selection_uses_its_own_hit_and_copies_exact_original():
    original = hits()
    decision = parse_organization('{"order":[1,0],"selections":[2,null]}', original)
    selected = selected_evidence(original[0], decision.selections[0])
    assert selected.text == 'SQLite 原始说明。'
    assert selected.text == original[0].snippet[selected.start:selected.end]
    assert selected_evidence(original[1], decision.selections[1]) is None
    with pytest.raises(ValueError):
        parse_organization('{"order":[0,1],"selections":[0,0],"summary":"invented"}', original)


@pytest.mark.parametrize('text', ['', ' \n\t', '版本 3.46 可用。Second sentence!\n尾句', '  原文一。  原文二？  '])
def test_every_segment_is_an_exact_slice_including_unicode_and_whitespace(text):
    segments = snippet_segments(text)
    assert bool(segments) == bool(text.strip())
    for item in segments:
        assert 0 <= item.start < item.end <= len(text)
        assert item.text == text[item.start:item.end]


def test_empty_snippet_requires_null_selection():
    original = [SearchHit(1, 'title', 'https://a.test/', '', 'a')]
    assert parse_organization('{"order":[0],"selections":[null]}', original).selections == [None]
    with pytest.raises(ValueError):
        parse_organization('{"order":[0],"selections":[0]}', original)


@pytest.mark.parametrize('text', ['', ' \n\t', 'single', '  single  ', '第一句。\n第二句！', '  原文一。  原文二？  '])
def test_compact_prompt_preserves_all_evidence_and_original_selection_ids(text):
    originals = [SearchHit(1, 'title', 'https://a.test/', text, 'a')]
    payload = json.loads(organization_messages('query', originals)[1]['content'])
    sent = payload['results'][0]['segments']
    original_segments = snippet_segments(text)
    assert [segment['id'] for segment in sent] == list(range(len(original_segments)))
    visible = ''.join(segment.get('text', '') for segment in sent)
    assert ''.join(visible.split()) == ''.join(text.split())
    for segment in sent:
        evidence = selected_evidence(originals[0], segment['id'])
        if segment.get('whole_snippet'):
            assert evidence.text == text
        else:
            assert segment['text'] == evidence.text
    if len(original_segments) > 1:
        assert 'text' not in sent[0]


def test_json_roundtrip_for_bounded_unicode_and_control_character_domain():
    alphabet = ['中', '😀', '\\', '"', '`', '\n', '\r', '\t', '\0']
    texts = [''.join(chars) for length in range(3) for chars in product(alphabet, repeat=length)]
    texts.extend(['```json\n{}\n```', '  前后空格  '])
    for text in texts:
        value = {'text': text, 'nested': [None, True, False, 0, -1, 1.25, {'value': text}]}
        encoded = json.dumps(value, ensure_ascii=False)
        assert load_model_json(encoded) == value
        assert load_model_json('```json\n' + encoded + '\n```') == value


def test_all_small_permutations_keep_nullable_evidence_bound_to_its_source():
    for count in range(5):
        originals = [SearchHit(i + 1, f'title {i}', f'https://source.test/{i}', f'原文来源 {i}。', 'test') for i in range(count)]
        for order in permutations(range(count)):
            for choices in product([None, 0], repeat=count):
                decision = parse_organization(json.dumps({'order': order, 'selections': choices}), originals)
                rendered = {originals[i].url: selected_evidence(originals[i], decision.selections[i]) for i in decision.order}
                assert set(rendered) == {hit.url for hit in originals}
                for i, hit in enumerate(originals):
                    if choices[i] is None:
                        assert rendered[hit.url] is None
                    else:
                        assert rendered[hit.url].text == f'原文来源 {i}。'
