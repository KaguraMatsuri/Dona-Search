import pytest
from transformers import AutoTokenizer

from dona_search.core import BASE_MODEL, BASE_REVISION


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
