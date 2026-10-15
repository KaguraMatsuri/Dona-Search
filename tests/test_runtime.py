from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from weakref import ref

import pytest

from dona_search.runtime import ModelRuntime


def test_single_worker_keeps_instances_and_cleans_after_failure(monkeypatch):
    events = []
    class Model:
        def __init__(self, *args):
            events.append(("load", get_ident()))
        def warmup(self):
            events.append(("warmup", get_ident()))
        def cleanup(self):
            events.append(("cleanup", get_ident()))
        def close(self):
            events.append(("close", get_ident()))
    monkeypatch.setattr("dona_search.runtime.CoreModel", Model)
    runtime = ModelRuntime(Model)
    original = (id(runtime.query_processor), id(runtime.core))
    def task(value):
        events.append(("call", get_ident()))
        if value < 0:
            raise ValueError("fixture_failure")
        return value
    with ThreadPoolExecutor(max_workers=3) as callers:
        assert list(callers.map(lambda v: runtime.call(task, v), range(6))) == list(range(6))
    with pytest.raises(ValueError):
        runtime.call(task, -1)
    assert original == (id(runtime.query_processor), id(runtime.core))
    assert runtime.call(task, 10) == 10
    assert sum(name == "load" for name, _ in events) == 2
    assert len({thread for _, thread in events}) == 1
    assert sum(name == "cleanup" for name, _ in events) == 18

    buffers = []
    class RequestBuffer:
        pass
    def failure_with_buffer():
        buffer = RequestBuffer()
        buffers.append(ref(buffer))
        raise ValueError("fixture_buffer_failure")
    with pytest.raises(ValueError) as saved_exception:
        runtime.call(failure_with_buffer)
    assert saved_exception.value.__traceback__ is not None
    assert buffers[0]() is None
    assert original == (id(runtime.query_processor), id(runtime.core))
    runtime.close()
    runtime.close()
    assert sum(name == "close" for name, _ in events) == 2
    with pytest.raises(RuntimeError, match="runtime_closed"):
        runtime.call(task, 1)


def test_startup_failure_releases_already_loaded_core(monkeypatch):
    closed = []
    class Query:
        def __init__(self):
            raise ValueError("fixture_startup_failure")
    class Core:
        def __init__(self, path):
            pass
        def close(self):
            closed.append(True)
    monkeypatch.setattr("dona_search.runtime.CoreModel", Core)
    with pytest.raises(ValueError):
        ModelRuntime(Query)
    assert closed == [True]
