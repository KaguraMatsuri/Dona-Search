import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event, Lock

import httpx

from ..interfaces import (
    ProviderResult,
    SearchHit,
    SearchProvider,
    SearchTask,
    TaskResult,
)

logger = logging.getLogger(__name__)


class SearchScheduler:
    def __init__(self, providers: list[SearchProvider]):
        self.providers = providers
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dona-search')
        self._cancel = Event()
        self._closed = False
        self._lock = Lock()
        self._pending = set()

    def call(self, fn, *args):
        with self._lock:
            if self._closed:
                raise RuntimeError('scheduler_closed')
            future = self._worker.submit(fn, *args)
            self._pending.add(future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._pending.discard(future)

    def collect(self, queries):
        return self.call(self._collect, queries)

    def _collect(self, queries):
        tasks = []
        blocked = set()
        for index, query in enumerate(queries):
            for provider in self.providers:
                if self._cancel.is_set():
                    return tasks
                task = SearchTask(len(tasks), provider.name, query, index, provider.config.top_k)
                if provider.name in blocked:
                    outcome = ProviderResult(provider.name, 'skipped', error_code='engine_blocked')
                else:
                    outcome = self._execute(provider, query)
                    if outcome.status == 'blocked':
                        blocked.add(provider.name)
                acquired = datetime.now(timezone.utc).isoformat()
                for hit in outcome.hits:
                    hit.origins = [{'provider': provider.name, 'rank': hit.rank,
                                    'retrieved_at': acquired, 'query': query, 'task_id': task.id}]
                tasks.append(TaskResult(task, outcome))
        return tasks

    @staticmethod
    def _execute(provider, query):
        started = time.perf_counter()
        try:
            outcome = provider.search(query, provider.config.top_k)
            if (not isinstance(outcome, ProviderResult) or outcome.provider != provider.name
                    or not isinstance(outcome.hits, list)
                    or any(not isinstance(hit, SearchHit) or any(not isinstance(value, str)
                        for value in (hit.title, hit.url, hit.snippet)) for hit in outcome.hits)):
                raise TypeError('invalid_provider_result')
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            blocked = status in (403, 429) or 'captcha' in error.response.headers.get('location', '').lower()
            outcome = ProviderResult(provider.name, 'blocked' if blocked else 'http_error',
                                     http_status=status, error_code=f'http_{status}')
        except httpx.TimeoutException:
            outcome = ProviderResult(provider.name, 'timeout', error_code='request_timeout')
        except httpx.HTTPError as error:
            outcome = ProviderResult(provider.name, 'network_error', error_code=type(error).__name__)
        except Exception as error:
            logger.debug('provider_failed', exc_info=True)
            outcome = ProviderResult(provider.name, 'parse_error', error_code=type(error).__name__)
        outcome.elapsed_seconds = time.perf_counter() - started
        return outcome

    def close(self, release=None):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancel.set()
            for pending in self._pending:
                pending.cancel()
            future = self._worker.submit(release) if release else None
        try:
            if future is not None:
                future.result()
        finally:
            self._worker.shutdown(wait=True, cancel_futures=True)


def _round_robin(groups, limit):
    result, seen = [], set()
    for row in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if row < len(group) and group[row].url not in seen:
                result.append(group[row])
                seen.add(group[row].url)
                if len(result) == limit:
                    return result
    return result


def merge_tasks(tasks, names):
    origins = {}
    for record in tasks:
        for hit in record.outcome.hits:
            origins.setdefault(hit.url, []).extend(hit.origins)
    providers = []
    for name in names:
        records = [t.outcome for t in tasks if t.task.provider == name]
        hits = _round_robin([r.hits for r in records], 20)
        statuses = {r.status for r in records if r.status != 'skipped'}
        status = next(iter(statuses)) if len(statuses) == 1 else 'partial'
        first_error = next((r.error_code for r in records if r.error_code), None)
        cleanup_error = next((r.cleanup_error for r in records if r.cleanup_error), None)
        providers.append(ProviderResult(name, status if records else 'skipped', hits,
                                        error_code=first_error,
                                        elapsed_seconds=sum(r.elapsed_seconds for r in records),
                                        cleanup_error=cleanup_error))
    candidates = _round_robin([p.hits for p in providers], 40)
    for hit in candidates:
        hit.origins = list(origins[hit.url])
    return candidates, providers
