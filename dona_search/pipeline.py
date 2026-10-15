import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import httpx
import yaml

from .config import HTTP_HEADERS, read_config, validate_config
from .interfaces import ProviderResult, SearchReport, SearchResult
from .query.planner import MLXQueryPlanner, validate_plan
from .registry import get
from .rerank.gemma import GemmaReranker, validate_order
from .runtime import ModelRuntime
from .search.scheduler import SearchScheduler, merge_tasks

logger = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(self, config_path: str | Path | None = None, *, http_client=None, load_model=True,
                 build_query=None, build_ranker=None, build_organizer=None, browser_factory=None,
                 browser_kind=None, show_browser=False):
        raw, base = read_config(config_path)
        self._search_rank = 'schema_version' in raw or 'query_processor' not in raw
        self._complete = raw.get('schema_version') in (3, 4)
        self._organizing = raw.get('schema_version') == 4
        if build_organizer is not None and (not self._organizing or build_ranker is not None):
            raise ValueError('organizer_requires_schema_v4')
        if self._organizing and build_ranker is not None:
            raise ValueError('schema_v4_requires_organizer')
        self._closed = False
        self._request_lock = Lock()
        if self._complete:
            config = validate_config(raw, base, require_model_paths=load_model)
            if browser_kind is not None:
                config.browser.kind = browser_kind
            if show_browser:
                config.browser.headless = False
            self._init_complete(config, http_client, load_model, build_query, build_ranker, browser_factory, build_organizer)
            return
        if any(value is not None for value in (build_query, build_ranker, browser_factory, browser_kind)) or show_browser:
            raise ValueError('pipeline_options_require_schema_v3')
        if self._search_rank:
            self._init_search_rank(validate_config(raw, base), http_client, load_model)
            return
        if not load_model:
            raise ValueError('search_only_requires_schema_v2')
        # Versionless research configurations retain their original behavior.
        from . import query, search, utils  # noqa: F401
        from .extract.sentence import SentenceExtractor
        from .rerank.cross_encoder import CrossEncoderReranker

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        if "reranker" in cfg or "extractor" in cfg:
            raise ValueError("core_configuration_fixed")
        self.providers = [self._build(p) for p in cfg["search"]["providers"]]
        self.fetcher = self._build(cfg["fetcher"])
        self.runtime = ModelRuntime(lambda: self._build(cfg["query_processor"]))
        self.query_processor = self.runtime.query_processor
        self.reranker = CrossEncoderReranker(self.runtime.core)
        self.extractor = SentenceExtractor(self.runtime.core)

        self.cfg = cfg

    def _build(self, spec: dict):
        category, name = spec["type"].split("_", 1)
        cls = get(category, name)
        return cls(**{k: v for k, v in spec.items() if k != "type"})

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self._search_rank:
            return self.search_report(query, top_k).results
        if top_k < 1:
            return []
        pq = self.runtime.call(self.query_processor.process, query)

        # 2. Multi-provider search
        hits = []
        for provider in self.providers:
            hits.extend(provider.search(" ".join(pq.keywords), k=100))

        # Deduplicate by URL
        seen = {}
        for h in hits:
            if h.url not in seen:
                seen[h.url] = h
        hits = list(seen.values())

        # 3. Stage 1: Snippet reranking → top-50
        snippets = [{"title": h.title, "snippet": h.snippet} for h in hits]
        scores_1 = self.runtime.call(self.reranker.score_batch, pq.raw, snippets)
        ranked_1 = sorted(zip(hits, scores_1), key=lambda x: x[1], reverse=True)[:50]

        # 4. Fetch full text
        docs = []
        for hit, snippet_score in ranked_1:
            doc = self.fetcher.fetch(hit.url)
            if doc and doc.url != hit.url:
                doc = None
            docs.append((hit, doc, snippet_score))

        # 5. Stage 2: Full-text reranking → top-10
        full_texts = [{"title": d.title, "text": d.text} for _, d, _ in docs if d is not None]
        scores_2 = self.runtime.call(self.reranker.score_batch, pq.raw, full_texts)
        scores_iter = iter(scores_2)
        ranked_docs = [((hit, doc), next(scores_iter) if doc is not None else score) for hit, doc, score in docs]
        ranked_2 = sorted(ranked_docs, key=lambda x: x[1], reverse=True)[:top_k]

        # 6. Extract evidence
        results = []
        for rank, ((hit, doc), score) in enumerate(ranked_2, 1):
            evidences = self.runtime.call(self.extractor.extract, pq.raw, doc, top_k=1) if doc is not None else []
            ev = evidences[0] if evidences else None
            if ev is not None and not (
                type(ev.start) is int
                and type(ev.end) is int
                and 0 <= ev.start < ev.end <= len(doc.text)
                and ev.text == doc.text[ev.start:ev.end]
                and math.isfinite(ev.confidence)
                and self.runtime.core.threshold <= ev.confidence <= 1
            ):
                ev = None

            results.append(SearchResult(
                rank=rank,
                preview=hit.snippet,
                url=hit.url,
                summary=ev,
            ))

        return results

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._complete and self.scheduler is not None:
                self.scheduler.close(self.browser.close if self.browser is not None else None)
        finally:
            try:
                if self.runtime is not None:
                    self.runtime.close()
            finally:
                if self._search_rank and self._owns_client:
                    self.client.close()

    def _init_complete(self, config, http_client, load_model, build_query, build_ranker, browser_factory, build_organizer=None):
        from . import search  # noqa: F401
        from .search.browser import BrowserSession

        started = time.perf_counter()
        if self._organizing and load_model:
            required = [name for name, factory in (('query_model', build_query), ('ranking', build_organizer))
                        if factory is None]
            if required:
                from .models import ModelPreparer
                config, _ = ModelPreparer().prepare(config, required_models=required)
        self.cfg = config
        self.runtime = self.scheduler = self.browser = None
        self.ranking_error = None
        self._load_model = load_model
        self._owns_client = http_client is None
        self.client = http_client if http_client is not None else httpx.Client(timeout=config.timeout, headers=HTTP_HEADERS)
        self.providers = []
        try:
            active = [p for p in config.providers if p.enabled]
            if any(p.type not in ('search_so360', 'search_mwmbl') for p in active):
                self.browser = (browser_factory or BrowserSession)(config.browser, config.timeout)
            self.providers = [get('search', p.type.split('_', 1)[1])(
                p, self.client if p.type in ('search_so360', 'search_mwmbl') else self.browser) for p in active]
            self.scheduler = SearchScheduler(self.providers)
            if self.browser is not None:
                self.scheduler.call(self.browser.start)
            if load_model:
                if self._organizing:
                    from .organizer import MLXOrganizer
                    self.runtime = ModelRuntime(
                        build_query or (lambda: MLXQueryPlanner(config.query_model, max(1, config.max_queries - 1))),
                        build_organizer=build_organizer or (lambda: MLXOrganizer(config.ranking)),
                    )
                else:
                    self.runtime = ModelRuntime(
                        build_query or (lambda: MLXQueryPlanner(config.query_model, config.max_queries)),
                        build_ranker=build_ranker or (lambda: GemmaReranker(config.ranking)),
                    )
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.debug('startup_cleanup_failed', exc_info=True)
            raise
        self.startup_seconds = time.perf_counter() - started

    def _search_complete(self, query, top_k):
        started = time.perf_counter()
        if not query.strip() or top_k < 1:
            return SearchReport(query, [], [], 0, 'not_requested')
        if self._load_model:
            try:
                plan = validate_plan(self.runtime.call(self.runtime.query_processor.plan, query), query, self.cfg.max_queries)
                queries, query_status = plan.queries, 'ok'
                if self._organizing:
                    queries = [query] + [item for item in queries if item.strip() != query.strip()]
                    queries = queries[:self.cfg.max_queries]
            except Exception as error:
                logger.debug('query_failed', exc_info=True)
                return SearchReport(query, [], [], 0, 'not_requested',
                                    timings={'plan': time.perf_counter() - started, 'total': time.perf_counter() - started},
                                    query_status='failed', query_error=str(error) or type(error).__name__)
        else:
            queries, query_status = [query], 'not_requested'
        planned = time.perf_counter()
        tasks = self.scheduler.collect(queries)
        hits, providers = merge_tasks(tasks, [p.name for p in self.providers])
        collected = time.perf_counter()
        if self._closed:
            raise RuntimeError('pipeline_closed')
        order, error_code = list(range(len(hits))), None
        evidence = {}
        if not hits:
            status = 'no_results'
        elif not self._load_model:
            status = 'not_requested'
        else:
            try:
                if self._organizing:
                    from .organizer import selected_evidence, validate_decision
                    decision = validate_decision(self.runtime.call(self.runtime.organizer.organize, query, hits), hits)
                    evidence = {i: selected_evidence(hit, decision.selections[i]) for i, hit in enumerate(hits)}
                    order = decision.order
                else:
                    order = validate_order(self.runtime.call(self.runtime.ranker.rank, query, hits), len(hits))
                status = 'ok'
            except Exception as error:
                logger.debug('organizing_failed', exc_info=True)
                status, error_code = 'failed', str(error) or type(error).__name__
        results = [SearchResult(rank, hits[i].snippet, hits[i].url, None, hits[i].title, list(hits[i].origins), evidence.get(i))
                   for rank, i in enumerate(order[:min(top_k, self.cfg.display_limit)], 1)]
        finished = time.perf_counter()
        return SearchReport(query, results, providers, len(hits), status, error_code,
                            {'plan': planned - started, 'search': collected - planned,
                             'rank': finished - collected, 'total': finished - started},
                            hits, query_status=query_status, search_queries=queries, tasks=tasks,
                            organizing_status=status if self._organizing else 'not_requested',
                            organizing_error=error_code if self._organizing else None)

    def _init_search_rank(self, config, http_client, load_model):
        from . import search  # noqa: F401

        started = time.perf_counter()
        self.cfg = config
        self.runtime = None
        self.ranking_error = None
        self._load_model = load_model
        self._owns_client = http_client is None
        self.client = http_client if http_client is not None else httpx.Client(
            timeout=config.timeout,
            headers=HTTP_HEADERS,
        )
        try:
            self.providers = [get('search', p.type.split('_', 1)[1])(p, self.client)
                              for p in config.providers if p.enabled]
            if load_model:
                try:
                    self.runtime = ModelRuntime(build_ranker=lambda: GemmaReranker(config.ranking))
                except Exception as error:
                    logger.debug('ranking_startup_failed', exc_info=True)
                    self.ranking_error = type(error).__name__
        except BaseException:
            if self._owns_client:
                self.client.close()
            raise
        self.startup_seconds = time.perf_counter() - started

    def _collect(self, query):
        outcomes = []
        unique = {}
        for provider in self.providers:
            started = time.perf_counter()
            try:
                outcome = provider.search(query, provider.config.top_k)
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
            acquired = datetime.now(timezone.utc).isoformat()
            for hit in outcome.hits:
                origin = {'provider': provider.name, 'rank': hit.rank, 'retrieved_at': acquired}
                if hit.url in unique:
                    unique[hit.url].origins.append(origin)
                else:
                    hit.origins = [origin]
                    unique[hit.url] = hit
            outcomes.append(outcome)
        return list(unique.values()), outcomes

    def search_report(self, query: str, top_k: int = 10) -> SearchReport:
        if self._closed:
            raise RuntimeError('pipeline_closed')
        if self._complete:
            with self._request_lock:
                if self._closed:
                    raise RuntimeError('pipeline_closed')
                return self._search_complete(query, top_k)
        if not self._search_rank:
            results = self.search(query, top_k)
            return SearchReport(query, results, [], len(results), 'research')
        started = time.perf_counter()
        if not query.strip() or top_k < 1:
            return SearchReport(query, [], [], 0, 'not_requested')
        hits, providers = self._collect(query)
        collected = time.perf_counter()
        order = list(range(len(hits)))
        error_code = None
        if not hits:
            status = 'no_results'
        elif not self._load_model:
            status = 'not_requested'
        elif self.runtime is None:
            status, error_code = 'unavailable', self.ranking_error
        else:
            try:
                order = validate_order(self.runtime.call(self.runtime.ranker.rank, query, hits), len(hits))
                status = 'ok'
            except Exception as error:
                logger.debug('ranking_failed', exc_info=True)
                status, error_code = 'failed', type(error).__name__
        results = [SearchResult(
            rank=rank, preview=hits[i].snippet, url=hits[i].url, summary=None,
            title=hits[i].title, origins=list(hits[i].origins),
        ) for rank, i in enumerate(order[:min(top_k, self.cfg.display_limit)], 1)]
        finished = time.perf_counter()
        return SearchReport(query, results, providers, len(hits), status, error_code,
                            {'search': collected - started, 'rank': finished - collected,
                             'total': finished - started}, hits)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
