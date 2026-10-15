import argparse
import ctypes
import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dona_search.config import read_config, validate_config
from dona_search.interfaces import SearchHit
from dona_search.rerank.gemma import GemmaReranker, validate_order
from dona_search.runtime import ModelRuntime


class MemorySampler:
    def __init__(self, pid=None):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = None
        self.reader = None
        if sys.platform == 'darwin':
            class Usage(ctypes.Structure):
                _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
                    'user', 'system', 'idle', 'interrupt', 'pageins', 'wired', 'resident', 'physical', 'start', 'exit')]
            function = ctypes.CDLL('/usr/lib/libproc.dylib').proc_pid_rusage
            function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            function.restype = ctypes.c_int
            target_pid = os.getpid() if pid is None else pid
            def read():
                value = Usage()
                return value.physical if function(target_pid, 0, ctypes.byref(value)) == 0 else None
            self.reader = read

    def current(self):
        return self.reader() if self.reader else None

    def __enter__(self):
        def sample():
            while not self.stop.is_set():
                self.peak = max(self.peak, self.current() or 0)
                self.stop.wait(.1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def validate_fixture(row):
    if not isinstance(row.get('query'), str) or not isinstance(row.get('candidates'), list):
        raise ValueError('invalid_evaluation_fixture')
    if len(row['candidates']) > 40:
        raise ValueError('candidate_limit_exceeded')
    hits = []
    for i, value in enumerate(row['candidates']):
        if any(not isinstance(value.get(k), str) for k in ('title', 'url', 'snippet')):
            raise ValueError('invalid_evaluation_candidate')
        hits.append(SearchHit(i + 1, value['title'], value['url'], value['snippet'], value.get('provider', 'fixture')))
    return hits


def quality_metrics(grades, order, k=10):
    import numpy as np
    from sklearn.metrics import ndcg_score

    validate_order(order, len(grades))
    if any(type(g) is not int or not 0 <= g <= 3 for g in grades):
        raise ValueError('invalid_relevance_grade')
    if len(grades) < 2:
        ndcg = float(bool(grades and grades[0] > 0))
    else:
        predicted = np.empty(len(order), dtype=float)
        for rank, candidate in enumerate(order):
            predicted[candidate] = len(order) - rank
        ndcg = float(ndcg_score([[(2 ** g) - 1 for g in grades]], [predicted], k=k))
    reciprocal = next((1 / (i + 1) for i, candidate in enumerate(order[:k]) if grades[candidate] > 0), 0.0)
    return {'ndcg_at_10': ndcg, 'mrr_at_10': reciprocal}


def evaluate_rows(rows, ranker):
    reports = []
    for row in rows:
        hits = validate_fixture(row)
        started = time.perf_counter()
        record = {'id': row['id'], 'query': row['query'], 'candidate_count': len(hits)}
        try:
            order = validate_order(ranker.rank(row['query'], hits), len(hits))
            record.update(ranking_status='ok', order=order)
        except Exception as error:
            order = list(range(len(hits)))
            record.update(ranking_status='failed', error=type(error).__name__, order=order)
        record['seconds'] = time.perf_counter() - started
        grades = row.get('judgments')
        if grades is None:
            record['quality_status'] = 'unjudged'
        elif row.get('judgment_source') not in ('human_confirmed', 'published_human'):
            record['quality_status'] = 'unconfirmed_judgments'
        elif not isinstance(grades, list) or len(grades) != len(hits):
            raise ValueError('judgments_length_mismatch')
        else:
            if row.get('judgment_source') == 'published_human':
                from scripts.training_data import validate_record
                validate_record(row)
            record['quality_status'] = 'judged'
            record['baseline'] = quality_metrics(grades, list(range(len(hits))))
            record['ranked'] = quality_metrics(grades, order) if record['ranking_status'] == 'ok' else None
            if record['ranking_status'] != 'ok':
                record['fallback'] = quality_metrics(grades, order)
            threshold = 1 if row.get('source', {}).get('name') == 'miracl' else 2
            record['has_relevant_candidate'] = any(g >= threshold for g in grades)
            record['top1_relevant'] = (bool(grades and grades[order[0]] >= threshold)
                                      if record['ranking_status'] == 'ok' else False)
        record['language'] = row.get('language')
        record['source'] = row.get('source', {}).get('name')
        reports.append(record)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config')
    args = parser.parse_args()
    source = args.input.read_bytes()
    rows = [json.loads(line) for line in source.decode('utf-8').splitlines() if line.strip()]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_fixture_id')
    for row in rows:
        validate_fixture(row)
    raw, base = read_config(args.config)
    cfg = validate_config(raw, base, required_models=('ranking',))
    with MemorySampler() as memory:
        started = time.perf_counter()
        with_runtime = ModelRuntime(build_ranker=lambda: GemmaReranker(cfg.ranking))
        loaded = time.perf_counter()
        class Runner:
            def rank(self, query, hits):
                return with_runtime.call(with_runtime.ranker.rank, query, hits)
        try:
            records = evaluate_rows(rows, Runner())
            def allocator_stats():
                import mlx.core as mx
                return {'mlx_active_bytes': mx.get_active_memory(), 'mlx_peak_bytes': mx.get_peak_memory(),
                        'mlx_cache_bytes': mx.get_cache_memory()}
            allocator = with_runtime.call(allocator_stats)
            idle_physical = memory.current()
        finally:
            with_runtime.close()
        released_physical = memory.current()
        import mlx.core as mx
        released_active = mx.get_active_memory()
    report = {
        'schema_version': 1, 'dataset_sha256': hashlib.sha256(source).hexdigest(),
        'model': cfg.ranking.model, 'revision': cfg.ranking.revision,
        'local_model_path': str(cfg.ranking.model_path) if cfg.ranking.model_path else None,
        'adapter_path': str(cfg.ranking.adapter_path) if cfg.ranking.adapter_path else None,
        'startup_seconds': loaded - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        'memory': {'sample_interval_seconds': .1, 'sampled_physical_peak_bytes': memory.peak or None,
                   'idle_physical_bytes': idle_physical, 'after_close_physical_bytes': released_physical,
                   'after_close_mlx_active_bytes': released_active, **allocator},
        'search_requests': 0, 'records': records,
        'judged_queries': sum(r['quality_status'] == 'judged' for r in records),
        'ranking_failures': sum(r['ranking_status'] != 'ok' for r in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
