"""Fixture-bound v4 evaluation; never substitute ranking labels for evidence labels."""
import gc
import json

from dona_search.adapters import file_digest, model_fingerprint
from dona_search.config import ModelSelection
from dona_search.memory import check_model_memory
from dona_search.organizer import MLXOrganizer, selected_evidence, validate_decision
from dona_search.query.planner import MLXQueryPlanner
from scripts.eval_query import evaluate_queries, evaluation_runner, judge_report
from scripts.eval_search import MemorySampler, quality_metrics
from scripts.train_models import write_status
from scripts.training_data import positive_grade, read_rows
from scripts.v4_data import candidate_hits, validate_v4_record


def evaluate_organizing(rows, organizer):
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate_fixture_id')
    records = []
    for row in rows:
        validate_v4_record(row)
        record = {'id': row['id'], 'language': row['language'], 'quality_status': 'judged',
                  'ranking_status': 'failed', 'evidence_match': False, 'ranked': None,
                  'has_relevant_candidate': max(row['judgments']) >= positive_grade(row), 'top1_relevant': False}
        hits = candidate_hits(row)
        try:
            decision = validate_decision(organizer.organize(row['query'], hits), hits)
            evidence = [selected_evidence(hit, decision.selections[i]) for i, hit in enumerate(hits)]
            record.update(ranking_status='ok', order=decision.order, selections=decision.selections,
                evidence_match=decision.selections == row['selections'],
                source_match=all(item is None or item.text == hits[i].snippet[item.start:item.end]
                                 for i, item in enumerate(evidence)),
                top1_relevant=row['judgments'][decision.order[0]] >= positive_grade(row),
                ranked=quality_metrics(row['judgments'], decision.order))
        except Exception as error:  # noqa: BLE001 - retain every failed inference in the denominator
            record['error'] = type(error).__name__
        records.append(record)
    return records


def summarize_v4(output, dataset_hash, reviews=None):
    from scripts.eval_models import ranking_acceptance
    reports = {}
    for variant in ('bf16', 'quantized', 'adapted'):
        report = json.loads((output / f'{variant}.organizing.json').read_text())
        if report.get('framework_schema_version') != 4 or report.get('dataset_sha256') != dataset_hash:
            raise ValueError('evaluation_dataset_mismatch')
        reports[variant] = report
    ranking = ranking_acceptance(reports)
    evidence = all(record.get('evidence_match') and record.get('source_match')
                   and record['ranking_status'] == 'ok'
                   for subset in ('public', 'web') for record in reports['adapted'][subset])
    query_path = output / 'adapted.query.json'
    query = {'accepted': False}
    if query_path.exists():
        query = json.loads(query_path.read_text())
        if query.get('dataset_sha256') != dataset_hash or query.get('framework_schema_version') != 4:
            raise ValueError('evaluation_dataset_mismatch')
        if reviews is None:
            reviews_rows = [{'id': row['id'], 'prediction_sha256': row.get('prediction_sha256'),
                             'reviewer': row.get('reviewer'), 'intent_preserved': row['intent_status'] == 'confirmed'}
                            for row in query['records'] if row.get('intent_status') in ('confirmed', 'rejected')]
        else:
            reviews_rows = read_rows(reviews)
        query = judge_report(query, reviews_rows)
        write_status(query_path, **query)
    result = {'framework_schema_version': 4, 'acceptance_scope': 'reviewed_fixture_only',
              'organizing': {'ranking': ranking, 'evidence_match': evidence,
                             'accepted': ranking['accepted'] and evidence},
              'query': {'accepted': query['accepted']},
              'accepted': bool(ranking['accepted'] and evidence and query['accepted']),
              'dataset_sha256': dataset_hash}
    write_status(output / 'acceptance.json', **result)
    return result


def evaluate_v4(args, manifest, dataset_hash):
    if args.summarize_only:
        return summarize_v4(args.output, dataset_hash, args.query_reviews)
    args.output.mkdir(parents=True, exist_ok=False)
    for variant, path in [('bf16', args.bf16), ('quantized', args.base), ('adapted', args.base)]:
        for task in manifest['tasks']:
            adapter = args.adapters / task if variant == 'adapted' else None
            config = ModelSelection(model=str(path), weights={'path': path},
                                    adapter={'path': adapter} if adapter else None,
                                    max_prompt_tokens=2048 if task == 'query' else 8192)

            def build(config=config, task=task, path=path):
                check_model_memory([(path, config.max_prompt_tokens)])
                return MLXQueryPlanner(config, 2) if task == 'query' else MLXOrganizer(config)

            builders = {'build_query': build, 'query_only': True} if task == 'query' else {
                'build_organizer': build, 'organizer_only': True}
            with MemorySampler() as memory, evaluation_runner(**builders) as runner:
                if task == 'query':
                    report = evaluate_queries(read_rows(args.data / 'query/test.records.jsonl'), runner, max_queries=2)
                else:
                    report = {'public': evaluate_organizing(read_rows(args.data / 'organizing/test.records.jsonl'), runner),
                              'web': evaluate_organizing(read_rows(args.data / 'web_test.records.jsonl'), runner)}
                report['startup_error'] = runner.startup_error
            report.update(framework_schema_version=4, model_fingerprint=model_fingerprint(path),
                          adapter_manifest_sha256=file_digest(adapter / 'dona_adapter_manifest.json') if adapter else None,
                          dataset_sha256=dataset_hash, peak_physical_bytes=memory.peak)
            write_status(args.output / f'{variant}.{task}.json', **report)
            gc.collect()
    return summarize_v4(args.output, dataset_hash, args.query_reviews)
