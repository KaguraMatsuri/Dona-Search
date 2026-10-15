"""Bounded local MLX training; preflight adapters never become production weights."""
import argparse
import gc
import json
import math
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dona_search.adapters import file_digest, model_fingerprint
from scripts.training_data import (
    PROMPT_LIMIT,
    read_rows,
    tokenize_example,
    verify_dataset,
)

MEMORY_LIMIT = 24 * 1024 ** 3
LORA = {'rank': 8, 'scale': 20.0, 'dropout': 0.0}


def write_status(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2))
    temporary.replace(path)


def probe_messages(task, tokenizer, long=False):
    from dona_search.query.planner import query_messages
    from dona_search.rerank.gemma import ranking_messages
    def make(repetitions):
        if task == 'query':
            messages = query_messages('SQLite' + ' documentation' * repetitions, 2)
            answer = '{"queries": ["SQLite"]}'
        elif task == 'organizing':
            from dona_search.interfaces import SearchHit
            from dona_search.organizer import organization_messages
            messages = organization_messages('SQLite', [
                SearchHit(1, 'SQLite', 'https://sqlite.org/', 'SQLite documentation.' * repetitions, 'probe'),
                SearchHit(2, 'Minecraft', 'https://minecraft.net/', 'Minecraft', 'probe')])
            answer = '{"order": [0, 1], "selections": [0, null]}'
        else:
            messages = ranking_messages('SQLite', [
                {'title': 'SQLite', 'url': 'https://sqlite.org/', 'snippet': 'SQLite documentation.' * repetitions},
                {'title': 'Minecraft', 'url': 'https://www.minecraft.net/', 'snippet': 'Minecraft'},
            ])
            answer = '[0, 1]'
        return messages + [{'role': 'assistant', 'content': answer}]
    if not long:
        return make(1)
    low, high = 1, PROMPT_LIMIT[task] * 2
    while low < high:
        middle = (low + high + 1) // 2
        try:
            tokenize_example(tokenizer, make(middle), task)
            low = middle
        except ValueError as error:
            if str(error) != 'training_context_limit':
                raise
            high = middle - 1
    return make(low)


def train_child(args):
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.trainer import default_loss, iterate_batches

    config = json.loads((args.base / 'config.json').read_text())
    quantization = config.get('quantization', config.get('quantization_config', {}))
    if quantization.get('bits') != 8 or quantization.get('group_size') != 64:
        raise ValueError('eight_bit_base_required')
    base_fingerprint = model_fingerprint(args.base)
    if args.action == 'train':
        data_manifest = verify_dataset(args.data)
        gate_path = args.preflight_root / args.task / 'preflight.json'
        if not gate_path.is_file():
            raise ValueError('matching_preflight_required')
        gate = json.loads(gate_path.read_text())
        if not gate.get('passed') or gate.get('base_fingerprint') != base_fingerprint or gate.get('task') != args.task:
            raise ValueError('matching_preflight_required')
        if gate.get('max_prompt_tokens') != PROMPT_LIMIT[args.task]:
            raise ValueError('preflight_context_mismatch')
        if data_manifest.get('framework_schema_version') == 4 and gate.get('framework_schema_version') != 4:
            raise ValueError('preflight_protocol_mismatch')
    model, tokenizer = load(str(args.base), lazy=False)
    mx.set_cache_limit(0)
    max_tokens = PROMPT_LIMIT[args.task] + (1024 if args.task == 'organizing' else 256)
    if args.action == 'preflight':
        records = [{'messages': probe_messages(args.task, tokenizer, long=long)} for long in (False, True)]
        validation = []
        iterations = 4
    else:
        records = read_rows(args.data / args.task / 'train.jsonl')
        validation = read_rows(args.data / args.task / 'valid.jsonl')
        iterations = len(records)
        if iterations % 4:
            raise ValueError('incomplete_gradient_accumulation')

    class Dataset:
        # The trainer consumes this official process/itemlen protocol through CacheDataset.
        def __init__(self, rows):
            self.rows = rows
            for row in rows:
                tokenize_example(tokenizer, row['messages'], args.task)
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]
        def process(self, row): return tokenize_example(tokenizer, row['messages'], args.task)

    args.output.mkdir(parents=True, exist_ok=False)
    opts = SimpleNamespace(**(CONFIG_DEFAULTS | {
        'model': str(args.base), 'train': True, 'seed': 42, 'num_layers': 16,
        'data': str(args.data) if args.data else 'synthetic_preflight_only',
        'batch_size': 1, 'iters': iterations, 'val_batches': -1,
        'learning_rate': 1e-5, 'grad_accumulation_steps': 4, 'grad_checkpoint': True,
        'mask_prompt': True, 'lora_parameters': LORA, 'adapter_path': str(args.output),
        'max_seq_length': max_tokens, 'steps_per_report': 1, 'steps_per_eval': 200,
        'save_every': 200, 'report_to': None,
    }))
    frozen_before = dict(tree_flatten(model.parameters()))
    # Keep the frozen values for an exact comparison after the compiled update.
    events = []
    class Callback:
        def on_train_loss_report(self, values):
            if not math.isfinite(values['train_loss']):
                raise ValueError('training_nonfinite_loss')
            events.append({'event': 'train', **values})
            with (args.output / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(events[-1]) + '\n')
        def on_val_loss_report(self, values):
            if not math.isfinite(values['val_loss']):
                raise ValueError('validation_nonfinite_loss')
            with (args.output / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps({'event': 'validation', **values}) + '\n')

    dataset = Dataset(records)
    np.random.seed(42)
    train_model(opts, model, dataset, Dataset(validation), Callback())
    trained = dict(tree_flatten(model.trainable_parameters()))
    if not trained or any(not k.endswith(('.lora_a', '.lora_b')) for k in trained):
        raise ValueError('unexpected_trainable_parameters')
    if not all(bool(mx.all(mx.isfinite(v)).item()) for v in trained.values()):
        raise ValueError('nonfinite_adapter')
    if not any(bool(mx.any(v != 0).item()) for k, v in trained.items() if k.endswith('.lora_b')):
        raise ValueError('adapter_not_updated')
    after = dict(tree_flatten(model.parameters()))
    for name, array in frozen_before.items():
        # QuantizedLinear becomes LoRALinear.linear; no base tensor may be updated.
        stem, _, leaf = name.rpartition('.')
        updated = after.get(name, after.get(stem + '.linear.' + leaf))
        if updated is None or not bool(mx.array_equal(updated, array).item()):
            raise ValueError('frozen_base_changed:' + name)
    if args.action == 'preflight':
        processed = [dataset.process(r) for r in records]
        batch = next(iterate_batches([max(processed, key=lambda x: len(x[0]))], 1, max_tokens))
        (_, _), gradients = nn.value_and_grad(model, default_loss)(model, *batch)
        mx.eval(gradients)
        gradient_arrays = [v for _, v in tree_flatten(gradients)]
        if not all(bool(mx.all(mx.isfinite(v)).item()) for v in gradient_arrays):
            raise ValueError('nonfinite_gradient')
        if not any(bool(mx.any(v != 0).item()) for v in gradient_arrays):
            raise ValueError('zero_gradient')
        del gradients, gradient_arrays, batch
    before_reload = {k: np.array(v) for k, v in trained.items()}
    manifest = {'schema_version': 1, 'task': args.task, 'base_fingerprint': base_fingerprint,
        'quantization': quantization, 'accepted': False, 'probe_only': args.action == 'preflight',
        'data_manifest_sha256': file_digest(args.data / 'manifest.json') if args.action == 'train' else None,
        'files': {n: file_digest(args.output / n) for n in ('adapter_config.json', 'adapters.safetensors')}}
    write_status(args.output / 'dona_adapter_manifest.json', **manifest)
    del model, trained, after, frozen_before
    gc.collect()
    mx.clear_cache()
    # Exercise the actual resident inference loaders and their manifest validation.
    from dona_search.config import RankingConfig
    from dona_search.query.planner import MLXQueryPlanner
    from dona_search.rerank.gemma import GemmaReranker
    config = RankingConfig(model_path=args.base, adapter_path=args.output,
                           max_prompt_tokens=PROMPT_LIMIT[args.task])
    if args.task == 'organizing':
        from dona_search.organizer import MLXOrganizer
        runtime = MLXOrganizer(config)
    else:
        v4 = args.action == 'preflight' or data_manifest.get('framework_schema_version') == 4
        runtime = MLXQueryPlanner(config, 2 if v4 else 3) if args.task == 'query' else GemmaReranker(config)
    try:
        reloaded = runtime.backend.model if args.task in ('query', 'organizing') else runtime.model
        params = dict(tree_flatten(reloaded.parameters()))
        if not all(np.array_equal(v, np.asarray(params[k])) for k, v in before_reload.items()):
            raise ValueError('adapter_roundtrip_mismatch')
        try:
            runtime.warmup()
        except Exception as error:
            if args.action == 'preflight':
                write_status(args.output / 'preflight.json', passed=False, task=args.task,
                             base_fingerprint=base_fingerprint, numeric_checks_passed=True,
                             adapter_roundtrip_passed=True, inference_error=type(error).__name__,
                             max_prompt_tokens=PROMPT_LIMIT[args.task], probe_only=True)
            raise
    finally:
        runtime.close()
    if args.action == 'preflight':
        write_status(args.output / 'preflight.json', passed=True, task=args.task,
                     base_fingerprint=base_fingerprint, max_prompt_tokens=PROMPT_LIMIT[args.task],
                     max_sequence_tokens=max_tokens, iterations=iterations, probe_only=True, framework_schema_version=4)
    print(json.dumps({'status': 'preflight_passed' if args.action == 'preflight' else 'training_complete',
                      'task': args.task, 'accepted': False}), flush=True)


def child(args):
    if args.action == 'convert':
        from mlx_lm.convert import convert
        convert(str(args.base), str(args.output), quantize=True, q_bits=8,
                q_group_size=64, q_mode='affine', dtype='bfloat16')
        write_status(args.output / 'dona_base_manifest.json', schema_version=1,
                     source_fingerprint=model_fingerprint(args.base),
                     fingerprint=model_fingerprint(args.output), bits=8, group_size=64)
    else:
        train_child(args)


def supervise(args):
    from scripts.eval_search import MemorySampler
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('apple_silicon_required')
    if args.action == 'train':
        verify_dataset(args.data)
    if args.task == 'both' and args.action != 'convert':
        manifest = verify_dataset(args.data) if args.data is not None else {}
        tasks = tuple(manifest.get('tasks', ('query', 'ranking') if manifest.get('schema_version') == 1 else ('query', 'organizing')))
    else:
        tasks = (args.task,)
    for task in tasks:
        destination = args.output if args.action == 'convert' else args.output / task
        if destination.exists():
            raise ValueError('output_already_exists')
        log = destination.parent / (destination.name + '.process.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), args.action, '--child',
                   '--task', task, '--base', str(args.base), '--output', str(destination)]
        if args.data: command += ['--data', str(args.data)]
        if args.preflight_root: command += ['--preflight-root', str(args.preflight_root)]
        started, peak, reason = time.monotonic(), 0, None
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            sampler = MemorySampler(pid=process.pid)
            try:
                while process.poll() is None:
                    peak = max(peak, sampler.current() or 0)
                    if peak > MEMORY_LIMIT:
                        reason = 'memory_budget_exceeded'
                    if args.action == 'preflight' and time.monotonic() - started > 900:
                        reason = 'preflight_timeout'
                    if reason:
                        process.terminate()
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        break
                    time.sleep(.1)
            except BaseException:
                process.send_signal(signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill()
                raise
            finally:
                code = process.wait()
        report = {'action': args.action, 'task': task, 'exit_code': code,
                  'seconds': time.monotonic() - started, 'peak_physical_bytes': peak,
                  'memory_limit_bytes': MEMORY_LIMIT, 'error': reason or ('child_failed' if code else None),
                  'status': 'complete' if code == 0 and reason is None else 'failed'}
        write_status(log.with_suffix('.json'), **report)
        if args.action == 'preflight' and (code or reason):
            gate_path = destination / 'preflight.json'
            gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
            write_status(gate_path, **(gate | {'passed': False, 'task': task,
                                             'process_error': report['error']}))
        print(json.dumps(report), flush=True)
        if code or reason:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('convert', 'preflight', 'train'))
    parser.add_argument('--task', choices=('query', 'ranking', 'organizing', 'both'), default='both')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--preflight-root', type=Path)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == 'train' and (args.data is None or args.preflight_root is None):
        parser.error('--data --preflight-root')
    if args.child:
        child(args)
    else:
        supervise(args)


if __name__ == '__main__':
    main()
