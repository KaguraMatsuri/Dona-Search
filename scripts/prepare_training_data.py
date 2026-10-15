import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.training_data import build_dataset, read_rows, write_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('review', 'build'))
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--schema-version', type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    rows = [r for p in args.records for r in read_rows(p)]
    if args.action == 'review':
        if args.schema_version == 4:
            from dona_search.organizer import snippet_segments
            prepared = []
            for row in rows:
                if row['task'] == 'ranking':
                    row = row | {'task': 'organizing', 'selections': None, 'review_status': 'pending', 'reviewer': None}
                elif row['task'] == 'query' and len(row.get('queries') or []) > 2:
                    row = row | {'review_status': 'pending', 'reviewer': None}
                if row['task'] == 'organizing':
                    row = row | {'candidate_segments': [
                        [{'id': i, 'text': segment.text, 'start': segment.start, 'end': segment.end}
                         for i, segment in enumerate(snippet_segments(candidate['snippet']))]
                        for candidate in row['candidates']]}
                prepared.append(row)
            rows = prepared
        args.output.mkdir(parents=True, exist_ok=False)
        pending = [r for r in rows if r.get('review_status') != 'confirmed']
        for offset in range(0, len(pending), 50):
            write_rows(args.output / f'{offset // 50:04d}.jsonl', pending[offset:offset + 50])
        print(json.dumps({'status': 'awaiting_human_review', 'records': len(pending)}))
    else:
        if args.model is None:
            parser.error('--model')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if args.schema_version == 4:
            from scripts.v4_data import build_v4_dataset
            manifest = build_v4_dataset(rows, args.output, tokenizer)
        else:
            manifest = build_dataset(rows, args.output, tokenizer)
        print(json.dumps({'status': 'data_ready', 'counts': manifest['counts']}))


if __name__ == '__main__':
    main()
