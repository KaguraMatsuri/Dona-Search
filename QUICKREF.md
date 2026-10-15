# Dona Search v4 quick reference

| Task | Command or interface |
|---|---|
| Interactive setup and search | `.venv/bin/python -m dona_search` |
| Explicit config and query | `.venv/bin/python -m dona_search --config config.yaml 'query'` |
| Search without models | `.venv/bin/python -m dona_search --search-only 'query'` |
| Alternate browser | `--browser firefox` or `--browser chrome` |
| Show the browser | `--show-browser` |
| SDK with statuses | `SearchPipeline().search_report(query)` inside a context manager |
| v4 result formatting | `format_result(result)`, `format_results(report)`, `result.format(organized=True)` |
| Legacy formatting | `result.format()` |
| Offline verification | `bash scripts/validate.sh` |

Configuration priority: explicit → saved user configuration → bundled v4. Saved configuration: `~/Library/Application Support/dona-search/config.yaml`.

The query limit includes the original query. Default: original plus up to two planned queries, FIFO across configured engines. At most 40 candidates and 10 displayed results. Empty evidence remains an empty summary.

Both models are resident until close. Startup memory estimates stop oversized configurations; they are not hard allocation limits. Ctrl+C requests cooperative stop and exit. Minimal context overflow fails visibly rather than dropping candidates.

For engine XPath, complete model weights and adapter binding, see QUICKSTART.md. For human-reviewed v4 training and evaluation, see TRAINING.md. JSON field names and source offsets describe machine contracts; exact copying does not guarantee source truth or relevance.
