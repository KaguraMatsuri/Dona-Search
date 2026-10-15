# Framework v4 architecture

```text
Configuration resolution
  → artifact preparation and memory estimates
  → browser worker + resident model worker
  → both models loaded and warmed
  → query planning
  → original query plus bounded planned queries
  → FIFO engine tasks
  → URL deduplication and balanced fusion
  → organizer decision
  → validated IDs and copied source slices
```

## Ownership

- `config.py`: typed configuration, default resolution, v3 selection migration and path resolution.
- `models.py` and `memory.py`: source preparation, cache validation, default conversion and startup estimates.
- `runtime.py`: model loading, warmup, serialized inference, cooperative cancellation, cleanup and close.
- `search/scheduler.py`: FIFO tasks, per-engine failure isolation, blocked-engine skipping and candidate fusion.
- `search/browser.py`: browser lifecycle, owned profiles, DOM parsing and source URL extraction.
- `query/planner.py`: bounded query JSON protocol. v4 reserves one query slot for the original query.
- `organizer.py`: complete permutation and nullable per-input-result segment selections.
- `output.py`: shared v4 formatting and terminal control escaping. Internal source data remains unchanged.
- `console.py`: interactive selection, preparation feedback and result display. SDK calls do not prompt.

## Evidence contract

Each result retains its original title, URL, snippet and retrieval origins. `selections[i]` refers to input result i, independently of sorted position. Index 0 selects the complete original snippet; other IDs select indexed verbatim slices. Null means no supported selection.

The program validates the full order permutation and every selection before rendering any organized result. Failure falls back atomically to source results without a fabricated summary. Snippet offsets refer to the search snippet, not a fetched page body.

Exact source copying does not establish factual truth, relevance or faithful query planning. Those require reviewed model evaluation. Source content is treated as data in prompts, but semantic correctness is not guaranteed by JSON validation.

## Batching and bounds

Each provider contributes up to 20 merged hits; the final pool is at most 40. Large organizer inputs are split by the actual tokenizer budget. Each batch selects source IDs and ranks its candidates. Sorted batches are merged through pairwise comparisons of selected evidence. All candidate IDs remain in the final permutation.

An individual result or minimum comparison that cannot fit the context causes an explicit organization failure. No hidden truncation or unbounded retry is used. Cross-batch comparisons add inference calls; they do not establish model ranking consistency or global factual correctness.

## Resources and shutdown

Both application models stay resident after warmup. Per-call cleanup releases temporary inference allocations without unloading the weights. Memory estimates include both complete bundles, adapters, KV workspace and prefill workspace, with headroom. Default budget derives from Metal's recommended working set. Conversion is checked independently. These estimates are not hard allocation guarantees.

Cancellation closes the official stream generator between generated steps. A running prefill, generation step or browser operation can still delay shutdown. Browser and model calls stay on their owning workers; close releases browser profiles, workers and resident models.

## Compatibility

The supported product backend is MLX on Apple Silicon macOS. Explicit v2/v3 and versionless research configurations retain their separate compatibility behavior. Implicit saved v3 model selections migrate to v4. Legacy ordering/evidence classes and formatting remain available; they are not the v4 organizer protocol.
