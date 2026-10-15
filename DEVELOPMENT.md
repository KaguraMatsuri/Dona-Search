# Framework development

Use the source checkout and install the `dev` extra. The formal product path is schema v4. See ARCHITECTURE.md for ownership and QUICKSTART.md for configuration.

## Checks

```bash
bash scripts/validate.sh
```

The entry point compiles sources, validates the bundled configuration and runs offline tests. It avoids real browser sessions and model-weight loading. Legacy tokenizer test fixtures must already be cached. `DONA_PYTHON` can select the Python interpreter. Targeted lint and tests should cover changed modules during iteration; unrelated existing findings must not be hidden by disabling rules.

## Search extension

For an engine with an HTML results page, configure `search_xpath` and page/result XPath rules. This uses the existing browser lifecycle, timeout and status handling.

A native provider consumed by the current scheduler has:

```python
from dona_search.interfaces import ProviderResult

class Provider:
    def __init__(self, config, client):
        self.config, self.client, self.name = config, client, config.name

    def search(self, query: str, k: int) -> ProviderResult:
        return ProviderResult(self.name, 'empty')
```

This is an interface example, not a complete new-engine registration. New native engine types require explicit configuration/registry integration. Registering an arbitrary name does not bypass the schema's allowed provider types. `LegacySearchProvider` describes the old list-returning research interface.

Return source text and URLs rather than model-generated fields. Invalid provider results become isolated parse failures; another engine may continue. Preserve query parameter encoding and source provenance.

## Model extension

Custom complete MLX bundles use the existing model selection fields. They must support the chat template and strict role protocol. The planner returns a queries object; the organizer returns a complete order and nullable segment selections. Do not replace these contracts with generated prose.

Component injection through `build_query`, `build_organizer` and `browser_factory` supports isolated tests. The application runtime loads both models; `organizer_only` is an explicit component-evaluation path. Do not infer live model quality from injected-component tests.

## Failure and compatibility tests

Cover startup and warmup failures, invalid model JSON, source-ID mismatches, empty/blocked engines, cancellation, context overflow, model/cache preparation, configuration preservation and resource cleanup. Explicit legacy configurations should retain their documented behavior. Use fake models and temporary paths for control-flow tests, and real tokenizer metadata when measuring prompt size.
