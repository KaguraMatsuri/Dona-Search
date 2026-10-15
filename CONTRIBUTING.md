# Contributing

Read ARCHITECTURE.md and DEVELOPMENT.md before changing the framework. Preserve explicit compatibility paths and the v4 rule that the model chooses IDs while the program copies source text.

## Local verification

Install the dev extra and run targeted lint/tests for the changed boundary. Run `bash scripts/validate.sh` for the offline aggregate check. Model quality, real-engine availability and memory measurements require their own explicit runs; do not present unit tests as evidence for those outcomes.

## Changes to engines and models

Use configuration-based XPath engines when that capability is sufficient. Native providers implement the current `SearchProvider` contract with `config.top_k` and `ProviderResult`; the old list-returning contract is `LegacySearchProvider`. Registry entries alone are not configurable engine plugins. See DEVELOPMENT.md for the minimal interface example.

Adapters must be bound to the exact prepared base and correct role. The v4 second role is `organizing`, not legacy `ranking`. Dataset annotations and evaluation reports must identify their framework protocol.

## Scope and review

Describe the concrete problem, resulting behavior, changed contracts and relevant validation. Include failures and limitations. Do not remove compatibility behavior, change resource/concurrency policy or expand user-visible flows as incidental cleanup.

Keep personal settings, credentials, model weights, caches, logs and local agent rules out of product commits. Framework code uses the existing MIT license; third-party artifacts retain their own terms. Publishing, deployment and remote Git operations are separate actions.
