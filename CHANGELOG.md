# Changelog

All notable changes to ReflexRoute will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Expanded built-in priors to 130 current OpenRouter models across 23 providers.
- Added reproducible profile generation from an official Models API snapshot.
- Added profile provenance, modalities, context length, feature support, and
  token-price metadata.

### Changed

- Default routing now uses a bounded 14-model cross-provider shortlist instead
  of every bundled profile.
- Hard-budget estimates now use a documented 2,000-input/1,000-output token
  reference request.

## [0.1.0] - 2026-09-20

### Added

- Zero-shot routing from versioned built-in model priors.
- Few-shot adaptation from JSONL or CSV routing history.
- Balanced per-model TF-IDF retrieval with basic CJK support.
- Hard budget filtering before the Jev request.
- OpenRouter Jev structured Choice client using `typesafe/jev-1.13`.
- Python API, CLI, example history, typed errors, and test suite.

[Unreleased]: https://github.com/AIGNLAI/ReflexRoute/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AIGNLAI/ReflexRoute/releases/tag/v0.1.0
