# Contributing to ReflexRoute

Thanks for helping improve ReflexRoute. Small, focused changes with tests are
the easiest to review.

## Development setup

```bash
git clone https://github.com/AIGNLAI/ReflexRoute.git
cd ReflexRoute
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

## Pull requests

1. Open an issue first for large features or behavior changes.
2. Create a branch from `main`.
3. Add or update tests for observable behavior.
4. Run the complete test suite locally.
5. Update README or CHANGELOG when public behavior changes.
6. Keep commits and the pull request focused on one concern.

Do not commit API keys, routing histories containing private prompts, generated
build artifacts, or model-profile claims without a traceable source.

## Design principles

- Keep routing training-free and simple to adopt.
- Apply hard constraints such as budgets in deterministic code.
- Treat built-in model profiles as fallible, versioned priors.
- Prefer user evidence over generic priors when it is relevant.
- Avoid mandatory services beyond OpenRouter's Jev decision call.
- Preserve a small runtime dependency surface.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
