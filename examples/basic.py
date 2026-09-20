"""Run with: OPENROUTER_API_KEY=... python examples/basic.py"""

from pathlib import Path

from reflexroute import Router


models = [
    "openai/gpt-5.6-sol",
    "google/gemini-3.8-flash",
    "anthropic/claude-sonnet-5",
]

# Zero-shot: built-in, explicitly versioned priors only.
router = Router(models=models)
zero_shot = router.route("Implement a distributed rate limiter.")
print("zero-shot:", zero_shot.to_dict())

# In-context: balanced top-K evidence from your own observations.
router.load_history(Path(__file__).with_name("history.jsonl"))
adapted = router.route("Implement a distributed rate limiter.")
print("in-context:", adapted.to_dict())
