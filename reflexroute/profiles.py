"""Load the bundled, versioned model priors."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import ConfigurationError, ModelProfile


def load_profiles(path: str | Path | None = None) -> dict[str, ModelProfile]:
    """Load model profiles from the bundled YAML file or a user-supplied file."""

    if path is None:
        profile_path = Path(str(files("reflexroute").joinpath("profiles/models.yaml")))
    else:
        profile_path = Path(path)
    try:
        text = profile_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read model profiles: {profile_path}") from exc

    raw = _load_minimal_yaml(text)
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Model profiles must be a YAML mapping.")
    version = raw.get("profile_version")
    source = raw.get("source")
    models = raw.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ConfigurationError("Model profiles must contain a non-empty 'models' mapping.")

    result: dict[str, ModelProfile] = {}
    for raw_name, raw_profile in models.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_profile, Mapping):
            raise ConfigurationError(f"Invalid model profile: {raw_name!r}.")
        description = str(raw_profile.get("description", "")).strip()
        strengths = raw_profile.get("strengths", [])
        if not description or not isinstance(strengths, list):
            raise ConfigurationError(
                f"Profile {name!r} needs a description and a strengths list."
            )
        estimated_cost = _profile_cost(raw_profile)
        known = {
            "description",
            "strengths",
            "cost_level",
            "latency_level",
            "estimated_cost",
            "estimated_cost_per_request",
            "profile_version",
            "source",
            "default_candidate",
        }
        result[name] = ModelProfile(
            model=name,
            description=description,
            strengths=tuple(str(item) for item in strengths),
            cost_level=_optional_string(raw_profile.get("cost_level")),
            latency_level=_optional_string(raw_profile.get("latency_level")),
            estimated_cost=estimated_cost,
            profile_version=_optional_string(raw_profile.get("profile_version", version)),
            source=raw_profile.get("source", source),
            default_candidate=raw_profile.get("default_candidate") is True,
            extra={key: value for key, value in raw_profile.items() if key not in known},
        )
    return result


def unknown_profile(model: str) -> ModelProfile:
    """Create an explicit fallback prior without inventing model capabilities."""

    return ModelProfile(
        model=model,
        description=(
            "No built-in prior is available. Base the decision on relevant user "
            "history; otherwise treat this candidate as having unknown capabilities."
        ),
        source="user-supplied candidate; no bundled profile",
        extra={"profile_available": False},
    )


def _profile_cost(profile: Mapping[str, Any]) -> float | None:
    value = profile.get("estimated_cost", profile.get("estimated_cost_per_request"))
    if value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Profile estimated_cost must be numeric.") from exc
    if cost < 0:
        raise ConfigurationError("Profile estimated_cost cannot be negative.")
    return cost


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _load_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small mapping/list YAML subset used by bundled profiles.

    Keeping this format intentionally simple avoids making every ReflexRoute user
    install a full YAML stack merely to read a few built-in priors.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    pending_lists: dict[tuple[int, str], list[Any]] = {}
    for line_number, original in enumerate(text.splitlines(), start=1):
        if not original.strip() or original.lstrip().startswith("#"):
            continue
        if "\t" in original[: len(original) - len(original.lstrip())]:
            raise ConfigurationError(f"Tabs are not supported in profiles (line {line_number}).")
        indent = len(original) - len(original.lstrip(" "))
        stripped = original.strip()
        if stripped.startswith("- "):
            key = (indent, "list")
            target = pending_lists.get(key)
            if target is None:
                raise ConfigurationError(f"Unexpected YAML list item on line {line_number}.")
            target.append(_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            raise ConfigurationError(f"Invalid profile YAML on line {line_number}.")
        raw_key, raw_value = stripped.split(":", 1)
        name = str(_scalar(raw_key.strip()))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            raise ConfigurationError(f"Invalid indentation on line {line_number}.")
        parent = stack[-1][1]
        value_text = raw_value.strip()
        if value_text:
            parent[name] = _scalar(value_text)
            continue

        # A key with no inline value is a list when its next meaningful line is
        # a dash, otherwise it is a mapping. Look ahead without a YAML dependency.
        next_content = _next_content(text.splitlines(), line_number)
        is_list = bool(next_content and next_content[0] > indent and next_content[1].startswith("- "))
        if is_list:
            child_list: list[Any] = []
            parent[name] = child_list
            pending_lists[(next_content[0], "list")] = child_list
        else:
            child_map: dict[str, Any] = {}
            parent[name] = child_map
            stack.append((indent, child_map))
    return root


def _next_content(lines: list[str], current_line_number: int) -> tuple[int, str] | None:
    for line in lines[current_line_number:]:
        if line.strip() and not line.lstrip().startswith("#"):
            return len(line) - len(line.lstrip(" ")), line.strip()
    return None


def _scalar(value: str) -> Any:
    if not value:
        return ""
    if value[0:1] in {'"', "'"} and value[-1:] == value[0]:
        if value[0] == '"':
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value[1:-1]
    lowered = value.casefold()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return float(value) if any(character in value for character in ".eE") else int(value)
    except ValueError:
        return value
