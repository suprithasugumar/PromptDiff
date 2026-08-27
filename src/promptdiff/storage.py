"""Storage operations for run outputs and baseline tracking."""

from __future__ import annotations

import json
import re
from pathlib import Path
from promptdiff.models import RunOutput


def _slugify(text: str) -> str:
    """Convert a string to a safe filesystem identifier."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", text).strip("_").lower()


def save_run(run_output: RunOutput, output_dir: Path | str = "runs") -> Path:
    """Save a RunOutput instance to disk as formatted JSON."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    file_name = f"{run_output.run_id}.json"
    file_path = dir_path / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        data = run_output.model_dump(mode="json")
        json.dump(data, f, indent=2, ensure_ascii=False)

    return file_path


def save_baseline(run_output: RunOutput, output_dir: Path | str = "runs") -> Path:
    """Save a RunOutput instance as the primary baseline for its test suite."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    suite_slug = _slugify(run_output.suite_name)
    baseline_path = dir_path / f"baseline_{suite_slug}.json"

    with open(baseline_path, "w", encoding="utf-8") as f:
        data = run_output.model_dump(mode="json")
        json.dump(data, f, indent=2, ensure_ascii=False)

    return baseline_path


def get_latest_baseline(suite_name: str, output_dir: Path | str = "runs") -> RunOutput | None:
    """Find and load the latest baseline RunOutput for the given suite name."""
    dir_path = Path(output_dir)
    if not dir_path.exists():
        return None

    suite_slug = _slugify(suite_name)
    primary_baseline = dir_path / f"baseline_{suite_slug}.json"
    if primary_baseline.exists():
        return load_run(primary_baseline)

    # Fallback: look for any file matching baseline pattern for this suite
    matching = sorted(
        dir_path.glob(f"*baseline*{suite_slug}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matching:
        return load_run(matching[0])

    return None


def load_run(file_path: Path | str) -> RunOutput:
    """Load and parse a RunOutput from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return RunOutput.model_validate(data)

