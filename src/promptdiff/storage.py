"""Storage operations for run outputs."""

from __future__ import annotations

import json
from pathlib import Path
from promptdiff.models import RunOutput


def save_run(run_output: RunOutput, output_dir: Path | str = "runs") -> Path:
    """Save a RunOutput instance to disk as formatted JSON."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    file_name = f"{run_output.run_id}.json"
    file_path = dir_path / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        # Use Pydantic's model_dump_json or json.dump for clean indentation
        data = run_output.model_dump(mode="json")
        json.dump(data, f, indent=2, ensure_ascii=False)

    return file_path
