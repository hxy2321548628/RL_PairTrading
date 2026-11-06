"""Collision-free run directories and reproducibility metadata."""

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from rl_pairtrading.config import ProjectConfig


def create_run_directory(config: ProjectConfig) -> Path:
    name = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory = config.paths.output_dir / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("rl-pairtrading", "numpy", "pandas", "torch", "pydantic", "statsmodels")
        },
    }
    (directory / "runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory
