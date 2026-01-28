"""Configuration loading from YAML file."""

from pathlib import Path
from typing import Optional
import yaml

_config: Optional[dict] = None


def load_config(path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    global _config

    if _config is not None and path is None:
        return _config

    search_paths = [
        path,
        Path.cwd() / "config.yaml",
        Path(__file__).parent / "config.yaml",
    ]

    for p in search_paths:
        if p and p.exists():
            with open(p) as f:
                _config = yaml.safe_load(f)
            return _config

    return None
