"""
Config loader for PMIS v2.
Loads hyperparameters.yaml and provides typed access to all settings.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict


_CONFIG: Dict[str, Any] = {}
_CONFIG_PATH = Path(__file__).parent.parent / "hyperparameters.yaml"


def load_config(path: str = None) -> Dict[str, Any]:
    """Load config from YAML file. Caches after first load."""
    global _CONFIG
    if _CONFIG:
        return _CONFIG

    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)

    _validate_config(_CONFIG)
    return _CONFIG


def get(key: str, default: Any = None) -> Any:
    """Get a single config value by key."""
    if not _CONFIG:
        load_config()
    return _CONFIG.get(key, default)


def get_all() -> Dict[str, Any]:
    """Get entire config dict."""
    if not _CONFIG:
        load_config()
    return _CONFIG.copy()


def reload(path: str = None) -> Dict[str, Any]:
    """Force reload config from disk."""
    global _CONFIG
    _CONFIG = {}
    return load_config(path)


def _validate_config(cfg: Dict[str, Any]):
    """Basic validation of required keys and weight sums."""
    # Check precision weights sum to 1.0
    pw = (cfg.get("precision_weight_anchors", 0) +
          cfg.get("precision_weight_recency", 0) +
          cfg.get("precision_weight_consistency", 0))
    assert abs(pw - 1.0) < 0.01, f"Precision weights sum to {pw}, expected 1.0"

    # Check score weights sum to 1.0
    sw = (cfg.get("score_weight_semantic", 0) +
          cfg.get("score_weight_hierarchy", 0) +
          cfg.get("score_weight_temporal", 0) +
          cfg.get("score_weight_precision", 0))
    assert abs(sw - 1.0) < 0.01, f"Score weights sum to {sw}, expected 1.0"

    # Check required keys exist
    required = [
        "poincare_dimensions", "poincare_curvature",
        "gamma_temperature", "gamma_bias",
        "embedding_dimensions", "temporal_embedding_dim",
    ]
    for key in required:
        assert key in cfg, f"Missing required config key: {key}"
