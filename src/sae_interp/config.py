"""Typed configuration loading.

We keep all knobs in a YAML file (``configs/sae_gpt2small.yaml``) and parse it into
nested dataclasses here. Dataclasses give us attribute access (``cfg.train.lr``),
a single place to document every field, and light validation — much nicer than
passing raw dicts around.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    name: str = "gpt2"
    hook_name: str = "blocks.8.hook_resid_pre"
    layer: int = 8
    d_model: int = 768


@dataclass
class PretrainedSAEConfig:
    release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.8.hook_resid_pre"


@dataclass
class SAEConfig:
    expansion_factor: int = 4
    l1_coeff: float = 5.0e-4
    init_b_dec_to_mean: bool = True


@dataclass
class TrainConfig:
    lr: float = 4.0e-4
    steps: int = 20000
    batch_size: int = 4096
    seed: int = 0
    log_every: int = 100
    ckpt_every: int = 2000
    resample_dead: bool = True
    dead_feature_window: int = 2000


@dataclass
class DataConfig:
    dataset: str = "NeelNanda/pile-10k"
    ctx_len: int = 128
    n_sequences: int = 2000
    cache_dir: str = "activation_cache"
    store_dtype: str = "float16"


@dataclass
class MetricsConfig:
    l1_sweep: list[float] = field(
        default_factory=lambda: [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    )
    eval_batches: int = 50


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrained_sae: PretrainedSAEConfig = field(default_factory=PretrainedSAEConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    seed: int = 0
    results_dir: str = "results"

    # --- Derived convenience properties (not in the YAML) ---
    @property
    def d_sae(self) -> int:
        """Hidden width of the SAE dictionary = expansion_factor * d_model."""
        return self.sae.expansion_factor * self.model.d_model


def _build(dc_type: type, data: dict[str, Any]) -> Any:
    """Recursively construct a (possibly nested) dataclass from a plain dict,
    ignoring unknown keys and raising on type mismatches that matter."""
    if not is_dataclass(dc_type):
        return data
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(dc_type)}
    # Resolve real field types (with `from __future__ import annotations`, the raw
    # f.type is a *string*; get_type_hints turns it back into the actual class).
    hints = get_type_hints(dc_type)
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(
                f"Unknown config key '{key}' for {dc_type.__name__}. "
                f"Allowed: {sorted(known)}"
            )
        ftype = hints.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        else:
            kwargs[key] = value
    return dc_type(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a ``Config`` dataclass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(Config, raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Cheap sanity checks that catch the most common foot-guns early."""
    if cfg.sae.expansion_factor < 1:
        raise ValueError("sae.expansion_factor must be >= 1")
    if cfg.sae.l1_coeff < 0:
        raise ValueError("sae.l1_coeff must be >= 0")
    # The layer in the hook name should match the declared layer, otherwise the
    # SAE and the activations we harvest would be from different sites.
    expected = f"blocks.{cfg.model.layer}."
    if not cfg.model.hook_name.startswith(expected):
        raise ValueError(
            f"model.hook_name '{cfg.model.hook_name}' does not match "
            f"model.layer {cfg.model.layer} (expected prefix '{expected}'). "
            "Keep layer and hook_name in sync."
        )
    if cfg.data.store_dtype not in {"float16", "float32", "bfloat16"}:
        raise ValueError("data.store_dtype must be one of float16/float32/bfloat16")
