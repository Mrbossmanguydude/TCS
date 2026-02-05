"""
Run initialisation and persistence scaffolding.

Design intent:
- DB stores audit-worthy TRAIN runs (and explicit saves), linking seed/config to metrics.
- Configs are less critical; keep them in memory, and only persist when a TRAIN run is recorded.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import random


# Directories under project root used for outputs/replays/metrics.
DATA_DIRS = [
    "data/logs",
    "data/logs/configs",       # optional rolling configs
    "data/logs/saved_configs", # configs persisted alongside TRAIN runs
    "data/maps",
    "data/metrics",
    "data/models",
    "data/replays",
    "data/screenshots",
]


@dataclass
class RunContext:
    base_dir: Path
    data_dir: Path
    db_path: Path
    seed: int
    config: Dict[str, Any]
    run_id: Optional[int]
    created_at: datetime
    config_snapshot_path: Optional[Path]


def _project_root() -> Path:
    """Assumes this file is at src/utils, step up two levels to repo root."""
    return Path(__file__).resolve().parents[2]


def ensure_data_dirs(base_dir: Optional[Path] = None) -> Path:
    """Create required data folders if they do not exist."""
    root = base_dir or _project_root()
    for rel in DATA_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def resolve_seed(user_seed: Optional[int] = None) -> int:
    """Return a single seed integer."""
    if user_seed is not None:
        return int(user_seed)
    # Time-based but deterministic once captured.
    return random.randint(0, 2**31 - 1)


def load_config(config_path: Optional[Path] = None, seed: int = 0) -> Dict[str, Any]:
    """
    Load a JSON config if provided; otherwise return a minimal default snapshot.
    This mirrors the design's separation of config vs seed.
    """
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    else:
        cfg = {
            "scenario": {"map_size": "default", "vehicles": 10, "episode_len": 300},
            "options": {"fps_cap": 60, "overlays": True, "device": "auto"},
            "train": {
                "gamma": 0.99,
                "clip_eps": 0.2,
                "batch_size": 2048,
            },
        }
    cfg["seed"] = seed
    return cfg


def reseed_all(seed: int) -> None:
    """
    Apply seed to Python random, NumPy, and PyTorch (if available).
    For reproducibility.
    """
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database at db_path and ensure schema exists."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT    NOT NULL,
            mode       TEXT    DEFAULT 'UNSET',
            seed       INTEGER NOT NULL,
            config_json TEXT   NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id     INTEGER NOT NULL,
            mode       TEXT    NOT NULL,
            seed       INTEGER NOT NULL,
            created_at TEXT    NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id     INTEGER NOT NULL,
            episode_id INTEGER,
            key        TEXT    NOT NULL,
            value      REAL    NOT NULL,
            step       INTEGER,
            created_at TEXT    NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id),
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );
        """
    )
    conn.commit()
    return conn


def create_run_record(db_path: Path, seed: int, config: Dict[str, Any], mode: str) -> Tuple[int, str]:
    """Insert a run row (TRAIN or explicit save) and return run_id plus ISO timestamp."""
    conn = sqlite3.connect(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    config_json = json.dumps(config, sort_keys=True)
    cur = conn.execute(
        "INSERT INTO runs (created_at, mode, seed, config_json) VALUES (?, ?, ?, ?)",
        (created_at, mode, seed, config_json),
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id, created_at


def save_persisted_config(base_dir: Path, run_id: int, config: Dict[str, Any]) -> Path:
    """Persist a versioned config tied to a TRAIN run."""
    path = base_dir / "data" / "logs" / "saved_configs" / f"run_{run_id}_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def init_run(user_seed: Optional[int] = None, config_path: Optional[Path] = None) -> RunContext:
    """
      - Ensures data directories exist,
      - Resolves seed,
      - Opens/creates the SQLite DB schema (no run row yet),
      - Returns context; TRAIN run rows are created when training starts.
    """
    base_dir = ensure_data_dirs()
    seed = resolve_seed(user_seed)
    cfg = load_config(config_path, seed=seed)
    reseed_all(seed)
    db_path = base_dir / "data" / "metrics" / "tcs.db"
    open_db(db_path).close()
    cfg_snapshot = None  # only persist when a TRAIN run is recorded
    created_at = datetime.now(timezone.utc)
    return RunContext(
        base_dir=base_dir,
        data_dir=base_dir / "data",
        db_path=db_path,
        seed=seed,
        config=cfg,
        run_id=None,
        created_at=created_at,
        config_snapshot_path=cfg_snapshot,
    )


def start_train_run(ctx: RunContext) -> RunContext:
    """
    Create a DB run row for TRAIN and persist a saved config alongside it.
    """
    run_id, created_iso = create_run_record(ctx.db_path, ctx.seed, ctx.config, mode="TRAIN")
    saved_cfg = save_persisted_config(ctx.base_dir, run_id, ctx.config)
    return RunContext(
        base_dir=ctx.base_dir,
        data_dir=ctx.data_dir,
        db_path=ctx.db_path,
        seed=ctx.seed,
        config=ctx.config,
        run_id=run_id,
        created_at=datetime.fromisoformat(created_iso),
        config_snapshot_path=saved_cfg,
    )
