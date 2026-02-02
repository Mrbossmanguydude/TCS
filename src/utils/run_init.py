import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import random


# Directories under project root used for outputs/replays/metrics.
DATA_DIRS = [
    "data/logs",
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
    run_id: int
    created_at: datetime


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
    conn.commit()
    return conn


def create_run(conn: sqlite3.Connection, seed: int, config: Dict[str, Any], mode: str = "BOOT") -> int:
    """Insert a run row capturing seed + config snapshot; return new run_id."""
    created_at = datetime.now(timezone.utc).isoformat()
    config_json = json.dumps(config, sort_keys=True)
    cur = conn.execute(
        "INSERT INTO runs (created_at, mode, seed, config_json) VALUES (?, ?, ?, ?)",
        (created_at, mode, seed, config_json),
    )
    conn.commit()
    return cur.lastrowid


def init_run(user_seed: Optional[int] = None, config_path: Optional[Path] = None) -> RunContext:
    """
      - Ensures data directories exist,
      - Resolves seed + config snapshot,
      - Opens/creates the SQLite DB,
      - Inserts an initial run row.
    """
    base_dir = ensure_data_dirs()
    seed = resolve_seed(user_seed)
    cfg = load_config(config_path, seed=seed)
    db_path = base_dir / "data" / "metrics" / "tcs.db"
    conn = open_db(db_path)
    run_id = create_run(conn, seed, cfg, mode="BOOT")
    created_at = datetime.now(timezone.utc)
    return RunContext(
        base_dir=base_dir,
        data_dir=base_dir / "data",
        db_path=db_path,
        seed=seed,
        config=cfg,
        run_id=run_id,
        created_at=created_at,
    )
