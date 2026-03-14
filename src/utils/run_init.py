"""
Use:
Startup and runtime persistence utilities for the TCS application.

This module manages:
- deterministic seed + configuration loading,
- SQLite schema preparation for audit logs,
- rolling runtime config snapshots (single overwrite file),
- manual config-log saves when requested by the user.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# Runtime directories used by the project.
DATA_DIRS = [
    "data/logs",
    "data/logs/configs",
    "data/logs/saved_configs",
    "data/maps",
    "data/metrics",
    "data/models",
    "data/replays",
    "data/screenshots",
]

# Rolling runtime config path (single file replaced during runtime changes).
ROLLING_CONFIG_REL = Path("data/logs/configs/runtime_config.json")

# Manual config snapshots are written only when user explicitly saves logs.
MANUAL_CONFIG_DIR_REL = Path("data/logs/saved_configs")


DEFAULT_CONFIG: Dict[str, Any] = {
    "scenario": {
        "map_size": "default",
        "vehicles": 10,
        "episode_len": 300,
        "preview_road_density": 0.72,
        "preview_structure_density": 0.62,
    },
    "options": {"fps_cap": 60, "overlays": True, "device": "auto"},
    "train": {"gamma": 0.99, "clip_eps": 0.2, "batch_size": 2048},
}


@dataclass
class RunContext:
    """
    Use:
    Share startup/runtime state between GUI screens and backend helpers.

    Attributes:
    - base_dir: Project root directory.
    - data_dir: Runtime data folder (`base_dir/data`).
    - db_path: SQLite file path for run/episode/metric logs.
    - seed: Active deterministic seed for runtime.
    - config: Runtime configuration dictionary.
    - run_id: Active run primary key; `None` until TRAIN persistence begins.
    - created_at: UTC timestamp for context creation.
    - config_snapshot_path: Manual config-log save path when requested.
    """

    base_dir: Path
    data_dir: Path
    db_path: Path
    seed: int
    config: Dict[str, Any]
    run_id: Optional[int]
    created_at: datetime
    config_snapshot_path: Optional[Path]


def _project_root() -> Path:
    """
    Use:
    Resolve project root from `src/utils/run_init.py`.

    Inputs:
    - None.

    Output:
    Absolute project root path.
    """
    return Path(__file__).resolve().parents[2]


def ensure_data_dirs(base_dir: Optional[Path] = None) -> Path:
    """
    Use:
    Ensure all runtime data directories exist.

    Inputs:
    - base_dir: Optional override root path.

    Output:
    Root path used to create/validate runtime directories.
    """
    root = base_dir or _project_root()
    for rel_path in DATA_DIRS:
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    return root


def resolve_seed(user_seed: Optional[int] = None) -> int:
    """
    Use:
    Resolve active seed value for deterministic runtime behaviour.

    Inputs:
    - user_seed: Optional fixed seed from caller/UI.

    Output:
    Integer seed in [0, 2^31 - 1].
    """
    if user_seed is not None:
        return int(user_seed)
    return random.randint(0, 2**31 - 1)


def load_config(config_path: Optional[Path] = None, seed: int = 0) -> Dict[str, Any]:
    """
    Use:
    Load configuration from JSON file or defaults and attach active seed.

    Inputs:
    - config_path: Optional JSON config path.
    - seed: Seed value inserted into resulting config.

    Output:
    Mutable configuration dictionary for runtime.
    """
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        # Use JSON round-trip so nested defaults are copied, not referenced.
        config = json.loads(json.dumps(DEFAULT_CONFIG))

    config["seed"] = int(seed)
    return config


def reseed_all(seed: int) -> None:
    """
    Use:
    Apply active seed to available randomness providers.

    Inputs:
    - seed: Integer seed value.

    Output:
    None. Global RNG states are updated in-place.
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
    """
    Use:
    Open/create SQLite DB and ensure schema required for logging exists.

    Inputs:
    - db_path: SQLite file path.

    Output:
    Open sqlite3 connection with schema prepared.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT    NOT NULL,
            mode        TEXT    DEFAULT 'UNSET',
            seed        INTEGER NOT NULL,
            config_json TEXT    NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL,
            mode        TEXT    NOT NULL,
            seed        INTEGER NOT NULL,
            created_at  TEXT    NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL,
            episode_id  INTEGER,
            key         TEXT    NOT NULL,
            value       REAL    NOT NULL,
            step        INTEGER,
            created_at  TEXT    NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id),
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );
        """
    )
    conn.commit()
    return conn


def create_run_record(db_path: Path, seed: int, config: Dict[str, Any], mode: str) -> Tuple[int, str]:
    """
    Use:
    Insert a run row into SQLite logging table.

    Inputs:
    - db_path: SQLite database path.
    - seed: Run seed value.
    - config: Run configuration snapshot (serialized to JSON).
    - mode: Run mode label (e.g. TRAIN, EVAL).

    Output:
    Tuple `(run_id, created_at_iso)`.
    """
    conn = sqlite3.connect(db_path)
    created_at_iso = datetime.now(timezone.utc).isoformat()
    config_json = json.dumps(config, sort_keys=True)
    cursor = conn.execute(
        "INSERT INTO runs (created_at, mode, seed, config_json) VALUES (?, ?, ?, ?)",
        (created_at_iso, mode, int(seed), config_json),
    )
    conn.commit()
    run_id = int(cursor.lastrowid)
    conn.close()
    return run_id, created_at_iso


def create_episode_record(
    db_path: Path,
    run_id: int,
    mode: str,
    seed: int,
) -> Tuple[int, str]:
    """
    Use:
    Insert one episode row for a run before metrics are logged.

    Inputs:
    - db_path: SQLite database path.
    - run_id: Parent run primary key.
    - mode: Mode label (eg: TRAIN).
    - seed: Episode seed.

    Output:
    Tuple (episode_id, created_at_iso).
    """
    conn = sqlite3.connect(db_path)
    created_at_iso = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO episodes (run_id, mode, seed, created_at) VALUES (?, ?, ?, ?)",
        (int(run_id), mode, int(seed), created_at_iso),
    )
    conn.commit()
    episode_id = int(cursor.lastrowid)
    conn.close()
    return episode_id, created_at_iso


def insert_metric_record(
    db_path: Path,
    run_id: int,
    episode_id: Optional[int],
    key: str,
    value: float,
    step: Optional[int] = None,
) -> None:
    """
    Use:
    Insert one numeric metric row linked to run/episode.

    Inputs:
    - db_path: SQLite database path.
    - run_id: Parent run primary key.
    - episode_id: Optional parent episode id.
    - key: Metric key label.
    - value: Metric numeric value.
    - step: Optional step/iteration index.

    Output:
    None.
    """
    conn = sqlite3.connect(db_path)
    created_at_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO metrics (run_id, episode_id, key, value, step, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (int(run_id), episode_id, str(key), float(value), step, created_at_iso),
    )
    conn.commit()
    conn.close()


def write_rolling_config(base_dir: Path, config: Dict[str, Any]) -> Path:
    """
    Use:
    Overwrite rolling runtime config file with the latest in-memory config.

    Inputs:
    - base_dir: Project root directory.
    - config: Runtime config to persist.

    Output:
    Path to rolling config JSON file.
    """
    rolling_path = base_dir / ROLLING_CONFIG_REL
    rolling_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return rolling_path


def save_manual_config_log(
    base_dir: Path,
    config: Dict[str, Any],
    run_id: Optional[int] = None,
    label: str = "manual",
) -> Path:
    """
    Use:
    Save a manual config snapshot into log archive directory.

    Inputs:
    - base_dir: Project root directory.
    - config: Runtime config to archive.
    - run_id: Optional run id to include in filename.
    - label: Optional filename label.

    Output:
    Path to saved config archive file.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch for ch in label if ch.isalnum() or ch in ("_", "-")).strip() or "manual"
    prefix = f"run_{run_id}_" if run_id is not None else ""
    file_name = f"{prefix}{safe_label}_{timestamp}.json"

    target_dir = base_dir / MANUAL_CONFIG_DIR_REL
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / file_name
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def clear_log_artefacts(base_dir: Optional[Path] = None, clear_db: bool = True) -> None:
    """
    Use:
    Remove existing log/config artefacts when starting a clean development cycle.

    Inputs:
    - base_dir: Optional project root override.
    - clear_db: If True, remove `data/metrics/tcs.db` when present.

    Output:
    None. Files are deleted in-place when they exist.
    """
    root = ensure_data_dirs(base_dir=base_dir)
    for rel in (Path("data/logs/configs"), Path("data/logs/saved_configs")):
        folder = root / rel
        if not folder.exists():
            continue
        for file_path in folder.glob("*"):
            if file_path.is_file():
                file_path.unlink()

    if clear_db:
        db_path = root / "data/metrics/tcs.db"
        if db_path.exists():
            db_path.unlink()


def init_run(user_seed: Optional[int] = None, config_path: Optional[Path] = None) -> RunContext:
    """
    Use:
    Initialise startup context without inserting a run row.

    Inputs:
    - user_seed: Optional fixed seed.
    - config_path: Optional external config path.

    Output:
    `RunContext` ready for GUI usage.
    """
    base_dir = ensure_data_dirs()
    seed = resolve_seed(user_seed)
    config = load_config(config_path, seed=seed)

    # Ensure randomness providers use the same startup seed.
    reseed_all(seed)

    db_path = base_dir / "data/metrics/tcs.db"
    open_db(db_path).close()

    # Rolling runtime config is always overwritten with latest values.
    write_rolling_config(base_dir, config)

    return RunContext(
        base_dir=base_dir,
        data_dir=base_dir / "data",
        db_path=db_path,
        seed=seed,
        config=config,
        run_id=None,
        created_at=datetime.now(timezone.utc),
        config_snapshot_path=None,
    )


def start_train_run(ctx: RunContext) -> RunContext:
    """
    Use:
    Persist TRAIN run metadata to SQLite.

    Inputs:
    - ctx: Existing run context.

    Output:
    Updated context with `run_id` and creation timestamp.
    """
    run_id, created_iso = create_run_record(ctx.db_path, ctx.seed, ctx.config, mode="TRAIN")
    return RunContext(
        base_dir=ctx.base_dir,
        data_dir=ctx.data_dir,
        db_path=ctx.db_path,
        seed=ctx.seed,
        config=ctx.config,
        run_id=run_id,
        created_at=datetime.fromisoformat(created_iso),
        config_snapshot_path=ctx.config_snapshot_path,
    )


