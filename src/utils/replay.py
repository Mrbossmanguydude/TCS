"""
Replay persistence helpers for TRAIN episodes.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# INITIALISATION / STARTUP MATERIAL                                           #
# --------------------------------------------------------------------------- #

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# FUNCTIONS AND HELPERS                                                       #
# --------------------------------------------------------------------------- #

def save_episode_replay(
    base_dir: Path,
    replay_payload: Dict[str, Any],
    run_id: Optional[int],
    phase: int,
    level_index: int,
    episode_index: int,
) -> Path:
    """
    Type:
    FUNCTION.

    Purpose:
    Persist one episode replay payload to `data/replays` using a timestamped
    filename that includes run/phase/level/episode traceability.

    Inputs:
    - base_dir: Project root directory.
    - replay_payload: Replay data dictionary to serialize.
    - run_id: Optional run identifier linked to SQLite run rows.
    - phase: Curriculum phase number for the episode.
    - level_index: Map level index for the episode.
    - episode_index: Episode counter within the current run.

    Outputs:
    - Returnvalue: Absolute path of the written replay JSON file.
    """
    target_dir = base_dir / "data" / "replays"
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_token = f"run_{run_id}" if run_id is not None else "run_none"
    file_name = (
        f"{run_token}_p{int(phase)}_l{int(level_index) + 1}_"
        f"e{int(episode_index)}_{stamp}.json"
    )
    path = target_dir / file_name
    path.write_text(json.dumps(replay_payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
