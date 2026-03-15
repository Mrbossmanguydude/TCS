"""
Helpers for managing up to 5 named network checkpoint slots.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SLOT_COUNT = 5
SLOT_NAME_LIMIT = 16


def _models_dir(base_dir: Path) -> Path:
    """
    Resolve `data/models` directory.
    """
    return base_dir / "data" / "models"


def slots_index_path(base_dir: Path) -> Path:
    """
    Resolve slot-index JSON path.
    """
    return _models_dir(base_dir) / "network_slots.json"


def slot_checkpoint_path(base_dir: Path, slot_id: int) -> Path:
    """
    Resolve checkpoint path for a slot.
    """
    return _models_dir(base_dir) / f"network_slot_{int(slot_id)}.pt"


def _default_slots() -> List[Dict[str, Any]]:
    """
    Build empty slot records.
    """
    return [
        {
            "slot_id": int(idx + 1),
            "name": f"Network {idx + 1}",
            "occupied": False,
            "saved_at": None,
            "seed": None,
            "phase": None,
            "level_index": None,
        }
        for idx in range(SLOT_COUNT)
    ]


def normalise_slot_name(name: str, limit: int = SLOT_NAME_LIMIT) -> str:
    """
    Clamp and sanitise a slot display name.
    """
    cleaned = "".join(ch for ch in str(name).strip() if ch.isalnum() or ch in " _-")
    if not cleaned:
        cleaned = "Network"
    return cleaned[: max(1, int(limit))]


def load_slots(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Load slot-index JSON; create defaults when missing/corrupt.
    """
    models_dir = _models_dir(base_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    index_path = slots_index_path(base_dir)
    if not index_path.exists():
        slots = _default_slots()
        index_path.write_text(json.dumps(slots, indent=2), encoding="utf-8")
        return slots
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("network slot index must be a list")
    except Exception:
        slots = _default_slots()
        index_path.write_text(json.dumps(slots, indent=2), encoding="utf-8")
        return slots

    defaults = _default_slots()
    merged: List[Dict[str, Any]] = []
    for idx in range(SLOT_COUNT):
        base = dict(defaults[idx])
        if idx < len(raw) and isinstance(raw[idx], dict):
            base.update(raw[idx])
        base["slot_id"] = int(idx + 1)
        base["name"] = normalise_slot_name(str(base.get("name", defaults[idx]["name"])))
        base["occupied"] = bool(base.get("occupied", False))
        merged.append(base)
    return merged


def save_slots(base_dir: Path, slots: List[Dict[str, Any]]) -> None:
    """
    Persist slot-index JSON.
    """
    models_dir = _models_dir(base_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    sanitised: List[Dict[str, Any]] = []
    for idx in range(SLOT_COUNT):
        entry = dict(slots[idx]) if idx < len(slots) else _default_slots()[idx]
        entry["slot_id"] = int(idx + 1)
        entry["name"] = normalise_slot_name(str(entry.get("name", f"Network {idx + 1}")))
        entry["occupied"] = bool(entry.get("occupied", False))
        sanitised.append(entry)
    slots_index_path(base_dir).write_text(json.dumps(sanitised, indent=2), encoding="utf-8")


def first_empty_slot(slots: List[Dict[str, Any]]) -> Optional[int]:
    """
    Return first available slot id, or None when full.
    """
    for entry in slots:
        if not bool(entry.get("occupied", False)):
            return int(entry.get("slot_id", 0))
    return None


def mark_slot_saved(
    slots: List[Dict[str, Any]],
    slot_id: int,
    *,
    seed: int,
    phase: int,
    level_index: int,
) -> None:
    """
    Mark one slot as occupied with standard metadata.
    """
    target = max(1, min(SLOT_COUNT, int(slot_id)))
    stamp = datetime.now(timezone.utc).isoformat()
    for entry in slots:
        if int(entry.get("slot_id", 0)) == target:
            entry["occupied"] = True
            entry["saved_at"] = stamp
            entry["seed"] = int(seed)
            entry["phase"] = int(phase)
            entry["level_index"] = int(level_index)
            if not str(entry.get("name", "")).strip():
                entry["name"] = f"Network {target}"
            return


def delete_slot(base_dir: Path, slots: List[Dict[str, Any]], slot_id: int) -> None:
    """
    Delete checkpoint for slot and reset slot metadata.
    """
    target = max(1, min(SLOT_COUNT, int(slot_id)))
    checkpoint = slot_checkpoint_path(base_dir, target)
    try:
        if checkpoint.exists():
            checkpoint.unlink()
    except Exception:
        pass
    for entry in slots:
        if int(entry.get("slot_id", 0)) == target:
            entry["occupied"] = False
            entry["saved_at"] = None
            entry["seed"] = None
            entry["phase"] = None
            entry["level_index"] = None
            if not str(entry.get("name", "")).strip():
                entry["name"] = f"Network {target}"
            return

