"""
Episode replay persistence helpers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Slot-store constants                                                        #
# --------------------------------------------------------------------------- #

EPISODE_REPLAY_SLOT_COUNT = 5
EPISODE_REPLAY_NAME_LIMIT = 18

_SLOT_META_FILE = "episode_replay_slots.json"
_SLOT_FILE_TEMPLATE = "episode_slot_{slot_id}.json"


# --------------------------------------------------------------------------- #
# Path + naming helpers                                                       #
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    """
    Return the current UTC timestamp in ISO format.
    """
    return datetime.now(timezone.utc).isoformat()


def _replays_dir(base_dir: Path) -> Path:
    """
    Resolve and create the replay directory.
    """
    folder = base_dir / "data" / "replays"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _slot_meta_path(base_dir: Path) -> Path:
    """
    Resolve metadata JSON path for episode replay slots.
    """
    return _replays_dir(base_dir) / _SLOT_META_FILE


def episode_slot_path(base_dir: Path, slot_id: int) -> Path:
    """
    Resolve one slot JSON file path.
    """
    return _replays_dir(base_dir) / _SLOT_FILE_TEMPLATE.format(slot_id=int(slot_id))


def normalise_episode_replay_name(name: str, fallback: str) -> str:
    """
    Clamp and sanitise replay names for UI display.
    """
    clean = " ".join(str(name or "").split()).strip()
    if not clean:
        clean = fallback
    return clean[:EPISODE_REPLAY_NAME_LIMIT]


# --------------------------------------------------------------------------- #
# Slot metadata loading/saving                                                #
# --------------------------------------------------------------------------- #

def _default_slot_entry(slot_id: int) -> Dict[str, Any]:
    """
    Build one empty slot metadata entry.
    """
    return {
        "slot_id": int(slot_id),
        "name": f"Replay {int(slot_id)}",
        "kept": False,
        "has_data": False,
        "created_at": None,
        "updated_at": None,
        "seed": None,
        "network_name": "",
        "path": _SLOT_FILE_TEMPLATE.format(slot_id=int(slot_id)),
    }


def load_episode_replay_slots(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Load slot metadata, creating defaults when needed.
    """
    meta_path = _slot_meta_path(base_dir)
    slots: List[Dict[str, Any]] = []

    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                slots = [entry for entry in loaded if isinstance(entry, dict)]
        except Exception:
            slots = []

    by_id = {int(entry.get("slot_id", -1)): entry for entry in slots}
    normalised: List[Dict[str, Any]] = []
    for slot_id in range(1, EPISODE_REPLAY_SLOT_COUNT + 1):
        # Start from metadata if present, otherwise seed a default row.
        entry = dict(by_id.get(slot_id, _default_slot_entry(slot_id)))
        entry["slot_id"] = slot_id
        entry["name"] = normalise_episode_replay_name(entry.get("name", ""), f"Replay {slot_id}")
        entry["kept"] = bool(entry.get("kept", False))
        entry["path"] = _SLOT_FILE_TEMPLATE.format(slot_id=slot_id)
        path = episode_slot_path(base_dir, slot_id)
        # `has_data` is validated against disk existence to avoid stale UI
        # states after manual file deletion.
        entry["has_data"] = bool(entry.get("has_data", False)) and path.exists()
        if path.exists():
            entry["has_data"] = True
        normalised.append(entry)
    return normalised


def save_episode_replay_slots(base_dir: Path, slots: List[Dict[str, Any]]) -> Path:
    """
    Persist slot metadata to disk.
    """
    meta_path = _slot_meta_path(base_dir)
    meta_path.write_text(json.dumps(slots, indent=2), encoding="utf-8")
    return meta_path


def _parse_iso_or_min(value: Optional[str]) -> datetime:
    """
    Parse ISO timestamp with safe minimum fallback.
    """
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Save path selection                                                         #
# --------------------------------------------------------------------------- #

def _choose_episode_slot_for_save(slots: List[Dict[str, Any]]) -> Optional[int]:
    """
    Choose save target slot:
    1) first empty slot,
    2) oldest non-kept slot,
    3) None if all are kept.
    """
    for entry in slots:
        if not bool(entry.get("has_data", False)):
            return int(entry.get("slot_id", 0))

    candidates = [entry for entry in slots if not bool(entry.get("kept", False))]
    if not candidates:
        return None

    candidates.sort(
        key=lambda row: _parse_iso_or_min(
            row.get("updated_at") or row.get("created_at")
        )
    )
    return int(candidates[0].get("slot_id", 0))


# --------------------------------------------------------------------------- #
# Public slot operations                                                      #
# --------------------------------------------------------------------------- #

def save_episode_replay_to_slots(
    base_dir: Path,
    replay_data: Dict[str, Any],
    *,
    replay_name: str = "",
    seed: Optional[int] = None,
    network_name: str = "",
) -> Tuple[Optional[Path], str]:
    """
    Save one episode replay into the fixed-slot store.
    """
    slots = load_episode_replay_slots(base_dir)
    slot_id = _choose_episode_slot_for_save(slots)
    if slot_id is None:
        return None, "Replay not saved: all 5 slots are marked as kept."

    target = episode_slot_path(base_dir, slot_id)
    replacing = target.exists()
    now_iso = _now_iso()

    payload = dict(replay_data)
    # Store minimal provenance directly in payload so later exports/replays do
    # not depend on external metadata files.
    payload.setdefault("saved_at", now_iso)
    payload.setdefault("seed", seed)
    payload.setdefault("network_name", network_name)
    payload["slot_id"] = slot_id

    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    entry = next((row for row in slots if int(row.get("slot_id", 0)) == slot_id), None)
    if entry is None:
        entry = _default_slot_entry(slot_id)
        slots.append(entry)

    default_name = entry.get("name", f"Replay {slot_id}")
    entry["name"] = normalise_episode_replay_name(replay_name, str(default_name))
    entry["has_data"] = True
    entry["updated_at"] = now_iso
    entry["created_at"] = now_iso
    entry["seed"] = int(seed) if seed is not None else None
    entry["network_name"] = str(network_name or "")
    entry["path"] = _SLOT_FILE_TEMPLATE.format(slot_id=slot_id)
    entry["slot_id"] = slot_id

    # Slot metadata remains sorted for deterministic rendering in GUI lists.
    slots.sort(key=lambda row: int(row.get("slot_id", 0)))
    save_episode_replay_slots(base_dir, slots)

    if replacing:
        return target, f"Replay saved by replacing oldest non-kept slot ({slot_id})."
    return target, f"Replay saved to slot {slot_id}."


def delete_episode_replay_slot(base_dir: Path, slot_id: int) -> str:
    """
    Delete one replay slot payload and reset its metadata.
    """
    slot_id = int(slot_id)
    slots = load_episode_replay_slots(base_dir)
    target = episode_slot_path(base_dir, slot_id)
    if target.exists():
        target.unlink()

    entry = next((row for row in slots if int(row.get("slot_id", 0)) == slot_id), None)
    if entry is None:
        entry = _default_slot_entry(slot_id)
        slots.append(entry)
    else:
        default = _default_slot_entry(slot_id)
        entry.update(default)

    slots.sort(key=lambda row: int(row.get("slot_id", 0)))
    save_episode_replay_slots(base_dir, slots)
    return f"Deleted replay slot {slot_id}."


def rename_episode_replay_slot(base_dir: Path, slot_id: int, new_name: str) -> str:
    """
    Rename one replay slot entry.
    """
    slot_id = int(slot_id)
    slots = load_episode_replay_slots(base_dir)
    entry = next((row for row in slots if int(row.get("slot_id", 0)) == slot_id), None)
    if entry is None:
        return "Rename failed: slot does not exist."
    entry["name"] = normalise_episode_replay_name(new_name, f"Replay {slot_id}")
    save_episode_replay_slots(base_dir, slots)
    return f"Renamed replay slot {slot_id}."


def set_episode_replay_keep(base_dir: Path, slot_id: int, keep: bool) -> str:
    """
    Set keep/unkeep flag on one replay slot.
    """
    slot_id = int(slot_id)
    slots = load_episode_replay_slots(base_dir)
    entry = next((row for row in slots if int(row.get("slot_id", 0)) == slot_id), None)
    if entry is None:
        return "Keep toggle failed: slot does not exist."
    if not bool(entry.get("has_data", False)):
        return "Keep toggle ignored: slot is empty."
    entry["kept"] = bool(keep)
    save_episode_replay_slots(base_dir, slots)
    return f"Replay slot {slot_id} keep set to {bool(keep)}."


def load_episode_replay_slot_data(base_dir: Path, slot_id: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Load one replay slot payload JSON.

    Returns:
    - payload dict when load succeeds,
    - None when slot is empty/invalid,
    plus a user-facing status message.
    """
    slot_id = int(slot_id)
    path = episode_slot_path(base_dir, slot_id)
    if not path.exists():
        return None, f"Replay slot {slot_id} is empty."

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Replay slot {slot_id} load failed: {exc}"

    if not isinstance(payload, dict):
        return None, f"Replay slot {slot_id} payload is invalid."
    return payload, f"Replay slot {slot_id} loaded."
