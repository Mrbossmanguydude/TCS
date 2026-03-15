"""
Replay browser screen with fixed-slot episode replay management and playback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame

from src.utils.replay import (
    EPISODE_REPLAY_NAME_LIMIT,
    delete_episode_replay_slot,
    load_episode_replay_slot_data,
    load_episode_replay_slots,
    rename_episode_replay_slot,
    set_episode_replay_keep,
)
from src.utils.network_slots import (
    SLOT_NAME_LIMIT,
    delete_slot,
    load_slots,
    save_slots,
    slot_checkpoint_path,
)
from src.utils.map_generation import generate_phase_map


# --------------------------------------------------------------------------- #
# Theme constants                                                             #
# --------------------------------------------------------------------------- #

BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
PANEL_BG = (46, 53, 58)
PANEL_BORDER = (78, 91, 99)
HILITE = (255, 215, 0)
BUTTON_BG = (80, 92, 104)
BUTTON_BG_ACTIVE = (90, 150, 220)
RED = (165, 60, 60)
MAP_EMPTY_COLOUR = (34, 39, 44)
MAP_GRID_COLOUR = (56, 63, 70)
ROAD_COLOUR = (145, 162, 182)
ROAD_STRUCTURE_LIGHT = (166, 166, 166)
SPAWN_COLOUR = (112, 216, 104)
DEST_COLOUR = (86, 198, 92)


# --------------------------------------------------------------------------- #
# Replay browser screen                                                       #
# --------------------------------------------------------------------------- #

class ReplayScreen:
    """
    Replay browser UI with three main views:
    - chooser,
    - episode slot management,
    - per-step replay player.
    """

    def __init__(
        self,
        screen_rect: pygame.Rect,
        font_path: Path | None = None,
        run_ctx: Any = None,
        ui_offsets: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.run_ctx = run_ctx
        self.ui_offsets = ui_offsets or {}

        self.view = "chooser"
        self.status_message = "Select a replay browser mode."
        self.episode_slots: List[Dict[str, Any]] = []
        self.network_slots: List[Dict[str, Any]] = []

        # Episode slot name editing state.
        self.edit_slot_id: Optional[int] = None
        self.edit_buffer = ""
        self.edit_kind: Optional[str] = None

        # Player runtime state.
        self.player_slot_id: Optional[int] = None
        self.player_payload: Dict[str, Any] = {}
        self.player_frames: List[Dict[str, Any]] = []
        self.player_index = 0
        self.player_playing = False
        self.player_finished = False
        self.player_speed = 1.0
        self.player_steps_per_second = 12.0
        self.player_step_accumulator = 0.0
        self.player_last_tick_ms: Optional[int] = None
        self.player_generated_map_grid: Optional[List[List[int]]] = None
        self.player_generated_node_grid: Optional[List[List[str]]] = None

        # Compatibility hooks used by existing gui_main state flow.
        self.load_request_mode = False
        self.return_state = "MENU"
        self._loaded_network_selection: Optional[Tuple[Path, Dict[str, Any]]] = None

        # Training-style preview asset cache for replay rendering.
        self._scaled_cache: Dict[tuple[str, int], pygame.Surface] = {}
        self.road_h_img: Optional[pygame.Surface] = None
        self.road_v_img: Optional[pygame.Surface] = None
        self.car_img: Optional[pygame.Surface] = None

        self._build_layout()
        self._load_preview_assets()
        self._reload_network_slots()
        self._reload_episode_slots()

    def _font(self, size: int) -> pygame.font.Font:
        return pygame.font.Font(self.font_path, size) if self.font_path else pygame.font.SysFont(None, size)

    def _project_root(self) -> Path:
        """
        Resolve project root for asset loading.
        """
        if self.run_ctx is not None:
            return Path(self.run_ctx.base_dir)
        return Path(__file__).resolve().parents[2]

    def _load_image(self, candidates: List[Path]) -> Optional[pygame.Surface]:
        """
        Load first existing image from candidate paths.
        """
        for path in candidates:
            if path.exists():
                try:
                    return pygame.image.load(path.as_posix()).convert_alpha()
                except Exception:
                    return None
        return None

    def _trim_transparent_padding(self, image: Optional[pygame.Surface]) -> Optional[pygame.Surface]:
        """
        Crop transparent sprite borders so scaled road tiles fill cells.
        """
        if image is None:
            return None
        bounds = image.get_bounding_rect(min_alpha=1)
        if bounds.width <= 0 or bounds.height <= 0:
            return image
        if bounds.width == image.get_width() and bounds.height == image.get_height():
            return image
        return image.subsurface(bounds).copy()

    def _load_preview_assets(self) -> None:
        """
        Load the same road/car preview assets used in Train.
        """
        primary_root = self._project_root()
        fallback_root = Path(__file__).resolve().parents[2]
        roots = [primary_root]
        if fallback_root != primary_root:
            roots.append(fallback_root)

        self.road_h_img = self._trim_transparent_padding(
            self._load_image(
                [base / "assets" / "road_imgs" / "road_horizontal.png" for base in roots]
                + [base / "assets" / "road_horizontal.png" for base in roots]
            )
        )
        self.road_v_img = self._trim_transparent_padding(
            self._load_image(
                [base / "assets" / "road_imgs" / "road_vertical.png" for base in roots]
                + [base / "assets" / "road_vertical.png" for base in roots]
            )
        )
        self.car_img = self._load_image(
            [base / "assets" / "road_imgs" / "vehicle_imgs" / "car.png" for base in roots]
            + [base / "assets" / "vehicle_imgs" / "car.png" for base in roots]
        )

    def _scaled_asset(self, key: str, image: Optional[pygame.Surface], tile: int) -> Optional[pygame.Surface]:
        """
        Return cached sprite scaled to tile size.
        """
        if image is None:
            return None
        cache_key = (key, int(tile))
        cached = self._scaled_cache.get(cache_key)
        if cached is not None:
            return cached
        scaled = pygame.transform.smoothscale(image, (int(tile), int(tile)))
        self._scaled_cache[cache_key] = scaled
        return scaled

    def _build_layout(self) -> None:
        """
        Build static rectangles used by all replay views.
        """
        self.back_button = pygame.Rect(34, 28, 96, 46)
        self.title_rect = pygame.Rect(0, 18, self.screen_rect.width, 52)
        self.body_rect = pygame.Rect(34, 98, self.screen_rect.width - 68, self.screen_rect.height - 132)

        # Chooser buttons.
        choice_w = 320
        choice_h = 80
        gap = 30
        total_w = choice_w * 2 + gap
        left = self.screen_rect.centerx - total_w // 2
        y = self.screen_rect.centery - choice_h // 2
        self.networks_button = pygame.Rect(left, y, choice_w, choice_h)
        self.episodes_button = pygame.Rect(left + choice_w + gap, y, choice_w, choice_h)
        self.export_results_button = pygame.Rect(self.screen_rect.centerx - 210, self.networks_button.bottom + 24, 420, 62)

        # Episodes list layout.
        self.rows_top = self.body_rect.y + 70
        self.row_h = 72
        self.row_gap = 10
        self.row_x = self.body_rect.x + 20
        self.row_w = self.body_rect.width - 40

        # Player layout mirrors TrainScreen DURING TRAINING frontend geometry.
        self.player_back_button = pygame.Rect(34, 28, 96, 46)
        self.player_restart_button = pygame.Rect(160, 28, 146, 46)
        self.player_play_pause_button = pygame.Rect(472, 28, 130, 46)
        self.player_speed_minus_button = pygame.Rect(612, 28, 48, 46)
        self.player_speed_plus_button = pygame.Rect(788, 28, 48, 46)
        self.player_speed_label_rect = pygame.Rect(670, 28, 108, 46)
        self.player_left_panel = pygame.Rect(34, 108, 304, self.screen_rect.height - 132)
        self.player_right_panel = pygame.Rect(356, 108, self.screen_rect.width - 390, self.screen_rect.height - 132)
        self.player_map_rect = self.player_right_panel.inflate(-26, -84)
        self.player_map_rect.y += 28

    def _reload_episode_slots(self) -> None:
        base_dir = getattr(self.run_ctx, "base_dir", None)
        if base_dir is None:
            self.episode_slots = []
            return
        self.episode_slots = load_episode_replay_slots(base_dir)

    def _reload_network_slots(self) -> None:
        """
        Load network checkpoint slots used by TRAIN save/load workflow.
        """
        base_dir = getattr(self.run_ctx, "base_dir", None)
        if base_dir is None:
            self.network_slots = []
            return
        self.network_slots = load_slots(base_dir)

    @staticmethod
    def _safe_file_token(value: str, fallback: str = "network") -> str:
        """
        Build a filesystem-safe token for export filenames.
        """
        filtered = "".join(ch for ch in str(value) if ch.isalnum() or ch in ("-", "_"))
        return filtered or fallback

    def _selected_network_for_export(self) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """
        Return the currently selected network slot payload for export.
        """
        if self._loaded_network_selection is None:
            return None
        checkpoint_path, slot_meta = self._loaded_network_selection
        if not checkpoint_path.exists():
            return None
        return checkpoint_path, dict(slot_meta)

    def _collect_network_replay_metrics(
        self,
        base_dir: Path,
        network_name: str,
        network_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Collect replay metrics associated with one network.

        Primary match uses exact network name.
        Fallback match uses seed when replay network tag is missing/generic.
        """
        target_name = str(network_name).strip().lower()

        def _is_generic_name(value: str) -> bool:
            token = str(value).strip().lower()
            return token in {"", "current policy", "n/a", "none", "unknown"}

        slots = load_episode_replay_slots(base_dir)
        matching_slots: List[Tuple[Dict[str, Any], str]] = []
        for slot in slots:
            if not bool(slot.get("has_data", False)):
                continue
            slot_network_name = str(slot.get("network_name", "")).strip()
            slot_seed = slot.get("seed")
            slot_seed_int = int(slot_seed) if isinstance(slot_seed, (int, float)) else None
            slot_name_match = str(slot_network_name).strip().lower() == target_name and bool(target_name)
            slot_seed_match = (
                network_seed is not None
                and slot_seed_int is not None
                and int(slot_seed_int) == int(network_seed)
                and _is_generic_name(slot_network_name)
            )
            if slot_name_match:
                matching_slots.append((slot, "slot_name"))
            elif slot_seed_match:
                matching_slots.append((slot, "slot_seed_fallback"))

        metric_totals: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        replay_rows: List[Dict[str, Any]] = []

        for slot, slot_match_reason in matching_slots:
            slot_id = int(slot.get("slot_id", 0))
            payload, _ = load_episode_replay_slot_data(base_dir, slot_id)
            if not isinstance(payload, dict):
                continue

            payload_network_name = str(payload.get("network_name", "")).strip()
            payload_seed = payload.get("seed")
            payload_seed_int = int(payload_seed) if isinstance(payload_seed, (int, float)) else None
            payload_name_match = str(payload_network_name).strip().lower() == target_name and bool(target_name)
            payload_seed_match = (
                network_seed is not None
                and payload_seed_int is not None
                and int(payload_seed_int) == int(network_seed)
                and _is_generic_name(payload_network_name)
            )
            if payload_name_match:
                match_reason = "payload_name"
            elif payload_seed_match:
                match_reason = "payload_seed_fallback"
            else:
                match_reason = slot_match_reason

            metrics_final = payload.get("metrics_final")
            metrics_map = metrics_final if isinstance(metrics_final, dict) else {}
            replay_rows.append(
                {
                    "slot_id": slot_id,
                    "name": str(slot.get("name", f"Replay {slot_id}")),
                    "seed": payload.get("seed"),
                    "phase": payload.get("phase"),
                    "level_index": payload.get("level_index"),
                    "episode_index": payload.get("episode_index"),
                    "passed": bool(payload.get("passed", False)),
                    "created_at": payload.get("created_at"),
                    "match_reason": match_reason,
                    "metrics": {
                        key: float(value)
                        for key, value in metrics_map.items()
                        if isinstance(value, (int, float))
                    },
                }
            )

            for key, value in metrics_map.items():
                if isinstance(value, (int, float)):
                    metric_totals[key] = float(metric_totals.get(key, 0.0) + float(value))
                    metric_counts[key] = int(metric_counts.get(key, 0) + 1)

        metric_averages = {
            key: float(metric_totals[key] / float(metric_counts[key]))
            for key in metric_totals
            if metric_counts.get(key, 0) > 0
        }

        replay_rows.sort(key=lambda row: str(row.get("created_at") or ""))
        return {
            "matching_replay_count": len(replay_rows),
            "metrics_average": metric_averages,
            "replays": replay_rows,
        }

    def _export_loaded_network_results(self) -> None:
        """
        Export JSON summary for metrics relevant to the selected loaded network.
        """
        base_dir = getattr(self.run_ctx, "base_dir", None)
        if base_dir is None:
            self.status_message = "No network is chosen."
            return

        selected = self._selected_network_for_export()
        if selected is None:
            self.status_message = "No network is chosen."
            return

        checkpoint_path, slot_meta = selected
        network_name = str(slot_meta.get("name", "")).strip()
        if not network_name:
            self.status_message = "No network is chosen."
            return

        export_payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "network": {
                "slot_id": int(slot_meta.get("slot_id", 0)),
                "name": network_name,
                "seed": slot_meta.get("seed"),
                "phase": slot_meta.get("phase"),
                "level_index": slot_meta.get("level_index"),
                "saved_at": slot_meta.get("saved_at"),
                "checkpoint_path": checkpoint_path.as_posix(),
            },
            "results": self._collect_network_replay_metrics(
                base_dir=Path(base_dir),
                network_name=network_name,
                network_seed=int(slot_meta.get("seed")) if isinstance(slot_meta.get("seed"), (int, float)) else None,
            ),
        }

        export_dir = Path(base_dir) / "data" / "logs" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        token = self._safe_file_token(network_name, fallback=f"slot_{int(slot_meta.get('slot_id', 0))}")
        out_path = export_dir / f"network_results_{token}_{stamp}.json"
        out_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")

        replay_count = int(export_payload["results"].get("matching_replay_count", 0))
        self.status_message = f"Exported {replay_count} replay result(s) to {out_path.name}."

    def _row_rect(self, slot_index: int) -> pygame.Rect:
        y = self.rows_top + slot_index * (self.row_h + self.row_gap)
        return pygame.Rect(self.row_x, y, self.row_w, self.row_h)

    @staticmethod
    def _slot_by_id(slots: List[Dict[str, Any]], slot_id: int) -> Optional[Dict[str, Any]]:
        return next((slot for slot in slots if int(slot.get("slot_id", 0)) == int(slot_id)), None)

    @staticmethod
    def _circle_hit_test(pos: Tuple[int, int], centre: Tuple[int, int], radius: int) -> bool:
        dx = pos[0] - centre[0]
        dy = pos[1] - centre[1]
        return dx * dx + dy * dy <= radius * radius

    def _start_rename(self, slot_id: int) -> None:
        """
        Start rename flow for an episode replay slot.
        """
        slot = self._slot_by_id(self.episode_slots, slot_id)
        if slot is None:
            return
        self.edit_kind = "episode"
        self.edit_slot_id = slot_id
        self.edit_buffer = str(slot.get("name", f"Replay {slot_id}"))
        self.status_message = f"Renaming slot {slot_id}. Enter to save, ESC to cancel."

    def _start_network_rename(self, slot_id: int) -> None:
        """
        Start rename flow for a network checkpoint slot.
        """
        slot = self._slot_by_id(self.network_slots, slot_id)
        if slot is None:
            return
        self.edit_kind = "network"
        self.edit_slot_id = slot_id
        self.edit_buffer = str(slot.get("name", f"Network {slot_id}"))
        self.status_message = f"Renaming network slot {slot_id}. Enter to save, ESC to cancel."

    def _cancel_rename(self) -> None:
        """
        Cancel any active rename flow.
        """
        self.edit_slot_id = None
        self.edit_buffer = ""
        self.edit_kind = None
        self.status_message = "Rename cancelled."

    def _commit_rename(self) -> None:
        """
        Save rename edits to either episode slots or network slots.
        """
        if self.edit_slot_id is None:
            return
        base_dir = getattr(self.run_ctx, "base_dir", None)
        if base_dir is None:
            self._cancel_rename()
            return

        if self.edit_kind == "network":
            slot_id = int(self.edit_slot_id)
            self._reload_network_slots()
            slot = self._slot_by_id(self.network_slots, slot_id)
            if slot is not None:
                slot["name"] = str(self.edit_buffer or "").strip()[:SLOT_NAME_LIMIT] or f"Network {slot_id}"
                save_slots(base_dir, self.network_slots)
                self.status_message = f"Renamed network slot {slot_id}."
                self._reload_network_slots()
            else:
                self.status_message = f"Rename failed: network slot {slot_id} missing."
        else:
            self.status_message = rename_episode_replay_slot(base_dir, self.edit_slot_id, self.edit_buffer)
            self._reload_episode_slots()

        self.edit_slot_id = None
        self.edit_buffer = ""
        self.edit_kind = None

    @staticmethod
    def _extract_frames(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract an ordered frame list from a replay payload.

        This keeps playback compatible with multiple serialisation shapes.
        """
        if isinstance(payload.get("frames"), list):
            return [frame for frame in payload["frames"] if isinstance(frame, dict)]
        if isinstance(payload.get("steps"), list):
            return [frame for frame in payload["steps"] if isinstance(frame, dict)]
        if isinstance(payload.get("timeline"), list):
            return [frame for frame in payload["timeline"] if isinstance(frame, dict)]
        return []

    @staticmethod
    def _extract_grid(frame: Dict[str, Any], payload: Dict[str, Any]) -> Optional[List[List[Any]]]:
        """
        Read a tile grid from either frame-level or payload-level fields.
        """
        def read_from(source: Dict[str, Any]) -> Optional[List[List[Any]]]:
            for key in ("grid", "map_grid", "tiles"):
                grid = source.get(key)
                if isinstance(grid, list) and grid and isinstance(grid[0], list):
                    return grid  # type: ignore[return-value]
            map_block = source.get("map")
            if isinstance(map_block, dict):
                for key in ("grid", "map_grid", "tiles"):
                    grid = map_block.get(key)
                    if isinstance(grid, list) and grid and isinstance(grid[0], list):
                        return grid  # type: ignore[return-value]
            if isinstance(map_block, list) and map_block and isinstance(map_block[0], list):
                return map_block  # type: ignore[return-value]
            return None

        return read_from(frame) or read_from(payload)

    @staticmethod
    def _cell_is_road(value: Any) -> bool:
        """
        Classify grid-cell value as driveable road or non-road.
        """
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return float(value) > 0.0

        token = str(value).strip().lower()
        if token in {"1", "true", "road", "lane", "turn", "junction", "roundabout", "r"}:
            return True
        if any(part in token for part in ("road", "lane", "junction", "turn", "roundabout")):
            return True
        return False

    @staticmethod
    def _extract_points(items: Any) -> List[Tuple[float, float]]:
        """
        Extract `(x, y)` pairs from flexible dict/list point structures.
        """
        points: List[Tuple[float, float]] = []
        if not isinstance(items, list):
            return points
        for item in items:
            x_val: Optional[float] = None
            y_val: Optional[float] = None
            if isinstance(item, dict):
                if "x" in item and "y" in item:
                    x_val, y_val = float(item["x"]), float(item["y"])
                elif "col" in item and "row" in item:
                    x_val, y_val = float(item["col"]), float(item["row"])
                elif isinstance(item.get("pos"), list) and len(item["pos"]) >= 2:
                    x_val, y_val = float(item["pos"][0]), float(item["pos"][1])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x_val, y_val = float(item[0]), float(item[1])
            if x_val is not None and y_val is not None:
                points.append((x_val, y_val))
        return points

    @staticmethod
    def _extract_vehicle_rows(frame: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract structured vehicle rows for replay map rendering.
        """
        raw = frame.get("vehicles", payload.get("vehicles", []))
        if not isinstance(raw, list):
            return []
        rows: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            position = item.get("position", item.get("current", [0.0, 0.0]))
            destination = item.get("destination", [0, 0])
            if not (isinstance(position, (list, tuple)) and len(position) >= 2):
                continue
            if not (isinstance(destination, (list, tuple)) and len(destination) >= 2):
                destination = [int(round(float(position[0]))), int(round(float(position[1])))]
            rows.append(
                {
                    "position": (float(position[0]), float(position[1])),
                    "heading_deg": float(item.get("heading_deg", 0.0)),
                    "destination": (int(destination[0]), int(destination[1])),
                }
            )
        return rows

    def _road_orientation_from_grid(self, grid: List[List[Any]], x_pos: int, y_pos: int, node_type: str = "") -> str:
        """
        Infer road tile orientation from neighbouring road links.
        """
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        left = x_pos > 0 and self._cell_is_road(grid[y_pos][x_pos - 1])
        right = x_pos < (cols - 1) and self._cell_is_road(grid[y_pos][x_pos + 1])
        up = y_pos > 0 and self._cell_is_road(grid[y_pos - 1][x_pos])
        down = y_pos < (rows - 1) and self._cell_is_road(grid[y_pos + 1][x_pos])
        horizontal_links = int(left) + int(right)
        vertical_links = int(up) + int(down)
        if node_type == "road_two_lane":
            if horizontal_links >= vertical_links and horizontal_links > 0:
                return "horizontal"
            if vertical_links > 0:
                return "vertical"
        has_h = horizontal_links > 0
        has_v = vertical_links > 0
        if has_h and not has_v:
            return "horizontal"
        if has_v and not has_h:
            return "vertical"
        return "mixed"

    @staticmethod
    def _extract_node_type_grid(frame: Dict[str, Any], payload: Dict[str, Any]) -> Optional[List[List[str]]]:
        """
        Read optional node-type matrix used for junction/turn styling.
        """
        def read_from(source: Dict[str, Any]) -> Optional[List[List[str]]]:
            for key in ("map_node_types", "node_types_grid"):
                grid = source.get(key)
                if isinstance(grid, list) and grid and isinstance(grid[0], list):
                    return [[str(cell) for cell in row] for row in grid]
            map_block = source.get("map")
            if isinstance(map_block, dict):
                for key in ("map_node_types", "node_types_grid"):
                    grid = map_block.get(key)
                    if isinstance(grid, list) and grid and isinstance(grid[0], list):
                        return [[str(cell) for cell in row] for row in grid]
            return None

        return read_from(frame) or read_from(payload)

    def _draw_replay_map_preview(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        grid: List[List[Any]],
        node_types: Optional[List[List[str]]],
        vehicles: List[Dict[str, Any]],
    ) -> None:
        """
        Render replay map/vehicles using Train preview visual style.
        """
        pygame.draw.rect(screen, MAP_EMPTY_COLOUR, rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 1, border_radius=8)

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        if rows <= 0 or cols <= 0:
            return

        tile = max(4, min(rect.width // cols, rect.height // rows))
        grid_w = tile * cols
        grid_h = tile * rows
        ox = rect.x + (rect.width - grid_w) // 2
        oy = rect.y + (rect.height - grid_h) // 2

        road_h_cache: Dict[tuple[int, int], pygame.Surface] = {}
        road_v_cache: Dict[tuple[int, int], pygame.Surface] = {}
        car_sprite = self._scaled_asset("car", self.car_img, max(8, int(tile * 0.80)))

        def _blit_scaled(
            image: Optional[pygame.Surface],
            cache: Dict[tuple[int, int], pygame.Surface],
            cell_rect: pygame.Rect,
        ) -> bool:
            if image is None:
                return False
            key = (max(1, cell_rect.width), max(1, cell_rect.height))
            sprite = cache.get(key)
            if sprite is None:
                sprite = pygame.transform.smoothscale(image, key)
                cache[key] = sprite
            screen.blit(sprite, cell_rect.topleft)
            return True

        structure_fill_types = {
            "junction_turn_one_lane",
            "junction_turn_two_lane",
            "junction_t",
            "junction_cross",
            "junction_centre",
            "road_turn",
        }

        for y_pos in range(rows):
            for x_pos in range(cols):
                cell = pygame.Rect(ox + x_pos * tile, oy + y_pos * tile, tile, tile)
                pygame.draw.rect(screen, MAP_EMPTY_COLOUR, cell)
                pygame.draw.rect(screen, MAP_GRID_COLOUR, cell, 1)
                if not self._cell_is_road(grid[y_pos][x_pos]):
                    continue

                node_type = ""
                if node_types is not None and y_pos < len(node_types) and x_pos < len(node_types[y_pos]):
                    node_type = str(node_types[y_pos][x_pos] or "")
                orientation = self._road_orientation_from_grid(grid, x_pos, y_pos, node_type=node_type)
                if node_type in structure_fill_types or orientation == "mixed":
                    pygame.draw.rect(screen, ROAD_STRUCTURE_LIGHT, cell)
                elif orientation == "horizontal":
                    pygame.draw.rect(screen, ROAD_COLOUR, cell)
                    _blit_scaled(self.road_h_img, road_h_cache, cell)
                elif orientation == "vertical":
                    pygame.draw.rect(screen, ROAD_COLOUR, cell)
                    _blit_scaled(self.road_v_img, road_v_cache, cell)
                else:
                    pygame.draw.rect(screen, ROAD_COLOUR, cell)

        for destination_index, vehicle in enumerate(vehicles, start=1):
            position = vehicle.get("position", (0.0, 0.0))
            destination = vehicle.get("destination", (0, 0))
            heading = float(vehicle.get("heading_deg", 0.0))

            current_centre = (int(ox + (float(position[0]) * tile)), int(oy + (float(position[1]) * tile)))
            current_rect = pygame.Rect(0, 0, tile, tile)
            current_rect.center = current_centre
            dest_rect = pygame.Rect(
                ox + int(destination[0]) * tile,
                oy + int(destination[1]) * tile,
                tile,
                tile,
            )

            if car_sprite is not None:
                rotated = pygame.transform.rotozoom(car_sprite, -heading, 1.0)
                car_rect = rotated.get_rect(center=current_rect.center)
                screen.blit(rotated, car_rect)
            else:
                pygame.draw.circle(screen, SPAWN_COLOUR, current_rect.center, max(2, tile // 4))

            dest_radius = max(3, int(tile * 0.48))
            pygame.draw.circle(screen, DEST_COLOUR, dest_rect.center, dest_radius)
            pygame.draw.circle(screen, (26, 102, 32), dest_rect.center, dest_radius, 1)
            label = self._font(max(8, int(tile * 0.50))).render(str(destination_index), True, (0, 0, 0))
            screen.blit(label, label.get_rect(center=dest_rect.center))

    def _start_player(self, slot_id: int) -> None:
        """
        Enter player view and initialise frame-step playback state.
        """
        base_dir = getattr(self.run_ctx, "base_dir", None)
        if base_dir is None:
            self.status_message = "Replay start failed: base directory missing."
            return

        payload, status = load_episode_replay_slot_data(base_dir, slot_id)
        if payload is None:
            self.status_message = status
            return

        frames = self._extract_frames(payload)
        if not frames:
            self.status_message = f"Replay slot {slot_id} has no frame timeline to play."
            return

        self.player_slot_id = int(slot_id)
        self.player_payload = payload
        self.player_frames = frames
        self.player_index = 0
        self.player_playing = True
        self.player_finished = False
        self.player_speed = 1.0
        self.player_step_accumulator = 0.0
        self.player_steps_per_second = float(payload.get("replay_fps", 12.0))
        self.player_last_tick_ms = None
        self.player_generated_map_grid = None
        self.player_generated_node_grid = None

        # Legacy replays may not include map tiles. For those, regenerate the
        # map deterministically from saved seed/phase/level so playback visuals
        # still match training layout.
        if not isinstance(payload.get("map_grid"), list):
            self._build_generated_map_from_payload(payload)

        self.view = "player"
        self.status_message = f"Running replay slot {slot_id} ({len(frames)} steps)."

    def _restart_player(self) -> None:
        """
        Restart playback from step zero.
        """
        if not self.player_frames:
            return
        self.player_index = 0
        self.player_playing = True
        self.player_finished = False
        self.player_step_accumulator = 0.0
        self.player_last_tick_ms = None
        self.status_message = "Replay restarted from step 1."

    def _build_generated_map_from_payload(self, payload: Dict[str, Any]) -> None:
        """
        Reconstruct replay map tiles from payload metadata when explicit map
        grids are absent (legacy replay files).
        """
        seed = int(payload.get("seed", 0))
        phase = int(payload.get("phase", 1))
        level_index = int(payload.get("level_index", 0))
        road_density = float(payload.get("road_density", 0.72))
        structure_density = float(payload.get("structure_density", 0.62))
        try:
            generated = generate_phase_map(
                seed=seed,
                phase=phase,
                level_index=level_index,
                road_density=road_density,
                structure_density=structure_density,
            )
        except Exception:
            self.player_generated_map_grid = None
            self.player_generated_node_grid = None
            return

        width = int(generated.width)
        height = int(generated.height)
        self.player_generated_map_grid = [
            [1 if (x_pos, y_pos) in generated.roads else 0 for x_pos in range(width)]
            for y_pos in range(height)
        ]
        self.player_generated_node_grid = [
            [str(generated.node_types.get((x_pos, y_pos), "")) for x_pos in range(width)]
            for y_pos in range(height)
        ]

    def enter(self, view: str = "chooser", load_request_mode: bool = False, return_state: str = "MENU") -> None:
        """
        External screen-entry hook used by `gui_main`.

        Args:
        - view: Initial replay sub-view.
        - load_request_mode: True when entered from TRAIN for network loading flow.
        - return_state: Target state to return to on back/escape while in load mode.
        """
        self.view = view if view in {"chooser", "networks", "episodes", "player"} else "chooser"
        self.load_request_mode = bool(load_request_mode)
        self.return_state = str(return_state or "MENU")
        self.edit_slot_id = None
        self.edit_buffer = ""
        self.edit_kind = None
        self.player_last_tick_ms = None
        self._reload_network_slots()
        self._reload_episode_slots()
        if self.load_request_mode:
            self.status_message = "Replay load mode active: select a network slot to load."
        else:
            self.status_message = "Select a replay browser mode."

    def consume_loaded_network(self) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """
        Return and clear any selected network slot payload.

        This is a compatibility API for `gui_main`; network-slot selection is
        currently not implemented in this replay screen revision.
        """
        selected = self._loaded_network_selection
        self._loaded_network_selection = None
        return selected

    def _tick_player(self, dt_seconds: float) -> None:
        """
        Advance replay by frame-steps while playback is active.
        """
        if not self.player_playing or self.player_finished or not self.player_frames:
            return

        # Replay speed scales logical step rate; UI frame-rate only controls
        # how often we check for another step.
        steps_per_second = max(0.25, self.player_steps_per_second * self.player_speed)
        step_duration = 1.0 / steps_per_second
        self.player_step_accumulator += max(0.0, dt_seconds)

        while self.player_step_accumulator >= step_duration:
            self.player_step_accumulator -= step_duration
            self.player_index += 1
            if self.player_index >= len(self.player_frames):
                self.player_index = len(self.player_frames) - 1
                self.player_playing = False
                self.player_finished = True
                self.status_message = "Replay ended. Press RESTART to play again."
                break

    def handle_events(self, events: List[pygame.event.Event]) -> str:
        """
        Process input and return next GUI state token.
        """
        for event in events:
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                if self.edit_slot_id is not None:
                    self._cancel_rename()
                elif self.view == "player":
                    self.view = "episodes"
                    self.status_message = "Returned to episode replay slots."
                elif self.view == "chooser":
                    return self.return_state if self.load_request_mode else "MENU"
                else:
                    if self.load_request_mode and self.view == "networks":
                        return self.return_state
                    self.view = "chooser"
                    self.status_message = "Select a replay browser mode."
                continue

            if event.type == pygame.KEYDOWN:
                if self.edit_slot_id is not None:
                    name_limit = SLOT_NAME_LIMIT if self.edit_kind == "network" else EPISODE_REPLAY_NAME_LIMIT
                    if event.key == pygame.K_RETURN:
                        self._commit_rename()
                    elif event.key == pygame.K_ESCAPE:
                        self._cancel_rename()
                    elif event.key == pygame.K_BACKSPACE:
                        self.edit_buffer = self.edit_buffer[:-1]
                    else:
                        text = event.unicode or ""
                        if text.isprintable() and len(self.edit_buffer) < name_limit:
                            self.edit_buffer += text
                    continue

                if self.view == "player" and event.key == pygame.K_SPACE and self.player_frames:
                    self.player_playing = not self.player_playing
                    self.status_message = "Replay resumed." if self.player_playing else "Replay paused."
                continue

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                pos = event.pos
                # Back handling is centralised so behaviour stays consistent
                # across chooser, slots, and in-player views.
                back_hit = self.back_button.collidepoint(pos) or (
                    self.view == "player" and self.player_back_button.collidepoint(pos)
                )
                if back_hit:
                    if self.edit_slot_id is not None:
                        self._cancel_rename()
                    elif self.view == "player":
                        self.view = "episodes"
                        self.status_message = "Returned to episode replay slots."
                    elif self.view == "chooser":
                        return self.return_state if self.load_request_mode else "MENU"
                    else:
                        if self.load_request_mode and self.view == "networks":
                            return self.return_state
                        self.view = "chooser"
                        self.status_message = "Select a replay browser mode."
                    continue

                if self.view == "chooser":
                    if self.networks_button.collidepoint(pos):
                        self.view = "networks"
                        self._reload_network_slots()
                        if self.load_request_mode:
                            self.status_message = "Load mode: click LOAD on an occupied network slot."
                        else:
                            self.status_message = "Manage network checkpoint slots."
                    elif self.episodes_button.collidepoint(pos):
                        self.view = "episodes"
                        self._reload_episode_slots()
                        self.status_message = "Episode replay slots: click RUN on a slot to replay."
                    elif self.export_results_button.collidepoint(pos):
                        self._export_loaded_network_results()
                    continue

                if self.view == "networks":
                    base_dir = getattr(self.run_ctx, "base_dir", None)
                    if base_dir is None:
                        continue

                    # Slot controls are row-local so each operation maps to one
                    # deterministic checkpoint target.
                    for row_index, slot in enumerate(self.network_slots):
                        row = self._row_rect(row_index)
                        slot_id = int(slot.get("slot_id", 0))
                        occupied = bool(slot.get("occupied", False))
                        load_rect = pygame.Rect(row.right - 245, row.y + 18, 66, 36)
                        delete_rect = pygame.Rect(row.right - 170, row.y + 18, 64, 36)
                        rename_rect = pygame.Rect(row.right - 95, row.y + 18, 74, 36)

                        if load_rect.collidepoint(pos):
                            if not occupied:
                                self.status_message = f"Network slot {slot_id} is empty."
                                break
                            checkpoint = slot_checkpoint_path(base_dir, slot_id)
                            if not checkpoint.exists():
                                self.status_message = f"Checkpoint missing for slot {slot_id}."
                                break
                            self._loaded_network_selection = (checkpoint, dict(slot))
                            self.status_message = f"Selected network slot {slot_id}."
                            if self.load_request_mode:
                                return self.return_state
                            break

                        if delete_rect.collidepoint(pos):
                            delete_slot(base_dir, self.network_slots, slot_id)
                            save_slots(base_dir, self.network_slots)
                            self._reload_network_slots()
                            if self._loaded_network_selection is not None:
                                selected_slot_id = int(self._loaded_network_selection[1].get("slot_id", 0))
                                if selected_slot_id == slot_id:
                                    self._loaded_network_selection = None
                            if self.edit_slot_id == slot_id and self.edit_kind == "network":
                                self._cancel_rename()
                            self.status_message = f"Deleted network slot {slot_id}."
                            break

                        if rename_rect.collidepoint(pos):
                            self._start_network_rename(slot_id)
                            break

                if self.view == "episodes":
                    base_dir = getattr(self.run_ctx, "base_dir", None)
                    if base_dir is None:
                        continue

                    # Episode slot controls support keep/run/delete/rename from
                    # one row, matching network-slot interaction style.
                    for row_index, slot in enumerate(self.episode_slots):
                        row = self._row_rect(row_index)
                        slot_id = int(slot.get("slot_id", 0))
                        has_data = bool(slot.get("has_data", False))

                        keep_radius = 11
                        keep_centre = (row.right - 330, row.centery)
                        run_rect = pygame.Rect(row.right - 245, row.y + 18, 66, 36)
                        delete_rect = pygame.Rect(row.right - 170, row.y + 18, 64, 36)
                        rename_rect = pygame.Rect(row.right - 95, row.y + 18, 74, 36)

                        if self._circle_hit_test(pos, keep_centre, keep_radius):
                            new_keep = not bool(slot.get("kept", False))
                            self.status_message = set_episode_replay_keep(base_dir, slot_id, new_keep)
                            self._reload_episode_slots()
                            break

                        if run_rect.collidepoint(pos):
                            if not has_data:
                                self.status_message = f"Replay slot {slot_id} is empty."
                            else:
                                self._start_player(slot_id)
                            break

                        if delete_rect.collidepoint(pos):
                            self.status_message = delete_episode_replay_slot(base_dir, slot_id)
                            if self.edit_slot_id == slot_id:
                                self.edit_slot_id = None
                                self.edit_buffer = ""
                            self._reload_episode_slots()
                            break

                        if rename_rect.collidepoint(pos):
                            self._start_rename(slot_id)
                            break

                if self.view == "player":
                    if self.player_restart_button.collidepoint(pos):
                        self._restart_player()
                    elif self.player_play_pause_button.collidepoint(pos) and self.player_frames:
                        self.player_playing = not self.player_playing
                        self.status_message = "Replay resumed." if self.player_playing else "Replay paused."
                    elif self.player_speed_minus_button.collidepoint(pos):
                        self.player_speed = max(0.25, self.player_speed - 0.25)
                    elif self.player_speed_plus_button.collidepoint(pos):
                        self.player_speed = min(10.0, self.player_speed + 0.25)

        return "REPLAYS"

    def _draw_button(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        *,
        active: bool = False,
        text_size: int = 28,
        fill: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        colour = fill if fill is not None else (BUTTON_BG_ACTIVE if active else BUTTON_BG)
        pygame.draw.rect(screen, colour, rect, border_radius=8)
        pygame.draw.rect(screen, HILITE if active else PANEL_BORDER, rect, 2, border_radius=8)
        label = self._font(text_size).render(text, True, FG)
        screen.blit(label, label.get_rect(center=rect.center))

    def _draw_training_style_button(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        text_size: int,
        border_colour: tuple[int, int, int] = HILITE,
    ) -> None:
        """
        Draw a button using the same visual style as TrainScreen controls.
        """
        gradient = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for y_pos in range(rect.height):
            blend = y_pos / max(1, rect.height - 1)
            red = int(90 * (1 - blend) + 50 * blend)
            green = int(95 * (1 - blend) + 52 * blend)
            blue = int(100 * (1 - blend) + 55 * blend)
            pygame.draw.line(gradient, (red, green, blue), (0, y_pos), (rect.width, y_pos))

        mask = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=10)
        gradient.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(gradient, rect.topleft)
        pygame.draw.rect(screen, border_colour, rect, 2, border_radius=10)

        font = self._font(text_size)
        label = font.render(text, True, FG)
        shadow = font.render(text, True, (0, 0, 0))
        text_rect = label.get_rect(center=rect.center)
        screen.blit(shadow, text_rect.move(1, 1))
        screen.blit(label, text_rect)

    def _draw_chooser(self, screen: pygame.Surface) -> None:
        # Keep both chooser buttons visually identical so neither appears pre-selected.
        self._draw_button(screen, self.networks_button, "NETWORK SLOTS", active=True, text_size=34)
        self._draw_button(screen, self.episodes_button, "EPISODE REPLAYS", active=True, text_size=34)
        export_active = self._selected_network_for_export() is not None
        export_fill = (70, 135, 96) if export_active else BUTTON_BG
        self._draw_button(screen, self.export_results_button, "EXPORT RESULTS (JSON)", active=export_active, text_size=30, fill=export_fill)

        hint_font = self._font(24)
        hint = hint_font.render("Choose what to browse from REPLAYS.", True, ACCENT)
        screen.blit(hint, hint.get_rect(center=(self.screen_rect.centerx, self.export_results_button.bottom + 34)))

        selected = self._selected_network_for_export()
        selected_text = "Loaded network: none selected"
        if selected is not None:
            selected_text = f"Loaded network: {str(selected[1].get('name', 'N/A'))}"
        selected_label = self._font(20).render(selected_text, True, ACCENT)
        screen.blit(selected_label, selected_label.get_rect(center=(self.screen_rect.centerx, self.export_results_button.bottom + 62)))

    def _draw_networks(self, screen: pygame.Surface) -> None:
        """
        Draw network checkpoint slot management list.
        """
        header_font = self._font(26)
        row_font = self._font(24)
        small_font = self._font(20)

        title = "Network Slots (max 5)"
        if self.load_request_mode:
            title = "Network Slots (load mode)"
        screen.blit(header_font.render(title, True, FG), (self.body_rect.x + 18, self.body_rect.y + 18))

        for row_index, slot in enumerate(self.network_slots):
            row = self._row_rect(row_index)
            pygame.draw.rect(screen, (58, 66, 72), row, border_radius=8)
            pygame.draw.rect(screen, PANEL_BORDER, row, 2, border_radius=8)

            slot_id = int(slot.get("slot_id", 0))
            occupied = bool(slot.get("occupied", False))
            name = str(slot.get("name", f"Network {slot_id}"))
            seed = slot.get("seed", None)
            phase = slot.get("phase", None)
            level = slot.get("level_index", None)

            title_text = f"{slot_id}. {name}"
            subtitle = "Empty slot"
            if occupied:
                subtitle = f"Seed: {seed if seed is not None else 'N/A'} | Phase: {phase if phase is not None else 'N/A'} | Level: {level if level is not None else 'N/A'}"

            screen.blit(row_font.render(title_text, True, FG), (row.x + 14, row.y + 10))
            screen.blit(small_font.render(subtitle, True, ACCENT), (row.x + 14, row.y + 42))

            load_rect = pygame.Rect(row.right - 245, row.y + 18, 66, 36)
            delete_rect = pygame.Rect(row.right - 170, row.y + 18, 64, 36)
            rename_rect = pygame.Rect(row.right - 95, row.y + 18, 74, 36)
            self._draw_button(screen, load_rect, "LOAD", fill=(56, 130, 82), text_size=20, active=occupied)
            self._draw_button(screen, delete_rect, "DEL", fill=RED, text_size=24)
            self._draw_button(screen, rename_rect, "EDIT", text_size=22)

            if self.edit_kind == "network" and self.edit_slot_id == slot_id:
                edit_rect = pygame.Rect(row.x + 360, row.y + 16, 240, 40)
                pygame.draw.rect(screen, (36, 42, 47), edit_rect, border_radius=6)
                pygame.draw.rect(screen, HILITE, edit_rect, 2, border_radius=6)
                edit_text = small_font.render(self.edit_buffer or "_", True, FG)
                screen.blit(edit_text, (edit_rect.x + 8, edit_rect.y + 10))

    def _draw_episodes(self, screen: pygame.Surface) -> None:
        header_font = self._font(26)
        row_font = self._font(24)
        small_font = self._font(20)

        title = header_font.render("Episode Replay Slots (max 5)", True, FG)
        screen.blit(title, (self.body_rect.x + 18, self.body_rect.y + 18))

        for row_index, slot in enumerate(self.episode_slots):
            row = self._row_rect(row_index)
            pygame.draw.rect(screen, (58, 66, 72), row, border_radius=8)
            pygame.draw.rect(screen, PANEL_BORDER, row, 2, border_radius=8)

            slot_id = int(slot.get("slot_id", 0))
            has_data = bool(slot.get("has_data", False))
            name = str(slot.get("name", f"Replay {slot_id}"))
            seed = slot.get("seed", None)
            network_name = str(slot.get("network_name", "")) or "N/A"

            title_text = f"{slot_id}. {name}"
            subtitle = "Empty slot"
            if has_data:
                subtitle = f"Seed: {seed if seed is not None else 'N/A'} | Network: {network_name}"

            screen.blit(row_font.render(title_text, True, FG), (row.x + 14, row.y + 10))
            screen.blit(small_font.render(subtitle, True, ACCENT), (row.x + 14, row.y + 42))

            keep_radius = 11
            keep_centre = (row.right - 330, row.centery)
            pygame.draw.circle(screen, FG, keep_centre, keep_radius, 2)
            if bool(slot.get("kept", False)) and has_data:
                pygame.draw.circle(screen, HILITE, keep_centre, keep_radius - 4)
            keep_label = small_font.render("KEEP", True, FG)
            screen.blit(keep_label, (keep_centre[0] + 16, keep_centre[1] - keep_label.get_height() // 2))

            run_rect = pygame.Rect(row.right - 245, row.y + 18, 66, 36)
            delete_rect = pygame.Rect(row.right - 170, row.y + 18, 64, 36)
            rename_rect = pygame.Rect(row.right - 95, row.y + 18, 74, 36)
            self._draw_button(screen, run_rect, "RUN", fill=(56, 130, 82), text_size=22, active=has_data)
            self._draw_button(screen, delete_rect, "DEL", fill=RED, text_size=24)
            self._draw_button(screen, rename_rect, "EDIT", text_size=22)

            if self.edit_kind == "episode" and self.edit_slot_id == slot_id:
                edit_rect = pygame.Rect(row.x + 360, row.y + 16, 240, 40)
                pygame.draw.rect(screen, (36, 42, 47), edit_rect, border_radius=6)
                pygame.draw.rect(screen, HILITE, edit_rect, 2, border_radius=6)
                edit_text = small_font.render(self.edit_buffer or "_", True, FG)
                screen.blit(edit_text, (edit_rect.x + 8, edit_rect.y + 10))

    def _draw_player(self, screen: pygame.Surface) -> None:
        """
        Draw replay player view: controls, metrics panel, and map panel.
        """
        # Advance timeline in draw-time so playback speed remains frame-rate independent.
        now_ms = pygame.time.get_ticks()
        if self.player_last_tick_ms is None:
            self.player_last_tick_ms = now_ms
        dt = (now_ms - self.player_last_tick_ms) / 1000.0
        self.player_last_tick_ms = now_ms
        self._tick_player(dt)

        screen.fill(BG)
        pygame.draw.rect(screen, PANEL_BG, self.player_left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.player_left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.player_right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.player_right_panel, 2, border_radius=12)

        play_label = "PAUSE" if self.player_playing else "PLAY"
        self._draw_training_style_button(screen, self.player_back_button, "BACK", 22)
        self._draw_training_style_button(screen, self.player_restart_button, "RESTART", 20, border_colour=(95, 170, 200))
        self._draw_training_style_button(screen, self.player_play_pause_button, play_label, 22, border_colour=(95, 170, 200))
        self._draw_training_style_button(screen, self.player_speed_minus_button, "-", 28, border_colour=(95, 170, 200))
        self._draw_training_style_button(screen, self.player_speed_plus_button, "+", 28, border_colour=(95, 170, 200))

        pygame.draw.rect(screen, PANEL_BG, self.player_speed_label_rect, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, self.player_speed_label_rect, 2, border_radius=10)
        speed_text = self._font(22).render(f"{self.player_speed:.2f}x", True, FG)
        screen.blit(speed_text, speed_text.get_rect(center=self.player_speed_label_rect.center))

        frame_count = len(self.player_frames)
        if frame_count == 0:
            info = self._font(24).render("No replay data loaded.", True, ACCENT)
            screen.blit(info, info.get_rect(center=self.player_right_panel.center))
            return

        frame = self.player_frames[self.player_index]
        grid = self._extract_grid(frame, self.player_payload)
        node_types = self._extract_node_type_grid(frame, self.player_payload)

        if grid is None and self.player_generated_map_grid is not None:
            grid = self.player_generated_map_grid
        if node_types is None and self.player_generated_node_grid is not None:
            node_types = self.player_generated_node_grid
        vehicles = self._extract_vehicle_rows(frame, self.player_payload)

        # Fallback grid when replay payload does not include explicit map data.
        if not grid:
            width = 20
            height = 20
            grid = [[1 for _ in range(width)] for _ in range(height)]

        # Draw map/vehicles with the same rendering approach as Train preview.
        self._draw_replay_map_preview(
            screen=screen,
            rect=self.player_map_rect,
            grid=grid,
            node_types=node_types,
            vehicles=vehicles,
        )

        # Left metrics panel mirrors training-style summary metrics.
        h_font = self._font(22)
        t_font = self._font(20)
        screen.blit(h_font.render("REPLAY METRICS", True, FG), (self.player_left_panel.x + 18, self.player_left_panel.y + 72))

        metrics_lines = [
            f"Slot: {self.player_slot_id}",
            f"Step: {self.player_index + 1}/{frame_count}",
            f"State: {'PLAYING' if self.player_playing else 'PAUSED'}",
            f"Finished: {'YES' if self.player_finished else 'NO'}",
            f"Seed: {self.player_payload.get('seed', 'N/A')}",
            f"Network: {self.player_payload.get('network_name', 'N/A') or 'N/A'}",
        ]

        frame_metrics = frame.get("metrics")
        if isinstance(frame_metrics, dict):
            for key, value in frame_metrics.items():
                if len(metrics_lines) >= 14:
                    break
                metrics_lines.append(f"{str(key)}: {value}")

        y = self.player_left_panel.y + 104
        for line in metrics_lines:
            screen.blit(t_font.render(line, True, ACCENT), (self.player_left_panel.x + 14, y))
            y += 28

        if self.player_finished:
            overlay = self._font(32).render("EPISODE ENDED - PRESS RESTART", True, HILITE)
            screen.blit(overlay, overlay.get_rect(center=(self.player_map_rect.centerx, self.player_map_rect.y + 22)))

        phase_line = self._font(21).render("Replay Player", True, FG)
        screen.blit(phase_line, (self.player_left_panel.x + 16, self.player_left_panel.bottom - 60))
        runtime_line = self._font(20).render(self.status_message, True, ACCENT)
        screen.blit(runtime_line, (self.player_left_panel.x + 16, self.player_left_panel.bottom - 34))

    def draw(self, screen: pygame.Surface) -> None:
        """
        Render replay browser state.
        """
        if self.view == "player":
            self._draw_player(screen)
            return

        screen.fill(BG)
        pygame.draw.rect(screen, PANEL_BG, self.body_rect, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, self.body_rect, 2, border_radius=10)

        # Keep BACK consistent with training-style buttons across all replay views.
        self._draw_training_style_button(screen, self.back_button, "BACK", 22)

        title_font = self._font(42)
        title = title_font.render("REPLAYS", True, FG)
        screen.blit(title, title.get_rect(center=(self.title_rect.centerx, self.title_rect.centery)))

        if self.view == "chooser":
            self._draw_chooser(screen)
        elif self.view == "networks":
            self._draw_networks(screen)
        else:
            self._draw_episodes(screen)

        status_font = self._font(24)
        status = status_font.render(self.status_message, True, ACCENT)
        screen.blit(status, (self.body_rect.x + 16, self.body_rect.bottom - 34))
