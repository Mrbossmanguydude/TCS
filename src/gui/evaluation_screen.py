"""
Evaluate screen frontend.

This screen mirrors TRAIN/DEMO visual structure but runs inference-only episodes.
No PPO optimisation is performed during evaluation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pygame

from src.gui.baseline_demo_screen import BaselineDemoScreen
from src.gui.train_screen import (
    ACCENT,
    BG,
    FG,
    MAP_EMPTY_COLOUR,
    PANEL_BG,
    PANEL_BORDER,
    TrainScreen,
    map_level_count,
)
from src.utils.hold_repeat import HoldRepeatController
from src.utils.train_types import EpisodeSummary


class EvaluateScreen(BaselineDemoScreen):
    """
    Use:
    Frontend-focused Evaluate mode that:
    - requires a loaded controller before running,
    - allows seed control (type/regenerate),
    - runs a configurable batch of inference-only episodes,
    - shows a summary table aggregated across completed evaluation episodes.
    """

    def __init__(
        self,
        screen_rect: pygame.Rect,
        font_path: Path | None = None,
        run_ctx: Any = None,
        ui_offsets: Dict[str, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__(screen_rect=screen_rect, font_path=font_path, run_ctx=run_ctx, ui_offsets=ui_offsets)

        # Evaluate mode is inference-only, but visualisation can be toggled like TRAIN.
        self.status_message = "EVALUATE ready. Load a network to begin."
        self.auto_continue_training = False
        self.training_visualised = bool(getattr(self, "training_visualised", True))

        # Keep network save hidden in Evaluate; load remains available.
        self.save_network_button = self.save_network_button.move(-2000, 0)
        self._build_evaluate_frontend_layout()

        # Episode-count controls for batch evaluation.
        self.eval_target_episodes = 3
        self._eval_target_min = 1
        self._eval_target_max = 100
        self._eval_hold = HoldRepeatController()

        # Seed text-input state.
        self.seed_input_rect = self._offset_rect("seed_input_rect", self.seed_input_rect)
        self.seed_input_active = False
        self.seed_input_buffer = str(int(self.seed))

        # Batch progress and historical summary state.
        self._evaluation_batch_active = False
        self._evaluation_batch_completed = 0
        self._evaluation_history: List[EpisodeSummary] = []
        self._show_summary_table = False

    def _build_evaluate_frontend_layout(self) -> None:
        """
        Use:
        Reposition pre-evaluation controls into one large replay-sized panel
        with evenly distributed rows to prevent overlap.
        """
        shift_x, shift_y = self._offset("components_shift")
        panel_margin = int(self.ui_offsets.get("eval_panel_margin", 34))
        panel_top = int(self.ui_offsets.get("eval_panel_top", 98))
        panel_bottom_margin = int(self.ui_offsets.get("eval_panel_bottom_margin", 34))

        self.setup_back_button = self._offset_rect(
            "setup_back_button",
            pygame.Rect(
                self.back_button.x + shift_x,
                self.back_button.y + shift_y,
                self.back_button.width,
                self.back_button.height,
            ),
        )

        # Single large setup panel (same size style as Replay body panel).
        self.eval_panel = self._offset_rect(
            "eval_panel",
            pygame.Rect(
                panel_margin + shift_x,
                panel_top + shift_y,
                self.screen_rect.width - (panel_margin * 2),
                self.screen_rect.height - panel_top - panel_bottom_margin,
            ),
        )

        inner_pad = int(self.ui_offsets.get("eval_inner_pad", 28))
        column_gap = int(self.ui_offsets.get("eval_column_gap", 28))
        row_height = int(self.ui_offsets.get("eval_row_height", 46))
        row_gap = int(self.ui_offsets.get("eval_row_gap", 28))
        col_w = int((self.eval_panel.width - (inner_pad * 2) - column_gap) / 2)
        left_x = self.eval_panel.x + inner_pad
        right_x = left_x + col_w + column_gap
        row1_y = self.eval_panel.y + int(self.ui_offsets.get("eval_row1_y", 92))
        row2_y = row1_y + row_height + row_gap
        row3_y = row2_y + row_height + row_gap
        row4_y = row3_y + row_height + row_gap
        row5_y = row4_y + row_height + row_gap

        # Top control rows.
        self.load_network_button = self._offset_rect("load_network_button", pygame.Rect(left_x, row1_y, col_w, row_height))
        self.reset_button = self._offset_rect("reset_button", pygame.Rect(right_x, row1_y, col_w, row_height))
        self.seed_input_rect = self._offset_rect("seed_input_rect", pygame.Rect(left_x, row2_y, col_w, row_height))
        self.visualise_button = self._offset_rect("visualise_button", pygame.Rect(right_x, row2_y, col_w, row_height))

        # Phase controls row (left label, right label).
        pm_w = int(self.ui_offsets.get("eval_pm_w", 44))
        pm_h = int(self.ui_offsets.get("eval_pm_h", 44))
        phase_block_x = left_x
        level_block_x = right_x
        self.phase_minus_button = self._offset_rect("phase_minus_button", pygame.Rect(phase_block_x, row3_y + 1, pm_w, pm_h))
        self.phase_plus_button = self._offset_rect("phase_plus_button", pygame.Rect(phase_block_x + col_w - pm_w, row3_y + 1, pm_w, pm_h))
        self.level_minus_button = self._offset_rect("level_minus_button", pygame.Rect(level_block_x, row3_y + 1, pm_w, pm_h))
        self.level_plus_button = self._offset_rect("level_plus_button", pygame.Rect(level_block_x + col_w - pm_w, row3_y + 1, pm_w, pm_h))

        # Episode count + summary row.
        self.eval_minus_button = self._offset_rect("eval_minus_button", pygame.Rect(left_x, row4_y + 1, pm_w, pm_h))
        self.eval_plus_button = self._offset_rect("eval_plus_button", pygame.Rect(left_x + col_w - pm_w, row4_y + 1, pm_w, pm_h))
        self.summary_button = self._offset_rect("summary_button", pygame.Rect(right_x, row4_y, col_w, row_height))

        # Main start action centred at the bottom section.
        begin_w = min(int(self.ui_offsets.get("eval_begin_max_w", 420)), self.eval_panel.width - int(self.ui_offsets.get("eval_begin_side_pad", 56)))
        begin_h_extra = int(self.ui_offsets.get("eval_begin_h_extra", 6))
        self.begin_button = self._offset_rect(
            "begin_button",
            pygame.Rect(
                self.eval_panel.centerx - begin_w // 2,
                row5_y + int(self.ui_offsets.get("eval_begin_y_extra", 8)),
                begin_w,
                row_height + begin_h_extra,
            ),
        )

    def _controller_is_loaded(self) -> bool:
        """
        Use:
        Determine if Evaluate can run (explicit network selected/loaded).
        """
        loaded_name = str(getattr(self, "loaded_network_name", "")).strip()
        return bool(self.ppo is not None and loaded_name and loaded_name.lower() != "current policy")

    def _set_training_visualised(self, enabled: bool) -> None:
        """
        Use:
        Apply TRAIN-equivalent visualisation toggle behaviour in Evaluate.
        """
        TrainScreen._set_training_visualised(self, bool(enabled))

    def _apply_seed_value(self, seed_value: int) -> None:
        """
        Use:
        Apply a new evaluation seed and rebuild preview/runtime environment.
        """
        self.seed = int(max(0, min(2**31 - 1, int(seed_value))))
        self.seed_input_buffer = str(self.seed)
        if self.run_ctx is not None:
            self.run_ctx.seed = int(self.seed)
            self.run_ctx.config["seed"] = int(self.seed)
        self.reset_environment(initial=False)
        self.status_message = f"Evaluation seed set to {self.seed}."

    def _regenerate_seed(self) -> None:
        """
        Use:
        Generate a fresh random seed for Evaluate setup.
        """
        self._apply_seed_value(random.randint(0, 2**31 - 1))

    def _commit_seed_input(self) -> None:
        """
        Use:
        Validate and apply typed seed text.
        """
        token = self.seed_input_buffer.strip()
        if not token:
            self.status_message = "Seed input is empty."
            self.seed_input_active = False
            self.seed_input_buffer = str(int(self.seed))
            return
        try:
            value = int(token)
        except ValueError:
            self.status_message = "Seed must be an integer."
            self.seed_input_active = False
            self.seed_input_buffer = str(int(self.seed))
            return
        self.seed_input_active = False
        self._apply_seed_value(value)

    def _apply_eval_target_delta(self, direction: int) -> None:
        """
        Use:
        Increment/decrement evaluation episode batch target.
        """
        next_value = int(self.eval_target_episodes) + int(direction)
        self.eval_target_episodes = max(self._eval_target_min, min(self._eval_target_max, next_value))

    def _toggle_summary_table(self) -> None:
        """
        Use:
        Toggle summary table visibility when historical data exists.
        """
        if not self._evaluation_history:
            self.status_message = "No evaluation runs completed yet."
            return
        self._show_summary_table = not self._show_summary_table
        if self._show_summary_table:
            self.status_message = "Summary table opened."
        else:
            self.status_message = "Summary table closed."

    def _begin_evaluation_batch(self) -> None:
        """
        Use:
        Start batch evaluation using current seed/phase/level settings.
        """
        if not self._controller_is_loaded():
            self.status_message = "No network chosen. Load a controller first."
            return
        self._show_summary_table = False
        self._evaluation_batch_active = True
        self._evaluation_batch_completed = 0
        self.auto_continue_training = bool(self.eval_target_episodes > 1)
        self._begin_training_session()
        self.status_message = f"Evaluation started: {self.eval_target_episodes} episode(s)."

    def _finalise_episode(self) -> None:
        """
        Use:
        Extend baseline finalisation by tracking Evaluate history and batch progress.
        """
        super()._finalise_episode()

        if self.last_summary is not None:
            self._evaluation_history.append(
                EpisodeSummary(
                    passed=bool(self.last_summary.passed),
                    success_rate=float(self.last_summary.success_rate),
                    collision_rate=float(self.last_summary.collision_rate),
                    throughput=float(self.last_summary.throughput),
                    avg_journey_time=float(self.last_summary.avg_journey_time),
                    reward=float(self.last_summary.reward),
                    loss=float(self.last_summary.loss),
                )
            )

        if not self._evaluation_batch_active:
            return

        self._evaluation_batch_completed += 1
        if self._evaluation_batch_completed >= int(self.eval_target_episodes):
            self._evaluation_batch_active = False
            self.auto_continue_training = False
            self._next_auto_start_ms = 0
            self.status_message = (
                f"Evaluation complete: {self._evaluation_batch_completed}/{self.eval_target_episodes} episodes."
            )
        else:
            self.auto_continue_training = True

    def _exit_training_session(self) -> None:
        """
        Use:
        Exit Evaluate runtime view and stop any active batch continuation.
        """
        super()._exit_training_session()
        self._evaluation_batch_active = False
        self.auto_continue_training = False
        self._next_auto_start_ms = 0

    def load_network_from_path(self, checkpoint_path: Path, slot_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Use:
        Load controller and keep Evaluate status text clear.
        """
        loaded = super().load_network_from_path(checkpoint_path, slot_metadata)
        if loaded:
            self.status_message = f"Controller loaded for evaluation -> {self.loaded_network_name}"
        return loaded

    def handle_events(self, events: List[pygame.event.Event]) -> str:
        """
        Use:
        Process Evaluate interactions in setup/runtime views.
        """
        if self.training_view:
            next_state = super().handle_events(events)
            if next_state == "TRAIN":
                return "EVALUATE"
            return next_state

        self._phase_hold.update()
        self._level_hold.update()
        self._eval_hold.update()

        for event in events:
            if event.type == pygame.KEYDOWN and self.seed_input_active:
                if event.key == pygame.K_RETURN:
                    self._commit_seed_input()
                elif event.key == pygame.K_ESCAPE:
                    self.seed_input_active = False
                    self.seed_input_buffer = str(int(self.seed))
                elif event.key == pygame.K_BACKSPACE:
                    self.seed_input_buffer = self.seed_input_buffer[:-1]
                else:
                    text = event.unicode or ""
                    if text.isdigit() and len(self.seed_input_buffer) < 10:
                        self.seed_input_buffer += text
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.phase_minus_button.collidepoint(event.pos):
                    self._phase_hold.begin(lambda: self._apply_phase_delta(-1))
                    continue
                if self.phase_plus_button.collidepoint(event.pos):
                    self._phase_hold.begin(lambda: self._apply_phase_delta(+1))
                    continue
                if self.level_minus_button.collidepoint(event.pos):
                    self._level_hold.begin(lambda: self._apply_level_delta(-1))
                    continue
                if self.level_plus_button.collidepoint(event.pos):
                    self._level_hold.begin(lambda: self._apply_level_delta(+1))
                    continue
                if self.eval_minus_button.collidepoint(event.pos):
                    self._eval_hold.begin(lambda: self._apply_eval_target_delta(-1))
                    continue
                if self.eval_plus_button.collidepoint(event.pos):
                    self._eval_hold.begin(lambda: self._apply_eval_target_delta(+1))
                    continue

            if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
                continue

            self._phase_hold.stop()
            self._level_hold.stop()
            self._eval_hold.stop()

            if self.setup_back_button.collidepoint(event.pos):
                self.seed_input_active = False
                self._show_summary_table = False
                return "MENU"

            if self.seed_input_rect.collidepoint(event.pos):
                self.seed_input_active = True
                self.seed_input_buffer = str(int(self.seed))
                continue

            if self.seed_input_active and not self.seed_input_rect.collidepoint(event.pos):
                self._commit_seed_input()

            if self.load_network_button.collidepoint(event.pos):
                self.status_message = "Open Replays > Networks to choose controller for Evaluate."
                return "REPLAYS_NETWORK_LOAD"
            if self.reset_button.collidepoint(event.pos):
                self._regenerate_seed()
                continue
            if self.begin_button.collidepoint(event.pos):
                self._begin_evaluation_batch()
                continue
            if self.visualise_button.collidepoint(event.pos):
                self._set_training_visualised(not self.training_visualised)
                mode = "ON" if self.training_visualised else "OFF (full throttle)"
                self.status_message = f"Evaluation visualisation set to {mode}."
                continue
            if self.summary_button.collidepoint(event.pos):
                self._toggle_summary_table()
                continue

        return "EVALUATE"

    def _draw_summary_overlay(self, screen: pygame.Surface) -> None:
        """
        Use:
        Draw a summary table averaged over all completed evaluation episodes.
        """
        if not self._show_summary_table or not self._evaluation_history:
            return

        n = float(len(self._evaluation_history))
        avg_success = sum(item.success_rate for item in self._evaluation_history) / n
        avg_collision = sum(item.collision_rate for item in self._evaluation_history) / n
        avg_throughput = sum(item.throughput for item in self._evaluation_history) / n
        avg_journey = sum(item.avg_journey_time for item in self._evaluation_history) / n
        avg_reward = sum(item.reward for item in self._evaluation_history) / n

        panel = pygame.Rect(self.screen_rect.centerx - 320, self.screen_rect.centery - 180, 640, 360)
        dim = pygame.Surface((self.screen_rect.width, self.screen_rect.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        screen.blit(dim, (0, 0))

        pygame.draw.rect(screen, PANEL_BG, panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, panel, 2, border_radius=12)

        title = self._font(34).render("EVALUATION SUMMARY TABLE", True, FG)
        screen.blit(title, title.get_rect(center=(panel.centerx, panel.y + 34)))

        lines = [
            f"Episodes: {int(n)}",
            f"Average success rate: {avg_success:.3f}",
            f"Average collision rate: {avg_collision:.4f}",
            f"Average throughput: {avg_throughput:.3f}",
            f"Average journey time: {avg_journey:.3f}",
            f"Average reward: {avg_reward:.3f}",
            "Click SUMMARY TABLE again to close.",
        ]
        y_pos = panel.y + 86
        for line in lines:
            colour = FG if "close" not in line.lower() else ACCENT
            rendered = self._font(24).render(line, True, colour)
            screen.blit(rendered, (panel.x + 26, y_pos))
            y_pos += 40

    def _draw_training_runtime(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render DURING EVALUATION runtime view with baseline-style metrics.
        """
        screen.fill(BG)

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.right_panel, 2, border_radius=12)

        self._draw_button(screen, self.training_back_button, "BACK", 22)
        self._draw_button(screen, self.screenshot_button, "SCREENSHOT", 18, border_colour=(95, 170, 200))
        self._draw_button(
            screen,
            self.replay_button,
            f"SAVE REPLAY: {'ON' if self._pending_replay_save else 'OFF'}",
            16,
            border_colour=(110, 195, 130) if self._pending_replay_save else PANEL_BORDER,
        )
        self._draw_button(
            screen,
            self.play_pause_button,
            "PLAY" if self.training_paused else "PAUSE",
            22,
            border_colour=(80, 180, 110) if self.training_paused else (220, 140, 92),
        )
        speed_border = (95, 170, 200) if self.training_visualised else PANEL_BORDER
        self._draw_button(screen, self.speed_minus_button, "-", 28, border_colour=speed_border)
        self._draw_button(screen, self.speed_plus_button, "+", 28, border_colour=speed_border)

        pygame.draw.rect(screen, PANEL_BG, self.speed_label_rect, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, self.speed_label_rect, 2, border_radius=10)
        speed_text = f"{self.sim_speed:.2f}x" if self.training_visualised else "MAX"
        speed_label = self._font(22).render(speed_text, True, FG)
        screen.blit(speed_label, speed_label.get_rect(center=self.speed_label_rect.center))

        detail_font = self._font(20)
        start_x, start_y = self._text_pos("training_info_start", (self.left_panel.x + 18, self.left_panel.y + 72))
        y_pos = start_y
        for line in [
            f"Run ID: {getattr(self.run_ctx, 'run_id', None)}",
            f"Controller: {self.loaded_network_name}",
            f"Vehicles: {len(self.preview_map.vehicles)}",
            f"Batch progress: {self._evaluation_batch_completed}/{self.eval_target_episodes}",
        ]:
            txt = detail_font.render(line, True, FG)
            screen.blit(txt, (start_x, y_pos))
            y_pos += 30

        if self.episode_state is not None:
            throughput = float(self.episode_state.metrics.get("throughput", 0.0))
            success_rate = float(self.episode_state.metrics.get("success_rate", 0.0))
            collisions = float(self.episode_state.metrics.get("collisions", 0.0))
            avg_speed = float(self.episode_state.metrics.get("avg_speed", 0.0))
            reward_sum = float(self.episode_state.metrics.get("reward_sum", 0.0))
            avg_journey_time = 0.0 if self.last_summary is None else float(self.last_summary.avg_journey_time)
            last_pass = "N/A" if self.last_summary is None else ("YES" if self.last_summary.passed else "NO")

            for line in [
                f"Episode index: {self.episode_state.episode_index}",
                f"Step count: {self.episode_state.step_count}",
                f"Success rate: {success_rate:.2f}",
                f"Collisions: {collisions:.0f}",
                f"Throughput: {throughput:.3f}",
                f"Average journey time: {avg_journey_time:.2f}",
                f"Average speed: {avg_speed:.3f}",
                f"Reward sum: {reward_sum:.3f}",
                f"Last episode pass: {last_pass}",
            ]:
                txt = detail_font.render(line, True, ACCENT)
                screen.blit(txt, (start_x, y_pos))
                y_pos += 26

        map_rect = self.right_panel.inflate(-26, -84)
        map_rect.y += 28
        if self.training_visualised:
            self._draw_map_preview(screen, map_rect)
        else:
            pygame.draw.rect(screen, MAP_EMPTY_COLOUR, map_rect, border_radius=8)
            pygame.draw.rect(screen, PANEL_BORDER, map_rect, 1, border_radius=8)
            line_1 = self._font(26).render("VISUALISATION OFF", True, FG)
            line_2 = self._font(21).render("Headless evaluation running at maximum throughput.", True, ACCENT)
            screen.blit(line_1, line_1.get_rect(center=(map_rect.centerx, map_rect.centery - 16)))
            screen.blit(line_2, line_2.get_rect(center=(map_rect.centerx, map_rect.centery + 18)))

        phase_line = self._font(21).render(
            f"Phase {self.phase} | Level {self.level_index + 1}/{max(1, map_level_count(self.phase))}",
            True,
            FG,
        )
        phase_pos = self._text_pos("training_status", (self.left_panel.x + 16, self.left_panel.bottom - 60))
        screen.blit(phase_line, phase_pos)

        runtime_line = self._font(20).render(self.status_message, True, ACCENT)
        runtime_pos = self._text_pos("training_detail_status", (self.left_panel.x + 16, self.left_panel.bottom - 34))
        screen.blit(runtime_line, runtime_pos)

        if self._pending_screenshot:
            self._save_training_screenshot(screen)
            self._pending_screenshot = False

    def draw(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render Evaluate pre-runtime setup and runtime views.
        """
        self._tick_runtime()

        if self.training_view:
            self._draw_training_runtime(screen)
            return

        screen.fill(BG)
        title = self._font(96).render("EVALUATE", True, FG)
        title_pos = self._text_pos("title", (self.screen_rect.centerx, 8))
        screen.blit(title, title.get_rect(midtop=title_pos))

        # Single setup panel (no map preview in Evaluate setup screen).
        pygame.draw.rect(screen, PANEL_BG, self.eval_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.eval_panel, 2, border_radius=12)

        self._draw_button(screen, self.setup_back_button, "BACK", 22)
        self._draw_button(screen, self.load_network_button, "LOAD CONTROLLER", 20, border_colour=(100, 155, 220))
        self._draw_button(screen, self.reset_button, "REGENERATE SEED", 20, border_colour=(95, 170, 200))
        self._draw_button(screen, self.begin_button, "BEGIN EVALUATION", 20)
        self._draw_button(
            screen,
            self.visualise_button,
            f"VISUALISE EVALUATION: {'ON' if self.training_visualised else 'OFF'}",
            17,
            border_colour=(95, 170, 200) if self.training_visualised else PANEL_BORDER,
        )
        self._draw_button(screen, self.phase_minus_button, "-", 30)
        self._draw_button(screen, self.phase_plus_button, "+", 30)
        self._draw_button(screen, self.level_minus_button, "-", 30, border_colour=(95, 170, 200))
        self._draw_button(screen, self.level_plus_button, "+", 30, border_colour=(95, 170, 200))
        self._draw_button(screen, self.eval_minus_button, "-", 30, border_colour=(110, 170, 115))
        self._draw_button(screen, self.eval_plus_button, "+", 30, border_colour=(110, 170, 115))
        self._draw_button(screen, self.summary_button, "SUMMARY TABLE", 20, border_colour=(110, 170, 115))

        # Seed text input box.
        pygame.draw.rect(screen, (36, 42, 47), self.seed_input_rect, border_radius=8)
        seed_border = (95, 170, 200) if self.seed_input_active else PANEL_BORDER
        pygame.draw.rect(screen, seed_border, self.seed_input_rect, 2, border_radius=8)
        seed_text = self.seed_input_buffer if self.seed_input_active else str(int(self.seed))
        seed_label = self._font(24).render(seed_text or "0", True, FG)
        screen.blit(seed_label, (self.seed_input_rect.x + 10, self.seed_input_rect.y + 9))

        label_font = self._font(22)

        screen.blit(
            label_font.render(f"Controller: {self.loaded_network_name}", True, FG),
            self._text_pos("controller_label", (self.load_network_button.x + 2, self.load_network_button.y - 30)),
        )
        screen.blit(
            label_font.render("Seed (type to set):", True, FG),
            self._text_pos("seed_label", (self.seed_input_rect.x + 2, self.seed_input_rect.y - 30)),
        )

        phase_label_cx = (self.phase_minus_button.centerx + self.phase_plus_button.centerx) // 2
        level_label_cx = (self.level_minus_button.centerx + self.level_plus_button.centerx) // 2
        episodes_label_cx = (self.eval_minus_button.centerx + self.eval_plus_button.centerx) // 2

        screen.blit(
            label_font.render(f"START PHASE: {self.phase}", True, FG),
            self._text_pos("phase_label", (phase_label_cx - 80, self.phase_minus_button.y - 36)),
        )
        screen.blit(
            label_font.render(
                f"MAP LEVEL: {self.level_index + 1}/{max(1, map_level_count(self.phase))}",
                True,
                FG,
            ),
            self._text_pos("level_label", (level_label_cx - 92, self.level_minus_button.y - 36)),
        )
        screen.blit(
            label_font.render(f"Episodes: {self.eval_target_episodes}", True, FG),
            self._text_pos("episodes_label", (episodes_label_cx - 62, self.eval_minus_button.y - 36)),
        )

        status = self._font(20).render(self.status_message, True, ACCENT)
        screen.blit(status, self._text_pos("status", (self.eval_panel.x + 18, self.eval_panel.bottom - 34)))

        self._draw_summary_overlay(screen)
