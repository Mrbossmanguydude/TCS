"""
Baseline Demo screen.

This screen reuses the same episode simulation/runtime path as TRAIN, but
disables learning updates so users can observe how the current network runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pygame

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
from src.utils.ppo_controller import PPOUpdateStats


class BaselineDemoScreen(TrainScreen):
    """
    Use:
    Run TRAIN-like episodes for demonstration only (inference without PPO
    optimisation), so behaviour can be observed and reported.
    """

    def __init__(
        self,
        screen_rect: pygame.Rect,
        font_path: Path | None = None,
        run_ctx: Any = None,
        ui_offsets: Dict[str, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__(screen_rect=screen_rect, font_path=font_path, run_ctx=run_ctx, ui_offsets=ui_offsets)
        self.status_message = "BASELINE DEMO ready. Begin one episode to observe policy behaviour."
        self.auto_continue_training = False
        self.training_visualised = True

        # Demo must not expose network save/load or visualisation toggle controls.
        self.save_network_button = self.save_network_button.move(-2000, 0)
        self.load_network_button = self.load_network_button.move(-2000, 0)
        self.visualise_button = self.visualise_button.move(-2000, 0)

    def _run_ppo_update(self) -> PPOUpdateStats:
        """
        Use:
        Disable PPO optimisation for demo episodes while preserving episode flow.
        """
        return PPOUpdateStats(
            loss=0.0,
            actor_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            clip_fraction=0.0,
        )

    def _reset_episode_rollouts(self, vehicles) -> None:  # type: ignore[override]
        """
        Use:
        Clear rollout buffers; demo does not train from collected samples.
        """
        self._episode_rollouts = {}

    def _record_rollout_step(self, **kwargs) -> None:  # type: ignore[override]
        """
        Use:
        Ignore rollout recording in demo mode.
        """
        return

    def _advance_curriculum_if_ready(self) -> None:
        """
        Use:
        Keep Demo pinned to selected phase/level; no automatic advancement.
        """
        return

    def _set_training_visualised(self, enabled: bool) -> None:
        """
        Use:
        Keep visualisation permanently enabled in demo mode.
        """
        self.training_visualised = True

    def _draw_training_runtime(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render DURING DEMO runtime view with result-focused metrics.
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
        self._draw_button(screen, self.speed_minus_button, "-", 28, border_colour=(95, 170, 200))
        self._draw_button(screen, self.speed_plus_button, "+", 28, border_colour=(95, 170, 200))

        pygame.draw.rect(screen, PANEL_BG, self.speed_label_rect, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, self.speed_label_rect, 2, border_radius=10)
        speed_label = self._font(22).render(f"{self.sim_speed:.2f}x", True, FG)
        screen.blit(speed_label, speed_label.get_rect(center=self.speed_label_rect.center))

        detail_font = self._font(20)
        start_x, start_y = self._text_pos("training_info_start", (self.left_panel.x + 18, self.left_panel.y + 72))
        y_pos = start_y
        for line in [
            f"Run ID: {getattr(self.run_ctx, 'run_id', None)}",
            f"Vehicles: {len(self.preview_map.vehicles)}",
            f"Network ready: {'YES' if self.ppo is not None else 'NO'}",
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
        self._draw_map_preview(screen, map_rect)

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
        Render pre-demo setup view and runtime demo view.
        """
        self._tick_runtime()

        if self.training_view:
            self._draw_training_runtime(screen)
            return

        screen.fill(BG)
        title = self._font(82).render("BASELINE DEMO", True, FG)
        title_pos = self._text_pos("title", (self.screen_rect.centerx, 14))
        screen.blit(title, title.get_rect(midtop=title_pos))

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.right_panel, 2, border_radius=12)

        self._draw_button(screen, self.back_button, "BACK", 22)
        self._draw_button(screen, self.reset_button, "RESET ENV", 22)
        self._draw_button(screen, self.phase_minus_button, "-", 34)
        self._draw_button(screen, self.phase_plus_button, "+", 34)
        self._draw_button(screen, self.level_minus_button, "-", 34, border_colour=(95, 170, 200))
        self._draw_button(screen, self.level_plus_button, "+", 34, border_colour=(95, 170, 200))
        self._draw_button(screen, self.begin_button, "BEGIN DEMO", 22)

        phase_font = self._font(24)
        phase_text = phase_font.render(f"START PHASE: {self.phase}", True, FG)
        phase_pos = self._text_pos("phase_label", (self.left_panel.x + 108, self.phase_minus_button.y - 38))
        screen.blit(phase_text, phase_pos)

        level_font = self._font(24)
        level_text = level_font.render(
            f"MAP LEVEL: {self.level_index + 1}/{max(1, map_level_count(self.phase))}",
            True,
            FG,
        )
        level_pos = self._text_pos("level_label", (self.left_panel.x + 108, self.level_minus_button.y - 38))
        screen.blit(level_text, level_pos)

        detail_font = self._font(20)
        info_anchor_y = max(self.level_minus_button.bottom + 16, self.left_panel.y + 330)
        start_x, start_y = self._text_pos("info_start", (self.left_panel.x + 18, info_anchor_y))
        y_pos = start_y
        for line in [
            f"Run ID: {getattr(self.run_ctx, 'run_id', None)}",
            f"Vehicles: {len(self.preview_map.vehicles)}",
            f"Network ready: {'YES' if self.ppo is not None else 'NO'}",
        ]:
            txt = detail_font.render(line, True, FG)
            screen.blit(txt, (start_x, y_pos))
            y_pos += 30

        if self.episode_state is not None:
            for line in [
                f"Episode index: {self.episode_state.episode_index}",
                f"Step count: {self.episode_state.step_count}",
                f"Success rate: {float(self.episode_state.metrics.get('success_rate', 0.0)):.2f}",
                f"Collisions: {float(self.episode_state.metrics.get('collisions', 0.0)):.0f}",
                f"Throughput: {float(self.episode_state.metrics.get('throughput', 0.0)):.3f}",
            ]:
                txt = detail_font.render(line, True, ACCENT)
                screen.blit(txt, (start_x, y_pos))
                y_pos += 26

        map_rect = self.right_panel.inflate(-26, -84)
        map_rect.y += 28
        self._draw_map_preview(screen, map_rect)

        status = self._font(20).render(self.status_message, True, ACCENT)
        screen.blit(status, self._text_pos("status", (self.left_panel.x + 16, self.left_panel.bottom - 34)))

