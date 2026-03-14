"""
Reusable hold-to-repeat helper for plus/minus UI controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame


@dataclass(frozen=True)
class HoldRepeatConfig:
    """
    Use:
    Store timing thresholds for accelerating hold-repeat behaviour.

    Attributes:
    - first_threshold_ms: Time before switching from slow to medium repeat.
    - second_threshold_ms: Time before switching from medium to fast repeat.
    - slow_interval_ms: Repeat interval during early hold.
    - medium_interval_ms: Repeat interval during middle hold.
    - fast_interval_ms: Repeat interval during long hold.
    """

    first_threshold_ms: int = 700
    second_threshold_ms: int = 1800
    slow_interval_ms: int = 250
    medium_interval_ms: int = 120
    fast_interval_ms: int = 60


class HoldRepeatController:
    """
    Use:
    Drive one active hold-to-repeat action for UI plus/minus controls.

    This class is intentionally generic: any callable can be repeated while
    the left mouse button is held.

    Attributes:
    - config: Timing profile for repeat acceleration.
    - _active_action: Current action callback, or None when idle.
    - _hold_start_ms: Tick when current hold started.
    - _last_tick_ms: Tick when action was last repeated.
    """

    def __init__(self, config: Optional[HoldRepeatConfig] = None) -> None:
        """
        Use:
        Construct hold-repeat state with optional custom timing profile.

        Inputs:
        - config: Optional hold-repeat timing config.

        Output:
        None.
        """
        self.config = config or HoldRepeatConfig()
        self._active_action: Optional[Callable[[], None]] = None
        self._hold_start_ms: int = 0
        self._last_tick_ms: int = 0

    def begin(self, action: Callable[[], None], trigger_immediately: bool = True) -> None:
        """
        Use:
        Start repeating one action while mouse is held.

        Inputs:
        - action: Callback to execute on repeat ticks.
        - trigger_immediately: Run action once immediately when starting.

        Output:
        None.
        """
        now_ms = pygame.time.get_ticks()
        self._active_action = action
        self._hold_start_ms = now_ms
        self._last_tick_ms = now_ms
        if trigger_immediately:
            action()

    def stop(self) -> None:
        """
        Use:
        Stop any active hold-repeat action.

        Inputs:
        - None.

        Output:
        None.
        """
        self._active_action = None

    def update(self) -> None:
        """
        Use:
        Run repeated action tick when hold timing interval is reached.

        Inputs:
        - None.

        Output:
        None.
        """
        if self._active_action is None:
            return

        # Safety: stop repeating if left mouse is no longer physically held.
        if not pygame.mouse.get_pressed()[0]:
            self.stop()
            return

        now_ms = pygame.time.get_ticks()
        elapsed = now_ms - self._hold_start_ms
        if elapsed < self.config.first_threshold_ms:
            interval = self.config.slow_interval_ms
        elif elapsed < self.config.second_threshold_ms:
            interval = self.config.medium_interval_ms
        else:
            interval = self.config.fast_interval_ms

        if now_ms - self._last_tick_ms < interval:
            return

        self._active_action()
        self._last_tick_ms = now_ms


