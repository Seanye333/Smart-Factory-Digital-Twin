"""
Machine State Machine — models the discrete lifecycle of factory equipment.

States:
  IDLE        → machine is powered, no work queued
  RUNNING     → actively processing a part
  FAULT       → unplanned stoppage (requires maintenance/reset)
  MAINTENANCE → planned downtime (preventive or corrective)
  SETUP       → changeover / tool-change
  STARVED     → running but no upstream parts available
  BLOCKED     → finished part, downstream buffer full
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# State & colour definitions
# --------------------------------------------------------------------------

class MachineState(Enum):
    IDLE        = "IDLE"
    RUNNING     = "RUNNING"
    FAULT       = "FAULT"
    MAINTENANCE = "MAINTENANCE"
    SETUP       = "SETUP"
    STARVED     = "STARVED"
    BLOCKED     = "BLOCKED"


STATE_COLORS: Dict[MachineState, str] = {
    MachineState.IDLE:        "#3498db",   # Blue
    MachineState.RUNNING:     "#2ecc71",   # Green
    MachineState.FAULT:       "#e74c3c",   # Red
    MachineState.MAINTENANCE: "#f39c12",   # Orange
    MachineState.SETUP:       "#9b59b6",   # Purple
    MachineState.STARVED:     "#e67e22",   # Dark Orange
    MachineState.BLOCKED:     "#95a5a6",   # Grey
}

STATE_LABELS: Dict[MachineState, str] = {
    MachineState.IDLE:        "Idle",
    MachineState.RUNNING:     "Running",
    MachineState.FAULT:       "Fault",
    MachineState.MAINTENANCE: "Maintenance",
    MachineState.SETUP:       "Setup",
    MachineState.STARVED:     "Starved",
    MachineState.BLOCKED:     "Blocked",
}

# Legal transitions: {from_state: [allowed_to_states]}
VALID_TRANSITIONS: Dict[MachineState, List[MachineState]] = {
    MachineState.IDLE:        [MachineState.RUNNING, MachineState.MAINTENANCE, MachineState.SETUP, MachineState.STARVED],
    MachineState.RUNNING:     [MachineState.IDLE, MachineState.FAULT, MachineState.BLOCKED, MachineState.STARVED],
    MachineState.FAULT:       [MachineState.MAINTENANCE, MachineState.IDLE],
    MachineState.MAINTENANCE: [MachineState.IDLE, MachineState.SETUP],
    MachineState.SETUP:       [MachineState.RUNNING, MachineState.IDLE],
    MachineState.STARVED:     [MachineState.RUNNING, MachineState.IDLE, MachineState.FAULT],
    MachineState.BLOCKED:     [MachineState.RUNNING, MachineState.IDLE, MachineState.FAULT],
}


# --------------------------------------------------------------------------
# Duration tracking
# --------------------------------------------------------------------------

@dataclass
class StateDuration:
    state:      MachineState
    entered_at: float
    reason:     str = ""
    exited_at:  Optional[float] = None

    @property
    def duration(self) -> float:
        end = self.exited_at if self.exited_at is not None else time.time()
        return max(0.0, end - self.entered_at)

    def close(self) -> None:
        self.exited_at = time.time()


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------

class MachineStateMachine:
    """
    Tracks the current state of a machine, enforces valid transitions,
    accumulates duration history, and fires per-state callbacks.
    """

    def __init__(self, machine_id: str, initial_state: MachineState = MachineState.IDLE):
        self.machine_id = machine_id
        self._state = initial_state
        self._history: List[StateDuration] = []
        self._current = StateDuration(state=initial_state, entered_at=time.time())
        self._callbacks: Dict[str, List[Callable]] = {}
        self._global_callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state(self) -> MachineState:
        return self._state

    @property
    def color(self) -> str:
        return STATE_COLORS[self._state]

    @property
    def label(self) -> str:
        return STATE_LABELS[self._state]

    @property
    def time_in_current_state(self) -> float:
        return self._current.duration

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def can_transition(self, new_state: MachineState) -> bool:
        return new_state in VALID_TRANSITIONS.get(self._state, [])

    def transition(self, new_state: MachineState, reason: str = "") -> bool:
        if not self.can_transition(new_state):
            logger.debug(
                f"[{self.machine_id}] Blocked transition: {self._state.value} → {new_state.value}"
            )
            return False

        old_state = self._state
        self._current.close()
        self._history.append(self._current)

        self._state = new_state
        self._current = StateDuration(state=new_state, entered_at=time.time(), reason=reason)

        logger.info(
            f"[{self.machine_id}] {old_state.value} → {new_state.value}"
            + (f" ({reason})" if reason else "")
        )

        # Fire per-state callbacks
        for cb in self._callbacks.get(new_state.value, []):
            try:
                cb(old_state, new_state, reason)
            except Exception as exc:
                logger.error(f"State callback error: {exc}")

        # Fire global callbacks
        for cb in self._global_callbacks:
            try:
                cb(old_state, new_state, reason)
            except Exception as exc:
                logger.error(f"Global state callback error: {exc}")

        return True

    def force_state(self, state: MachineState, reason: str = "forced") -> None:
        """Override any state — used for simulation resets or external commands."""
        self._current.close()
        self._history.append(self._current)
        self._state = state
        self._current = StateDuration(state=state, entered_at=time.time(), reason=reason)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_state(self, state: MachineState, callback: Callable) -> None:
        self._callbacks.setdefault(state.value, []).append(callback)

    def on_any_transition(self, callback: Callable) -> None:
        self._global_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_uptime_pct(self, window_s: float = 3600.0) -> float:
        """Percentage of the last `window_s` seconds that the machine was RUNNING."""
        now = time.time()
        cutoff = now - window_s
        running = 0.0

        for rec in self._history:
            if rec.exited_at and rec.exited_at < cutoff:
                continue
            if rec.state == MachineState.RUNNING:
                start = max(rec.entered_at, cutoff)
                end = rec.exited_at or now
                running += end - start

        if self._state == MachineState.RUNNING:
            running += now - max(self._current.entered_at, cutoff)

        return min(100.0, running / window_s * 100)

    def get_state_distribution(self, window_s: float = 3600.0) -> Dict[str, float]:
        """Returns percentage of time spent in each state over last `window_s` seconds."""
        now = time.time()
        cutoff = now - window_s
        totals: Dict[str, float] = {s.value: 0.0 for s in MachineState}

        for rec in self._history:
            if rec.exited_at and rec.exited_at < cutoff:
                continue
            start = max(rec.entered_at, cutoff)
            end   = rec.exited_at or now
            totals[rec.state.value] += max(0.0, end - start)

        # Current state
        totals[self._state.value] += now - max(self._current.entered_at, cutoff)

        total_time = sum(totals.values())
        if total_time == 0:
            return totals

        return {k: round(v / total_time * 100, 2) for k, v in totals.items()}

    def get_fault_count(self) -> int:
        return sum(1 for r in self._history if r.state == MachineState.FAULT)

    def summary(self) -> Dict:
        dist = self.get_state_distribution()
        return {
            "machine_id":         self.machine_id,
            "current_state":      self._state.value,
            "color":              self.color,
            "uptime_pct":         round(self.get_uptime_pct(), 2),
            "fault_count":        self.get_fault_count(),
            "time_in_state_s":    round(self.time_in_current_state, 1),
            "state_distribution": dist,
        }
