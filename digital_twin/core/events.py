"""
Event Bus — Pub/Sub messaging between Digital Twin components.
Provides a decoupled, thread-safe event system for the factory simulation.
"""

import time
import threading
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    # Machine lifecycle
    MACHINE_STATE_CHANGE    = "machine.state_change"
    MACHINE_FAULT           = "machine.fault"
    MACHINE_FAULT_CLEARED   = "machine.fault_cleared"
    MACHINE_CYCLE_COMPLETE  = "machine.cycle_complete"
    MACHINE_METRICS_UPDATE  = "machine.metrics_update"

    # Material flow
    CONVEYOR_PART_ARRIVED   = "conveyor.part_arrived"
    CONVEYOR_PART_DEPARTED  = "conveyor.part_departed"
    CONVEYOR_JAMMED         = "conveyor.jammed"
    BUFFER_FULL             = "buffer.full"
    BUFFER_EMPTY            = "buffer.empty"

    # Robot
    ROBOT_OPERATION_COMPLETE = "robot.op_complete"
    ROBOT_FAULT              = "robot.fault"

    # Sensor / data
    SENSOR_DATA_UPDATE      = "sensor.data_update"
    ANOMALY_DETECTED        = "anomaly.detected"
    FAILURE_PREDICTED       = "failure.predicted"

    # Optimisation
    BOTTLENECK_DETECTED     = "bottleneck.detected"
    SCHEDULE_UPDATED        = "schedule.updated"

    # System
    SIMULATION_TICK         = "simulation.tick"
    PRODUCTION_ALERT        = "production.alert"
    SHIFT_START             = "shift.start"
    SHIFT_END               = "shift.end"


@dataclass
class Event:
    type:      EventType
    source:    str
    payload:   Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    severity:  str   = "INFO"          # INFO | WARNING | CRITICAL

    def __repr__(self) -> str:
        return f"<Event {self.type.value} from={self.source} at={self.timestamp:.2f}>"


class EventBus:
    """
    Thread-safe central event bus for the Digital Twin.
    Supports subscribe / publish / replay of recent history.
    """

    def __init__(self, max_history: int = 2000):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        self._wildcard_subscribers: List[Callable[[Event], None]] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._lock = threading.RLock()
        self._stats: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Subscription API
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe a callback to a specific event type."""
        with self._lock:
            self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """Subscribe a callback to ALL event types (wildcard)."""
        with self._lock:
            self._wildcard_subscribers.append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        with self._lock:
            lst = self._subscribers.get(event_type, [])
            if callback in lst:
                lst.remove(callback)

    # ------------------------------------------------------------------
    # Publish API
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Publish an event; calls all registered subscribers synchronously."""
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            self._stats[event.type.value] += 1
            handlers = list(self._subscribers.get(event.type, []))
            wildcards = list(self._wildcard_subscribers)

        for cb in handlers + wildcards:
            try:
                cb(event)
            except Exception as exc:
                logger.error(f"[EventBus] Handler error for {event}: {exc}", exc_info=True)

    def emit(
        self,
        event_type: EventType,
        source: str,
        payload: Dict[str, Any],
        severity: str = "INFO",
    ) -> None:
        """Convenience wrapper — creates and publishes an Event."""
        self.publish(Event(type=event_type, source=source, payload=payload, severity=severity))

    # ------------------------------------------------------------------
    # History / diagnostics
    # ------------------------------------------------------------------

    def get_recent(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        with self._lock:
            events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        return events[-limit:]

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._stats.clear()


# --------------------------------------------------------------------------
# Global singleton — import and use anywhere in the codebase
# --------------------------------------------------------------------------
event_bus = EventBus()
