"""
Mock PLC / Sensor Data Generator
=================================
Produces realistic, time-varying industrial sensor streams for seven
factory assets.  Simulates gradual degradation, spontaneous faults,
and maintenance recovery — no real hardware required.

Usage:
    from ingestion.mock_plc import MockPLCGenerator
    plc = MockPLCGenerator()
    readings = plc.generate_all()          # one snapshot
    plc.start_streaming(interval=2.0)      # background thread
"""

import math
import random
import threading
import time
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

class DegradationProfile(Enum):
    HEALTHY   = "healthy"
    DEGRADING = "degrading"
    CRITICAL  = "critical"
    FAULTED   = "faulted"


class FaultCode(Enum):
    NONE              = 0
    OVERHEAT          = 101
    HIGH_VIBRATION    = 102
    OVERCURRENT       = 103
    LOW_PRESSURE      = 104
    SPEED_DEVIATION   = 105
    ENCODER_ERROR     = 201
    COMMUNICATION     = 202
    ESTOP             = 301
    DRIVE_FAULT       = 401
    SERVO_FAULT       = 402
    PNEUMATIC_FAULT   = 501


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class MachineProfile:
    machine_id:         str
    machine_type:       str     # CNC | Robot | Conveyor | QA
    nominal_temp:       float   # °C
    nominal_vibration:  float   # mm/s RMS
    nominal_current:    float   # Amperes
    nominal_pressure:   float   # bar
    nominal_speed:      float   # RPM
    nominal_cycle_time: float   # seconds / cycle
    max_temp:           float = 90.0
    max_vibration:      float = 15.0
    failure_rate:       float = 0.0008   # prob of spontaneous fault per tick


@dataclass
class SensorReading:
    machine_id:     str
    timestamp:      float
    temperature:    float   # °C
    vibration:      float   # mm/s RMS
    current:        float   # Amperes
    pressure:       float   # bar
    speed:          float   # RPM
    cycle_time:     float   # seconds
    parts_produced: int
    fault_code:     int     # 0 = no fault
    profile:        str     # DegradationProfile.value

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Default machine fleet
# --------------------------------------------------------------------------

MACHINE_PROFILES: Dict[str, MachineProfile] = {
    "CNC-001": MachineProfile("CNC-001", "CNC",      45.0, 2.1, 18.5, 6.0, 3000, 45.0, failure_rate=0.0007),
    "CNC-002": MachineProfile("CNC-002", "CNC",      48.0, 2.3, 20.0, 6.2, 2900, 48.0, failure_rate=0.0009),
    "ROB-001": MachineProfile("ROB-001", "Robot",    38.0, 1.5, 25.0, 5.0, 1500, 12.0, failure_rate=0.0006),
    "ROB-002": MachineProfile("ROB-002", "Robot",    40.0, 1.8, 24.0, 5.1, 1450, 14.0, failure_rate=0.0006),
    "CNV-001": MachineProfile("CNV-001", "Conveyor", 35.0, 0.8,  8.0, 3.0,  500,  0.5, failure_rate=0.0003),
    "CNV-002": MachineProfile("CNV-002", "Conveyor", 36.0, 0.9,  8.5, 3.1,  480,  0.5, failure_rate=0.0003),
    "QAS-001": MachineProfile("QAS-001", "QA",       42.0, 1.2, 12.0, 4.0,  200,  8.0, failure_rate=0.0005),
}


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

class MockPLCGenerator:
    """
    Generates realistic sensor readings for each machine in the fleet.

    Internal state per machine:
      _degradation[id]  — 0.0 (fresh) → 1.0 (about to fail)
      _fault[id]        — True = active fault
      _running[id]      — True = machine is running (not paused externally)
      _parts[id]        — cumulative good-part counter
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, MachineProfile]] = None,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.profiles         = profiles or MACHINE_PROFILES
        self._degradation:    Dict[str, float] = {m: 0.0 for m in self.profiles}
        self._fault:          Dict[str, bool]  = {m: False for m in self.profiles}
        self._running:        Dict[str, bool]  = {m: True  for m in self.profiles}
        self._parts:          Dict[str, int]   = {m: 0     for m in self.profiles}
        self._fault_codes:    Dict[str, int]   = {m: 0     for m in self.profiles}
        self._callbacks:      List[Callable]   = []
        self._tick:           int              = 0
        self._lock            = threading.RLock()

        # Slight time offset so each machine degrades at a slightly different pace
        self._phase_offsets: Dict[str, float] = {
            m: random.uniform(0, 2 * math.pi) for m in self.profiles
        }

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def register_callback(self, cb: Callable[[Dict[str, SensorReading]], None]) -> None:
        """Register a function called after every generate_all()."""
        with self._lock:
            self._callbacks.append(cb)

    def set_running(self, machine_id: str, running: bool) -> None:
        """Pause or resume a machine externally (dashboard control)."""
        with self._lock:
            self._running[machine_id] = running
        logger.info(f"[MockPLC] {machine_id} running={running}")

    def inject_fault(self, machine_id: str, fault_code: Optional[int] = None) -> None:
        """Force a fault into a machine (for testing / drill)."""
        with self._lock:
            self._fault[machine_id] = True
            self._degradation[machine_id] = 1.0
            self._fault_codes[machine_id] = fault_code or random.choice(
                [c.value for c in FaultCode if c.value > 0]
            )
        logger.warning(f"[MockPLC] Fault injected → {machine_id}")

    def clear_fault(self, machine_id: str) -> None:
        """Simulate a maintenance reset."""
        with self._lock:
            self._fault[machine_id]       = False
            self._degradation[machine_id] = 0.0
            self._fault_codes[machine_id] = 0
        logger.info(f"[MockPLC] Fault cleared → {machine_id}")

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_reading(self, machine_id: str) -> SensorReading:
        """Generate one sensor snapshot for a single machine."""
        p   = self.profiles[machine_id]
        deg = self._degradation[machine_id]
        is_fault   = self._fault[machine_id]
        is_running = self._running[machine_id]

        # ---- classify degradation level ----
        if is_fault:
            deg_profile = DegradationProfile.FAULTED
            fault_code  = self._fault_codes[machine_id]
        elif deg > 0.7:
            deg_profile = DegradationProfile.CRITICAL
            fault_code  = 0
        elif deg > 0.3:
            deg_profile = DegradationProfile.DEGRADING
            fault_code  = 0
        else:
            deg_profile = DegradationProfile.HEALTHY
            fault_code  = 0

        noise = 0.04  # 4 % base noise

        if is_fault:
            # Dramatically abnormal readings
            temp       = p.nominal_temp      * (1.6 + deg * 0.5) + np.random.normal(0, 5)
            vibration  = p.nominal_vibration * (3.5 + deg * 4.0) + np.random.normal(0, 0.6)
            current    = p.nominal_current   * (1.9 + deg * 0.8) + np.random.normal(0, 2)
            pressure   = p.nominal_pressure  * max(0.05, 0.25 + np.random.normal(0, 0.1))
            speed      = p.nominal_speed     * max(0.0,  0.08 + np.random.normal(0, 0.05))
            cycle_time = p.nominal_cycle_time * (3.5 + np.random.normal(0, 0.5))

        elif not is_running:
            # Machine stopped — low-idle readings
            temp       = p.nominal_temp      * 0.65 + np.random.normal(0, 1.0)
            vibration  = p.nominal_vibration * 0.05 + abs(np.random.normal(0, 0.03))
            current    = p.nominal_current   * 0.04 + abs(np.random.normal(0, 0.15))
            pressure   = p.nominal_pressure  * 0.45 + np.random.normal(0, 0.08)
            speed      = 0.0
            cycle_time = 0.0

        else:
            # Normal / degraded production
            df         = 1.0 + deg * 0.9          # degradation factor
            phase      = self._phase_offsets[machine_id] + self._tick * 0.1
            ripple     = 1.0 + 0.02 * math.sin(phase)   # slight periodic drift

            temp       = p.nominal_temp      * df * ripple + np.random.normal(0, p.nominal_temp      * noise)
            vibration  = p.nominal_vibration * df * (1 + deg) + np.random.normal(0, p.nominal_vibration * noise)
            current    = p.nominal_current   * df + np.random.normal(0, p.nominal_current   * noise)
            pressure   = p.nominal_pressure  * (1 - deg * 0.2) + np.random.normal(0, p.nominal_pressure * noise * 0.5)
            speed      = p.nominal_speed     * (1 - deg * 0.15) + np.random.normal(0, p.nominal_speed * noise)
            cycle_time = p.nominal_cycle_time * (1 + deg * 0.6) + np.random.normal(0, p.nominal_cycle_time * noise)

        # Clamp to physical limits
        temp       = max(15.0, min(p.max_temp * 1.3, temp))
        vibration  = max(0.0,  min(p.max_vibration,  vibration))
        current    = max(0.0,  current)
        pressure   = max(0.0,  pressure)
        speed      = max(0.0,  speed)
        cycle_time = max(0.0,  cycle_time)

        if is_running and not is_fault:
            self._parts[machine_id] += 1

        return SensorReading(
            machine_id=machine_id,
            timestamp=time.time(),
            temperature=round(temp,       2),
            vibration=round(vibration,    3),
            current=round(current,        2),
            pressure=round(pressure,      3),
            speed=round(speed,            1),
            cycle_time=round(cycle_time,  2),
            parts_produced=self._parts[machine_id],
            fault_code=fault_code,
            profile=deg_profile.value,
        )

    def generate_all(self) -> Dict[str, SensorReading]:
        """Generate one snapshot for every machine; evolve degradation."""
        with self._lock:
            self._tick += 1
            self._evolve_degradation()
            readings = {mid: self.generate_reading(mid) for mid in self.profiles}

        for cb in self._callbacks:
            try:
                cb(readings)
            except Exception as exc:
                logger.error(f"[MockPLC] Callback error: {exc}")

        return readings

    def _evolve_degradation(self) -> None:
        """Randomly advance degradation state and trigger spontaneous faults."""
        for machine_id, profile in self.profiles.items():
            if self._fault[machine_id]:
                continue

            # Tiny stochastic drift in degradation
            delta = random.gauss(0.00025, 0.0001)
            self._degradation[machine_id] = max(0.0, min(0.97, self._degradation[machine_id] + delta))

            # Spontaneous fault when degradation reaches critical + dice roll
            if self._degradation[machine_id] > 0.5 and random.random() < profile.failure_rate:
                self._fault[machine_id] = True
                self._degradation[machine_id] = 1.0
                self._fault_codes[machine_id] = random.choice(
                    [c.value for c in FaultCode if c.value > 0]
                )
                logger.warning(f"[MockPLC] Spontaneous fault → {machine_id}")

            # Occasional partial self-recovery (simulates operator intervention)
            elif self._degradation[machine_id] > 0.6 and random.random() < 0.008:
                self._degradation[machine_id] *= 0.45
                logger.info(f"[MockPLC] Partial recovery → {machine_id}")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def start_streaming(self, interval: float = 2.0) -> threading.Thread:
        """
        Launch a background thread that calls generate_all() every `interval` seconds.
        Registered callbacks are invoked with each new batch of readings.
        """
        def _loop():
            while True:
                self.generate_all()
                time.sleep(interval)

        t = threading.Thread(target=_loop, daemon=True, name="MockPLC-stream")
        t.start()
        logger.info(f"[MockPLC] Streaming started  interval={interval}s")
        return t

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_degradation_states(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._degradation)

    def get_fault_states(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._fault)

    def get_parts_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._parts)
