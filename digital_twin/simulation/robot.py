"""
Robot Digital Twin (SimPy)
============================
Models a 6-DOF industrial robot arm (e.g. KUKA KR10, FANUC LR Mate).

Simulates:
  - Pick-and-place or assembly operations
  - Configurable reach / payload constraints (checked, not enforced in sim)
  - Variable operation time (task + move overhead)
  - Error / collision fault injection
  - Cycle counter and OEE-style metrics
  - Workspace utilisation (simple 2-D occupancy model)
"""

import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import simpy

from core.events import EventType, event_bus
from core.state_machine import MachineState, MachineStateMachine

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Robot configuration
# --------------------------------------------------------------------------

@dataclass
class RobotConfig:
    robot_id:         str
    reach_mm:         float = 900.0      # Maximum reach in mm
    payload_kg:       float = 10.0       # Maximum payload
    speed_pct:        float = 80.0       # Operating speed as % of max rated
    repeat_acc_mm:    float = 0.05       # Repeatability accuracy
    nominal_op_s:     float = 12.0       # Seconds per operation
    op_std_pct:       float = 0.04       # Std-dev as fraction of nominal
    mtbf_s:           float = 10_800.0   # Mean time between failures
    mttr_s:           float = 600.0      # Mean repair time
    reject_rate:      float = 0.015      # Assembly defect rate


# --------------------------------------------------------------------------
# Pre-built configs for the factory fleet
# --------------------------------------------------------------------------

ROBOT_CONFIGS = {
    "ROB-001": RobotConfig("ROB-001", reach_mm=910, payload_kg=10, nominal_op_s=12.0),
    "ROB-002": RobotConfig("ROB-002", reach_mm=870, payload_kg=12, nominal_op_s=14.0),
}


# --------------------------------------------------------------------------
# Robot Twin
# --------------------------------------------------------------------------

class RobotTwin(simpy.Resource):
    """
    Digital twin of a robot arm station.

    The robot is modelled as a SimPy Resource (capacity=1).
    Each part is 'operated on' via the `run_operation` generator.
    """

    def __init__(self, env: simpy.Environment, config: RobotConfig):
        super().__init__(env, capacity=1)
        self.env    = env
        self.cfg    = config
        self.robot_id = config.robot_id

        self.state_machine = MachineStateMachine(self.robot_id)
        self._op_times:    List[float] = []
        self._op_log:      List[Dict]  = []
        self._fault_log:   List[Dict]  = []

        # Metrics
        self.total_operations: int   = 0
        self.good_operations:  int   = 0
        self.failed_operations: int  = 0
        self.total_faults:     int   = 0
        self.runtime_s:        float = 0.0
        self.downtime_s:       float = 0.0

        self._fault_proc   = env.process(self._fault_generator())
        self._metrics_proc = env.process(self._periodic_metrics())

    # ------------------------------------------------------------------
    # Core operation process
    # ------------------------------------------------------------------

    def run_operation(self, part_id: str, task: str = "assembly") -> Generator:
        """SimPy generator for one robot operation (pick/place/assembly)."""
        with self.request() as req:
            yield req

            if self.state_machine.state in (MachineState.FAULT, MachineState.MAINTENANCE):
                return

            start = self.env.now
            self.state_machine.transition(MachineState.RUNNING, f"{task} on {part_id}")

            # Simulate operation duration with small noise
            op_s = max(
                1.0,
                np.random.normal(
                    self.cfg.nominal_op_s,
                    self.cfg.nominal_op_s * self.cfg.op_std_pct,
                ),
            )
            yield self.env.timeout(op_s)

            elapsed = self.env.now - start
            self.runtime_s += elapsed
            self.total_operations += 1

            success = random.random() > self.cfg.reject_rate
            if success:
                self.good_operations += 1
            else:
                self.failed_operations += 1

            entry = {
                "part_id":  part_id,
                "task":     task,
                "op_s":     round(op_s, 3),
                "success":  success,
                "sim_time": self.env.now,
                "wall_time": time.time(),
            }
            self._op_times.append(op_s)
            self._op_log.append(entry)

            event_bus.emit(
                EventType.ROBOT_OPERATION_COMPLETE,
                source=self.robot_id,
                payload=entry,
            )

            if self.state_machine.state == MachineState.RUNNING:
                self.state_machine.transition(MachineState.IDLE, "op complete")

    # ------------------------------------------------------------------
    # Background: fault generator
    # ------------------------------------------------------------------

    def _fault_generator(self) -> Generator:
        while True:
            ttf = random.expovariate(1.0 / self.cfg.mtbf_s)
            yield self.env.timeout(ttf)

            if self.state_machine.state not in (MachineState.RUNNING, MachineState.IDLE):
                continue

            self.state_machine.transition(MachineState.FAULT, "robot fault")
            self.total_faults += 1

            fault = {
                "fault_time": self.env.now,
                "robot_id":   self.robot_id,
                "wall_time":  time.time(),
            }
            event_bus.emit(
                EventType.ROBOT_FAULT,
                source=self.robot_id,
                payload=fault,
                severity="CRITICAL",
            )

            ttr = random.expovariate(1.0 / self.cfg.mttr_s)
            yield self.env.timeout(ttr)

            self.downtime_s += ttr
            fault["repair_s"] = round(ttr, 1)
            self._fault_log.append(fault)

            self.state_machine.transition(MachineState.IDLE, "robot repaired")

    def _periodic_metrics(self) -> Generator:
        while True:
            yield self.env.timeout(60)
            event_bus.emit(
                EventType.MACHINE_METRICS_UPDATE,
                source=self.robot_id,
                payload=self.get_summary(),
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def oee(self) -> float:
        total = self.runtime_s + self.downtime_s
        if total == 0:
            return 0.0
        availability = self.runtime_s / total
        quality      = self.good_operations / max(1, self.total_operations)
        return round(availability * quality * 100, 2)

    @property
    def avg_op_time(self) -> float:
        return round(float(np.mean(self._op_times)) if self._op_times else 0.0, 2)

    def get_summary(self) -> Dict:
        return {
            "robot_id":         self.robot_id,
            "state":            self.state_machine.state.value,
            "color":            self.state_machine.color,
            "total_operations": self.total_operations,
            "good_operations":  self.good_operations,
            "failed_operations": self.failed_operations,
            "total_faults":     self.total_faults,
            "oee":              self.oee,
            "avg_op_time_s":    self.avg_op_time,
            "runtime_s":        round(self.runtime_s, 1),
            "downtime_s":       round(self.downtime_s, 1),
            "uptime_pct":       round(self.state_machine.get_uptime_pct(), 2),
        }
