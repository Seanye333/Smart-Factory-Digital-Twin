"""
Machine Digital Twin (SimPy)
==============================
Models a factory machine as a SimPy Resource with:
  - State machine (IDLE / RUNNING / FAULT / MAINTENANCE / BLOCKED / STARVED)
  - Stochastic cycle times (normal distribution around nominal)
  - Quality inspection (reject rate per machine)
  - Exponential fault generation (MTBF / MTTR)
  - OEE, throughput, and reject-rate metrics
  - Event bus integration
"""

import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import numpy as np
import simpy

from core.events import EventType, event_bus
from core.state_machine import MachineState, MachineStateMachine

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metrics container
# --------------------------------------------------------------------------

@dataclass
class MachineMetrics:
    machine_id:     str
    total_cycles:   int   = 0
    good_parts:     int   = 0
    rejected_parts: int   = 0
    total_faults:   int   = 0
    downtime_s:     float = 0.0
    runtime_s:      float = 0.0

    @property
    def oee(self) -> float:
        """
        OEE = Availability × Performance × Quality
        Simplified: A = runtime / (runtime+downtime), Q = good/total, P = 1.0
        """
        total = self.downtime_s + self.runtime_s
        if total == 0:
            return 0.0
        availability = self.runtime_s / total
        quality      = self.good_parts / max(1, self.total_cycles)
        return round(availability * quality * 100, 2)

    @property
    def reject_rate(self) -> float:
        return round(self.rejected_parts / max(1, self.total_cycles) * 100, 2)

    @property
    def throughput_pph(self) -> float:
        """Good parts per hour."""
        return round(self.good_parts / max(1, self.runtime_s) * 3600, 2)

    def to_dict(self) -> Dict:
        return {
            "machine_id":     self.machine_id,
            "total_cycles":   self.total_cycles,
            "good_parts":     self.good_parts,
            "rejected_parts": self.rejected_parts,
            "reject_rate":    self.reject_rate,
            "total_faults":   self.total_faults,
            "downtime_s":     round(self.downtime_s, 1),
            "runtime_s":      round(self.runtime_s, 1),
            "oee":            self.oee,
            "throughput_pph": self.throughput_pph,
        }


# --------------------------------------------------------------------------
# Machine twin
# --------------------------------------------------------------------------

class MachineTwin(simpy.Resource):
    """
    Digital Twin of a single factory machine.

    Parameters
    ----------
    env               : SimPy environment
    machine_id        : unique identifier (e.g. "CNC-001")
    nominal_cycle_s   : mean seconds per cycle
    cycle_std_pct     : std-dev as fraction of nominal (default 5 %)
    reject_rate       : fraction of parts that fail QC (default 2 %)
    mtbf_s            : mean time between failures in sim-seconds
    mttr_s            : mean time to repair in sim-seconds
    """

    def __init__(
        self,
        env:             simpy.Environment,
        machine_id:      str,
        nominal_cycle_s: float = 30.0,
        cycle_std_pct:   float = 0.05,
        reject_rate:     float = 0.02,
        mtbf_s:          float = 7_200.0,
        mttr_s:          float =   900.0,
    ):
        super().__init__(env, capacity=1)

        self.machine_id      = machine_id
        self.env             = env
        self.nominal_cycle_s = nominal_cycle_s
        self.cycle_std_pct   = cycle_std_pct
        self.reject_rate     = reject_rate
        self.mtbf_s          = mtbf_s
        self.mttr_s          = mttr_s

        self.state_machine = MachineStateMachine(machine_id)
        self.metrics       = MachineMetrics(machine_id)

        self._cycle_times:    List[float] = []
        self._fault_log:      List[Dict]  = []
        self._production_log: List[Dict]  = []

        # Background SimPy processes
        self._fault_proc   = env.process(self._fault_generator())
        self._metrics_proc = env.process(self._periodic_metrics_emit())

    # ------------------------------------------------------------------
    # Core process — called by factory line for each part
    # ------------------------------------------------------------------

    def process_part(self, part_id: str, part_type: str = "standard") -> Generator:
        """
        SimPy generator process.
        Acquires the machine resource, runs the cycle, releases.
        """
        arrival_time = self.env.now

        with self.request() as req:
            # Wait if machine is busy or in fault / maintenance
            yield req

            if self.state_machine.state in (MachineState.FAULT, MachineState.MAINTENANCE):
                # Part has to wait until machine is back up
                return

            run_start = self.env.now
            self.state_machine.transition(MachineState.RUNNING, f"processing {part_id}")

            cycle_s = max(
                0.5,
                np.random.normal(self.nominal_cycle_s, self.nominal_cycle_s * self.cycle_std_pct),
            )
            yield self.env.timeout(cycle_s)

            run_end = self.env.now
            self.metrics.runtime_s += run_end - run_start

            # Quality gate
            is_good = random.random() > self.reject_rate
            if is_good:
                self.metrics.good_parts += 1
            else:
                self.metrics.rejected_parts += 1
            self.metrics.total_cycles += 1

            actual_ct = run_end - arrival_time
            self._cycle_times.append(cycle_s)

            log_entry = {
                "part_id":    part_id,
                "part_type":  part_type,
                "cycle_s":    round(cycle_s, 3),
                "actual_ct":  round(actual_ct, 3),
                "good":       is_good,
                "sim_time":   self.env.now,
                "wall_time":  time.time(),
            }
            self._production_log.append(log_entry)

            event_bus.emit(
                EventType.MACHINE_CYCLE_COMPLETE,
                source=self.machine_id,
                payload=log_entry,
            )

            if self.state_machine.state == MachineState.RUNNING:
                self.state_machine.transition(MachineState.IDLE, "cycle complete")

    # ------------------------------------------------------------------
    # Background: fault generator
    # ------------------------------------------------------------------

    def _fault_generator(self) -> Generator:
        """Inject random faults using exponential inter-arrival times."""
        while True:
            ttf = random.expovariate(1.0 / self.mtbf_s)
            yield self.env.timeout(ttf)

            if self.state_machine.state not in (MachineState.RUNNING, MachineState.IDLE,
                                                 MachineState.STARVED, MachineState.BLOCKED):
                continue   # already in fault/maintenance — skip

            self.state_machine.transition(MachineState.FAULT, "random failure")
            self.metrics.total_faults += 1

            fault_start = self.env.now
            fault_entry = {
                "fault_start": fault_start,
                "machine_id":  self.machine_id,
                "wall_time":   time.time(),
            }
            event_bus.emit(
                EventType.MACHINE_FAULT,
                source=self.machine_id,
                payload=fault_entry,
                severity="CRITICAL",
            )

            ttr = random.expovariate(1.0 / self.mttr_s)
            yield self.env.timeout(ttr)

            fault_entry["repair_s"] = round(ttr, 1)
            self._fault_log.append(fault_entry)
            self.metrics.downtime_s += ttr

            self.state_machine.transition(MachineState.IDLE, "repair complete")
            event_bus.emit(
                EventType.MACHINE_FAULT_CLEARED,
                source=self.machine_id,
                payload={"repair_s": round(ttr, 1)},
            )

    # ------------------------------------------------------------------
    # Background: periodic metrics reporter
    # ------------------------------------------------------------------

    def _periodic_metrics_emit(self) -> Generator:
        while True:
            yield self.env.timeout(60)   # every sim-minute
            event_bus.emit(
                EventType.MACHINE_METRICS_UPDATE,
                source=self.machine_id,
                payload=self.get_summary(),
            )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    @property
    def avg_cycle_time(self) -> float:
        return round(float(np.mean(self._cycle_times)) if self._cycle_times else 0.0, 2)

    @property
    def cycle_time_std(self) -> float:
        return round(float(np.std(self._cycle_times)) if len(self._cycle_times) > 1 else 0.0, 2)

    def get_summary(self) -> Dict:
        m = self.metrics.to_dict()
        m.update({
            "state":          self.state_machine.state.value,
            "color":          self.state_machine.color,
            "avg_cycle_s":    self.avg_cycle_time,
            "cycle_std_s":    self.cycle_time_std,
            "uptime_pct":     round(self.state_machine.get_uptime_pct(), 2),
            "state_dist":     self.state_machine.get_state_distribution(),
        })
        return m

    def recent_faults(self, n: int = 10) -> List[Dict]:
        return self._fault_log[-n:]

    def recent_production(self, n: int = 20) -> List[Dict]:
        return self._production_log[-n:]
