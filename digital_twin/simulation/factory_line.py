"""
Factory Line Orchestrator (SimPy)
====================================
Wires up the complete production line:

    [Raw Parts] → CNC-001 ─┐
                             ├─→ CNV-001 → ROB-001 ─┐
                   CNC-002 ─┘                         ├─→ CNV-002 → QAS-001 → [Output]
                                            ROB-002 ─┘

Parts arrive according to a Poisson process, join a shared CNC queue,
are machined → conveyed → assembled by a robot → inspected → shipped.

The FactoryLine object exposes a rich summary dict consumed by the
dashboard and API layer.
"""

import random
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import simpy

from simulation.machine  import MachineTwin
from simulation.conveyor import ConveyorTwin
from simulation.robot    import RobotTwin, RobotConfig
from core.events         import EventType, event_bus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Shared output buffer (inspection results)
# --------------------------------------------------------------------------

@dataclass
class OutputBuffer:
    good_parts:     int = 0
    rejected_parts: int = 0
    total_parts:    int = 0
    shift_good:     int = 0     # reset each shift

    def accept(self) -> None:
        self.good_parts  += 1
        self.total_parts += 1
        self.shift_good  += 1

    def reject(self) -> None:
        self.rejected_parts += 1
        self.total_parts    += 1

    @property
    def yield_pct(self) -> float:
        return round(self.good_parts / max(1, self.total_parts) * 100, 2)


# --------------------------------------------------------------------------
# Part arrival generator
# --------------------------------------------------------------------------

def part_source(
    env:          simpy.Environment,
    line:         "FactoryLine",
    arrival_rate: float,          # parts per sim-second (Poisson)
) -> Generator:
    """Poisson part arrival generator — feeds raw parts into the line."""
    part_idx = 0
    while True:
        inter = random.expovariate(arrival_rate)
        yield env.timeout(inter)
        part_idx += 1
        part_id = f"P{part_idx:06d}"
        env.process(line.process_part(part_id))


# --------------------------------------------------------------------------
# Factory Line
# --------------------------------------------------------------------------

class FactoryLine:
    """
    Complete digital twin of a two-machine CNC + two-robot assembly line.

    Parameters
    ----------
    sim_speed     : sim-seconds per real-second (1.0 = real-time, 60.0 = 1 min/sec)
    arrival_rate  : mean parts per sim-second entering the line
    """

    # Default line parameters
    DEFAULT_CONFIG = {
        "cnc_nominal_cycle_s":  45.0,
        "cnc_mtbf_s":        7_200.0,
        "cnc_mttr_s":          600.0,
        "robot_nominal_op_s":  12.0,
        "robot_mtbf_s":     10_800.0,
        "robot_mttr_s":        400.0,
        "conv1_speed_m_min":   12.0,
        "conv2_speed_m_min":   10.0,
        "conv_length_m":        5.0,
        "qa_cycle_s":           8.0,
        "qa_reject_rate":       0.03,
    }

    def __init__(
        self,
        sim_speed:    float = 60.0,
        arrival_rate: float = 0.022,   # ~1.3 parts/min
        config: Optional[Dict] = None,
    ):
        self.sim_speed    = sim_speed
        self.arrival_rate = arrival_rate
        self.cfg          = {**self.DEFAULT_CONFIG, **(config or {})}

        self.env       = simpy.Environment()
        self.output    = OutputBuffer()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._lock     = threading.RLock()
        self._started_at: Optional[float] = None

        self._build_line()

    # ------------------------------------------------------------------
    # Build / wire the production assets
    # ------------------------------------------------------------------

    def _build_line(self) -> None:
        cfg = self.cfg
        env = self.env

        # CNC machines (shared resource pool, capacity=2)
        self.cnc1 = MachineTwin(
            env, "CNC-001",
            nominal_cycle_s=cfg["cnc_nominal_cycle_s"],
            mtbf_s=cfg["cnc_mtbf_s"],
            mttr_s=cfg["cnc_mttr_s"],
            reject_rate=0.02,
        )
        self.cnc2 = MachineTwin(
            env, "CNC-002",
            nominal_cycle_s=cfg["cnc_nominal_cycle_s"] * 1.07,  # slightly slower
            mtbf_s=cfg["cnc_mtbf_s"] * 0.9,
            mttr_s=cfg["cnc_mttr_s"],
            reject_rate=0.025,
        )

        # Conveyor between CNC and robots
        self.conv1 = ConveyorTwin(
            env, "CNV-001",
            speed_m_min=cfg["conv1_speed_m_min"],
            length_m=cfg["conv_length_m"],
            capacity=8,
        )

        # Assembly robots
        self.rob1 = RobotTwin(env, RobotConfig(
            "ROB-001",
            nominal_op_s=cfg["robot_nominal_op_s"],
            mtbf_s=cfg["robot_mtbf_s"],
            mttr_s=cfg["robot_mttr_s"],
        ))
        self.rob2 = RobotTwin(env, RobotConfig(
            "ROB-002",
            nominal_op_s=cfg["robot_nominal_op_s"] * 1.17,
            mtbf_s=cfg["robot_mtbf_s"] * 0.85,
            mttr_s=cfg["robot_mttr_s"],
        ))

        # Conveyor between robots and QA
        self.conv2 = ConveyorTwin(
            env, "CNV-002",
            speed_m_min=cfg["conv2_speed_m_min"],
            length_m=cfg["conv_length_m"],
            capacity=6,
        )

        # QA inspection (models as a machine)
        self.qa = MachineTwin(
            env, "QAS-001",
            nominal_cycle_s=cfg["qa_cycle_s"],
            reject_rate=cfg["qa_reject_rate"],
            mtbf_s=20_000.0,    # QA station very reliable
            mttr_s=200.0,
        )

        # All machine twins for unified querying
        self.machines: Dict[str, object] = {
            "CNC-001": self.cnc1,
            "CNC-002": self.cnc2,
            "CNV-001": self.conv1,
            "ROB-001": self.rob1,
            "ROB-002": self.rob2,
            "CNV-002": self.conv2,
            "QAS-001": self.qa,
        }

        # Part arrival source
        self.env.process(part_source(self.env, self, self.arrival_rate))

    # ------------------------------------------------------------------
    # Part flow process
    # ------------------------------------------------------------------

    def process_part(self, part_id: str) -> Generator:
        """Full journey of one part through the factory line."""

        # 1. CNC machining — pick the less-loaded machine
        if len(self.cnc1.queue) <= len(self.cnc2.queue):
            yield from self.cnc1.process_part(part_id, "cnc_machining")
        else:
            yield from self.cnc2.process_part(part_id, "cnc_machining")

        # 2. Transport to robot cell
        yield from self.conv1.transport(part_id)

        # 3. Assembly robot — pick the less-loaded robot
        if len(self.rob1.queue) <= len(self.rob2.queue):
            yield from self.rob1.run_operation(part_id, "assembly")
        else:
            yield from self.rob2.run_operation(part_id, "assembly")

        # 4. Transport to QA
        yield from self.conv2.transport(part_id)

        # 5. Quality inspection
        yield from self.qa.process_part(part_id, "qa_inspection")

        # 6. Record outcome
        with self._lock:
            if random.random() > self.cfg["qa_reject_rate"]:
                self.output.accept()
                event_bus.emit(
                    EventType.PRODUCTION_ALERT,
                    source="QAS-001",
                    payload={"part_id": part_id, "result": "PASS", "sim_time": self.env.now},
                    severity="INFO",
                )
            else:
                self.output.reject()
                event_bus.emit(
                    EventType.PRODUCTION_ALERT,
                    source="QAS-001",
                    payload={"part_id": part_id, "result": "FAIL", "sim_time": self.env.now},
                    severity="WARNING",
                )

    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------

    def start(self, run_until: float = float("inf")) -> None:
        """Start the simulation in a background thread."""
        self._running = True
        self._started_at = time.time()

        def _loop():
            step_s = 1.0 / self.sim_speed   # real-time seconds per sim-second
            while self._running and self.env.now < run_until:
                try:
                    self.env.step()
                except simpy.core.EmptySchedule:
                    break
                except Exception as exc:
                    logger.error(f"[FactoryLine] Simulation error: {exc}")
                    break
                time.sleep(step_s)

        self._thread = threading.Thread(target=_loop, daemon=True, name="FactoryLine-sim")
        self._thread.start()
        logger.info(f"[FactoryLine] Simulation started  speed={self.sim_speed}x")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[FactoryLine] Simulation stopped")

    # ------------------------------------------------------------------
    # State queries (thread-safe reads)
    # ------------------------------------------------------------------

    def get_machine_summaries(self) -> Dict[str, Dict]:
        summaries = {}
        for mid, twin in self.machines.items():
            if hasattr(twin, "get_summary"):
                summaries[mid] = twin.get_summary()
        return summaries

    def get_line_summary(self) -> Dict:
        cnc_throughput  = self.cnc1.metrics.throughput_pph + self.cnc2.metrics.throughput_pph
        rob_throughput  = self.rob1.good_operations + self.rob2.good_operations
        wall_elapsed    = time.time() - (self._started_at or time.time())

        return {
            "sim_time":          round(self.env.now, 1),
            "wall_elapsed_s":    round(wall_elapsed, 1),
            "sim_speed_x":       self.sim_speed,
            "output_good":       self.output.good_parts,
            "output_rejected":   self.output.rejected_parts,
            "output_total":      self.output.total_parts,
            "yield_pct":         self.output.yield_pct,
            "cnc_throughput_pph": round(cnc_throughput, 2),
            "conv1_load_pct":    self.conv1.load_pct,
            "conv2_load_pct":    self.conv2.load_pct,
            "conv1_jams":        self.conv1._jam_count,
            "conv2_jams":        self.conv2._jam_count,
        }
