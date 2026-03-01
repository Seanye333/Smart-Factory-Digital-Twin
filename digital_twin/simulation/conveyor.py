"""
Conveyor Digital Twin (SimPy)
================================
Models a belt / roller conveyor as a bounded FIFO queue.

Features:
  - Configurable belt speed (m/min) and length (m)
  - Capacity-limited queue — signals BLOCKED upstream when full
  - Stochastic jam events (exponential inter-arrival)
  - Part tracking and throughput metrics
  - Event bus integration
"""

import random
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import simpy

from core.events import EventType, event_bus
from core.state_machine import MachineState, MachineStateMachine

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Conveyor twin
# --------------------------------------------------------------------------

class ConveyorTwin:
    """
    Digital twin of a factory conveyor segment.

    Parts are modelled as tokens.  The conveyor has a physical capacity;
    attempting to place a part when full blocks the upstream machine.

    Parameters
    ----------
    env          : SimPy environment
    conveyor_id  : unique identifier (e.g. "CNV-001")
    speed_m_min  : belt speed in metres per minute
    length_m     : belt length in metres
    capacity     : maximum parts on belt simultaneously
    jam_mtbf_s   : mean time between jams (sim-seconds); 0 = no jams
    jam_mttr_s   : mean repair time (sim-seconds)
    """

    def __init__(
        self,
        env:         simpy.Environment,
        conveyor_id: str,
        speed_m_min: float = 12.0,
        length_m:    float = 5.0,
        capacity:    int   = 10,
        jam_mtbf_s:  float = 14_400.0,
        jam_mttr_s:  float = 300.0,
    ):
        self.env         = env
        self.conveyor_id = conveyor_id
        self.speed_m_min = speed_m_min
        self.length_m    = length_m
        self.capacity    = capacity
        self.jam_mtbf_s  = jam_mtbf_s
        self.jam_mttr_s  = jam_mttr_s

        # Transit time in sim-seconds
        self.transit_s = (length_m / speed_m_min) * 60.0

        self.state_machine = MachineStateMachine(conveyor_id)
        self._container    = simpy.Container(env, capacity=capacity, init=0)
        self._is_jammed    = False

        # Metrics
        self._parts_in:    int   = 0
        self._parts_out:   int   = 0
        self._total_jam_s: float = 0.0
        self._jam_count:   int   = 0
        self._part_log:    List[Dict] = []

        if jam_mtbf_s > 0:
            self._jam_proc = env.process(self._jam_generator())

    # ------------------------------------------------------------------
    # Core transport process
    # ------------------------------------------------------------------

    def transport(self, part_id: str) -> Generator:
        """
        SimPy generator: place part on belt, wait transit time, deliver.
        Blocks if conveyor is at capacity.
        """
        # Wait for a slot on the belt
        yield self._container.put(1)
        self._parts_in += 1

        event_bus.emit(
            EventType.CONVEYOR_PART_ARRIVED,
            source=self.conveyor_id,
            payload={"part_id": part_id, "sim_time": self.env.now, "load": self._container.level},
        )

        # If jammed, wait until unjammed
        while self._is_jammed:
            yield self.env.timeout(1.0)

        # Normal transit
        yield self.env.timeout(self.transit_s)

        yield self._container.get(1)
        self._parts_out += 1

        entry = {
            "part_id":    part_id,
            "transit_s":  self.transit_s,
            "sim_time":   self.env.now,
            "wall_time":  time.time(),
        }
        self._part_log.append(entry)

        event_bus.emit(
            EventType.CONVEYOR_PART_DEPARTED,
            source=self.conveyor_id,
            payload=entry,
        )

    # ------------------------------------------------------------------
    # Jam generator
    # ------------------------------------------------------------------

    def _jam_generator(self) -> Generator:
        while True:
            ttj = random.expovariate(1.0 / self.jam_mtbf_s)
            yield self.env.timeout(ttj)

            self._is_jammed = True
            self._jam_count += 1
            self.state_machine.transition(MachineState.FAULT, "conveyor jam")

            event_bus.emit(
                EventType.CONVEYOR_JAMMED,
                source=self.conveyor_id,
                payload={"jam_count": self._jam_count, "sim_time": self.env.now},
                severity="WARNING",
            )

            ttr = random.expovariate(1.0 / self.jam_mttr_s)
            yield self.env.timeout(ttr)

            self._total_jam_s += ttr
            self._is_jammed = False
            self.state_machine.transition(MachineState.RUNNING, "unjammed")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def current_load(self) -> int:
        return self._container.level

    @property
    def load_pct(self) -> float:
        return round(self.current_load / self.capacity * 100, 1)

    @property
    def throughput_pph(self) -> float:
        elapsed = self.env.now
        if elapsed == 0:
            return 0.0
        return round(self._parts_out / elapsed * 3600, 2)

    def get_summary(self) -> Dict:
        return {
            "conveyor_id":   self.conveyor_id,
            "state":         self.state_machine.state.value,
            "speed_m_min":   self.speed_m_min,
            "length_m":      self.length_m,
            "capacity":      self.capacity,
            "current_load":  self.current_load,
            "load_pct":      self.load_pct,
            "parts_in":      self._parts_in,
            "parts_out":     self._parts_out,
            "jam_count":     self._jam_count,
            "total_jam_s":   round(self._total_jam_s, 1),
            "transit_s":     self.transit_s,
            "throughput_pph": self.throughput_pph,
            "is_jammed":     self._is_jammed,
        }
