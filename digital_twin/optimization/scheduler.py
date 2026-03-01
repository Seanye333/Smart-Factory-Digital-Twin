"""
Production Scheduler
======================
Generates and optimises production schedules for the factory line.

Features:
  - Earliest Due Date (EDD) scheduling heuristic
  - Shortest Processing Time (SPT) heuristic
  - Weighted Shortest Job First (WSJF)
  - Genetic Algorithm optimiser for makespan minimisation
  - What-if scenario comparison

The scheduler operates on job lists and returns sequenced schedules
with KPI estimates (makespan, tardiness, utilisation).
"""

import logging
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Job data model
# --------------------------------------------------------------------------

@dataclass
class Job:
    job_id:       str
    part_type:    str
    quantity:     int
    process_time: float         # seconds per unit
    due_date:     float         # Unix timestamp
    priority:     int = 5       # 1 (highest) – 10 (lowest)
    setup_time:   float = 0.0   # changeover seconds

    @property
    def total_time(self) -> float:
        return self.process_time * self.quantity + self.setup_time

    @property
    def urgency(self) -> float:
        slack = self.due_date - time.time()
        return max(0.0, self.total_time / max(1.0, slack))


@dataclass
class ScheduledJob:
    job:        Job
    start_time: float
    end_time:   float
    machine:    str

    @property
    def tardiness(self) -> float:
        return max(0.0, self.end_time - self.job.due_date)

    @property
    def is_late(self) -> bool:
        return self.end_time > self.job.due_date


@dataclass
class Schedule:
    jobs:       List[ScheduledJob]
    makespan:   float
    total_tardiness: float
    on_time_pct:    float
    algorithm:  str

    def to_dict(self) -> Dict:
        return {
            "algorithm":        self.algorithm,
            "makespan_s":       round(self.makespan, 1),
            "total_tardiness_s": round(self.total_tardiness, 1),
            "on_time_pct":      round(self.on_time_pct, 1),
            "n_jobs":           len(self.jobs),
            "jobs": [
                {
                    "job_id":     sj.job.job_id,
                    "part_type":  sj.job.part_type,
                    "quantity":   sj.job.quantity,
                    "start_time": sj.start_time,
                    "end_time":   sj.end_time,
                    "machine":    sj.machine,
                    "tardiness_s": round(sj.tardiness, 1),
                    "on_time":    not sj.is_late,
                }
                for sj in self.jobs
            ],
        }


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------

class ProductionScheduler:
    """
    Multi-algorithm production scheduler.

    Supports single-machine scheduling with multiple heuristics and
    a genetic algorithm meta-heuristic for harder instances.
    """

    def __init__(self, machine_ids: Optional[List[str]] = None):
        self.machine_ids = machine_ids or ["CNC-001", "CNC-002"]

    # ------------------------------------------------------------------
    # Generate sample jobs (demo / testing)
    # ------------------------------------------------------------------

    def generate_sample_jobs(self, n: int = 12) -> List[Job]:
        part_types  = ["PartA", "PartB", "PartC", "PartD"]
        now = time.time()
        jobs = []
        rng  = np.random.default_rng(int(now) % 10_000)

        for i in range(n):
            due_in_s = float(rng.uniform(3_600, 28_800))  # 1–8 hours from now
            jobs.append(Job(
                job_id=f"JOB-{i+1:03d}",
                part_type=random.choice(part_types),
                quantity=int(rng.integers(10, 80)),
                process_time=float(rng.uniform(30, 90)),
                due_date=now + due_in_s,
                priority=int(rng.integers(1, 11)),
                setup_time=float(rng.uniform(0, 600)),
            ))
        return jobs

    # ------------------------------------------------------------------
    # Single-machine scheduling heuristics
    # ------------------------------------------------------------------

    def schedule_edd(self, jobs: List[Job], machine: str = "CNC-001") -> Schedule:
        """Earliest Due Date — minimises maximum tardiness."""
        ordered = sorted(jobs, key=lambda j: j.due_date)
        return self._build_schedule(ordered, machine, "EDD")

    def schedule_spt(self, jobs: List[Job], machine: str = "CNC-001") -> Schedule:
        """Shortest Processing Time — minimises mean flow-time."""
        ordered = sorted(jobs, key=lambda j: j.total_time)
        return self._build_schedule(ordered, machine, "SPT")

    def schedule_wsjf(self, jobs: List[Job], machine: str = "CNC-001") -> Schedule:
        """Weighted Shortest Job First — balances priority and urgency."""
        ordered = sorted(jobs, key=lambda j: j.urgency * (11 - j.priority), reverse=True)
        return self._build_schedule(ordered, machine, "WSJF")

    def _build_schedule(self, jobs: List[Job], machine: str, algo: str) -> Schedule:
        scheduled = []
        cursor    = time.time()
        total_tar = 0.0

        for job in jobs:
            start = cursor + job.setup_time
            end   = start + job.process_time * job.quantity
            sj    = ScheduledJob(job=job, start_time=start, end_time=end, machine=machine)
            scheduled.append(sj)
            total_tar += sj.tardiness
            cursor     = end

        makespan    = scheduled[-1].end_time - scheduled[0].start_time if scheduled else 0
        on_time_pct = sum(1 for sj in scheduled if not sj.is_late) / max(1, len(scheduled)) * 100

        return Schedule(
            jobs=scheduled,
            makespan=makespan,
            total_tardiness=total_tar,
            on_time_pct=on_time_pct,
            algorithm=algo,
        )

    # ------------------------------------------------------------------
    # Genetic Algorithm optimiser
    # ------------------------------------------------------------------

    def schedule_genetic(
        self,
        jobs:         List[Job],
        machine:      str = "CNC-001",
        pop_size:     int = 60,
        n_generations: int = 150,
        mutation_rate: float = 0.15,
    ) -> Schedule:
        """
        Genetic Algorithm for job sequence optimisation.
        Minimises: makespan + 0.3 × total_tardiness
        """
        if len(jobs) <= 2:
            return self.schedule_edd(jobs, machine)

        rng = np.random.default_rng(42)

        def fitness(seq: List[int]) -> float:
            ordered   = [jobs[i] for i in seq]
            sched     = self._build_schedule(ordered, machine, "GA")
            return sched.makespan + 0.3 * sched.total_tardiness

        def crossover(p1: List[int], p2: List[int]) -> List[int]:
            a, b = sorted(rng.choice(len(p1), 2, replace=False))
            child = [-1] * len(p1)
            child[a:b] = p1[a:b]
            fill  = [g for g in p2 if g not in child]
            j = 0
            for i in range(len(child)):
                if child[i] == -1:
                    child[i] = fill[j]; j += 1
            return child

        def mutate(seq: List[int]) -> List[int]:
            if rng.random() < mutation_rate:
                a, b = sorted(rng.choice(len(seq), 2, replace=False))
                seq[a], seq[b] = seq[b], seq[a]
            return seq

        # Initialise population
        n = len(jobs)
        population = [rng.permutation(n).tolist() for _ in range(pop_size)]

        best_seq   = min(population, key=fitness)
        best_score = fitness(best_seq)

        for gen in range(n_generations):
            # Tournament selection
            new_pop = []
            for _ in range(pop_size):
                a, b   = rng.choice(pop_size, 2, replace=False)
                winner = population[a] if fitness(population[a]) < fitness(population[b]) else population[b]
                new_pop.append(winner)

            # Crossover + mutation
            children = []
            for i in range(0, pop_size - 1, 2):
                c1 = mutate(crossover(new_pop[i], new_pop[i+1]))
                c2 = mutate(crossover(new_pop[i+1], new_pop[i]))
                children.extend([c1, c2])

            population = children[:pop_size]

            gen_best = min(population, key=fitness)
            gen_score = fitness(gen_best)
            if gen_score < best_score:
                best_score = gen_score
                best_seq   = gen_best

        logger.info(f"[Scheduler] GA converged: score={best_score:.1f} after {n_generations} gen")
        ordered = [jobs[i] for i in best_seq]
        return self._build_schedule(ordered, machine, "GeneticAlgorithm")

    # ------------------------------------------------------------------
    # Algorithm comparison
    # ------------------------------------------------------------------

    def compare_algorithms(self, jobs: List[Job], machine: str = "CNC-001") -> Dict:
        schedules = {
            "EDD":  self.schedule_edd(jobs,     machine),
            "SPT":  self.schedule_spt(jobs,     machine),
            "WSJF": self.schedule_wsjf(jobs,    machine),
            "GA":   self.schedule_genetic(jobs, machine),
        }
        best_algo = min(schedules, key=lambda a: schedules[a].makespan + schedules[a].total_tardiness)

        return {
            "best_algorithm": best_algo,
            "comparison": {
                name: {
                    "makespan_s":       round(s.makespan, 1),
                    "total_tardiness_s": round(s.total_tardiness, 1),
                    "on_time_pct":      round(s.on_time_pct, 1),
                }
                for name, s in schedules.items()
            },
            "recommended_schedule": schedules[best_algo].to_dict(),
        }
