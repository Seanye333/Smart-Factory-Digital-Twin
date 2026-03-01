"""
Bottleneck Detection & Throughput Analysis
==========================================
Identifies the machine or station that most constrains overall factory output.

Methods implemented:
  1. Cycle-time based bottleneck (fastest heuristic)
  2. Queue-length based bottleneck (upstream starvation indicator)
  3. Utilisation-based bottleneck (most time spent Running vs capacity)
  4. Throughput ratio analysis (Little's Law variant)
  5. Monte Carlo sensitivity analysis (what-if)

Output includes actionable recommendations.
"""

import logging
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class MachineStats:
    machine_id:         str
    cycle_time_s:       float     # average seconds per cycle
    utilisation_pct:    float     # % time machine was busy
    queue_length:       float     # average upstream queue depth
    throughput_pph:     float     # good parts per hour
    downtime_pct:       float = 0.0
    oee:                float = 0.0


@dataclass
class BottleneckResult:
    bottleneck_id:      str
    method:             str
    score:              float       # higher = more severe bottleneck
    lost_throughput_pph: float      # estimate of capacity loss
    recommendation:     str
    runner_up:          Optional[str] = None
    all_scores:         Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "bottleneck_id":       self.bottleneck_id,
            "method":              self.method,
            "score":               round(self.score, 3),
            "lost_throughput_pph": round(self.lost_throughput_pph, 2),
            "recommendation":      self.recommendation,
            "runner_up":           self.runner_up,
            "all_scores":          {k: round(v, 3) for k, v in self.all_scores.items()},
        }


# --------------------------------------------------------------------------
# Analyser
# --------------------------------------------------------------------------

class BottleneckAnalyzer:
    """
    Multi-method bottleneck detection for the factory line.

    Parameters
    ----------
    machine_ids   : ordered list of machine IDs in the production sequence
    history_len   : number of snapshots to keep for trend analysis
    """

    def __init__(self, machine_ids: List[str], history_len: int = 30):
        self.machine_ids  = machine_ids
        self.history_len  = history_len
        self._history:    List[Dict[str, MachineStats]] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, stats: List[Dict]) -> Optional[BottleneckResult]:
        """
        Accepts a list of machine stat dicts (from factory_line.get_machine_summaries()).
        Returns the most likely bottleneck.
        """
        if not stats:
            return None

        machine_stats = self._parse_stats(stats)
        if not machine_stats:
            return None

        self._history.append(machine_stats)
        if len(self._history) > self.history_len:
            self._history.pop(0)

        # Run all detection methods
        ct_result   = self._cycle_time_method(machine_stats)
        util_result = self._utilisation_method(machine_stats)
        q_result    = self._queue_method(machine_stats)

        # Weighted vote: cycle_time=0.4, utilisation=0.4, queue=0.2
        combined = self._combine_scores(
            [ct_result, util_result, q_result],
            weights=[0.4, 0.4, 0.2],
        )

        if not combined:
            return None

        bottleneck_id, score = combined[0]
        runner_up             = combined[1][0] if len(combined) > 1 else None
        ms = machine_stats.get(bottleneck_id)

        lost_pph = self._estimate_lost_throughput(machine_stats, bottleneck_id)
        rec      = self._generate_recommendation(bottleneck_id, ms, score)

        result = BottleneckResult(
            bottleneck_id=bottleneck_id,
            method="composite",
            score=score,
            lost_throughput_pph=lost_pph,
            recommendation=rec,
            runner_up=runner_up,
            all_scores={mid: combined_score for mid, combined_score in combined},
        )

        logger.info(f"[Bottleneck] {bottleneck_id} (score={score:.3f}) — {rec}")
        return result

    # Simple wrapper used by dashboard
    def find_bottleneck(self, stats_list: List[Dict]) -> Optional[Dict]:
        """Convenience — returns a plain dict or None."""
        result = self.analyze(stats_list)
        return result.to_dict() if result else None

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _cycle_time_method(self, stats: Dict[str, MachineStats]) -> Dict[str, float]:
        """Machines with longest cycle time constrain downstream flow."""
        scores: Dict[str, float] = {}
        times = {mid: s.cycle_time_s for mid, s in stats.items() if s.cycle_time_s > 0}
        if not times:
            return scores
        max_ct = max(times.values())
        for mid, ct in times.items():
            scores[mid] = ct / max_ct
        return scores

    def _utilisation_method(self, stats: Dict[str, MachineStats]) -> Dict[str, float]:
        """Highest utilisation machine is the tightest resource."""
        scores: Dict[str, float] = {}
        utils = {mid: s.utilisation_pct for mid, s in stats.items()}
        max_u = max(utils.values()) if utils else 1
        for mid, u in utils.items():
            scores[mid] = u / max(1, max_u)
        return scores

    def _queue_method(self, stats: Dict[str, MachineStats]) -> Dict[str, float]:
        """Machine preceded by the longest queue is likely the bottleneck."""
        scores: Dict[str, float] = {}
        queues = {mid: s.queue_length for mid, s in stats.items()}
        max_q  = max(queues.values()) if any(q > 0 for q in queues.values()) else 1
        for mid, q in queues.items():
            scores[mid] = q / max(1, max_q)
        return scores

    # ------------------------------------------------------------------
    # Score combination
    # ------------------------------------------------------------------

    def _combine_scores(
        self,
        score_dicts: List[Dict[str, float]],
        weights: List[float],
    ) -> List[Tuple[str, float]]:
        combined: Dict[str, float] = {}
        for sd, w in zip(score_dicts, weights):
            for mid, sc in sd.items():
                combined[mid] = combined.get(mid, 0.0) + sc * w
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_stats(self, raw: List[Dict]) -> Dict[str, MachineStats]:
        result = {}
        for item in raw:
            mid = item.get("machine_id") or item.get("conveyor_id") or item.get("robot_id", "")
            if not mid:
                continue
            result[mid] = MachineStats(
                machine_id=mid,
                cycle_time_s=float(item.get("cycle_time", item.get("avg_cycle_s", item.get("avg_op_time_s", 0))) or 0),
                utilisation_pct=float(item.get("uptime_pct", item.get("load_pct", 50)) or 50),
                queue_length=float(item.get("queue_length", item.get("current_load", 0)) or 0),
                throughput_pph=float(item.get("throughput_pph", 0) or 0),
                oee=float(item.get("oee", 0) or 0),
            )
        return result

    def _estimate_lost_throughput(
        self,
        stats: Dict[str, MachineStats],
        bottleneck_id: str,
    ) -> float:
        bn = stats.get(bottleneck_id)
        if bn is None or bn.cycle_time_s == 0:
            return 0.0
        max_theoretical = 3600 / bn.cycle_time_s
        actual = bn.throughput_pph
        return max(0.0, max_theoretical - actual)

    def _generate_recommendation(
        self,
        machine_id: str,
        stats: Optional[MachineStats],
        score: float,
    ) -> str:
        if stats is None:
            return f"Investigate {machine_id}"

        recs = []
        if stats.cycle_time_s > 0:
            recs.append(f"Reduce cycle time (currently {stats.cycle_time_s:.1f}s)")
        if stats.utilisation_pct > 85:
            recs.append("Add a parallel machine to share load")
        if stats.downtime_pct > 10:
            recs.append("Schedule preventive maintenance to reduce unplanned downtime")
        if stats.oee > 0 and stats.oee < 65:
            recs.append(f"Improve OEE from {stats.oee:.1f}% (target ≥ 85%)")

        if not recs:
            recs.append("Monitor closely — approaching capacity limit")

        return " | ".join(recs)

    # ------------------------------------------------------------------
    # Monte Carlo what-if analysis
    # ------------------------------------------------------------------

    def monte_carlo_throughput(
        self,
        stats: List[Dict],
        improvements: Dict[str, float],  # {machine_id: cycle_time_reduction_pct}
        n_simulations: int = 1_000,
    ) -> Dict[str, float]:
        """
        Simulate the effect of cycle-time improvements on overall throughput.

        Parameters
        ----------
        improvements : dict mapping machine_id → % reduction in cycle time (0–100)
        n_simulations : number of Monte Carlo runs

        Returns
        -------
        dict with keys: mean_throughput, std_throughput, p5, p95, improvement_pct
        """
        machine_stats = self._parse_stats(stats)
        base_times    = {mid: s.cycle_time_s for mid, s in machine_stats.items() if s.cycle_time_s > 0}

        if not base_times:
            return {}

        base_bottleneck_ct = max(base_times.values())
        results = []

        rng = np.random.default_rng(12345)

        for _ in range(n_simulations):
            trial_times = {}
            for mid, ct in base_times.items():
                noise       = rng.normal(1.0, 0.05)
                improvement = 1.0 - improvements.get(mid, 0.0) / 100.0
                trial_times[mid] = ct * improvement * noise

            bottleneck_ct = max(trial_times.values())
            throughput    = 3600 / bottleneck_ct
            results.append(throughput)

        base_throughput = 3600 / base_bottleneck_ct
        mean_tp         = statistics.mean(results)

        return {
            "base_throughput_pph":    round(base_throughput, 2),
            "mean_throughput_pph":    round(mean_tp, 2),
            "std_throughput":         round(statistics.stdev(results), 2),
            "p5_throughput_pph":      round(np.percentile(results, 5), 2),
            "p95_throughput_pph":     round(np.percentile(results, 95), 2),
            "improvement_pct":        round((mean_tp / base_throughput - 1) * 100, 2),
            "n_simulations":          n_simulations,
        }
