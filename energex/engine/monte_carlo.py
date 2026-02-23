"""
EnergeX — Monte Carlo Engine (Phase C)

Runs N independent simulations with different seeds to build a statistical
distribution of outcomes and compute P10/P50/P90/P99 confidence bands.

Why this matters
----------------
A point-estimate reliability number ("ENS = 245 kWh/yr") is misleading because
outage patterns vary stochastically year-to-year and trial-to-trial. A P90 band
("90 % of years will see ENS ≤ 520 kWh") is bankable — lenders and insurers
require it.

Design
------
Each trial i:
  1. Uses seed = base_seed + i × PRIME so seeds are well-separated.
  2. Runs the full simulation (all project years) to get per-year reliability
     metrics and aggregate financials.
  3. Records per-trial summaries: mean annual ENS, mean annual downtime, NPV,
     total outage cost, LCOE.

After N trials the engine computes configurable percentile bands.

Usage
-----
    from energex.engine.monte_carlo import MonteCarloEngine

    engine = MonteCarloEngine(config, n_trials=200, base_seed=42)
    mc_result = engine.run(progress_callback=lambda p: print(f"{p*100:.0f}%"))

    print(mc_result.ens_p50, mc_result.ens_p90, mc_result.ens_p99)
    print(mc_result.npv_p10, mc_result.npv_p50)   # downside risk for finance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from energex.domain.schemas import ProjectConfig


# Large prime for seed spacing — ensures trials get well-separated seeds
_SEED_PRIME = 31337


# ---------------------------------------------------------------------------
# Per-trial record
# ---------------------------------------------------------------------------

@dataclass
class MCTrialResult:
    """Summary metrics for one Monte Carlo trial."""
    trial: int
    seed: int

    # Reliability — mean across all project years
    mean_ens_kwh_yr: float
    mean_downtime_hrs_yr: float
    mean_continuity_pct: float
    mean_outage_events_yr: float

    # Financials
    npv: float
    total_outage_cost: float
    lcoe_rs_kwh: float
    total_capex: float

    # Year-1 values (representative)
    y1_ens_kwh: float
    y1_downtime_hrs: float
    y1_continuity_pct: float


# ---------------------------------------------------------------------------
# Monte Carlo aggregate result
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """
    Full Monte Carlo result with per-trial records and percentile bands.

    Percentile naming convention used throughout:
      P50  → median (50th percentile)
      P90  → 90th percentile (90 % of trials ≤ this value)
      P99  → 99th percentile (near-worst-case)
      P10  → 10th percentile (used for NPV downside risk)

    For reliability metrics (ENS, downtime) higher is worse → P90/P99 are
    the risk bands.

    For NPV lower is worse → P10/P25 are the risk bands.
    """
    n_trials: int
    base_seed: int
    confidence_levels: list[float]

    # Per-trial records
    trials: list[MCTrialResult] = field(default_factory=list)

    # Numpy arrays of per-trial metrics (populated by engine)
    ens_trials: np.ndarray = field(default_factory=lambda: np.array([]))
    downtime_trials: np.ndarray = field(default_factory=lambda: np.array([]))
    continuity_trials: np.ndarray = field(default_factory=lambda: np.array([]))
    npv_trials: np.ndarray = field(default_factory=lambda: np.array([]))
    outage_cost_trials: np.ndarray = field(default_factory=lambda: np.array([]))
    lcoe_trials: np.ndarray = field(default_factory=lambda: np.array([]))

    def percentile(self, arr: np.ndarray, p: float) -> float:
        """Return p-th percentile (p in [0, 1])."""
        if len(arr) == 0:
            return float("nan")
        return float(np.percentile(arr, p * 100))

    # --- ENS bands ---
    @property
    def ens_p50(self) -> float:
        return self.percentile(self.ens_trials, 0.50)

    @property
    def ens_p90(self) -> float:
        return self.percentile(self.ens_trials, 0.90)

    @property
    def ens_p99(self) -> float:
        return self.percentile(self.ens_trials, 0.99)

    @property
    def ens_p10(self) -> float:
        return self.percentile(self.ens_trials, 0.10)

    # --- Downtime bands ---
    @property
    def downtime_p50(self) -> float:
        return self.percentile(self.downtime_trials, 0.50)

    @property
    def downtime_p90(self) -> float:
        return self.percentile(self.downtime_trials, 0.90)

    @property
    def downtime_p99(self) -> float:
        return self.percentile(self.downtime_trials, 0.99)

    # --- Continuity bands (higher is better → P10 is risk band) ---
    @property
    def continuity_p50(self) -> float:
        return self.percentile(self.continuity_trials, 0.50)

    @property
    def continuity_p10(self) -> float:
        return self.percentile(self.continuity_trials, 0.10)

    # --- NPV bands (lower is worse → P10 / P25 are risk bands) ---
    @property
    def npv_p50(self) -> float:
        return self.percentile(self.npv_trials, 0.50)

    @property
    def npv_p10(self) -> float:
        return self.percentile(self.npv_trials, 0.10)

    @property
    def npv_p90(self) -> float:
        return self.percentile(self.npv_trials, 0.90)

    # --- Outage cost bands ---
    @property
    def outage_cost_p50(self) -> float:
        return self.percentile(self.outage_cost_trials, 0.50)

    @property
    def outage_cost_p90(self) -> float:
        return self.percentile(self.outage_cost_trials, 0.90)

    @property
    def outage_cost_p99(self) -> float:
        return self.percentile(self.outage_cost_trials, 0.99)

    # --- LCOE bands ---
    @property
    def lcoe_p50(self) -> float:
        return self.percentile(self.lcoe_trials, 0.50)

    @property
    def lcoe_p90(self) -> float:
        return self.percentile(self.lcoe_trials, 0.90)

    def get_bands(self, metric: str) -> dict[str, float]:
        """
        Return a dict of percentile → value for the given metric.

        metric: 'ens', 'downtime', 'continuity', 'npv', 'outage_cost', 'lcoe'
        """
        arr_map = {
            "ens": self.ens_trials,
            "downtime": self.downtime_trials,
            "continuity": self.continuity_trials,
            "npv": self.npv_trials,
            "outage_cost": self.outage_cost_trials,
            "lcoe": self.lcoe_trials,
        }
        arr = arr_map.get(metric, np.array([]))
        return {
            f"P{int(p * 100)}": self.percentile(arr, p)
            for p in self.confidence_levels
        }

    def summary_dict(self) -> dict:
        """Return a flat summary dict suitable for JSON export."""
        return {
            "n_trials": self.n_trials,
            "base_seed": self.base_seed,
            "ens_kWh_yr": {
                "p10": round(self.ens_p10, 2),
                "p50": round(self.ens_p50, 2),
                "p90": round(self.ens_p90, 2),
                "p99": round(self.ens_p99, 2),
                "mean": round(float(self.ens_trials.mean()), 2) if len(self.ens_trials) else None,
            },
            "downtime_hrs_yr": {
                "p50": round(self.downtime_p50, 4),
                "p90": round(self.downtime_p90, 4),
                "p99": round(self.downtime_p99, 4),
                "mean": round(float(self.downtime_trials.mean()), 4) if len(self.downtime_trials) else None,
            },
            "continuity_pct": {
                "p10": round(self.continuity_p10, 4),
                "p50": round(self.continuity_p50, 4),
            },
            "npv_rs": {
                "p10": round(self.npv_p10, 0),
                "p50": round(self.npv_p50, 0),
                "p90": round(self.npv_p90, 0),
                "mean": round(float(self.npv_trials.mean()), 0) if len(self.npv_trials) else None,
            },
            "outage_cost_rs": {
                "p50": round(self.outage_cost_p50, 0),
                "p90": round(self.outage_cost_p90, 0),
                "p99": round(self.outage_cost_p99, 0),
            },
            "lcoe_rs_kwh": {
                "p50": round(self.lcoe_p50, 2),
                "p90": round(self.lcoe_p90, 2),
            },
        }


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Monte Carlo simulation engine.

    Parameters
    ----------
    config : ProjectConfig
        Base project configuration. Outage mode must be 'stochastic' for MC
        to produce meaningful variation (deterministic mode gives identical
        results every trial).
    n_trials : int
        Number of independent trials. 200 is a practical minimum for stable
        P99 estimates; 500 is recommended for bankable studies.
    base_seed : int
        Seed for the first trial. Subsequent seeds are spaced by PRIME steps.
    confidence_levels : list[float]
        Percentile levels to compute (e.g. [0.10, 0.50, 0.90, 0.99]).
    """

    def __init__(
        self,
        config: ProjectConfig,
        n_trials: int = 200,
        base_seed: int = 42,
        confidence_levels: Optional[list[float]] = None,
    ) -> None:
        self.config = config
        self.n_trials = n_trials
        self.base_seed = base_seed
        self.confidence_levels = confidence_levels or [0.10, 0.50, 0.90, 0.99]

    def run(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
        baseline_npv: Optional[float] = None,
    ) -> MonteCarloResult:
        """
        Execute all Monte Carlo trials.

        Parameters
        ----------
        progress_callback : callable | None
            Called with fractional progress (0.0 – 1.0) after each trial.
        baseline_npv : float | None
            If provided, included in finance engine for incremental NPV tracking.

        Returns
        -------
        MonteCarloResult
        """
        # Lazy import to avoid circular imports
        from energex.engine.runner import run_simulation

        mc = MonteCarloResult(
            n_trials=self.n_trials,
            base_seed=self.base_seed,
            confidence_levels=self.confidence_levels,
        )

        ens_list: list[float] = []
        downtime_list: list[float] = []
        continuity_list: list[float] = []
        npv_list: list[float] = []
        outage_cost_list: list[float] = []
        lcoe_list: list[float] = []

        for i in range(self.n_trials):
            trial_seed = self.base_seed + i * _SEED_PRIME

            try:
                result = run_simulation(
                    self.config,
                    seed=trial_seed,
                    baseline_npv=baseline_npv,
                )

                # Aggregate reliability across all simulation years
                all_sims = result.all_sim_results
                mean_ens = float(np.mean([s.ens_kwh for s in all_sims]))
                mean_dt = float(np.mean([s.downtime_hours for s in all_sims]))
                mean_cont = float(np.mean([s.continuity_pct for s in all_sims]))
                mean_events = float(np.mean([s.outage_events_total for s in all_sims]))

                trial_rec = MCTrialResult(
                    trial=i,
                    seed=trial_seed,
                    mean_ens_kwh_yr=mean_ens,
                    mean_downtime_hrs_yr=mean_dt,
                    mean_continuity_pct=mean_cont,
                    mean_outage_events_yr=mean_events,
                    npv=result.finance.npv,
                    total_outage_cost=result.finance.total_outage_cost,
                    lcoe_rs_kwh=result.finance.lcoe_rs_kwh,
                    total_capex=result.finance.total_capex,
                    y1_ens_kwh=result.sim_year1.ens_kwh,
                    y1_downtime_hrs=result.sim_year1.downtime_hours,
                    y1_continuity_pct=result.sim_year1.continuity_pct,
                )

                mc.trials.append(trial_rec)
                ens_list.append(mean_ens)
                downtime_list.append(mean_dt)
                continuity_list.append(mean_cont)
                npv_list.append(result.finance.npv)
                outage_cost_list.append(result.finance.total_outage_cost)
                lcoe_list.append(result.finance.lcoe_rs_kwh)

            except Exception:
                # Skip failed trials but still count progress
                pass

            if progress_callback is not None:
                progress_callback((i + 1) / self.n_trials)

        # Assemble numpy arrays
        mc.ens_trials = np.array(ens_list)
        mc.downtime_trials = np.array(downtime_list)
        mc.continuity_trials = np.array(continuity_list)
        mc.npv_trials = np.array(npv_list)
        mc.outage_cost_trials = np.array(outage_cost_list)
        mc.lcoe_trials = np.array(lcoe_list)

        return mc
