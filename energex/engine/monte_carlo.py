"""
EnergeX — Monte Carlo Uncertainty Engine

Runs N independent simulations with different random seeds to produce
statistically defensible confidence intervals (P10/P25/P50/P75/P90/P99)
for reliability and financial metrics.

Design choices
--------------
* Year-1 only simulation per run (speed; captures stochastic outage variability)
* NPV proxy = deterministic_base_npv + stochastic_outage_cost × annuity_factor
* Each run gets seed = base_seed + run_index (reproducible)
* SLA pass-rate reported as fraction of runs where all SLA targets are met
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from energex.domain.schemas import ProjectConfig, MonteCarloConfig
from energex.engine.outage_generator import OutageGenerator
from energex.engine.dispatch_backup import (
    BackupDispatchEngine,
    SimulationResult,
    build_hourly_loads,
    build_hourly_solar,
)
from energex.engine.finance_engine import FinanceEngine


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloStats:
    """Percentile statistics for a single metric across all MC runs."""
    metric: str
    unit: str
    mean: float
    std: float
    min: float
    max: float
    percentiles: dict[float, float] = field(default_factory=dict)
    # e.g. {10: val, 25: val, 50: val, 75: val, 90: val, 99: val}

    @property
    def p50(self) -> float:
        return self.percentiles.get(50.0, self.mean)

    @property
    def p90(self) -> float:
        return self.percentiles.get(90.0, self.max)

    @property
    def p99(self) -> float:
        return self.percentiles.get(99.0, self.max)

    @property
    def p10(self) -> float:
        return self.percentiles.get(10.0, self.min)


@dataclass
class MonteCarloResult:
    """Full Monte Carlo analysis result."""

    n_runs: int
    base_seed: int
    percentiles_requested: list[float]

    # Raw arrays (length = n_runs)
    ens_kwh_runs: list[float] = field(default_factory=list)
    downtime_hours_runs: list[float] = field(default_factory=list)
    continuity_pct_runs: list[float] = field(default_factory=list)
    npv_proxy_runs: list[float] = field(default_factory=list)
    sla_pass_runs: list[bool] = field(default_factory=list)

    # Computed statistics (one per metric)
    ens_stats: Optional[MonteCarloStats] = None
    downtime_stats: Optional[MonteCarloStats] = None
    continuity_stats: Optional[MonteCarloStats] = None
    npv_stats: Optional[MonteCarloStats] = None

    # Summary
    sla_pass_rate_pct: float = 0.0
    total_outage_cost_p50: float = 0.0

    def to_dict(self) -> dict:
        """Serializable summary for export."""
        return {
            "n_runs": self.n_runs,
            "base_seed": self.base_seed,
            "sla_pass_rate_pct": round(self.sla_pass_rate_pct, 2),
            "ens_kwh": self._stats_dict(self.ens_stats),
            "downtime_hours": self._stats_dict(self.downtime_stats),
            "continuity_pct": self._stats_dict(self.continuity_stats),
            "npv_proxy_rs": self._stats_dict(self.npv_stats),
        }

    @staticmethod
    def _stats_dict(s: Optional[MonteCarloStats]) -> dict:
        if s is None:
            return {}
        return {
            "mean": round(s.mean, 3),
            "std": round(s.std, 3),
            "min": round(s.min, 3),
            "max": round(s.max, 3),
            **{f"p{int(p)}": round(v, 3) for p, v in s.percentiles.items()},
        }


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Runs N stochastic simulations and computes reliability + financial
    confidence intervals.

    Parameters
    ----------
    config : ProjectConfig
    mc_config : MonteCarloConfig  (or uses config.monte_carlo if None)
    baseline_npv : float | None   deterministic baseline for incremental NPV
    """

    def __init__(
        self,
        config: ProjectConfig,
        mc_config: Optional[MonteCarloConfig] = None,
        baseline_npv: Optional[float] = None,
    ) -> None:
        self.config = config
        self.mc_cfg = mc_config or config.monte_carlo or MonteCarloConfig()
        self.baseline_npv = baseline_npv

        # Pre-build load and solar arrays (same for all runs; only outages vary)
        self.total_kw, self.critical_kw = build_hourly_loads(config, year=0)
        self.solar_kw = build_hourly_solar(config, year=0)

        # Annuity factor: PV of 1 ₹/yr for n_years at discount rate
        dr = config.discount_rate_pct / 100.0
        n = config.project_lifetime_years
        self._annuity_factor = (1 - (1 + dr) ** (-n)) / dr if dr > 0 else float(n)

    def run(self) -> MonteCarloResult:
        """Execute all Monte Carlo runs. Returns MonteCarloResult."""
        cfg = self.mc_cfg
        result = MonteCarloResult(
            n_runs=cfg.n_runs,
            base_seed=cfg.base_seed,
            percentiles_requested=cfg.percentiles,
        )

        ens_list: list[float] = []
        dt_list: list[float] = []
        cont_list: list[float] = []
        npv_list: list[float] = []
        sla_list: list[bool] = []

        for i in range(cfg.n_runs):
            seed = cfg.base_seed + i
            sim = self._run_single(seed)

            ens_list.append(sim.ens_kwh)
            dt_list.append(sim.downtime_hours)
            cont_list.append(sim.continuity_pct)
            sla_list.append(sim.sla_pass)

            # NPV proxy: outage cost savings dominate stochastic variability
            npv_proxy = self._compute_npv_proxy(sim)
            npv_list.append(npv_proxy)

        # Store raw runs
        result.ens_kwh_runs = ens_list
        result.downtime_hours_runs = dt_list
        result.continuity_pct_runs = cont_list
        result.npv_proxy_runs = npv_list
        result.sla_pass_runs = sla_list

        # Compute statistics
        pcts = cfg.percentiles
        result.ens_stats = self._compute_stats(ens_list, "ens_kwh", "kWh/yr", pcts)
        result.downtime_stats = self._compute_stats(dt_list, "downtime_hours", "hrs/yr", pcts)
        result.continuity_stats = self._compute_stats(cont_list, "continuity_pct", "%", pcts)
        result.npv_stats = self._compute_stats(npv_list, "npv_proxy", "₹", pcts)

        result.sla_pass_rate_pct = 100.0 * sum(sla_list) / len(sla_list) if sla_list else 0.0

        # P50 outage cost (₹/yr × annuity factor)
        p50_ens = result.ens_stats.p50
        oc = self.config.outage_cost
        from energex.domain.schemas import OutageCostMode
        if oc.mode == OutageCostMode.VOLL and oc.voll_rs_kwh:
            result.total_outage_cost_p50 = p50_ens * oc.voll_rs_kwh * self._annuity_factor
        elif oc.mode == OutageCostMode.DOWNTIME and oc.downtime_cost_rs_hr:
            p50_dt = result.downtime_stats.p50
            result.total_outage_cost_p50 = p50_dt * oc.downtime_cost_rs_hr * self._annuity_factor

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single(self, seed: int) -> SimulationResult:
        """Run a single year-1 simulation with given seed."""
        gen = OutageGenerator(self.config.outage, seed=seed)
        events = gen.generate()

        engine = BackupDispatchEngine(
            config=self.config,
            outage_events=events,
            year=0,
            seed=seed,
            total_kw=self.total_kw.copy(),
            critical_kw=self.critical_kw.copy(),
            solar_kw=self.solar_kw.copy(),
        )
        return engine.run()

    def _compute_npv_proxy(self, sim: SimulationResult) -> float:
        """
        Compute an NPV proxy that captures stochastic variability.

        NPV_proxy ≈ -CAPEX + (annual_savings - outage_cost_this_run) × annuity
        where annual_savings = (grid_energy_cost_baseline - grid_energy_cost_scenario)
        """
        oc = self.config.outage_cost
        from energex.domain.schemas import OutageCostMode

        # Stochastic outage cost for this run (₹/yr)
        if oc.mode == OutageCostMode.VOLL and oc.voll_rs_kwh:
            annual_outage_cost = sim.ens_kwh * oc.voll_rs_kwh
        elif oc.mode == OutageCostMode.DOWNTIME and oc.downtime_cost_rs_hr:
            annual_outage_cost = sim.downtime_hours * oc.downtime_cost_rs_hr
        else:
            annual_outage_cost = 0.0

        # CAPEX
        capex = 0.0
        if self.config.bess:
            capex += self.config.bess.capex_rs
        if self.config.dg:
            capex += self.config.dg.capex_rs
        if self.config.solar:
            capex += self.config.solar.capex_rs

        # Annual grid cost
        grid_rate = self.config.grid.energy_rate_rs_kwh
        annual_grid_cost = sim.total_grid_kwh * grid_rate

        # Approximate NPV proxy
        annual_net = -annual_grid_cost - annual_outage_cost
        npv = -capex + annual_net * self._annuity_factor

        if self.baseline_npv is not None:
            return npv - self.baseline_npv
        return npv

    @staticmethod
    def _compute_stats(
        values: list[float],
        metric: str,
        unit: str,
        percentiles: list[float],
    ) -> MonteCarloStats:
        arr = np.array(values, dtype=float)
        pct_dict = {p: float(np.percentile(arr, p)) for p in percentiles}
        return MonteCarloStats(
            metric=metric,
            unit=unit,
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=float(arr.max()),
            percentiles=pct_dict,
        )
