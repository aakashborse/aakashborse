"""
EnergeX — Automated Optimal Sizing Engine

Grid sweep over BESS / DG / Solar size combinations to find:
  1. The Pareto frontier in (lifecycle_cost, ENS) space
  2. The least-cost SLA-feasible configuration ("optimal")
  3. The minimum-ENS configuration regardless of cost

Design choices
--------------
* n_years_per_eval simulations (default 3) per candidate for speed
* Each evaluation uses seed = hash(bess, dg, solar) for reproducibility
* Pareto dominance: point A dominates B if A is cheaper AND has lower ENS
* SLA feasibility = sim.sla_pass for all evaluated years
* Capex proxy = BESS_kwh × capex_rs_kwh + DG_kW × DG_capex_factor + Solar_kWp × capex_rs_kw
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from energex.domain.schemas import (
    ProjectConfig,
    SizingBounds,
    BESSBackupModel,
    DGBackupModel,
    SolarModel,
)
from energex.engine.outage_generator import OutageGenerator
from energex.engine.dispatch_backup import (
    BackupDispatchEngine,
    build_hourly_loads,
    build_hourly_solar,
)
from energex.engine.finance_engine import FinanceEngine


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SizingPoint:
    """One evaluated candidate in the search space."""
    bess_kwh: float
    bess_kw: float
    dg_kw: float
    solar_kwp: float

    # Metrics (averages across n_years_per_eval)
    avg_ens_kwh: float = 0.0
    avg_downtime_hours: float = 0.0
    avg_continuity_pct: float = 0.0
    sla_pass: bool = False

    # Financials
    capex_rs: float = 0.0
    npv_rs: float = 0.0
    lcoe_rs_kwh: float = 0.0

    # Pareto frontier flag
    is_pareto: bool = False

    def to_dict(self) -> dict:
        return {
            "bess_kwh": self.bess_kwh,
            "bess_kw": self.bess_kw,
            "dg_kw": self.dg_kw,
            "solar_kwp": self.solar_kwp,
            "avg_ens_kwh": round(self.avg_ens_kwh, 2),
            "avg_downtime_hours": round(self.avg_downtime_hours, 3),
            "avg_continuity_pct": round(self.avg_continuity_pct, 4),
            "sla_pass": self.sla_pass,
            "capex_rs": round(self.capex_rs, 0),
            "npv_rs": round(self.npv_rs, 0),
            "lcoe_rs_kwh": round(self.lcoe_rs_kwh, 4),
            "is_pareto": self.is_pareto,
        }


@dataclass
class OptimizationResult:
    """Full optimizer output."""

    bounds: SizingBounds
    all_points: list[SizingPoint] = field(default_factory=list)
    pareto_points: list[SizingPoint] = field(default_factory=list)

    # Best configurations
    optimal_point: Optional[SizingPoint] = None      # least-cost SLA-feasible
    min_ens_point: Optional[SizingPoint] = None       # lowest ENS regardless of cost
    min_capex_point: Optional[SizingPoint] = None     # cheapest feasible

    # Metadata
    n_candidates_evaluated: int = 0
    n_sla_feasible: int = 0

    def to_dict(self) -> dict:
        return {
            "n_candidates_evaluated": self.n_candidates_evaluated,
            "n_sla_feasible": self.n_sla_feasible,
            "n_pareto_points": len(self.pareto_points),
            "optimal": self.optimal_point.to_dict() if self.optimal_point else None,
            "min_ens": self.min_ens_point.to_dict() if self.min_ens_point else None,
            "min_capex": self.min_capex_point.to_dict() if self.min_capex_point else None,
            "pareto_frontier": [p.to_dict() for p in self.pareto_points],
        }


# ---------------------------------------------------------------------------
# Sizing Optimizer
# ---------------------------------------------------------------------------

class SizingOptimizer:
    """
    Grid sweep optimizer for BESS / DG / Solar sizing.

    Parameters
    ----------
    base_config : ProjectConfig
        Template configuration. BESS/DG/Solar sizes will be overwritten.
    bounds : SizingBounds
        Search space definition (or use base_config.sizing_bounds).
    base_seed : int
        Base seed for reproducible evaluation (actual seed = hash(bess_kwh, dg_kw, solar_kw)).
    """

    # Default DG capex per kW (₹/kW) if not in base_config
    _DEFAULT_DG_CAPEX_RS_KW = 30_000.0
    # Default Solar capex per kWp (₹/kWp) if not in base_config
    _DEFAULT_SOLAR_CAPEX_RS_KW = 45_000.0

    def __init__(
        self,
        base_config: ProjectConfig,
        bounds: Optional[SizingBounds] = None,
        base_seed: int = 42,
    ) -> None:
        self.base_config = base_config
        self.bounds = bounds or base_config.sizing_bounds or SizingBounds()
        self.base_seed = base_seed

        # Pre-build load arrays (same for all candidates)
        self._total_kw, self._critical_kw = build_hourly_loads(base_config, year=0)

    def run(self) -> OptimizationResult:
        """Execute grid sweep. Returns OptimizationResult."""
        b = self.bounds
        result = OptimizationResult(bounds=b)

        # Build candidate grid
        bess_sizes = self._arange(b.bess_capacity_min_kwh, b.bess_capacity_max_kwh, b.bess_capacity_step_kwh)
        dg_sizes = self._arange(b.dg_kw_min, b.dg_kw_max, b.dg_kw_step) if b.include_dg else [0.0]
        solar_sizes = self._arange(b.solar_kw_min, b.solar_kw_max, b.solar_kw_step) if b.include_solar else [0.0]

        # Ensure at least zero included
        if 0.0 not in bess_sizes:
            bess_sizes = [0.0] + list(bess_sizes)

        points: list[SizingPoint] = []

        for bess_kwh in bess_sizes:
            for dg_kw in dg_sizes:
                for solar_kwp in solar_sizes:
                    bess_kw = bess_kwh * b.bess_power_to_capacity_ratio if bess_kwh > 0 else 0.0
                    pt = self._evaluate(bess_kwh, bess_kw, dg_kw, solar_kwp)
                    points.append(pt)

        result.all_points = points
        result.n_candidates_evaluated = len(points)
        result.n_sla_feasible = sum(1 for p in points if p.sla_pass)

        # Pareto frontier
        result.pareto_points = self._extract_pareto(points)
        for p in result.pareto_points:
            p.is_pareto = True

        # Best configurations
        feasible = [p for p in points if p.sla_pass]
        if feasible:
            result.optimal_point = min(feasible, key=lambda p: p.capex_rs)
            result.min_capex_point = result.optimal_point
        if points:
            result.min_ens_point = min(points, key=lambda p: p.avg_ens_kwh)

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        bess_kwh: float,
        bess_kw: float,
        dg_kw: float,
        solar_kwp: float,
    ) -> SizingPoint:
        """Evaluate one candidate configuration."""
        cfg = self._build_config(bess_kwh, bess_kw, dg_kw, solar_kwp)
        seed = self._candidate_seed(bess_kwh, dg_kw, solar_kwp)

        ens_vals, dt_vals, cont_vals = [], [], []
        sla_pass_all = True
        sim_results = []

        for yr in range(self.bounds.n_years_per_eval):
            yr_seed = seed + yr * 7919
            gen = OutageGenerator(cfg.outage, seed=yr_seed)
            events = gen.generate()

            solar_kw = build_hourly_solar(cfg, year=yr)

            engine = BackupDispatchEngine(
                config=cfg,
                outage_events=events,
                year=yr,
                seed=yr_seed,
                total_kw=self._total_kw.copy(),
                critical_kw=self._critical_kw.copy(),
                solar_kw=solar_kw,
            )
            sim = engine.run()
            sim_results.append(sim)

            ens_vals.append(sim.ens_kwh)
            dt_vals.append(sim.downtime_hours)
            cont_vals.append(sim.continuity_pct)
            if not sim.sla_pass:
                sla_pass_all = False

        # Finance (lightweight — use just the evaluated years repeated)
        full_sims = [sim_results[i % len(sim_results)] for i in range(cfg.project_lifetime_years)]
        fe = FinanceEngine(cfg, full_sims, scenario_name="sizing_eval")
        fr = fe.run()

        capex = self._compute_capex(bess_kwh, bess_kw, dg_kw, solar_kwp)

        return SizingPoint(
            bess_kwh=bess_kwh,
            bess_kw=bess_kw,
            dg_kw=dg_kw,
            solar_kwp=solar_kwp,
            avg_ens_kwh=float(np.mean(ens_vals)),
            avg_downtime_hours=float(np.mean(dt_vals)),
            avg_continuity_pct=float(np.mean(cont_vals)),
            sla_pass=sla_pass_all,
            capex_rs=capex,
            npv_rs=fr.npv,
            lcoe_rs_kwh=fr.lcoe_rs_kwh,
        )

    def _build_config(
        self,
        bess_kwh: float,
        bess_kw: float,
        dg_kw: float,
        solar_kwp: float,
    ) -> ProjectConfig:
        """Deep-copy base config and override asset sizes."""
        cfg = copy.deepcopy(self.base_config)

        # BESS
        if bess_kwh > 0:
            if cfg.bess is not None:
                cfg.bess.capacity_kwh = bess_kwh
                cfg.bess.power_kw = max(bess_kw, 1.0)
            else:
                # Create a minimal BESS using defaults from base BESS capex
                capex_rs_kwh = 25_000.0  # default ₹/kWh if no BESS template
                cfg.bess = BESSBackupModel(
                    capacity_kwh=bess_kwh,
                    power_kw=max(bess_kw, 1.0),
                    capex_rs_kwh=capex_rs_kwh,
                )
        else:
            cfg.bess = None

        # DG
        if dg_kw > 0 and self.bounds.include_dg:
            if cfg.dg is not None:
                cfg.dg.rated_kw = dg_kw
            else:
                cfg.dg = DGBackupModel(
                    rated_kw=dg_kw,
                    fuel_l_per_kwh=0.35,
                    diesel_price_rs_l=90.0,
                    capex_rs=dg_kw * self._DEFAULT_DG_CAPEX_RS_KW,
                )
        else:
            cfg.dg = None

        # Solar
        if solar_kwp > 0 and self.bounds.include_solar:
            if cfg.solar is not None:
                cfg.solar.size_kw = solar_kwp
            else:
                cfg.solar = SolarModel(
                    size_kw=solar_kwp,
                    yield_kwh_kw_day=4.5,
                    capex_rs_kw=self._DEFAULT_SOLAR_CAPEX_RS_KW,
                )
        else:
            cfg.solar = None

        # Re-derive scenario by setting to None and re-running the validator logic
        cfg.scenario = None
        has_dg = cfg.dg is not None
        has_bess = cfg.bess is not None
        has_solar = cfg.solar is not None
        from energex.domain.schemas import ScenarioType
        if has_dg and has_bess and has_solar:
            cfg.scenario = ScenarioType.GRID_SOLAR_DG_BESS
        elif has_dg and has_bess:
            cfg.scenario = ScenarioType.GRID_DG_BESS
        elif has_solar and has_bess:
            cfg.scenario = ScenarioType.GRID_SOLAR_BESS
        elif has_bess:
            cfg.scenario = ScenarioType.GRID_BESS
        elif has_dg:
            cfg.scenario = ScenarioType.GRID_DG
        else:
            cfg.scenario = ScenarioType.GRID_ONLY

        return cfg

    def _compute_capex(
        self,
        bess_kwh: float,
        bess_kw: float,
        dg_kw: float,
        solar_kwp: float,
    ) -> float:
        """Compute total capex for a candidate."""
        capex = 0.0
        cfg = self.base_config

        if bess_kwh > 0:
            rate = cfg.bess.capex_rs_kwh if cfg.bess else 25_000.0
            capex += bess_kwh * rate

        if dg_kw > 0 and self.bounds.include_dg:
            rate = cfg.dg.capex_rs / cfg.dg.rated_kw if (cfg.dg and cfg.dg.rated_kw > 0) else self._DEFAULT_DG_CAPEX_RS_KW
            capex += dg_kw * rate

        if solar_kwp > 0 and self.bounds.include_solar:
            rate = cfg.solar.capex_rs_kw if cfg.solar else self._DEFAULT_SOLAR_CAPEX_RS_KW
            capex += solar_kwp * rate

        return capex

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pareto(points: list[SizingPoint]) -> list[SizingPoint]:
        """
        Extract non-dominated Pareto frontier in (capex, avg_ens_kwh) space.

        Point A dominates B if A.capex <= B.capex AND A.ens <= B.ens with at least one strict.
        """
        pareto: list[SizingPoint] = []
        for candidate in points:
            dominated = False
            for other in points:
                if other is candidate:
                    continue
                if (
                    other.capex_rs <= candidate.capex_rs
                    and other.avg_ens_kwh <= candidate.avg_ens_kwh
                    and (
                        other.capex_rs < candidate.capex_rs
                        or other.avg_ens_kwh < candidate.avg_ens_kwh
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        # Sort by capex ascending
        return sorted(pareto, key=lambda p: p.capex_rs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _arange(lo: float, hi: float, step: float) -> list[float]:
        """Generate inclusive range from lo to hi with given step."""
        vals = []
        v = lo
        while v <= hi + 1e-9:
            vals.append(round(v, 6))
            v += step
        return vals

    @staticmethod
    def _candidate_seed(bess_kwh: float, dg_kw: float, solar_kwp: float) -> int:
        """Deterministic seed from candidate dimensions."""
        return abs(hash((round(bess_kwh, 1), round(dg_kw, 1), round(solar_kwp, 1)))) % (2**31)
