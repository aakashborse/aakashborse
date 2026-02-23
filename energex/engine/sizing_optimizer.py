"""
EnergeX — Automated Sizing Optimizer (Phase C)

Answers the question every client actually asks: "What should I buy?"

The optimizer sweeps a user-defined grid of BESS capacity × DG rated-power
combinations, runs the full simulation for each, and returns a ranked list
of candidates with NPV, LCOE, CAPEX, payback, and SLA pass/fail.

Algorithm
---------
1. Build a search grid: N_bess × N_dg combinations.
2. Skip combinations that violate hard CAPEX ceilings (fast prune).
3. For each surviving combination:
   a. Clone the base config with the new BESS/DG sizes.
   b. Run run_simulation() (full 25-year dispatch + finance).
   c. Record reliability and financial metrics.
4. Rank by the chosen objective (incremental NPV by default).
   SLA-passing candidates are placed above SLA-failing ones.

Design notes
------------
- BESS power (kW) is derived from capacity via bess_power_ratio (default 0.5 = C/2).
- If the base config has a BESS/DG template, its parameters (capex, efficiency,
  fuel price, etc.) are reused; only the size is changed.
- If the base config has no BESS template and a non-zero BESS size is requested,
  a sensible Indian-market default BESS config is injected.
- Same logic for DG.

Usage
-----
    from energex.engine.sizing_optimizer import SizingOptimizer

    opt = SizingOptimizer(
        config=cfg,
        bess_sizes_kwh=[0, 100, 200, 300],
        dg_sizes_kw=[0, 100, 200],
        optimize_for="incremental_npv",
    )
    candidates = opt.run(progress_callback=lambda p: print(f"{p*100:.0f}%"))
    best = candidates[0]
    print(f"Recommended: BESS {best.bess_kwh} kWh + DG {best.dg_kw} kW")
    print(f"  NPV ₹{best.npv:,.0f}  |  Payback {best.payback_years:.1f} yr")
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from energex.domain.schemas import (
    ProjectConfig,
    BESSBackupModel,
    DGBackupModel,
    SizingObjective,
)


# ---------------------------------------------------------------------------
# Default templates (used when base config has no BESS/DG)
# ---------------------------------------------------------------------------

_DEFAULT_BESS_CAPEX_RS_KWH = 35_000.0       # ₹35k/kWh (typical LFP 2025)
_DEFAULT_BESS_OM_RS_KWH_YR = 500.0
_DEFAULT_DG_CAPEX_RS_KW = 8_000.0           # ₹8k/kW installed
_DEFAULT_DG_FUEL_L_PER_KWH = 0.27
_DEFAULT_DG_PRICE_RS_L = 95.0


# ---------------------------------------------------------------------------
# Sizing candidate
# ---------------------------------------------------------------------------

@dataclass
class SizingCandidate:
    """
    One evaluated BESS + DG sizing combination.

    Attributes
    ----------
    bess_kwh : float
        BESS energy capacity (kWh). 0 = no BESS.
    bess_kw : float
        BESS power rating (kW).
    dg_kw : float
        DG rated power (kW). 0 = no DG.
    ens_kwh_yr : float
        Mean annual Energy Not Served (kWh/yr) — Year 1.
    downtime_hrs_yr : float
        Mean annual downtime (hrs/yr) — Year 1.
    continuity_pct : float
        Power continuity (%) — Year 1.
    backup_autonomy_hrs : float
        Backup autonomy at peak critical load (hours).
    npv : float
        Project NPV over full lifetime (₹).
    incremental_npv : float | None
        NPV improvement vs grid-only baseline (₹). Positive = investment worthwhile.
    lcoe_rs_kwh : float
        Levelized Cost of Energy (₹/kWh).
    capex_rs : float
        Total capital expenditure (₹).
    payback_years : float | None
        Simple payback period (years).
    sla_pass : bool
        Whether SLA targets are met.
    sla_failures : list[str]
        List of SLA failure messages.
    label : str
        Human-readable description, e.g. "BESS 200 kWh + DG 100 kW".
    """
    bess_kwh: float
    bess_kw: float
    dg_kw: float

    ens_kwh_yr: float
    downtime_hrs_yr: float
    continuity_pct: float
    backup_autonomy_hrs: float

    npv: float
    incremental_npv: Optional[float]
    lcoe_rs_kwh: float
    capex_rs: float
    payback_years: Optional[float]

    sla_pass: bool
    sla_failures: list[str] = field(default_factory=list)

    # Populated after ranking
    rank: int = 0

    @property
    def label(self) -> str:
        parts = []
        if self.bess_kwh > 0:
            parts.append(f"BESS {self.bess_kwh:.0f} kWh / {self.bess_kw:.0f} kW")
        if self.dg_kw > 0:
            parts.append(f"DG {self.dg_kw:.0f} kW")
        return " + ".join(parts) if parts else "Grid Only (no backup)"

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "label": self.label,
            "bess_kwh": self.bess_kwh,
            "bess_kw": self.bess_kw,
            "dg_kw": self.dg_kw,
            "capex_rs": round(self.capex_rs, 0),
            "npv_rs": round(self.npv, 0),
            "incremental_npv_rs": round(self.incremental_npv, 0) if self.incremental_npv is not None else None,
            "lcoe_rs_kwh": round(self.lcoe_rs_kwh, 2),
            "payback_years": round(self.payback_years, 2) if self.payback_years else None,
            "ens_kwh_yr": round(self.ens_kwh_yr, 2),
            "downtime_hrs_yr": round(self.downtime_hrs_yr, 4),
            "continuity_pct": round(self.continuity_pct, 4),
            "backup_autonomy_hrs": round(self.backup_autonomy_hrs, 2),
            "sla_pass": self.sla_pass,
            "sla_failures": self.sla_failures,
        }


# ---------------------------------------------------------------------------
# Sizing Optimizer
# ---------------------------------------------------------------------------

class SizingOptimizer:
    """
    Grid-search optimal BESS + DG sizing.

    Parameters
    ----------
    config : ProjectConfig
        Base project configuration used as template.
    bess_sizes_kwh : list[float]
        BESS energy capacity candidates (kWh). Include 0 for no-BESS option.
    dg_sizes_kw : list[float]
        DG rated power candidates (kW). Include 0 for no-DG option.
    bess_power_ratio : float
        BESS power (kW) = capacity_kwh × ratio. Default 0.5 (C/2 = 2-hour battery).
    optimize_for : str
        Ranking objective: 'incremental_npv', 'lcoe', 'capex', 'ens'.
    max_capex_rs : float | None
        Optional hard CAPEX ceiling (₹). Combinations above this are skipped.
    require_sla_pass : bool
        If True, SLA-passing candidates are ranked above SLA-failing ones.
    seed : int
        Random seed passed to run_simulation().
    """

    DEFAULT_BESS_SIZES_KWH = [0, 50, 100, 150, 200, 300, 400, 500]
    DEFAULT_DG_SIZES_KW = [0, 50, 100, 150, 200]

    def __init__(
        self,
        config: ProjectConfig,
        bess_sizes_kwh: Optional[list[float]] = None,
        dg_sizes_kw: Optional[list[float]] = None,
        bess_power_ratio: float = 0.5,
        optimize_for: str = "incremental_npv",
        max_capex_rs: Optional[float] = None,
        require_sla_pass: bool = True,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.bess_sizes = sorted(set(bess_sizes_kwh or self.DEFAULT_BESS_SIZES_KWH))
        self.dg_sizes = sorted(set(dg_sizes_kw or self.DEFAULT_DG_SIZES_KW))
        self.bess_power_ratio = bess_power_ratio
        self.optimize_for = optimize_for
        self.max_capex_rs = max_capex_rs
        self.require_sla_pass = require_sla_pass
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[SizingCandidate]:
        """
        Run the full sizing grid search.

        Returns
        -------
        list[SizingCandidate]
            All evaluated candidates, ranked by objective. Best candidate is [0].
        """
        from energex.engine.runner import run_simulation, run_grid_only_baseline

        # Compute baseline NPV once
        try:
            bl_finance = run_grid_only_baseline(self.config, seed=self.seed)
            baseline_npv = bl_finance.npv
        except Exception:
            baseline_npv = None

        candidates: list[SizingCandidate] = []
        total = len(self.bess_sizes) * len(self.dg_sizes)
        done = 0

        for bess_kwh in self.bess_sizes:
            for dg_kw in self.dg_sizes:

                # Fast CAPEX prune
                estimated_capex = self._estimate_capex(bess_kwh, dg_kw)
                if self.max_capex_rs is not None and estimated_capex > self.max_capex_rs:
                    done += 1
                    if progress_callback:
                        progress_callback(done / total)
                    continue

                try:
                    cfg = self._build_config(bess_kwh, dg_kw)
                    result = run_simulation(cfg, seed=self.seed, baseline_npv=baseline_npv)
                    sim = result.sim_year1
                    fin = result.finance

                    candidate = SizingCandidate(
                        bess_kwh=bess_kwh,
                        bess_kw=bess_kwh * self.bess_power_ratio if bess_kwh > 0 else 0.0,
                        dg_kw=dg_kw,
                        ens_kwh_yr=sim.ens_kwh,
                        downtime_hrs_yr=sim.downtime_hours,
                        continuity_pct=sim.continuity_pct,
                        backup_autonomy_hrs=sim.backup_autonomy_hours,
                        npv=fin.npv,
                        incremental_npv=fin.incremental_npv,
                        lcoe_rs_kwh=fin.lcoe_rs_kwh,
                        capex_rs=fin.total_capex,
                        payback_years=fin.simple_payback_years,
                        sla_pass=sim.sla_pass,
                        sla_failures=list(sim.sla_failures),
                    )
                    candidates.append(candidate)

                except Exception:
                    # Skip failed combinations silently
                    pass

                done += 1
                if progress_callback:
                    progress_callback(done / total)

        ranked = self._rank(candidates)
        for i, c in enumerate(ranked):
            c.rank = i + 1

        return ranked

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_config(self, bess_kwh: float, dg_kw: float) -> ProjectConfig:
        """
        Clone the base config and inject the specified BESS/DG sizes.

        - If bess_kwh == 0, BESS is removed.
        - If dg_kw == 0, DG is removed.
        - If the base config has an existing BESS/DG template, its parameters
          are reused (only capacity/rated_kw is changed).
        - If no template exists, sensible Indian-market defaults are injected.
        """
        cfg = copy.deepcopy(self.config)

        # --- BESS ---
        if bess_kwh <= 0:
            cfg.bess = None
        else:
            bess_kw = bess_kwh * self.bess_power_ratio
            if cfg.bess is not None:
                # Reuse existing template parameters
                cfg.bess.capacity_kwh = bess_kwh
                cfg.bess.power_kw = bess_kw
            else:
                # Inject default BESS
                cfg.bess = BESSBackupModel(
                    capacity_kwh=bess_kwh,
                    power_kw=bess_kw,
                    dod_pct=90.0,
                    roundtrip_efficiency=0.92,
                    min_reserve_soc_pct=10.0,
                    initial_soc_pct=90.0,
                    calendar_fade_pct_yr=2.0,
                    cycle_fade_per_kwh=0.000025,
                    calendar_life_yr=12.0,
                    eol_capacity_pct=80.0,
                    capex_rs_kwh=_DEFAULT_BESS_CAPEX_RS_KWH,
                    replacement_cost_fraction=0.6,
                    om_rs_kwh_yr=_DEFAULT_BESS_OM_RS_KWH_YR,
                    availability_pct=99.0,
                )

        # --- DG ---
        if dg_kw <= 0:
            cfg.dg = None
        else:
            if cfg.dg is not None:
                cfg.dg.rated_kw = dg_kw
                cfg.dg.capex_rs = dg_kw * _DEFAULT_DG_CAPEX_RS_KW
            else:
                # Inject default DG
                cfg.dg = DGBackupModel(
                    rated_kw=dg_kw,
                    min_loading_pct=30.0,
                    start_delay_seconds=10.0,
                    ramp_rate_kw_s=10.0,
                    fuel_l_per_kwh=_DEFAULT_DG_FUEL_L_PER_KWH,
                    diesel_price_rs_l=_DEFAULT_DG_PRICE_RS_L,
                    diesel_escalation_pct_yr=5.0,
                    om_rs_kwh=1.5,
                    om_rs_hour=0.0,
                    availability_pct=95.0,
                    maintenance_hours_yr=200.0,
                    capex_rs=dg_kw * _DEFAULT_DG_CAPEX_RS_KW,
                )

        # Re-derive scenario
        cfg.scenario = None
        return cfg

    def _estimate_capex(self, bess_kwh: float, dg_kw: float) -> float:
        """Quick CAPEX estimate for pre-filtering (no simulation needed)."""
        total = 0.0
        if bess_kwh > 0:
            if self.config.bess:
                total += bess_kwh * self.config.bess.capex_rs_kwh
            else:
                total += bess_kwh * _DEFAULT_BESS_CAPEX_RS_KWH
        if dg_kw > 0:
            if self.config.dg:
                total += dg_kw * (self.config.dg.capex_rs / max(self.config.dg.rated_kw, 1))
            else:
                total += dg_kw * _DEFAULT_DG_CAPEX_RS_KW
        if self.config.solar:
            total += self.config.solar.capex_rs
        return total

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(self, candidates: list[SizingCandidate]) -> list[SizingCandidate]:
        """Rank candidates by objective. SLA-passing candidates ranked first."""
        if not candidates:
            return []

        if self.require_sla_pass:
            sla_pass = [c for c in candidates if c.sla_pass]
            sla_fail = [c for c in candidates if not c.sla_pass]
        else:
            sla_pass = candidates
            sla_fail = []

        def sort_key(c: SizingCandidate):
            if self.optimize_for == "incremental_npv":
                # Highest incremental NPV first; fall back to NPV if no baseline
                v = c.incremental_npv if c.incremental_npv is not None else c.npv
                return -v  # negate for ascending sort
            elif self.optimize_for == "lcoe":
                return c.lcoe_rs_kwh   # lowest first
            elif self.optimize_for == "capex":
                return c.capex_rs      # lowest first
            elif self.optimize_for == "ens":
                return c.ens_kwh_yr    # lowest first
            return 0.0

        sla_pass.sort(key=sort_key)
        sla_fail.sort(key=sort_key)

        return sla_pass + sla_fail

    # ------------------------------------------------------------------
    # Recommendation narrative
    # ------------------------------------------------------------------

    @staticmethod
    def recommendation_text(candidates: list[SizingCandidate]) -> str:
        """
        Generate a plain-English recommendation summary for the top candidate.
        """
        if not candidates:
            return "No viable configurations found."

        best = candidates[0]

        lines = [
            f"RECOMMENDED: {best.label}",
            "",
            f"  CAPEX         : ₹{best.capex_rs/1e6:.2f}M",
            f"  Incremental NPV: ₹{best.incremental_npv/1e6:.2f}M" if best.incremental_npv is not None else "",
            f"  Payback        : {best.payback_years:.1f} yr" if best.payback_years else "  Payback        : N/A",
            f"  LCOE           : ₹{best.lcoe_rs_kwh:.2f}/kWh",
            f"  ENS            : {best.ens_kwh_yr:.1f} kWh/yr",
            f"  Downtime       : {best.downtime_hrs_yr:.2f} hrs/yr",
            f"  Continuity     : {best.continuity_pct:.3f}%",
            f"  Backup autonomy: {best.backup_autonomy_hrs:.1f} hrs",
            f"  SLA            : {'PASS' if best.sla_pass else 'FAIL'}",
        ]
        lines = [l for l in lines if l.strip()]  # remove blank entries

        if len(candidates) > 1:
            lines.append("")
            lines.append("Alternatives (ranked):")
            for c in candidates[1:min(5, len(candidates))]:
                npv_str = (
                    f"ΔNPV ₹{c.incremental_npv/1e6:+.2f}M"
                    if c.incremental_npv is not None
                    else f"NPV ₹{c.npv/1e6:.2f}M"
                )
                pb_str = f"{c.payback_years:.1f} yr" if c.payback_years else "N/A"
                lines.append(
                    f"  #{c.rank}  {c.label:<40s}  CAPEX ₹{c.capex_rs/1e6:.2f}M  "
                    f"{npv_str}  PB {pb_str}  {'✓' if c.sla_pass else '✗'}"
                )

        return "\n".join(lines)
