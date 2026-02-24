"""
EnergeX — Finance Engine

Computes project cashflows, NPV, IRR, payback, and LCOE.
Reliability-aware: outage costs are a first-class cashflow component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from energex.domain.schemas import (
    ProjectConfig,
    OutageCostMode,
)
from energex.engine.dispatch_backup import SimulationResult


# ---------------------------------------------------------------------------
# Year-level cashflow
# ---------------------------------------------------------------------------

@dataclass
class YearCashflow:
    """Cashflow breakdown for one project year."""

    year: int

    # Capital
    capex: float = 0.0              # negative at year 0 (investment)
    replacement_cost: float = 0.0   # battery replacement

    # Operating costs (negative = cost)
    grid_energy_cost: float = 0.0
    demand_charge_rs: float = 0.0   # monthly peak demand charges
    diesel_fuel_cost: float = 0.0
    om_bess: float = 0.0
    om_solar: float = 0.0
    om_dg: float = 0.0
    om_dg_variable: float = 0.0     # ₹/kWh generated

    # Outage cost
    outage_cost: float = 0.0

    # Salvage (positive, end of life)
    salvage: float = 0.0

    @property
    def total_opex(self) -> float:
        return (
            self.grid_energy_cost
            + self.demand_charge_rs
            + self.diesel_fuel_cost
            + self.om_bess
            + self.om_solar
            + self.om_dg
            + self.om_dg_variable
            + self.outage_cost
        )

    @property
    def net_cashflow(self) -> float:
        """Total cashflow for this year (negative = outflow)."""
        return -(self.capex + self.replacement_cost) + self.salvage - self.total_opex


# ---------------------------------------------------------------------------
# Finance result
# ---------------------------------------------------------------------------

@dataclass
class FinanceResult:
    """Full financial result for a project."""

    scenario_name: str
    cashflows: list[YearCashflow] = field(default_factory=list)

    # Summary
    total_capex: float = 0.0
    npv: float = 0.0
    irr: Optional[float] = None
    simple_payback_years: Optional[float] = None
    discounted_payback_years: Optional[float] = None
    lcoe_rs_kwh: float = 0.0

    # Cost breakdown
    total_cost_with_outages: float = 0.0
    total_cost_without_outages: float = 0.0
    total_outage_cost: float = 0.0

    # Incremental vs baseline
    incremental_npv: Optional[float] = None

    @property
    def net_cashflow_series(self) -> list[float]:
        return [cf.net_cashflow for cf in self.cashflows]


# ---------------------------------------------------------------------------
# Finance Engine
# ---------------------------------------------------------------------------

class FinanceEngine:
    """
    Compute financial metrics for one scenario over project lifetime.

    Parameters
    ----------
    config : ProjectConfig
    sim_results : list[SimulationResult]
        One SimulationResult per project year (length == project_lifetime_years).
        Year 0 result is used for capex only; year 1..N for opex.
    scenario_name : str
    baseline_npv : float | None
        NPV of baseline (grid-only) scenario for incremental comparison.
    """

    def __init__(
        self,
        config: ProjectConfig,
        sim_results: list[SimulationResult],
        scenario_name: str = "",
        baseline_npv: Optional[float] = None,
    ) -> None:
        self.cfg = config
        self.results = sim_results
        self.scenario_name = scenario_name
        self.baseline_npv = baseline_npv
        self.n_years = config.project_lifetime_years
        self.discount_rate = config.discount_rate_pct / 100.0

    def run(self) -> FinanceResult:
        fr = FinanceResult(scenario_name=self.scenario_name)

        # Year 0: CAPEX
        capex_cf = YearCashflow(year=0)
        capex_cf.capex = self._compute_capex()
        fr.cashflows.append(capex_cf)
        fr.total_capex = capex_cf.capex

        total_energy_kwh = 0.0

        for yr in range(1, self.n_years + 1):
            sim = self.results[min(yr - 1, len(self.results) - 1)]
            cf = YearCashflow(year=yr)

            # Grid energy cost — use TOU hourly rates if configured
            escalation = (1 + self.cfg.grid.escalation_pct_yr / 100) ** yr
            if self.cfg.grid.tou_schedule is not None and sim.hourly:
                # Sum per-hour grid draw × hour-specific TOU rate
                grid_energy_cost = sum(
                    (h.grid_to_load + h.grid_to_bess) * h.tou_rate_rs_kwh * escalation
                    for h in sim.hourly
                )
            else:
                grid_rate = self.cfg.grid.energy_rate_rs_kwh * escalation
                grid_energy_cost = sim.total_grid_kwh * grid_rate
            cf.grid_energy_cost = grid_energy_cost + self.cfg.grid.fixed_charge_rs_month * 12

            # Demand charge — monthly peak kW × ₹/kVA/month
            if self.cfg.grid.demand_charge is not None and sim.monthly_peak_grid_kw:
                dc = self.cfg.grid.demand_charge
                pf = dc.power_factor
                annual_demand_charge = 0.0
                for m, peak_kw in enumerate(sim.monthly_peak_grid_kw):
                    if dc.peak_hours_only and self.cfg.grid.tou_schedule is not None:
                        # Only bill months where peak occurred during TOU peak hours
                        # Simplified: use the full monthly peak (conservative)
                        pass
                    peak_kva = peak_kw / pf
                    annual_demand_charge += peak_kva * dc.charge_rs_kva_month
                cf.demand_charge_rs = annual_demand_charge * escalation

            # Diesel fuel cost
            if self.cfg.dg is not None:
                dg_esc = (1 + self.cfg.dg.diesel_escalation_pct_yr / 100) ** yr
                fuel_price = self.cfg.dg.diesel_price_rs_l * dg_esc
                cf.diesel_fuel_cost = sim.dg_fuel_liters * fuel_price
                # DG variable O&M
                cf.om_dg_variable = sim.total_dg_kwh * self.cfg.dg.om_rs_kwh
                # DG fixed O&M (per running hour estimate)
                dg_hours = sum(
                    1 for h in sim.hourly if h.dg_running
                )
                cf.om_dg = dg_hours * self.cfg.dg.om_rs_hour
                # Annual DG CAPEX amortized (treated as year-0 capex for simplicity)

            # BESS O&M
            if self.cfg.bess is not None:
                cf.om_bess = self.cfg.bess.om_rs_kwh_yr * self.cfg.bess.capacity_kwh

            # Solar O&M
            if self.cfg.solar is not None:
                cf.om_solar = self.cfg.solar.om_rs_kw_yr * self.cfg.solar.size_kw

            # Outage cost
            cf.outage_cost = self._compute_outage_cost(sim, yr)

            # BESS replacement check
            if self.cfg.bess is not None:
                if self._bess_needs_replacement(sim, yr):
                    replacement = (
                        self.cfg.bess.capex_rs
                        * self.cfg.bess.replacement_cost_fraction
                    )
                    cf.replacement_cost = replacement

            # Salvage at end of project
            if yr == self.n_years:
                cf.salvage = self._compute_salvage()

            fr.cashflows.append(cf)
            total_energy_kwh += sim.total_grid_kwh + sim.total_solar_kwh + sim.total_dg_kwh

        # Financial metrics
        fr.npv = self._compute_npv(fr.cashflows)
        fr.irr = self._compute_irr(fr.cashflows)
        fr.simple_payback_years = self._compute_simple_payback(fr.cashflows)
        fr.discounted_payback_years = self._compute_discounted_payback(fr.cashflows)
        fr.lcoe_rs_kwh = self._compute_lcoe(fr.cashflows, total_energy_kwh)

        fr.total_outage_cost = sum(cf.outage_cost for cf in fr.cashflows)
        fr.total_cost_with_outages = sum(cf.capex + cf.replacement_cost + cf.total_opex for cf in fr.cashflows)
        fr.total_cost_without_outages = fr.total_cost_with_outages - fr.total_outage_cost

        if self.baseline_npv is not None:
            fr.incremental_npv = fr.npv - self.baseline_npv

        return fr

    # ------------------------------------------------------------------
    # CAPEX
    # ------------------------------------------------------------------

    def _compute_capex(self) -> float:
        total = 0.0
        if self.cfg.bess:
            total += self.cfg.bess.capex_rs
        if self.cfg.solar:
            total += self.cfg.solar.capex_rs
        if self.cfg.dg:
            total += self.cfg.dg.capex_rs
        return total

    # ------------------------------------------------------------------
    # Outage cost
    # ------------------------------------------------------------------

    def _compute_outage_cost(self, sim: SimulationResult, year: int) -> float:
        oc = self.cfg.outage_cost

        if oc.tiered_penalties:
            return self._tiered_outage_cost(sim)

        if oc.mode == OutageCostMode.VOLL:
            return sim.ens_kwh * oc.voll_rs_kwh

        else:  # DOWNTIME
            return sim.downtime_hours * oc.downtime_cost_rs_hr

    def _tiered_outage_cost(self, sim: SimulationResult) -> float:
        """Compute outage cost using tiered penalty schedule."""
        oc = self.cfg.outage_cost
        if not oc.tiered_penalties:
            return 0.0

        tiers = sorted(oc.tiered_penalties, key=lambda t: t.threshold_minutes)
        total_cost = 0.0

        # Group by outage event and compute per-event cost
        # Simplified: compute based on total downtime hours
        downtime_min = sim.downtime_hours * 60.0
        if downtime_min <= 0:
            return 0.0

        # Apply tiers sequentially
        remaining_min = downtime_min
        prev_threshold = 0.0

        for tier in tiers:
            in_tier = min(remaining_min, tier.threshold_minutes - prev_threshold)
            if in_tier <= 0:
                break
            total_cost += (in_tier / 60.0) * tier.cost_rs_hr
            remaining_min -= in_tier
            prev_threshold = tier.threshold_minutes

        # Beyond last tier
        if remaining_min > 0 and tiers:
            total_cost += (remaining_min / 60.0) * tiers[-1].cost_rs_hr

        return total_cost

    # ------------------------------------------------------------------
    # BESS replacement
    # ------------------------------------------------------------------

    def _bess_needs_replacement(self, sim: SimulationResult, year: int) -> bool:
        """Simple rule: replace at calendar EOL or when capacity degraded to EOL%."""
        if not self.cfg.bess:
            return False
        b = self.cfg.bess
        if year == int(b.calendar_life_yr):
            return True
        cap_pct = sim.bess_capacity_degraded_kwh / b.capacity_kwh * 100
        return cap_pct < b.eol_capacity_pct

    # ------------------------------------------------------------------
    # Salvage
    # ------------------------------------------------------------------

    def _compute_salvage(self) -> float:
        """Residual value at end of project (10% of CAPEX)."""
        return self._compute_capex() * 0.10

    # ------------------------------------------------------------------
    # NPV / IRR / Payback / LCOE
    # ------------------------------------------------------------------

    def _compute_npv(self, cashflows: list[YearCashflow]) -> float:
        npv = 0.0
        for cf in cashflows:
            pv = cf.net_cashflow / (1 + self.discount_rate) ** cf.year
            npv += pv
        return npv

    def _compute_irr(self, cashflows: list[YearCashflow]) -> Optional[float]:
        """
        Compute IRR using Newton-Raphson / bisection on NPV(r)=0.
        Returns None if no real positive IRR exists.
        """
        series = [cf.net_cashflow for cf in cashflows]
        if series[0] >= 0:
            return None  # No initial investment

        def npv_at_rate(r: float) -> float:
            if r <= -1:
                return float("inf")
            return sum(c / (1 + r) ** i for i, c in enumerate(series))

        # Bisect between -0.99 and 5.0
        try:
            lo, hi = -0.99, 5.0
            if npv_at_rate(lo) * npv_at_rate(hi) > 0:
                return None  # No sign change → no IRR
            for _ in range(100):
                mid = (lo + hi) / 2
                if abs(hi - lo) < 1e-8:
                    break
                if npv_at_rate(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            return mid * 100  # return as percentage
        except Exception:
            return None

    def _compute_simple_payback(self, cashflows: list[YearCashflow]) -> Optional[float]:
        """Simple payback: when cumulative cashflow turns positive."""
        capex = cashflows[0].capex
        if capex <= 0:
            return 0.0
        cumulative = -capex
        for cf in cashflows[1:]:
            savings = cf.net_cashflow
            if cumulative + savings >= 0:
                # Interpolate
                frac = -cumulative / savings if savings != 0 else 0
                return cf.year - 1 + frac
            cumulative += savings
        return None

    def _compute_discounted_payback(self, cashflows: list[YearCashflow]) -> Optional[float]:
        """Discounted payback period."""
        capex = cashflows[0].capex
        if capex <= 0:
            return 0.0
        cumulative = -capex
        for cf in cashflows[1:]:
            pv = cf.net_cashflow / (1 + self.discount_rate) ** cf.year
            if cumulative + pv >= 0:
                frac = -cumulative / pv if pv != 0 else 0
                return cf.year - 1 + frac
            cumulative += pv
        return None

    def _compute_lcoe(self, cashflows: list[YearCashflow], total_energy_kwh: float) -> float:
        """
        LCOE = NPV of all costs / NPV of all energy served (₹/kWh).
        Excludes outage costs in numerator (cost of supply, not reliability penalty).
        """
        if total_energy_kwh <= 0:
            return 0.0
        # Total cost NPV (costs are positive in this computation)
        pv_costs = 0.0
        for cf in cashflows:
            cost = cf.capex + cf.replacement_cost + cf.total_opex - cf.outage_cost - cf.salvage
            pv_costs += cost / (1 + self.discount_rate) ** cf.year
        return pv_costs / (total_energy_kwh / self.n_years * self.n_years)  # annualised


# ---------------------------------------------------------------------------
# Sensitivity Engine
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Result of a single sensitivity sweep."""
    parameter: str
    delta_pct: float
    npv: float
    ens_kwh: float
    delta_npv: float
    delta_ens_kwh: float


class SensitivityEngine:
    """
    Tornado / single-parameter sensitivity analysis.

    For each parameter, perturbs it ±delta% and recomputes NPV and ENS.
    """

    PARAMETERS = [
        ("saidi", 50),
        ("saifi", 50),
        ("voll_or_downtime_cost", 50),
        ("diesel_price", 25),
        ("battery_capex", 20),
        ("dg_availability", 5),
        ("critical_load", 20),
    ]

    def __init__(self, base_config: ProjectConfig, runner_fn) -> None:
        """
        Parameters
        ----------
        base_config : ProjectConfig
        runner_fn : callable
            Signature: (config) -> (FinanceResult, SimulationResult)
            Used to run full simulation with modified config.
        """
        self.base_config = base_config
        self.runner_fn = runner_fn

    def run(self) -> list[SensitivityResult]:
        base_fr, base_sim = self.runner_fn(self.base_config)
        base_npv = base_fr.npv
        base_ens = base_sim.ens_kwh

        results: list[SensitivityResult] = []
        for param, delta_pct in self.PARAMETERS:
            for sign in [+1, -1]:
                try:
                    mod_cfg = self._perturb(self.base_config, param, delta_pct * sign)
                    fr, sim = self.runner_fn(mod_cfg)
                    results.append(SensitivityResult(
                        parameter=param,
                        delta_pct=delta_pct * sign,
                        npv=fr.npv,
                        ens_kwh=sim.ens_kwh,
                        delta_npv=fr.npv - base_npv,
                        delta_ens_kwh=sim.ens_kwh - base_ens,
                    ))
                except Exception:
                    pass

        return results

    def _perturb(self, cfg: ProjectConfig, param: str, delta_pct: float) -> ProjectConfig:
        """Return a deep-copied config with one parameter perturbed."""
        import copy
        c = copy.deepcopy(cfg)
        factor = 1.0 + delta_pct / 100.0

        if param == "saidi" and c.outage.stochastic:
            c.outage.stochastic.saidi_minutes_per_year *= factor
        elif param == "saifi" and c.outage.stochastic:
            c.outage.stochastic.saifi_events_per_year *= factor
        elif param == "voll_or_downtime_cost":
            if c.outage_cost.voll_rs_kwh:
                c.outage_cost.voll_rs_kwh *= factor
            if c.outage_cost.downtime_cost_rs_hr:
                c.outage_cost.downtime_cost_rs_hr *= factor
        elif param == "diesel_price" and c.dg:
            c.dg.diesel_price_rs_l *= factor
        elif param == "battery_capex" and c.bess:
            c.bess.capex_rs_kwh *= factor
        elif param == "dg_availability" and c.dg:
            c.dg.availability_pct = min(100.0, c.dg.availability_pct * factor)
        elif param == "critical_load":
            c.load.critical_fraction = min(1.0, c.load.critical_fraction * factor)

        return c
