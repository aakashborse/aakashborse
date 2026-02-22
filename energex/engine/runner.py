"""
EnergeX — Simulation Runner

Orchestrates the full simulation pipeline:
  1. Load config
  2. Generate outages
  3. Run dispatch simulation (per year)
  4. Compute financials
  5. Return results
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from energex.domain.schemas import ProjectConfig
from energex.engine.outage_generator import OutageGenerator, OutageEvent
from energex.engine.dispatch_backup import (
    BackupDispatchEngine,
    SimulationResult,
    build_hourly_loads,
    build_hourly_solar,
)
from energex.engine.finance_engine import FinanceEngine, FinanceResult
from energex.adapters.csv_parser import load_load_timeseries, load_solar_timeseries


@dataclass
class RunResult:
    """Full run result bundle."""
    run_id: str
    seed: int
    config: ProjectConfig
    outage_events: list[OutageEvent]       # Year-1 events (representative)
    sim_year1: SimulationResult            # Year 1 simulation
    all_sim_results: list[SimulationResult]
    finance: FinanceResult
    warnings: list[str]


def run_simulation(
    config: ProjectConfig,
    seed: int = 42,
    baseline_npv: Optional[float] = None,
) -> RunResult:
    """
    Run the complete EnergeX simulation for the given config.

    Parameters
    ----------
    config : ProjectConfig
    seed : int
        Random seed (determines outage events; same seed = same results).
    baseline_npv : float | None
        NPV of the grid-only baseline for incremental comparisons.

    Returns
    -------
    RunResult
    """
    run_id = str(uuid.uuid4())[:8]
    warnings: list[str] = []

    # --- Load arrays ---
    if config.load.mode.value == "timeseries" and config.load.timeseries_csv:
        total_kw, critical_kw = load_load_timeseries(config.load.timeseries_csv)
    else:
        total_kw, critical_kw = build_hourly_loads(config, year=0)

    if config.solar and config.solar.yield_timeseries_csv:
        solar_kw_base = load_solar_timeseries(config.solar.yield_timeseries_csv)
    else:
        solar_kw_base = build_hourly_solar(config, year=0)

    # --- Generate outages (stochastic: one draw per year) ---
    outage_gen = OutageGenerator(config.outage, seed=seed)
    # Use same generator for year 1 (representative)
    year1_events = outage_gen.generate()

    # --- Simulate each project year ---
    all_sims: list[SimulationResult] = []
    n_years = config.project_lifetime_years

    for yr in range(n_years):
        # Each year gets a slightly different seed for outage variation
        yr_seed = seed + yr * 7919
        yr_gen = OutageGenerator(config.outage, seed=yr_seed)
        yr_events = yr_gen.generate()

        # Solar degrades each year
        solar_kw = build_hourly_solar(config, year=yr)

        engine = BackupDispatchEngine(
            config=config,
            outage_events=yr_events,
            year=yr,
            seed=yr_seed,
            total_kw=total_kw,
            critical_kw=critical_kw,
            solar_kw=solar_kw,
        )
        sim = engine.run()
        all_sims.append(sim)

    sim_year1 = all_sims[0]

    # --- Warnings ---
    if sim_year1.ens_kwh > 0:
        warnings.append(
            f"Energy Not Served in year 1: {sim_year1.ens_kwh:.2f} kWh — "
            "consider increasing BESS capacity or adding DG"
        )
    if not sim_year1.sla_pass:
        for failure in sim_year1.sla_failures:
            warnings.append(f"SLA VIOLATION: {failure}")
    if config.dg and config.dg.rated_kw < (critical_kw.max() if len(critical_kw) > 0 else 0):
        warnings.append(
            f"DG rated power ({config.dg.rated_kw} kW) is less than peak critical load "
            f"({critical_kw.max():.1f} kW) — DG may not fully cover peak outage demand"
        )
    if config.bess and config.bess.min_reserve_soc_pct > 0:
        warnings.append(
            f"BESS reserve SOC of {config.bess.min_reserve_soc_pct}% "
            f"({config.bess.capacity_kwh * config.bess.min_reserve_soc_pct / 100:.1f} kWh) "
            "is held as emergency buffer and not available for regular backup"
        )

    # --- Finance ---
    fin_engine = FinanceEngine(
        config=config,
        sim_results=all_sims,
        scenario_name=config.scenario.value if config.scenario else "unknown",
        baseline_npv=baseline_npv,
    )
    finance = fin_engine.run()

    return RunResult(
        run_id=run_id,
        seed=seed,
        config=config,
        outage_events=year1_events,
        sim_year1=sim_year1,
        all_sim_results=all_sims,
        finance=finance,
        warnings=warnings,
    )


def run_grid_only_baseline(config: ProjectConfig, seed: int = 42) -> FinanceResult:
    """
    Run grid-only scenario (no backup) to establish baseline NPV.
    Used for incremental cost comparison.
    """
    import copy
    base = copy.deepcopy(config)
    base.dg = None
    base.bess = None
    base.solar = None
    base.scenario = None
    result = run_simulation(base, seed=seed)
    return result.finance
