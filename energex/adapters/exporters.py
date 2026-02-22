"""
EnergeX — Exporters

Exports simulation results to JSON, CSV, and warnings files.
"""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from energex.domain.schemas import ProjectConfig
from energex.engine.dispatch_backup import SimulationResult
from energex.engine.finance_engine import FinanceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# results_summary.json
# ---------------------------------------------------------------------------

def export_results_summary(
    config: ProjectConfig,
    sim: SimulationResult,
    finance: FinanceResult,
    output_dir: str | Path,
    seed: int,
    run_id: str | None = None,
    model_version: str = "1.0.0",
    warnings: list[str] | None = None,
) -> Path:
    """Write results_summary.json."""
    out = Path(output_dir)
    _ensure_dir(out)

    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    summary: dict[str, Any] = {
        "meta": {
            "run_id": run_id,
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seed": seed,
            "scenario": config.scenario.value if config.scenario else "unknown",
            "project_name": config.name,
            "project_lifetime_years": config.project_lifetime_years,
        },
        "reliability": {
            "ens_critical_kwh_yr": round(sim.ens_kwh, 2),
            "downtime_critical_hrs_yr": round(sim.downtime_hours, 2),
            "outage_events_total": sim.outage_events_total,
            "outage_events_served": sim.outage_events_served,
            "outage_served_pct": round(sim.outage_served_pct, 2),
            "backup_autonomy_hours": round(sim.backup_autonomy_hours, 2),
            "continuity_pct": round(sim.continuity_pct, 4),
            "sla_pass": sim.sla_pass,
            "sla_failures": sim.sla_failures,
        },
        "energy_flows_kwh_yr": {
            "grid_to_load": round(sim.total_grid_kwh, 1),
            "solar_total": round(sim.total_solar_kwh, 1),
            "solar_to_bess": round(sim.total_solar_to_bess_kwh, 1),
            "grid_to_bess": round(sim.total_grid_to_bess_kwh, 1),
            "bess_to_load": round(sim.total_bess_discharge_kwh, 1),
            "dg_to_load": round(sim.total_dg_kwh, 1),
            "unserved_critical": round(sim.ens_kwh, 1),
        },
        "dg_fuel_liters_yr": round(sim.dg_fuel_liters, 1),
        "financial": {
            "total_capex_rs": round(finance.total_capex, 0),
            "npv_rs": round(finance.npv, 0),
            "irr_pct": round(finance.irr, 2) if finance.irr is not None else None,
            "simple_payback_years": round(finance.simple_payback_years, 2) if finance.simple_payback_years else None,
            "discounted_payback_years": round(finance.discounted_payback_years, 2) if finance.discounted_payback_years else None,
            "lcoe_rs_kwh": round(finance.lcoe_rs_kwh, 2),
            "total_cost_with_outages_rs": round(finance.total_cost_with_outages, 0),
            "total_cost_without_outages_rs": round(finance.total_cost_without_outages, 0),
            "total_outage_cost_rs": round(finance.total_outage_cost, 0),
            "incremental_npv_rs": round(finance.incremental_npv, 0) if finance.incremental_npv is not None else None,
        },
        "assumptions": _build_assumptions(config),
        "warnings": warnings or [],
    }

    out_file = out / "results_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return out_file


# ---------------------------------------------------------------------------
# reliability_metrics.csv
# ---------------------------------------------------------------------------

def export_reliability_metrics(
    sim: SimulationResult,
    output_dir: str | Path,
) -> Path:
    """Write reliability_metrics.csv with per-event breakdown."""
    out = Path(output_dir)
    _ensure_dir(out)

    out_file = out / "reliability_metrics.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric", "value", "unit"
        ])
        writer.writerow(["ens_critical", round(sim.ens_kwh, 2), "kWh/yr"])
        writer.writerow(["downtime_critical", round(sim.downtime_hours, 4), "hrs/yr"])
        writer.writerow(["outage_events_total", sim.outage_events_total, "events/yr"])
        writer.writerow(["outage_events_served", sim.outage_events_served, "events/yr"])
        writer.writerow(["outage_served_pct", round(sim.outage_served_pct, 2), "%"])
        writer.writerow(["backup_autonomy_hours", round(sim.backup_autonomy_hours, 2), "hrs"])
        writer.writerow(["continuity_pct", round(sim.continuity_pct, 4), "%"])
        writer.writerow(["sla_pass", "yes" if sim.sla_pass else "no", ""])
        for i, failure in enumerate(sim.sla_failures):
            writer.writerow([f"sla_failure_{i+1}", failure, ""])

    return out_file


# ---------------------------------------------------------------------------
# energy_flows.csv  (hourly)
# ---------------------------------------------------------------------------

def export_energy_flows(
    sim: SimulationResult,
    output_dir: str | Path,
) -> Path:
    """Write energy_flows.csv with hourly energy flows."""
    out = Path(output_dir)
    _ensure_dir(out)

    out_file = out / "energy_flows.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hour",
            "grid_available",
            "outage_fraction",
            "total_load_kwh",
            "critical_load_kwh",
            "grid_to_load_kwh",
            "solar_to_load_kwh",
            "solar_to_bess_kwh",
            "grid_to_bess_kwh",
            "bess_to_load_kwh",
            "dg_to_load_kwh",
            "unserved_critical_kwh",
            "bess_soc_end_kwh",
            "downtime_hours",
            "dg_fuel_liters",
        ])
        for h in sim.hourly:
            writer.writerow([
                h.hour,
                1 if h.grid_available else 0,
                round(h.outage_fraction, 4),
                round(h.total_load_kwh, 3),
                round(h.critical_load_kwh, 3),
                round(h.grid_to_load, 3),
                round(h.solar_to_load, 3),
                round(h.solar_to_bess, 3),
                round(h.grid_to_bess, 3),
                round(h.bess_to_load, 3),
                round(h.dg_to_load, 3),
                round(h.unserved_critical, 3),
                round(h.bess_soc_end, 3),
                round(h.downtime_hours, 4),
                round(h.dg_fuel_liters, 4),
            ])

    return out_file


# ---------------------------------------------------------------------------
# cashflows.csv
# ---------------------------------------------------------------------------

def export_cashflows(
    finance: FinanceResult,
    output_dir: str | Path,
) -> Path:
    """Write cashflows.csv with yearly financial breakdown."""
    out = Path(output_dir)
    _ensure_dir(out)

    out_file = out / "cashflows.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year",
            "capex_rs",
            "replacement_cost_rs",
            "grid_energy_cost_rs",
            "diesel_fuel_cost_rs",
            "om_bess_rs",
            "om_solar_rs",
            "om_dg_rs",
            "om_dg_variable_rs",
            "outage_cost_rs",
            "salvage_rs",
            "total_opex_rs",
            "net_cashflow_rs",
        ])
        for cf in finance.cashflows:
            writer.writerow([
                cf.year,
                round(cf.capex, 0),
                round(cf.replacement_cost, 0),
                round(cf.grid_energy_cost, 0),
                round(cf.diesel_fuel_cost, 0),
                round(cf.om_bess, 0),
                round(cf.om_solar, 0),
                round(cf.om_dg, 0),
                round(cf.om_dg_variable, 0),
                round(cf.outage_cost, 0),
                round(cf.salvage, 0),
                round(cf.total_opex, 0),
                round(cf.net_cashflow, 0),
            ])

    return out_file


# ---------------------------------------------------------------------------
# warnings.json
# ---------------------------------------------------------------------------

def export_warnings(
    warnings: list[str],
    output_dir: str | Path,
) -> Path:
    """Write warnings.json."""
    out = Path(output_dir)
    _ensure_dir(out)

    out_file = out / "warnings.json"
    with open(out_file, "w") as f:
        json.dump({"warnings": warnings, "count": len(warnings)}, f, indent=2)

    return out_file


# ---------------------------------------------------------------------------
# Assumptions builder
# ---------------------------------------------------------------------------

def _build_assumptions(config: ProjectConfig) -> list[str]:
    """Generate assumptions list from config."""
    assumptions = [
        f"Simulation year: {config.simulation_year}",
        f"Project lifetime: {config.project_lifetime_years} years",
        f"Discount rate: {config.discount_rate_pct}%/yr",
        f"Grid tariff: ₹{config.grid.energy_rate_rs_kwh}/kWh, escalation {config.grid.escalation_pct_yr}%/yr",
        f"Simulation timebase: hourly (1h)",
        f"Load mode: {config.load.mode.value}",
        f"Outage mode: {config.outage.mode.value}",
    ]
    if config.outage.stochastic:
        assumptions.append(
            f"SAIDI: {config.outage.stochastic.saidi_minutes_per_year} min/yr, "
            f"SAIFI: {config.outage.stochastic.saifi_events_per_year} events/yr"
        )
    if config.bess:
        assumptions.append(
            f"BESS: {config.bess.capacity_kwh} kWh / {config.bess.power_kw} kW, "
            f"DoD {config.bess.dod_pct}%, RTE {config.bess.roundtrip_efficiency*100}%"
        )
    if config.dg:
        assumptions.append(
            f"DG: {config.dg.rated_kw} kW, start delay {config.dg.start_delay_seconds}s, "
            f"availability {config.dg.availability_pct}%"
        )
    if config.solar:
        assumptions.append(
            f"Solar: {config.solar.size_kw} kWp, degradation {config.solar.degradation_pct_yr}%/yr"
        )
    if config.outage_cost.mode.value == "voll":
        assumptions.append(f"VoLL: ₹{config.outage_cost.voll_rs_kwh}/kWh unserved")
    else:
        assumptions.append(f"Downtime cost: ₹{config.outage_cost.downtime_cost_rs_hr}/hr")
    return assumptions


# ---------------------------------------------------------------------------
# Convenience: export all at once
# ---------------------------------------------------------------------------

def export_all(
    config: ProjectConfig,
    sim: SimulationResult,
    finance: FinanceResult,
    output_dir: str | Path,
    seed: int,
    warnings: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Export all result files. Returns dict of {name: path}."""
    out = Path(output_dir)
    warnings = warnings or []

    files = {}
    files["results_summary"] = export_results_summary(
        config, sim, finance, out, seed=seed, run_id=run_id, warnings=warnings
    )
    files["reliability_metrics"] = export_reliability_metrics(sim, out)
    files["energy_flows"] = export_energy_flows(sim, out)
    files["cashflows"] = export_cashflows(finance, out)
    files["warnings"] = export_warnings(warnings, out)

    return files
