"""
EnergeX — CLI Interface

Usage:
    energex run --config config.json --seed 42 --export out/
    energex validate --config config.json
    energex generate-sample-csv --output-dir samples/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(
    name="energex",
    help="EnergeX — Backup/Outage Energy ROI Calculator",
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to config JSON/YAML file",
        exists=True, readable=True, resolve_path=True,
    ),
    seed: int = typer.Option(
        42, "--seed", "-s",
        help="Random seed for reproducible stochastic outages",
    ),
    export: Path = typer.Option(
        Path("energex_output"),
        "--export", "-e",
        help="Output directory for exported results",
    ),
    baseline: bool = typer.Option(
        True,
        "--baseline/--no-baseline",
        help="Compute grid-only baseline NPV for incremental comparison",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed hourly stats",
    ),
) -> None:
    """Run a full EnergeX simulation and export results."""
    from energex.adapters.config_loader import load_config
    from energex.adapters.exporters import export_all
    from energex.engine.runner import run_simulation, run_grid_only_baseline

    console.rule("[bold cyan]EnergeX — Backup/Outage Energy ROI Engine[/bold cyan]")

    # Load config
    with console.status("Loading configuration..."):
        try:
            cfg = load_config(config)
        except Exception as e:
            console.print(f"[red]Config error: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"  Project : [bold]{cfg.name}[/bold]")
    console.print(f"  Scenario: [bold yellow]{cfg.scenario.value if cfg.scenario else 'auto'}[/bold yellow]")
    console.print(f"  Lifetime: {cfg.project_lifetime_years} years | Discount: {cfg.discount_rate_pct}%")
    console.print(f"  Seed    : {seed}")
    console.print()

    # Baseline
    baseline_npv = None
    if baseline:
        with console.status("Computing grid-only baseline..."):
            try:
                bl_finance = run_grid_only_baseline(cfg, seed=seed)
                baseline_npv = bl_finance.npv
                console.print(f"  Baseline NPV (grid-only): [cyan]₹{baseline_npv:,.0f}[/cyan]")
            except Exception as e:
                console.print(f"  [yellow]Baseline failed: {e}[/yellow]")

    # Run simulation
    with console.status(f"Simulating {cfg.project_lifetime_years} years..."):
        try:
            result = run_simulation(cfg, seed=seed, baseline_npv=baseline_npv)
        except Exception as e:
            console.print(f"[red]Simulation error: {e}[/red]")
            raise typer.Exit(1)

    # Display results
    _display_results(result, verbose)

    # Export
    with console.status("Exporting results..."):
        export.mkdir(parents=True, exist_ok=True)
        files = export_all(
            config=cfg,
            sim=result.sim_year1,
            finance=result.finance,
            output_dir=export,
            seed=seed,
            warnings=result.warnings,
            run_id=result.run_id,
        )

    console.print()
    console.rule("[green]Exported Files[/green]")
    for name, path in files.items():
        console.print(f"  [green]✓[/green] {name:30s} → {path}")

    console.print()
    if result.sim_year1.sla_pass:
        console.print(Panel("[bold green]✓ SLA PASSED[/bold green]", expand=False))
    else:
        console.print(Panel(
            "[bold red]✗ SLA FAILED[/bold red]\n"
            + "\n".join(f"  • {f}" for f in result.sim_year1.sla_failures),
            expand=False,
        ))


def _display_results(result, verbose: bool) -> None:
    """Pretty-print key results to console."""
    sim = result.sim_year1
    fin = result.finance

    console.print()
    console.rule("[bold]Reliability Results (Year 1)[/bold]")

    rel_table = Table(box=box.SIMPLE, show_header=True)
    rel_table.add_column("Metric", style="cyan")
    rel_table.add_column("Value", justify="right")
    rel_table.add_column("Unit")
    rel_table.add_row("ENS (critical)", f"{sim.ens_kwh:.2f}", "kWh/yr")
    rel_table.add_row("Downtime", f"{sim.downtime_hours:.2f}", "hrs/yr")
    rel_table.add_row("Outage events", str(sim.outage_events_total), "events/yr")
    rel_table.add_row("Events served", f"{sim.outage_served_pct:.1f}%", "")
    rel_table.add_row("Backup autonomy", f"{sim.backup_autonomy_hours:.2f}", "hrs")
    rel_table.add_row("Continuity", f"{sim.continuity_pct:.4f}%", "")
    console.print(rel_table)

    console.rule("[bold]Energy Flows (Year 1)[/bold]")
    ef_table = Table(box=box.SIMPLE, show_header=True)
    ef_table.add_column("Flow", style="cyan")
    ef_table.add_column("kWh/yr", justify="right")
    ef_table.add_row("Grid → load", f"{sim.total_grid_kwh:,.0f}")
    ef_table.add_row("Solar → load/BESS", f"{sim.total_solar_kwh:,.0f}")
    ef_table.add_row("BESS → load (backup)", f"{sim.total_bess_discharge_kwh:,.0f}")
    ef_table.add_row("DG → load (backup)", f"{sim.total_dg_kwh:,.0f}")
    ef_table.add_row("[red]Unserved critical[/red]", f"{sim.ens_kwh:,.1f}")
    console.print(ef_table)

    console.rule("[bold]Financial Summary[/bold]")
    fin_table = Table(box=box.SIMPLE, show_header=True)
    fin_table.add_column("Metric", style="cyan")
    fin_table.add_column("Value", justify="right")
    fin_table.add_row("Total CAPEX", f"₹{fin.total_capex:,.0f}")
    fin_table.add_row("NPV (25yr)", f"₹{fin.npv:,.0f}")
    fin_table.add_row("IRR", f"{fin.irr:.2f}%" if fin.irr is not None else "N/A")
    fin_table.add_row("Simple payback", f"{fin.simple_payback_years:.1f} yr" if fin.simple_payback_years else "N/A")
    fin_table.add_row("LCOE", f"₹{fin.lcoe_rs_kwh:.2f}/kWh")
    fin_table.add_row("Total cost (with outages)", f"₹{fin.total_cost_with_outages:,.0f}")
    fin_table.add_row("Total outage cost", f"₹{fin.total_outage_cost:,.0f}")
    if fin.incremental_npv is not None:
        label = "Incremental NPV vs baseline"
        colour = "green" if fin.incremental_npv > 0 else "red"
        fin_table.add_row(label, f"[{colour}]₹{fin.incremental_npv:,.0f}[/{colour}]")
    console.print(fin_table)

    if result.warnings:
        console.rule("[bold yellow]Warnings[/bold yellow]")
        for w in result.warnings:
            console.print(f"  [yellow]⚠[/yellow]  {w}")


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------

@app.command()
def validate(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to config JSON/YAML file",
        exists=True, readable=True, resolve_path=True,
    ),
) -> None:
    """Validate a config file without running the simulation."""
    from energex.adapters.config_loader import load_config

    console.rule("[cyan]EnergeX — Config Validation[/cyan]")

    try:
        cfg = load_config(config)
        console.print(f"[green]✓[/green] Config is valid")
        console.print(f"  Project  : {cfg.name}")
        console.print(f"  Scenario : {cfg.scenario.value if cfg.scenario else 'auto'}")
        console.print(f"  Load mode: {cfg.load.mode.value}")
        if cfg.outage.stochastic:
            console.print(
                f"  Outages  : SAIDI={cfg.outage.stochastic.saidi_minutes_per_year} min/yr, "
                f"SAIFI={cfg.outage.stochastic.saifi_events_per_year} events/yr"
            )
        elif cfg.outage.events:
            console.print(f"  Outages  : {len(cfg.outage.events)} deterministic events")
        assets = []
        if cfg.dg:
            assets.append(f"DG {cfg.dg.rated_kw} kW")
        if cfg.bess:
            assets.append(f"BESS {cfg.bess.capacity_kwh} kWh / {cfg.bess.power_kw} kW")
        if cfg.solar:
            assets.append(f"Solar {cfg.solar.size_kw} kWp")
        console.print(f"  Assets   : {', '.join(assets) if assets else 'Grid only'}")
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# generate-sample-csv command
# ---------------------------------------------------------------------------

@app.command(name="generate-sample-csv")
def generate_sample_csv(
    output_dir: Path = typer.Option(
        Path("sample_data"),
        "--output-dir", "-o",
        help="Directory to write sample CSVs",
    ),
    avg_kw: float = typer.Option(100.0, help="Average total load (kW)"),
    critical_fraction: float = typer.Option(0.3, help="Critical load fraction"),
    solar_kw: float = typer.Option(50.0, help="Solar installed capacity (kWp)"),
) -> None:
    """Generate sample load and solar timeseries CSV files."""
    from energex.adapters.csv_parser import generate_sample_load_csv, generate_sample_solar_csv

    output_dir.mkdir(parents=True, exist_ok=True)

    load_path = output_dir / "load_timeseries.csv"
    solar_path = output_dir / "solar_timeseries.csv"

    generate_sample_load_csv(load_path, avg_kw=avg_kw, critical_fraction=critical_fraction)
    generate_sample_solar_csv(solar_path, size_kw=solar_kw)

    console.print(f"[green]✓[/green] Load CSV   → {load_path}")
    console.print(f"[green]✓[/green] Solar CSV  → {solar_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
