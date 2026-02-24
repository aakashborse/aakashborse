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
# ui command  (Phase B)
# ---------------------------------------------------------------------------

@app.command()
def ui(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit server port"),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Open browser automatically"),
) -> None:
    """Launch the EnergeX interactive Streamlit dashboard (Phase B)."""
    import subprocess, sys
    from pathlib import Path

    app_path = Path(__file__).parent / "streamlit_app.py"
    if not app_path.exists():
        console.print(f"[red]Streamlit app not found at {app_path}[/red]")
        raise typer.Exit(1)

    console.rule("[bold cyan]EnergeX — Streamlit Dashboard[/bold cyan]")
    console.print(f"  Launching UI on [link]http://localhost:{port}[/link]")
    console.print("  Press Ctrl+C to stop.\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        f"--server.headless={'false' if browser else 'true'}",
        "--theme.primaryColor", "#1A237E",
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]UI stopped.[/yellow]")


# ---------------------------------------------------------------------------
# pdf command  (Phase B)
# ---------------------------------------------------------------------------

@app.command()
def pdf(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to config JSON/YAML file",
        exists=True, readable=True, resolve_path=True,
    ),
    output: Path = typer.Option(
        Path("energex_report.pdf"), "--output", "-o",
        help="Output PDF path",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    sensitivity: bool = typer.Option(
        False, "--sensitivity/--no-sensitivity",
        help="Include sensitivity tornado charts in PDF",
    ),
) -> None:
    """Generate a PDF report (management summary + engineering appendix)."""
    from energex.adapters.config_loader import load_config
    from energex.engine.runner import run_simulation, run_grid_only_baseline
    from energex.adapters.pdf_report import generate_pdf
    from energex.engine.finance_engine import SensitivityEngine

    console.rule("[bold cyan]EnergeX — PDF Report Generator[/bold cyan]")

    with console.status("Loading config..."):
        cfg = load_config(config)

    with console.status("Running simulation..."):
        bl = run_grid_only_baseline(cfg, seed=seed)
        result = run_simulation(cfg, seed=seed, baseline_npv=bl.npv)

    sens_results = None
    if sensitivity:
        with console.status("Running sensitivity analysis..."):
            try:
                def runner_fn(c):
                    r = run_simulation(c, seed=seed)
                    return r.finance, r.sim_year1
                se = SensitivityEngine(cfg, runner_fn)
                sens_results = se.run()
            except Exception as e:
                console.print(f"  [yellow]Sensitivity failed: {e}[/yellow]")

    with console.status("Generating PDF..."):
        pdf_bytes = generate_pdf(result, sens_results)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(pdf_bytes)

    console.print(f"\n[green]✓[/green] PDF report written to: {output}")
    console.print(f"  Pages: management summary + engineering appendix")


# ---------------------------------------------------------------------------
# monte-carlo command  (Phase C)
# ---------------------------------------------------------------------------

@app.command(name="monte-carlo")
def monte_carlo(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to config JSON/YAML file",
        exists=True, readable=True, resolve_path=True,
    ),
    n_runs: int = typer.Option(200, "--n-runs", "-n", help="Number of MC runs"),
    seed: int = typer.Option(42, "--seed", "-s", help="Base random seed"),
    export: Path = typer.Option(
        Path("energex_output"),
        "--export", "-e",
        help="Output directory for MC results",
    ),
    percentiles: str = typer.Option(
        "10,25,50,75,90,99",
        "--percentiles",
        help="Comma-separated percentile list (e.g. '10,50,90')",
    ),
) -> None:
    """Run Monte Carlo uncertainty analysis (P10/P50/P90/P99 confidence bands)."""
    from energex.adapters.config_loader import load_config
    from energex.engine.monte_carlo import MonteCarloEngine
    from energex.domain.schemas import MonteCarloConfig
    from energex.adapters.exporters import export_monte_carlo
    from energex.engine.runner import run_grid_only_baseline

    console.rule("[bold cyan]EnergeX — Monte Carlo Analysis[/bold cyan]")

    with console.status("Loading config..."):
        cfg = load_config(config)

    pct_list = [float(p.strip()) for p in percentiles.split(",")]

    mc_cfg = MonteCarloConfig(
        n_runs=n_runs,
        base_seed=seed,
        percentiles=pct_list,
    )

    console.print(f"  Runs       : {n_runs}")
    console.print(f"  Base seed  : {seed}")
    console.print(f"  Percentiles: {pct_list}")
    console.print()

    with console.status("Computing baseline..."):
        try:
            bl = run_grid_only_baseline(cfg, seed=seed)
            baseline_npv = bl.npv
        except Exception:
            baseline_npv = None

    with console.status(f"Running {n_runs} Monte Carlo simulations..."):
        engine = MonteCarloEngine(cfg, mc_config=mc_cfg, baseline_npv=baseline_npv)
        mc_result = engine.run()

    # Display summary
    console.print()
    console.rule("[bold]Monte Carlo Results[/bold]")
    mc_table = Table(box=box.SIMPLE, show_header=True)
    mc_table.add_column("Metric", style="cyan")
    mc_table.add_column("Mean", justify="right")
    mc_table.add_column("P50", justify="right")
    mc_table.add_column("P90", justify="right")
    mc_table.add_column("P99", justify="right")
    mc_table.add_column("Unit")

    for stats in [mc_result.ens_stats, mc_result.downtime_stats, mc_result.continuity_stats]:
        if stats:
            mc_table.add_row(
                stats.metric,
                f"{stats.mean:.2f}",
                f"{stats.p50:.2f}",
                f"{stats.p90:.2f}",
                f"{stats.p99:.2f}",
                stats.unit,
            )
    console.print(mc_table)

    sla_color = "green" if mc_result.sla_pass_rate_pct >= 90 else "yellow" if mc_result.sla_pass_rate_pct >= 50 else "red"
    console.print(f"  SLA Pass Rate: [{sla_color}]{mc_result.sla_pass_rate_pct:.1f}%[/{sla_color}]")

    # Export
    export.mkdir(parents=True, exist_ok=True)
    mc_path = export_monte_carlo(mc_result, export)
    console.print(f"\n[green]✓[/green] MC results → {mc_path}")


# ---------------------------------------------------------------------------
# optimize command  (Phase C)
# ---------------------------------------------------------------------------

@app.command()
def optimize(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to config JSON/YAML file",
        exists=True, readable=True, resolve_path=True,
    ),
    bess_max: float = typer.Option(500.0, "--bess-max", help="Max BESS capacity to sweep (kWh)"),
    bess_step: float = typer.Option(50.0, "--bess-step", help="BESS step size (kWh)"),
    include_dg: bool = typer.Option(False, "--include-dg/--no-dg", help="Include DG in sweep"),
    include_solar: bool = typer.Option(False, "--include-solar/--no-solar", help="Include Solar in sweep"),
    n_years: int = typer.Option(3, "--n-years", help="Years per candidate evaluation"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    export: Path = typer.Option(
        Path("energex_output"),
        "--export", "-e",
        help="Output directory for optimization results",
    ),
) -> None:
    """Find optimal BESS/DG/Solar sizing using grid sweep + Pareto frontier."""
    from energex.adapters.config_loader import load_config
    from energex.engine.optimizer import SizingOptimizer
    from energex.domain.schemas import SizingBounds
    from energex.adapters.exporters import export_optimizer

    console.rule("[bold cyan]EnergeX — Sizing Optimizer[/bold cyan]")

    with console.status("Loading config..."):
        cfg = load_config(config)

    bounds = SizingBounds(
        bess_capacity_max_kwh=bess_max,
        bess_capacity_step_kwh=bess_step,
        include_dg=include_dg,
        include_solar=include_solar,
        n_years_per_eval=n_years,
    )

    n_candidates = len([
        (b, d, s)
        for b in [bounds.bess_capacity_min_kwh + i * bess_step for i in range(int((bess_max - 0) / bess_step) + 2)]
        for d in ([0, 50, 100, 150, 200, 250, 300] if include_dg else [0])
        for s in ([0, 50, 100, 150, 200, 250, 300] if include_solar else [0])
    ])
    console.print(f"  BESS sweep : 0 → {bess_max:.0f} kWh step {bess_step:.0f}")
    console.print(f"  DG sweep   : {'Yes' if include_dg else 'No'}")
    console.print(f"  Solar sweep: {'Yes' if include_solar else 'No'}")
    console.print(f"  Years/eval : {n_years}")
    console.print()

    with console.status("Running optimizer (this may take a minute)..."):
        optimizer = SizingOptimizer(cfg, bounds=bounds, base_seed=seed)
        opt_result = optimizer.run()

    # Display results
    console.print()
    console.rule("[bold]Optimization Results[/bold]")
    console.print(f"  Candidates evaluated  : {opt_result.n_candidates_evaluated}")
    console.print(f"  SLA-feasible configs  : {opt_result.n_sla_feasible}")
    console.print(f"  Pareto frontier points: {len(opt_result.pareto_points)}")

    if opt_result.optimal_point:
        opt = opt_result.optimal_point
        console.print()
        console.print(Panel(
            f"[bold green]OPTIMAL CONFIGURATION[/bold green]\n"
            f"  BESS: {opt.bess_kwh:.0f} kWh / {opt.bess_kw:.0f} kW\n"
            f"  DG  : {opt.dg_kw:.0f} kW\n"
            f"  Solar: {opt.solar_kwp:.0f} kWp\n"
            f"  CAPEX: ₹{opt.capex_rs:,.0f}\n"
            f"  Avg ENS: {opt.avg_ens_kwh:.1f} kWh/yr\n"
            f"  Continuity: {opt.avg_continuity_pct:.4f}%",
            expand=False,
        ))

    # Export
    export.mkdir(parents=True, exist_ok=True)
    files = export_optimizer(opt_result, export)
    console.print()
    for name, path in files.items():
        console.print(f"  [green]✓[/green] {name:30s} → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
