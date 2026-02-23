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
# size command  (Phase C)
# ---------------------------------------------------------------------------

@app.command()
def size(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to config JSON/YAML file",
        exists=True, readable=True, resolve_path=True,
    ),
    bess_max_kwh: float = typer.Option(
        500.0, "--bess-max", help="Max BESS capacity to evaluate (kWh)"
    ),
    dg_max_kw: float = typer.Option(
        200.0, "--dg-max", help="Max DG rated power to evaluate (kW)"
    ),
    steps_bess: int = typer.Option(
        6, "--bess-steps", help="Number of BESS size steps to evaluate"
    ),
    steps_dg: int = typer.Option(
        5, "--dg-steps", help="Number of DG size steps to evaluate"
    ),
    objective: str = typer.Option(
        "incremental_npv", "--objective", "-o",
        help="Ranking objective: incremental_npv | lcoe | capex | ens",
    ),
    max_capex_m: Optional[float] = typer.Option(
        None, "--max-capex-m", help="Max CAPEX ceiling (₹ millions). None = no limit."
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    export: Path = typer.Option(
        Path("energex_output"), "--export", "-e",
        help="Output directory",
    ),
    top_n: int = typer.Option(10, "--top-n", help="Number of top candidates to show"),
) -> None:
    """
    [Phase C] Automated optimal sizing — find the best BESS + DG combination.

    Sweeps a grid of BESS capacity × DG rated-power combinations,
    runs the full simulation for each, and ranks by the chosen objective.

    Example:
        energex size --config configs/grid_bess.json --bess-max 400 --dg-max 150
    """
    import json
    from energex.adapters.config_loader import load_config
    from energex.engine.sizing_optimizer import SizingOptimizer

    console.rule("[bold cyan]EnergeX — Sizing Optimizer[/bold cyan]")

    with console.status("Loading config..."):
        try:
            cfg = load_config(config)
        except Exception as e:
            console.print(f"[red]Config error: {e}[/red]")
            raise typer.Exit(1)

    # Build size grids
    import numpy as np
    bess_sizes = [0.0] + [round(v, 1) for v in np.linspace(
        max(50, bess_max_kwh / steps_bess), bess_max_kwh, steps_bess
    ).tolist()]
    dg_sizes = [0.0] + [round(v, 1) for v in np.linspace(
        max(50, dg_max_kw / steps_dg), dg_max_kw, steps_dg
    ).tolist()]

    # Override from config sizing block if present
    if cfg.sizing:
        bess_sizes = cfg.sizing.bess_sizes_kwh
        dg_sizes = cfg.sizing.dg_sizes_kw
        objective = cfg.sizing.optimize_for.value

    max_capex_rs = max_capex_m * 1e6 if max_capex_m is not None else None

    total = len(bess_sizes) * len(dg_sizes)
    console.print(f"  Evaluating [bold]{total}[/bold] combinations "
                  f"({len(bess_sizes)} BESS sizes × {len(dg_sizes)} DG sizes)")
    console.print(f"  BESS sizes : {bess_sizes}")
    console.print(f"  DG sizes   : {dg_sizes}")
    console.print(f"  Objective  : [bold yellow]{objective}[/bold yellow]")
    console.print()

    optimizer = SizingOptimizer(
        config=cfg,
        bess_sizes_kwh=bess_sizes,
        dg_sizes_kw=dg_sizes,
        optimize_for=objective,
        max_capex_rs=max_capex_rs,
        seed=seed,
    )

    done_count = 0
    with console.status("Running sizing grid search...") as status:
        def progress(p: float) -> None:
            nonlocal done_count
            done_count = int(p * total)
            status.update(f"Running sizing grid search... {done_count}/{total}")

        candidates = optimizer.run(progress_callback=progress)

    if not candidates:
        console.print("[red]No valid candidates found.[/red]")
        raise typer.Exit(1)

    console.print()
    console.rule("[bold green]Sizing Results[/bold green]")
    console.print(SizingOptimizer.recommendation_text(candidates))

    # Table of top N
    console.print()
    table = Table(title=f"Top {min(top_n, len(candidates))} Candidates", box=box.SIMPLE)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Configuration")
    table.add_column("CAPEX (₹M)", justify="right")
    table.add_column("Incr. NPV (₹M)", justify="right")
    table.add_column("Payback (yr)", justify="right")
    table.add_column("ENS (kWh/yr)", justify="right")
    table.add_column("SLA", justify="center")

    for c in candidates[:top_n]:
        incr_npv = f"{c.incremental_npv/1e6:.2f}" if c.incremental_npv else "N/A"
        pb = f"{c.payback_years:.1f}" if c.payback_years else "N/A"
        sla_str = "[green]PASS[/green]" if c.sla_pass else "[red]FAIL[/red]"
        table.add_row(
            str(c.rank),
            c.label,
            f"{c.capex_rs/1e6:.2f}",
            incr_npv,
            pb,
            f"{c.ens_kwh_yr:.1f}",
            sla_str,
        )
    console.print(table)

    # Export
    export.mkdir(parents=True, exist_ok=True)
    out_path = export / "sizing_results.json"
    out_path.write_text(
        json.dumps(
            {"candidates": [c.to_dict() for c in candidates]},
            indent=2, ensure_ascii=False,
        )
    )
    console.print(f"\n[green]✓[/green] Results exported to {out_path}")


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
    trials: int = typer.Option(
        200, "--trials", "-n",
        help="Number of Monte Carlo trials (min 50, recommended 200+)",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Base random seed"),
    export: Path = typer.Option(
        Path("energex_output"), "--export", "-e",
        help="Output directory",
    ),
) -> None:
    """
    [Phase C] Monte Carlo simulation — compute P50/P90/P99 confidence bands.

    Runs N independent trials with different seeds to build a distribution of
    ENS, downtime, NPV, and outage cost outcomes.

    Requires stochastic outage mode. Results are exported as JSON.

    Example:
        energex monte-carlo --config configs/grid_bess.json --trials 500
    """
    import json
    from energex.adapters.config_loader import load_config
    from energex.engine.monte_carlo import MonteCarloEngine
    from energex.engine.runner import run_grid_only_baseline

    console.rule("[bold cyan]EnergeX — Monte Carlo Analysis[/bold cyan]")

    with console.status("Loading config..."):
        try:
            cfg = load_config(config)
        except Exception as e:
            console.print(f"[red]Config error: {e}[/red]")
            raise typer.Exit(1)

    if cfg.outage.mode.value != "stochastic":
        console.print("[yellow]⚠ Config uses deterministic outages — MC will produce identical trials.[/yellow]")

    # Override n_trials from config if present
    if cfg.monte_carlo:
        trials = cfg.monte_carlo.n_trials

    console.print(f"  Trials     : [bold]{trials}[/bold]")
    console.print(f"  Scenario   : {cfg.scenario.value if cfg.scenario else 'auto'}")
    console.print(f"  Seed base  : {seed}")
    console.print()

    with console.status("Computing baseline NPV..."):
        try:
            bl = run_grid_only_baseline(cfg, seed=seed)
            baseline_npv = bl.npv
        except Exception:
            baseline_npv = None

    engine = MonteCarloEngine(config=cfg, n_trials=trials, base_seed=seed)

    done_count = 0
    with console.status(f"Running {trials} trials...") as status:
        def progress(p: float) -> None:
            nonlocal done_count
            done_count = int(p * trials)
            status.update(f"Running trial {done_count}/{trials} ({p*100:.0f}%)...")

        mc = engine.run(progress_callback=progress, baseline_npv=baseline_npv)

    console.print()
    console.rule("[bold green]Monte Carlo Results[/bold green]")

    mc_table = Table(box=box.SIMPLE, show_header=True)
    mc_table.add_column("Metric", style="cyan")
    mc_table.add_column("P10", justify="right")
    mc_table.add_column("P50 (median)", justify="right")
    mc_table.add_column("P90", justify="right")
    mc_table.add_column("P99", justify="right")

    mc_table.add_row(
        "ENS (kWh/yr)",
        f"{mc.ens_p10:.1f}",
        f"{mc.ens_p50:.1f}",
        f"{mc.ens_p90:.1f}",
        f"{mc.ens_p99:.1f}",
    )
    mc_table.add_row(
        "Downtime (hrs/yr)",
        "—",
        f"{mc.downtime_p50:.3f}",
        f"{mc.downtime_p90:.3f}",
        f"{mc.downtime_p99:.3f}",
    )
    mc_table.add_row(
        "Continuity (%)",
        f"{mc.continuity_p10:.3f}",
        f"{mc.continuity_p50:.3f}",
        "—", "—",
    )
    mc_table.add_row(
        "Project NPV (₹M)",
        f"{mc.npv_p10/1e6:.2f}",
        f"{mc.npv_p50/1e6:.2f}",
        f"{mc.npv_p90/1e6:.2f}",
        "—",
    )
    mc_table.add_row(
        "Outage Cost 25yr (₹M)",
        "—",
        f"{mc.outage_cost_p50/1e6:.2f}",
        f"{mc.outage_cost_p90/1e6:.2f}",
        f"{mc.outage_cost_p99/1e6:.2f}",
    )
    mc_table.add_row(
        "LCOE (₹/kWh)",
        "—",
        f"{mc.lcoe_p50:.2f}",
        f"{mc.lcoe_p90:.2f}",
        "—",
    )
    console.print(mc_table)
    console.print(f"\n  Trials completed: {len(mc.trials)}/{trials}")

    # Export
    export.mkdir(parents=True, exist_ok=True)
    out_path = export / "monte_carlo_results.json"
    summary = mc.summary_dict()
    summary["config_name"] = cfg.name
    summary["trials_completed"] = len(mc.trials)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(f"[green]✓[/green] Results exported to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
