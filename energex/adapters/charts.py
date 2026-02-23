"""
EnergeX — Chart Generators (Phase B + Phase C)

Two chart families:
  1. Plotly (interactive) — used in Streamlit UI
  2. Matplotlib (static PNG) — embedded in PDF reports

All plotly functions return go.Figure.
All matplotlib functions write to an Axes and return a BytesIO PNG for PDF.

Phase C additions:
  - plotly_mc_distribution  — histogram + P50/P90/P99 bands for Monte Carlo
  - plotly_mc_bands_bar     — side-by-side P50/P90/P99 bar chart
  - plotly_sizing_scatter   — CAPEX vs NPV scatter for sizing candidates
  - plotly_sizing_table_fig — ranking table for sizing candidates
  - plotly_tou_profile      — hourly TOU rate profile heatmap
  - plotly_demand_charge_monthly — monthly peak demand vs demand charge
"""

from __future__ import annotations

import io
import math
from typing import Optional

import numpy as np

# Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Matplotlib (for PDF)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure as MplFigure

from energex.engine.dispatch_backup import SimulationResult, HourlyResult
from energex.engine.finance_engine import FinanceResult, SensitivityResult
from energex.engine.outage_generator import OutageEvent


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "grid": "#2196F3",
    "solar": "#FFC107",
    "bess": "#4CAF50",
    "dg": "#FF5722",
    "unserved": "#F44336",
    "bess_charge": "#81C784",
    "outage": "#EF9A9A",
    "positive": "#43A047",
    "negative": "#E53935",
    "neutral": "#90A4AE",
    "bg": "#F8F9FA",
}


# ============================================================
# Plotly Charts
# ============================================================

def plotly_energy_flows_area(
    sim: SimulationResult,
    sample_hours: int = 8760,
) -> go.Figure:
    """
    Stacked area chart of hourly energy flows.
    Sampled to first `sample_hours` hours for display.
    """
    n = min(sample_hours, len(sim.hourly))
    hrs = list(range(n))
    h_data = sim.hourly[:n]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hrs, y=[h.grid_to_load for h in h_data],
        name="Grid → Load", stackgroup="load",
        line=dict(width=0), fillcolor=COLORS["grid"],
        mode="lines", hovertemplate="%{y:.1f} kWh<extra>Grid→Load</extra>",
    ))
    if any(h.solar_to_load > 0 for h in h_data):
        fig.add_trace(go.Scatter(
            x=hrs, y=[h.solar_to_load for h in h_data],
            name="Solar → Load", stackgroup="load",
            line=dict(width=0), fillcolor=COLORS["solar"],
            mode="lines", hovertemplate="%{y:.1f} kWh<extra>Solar→Load</extra>",
        ))
    if any(h.bess_to_load > 0 for h in h_data):
        fig.add_trace(go.Scatter(
            x=hrs, y=[h.bess_to_load for h in h_data],
            name="BESS → Load", stackgroup="load",
            line=dict(width=0), fillcolor=COLORS["bess"],
            mode="lines", hovertemplate="%{y:.1f} kWh<extra>BESS→Load</extra>",
        ))
    if any(h.dg_to_load > 0 for h in h_data):
        fig.add_trace(go.Scatter(
            x=hrs, y=[h.dg_to_load for h in h_data],
            name="DG → Load", stackgroup="load",
            line=dict(width=0), fillcolor=COLORS["dg"],
            mode="lines", hovertemplate="%{y:.1f} kWh<extra>DG→Load</extra>",
        ))
    if any(h.unserved_critical > 0 for h in h_data):
        fig.add_trace(go.Scatter(
            x=hrs, y=[h.unserved_critical for h in h_data],
            name="Unserved Critical", stackgroup="load",
            line=dict(width=0), fillcolor=COLORS["unserved"],
            mode="lines", hovertemplate="%{y:.1f} kWh<extra>Unserved</extra>",
        ))

    # Critical load line
    fig.add_trace(go.Scatter(
        x=hrs, y=[h.critical_load_kwh for h in h_data],
        name="Critical Load", line=dict(color="black", width=1.5, dash="dash"),
        mode="lines", hovertemplate="%{y:.1f} kW<extra>Critical Load</extra>",
    ))

    fig.update_layout(
        title="Hourly Energy Flows",
        xaxis_title="Hour of Year",
        yaxis_title="Energy (kWh/hr)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=450,
        plot_bgcolor=COLORS["bg"],
    )
    return fig


def plotly_bess_soc(sim: SimulationResult, bess_capacity: float = 0.0) -> go.Figure:
    """BESS state-of-charge over the year."""
    hrs = list(range(len(sim.hourly)))
    soc_vals = [h.bess_soc_end for h in sim.hourly]
    outage_mask = [h.outage_fraction > 0.5 for h in sim.hourly]

    # Shade outage hours
    fig = go.Figure()

    # Outage shading
    in_outage = False
    outage_start = 0
    shapes = []
    for i, is_out in enumerate(outage_mask + [False]):
        if is_out and not in_outage:
            outage_start = i
            in_outage = True
        elif not is_out and in_outage:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=outage_start, x1=i, y0=0, y1=1,
                fillcolor="rgba(239,154,154,0.3)", line_width=0,
            ))
            in_outage = False

    fig.add_trace(go.Scatter(
        x=hrs, y=soc_vals,
        name="BESS SOC", line=dict(color=COLORS["bess"], width=1.5),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.15)",
        mode="lines", hovertemplate="%{y:.1f} kWh<extra>SOC</extra>",
    ))

    if bess_capacity > 0:
        fig.add_hline(
            y=bess_capacity * 0.1,
            line_dash="dot", line_color="orange",
            annotation_text="Reserve (10%)",
            annotation_position="right",
        )

    fig.update_layout(
        title="BESS State of Charge",
        xaxis_title="Hour of Year",
        yaxis_title="SOC (kWh)",
        shapes=shapes,
        height=350,
        plot_bgcolor=COLORS["bg"],
    )
    return fig


def plotly_outage_timeline(
    outage_events: list[OutageEvent],
    total_hours: int = 8760,
) -> go.Figure:
    """Gantt-style outage timeline for one year."""
    if not outage_events:
        fig = go.Figure()
        fig.add_annotation(text="No outage events", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font_size=16)
        fig.update_layout(title="Outage Timeline", height=200)
        return fig

    starts = [ev.start_hour for ev in outage_events]
    durations = [ev.effective_duration for ev in outage_events]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=durations,
        y=["Outages"] * len(outage_events),
        base=starts,
        orientation="h",
        name="Outage",
        marker_color=COLORS["unserved"],
        hovertemplate="Start: %{base:.1f} h<br>Duration: %{x:.2f} h<extra></extra>",
    ))
    fig.update_layout(
        title=f"Outage Events Timeline ({len(outage_events)} events)",
        xaxis_title="Hour of Year",
        xaxis_range=[0, total_hours],
        height=200,
        plot_bgcolor=COLORS["bg"],
    )
    return fig


def plotly_cashflows(finance: FinanceResult) -> go.Figure:
    """Stacked bar chart of annual cashflows."""
    years = [cf.year for cf in finance.cashflows]
    grid_costs = [-cf.grid_energy_cost for cf in finance.cashflows]
    fuel_costs = [-cf.diesel_fuel_cost for cf in finance.cashflows]
    om_costs = [-(cf.om_bess + cf.om_solar + cf.om_dg + cf.om_dg_variable) for cf in finance.cashflows]
    outage_costs = [-cf.outage_cost for cf in finance.cashflows]
    capex = [-cf.capex - cf.replacement_cost for cf in finance.cashflows]
    salvage = [cf.salvage for cf in finance.cashflows]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=capex, name="CAPEX / Replacement",
                         marker_color=COLORS["negative"]))
    fig.add_trace(go.Bar(x=years, y=grid_costs, name="Grid Energy",
                         marker_color=COLORS["grid"]))
    fig.add_trace(go.Bar(x=years, y=fuel_costs, name="Diesel Fuel",
                         marker_color=COLORS["dg"]))
    fig.add_trace(go.Bar(x=years, y=om_costs, name="O&M",
                         marker_color=COLORS["neutral"]))
    fig.add_trace(go.Bar(x=years, y=outage_costs, name="Outage Cost",
                         marker_color=COLORS["unserved"]))
    fig.add_trace(go.Bar(x=years, y=salvage, name="Salvage",
                         marker_color=COLORS["positive"]))

    fig.update_layout(
        title="Annual Cashflows",
        xaxis_title="Year",
        yaxis_title="₹",
        barmode="relative",
        height=400,
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def plotly_cumulative_cashflow(finance: FinanceResult) -> go.Figure:
    """Cumulative cashflow / payback chart."""
    years = [cf.year for cf in finance.cashflows]
    cumulative = []
    total = 0.0
    for cf in finance.cashflows:
        total += cf.net_cashflow
        cumulative.append(total)

    color = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in cumulative]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=cumulative,
        name="Cumulative Cashflow",
        marker_color=color,
        hovertemplate="Year %{x}: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1.5)
    fig.update_layout(
        title="Cumulative Cashflow (Payback Curve)",
        xaxis_title="Year",
        yaxis_title="Cumulative ₹",
        height=350,
        plot_bgcolor=COLORS["bg"],
    )
    return fig


def plotly_npv_comparison(results: dict[str, FinanceResult]) -> go.Figure:
    """Bar chart comparing NPV across scenarios."""
    scenarios = list(results.keys())
    npvs = [results[s].npv for s in scenarios]
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in npvs]

    fig = go.Figure(go.Bar(
        x=scenarios, y=npvs,
        marker_color=colors,
        hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
        text=[f"₹{v/1e6:.1f}M" for v in npvs],
        textposition="auto",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title="NPV Comparison by Scenario",
        yaxis_title="NPV (₹)",
        height=400,
        plot_bgcolor=COLORS["bg"],
    )
    return fig


def plotly_sensitivity_tornado(
    sensitivity_results: list[SensitivityResult],
    metric: str = "npv",
) -> go.Figure:
    """
    Tornado chart for sensitivity analysis.

    Parameters
    ----------
    sensitivity_results : list[SensitivityResult]
    metric : "npv" or "ens"
    """
    from collections import defaultdict

    # Group by parameter, find +/- range
    grouped: dict[str, dict] = defaultdict(lambda: {"pos": 0.0, "neg": 0.0})

    for sr in sensitivity_results:
        delta = sr.delta_npv if metric == "npv" else sr.delta_ens_kwh
        if sr.delta_pct > 0:
            grouped[sr.parameter]["pos"] = delta
        else:
            grouped[sr.parameter]["neg"] = delta

    # Sort by total swing (abs range)
    params = sorted(
        grouped.keys(),
        key=lambda p: abs(grouped[p]["pos"]) + abs(grouped[p]["neg"]),
    )

    labels = {
        "saidi": "SAIDI (±50%)",
        "saifi": "SAIFI (±50%)",
        "voll_or_downtime_cost": "VoLL / Downtime Cost (±50%)",
        "diesel_price": "Diesel Price (±25%)",
        "battery_capex": "Battery CAPEX (±20%)",
        "dg_availability": "DG Availability (±5%)",
        "critical_load": "Critical Load (±20%)",
    }

    y_labels = [labels.get(p, p) for p in params]
    pos_vals = [grouped[p]["pos"] for p in params]
    neg_vals = [grouped[p]["neg"] for p in params]

    unit = "₹" if metric == "npv" else "kWh"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Upside (+Δ)",
        y=y_labels,
        x=pos_vals,
        orientation="h",
        marker_color=COLORS["positive"],
        hovertemplate=f"%{{y}}: %{{x:,.0f}} {unit}<extra>+Δ</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Downside (−Δ)",
        y=y_labels,
        x=neg_vals,
        orientation="h",
        marker_color=COLORS["negative"],
        hovertemplate=f"%{{y}}: %{{x:,.0f}} {unit}<extra>−Δ</extra>",
    ))

    title_metric = "NPV (₹)" if metric == "npv" else "ENS (kWh/yr)"
    fig.update_layout(
        title=f"Sensitivity Tornado — {title_metric}",
        xaxis_title=f"Δ {title_metric}",
        barmode="relative",
        height=max(300, len(params) * 60 + 100),
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.15),
    )
    fig.add_vline(x=0, line_color="black", line_width=1.5)
    return fig


def plotly_reliability_gauge(sim: SimulationResult) -> go.Figure:
    """Gauge chart for continuity percentage."""
    cont = sim.continuity_pct

    if cont >= 99.9:
        color = "green"
    elif cont >= 99.0:
        color = "orange"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=cont,
        title={"text": "Continuity (%)"},
        delta={"reference": 99.9, "increasing": {"color": "green"}},
        gauge={
            "axis": {"range": [95, 100], "tickformat": ".2f"},
            "bar": {"color": color},
            "steps": [
                {"range": [95, 99], "color": "#FFCDD2"},
                {"range": [99, 99.9], "color": "#FFF9C4"},
                {"range": [99.9, 100], "color": "#C8E6C9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 99.9,
            },
        },
        number={"suffix": "%", "valueformat": ".4f"},
    ))
    fig.update_layout(height=280)
    return fig


def plotly_monthly_ens(sim: SimulationResult) -> go.Figure:
    """Bar chart of ENS per month."""
    HOURS_PER_MONTH = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_ens = [0.0] * 12
    monthly_downtime = [0.0] * 12
    h = 0
    for m, hrs in enumerate(HOURS_PER_MONTH):
        for hour in sim.hourly[h: h + hrs]:
            monthly_ens[m] += hour.unserved_critical
            monthly_downtime[m] += hour.downtime_hours
        h += hrs

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=MONTHS, y=monthly_ens,
        name="ENS (kWh)",
        marker_color=COLORS["unserved"],
        hovertemplate="%{x}: %{y:.1f} kWh<extra>ENS</extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=MONTHS, y=monthly_downtime,
        name="Downtime (hrs)",
        line=dict(color=COLORS["dg"], width=2),
        mode="lines+markers",
        hovertemplate="%{x}: %{y:.2f} hrs<extra>Downtime</extra>",
    ), secondary_y=True)

    fig.update_layout(
        title="Monthly ENS & Downtime Distribution",
        height=350,
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(title_text="ENS (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Downtime (hrs)", secondary_y=True)
    return fig


# ============================================================
# Matplotlib Charts (for PDF embedding)
# ============================================================

def _mpl_figure(width: float = 8.0, height: float = 4.0) -> tuple[MplFigure, any]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def mpl_cashflows_bar(finance: FinanceResult) -> bytes:
    """Stacked bar cashflow chart as PNG bytes."""
    fig, ax = _mpl_figure(8, 4)
    years = [cf.year for cf in finance.cashflows]

    grid_costs = [cf.grid_energy_cost for cf in finance.cashflows]
    fuel_costs = [cf.diesel_fuel_cost for cf in finance.cashflows]
    capex = [cf.capex + cf.replacement_cost for cf in finance.cashflows]
    outage_costs = [cf.outage_cost for cf in finance.cashflows]
    om = [cf.om_bess + cf.om_solar + cf.om_dg + cf.om_dg_variable for cf in finance.cashflows]

    bottoms = np.zeros(len(years))
    for vals, label, color in [
        (capex, "CAPEX/Replacement", "#E53935"),
        (grid_costs, "Grid Energy", "#2196F3"),
        (fuel_costs, "Diesel Fuel", "#FF5722"),
        (om, "O&M", "#90A4AE"),
        (outage_costs, "Outage Cost", "#F44336"),
    ]:
        if any(v > 0 for v in vals):
            ax.bar(years, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
            bottoms += np.array(vals)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Cost (₹)", fontsize=10)
    ax.set_title("Annual Cost Breakdown", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    return _fig_to_bytes(fig)


def mpl_cumulative_cashflow(finance: FinanceResult) -> bytes:
    """Cumulative cashflow chart as PNG bytes."""
    fig, ax = _mpl_figure(8, 3.5)
    years = [cf.year for cf in finance.cashflows]
    cumulative = []
    total = 0.0
    for cf in finance.cashflows:
        total += cf.net_cashflow
        cumulative.append(total)

    ax.fill_between(years, 0, cumulative,
                    where=[v >= 0 for v in cumulative],
                    color="#43A047", alpha=0.3, label="Positive")
    ax.fill_between(years, 0, cumulative,
                    where=[v < 0 for v in cumulative],
                    color="#E53935", alpha=0.3, label="Negative")
    ax.plot(years, cumulative, color="#1A237E", linewidth=2, zorder=3)
    ax.axhline(0, color="black", linewidth=1.5)

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Cumulative (₹)", fontsize=10)
    ax.set_title("Cumulative Cashflow — Payback Curve", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    return _fig_to_bytes(fig)


def mpl_tornado(sensitivity_results: list[SensitivityResult], metric: str = "npv") -> bytes:
    """Tornado chart as PNG bytes for PDF."""
    if not sensitivity_results:
        fig, ax = _mpl_figure(8, 3)
        ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return _fig_to_bytes(fig)

    from collections import defaultdict
    grouped: dict[str, dict] = defaultdict(lambda: {"pos": 0.0, "neg": 0.0})
    for sr in sensitivity_results:
        delta = sr.delta_npv if metric == "npv" else sr.delta_ens_kwh
        if sr.delta_pct > 0:
            grouped[sr.parameter]["pos"] = delta
        else:
            grouped[sr.parameter]["neg"] = delta

    labels = {
        "saidi": "SAIDI (±50%)",
        "saifi": "SAIFI (±50%)",
        "voll_or_downtime_cost": "VoLL / Downtime (±50%)",
        "diesel_price": "Diesel Price (±25%)",
        "battery_capex": "Battery CAPEX (±20%)",
        "dg_availability": "DG Availability (±5%)",
        "critical_load": "Critical Load (±20%)",
    }

    params = sorted(
        grouped.keys(),
        key=lambda p: abs(grouped[p]["pos"]) + abs(grouped[p]["neg"]),
    )
    y_labels = [labels.get(p, p) for p in params]
    pos_vals = [grouped[p]["pos"] for p in params]
    neg_vals = [grouped[p]["neg"] for p in params]

    fig, ax = _mpl_figure(8, max(3.5, len(params) * 0.6 + 1.5))
    y_pos = range(len(params))

    ax.barh(list(y_pos), pos_vals, color="#43A047", alpha=0.85, label="+Δ")
    ax.barh(list(y_pos), neg_vals, color="#E53935", alpha=0.85, label="−Δ")
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(y_labels, fontsize=9)
    unit = "₹" if metric == "npv" else "kWh"
    ax.set_xlabel(f"Δ {'NPV' if metric == 'npv' else 'ENS'} ({unit})", fontsize=10)
    ax.set_title(f"Sensitivity Tornado — {'NPV' if metric == 'npv' else 'ENS'}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M" if abs(x) >= 1e6 else f"₹{x/1e3:.0f}k"))
    plt.tight_layout()
    return _fig_to_bytes(fig)


def mpl_energy_pie(sim: SimulationResult) -> bytes:
    """Pie chart of energy supply mix for PDF."""
    fig, ax = _mpl_figure(5, 4)
    labels, sizes, colors = [], [], []

    if sim.total_grid_kwh > 0:
        labels.append(f"Grid\n{sim.total_grid_kwh/1000:.0f} MWh")
        sizes.append(sim.total_grid_kwh)
        colors.append(COLORS["grid"])
    if sim.total_solar_kwh > 0:
        labels.append(f"Solar\n{sim.total_solar_kwh/1000:.0f} MWh")
        sizes.append(sim.total_solar_kwh)
        colors.append(COLORS["solar"])
    if sim.total_dg_kwh > 0:
        labels.append(f"DG\n{sim.total_dg_kwh/1000:.0f} MWh")
        sizes.append(sim.total_dg_kwh)
        colors.append(COLORS["dg"])
    if sim.total_bess_discharge_kwh > 0:
        labels.append(f"BESS\n{sim.total_bess_discharge_kwh/1000:.0f} MWh")
        sizes.append(sim.total_bess_discharge_kwh)
        colors.append(COLORS["bess"])

    if sizes:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
    ax.set_title("Annual Energy Supply Mix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _fig_to_bytes(fig)


def _fig_to_bytes(fig: MplFigure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ============================================================
# Phase C: Monte Carlo Charts
# ============================================================

def plotly_mc_distribution(
    trials: "np.ndarray",
    metric_name: str,
    unit: str = "",
    confidence_levels: Optional[list[float]] = None,
) -> go.Figure:
    """
    Histogram with P10/P50/P90/P99 vertical lines for a Monte Carlo metric.

    Parameters
    ----------
    trials : np.ndarray   per-trial values
    metric_name : str     e.g. "ENS (kWh/yr)"
    unit : str            e.g. "kWh/yr"
    confidence_levels : list[float] | None   percentiles to mark (0–1)
    """
    if len(trials) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No MC data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    cls = confidence_levels or [0.10, 0.50, 0.90, 0.99]
    pct_vals = {int(p * 100): float(np.percentile(trials, p * 100)) for p in cls}

    pct_colors = {10: "blue", 50: "green", 90: "orange", 99: "red"}
    pct_dash = {10: "dash", 50: "solid", 90: "dash", 99: "dot"}

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trials,
        nbinsx=40,
        name="Trials",
        marker_color="#90CAF9",
        opacity=0.75,
    ))

    for pct, val in pct_vals.items():
        fig.add_vline(
            x=val,
            line_color=pct_colors.get(pct, "gray"),
            line_dash=pct_dash.get(pct, "dash"),
            line_width=2.5,
            annotation_text=f"P{pct}: {val:.1f}",
            annotation_position="top right",
            annotation_font_size=11,
        )

    fig.update_layout(
        title=f"Monte Carlo Distribution — {metric_name}",
        xaxis_title=unit or metric_name,
        yaxis_title="Number of Trials",
        height=360,
        plot_bgcolor=COLORS["bg"],
        showlegend=False,
    )
    return fig


def plotly_mc_bands_bar(mc_result: "MonteCarloResult") -> go.Figure:
    """
    Side-by-side bar chart comparing P10/P50/P90/P99 for key metrics.
    Uses a 2-row subplot: reliability (ENS, downtime) and financial (NPV, outage cost).
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Reliability Bands", "Financial Bands"),
    )

    metrics_rel = [
        ("ENS P50", mc_result.ens_p50, "kWh/yr", "#4CAF50"),
        ("ENS P90", mc_result.ens_p90, "kWh/yr", "#FF9800"),
        ("ENS P99", mc_result.ens_p99, "kWh/yr", "#F44336"),
        ("Downtime P50", mc_result.downtime_p50 * 60, "min/yr", "#90A4AE"),
        ("Downtime P90", mc_result.downtime_p90 * 60, "min/yr", "#607D8B"),
    ]

    labels_rel = [m[0] for m in metrics_rel]
    vals_rel = [m[1] for m in metrics_rel]
    clrs_rel = [m[3] for m in metrics_rel]

    fig.add_trace(
        go.Bar(x=labels_rel, y=vals_rel, marker_color=clrs_rel,
               name="Reliability", hovertemplate="%{x}: %{y:.2f}<extra></extra>"),
        row=1, col=1,
    )

    metrics_fin = [
        ("NPV P10", mc_result.npv_p10 / 1e6, "₹M"),
        ("NPV P50", mc_result.npv_p50 / 1e6, "₹M"),
        ("NPV P90", mc_result.npv_p90 / 1e6, "₹M"),
        ("OC P50", mc_result.outage_cost_p50 / 1e6, "₹M"),
        ("OC P90", mc_result.outage_cost_p90 / 1e6, "₹M"),
    ]
    labels_fin = [m[0] for m in metrics_fin]
    vals_fin = [m[1] for m in metrics_fin]
    fin_colors = [
        COLORS["negative"] if "P10" in l else
        COLORS["positive"] if "P90" in l or "P50" in l else COLORS["neutral"]
        for l in labels_fin
    ]

    fig.add_trace(
        go.Bar(x=labels_fin, y=vals_fin, marker_color=fin_colors,
               name="Financial", hovertemplate="%{x}: ₹%{y:.2f}M<extra></extra>"),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Monte Carlo Confidence Bands ({mc_result.n_trials} trials)",
        height=380,
        plot_bgcolor=COLORS["bg"],
        showlegend=False,
    )
    return fig


# ============================================================
# Phase C: Sizing Charts
# ============================================================

def plotly_sizing_scatter(candidates: list) -> go.Figure:
    """
    CAPEX vs Incremental NPV scatter plot for sizing candidates.
    Bubble size = backup autonomy. Color = SLA pass/fail.
    """
    if not candidates:
        fig = go.Figure()
        fig.add_annotation(text="No sizing results", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    x = [c.capex_rs / 1e6 for c in candidates]
    y = [(c.incremental_npv or c.npv) / 1e6 for c in candidates]
    sizes = [max(8, min(40, c.backup_autonomy_hrs * 4)) for c in candidates]
    colors = [COLORS["positive"] if c.sla_pass else COLORS["negative"] for c in candidates]
    labels = [c.label for c in candidates]
    ranks = [c.rank for c in candidates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color="white"),
        ),
        text=[f"#{r}" for r in ranks],
        textposition="top center",
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "CAPEX: ₹%{x:.2f}M<br>"
            "Incr. NPV: ₹%{y:.2f}M<extra></extra>"
        ),
        customdata=labels,
    ))

    fig.add_hline(y=0, line_color="black", line_width=1.5, line_dash="dash",
                  annotation_text="NPV break-even", annotation_position="right")

    fig.update_layout(
        title="Sizing Candidates — CAPEX vs Incremental NPV",
        xaxis_title="Total CAPEX (₹M)",
        yaxis_title="Incremental NPV vs Grid-Only (₹M)",
        height=480,
        plot_bgcolor=COLORS["bg"],
    )
    return fig


def plotly_sizing_pareto(candidates: list) -> go.Figure:
    """
    Pareto chart: sorted candidates by incremental NPV with CAPEX overlay.
    Shows top-10 candidates for easy comparison.
    """
    if not candidates:
        fig = go.Figure()
        fig.add_annotation(text="No sizing results", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    top = candidates[:min(10, len(candidates))]
    labels = [c.label for c in top]
    incr_npv = [(c.incremental_npv or c.npv) / 1e6 for c in top]
    capex = [c.capex_rs / 1e6 for c in top]
    ens = [c.ens_kwh_yr for c in top]
    sla_colors = [COLORS["positive"] if c.sla_pass else COLORS["negative"] for c in top]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=labels, y=incr_npv,
        name="Incremental NPV (₹M)",
        marker_color=sla_colors,
        hovertemplate="%{x}<br>Incr. NPV: ₹%{y:.2f}M<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=capex,
        name="CAPEX (₹M)",
        line=dict(color="#1A237E", width=2),
        mode="lines+markers",
        marker=dict(size=8),
        hovertemplate="%{x}<br>CAPEX: ₹%{y:.2f}M<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title="Top Sizing Candidates — Incremental NPV vs CAPEX",
        height=420,
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.25),
        xaxis_tickangle=-30,
    )
    fig.update_yaxes(title_text="Incremental NPV (₹M)", secondary_y=False)
    fig.update_yaxes(title_text="CAPEX (₹M)", secondary_y=True)
    return fig


def plotly_sizing_reliability(candidates: list) -> go.Figure:
    """
    ENS vs Downtime scatter for all sizing candidates.
    Helps users understand the reliability trade-off.
    """
    if not candidates:
        fig = go.Figure()
        fig.add_annotation(text="No sizing results", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    x = [c.ens_kwh_yr for c in candidates]
    y = [c.downtime_hrs_yr * 60 for c in candidates]  # convert to minutes
    colors = [COLORS["positive"] if c.sla_pass else COLORS["negative"] for c in candidates]
    labels = [f"#{c.rank} {c.label}" for c in candidates]

    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color=colors, size=12, opacity=0.85,
                    line=dict(width=1, color="white")),
        hovertemplate="%{customdata}<br>ENS: %{x:.1f} kWh/yr<br>Downtime: %{y:.1f} min/yr<extra></extra>",
        customdata=labels,
    ))

    fig.update_layout(
        title="Sizing Trade-off — Reliability (lower = better)",
        xaxis_title="ENS (kWh/yr)",
        yaxis_title="Downtime (min/yr)",
        height=380,
        plot_bgcolor=COLORS["bg"],
    )
    return fig


# ============================================================
# Phase C: TOU / Demand Charge Charts
# ============================================================

def plotly_tou_rate_profile(config) -> go.Figure:
    """
    Heatmap showing hourly TOU rates across a week (Mon–Sun).
    Helps visualize the TOU structure.
    """
    grid = config.grid

    if not grid.tou_blocks:
        fig = go.Figure()
        fig.add_annotation(
            text="Flat tariff — no TOU blocks configured",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=14,
        )
        fig.update_layout(title="Tariff Profile", height=200)
        return fig

    hours = list(range(24))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    z = []
    for d in range(7):
        is_weekday = d < 5
        row = []
        for h in hours:
            # Simulate an arbitrary hour of year to get the rate
            # day_of_week computed from hour // 24 % 7
            rate = grid.energy_rate_rs_kwh
            if grid.tou_blocks:
                for block in grid.tou_blocks:
                    if block.applies_to_hour(h, is_weekday):
                        rate = block.rate_rs_kwh
                        break
            row.append(rate)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{h:02d}:00" for h in hours],
        y=days,
        colorscale="RdYlGn_r",
        text=[[f"₹{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="₹/kWh"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Rate: ₹%{z:.2f}/kWh<extra></extra>",
    ))

    fig.update_layout(
        title="TOU Rate Profile (₹/kWh per hour × day)",
        xaxis_title="Hour of Day",
        yaxis_title="Day",
        height=320,
    )
    return fig


def plotly_demand_charge_monthly(sim: "SimulationResult", config) -> go.Figure:
    """
    Monthly peak demand bar chart with demand charge cost overlay.
    Illustrates the financial impact of demand charges per month.
    """
    dc_model = config.grid.demand_charge
    if dc_model is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No demand charge configured",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=14,
        )
        fig.update_layout(title="Monthly Demand Charges", height=200)
        return fig

    HOURS_PER_MONTH = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_peaks = []
    h = 0
    annual_peak = 0.0
    for hrs in HOURS_PER_MONTH:
        month_hrs = sim.hourly[h: h + hrs]
        if month_hrs:
            peak = max(hr.grid_to_load + hr.grid_to_bess for hr in month_hrs)
            monthly_peaks.append(peak)
            annual_peak = max(annual_peak, peak)
        else:
            monthly_peaks.append(0.0)
        h += hrs

    # Compute billable demand per month
    billable = []
    for pk in monthly_peaks:
        bd = pk
        if dc_model.ratchet_pct is not None:
            bd = max(bd, dc_model.ratchet_pct / 100.0 * annual_peak)
        if dc_model.contracted_demand_kw is not None:
            bd = max(bd, dc_model.contracted_demand_kw)
        billable.append(bd)

    charges = [b * dc_model.charge_rs_kw_month for b in billable]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=MONTHS, y=monthly_peaks,
        name="Peak Demand (kW)",
        marker_color=COLORS["grid"],
        hovertemplate="%{x}: %{y:.1f} kW<extra>Peak Demand</extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=MONTHS, y=[b - p for b, p in zip(billable, monthly_peaks)],
        name="Ratchet/Contract Floor",
        marker_color=COLORS["neutral"],
        base=monthly_peaks,
        hovertemplate="%{x}: +%{y:.1f} kW ratchet<extra>Ratchet</extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=MONTHS, y=charges,
        name="Demand Charge (₹)",
        line=dict(color=COLORS["negative"], width=2),
        mode="lines+markers",
        marker=dict(size=8),
        hovertemplate="%{x}: ₹%{y:,.0f}<extra>Demand Charge</extra>",
    ), secondary_y=True)

    fig.update_layout(
        title="Monthly Peak Demand & Demand Charges",
        height=380,
        plot_bgcolor=COLORS["bg"],
        barmode="stack",
        legend=dict(orientation="h", y=-0.25),
    )
    fig.update_yaxes(title_text="Peak Demand (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Demand Charge (₹/month)", secondary_y=True)
    return fig
