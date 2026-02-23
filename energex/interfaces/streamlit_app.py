"""
EnergeX — Streamlit UI (Phase B + Phase C)

Interactive dashboard for Backup/Outage Energy ROI analysis.

Phase C additions (new tabs):
  - 📐 Sizing    — Automated optimal sizing engine (BESS + DG sweep)
  - 🎲 Monte Carlo — P50/P90/P99 confidence bands with histograms
  - 💡 TOU/Demand  — TOU tariff profile and monthly demand charge breakdown

Launch with:
    streamlit run energex/interfaces/streamlit_app.py

Or via CLI:
    energex ui
"""

from __future__ import annotations

import json
import sys
import io
import copy
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np

# Ensure the parent path is available for the energex package import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from energex.adapters.config_loader import load_config, load_config_dict
from energex.adapters.exporters import export_all
from energex.engine.runner import run_simulation, run_grid_only_baseline, RunResult
from energex.engine.finance_engine import SensitivityEngine
from energex.domain.schemas import ProjectConfig


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

def _page_config() -> None:
    st.set_page_config(
        page_title="EnergeX — Energy ROI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "result": None,
        "baseline_npv": None,
        "config_dict": None,
        "sensitivity_results": None,
        "run_seed": 42,
        # Phase C
        "mc_result": None,
        "sizing_candidates": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar — config upload + quick overrides
# ---------------------------------------------------------------------------

PRESETS = {
    "Grid + BESS (Commercial)": "configs/grid_bess.json",
    "Grid + DG (Industrial)": "configs/grid_dg.json",
    "Grid + Solar + BESS (EV Station)": "configs/grid_solar_bess.json",
    "DG + BESS Hybrid (Telecom)": "configs/dg_bess_hybrid.json",
    "Grid + BESS + TOU + Demand Charges": "configs/grid_bess_tou.json",
}


def _sidebar() -> Optional[dict]:
    """Render sidebar and return config dict (or None)."""
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/lightning-bolt.png",
        width=60,
    ) if False else st.sidebar.markdown("## ⚡ EnergeX")

    st.sidebar.markdown("## ⚡ EnergeX")
    st.sidebar.caption("Backup/Outage Energy ROI Calculator")
    st.sidebar.divider()

    # --- Config source ---
    config_source = st.sidebar.radio(
        "Config source",
        ["Upload JSON", "Use preset", "Paste JSON"],
        index=1,
    )

    raw_cfg: Optional[dict] = None
    base_dir = Path(__file__).parent.parent

    if config_source == "Use preset":
        preset = st.sidebar.selectbox("Preset scenario", list(PRESETS.keys()))
        preset_path = base_dir / PRESETS[preset]
        if preset_path.exists():
            with open(preset_path) as f:
                raw_cfg = json.load(f)
        else:
            st.sidebar.error(f"Preset file not found: {preset_path}")

    elif config_source == "Upload JSON":
        uploaded = st.sidebar.file_uploader("Upload config.json", type=["json"])
        if uploaded:
            raw_cfg = json.load(uploaded)
        else:
            st.sidebar.info("Upload a config JSON to proceed.")

    else:  # Paste JSON
        pasted = st.sidebar.text_area("Paste JSON config", height=200)
        if pasted:
            try:
                raw_cfg = json.loads(pasted)
            except json.JSONDecodeError as e:
                st.sidebar.error(f"Invalid JSON: {e}")

    if raw_cfg is None:
        return None

    # Remove comments key
    raw_cfg.pop("_comment", None)

    # --- Quick overrides ---
    st.sidebar.divider()
    st.sidebar.markdown("### Quick Overrides")

    with st.sidebar.expander("Load & Outage", expanded=True):
        avg_kw = st.slider(
            "Avg load (kW)",
            10.0, 1000.0,
            float(raw_cfg.get("load", {}).get("avg_kw", 100)),
            step=10.0,
        )
        raw_cfg.setdefault("load", {})["avg_kw"] = avg_kw
        raw_cfg.setdefault("load", {})["mode"] = raw_cfg.get("load", {}).get("mode", "simple")

        if raw_cfg.get("outage", {}).get("mode") == "stochastic":
            stoch = raw_cfg.get("outage", {}).get("stochastic", {})
            saidi = st.slider(
                "SAIDI (min/yr)",
                0.0, 5000.0,
                float(stoch.get("saidi_minutes_per_year", 600)),
                step=60.0,
            )
            saifi = st.slider(
                "SAIFI (events/yr)",
                0.0, 100.0,
                float(stoch.get("saifi_events_per_year", 10)),
                step=1.0,
            )
            stoch["saidi_minutes_per_year"] = saidi
            stoch["saifi_events_per_year"] = saifi

    if raw_cfg.get("bess"):
        with st.sidebar.expander("BESS", expanded=False):
            cap = st.slider(
                "Capacity (kWh)",
                10.0, 2000.0,
                float(raw_cfg["bess"].get("capacity_kwh", 100)),
                step=10.0,
            )
            pwr = st.slider(
                "Power (kW)",
                5.0, 1000.0,
                float(raw_cfg["bess"].get("power_kw", 50)),
                step=5.0,
            )
            raw_cfg["bess"]["capacity_kwh"] = cap
            raw_cfg["bess"]["power_kw"] = pwr

    oc = raw_cfg.get("outage_cost", {})
    with st.sidebar.expander("Outage Cost", expanded=False):
        if oc.get("mode") == "voll":
            voll = st.slider(
                "VoLL (₹/kWh)",
                10.0, 500.0,
                float(oc.get("voll_rs_kwh", 100)),
                step=10.0,
            )
            raw_cfg["outage_cost"]["voll_rs_kwh"] = voll
        else:
            dt_cost = st.slider(
                "Downtime cost (₹/hr)",
                1000.0, 500000.0,
                float(oc.get("downtime_cost_rs_hr", 50000)),
                step=1000.0,
            )
            raw_cfg["outage_cost"]["downtime_cost_rs_hr"] = dt_cost

    st.sidebar.divider()
    st.sidebar.markdown("### Simulation")
    st.session_state.run_seed = st.sidebar.number_input(
        "Random seed", value=st.session_state.run_seed, step=1
    )

    run_sensitivity = st.sidebar.checkbox("Run sensitivity analysis", value=False)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_btn = st.button("▶ Run", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("✖ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.result = None
        st.session_state.sensitivity_results = None
        st.session_state.mc_result = None
        st.session_state.sizing_candidates = None
        st.rerun()

    if run_btn:
        _run_simulation(raw_cfg, st.session_state.run_seed, run_sensitivity)

    return raw_cfg


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

def _run_simulation(cfg_dict: dict, seed: int, run_sensitivity: bool) -> None:
    """Execute simulation and store results in session state."""
    progress = st.sidebar.progress(0, text="Validating config...")

    try:
        cfg = load_config_dict(cfg_dict)
    except Exception as e:
        st.sidebar.error(f"Config error: {e}")
        return

    progress.progress(15, text="Computing baseline...")
    try:
        bl = run_grid_only_baseline(cfg, seed=seed)
        baseline_npv = bl.npv
    except Exception:
        baseline_npv = None

    progress.progress(30, text=f"Simulating {cfg.project_lifetime_years} years...")
    try:
        result = run_simulation(cfg, seed=seed, baseline_npv=baseline_npv)
    except Exception as e:
        st.sidebar.error(f"Simulation error: {e}")
        progress.empty()
        return

    progress.progress(85, text="Finishing...")

    if run_sensitivity:
        progress.progress(90, text="Running sensitivity...")
        try:
            def runner_fn(c: ProjectConfig):
                r = run_simulation(c, seed=seed)
                return r.finance, r.sim_year1

            sens_engine = SensitivityEngine(cfg, runner_fn)
            st.session_state.sensitivity_results = sens_engine.run()
        except Exception:
            st.session_state.sensitivity_results = None

    st.session_state.result = result
    st.session_state.baseline_npv = baseline_npv
    st.session_state.config_dict = cfg_dict

    progress.progress(100, text="Done!")
    progress.empty()
    st.rerun()


# ---------------------------------------------------------------------------
# Result display helpers
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, delta: str | None = None, good: bool = True) -> None:
    color = "#43A047" if good else "#E53935"
    st.markdown(
        f"""
        <div style="
            background: white;
            border-left: 4px solid {color};
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px">{label}</div>
            <div style="font-size: 22px; font-weight: 700; color: #1A237E; margin-top: 2px">{value}</div>
            {"<div style='font-size: 12px; color: " + color + "; margin-top: 2px'>" + delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sla_badge(passed: bool) -> str:
    if passed:
        return '<span style="background:#43A047;color:white;padding:4px 12px;border-radius:12px;font-weight:700">✓ SLA PASSED</span>'
    return '<span style="background:#E53935;color:white;padding:4px 12px;border-radius:12px;font-weight:700">✗ SLA FAILED</span>'


# ---------------------------------------------------------------------------
# Tab: Overview Dashboard
# ---------------------------------------------------------------------------

def _tab_overview(result: RunResult) -> None:
    from energex.adapters.charts import plotly_reliability_gauge, plotly_energy_flows_area

    sim = result.sim_year1
    fin = result.finance
    cfg = result.config

    # SLA badge
    st.markdown(_sla_badge(sim.sla_pass), unsafe_allow_html=True)
    if not sim.sla_pass:
        for f in sim.sla_failures:
            st.warning(f"⚠ {f}")
    st.markdown("")

    # Top metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _metric_card("ENS (kWh/yr)", f"{sim.ens_kwh:,.1f}", good=(sim.ens_kwh < 100))
    with c2:
        _metric_card("Downtime (hrs/yr)", f"{sim.downtime_hours:.2f}", good=(sim.downtime_hours < 10))
    with c3:
        _metric_card("Backup Autonomy", f"{sim.backup_autonomy_hours:.1f} hrs",
                     good=(sim.backup_autonomy_hours >= 4))
    with c4:
        _metric_card("Total CAPEX", f"₹{fin.total_capex/1e6:.2f}M")
    with c5:
        npv_good = fin.npv > -fin.total_capex * 0.5
        _metric_card("Project NPV", f"₹{fin.npv/1e6:.2f}M", good=npv_good)
    with c6:
        _metric_card("LCOE", f"₹{fin.lcoe_rs_kwh:.2f}/kWh")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.plotly_chart(
            plotly_energy_flows_area(sim, sample_hours=720),  # first month
            use_container_width=True,
        )
        st.caption("Showing first 720 hours (1 month). Full year available in Energy Flows tab.")

    with col_right:
        st.plotly_chart(plotly_reliability_gauge(sim), use_container_width=True)

        # Outage stats
        st.markdown("**Outage Statistics (Year 1)**")
        st.dataframe({
            "Metric": ["Total outage events", "Events fully served", "Outages served %",
                       "Outage events total"],
            "Value": [
                sim.outage_events_total,
                sim.outage_events_served,
                f"{sim.outage_served_pct:.1f}%",
                sim.outage_events_total,
            ],
        }, use_container_width=True, hide_index=True)

    # Warnings
    if result.warnings:
        with st.expander(f"⚠ {len(result.warnings)} Warning(s)", expanded=False):
            for w in result.warnings:
                st.warning(w)


# ---------------------------------------------------------------------------
# Tab: Reliability
# ---------------------------------------------------------------------------

def _tab_reliability(result: RunResult) -> None:
    from energex.adapters.charts import plotly_outage_timeline, plotly_monthly_ens

    sim = result.sim_year1

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plotly_monthly_ens(sim), use_container_width=True)
    with col2:
        st.plotly_chart(plotly_outage_timeline(result.outage_events), use_container_width=True)

    st.subheader("Reliability Metrics")
    metrics_data = {
        "Metric": [
            "Energy Not Served (ENS)",
            "Downtime (critical load unmet)",
            "Outage events (year 1)",
            "Events fully served",
            "Outage served %",
            "Backup autonomy at peak critical",
            "Continuity",
        ],
        "Value": [
            f"{sim.ens_kwh:.2f} kWh/yr",
            f"{sim.downtime_hours:.4f} hrs/yr",
            str(sim.outage_events_total),
            str(sim.outage_events_served),
            f"{sim.outage_served_pct:.2f}%",
            f"{sim.backup_autonomy_hours:.2f} hrs",
            f"{sim.continuity_pct:.4f}%",
        ],
        "SLA Target": [
            f"≤ {result.config.outage_cost.sla_max_ens_kwh_yr}" if result.config.outage_cost.sla_max_ens_kwh_yr else "—",
            f"≤ {result.config.outage_cost.sla_max_downtime_hrs_yr}" if result.config.outage_cost.sla_max_downtime_hrs_yr else "—",
            "—", "—", "—",
            f"≥ {result.config.outage_cost.sla_min_autonomy_hours}" if result.config.outage_cost.sla_min_autonomy_hours else "—",
            f"≥ {result.config.outage_cost.sla_continuity_pct}%" if result.config.outage_cost.sla_continuity_pct else "—",
        ],
    }
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)

    if sim.sla_failures:
        st.error("**SLA Failures:**")
        for f in sim.sla_failures:
            st.error(f"• {f}")
    else:
        st.success("✓ All SLA targets met")


# ---------------------------------------------------------------------------
# Tab: Energy Flows
# ---------------------------------------------------------------------------

def _tab_energy_flows(result: RunResult) -> None:
    from energex.adapters.charts import (
        plotly_energy_flows_area, plotly_bess_soc, mpl_energy_pie,
    )
    import plotly.graph_objects as go

    sim = result.sim_year1
    cfg = result.config

    hour_range = st.slider(
        "Display hour range",
        0, 8760,
        (0, min(720, 8760)),
        step=24,
        format="%d hr",
    )
    start_h, end_h = hour_range
    partial_sim = type("Obj", (), {"hourly": sim.hourly[start_h:end_h]})()
    partial_sim.outage_events_total = sim.outage_events_total
    partial_sim.outage_events_served = sim.outage_events_served
    partial_sim.outage_served_pct = sim.outage_served_pct

    st.plotly_chart(
        plotly_energy_flows_area(partial_sim, sample_hours=end_h - start_h),
        use_container_width=True,
    )

    if cfg.bess:
        st.plotly_chart(
            plotly_bess_soc(sim, bess_capacity=cfg.bess.capacity_kwh),
            use_container_width=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annual Energy Flows (Year 1)")
        flows = {
            "Flow": ["Grid → Load", "Solar → Load", "Solar → BESS", "Grid → BESS",
                     "BESS → Load (backup)", "DG → Load (backup)", "Unserved Critical"],
            "kWh/yr": [
                f"{sim.total_grid_kwh:,.0f}",
                f"{sim.total_solar_kwh:,.0f}",
                f"{sim.total_solar_to_bess_kwh:,.0f}",
                f"{sim.total_grid_to_bess_kwh:,.0f}",
                f"{sim.total_bess_discharge_kwh:,.0f}",
                f"{sim.total_dg_kwh:,.0f}",
                f"{sim.ens_kwh:,.1f}",
            ],
            "MWh/yr": [
                f"{sim.total_grid_kwh/1000:,.1f}",
                f"{sim.total_solar_kwh/1000:,.1f}",
                f"{sim.total_solar_to_bess_kwh/1000:,.1f}",
                f"{sim.total_grid_to_bess_kwh/1000:,.1f}",
                f"{sim.total_bess_discharge_kwh/1000:,.1f}",
                f"{sim.total_dg_kwh/1000:,.1f}",
                f"{sim.ens_kwh/1000:,.2f}",
            ],
        }
        st.dataframe(flows, use_container_width=True, hide_index=True)

    with col2:
        png = mpl_energy_pie(sim)
        st.image(png, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Financial
# ---------------------------------------------------------------------------

def _tab_financial(result: RunResult) -> None:
    from energex.adapters.charts import plotly_cashflows, plotly_cumulative_cashflow

    fin = result.finance
    bl_npv = result.finance.incremental_npv

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Total CAPEX", f"₹{fin.total_capex/1e6:.2f}M")
    with c2:
        _metric_card("NPV (25yr)", f"₹{fin.npv/1e6:.2f}M", good=(fin.npv > 0))
    with c3:
        irr_str = f"{fin.irr:.2f}%" if fin.irr is not None else "N/A"
        _metric_card("IRR", irr_str, good=(fin.irr or 0) > 10)
    with c4:
        pb_str = f"{fin.simple_payback_years:.1f} yr" if fin.simple_payback_years else "N/A"
        _metric_card("Simple Payback", pb_str, good=(fin.simple_payback_years or 99) < 10)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("LCOE", f"₹{fin.lcoe_rs_kwh:.2f}/kWh")
    with c2:
        _metric_card("Total Outage Cost (25yr)", f"₹{fin.total_outage_cost/1e6:.2f}M",
                     good=(fin.total_outage_cost < fin.total_capex))
    with c3:
        _metric_card("Total Cost w/ Outages", f"₹{fin.total_cost_with_outages/1e6:.2f}M")
    with c4:
        if bl_npv is not None:
            _metric_card("Incremental NPV vs Baseline",
                        f"₹{bl_npv/1e6:.2f}M", good=(bl_npv > 0))

    st.plotly_chart(plotly_cashflows(fin), use_container_width=True)
    st.plotly_chart(plotly_cumulative_cashflow(fin), use_container_width=True)

    with st.expander("Full Cashflow Table", expanded=False):
        rows = []
        for cf in fin.cashflows:
            rows.append({
                "Year": cf.year,
                "CAPEX (₹)": f"{cf.capex:,.0f}",
                "Replacement (₹)": f"{cf.replacement_cost:,.0f}",
                "Grid Energy (₹)": f"{cf.grid_energy_cost:,.0f}",
                "Diesel Fuel (₹)": f"{cf.diesel_fuel_cost:,.0f}",
                "O&M (₹)": f"{cf.om_bess + cf.om_solar + cf.om_dg + cf.om_dg_variable:,.0f}",
                "Outage Cost (₹)": f"{cf.outage_cost:,.0f}",
                "Salvage (₹)": f"{cf.salvage:,.0f}",
                "Net Cashflow (₹)": f"{cf.net_cashflow:,.0f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Sensitivity
# ---------------------------------------------------------------------------

def _tab_sensitivity(result: RunResult) -> None:
    from energex.adapters.charts import plotly_sensitivity_tornado

    sens = st.session_state.sensitivity_results

    if sens is None:
        st.info(
            "Sensitivity analysis was not run. "
            "Enable **Run sensitivity analysis** in the sidebar and re-run."
        )
        return

    metric = st.radio("Sensitivity metric", ["NPV", "ENS"], horizontal=True)
    m = "npv" if metric == "NPV" else "ens"

    st.plotly_chart(plotly_sensitivity_tornado(sens, metric=m), use_container_width=True)

    # Table
    with st.expander("Raw sensitivity data", expanded=False):
        rows = []
        for sr in sens:
            rows.append({
                "Parameter": sr.parameter,
                "Δ%": f"{sr.delta_pct:+.0f}%",
                "NPV (₹)": f"{sr.npv:,.0f}",
                "ΔNPV (₹)": f"{sr.delta_npv:+,.0f}",
                "ENS (kWh/yr)": f"{sr.ens_kwh:.2f}",
                "ΔENS (kWh/yr)": f"{sr.delta_ens_kwh:+.2f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Export & Download
# ---------------------------------------------------------------------------

def _tab_export(result: RunResult) -> None:
    import tempfile, zipfile

    st.subheader("Export Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Available exports:**")
        st.markdown("- `results_summary.json` — Full result bundle")
        st.markdown("- `reliability_metrics.csv` — Reliability KPIs")
        st.markdown("- `energy_flows.csv` — 8760-hour hourly flows")
        st.markdown("- `cashflows.csv` — Year-by-year financials")
        st.markdown("- `warnings.json` — Warnings list")
        st.markdown("- `report.pdf` — Management + Engineering PDF report")

    with col2:
        if st.button("📦 Download all as ZIP", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                # Export CSV/JSON files
                exported = export_all(
                    config=result.config,
                    sim=result.sim_year1,
                    finance=result.finance,
                    output_dir=tmp_path,
                    seed=result.seed,
                    warnings=result.warnings,
                    run_id=result.run_id,
                )
                # Generate PDF
                try:
                    from energex.adapters.pdf_report import generate_pdf
                    pdf_bytes = generate_pdf(result, st.session_state.sensitivity_results)
                    pdf_path = tmp_path / "report.pdf"
                    pdf_path.write_bytes(pdf_bytes)
                except Exception as e:
                    st.warning(f"PDF generation skipped: {e}")

                # Zip all files
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in tmp_path.glob("*"):
                        zf.write(f, f.name)
                zip_buf.seek(0)

                st.download_button(
                    "⬇ Click to download ZIP",
                    data=zip_buf.getvalue(),
                    file_name=f"energex_{result.run_id}.zip",
                    mime="application/zip",
                )

    st.divider()
    st.subheader("PDF Report Preview")

    if st.button("Generate PDF report"):
        with st.spinner("Generating PDF..."):
            try:
                from energex.adapters.pdf_report import generate_pdf
                pdf_bytes = generate_pdf(result, st.session_state.sensitivity_results)
                st.download_button(
                    "⬇ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"energex_report_{result.run_id}.pdf",
                    mime="application/pdf",
                )
                st.success("PDF ready for download!")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    # JSON summary inline view
    with st.expander("View results_summary.json", expanded=False):
        from energex.adapters.exporters import export_results_summary
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = export_results_summary(
                config=result.config,
                sim=result.sim_year1,
                finance=result.finance,
                output_dir=tmp,
                seed=result.seed,
                run_id=result.run_id,
                warnings=result.warnings,
            )
            content = p.read_text()
            st.code(content, language="json")


# ---------------------------------------------------------------------------
# Tab: Config Editor
# ---------------------------------------------------------------------------

def _tab_config(cfg_dict: dict) -> None:
    st.subheader("Active Configuration")
    st.caption("This is the config currently loaded (after quick overrides).")
    cfg_clean = {k: v for k, v in cfg_dict.items() if k != "_comment"}
    st.code(json.dumps(cfg_clean, indent=2), language="json")

    st.download_button(
        "⬇ Download this config",
        data=json.dumps(cfg_clean, indent=2),
        file_name="energex_config.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _page_config()
    _init_state()

    st.title("⚡ EnergeX — Backup/Outage Energy ROI Calculator")
    st.caption(
        "Model grid outages, backup assets (BESS, DG, Solar), and their true economics. "
        "Reliability is a first-class output."
    )

    cfg_dict = _sidebar()

    result: Optional[RunResult] = st.session_state.result

    if result is None:
        _show_landing(cfg_dict)
        return

    # Tabs — Phase B + Phase C
    tabs = st.tabs([
        "📊 Overview",
        "🔴 Reliability",
        "⚡ Energy Flows",
        "💰 Financial",
        "🌪 Sensitivity",
        "📐 Sizing",
        "🎲 Monte Carlo",
        "💡 TOU & Demand",
        "📥 Export",
        "🔧 Config",
    ])

    with tabs[0]:
        _tab_overview(result)
    with tabs[1]:
        _tab_reliability(result)
    with tabs[2]:
        _tab_energy_flows(result)
    with tabs[3]:
        _tab_financial(result)
    with tabs[4]:
        _tab_sensitivity(result)
    with tabs[5]:
        _tab_sizing(result)
    with tabs[6]:
        _tab_monte_carlo(result)
    with tabs[7]:
        _tab_tou_demand(result)
    with tabs[8]:
        _tab_export(result)
    with tabs[9]:
        _tab_config(st.session_state.config_dict or {})


# ---------------------------------------------------------------------------
# Tab: Sizing (Phase C)
# ---------------------------------------------------------------------------

def _tab_sizing(result: RunResult) -> None:
    from energex.engine.sizing_optimizer import SizingOptimizer
    from energex.adapters.charts import (
        plotly_sizing_scatter, plotly_sizing_pareto, plotly_sizing_reliability,
    )

    st.subheader("📐 Automated Optimal Sizing")
    st.caption(
        "Find the best BESS + DG combination for your load and outage profile. "
        "Sweep a grid of sizes and rank by NPV, LCOE, or reliability."
    )

    cfg = result.config

    col1, col2, col3 = st.columns(3)
    with col1:
        bess_max = st.slider("Max BESS capacity (kWh)", 50, 1000, 400, step=50)
        bess_steps = st.slider("BESS size steps", 3, 8, 5)
    with col2:
        dg_max = st.slider("Max DG rating (kW)", 0, 500, 200, step=25)
        dg_steps = st.slider("DG size steps", 2, 6, 4)
    with col3:
        objective = st.selectbox(
            "Optimize for",
            ["incremental_npv", "lcoe", "capex", "ens"],
            index=0,
        )
        max_capex_m = st.number_input("Max CAPEX (₹M, 0 = no limit)", 0.0, 100.0, 0.0, step=0.5)
        max_capex_rs = max_capex_m * 1e6 if max_capex_m > 0 else None

    run_sizing = st.button("▶ Run Sizing Optimizer", type="primary")

    if run_sizing:
        import numpy as np
        bess_sizes = [0.0] + [
            round(v, 0) for v in np.linspace(
                max(50, bess_max / bess_steps), bess_max, bess_steps
            )
        ]
        dg_sizes = [0.0] + [
            round(v, 0) for v in np.linspace(
                max(50, dg_max / dg_steps), dg_max, dg_steps
            )
        ] if dg_max > 0 else [0.0]

        total_combos = len(bess_sizes) * len(dg_sizes)
        progress_bar = st.progress(0, text=f"Evaluating 0/{total_combos} combinations...")
        progress_count = [0]

        def _prog(p: float) -> None:
            progress_count[0] = int(p * total_combos)
            progress_bar.progress(p, text=f"Evaluating {progress_count[0]}/{total_combos} combinations...")

        optimizer = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=bess_sizes,
            dg_sizes_kw=dg_sizes,
            optimize_for=objective,
            max_capex_rs=max_capex_rs,
            seed=st.session_state.run_seed,
        )
        with st.spinner("Running sizing optimizer..."):
            candidates = optimizer.run(progress_callback=_prog)

        progress_bar.empty()
        st.session_state.sizing_candidates = candidates
        st.rerun()

    candidates = st.session_state.sizing_candidates
    if candidates is None:
        st.info("Configure sizing parameters above and click **▶ Run Sizing Optimizer** to find the best configuration.")
        return

    if not candidates:
        st.error("No valid candidates found. Try relaxing constraints.")
        return

    best = candidates[0]
    st.success(f"**Recommended:** {best.label}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric_card("CAPEX", f"₹{best.capex_rs/1e6:.2f}M")
    with c2:
        incr = best.incremental_npv if best.incremental_npv is not None else best.npv
        _metric_card("Incr. NPV", f"₹{incr/1e6:.2f}M", good=(incr > 0))
    with c3:
        pb = f"{best.payback_years:.1f} yr" if best.payback_years else "N/A"
        _metric_card("Payback", pb, good=(best.payback_years or 99) < 10)
    with c4:
        _metric_card("ENS", f"{best.ens_kwh_yr:.1f} kWh/yr", good=(best.ens_kwh_yr < 100))
    with c5:
        _metric_card("SLA", "PASS" if best.sla_pass else "FAIL", good=best.sla_pass)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(plotly_sizing_pareto(candidates), use_container_width=True)
    with col_r:
        st.plotly_chart(plotly_sizing_reliability(candidates), use_container_width=True)

    st.plotly_chart(plotly_sizing_scatter(candidates), use_container_width=True)

    with st.expander(f"Full Ranking Table ({len(candidates)} candidates)", expanded=True):
        rows = [c.to_dict() for c in candidates]
        import pandas as pd
        df = pd.DataFrame(rows)
        # Format numeric columns
        for col in ["capex_rs", "npv_rs", "incremental_npv_rs"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: f"₹{v/1e6:.2f}M" if v is not None else "N/A")
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Monte Carlo (Phase C)
# ---------------------------------------------------------------------------

def _tab_monte_carlo(result: RunResult) -> None:
    from energex.engine.monte_carlo import MonteCarloEngine
    from energex.adapters.charts import plotly_mc_distribution, plotly_mc_bands_bar

    st.subheader("🎲 Monte Carlo Analysis")
    st.caption(
        "Run N independent simulations to build a statistical distribution of outcomes. "
        "Outputs P10/P50/P90/P99 confidence bands — required for bankable studies."
    )

    cfg = result.config

    if cfg.outage.mode.value != "stochastic":
        st.warning(
            "⚠ This config uses **deterministic** outages. "
            "Monte Carlo requires stochastic mode to produce meaningful variation."
        )

    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.slider(
            "Number of trials",
            min_value=50, max_value=1000,
            value=int(cfg.monte_carlo.n_trials) if cfg.monte_carlo else 200,
            step=50,
            help="200 is sufficient for stable P99. 500+ for bankable studies.",
        )
    with col2:
        st.markdown("**What each percentile means:**")
        st.markdown(
            "- **P50** — Median. Half of years are better, half worse.\n"
            "- **P90** — 90% chance of being at or below this value.\n"
            "- **P99** — Near-worst-case (1-in-100 year event).\n"
            "- **P10** — Used for NPV downside risk (worst-case financial view)."
        )

    run_mc = st.button("▶ Run Monte Carlo", type="primary")

    if run_mc:
        progress_bar = st.progress(0, text="Initializing Monte Carlo...")

        def _prog(p: float) -> None:
            progress_bar.progress(p, text=f"Running trial {int(p*n_trials)}/{n_trials}...")

        engine = MonteCarloEngine(
            config=cfg,
            n_trials=n_trials,
            base_seed=st.session_state.run_seed,
        )

        with st.spinner(f"Running {n_trials} Monte Carlo trials..."):
            try:
                from energex.engine.runner import run_grid_only_baseline
                bl = run_grid_only_baseline(cfg, seed=st.session_state.run_seed)
                bl_npv = bl.npv
            except Exception:
                bl_npv = None

            mc = engine.run(progress_callback=_prog, baseline_npv=bl_npv)

        progress_bar.empty()
        st.session_state.mc_result = mc
        st.rerun()

    mc = st.session_state.mc_result
    if mc is None:
        st.info("Click **▶ Run Monte Carlo** to compute confidence bands.")
        return

    # Summary metrics
    st.success(f"✓ {len(mc.trials)} trials completed")
    st.markdown("---")
    st.subheader("Confidence Bands Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("ENS P50", f"{mc.ens_p50:.1f} kWh/yr")
    with col2:
        _metric_card("ENS P90", f"{mc.ens_p90:.1f} kWh/yr",
                     good=(mc.ens_p90 < mc.ens_p50 * 3))
    with col3:
        _metric_card("ENS P99", f"{mc.ens_p99:.1f} kWh/yr")
    with col4:
        _metric_card("Downtime P90", f"{mc.downtime_p90*60:.1f} min/yr")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("NPV P10", f"₹{mc.npv_p10/1e6:.2f}M",
                     good=(mc.npv_p10 > 0))
    with col2:
        _metric_card("NPV P50", f"₹{mc.npv_p50/1e6:.2f}M", good=(mc.npv_p50 > 0))
    with col3:
        _metric_card("OC P90 (25yr)", f"₹{mc.outage_cost_p90/1e6:.2f}M")
    with col4:
        _metric_card("LCOE P50", f"₹{mc.lcoe_p50:.2f}/kWh")

    st.markdown("---")
    st.plotly_chart(plotly_mc_bands_bar(mc), use_container_width=True)

    st.subheader("Distributions")
    metric_choice = st.selectbox(
        "Show distribution for",
        ["ENS (kWh/yr)", "Downtime (hrs/yr)", "NPV (₹)", "Outage Cost (₹)", "LCOE (₹/kWh)"],
    )
    arr_map = {
        "ENS (kWh/yr)": (mc.ens_trials, "kWh/yr"),
        "Downtime (hrs/yr)": (mc.downtime_trials, "hrs/yr"),
        "NPV (₹)": (mc.npv_trials, "₹"),
        "Outage Cost (₹)": (mc.outage_cost_trials, "₹"),
        "LCOE (₹/kWh)": (mc.lcoe_trials, "₹/kWh"),
    }
    arr, unit = arr_map[metric_choice]
    st.plotly_chart(
        plotly_mc_distribution(arr, metric_choice, unit, mc.confidence_levels),
        use_container_width=True,
    )

    with st.expander("Full Confidence Band Table", expanded=False):
        import pandas as pd
        summary = mc.summary_dict()
        rows_data = []
        for metric_key, bands in summary.items():
            if isinstance(bands, dict):
                row = {"Metric": metric_key}
                row.update(bands)
                rows_data.append(row)
        if rows_data:
            st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: TOU & Demand Charges (Phase C)
# ---------------------------------------------------------------------------

def _tab_tou_demand(result: RunResult) -> None:
    from energex.adapters.charts import plotly_tou_rate_profile, plotly_demand_charge_monthly

    st.subheader("💡 TOU Tariff & Demand Charges")
    st.caption(
        "Time-of-Use tariff profile and monthly peak demand breakdown. "
        "Demand charges can be 40–60% of the bill for large C&I consumers in India."
    )

    cfg = result.config
    sim = result.sim_year1

    # TOU profile
    st.subheader("Tariff Structure")
    if cfg.grid.tou_blocks:
        cols = st.columns(len(cfg.grid.tou_blocks) + 1)
        with cols[0]:
            _metric_card("Flat Rate (fallback)", f"₹{cfg.grid.energy_rate_rs_kwh:.2f}/kWh")
        for i, blk in enumerate(cfg.grid.tou_blocks):
            with cols[i + 1]:
                _metric_card(
                    f"{blk.name.upper()} ({blk.day_type.value})",
                    f"₹{blk.rate_rs_kwh:.2f}/kWh",
                    delta=f"{blk.start_hour:02d}:00–{blk.end_hour:02d}:00",
                )
    else:
        st.info(f"**Flat tariff:** ₹{cfg.grid.energy_rate_rs_kwh:.2f}/kWh — no TOU blocks configured.")
        st.markdown("To enable TOU billing, add `tou_blocks` to the `grid` section of your config.")

    st.plotly_chart(plotly_tou_rate_profile(cfg), use_container_width=True)

    st.markdown("---")

    # Demand charges
    st.subheader("Monthly Peak Demand & Demand Charges")
    if cfg.grid.demand_charge:
        dc = cfg.grid.demand_charge
        c1, c2, c3 = st.columns(3)
        with c1:
            _metric_card("Demand Charge Rate", f"₹{dc.charge_rs_kw_month:,.0f}/kW/month")
        with c2:
            ratchet_str = f"{dc.ratchet_pct}% of annual peak" if dc.ratchet_pct else "None"
            _metric_card("Ratchet Clause", ratchet_str)
        with c3:
            contract_str = f"{dc.contracted_demand_kw:.0f} kW" if dc.contracted_demand_kw else "None"
            _metric_card("Contracted Demand", contract_str)

        # Compute annual demand charge cost
        from energex.engine.finance_engine import _HOURS_PER_MONTH
        monthly_peaks = []
        h = 0
        annual_peak = 0.0
        for hrs in _HOURS_PER_MONTH:
            month_hrs = sim.hourly[h: h + hrs]
            if month_hrs:
                peak = max(hr.grid_to_load + hr.grid_to_bess for hr in month_hrs)
                monthly_peaks.append(peak)
                annual_peak = max(annual_peak, peak)
            else:
                monthly_peaks.append(0.0)
            h += hrs

        annual_dc_cost = 0.0
        for pk in monthly_peaks:
            bd = pk
            if dc.ratchet_pct:
                bd = max(bd, dc.ratchet_pct / 100.0 * annual_peak)
            if dc.contracted_demand_kw:
                bd = max(bd, dc.contracted_demand_kw)
            annual_dc_cost += bd * dc.charge_rs_kw_month

        # Energy cost comparison
        energy_cost_yr1 = result.finance.cashflows[1].grid_energy_cost if len(result.finance.cashflows) > 1 else 0
        total_bill = energy_cost_yr1 + annual_dc_cost
        dc_pct = (annual_dc_cost / total_bill * 100) if total_bill > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            _metric_card("Annual Demand Charge (Yr 1)", f"₹{annual_dc_cost/1e6:.3f}M")
        with c2:
            _metric_card("Annual Energy Bill (Yr 1)", f"₹{energy_cost_yr1/1e6:.3f}M")
        with c3:
            _metric_card("Demand Charge Share", f"{dc_pct:.1f}%",
                         good=(dc_pct < 40))

        st.plotly_chart(
            plotly_demand_charge_monthly(sim, cfg),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("25-Year Demand Charge Projection")
        years = list(range(1, cfg.project_lifetime_years + 1))
        esc = [(1 + cfg.grid.escalation_pct_yr / 100) ** yr for yr in years]
        annual_dcs = [annual_dc_cost * e for e in esc]
        total_dc_lifetime = result.finance.total_demand_charge_cost

        _metric_card(
            "Total Demand Charges (25yr PV)",
            f"₹{total_dc_lifetime/1e6:.2f}M",
        )

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=years, y=annual_dcs,
            marker_color="#E53935",
            hovertemplate="Year %{x}: ₹%{y:,.0f}<extra>Demand Charge</extra>",
        ))
        fig.update_layout(
            title="Annual Demand Charge Projection (escalated)",
            xaxis_title="Year",
            yaxis_title="₹/yr",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "**No demand charge configured.** For large commercial/industrial consumers "
            "in India (HT/EHT category), demand charges can be 40–60% of the total bill.\n\n"
            "Add a `demand_charge` block to your `grid` config to model this cost component."
        )
        with st.expander("Example demand_charge config block"):
            st.code("""{
  "demand_charge": {
    "charge_rs_kw_month": 300.0,
    "ratchet_pct": 80.0,
    "contracted_demand_kw": 100.0
  }
}""", language="json")


def _show_landing(cfg_dict: Optional[dict]) -> None:
    """Show landing / instructions when no simulation has been run yet."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Step 1 — Load Config**\n\n"
            "Choose a preset scenario from the sidebar, upload a JSON config, "
            "or paste one directly."
        )
    with col2:
        st.info(
            "**Step 2 — Adjust Parameters**\n\n"
            "Use the Quick Overrides sliders to tune SAIDI, BESS size, VoLL, "
            "and other key inputs."
        )
    with col3:
        st.info(
            "**Step 3 — Run**\n\n"
            "Click **▶ Run** in the sidebar. Results appear in tabs: "
            "Overview, Reliability, Energy Flows, Financial, Sensitivity, Export."
        )

    if cfg_dict:
        st.success("✓ Config loaded. Press **▶ Run** in the sidebar to simulate.")
    else:
        st.warning("No config loaded yet. Select a preset or upload a JSON file.")

    st.divider()
    col_sc, col_c = st.columns(2)
    with col_sc:
        st.subheader("Supported Scenarios")
        st.markdown(
            "- **Grid + BESS** — fast response, zero start delay\n"
            "- **Grid + DG** — cost-effective, fuel-constrained\n"
            "- **Grid + Solar + BESS** — lowest LCOE, solar-charged BESS\n"
            "- **DG + BESS Hybrid** — BESS covers start delay, DG handles sustained outages\n"
            "- **Full Hybrid** — Solar + DG + BESS for maximum resilience\n"
            "- **Grid Only** — baseline for incremental NPV comparison"
        )
    with col_c:
        st.subheader("Phase C Features")
        st.markdown(
            "- **📐 Sizing** — Automated optimal sizing sweeps BESS + DG combinations\n"
            "  and ranks by NPV, LCOE, or CAPEX. Answers: *what should I buy?*\n\n"
            "- **🎲 Monte Carlo** — P50/P90/P99 confidence bands from N independent\n"
            "  simulation runs. Required for bankable financial studies.\n\n"
            "- **💡 TOU & Demand** — Time-of-Use tariff billing + monthly peak demand\n"
            "  charges. Corrects the financial model for Indian C&I consumers\n"
            "  where demand charges are 40–60% of the electricity bill."
        )


if __name__ == "__main__":
    main()
