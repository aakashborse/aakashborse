"""
EnergeX — Streamlit UI (Phase B)

Interactive dashboard for Backup/Outage Energy ROI analysis.

Launch with:
    streamlit run energex/interfaces/streamlit_app.py

Or via CLI (if wired up):
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
        "mc_result": None,      # Phase C: Monte Carlo result
        "opt_result": None,     # Phase C: Optimizer result
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

    # TOU demand profile (Phase C)
    if result.config.grid.tou_schedule is not None or sim.total_peak_shaving_kwh > 0:
        st.divider()
        st.subheader("TOU & Peak Shaving Profile")
        from energex.adapters.charts import plotly_tou_demand_profile
        st.plotly_chart(
            plotly_tou_demand_profile(sim, result.config),
            use_container_width=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total BESS Peak Shaving", f"{sim.total_peak_shaving_kwh:,.0f} kWh/yr")
        with col2:
            if sim.monthly_peak_grid_kw:
                st.metric(
                    "Peak Monthly Grid Demand",
                    f"{max(sim.monthly_peak_grid_kw):,.1f} kW",
                )


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

    # Tabs (Phase C adds Monte Carlo and Optimizer tabs)
    tabs = st.tabs([
        "📊 Overview",
        "🔴 Reliability",
        "⚡ Energy Flows",
        "💰 Financial",
        "🌪 Sensitivity",
        "🎲 Monte Carlo",
        "🎯 Optimizer",
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
        _tab_monte_carlo(result)
    with tabs[6]:
        _tab_optimizer(result)
    with tabs[7]:
        _tab_export(result)
    with tabs[8]:
        _tab_config(st.session_state.config_dict or {})


def _tab_monte_carlo(result: RunResult) -> None:
    """Phase C — Monte Carlo uncertainty analysis tab."""
    import streamlit as st
    from energex.engine.monte_carlo import MonteCarloEngine
    from energex.domain.schemas import MonteCarloConfig
    from energex.adapters.charts import (
        plotly_mc_distribution, plotly_mc_bands, plotly_mc_summary_table
    )

    st.subheader("🎲 Monte Carlo Uncertainty Analysis")
    st.caption(
        "Runs N independent simulations with different outage seeds to produce "
        "statistically defensible P10/P50/P90/P99 confidence intervals."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_runs = st.slider("Number of runs", 50, 1000, 200, step=50,
                           help="More runs = tighter confidence intervals (slower)")
    with col2:
        base_seed = st.number_input("Base seed", value=42, min_value=0,
                                    help="Each run gets base_seed + run_index")
    with col3:
        n_years = st.slider("Years per run", 1, 3, 1,
                            help="1 = fastest; 3 = more accurate multi-year average")

    if st.button("▶ Run Monte Carlo", type="primary"):
        with st.spinner(f"Running {n_runs} simulations..."):
            try:
                mc_cfg = MonteCarloConfig(
                    n_runs=n_runs,
                    base_seed=int(base_seed),
                    n_years_per_run=n_years,
                )
                engine = MonteCarloEngine(result.config, mc_config=mc_cfg)
                mc_result = engine.run()
                st.session_state["mc_result"] = mc_result
            except Exception as e:
                st.error(f"Monte Carlo failed: {e}")
                return

    mc_result = st.session_state.get("mc_result")
    if mc_result is None:
        st.info("Configure settings above and click **▶ Run Monte Carlo** to start.")
        return

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SLA Pass Rate", f"{mc_result.sla_pass_rate_pct:.1f}%",
                  delta=None,
                  help="% of runs where all SLA targets are met")
    with col2:
        if mc_result.ens_stats:
            st.metric("ENS P50", f"{mc_result.ens_stats.p50:,.1f} kWh/yr")
    with col3:
        if mc_result.ens_stats:
            st.metric("ENS P90", f"{mc_result.ens_stats.p90:,.1f} kWh/yr")
    with col4:
        if mc_result.ens_stats:
            st.metric("ENS P99", f"{mc_result.ens_stats.p99:,.1f} kWh/yr")

    st.divider()

    # Summary table
    st.plotly_chart(plotly_mc_summary_table(mc_result), use_container_width=True)

    # Distribution charts
    st.subheader("ENS Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plotly_mc_distribution(mc_result, "ens_kwh", "ENS Distribution (kWh/yr)"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(plotly_mc_bands(mc_result), use_container_width=True)

    st.subheader("Downtime & Continuity Distributions")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plotly_mc_distribution(mc_result, "downtime_hours", "Downtime Distribution (hrs/yr)"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            plotly_mc_distribution(mc_result, "continuity_pct", "Continuity Distribution (%)"),
            use_container_width=True,
        )

    # Download raw data
    import json
    st.download_button(
        "⬇ Download MC Results (JSON)",
        data=json.dumps(mc_result.to_dict(), indent=2),
        file_name="monte_carlo_results.json",
        mime="application/json",
    )


def _tab_optimizer(result: RunResult) -> None:
    """Phase C — Automated sizing optimizer tab."""
    import streamlit as st
    from energex.engine.optimizer import SizingOptimizer
    from energex.domain.schemas import SizingBounds
    from energex.adapters.charts import plotly_pareto_frontier, plotly_sizing_heatmap

    st.subheader("🎯 Automated Optimal Sizing")
    st.caption(
        "Grid sweep over BESS / DG / Solar sizes. "
        "Finds the Pareto frontier in (CAPEX, ENS) space and the least-cost SLA-feasible configuration."
    )

    with st.expander("🔧 Search Space Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            bess_max = st.slider("Max BESS (kWh)", 50, 1000, 400, step=50)
            bess_step = st.slider("BESS step (kWh)", 25, 200, 50, step=25)
        with col2:
            n_years = st.slider("Years per evaluation", 1, 5, 3)
            include_dg = st.checkbox("Include DG in sweep", value=False)
            include_solar = st.checkbox("Include Solar in sweep", value=False)

        # Estimate candidate count
        n_bess = len(range(0, int(bess_max) + 1, int(bess_step))) + 1
        n_dg = 7 if include_dg else 1
        n_solar = 7 if include_solar else 1
        est = n_bess * n_dg * n_solar
        st.info(f"Estimated candidates: ~{est} × {n_years} years = ~{est * n_years} simulations")

    if st.button("▶ Run Optimizer", type="primary"):
        with st.spinner("Optimizing... (this may take a minute for large grids)"):
            try:
                bounds = SizingBounds(
                    bess_capacity_max_kwh=float(bess_max),
                    bess_capacity_step_kwh=float(bess_step),
                    include_dg=include_dg,
                    include_solar=include_solar,
                    n_years_per_eval=n_years,
                )
                optimizer = SizingOptimizer(result.config, bounds=bounds)
                opt_result = optimizer.run()
                st.session_state["opt_result"] = opt_result
            except Exception as e:
                st.error(f"Optimizer failed: {e}")
                return

    opt_result = st.session_state.get("opt_result")
    if opt_result is None:
        st.info("Configure settings and click **▶ Run Optimizer** to start.")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Candidates Evaluated", opt_result.n_candidates_evaluated)
    with col2:
        st.metric("SLA-Feasible Configs", opt_result.n_sla_feasible)
    with col3:
        st.metric("Pareto Points", len(opt_result.pareto_points))

    st.divider()

    # Optimal config
    if opt_result.optimal_point:
        opt = opt_result.optimal_point
        st.success(
            f"**Optimal Configuration (least-cost SLA-feasible)**  \n"
            f"BESS: **{opt.bess_kwh:.0f} kWh / {opt.bess_kw:.0f} kW** | "
            f"DG: **{opt.dg_kw:.0f} kW** | "
            f"Solar: **{opt.solar_kwp:.0f} kWp**  \n"
            f"CAPEX: **₹{opt.capex_rs/1e5:.1f}L** | "
            f"ENS: **{opt.avg_ens_kwh:.1f} kWh/yr** | "
            f"Continuity: **{opt.avg_continuity_pct:.4f}%**"
        )
    else:
        st.warning("No SLA-feasible configuration found. Consider relaxing SLA targets or expanding the search space.")

    # Pareto frontier chart
    st.plotly_chart(plotly_pareto_frontier(opt_result), use_container_width=True)

    # Heatmap
    st.plotly_chart(plotly_sizing_heatmap(opt_result), use_container_width=True)

    # Pareto table
    if opt_result.pareto_points:
        st.subheader("Pareto Frontier Points")
        import pandas as pd
        df = pd.DataFrame([p.to_dict() for p in opt_result.pareto_points])
        st.dataframe(df, use_container_width=True)

    # Download
    import json
    st.download_button(
        "⬇ Download Optimizer Results (JSON)",
        data=json.dumps(opt_result.to_dict(), indent=2),
        file_name="optimizer_results.json",
        mime="application/json",
    )


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
    st.subheader("Supported Scenarios")
    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        st.markdown(
            "- **Grid + BESS** — fast response, zero start delay\n"
            "- **Grid + DG** — cost-effective, fuel-constrained\n"
            "- **Grid + Solar + BESS** — lowest LCOE, solar-charged BESS"
        )
    with sc_col2:
        st.markdown(
            "- **DG + BESS Hybrid** — BESS covers start delay, DG handles sustained outages\n"
            "- **Full Hybrid** — Solar + DG + BESS for maximum resilience\n"
            "- **Grid Only** — baseline for incremental NPV comparison"
        )


if __name__ == "__main__":
    main()
