"""
EnergeX — PDF Report Generator (Phase B)

Generates a professional two-section PDF:
  1. Executive Summary  (management view, ~3 pages)
     - Project overview, key metrics, SLA status
     - Financial summary, cost comparison
     - Reliability summary with outage stats

  2. Engineering Appendix (~3–5 pages)
     - Dispatch charts (energy flows, BESS SOC)
     - Full cashflow table
     - Outage event details
     - Assumptions list + warnings

Uses ReportLab for layout and Matplotlib for chart images.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor, white, black

from energex.engine.runner import RunResult
from energex.engine.finance_engine import SensitivityResult


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BLUE = HexColor("#1A237E")
LIGHT_BLUE = HexColor("#E3F2FD")
GREEN = HexColor("#43A047")
RED = HexColor("#E53935")
ORANGE = HexColor("#FF9800")
GREY = HexColor("#90A4AE")
DARK_GREY = HexColor("#455A64")
LIGHT_GREY = HexColor("#F5F5F5")
YELLOW_BG = HexColor("#FFFDE7")


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        parent=base["Heading1"],
        fontSize=28, textColor=white,
        alignment=TA_CENTER, spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        parent=base["Normal"],
        fontSize=14, textColor=HexColor("#B3E5FC"),
        alignment=TA_CENTER, spaceAfter=4,
        fontName="Helvetica",
    )
    styles["section_title"] = ParagraphStyle(
        "section_title",
        parent=base["Heading1"],
        fontSize=16, textColor=BLUE,
        spaceBefore=14, spaceAfter=6,
        fontName="Helvetica-Bold",
        borderPad=4,
    )
    styles["subsection_title"] = ParagraphStyle(
        "sub_title",
        parent=base["Heading2"],
        fontSize=13, textColor=DARK_GREY,
        spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontSize=10, textColor=DARK_GREY,
        leading=14, spaceAfter=4,
        fontName="Helvetica",
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        parent=base["Normal"],
        fontSize=8, textColor=GREY,
        alignment=TA_CENTER, spaceAfter=2,
        fontName="Helvetica-Oblique",
    )
    styles["metric_value"] = ParagraphStyle(
        "metric_value",
        parent=base["Normal"],
        fontSize=20, textColor=BLUE,
        alignment=TA_CENTER, spaceAfter=0,
        fontName="Helvetica-Bold",
    )
    styles["metric_label"] = ParagraphStyle(
        "metric_label",
        parent=base["Normal"],
        fontSize=8, textColor=GREY,
        alignment=TA_CENTER,
        fontName="Helvetica",
    )
    styles["warning"] = ParagraphStyle(
        "warning",
        parent=base["Normal"],
        fontSize=9, textColor=HexColor("#E65100"),
        leading=12, spaceAfter=3,
        fontName="Helvetica",
        leftIndent=12,
    )
    styles["assumption"] = ParagraphStyle(
        "assumption",
        parent=base["Normal"],
        fontSize=9, textColor=DARK_GREY,
        leading=12, spaceAfter=2,
        fontName="Helvetica",
        leftIndent=12,
    )
    return styles


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _key_table(data: list[tuple[str, str]], col_widths=None) -> Table:
    """Two-column key-value table."""
    if col_widths is None:
        col_widths = [7 * cm, 8 * cm]

    styled_data = [
        [
            Paragraph(f"<b>{k}</b>", ParagraphStyle("kv_key", fontSize=9, textColor=DARK_GREY,
                                                     fontName="Helvetica-Bold")),
            Paragraph(str(v), ParagraphStyle("kv_val", fontSize=9, textColor=BLUE,
                                             fontName="Helvetica-Bold")),
        ]
        for k, v in data
    ]
    t = Table(styled_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.3, GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, LIGHT_GREY]),
    ]))
    return t


def _header_table(headers: list[str], rows: list[list[str]], col_widths=None) -> Table:
    """Multi-column table with styled header row."""
    header_style = ParagraphStyle(
        "th", fontSize=9, textColor=white, fontName="Helvetica-Bold", alignment=TA_CENTER
    )
    cell_style = ParagraphStyle(
        "td", fontSize=9, textColor=DARK_GREY, fontName="Helvetica", alignment=TA_CENTER
    )

    data = [[Paragraph(h, header_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), cell_style) for c in row])

    if col_widths is None:
        total_w = 17 * cm
        col_widths = [total_w / len(headers)] * len(headers)

    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.3, GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _png_image(png_bytes: bytes, width: float = 16 * cm) -> Image:
    """Wrap PNG bytes in a ReportLab Image."""
    buf = io.BytesIO(png_bytes)
    img = Image(buf)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


# ---------------------------------------------------------------------------
# Page numbering
# ---------------------------------------------------------------------------

class _PageNum(Flowable):
    """Simple page number footnote."""
    def __init__(self, doc_title: str):
        self.doc_title = doc_title
        super().__init__()
        self.width = 0
        self.height = 0

    def draw(self):
        pass  # handled by onFirstPage / onLaterPages


def _on_page(canvas, doc, project_name: str):
    canvas.saveState()
    # Footer bar
    canvas.setFillColor(HexColor("#E3F2FD"))
    canvas.rect(0, 0, A4[0], 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(DARK_GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.5 * cm, 0.45 * cm, f"EnergeX — {project_name}")
    canvas.drawRightString(A4[0] - 1.5 * cm, 0.45 * cm,
                           f"Page {doc.page}  |  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf(
    result: RunResult,
    sensitivity_results: Optional[list[SensitivityResult]] = None,
) -> bytes:
    """
    Generate a two-section PDF report and return as bytes.

    Parameters
    ----------
    result : RunResult
    sensitivity_results : list[SensitivityResult] | None

    Returns
    -------
    bytes — PDF content
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = _make_styles()
    project_name = result.config.name
    story = []

    _build_cover(story, styles, result)
    story.append(PageBreak())

    # Section 1 — Executive Summary
    story.append(Paragraph("Section 1 — Executive Summary", styles["section_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    story.append(Spacer(1, 8))
    _build_exec_summary(story, styles, result)
    story.append(PageBreak())

    # Section 2 — Engineering Appendix
    story.append(Paragraph("Section 2 — Engineering Appendix", styles["section_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_GREY))
    story.append(Spacer(1, 8))
    _build_engineering_appendix(story, styles, result, sensitivity_results)

    # Build
    doc.build(
        story,
        onFirstPage=lambda c, d: _on_page(c, d, project_name),
        onLaterPages=lambda c, d: _on_page(c, d, project_name),
    )
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def _build_cover(story, styles, result: RunResult) -> None:
    sim = result.sim_year1
    fin = result.finance
    cfg = result.config

    # Blue header band (simulated with a table)
    cover_data = [[
        Paragraph("⚡ EnergeX", styles["cover_title"]),
    ], [
        Paragraph("Backup / Outage Energy ROI Report", styles["cover_sub"]),
    ], [
        Paragraph(cfg.name, ParagraphStyle(
            "cn", fontSize=18, textColor=white, alignment=TA_CENTER, spaceAfter=4,
            fontName="Helvetica-Bold",
        )),
    ]]
    cover_t = Table(cover_data, colWidths=[17 * cm])
    cover_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(cover_t)
    story.append(Spacer(1, 1.5 * cm))

    # Key project info
    scenario = cfg.scenario.value.replace("_", " ").title() if cfg.scenario else "—"
    info = [
        ("Scenario", scenario),
        ("Project Lifetime", f"{cfg.project_lifetime_years} years"),
        ("Discount Rate", f"{cfg.discount_rate_pct}%/yr"),
        ("Simulation Seed", str(result.seed)),
        ("Run ID", result.run_id),
        ("Report Date", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
    ]
    story.append(_key_table(info, col_widths=[7 * cm, 10 * cm]))
    story.append(Spacer(1, 1.0 * cm))

    # Headline metrics (2x3 table)
    sla_text = "✓ PASSED" if sim.sla_pass else "✗ FAILED"
    sla_color = GREEN if sim.sla_pass else RED
    metrics = [
        [
            _metric_cell(f"{sim.ens_kwh:,.1f}", "kWh/yr", "ENS (Year 1)"),
            _metric_cell(f"{sim.downtime_hours:.2f}", "hrs/yr", "Downtime (Year 1)"),
            _metric_cell(f"{sim.backup_autonomy_hours:.1f}", "hrs", "Backup Autonomy"),
        ],
        [
            _metric_cell(f"₹{fin.total_capex/1e6:.2f}M", "", "Total CAPEX"),
            _metric_cell(f"₹{fin.npv/1e6:.2f}M", "", "NPV"),
            _metric_cell(f"{fin.lcoe_rs_kwh:.2f}", "₹/kWh", "LCOE"),
        ],
    ]
    metrics_t = Table(metrics, colWidths=[5.5 * cm, 5.5 * cm, 6 * cm])
    metrics_t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
    ]))
    story.append(metrics_t)
    story.append(Spacer(1, 0.8 * cm))

    # SLA badge
    sla_t = Table([[
        Paragraph(
            f'<font color="{"#43A047" if sim.sla_pass else "#E53935"}" size="14"><b>{sla_text}</b></font>'
            f' — SLA Evaluation',
            ParagraphStyle("sla", fontSize=14, fontName="Helvetica-Bold",
                           alignment=TA_CENTER, textColor=sla_color),
        ),
    ]], colWidths=[17 * cm])
    sla_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 2, sla_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(sla_t)

    if not sim.sla_pass:
        story.append(Spacer(1, 4))
        for f in sim.sla_failures:
            story.append(Paragraph(f"• {f}", styles["warning"]))


def _metric_cell(value: str, unit: str, label: str):
    """Build a metric display cell for cover page."""
    return [
        Paragraph(value, ParagraphStyle(
            "mv", fontSize=18, textColor=BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold"
        )),
        Paragraph(f"{unit}\n{label}" if unit else label, ParagraphStyle(
            "ml", fontSize=8, textColor=GREY, alignment=TA_CENTER, fontName="Helvetica"
        )),
    ]


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------

def _build_exec_summary(story, styles, result: RunResult) -> None:
    sim = result.sim_year1
    fin = result.finance
    cfg = result.config
    from energex.adapters.charts import mpl_cashflows_bar, mpl_cumulative_cashflow, mpl_energy_pie

    # --- Project Overview ---
    story.append(Paragraph("1.1 Project Overview", styles["subsection_title"]))
    story.append(Paragraph(
        f"This report presents the techno-economic analysis for <b>{cfg.name}</b>. "
        f"The simulation modelled a <b>{cfg.scenario.value.replace('_', ' ').upper()}</b> "
        f"configuration over a {cfg.project_lifetime_years}-year project lifetime. "
        f"Grid outages were modelled using a <b>{'stochastic (SAIDI/SAIFI)' if cfg.outage.mode.value == 'stochastic' else 'deterministic schedule'}</b> model "
        f"with a random seed of {result.seed} for reproducibility.",
        styles["body"],
    ))
    story.append(Spacer(1, 6))

    # Asset summary
    assets = []
    if cfg.bess:
        assets.append(f"BESS: {cfg.bess.capacity_kwh} kWh / {cfg.bess.power_kw} kW "
                      f"(DoD {cfg.bess.dod_pct}%, RTE {cfg.bess.roundtrip_efficiency*100:.0f}%)")
    if cfg.dg:
        assets.append(f"DG: {cfg.dg.rated_kw} kW, start delay {cfg.dg.start_delay_seconds}s, "
                      f"availability {cfg.dg.availability_pct}%")
    if cfg.solar:
        assets.append(f"Solar: {cfg.solar.size_kw} kWp, "
                      f"degradation {cfg.solar.degradation_pct_yr}%/yr")

    if assets:
        story.append(Paragraph("Backup assets configured:", styles["body"]))
        for a in assets:
            story.append(Paragraph(f"  • {a}", styles["body"]))
    else:
        story.append(Paragraph("Grid only — no backup assets configured.", styles["body"]))

    story.append(Spacer(1, 10))

    # --- Reliability Summary ---
    story.append(Paragraph("1.2 Reliability Summary", styles["subsection_title"]))

    rel_rows = [
        ["Energy Not Served (ENS)", f"{sim.ens_kwh:,.2f} kWh/yr",
         f"≤ {cfg.outage_cost.sla_max_ens_kwh_yr}" if cfg.outage_cost.sla_max_ens_kwh_yr else "—",
         "✓" if (not cfg.outage_cost.sla_max_ens_kwh_yr or sim.ens_kwh <= cfg.outage_cost.sla_max_ens_kwh_yr) else "✗"],
        ["Downtime (critical load)", f"{sim.downtime_hours:.4f} hrs/yr",
         f"≤ {cfg.outage_cost.sla_max_downtime_hrs_yr}" if cfg.outage_cost.sla_max_downtime_hrs_yr else "—",
         "✓" if (not cfg.outage_cost.sla_max_downtime_hrs_yr or sim.downtime_hours <= cfg.outage_cost.sla_max_downtime_hrs_yr) else "✗"],
        ["Continuity", f"{sim.continuity_pct:.4f}%",
         f"≥ {cfg.outage_cost.sla_continuity_pct}%" if cfg.outage_cost.sla_continuity_pct else "—",
         "✓" if (not cfg.outage_cost.sla_continuity_pct or sim.continuity_pct >= cfg.outage_cost.sla_continuity_pct) else "✗"],
        ["Backup Autonomy", f"{sim.backup_autonomy_hours:.2f} hrs",
         f"≥ {cfg.outage_cost.sla_min_autonomy_hours} hrs" if cfg.outage_cost.sla_min_autonomy_hours else "—",
         "✓" if (not cfg.outage_cost.sla_min_autonomy_hours or sim.backup_autonomy_hours >= cfg.outage_cost.sla_min_autonomy_hours) else "✗"],
        ["Outage events (Year 1)", str(sim.outage_events_total), "—", "—"],
        ["Events fully served", f"{sim.outage_events_served} ({sim.outage_served_pct:.1f}%)", "—", "—"],
    ]
    story.append(_header_table(
        ["Metric", "Value", "SLA Target", "Pass/Fail"],
        rel_rows,
        col_widths=[6.5 * cm, 4 * cm, 3.5 * cm, 3 * cm],
    ))
    story.append(Spacer(1, 10))

    # --- Financial Summary ---
    story.append(Paragraph("1.3 Financial Summary", styles["subsection_title"]))

    fin_rows = [
        ["Total CAPEX", f"₹{fin.total_capex:,.0f}"],
        ["NPV (project lifetime)", f"₹{fin.npv:,.0f}"],
        ["IRR", f"{fin.irr:.2f}%" if fin.irr else "N/A"],
        ["Simple Payback", f"{fin.simple_payback_years:.1f} yr" if fin.simple_payback_years else "N/A"],
        ["Discounted Payback", f"{fin.discounted_payback_years:.1f} yr" if fin.discounted_payback_years else "N/A"],
        ["LCOE", f"₹{fin.lcoe_rs_kwh:.2f}/kWh"],
        ["Total Cost with Outages", f"₹{fin.total_cost_with_outages:,.0f}"],
        ["Total Cost without Outages", f"₹{fin.total_cost_without_outages:,.0f}"],
        ["Total Outage Cost (lifetime)", f"₹{fin.total_outage_cost:,.0f}"],
        ["Outage Cost as % of Total Cost", f"{fin.total_outage_cost / max(fin.total_cost_with_outages, 1) * 100:.1f}%"],
    ]
    if fin.incremental_npv is not None:
        fin_rows.append(["Incremental NPV vs Grid-Only", f"₹{fin.incremental_npv:,.0f}"])

    story.append(_key_table(fin_rows, col_widths=[9 * cm, 8 * cm]))
    story.append(Spacer(1, 10))

    # Cashflow bar chart
    try:
        cashflow_png = mpl_cashflows_bar(fin)
        story.append(KeepTogether([
            _png_image(cashflow_png, width=16 * cm),
            Paragraph("Figure 1: Annual cost breakdown by category", styles["caption"]),
            Spacer(1, 8),
        ]))
    except Exception:
        pass

    # Cumulative cashflow
    try:
        cum_png = mpl_cumulative_cashflow(fin)
        story.append(KeepTogether([
            _png_image(cum_png, width=16 * cm),
            Paragraph("Figure 2: Cumulative cashflow — payback curve", styles["caption"]),
            Spacer(1, 8),
        ]))
    except Exception:
        pass

    # Energy mix pie
    try:
        pie_png = mpl_energy_pie(sim)
        story.append(KeepTogether([
            _png_image(pie_png, width=10 * cm),
            Paragraph("Figure 3: Annual energy supply mix (Year 1)", styles["caption"]),
        ]))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Engineering Appendix
# ---------------------------------------------------------------------------

def _build_engineering_appendix(
    story, styles, result: RunResult,
    sensitivity_results: Optional[list[SensitivityResult]],
) -> None:
    sim = result.sim_year1
    fin = result.finance
    cfg = result.config
    from energex.adapters.charts import mpl_tornado

    story.append(Paragraph("2.1 Energy Flow Detail", styles["subsection_title"]))

    # Energy flows table
    flow_rows = [
        ["Grid → Load", f"{sim.total_grid_kwh:,.0f}", f"{sim.total_grid_kwh/1000:,.1f}"],
        ["Solar → Load", f"{sim.total_solar_kwh:,.0f}", f"{sim.total_solar_kwh/1000:,.1f}"],
        ["Solar → BESS (charge)", f"{sim.total_solar_to_bess_kwh:,.0f}", f"{sim.total_solar_to_bess_kwh/1000:,.1f}"],
        ["Grid → BESS (charge)", f"{sim.total_grid_to_bess_kwh:,.0f}", f"{sim.total_grid_to_bess_kwh/1000:,.1f}"],
        ["BESS → Load (backup)", f"{sim.total_bess_discharge_kwh:,.0f}", f"{sim.total_bess_discharge_kwh/1000:,.1f}"],
        ["DG → Load (backup)", f"{sim.total_dg_kwh:,.0f}", f"{sim.total_dg_kwh/1000:,.1f}"],
        ["Unserved Critical (ENS)", f"{sim.ens_kwh:,.1f}", f"{sim.ens_kwh/1000:,.3f}"],
        ["DG Fuel Consumed", f"{sim.dg_fuel_liters:,.0f} L", "—"],
    ]
    story.append(_header_table(
        ["Flow", "kWh/yr", "MWh/yr"],
        flow_rows,
        col_widths=[8 * cm, 4.5 * cm, 4.5 * cm],
    ))
    story.append(Spacer(1, 12))

    # --- Outage events ---
    story.append(Paragraph("2.2 Outage Events (Year 1 Sample)", styles["subsection_title"]))
    story.append(Paragraph(
        f"Year 1 outage profile: {len(result.outage_events)} events "
        f"({result.sim_year1.outage_events_served} fully served). "
        f"Showing up to 20 events.",
        styles["body"],
    ))
    story.append(Spacer(1, 4))

    ev_rows = []
    for i, ev in enumerate(result.outage_events[:20]):
        ev_rows.append([
            str(i + 1),
            f"{ev.start_hour:.2f}",
            f"{ev.effective_duration:.3f}",
            f"{ev.start_hour / 24:.1f}",
        ])

    story.append(_header_table(
        ["#", "Start (hr)", "Duration (hr)", "Day of Year"],
        ev_rows,
        col_widths=[2 * cm, 4.5 * cm, 4.5 * cm, 6 * cm],
    ))
    story.append(Spacer(1, 12))

    # --- Sensitivity ---
    if sensitivity_results:
        story.append(Paragraph("2.3 Sensitivity Analysis", styles["subsection_title"]))
        story.append(Paragraph(
            "Tornado chart showing NPV and ENS sensitivity to key input parameters "
            "(each perturbed individually at ±20–50%).",
            styles["body"],
        ))
        try:
            tornado_npv = mpl_tornado(sensitivity_results, metric="npv")
            story.append(KeepTogether([
                _png_image(tornado_npv, width=15 * cm),
                Paragraph("Figure 4: NPV Sensitivity Tornado", styles["caption"]),
                Spacer(1, 8),
            ]))
        except Exception:
            pass

        try:
            tornado_ens = mpl_tornado(sensitivity_results, metric="ens")
            story.append(KeepTogether([
                _png_image(tornado_ens, width=15 * cm),
                Paragraph("Figure 5: ENS Sensitivity Tornado", styles["caption"]),
            ]))
        except Exception:
            pass
        story.append(Spacer(1, 12))

    # --- Full cashflow table ---
    story.append(Paragraph("2.4 Full Cashflow Table", styles["subsection_title"]))
    cf_rows = []
    for cf in fin.cashflows:
        cf_rows.append([
            str(cf.year),
            f"₹{cf.capex + cf.replacement_cost:,.0f}",
            f"₹{cf.grid_energy_cost:,.0f}",
            f"₹{cf.diesel_fuel_cost:,.0f}",
            f"₹{cf.om_bess + cf.om_solar + cf.om_dg:,.0f}",
            f"₹{cf.outage_cost:,.0f}",
            f"₹{cf.net_cashflow:,.0f}",
        ])
    story.append(_header_table(
        ["Yr", "CAPEX/Repl", "Grid", "Fuel", "O&M", "Outage", "Net CF"],
        cf_rows,
        col_widths=[1.2*cm, 3.1*cm, 3.0*cm, 2.8*cm, 2.6*cm, 2.8*cm, 3.0*cm],
    ))
    story.append(Spacer(1, 12))

    # --- Assumptions ---
    story.append(Paragraph("2.5 Assumptions", styles["subsection_title"]))
    from energex.adapters.exporters import _build_assumptions
    for a in _build_assumptions(cfg):
        story.append(Paragraph(f"• {a}", styles["assumption"]))
    story.append(Spacer(1, 8))

    # --- Warnings ---
    if result.warnings:
        story.append(Paragraph("2.6 Warnings", styles["subsection_title"]))
        for w in result.warnings:
            story.append(Paragraph(f"⚠ {w}", styles["warning"]))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=GREY))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"EnergeX v1.0.0 — Run ID: {result.run_id} — Seed: {result.seed}",
        ParagraphStyle("footer", fontSize=8, textColor=GREY, alignment=TA_CENTER,
                       fontName="Helvetica-Oblique"),
    ))
