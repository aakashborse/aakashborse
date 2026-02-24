# EnergeX — Backup/Outage Energy ROI Calculator

**Production-grade platform for reliability-aware energy system techno-economics.**

EnergeX models grid outages, backup assets (DG, BESS, Solar), and their combined economics — answering *"How much does unreliability actually cost, what is the cheapest configuration to fix it, and how confident are we?"*

---

## Key Capabilities

| Domain | What EnergeX Computes |
|---|---|
| **Reliability** | ENS (kWh/yr), Downtime (hrs/yr), Continuity (%), Backup Autonomy (hrs) |
| **Dispatch** | Hourly BESS/DG backup with SOC tracking, start delays, fuel constraints, peak shaving |
| **Finance** | NPV, IRR, Payback, LCOE — outage cost as a first-class cashflow |
| **Outage Modelling** | Deterministic schedule OR stochastic SAIDI/SAIFI with seasonal weighting |
| **Sensitivity** | Tornado sweeps on SAIDI, VoLL, diesel price, BESS capex, critical load |
| **TOU Tariffs** | Peak/off-peak rate schedules, monthly demand charges, BESS peak shaving |
| **Monte Carlo** | P10/P25/P50/P75/P90/P99 confidence bands — statistically defensible output |
| **Optimal Sizing** | Grid sweep + Pareto frontier — finds least-cost SLA-feasible configuration |
| **Streamlit UI** | Interactive 9-tab dashboard — charts, sensitivity, MC, optimizer, PDF export |
| **PDF Reports** | ReportLab management summary + engineering appendix with all charts |

---

## Supported Scenarios

| Scenario | Config Example |
|---|---|
| Grid only (baseline) | Implicit — computed automatically |
| Grid + DG backup | `configs/grid_dg.json` |
| Grid + BESS backup | `configs/grid_bess.json` |
| Grid + Solar + BESS | `configs/grid_solar_bess.json` |
| Grid + DG + BESS hybrid | `configs/dg_bess_hybrid.json` |
| Grid + BESS + TOU + Demand Charges | `configs/grid_tou_demand.json` |
| Grid + Solar + DG + BESS | Extend any config with all assets |

---

## Quick Start

### Install

```bash
cd energex
pip install -e ".[dev]"          # core + dev (tests)
pip install -e ".[all]"          # core + UI (Streamlit, Plotly, PDF)
```

### Run a scenario

```bash
energex run --config configs/grid_bess.json --seed 42 --export out/
```

### Launch the interactive UI

```bash
energex ui
# opens http://localhost:8501 in your browser
```

### Generate a PDF report

```bash
energex pdf --config configs/grid_bess.json --output report.pdf --sensitivity
```

### Run Monte Carlo analysis

```bash
energex monte-carlo --config configs/grid_tou_demand.json --n-runs 500 --export out/
```

### Run the Sizing Optimizer

```bash
energex optimize --config configs/grid_bess.json --bess-max 600 --bess-step 50 --export out/
```

### Validate config

```bash
energex validate --config configs/grid_dg.json
```

### Generate sample CSV timeseries

```bash
energex generate-sample-csv --output-dir sample_data/ --avg-kw 150 --solar-kw 80
```

---

## CLI Reference

```
energex run        --config  <path>   # JSON or YAML config file
                   --seed    <int>    # Random seed (default: 42)
                   --export  <dir>    # Output directory (default: energex_output/)
                   --baseline/--no-baseline

energex ui         --port <int>       # Streamlit port (default: 8501)

energex pdf        --config  <path>
                   --output  <path>   # PDF output path
                   --seed    <int>
                   --sensitivity/--no-sensitivity

energex monte-carlo
                   --config      <path>
                   --n-runs      <int>    # Number of MC runs (default: 200)
                   --seed        <int>
                   --percentiles <str>    # e.g. "10,50,90,99"
                   --export      <dir>

energex optimize   --config      <path>
                   --bess-max    <kWh>   # Max BESS to sweep
                   --bess-step   <kWh>   # Step size
                   --include-dg / --no-dg
                   --include-solar / --no-solar
                   --n-years     <int>   # Years per candidate eval (default: 3)
                   --seed        <int>
                   --export      <dir>

energex validate   --config <path>
energex generate-sample-csv --output-dir <dir> --avg-kw <kW> --solar-kw <kW>
```

---

## Output Files

| File | Contents |
|---|---|
| `results_summary.json` | Full result: reliability, energy flows, financials, assumptions, warnings |
| `reliability_metrics.csv` | ENS, downtime, autonomy, continuity, SLA pass/fail |
| `energy_flows.csv` | Hourly flows (8760 rows): grid, solar, BESS, DG, TOU rate, peak shaving |
| `cashflows.csv` | Year-by-year breakdown: grid cost, demand charge, fuel, O&M, outage cost |
| `warnings.json` | Warnings list |
| `monte_carlo_results.json` | MC stats: P10/P50/P90/P99 for ENS, downtime, continuity, NPV |
| `optimizer_results.json` | Optimal config, Pareto frontier, all candidates summary |
| `optimizer_all_candidates.csv` | Full grid sweep results (one row per candidate) |

---

## Config Format

All configs are JSON (or YAML). Full schema defined in `domain/schemas.py`.

### Minimal Grid + BESS config

```json
{
  "name": "My Project",
  "project_lifetime_years": 25,
  "discount_rate_pct": 10.0,
  "load": {"mode": "simple", "avg_kw": 100.0, "critical_fraction": 0.35},
  "outage": {
    "mode": "stochastic",
    "stochastic": {"saidi_minutes_per_year": 1200, "saifi_events_per_year": 18}
  },
  "grid": {"energy_rate_rs_kwh": 8.5, "escalation_pct_yr": 5.0},
  "bess": {"capacity_kwh": 150.0, "power_kw": 75.0, "capex_rs_kwh": 35000},
  "outage_cost": {"mode": "voll", "voll_rs_kwh": 150.0, "sla_continuity_pct": 99.9}
}
```

### TOU tariff + demand charges

```json
"grid": {
  "energy_rate_rs_kwh": 8.0,
  "tou_schedule": {
    "rate_bands": [
      {"name": "peak",     "hours_of_day": [9,10,11,12,13,14,15,16,17], "rate_rs_kwh": 12.0},
      {"name": "off_peak", "hours_of_day": [22,23,0,1,2,3,4,5],        "rate_rs_kwh": 5.0},
      {"name": "shoulder", "hours_of_day": [6,7,8,18,19,20,21],        "rate_rs_kwh": 8.0}
    ],
    "default_rate_rs_kwh": 8.0
  },
  "demand_charge": {
    "charge_rs_kva_month": 350.0,
    "power_factor": 0.9
  },
  "peak_shaving_enabled": true,
  "peak_shaving_target_kw": 65.0
}
```

When `tou_schedule` is set, BESS automatically:
- Charges from grid only during off-peak hours
- Discharges for peak shaving when grid draw exceeds `peak_shaving_target_kw`

### Monte Carlo config (inline)

```json
"monte_carlo": {
  "n_runs": 200,
  "base_seed": 42,
  "percentiles": [10.0, 25.0, 50.0, 75.0, 90.0, 99.0],
  "n_years_per_run": 1
}
```

### Sizing optimizer bounds (inline)

```json
"sizing_bounds": {
  "bess_capacity_min_kwh": 0.0,
  "bess_capacity_max_kwh": 500.0,
  "bess_capacity_step_kwh": 50.0,
  "bess_power_to_capacity_ratio": 0.5,
  "include_dg": false,
  "include_solar": false,
  "n_years_per_eval": 3
}
```

See `configs/grid_tou_demand.json` for a full TOU + demand charge + MC + sizing example.

### Deterministic outages

```json
"outage": {
  "mode": "deterministic",
  "events": [
    {"start_hour": 1200.0, "duration_hours": 4.0},
    {"start_hour": 4500.0, "duration_hours": 12.0}
  ]
}
```

---

## Architecture

```
energex/
├── domain/
│   └── schemas.py              # Pydantic v2 models (TOU, DemandCharge, MC, Sizing)
├── engine/
│   ├── outage_generator.py     # Deterministic + SAIDI/SAIFI stochastic outages
│   ├── dispatch_backup.py      # Hourly dispatch: backup + TOU + peak shaving
│   ├── finance_engine.py       # NPV, IRR, LCOE, demand charges, sensitivity
│   ├── monte_carlo.py          # Monte Carlo engine (P10/P50/P90/P99 bands)
│   ├── optimizer.py            # Sizing optimizer (grid sweep + Pareto frontier)
│   └── runner.py               # Full pipeline orchestration
├── adapters/
│   ├── config_loader.py        # JSON/YAML config loading
│   ├── csv_parser.py           # Timeseries CSV parsing + sample generation
│   ├── exporters.py            # JSON/CSV/MC/optimizer result export
│   ├── charts.py               # Plotly (interactive) + Matplotlib (PDF) charts
│   └── pdf_report.py           # ReportLab PDF generator
├── interfaces/
│   ├── cli.py                  # Typer CLI (run, ui, pdf, monte-carlo, optimize)
│   └── streamlit_app.py        # 9-tab Streamlit dashboard
├── tests/
│   ├── test_outage_generator.py
│   ├── test_dispatch.py
│   ├── test_finance.py
│   ├── test_golden.py          # End-to-end golden runs for all example configs
│   ├── test_tou.py             # TOU tariff + demand charge tests
│   ├── test_monte_carlo.py     # Monte Carlo engine tests
│   └── test_optimizer.py       # Sizing optimizer tests
└── configs/
    ├── grid_bess.json
    ├── grid_dg.json
    ├── grid_solar_bess.json
    ├── dg_bess_hybrid.json
    └── grid_tou_demand.json    # TOU + demand charges + MC + sizing example
```

---

## Dispatch Logic

### Grid-Up (normal operation)
```
Solar → Load (priority)
BESS peak shaving (if demand_charge configured and load > peak_shaving_target_kw)
Grid → remaining load
Solar surplus → BESS charge
Grid → BESS charge (off-peak hours only, when TOU configured)
```

### Grid-Down (outage)
```
Outage detected
    │
    ├─► Solar → critical load (if available)
    ├─► BESS (instant, no delay) → remaining critical load
    └─► DG (after start_delay_seconds) → remaining critical load
            │
            └─► If all insufficient → ENS recorded + downtime logged
```

---

## Financial Model

- **Year 0**: CAPEX (BESS + Solar + DG)
- **Year 1–N**:
  - Grid energy cost — hourly TOU rates × grid draw, escalated annually
  - Demand charges — monthly peak kVA × ₹/kVA/month (when configured)
  - Diesel fuel cost with annual price escalation
  - BESS/Solar/DG O&M
  - Outage cost: `ENS × VoLL` OR `downtime × ₹/hr` OR tiered penalties
  - BESS replacement at end-of-calendar-life
- **Year N**: 10% salvage value
- **Metrics**: NPV, IRR, simple payback, discounted payback, LCOE, incremental NPV vs baseline

---

## Phase B — Interactive UI & Reports

### Streamlit Dashboard

```bash
energex ui          # launches http://localhost:8501
```

**9 tabs:**
| Tab | Content |
|---|---|
| Overview | KPI cards, SLA badge, scenario summary, gauge charts |
| Reliability | ENS/downtime metrics, outage timeline, autonomy analysis |
| Energy Flows | Hourly area chart, BESS SOC, TOU demand profile, energy mix pie |
| Financial | Cashflow bars, cumulative NPV, payback line |
| Sensitivity | Tornado charts, sensitivity table |
| Monte Carlo | P10/P50/P90 distribution histograms, percentile bands, summary table |
| Optimizer | Pareto frontier scatter, ENS heatmap, optimal config panel |
| Export | ZIP download (all files + PDF), individual PDF button |
| Config | Raw config JSON viewer |

**Sidebar:** preset selector, JSON upload, quick sliders (SAIDI, BESS size, VoLL, load), seed, run/clear buttons

### PDF Report

```bash
energex pdf --config <path> --output report.pdf --sensitivity
```

**Sections:**
1. **Management Summary** — cover page, headline KPIs, SLA badge, reliability table, financial summary, cashflow and payback charts
2. **Engineering Appendix** — energy flow table, outage event log, sensitivity tornado, year-by-year cashflow table, assumptions list

---

## Phase C — Industry-Grade Extensions

### 1. TOU Tariffs + Demand Charges

Models the real commercial/industrial electricity bill in India (and globally), where demand charges can be **40–60% of the monthly bill**.

**Features:**
- Time-of-use rate bands (peak/shoulder/off-peak) by hour-of-day
- Monthly peak demand charge (₹/kVA/month) with power factor correction
- BESS automatically shifts charging to off-peak hours
- BESS peak shaving during high-demand periods to reduce billing demand
- All demand charges flow into the NPV/LCOE calculation

**Config:** `configs/grid_tou_demand.json`

### 2. Monte Carlo with P50/P90/P99 Bands

Converts point-estimate reliability numbers into **statistically defensible confidence intervals** required for bankable DPRs.

**Features:**
- N independent year-1 simulations (each with a different seed)
- Percentile statistics for ENS, downtime, continuity, NPV
- SLA pass rate — fraction of stochastic runs where all SLA targets are met
- Reproducible: same `base_seed` always gives the same set of runs
- Streamlit tab with histogram, percentile band chart, summary table
- CLI: `energex monte-carlo --n-runs 500`
- Export: `monte_carlo_results.json` with full raw runs + percentile stats

**Interpretation:**
- **P50 ENS** = median expected energy not served (50% of scenarios are better, 50% worse)
- **P90 ENS** = worst-case ENS that is exceeded in only 10% of scenarios (conservative design point)
- **SLA pass rate 95%** = configuration meets reliability targets in 95 out of 100 independent years

### 3. Automated Optimal Sizing

Answers the question clients actually ask: **"What should I buy?"**

**Features:**
- Grid sweep over BESS (kWh/kW), DG (kW), Solar (kWp) — configurable step sizes
- Pareto frontier extraction in (CAPEX, ENS) space
- Reports: optimal (least-cost SLA-feasible), min-ENS, min-CAPEX configurations
- Streamlit tab with Pareto frontier scatter and ENS vs BESS heatmap
- CLI: `energex optimize --bess-max 600 --include-dg`
- Export: `optimizer_results.json` + `optimizer_all_candidates.csv`

**Search space example:** BESS 0→500 kWh (step 50) = 11 candidates × 3 years per eval = 33 simulations total. Add DG sweep (0→300 kW step 50) → 11×7 = 77 candidates.

---

## Running Tests

```bash
# All 117 tests
pytest tests/ -v

# By category
pytest tests/test_tou.py -v           # TOU + demand charge (16 tests)
pytest tests/test_monte_carlo.py -v   # Monte Carlo engine (11 tests)
pytest tests/test_optimizer.py -v     # Sizing optimizer (15 tests)
pytest tests/test_golden.py -v        # End-to-end golden runs (25 tests)

# With coverage
pytest tests/ --cov=energex --cov-report=term-missing
```

**Test breakdown (117 total):**
| File | Tests | Coverage |
|---|---|---|
| `test_outage_generator.py` | 16 | Stochastic/deterministic outage generation |
| `test_dispatch.py` | 22 | BESS/DG unit tests, dispatch integration |
| `test_finance.py` | 12 | Cashflow, NPV, IRR, LCOE |
| `test_golden.py` | 25 | End-to-end golden runs, all 4 configs |
| `test_tou.py` | 16 | TOU schemas, dispatch integration, demand charges |
| `test_monte_carlo.py` | 11 | MC engine, reproducibility, percentile ordering |
| `test_optimizer.py` | 15 | Sizing sweep, Pareto frontier, serialisation |

---

## Key Outputs Explained

| Metric | Definition |
|---|---|
| **ENS** | Energy Not Served — kWh of critical load that could not be met during outages |
| **Downtime** | Hours during which critical load was fully or partially unserved |
| **Continuity %** | `(1 - ENS / total_critical_kWh) × 100` |
| **Backup Autonomy** | Hours BESS (+ DG if present) can serve peak critical load continuously |
| **LCOE** | Levelised Cost of Energy = PV(all supply costs) / total energy |
| **VoLL** | Value of Lost Load — opportunity cost of 1 kWh unserved (₹/kWh) |
| **Incremental NPV** | NPV of this scenario minus NPV of grid-only baseline |
| **P90 ENS** | ENS exceeded in only 10% of stochastic scenarios (conservative design point) |
| **SLA Pass Rate** | % of MC runs where all SLA targets (downtime, ENS, continuity, autonomy) are met |
| **Demand Charge** | Monthly billing based on peak kVA demand × ₹/kVA/month |
| **Pareto Optimal** | Configuration where no alternative is cheaper AND has lower ENS simultaneously |

---

## License

MIT

---

*EnergeX v1.2.0 — Phase A (Engine) + Phase B (UI/PDF) + Phase C (TOU, Monte Carlo, Optimizer)*
