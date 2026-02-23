# EnergeX — Backup/Outage Energy ROI Calculator

**Production-grade platform for reliability-aware energy system techno-economics.**

EnergeX models grid outages, backup assets (DG, BESS, Solar), and their combined economics — answering *"How much does unreliability actually cost, and what is the cheapest configuration to fix it?"*

---

## Key Capabilities

| Domain | What EnergeX Computes |
|---|---|
| **Reliability** | ENS (kWh/yr), Downtime (hrs/yr), Continuity (%), Backup Autonomy (hrs) |
| **Dispatch** | Hourly BESS/DG backup dispatch with SOC tracking, start delays, fuel constraints |
| **Finance** | NPV, IRR, Payback, LCOE — including outage cost as a first-class cashflow |
| **Outage Modelling** | Deterministic schedule OR stochastic SAIDI/SAIFI with seasonal weighting |
| **Sensitivity** | Tornado sweeps on SAIDI, VoLL, diesel price, BESS capex, critical load |
| **TOU Tariffs** | Time-of-Use billing (peak/shoulder/off-peak) with midnight-wrap support |
| **Demand Charges** | Monthly peak kW billing + ratchet clauses (40–60% of C&I bills in India) |
| **Monte Carlo** | P50/P90/P99 confidence bands from N independent trials — bankable output |
| **Auto Sizing** | Automated BESS + DG sizing sweep: tells you *what to buy*, not just pass/fail |

---

## Supported Scenarios

| Scenario | Config Example |
|---|---|
| Grid only (baseline) | Implicit — computed automatically |
| Grid + DG backup | `configs/grid_dg.json` |
| Grid + BESS backup | `configs/grid_bess.json` |
| Grid + Solar + BESS | `configs/grid_solar_bess.json` |
| Grid + DG + BESS hybrid | `configs/dg_bess_hybrid.json` |
| Grid + BESS + TOU + Demand Charge | `configs/grid_bess_tou.json` |
| Grid + Solar + DG + BESS | Extend any config with all assets |

---

## Quick Start

### Install

```bash
cd energex
pip install -e ".[dev]"
```

### Run a scenario

```bash
energex run --config configs/grid_bess.json --seed 42 --export out/
```

### Run with TOU tariffs and demand charges

```bash
energex run --config configs/grid_bess_tou.json --export out/
```

### Automated optimal sizing

```bash
energex size --config configs/grid_bess_tou.json \
             --bess-max 500 --bess-steps 6 \
             --dg-max 200 --dg-steps 4 \
             --objective incremental_npv
```

### Monte Carlo confidence bands

```bash
energex monte-carlo --config configs/grid_bess.json --trials 500
```

### Launch interactive UI

```bash
energex ui
```

---

## CLI Reference

```
energex run              --config  <path>    JSON or YAML config
                         --seed    <int>     Random seed (default: 42)
                         --export  <dir>     Output directory (default: energex_output/)
                         --baseline/--no-baseline

energex validate         --config  <path>

energex generate-sample-csv
                         --output-dir <dir>
                         --avg-kw <kW>
                         --solar-kw <kW>

energex ui               --port <int>        Launch Streamlit dashboard (Phase B)

energex pdf              --config <path>     Generate PDF report (Phase B)
                         --output <path>
                         --sensitivity

energex size             --config  <path>    Automated sizing optimizer (Phase C)
                         --bess-max <kWh>    Max BESS capacity to sweep
                         --bess-steps <int>  Number of BESS size steps
                         --dg-max <kW>       Max DG rating to sweep
                         --dg-steps <int>    Number of DG size steps
                         --objective <str>   incremental_npv | lcoe | capex | ens
                         --max-capex-m <₹M>  Hard CAPEX ceiling (optional)
                         --top-n <int>       Top N candidates to display

energex monte-carlo      --config  <path>    Monte Carlo analysis (Phase C)
                         --trials  <int>     Number of trials (default: 200)
                         --seed    <int>
                         --export  <dir>
```

---

## Output Files

| File | Contents |
|---|---|
| `results_summary.json` | Full result: reliability, energy flows, financials, assumptions, warnings, run metadata |
| `reliability_metrics.csv` | ENS, downtime, autonomy, continuity, SLA pass/fail |
| `energy_flows.csv` | Hourly energy flows (8760 rows): grid, solar, BESS, DG, unserved |
| `cashflows.csv` | Year-by-year financial breakdown: grid cost, fuel, O&M, outage cost, NPV cashflows |
| `warnings.json` | Warnings list (undersized BESS, SLA violations, etc.) |
| `sizing_results.json` | Sizing optimizer: all candidates ranked by objective *(Phase C)* |
| `monte_carlo_results.json` | P10/P50/P90/P99 bands for all key metrics *(Phase C)* |

---

## Phase B — Interactive UI & Reporting

### Streamlit Dashboard

```bash
energex ui
# or: streamlit run energex/interfaces/streamlit_app.py
```

The dashboard provides 10 interactive tabs:

| Tab | Contents |
|---|---|
| 📊 Overview | Key metrics, SLA badge, energy flow chart |
| 🔴 Reliability | Monthly ENS, outage timeline, SLA targets |
| ⚡ Energy Flows | Hourly stacked area, BESS SOC, energy mix pie |
| 💰 Financial | Cashflows, cumulative payback, NPV metrics |
| 🌪 Sensitivity | Tornado charts for NPV and ENS |
| **📐 Sizing** | **Automated BESS + DG sizing optimizer** *(Phase C)* |
| **🎲 Monte Carlo** | **P50/P90/P99 distributions and bands** *(Phase C)* |
| **💡 TOU & Demand** | **TOU rate heatmap + monthly demand charge breakdown** *(Phase C)* |
| 📥 Export | Download ZIP, PDF report, JSON summary |
| 🔧 Config | View/download active configuration |

### PDF Report

```bash
energex pdf --config configs/grid_bess.json --sensitivity
```

Generates a professional PDF with executive summary and engineering appendix.

---

## Phase C — Automated Sizing, Monte Carlo, TOU/Demand Charges

### 1. Automated Optimal Sizing

**Problem**: Clients always ask *"What should I buy?"*, not just *"Is what I specified good enough?"*

**Solution**: `SizingOptimizer` sweeps a configurable grid of BESS capacity × DG rated-power combinations, runs the full simulation for each, and returns a ranked list of candidates.

#### Config block (optional, overrides CLI flags)

```json
"sizing": {
  "bess_sizes_kwh": [0, 100, 150, 200, 250, 300],
  "dg_sizes_kw":    [0, 100, 150],
  "bess_power_ratio": 0.5,
  "optimize_for": "incremental_npv",
  "max_capex_rs": 15000000,
  "require_sla_pass": true
}
```

#### Objectives

| `optimize_for` | Meaning |
|---|---|
| `incremental_npv` | Highest NPV improvement vs grid-only baseline *(default)* |
| `lcoe` | Lowest Levelized Cost of Energy |
| `capex` | Lowest capital cost (with SLA as constraint) |
| `ens` | Best reliability (lowest ENS) |

#### Output example

```
RECOMMENDED: BESS 200 kWh / 100 kW + DG 100 kW

  CAPEX          : ₹9.60M
  Incremental NPV: ₹6.40M
  Payback        : 7.2 yr
  LCOE           : ₹8.40/kWh
  ENS            : 12.5 kWh/yr
  Downtime       : 0.14 hrs/yr
  Continuity     : 99.998%
  Backup autonomy: 4.1 hrs
  SLA            : PASS

Alternatives (ranked):
  #2  BESS 300 kWh / 150 kW                   CAPEX ₹12.50M  ΔNPV +₹6.20M  PB 8.0 yr  ✓
  #3  BESS 150 kWh / 75 kW + DG 100 kW        CAPEX ₹7.45M   ΔNPV +₹5.80M  PB 6.8 yr  ✓
```

---

### 2. Monte Carlo with P50/P90/P99 Bands

**Problem**: A single-run ENS of "245 kWh/yr" is misleading — actual outcomes vary stochastically year-to-year. Lenders and insurers require statistically defensible confidence intervals.

**Solution**: `MonteCarloEngine` runs N independent trials (different seeds), collects the distribution, and computes P10/P50/P90/P99 bands.

#### Config block (optional)

```json
"monte_carlo": {
  "n_trials": 200,
  "confidence_levels": [0.10, 0.50, 0.90, 0.99]
}
```

#### Interpreting the bands

| Percentile | Meaning |
|---|---|
| **P50** | Median. Half of years are better, half worse. |
| **P90** | 90% of years will see this ENS or less. Use for SLA sizing. |
| **P99** | Near-worst-case (1-in-100 year event). Use for worst-case contracts. |
| **P10** | Used for NPV downside risk — the pessimistic financial case. |

#### Output example

```
Monte Carlo Results (200 trials)

Metric               P10      P50 (median)   P90      P99
ENS (kWh/yr)         42.1     158.4          380.2    621.8
Downtime (hrs/yr)    —        0.312          0.748    1.280
Continuity (%)       99.988   99.996         —        —
Project NPV (₹M)     −1.8     +3.2           +6.1     —
Outage Cost 25yr (₹M)—        4.8            11.2     18.6
LCOE (₹/kWh)         —        8.12           9.40     —
```

---

### 3. TOU Tariffs + Demand Charges

**Problem**: Without TOU and demand charge modelling, the financial model is wrong for most commercial/industrial consumers in India. Demand charges are 40–60% of the electricity bill for HT/EHT category consumers.

**Solution**: Extended `GridTariffModel` with `tou_blocks` and `demand_charge` fields.

#### TOU blocks

```json
"grid": {
  "energy_rate_rs_kwh": 7.50,
  "fixed_charge_rs_month": 2000.0,
  "escalation_pct_yr": 5.5,
  "tou_blocks": [
    {
      "name": "peak",
      "rate_rs_kwh": 9.50,
      "start_hour": 8,
      "end_hour": 22,
      "day_type": "weekday"
    },
    {
      "name": "shoulder",
      "rate_rs_kwh": 7.50,
      "start_hour": 6,
      "end_hour": 22,
      "day_type": "weekend"
    },
    {
      "name": "off_peak",
      "rate_rs_kwh": 5.50,
      "start_hour": 22,
      "end_hour": 6,
      "day_type": "all"
    }
  ]
}
```

- `start_hour` and `end_hour` are 0–23 (inclusive).
- If `start_hour > end_hour`, the block wraps midnight (e.g., 22:00–06:00).
- `day_type`: `"all"`, `"weekday"` (Mon–Fri), or `"weekend"` (Sat–Sun).
- Hours not covered by any block fall back to `energy_rate_rs_kwh`.

#### Demand charges

```json
"demand_charge": {
  "charge_rs_kw_month": 300.0,
  "ratchet_pct": 80.0,
  "contracted_demand_kw": 100.0
}
```

| Field | Meaning |
|---|---|
| `charge_rs_kw_month` | ₹/kW/month applied to monthly peak grid draw |
| `ratchet_pct` | Min billable demand = ratchet_pct% × annual peak. Protects DISCOM revenue. |
| `contracted_demand_kw` | Contracted/sanctioned capacity — floor for monthly billing. |

The finance engine applies these charges based on hourly simulation data — computing monthly peak demand from the 8760 hourly grid draws.

---

## Config Format

### Minimal Grid + BESS config

```json
{
  "name": "My Project",
  "project_lifetime_years": 25,
  "discount_rate_pct": 10.0,
  "simulation_year": 2025,

  "load": {
    "mode": "simple",
    "avg_kw": 100.0,
    "peak_kw": 140.0,
    "critical_fraction": 0.35
  },

  "outage": {
    "mode": "stochastic",
    "stochastic": {
      "saidi_minutes_per_year": 1200,
      "saifi_events_per_year": 18,
      "duration_distribution": "lognormal",
      "duration_sigma": 0.9
    }
  },

  "grid": {
    "energy_rate_rs_kwh": 8.5,
    "fixed_charge_rs_month": 5000,
    "escalation_pct_yr": 5.0
  },

  "bess": {
    "capacity_kwh": 150.0,
    "power_kw": 75.0,
    "dod_pct": 90.0,
    "roundtrip_efficiency": 0.92,
    "min_reserve_soc_pct": 10.0,
    "initial_soc_pct": 90.0,
    "calendar_fade_pct_yr": 2.0,
    "cycle_fade_per_kwh": 0.000025,
    "calendar_life_yr": 12.0,
    "eol_capacity_pct": 80.0,
    "capex_rs_kwh": 35000,
    "replacement_cost_fraction": 0.60,
    "om_rs_kwh_yr": 500,
    "availability_pct": 99.0
  },

  "outage_cost": {
    "mode": "voll",
    "voll_rs_kwh": 150.0,
    "sla_max_downtime_hrs_yr": 8.0,
    "sla_continuity_pct": 99.9,
    "sla_min_autonomy_hours": 4.0
  }
}
```

### Load timeseries CSV format

```
hour,total_kw,critical_kw
0,95.2,28.5
1,88.0,26.4
...
8759,102.1,30.6
```

Set `"load": {"mode": "timeseries", "timeseries_csv": "path/to/load.csv"}` in config.

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
│   └── schemas.py              # Pydantic v2 models (TOU, demand, MC, sizing)
├── engine/
│   ├── outage_generator.py     # Deterministic + SAIDI/SAIFI stochastic outages
│   ├── dispatch_backup.py      # Hourly backup dispatch (BESS, DG, Solar)
│   ├── finance_engine.py       # NPV, IRR, payback, LCOE, TOU billing, demand charges
│   ├── runner.py               # Full pipeline orchestration
│   ├── monte_carlo.py          # Monte Carlo engine (P50/P90/P99) [Phase C]
│   └── sizing_optimizer.py     # Automated BESS+DG sizing [Phase C]
├── adapters/
│   ├── config_loader.py        # JSON/YAML config loading
│   ├── csv_parser.py           # Timeseries CSV parsing
│   ├── charts.py               # Plotly + Matplotlib charts (incl. MC/sizing)
│   ├── pdf_report.py           # PDF report generation [Phase B]
│   └── exporters.py            # JSON/CSV result export
├── interfaces/
│   ├── cli.py                  # Typer CLI (run, size, monte-carlo, ui, pdf)
│   └── streamlit_app.py        # Interactive dashboard (10 tabs) [Phase B+C]
├── tests/
│   ├── test_outage_generator.py
│   ├── test_dispatch.py
│   ├── test_finance.py
│   ├── test_golden.py
│   ├── test_tou.py             # TOU + demand charge tests [Phase C]
│   ├── test_monte_carlo.py     # Monte Carlo tests [Phase C]
│   └── test_sizing_optimizer.py # Sizing optimizer tests [Phase C]
└── configs/
    ├── grid_bess.json
    ├── grid_dg.json
    ├── grid_solar_bess.json
    ├── dg_bess_hybrid.json
    └── grid_bess_tou.json      # TOU + demand charge example [Phase C]
```

### Dispatch logic (grid-down)

```
Outage detected
    │
    ├─► Solar (if available) → critical load
    │
    ├─► BESS (instant) → remaining critical load up to power_kw and SOC floor
    │
    └─► DG (after start_delay_seconds) → remaining critical load
            │
            └─► If all sources insufficient → ENS recorded, downtime logged
```

### Financial model

- **Year 0**: CAPEX (BESS + Solar + DG)
- **Year 1–N**: Grid energy (TOU or flat), diesel fuel, O&M, demand charges, BESS replacement at EOL, outage cost
- **Outage cost**: `ENS_kWh × VoLL (₹/kWh)` OR `downtime_hrs × ₹/hr` OR tiered penalties
- **TOU billing**: Per-hour energy rate × kWh consumed (if `tou_blocks` configured)
- **Demand charges**: Monthly peak kW × rate + ratchet clause (if `demand_charge` configured)
- **NPV**: Discounted at project discount rate
- **LCOE**: PV(all supply costs) / total energy served

---

## Engine Design Notes

- **Timebase**: Hourly (8760 slots/year).
- **Determinism**: All stochastic runs are seed-controlled. Same seed → identical results.
- **BESS degradation**: Calendar fade (%/yr) + cycle fade (per kWh throughput) per year.
- **DG availability**: Each hour randomly drawn vs `availability_pct`. Failed hours → DG unavailable.
- **TOU billing**: `rate_for_hour()` on `GridTariffModel` resolves the applicable block per hour. Falls back to flat rate when no block matches.
- **Demand charges**: Finance engine groups hourly grid draws into 12 calendar months, finds monthly peak, applies ratchet and contracted demand floor.
- **Monte Carlo seed spacing**: Trial seeds are spaced by prime step (31337) for statistical independence.
- **Sizing optimizer**: Clones base config per combination; injects default BESS/DG templates if none present in base config.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just Phase C tests
pytest tests/test_tou.py tests/test_monte_carlo.py tests/test_sizing_optimizer.py -v

# With coverage
pytest tests/ --cov=energex --cov-report=term-missing
```

**Test count: 120 tests** across 7 test files:

| Test file | Count | Scope |
|---|---|---|
| `test_outage_generator.py` | 16 | Deterministic/stochastic outage generation |
| `test_dispatch.py` | 22 | BESS/DG unit tests + dispatch integration |
| `test_finance.py` | 12 | Cashflow, NPV, IRR, LCOE |
| `test_golden.py` | 25 | End-to-end golden runs for all 4 example configs |
| `test_tou.py` | 16 | TOU block logic, demand charge, finance integration |
| `test_monte_carlo.py` | 12 | MC engine, percentile bands, trial variation |
| `test_sizing_optimizer.py` | 17 | Sizing candidate, config builder, ranking, run |

---

## Key Outputs Explained

| Metric | Definition |
|---|---|
| **ENS** | Energy Not Served — kWh of critical load that could not be met during outages |
| **Downtime** | Hours during which critical load was fully or partially unserved |
| **Continuity %** | `1 - ENS / total_critical_kWh` × 100 |
| **Backup Autonomy** | Hours BESS (+ DG if present) can serve peak critical load continuously |
| **LCOE** | Levelised Cost of Energy = PV(all supply costs) / total energy |
| **VoLL** | Value of Lost Load — opportunity cost of 1 kWh unserved (₹/kWh) |
| **Incremental NPV** | NPV of this scenario minus NPV of grid-only baseline |
| **Demand Charge** | Monthly kW-based billing component (₹/kW/month × peak demand) |
| **P50/P90/P99** | Monte Carlo percentile bands — P90 means 90% of years are at or below this value |

---

## Phase B — Completed Features

- ✅ Streamlit interactive dashboard (10 tabs including Phase C)
- ✅ PDF report generation (executive summary + engineering appendix)
- ✅ Sensitivity tornado charts
- ✅ Interactive Plotly charts (energy flows, BESS SOC, outage timeline, payback curve)
- ✅ ZIP export (CSV/JSON/PDF bundle)

## Phase C — Completed Features

- ✅ **Automated optimal sizing** — `SizingOptimizer` sweeps BESS × DG grid, ranks by objective
- ✅ **Monte Carlo P50/P90/P99** — `MonteCarloEngine` builds statistical confidence intervals
- ✅ **TOU tariffs** — `GridTariffModel.tou_blocks` with midnight-wrap, weekday/weekend support
- ✅ **Demand charges** — `GridTariffModel.demand_charge` with ratchet + contracted demand floor
- ✅ **CLI commands** — `energex size`, `energex monte-carlo`
- ✅ **Streamlit tabs** — 📐 Sizing, 🎲 Monte Carlo, 💡 TOU & Demand
- ✅ **New config preset** — `configs/grid_bess_tou.json` (Maharashtra HT consumer example)
- ✅ **45 new tests** — TOU, Monte Carlo, and sizing test suites

## Future Roadmap

- 15-minute simulation timebase for sub-hourly demand peaks
- Multi-site portfolio aggregation
- LOLP (Loss of Load Probability) computation
- Real-time tariff data integration (DISCOM API)
- Battery second-life economics

---

## License

MIT

---

*EnergeX v1.2.0 — Phase C: Automated Sizing · Monte Carlo · TOU/Demand Charges*
