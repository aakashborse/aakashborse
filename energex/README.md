# EnergeX — Backup/Outage Energy ROI Calculator

**Production-grade platform for reliability-aware energy system techno-economics.**

EnergeX models grid outages, backup assets (DG, BESS, Solar), and their combined economics — answering *"How much does unreliability actually cost, and what is the cheapest configuration to fix it?"*

---

## Key Capabilities

| Domain | What EnergeX Computes |
|---|---|
| **Reliability** | ENS (kWh/yr), Downtime (hrs/yr), Continuity (%), Backup Autonomy (hrs) |
| **Dispatch** | Hourly BESS/DG backup dispatch with SOC tracking, start delays, fuel constraints |
| **Finance** | NPV, IRR, Payback, LCOE — all including outage cost as a first-class cashflow |
| **Outage Modelling** | Deterministic schedule OR stochastic SAIDI/SAIFI with seasonal weighting |
| **Sensitivity** | Tornado sweeps on SAIDI, VoLL, diesel price, BESS capex, critical load |

---

## Supported Scenarios

| Scenario | Config Example |
|---|---|
| Grid only (baseline) | Implicit — computed automatically |
| Grid + DG backup | `configs/grid_dg.json` |
| Grid + BESS backup | `configs/grid_bess.json` |
| Grid + Solar + BESS | `configs/grid_solar_bess.json` |
| Grid + DG + BESS hybrid | `configs/dg_bess_hybrid.json` |
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
energex run   --config  <path>   # JSON or YAML config file
              --seed    <int>    # Random seed (default: 42)
              --export  <dir>    # Output directory (default: energex_output/)
              --baseline/--no-baseline  # Compute grid-only NPV baseline

energex validate --config <path>

energex generate-sample-csv --output-dir <dir> --avg-kw <kW> --solar-kw <kW>
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

---

## Config Format

All configs are JSON (or YAML). Full schema is defined in `domain/schemas.py`.

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

### Solar timeseries CSV format

```
hour,yield_kw
0,0.0
6,12.5
12,48.3
...
8759,0.0
```

Set in `bess.yield_timeseries_csv`.

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
│   └── schemas.py          # Pydantic v2 models for all inputs
├── engine/
│   ├── outage_generator.py # Deterministic + SAIDI/SAIFI stochastic outages
│   ├── dispatch_backup.py  # Hourly backup dispatch (BESS, DG, Solar)
│   ├── finance_engine.py   # NPV, IRR, payback, LCOE, sensitivity
│   └── runner.py           # Full pipeline orchestration
├── adapters/
│   ├── config_loader.py    # JSON/YAML config loading
│   ├── csv_parser.py       # Timeseries CSV parsing + sample generation
│   └── exporters.py        # JSON/CSV result export
├── interfaces/
│   └── cli.py              # Typer CLI
├── tests/
│   ├── test_outage_generator.py
│   ├── test_dispatch.py
│   ├── test_finance.py
│   └── test_golden.py      # End-to-end golden tests for all 4 configs
└── configs/
    ├── grid_bess.json
    ├── grid_dg.json
    ├── grid_solar_bess.json
    └── dg_bess_hybrid.json
```

### Dispatch logic (grid-down)

```
Outage detected
    │
    ├─► BESS (instant) → serves critical load up to power_kw and SOC floor
    │
    └─► DG (after start_delay_seconds) → serves remaining critical load
            │
            └─► If both insufficient → ENS recorded, downtime logged
```

### Financial model

- **Year 0**: CAPEX (BESS + Solar + DG)
- **Year 1–N**: Grid energy, diesel fuel, O&M, BESS replacement at EOL, outage cost
- **Outage cost**: `ENS_kWh × VoLL (₹/kWh)` OR `downtime_hrs × ₹/hr` OR tiered penalties
- **NPV**: Discounted at project discount rate
- **LCOE**: PV(all costs) / total energy served

---

## Engine Design Notes

- **Timebase**: Hourly (8760 slots/year). Sub-hourly (15-min) planned for Phase B.
- **Determinism**: All stochastic runs are seed-controlled. Same seed → identical outages and results.
- **BESS degradation**: Calendar fade (%/yr) + cycle fade (per kWh throughput) applied per year.
- **DG availability**: Each hour randomly drawn vs `availability_pct`. Failed hours → DG unavailable.
- **Dispatch priority**: BESS first (instant), DG second (after start delay). Solar offsets load and charges BESS during grid-up periods.
- **SLA evaluation**: `downtime ≤ max`, `ENS ≤ max`, `continuity ≥ target`, `autonomy ≥ required`.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just golden end-to-end tests
pytest tests/test_golden.py -v

# With coverage
pytest tests/ --cov=energex --cov-report=term-missing
```

**Test count: 75 tests** across 4 test files:
- `test_outage_generator.py` — deterministic/stochastic outage generation (16 tests)
- `test_dispatch.py` — BESS/DG unit tests + dispatch integration (22 tests)
- `test_finance.py` — cashflow, NPV, IRR, LCOE (12 tests)
- `test_golden.py` — end-to-end golden runs for all 4 example configs (25 tests)

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

---

## Phase B (Roadmap)

- Streamlit interactive UI with charts
- PDF report generation (executive summary + engineering appendix)
- Sensitivity tornado charts (visualised)
- 15-minute simulation timebase
- TOU tariff support
- Multi-site aggregation
- LOLP (Loss of Load Probability) computation

---

## License

MIT

---

*EnergeX v1.0.0 — Built for reliability-first energy economics.*
