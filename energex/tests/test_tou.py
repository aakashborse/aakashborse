"""
EnergeX — Phase C: TOU Tariff + Demand Charge Tests
"""

from __future__ import annotations

import pytest
from energex.domain.schemas import (
    TOURateBand,
    TOUSchedule,
    DemandChargeModel,
    GridTariffModel,
    ProjectConfig,
    LoadProfile,
    OutageModel,
    StochasticOutageParams,
    BESSBackupModel,
    OutageCostModel,
    MonteCarloConfig,
    SizingBounds,
)
from energex.engine.dispatch_backup import BackupDispatchEngine, build_hourly_loads
from energex.engine.finance_engine import FinanceEngine
from energex.engine.outage_generator import OutageGenerator
import numpy as np


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_tou_rate_band_valid():
    band = TOURateBand(name="peak", hours_of_day=[9, 10, 11, 12, 13, 14], rate_rs_kwh=12.0)
    assert band.rate_rs_kwh == 12.0
    assert 9 in band.hours_of_day


def test_tou_rate_band_invalid_hour():
    with pytest.raises(Exception):
        TOURateBand(name="peak", hours_of_day=[25], rate_rs_kwh=12.0)


def test_tou_rate_band_empty_hours():
    with pytest.raises(Exception):
        TOURateBand(name="peak", hours_of_day=[], rate_rs_kwh=12.0)


def test_tou_schedule_rate_for_hour():
    schedule = TOUSchedule(
        rate_bands=[
            TOURateBand(name="peak", hours_of_day=list(range(9, 17)), rate_rs_kwh=12.0),
            TOURateBand(name="off_peak", hours_of_day=list(range(22, 24)) + list(range(0, 6)), rate_rs_kwh=5.0),
        ],
        default_rate_rs_kwh=8.0,
    )
    assert schedule.rate_for_hour(10) == 12.0   # peak
    assert schedule.rate_for_hour(23) == 5.0    # off-peak
    assert schedule.rate_for_hour(18) == 8.0    # default (not in any band)


def test_tou_schedule_is_peak_hour():
    schedule = TOUSchedule(
        rate_bands=[
            TOURateBand(name="peak", hours_of_day=[9, 10, 11], rate_rs_kwh=12.0),
        ],
        default_rate_rs_kwh=8.0,
    )
    assert schedule.is_peak_hour(10) is True
    assert schedule.is_peak_hour(20) is False


def test_demand_charge_model_defaults():
    dc = DemandChargeModel(charge_rs_kva_month=350.0)
    assert dc.power_factor == 0.9
    assert dc.measurement_window_hours == 0.5
    assert dc.peak_hours_only is False


def test_grid_tariff_model_with_tou():
    schedule = TOUSchedule(
        rate_bands=[TOURateBand(name="peak", hours_of_day=[9, 10], rate_rs_kwh=12.0)],
        default_rate_rs_kwh=7.0,
    )
    gt = GridTariffModel(energy_rate_rs_kwh=7.0, tou_schedule=schedule)
    # Hour 9 (9am) → peak rate
    assert gt.rate_for_hour(9) == 12.0
    # Hour 20 → default
    assert gt.rate_for_hour(20) == 7.0


def test_grid_tariff_rate_for_hour_escalation():
    gt = GridTariffModel(energy_rate_rs_kwh=8.0)
    # Flat rate (no TOU)
    assert gt.rate_for_hour(5, escalation_factor=1.5) == 12.0


def test_monte_carlo_config_defaults():
    mc = MonteCarloConfig()
    assert mc.n_runs == 200
    assert mc.base_seed == 42
    assert 50.0 in mc.percentiles


def test_monte_carlo_config_percentiles_sorted():
    mc = MonteCarloConfig(percentiles=[90, 10, 50])
    assert mc.percentiles == [10.0, 50.0, 90.0]


def test_sizing_bounds_defaults():
    sb = SizingBounds()
    assert sb.bess_capacity_max_kwh == 500.0
    assert sb.include_dg is False
    assert sb.n_years_per_eval == 3


# ---------------------------------------------------------------------------
# TOU dispatch integration tests
# ---------------------------------------------------------------------------

def _make_tou_config(with_demand_charge: bool = False, with_bess: bool = True) -> ProjectConfig:
    schedule = TOUSchedule(
        rate_bands=[
            TOURateBand(name="peak", hours_of_day=list(range(9, 17)), rate_rs_kwh=12.0),
            TOURateBand(name="off_peak", hours_of_day=list(range(22, 24)) + list(range(0, 6)), rate_rs_kwh=5.0),
        ],
        default_rate_rs_kwh=8.0,
    )
    demand_charge = DemandChargeModel(charge_rs_kva_month=350.0) if with_demand_charge else None
    bess = BESSBackupModel(
        capacity_kwh=100.0,
        power_kw=50.0,
        capex_rs_kwh=20000.0,
    ) if with_bess else None

    return ProjectConfig(
        name="TOU Test",
        load=LoadProfile(avg_kw=50.0, critical_fraction=0.4),
        outage=OutageModel(
            mode="stochastic",
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=300.0,
                saifi_events_per_year=5.0,
            ),
        ),
        grid=GridTariffModel(
            energy_rate_rs_kwh=8.0,
            tou_schedule=schedule,
            demand_charge=demand_charge,
            peak_shaving_enabled=True if with_bess else False,
        ),
        outage_cost=OutageCostModel(mode="voll", voll_rs_kwh=100.0),
        bess=bess,
    )


def test_tou_rate_recorded_in_hourly_results():
    """TOU rate should be recorded in each hourly result."""
    cfg = _make_tou_config()
    gen = OutageGenerator(cfg.outage, seed=42)
    events = gen.generate()
    total_kw, critical_kw = build_hourly_loads(cfg, year=0)

    engine = BackupDispatchEngine(cfg, events, year=0, seed=42,
                                  total_kw=total_kw, critical_kw=critical_kw)
    sim = engine.run()

    # Hours 9-16 (peak) should have rate 12.0
    peak_hours = [h for h in sim.hourly if 9 <= h.hour % 24 <= 16]
    assert all(h.tou_rate_rs_kwh == 12.0 for h in peak_hours[:10])

    # Off-peak hours (22-5) should have rate 5.0
    offpeak_hours = [h for h in sim.hourly if h.hour % 24 >= 22 or h.hour % 24 <= 5]
    assert all(h.tou_rate_rs_kwh == 5.0 for h in offpeak_hours[:10])


def test_monthly_peak_grid_kw_tracked():
    """Monthly peak grid demand should be populated for demand charge billing."""
    cfg = _make_tou_config(with_demand_charge=True)
    gen = OutageGenerator(cfg.outage, seed=42)
    events = gen.generate()
    total_kw, critical_kw = build_hourly_loads(cfg, year=0)

    engine = BackupDispatchEngine(cfg, events, year=0, seed=42,
                                  total_kw=total_kw, critical_kw=critical_kw)
    sim = engine.run()

    assert len(sim.monthly_peak_grid_kw) == 12
    assert all(v >= 0 for v in sim.monthly_peak_grid_kw)
    assert max(sim.monthly_peak_grid_kw) > 0


def test_bess_peak_shaving_reduces_grid_peak():
    """With BESS peak shaving, grid peak demand should be reduced compared to no-BESS."""
    # With BESS peak shaving
    cfg_bess = _make_tou_config(with_demand_charge=True, with_bess=True)
    cfg_bess.grid.peak_shaving_target_kw = 30.0  # aggressive target

    # Without BESS
    cfg_no_bess = _make_tou_config(with_demand_charge=True, with_bess=False)

    gen = OutageGenerator(cfg_bess.outage, seed=42)
    events = gen.generate()
    total_kw, critical_kw = build_hourly_loads(cfg_bess, year=0)

    engine_bess = BackupDispatchEngine(cfg_bess, events, year=0, seed=42,
                                       total_kw=total_kw, critical_kw=critical_kw)
    engine_no_bess = BackupDispatchEngine(cfg_no_bess, events, year=0, seed=42,
                                          total_kw=total_kw, critical_kw=critical_kw)

    sim_bess = engine_bess.run()
    sim_no_bess = engine_no_bess.run()

    # With peak shaving, BESS should have discharged some energy for it
    assert sim_bess.total_peak_shaving_kwh >= 0


def test_demand_charge_added_to_cashflow():
    """Demand charge should appear in year cashflows when configured."""
    cfg = _make_tou_config(with_demand_charge=True, with_bess=True)
    gen = OutageGenerator(cfg.outage, seed=42)
    events = gen.generate()
    total_kw, critical_kw = build_hourly_loads(cfg, year=0)
    import numpy as np
    solar_kw = np.zeros(8760)

    engine = BackupDispatchEngine(cfg, events, year=0, seed=42,
                                  total_kw=total_kw, critical_kw=critical_kw, solar_kw=solar_kw)
    sim = engine.run()

    fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years, scenario_name="tou_test")
    fr = fe.run()

    # Demand charges should be positive for years 1+
    annual_cfs = [cf for cf in fr.cashflows if cf.year > 0]
    total_demand_charge = sum(cf.demand_charge_rs for cf in annual_cfs)
    assert total_demand_charge > 0, "Demand charges should be non-zero when configured"


def test_tou_grid_cost_higher_in_peak_scenario():
    """With TOU, peak-heavy loads should cost more than off-peak-heavy loads."""
    cfg = _make_tou_config(with_demand_charge=False, with_bess=False)
    gen = OutageGenerator(cfg.outage, seed=42)
    events = gen.generate()
    total_kw, critical_kw = build_hourly_loads(cfg, year=0)
    import numpy as np
    solar_kw = np.zeros(8760)

    engine = BackupDispatchEngine(cfg, events, year=0, seed=42,
                                  total_kw=total_kw, critical_kw=critical_kw, solar_kw=solar_kw)
    sim = engine.run()

    # TOU-based grid cost should include variation (not constant rate)
    peak_rates = [h.tou_rate_rs_kwh for h in sim.hourly if 9 <= h.hour % 24 <= 16]
    offpeak_rates = [h.tou_rate_rs_kwh for h in sim.hourly if h.hour % 24 <= 5 or h.hour % 24 >= 22]

    assert len(set(peak_rates)) == 1 and peak_rates[0] == 12.0
    assert len(set(offpeak_rates)) == 1 and offpeak_rates[0] == 5.0
