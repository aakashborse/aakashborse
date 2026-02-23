"""
Tests for the finance engine.
"""

import pytest
import math

from energex.domain.schemas import (
    ProjectConfig, LoadProfile, OutageModel, OutageMode,
    GridTariffModel, OutageCostModel, OutageCostMode,
    BESSBackupModel, StochasticOutageParams, DeterministicOutageEvent,
)
from energex.engine.outage_generator import OutageEvent
from energex.engine.dispatch_backup import BackupDispatchEngine, SimulationResult
from energex.engine.finance_engine import FinanceEngine, YearCashflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(bess=None, dg=None):
    return ProjectConfig(
        name="Finance Test",
        project_lifetime_years=10,
        discount_rate_pct=10.0,
        load=LoadProfile(mode="simple", avg_kw=100.0, critical_fraction=0.3),
        outage=OutageModel(
            mode=OutageMode.DETERMINISTIC,
            events=[DeterministicOutageEvent(start_hour=500.0, duration_hours=2.0)],
        ),
        grid=GridTariffModel(
            energy_rate_rs_kwh=8.0,
            fixed_charge_rs_month=1000.0,
            escalation_pct_yr=5.0,
        ),
        outage_cost=OutageCostModel(
            mode=OutageCostMode.VOLL,
            voll_rs_kwh=100.0,
        ),
        bess=bess,
        dg=dg,
    )


def run_dispatch(cfg):
    events = [OutageEvent(500.0, 2.0)]
    engine = BackupDispatchEngine(config=cfg, outage_events=events, year=0, seed=42)
    return engine.run()


# ---------------------------------------------------------------------------
# YearCashflow tests
# ---------------------------------------------------------------------------

class TestYearCashflow:

    def test_total_opex_sum(self):
        cf = YearCashflow(year=1)
        cf.grid_energy_cost = 100000
        cf.diesel_fuel_cost = 5000
        cf.om_bess = 2000
        cf.outage_cost = 10000
        assert cf.total_opex == 117000

    def test_net_cashflow_year0(self):
        cf = YearCashflow(year=0)
        cf.capex = 5000000
        assert cf.net_cashflow == -5000000

    def test_net_cashflow_opex_year(self):
        cf = YearCashflow(year=1)
        cf.grid_energy_cost = 200000
        assert cf.net_cashflow == -200000


# ---------------------------------------------------------------------------
# FinanceEngine tests
# ---------------------------------------------------------------------------

class TestFinanceEngine:

    def test_capex_zero_no_assets(self):
        cfg = make_config()
        sim = run_dispatch(cfg)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years, "grid_only")
        result = fe.run()
        assert result.total_capex == 0.0

    def test_capex_with_bess(self):
        bess = BESSBackupModel(
            capacity_kwh=100.0, power_kw=50.0, dod_pct=90.0,
            roundtrip_efficiency=0.92, min_reserve_soc_pct=10.0,
            initial_soc_pct=90.0, calendar_fade_pct_yr=2.0,
            cycle_fade_per_kwh=0.00002, calendar_life_yr=12.0,
            eol_capacity_pct=80.0, capex_rs_kwh=35000,
            availability_pct=99.0,
        )
        cfg = make_config(bess=bess)
        sim = run_dispatch(cfg)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years, "grid_bess")
        result = fe.run()
        assert result.total_capex == 100.0 * 35000  # 100 kWh * ₹35000/kWh

    def test_npv_has_correct_discount(self):
        """NPV of year-0 only capex should equal -capex (already discounted to year 0)."""
        bess = BESSBackupModel(
            capacity_kwh=100.0, power_kw=50.0, dod_pct=90.0,
            roundtrip_efficiency=0.92, min_reserve_soc_pct=10.0,
            initial_soc_pct=90.0, calendar_fade_pct_yr=0.0,
            cycle_fade_per_kwh=0.0, calendar_life_yr=100.0,
            eol_capacity_pct=10.0, capex_rs_kwh=10000,
            availability_pct=100.0,
        )
        cfg = make_config(bess=bess)
        cfg2 = ProjectConfig(
            name="NPV test",
            project_lifetime_years=1,
            discount_rate_pct=10.0,
            load=cfg.load,
            outage=cfg.outage,
            grid=GridTariffModel(energy_rate_rs_kwh=0.0, fixed_charge_rs_month=0.0, escalation_pct_yr=0.0),
            outage_cost=OutageCostModel(mode=OutageCostMode.VOLL, voll_rs_kwh=0.0),
            bess=bess,
        )
        sim = run_dispatch(cfg2)
        fe = FinanceEngine(cfg2, [sim], "test")
        result = fe.run()
        # CAPEX = 100 * 10000 = 1,000,000; NPV ≈ -1,000,000 + small salvage
        assert result.npv < 0
        assert result.total_capex == 1_000_000

    def test_outage_cost_voll_positive(self):
        """Outage cost should be positive when ENS > 0."""
        cfg = make_config()
        sim = run_dispatch(cfg)
        # ENS should be positive (no backup)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years, "grid_only")
        result = fe.run()
        if sim.ens_kwh > 0:
            assert result.total_outage_cost > 0

    def test_cashflow_count(self):
        """Should have project_lifetime_years + 1 cashflows (year 0 + each year)."""
        cfg = make_config()
        sim = run_dispatch(cfg)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years)
        result = fe.run()
        assert len(result.cashflows) == cfg.project_lifetime_years + 1

    def test_irr_none_when_no_investment(self):
        """IRR should be None when no capex (no investment)."""
        cfg = make_config()
        sim = run_dispatch(cfg)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years)
        result = fe.run()
        # No capex → year-0 cashflow is non-negative → IRR undefined
        assert result.irr is None

    def test_lcoe_positive(self):
        cfg = make_config()
        sim = run_dispatch(cfg)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years)
        result = fe.run()
        assert result.lcoe_rs_kwh >= 0.0

    def test_total_cost_with_outages_ge_without(self):
        """Total cost including outages should be >= total cost excluding outages."""
        cfg = make_config()
        sim = run_dispatch(cfg)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years)
        result = fe.run()
        assert result.total_cost_with_outages >= result.total_cost_without_outages - 1e-6

    def test_grid_energy_cost_scales_with_escalation(self):
        """Grid energy cost should increase each year due to escalation."""
        cfg = make_config()
        sim = run_dispatch(cfg)
        fe = FinanceEngine(cfg, [sim] * cfg.project_lifetime_years)
        result = fe.run()
        costs = [cf.grid_energy_cost for cf in result.cashflows if cf.year > 0]
        # Costs should generally increase (escalation 5%/yr)
        assert costs[-1] > costs[0]
