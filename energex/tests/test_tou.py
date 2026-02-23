"""
Tests for TOU tariff blocks and demand charges.
"""
from __future__ import annotations

import pytest
from energex.domain.schemas import (
    GridTariffModel,
    TOUBlock,
    DemandChargeModel,
    TOUDayType,
    ProjectConfig,
)


# ---------------------------------------------------------------------------
# TOUBlock.applies_to_hour
# ---------------------------------------------------------------------------

class TestTOUBlockApplies:
    def test_simple_daytime_weekday(self):
        blk = TOUBlock(name="peak", rate_rs_kwh=9.5, start_hour=8, end_hour=22,
                       day_type=TOUDayType.WEEKDAY)
        assert blk.applies_to_hour(8, is_weekday=True)
        assert blk.applies_to_hour(12, is_weekday=True)
        assert blk.applies_to_hour(22, is_weekday=True)
        assert not blk.applies_to_hour(7, is_weekday=True)
        assert not blk.applies_to_hour(23, is_weekday=True)
        assert not blk.applies_to_hour(12, is_weekday=False)  # weekend

    def test_midnight_wrapping(self):
        # Off-peak: 22:00–06:00 (wraps midnight)
        blk = TOUBlock(name="off_peak", rate_rs_kwh=5.5, start_hour=22, end_hour=6,
                       day_type=TOUDayType.ALL)
        assert blk.applies_to_hour(22, is_weekday=True)
        assert blk.applies_to_hour(0, is_weekday=True)
        assert blk.applies_to_hour(3, is_weekday=True)
        assert blk.applies_to_hour(6, is_weekday=True)
        assert not blk.applies_to_hour(7, is_weekday=True)
        assert not blk.applies_to_hour(12, is_weekday=True)

    def test_weekend_only(self):
        blk = TOUBlock(name="shoulder", rate_rs_kwh=7.5, start_hour=8, end_hour=20,
                       day_type=TOUDayType.WEEKEND)
        assert blk.applies_to_hour(10, is_weekday=False)
        assert not blk.applies_to_hour(10, is_weekday=True)

    def test_all_day_type(self):
        blk = TOUBlock(name="all", rate_rs_kwh=6.0, start_hour=0, end_hour=23,
                       day_type=TOUDayType.ALL)
        assert blk.applies_to_hour(0, is_weekday=True)
        assert blk.applies_to_hour(23, is_weekday=False)


# ---------------------------------------------------------------------------
# GridTariffModel.rate_for_hour
# ---------------------------------------------------------------------------

class TestGridTariffRateForHour:
    def _make_tou_tariff(self):
        return GridTariffModel(
            energy_rate_rs_kwh=7.5,
            fixed_charge_rs_month=2000.0,
            escalation_pct_yr=5.0,
            tou_blocks=[
                TOUBlock(name="peak", rate_rs_kwh=9.5, start_hour=8, end_hour=22,
                         day_type=TOUDayType.WEEKDAY),
                TOUBlock(name="off_peak", rate_rs_kwh=5.5, start_hour=22, end_hour=7,
                         day_type=TOUDayType.ALL),
            ],
        )

    def test_flat_rate_no_tou(self):
        tariff = GridTariffModel(energy_rate_rs_kwh=8.0, escalation_pct_yr=5.0)
        assert tariff.rate_for_hour(100) == pytest.approx(8.0)
        assert tariff.rate_for_hour(0) == pytest.approx(8.0)

    def test_tou_peak_weekday(self):
        tariff = self._make_tou_tariff()
        # Hour 8 on a Monday (day_of_week=0): peak
        # day_of_week = (hour // 24) % 7 ; 8//24=0, 0%7=0 (Monday) → weekday
        assert tariff.rate_for_hour(8) == pytest.approx(9.5)

    def test_tou_offpeak(self):
        tariff = self._make_tou_tariff()
        # Hour 23 → hod=23, day_of_week=0 (Mon)
        # off_peak covers 22-7 (wrapping), so hour 23 → off_peak
        assert tariff.rate_for_hour(23) == pytest.approx(5.5)

    def test_tou_fallback_on_unmatched_hour(self):
        # No block covers 8-22 on weekends in this config → fallback to flat rate 7.5
        tariff = GridTariffModel(
            energy_rate_rs_kwh=7.5,
            tou_blocks=[
                TOUBlock(name="peak", rate_rs_kwh=9.5, start_hour=8, end_hour=22,
                         day_type=TOUDayType.WEEKDAY),
            ],
        )
        # Hour 8 on Saturday: day_of_week = (8//24=0) % 7 = 0 → Monday (weekday)
        # Actually, let's use hour 8 + 5*24=128 (Saturday):
        # 128//24=5, 5%7=5 (Saturday) → weekend
        assert tariff.rate_for_hour(128) == pytest.approx(7.5)  # fallback


# ---------------------------------------------------------------------------
# DemandChargeModel validation
# ---------------------------------------------------------------------------

class TestDemandChargeModel:
    def test_basic_fields(self):
        dc = DemandChargeModel(charge_rs_kw_month=300.0, ratchet_pct=80.0,
                               contracted_demand_kw=100.0)
        assert dc.charge_rs_kw_month == pytest.approx(300.0)
        assert dc.ratchet_pct == pytest.approx(80.0)
        assert dc.contracted_demand_kw == pytest.approx(100.0)

    def test_no_ratchet(self):
        dc = DemandChargeModel(charge_rs_kw_month=200.0)
        assert dc.ratchet_pct is None
        assert dc.contracted_demand_kw is None

    def test_invalid_ratchet(self):
        with pytest.raises(Exception):
            DemandChargeModel(charge_rs_kw_month=300.0, ratchet_pct=150.0)  # >100%


# ---------------------------------------------------------------------------
# Finance engine: TOU + demand charge integration
# ---------------------------------------------------------------------------

def _make_project_config_with_tou():
    """Build a minimal ProjectConfig with TOU blocks and demand charge."""
    from energex.domain.schemas import (
        ProjectConfig, LoadProfile, OutageModel, StochasticOutageParams,
        OutageCostModel, BESSBackupModel,
    )
    return ProjectConfig(
        name="TOU Test",
        project_lifetime_years=5,
        discount_rate_pct=10.0,
        load=LoadProfile(mode="simple", avg_kw=100.0, critical_fraction=0.3),
        outage=OutageModel(
            mode="stochastic",
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=600.0,
                saifi_events_per_year=10.0,
            ),
        ),
        grid=GridTariffModel(
            energy_rate_rs_kwh=7.5,
            fixed_charge_rs_month=1000.0,
            escalation_pct_yr=5.0,
            tou_blocks=[
                TOUBlock(name="peak", rate_rs_kwh=10.0, start_hour=8, end_hour=22,
                         day_type=TOUDayType.WEEKDAY),
                TOUBlock(name="off_peak", rate_rs_kwh=5.0, start_hour=22, end_hour=7,
                         day_type=TOUDayType.ALL),
            ],
            demand_charge=DemandChargeModel(
                charge_rs_kw_month=300.0,
                ratchet_pct=80.0,
                contracted_demand_kw=50.0,
            ),
        ),
        outage_cost=OutageCostModel(mode="voll", voll_rs_kwh=100.0),
        bess=BESSBackupModel(
            capacity_kwh=100.0,
            power_kw=50.0,
            capex_rs_kwh=35000.0,
        ),
    )


class TestTOUFinanceIntegration:
    def test_tou_run_succeeds(self):
        """Full simulation with TOU should complete without error."""
        from energex.engine.runner import run_simulation
        cfg = _make_project_config_with_tou()
        result = run_simulation(cfg, seed=42)
        assert result is not None
        assert result.finance is not None
        assert result.finance.total_capex > 0

    def test_tou_grid_cost_computed(self):
        """TOU energy cost should be present and positive."""
        from energex.engine.runner import run_simulation
        cfg = _make_project_config_with_tou()
        result = run_simulation(cfg, seed=42)
        yr1_cf = result.finance.cashflows[1]
        # Grid energy cost should be nonzero
        assert yr1_cf.grid_energy_cost > 0

    def test_demand_charge_cost_computed(self):
        """Demand charge cost should be positive when demand_charge is configured."""
        from energex.engine.runner import run_simulation
        cfg = _make_project_config_with_tou()
        result = run_simulation(cfg, seed=42)
        yr1_cf = result.finance.cashflows[1]
        assert yr1_cf.demand_charge_cost > 0

    def test_no_demand_charge_when_not_configured(self):
        """Demand charge should be 0 for configs without demand_charge."""
        from energex.engine.runner import run_simulation
        from energex.domain.schemas import (
            ProjectConfig, LoadProfile, OutageModel, StochasticOutageParams,
            OutageCostModel, BESSBackupModel,
        )
        cfg = ProjectConfig(
            name="No DC",
            project_lifetime_years=3,
            discount_rate_pct=10.0,
            load=LoadProfile(mode="simple", avg_kw=100.0, critical_fraction=0.3),
            outage=OutageModel(
                mode="stochastic",
                stochastic=StochasticOutageParams(
                    saidi_minutes_per_year=600.0,
                    saifi_events_per_year=10.0,
                ),
            ),
            grid=GridTariffModel(energy_rate_rs_kwh=7.5, escalation_pct_yr=5.0),
            outage_cost=OutageCostModel(mode="voll", voll_rs_kwh=100.0),
            bess=BESSBackupModel(
                capacity_kwh=100.0, power_kw=50.0, capex_rs_kwh=35000.0,
            ),
        )
        result = run_simulation(cfg, seed=42)
        for cf in result.finance.cashflows[1:]:
            assert cf.demand_charge_cost == pytest.approx(0.0)

    def test_demand_charge_total_in_finance_result(self):
        """total_demand_charge_cost field should aggregate correctly."""
        from energex.engine.runner import run_simulation
        cfg = _make_project_config_with_tou()
        result = run_simulation(cfg, seed=42)
        fin = result.finance
        # Sum of per-year demand charges should equal the aggregate
        computed = sum(cf.demand_charge_cost for cf in fin.cashflows)
        assert computed == pytest.approx(fin.total_demand_charge_cost, rel=1e-6)
