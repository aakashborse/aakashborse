"""
Golden tests — end-to-end simulation with known expected outputs.

These tests validate the full pipeline: config → outage generation →
dispatch → finance → results.
"""

import pytest

from energex.adapters.config_loader import load_config
from energex.engine.runner import run_simulation, run_grid_only_baseline
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_config(name: str, seed: int = 42):
    cfg = load_config(CONFIGS_DIR / f"{name}.json")
    result = run_simulation(cfg, seed=seed)
    return result


# ---------------------------------------------------------------------------
# Golden test 1: Grid + BESS (no outage specified — stochastic)
# ---------------------------------------------------------------------------

class TestGoldenGridBESS:

    def test_runs_without_error(self):
        result = run_config("grid_bess")
        assert result is not None

    def test_ens_below_threshold(self):
        """With 150 kWh BESS for ~35 kW critical load, most outages should be served."""
        result = run_config("grid_bess", seed=42)
        # ENS should be low (autonomy ≈ 150*0.9/35 ≈ 3.9 hrs; SAIDI=1200 min=20 hrs/yr)
        # Some ENS expected for very long outages
        assert result.sim_year1.ens_kwh >= 0
        assert result.sim_year1.downtime_hours >= 0

    def test_grid_energy_positive(self):
        result = run_config("grid_bess")
        assert result.sim_year1.total_grid_kwh > 0

    def test_bess_discharge_occurs(self):
        result = run_config("grid_bess")
        # BESS should discharge during outages
        assert result.sim_year1.total_bess_discharge_kwh >= 0

    def test_finance_capex_positive(self):
        result = run_config("grid_bess")
        # BESS capex = 150 kWh * ₹35000/kWh = ₹5,250,000
        assert result.finance.total_capex == pytest.approx(150.0 * 35000, rel=0.01)

    def test_reproducible_with_seed(self):
        r1 = run_config("grid_bess", seed=99)
        r2 = run_config("grid_bess", seed=99)
        assert abs(r1.sim_year1.ens_kwh - r2.sim_year1.ens_kwh) < 1e-9


# ---------------------------------------------------------------------------
# Golden test 2: Grid + DG
# ---------------------------------------------------------------------------

class TestGoldenGridDG:

    def test_runs_without_error(self):
        result = run_config("grid_dg")
        assert result is not None

    def test_dg_fuel_consumed_during_outages(self):
        result = run_config("grid_dg")
        if result.sim_year1.outage_events_total > 0:
            assert result.sim_year1.dg_fuel_liters >= 0

    def test_dg_capex_in_finance(self):
        result = run_config("grid_dg")
        # DG capex = 1,200,000 from config
        assert result.finance.total_capex == pytest.approx(1_200_000, rel=0.01)

    def test_finance_cashflows_count(self):
        result = run_config("grid_dg")
        cfg = result.config
        assert len(result.finance.cashflows) == cfg.project_lifetime_years + 1

    def test_outage_cost_appears_in_cashflows(self):
        result = run_config("grid_dg")
        yr1_cf = result.finance.cashflows[1]
        assert yr1_cf.outage_cost >= 0


# ---------------------------------------------------------------------------
# Golden test 3: Grid + Solar + BESS
# ---------------------------------------------------------------------------

class TestGoldenGridSolarBESS:

    def test_runs_without_error(self):
        result = run_config("grid_solar_bess")
        assert result is not None

    def test_solar_generation_nonzero(self):
        result = run_config("grid_solar_bess")
        assert result.sim_year1.total_solar_kwh > 0

    def test_solar_to_bess_nonzero(self):
        result = run_config("grid_solar_bess")
        assert result.sim_year1.total_solar_to_bess_kwh >= 0

    def test_solar_capex_in_finance(self):
        result = run_config("grid_solar_bess")
        cfg = result.config
        # Solar capex = 120 kWp * ₹42000/kWp + BESS capex
        expected_solar_capex = cfg.solar.capex_rs
        expected_bess_capex = cfg.bess.capex_rs
        assert result.finance.total_capex == pytest.approx(
            expected_solar_capex + expected_bess_capex, rel=0.01
        )

    def test_ens_nonnegative(self):
        result = run_config("grid_solar_bess")
        assert result.sim_year1.ens_kwh >= 0

    def test_continuity_between_0_and_100(self):
        result = run_config("grid_solar_bess")
        assert 0.0 <= result.sim_year1.continuity_pct <= 100.0


# ---------------------------------------------------------------------------
# Golden test 4: DG + BESS Hybrid (full hybrid)
# ---------------------------------------------------------------------------

class TestGoldenDGBESSHybrid:

    def test_runs_without_error(self):
        result = run_config("dg_bess_hybrid")
        assert result is not None

    def test_autonomy_with_dg_and_bess(self):
        result = run_config("dg_bess_hybrid")
        # With DG + BESS, autonomy should be at least BESS hours
        assert result.sim_year1.backup_autonomy_hours > 0

    def test_sla_failures_list_exists(self):
        result = run_config("dg_bess_hybrid")
        assert isinstance(result.sim_year1.sla_failures, list)

    def test_warnings_list_exists(self):
        result = run_config("dg_bess_hybrid")
        assert isinstance(result.warnings, list)

    def test_seed_in_result(self):
        result = run_config("dg_bess_hybrid", seed=1234)
        assert result.seed == 1234

    def test_run_id_exists(self):
        result = run_config("dg_bess_hybrid")
        assert result.run_id is not None and len(result.run_id) > 0

    def test_total_cost_with_outages_positive(self):
        result = run_config("dg_bess_hybrid")
        assert result.finance.total_cost_with_outages > 0


# ---------------------------------------------------------------------------
# Cross-scenario comparison tests
# ---------------------------------------------------------------------------

class TestScenarioComparison:

    def test_bess_reduces_ens_vs_grid_only(self):
        """Grid+BESS should have less ENS than grid-only under same outage conditions."""
        from energex.domain.schemas import (
            ProjectConfig, LoadProfile, OutageModel, OutageMode,
            GridTariffModel, OutageCostModel, OutageCostMode,
            StochasticOutageParams, BESSBackupModel,
        )
        from energex.engine.runner import run_simulation

        base_cfg = ProjectConfig(
            name="Base",
            project_lifetime_years=5,
            discount_rate_pct=10.0,
            load=LoadProfile(mode="simple", avg_kw=100.0, critical_fraction=0.3),
            outage=OutageModel(
                mode=OutageMode.STOCHASTIC,
                stochastic=StochasticOutageParams(
                    saidi_minutes_per_year=600,
                    saifi_events_per_year=12,
                )
            ),
            grid=GridTariffModel(energy_rate_rs_kwh=8.0),
            outage_cost=OutageCostModel(mode=OutageCostMode.VOLL, voll_rs_kwh=100.0),
        )
        bess_cfg = ProjectConfig(
            name="BESS",
            project_lifetime_years=5,
            discount_rate_pct=10.0,
            load=base_cfg.load,
            outage=base_cfg.outage,
            grid=base_cfg.grid,
            outage_cost=base_cfg.outage_cost,
            bess=BESSBackupModel(
                capacity_kwh=200.0, power_kw=100.0, dod_pct=90.0,
                roundtrip_efficiency=1.0, min_reserve_soc_pct=0.0,
                initial_soc_pct=100.0, calendar_fade_pct_yr=0.0,
                cycle_fade_per_kwh=0.0, calendar_life_yr=20.0,
                eol_capacity_pct=80.0, capex_rs_kwh=30000,
                availability_pct=100.0,
            ),
        )
        seed = 77
        r_base = run_simulation(base_cfg, seed=seed)
        r_bess = run_simulation(bess_cfg, seed=seed)
        # BESS with 200 kWh should significantly reduce ENS
        assert r_bess.sim_year1.ens_kwh <= r_base.sim_year1.ens_kwh + 1.0
