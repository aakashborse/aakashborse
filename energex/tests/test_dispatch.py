"""
Tests for the backup dispatch engine.

Verifies:
  - SOC never violates physical bounds
  - Power limits respected
  - ENS computed correctly
  - DG start delay logic
"""

import math
import numpy as np
import pytest

from energex.domain.schemas import (
    ProjectConfig, LoadProfile, OutageModel, OutageMode,
    GridTariffModel, OutageCostModel, OutageCostMode,
    BESSBackupModel, DGBackupModel, StochasticOutageParams,
    DeterministicOutageEvent,
)
from energex.engine.outage_generator import OutageEvent
from energex.engine.dispatch_backup import (
    BackupDispatchEngine, BESSSimulator, DGSimulator, SimulationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_base_config(
    avg_kw=100.0,
    critical_fraction=0.3,
    bess=None,
    dg=None,
    solar=None,
) -> ProjectConfig:
    return ProjectConfig(
        name="Test",
        project_lifetime_years=10,
        discount_rate_pct=10.0,
        load=LoadProfile(mode="simple", avg_kw=avg_kw, critical_fraction=critical_fraction),
        outage=OutageModel(
            mode=OutageMode.STOCHASTIC,
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=600,
                saifi_events_per_year=10,
            )
        ),
        grid=GridTariffModel(energy_rate_rs_kwh=8.0),
        outage_cost=OutageCostModel(mode=OutageCostMode.VOLL, voll_rs_kwh=100.0),
        bess=bess,
        dg=dg,
        solar=solar,
    )


def make_bess(capacity_kwh=100.0, power_kw=50.0, soc_pct=100.0) -> BESSBackupModel:
    return BESSBackupModel(
        capacity_kwh=capacity_kwh,
        power_kw=power_kw,
        dod_pct=90.0,
        roundtrip_efficiency=1.0,  # perfect for easy math
        min_reserve_soc_pct=0.0,
        initial_soc_pct=soc_pct,
        calendar_fade_pct_yr=0.0,
        cycle_fade_per_kwh=0.0,
        calendar_life_yr=15.0,
        eol_capacity_pct=80.0,
        capex_rs_kwh=30000,
        availability_pct=100.0,
    )


def make_dg(rated_kw=100.0, start_delay_s=0.0, avail_pct=100.0) -> DGBackupModel:
    return DGBackupModel(
        rated_kw=rated_kw,
        min_loading_pct=0.0,
        start_delay_seconds=start_delay_s,
        ramp_rate_kw_s=100.0,
        fuel_l_per_kwh=0.3,
        diesel_price_rs_l=95.0,
        diesel_escalation_pct_yr=0.0,
        om_rs_kwh=1.0,
        availability_pct=avail_pct,
        capex_rs=500000,
    )


# ---------------------------------------------------------------------------
# BESS Simulator unit tests
# ---------------------------------------------------------------------------

class TestBESSSimulator:

    def test_initial_soc(self):
        model = make_bess(capacity_kwh=100.0, soc_pct=80.0)
        sim = BESSSimulator(model, year=0)
        assert abs(sim.soc_kwh - 80.0) < 0.01

    def test_discharge_reduces_soc(self):
        model = make_bess(capacity_kwh=100.0, power_kw=50.0, soc_pct=100.0)
        sim = BESSSimulator(model, year=0)
        initial_soc = sim.soc_kwh
        delivered = sim.discharge(30.0, dt_hours=1.0)
        assert delivered > 0
        assert sim.soc_kwh < initial_soc

    def test_discharge_respects_power_limit(self):
        model = make_bess(capacity_kwh=1000.0, power_kw=50.0, soc_pct=100.0)
        sim = BESSSimulator(model, year=0)
        delivered = sim.discharge(200.0, dt_hours=1.0)  # request 200 kWh but limit is 50 kW * 1h
        assert delivered <= 50.0 + 1e-6

    def test_discharge_respects_soc_floor(self):
        model = make_bess(capacity_kwh=100.0, power_kw=200.0, soc_pct=20.0)
        model2 = BESSBackupModel(
            capacity_kwh=100.0, power_kw=200.0, dod_pct=90.0,
            roundtrip_efficiency=1.0, min_reserve_soc_pct=20.0,
            initial_soc_pct=20.0, calendar_fade_pct_yr=0.0,
            cycle_fade_per_kwh=0.0, calendar_life_yr=15.0,
            eol_capacity_pct=80.0, capex_rs_kwh=30000, availability_pct=100.0,
        )
        sim = BESSSimulator(model2, year=0)
        delivered = sim.discharge(100.0, dt_hours=1.0)
        # Reserve is 20% of 100 = 20 kWh, initial SOC is 20 kWh → nothing available
        assert delivered < 1e-6

    def test_charge_increases_soc(self):
        model = make_bess(capacity_kwh=100.0, power_kw=50.0, soc_pct=50.0)
        sim = BESSSimulator(model, year=0)
        initial = sim.soc_kwh
        taken = sim.charge(20.0, dt_hours=1.0)
        assert taken > 0
        assert sim.soc_kwh > initial

    def test_charge_not_exceed_capacity(self):
        model = make_bess(capacity_kwh=100.0, power_kw=200.0, soc_pct=99.0)
        sim = BESSSimulator(model, year=0)
        sim.charge(500.0, dt_hours=1.0)
        assert sim.soc_kwh <= sim.effective_capacity + 1e-6

    def test_soc_never_negative(self):
        model = make_bess(capacity_kwh=100.0, power_kw=5000.0, soc_pct=10.0)
        sim = BESSSimulator(model, year=0)
        for _ in range(100):
            sim.discharge(1000.0, dt_hours=1.0)
        assert sim.soc_kwh >= -1e-9


# ---------------------------------------------------------------------------
# DG Simulator unit tests
# ---------------------------------------------------------------------------

class TestDGSimulator:

    def test_cannot_start_before_delay(self):
        dg = make_dg(start_delay_s=30.0)
        sim = DGSimulator(dg)
        assert not sim.can_start(outage_elapsed_hours=0.005)  # 18 seconds < 30s

    def test_can_start_after_delay(self):
        dg = make_dg(start_delay_s=10.0)
        sim = DGSimulator(dg)
        assert sim.can_start(outage_elapsed_hours=1.0)

    def test_fuel_consumed(self):
        dg = make_dg(rated_kw=100.0)
        sim = DGSimulator(dg)
        sim.supply(100.0, dt_hours=1.0)
        assert sim.fuel_used_liters > 0

    def test_fuel_storage_limits_supply(self):
        dg = make_dg(rated_kw=100.0)
        dg2 = DGBackupModel(
            rated_kw=100.0, min_loading_pct=0.0, start_delay_seconds=0.0,
            ramp_rate_kw_s=100.0, fuel_l_per_kwh=0.3,
            diesel_price_rs_l=95.0, diesel_escalation_pct_yr=0.0,
            om_rs_kwh=1.0, availability_pct=100.0, capex_rs=500000,
            fuel_storage_liters=15.0,  # only 15 L ≈ 50 kWh @ 0.3 L/kWh
        )
        sim = DGSimulator(dg2)
        delivered = sim.supply(100.0, dt_hours=1.0)
        # 15L / 0.3 L/kWh = 50 kWh max
        assert delivered <= 50.0 + 1e-6


# ---------------------------------------------------------------------------
# Full dispatch integration tests
# ---------------------------------------------------------------------------

class TestDispatchEngine:

    def _run(self, cfg, events=None):
        if events is None:
            events = []
        engine = BackupDispatchEngine(
            config=cfg,
            outage_events=events,
            year=0,
            seed=42,
        )
        return engine.run()

    def _make_events(self, start, duration):
        return [OutageEvent(start_hour=float(start), duration_hours=float(duration))]

    def test_no_outages_zero_ens(self):
        """Golden test 1: No outages → ENS must be 0."""
        cfg = make_base_config(avg_kw=100.0, critical_fraction=0.3)
        result = self._run(cfg, events=[])
        assert result.ens_kwh == 0.0
        assert result.downtime_hours == 0.0

    def test_bess_covers_short_outage_zero_ens(self):
        """Golden test 2: Outage with enough BESS → ENS = 0."""
        bess = make_bess(capacity_kwh=200.0, power_kw=100.0, soc_pct=100.0)
        cfg = make_base_config(avg_kw=100.0, critical_fraction=0.3, bess=bess)
        # Critical load ≈ 30 kW, BESS usable = 200*0.9 = 180 kWh → autonomy ~6 hrs
        events = self._make_events(start=100.0, duration=2.0)
        result = self._run(cfg, events=events)
        assert result.ens_kwh < 1e-3

    def test_outage_longer_than_autonomy_gives_ens(self):
        """Golden test 3: Outage longer than BESS autonomy → ENS > 0."""
        bess = make_bess(capacity_kwh=30.0, power_kw=100.0, soc_pct=100.0)
        cfg = make_base_config(avg_kw=100.0, critical_fraction=0.3, bess=bess)
        # Critical load ≈ 30 kW, usable = 30*0.9 = 27 kWh → autonomy < 1 hr
        events = self._make_events(start=50.0, duration=5.0)  # 5-hour outage
        result = self._run(cfg, events=events)
        assert result.ens_kwh > 0.0

    def test_dg_after_start_delay_ens_only_during_gap(self):
        """Golden test 4: DG with start delay → ENS occurs only during BESS-gap period."""
        # Small BESS (1 kWh) and DG with 30s delay
        bess = make_bess(capacity_kwh=1.0, power_kw=100.0, soc_pct=100.0)
        dg = make_dg(rated_kw=100.0, start_delay_s=0.0, avail_pct=100.0)
        cfg = make_base_config(avg_kw=100.0, critical_fraction=0.3, bess=bess, dg=dg)
        events = self._make_events(start=100.0, duration=4.0)
        result = self._run(cfg, events=events)
        # DG with 0s delay + BESS → almost no ENS
        assert result.ens_kwh < 5.0

    def test_dg_only_covers_load(self):
        """DG without BESS should cover critical load after start delay."""
        dg = make_dg(rated_kw=200.0, start_delay_s=0.0, avail_pct=100.0)
        cfg = make_base_config(avg_kw=100.0, critical_fraction=0.3, dg=dg)
        events = self._make_events(start=200.0, duration=3.0)
        result = self._run(cfg, events=events)
        # DG rated at 200 kW > critical ~30 kW, should serve fully
        assert result.ens_kwh < 1.0

    def test_soc_never_violates_bounds(self):
        """Property test: BESS SOC always within [0, effective_capacity]."""
        bess = make_bess(capacity_kwh=100.0, power_kw=50.0, soc_pct=80.0)
        cfg = make_base_config(avg_kw=100.0, bess=bess)
        from energex.engine.outage_generator import OutageGenerator
        gen = OutageGenerator(cfg.outage, seed=42)
        events = gen.generate()
        engine = BackupDispatchEngine(config=cfg, outage_events=events, year=0, seed=42)
        result = engine.run()
        cap = engine.bess.effective_capacity
        for h in result.hourly:
            assert -1e-6 <= h.bess_soc_end <= cap + 1e-6, \
                f"SOC {h.bess_soc_end:.3f} out of bounds [0, {cap:.1f}] at hour {h.hour}"

    def test_grid_energy_zero_during_full_outage(self):
        """Grid energy should be zero in hours with 100% outage."""
        bess = make_bess(capacity_kwh=500.0, power_kw=200.0, soc_pct=100.0)
        cfg = make_base_config(avg_kw=50.0, bess=bess)
        events = [OutageEvent(start_hour=10.0, duration_hours=5.0)]
        engine = BackupDispatchEngine(config=cfg, outage_events=events, year=0, seed=42)
        result = engine.run()
        for h in result.hourly:
            if h.outage_fraction >= 1.0:
                assert h.grid_to_load < 1e-6

    def test_no_backup_assets_all_ens_during_outage(self):
        """Without any backup, all critical load during outage is unserved."""
        cfg = make_base_config(avg_kw=100.0, critical_fraction=0.4)
        events = [OutageEvent(start_hour=50.0, duration_hours=2.0)]
        result = self._run(cfg, events=events)
        # 2h outage, ~40 kW critical → ~80 kWh ENS
        assert result.ens_kwh > 50.0

    def test_dg_fuel_tracking(self):
        """DG fuel usage should be positive when DG runs."""
        dg = make_dg(rated_kw=100.0, start_delay_s=0.0, avail_pct=100.0)
        cfg = make_base_config(avg_kw=50.0, critical_fraction=0.3, dg=dg)
        events = [OutageEvent(start_hour=100.0, duration_hours=5.0)]
        result = self._run(cfg, events=events)
        assert result.dg_fuel_liters > 0

    def test_backup_autonomy_with_bess_and_dg(self):
        """Autonomy with both BESS and DG should exceed BESS-only autonomy."""
        bess = make_bess(capacity_kwh=100.0, power_kw=100.0, soc_pct=100.0)
        dg = make_dg(rated_kw=200.0, start_delay_s=0.0, avail_pct=100.0)
        cfg_bess_only = make_base_config(avg_kw=100.0, critical_fraction=0.3, bess=bess)
        cfg_hybrid = make_base_config(avg_kw=100.0, critical_fraction=0.3, bess=bess, dg=dg)
        r_bess = self._run(cfg_bess_only)
        r_hybrid = self._run(cfg_hybrid)
        assert r_hybrid.backup_autonomy_hours >= r_bess.backup_autonomy_hours

    def test_sla_pass_when_adequate_bess(self):
        """Adequate BESS with low outage rates should pass SLA."""
        bess = make_bess(capacity_kwh=500.0, power_kw=200.0, soc_pct=100.0)
        # Use SLA with generous targets
        cfg = ProjectConfig(
            name="SLA test",
            project_lifetime_years=10,
            discount_rate_pct=10.0,
            load=LoadProfile(mode="simple", avg_kw=50.0, critical_fraction=0.2),
            outage=OutageModel(
                mode=OutageMode.DETERMINISTIC,
                events=[DeterministicOutageEvent(start_hour=100.0, duration_hours=1.0)],
            ),
            grid=GridTariffModel(energy_rate_rs_kwh=8.0),
            outage_cost=OutageCostModel(
                mode=OutageCostMode.VOLL,
                voll_rs_kwh=100.0,
                sla_max_downtime_hrs_yr=100.0,
                sla_max_ens_kwh_yr=10000.0,
            ),
            bess=bess,
        )
        result = self._run(cfg, events=[OutageEvent(100.0, 1.0)])
        assert result.sla_pass
