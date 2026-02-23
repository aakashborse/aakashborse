"""
Tests for Monte Carlo engine (P50/P90/P99 confidence bands).
"""
from __future__ import annotations

import pytest
import numpy as np

from energex.engine.monte_carlo import MonteCarloEngine, MonteCarloResult


def _make_minimal_config(n_years: int = 3):
    """Build a minimal stochastic config for MC tests."""
    from energex.domain.schemas import (
        ProjectConfig, LoadProfile, OutageModel, StochasticOutageParams,
        OutageCostModel, GridTariffModel, BESSBackupModel,
    )
    return ProjectConfig(
        name="MC Test",
        project_lifetime_years=n_years,
        discount_rate_pct=10.0,
        load=LoadProfile(mode="simple", avg_kw=50.0, critical_fraction=0.3),
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


class TestMonteCarloResult:
    def test_percentile_empty(self):
        mc = MonteCarloResult(n_trials=0, base_seed=42,
                              confidence_levels=[0.50, 0.90])
        assert np.isnan(mc.percentile(np.array([]), 0.5))

    def test_percentile_basic(self):
        mc = MonteCarloResult(n_trials=10, base_seed=42,
                              confidence_levels=[0.50, 0.90])
        mc.ens_trials = np.array([1.0, 2.0, 3.0, 4.0, 5.0,
                                   6.0, 7.0, 8.0, 9.0, 10.0])
        assert mc.ens_p50 == pytest.approx(5.5)
        assert mc.ens_p90 == pytest.approx(9.1)

    def test_get_bands_keys(self):
        mc = MonteCarloResult(n_trials=5, base_seed=42,
                              confidence_levels=[0.10, 0.50, 0.90])
        mc.ens_trials = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bands = mc.get_bands("ens")
        assert "P10" in bands
        assert "P50" in bands
        assert "P90" in bands

    def test_summary_dict_structure(self):
        mc = MonteCarloResult(n_trials=5, base_seed=42,
                              confidence_levels=[0.10, 0.50, 0.90, 0.99])
        mc.ens_trials = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        mc.downtime_trials = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mc.continuity_trials = np.array([99.0, 99.5, 99.8, 99.9, 100.0])
        mc.npv_trials = np.array([-1e6, 0.0, 1e6, 2e6, 3e6])
        mc.outage_cost_trials = np.array([1e5, 2e5, 3e5, 4e5, 5e5])
        mc.lcoe_trials = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
        summary = mc.summary_dict()
        assert "n_trials" in summary
        assert "ens_kWh_yr" in summary
        assert "npv_rs" in summary
        assert summary["ens_kWh_yr"]["p50"] > 0


class TestMonteCarloEngine:
    def test_mc_runs_n_trials(self):
        cfg = _make_minimal_config(n_years=2)
        engine = MonteCarloEngine(config=cfg, n_trials=10, base_seed=42)
        mc = engine.run()
        assert len(mc.trials) == 10

    def test_mc_returns_arrays(self):
        cfg = _make_minimal_config(n_years=2)
        engine = MonteCarloEngine(config=cfg, n_trials=10, base_seed=42)
        mc = engine.run()
        assert len(mc.ens_trials) == 10
        assert len(mc.npv_trials) == 10
        assert len(mc.downtime_trials) == 10

    def test_mc_different_seeds_give_variation(self):
        """With stochastic outages, different seeds should yield different ENS values."""
        cfg = _make_minimal_config(n_years=3)
        engine = MonteCarloEngine(config=cfg, n_trials=20, base_seed=42)
        mc = engine.run()
        # ENS values should not all be identical
        assert len(set(round(v, 6) for v in mc.ens_trials)) > 1

    def test_mc_p50_less_than_p90(self):
        cfg = _make_minimal_config(n_years=3)
        engine = MonteCarloEngine(config=cfg, n_trials=30, base_seed=42)
        mc = engine.run()
        # P50 ≤ P90 ≤ P99 for ENS (higher = worse)
        assert mc.ens_p50 <= mc.ens_p90
        assert mc.ens_p90 <= mc.ens_p99

    def test_mc_p10_less_than_p50_for_npv(self):
        """NPV distribution: P10 ≤ P50 (P10 is the downside)."""
        cfg = _make_minimal_config(n_years=3)
        engine = MonteCarloEngine(config=cfg, n_trials=30, base_seed=42)
        mc = engine.run()
        assert mc.npv_p10 <= mc.npv_p50

    def test_mc_progress_callback(self):
        cfg = _make_minimal_config(n_years=2)
        engine = MonteCarloEngine(config=cfg, n_trials=5, base_seed=42)
        progress_values = []
        mc = engine.run(progress_callback=lambda p: progress_values.append(p))
        assert len(progress_values) == 5
        assert progress_values[-1] == pytest.approx(1.0)

    def test_mc_capex_consistent(self):
        """All trials should have the same CAPEX (deterministic)."""
        cfg = _make_minimal_config(n_years=2)
        engine = MonteCarloEngine(config=cfg, n_trials=10, base_seed=42)
        mc = engine.run()
        capex_values = [t.total_capex for t in mc.trials]
        # All capex should be identical
        assert all(c == pytest.approx(capex_values[0]) for c in capex_values)

    def test_mc_summary_dict_has_all_keys(self):
        cfg = _make_minimal_config(n_years=2)
        engine = MonteCarloEngine(config=cfg, n_trials=10, base_seed=42)
        mc = engine.run()
        summary = mc.summary_dict()
        for key in ["n_trials", "ens_kWh_yr", "downtime_hrs_yr", "npv_rs", "outage_cost_rs"]:
            assert key in summary, f"Missing key: {key}"
