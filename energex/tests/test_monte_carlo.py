"""
EnergeX — Phase C: Monte Carlo Engine Tests
"""

from __future__ import annotations

import pytest
import numpy as np

from energex.domain.schemas import (
    ProjectConfig,
    LoadProfile,
    OutageModel,
    StochasticOutageParams,
    GridTariffModel,
    BESSBackupModel,
    OutageCostModel,
    MonteCarloConfig,
)
from energex.engine.monte_carlo import MonteCarloEngine, MonteCarloStats, MonteCarloResult


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_config(with_bess: bool = True, n_lifetime: int = 5) -> ProjectConfig:
    bess = BESSBackupModel(
        capacity_kwh=100.0,
        power_kw=50.0,
        capex_rs_kwh=20000.0,
    ) if with_bess else None

    return ProjectConfig(
        name="MC Test",
        project_lifetime_years=n_lifetime,
        load=LoadProfile(avg_kw=50.0, critical_fraction=0.4),
        outage=OutageModel(
            mode="stochastic",
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=600.0,
                saifi_events_per_year=8.0,
            ),
        ),
        grid=GridTariffModel(energy_rate_rs_kwh=8.0),
        outage_cost=OutageCostModel(mode="voll", voll_rs_kwh=100.0),
        bess=bess,
    )


# ---------------------------------------------------------------------------
# MonteCarloStats tests
# ---------------------------------------------------------------------------

def test_mc_stats_percentiles():
    stats = MonteCarloStats(
        metric="ens_kwh",
        unit="kWh/yr",
        mean=100.0,
        std=20.0,
        min=50.0,
        max=200.0,
        percentiles={10.0: 60.0, 50.0: 100.0, 90.0: 150.0, 99.0: 190.0},
    )
    assert stats.p10 == 60.0
    assert stats.p50 == 100.0
    assert stats.p90 == 150.0
    assert stats.p99 == 190.0


def test_mc_stats_fallback():
    """When percentile not in dict, should fallback to mean/min/max."""
    stats = MonteCarloStats(
        metric="test", unit="u", mean=50.0, std=5.0, min=10.0, max=90.0,
        percentiles={},
    )
    assert stats.p50 == 50.0   # fallback to mean
    assert stats.p90 == 90.0   # fallback to max
    assert stats.p10 == 10.0   # fallback to min


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

def test_mc_engine_returns_n_runs():
    cfg = _make_config()
    mc_cfg = MonteCarloConfig(n_runs=20, base_seed=42)
    engine = MonteCarloEngine(cfg, mc_config=mc_cfg)
    result = engine.run()

    assert result.n_runs == 20
    assert len(result.ens_kwh_runs) == 20
    assert len(result.downtime_hours_runs) == 20
    assert len(result.continuity_pct_runs) == 20
    assert len(result.sla_pass_runs) == 20


def test_mc_results_positive():
    """All ENS, downtime values should be non-negative."""
    cfg = _make_config()
    mc_cfg = MonteCarloConfig(n_runs=30, base_seed=42)
    engine = MonteCarloEngine(cfg, mc_config=mc_cfg)
    result = engine.run()

    assert all(v >= 0 for v in result.ens_kwh_runs)
    assert all(v >= 0 for v in result.downtime_hours_runs)
    assert all(0 <= v <= 100 for v in result.continuity_pct_runs)


def test_mc_percentile_ordering():
    """P10 ≤ P50 ≤ P90 for all metrics."""
    cfg = _make_config()
    mc_cfg = MonteCarloConfig(n_runs=50, base_seed=42, percentiles=[10, 50, 90])
    engine = MonteCarloEngine(cfg, mc_config=mc_cfg)
    result = engine.run()

    for stats in [result.ens_stats, result.downtime_stats]:
        if stats and stats.percentiles:
            p10 = stats.percentiles.get(10.0, 0)
            p50 = stats.percentiles.get(50.0, 0)
            p90 = stats.percentiles.get(90.0, 0)
            assert p10 <= p50 <= p90, f"Percentile ordering violated for {stats.metric}"


def test_mc_sla_pass_rate_in_range():
    """SLA pass rate should be in [0, 100]."""
    cfg = _make_config()
    mc_cfg = MonteCarloConfig(n_runs=20, base_seed=42)
    engine = MonteCarloEngine(cfg, mc_config=mc_cfg)
    result = engine.run()

    assert 0.0 <= result.sla_pass_rate_pct <= 100.0


def test_mc_reproducible_with_same_seed():
    """Same base seed should produce identical results."""
    cfg = _make_config()
    mc_cfg = MonteCarloConfig(n_runs=15, base_seed=99)

    e1 = MonteCarloEngine(cfg, mc_config=mc_cfg)
    r1 = e1.run()
    e2 = MonteCarloEngine(cfg, mc_config=mc_cfg)
    r2 = e2.run()

    assert r1.ens_kwh_runs == r2.ens_kwh_runs
    assert r1.downtime_hours_runs == r2.downtime_hours_runs


def test_mc_different_seeds_give_different_results():
    """Different base seeds should give different run lists."""
    cfg = _make_config()
    mc_cfg_a = MonteCarloConfig(n_runs=20, base_seed=42)
    mc_cfg_b = MonteCarloConfig(n_runs=20, base_seed=999)

    e1 = MonteCarloEngine(cfg, mc_config=mc_cfg_a)
    r1 = e1.run()
    e2 = MonteCarloEngine(cfg, mc_config=mc_cfg_b)
    r2 = e2.run()

    # At least some runs should differ
    assert r1.ens_kwh_runs != r2.ens_kwh_runs


def test_mc_bess_reduces_p90_ens():
    """BESS should reduce P90 ENS vs no backup (statistical expectation)."""
    cfg_bess = _make_config(with_bess=True)
    cfg_no = _make_config(with_bess=False)

    mc_cfg = MonteCarloConfig(n_runs=50, base_seed=42)

    r_bess = MonteCarloEngine(cfg_bess, mc_config=mc_cfg).run()
    r_no = MonteCarloEngine(cfg_no, mc_config=mc_cfg).run()

    p90_bess = r_bess.ens_stats.p90 if r_bess.ens_stats else 0
    p90_no = r_no.ens_stats.p90 if r_no.ens_stats else 0

    assert p90_bess <= p90_no, "BESS should reduce P90 ENS vs no backup"


def test_mc_to_dict_serializable():
    """to_dict should return a JSON-serializable structure."""
    import json
    cfg = _make_config()
    mc_cfg = MonteCarloConfig(n_runs=10, base_seed=42)
    result = MonteCarloEngine(cfg, mc_config=mc_cfg).run()

    d = result.to_dict()
    json_str = json.dumps(d)  # should not raise
    assert "n_runs" in d
    assert "sla_pass_rate_pct" in d


def test_mc_stats_computation():
    """_compute_stats produces correct mean/std."""
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = MonteCarloEngine._compute_stats(vals, "test", "u", [50.0])
    assert abs(stats.mean - 30.0) < 0.01
    assert abs(stats.percentiles[50.0] - 30.0) < 0.01
    assert stats.min == 10.0
    assert stats.max == 50.0
