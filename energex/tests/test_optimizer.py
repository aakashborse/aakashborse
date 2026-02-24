"""
EnergeX — Phase C: Sizing Optimizer Tests
"""

from __future__ import annotations

import pytest

from energex.domain.schemas import (
    ProjectConfig,
    LoadProfile,
    OutageModel,
    StochasticOutageParams,
    GridTariffModel,
    BESSBackupModel,
    OutageCostModel,
    SizingBounds,
)
from energex.engine.optimizer import SizingOptimizer, SizingPoint, OptimizationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config() -> ProjectConfig:
    return ProjectConfig(
        name="Optimizer Test",
        project_lifetime_years=5,
        load=LoadProfile(avg_kw=50.0, critical_fraction=0.4),
        outage=OutageModel(
            mode="stochastic",
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=600.0,
                saifi_events_per_year=8.0,
            ),
        ),
        grid=GridTariffModel(energy_rate_rs_kwh=8.0),
        outage_cost=OutageCostModel(
            mode="voll",
            voll_rs_kwh=100.0,
            sla_max_ens_kwh_yr=500.0,
        ),
        bess=BESSBackupModel(
            capacity_kwh=50.0,
            power_kw=25.0,
            capex_rs_kwh=20000.0,
        ),
    )


def _make_small_bounds() -> SizingBounds:
    """Tiny bounds for fast tests."""
    return SizingBounds(
        bess_capacity_min_kwh=0.0,
        bess_capacity_max_kwh=100.0,
        bess_capacity_step_kwh=50.0,
        include_dg=False,
        include_solar=False,
        n_years_per_eval=1,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_sizing_point_to_dict():
    pt = SizingPoint(bess_kwh=100.0, bess_kw=50.0, dg_kw=0.0, solar_kwp=0.0)
    d = pt.to_dict()
    assert d["bess_kwh"] == 100.0
    assert "capex_rs" in d
    assert "is_pareto" in d


def test_arange_inclusive():
    vals = SizingOptimizer._arange(0, 100, 50)
    assert 0.0 in vals
    assert 50.0 in vals
    assert 100.0 in vals


def test_arange_single():
    vals = SizingOptimizer._arange(50, 50, 10)
    assert vals == [50.0]


def test_candidate_seed_deterministic():
    s1 = SizingOptimizer._candidate_seed(100.0, 0.0, 0.0)
    s2 = SizingOptimizer._candidate_seed(100.0, 0.0, 0.0)
    assert s1 == s2


def test_candidate_seed_different_for_different_sizes():
    s1 = SizingOptimizer._candidate_seed(100.0, 0.0, 0.0)
    s2 = SizingOptimizer._candidate_seed(150.0, 0.0, 0.0)
    assert s1 != s2


def test_pareto_extraction_basic():
    """Point A dominates B if A has lower cost AND lower ENS."""
    pts = [
        SizingPoint(bess_kwh=0, bess_kw=0, dg_kw=0, solar_kwp=0, capex_rs=0, avg_ens_kwh=100),   # cheap, bad ENS
        SizingPoint(bess_kwh=50, bess_kw=25, dg_kw=0, solar_kwp=0, capex_rs=50000, avg_ens_kwh=50),  # better ENS
        SizingPoint(bess_kwh=100, bess_kw=50, dg_kw=0, solar_kwp=0, capex_rs=200000, avg_ens_kwh=10), # best ENS
        SizingPoint(bess_kwh=75, bess_kw=37, dg_kw=0, solar_kwp=0, capex_rs=300000, avg_ens_kwh=50),  # dominated (more expensive, same ENS as 50kWh)
    ]
    pareto = SizingOptimizer._extract_pareto(pts)
    pareto_costs = {p.capex_rs for p in pareto}
    # The dominated point (capex=300000, ens=50) should not be in Pareto
    assert 300000 not in pareto_costs


def test_pareto_all_on_frontier_when_tradeoff():
    """When each point is better on one axis, all should be on frontier."""
    pts = [
        SizingPoint(bess_kwh=0, bess_kw=0, dg_kw=0, solar_kwp=0, capex_rs=0, avg_ens_kwh=100),
        SizingPoint(bess_kwh=50, bess_kw=25, dg_kw=0, solar_kwp=0, capex_rs=100000, avg_ens_kwh=50),
        SizingPoint(bess_kwh=100, bess_kw=50, dg_kw=0, solar_kwp=0, capex_rs=200000, avg_ens_kwh=10),
    ]
    pareto = SizingOptimizer._extract_pareto(pts)
    assert len(pareto) == 3  # all are non-dominated


# ---------------------------------------------------------------------------
# Integration tests (small grid)
# ---------------------------------------------------------------------------

def test_optimizer_runs_and_returns_result():
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()

    assert isinstance(result, OptimizationResult)
    assert result.n_candidates_evaluated > 0
    assert len(result.all_points) == result.n_candidates_evaluated


def test_optimizer_evaluates_correct_number_of_candidates():
    """0kWh, 50kWh, 100kWh = 3 candidates."""
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()
    # 0, 50, 100 kWh (step=50 from 0 to 100)
    assert result.n_candidates_evaluated == 3


def test_optimizer_pareto_points_are_non_dominated():
    """Every pareto point must flag is_pareto=True."""
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()

    for pt in result.pareto_points:
        assert pt.is_pareto is True


def test_optimizer_all_points_have_valid_metrics():
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()

    for pt in result.all_points:
        assert pt.avg_ens_kwh >= 0
        assert pt.avg_downtime_hours >= 0
        assert 0 <= pt.avg_continuity_pct <= 100
        assert pt.capex_rs >= 0


def test_optimizer_zero_bess_has_zero_capex():
    """Zero-BESS candidate should have zero capex."""
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()

    zero_bess = [p for p in result.all_points if p.bess_kwh == 0]
    assert len(zero_bess) > 0
    assert zero_bess[0].capex_rs == 0.0


def test_optimizer_larger_bess_higher_capex():
    """Increasing BESS size should increase capex."""
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()

    pts_sorted = sorted(result.all_points, key=lambda p: p.bess_kwh)
    for i in range(len(pts_sorted) - 1):
        if pts_sorted[i + 1].bess_kwh > pts_sorted[i].bess_kwh:
            assert pts_sorted[i + 1].capex_rs >= pts_sorted[i].capex_rs


def test_optimizer_min_ens_not_none():
    """min_ens_point should always be set (= point with lowest ENS)."""
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()

    assert result.min_ens_point is not None
    min_ens = min(p.avg_ens_kwh for p in result.all_points)
    assert abs(result.min_ens_point.avg_ens_kwh - min_ens) < 1e-6


def test_optimizer_to_dict_serializable():
    """to_dict should produce JSON-serializable output."""
    import json
    cfg = _make_config()
    bounds = _make_small_bounds()
    opt = SizingOptimizer(cfg, bounds=bounds, base_seed=42)
    result = opt.run()

    d = result.to_dict()
    json_str = json.dumps(d)  # should not raise
    assert "n_candidates_evaluated" in d
    assert "pareto_frontier" in d
