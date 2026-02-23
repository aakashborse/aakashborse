"""
Tests for the Automated Sizing Optimizer.
"""
from __future__ import annotations

import pytest
from energex.engine.sizing_optimizer import SizingOptimizer, SizingCandidate


def _make_base_config(has_bess: bool = False, has_dg: bool = False):
    """Minimal config for sizing tests."""
    from energex.domain.schemas import (
        ProjectConfig, LoadProfile, OutageModel, StochasticOutageParams,
        OutageCostModel, GridTariffModel, BESSBackupModel, DGBackupModel,
    )
    cfg = ProjectConfig(
        name="Sizing Test",
        project_lifetime_years=5,
        discount_rate_pct=10.0,
        load=LoadProfile(mode="simple", avg_kw=80.0, critical_fraction=0.35),
        outage=OutageModel(
            mode="stochastic",
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=900.0,
                saifi_events_per_year=12.0,
            ),
        ),
        grid=GridTariffModel(energy_rate_rs_kwh=7.5, escalation_pct_yr=5.0),
        outage_cost=OutageCostModel(
            mode="voll",
            voll_rs_kwh=100.0,
            sla_min_autonomy_hours=2.0,
        ),
        bess=BESSBackupModel(
            capacity_kwh=100.0, power_kw=50.0, capex_rs_kwh=35000.0,
        ) if has_bess else None,
        dg=DGBackupModel(
            rated_kw=100.0, fuel_l_per_kwh=0.27, diesel_price_rs_l=95.0,
        ) if has_dg else None,
    )
    return cfg


class TestSizingCandidate:
    def test_label_bess_only(self):
        c = SizingCandidate(
            bess_kwh=200.0, bess_kw=100.0, dg_kw=0.0,
            ens_kwh_yr=50.0, downtime_hrs_yr=0.5, continuity_pct=99.9,
            backup_autonomy_hrs=2.0,
            npv=-1e6, incremental_npv=5e5, lcoe_rs_kwh=8.0,
            capex_rs=7e6, payback_years=8.0,
            sla_pass=True,
        )
        assert "BESS" in c.label
        assert "200" in c.label
        assert "DG" not in c.label

    def test_label_dg_only(self):
        c = SizingCandidate(
            bess_kwh=0.0, bess_kw=0.0, dg_kw=150.0,
            ens_kwh_yr=10.0, downtime_hrs_yr=0.1, continuity_pct=99.95,
            backup_autonomy_hrs=10.0,
            npv=-5e5, incremental_npv=3e5, lcoe_rs_kwh=9.0,
            capex_rs=1.2e6, payback_years=5.0,
            sla_pass=True,
        )
        assert "DG" in c.label
        assert "150" in c.label
        assert "BESS" not in c.label

    def test_label_grid_only(self):
        c = SizingCandidate(
            bess_kwh=0.0, bess_kw=0.0, dg_kw=0.0,
            ens_kwh_yr=500.0, downtime_hrs_yr=5.0, continuity_pct=97.0,
            backup_autonomy_hrs=0.0,
            npv=-2e6, incremental_npv=None, lcoe_rs_kwh=7.5,
            capex_rs=0.0, payback_years=None,
            sla_pass=False,
        )
        assert "Grid Only" in c.label

    def test_to_dict_keys(self):
        c = SizingCandidate(
            bess_kwh=200.0, bess_kw=100.0, dg_kw=100.0,
            ens_kwh_yr=10.0, downtime_hrs_yr=0.1, continuity_pct=99.9,
            backup_autonomy_hrs=4.0,
            npv=2e6, incremental_npv=3e5, lcoe_rs_kwh=8.5,
            capex_rs=8.5e6, payback_years=7.5,
            sla_pass=True,
        )
        d = c.to_dict()
        for key in ["rank", "label", "bess_kwh", "dg_kw", "capex_rs",
                    "npv_rs", "lcoe_rs_kwh", "ens_kwh_yr", "sla_pass"]:
            assert key in d


class TestSizingOptimizerBuildConfig:
    def test_inject_bess_on_empty_config(self):
        cfg = _make_base_config(has_bess=False)
        opt = SizingOptimizer(config=cfg, bess_sizes_kwh=[200.0], dg_sizes_kw=[0.0])
        new_cfg = opt._build_config(200.0, 0.0)
        assert new_cfg.bess is not None
        assert new_cfg.bess.capacity_kwh == pytest.approx(200.0)
        assert new_cfg.bess.power_kw == pytest.approx(100.0)  # 0.5 ratio

    def test_inject_dg_on_empty_config(self):
        cfg = _make_base_config(has_dg=False)
        opt = SizingOptimizer(config=cfg, bess_sizes_kwh=[0.0], dg_sizes_kw=[100.0])
        new_cfg = opt._build_config(0.0, 100.0)
        assert new_cfg.dg is not None
        assert new_cfg.dg.rated_kw == pytest.approx(100.0)

    def test_no_bess_when_zero_kwh(self):
        cfg = _make_base_config(has_bess=True)
        opt = SizingOptimizer(config=cfg, bess_sizes_kwh=[0.0], dg_sizes_kw=[0.0])
        new_cfg = opt._build_config(0.0, 0.0)
        assert new_cfg.bess is None

    def test_no_dg_when_zero_kw(self):
        cfg = _make_base_config(has_dg=True)
        opt = SizingOptimizer(config=cfg, bess_sizes_kwh=[0.0], dg_sizes_kw=[0.0])
        new_cfg = opt._build_config(0.0, 0.0)
        assert new_cfg.dg is None

    def test_reuse_existing_bess_template(self):
        cfg = _make_base_config(has_bess=True)
        original_capex = cfg.bess.capex_rs_kwh
        opt = SizingOptimizer(config=cfg, bess_sizes_kwh=[300.0], dg_sizes_kw=[0.0])
        new_cfg = opt._build_config(300.0, 0.0)
        assert new_cfg.bess.capacity_kwh == pytest.approx(300.0)
        assert new_cfg.bess.capex_rs_kwh == pytest.approx(original_capex)  # preserved


class TestSizingOptimizerRun:
    def test_run_returns_candidates(self):
        cfg = _make_base_config()
        opt = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=[0.0, 100.0],
            dg_sizes_kw=[0.0],
            seed=42,
        )
        candidates = opt.run()
        assert len(candidates) >= 1

    def test_candidates_have_ranks(self):
        cfg = _make_base_config()
        opt = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=[0.0, 100.0, 200.0],
            dg_sizes_kw=[0.0],
            seed=42,
        )
        candidates = opt.run()
        ranks = [c.rank for c in candidates]
        assert ranks == list(range(1, len(candidates) + 1))

    def test_sla_pass_ranked_first(self):
        """SLA-passing candidates should be ranked before SLA-failing ones."""
        cfg = _make_base_config()
        opt = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=[0.0, 200.0],
            dg_sizes_kw=[0.0],
            require_sla_pass=True,
            seed=42,
        )
        candidates = opt.run()
        if not candidates:
            return
        # Find first failing candidate
        pass_ranks = [c.rank for c in candidates if c.sla_pass]
        fail_ranks = [c.rank for c in candidates if not c.sla_pass]
        if pass_ranks and fail_ranks:
            assert max(pass_ranks) < min(fail_ranks)

    def test_capex_ceiling_filters(self):
        """Candidates above CAPEX ceiling should be excluded."""
        cfg = _make_base_config()
        opt = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=[0.0, 50.0, 200.0, 500.0],
            dg_sizes_kw=[0.0],
            max_capex_rs=5_000_000,  # 5M cap — should exclude 500 kWh BESS
            seed=42,
        )
        candidates = opt.run()
        for c in candidates:
            assert c.capex_rs <= 5_000_000 + 1  # +1 for floating point

    def test_progress_callback(self):
        cfg = _make_base_config()
        opt = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=[0.0, 100.0],
            dg_sizes_kw=[0.0, 100.0],
            seed=42,
        )
        progress_calls = []
        opt.run(progress_callback=lambda p: progress_calls.append(p))
        assert len(progress_calls) > 0
        assert progress_calls[-1] == pytest.approx(1.0)

    def test_recommendation_text_nonempty(self):
        cfg = _make_base_config()
        opt = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=[0.0, 100.0],
            dg_sizes_kw=[0.0],
            seed=42,
        )
        candidates = opt.run()
        text = SizingOptimizer.recommendation_text(candidates)
        assert len(text) > 0
        assert "RECOMMENDED" in text

    def test_recommendation_text_empty(self):
        text = SizingOptimizer.recommendation_text([])
        assert "No viable" in text

    def test_incremental_npv_objective_ordering(self):
        """With 'incremental_npv', candidates should be sorted descending by incr. NPV."""
        cfg = _make_base_config()
        opt = SizingOptimizer(
            config=cfg,
            bess_sizes_kwh=[0.0, 50.0, 100.0, 200.0],
            dg_sizes_kw=[0.0],
            optimize_for="incremental_npv",
            require_sla_pass=False,
            seed=42,
        )
        candidates = opt.run()
        if len(candidates) < 2:
            return
        # Check that first candidate has highest incremental NPV
        incr_npvs = [
            (c.incremental_npv if c.incremental_npv is not None else c.npv)
            for c in candidates
        ]
        assert incr_npvs[0] >= incr_npvs[1]
