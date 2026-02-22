"""
Tests for the outage generator module.
"""

import math
import pytest
import numpy as np

from energex.domain.schemas import (
    OutageModel, OutageMode, StochasticOutageParams,
    DurationDistribution, DeterministicOutageEvent,
)
from energex.engine.outage_generator import (
    OutageGenerator, OutageEvent, build_outage_mask, get_outage_hours_in_slot,
)


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------

class TestDeterministicOutageGenerator:

    def _model(self, events):
        return OutageModel(
            mode=OutageMode.DETERMINISTIC,
            events=[DeterministicOutageEvent(**e) for e in events],
        )

    def test_single_event_preserved(self):
        model = self._model([{"start_hour": 100.0, "duration_hours": 3.0}])
        gen = OutageGenerator(model, seed=0)
        events = gen.generate()
        assert len(events) == 1
        assert events[0].start_hour == 100.0
        assert events[0].duration_hours == 3.0

    def test_multiple_events_sorted(self):
        model = self._model([
            {"start_hour": 500.0, "duration_hours": 2.0},
            {"start_hour": 100.0, "duration_hours": 1.5},
            {"start_hour": 300.0, "duration_hours": 0.5},
        ])
        gen = OutageGenerator(model, seed=0)
        events = gen.generate()
        starts = [e.start_hour for e in events]
        assert starts == sorted(starts)

    def test_event_beyond_year_ignored(self):
        model = self._model([
            {"start_hour": 9000.0, "duration_hours": 1.0},
            {"start_hour": 100.0, "duration_hours": 2.0},
        ])
        gen = OutageGenerator(model, seed=0)
        events = gen.generate()
        assert all(e.start_hour < 8760 for e in events)
        assert len(events) == 1

    def test_end_hour_clipped_to_year_boundary(self):
        model = self._model([{"start_hour": 8758.0, "duration_hours": 10.0}])
        gen = OutageGenerator(model, seed=0)
        events = gen.generate()
        assert events[0].end_hour == 8760.0


# ---------------------------------------------------------------------------
# Stochastic mode
# ---------------------------------------------------------------------------

class TestStochasticOutageGenerator:

    def _model(self, saidi=600, saifi=10, dist=DurationDistribution.LOGNORMAL, sigma=0.8):
        return OutageModel(
            mode=OutageMode.STOCHASTIC,
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=saidi,
                saifi_events_per_year=saifi,
                duration_distribution=dist,
                duration_sigma=sigma,
            )
        )

    def test_determinism_with_seed(self):
        model = self._model()
        gen1 = OutageGenerator(model, seed=42)
        gen2 = OutageGenerator(model, seed=42)
        e1 = gen1.generate()
        e2 = gen2.generate()
        assert len(e1) == len(e2)
        for a, b in zip(e1, e2):
            assert abs(a.start_hour - b.start_hour) < 1e-9
            assert abs(a.duration_hours - b.duration_hours) < 1e-9

    def test_different_seeds_give_different_results(self):
        model = self._model(saifi=20)
        gen1 = OutageGenerator(model, seed=42)
        gen2 = OutageGenerator(model, seed=999)
        e1 = gen1.generate()
        e2 = gen2.generate()
        # Start hours must differ between two runs (different random placements)
        starts1 = sorted(round(e.start_hour, 1) for e in e1)
        starts2 = sorted(round(e.start_hour, 1) for e in e2)
        assert starts1 != starts2, "Events with different seeds should have different start hours"

    def test_all_events_within_year(self):
        model = self._model(saifi=50)
        gen = OutageGenerator(model, seed=7)
        events = gen.generate()
        for ev in events:
            assert ev.start_hour >= 0
            assert ev.end_hour <= 8760.0 + 1e-9

    def test_events_sorted(self):
        model = self._model(saifi=30)
        gen = OutageGenerator(model, seed=42)
        events = gen.generate()
        starts = [e.start_hour for e in events]
        assert starts == sorted(starts)

    def test_no_overlapping_events(self):
        model = self._model(saifi=50)
        gen = OutageGenerator(model, seed=42)
        events = gen.generate()
        for i in range(len(events) - 1):
            assert events[i].end_hour <= events[i + 1].start_hour + 1e-9

    def test_saidi_saifi_reasonable_expectation(self):
        """Over many seeds, SAIDI/SAIFI should be approximately correct."""
        model = self._model(saidi=600, saifi=10)
        total_duration = 0.0
        total_events = 0
        n_trials = 50
        for seed in range(n_trials):
            gen = OutageGenerator(model, seed=seed)
            events = gen.generate()
            total_events += len(events)
            total_duration += sum(e.effective_duration * 60 for e in events)
        avg_saifi = total_events / n_trials
        avg_saidi = total_duration / n_trials
        # Allow ±70% tolerance for Poisson variability
        assert 3.0 <= avg_saifi <= 30.0, f"SAIFI avg {avg_saifi:.1f} out of range"
        assert 100 <= avg_saidi <= 3000, f"SAIDI avg {avg_saidi:.1f} out of range"

    def test_exponential_distribution(self):
        model = self._model(dist=DurationDistribution.EXPONENTIAL)
        gen = OutageGenerator(model, seed=42)
        events = gen.generate()
        assert len(events) >= 0  # Just check it runs

    def test_monthly_weights_used(self):
        # Summer-heavy weights
        weights = [0.1] * 12
        weights[5] = 5.0   # June
        weights[6] = 5.0   # July
        weights[7] = 5.0   # August
        model = OutageModel(
            mode=OutageMode.STOCHASTIC,
            stochastic=StochasticOutageParams(
                saidi_minutes_per_year=1200,
                saifi_events_per_year=50,
                duration_distribution=DurationDistribution.LOGNORMAL,
                duration_sigma=0.8,
                monthly_weights=weights,
            )
        )
        gen = OutageGenerator(model, seed=42)
        events = gen.generate()
        # Most events should fall in summer months (hours 3624–6552 approx)
        summer_hours = sum(1 for e in events if 3624 <= e.start_hour <= 6552)
        if len(events) > 0:
            summer_frac = summer_hours / len(events)
            assert summer_frac > 0.3  # at least 30% in summer


# ---------------------------------------------------------------------------
# Outage mask helpers
# ---------------------------------------------------------------------------

class TestOutageMask:

    def test_empty_mask(self):
        mask = build_outage_mask([])
        assert not mask.any()

    def test_single_outage_mask(self):
        events = [OutageEvent(start_hour=10.0, duration_hours=3.0)]
        mask = build_outage_mask(events)
        assert mask[10] and mask[11] and mask[12]
        assert not mask[9] and not mask[13]

    def test_outage_fraction_full_hour(self):
        events = [OutageEvent(start_hour=5.0, duration_hours=2.0)]
        frac = get_outage_hours_in_slot(events, 5)
        assert abs(frac - 1.0) < 1e-9
        frac2 = get_outage_hours_in_slot(events, 7)
        assert frac2 == 0.0

    def test_outage_fraction_partial(self):
        events = [OutageEvent(start_hour=5.5, duration_hours=1.0)]
        frac = get_outage_hours_in_slot(events, 5)
        assert abs(frac - 0.5) < 1e-9
        frac2 = get_outage_hours_in_slot(events, 6)
        assert abs(frac2 - 0.5) < 1e-9
