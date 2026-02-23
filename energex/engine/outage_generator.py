"""
EnergeX — Outage Generator

Generates a sequence of outage events for a simulated year.

Supports:
  - Deterministic: user-supplied known outage schedule
  - Stochastic:    SAIDI/SAIFI-based random generation (reproducible via seed)

Output is a list of OutageEvent objects (start_hour, end_hour).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from energex.domain.schemas import OutageMode, DurationDistribution, OutageModel


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(order=True)
class OutageEvent:
    """A single grid outage event."""
    start_hour: float        # hour-of-year when outage begins  [0, 8760)
    duration_hours: float    # duration in hours
    end_hour: float = field(init=False)

    def __post_init__(self) -> None:
        self.end_hour = min(self.start_hour + self.duration_hours, 8760.0)

    @property
    def effective_duration(self) -> float:
        """Clipped to year boundary."""
        return self.end_hour - self.start_hour


# ---------------------------------------------------------------------------
# Outage Generator
# ---------------------------------------------------------------------------

class OutageGenerator:
    """
    Generate outage events for one simulation year.

    Parameters
    ----------
    outage_model : OutageModel
    seed : int
        Random seed for reproducibility.
    """

    HOURS_PER_YEAR: int = 8760
    HOURS_PER_MONTH: list[int] = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

    def __init__(self, outage_model: OutageModel, seed: int = 42) -> None:
        self.model = outage_model
        self.seed = seed

    def generate(self) -> list[OutageEvent]:
        """Return list of OutageEvent for one simulated year, sorted by start_hour."""
        if self.model.mode == OutageMode.DETERMINISTIC:
            return self._generate_deterministic()
        else:
            return self._generate_stochastic()

    # ------------------------------------------------------------------
    # Deterministic
    # ------------------------------------------------------------------

    def _generate_deterministic(self) -> list[OutageEvent]:
        events: list[OutageEvent] = []
        for ev in self.model.events:
            if ev.start_hour >= self.HOURS_PER_YEAR:
                continue
            events.append(OutageEvent(
                start_hour=ev.start_hour,
                duration_hours=ev.duration_hours,
            ))
        events.sort()
        return events

    # ------------------------------------------------------------------
    # Stochastic (SAIDI / SAIFI)
    # ------------------------------------------------------------------

    def _generate_stochastic(self) -> list[OutageEvent]:
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        params = self.model.stochastic
        saidi_hours = params.saidi_minutes_per_year / 60.0   # convert to hours
        saifi = params.saifi_events_per_year                  # events/yr

        # Number of outage events — Poisson distributed around SAIFI
        n_events = int(np_rng.poisson(saifi))
        if n_events == 0:
            return []

        # Target mean duration from SAIDI / SAIFI
        mean_duration_hours = saidi_hours / saifi

        # Sample event durations
        durations = self._sample_durations(
            np_rng,
            n=n_events,
            mean_hours=mean_duration_hours,
            distribution=params.duration_distribution,
            sigma=params.duration_sigma,
        )

        # Scale durations so sum ≈ SAIDI (preserve expected total ENS)
        total = sum(durations)
        if total > 0:
            scale = (saifi * mean_duration_hours) / total
            # Don't scale too aggressively — cap at 3×
            scale = min(scale, 3.0)
            durations = [d * scale for d in durations]

        # Sample start times (weighted by monthly pattern)
        start_hours = self._sample_start_hours(rng, np_rng, n_events, params.monthly_weights)

        # Build events, avoid obvious overlap (merge if needed)
        raw_events = sorted(
            [OutageEvent(start_hour=s, duration_hours=max(d, 0.0167))  # min 1 min
             for s, d in zip(start_hours, durations)],
            key=lambda e: e.start_hour,
        )

        merged = self._merge_overlapping(raw_events)
        return merged

    def _sample_durations(
        self,
        rng: np.random.Generator,
        n: int,
        mean_hours: float,
        distribution: DurationDistribution,
        sigma: float,
    ) -> list[float]:
        """Sample n outage durations."""
        if distribution == DurationDistribution.LOGNORMAL:
            # lognormal: mean of underlying normal = ln(mean) - sigma²/2
            mu = math.log(mean_hours) - 0.5 * sigma ** 2
            raw = rng.lognormal(mean=mu, sigma=sigma, size=n)
        elif distribution == DurationDistribution.EXPONENTIAL:
            raw = rng.exponential(scale=mean_hours, size=n)
        else:  # EMPIRICAL — fallback to exponential
            raw = rng.exponential(scale=mean_hours, size=n)
        return [max(float(d), 0.0167) for d in raw]  # minimum 1 min

    def _sample_start_hours(
        self,
        rng: random.Random,
        np_rng: np.random.Generator,
        n: int,
        monthly_weights: Optional[list[float]],
    ) -> list[float]:
        """Sample n start hours distributed across the year."""
        if monthly_weights is None or len(monthly_weights) != 12:
            # Uniform across year
            return [rng.uniform(0, self.HOURS_PER_YEAR) for _ in range(n)]

        # Weighted monthly selection
        weights = [max(w, 0.0) for w in monthly_weights]
        total_w = sum(weights)
        if total_w == 0:
            weights = [1.0] * 12

        month_starts = [0] + list(np.cumsum(self.HOURS_PER_MONTH[:-1]))
        start_hours: list[float] = []

        for _ in range(n):
            # Pick month by weight
            month_idx = rng.choices(range(12), weights=weights, k=1)[0]
            month_start = month_starts[month_idx]
            month_len = self.HOURS_PER_MONTH[month_idx]
            hour = rng.uniform(month_start, month_start + month_len)
            start_hours.append(min(hour, self.HOURS_PER_YEAR - 0.5))

        return start_hours

    @staticmethod
    def _merge_overlapping(events: list[OutageEvent]) -> list[OutageEvent]:
        """Merge overlapping/adjacent outage events."""
        if not events:
            return []
        merged: list[OutageEvent] = []
        cur_start = events[0].start_hour
        cur_end = events[0].end_hour

        for ev in events[1:]:
            if ev.start_hour <= cur_end:
                # Overlap — extend
                cur_end = max(cur_end, ev.end_hour)
            else:
                merged.append(OutageEvent(
                    start_hour=cur_start,
                    duration_hours=cur_end - cur_start,
                ))
                cur_start = ev.start_hour
                cur_end = ev.end_hour

        merged.append(OutageEvent(
            start_hour=cur_start,
            duration_hours=cur_end - cur_start,
        ))
        return merged


# ---------------------------------------------------------------------------
# Outage coverage helper — used by dispatch engine
# ---------------------------------------------------------------------------

def build_outage_mask(events: list[OutageEvent], n_hours: int = 8760) -> np.ndarray:
    """
    Return boolean numpy array of length n_hours where True = outage active.

    Uses integer hour slots (hour i covers [i, i+1)).
    """
    mask = np.zeros(n_hours, dtype=bool)
    for ev in events:
        h_start = int(ev.start_hour)
        h_end = min(int(math.ceil(ev.end_hour)), n_hours)
        mask[h_start:h_end] = True
    return mask


def outage_fraction_in_hour(event: OutageEvent, hour: int) -> float:
    """
    Return fraction of hour `hour` that is in outage (0..1).

    `hour` is the integer hour index (slot [hour, hour+1)).
    """
    slot_start = float(hour)
    slot_end = float(hour + 1)
    overlap_start = max(event.start_hour, slot_start)
    overlap_end = min(event.end_hour, slot_end)
    return max(0.0, overlap_end - overlap_start)


def get_outage_hours_in_slot(events: list[OutageEvent], hour: int) -> float:
    """Total outage hours within the integer hour slot."""
    total = 0.0
    for ev in events:
        total += outage_fraction_in_hour(ev, hour)
    return min(total, 1.0)  # can't exceed 1 hr per slot
