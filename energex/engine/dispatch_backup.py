"""
EnergeX — Backup Dispatch Engine

Hourly simulation of energy flows and backup asset operation.

Dispatch priority:
  Grid-UP:   solar → load, excess solar → BESS charge, grid fills remaining
  Grid-DOWN: BESS first (instant), then DG after start delay
              If critical load unmet → record ENS + downtime
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from energex.domain.schemas import (
    LoadProfile,
    DGBackupModel,
    BESSBackupModel,
    SolarModel,
    ProjectConfig,
)
from energex.engine.outage_generator import OutageEvent, get_outage_hours_in_slot


# ---------------------------------------------------------------------------
# Hourly state and results
# ---------------------------------------------------------------------------

@dataclass
class HourlyState:
    """State of BESS and DG at start of each simulation hour."""
    soc_kwh: float              # BESS state of charge (kWh)
    dg_fuel_used_liters: float  # Cumulative DG fuel used (L)
    bess_throughput_kwh: float  # Cumulative BESS throughput (kWh)
    year: int                   # Simulation year (0-based)


@dataclass
class HourlyResult:
    """Energy flows and reliability metrics for one hour."""
    hour: int
    # Availability
    grid_available: bool
    outage_fraction: float      # fraction of hour in outage (0..1)

    # Energy flows (kWh)
    grid_to_load: float = 0.0
    solar_to_load: float = 0.0
    solar_to_bess: float = 0.0
    grid_to_bess: float = 0.0
    bess_to_load: float = 0.0
    dg_to_load: float = 0.0
    unserved_critical: float = 0.0     # ENS (kWh)

    # Load served (kWh)
    total_load_kwh: float = 0.0
    critical_load_kwh: float = 0.0

    # DG
    dg_fuel_liters: float = 0.0
    dg_running: bool = False

    # BESS SOC at end of hour (kWh)
    bess_soc_end: float = 0.0

    # Reliability
    downtime_hours: float = 0.0        # hours of critical load unmet


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Full simulation result for one year."""
    hourly: list[HourlyResult] = field(default_factory=list)

    # Aggregated reliability
    ens_kwh: float = 0.0
    downtime_hours: float = 0.0
    outage_events_total: int = 0
    outage_events_served: int = 0       # events with zero ENS
    backup_autonomy_hours: float = 0.0  # at peak critical load

    # Aggregated energy flows (kWh/yr)
    total_grid_kwh: float = 0.0
    total_solar_kwh: float = 0.0
    total_dg_kwh: float = 0.0
    total_bess_discharge_kwh: float = 0.0
    total_solar_to_bess_kwh: float = 0.0
    total_grid_to_bess_kwh: float = 0.0

    # Total DG fuel
    dg_fuel_liters: float = 0.0

    # Final BESS state
    bess_soc_final_kwh: float = 0.0
    bess_capacity_degraded_kwh: float = 0.0   # effective capacity after degradation

    # SLA pass/fail
    sla_pass: bool = True
    sla_failures: list[str] = field(default_factory=list)

    @property
    def continuity_pct(self) -> float:
        total_critical_kwh = sum(h.critical_load_kwh for h in self.hourly)
        if total_critical_kwh <= 0:
            return 100.0
        return 100.0 * (1.0 - self.ens_kwh / total_critical_kwh)

    @property
    def outage_served_pct(self) -> float:
        if self.outage_events_total == 0:
            return 100.0
        return 100.0 * self.outage_events_served / self.outage_events_total


# ---------------------------------------------------------------------------
# Load profile generator
# ---------------------------------------------------------------------------

def build_hourly_loads(cfg: ProjectConfig, year: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (total_kw, critical_kw) arrays of length 8760 for given sim year.

    Simple mode: flat profile scaled to avg_kw.
    Timeseries mode: loaded from CSV (year-0 data repeated each year).
    """
    lp = cfg.load
    if lp.mode.value == "simple":
        avg = lp.avg_kw
        crit_avg = avg * lp.critical_fraction
        # Simple sinusoidal day/night variation around avg
        hours = np.arange(8760)
        hour_of_day = hours % 24
        # Day-night shape: peaks at 14:00, troughs at 03:00
        shape = 1.0 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 3) / 24)
        total = avg * shape
        # Scale so mean ≈ avg_kw
        total = total * (avg / total.mean())
        critical = total * lp.critical_fraction
        return total, critical
    else:
        # Timeseries CSV — loaded by adapter; not handled here at engine level
        raise RuntimeError("Timeseries mode: load arrays must be provided externally")


def build_hourly_solar(cfg: ProjectConfig, year: int) -> np.ndarray:
    """
    Return solar generation array (kW) of length 8760.
    Applies annual degradation.
    """
    solar = cfg.solar
    if solar is None:
        return np.zeros(8760)

    degrad_factor = (1.0 - solar.degradation_pct_yr / 100.0) ** year

    if solar.monthly_yield is not None:
        # Build hourly profile from monthly averages
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        gen = np.zeros(8760)
        h = 0
        for m, (monthly_kwh_kwp_day, hrs) in enumerate(zip(solar.monthly_yield, hours_per_month)):
            # Convert kWh/kWp/day → kW (assume 8 sun-hours per day, scaled)
            daily_kwh = monthly_kwh_kwp_day * solar.size_kw
            hourly_avg = daily_kwh / 24.0
            # Simple bell curve for solar (daylight hours 6–18)
            for offset in range(hrs):
                hod = (h + offset) % 24
                if 6 <= hod <= 18:
                    factor = math.sin(math.pi * (hod - 6) / 12)
                    gen[h + offset] = hourly_avg * factor * (24.0 / 8.0)  # scale to daily
                else:
                    gen[h + offset] = 0.0
            h += hrs
        gen = np.clip(gen, 0, None) * degrad_factor
        return gen

    elif solar.yield_kwh_kw_day is not None:
        # Flat daily yield with day/night shape
        daily_kwh = solar.yield_kwh_kw_day * solar.size_kw
        hourly_avg = daily_kwh / 24.0
        hours = np.arange(8760)
        hod = hours % 24
        gen = np.where(
            (hod >= 6) & (hod <= 18),
            hourly_avg * np.sin(np.pi * (hod - 6) / 12) * (24.0 / 8.0),
            0.0
        )
        return np.clip(gen, 0, None) * degrad_factor

    return np.zeros(8760)


# ---------------------------------------------------------------------------
# BESS helper
# ---------------------------------------------------------------------------

class BESSSimulator:
    """Tracks BESS SOC and degradation during simulation."""

    def __init__(self, model: BESSBackupModel, year: int) -> None:
        self.model = m = model
        # Apply calendar degradation for this year
        cal_fade = (1.0 - m.calendar_fade_pct_yr / 100.0) ** year
        self.effective_capacity = m.capacity_kwh * cal_fade
        self.usable_kwh = self.effective_capacity * m.dod_pct / 100.0
        self.reserve_kwh = self.effective_capacity * m.min_reserve_soc_pct / 100.0

        # SOC tracking
        self.soc_kwh = m.initial_soc_pct / 100.0 * self.effective_capacity
        self.soc_kwh = min(self.soc_kwh, self.effective_capacity)

        self.throughput_kwh = 0.0
        self.charge_eff = math.sqrt(m.roundtrip_efficiency)
        self.discharge_eff = math.sqrt(m.roundtrip_efficiency)

    @property
    def available_kwh(self) -> float:
        """Energy available to dispatch (above reserve)."""
        return max(0.0, self.soc_kwh - self.reserve_kwh)

    @property
    def headroom_kwh(self) -> float:
        """Space available for charging."""
        return max(0.0, self.effective_capacity - self.soc_kwh)

    def discharge(self, requested_kw: float, dt_hours: float = 1.0) -> float:
        """
        Discharge BESS. Returns actual kWh delivered to load.
        Updates SOC. Applies cycle degradation.
        """
        max_discharge_kwh = min(
            self.model.power_kw * dt_hours,
            self.available_kwh / self.discharge_eff,
        )
        actual_kwh = min(requested_kw * dt_hours, max_discharge_kwh)
        actual_kwh = max(0.0, actual_kwh)
        soc_reduction = actual_kwh * self.discharge_eff
        self.soc_kwh -= soc_reduction
        self.soc_kwh = max(self.soc_kwh, 0.0)
        self.throughput_kwh += actual_kwh
        # Cycle-based capacity fade (update effective capacity)
        fade = self.model.cycle_fade_per_kwh * actual_kwh
        self.effective_capacity *= (1.0 - fade)
        return actual_kwh  # delivered to load

    def charge(self, available_kw: float, dt_hours: float = 1.0) -> float:
        """
        Charge BESS with available power. Returns kWh taken from source.
        Updates SOC.
        """
        max_charge_kwh = min(
            self.model.power_kw * dt_hours,
            self.headroom_kwh / self.charge_eff,
        )
        actual_from_source = min(available_kw * dt_hours, max_charge_kwh)
        actual_from_source = max(0.0, actual_from_source)
        self.soc_kwh += actual_from_source * self.charge_eff
        self.soc_kwh = min(self.soc_kwh, self.effective_capacity)
        return actual_from_source  # taken from source


# ---------------------------------------------------------------------------
# DG helper
# ---------------------------------------------------------------------------

class DGSimulator:
    """Tracks DG state during simulation."""

    def __init__(self, model: DGBackupModel) -> None:
        self.model = model
        self.fuel_used_liters = 0.0
        self.fuel_available = model.fuel_storage_liters  # None = unlimited
        self.start_delay_hours = model.start_delay_seconds / 3600.0
        self.min_load_kw = model.rated_kw * model.min_loading_pct / 100.0

    def can_start(self, outage_elapsed_hours: float, random_fail: bool = False) -> bool:
        """Return True if DG can supply power at this outage elapsed time."""
        if outage_elapsed_hours < self.start_delay_hours:
            return False
        if random_fail:
            return False  # availability handled externally
        return True

    def supply(self, required_kw: float, dt_hours: float = 1.0) -> float:
        """
        Supply required_kw from DG. Returns actual kWh supplied.
        Enforces min-load (DG runs at min load if required < min; surplus goes to BESS).
        Deducts fuel.
        """
        if required_kw <= 0:
            return 0.0

        actual_kw = max(required_kw, self.min_load_kw)
        actual_kw = min(actual_kw, self.model.rated_kw)

        actual_kwh = actual_kw * dt_hours
        fuel_liters = self.model.fuel_consumption_lph(actual_kw) * dt_hours

        # Fuel constraint
        if self.fuel_available is not None:
            if self.fuel_available <= 0:
                return 0.0
            fuel_liters = min(fuel_liters, self.fuel_available)
            # Scale energy proportionally
            if fuel_liters < self.model.fuel_consumption_lph(actual_kw) * dt_hours:
                scale = fuel_liters / (self.model.fuel_consumption_lph(actual_kw) * dt_hours)
                actual_kwh *= scale
            self.fuel_available -= fuel_liters

        self.fuel_used_liters += fuel_liters
        return min(actual_kwh, required_kw * dt_hours)   # deliver only what was needed


# ---------------------------------------------------------------------------
# Core dispatch engine
# ---------------------------------------------------------------------------

class BackupDispatchEngine:
    """
    Hourly backup dispatch simulation.

    Parameters
    ----------
    config : ProjectConfig
    outage_events : list[OutageEvent]  for this simulation year
    year : int  (0-based)
    seed : int  (for DG availability randomisation)
    total_kw : np.ndarray  optional pre-built load array (8760)
    critical_kw : np.ndarray  optional pre-built critical load array (8760)
    solar_kw : np.ndarray  optional pre-built solar array (8760)
    """

    def __init__(
        self,
        config: ProjectConfig,
        outage_events: list[OutageEvent],
        year: int = 0,
        seed: int = 42,
        total_kw: Optional[np.ndarray] = None,
        critical_kw: Optional[np.ndarray] = None,
        solar_kw: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config
        self.outage_events = outage_events
        self.year = year
        self.seed = seed

        # Build load arrays if not provided
        if total_kw is not None and critical_kw is not None:
            self.total_kw = total_kw
            self.critical_kw = critical_kw
        else:
            self.total_kw, self.critical_kw = build_hourly_loads(config, year)

        # Solar
        if solar_kw is not None:
            self.solar_kw = solar_kw
        else:
            self.solar_kw = build_hourly_solar(config, year)

        # Asset simulators
        self.bess: Optional[BESSSimulator] = (
            BESSSimulator(config.bess, year) if config.bess else None
        )
        self.dg: Optional[DGSimulator] = (
            DGSimulator(config.dg) if config.dg else None
        )

        # DG availability random draws (per hour — True = DG failed)
        rng = np.random.default_rng(seed + year * 1000)
        if config.dg is not None:
            avail = config.dg.availability_pct / 100.0
            self.dg_failed = rng.random(8760) > avail
        else:
            self.dg_failed = np.zeros(8760, dtype=bool)

        # BESS availability
        if config.bess is not None:
            bess_avail = config.bess.availability_pct / 100.0
            self.bess_failed = rng.random(8760) > bess_avail
        else:
            self.bess_failed = np.zeros(8760, dtype=bool)

        # Build outage state per hour
        # outage_hours_in_slot[h] = fraction of hour h in outage
        self._outage_fractions = np.array([
            get_outage_hours_in_slot(outage_events, h) for h in range(8760)
        ])

        # Map from outage event → which integer hours it spans
        self._build_event_hour_map()

    def _build_event_hour_map(self) -> None:
        """Map each outage event to its start hour (for elapsed time tracking)."""
        self.hour_outage_start: dict[int, float] = {}  # hour → outage start_hour
        for ev in self.outage_events:
            h_start = int(ev.start_hour)
            h_end = min(int(math.ceil(ev.end_hour)), 8760)
            for h in range(h_start, h_end):
                if h not in self.hour_outage_start:
                    self.hour_outage_start[h] = ev.start_hour

    def _outage_elapsed(self, hour: int) -> float:
        """Hours elapsed since outage start at this hour."""
        if hour not in self.hour_outage_start:
            return 0.0
        return float(hour) - self.hour_outage_start[hour]

    def run(self) -> SimulationResult:
        """Run the full 8760-hour simulation. Returns SimulationResult."""
        results: list[HourlyResult] = []
        # Track per-event ENS for outage_events_served metric
        event_ens: dict[int, float] = {i: 0.0 for i in range(len(self.outage_events))}
        hour_to_event: dict[int, int] = {}
        for i, ev in enumerate(self.outage_events):
            h_start = int(ev.start_hour)
            h_end = min(int(math.ceil(ev.end_hour)), 8760)
            for h in range(h_start, h_end):
                hour_to_event[h] = i

        for h in range(8760):
            of = self._outage_fractions[h]
            grid_up_frac = 1.0 - of  # fraction of hour grid is available
            grid_down_frac = of      # fraction of hour grid is out

            hr = HourlyResult(
                hour=h,
                grid_available=(of < 1.0),
                outage_fraction=of,
                total_load_kwh=self.total_kw[h] * 1.0,
                critical_load_kwh=self.critical_kw[h] * 1.0,
                bess_soc_end=self.bess.soc_kwh if self.bess else 0.0,
            )

            solar_gen = self.solar_kw[h]  # kWh (1h slot)

            # ---- GRID-UP portion of this hour ----
            if grid_up_frac > 0:
                self._dispatch_grid_up(h, hr, solar_gen * grid_up_frac, grid_up_frac)

            # ---- GRID-DOWN portion of this hour ----
            if grid_down_frac > 0:
                elapsed = self._outage_elapsed(h)
                ens = self._dispatch_grid_down(h, hr, solar_gen * grid_down_frac, grid_down_frac, elapsed)
                if h in hour_to_event:
                    event_ens[hour_to_event[h]] += ens

            hr.bess_soc_end = self.bess.soc_kwh if self.bess else 0.0
            results.append(hr)

        # Aggregate
        result = SimulationResult(hourly=results)
        result.ens_kwh = sum(h.unserved_critical for h in results)
        result.downtime_hours = sum(h.downtime_hours for h in results)
        result.total_grid_kwh = sum(h.grid_to_load + h.grid_to_bess for h in results)
        result.total_solar_kwh = sum(h.solar_to_load + h.solar_to_bess for h in results)
        result.total_dg_kwh = sum(h.dg_to_load for h in results)
        result.total_bess_discharge_kwh = sum(h.bess_to_load for h in results)
        result.total_solar_to_bess_kwh = sum(h.solar_to_bess for h in results)
        result.total_grid_to_bess_kwh = sum(h.grid_to_bess for h in results)
        result.dg_fuel_liters = self.dg.fuel_used_liters if self.dg else 0.0
        result.bess_soc_final_kwh = self.bess.soc_kwh if self.bess else 0.0
        result.bess_capacity_degraded_kwh = self.bess.effective_capacity if self.bess else 0.0
        result.outage_events_total = len(self.outage_events)
        result.outage_events_served = sum(1 for ens in event_ens.values() if ens < 1e-6)

        # Backup autonomy: BESS (initial usable) + DG rated at critical peak
        result.backup_autonomy_hours = self._compute_autonomy()

        # SLA check
        self._check_sla(result)

        return result

    def _dispatch_grid_up(
        self,
        h: int,
        hr: HourlyResult,
        solar_kwh: float,
        dt: float,
    ) -> None:
        """Dispatch for grid-up fraction of the hour."""
        load_kwh = self.total_kw[h] * dt

        # 1. Solar → load
        solar_to_load = min(solar_kwh, load_kwh)
        load_remaining = load_kwh - solar_to_load
        solar_remaining = solar_kwh - solar_to_load

        # 2. Grid → remaining load
        hr.grid_to_load += load_remaining
        hr.solar_to_load += solar_to_load

        # 3. Solar → BESS charge
        if self.bess and not self.bess_failed[h] and solar_remaining > 0:
            taken = self.bess.charge(solar_remaining / dt, dt)
            hr.solar_to_bess += taken
            solar_remaining -= taken

        # 4. Grid → BESS charge (to maintain target SOC)
        if self.bess and not self.bess_failed[h]:
            target_soc = self.bess.effective_capacity * 0.95
            deficit_kwh = target_soc - self.bess.soc_kwh
            if deficit_kwh > 0.01:
                taken = self.bess.charge(deficit_kwh / dt, dt)
                hr.grid_to_bess += taken

    def _dispatch_grid_down(
        self,
        h: int,
        hr: HourlyResult,
        solar_kwh: float,
        dt: float,
        elapsed_hours: float,
    ) -> float:
        """
        Dispatch for grid-down fraction. Returns ENS (kWh).
        Priority: BESS → DG (after start delay).
        """
        critical_kwh = self.critical_kw[h] * dt
        remaining = critical_kwh

        # Solar → critical load (if BESS not available as buffer)
        if solar_kwh > 0:
            solar_share = min(solar_kwh, remaining)
            hr.solar_to_load += solar_share
            remaining -= solar_share
            # Excess solar → charge BESS during outage
            excess_solar = solar_kwh - solar_share
            if self.bess and not self.bess_failed[h] and excess_solar > 0:
                taken = self.bess.charge(excess_solar / dt, dt)
                hr.solar_to_bess += taken

        # BESS supplies (instant response)
        if self.bess and not self.bess_failed[h] and remaining > 0:
            delivered = self.bess.discharge(remaining / dt, dt)
            hr.bess_to_load += delivered
            remaining -= delivered

        # DG supplies after start delay
        dg_available = (
            self.dg is not None
            and not self.dg_failed[h]
            and self.dg.can_start(elapsed_hours)
        )
        if dg_available and remaining > 0:
            delivered = self.dg.supply(remaining / dt, dt)
            hr.dg_to_load += delivered
            hr.dg_running = True
            hr.dg_fuel_liters += self.dg.model.fuel_consumption_lph(
                delivered / dt if dt > 0 else 0
            ) * dt
            remaining -= delivered

        # Unserved critical load
        ens = max(0.0, remaining)
        hr.unserved_critical += ens
        if ens > 1e-6:
            hr.downtime_hours += dt

        return ens

    def _compute_autonomy(self) -> float:
        """
        Backup autonomy at peak critical load (hours).
        Accounts for BESS usable energy and DG rated power.
        """
        peak_critical_kw = float(self.critical_kw.max()) if len(self.critical_kw) > 0 else 0.0
        if peak_critical_kw <= 0:
            return float("inf")

        bess_hours = 0.0
        if self.bess:
            bess_usable = self.bess.usable_kwh
            bess_power = min(self.bess.model.power_kw, peak_critical_kw)
            if bess_power > 0:
                bess_hours = bess_usable / bess_power

        dg_hours = float("inf") if (
            self.dg and self.dg.model.rated_kw >= peak_critical_kw
        ) else 0.0
        # DG with fuel storage
        if self.dg and self.dg.model.fuel_storage_liters:
            fuel_avail = self.dg.model.fuel_storage_liters
            fuel_rate = self.dg.model.fuel_consumption_lph(peak_critical_kw)
            if fuel_rate > 0:
                dg_hours = min(dg_hours, fuel_avail / fuel_rate)

        # BESS + DG in sequence
        if self.dg and self.dg.model.rated_kw >= peak_critical_kw:
            return bess_hours + (dg_hours if dg_hours != float("inf") else 999.0)
        elif self.bess:
            return bess_hours
        elif self.dg:
            return dg_hours
        return 0.0

    def _check_sla(self, result: SimulationResult) -> None:
        """Evaluate SLA targets and record pass/fail."""
        sla = self.config.outage_cost
        failures = []

        if sla.sla_max_downtime_hrs_yr is not None:
            if result.downtime_hours > sla.sla_max_downtime_hrs_yr:
                failures.append(
                    f"Downtime {result.downtime_hours:.2f} hrs/yr exceeds SLA {sla.sla_max_downtime_hrs_yr} hrs/yr"
                )

        if sla.sla_max_ens_kwh_yr is not None:
            if result.ens_kwh > sla.sla_max_ens_kwh_yr:
                failures.append(
                    f"ENS {result.ens_kwh:.2f} kWh/yr exceeds SLA {sla.sla_max_ens_kwh_yr} kWh/yr"
                )

        if sla.sla_continuity_pct is not None:
            if result.continuity_pct < sla.sla_continuity_pct:
                failures.append(
                    f"Continuity {result.continuity_pct:.3f}% below SLA {sla.sla_continuity_pct}%"
                )

        if sla.sla_min_autonomy_hours is not None:
            if result.backup_autonomy_hours < sla.sla_min_autonomy_hours:
                failures.append(
                    f"Autonomy {result.backup_autonomy_hours:.2f} hrs below SLA {sla.sla_min_autonomy_hours} hrs"
                )

        result.sla_failures = failures
        result.sla_pass = len(failures) == 0
