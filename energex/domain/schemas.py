"""
EnergeX — Domain Schemas (Pydantic v2)

All input models for the Backup/Outage Energy ROI Calculator.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LoadMode(str, Enum):
    SIMPLE = "simple"
    TIMESERIES = "timeseries"


class OutageMode(str, Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


class DurationDistribution(str, Enum):
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    EMPIRICAL = "empirical"


class OutageCostMode(str, Enum):
    VOLL = "voll"
    DOWNTIME = "downtime"


class ScenarioType(str, Enum):
    GRID_ONLY = "grid_only"
    GRID_DG = "grid_dg"
    GRID_BESS = "grid_bess"
    GRID_SOLAR_BESS = "grid_solar_bess"
    GRID_DG_BESS = "grid_dg_bess"
    GRID_SOLAR_DG_BESS = "grid_solar_dg_bess"


# ---------------------------------------------------------------------------
# 3.1 Load Profile
# ---------------------------------------------------------------------------

class DeterministicOutageEvent(BaseModel):
    """A known outage event (deterministic mode)."""
    start_hour: float = Field(..., ge=0, description="Hour of year when outage starts")
    duration_hours: float = Field(..., gt=0, description="Duration of outage in hours")


class LoadProfile(BaseModel):
    """Load profile configuration."""

    mode: LoadMode = LoadMode.SIMPLE

    # Simple mode
    avg_kw: Optional[float] = Field(None, gt=0, description="Average total load (kW)")
    peak_kw: Optional[float] = Field(None, gt=0, description="Peak total load (kW)")
    critical_fraction: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Fraction of avg load that is critical (0–1)"
    )

    # Timeseries mode
    timeseries_csv: Optional[str] = Field(
        None, description="Path to CSV with columns: hour, total_kw, critical_kw"
    )

    @model_validator(mode="after")
    def check_mode_fields(self) -> "LoadProfile":
        if self.mode == LoadMode.SIMPLE:
            if self.avg_kw is None:
                raise ValueError("avg_kw is required for simple load mode")
            if self.peak_kw is None:
                self.peak_kw = self.avg_kw * 1.3
            if self.peak_kw < self.avg_kw:
                raise ValueError("peak_kw must be >= avg_kw")
        else:
            if not self.timeseries_csv:
                raise ValueError("timeseries_csv is required for timeseries load mode")
        return self


# ---------------------------------------------------------------------------
# 3.2 Outage Model
# ---------------------------------------------------------------------------

class StochasticOutageParams(BaseModel):
    """Parameters for stochastic outage generation via SAIDI/SAIFI."""

    saidi_minutes_per_year: float = Field(
        ..., gt=0, description="System Average Interruption Duration Index (min/yr)"
    )
    saifi_events_per_year: float = Field(
        ..., gt=0, description="System Average Interruption Frequency Index (events/yr)"
    )
    duration_distribution: DurationDistribution = DurationDistribution.LOGNORMAL
    # For lognormal: sigma of log-normal distribution of event durations
    duration_sigma: float = Field(0.8, gt=0, description="Shape parameter for duration distribution")
    # Optional monthly weights [12 values, unnormalized]
    monthly_weights: Optional[list[float]] = Field(
        None,
        description="12 monthly relative weights for outage timing (e.g. monsoon season heavier)"
    )

    @field_validator("monthly_weights")
    @classmethod
    def validate_monthly_weights(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None:
            if len(v) != 12:
                raise ValueError("monthly_weights must have exactly 12 values")
            if any(w < 0 for w in v):
                raise ValueError("monthly_weights values must be non-negative")
        return v


class OutageModel(BaseModel):
    """Grid outage model — reliability driver."""

    mode: OutageMode = OutageMode.STOCHASTIC

    # Deterministic
    events: Optional[list[DeterministicOutageEvent]] = Field(
        None, description="Known outage schedule (deterministic mode)"
    )

    # Stochastic
    stochastic: Optional[StochasticOutageParams] = None

    @model_validator(mode="after")
    def check_mode_params(self) -> "OutageModel":
        if self.mode == OutageMode.DETERMINISTIC:
            if not self.events:
                raise ValueError("events list is required for deterministic outage mode")
        else:
            if self.stochastic is None:
                raise ValueError("stochastic params are required for stochastic outage mode")
        return self


# ---------------------------------------------------------------------------
# 3.3 Grid Tariff Model
# ---------------------------------------------------------------------------

class GridTariffModel(BaseModel):
    """Grid electricity tariff."""

    energy_rate_rs_kwh: float = Field(..., ge=0, description="Energy rate (₹/kWh)")
    fixed_charge_rs_month: float = Field(0.0, ge=0, description="Fixed monthly charge (₹/month)")
    escalation_pct_yr: float = Field(5.0, ge=0, description="Annual tariff escalation (%/yr)")


# ---------------------------------------------------------------------------
# 3.4 DG Backup Model
# ---------------------------------------------------------------------------

class DGPartLoadPoint(BaseModel):
    """A point on the DG fuel consumption part-load curve."""
    load_fraction: float = Field(..., ge=0, le=1)
    fuel_lph: float = Field(..., ge=0, description="Fuel consumption (L/hr) at this load fraction")


class DGBackupModel(BaseModel):
    """Diesel Generator backup model."""

    rated_kw: float = Field(..., gt=0, description="Rated power output (kW)")
    min_loading_pct: float = Field(
        30.0, ge=0, le=100,
        description="Minimum loading to avoid wet-stacking (%)"
    )
    start_delay_seconds: float = Field(
        10.0, ge=0,
        description="Delay from outage detection to DG supplying power (s)"
    )
    ramp_rate_kw_s: float = Field(
        10.0, gt=0, description="Ramp rate (kW/s)"
    )

    # Fuel consumption — simple or part-load curve
    fuel_l_per_kwh: Optional[float] = Field(
        None, gt=0, description="Flat fuel consumption rate (L/kWh)"
    )
    part_load_curve: Optional[list[DGPartLoadPoint]] = Field(
        None, description="Part-load fuel curve (sorted by load_fraction)"
    )

    diesel_price_rs_l: float = Field(..., gt=0, description="Diesel price (₹/L)")
    diesel_escalation_pct_yr: float = Field(5.0, ge=0, description="Diesel price escalation (%/yr)")

    om_rs_kwh: float = Field(1.5, ge=0, description="O&M cost per kWh generated (₹/kWh)")
    om_rs_hour: float = Field(0.0, ge=0, description="O&M cost per running hour (₹/hr)")

    availability_pct: float = Field(
        95.0, gt=0, le=100, description="DG availability (% uptime, excluding planned maint)"
    )
    maintenance_hours_yr: float = Field(
        200.0, ge=0, description="Planned maintenance hours per year"
    )

    capex_rs: float = Field(0.0, ge=0, description="Capital cost of DG (₹)")
    fuel_storage_liters: Optional[float] = Field(
        None, gt=0, description="Available fuel storage (L). Unlimited if None."
    )

    @model_validator(mode="after")
    def check_fuel_model(self) -> "DGBackupModel":
        if self.fuel_l_per_kwh is None and self.part_load_curve is None:
            raise ValueError("Either fuel_l_per_kwh or part_load_curve must be provided")
        return self

    def fuel_consumption_lph(self, load_kw: float) -> float:
        """Return fuel consumption in L/hr for a given load (kW)."""
        if load_kw <= 0:
            return 0.0
        if self.fuel_l_per_kwh is not None:
            return self.fuel_l_per_kwh * load_kw
        # Interpolate part-load curve
        curve = sorted(self.part_load_curve, key=lambda p: p.load_fraction)
        frac = min(load_kw / self.rated_kw, 1.0)
        if frac <= curve[0].load_fraction:
            return curve[0].fuel_lph
        if frac >= curve[-1].load_fraction:
            return curve[-1].fuel_lph
        for i in range(len(curve) - 1):
            if curve[i].load_fraction <= frac <= curve[i + 1].load_fraction:
                t = (frac - curve[i].load_fraction) / (curve[i + 1].load_fraction - curve[i].load_fraction)
                return curve[i].fuel_lph + t * (curve[i + 1].fuel_lph - curve[i].fuel_lph)
        return curve[-1].fuel_lph


# ---------------------------------------------------------------------------
# 3.5 BESS Backup Model
# ---------------------------------------------------------------------------

class BESSBackupModel(BaseModel):
    """Battery Energy Storage System model."""

    capacity_kwh: float = Field(..., gt=0, description="Nominal capacity (kWh)")
    power_kw: float = Field(..., gt=0, description="Max charge/discharge power (kW)")

    dod_pct: float = Field(
        90.0, gt=0, le=100, description="Usable depth of discharge (%)"
    )
    roundtrip_efficiency: float = Field(
        0.92, gt=0, le=1.0, description="Round-trip efficiency (fraction)"
    )
    min_reserve_soc_pct: float = Field(
        10.0, ge=0, lt=100,
        description="Minimum SOC to hold as emergency reserve (%)"
    )
    initial_soc_pct: float = Field(
        90.0, ge=0, le=100, description="Initial SOC at simulation start (%)"
    )

    # Degradation
    calendar_fade_pct_yr: float = Field(
        2.0, ge=0, description="Calendar fade per year (% capacity loss/yr)"
    )
    cycle_fade_per_kwh: float = Field(
        0.000025, ge=0,
        description="Capacity fade per kWh throughput (fraction/kWh)"
    )
    cycle_life_kwh: float = Field(
        0.0, ge=0,
        description="Total lifetime throughput (kWh). 0 = use cycle_fade_per_kwh only."
    )
    calendar_life_yr: float = Field(
        12.0, gt=0, description="Calendar life (years)"
    )
    eol_capacity_pct: float = Field(
        80.0, gt=0, le=100, description="EOL threshold — % of nameplate capacity"
    )

    # Economics
    capex_rs_kwh: float = Field(..., gt=0, description="Capital cost (₹/kWh)")
    replacement_cost_fraction: float = Field(
        0.6, gt=0, le=1.0,
        description="Replacement cost as fraction of original CAPEX"
    )
    om_rs_kwh_yr: float = Field(500.0, ge=0, description="Annual O&M (₹/kWh/yr of capacity)")

    availability_pct: float = Field(
        99.0, gt=0, le=100, description="BESS system availability (%)"
    )

    @property
    def usable_kwh(self) -> float:
        return self.capacity_kwh * self.dod_pct / 100.0

    @property
    def capex_rs(self) -> float:
        return self.capex_rs_kwh * self.capacity_kwh


# ---------------------------------------------------------------------------
# 3.6 Solar Model
# ---------------------------------------------------------------------------

class SolarModel(BaseModel):
    """Solar PV model (optional add-on)."""

    size_kw: float = Field(..., gt=0, description="Installed capacity (kWp)")
    # Simple: flat daily yield
    yield_kwh_kw_day: Optional[float] = Field(
        None, gt=0,
        description="Average daily yield per kWp (kWh/kWp/day). Used if monthly not given."
    )
    # Monthly profile: 12 monthly averages (kWh/kWp/day)
    monthly_yield: Optional[list[float]] = Field(
        None, description="12 monthly average yield values (kWh/kWp/day)"
    )
    # Timeseries CSV
    yield_timeseries_csv: Optional[str] = Field(
        None, description="CSV with columns: hour, yield_kw (actual generation, not specific yield)"
    )

    degradation_pct_yr: float = Field(0.5, ge=0, description="Annual degradation (%/yr)")
    om_rs_kw_yr: float = Field(500.0, ge=0, description="Annual O&M (₹/kW/yr)")
    capex_rs_kw: float = Field(45000.0, gt=0, description="Capital cost (₹/kWp)")

    @model_validator(mode="after")
    def check_yield_source(self) -> "SolarModel":
        sources = [
            self.yield_kwh_kw_day is not None,
            self.monthly_yield is not None,
            self.yield_timeseries_csv is not None,
        ]
        if not any(sources):
            raise ValueError(
                "At least one yield source must be provided: "
                "yield_kwh_kw_day, monthly_yield, or yield_timeseries_csv"
            )
        if self.monthly_yield is not None and len(self.monthly_yield) != 12:
            raise ValueError("monthly_yield must have exactly 12 values")
        return self

    @property
    def capex_rs(self) -> float:
        return self.capex_rs_kw * self.size_kw


# ---------------------------------------------------------------------------
# 3.7 Outage Cost Model
# ---------------------------------------------------------------------------

class TieredPenalty(BaseModel):
    """A tier in the outage cost penalty schedule."""
    threshold_minutes: float = Field(..., ge=0, description="Threshold (minutes). Apply this rate beyond this duration.")
    cost_rs_hr: float = Field(..., ge=0, description="Cost rate (₹/hr) for this tier")


class OutageCostModel(BaseModel):
    """Economics of outages — VoLL or downtime cost."""

    mode: OutageCostMode = OutageCostMode.VOLL

    # VoLL mode
    voll_rs_kwh: Optional[float] = Field(
        None, ge=0, description="Value of Lost Load (₹/kWh unserved)"
    )

    # Downtime mode
    downtime_cost_rs_hr: Optional[float] = Field(
        None, gt=0, description="Downtime cost (₹/hr of critical load unmet)"
    )

    # Optional tiered penalties
    tiered_penalties: Optional[list[TieredPenalty]] = Field(
        None,
        description="Tiered outage cost schedule. Overrides flat rate if provided."
    )

    # SLA targets
    sla_max_downtime_hrs_yr: Optional[float] = Field(
        None, ge=0, description="SLA: max allowable downtime (hrs/yr)"
    )
    sla_max_ens_kwh_yr: Optional[float] = Field(
        None, ge=0, description="SLA: max allowable ENS (kWh/yr)"
    )
    sla_continuity_pct: Optional[float] = Field(
        None, ge=0, le=100, description="SLA: min continuity target (%)"
    )
    sla_min_autonomy_hours: Optional[float] = Field(
        None, ge=0, description="SLA: minimum backup autonomy required (hours)"
    )

    @model_validator(mode="after")
    def check_mode_params(self) -> "OutageCostModel":
        if self.mode == OutageCostMode.VOLL and self.voll_rs_kwh is None:
            raise ValueError("voll_rs_kwh is required for VoLL mode")
        if self.mode == OutageCostMode.DOWNTIME and self.downtime_cost_rs_hr is None:
            raise ValueError("downtime_cost_rs_hr is required for downtime mode")
        return self


# ---------------------------------------------------------------------------
# Top-level Project Config
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Top-level EnergeX project configuration."""

    name: str = Field("EnergeX Project", description="Project name")
    project_lifetime_years: int = Field(25, gt=0, description="Project lifetime (years)")
    discount_rate_pct: float = Field(10.0, gt=0, description="Discount rate for NPV (%)")
    simulation_year: int = Field(2025, description="Base simulation year")

    # Sub-models
    load: LoadProfile
    outage: OutageModel
    grid: GridTariffModel
    outage_cost: OutageCostModel

    # Optional backup assets
    dg: Optional[DGBackupModel] = None
    bess: Optional[BESSBackupModel] = None
    solar: Optional[SolarModel] = None

    # Scenario type (auto-derived if not given)
    scenario: Optional[ScenarioType] = None

    @model_validator(mode="after")
    def derive_scenario(self) -> "ProjectConfig":
        if self.scenario is None:
            has_dg = self.dg is not None
            has_bess = self.bess is not None
            has_solar = self.solar is not None
            if has_dg and has_bess and has_solar:
                self.scenario = ScenarioType.GRID_SOLAR_DG_BESS
            elif has_dg and has_bess:
                self.scenario = ScenarioType.GRID_DG_BESS
            elif has_solar and has_bess:
                self.scenario = ScenarioType.GRID_SOLAR_BESS
            elif has_bess:
                self.scenario = ScenarioType.GRID_BESS
            elif has_dg:
                self.scenario = ScenarioType.GRID_DG
            else:
                self.scenario = ScenarioType.GRID_ONLY
        return self
