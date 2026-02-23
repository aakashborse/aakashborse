"""
EnergeX — CSV Parsers

Loads timeseries load and solar yield data from CSV files.

Expected CSV format for load timeseries:
    hour,total_kw,critical_kw
    0,100.5,30.2
    1,95.0,28.5
    ...
    8759,...,...

Expected CSV format for solar timeseries:
    hour,yield_kw
    0,0.0
    ...
    8759,...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_load_timeseries(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load load timeseries CSV.

    Returns
    -------
    (total_kw, critical_kw) arrays of length 8760.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Load timeseries CSV not found: {p}")

    import csv
    total_kw = np.zeros(8760)
    critical_kw = np.zeros(8760)

    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = int(float(row.get("hour", 0)))
            if 0 <= h < 8760:
                total_kw[h] = float(row.get("total_kw", 0))
                critical_kw[h] = float(row.get("critical_kw", 0))

    return total_kw, critical_kw


def load_solar_timeseries(csv_path: str | Path) -> np.ndarray:
    """
    Load solar generation timeseries CSV.

    Returns
    -------
    yield_kw array of length 8760.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Solar timeseries CSV not found: {p}")

    import csv
    yield_kw = np.zeros(8760)

    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = int(float(row.get("hour", 0)))
            if 0 <= h < 8760:
                yield_kw[h] = float(row.get("yield_kw", 0))

    return yield_kw


def generate_sample_load_csv(
    output_path: str | Path,
    avg_kw: float = 100.0,
    critical_fraction: float = 0.3,
) -> None:
    """Generate a sample load CSV for testing."""
    import csv
    import math

    hours = list(range(8760))
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "total_kw", "critical_kw"])
        for h in hours:
            hod = h % 24
            shape = 1.0 + 0.3 * math.sin(2 * math.pi * (hod - 3) / 24)
            total = round(avg_kw * shape, 2)
            critical = round(total * critical_fraction, 2)
            writer.writerow([h, total, critical])


def generate_sample_solar_csv(
    output_path: str | Path,
    size_kw: float = 100.0,
    yield_kwh_kw_day: float = 5.0,
) -> None:
    """Generate a sample solar generation CSV for testing."""
    import csv
    import math

    daily_kwh = yield_kwh_kw_day * size_kw
    hourly_avg = daily_kwh / 24.0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "yield_kw"])
        for h in range(8760):
            hod = h % 24
            if 6 <= hod <= 18:
                factor = math.sin(math.pi * (hod - 6) / 12)
                gen = hourly_avg * factor * (24.0 / 8.0)
            else:
                gen = 0.0
            writer.writerow([h, round(gen, 3)])
