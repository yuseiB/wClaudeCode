#!/usr/bin/env python3
"""Report sensitive dependence on initial conditions for double pendulum CSV outputs.

Usage:
  python cpp/examples/sensitivity_report.py sensitivity_a.csv sensitivity_b.csv
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass


@dataclass
class Row:
    t: float
    theta1: float
    theta2: float
    x2: float
    y2: float


def load_rows(path: str) -> list[Row]:
    out: list[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(
                Row(
                    t=float(r["t"]),
                    theta1=float(r["theta1"]),
                    theta2=float(r["theta2"]),
                    x2=float(r["x2"]),
                    y2=float(r["y2"]),
                )
            )
    return out


def nearest_at(rows: list[Row], target_t: float) -> Row:
    return min(rows, key=lambda r: abs(r.t - target_t))


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python cpp/examples/sensitivity_report.py sensitivity_a.csv sensitivity_b.csv")
        return 2

    a = load_rows(sys.argv[1])
    b = load_rows(sys.argv[2])
    if len(a) != len(b):
        print(f"Row count mismatch: {len(a)} vs {len(b)}")
        return 1

    print("Sensitive dependence report (tiny initial perturbation in theta1)")
    print("time[s]  |Δtheta1|[deg]  |Δtheta2|[deg]  bob2 distance[m]")
    print("-" * 61)

    checkpoints = [0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 25]
    for t in checkpoints:
        ra = nearest_at(a, t)
        rb = nearest_at(b, t)
        dth1 = abs((ra.theta1 - rb.theta1) * 180.0 / math.pi)
        dth2 = abs((ra.theta2 - rb.theta2) * 180.0 / math.pi)
        dist = math.hypot(ra.x2 - rb.x2, ra.y2 - rb.y2)
        print(f"{ra.t:6.2f}   {dth1:12.6f}   {dth2:12.6f}   {dist:15.6f}")

    thresholds = [0.01, 0.1, 0.5, 1.0]
    print("\nFirst time bob-2 separation exceeds threshold:")
    for th in thresholds:
        crossing = next((r for r, s in zip(a, b) if math.hypot(r.x2 - s.x2, r.y2 - s.y2) > th), None)
        if crossing is None:
            print(f"  > {th:.2f} m : never")
        else:
            print(f"  > {th:.2f} m : t ≈ {crossing.t:.3f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
