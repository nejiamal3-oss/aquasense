"""Time-series augmentation (AquaSense-ICS).

Five techniques applied to the four sensor channels:
  jitter    : Gaussian noise, sigma = 0.05 (raw units of each channel)
  scaled    : random gain in [0.9, 1.1] (one factor per channel)
  warped    : time warping through a smooth curve with 4 random knots
  sliced    : random window of 90 % of the series, NaN elsewhere
  permuted  : series cut into 4 segments, segments shuffled

Output columns: <channel>_jitter, _scaled, _warped, _sliced, _permuted.
The augmented columns must be used on TRAINING data only.

Usage:
  python aquasense/augment.py --input data/Real_data.csv \
      --output data/augmented_data.csv --seed 42
"""
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

CHANNELS = ["tds_value", "temperature", "humidity", "gas_resistance"]
JITTER_SIGMA = 0.05
SCALE_RANGE = (0.9, 1.1)
N_KNOTS = 4
WARP_SIGMA = 0.2          # std of the warping speed at the knots
SLICE_RATIO = 0.9
N_SEGMENTS = 4


def jitter(x, rng):
    return x + rng.normal(0.0, JITTER_SIGMA, len(x))


def scaling(x, rng):
    return x * rng.uniform(*SCALE_RANGE)


def time_warp(x, rng):
    n = len(x)
    steps = np.arange(n)
    knots = np.linspace(0, n - 1, N_KNOTS + 2)
    speed = CubicSpline(knots, rng.normal(1.0, WARP_SIGMA, N_KNOTS + 2))(steps)
    speed = np.clip(speed, 0.1, None)
    warped_t = np.cumsum(speed)
    warped_t = (warped_t - warped_t[0]) / (warped_t[-1] - warped_t[0]) * (n - 1)
    return np.interp(steps, warped_t, x)


def window_slice(x, rng):
    n = len(x)
    w = int(SLICE_RATIO * n)
    s = int(rng.integers(0, n - w + 1))
    out = np.full(n, np.nan)
    out[s:s + w] = x[s:s + w]
    return out


def permutation(x, rng):
    parts = np.array_split(x, N_SEGMENTS)
    order = rng.permutation(N_SEGMENTS)
    return np.concatenate([parts[i] for i in order])


TECHNIQUES = {"jitter": jitter, "scaled": scaling, "warped": time_warp,
              "sliced": window_slice, "permuted": permutation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Real_data.csv")
    ap.add_argument("--output", default="data/augmented_data.csv")
    ap.add_argument("--seed", type=int, default=42)
    a, _ = ap.parse_known_args()

    df = pd.read_csv(a.input)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    rng = np.random.default_rng(a.seed)

    for ch in CHANNELS:
        x = df[ch].to_numpy(float)
        for name, fn in TECHNIQUES.items():
            df[f"{ch}_{name}"] = fn(x, rng)

    df.to_csv(a.output, index=False)
    print(f"Saved: {a.output} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
