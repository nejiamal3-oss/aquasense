"""Co-registration of the real (perturbed) record and the EPANET-MSX output,
water-quality labels, and TimeGAN rows (AquaSense-ICS).

  1. tau_real = elapsed seconds since the first timestamp
  2. tau_sim  = round(3600 * time_h), offset Delta_0 = 0
  3. backward as-of join, tolerance 3,600 s
  4. simulated columns whose name already exists are renamed *_epanet;
     sim_state_h records the hour of the matched simulated state
  5. wq_score / wq_class from J3 (pH, turbidity, chlorine, THM, conductivity)
     and from the RAW tds_value (before fault injection)
  6. TimeGAN rows appended with origin = timegan

Usage:
  python aquasense/fuse.py --raw data/Real_data.csv --perturbed data/data_full.csv \
      --sim data/simulated_data.csv --synthetic data/synthetic_data.csv \
      --out data/fused_dataset.csv
"""
import argparse
import numpy as np
import pandas as pd

SENSORS = ["tds_value", "temperature", "humidity", "gas_resistance"]
DROP = ["Time", "Hour", "Minute", "time_h", "tau_s", "tau_real",
        "quality_score", "tds_raw"]
AUG_SUFFIXES = ("_jitter", "_scaled", "_warped", "_sliced", "_permuted")
INF = np.inf

# (optimal range, acceptable range), inclusive bounds (Table quality_thresholds)
RULES = {
    "ph":   ((6.5, 8.5), (6.0, 9.0)),
    "turb": ((-INF, 1.0), (-INF, 5.0)),
    "cl":   ((0.2, 0.5), (0.1, 5.0)),
    "thm":  ((-INF, 60.0), (-INF, 100.0)),
    "cond": ((-INF, 800.0), (-INF, 2500.0)),
    "tds":  ((-INF, 300.0), (-INF, 600.0)),
}


def find_col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"None of {names} found in the fused table")


def points(x, opt, acc):
    s = np.zeros(len(x))
    s[(x >= acc[0]) & (x <= acc[1])] = 1
    s[(x >= opt[0]) & (x <= opt[1])] = 2
    return s


def add_wq_labels(df):
    v = {
        "ph": df[find_col(df, "PH_J3", "pH_J3")],
        "turb": df[find_col(df, "TURB_J3_epanet")],
        "cl": df[find_col(df, "Cl_J3_epanet")],
        "thm": df[find_col(df, "THM_J3")],
        "cond": df[find_col(df, "COND_J3")],
        "tds": df["tds_raw"],
    }
    v = {k: np.round(s.to_numpy(float), 6) for k, s in v.items()}   # float noise
    valid = np.all([~np.isnan(s) for s in v.values()], axis=0)
    score = sum(points(v[k], *RULES[k]) for k in RULES)
    critical = ((v["cl"] < 0.1) | (v["cl"] > 5.0) |
                (v["turb"] > 5.0) | (v["thm"] > 100.0))
    cls = np.where(critical | (score < 8), "Poor",
                   np.where(score >= 11, "Very Good", "Good")).astype(object)
    score = score.astype(float)
    score[~valid] = np.nan
    cls[~valid] = np.nan
    df["wq_score"], df["wq_class"] = score, cls
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/Real_data.csv")
    ap.add_argument("--perturbed", default="data/data_full.csv")
    ap.add_argument("--sim", default="data/simulated_data.csv")
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--out", default="data/fused_dataset.csv")
    ap.add_argument("--tolerance", type=int, default=3600)
    a, _ = ap.parse_known_args()

    # ---- real (perturbed) table + raw TDS, aligned row by row
    real = pd.read_csv(a.perturbed)
    raw = pd.read_csv(a.raw)
    for d in (real, raw):
        d["timestamp"] = pd.to_datetime(d["timestamp"])
    real = real.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    assert len(real) == len(raw) and (real["timestamp"] == raw["timestamp"]).all(), \
        "perturbed and raw files are not aligned row by row"
    real["tds_raw"] = raw["tds_value"].to_numpy()
    real["tau_s"] = (real["timestamp"] - real["timestamp"].iloc[0]).dt.total_seconds().round().astype("int64")

    # ---- simulation (wide format, one state per hour)
    sim = pd.read_csv(a.sim).dropna(subset=["time_h"])
    sim["tau_s"] = np.round(3600 * sim["time_h"]).astype("int64")      # Delta_0 = 0
    sim = sim.rename(columns={"time_h": "sim_state_h"})
    existing = set(real.columns)
    sim = sim.rename(columns={c: c + "_epanet" for c in sim.columns
                              if c not in ("tau_s", "sim_state_h") and c in existing})
    sim = sim.sort_values("tau_s")

    fused = pd.merge_asof(real, sim, on="tau_s", direction="backward",
                          tolerance=a.tolerance)
    fused["origin"] = "real"

    matched = int(fused["sim_state_h"].notna().sum())
    print(f"real rows: {len(fused)} | with a simulated state: {matched} "
          f"({100 * matched / len(fused):.1f} %)   [article: 20,795 = 87.8 %]")
    print(f"distinct hourly states matched: {fused['sim_state_h'].nunique()}   [article: 11]")
    print(f"largest share of one state: {fused['sim_state_h'].value_counts().max()}   [article: 2,816]")

    fused = add_wq_labels(fused)
    print(fused["wq_class"].value_counts(dropna=False).to_string())

    # ---- legacy label, cleaning
    if "quality_final" not in fused.columns:
        for alt in ("qualite", "quality"):
            if alt in fused.columns:
                fused = fused.rename(columns={alt: "quality_final"})
                print(f"legacy column '{alt}' renamed quality_final")
                break
    drop = [c for c in fused.columns if c in DROP or
            (c.endswith(AUG_SUFFIXES) and c.rsplit("_", 1)[0] in SENSORS)]
    fused = fused.drop(columns=drop)

    # ---- TimeGAN rows (training only)
    if a.synthetic:
        syn = pd.read_csv(a.synthetic)[SENSORS]
        syn["origin"] = "timegan"
        syn["attack_type"] = "none"
        syn["fault_type"] = "none"
        fused = pd.concat([fused, syn], ignore_index=True)

    if "quality_final" not in fused.columns:
        fused["quality_final"] = np.nan
    cols = [c for c in fused.columns if c != "quality_final"] + ["quality_final"]
    fused = fused[cols]

    fused.to_csv(a.out, index=False)
    print(f"fused: {fused.shape[0]} rows, {fused.shape[1]} columns   "
          f"[article: 47,376 rows, 97 columns]")


if __name__ == "__main__":
    main()
