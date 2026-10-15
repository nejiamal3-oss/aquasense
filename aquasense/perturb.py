"""Seeded generator of cyber-attacks and sensor faults (AquaSense-ICS).

Part 1 - Attacks (labels: attack_type, attack_id, attack_channel)
  FDI (bias), FDI (stealthy), replay, DoS (frozen / NaN), drift.
  - Intervals A = [t_s, t_s + L), L ~ U[L_MIN, L_MAX], t_s uniform, no overlap.
  - About 10 % of the samples are attacked.
  - sigma_x is computed on the first 70 % of the record (TRAIN_FRAC).
  - Attacks target 8 channels: 4 real channels and 4 row-aligned synthetic
    channels (Cl and TURB at nodes J3 and J7). These synthetic channels are
    generated here (baseline + slow cycle + AR(1) noise) and are NOT the
    EPANET-MSX outputs, which are released separately with the suffix
    "_epanet" after fusion.
  - Stealthy FDI is clipped to [l, u], the 0.5-99.5 % percentiles of the
    training portion (THRESHOLDS can override them per channel).
  - If no attack-free source window exists, a replay falls back to FDI.

Part 2 - Unintentional sensor faults (labels: fault_type, fault_channel)
  - White noise: x + eps * sigma_x, eps ~ N(0, 0.2^2), applied to ALL channels
    during noisy periods of 24-360 samples (fault_channel = "all").
  - Spike: +5 sigma_x on one isolated sample and one random channel.
  - Extreme: replacement by a full-scale limit on one sample and one channel.
  - Faults never overlap an attack or each other.

Usage:
  python aquasense/perturb.py --input data/Real_data.csv \
      --output data/data_full.csv --seed 42 --plot
"""
import argparse
import numpy as np
import pandas as pd

# ================================================================== PARAMETERS
REAL_CHANNELS = ["tds_value", "temperature", "humidity", "gas_resistance"]
SIM_CHANNELS = ["Cl_J3", "TURB_J3", "Cl_J7", "TURB_J7"]   # synthetic channels
ATTACK_TYPES = ["FDI", "FDI-stealth", "replay", "DoS", "drift"]

# ---- Attacks
L_MIN, L_MAX = 24, 360        # attack length (samples)
ATTACK_RATIO = 0.10           # share of attacked samples
TRAIN_FRAC = 0.70             # sigma_x / bounds computed on the first 70 % of rows
BETA_RANGE = (0.5, 2.0)       # beta ~ U{ +-[0.5, 2] }
GAMMA_RANGE = (1.0, 3.0)      # gamma ~ U[1, 3]
SEED = 42

# Optional fixed operating ranges [l, u] per channel. If a channel is absent,
# the 0.5 %-99.5 % percentiles of the training portion are used.
THRESHOLDS = {
    # "humidity": (25, 80),
}

# ---- Faults
NOISE_STD_Z = 0.2          # N(0, 0.2) in z-units
SPIKE_Z = 5.0              # +5 sigma_x
NOISE_RATIO = 0.10         # share of samples in noisy periods
SPIKE_RATIO = 0.005        # share of samples with a spike
EXTREME_RATIO = 0.002      # share of samples with an extreme value
NOISE_L_MIN, NOISE_L_MAX = 24, 360   # length of noisy periods
NOISE_GLOBAL = False       # True: white noise on the whole series

# Full-scale limits [min, max] per channel.
FULL_SCALE = {
    "tds_value": (0.0, 1000.0),            # scaled voltage (no unit)
    "temperature": (-40.0, 85.0),          # degC (BME680 range)
    "humidity": (0.0, 100.0),              # %RH
    "gas_resistance": (0.0, 5.0e5),        # ohm
    "Cl_J3": (0.0, 5.0), "Cl_J7": (0.0, 5.0),            # mg/L
    "TURB_J3": (0.0, 1000.0), "TURB_J7": (0.0, 1000.0),  # NTU
}


# ================================================================ SYNTHETIC CHANNELS
def add_simulated_channels(df, rng):
    """Row-aligned synthetic Cl (mg/L) and TURB (NTU) at J3 and J7:
    baseline + slow cycle + AR(1) noise. Not EPANET-MSX outputs."""
    n = len(df)
    t = np.arange(n)
    params = {"Cl_J3": (0.50, 0.05), "TURB_J3": (0.40, 0.08),
              "Cl_J7": (0.35, 0.05), "TURB_J7": (0.60, 0.10)}
    for name, (base, amp) in params.items():
        noise = np.zeros(n)
        eps = rng.normal(0, amp * 0.2, n)
        for i in range(1, n):
            noise[i] = 0.9 * noise[i - 1] + eps[i]
        phase = rng.uniform(0, 2 * np.pi)
        df[name] = np.clip(base + amp * np.sin(2 * np.pi * t / 288 + phase) + noise, 0, None)
    return df


# ======================================================= NON-OVERLAPPING INTERVALS
def draw_intervals(n, ratio, l_min, l_max, rng):
    """Draw disjoint intervals [t_s, t_s + L) until ratio * n samples are attacked."""
    l_max = min(l_max, max(l_min, n // 20))
    target = int(ratio * n)
    mask = np.zeros(n, dtype=bool)
    intervals = []
    for _ in range(100000):
        if mask.sum() >= target:
            break
        L = int(rng.integers(l_min, l_max + 1))
        L = min(L, max(l_min, target - int(mask.sum())))   # do not exceed the target
        ts = int(rng.integers(1, n - L + 1))          # ts >= 1 (frozen DoS uses x_{ts-1})
        if mask[ts:ts + L].any():
            continue
        mask[ts:ts + L] = True
        intervals.append((ts, L))
    intervals.sort()
    return intervals, mask


# ==================================================================== ATTACKS
def inject(df, channels, train_frac=TRAIN_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(df)
    clean = df[channels].astype(float).copy()
    out = clean.copy()

    n_train = int(train_frac * n)
    sigma = clean.iloc[:n_train].std()
    lo = {c: THRESHOLDS[c][0] if c in THRESHOLDS else clean[c].iloc[:n_train].quantile(0.005) for c in channels}
    hi = {c: THRESHOLDS[c][1] if c in THRESHOLDS else clean[c].iloc[:n_train].quantile(0.995) for c in channels}

    intervals, mask = draw_intervals(n, ATTACK_RATIO, L_MIN, L_MAX, rng)
    cs = np.concatenate([[0], np.cumsum(mask)])       # cumulative sum: "attack-free" test

    attack_type = np.full(n, "none", dtype=object)
    attack_id = np.zeros(n, dtype=int)
    attack_channel = np.full(n, "none", dtype=object)

    for k, (ts, L) in enumerate(intervals, start=1):
        te = ts + L
        ch = channels[rng.integers(len(channels))]
        atype = ATTACK_TYPES[rng.integers(len(ATTACK_TYPES))]
        x = clean[ch].to_numpy()
        seg = slice(ts, te)

        if atype in ("FDI", "FDI-stealth"):
            beta = rng.choice([-1, 1]) * rng.uniform(*BETA_RANGE)
            y = x[seg] + beta * sigma[ch]
            if atype == "FDI-stealth":
                y = np.clip(y, lo[ch], hi[ch])
            out.iloc[seg, out.columns.get_loc(ch)] = y

        elif atype == "replay":
            # source s: attack-free window of length L with Delta = ts - s >= L
            cand = [s for s in range(0, ts - L + 1) if cs[s + L] - cs[s] == 0]
            if not cand:                                # no clean window -> fall back to FDI
                atype = "FDI"
                beta = rng.choice([-1, 1]) * rng.uniform(*BETA_RANGE)
                out.iloc[seg, out.columns.get_loc(ch)] = x[seg] + beta * sigma[ch]
            else:
                s = cand[rng.integers(len(cand))]
                out.iloc[seg, out.columns.get_loc(ch)] = x[s:s + L]

        elif atype == "DoS":
            if rng.random() < 0.5:                      # frozen: x_{ts-1}
                out.iloc[seg, out.columns.get_loc(ch)] = x[ts - 1]
            else:                                       # dropped: NaN
                out.iloc[seg, out.columns.get_loc(ch)] = np.nan

        elif atype == "drift":
            gamma = rng.uniform(*GAMMA_RANGE)
            ramp = gamma * sigma[ch] * (np.arange(ts, te) - ts) / L
            out.iloc[seg, out.columns.get_loc(ch)] = x[seg] + ramp

        attack_type[seg] = atype
        attack_id[seg] = k
        attack_channel[seg] = ch

    res = df.copy()
    res[channels] = out
    res["attack_type"] = attack_type
    res["attack_id"] = attack_id
    res["attack_channel"] = attack_channel
    return res


# ===================================================================== FAULTS
def inject_faults(att_df, clean_df, channels, train_frac=TRAIN_FRAC, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    n = len(att_df)
    res = att_df.copy()
    n_train = int(train_frac * n)
    sigma = clean_df[channels].iloc[:n_train].std()      # sigma_x on the training portion

    fault_type = np.full(n, "none", dtype=object)
    fault_channel = np.full(n, "none", dtype=object)
    occupied = (att_df["attack_type"] != "none").to_numpy().copy()   # never on an attack

    # ---------- 1) White noise: noisy periods, all channels, in z-units
    if NOISE_GLOBAL:
        free = ~occupied
        for c in channels:
            res.loc[free, c] = res.loc[free, c] + rng.normal(0, NOISE_STD_Z, free.sum()) * sigma[c]
        fault_type[free] = "white_noise"
        fault_channel[free] = "all"
        occupied |= free
    else:
        target = int(NOISE_RATIO * n)
        done = 0
        for _ in range(100000):
            if done >= target:
                break
            L = int(rng.integers(NOISE_L_MIN, min(NOISE_L_MAX, max(NOISE_L_MIN, n // 20)) + 1))
            L = min(L, max(NOISE_L_MIN, target - done))
            ts = int(rng.integers(0, n - L + 1))
            if occupied[ts:ts + L].any():
                continue
            for c in channels:
                res.iloc[ts:ts + L, res.columns.get_loc(c)] += rng.normal(0, NOISE_STD_Z, L) * sigma[c]
            fault_type[ts:ts + L] = "white_noise"
            fault_channel[ts:ts + L] = "all"
            occupied[ts:ts + L] = True
            done += L

    # ---------- 2) Spikes: +5 sigma_x, isolated sample, random channel
    free_idx = np.flatnonzero(~occupied)
    n_spk = min(int(SPIKE_RATIO * n), len(free_idx))
    spk = rng.choice(free_idx, n_spk, replace=False)
    for t in spk:
        c = channels[rng.integers(len(channels))]
        res.iloc[t, res.columns.get_loc(c)] += SPIKE_Z * sigma[c]
        fault_type[t], fault_channel[t] = "spike", c
        occupied[t] = True

    # ---------- 3) Extreme values: full-scale limit (min or max) of the sensor
    free_idx = np.flatnonzero(~occupied)
    n_ext = min(int(EXTREME_RATIO * n), len(free_idx))
    ext = rng.choice(free_idx, n_ext, replace=False)
    for t in ext:
        c = channels[rng.integers(len(channels))]
        res.iloc[t, res.columns.get_loc(c)] = FULL_SCALE[c][rng.integers(2)]
        fault_type[t], fault_channel[t] = "extreme", c
        occupied[t] = True

    res["fault_type"] = fault_type
    res["fault_channel"] = fault_channel
    return res


# ====================================================================== PLOTS
def plot_attacks(clean_df, att_df, channels, path="attacks.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.2 * len(channels)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), channels):
        ax.plot(clean_df[c].values, color="blue", lw=0.8, label="clean " + c)
        m = att_df["attack_channel"] == c
        y = att_df[c].where(m)
        ax.plot(y.values, color="red", lw=1.0, label="attacked")
        ax.set_ylabel(c); ax.grid(True); ax.legend(loc="upper right")
    plt.xlabel("Index / Time"); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_full(clean_df, full_df, channels, path="faults.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.2 * len(channels)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), channels):
        ax.plot(clean_df[c].values, color="blue", lw=0.7)
        for typ, col in [("white_noise", "green"), ("spike", "orange"), ("extreme", "purple")]:
            m = (full_df.fault_type == typ) & ((full_df.fault_channel == c) | (full_df.fault_channel == "all"))
            ax.plot(np.where(m, full_df[c], np.nan), ".", color=col, ms=2, label=typ)
        m = full_df.attack_channel == c
        ax.plot(np.where(m, full_df[c], np.nan), color="red", lw=1, label="attack")
        ax.set_ylabel(c); ax.grid(True)
    np.atleast_1d(axes)[0].legend(ncol=4, loc="upper right")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# ======================================================================= MAIN
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Real_data.csv")
    ap.add_argument("--output", default="data/data_full.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--plot", action="store_true")
    # parse_known_args: ignore Colab/Jupyter internal arguments (e.g. -f kernel.json)
    a, _ = ap.parse_known_args()

    df = pd.read_csv(a.input)
    missing = [c for c in REAL_CHANNELS if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns missing from {a.input}: {missing}\nAvailable: {list(df.columns)}")

    # Chronological order is required. Sorting is applied only if needed, so
    # an already-sorted file is left untouched.
    if "timestamp" in df.columns:
        ts_parsed = pd.to_datetime(df["timestamp"])
        if not ts_parsed.is_monotonic_increasing:
            print("Warning: input not sorted by timestamp; sorting.")
            df = df.assign(timestamp=ts_parsed).sort_values("timestamp").reset_index(drop=True)

    df = add_simulated_channels(df, np.random.default_rng(a.seed))
    channels = REAL_CHANNELS + SIM_CHANNELS
    clean = df.copy()

    att = inject(df, channels, seed=a.seed)                      # 1) attacks (attack_*)
    full = inject_faults(att, clean, channels, seed=a.seed)      # 2) faults  (fault_*)
    full.to_csv(a.output, index=False)

    print(f"Saved: {a.output} ({len(full)} rows)")
    print("Attacks:", f"{(full.attack_type != 'none').mean():.1%}")
    print("Faults :", f"{(full.fault_type != 'none').mean():.1%}")
    print(full.attack_type.value_counts().to_string())
    print(full.fault_type.value_counts().to_string())
    assert not ((full.attack_type != "none") & (full.fault_type != "none")).any()

    if a.plot:
        plot_attacks(clean, att, channels)
        plot_full(clean, full, channels)
