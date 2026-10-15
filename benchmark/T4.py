"""
AquaSense-ICS - Benchmark T4 (utilite des donnees synthetiques)
Usage:
    python aquasense_t4_v1.py --csv fused_dataset.csv --seeds 5
Options:  --dl   active LSTM (necessite tensorflow)
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

CH = ["temperature", "humidity", "gas_resistance", "tds_value",
      "Cl_J7", "Cl_J3", "TURB_J3", "TURB_J7"]          # 8 canaux observes (article)
SENSOR4 = ["tds_value", "temperature", "humidity", "gas_resistance"]
W, STRIDE = 30, 5
TARGET_WITH_FAULTS = False
TR, VA = 0.6, 0.2                                       # 60/20/20 chronologique


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path)
    for c in ["attack_type", "fault_type"]:
        df[c] = df[c].fillna("none").astype(str)
    if "origin" not in df.columns:
        df["origin"] = "real"
    return df


def get_real(df):
    real = df[df.origin == "real"].copy()
    real["timestamp"] = pd.to_datetime(real["timestamp"])
    return real.sort_values("timestamp").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Pretraitement et fenetrage
# ----------------------------------------------------------------------------
def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def zscore_fit(real, n_train):
    X = real[CH].astype(float).copy()
    X["gas_resistance"] = slog(X["gas_resistance"])
    mu, sd = X.iloc[:n_train].mean(), X.iloc[:n_train].std().replace(0, 1)   # fit sur TRAIN seulement
    return mu, sd


def zscore_apply(df, mu, sd):
    X = df[CH].astype(float).copy()
    X["gas_resistance"] = slog(X["gas_resistance"])
    return ((X - mu) / sd).values


def make_windows(Z, lab, atype, aid, ts, lo, hi, w=W, stride=STRIDE):
    """Fenetres construites a l'interieur de [lo,hi) -> aucune fenetre a cheval sur deux splits.
    Label = dernier echantillon."""
    Xs, ys, at, ai, te = [], [], [], [], []
    for end in range(lo + w - 1, hi, stride):
        Xs.append(Z[end - w + 1:end + 1]); ys.append(lab[end]); at.append(atype[end])
        ai.append(aid[end]); te.append(ts[end])
    return dict(X=np.array(Xs), y=np.array(ys), atype=np.array(at), aid=np.array(ai), ts=np.array(te))


def build_splits(real):
    n = len(real); i1, i2 = int(n * TR), int(n * (TR + VA))
    mu, sd = zscore_fit(real, i1)
    Z = zscore_apply(real, mu, sd)
    atype = real["attack_type"].values
    ftype = real["fault_type"].values
    # cible T3 de l'article = attack_type ; les fautes (fault_type) sont du bruit => "normal"
    # (mettre TARGET_WITH_FAULTS=True pour classer aussi les fautes)
    lab = (np.where(atype != "none", atype, np.where(ftype != "none", ftype, "normal"))
           if TARGET_WITH_FAULTS else np.where(atype != "none", atype, "normal"))
    aid = real["attack_id"].astype(str).values if "attack_id" in real else np.array(["0"] * n)
    ts = real["timestamp"].values
    sp = {k: make_windows(Z, lab, atype, aid, ts, a, b)
          for k, (a, b) in dict(train=(0, i1), val=(i1, i2), test=(i2, n)).items()}
    print(f"Fenetres: train={len(sp['train']['y'])} val={len(sp['val']['y'])} test={len(sp['test']['y'])}")
    for k in sp:
        print(f"  {k:5s}", dict(zip(*np.unique(sp[k]['y'], return_counts=True))))
    seg_start = (real[real.attack_type != "none"].groupby(real["attack_id"].astype(str))["timestamp"].min()
                 .to_dict()) if "attack_id" in real else {}
    return sp, (mu, sd), seg_start


def feats(Wn):
    """mean, std, min, max, slope, last par canal (+ fraction de NaN)."""
    nanf = np.isnan(Wn).mean(1)
    m, s = np.nanmean(Wn, 1), np.nanstd(Wn, 1)
    mn, mx = np.nanmin(Wn, 1), np.nanmax(Wn, 1)
    Wf = np.where(np.isnan(Wn), np.nanmean(Wn, 1, keepdims=True), Wn)
    Wf = np.nan_to_num(Wf)
    t = np.arange(Wn.shape[1]) - (Wn.shape[1] - 1) / 2
    slope = (Wf * t[None, :, None]).sum(1) / (t ** 2).sum()
    last = np.nan_to_num(Wn[:, -1])
    return np.nan_to_num(np.hstack([m, s, mn, mx, slope, last, nanf]))


# ----------------------------------------------------------------------------
# Modeles
# ----------------------------------------------------------------------------
def tabular(name, seed):
    if name == "RF":
        return RandomForestClassifier(200, random_state=seed, n_jobs=-1, class_weight="balanced")
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed,
                             eval_metric="mlogloss", n_jobs=-1)
    raise ValueError(name)


def fit_predict_tab(name, Wtr, ytr, Wte, seed):
    le = LabelEncoder().fit(ytr)
    clf = tabular(name, seed).fit(feats(Wtr), le.transform(ytr))
    return le.inverse_transform(clf.predict(feats(Wte)))


def build_dl(kind, c, ncls):
    import tensorflow as tf
    L = tf.keras.layers
    m = tf.keras.Sequential([L.Input((W, c))])
    if kind == "CNN":
        m.add(L.Conv1D(32, 5, activation="relu")); m.add(L.MaxPooling1D(2))
        m.add(L.Conv1D(64, 3, activation="relu")); m.add(L.GlobalAveragePooling1D())
    elif kind == "LSTM":
        m.add(L.LSTM(64))
    m.add(L.Dense(32, activation="relu")); m.add(L.Dense(ncls, activation="softmax"))
    m.compile("adam", "sparse_categorical_crossentropy")
    return m


def fit_predict_dl(kind, Wtr, ytr, Wva, yva, Wte, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    le = LabelEncoder().fit(ytr)
    ok = np.isin(yva, le.classes_)
    cw = {i: len(ytr) / (len(le.classes_) * (ytr == c).sum()) for i, c in enumerate(le.classes_)}
    m = build_dl(kind, Wtr.shape[2], len(le.classes_))
    m.fit(np.nan_to_num(Wtr), le.transform(ytr), epochs=30, batch_size=64, verbose=0, class_weight=cw,
          validation_data=(np.nan_to_num(Wva[ok]), le.transform(yva[ok])) if ok.any() else None,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)] if ok.any() else None)
    return le.inverse_transform(m.predict(np.nan_to_num(Wte), verbose=0).argmax(1))


# ----------------------------------------------------------------------------
# T4 : utilite des donnees synthetiques (TSTR)
# ----------------------------------------------------------------------------
def augment(Wn, rng):
    """Jittering (sigma=0.05 en unites z) et scaling [0.9,1.1], labels conserves (training only)."""
    j = Wn + rng.normal(0, 0.05, Wn.shape)
    s = Wn * rng.uniform(0.9, 1.1, (Wn.shape[0], 1, Wn.shape[2]))
    return np.concatenate([j, s])


def timegan_windows(df, mu, sd):
    g = df[df.origin == "timegan"]
    if len(g) < W:
        return None
    Z = np.zeros((len(g), len(CH)))                 # canaux absents -> 0 (= moyenne train apres z-score)
    full = pd.DataFrame({c: g[c].values if c in g and g[c].notna().any() else np.nan for c in CH})
    z = zscore_apply(full, mu, sd)
    Z = np.where(np.isnan(z), 0.0, z)
    X = np.array([Z[i:i + W] for i in range(0, len(Z) - W + 1, W)])
    return X, np.array(["normal"] * len(X))


def run_T4(df, real, seeds, dl):
    print("\n=== T4 : utilite des donnees synthetiques (train synth / test reel) ===")
    sp, (mu, sd), _ = build_splits(real)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    tg = timegan_windows(df, mu, sd)
    models = ["RF"] + (["LSTM"] if dl and _has("tensorflow") else [])
    base = {}
    print(f"{'Entrainement':26s}" + "".join(f"{m:>18s}" for m in models))
    for setup in ["Real only", "Real + augmentation", "Real + TimeGAN", "TimeGAN only (TSTR)"]:
        line = f"{setup:26s}"
        for mname in models:
            f1s = []
            for s in range(seeds):
                rng = np.random.default_rng(42 + s)
                Xtr, ytr = tr["X"], tr["y"]
                if setup == "Real + augmentation":
                    Xtr = np.concatenate([Xtr, augment(tr["X"], rng)]); ytr = np.concatenate([ytr, ytr, ytr])
                elif setup == "Real + TimeGAN" and tg is not None:
                    Xtr = np.concatenate([Xtr, tg[0]]); ytr = np.concatenate([ytr, tg[1]])
                elif setup == "TimeGAN only (TSTR)":
                    if tg is None or len(np.unique(tg[1])) < 2:
                        f1s = None; break               # une seule classe -> n/a (cf. article)
                    Xtr, ytr = tg
                p = (fit_predict_tab("RF", Xtr, ytr, te["X"], 42 + s) if mname == "RF"
                     else fit_predict_dl("LSTM", Xtr, ytr, va["X"], va["y"], te["X"], 42 + s))
                f1s.append(f1_score(te["y"], p, labels=np.unique(te["y"]), average="macro", zero_division=0))
            if f1s is None:
                line += f"{'n/a':>18s}"; continue
            mu_f, sd_f = np.mean(f1s), np.std(f1s)
            if setup == "Real only": base[mname] = mu_f
            line += f"{mu_f:9.3f}+-{sd_f:.3f}   "
            if setup != "Real only" and mname in base:
                line = line.rstrip() + f"(d={mu_f - base[mname]:+.3f})"
        print(line)
    if len(np.unique(te["y"])) <= 2:
        print("Rappel: le test ne contient que (normal, replay) -> comparer au predicteur constant.")


def _has(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def main(csv="fused_data.csv", seeds=5, dl=False):
    df = pd.read_csv(csv)                              # lecture directe avec pd.read_csv
    for c in ["attack_type", "fault_type"]:
        df[c] = df[c].fillna("none").astype(str)
    if "origin" not in df.columns:
        df["origin"] = "real"
    real = get_real(df)
    print(f"Lignes: {len(df)} (reel: {len(real)})")
    run_T4(df, real, seeds, dl)


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules or any(a == "-f" for a in sys.argv):
        # Colab / Jupyter : modifier ici
        main(csv="fused_data.csv", seeds=5, dl=False)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--csv", default="fused_data.csv")
        ap.add_argument("--seeds", type=int, default=5)
        ap.add_argument("--dl", action="store_true")
        a = ap.parse_args()
        main(a.csv, a.seeds, a.dl)
