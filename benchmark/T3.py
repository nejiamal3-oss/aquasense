"""
AquaSense-ICS - Benchmark T3 (attaques/fautes)
Usage:
    python aquasense_t3_v1.py --csv fused_dataset.csv --seeds 5
Options:  --dl   active 1D-CNN / LSTM (necessite tensorflow)
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, recall_score, roc_auc_score

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
# Metriques T3
# ----------------------------------------------------------------------------
def t3_metrics(te, pred, seg_start):
    y = te["y"]
    mf1 = f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)  # classes presentes
    alarm = pred != "normal"
    far = 100 * alarm[y == "normal"].mean() if (y == "normal").any() else np.nan
    rec = {c: recall_score(y == c, pred == c) for c in np.unique(y) if c != "normal"}
    delays = []
    for i in np.unique(te["aid"][te["atype"] != "none"]):
        idx = np.where((te["aid"] == i) & alarm)[0]
        if len(idx) and i in seg_start:
            d = (te["ts"][idx[0]] - np.datetime64(seg_start[i])) / np.timedelta64(1, "s")
            delays.append(max(d, 0))
    return dict(macroF1=mf1, FAR=far, delay=np.mean(delays) if delays else np.nan, recall=rec)


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return np.nanmean(v) if not np.all(np.isnan(v)) else np.nan, np.nanstd(v) if not np.all(np.isnan(v)) else np.nan


# ----------------------------------------------------------------------------
# T3
# ----------------------------------------------------------------------------
def run_T3(real, seeds, dl):
    print("\n=== T3 : classification attaques/fautes (split chronologique) ===")
    sp, _, seg_start = build_splits(real)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    const = np.array(["normal"] * len(te["y"]))
    m = t3_metrics(te, const, seg_start)
    print(f"Constant 'normal'      macroF1={m['macroF1']:.3f}  FAR=0.00")

    names = ["LR", "RF"] + (["XGB"] if _has("xgboost") else [])
    results = {}
    for nm in names:
        rows = [t3_metrics(te, fit_predict_tab(nm, tr["X"], tr["y"], te["X"], 42 + s), seg_start)
                for s in range(seeds)]
        results[nm] = rows
    if dl and _has("tensorflow"):
        for kind in ["CNN", "LSTM"]:
            results[kind] = [t3_metrics(te, fit_predict_dl(kind, tr["X"], tr["y"], va["X"], va["y"], te["X"], s), seg_start)
                             for s in range(seeds)]
    # modeles non supervises : entraines sur fenetres normales du train
    Ftr_norm = feats(tr["X"][tr["atype"] == "none"])
    sc = StandardScaler().fit(Ftr_norm)
    for nm, mk in [("IsolationForest", lambda s: IsolationForest(n_estimators=200, contamination=0.05, random_state=s)),
                   ("OneClassSVM", lambda s: OneClassSVM(nu=0.05, gamma="scale"))]:
        rows = []
        for s in range(seeds):
            mod = mk(42 + s).fit(sc.transform(Ftr_norm))
            pred = np.where(mod.predict(sc.transform(feats(te["X"]))) == -1, "attack", "normal")
            r = t3_metrics(te, pred, seg_start); r["macroF1"] = np.nan; rows.append(r)
        results[nm] = rows

    print(f"{'Modele':16s} {'macroF1':>16s} {'Delay(s)':>14s} {'FAR(%)':>16s}")
    for nm, rows in results.items():
        f, d, a = agg(rows, "macroF1"), agg(rows, "delay"), agg(rows, "FAR")
        print(f"{nm:16s} {f[0]:7.3f}+-{f[1]:.3f} {d[0]:8.1f}+-{d[1]:.1f} {a[0]:8.2f}+-{a[1]:.2f}")
    print("Recall par classe (dernier seed, RF):",
          {k: round(v, 3) for k, v in results["RF"][-1]["recall"].items()})

    # ---------- LOAO : detection binaire d'un type d'attaque non vu ----------
    print("\n-- LOAO (AUROC, detection d'un type d'attaque non vu a l'entrainement) --")
    allw = {k: np.concatenate([sp[s][k] for s in sp]) for k in ["X", "y", "atype"]}
    vt_norm = np.concatenate([sp["val"]["y"], sp["test"]["y"]]) == "normal"
    Xvt = np.concatenate([sp["val"]["X"], sp["test"]["X"]])[vt_norm]
    types = [t for t in np.unique(allw["atype"]) if t != "none"]
    for k in types:
        pos = allw["X"][allw["atype"] == k]
        keep = tr["atype"] != k                       # type k retire du train
        Xk, yk = tr["X"][keep], (tr["atype"][keep] != "none").astype(int)
        if yk.sum() == 0 or len(pos) == 0 or len(Xvt) == 0:
            print(f"{k:12s} non evaluable (pas assez de donnees)"); continue
        Fe = feats(np.concatenate([pos, Xvt])); ye = np.r_[np.ones(len(pos)), np.zeros(len(Xvt))]
        out = {}
        for nm in ["RF", "LR"]:
            clf = tabular(nm, 42).fit(feats(Xk), yk)
            out[nm] = roc_auc_score(ye, clf.predict_proba(Fe)[:, 1])
        sc2 = StandardScaler().fit(feats(Xk[yk == 0]))
        iso = IsolationForest(n_estimators=200, random_state=42).fit(sc2.transform(feats(Xk[yk == 0])))
        out["IF"] = roc_auc_score(ye, -iso.score_samples(sc2.transform(Fe)))
        print(f"{k:12s} n_pos={len(pos):4d}  " + "  ".join(f"{a}={v:.3f}" for a, v in out.items()))
    return sp


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
    run_T3(real, seeds, dl)


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
