"""
AquaSense-ICS - Benchmark T1 (audit v1)
Usage:
    python aquasense_t1_v1.py --csv fused_dataset.csv --seeds 5
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")

CH = ["temperature", "humidity", "gas_resistance", "tds_value",
      "Cl_J7", "Cl_J3", "TURB_J3", "TURB_J7"]          # 8 canaux observes (article)
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
# T1 : audit de la premiere version (split aleatoire, label derive des entrees)
# ----------------------------------------------------------------------------
def run_T1(df, seeds):
    real = get_real(df)
    d = real.dropna(subset=["quality_final"])
    X = d[CH].fillna(0).values
    y = d["quality_final"].values
    print("\n=== T1 : audit v1 (label legacy 'quality_final', 8 colonnes label-defining) ===")
    models = {
        "RandomForest": lambda s: RandomForestClassifier(200, random_state=s, n_jobs=-1),
        "DecisionTree": lambda s: DecisionTreeClassifier(random_state=s),
        "KNN(k=5)": lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "LogReg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }
    print("-- split ALEATOIRE stratifie 80/20 --")
    for name, mk in models.items():
        f1s, accs = [], []
        for s in range(seeds):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42 + s)
            p = mk(42 + s).fit(Xtr, ytr).predict(Xte)
            f1s.append(f1_score(yte, p, average="macro")); accs.append(accuracy_score(yte, p))
        print(f"{name:14s} acc={np.mean(accs):.4f}  macroF1={np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
    print("-- split CHRONOLOGIQUE 60/20/20 (train -> test) --")
    n = len(d); i2 = int(n * (TR + VA))
    for name, mk in models.items():
        p = mk(42).fit(X[:int(n * TR)], y[:int(n * TR)]).predict(X[i2:])
        print(f"{name:14s} macroF1={f1_score(y[i2:], p, average='macro'):.4f}")
    print("Note: sans les 8 colonnes label-defining il ne reste aucune entree -> ablation indefinie.")


def main(csv="fused_data.csv", seeds=5):
    df = pd.read_csv(csv)                              # lecture directe avec pd.read_csv
    for c in ["attack_type", "fault_type"]:
        df[c] = df[c].fillna("none").astype(str)
    if "origin" not in df.columns:
        df["origin"] = "real"
    real = get_real(df)
    print(f"Lignes: {len(df)} (reel: {len(real)})")
    run_T1(df, seeds)


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules or any(a == "-f" for a in sys.argv):
        # Colab / Jupyter : modifier ici
        main(csv="fused_data.csv", seeds=5)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--csv", default="fused_data.csv")
        ap.add_argument("--seeds", type=int, default=5)
        a = ap.parse_args()
        main(a.csv, a.seeds)
