"""Synthetic sequences for AquaSense-ICS: i.i.d. bootstrap and TimeGAN.

Both generators are fitted on the first 60 % of the real record only; the
remaining 40 % is held out for the fidelity metrics (Yoon et al., 2019):
  - discriminative score = |0.5 - accuracy| of a post-hoc GRU classifier
  - predictive score     = MAE of a one-step GRU forecaster, train on
                           synthetic / test on real (TSTR)
  - PCA / t-SNE overlay of 500 windows

Usage:
  python aquasense/synthetic.py --real data/Real_data.csv \
      --out data/synthetic_data.csv --iters 5000 --evaluate
"""
import argparse
import os
import numpy as np
import pandas as pd

CHANNELS = ["tds_value", "temperature", "humidity", "gas_resistance"]
WINDOW = 24
GAP_S = 60
TRAIN_FRAC = 0.60


# ------------------------------------------------------------------ data
def load_real(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def make_windows(df, length=WINDOW, stride=1):
    """Windows never cross a gap longer than GAP_S seconds."""
    dt = df["timestamp"].diff().dt.total_seconds().fillna(0).to_numpy()
    seg = np.cumsum(dt > GAP_S)
    x = df[CHANNELS].to_numpy(float)
    out = []
    for s in np.unique(seg):
        xs = x[seg == s]
        for i in range(0, len(xs) - length + 1, stride):
            out.append(xs[i:i + length])
    return np.asarray(out)


def fit_scaler(df):
    v = df[CHANNELS].to_numpy(float)
    return v.min(0), v.max(0)


def scale(x, mn, mx):
    return (x - mn) / (mx - mn + 1e-9)


def unscale(x, mn, mx):
    return x * (mx - mn + 1e-9) + mn


# ------------------------------------------------------------- bootstrap
def bootstrap(df, n_syn, seed=42, noise_sigma=0.2):
    """i.i.d. bootstrap: rows drawn uniformly with replacement (temporal order
    and autocorrelation are NOT preserved). Gaussian noise (0, 0.2) and a
    linear drift from 0 to 1 are added to tds_value."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), n_syn)
    syn = df.iloc[idx][CHANNELS].reset_index(drop=True).copy()
    syn["tds_value"] = (syn["tds_value"]
                        + rng.normal(0.0, noise_sigma, n_syn)
                        + np.linspace(0.0, 1.0, n_syn))
    return syn


# --------------------------------------------------------------- TimeGAN
class TimeGAN:
    """Compact TimeGAN (Yoon et al., 2019) with GRU networks."""

    def __init__(self, dim, hidden=24, n_layers=3, gamma=1.0, lr=1e-3, seed=42):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.tf, self.dim = tf, dim

        def rnn(in_dim, n, out_dim, act, name):
            inp = keras.Input(shape=(None, in_dim))
            x = inp
            for _ in range(n):
                x = layers.GRU(hidden, return_sequences=True)(x)
            return keras.Model(inp, layers.Dense(out_dim, activation=act)(x), name=name)

        E = rnn(dim, n_layers, hidden, "sigmoid", "embedder")
        R = rnn(hidden, n_layers, dim, "sigmoid", "recovery")
        G = rnn(dim, n_layers, hidden, "sigmoid", "generator")
        S = rnn(hidden, n_layers - 1, hidden, "sigmoid", "supervisor")
        D = rnn(hidden, n_layers, 1, None, "discriminator")
        self.E, self.R, self.G, self.S, self.D = E, R, G, S, D

        opt = {k: keras.optimizers.Adam(lr) for k in ("e0", "s", "g", "e", "d")}
        mse = keras.losses.MeanSquaredError()
        bce = keras.losses.BinaryCrossentropy(from_logits=True)
        ones = tf.ones_like
        zeros = tf.zeros_like

        @tf.function
        def step_embed(x):
            with tf.GradientTape() as t:
                loss = 10.0 * tf.sqrt(mse(x, R(E(x))))
            v = E.trainable_variables + R.trainable_variables
            opt["e0"].apply_gradients(zip(t.gradient(loss, v), v))
            return loss

        @tf.function
        def step_sup(x):
            with tf.GradientTape() as t:
                h = E(x)
                hs = S(h)
                loss = mse(h[:, 1:], hs[:, :-1])
            v = S.trainable_variables
            opt["s"].apply_gradients(zip(t.gradient(loss, v), v))
            return loss

        @tf.function
        def step_g(x, z):
            with tf.GradientTape() as t:
                h = E(x)
                e_hat = G(z)
                h_hat = S(e_hat)
                hs = S(h)
                x_hat = R(h_hat)
                yf, yfe = D(h_hat), D(e_hat)
                g_u = bce(ones(yf), yf)
                g_ue = bce(ones(yfe), yfe)
                g_s = mse(h[:, 1:], hs[:, :-1])
                sd = lambda a: tf.sqrt(tf.math.reduce_variance(a, [0]) + 1e-6)
                g_v1 = tf.reduce_mean(tf.abs(sd(x_hat) - sd(x)))
                g_v2 = tf.reduce_mean(tf.abs(tf.reduce_mean(x_hat, [0]) - tf.reduce_mean(x, [0])))
                loss = g_u + gamma * g_ue + 100.0 * tf.sqrt(g_s) + 100.0 * (g_v1 + g_v2)
            v = G.trainable_variables + S.trainable_variables
            opt["g"].apply_gradients(zip(t.gradient(loss, v), v))
            return loss

        @tf.function
        def step_e(x):
            with tf.GradientTape() as t:
                h = E(x)
                hs = S(h)
                loss = (10.0 * tf.sqrt(mse(x, R(h)))
                        + 0.1 * mse(h[:, 1:], hs[:, :-1]))
            v = E.trainable_variables + R.trainable_variables
            opt["e"].apply_gradients(zip(t.gradient(loss, v), v))
            return loss

        @tf.function
        def d_loss(x, z):
            h = E(x)
            e_hat = G(z)
            h_hat = S(e_hat)
            yr, yf, yfe = D(h), D(h_hat), D(e_hat)
            return bce(ones(yr), yr) + bce(zeros(yf), yf) + gamma * bce(zeros(yfe), yfe)

        @tf.function
        def step_d(x, z):
            with tf.GradientTape() as t:
                loss = d_loss(x, z)
            v = D.trainable_variables
            opt["d"].apply_gradients(zip(t.gradient(loss, v), v))
            return loss

        self._s = dict(embed=step_embed, sup=step_sup, g=step_g, e=step_e,
                       dl=d_loss, d=step_d)

    def fit(self, data, iters=5000, batch=128, log=500):
        data = data.astype("float32")
        n, T, _ = data.shape
        bx = lambda: data[np.random.randint(0, n, batch)]
        bz = lambda: np.random.uniform(0, 1, (batch, T, self.dim)).astype("float32")
        for i in range(iters):                       # phase 1: embedding
            l = self._s["embed"](bx())
            if i % log == 0:
                print(f"[embedding] {i}/{iters} loss={float(l):.4f}", flush=True)
        for i in range(iters):                       # phase 2: supervisor
            l = self._s["sup"](bx())
            if i % log == 0:
                print(f"[supervised] {i}/{iters} loss={float(l):.4f}", flush=True)
        for i in range(iters):                       # phase 3: joint training
            for _ in range(2):
                self._s["g"](bx(), bz())
                self._s["e"](bx())
            x, z = bx(), bz()
            if float(self._s["dl"](x, z)) > 0.15:
                self._s["d"](x, z)
            if i % log == 0:
                print(f"[joint] {i}/{iters}", flush=True)
        return self

    def generate(self, n, length):
        z = np.random.uniform(0, 1, (n, length, self.dim)).astype("float32")
        return self.R(self.S(self.G(z))).numpy()


# -------------------------------------------------------------- fidelity
def discriminative_score(real, fake, seed):
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    m = min(len(real), len(fake))
    X = np.concatenate([real[:m], fake[:m]]).astype("float32")
    y = np.r_[np.ones(m), np.zeros(m)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=X.shape[1:]),
                              keras.layers.GRU(12),
                              keras.layers.Dense(1, activation="sigmoid")])
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, epochs=15, batch_size=128, verbose=0)
    return abs(0.5 - model.evaluate(Xte, yte, verbose=0)[1])


def predictive_score(train_w, test_w, seed):
    """One-step forecaster trained on train_w, MAE on real test_w (TSTR)."""
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    dim = train_w.shape[2]
    model = keras.Sequential([keras.Input(shape=(None, dim)),
                              keras.layers.GRU(12, return_sequences=True),
                              keras.layers.Dense(dim)])
    model.compile("adam", "mae")
    model.fit(train_w[:, :-1], train_w[:, 1:], epochs=15, batch_size=128, verbose=0)
    pred = model.predict(test_w[:, :-1], verbose=0)
    return float(np.mean(np.abs(pred - test_w[:, 1:])))


def evaluate(train_w, held_w, tg_w, bs_w, outdir, seeds=range(5)):
    os.makedirs(outdir, exist_ok=True)
    res = {"Real data (reference)": [], "Bootstrap": [], "TimeGAN": []}
    m = min(len(tg_w), len(bs_w), len(held_w))
    for s in seeds:
        rng = np.random.default_rng(s)
        pick = lambda w: w[rng.choice(len(w), m, replace=False)]
        real_tr, real_he = pick(train_w), pick(held_w)
        res["Real data (reference)"].append((np.nan, predictive_score(real_tr, real_he, s)))
        for name, w in (("Bootstrap", bs_w), ("TimeGAN", tg_w)):
            w = pick(w)
            res[name].append((discriminative_score(real_he, w, s),
                              predictive_score(w, real_he, s)))
    rows = []
    for name, v in res.items():
        v = np.array(v, float)
        rows.append({"generator": name,
                     "discriminative_mean": np.nanmean(v[:, 0]) if not np.isnan(v[:, 0]).all() else np.nan,
                     "discriminative_std": np.nanstd(v[:, 0]) if not np.isnan(v[:, 0]).all() else np.nan,
                     "predictive_mae_mean": v[:, 1].mean(),
                     "predictive_mae_std": v[:, 1].std()})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(outdir, "synth_fidelity.csv"), index=False)
    print(table.to_string(index=False))

    # PCA / t-SNE overlay (500 windows each)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(0)
    sets = {"Real (held-out)": held_w, "Bootstrap": bs_w, "TimeGAN": tg_w}
    flat = {k: w[rng.choice(len(w), min(500, len(w)), replace=False)].reshape(
        min(500, len(w)), -1) for k, w in sets.items()}
    allx = np.concatenate(list(flat.values()))
    pca = PCA(2).fit(flat["Real (held-out)"]).transform(allx)
    tsne = TSNE(2, random_state=0, init="pca").fit_transform(allx)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    i = 0
    for (k, f), c in zip(flat.items(), ("tab:blue", "tab:orange", "tab:green")):
        sl = slice(i, i + len(f)); i += len(f)
        ax[0].scatter(pca[sl, 0], pca[sl, 1], s=5, alpha=0.5, c=c, label=k)
        ax[1].scatter(tsne[sl, 0], tsne[sl, 1], s=5, alpha=0.5, c=c, label=k)
    ax[0].set_title("(a) PCA"); ax[1].set_title("(b) t-SNE"); ax[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_tsne.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/Real_data.csv")
    ap.add_argument("--out", default="data/synthetic_data.csv")
    ap.add_argument("--out-bootstrap", default=None)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--results-dir", default="results")
    a, _ = ap.parse_known_args()

    df = load_real(a.real)
    n = len(df)
    cut = int(TRAIN_FRAC * n)
    train, held = df.iloc[:cut], df.iloc[cut:]
    mn, mx = fit_scaler(train)
    train_w = scale(make_windows(train), mn, mx)
    held_w = scale(make_windows(held), mn, mx)
    print("windows: train", train_w.shape, "held-out", held_w.shape)

    tg = TimeGAN(dim=len(CHANNELS), seed=a.seed).fit(train_w, iters=a.iters)
    n_win = int(np.ceil(n / WINDOW))                 # 987 windows -> 23,688 rows
    fake = tg.generate(n_win, WINDOW)
    rows = unscale(fake.reshape(-1, len(CHANNELS))[:n], mn, mx)
    pd.DataFrame(rows, columns=CHANNELS).to_csv(a.out, index=False)
    print(f"Saved TimeGAN sequences: {a.out} ({n} rows)")

    bs = bootstrap(train, n_syn=n, seed=a.seed)
    if a.out_bootstrap:
        bs.to_csv(a.out_bootstrap, index=False)

    if a.evaluate:
        bs_arr = scale(bs[CHANNELS].to_numpy(float), mn, mx)
        k = (len(bs_arr) // WINDOW) * WINDOW
        bs_w = bs_arr[:k].reshape(-1, WINDOW, len(CHANNELS))
        evaluate(train_w, held_w, fake, bs_w, a.results_dir)


if __name__ == "__main__":
    main()
