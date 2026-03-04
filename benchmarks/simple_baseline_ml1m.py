"""
Flat MLP baseline for MovieLens 1M rating prediction.

Same task as RelML: predict whether a rating is >= 4 stars.
Same temporal split: 70 / 15 / 15 by timestamp.
Same feature encoding: numerical standardization, one-hot categoricals,
  cyclic timestamp. The key difference: features come from a flat JOIN
  of all three tables -- no relational message passing, no graph.

Usage:
    python baseline_mlp.py --data ./ml-1m
"""

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def load_movielens(data_dir: str):
    users   = pd.read_csv(f"{data_dir}/users.csv")
    movies  = pd.read_csv(f"{data_dir}/movies.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    return users, movies, ratings


def cyclic(series, period):
    return np.sin(2 * np.pi * series / period), np.cos(2 * np.pi * series / period)


def build_features(df: pd.DataFrame):
    t = pd.to_datetime(df["timestamp"], unit="s")
    m_sin, m_cos = cyclic(t.dt.month, 12)
    d_sin, d_cos = cyclic(t.dt.day,   31)

    age      = df["age"].values.reshape(-1, 1).astype(np.float32)
    gender   = pd.get_dummies(df["gender"],     prefix="gender",  dtype=np.float32).values
    occ      = pd.get_dummies(df["occupation"], prefix="occ",     dtype=np.float32).values
    genres   = df["genres"].str.get_dummies(sep="|").astype(np.float32).values
    ts_block = np.column_stack([m_sin, m_cos, d_sin, d_cos]).astype(np.float32)

    return np.hstack([age, gender, occ, genres, ts_block])


def temporal_split(X, y, r1=0.70, r2=0.85):
    N  = len(X)
    t1 = int(N * r1)
    t2 = int(N * r2)
    return (X[:t1],  y[:t1],
            X[t1:t2], y[t1:t2],
            X[t2:],  y[t2:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="./ml-1m-data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", default="128,64")
    parser.add_argument("--batch",  type=int, default=512)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(","))

    users, movies, ratings = load_movielens(args.data)

    df = (ratings
          .merge(users,  on="userId")
          .merge(movies, on="movieId")
          .sort_values("timestamp")
          .reset_index(drop=True))

    y = (df["rating"] >= 4).astype(int).values
    X = build_features(df)

    X_tr, y_tr, X_val, y_val, X_te, y_te = temporal_split(X, y)

    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_tr)
    X_val   = scaler.transform(X_val)
    X_te    = scaler.transform(X_te)

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    # sample_weight in MLPClassifier.fit() requires scikit-learn >= 0.23
    sample_weight = np.where(y_tr == 1, n_neg / n_pos, 1.0).astype(np.float64)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=args.lr,
        alpha=1e-4,
        batch_size=args.batch,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    print(f"\nFeatures: {X_tr.shape[1]}  "
          f"Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}")
    print(f"Positive rate — train: {y_tr.mean():.3f}  "
          f"val: {y_val.mean():.3f}  test: {y_te.mean():.3f}")
    print(f"Positive-class weight: {n_neg/n_pos:.3f}\n")

    col = f"{'Epoch':>5}  {'Train Loss':>12}  {'Val AP':>8}  {'Val AUC':>8}  {'Val Acc':>8}  {'Time (s)':>9}"
    sep = "-" * len(col)
    print(col)
    print(sep)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        try:
            clf.fit(X_tr, y_tr, sample_weight=sample_weight)
        except TypeError:
            clf.fit(X_tr, y_tr)  # older sklearn: no sample_weight
        elapsed = time.time() - t0

        p_val   = clf.predict_proba(X_val)[:, 1]
        val_ap  = average_precision_score(y_val, p_val)
        val_auc = roc_auc_score(y_val, p_val)
        val_acc = accuracy_score(y_val, p_val >= 0.5)

        print(f"{epoch:>5}  {clf.loss_:>12.4f}  "
              f"{val_ap:>8.4f}  {val_auc:>8.4f}  {val_acc:>8.4f}  {elapsed:>9.2f}")

    print(sep)
    p_te   = clf.predict_proba(X_te)[:, 1]
    te_ap  = average_precision_score(y_te, p_te)
    te_auc = roc_auc_score(y_te, p_te)
    te_acc = accuracy_score(y_te, p_te >= 0.5)
    print(f"{'Test':>5}  {'':>12}  {te_ap:>8.4f}  {te_auc:>8.4f}  {te_acc:>8.4f}")


if __name__ == "__main__":
    main()