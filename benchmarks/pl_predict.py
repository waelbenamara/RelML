"""
pl_predict.py — Standalone Premier League outcome predictor

Usage:
    python pl_predict.py <data_dir>                  # train + evaluate
    python pl_predict.py <data_dir> --predict        # train + predict next fixtures
    python pl_predict.py <data_dir> --match "Arsenal vs Chelsea"

Data dir must contain: games.csv, teams.csv, players.csv, player.csv

Model: LightGBM multiclass classifier (Home Win / Draw / Away Win)
Features: rolling last-5 form, season table, head-to-head, days rest
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

WINDOW      = 5       # rolling form window
MAX_REST    = 14      # days rest cap
CUTOFF      = "2026-03-01"   # exclude on-or-after this date from training
RANDOM_SEED = 42

# ── class labels ──────────────────────────────────────────────────────────────
LABEL_NAMES = {0: "Home Win", 1: "Draw", 2: "Away Win"}


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    games   = pd.read_csv(d / "games.csv")
    teams   = pd.read_csv(d / "teams.csv")
    players = pd.read_csv(d / "players.csv")

    # Parse dates — ISO 8601 with timezone
    games["date"] = pd.to_datetime(games["date"], utc=True).dt.tz_localize(None)

    # Outcome label (only for completed fixtures)
    mask = games["goals_home"].notna() & games["goals_away"].notna()
    games.loc[mask, "outcome"] = np.where(
        games.loc[mask, "goals_home"] > games.loc[mask, "goals_away"], 0,
        np.where(games.loc[mask, "goals_home"] == games.loc[mask, "goals_away"], 1, 2)
    ).astype(float)

    games = games.sort_values("date").reset_index(drop=True)
    return games, teams, players


# ── feature engineering ───────────────────────────────────────────────────────

def date_to_days(dt) -> int:
    if pd.isna(dt):
        return -1
    return (dt - datetime(2000, 1, 1)).days


def build_features(games: pd.DataFrame, players: pd.DataFrame,
                   teams: pd.DataFrame) -> pd.DataFrame:
    """
    Sweeps games chronologically. For each fixture records all pre-match
    features before updating the accumulators with the result.
    Returns a DataFrame aligned with the input games index.
    """
    N = len(games)

    # --- team season-aggregate player stats (from players.csv) ---------------
    # Average rating, goals, assists per player per (team, season)
    player_team_stats = (
        players.groupby(["team_id", "season"])[
            ["rating", "goals", "assists", "appearances",
             "tackles_total", "passes_accuracy", "dribbles_success"]
        ].mean()
        .rename(columns=lambda c: f"squad_{c}")
        .reset_index()
    )

    # --- accumulators ---------------------------------------------------------
    # team_history[(team_id)] → deque of dicts
    team_hist  = defaultdict(lambda: deque(maxlen=WINDOW * 4))
    h2h_hist   = defaultdict(lambda: deque(maxlen=WINDOW * 2))
    # (team_id, season) → SeasonRecord
    season_rec = defaultdict(lambda: {"pts": 0, "gd": 0, "played": 0})

    rows = []

    for _, g in games.iterrows():
        hid   = g["home_team_id"]
        aid   = g["away_team_id"]
        d     = g["date"]
        s     = g.get("season", -1)
        gh    = g["goals_home"]
        ga    = g["goals_away"]
        day   = date_to_days(d)
        completed = pd.notna(gh) and pd.notna(ga)
        future    = (d >= pd.Timestamp(CUTOFF)) if pd.notna(d) else False

        # ── rolling form helper ──
        def team_form(tid):
            hist = list(team_hist[tid])[-WINDOW:]
            if not hist:
                return dict(
                    wins=0, draws=0, losses=0,
                    gf=0, ga_f=0, clean_sheets=0,
                    home_wins=0, away_wins=0, days_rest=MAX_REST
                )
            wins = draws = losses = gf_sum = ga_sum = cs = hw = aw = 0
            last_day = -1
            for m in reversed(hist):
                if last_day < 0:
                    last_day = m["day"]
                if m["gf"] > m["ga"]:
                    wins += 1
                    if m["home"]:
                        hw += 1
                    else:
                        aw += 1
                elif m["gf"] == m["ga"]:
                    draws += 1
                else:
                    losses += 1
                gf_sum += m["gf"]
                ga_sum += m["ga"]
                if m["ga"] == 0:
                    cs += 1
            n   = len(hist)
            rest = min(day - last_day, MAX_REST) if (last_day >= 0 and day >= 0) else MAX_REST
            return dict(
                wins=wins/WINDOW, draws=draws/WINDOW, losses=losses/WINDOW,
                gf=gf_sum/(WINDOW*5), ga_f=ga_sum/(WINDOW*5),
                clean_sheets=cs/WINDOW,
                home_wins=hw/WINDOW, away_wins=aw/WINDOW,
                days_rest=rest/MAX_REST
            )

        hf = team_form(hid)
        af = team_form(aid)

        # ── head-to-head ──
        h2h_key = (min(hid, aid), max(hid, aid))
        h2h = list(h2h_hist[h2h_key])[-WINDOW:]
        h2h_hw = sum(1 for x in h2h if x["outcome"] == 0) / WINDOW
        h2h_d  = sum(1 for x in h2h if x["outcome"] == 1) / WINDOW
        h2h_aw = sum(1 for x in h2h if x["outcome"] == 2) / WINDOW

        # ── season table ──
        MAX_PTS = 99.0
        MAX_GD  = 50.0
        hr = season_rec[(hid, s)]
        ar = season_rec[(aid, s)]
        h_pts  = hr["pts"] / MAX_PTS
        h_gd   = np.clip(hr["gd"] / MAX_GD, -1, 1) * 0.5 + 0.5
        h_ppg  = (hr["pts"] / (hr["played"] * 3)) if hr["played"] > 0 else 0.0
        a_pts  = ar["pts"] / MAX_PTS
        a_gd   = np.clip(ar["gd"] / MAX_GD, -1, 1) * 0.5 + 0.5
        a_ppg  = (ar["pts"] / (ar["played"] * 3)) if ar["played"] > 0 else 0.0
        pts_diff = (hr["pts"] - ar["pts"] + MAX_PTS) / (2 * MAX_PTS)

        # ── squad quality from players.csv ──
        def squad_stats(tid):
            sub = player_team_stats[
                (player_team_stats["team_id"] == tid) &
                (player_team_stats["season"] == s)
            ]
            if sub.empty:
                return {c: 0.0 for c in [
                    "squad_rating", "squad_goals", "squad_assists",
                    "squad_appearances", "squad_tackles_total",
                    "squad_passes_accuracy", "squad_dribbles_success"
                ]}
            return sub.iloc[0].drop(["team_id", "season"]).to_dict()

        hs = squad_stats(hid)
        as_ = squad_stats(aid)

        # ── venue capacity (proxy for home atmosphere) ──
        def venue_cap(tid):
            row = teams[teams["team_id"] == tid]
            if row.empty:
                return 0.5
            cap = row.iloc[0].get("venue_capacity", 40000)
            return float(cap) / 90000.0   # normalise to ~[0,1]

        row = {
            "fixture_id":    g["fixture_id"],
            "date":          d,
            "season":        s,
            "home_team_id":  hid,
            "away_team_id":  aid,
            "outcome":       g.get("outcome", np.nan),
            "future":        future,

            # home rolling form
            "h_wins":        hf["wins"],
            "h_draws":       hf["draws"],
            "h_losses":      hf["losses"],
            "h_gf":          hf["gf"],
            "h_ga":          hf["ga_f"],
            "h_cs":          hf["clean_sheets"],
            "h_home_wins":   hf["home_wins"],
            "h_rest":        hf["days_rest"],

            # away rolling form
            "a_wins":        af["wins"],
            "a_draws":       af["draws"],
            "a_losses":      af["losses"],
            "a_gf":          af["gf"],
            "a_ga":          af["ga_f"],
            "a_cs":          af["clean_sheets"],
            "a_away_wins":   af["away_wins"],
            "a_rest":        af["days_rest"],

            # differential form signals
            "diff_wins":     hf["wins"]   - af["wins"],
            "diff_gf":       hf["gf"]     - af["gf"],
            "diff_ga":       hf["ga_f"]   - af["ga_f"],
            "diff_rest":     hf["days_rest"] - af["days_rest"],

            # head-to-head
            "h2h_hw":        h2h_hw,
            "h2h_d":         h2h_d,
            "h2h_aw":        h2h_aw,

            # season table
            "h_pts":         h_pts,
            "h_gd":          h_gd,
            "h_ppg":         h_ppg,
            "a_pts":         a_pts,
            "a_gd":          a_gd,
            "a_ppg":         a_ppg,
            "pts_diff":      pts_diff,

            # venue
            "home_capacity": venue_cap(hid),
        }

        # squad quality (home and away, and diffs)
        for k, v in hs.items():
            row[f"h_{k}"] = float(v) if pd.notna(v) else 0.0
        for k, v in as_.items():
            row[f"a_{k}"] = float(v) if pd.notna(v) else 0.0
        for k in hs:
            h_val = float(hs[k]) if pd.notna(hs[k]) else 0.0
            a_val = float(as_[k]) if pd.notna(as_[k]) else 0.0
            row[f"diff_{k}"] = h_val - a_val

        rows.append(row)

        # ── update accumulators (only for completed, non-future games) ──
        if completed and not future:
            gh_i = int(gh)
            ga_i = int(ga)
            outcome = 0 if gh_i > ga_i else (1 if gh_i == ga_i else 2)

            team_hist[hid].append({"day": day, "gf": gh_i, "ga": ga_i, "home": True})
            team_hist[aid].append({"day": day, "gf": ga_i, "ga": gh_i, "home": False})

            h2h_hist[h2h_key].append({"outcome": outcome})

            hpts = 3 if gh_i > ga_i else (1 if gh_i == ga_i else 0)
            apts = 3 if ga_i > gh_i else (1 if gh_i == ga_i else 0)
            season_rec[(hid, s)]["pts"]    += hpts
            season_rec[(hid, s)]["gd"]     += gh_i - ga_i
            season_rec[(hid, s)]["played"] += 1
            season_rec[(aid, s)]["pts"]    += apts
            season_rec[(aid, s)]["gd"]     += ga_i - gh_i
            season_rec[(aid, s)]["played"] += 1

    return pd.DataFrame(rows)


# ── model training ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "h_wins", "h_draws", "h_losses", "h_gf", "h_ga", "h_cs",
    "h_home_wins", "h_rest",
    "a_wins", "a_draws", "a_losses", "a_gf", "a_ga", "a_cs",
    "a_away_wins", "a_rest",
    "diff_wins", "diff_gf", "diff_ga", "diff_rest",
    "h2h_hw", "h2h_d", "h2h_aw",
    "h_pts", "h_gd", "h_ppg", "a_pts", "a_gd", "a_ppg", "pts_diff",
    "home_capacity",
    "h_squad_rating", "h_squad_goals", "h_squad_assists",
    "h_squad_passes_accuracy", "h_squad_tackles_total",
    "a_squad_rating", "a_squad_goals", "a_squad_assists",
    "a_squad_passes_accuracy", "a_squad_tackles_total",
    "diff_squad_rating", "diff_squad_goals", "diff_squad_assists",
]


def train_model(df: pd.DataFrame, test_cutoff: str = "2024-08-01"):
    """
    Train on completed games strictly before test_cutoff.
    Evaluate on completed games on-or-after test_cutoff (honest holdout).
    test_cutoff defaults to the start of the 2024-25 season so the model
    is always evaluated on a season it has never seen during training.
    """
    labeled = df[df["outcome"].notna() & ~df["future"]].copy()

    train_mask = labeled["date"] < pd.Timestamp(test_cutoff)
    test_mask  = labeled["date"] >= pd.Timestamp(test_cutoff)

    train_df = labeled[train_mask]
    test_df  = labeled[test_mask]

    print(f"\nTraining on {len(train_df)} games "
          f"(up to {test_cutoff})")
    print(f"Testing  on {len(test_df)} games "
          f"({test_cutoff} onward — never seen during training)")

    # Only keep features that actually exist in this df
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    X_tr = train_df[feat_cols].fillna(0).values
    y_tr = train_df["outcome"].astype(int).values
    X_te = test_df[feat_cols].fillna(0).values
    y_te = test_df["outcome"].astype(int).values

    # Class weights — sqrt of inverse frequency on TRAINING set only.
    # Full inverse-frequency ("balanced") weights push draw recall to 0.72
    # at the cost of precision 0.28, driving overall accuracy below the naive
    # home-win baseline. Square-root weighting gives draws a meaningful boost
    # (w~1.17 vs 1.0) without swamping the signal from the majority classes.
    counts    = np.bincount(y_tr, minlength=3)
    raw_w     = len(y_tr) / (3 * counts.clip(1))
    weights   = np.sqrt(raw_w)
    weights  /= weights.mean()
    sample_wt = np.array([weights[yi] for yi in y_tr])

    # Use a small random val split from training data only for early stopping
    X_tr2, X_val, y_tr2, y_val, w_tr2, _ = train_test_split(
        X_tr, y_tr, sample_wt,
        test_size=0.15, random_state=RANDOM_SEED, stratify=y_tr
    )

    dist = np.bincount(y_tr, minlength=3)
    print(f"  Train distribution — Home: {dist[0]}  Draw: {dist[1]}  Away: {dist[2]}")
    print(f"  Class weights      — Home: {weights[0]:.3f}  Draw: {weights[1]:.3f}  Away: {weights[2]:.3f}")
    dist_te = np.bincount(y_te, minlength=3)
    print(f"  Test  distribution — Home: {dist_te[0]}  Draw: {dist_te[1]}  Away: {dist_te[2]}")

    model = lgb.LGBMClassifier(
        n_estimators      = 1000,
        learning_rate     = 0.02,
        num_leaves        = 20,      # conservative for ~2177 samples
        max_depth         = 4,
        min_child_samples = 30,
        subsample         = 0.7,
        colsample_bytree  = 0.7,
        reg_alpha         = 0.5,
        reg_lambda        = 2.0,
        class_weight      = None,   # handled via sample_weight above
        random_state      = RANDOM_SEED,
        verbose           = -1,
    )

    model.fit(
        X_tr2, y_tr2,
        sample_weight      = w_tr2,
        eval_set           = [(X_val, y_val)],
        callbacks          = [lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)],
    )

    # Evaluate on the honest temporal holdout
    y_pred_te = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred_te)
    f1  = f1_score(y_te, y_pred_te, average="macro")
    print(f"\nTest (honest holdout — {test_cutoff} onward):")
    print(f"  Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print(classification_report(y_te, y_pred_te,
                                 target_names=["Home Win", "Draw", "Away Win"]))

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("Top 10 features:")
    print(fi.head(10).to_string(index=False))

    return model, feat_cols, fi, test_df


# ── prediction ────────────────────────────────────────────────────────────────

def predict_fixtures(model, df: pd.DataFrame, feat_cols: list,
                     teams: pd.DataFrame, fixtures=None):
    """
    If fixtures is None: predict all future games in df (date >= CUTOFF).
    If fixtures is a list of (home_name, away_name): look up team ids and
    find the most recent game row for each pairing.
    """
    team_name_map = teams.set_index("name")["team_id"].to_dict()
    team_id_map   = teams.set_index("team_id")["name"].to_dict()

    if fixtures:
        # Manual fixture list
        results = []
        for home_name, away_name in fixtures:
            hid = _fuzzy_team_id(home_name, team_name_map)
            aid = _fuzzy_team_id(away_name, team_name_map)
            if hid is None or aid is None:
                print(f"  Unknown team in: {home_name} vs {away_name}")
                continue
            # Find the most recent feature row for this pairing
            mask = (df["home_team_id"] == hid) & (df["away_team_id"] == aid)
            if not mask.any():
                mask = (df["home_team_id"] == hid) | (df["away_team_id"] == aid)
            row = df[mask].iloc[-1]
            x   = np.array([[row.get(c, 0.0) for c in feat_cols]])
            proba = model.predict_proba(x)[0]
            pred  = int(np.argmax(proba))
            results.append({
                "Home":       home_name,
                "Away":       away_name,
                "Prediction": LABEL_NAMES[pred],
                "P(Home)":    f"{proba[0]:.1%}",
                "P(Draw)":    f"{proba[1]:.1%}",
                "P(Away)":    f"{proba[2]:.1%}",
            })
        return pd.DataFrame(results)

    # Predict all future fixtures
    future = df[df["future"]].copy()
    if future.empty:
        print("No future fixtures found in dataset.")
        return pd.DataFrame()

    X_fut = future[[c for c in feat_cols if c in future.columns]].fillna(0).values
    proba  = model.predict_proba(X_fut)
    preds  = np.argmax(proba, axis=1)

    out = []
    for i, (_, row) in enumerate(future.iterrows()):
        out.append({
            "Date":       row["date"].strftime("%a %d %b") if pd.notna(row["date"]) else "?",
            "Home":       team_id_map.get(row["home_team_id"], str(row["home_team_id"])),
            "Away":       team_id_map.get(row["away_team_id"], str(row["away_team_id"])),
            "Prediction": LABEL_NAMES[preds[i]],
            "P(Home)":    f"{proba[i][0]:.1%}",
            "P(Draw)":    f"{proba[i][1]:.1%}",
            "P(Away)":    f"{proba[i][2]:.1%}",
        })
    return pd.DataFrame(out)


def _fuzzy_team_id(name: str, name_map: dict):
    """Match team name case-insensitively, with partial matching fallback."""
    name_lower = name.lower()
    for k, v in name_map.items():
        if k.lower() == name_lower:
            return v
    for k, v in name_map.items():
        if name_lower in k.lower() or k.lower() in name_lower:
            return v
    return None


# ── pretty print ──────────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Premier League Outcome Predictor")
    parser.add_argument("data_dir",  help="Directory containing PL CSVs")
    parser.add_argument("--predict", action="store_true",
                        help="Print predictions for all upcoming fixtures")
    parser.add_argument("--match",   type=str, default=None,
                        help='Predict a specific match e.g. "Arsenal vs Chelsea"')
    args = parser.parse_args()

    print("Loading data...")
    games, teams, players = load_data(args.data_dir)
    print(f"  {len(games)} games  |  {len(teams)} teams  |  {len(players)} player-seasons")

    print("\nEngineering features...")
    df = build_features(games, players, teams)
    completed = df["outcome"].notna() & ~df["future"]
    future    = df["future"]
    print(f"  {completed.sum()} labeled games  |  {future.sum()} future fixtures")

    model, feat_cols, _, test_df = train_model(df)

    if args.match:
        parts = [p.strip() for p in args.match.split("vs")]
        if len(parts) != 2:
            print("--match format: 'Home Team vs Away Team'")
            sys.exit(1)
        result = predict_fixtures(model, df, feat_cols, teams, [tuple(parts)])
        print_table(result, f"Prediction: {args.match}")

    elif args.predict:
        result = predict_fixtures(model, df, feat_cols, teams)
        if not result.empty:
            print_table(result, "Upcoming Fixture Predictions")

    else:
        # Default: show global prediction distribution on training data
        X_all = df[[c for c in feat_cols if c in df.columns]].fillna(0).values
        preds = model.predict(X_all)
        dist  = np.bincount(preds, minlength=3)
        total = len(preds)
        print(f"\nGlobal distribution over all {total} games:")
        for i, name in LABEL_NAMES.items():
            print(f"  {name:12s}: {dist[i]:4d}  ({dist[i]/total:.1%})")

        # Show a sample of the honest temporal holdout (last 20 games in test set)
        team_id_map = teams.set_index("team_id")["name"].to_dict()
        sample = test_df.tail(20).copy()
        X_bt = sample[[c for c in feat_cols if c in sample.columns]].fillna(0).values
        p_bt = model.predict(X_bt)
        proba_bt = model.predict_proba(X_bt)
        sample["Prediction"] = [LABEL_NAMES[p] for p in p_bt]
        sample["Actual"]     = sample["outcome"].map(LABEL_NAMES)
        sample["Correct"]    = sample["Prediction"] == sample["Actual"]
        sample["P(Home)"]    = [f"{p[0]:.0%}" for p in proba_bt]
        sample["P(Draw)"]    = [f"{p[1]:.0%}" for p in proba_bt]
        sample["P(Away)"]    = [f"{p[2]:.0%}" for p in proba_bt]
        sample["Home"]       = sample["home_team_id"].map(team_id_map)
        sample["Away"]       = sample["away_team_id"].map(team_id_map)
        sample["Date"]       = sample["date"].dt.strftime("%d %b %Y")

        correct = sample["Correct"].sum()
        print_table(
            sample[["Date", "Home", "Away", "Prediction", "P(Home)", "P(Draw)", "P(Away)", "Actual", "Correct"]],
            f"Holdout sample — last 20 games of test set  ({correct}/20 correct)"
        )

        # Show upcoming fixtures if any
        result = predict_fixtures(model, df, feat_cols, teams)
        if not result.empty:
            print_table(result, "Upcoming Fixture Predictions")


if __name__ == "__main__":
    main()