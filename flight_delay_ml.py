"""
Flight Delay Prediction Using Machine Learning
==============================================
Mini project - Machine Learning (TY Div A), JSPM's Rajarshi Shahu College of
Engineering, Pune.  Author: Shreyas Jadhav (Roll No: CS3154)

Predicts, using only information known BEFORE take-off, whether a flight from
a New York City airport (EWR / JFK / LGA, 2013) arrives more than 15 minutes
late.  Compares Logistic Regression, KNN, SVM and Random Forest with 5-fold
cross-validation, before and after GridSearchCV tuning.

Usage
-----
    python flight_delay_ml.py                 # real nycflights13 data
    python flight_delay_ml.py --synthetic     # quick smoke test, fake data
    python flight_delay_ml.py --sample-size 6000 --out results/
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
CAT_COLS = ["month", "dow", "carrier", "origin", "dest"]
WEATHER_COLS = ["temp", "dewp", "humid", "wind_dir", "wind_speed",
                "precip", "pressure", "visib"]
# Only features known BEFORE take-off. dep_time, dep_delay, arr_time and
# air_time are deliberately excluded (they would leak the answer).
KEEP = ["month", "dow", "hour", "carrier", "origin", "dest", "distance",
        *WEATHER_COLS, "delayed"]


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_real_data(sample_size, cache_path):
    """Load nycflights13, clean, merge weather, and draw a stratified sample."""
    if os.path.exists(cache_path):
        print(f"Loading cached sample from {cache_path}")
        return pd.read_csv(cache_path)

    from nycflights13 import flights, weather  # pip install nycflights13

    f = flights.dropna(subset=["arr_delay"]).copy()      # drop cancelled/diverted
    print(f"Flights: {len(flights):,} -> {len(f):,} after removing cancelled/diverted")
    f["delayed"] = (f["arr_delay"] > 15).astype(int)     # 1 = delayed

    w = weather[["origin", "time_hour", *WEATHER_COLS]]  # wind_gust dropped (~80% NaN)
    f = f.merge(w, on=["origin", "time_hour"], how="left")
    f["dow"] = pd.to_datetime(f["time_hour"]).dt.dayofweek

    df, _ = train_test_split(f[KEEP], train_size=sample_size,
                             random_state=RANDOM_STATE, stratify=f["delayed"])
    df = df.reset_index(drop=True)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved sample to {cache_path}")
    return df


def make_synthetic_data(n=3000):
    """Fake data with the same schema - ONLY for testing that the code runs."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "month": rng.integers(1, 13, n),
        "dow": rng.integers(0, 7, n),
        "hour": rng.integers(5, 24, n),
        "carrier": rng.choice(["UA", "B6", "EV", "DL", "AA", "US"], n),
        "origin": rng.choice(["EWR", "JFK", "LGA"], n),
        "dest": rng.choice([f"D{i:02d}" for i in range(30)], n),
        "distance": rng.integers(94, 4983, n),
        "temp": rng.normal(55, 18, n), "dewp": rng.normal(41, 18, n),
        "humid": rng.uniform(13, 100, n), "wind_dir": rng.uniform(0, 360, n),
        "wind_speed": rng.uniform(0, 30, n), "precip": rng.exponential(0.01, n),
        "pressure": rng.normal(1017, 7, n), "visib": rng.uniform(0, 10, n),
    })
    logit = -2.3 + 0.08 * (df["hour"] - 5) + 0.02 * (df["humid"] - 60) - 0.1 * df["visib"] / 10
    df["delayed"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[rng.choice(n, 100, replace=False), "pressure"] = np.nan  # some gaps
    return df


# --------------------------------------------------------------------------
# 2. Exploratory data analysis (Figures 2-5)
# --------------------------------------------------------------------------
def run_eda(df, out):
    sns.set_style("whitegrid")

    plt.figure(figsize=(5, 4))
    sns.countplot(x=df["delayed"].map({0: "On time", 1: "Delayed"}),
                  order=["On time", "Delayed"])
    plt.title("Figure 2: Class distribution"); plt.xlabel(""); plt.tight_layout()
    plt.savefig(f"{out}/fig2_class_distribution.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    (df.groupby("hour")["delayed"].mean() * 100).plot(kind="bar", ax=ax[0], color="steelblue")
    ax[0].set(title="Delay rate by scheduled hour", xlabel="Hour", ylabel="% delayed")
    (df.groupby("month")["delayed"].mean() * 100).plot(kind="bar", ax=ax[1], color="indianred")
    ax[1].set(title="Delay rate by month", xlabel="Month", ylabel="% delayed")
    fig.suptitle("Figure 3"); plt.tight_layout()
    plt.savefig(f"{out}/fig3_delay_by_time.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    car = df.groupby("carrier")["delayed"].agg(["mean", "size"])
    (car[car["size"] >= 30]["mean"].sort_values() * 100).plot(kind="barh", ax=ax[0], color="teal")
    ax[0].set(title="Delay rate by airline (>= 30 flights)", xlabel="% delayed")
    (df.groupby("origin")["delayed"].mean() * 100).plot(kind="bar", ax=ax[1], color="orange")
    ax[1].set(title="Delay rate by origin airport", ylabel="% delayed")
    fig.suptitle("Figure 4"); plt.tight_layout()
    plt.savefig(f"{out}/fig4_delay_by_carrier_origin.png", dpi=150); plt.close()

    num = ["hour", "distance", *WEATHER_COLS, "delayed"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Figure 5: Correlation of numeric features"); plt.tight_layout()
    plt.savefig(f"{out}/fig5_correlation_heatmap.png", dpi=150); plt.close()

    print(f"Delay rate: {df['delayed'].mean():.1%}")
    print("Top correlations with delay:\n",
          df[num].corr()["delayed"].drop("delayed").sort_values(key=abs, ascending=False).head(5).round(3))


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def make_preprocessor(num_feats):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ], sparse_threshold=0)


def build_models(num_feats):
    def pipe(clf):
        return Pipeline([("prep", make_preprocessor(num_feats)), ("clf", clf)])
    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=1000)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=5)),
        "SVM": pipe(SVC(kernel="rbf")),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=200,
                                                     random_state=RANDOM_STATE)),
    }


# Table 3. Every grid contains the default setting.
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1, 10],
                            "clf__class_weight": [None, "balanced"]},
    "KNN": {"clf__n_neighbors": [5, 11, 21, 41],
            "clf__weights": ["uniform", "distance"]},
    "SVM": {"clf__C": [0.5, 1, 2],
            "clf__class_weight": [None, "balanced"]},
    "Random Forest": {"clf__max_depth": [8, None],
                      "clf__min_samples_leaf": [1, 5],
                      "clf__class_weight": [None, "balanced"]},
}


def scores_for(model, X):
    """Continuous scores for ROC-AUC (SVM has no predict_proba by default)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X_test, y_test, cv_f1):
    pred = model.predict(X_test)
    return {
        "CV F1": cv_f1,
        "Test Acc.": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, scores_for(model, X_test)),
    }


# --------------------------------------------------------------------------
# 4. Plots for results (Figures 6-9)
# --------------------------------------------------------------------------
def plot_results(default_df, tuned_df, tuned_models, X_test, y_test, num_feats, out):
    names = list(tuned_models)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, metric in zip(ax, ["F1-score", "Test Acc."]):
        x = np.arange(len(names))
        a.bar(x - 0.2, default_df.loc[names, metric], 0.4, label="Default")
        a.bar(x + 0.2, tuned_df.loc[names, metric], 0.4, label="Tuned")
        a.set_xticks(x); a.set_xticklabels(names, rotation=15)
        a.set_title(metric); a.legend()
    fig.suptitle("Figure 6: Default vs tuned"); plt.tight_layout()
    plt.savefig(f"{out}/fig6_default_vs_tuned.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, n in zip(ax, names):
        cm = confusion_matrix(y_test, tuned_models[n].predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                    xticklabels=["On time", "Delayed"], yticklabels=["On time", "Delayed"])
        a.set(title=n, xlabel="Predicted", ylabel="Actual")
    fig.suptitle("Figure 7: Confusion matrices (tuned)"); plt.tight_layout()
    plt.savefig(f"{out}/fig7_confusion_matrices.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for n in names:
        fpr, tpr, _ = roc_curve(y_test, scores_for(tuned_models[n], X_test))
        plt.plot(fpr, tpr, label=f"{n} (AUC={tuned_df.loc[n, 'ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Figure 8: ROC curves (tuned)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/fig8_roc_curves.png", dpi=150); plt.close()

    rf = tuned_models["Random Forest"]
    prep, clf = rf.named_steps["prep"], rf.named_steps["clf"]
    feat_names = list(num_feats) + list(prep.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
    imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
    imp.head(15)[::-1].plot(kind="barh", figsize=(7, 5), color="seagreen")
    plt.title("Figure 9: Random Forest feature importance (top 15)")
    plt.tight_layout(); plt.savefig(f"{out}/fig9_feature_importance.png", dpi=150); plt.close()
    imp.head(15).to_csv(f"{out}/feature_importance.csv", header=["importance"])
    print("\nTop 5 features (Random Forest):\n", imp.head(5).round(3))


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-size", type=int, default=12000)
    ap.add_argument("--data", default="data/flights_sample.csv",
                    help="cache file for the sampled data")
    ap.add_argument("--out", default="outputs", help="folder for figures/tables")
    ap.add_argument("--synthetic", action="store_true",
                    help="use fake data to check the code runs (results meaningless)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = make_synthetic_data() if args.synthetic else load_real_data(args.sample_size, args.data)
    print(f"\nSample: {len(df):,} flights, {df['delayed'].mean():.1%} delayed")
    run_eda(df, args.out)

    X = df.drop("delayed", axis=1).copy()
    y = df["delayed"]
    X["month"] = X["month"].astype(str)
    X["dow"] = X["dow"].astype(str)
    num_feats = [c for c in X.columns if c not in CAT_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(num_feats)

    # --- Default models (Table 5) ---
    default_rows = {}
    for name, model in models.items():
        print(f"Training default {name} ...")
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()
        model.fit(X_train, y_train)
        default_rows[name] = evaluate(model, X_test, y_test, cv_f1)
    default_df = pd.DataFrame(default_rows).T.round(3)
    print("\nTable 5: default settings\n", default_df)

    # --- Tuned models (Table 6) ---
    tuned_models, tuned_rows, best_params = {}, {}, {}
    for name, model in models.items():
        print(f"Tuning {name} ...")
        gs = GridSearchCV(model, PARAM_GRIDS[name], cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        tuned_models[name] = gs.best_estimator_
        best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        tuned_rows[name] = evaluate(gs.best_estimator_, X_test, y_test, gs.best_score_)
    tuned_df = pd.DataFrame(tuned_rows).T.round(3)
    print("\nTable 6: after tuning\n", tuned_df)
    print("\nBest parameters:", best_params)

    # --- Confusion counts (Table 7) ---
    cm_rows = {}
    for name, m in tuned_models.items():
        tn, fp, fn, tp = confusion_matrix(y_test, m.predict(X_test)).ravel()
        cm_rows[name] = {"True Neg.": tn, "False Pos.": fp, "False Neg.": fn, "True Pos.": tp}
    cm_df = pd.DataFrame(cm_rows).T
    print("\nTable 7: confusion matrix counts\n", cm_df)

    default_df.to_csv(f"{args.out}/table5_default_results.csv")
    tuned_df.to_csv(f"{args.out}/table6_tuned_results.csv")
    cm_df.to_csv(f"{args.out}/table7_confusion_counts.csv")
    plot_results(default_df, tuned_df, tuned_models, X_test, y_test, num_feats, args.out)
    print(f"\nDone. Figures and tables saved in '{args.out}/'.")


if __name__ == "__main__":
    main()
