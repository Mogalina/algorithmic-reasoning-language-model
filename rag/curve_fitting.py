"""
Company clustering and interpolation for problem count prediction.

Loads the aggregated dataset, normalizes company curves, clusters them
with K-Means (best k chosen via silhouette score), fits PCHIP curves
per cluster, and exposes a predict_counts() function.

All heavy work runs once at module load time (~milliseconds on the CSV).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGGREGATED_PATH = PROJECT_ROOT / "datasets" / "dataset_aggregated.csv"

TIME_BUCKETS = [30, 90, 180, 360]
ANCHORED_BUCKETS = [0] + TIME_BUCKETS
K_RANGE = range(2, 9)


# Step 1: Load and normalize

def _load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["total"] = df["easycount"] + df["mediumcount"] + df["hardcount"]

    max_total = df.groupby("company")["total"].transform("max")
    df["proportion"] = df["total"] / max_total.clip(lower=1)

    for col in ("easycount", "mediumcount", "hardcount"):
        df[f"{col}_pct"] = df[col] / df["total"].clip(lower=1)

    return df


# Step 2: Build feature matrix for clustering

def _build_feature_matrix(df: pd.DataFrame):
    """
    Pivot companies into a matrix of shape (n_companies, len(TIME_BUCKETS))
    where each cell is the proportion of total problems at that time bucket.
    """
    pivot = df.pivot_table(
        index="company",
        columns="days_until",
        values="proportion",
        aggfunc="first",
    ).reindex(columns=TIME_BUCKETS)

    full_mask = pivot.notna().sum(axis=1) >= 2
    clusterable = pivot[full_mask].copy()
    sparse = pivot[~full_mask].copy()

    clusterable_filled = clusterable.fillna(0)

    return clusterable_filled, sparse, pivot


# Step 3: K-Means with elbow + silhouette

def _find_best_k(X: np.ndarray, k_range=K_RANGE):
    if len(X) < max(k_range):
        k_range = range(2, max(3, len(X)))

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "model": km})

    best = max(results, key=lambda r: r["silhouette"])
    logger.info(
        "K-Means results: %s",
        [(r["k"], round(r["silhouette"], 3)) for r in results],
    )
    logger.info("Best k=%d (silhouette=%.3f)", best["k"], best["silhouette"])
    return best["model"], results


# Step 4: Fit PCHIP curves per cluster

def _fit_cluster_curves(df: pd.DataFrame, company_labels: dict):
    """
    For each cluster, compute average proportion and difficulty mix at each
    time bucket, then fit PCHIP interpolators.
    """
    df = df[df["company"].isin(company_labels)].copy()
    df["cluster"] = df["company"].map(company_labels)

    cluster_ids = sorted(df["cluster"].unique())
    curves = {}

    for cid in cluster_ids:
        cdf = df[df["cluster"] == cid]

        avg = cdf.groupby("days_until").agg(
            proportion=("proportion", "mean"),
            easy_pct=("easycount_pct", "mean"),
            medium_pct=("mediumcount_pct", "mean"),
            hard_pct=("hardcount_pct", "mean"),
        ).reindex(TIME_BUCKETS)

        x = np.array(ANCHORED_BUCKETS, dtype=float)
        y_prop = np.concatenate([[0.0], avg["proportion"].fillna(0).values])
        y_prop[-1] = max(y_prop[-1], y_prop.max())

        y_easy = np.concatenate([[0.0], avg["easy_pct"].fillna(0).values])
        y_med = np.concatenate([[0.0], avg["medium_pct"].fillna(0).values])
        y_hard = np.concatenate([[0.0], avg["hard_pct"].fillna(0).values])

        curves[cid] = {
            "proportion": PchipInterpolator(x, y_prop),
            "easy_pct": PchipInterpolator(x, y_easy),
            "medium_pct": PchipInterpolator(x, y_med),
            "hard_pct": PchipInterpolator(x, y_hard),
            "n_companies": cdf["company"].nunique(),
        }

    return curves


# Step 5: Assign sparse companies to nearest cluster

def _assign_sparse(sparse_df: pd.DataFrame, curves: dict, df: pd.DataFrame):
    assignments = {}

    for company in sparse_df.index:
        rows = df[df["company"] == company]
        if rows.empty:
            assignments[company] = 0
            continue

        best_cluster = 0
        best_dist = float("inf")

        for row_idx, row in rows.iterrows():
            day = row["days_until"]
            prop = row["proportion"]
            for cid, curve_data in curves.items():
                predicted = float(curve_data["proportion"](day))
                dist = abs(prop - predicted)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cid

        assignments[company] = best_cluster

    return assignments


# ---------------------------------------------------------------------------
# Step 6: Build the predictor
# ---------------------------------------------------------------------------

class ProblemCountPredictor:
    def __init__(self, csv_path: Path = AGGREGATED_PATH):
        self.df = _load_and_normalize(csv_path)
        self._fit()

    def _fit(self):
        clusterable, sparse, pivot = _build_feature_matrix(self.df)
        X = clusterable.values

        self.km_model, self.km_results = _find_best_k(X)
        labels = self.km_model.predict(X)

        self.company_cluster = {
            company: int(label)
            for company, label in zip(clusterable.index, labels)
        }

        self.curves = _fit_cluster_curves(self.df, self.company_cluster)

        sparse_assignments = _assign_sparse(sparse, self.curves, self.df)
        self.company_cluster.update(sparse_assignments)

        self.max_totals = (
            self.df.groupby("company")["total"].max().to_dict()
        )

    def predict_counts(self, company: str, days_until: int) -> dict:
        """
        Predict how many easy, medium, hard problems to recommend.

        Returns {"easy": int, "medium": int, "hard": int, "total": int}.
        """
        days_until = max(1, min(days_until, 360))

        cluster_id = self.company_cluster.get(company)
        if cluster_id is None or cluster_id not in self.curves:
            return {"easy": 3, "medium": 5, "hard": 2, "total": 10}

        curve = self.curves[cluster_id]
        max_total = self.max_totals.get(company, 10)

        proportion = float(np.clip(curve["proportion"](days_until), 0, 1))
        total = max(1, round(proportion * max_total))

        easy_pct = float(np.clip(curve["easy_pct"](days_until), 0, 1))
        med_pct = float(np.clip(curve["medium_pct"](days_until), 0, 1))
        hard_pct = float(np.clip(curve["hard_pct"](days_until), 0, 1))

        pct_sum = easy_pct + med_pct + hard_pct
        if pct_sum > 0:
            easy_pct /= pct_sum
            med_pct /= pct_sum
            hard_pct /= pct_sum
        else:
            easy_pct, med_pct, hard_pct = 0.3, 0.5, 0.2

        easy = round(total * easy_pct)
        hard = round(total * hard_pct)
        medium = total - easy - hard

        easy = max(0, easy)
        medium = max(0, medium)
        hard = max(0, hard)

        return {"easy": easy, "medium": medium, "hard": hard, "total": total}

    def get_cluster_info(self, company: str) -> dict:
        """Return cluster assignment and metadata for a company."""
        cid = self.company_cluster.get(company)
        if cid is None:
            return {"cluster": None, "n_companies_in_cluster": 0}
        curve = self.curves.get(cid, {})
        return {
            "cluster": cid,
            "n_companies_in_cluster": curve.get("n_companies", 0),
        }


# Module-level singleton (precomputed on first import)

_predictor = None


def get_predictor() -> ProblemCountPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ProblemCountPredictor()
    return _predictor
