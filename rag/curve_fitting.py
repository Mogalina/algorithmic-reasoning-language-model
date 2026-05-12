import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGGREGATED_PATH = PROJECT_ROOT / "datasets" / "dataset_aggregated.csv"
COMPANY_META_PATH = PROJECT_ROOT / "datasets" / "company_meta.csv"

TIME_BUCKETS = [30, 90, 180, 360]
ANCHORED_BUCKETS = [0] + TIME_BUCKETS
K_RANGE = range(2, 9)

SIZE_RANGE_ORDER = {
    "1-10": 0,
    "11-50": 1,
    "51-200": 2,
    "201-500": 3,
    "501-1000": 4,
    "1001-5000": 5,
    "5001-10000": 6,
    "10001+": 7,
}

# Column names for the company feature dimensions (used for clustering)
COMPANY_FEATURE_COLS = ["industry_enc", "size_rank", "log_employees", "year_founded"]

# Industry is the primary signal for "what kind of company is this"
INDUSTRY_WEIGHT = 2.0


def _load_company_features(path: Path = COMPANY_META_PATH) -> pd.DataFrame | None:
    """
    Load datasets/company_meta.csv and return a DataFrame with numeric
    columns suitable for clustering, indexed by company name.

    Returns None if the file doesn't exist (graceful degradation).
    """
    if not path.exists():
        logger.info("company_meta.csv not found — clustering without company features")
        return None

    meta = pd.read_csv(path)
    if meta.empty:
        return None

    meta = meta.set_index("company")

    le = LabelEncoder()
    industries = meta["industry"].fillna("unknown").astype(str)
    meta["industry_enc"] = le.fit_transform(industries)

    meta["size_rank"] = (
        meta["size_range"]
        .fillna("")
        .map(SIZE_RANGE_ORDER)
        .fillna(len(SIZE_RANGE_ORDER) // 2)
        .astype(float)
    )

    employees = pd.to_numeric(meta["employee_count"], errors="coerce").fillna(1).clip(lower=1)
    meta["log_employees"] = np.log1p(employees)

    meta["year_founded"] = (
        pd.to_numeric(meta["year_founded"], errors="coerce")
        .fillna(2000)
        .astype(float)
    )

    return meta[COMPANY_FEATURE_COLS]



def _load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["total"] = df["easycount"] + df["mediumcount"] + df["hardcount"]

    max_total = df.groupby("company")["total"].transform("max")
    df["proportion"] = df["total"] / max_total.clip(lower=1)

    for col in ("easycount", "mediumcount", "hardcount"):
        df[f"{col}_pct"] = df[col] / df["total"].clip(lower=1)

    return df



def _build_feature_matrix(df: pd.DataFrame, company_features: pd.DataFrame | None):
    """
    Build the clustering feature matrix.

    When company_features is available, clustering uses ONLY company
    metadata (industry, size, employees, year founded) — volume features
    (30/90/180/360 day proportions) are excluded because they reflect
    data availability, not company identity.  Industry is weighted
    higher via INDUSTRY_WEIGHT so it dominates cluster formation.

    The volume pivot is still returned for PCHIP curve fitting.
    """
    pivot = df.pivot_table(
        index="company",
        columns="days_until",
        values="proportion",
        aggfunc="first",
    ).reindex(columns=TIME_BUCKETS)

    if company_features is not None:
        has_meta = company_features.index.intersection(pivot.index)
        clusterable = company_features.loc[has_meta].copy()
        for col in COMPANY_FEATURE_COLS:
            if col in clusterable.columns:
                clusterable[col] = clusterable[col].fillna(clusterable[col].median())
            else:
                clusterable[col] = 0.0

        sparse_names = set(pivot.index) - set(has_meta)
        sparse = pivot.loc[pivot.index.isin(sparse_names)]
    else:
        # No company metadata — fall back to volume-only clustering
        full_mask = pivot.notna().sum(axis=1) >= 2
        clusterable = pivot[full_mask].fillna(0)
        sparse = pivot[~full_mask]

    return clusterable, sparse, pivot



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


class ProblemCountPredictor:
    def __init__(self, csv_path: Path = AGGREGATED_PATH):
        self.df = _load_and_normalize(csv_path)
        self.company_features = _load_company_features()
        self._fit()

    def _fit(self):
        clusterable, sparse, pivot = _build_feature_matrix(self.df, self.company_features)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(clusterable.values)
        self._feature_columns = list(clusterable.columns)

        # Amplify industry so it dominates cluster formation
        if "industry_enc" in self._feature_columns:
            idx = self._feature_columns.index("industry_enc")
            X[:, idx] *= INDUSTRY_WEIGHT

        # Keep per-company scaled vectors for distance-ranked peer lookups
        self._company_vectors: dict[str, np.ndarray] = {
            company: X[i]
            for i, company in enumerate(clusterable.index)
        }

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


    def predict_cluster_for_unknown(self, meta: dict) -> int:
        """
        Assign an unknown company to the nearest existing cluster using
        its Kaggle features (industry, size_range, employee_count,
        year_founded).

        Returns the cluster ID.
        """
        from rag.company_lookup import encode_company_features

        feat_values = encode_company_features(meta, self._feature_columns, self.company_features)

        feat_scaled = self.scaler.transform([feat_values])

        if "industry_enc" in self._feature_columns:
            idx = self._feature_columns.index("industry_enc")
            feat_scaled[0, idx] *= INDUSTRY_WEIGHT

        distances = np.linalg.norm(self.km_model.cluster_centers_ - feat_scaled, axis=1)
        cluster_id = int(np.argmin(distances))

        # persist so future lookups are instant
        self._persist_company(meta, cluster_id)
        self.company_cluster[meta.get("company", "")] = cluster_id

        return cluster_id

    @staticmethod
    def _persist_company(meta: dict, cluster_id: int):
        """Append a newly resolved company to company_meta.csv."""
        row = {
            "company": meta.get("company", ""),
            "industry": meta.get("industry", ""),
            "size_range": meta.get("size_range", ""),
            "employee_count": meta.get("employee_count", 0),
            "year_founded": meta.get("year_founded", 0),
            "locality": meta.get("locality", ""),
        }
        row_df = pd.DataFrame([row])

        if COMPANY_META_PATH.exists():
            row_df.to_csv(COMPANY_META_PATH, mode="a", header=False, index=False)
        else:
            row_df.to_csv(COMPANY_META_PATH, index=False)

        logger.info(
            "Persisted unknown company '%s' -> cluster %d",
            row["company"], cluster_id,
        )


    def predict_counts(self, company: str, days_until: int) -> dict:
        """
        Predict how many easy, medium, hard problems to recommend.

        Returns {"easy": int, "medium": int, "hard": int, "total": int}.
        """
        days_until = max(1, min(days_until, 360))

        cluster_id = self.company_cluster.get(company)

        # Unknown company -> try Kaggle lookup
        if cluster_id is None:
            cluster_id = self._try_kaggle_lookup(company)

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

    def _try_kaggle_lookup(self, company: str) -> int | None:
        """Attempt to resolve an unknown company via the Kaggle dataset."""
        try:
            from rag.company_lookup import lookup_company
        except ImportError:
            return None

        meta = lookup_company(company)
        if meta is None:
            return None

        meta["company"] = company
        return self.predict_cluster_for_unknown(meta)


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

    def get_peer_company_names(self, company: str) -> list[str]:
        """
        Return names of companies in the same cluster, sorted by
        Euclidean distance to *company* in the standardized feature
        space (closest first).
        """
        cid = self.company_cluster.get(company)
        if cid is None:
            return []

        peers = [
            name for name, cluster in self.company_cluster.items()
            if cluster == cid and name != company
        ]

        ref_vec = self._company_vectors.get(company)
        if ref_vec is None:
            return peers

        def _distance(peer_name: str) -> float:
            peer_vec = self._company_vectors.get(peer_name)
            if peer_vec is None:
                return float("inf")
            return float(np.linalg.norm(ref_vec - peer_vec))

        peers.sort(key=_distance)
        return peers



_predictor = None


def get_predictor() -> ProblemCountPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ProblemCountPredictor()
    return _predictor
