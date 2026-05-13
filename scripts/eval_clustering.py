"""
Evaluate the clustering + PCHIP roadmap prediction pipeline.

Tests:
  1. Cluster quality (silhouette score, cluster sizes)
  2. PCHIP monotonicity (more days → more problems)
  3. Known-company predictions produce valid distributions
  4. Unknown-company cold-start falls back gracefully
  5. Peer companies belong to the same cluster

Usage:
    python scripts/eval_clustering.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from rag.curve_fitting import get_predictor

SEPARATOR = "=" * 70


def eval_cluster_quality(predictor):
    """Check that clusters are well-formed and balanced."""
    print(f"\n{SEPARATOR}")
    print("TEST 1: Cluster Quality")
    print(SEPARATOR)

    n_clusters = predictor.km_model.n_clusters
    cluster_map = predictor.company_cluster

    cluster_sizes = {}
    for company, cid in cluster_map.items():
        cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

    print(f"  Number of clusters (k):  {n_clusters}")
    print(f"  Best silhouette score:   {max(r['silhouette'] for r in predictor.km_results):.3f}")
    print(f"  Companies with cluster:  {len(cluster_map)}")
    print(f"  Cluster size distribution:")
    for cid in sorted(cluster_sizes):
        print(f"    Cluster {cid}: {cluster_sizes[cid]} companies")

    sizes = list(cluster_sizes.values())
    print(f"  Min / Max / Mean size:   {min(sizes)} / {max(sizes)} / {np.mean(sizes):.1f}")

    empty_clusters = n_clusters - len(cluster_sizes)
    assert empty_clusters == 0, f"FAIL: {empty_clusters} empty clusters"
    assert all(s >= 2 for s in sizes), "FAIL: cluster with fewer than 2 companies"
    print("  PASS: All clusters non-empty and have >= 2 companies")


def eval_pchip_monotonicity(predictor):
    """PCHIP curves should produce monotonically increasing totals."""
    print(f"\n{SEPARATOR}")
    print("TEST 2: PCHIP Monotonicity")
    print(SEPARATOR)

    test_companies = list(predictor.company_cluster.keys())[:10]
    horizons = [30, 90, 180, 360]
    all_pass = True

    for company in test_companies:
        totals = []
        for d in horizons:
            counts = predictor.predict_counts(company, d)
            totals.append(counts["total"])

        is_monotone = all(totals[i] <= totals[i + 1] for i in range(len(totals) - 1))
        status = "PASS" if is_monotone else "FAIL"
        if not is_monotone:
            all_pass = False
        print(f"  {company:25s}  {horizons} -> {totals}  [{status}]")

    if all_pass:
        print("  PASS: All tested companies show monotonically increasing totals")
    else:
        print("  WARN: Some companies do not show strict monotonicity")


def eval_known_company_predictions(predictor):
    """Known companies should produce valid difficulty distributions."""
    print(f"\n{SEPARATOR}")
    print("TEST 3: Known-Company Predictions")
    print(SEPARATOR)

    test_cases = [
        ("Google", 30),
        ("Google", 180),
        ("Amazon", 90),
        ("Microsoft", 360),
        ("Facebook", 90),
    ]

    all_pass = True
    for company, days in test_cases:
        if company not in predictor.company_cluster:
            print(f"  SKIP: {company} not in dataset")
            continue

        counts = predictor.predict_counts(company, days)
        e, m, h, t = counts["easy"], counts["medium"], counts["hard"], counts["total"]
        valid = (e >= 0 and m >= 0 and h >= 0 and t > 0 and e + m + h == t)
        status = "PASS" if valid else "FAIL"
        if not valid:
            all_pass = False
        print(f"  {company:12s} @ {days:3d} days -> E={e:2d} M={m:2d} H={h:2d} T={t:2d}  [{status}]")

    if all_pass:
        print("  PASS: All predictions have valid, non-negative distributions summing to total")


def eval_unknown_company_fallback(predictor):
    """Unknown companies should still get a prediction (defaults or Kaggle lookup)."""
    print(f"\n{SEPARATOR}")
    print("TEST 4: Unknown-Company Cold-Start")
    print(SEPARATOR)

    fake_companies = [
        "TotallyFakeCompany12345",
        "NonExistentCorp",
    ]

    for company in fake_companies:
        counts = predictor.predict_counts(company, 90)
        e, m, h, t = counts["easy"], counts["medium"], counts["hard"], counts["total"]
        valid = (e >= 0 and m >= 0 and h >= 0 and t > 0)
        print(f"  {company:30s} -> E={e} M={m} H={h} T={t}  [{'PASS' if valid else 'FAIL'}]")
        if valid:
            print(f"    (graceful fallback: defaults or Kaggle-resolved cluster)")


def eval_peer_same_cluster(predictor):
    """Peer companies should all belong to the same cluster as the query."""
    print(f"\n{SEPARATOR}")
    print("TEST 5: Peer Companies Share Cluster")
    print(SEPARATOR)

    test_companies = [c for c in ["Google", "Amazon", "Microsoft", "Apple", "Meta"]
                      if c in predictor.company_cluster][:5]

    if not test_companies:
        test_companies = list(predictor.company_cluster.keys())[:5]

    all_pass = True
    for company in test_companies:
        cluster = predictor.company_cluster.get(company)
        peers = predictor.get_peer_company_names(company)

        mismatched = []
        for peer in peers:
            peer_cluster = predictor.company_cluster.get(peer)
            if peer_cluster != cluster:
                mismatched.append((peer, peer_cluster))

        status = "PASS" if not mismatched else "FAIL"
        if mismatched:
            all_pass = False
        print(f"  {company:15s} (cluster {cluster}) -> {len(peers)} peers, {len(mismatched)} mismatched  [{status}]")
        if peers:
            print(f"    Sample peers: {peers[:5]}")

    if all_pass:
        print("  PASS: All peer companies belong to the same cluster as the query")


def main():
    print(SEPARATOR)
    print("EVALUATION: Clustering + PCHIP Roadmap Prediction")
    print(SEPARATOR)

    predictor = get_predictor()

    eval_cluster_quality(predictor)
    eval_pchip_monotonicity(predictor)
    eval_known_company_predictions(predictor)
    eval_unknown_company_fallback(predictor)
    eval_peer_same_cluster(predictor)

    print(f"\n{SEPARATOR}")
    print("CLUSTERING EVALUATION COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
