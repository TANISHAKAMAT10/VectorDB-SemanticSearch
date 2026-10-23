"""Review 3: HNSW query-time ef tuning against FLAT exact search.

Run from project root with Milvus running and both collections populated:
  $env:PYTHONPATH = (Get-Location).Path
  python scripts/tune_hnsw.py --k 10 --repetitions 10 --ef-values 16,32,64,128,200

This script does not rebuild or modify indexes. It temporarily passes each ef value
to Milvus search for the HNSW collection and compares results against FLAT.
"""
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from app.config import COLLECTION, FLAT_COLLECTION
from app.db import get_client, search_documents
from app.embedding import encode_query

DEFAULT_QUERIES = [
    "computer graphics rendering and image processing",
    "space exploration and NASA missions",
    "motorcycle maintenance and repair",
    "medical treatment and health conditions",
    "baseball teams and season statistics",
    "computer hardware components and upgrades",
    "cryptography and secure communication",
    "car engines and automobile performance",
    "religious beliefs and Christian theology",
    "electronic circuits and electrical components",
]


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return float(ordered[idx])


def safe_filter(category):
    if not category:
        return ""
    escaped = category.replace("\\", "\\\\").replace('"', '\\"')
    return f'category == "{escaped}"'


def flat_search(client, vector, k, filter_expr):
    start = time.perf_counter()
    result = search_documents(
        client, FLAT_COLLECTION, vector, k,
        filter_expr=filter_expr, hnsw=False
    )
    return [int(x["id"]) for x in result], (time.perf_counter() - start) * 1000


def hnsw_search_with_ef(client, vector, k, filter_expr, ef):
    """Call Milvus directly so each trial uses the requested query-time ef."""
    start = time.perf_counter()
    response = client.search(
        collection_name=COLLECTION,
        data=[vector],
        anns_field="embedding",
        limit=k,
        filter=filter_expr or "",
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": max(ef, k)},
        },
        output_fields=["category"],
    )
    elapsed = (time.perf_counter() - start) * 1000
    hits = response[0] if response else []
    ids = [int(hit.get("id", hit.get("pk"))) for hit in hits]
    return ids, elapsed


def recall_at_k(exact_ids, ann_ids):
    if not exact_ids:
        return 1.0 if not ann_ids else 0.0
    return len(set(exact_ids) & set(ann_ids)) / len(set(exact_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--ef-values", default="16,32,64,128,200")
    parser.add_argument("--category", default="")
    parser.add_argument("--output", default="results/hnsw_tuning.csv")
    parser.add_argument("--queries-file", default="")
    args = parser.parse_args()

    if args.k < 1 or args.repetitions < 1:
        parser.error("--k and --repetitions must be positive")
    try:
        ef_values = sorted(set(int(x.strip()) for x in args.ef_values.split(",") if x.strip()))
    except ValueError:
        parser.error("--ef-values must be comma-separated integers")
    if not ef_values or any(x < args.k for x in ef_values):
        parser.error("Each ef value must be >= --k")

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [
            line.strip()
            for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not queries:
            parser.error("Query file contains no non-empty queries")

    client = get_client()
    filter_expr = safe_filter(args.category)
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Embed once per query; embedding cost is not part of Milvus search latency.
    vectors = []
    embedding_ms = []
    for query in queries:
        start = time.perf_counter()
        vectors.append(encode_query(query))
        embedding_ms.append((time.perf_counter() - start) * 1000)

    # Warm up both paths before collecting measurements.
    for vector in vectors[:2]:
        flat_search(client, vector, args.k, filter_expr)
        for ef in ef_values:
            hnsw_search_with_ef(client, vector, args.k, filter_expr, ef)

    for ef in ef_values:
        flat_times, hnsw_times, recalls = [], [], []
        for qi, (query, vector) in enumerate(zip(queries, vectors), start=1):
            for trial in range(args.repetitions):
                # Alternate order to reduce systematic order bias.
                if trial % 2 == 0:
                    exact_ids, flat_ms = flat_search(client, vector, args.k, filter_expr)
                    ann_ids, hnsw_ms = hnsw_search_with_ef(
                        client, vector, args.k, filter_expr, ef
                    )
                else:
                    ann_ids, hnsw_ms = hnsw_search_with_ef(
                        client, vector, args.k, filter_expr, ef
                    )
                    exact_ids, flat_ms = flat_search(client, vector, args.k, filter_expr)
                flat_times.append(flat_ms)
                hnsw_times.append(hnsw_ms)
                recalls.append(recall_at_k(exact_ids, ann_ids))

        flat_mean = statistics.mean(flat_times)
        hnsw_mean = statistics.mean(hnsw_times)
        row = {
            "ef": ef,
            "k": args.k,
            "queries_count": len(queries),
            "repetitions_per_query": args.repetitions,
            "category_filter": args.category or "",
            "flat_mean_latency_ms": round(flat_mean, 4),
            "flat_median_latency_ms": round(statistics.median(flat_times), 4),
            "flat_p95_latency_ms": round(percentile(flat_times, 95), 4),
            "hnsw_mean_latency_ms": round(hnsw_mean, 4),
            "hnsw_median_latency_ms": round(statistics.median(hnsw_times), 4),
            "hnsw_p95_latency_ms": round(percentile(hnsw_times, 95), 4),
            "mean_recall_at_k": round(statistics.mean(recalls), 6),
            "min_recall_at_k": round(min(recalls), 6),
            "hnsw_queries_per_second": round(1000 / hnsw_mean, 4) if hnsw_mean else 0,
            "speedup_vs_flat": round(flat_mean / hnsw_mean, 4) if hnsw_mean else None,
        }
        rows.append(row)
        print(
            f"ef={ef:>3} | Recall@{args.k}={row['mean_recall_at_k']:.3f} "
            f"| FLAT mean={flat_mean:.2f} ms | HNSW mean={hnsw_mean:.2f} ms "
            f"| speedup={row['speedup_vs_flat']:.2f}x"
        )

    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "HNSW query-time ef tuning vs FLAT exact search",
        "hnsw_collection": COLLECTION,
        "flat_collection": FLAT_COLLECTION,
        "queries_count": len(queries),
        "repetitions_per_query": args.repetitions,
        "k": args.k,
        "ef_values": ef_values,
        "category_filter": args.category or None,
        "mean_embedding_latency_ms": round(statistics.mean(embedding_ms), 4),
        "results": rows,
        "notes": "Search latency excludes embedding. Existing indexes were not rebuilt. Results are hardware and dataset dependent.",
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved CSV: {output.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
