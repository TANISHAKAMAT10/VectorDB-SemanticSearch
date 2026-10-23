"""Review 3 benchmark: Milvus FLAT exact search vs HNSW ANN.

Run from the project root after Milvus is running and both collections are loaded:
  $env:PYTHONPATH = (Get-Location).Path
  python scripts/benchmark.py --k 10 --repetitions 10
  python scripts/benchmark.py --k 10 --repetitions 10 --category sci.space --output results/benchmark_sci_space.csv

Recall@K is computed for every paired trial against FLAT exact results, then averaged.
Latency includes the Milvus search call (embedding generation is measured separately).
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
    """Nearest-rank percentile, deterministic and dependency-free."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((p / 100.0) * len(ordered)) - 1))
    return float(ordered[index])


def safe_filter(category):
    if not category:
        return ""
    escaped = category.replace("\\", "\\\\").replace('"', '\\"')
    return f'category == "{escaped}"'


def timed_search(client, collection, vector, k, hnsw, filter_expr):
    start = time.perf_counter()
    result = search_documents(
        client, collection, vector, k, filter_expr=filter_expr, hnsw=hnsw
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    ids = [int(item["id"]) for item in result]
    return ids, elapsed_ms


def recall_at_k(exact_ids, ann_ids):
    if not exact_ids:
        return 1.0 if not ann_ids else 0.0
    return len(set(exact_ids).intersection(ann_ids)) / len(set(exact_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="Top-K result count")
    parser.add_argument("--repetitions", type=int, default=10, help="Paired measured trials per query")
    parser.add_argument("--category", default="", help="Optional category filter, e.g. sci.space")
    parser.add_argument("--output", default="results/benchmark.csv", help="CSV output path")
    parser.add_argument("--queries-file", default="", help="Optional UTF-8 text file with one query per line")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [line.strip() for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not queries:
            parser.error("The query file contains no non-empty queries")

    client = get_client()
    filter_expr = safe_filter(args.category)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_flat_times, all_hnsw_times, all_recalls = [], [], []
    embedding_times = []

    for query_index, query in enumerate(queries, start=1):
        t0 = time.perf_counter()
        vector = encode_query(query)
        embedding_ms = (time.perf_counter() - t0) * 1000.0
        embedding_times.append(embedding_ms)

        # Warm-up calls are excluded from measured values.
        timed_search(client, FLAT_COLLECTION, vector, args.k, False, filter_expr)
        timed_search(client, COLLECTION, vector, args.k, True, filter_expr)

        flat_times, hnsw_times, recalls = [], [], []
        for trial in range(args.repetitions):
            # Alternate call order to reduce systematic order bias.
            if trial % 2 == 0:
                exact_ids, flat_ms = timed_search(client, FLAT_COLLECTION, vector, args.k, False, filter_expr)
                ann_ids, hnsw_ms = timed_search(client, COLLECTION, vector, args.k, True, filter_expr)
            else:
                ann_ids, hnsw_ms = timed_search(client, COLLECTION, vector, args.k, True, filter_expr)
                exact_ids, flat_ms = timed_search(client, FLAT_COLLECTION, vector, args.k, False, filter_expr)
            flat_times.append(flat_ms)
            hnsw_times.append(hnsw_ms)
            recalls.append(recall_at_k(exact_ids, ann_ids))

        all_flat_times.extend(flat_times)
        all_hnsw_times.extend(hnsw_times)
        all_recalls.extend(recalls)
        row = {
            "query_index": query_index,
            "query": query,
            "category_filter": args.category,
            "k": args.k,
            "repetitions": args.repetitions,
            "embedding_latency_ms": round(embedding_ms, 4),
            "flat_latency_median_ms": round(statistics.median(flat_times), 4),
            "flat_latency_mean_ms": round(statistics.mean(flat_times), 4),
            "flat_latency_p95_ms": round(percentile(flat_times, 95), 4),
            "hnsw_latency_median_ms": round(statistics.median(hnsw_times), 4),
            "hnsw_latency_mean_ms": round(statistics.mean(hnsw_times), 4),
            "hnsw_latency_p95_ms": round(percentile(hnsw_times, 95), 4),
            "recall_at_k_mean": round(statistics.mean(recalls), 6),
            "recall_at_k_min": round(min(recalls), 6),
            "flat_queries_per_second": round(1000.0 / statistics.mean(flat_times), 4) if statistics.mean(flat_times) else 0.0,
            "hnsw_queries_per_second": round(1000.0 / statistics.mean(hnsw_times), 4) if statistics.mean(hnsw_times) else 0.0,
        }
        rows.append(row)
        print(f"[{query_index}/{len(queries)}] Recall@{args.k}={row['recall_at_k_mean']:.3f}; FLAT median={row['flat_latency_median_ms']:.2f} ms; HNSW median={row['hnsw_latency_median_ms']:.2f} ms")

    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    flat_mean = statistics.mean(all_flat_times) if all_flat_times else 0.0
    hnsw_mean = statistics.mean(all_hnsw_times) if all_hnsw_times else 0.0
    summary = {
        "experiment": "Milvus FLAT exact search vs HNSW approximate search",
        "flat_collection": FLAT_COLLECTION,
        "hnsw_collection": COLLECTION,
        "queries_count": len(queries),
        "repetitions_per_query": args.repetitions,
        "k": args.k,
        "category_filter": args.category or None,
        "mean_embedding_latency_ms": round(statistics.mean(embedding_times), 4),
        "flat_latency_mean_ms": round(flat_mean, 4),
        "flat_latency_median_ms": round(statistics.median(all_flat_times), 4),
        "flat_latency_p95_ms": round(percentile(all_flat_times, 95), 4),
        "hnsw_latency_mean_ms": round(hnsw_mean, 4),
        "hnsw_latency_median_ms": round(statistics.median(all_hnsw_times), 4),
        "hnsw_latency_p95_ms": round(percentile(all_hnsw_times, 95), 4),
        "mean_recall_at_k": round(statistics.mean(all_recalls), 6),
        "flat_queries_per_second": round(1000.0 / flat_mean, 4) if flat_mean else 0.0,
        "hnsw_queries_per_second": round(1000.0 / hnsw_mean, 4) if hnsw_mean else 0.0,
        "hnsw_speedup_vs_flat": round(flat_mean / hnsw_mean, 4) if hnsw_mean else None,
        "raw_csv": str(output),
        "notes": "Latency measures the Milvus search call only; embedding latency is reported separately. Results depend on hardware, collection state, and server configuration.",
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved CSV: {output.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
