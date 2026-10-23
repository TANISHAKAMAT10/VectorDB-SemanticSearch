import argparse
import csv
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

from app.config import COLLECTION, FLAT_COLLECTION
from app.db import get_client, search_documents
from app.embedding import encode_query

QUERIES = [
    "computer graphics rendering and image processing",
    "space exploration and NASA missions",
    "motorcycle maintenance and repair",
    "medical treatment and health conditions",
    "computer hardware components and upgrades",
    "cryptography and secure communication",
    "car engines and automobile performance",
    "electronic circuits and electrical components",
]
thread_state = local()


def percentile(values, p):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[index]


def get_thread_client():
    if not hasattr(thread_state, "client"):
        thread_state.client = get_client()
    return thread_state.client


def one_request(collection, is_hnsw, request_index, vectors, references, k):
    client = get_thread_client()
    query_index = request_index % len(vectors)
    start = time.perf_counter()
    try:
        results = search_documents(
            client, collection, vectors[query_index], k, hnsw=is_hnsw
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        found = {int(item["id"]) for item in results}
        reference = references[query_index]
        recall = len(reference & found) / max(1, len(reference))
        return elapsed_ms, recall, None
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, None, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="1,2,5,10")
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="results/concurrency_benchmark.csv")
    args = parser.parse_args()

    workers_list = [int(x) for x in args.workers.split(",")]
    if args.requests < 1 or args.k < 1 or any(w < 1 for w in workers_list):
        parser.error("Requests, k, and worker counts must be positive")

    print("Encoding test queries...")
    vectors = [encode_query(q) for q in QUERIES]

    client = get_client()
    try:
        references = []
        for vector in vectors:
            results = search_documents(
                client, FLAT_COLLECTION, vector, args.k, hnsw=False
            )
            references.append({int(item["id"]) for item in results})
    finally:
        client.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for collection, label, is_hnsw in [
        (FLAT_COLLECTION, "FLAT", False),
        (COLLECTION, "HNSW", True),
    ]:
        for workers in workers_list:
            warm_client = get_client()
            try:
                search_documents(
                    warm_client, collection, vectors[0], args.k,
                    hnsw=is_hnsw
                )
            finally:
                warm_client.close()

            print(
                f"Testing {label}: workers={workers}, "
                f"requests={args.requests}"
            )
            start_all = time.perf_counter()

            # Correct: enforce this specific worker count for this test.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        one_request, collection, is_hnsw, i,
                        vectors, references, args.k
                    )
                    for i in range(args.requests)
                ]
                outcomes = [
                    future.result() for future in as_completed(futures)
                ]

            wall_seconds = time.perf_counter() - start_all
            successful = [o for o in outcomes if o[2] is None]
            failures = [o for o in outcomes if o[2] is not None]
            latencies = [o[0] for o in successful]
            recalls = [o[1] for o in successful]

            for failure in failures[:3]:
                print("Request error:", failure[2])

            row = {
                "index": label,
                "workers": workers,
                "requests_submitted": args.requests,
                "successful_requests": len(successful),
                "failed_requests": len(failures),
                "wall_time_seconds": round(wall_seconds, 4),
                "successful_throughput_qps": round(
                    len(successful) / wall_seconds, 4
                ) if wall_seconds else 0,
                "mean_latency_ms": round(
                    statistics.mean(latencies), 4
                ) if latencies else None,
                "p95_latency_ms": round(
                    percentile(latencies, 95), 4
                ) if latencies else None,
                "mean_recall_at_k": round(
                    statistics.mean(recalls), 6
                ) if recalls else None,
            }
            rows.append(row)
            print(row)

    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output.resolve())


if __name__ == "__main__":
    main()
