"""Review 3 scalability experiment using independent Milvus collections.

This script does NOT delete or alter the existing semantic_docs_hnsw / semantic_docs_flat
collections. It creates temporary experiment collections for each requested corpus size,
loads a deterministic subset of data/semantic_docs.csv, benchmarks FLAT and HNSW, then
drops only the temporary collections it created.

Run from project root:
  $env:PYTHONPATH = (Get-Location).Path
  python scripts/scalability_review3.py --sizes 1000,2500,5000,10000 --k 10 --repetitions 5

Expected CSV columns include corpus size, index type, latency, throughput and HNSW recall
against FLAT. If the dataset schema differs, the script stops with a helpful error.
"""
import argparse
import csv
import json
import math
import statistics
import time
import uuid
from pathlib import Path

import pandas as pd
from pymilvus import DataType, MilvusClient

from app.config import MILVUS_URI, MILVUS_TOKEN
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
    i = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return float(ordered[i])


def pick_column(columns, candidates, purpose):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Cannot identify the {purpose} column in CSV. Columns found: {list(columns)}. "
        f"Expected one of: {candidates}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/semantic_docs.csv")
    parser.add_argument("--sizes", default="1000,2500,5000,10000")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", default="results/scalability.csv")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary experiment collections for inspection")
    args = parser.parse_args()

    sizes = sorted(set(int(x.strip()) for x in args.sizes.split(",") if x.strip()))
    if not sizes or any(n < args.k for n in sizes):
        parser.error("Every --sizes value must be >= --k")
    if args.k < 1 or args.repetitions < 1:
        parser.error("--k and --repetitions must be positive")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if df.empty:
        parser.error("Dataset is empty")

    text_col = pick_column(
    df.columns,
    ["text_content", "text", "document", "content", "body", "message", "title"],
    "text"
    )
    category_col = next((c for c in ["category", "label", "target", "newsgroup"] if c in df.columns), None)
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip().ne("")].reset_index(drop=True)
    max_size = max(sizes)
    if len(df) < max_size:
        parser.error(
            f"Requested maximum size {max_size}, but dataset contains only {len(df)} usable rows. "
            "Reduce --sizes or provide more unique records."
        )

    # Deterministic sample/order so runs are repeatable.
    df = df.head(max_size).copy()
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    suffix = uuid.uuid4().hex[:8]
    temp_names = []
    rows = []

    try:
        # Encode each document once, then use nested prefixes for comparable corpus sizes.
        print(f"Embedding {len(df)} documents from {dataset_path}...")
        texts = df[text_col].tolist()
        vectors = []
        batch_size = 128
        for start in range(0, len(texts), batch_size):
            vectors.extend(encode_query(texts[i]) for i in range(start, min(start + batch_size, len(texts))))
            if start % 1024 == 0:
                print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")
        dim = len(vectors[0])
        queries = [encode_query(q) for q in DEFAULT_QUERIES]

        for size in sizes:
            sample = df.iloc[:size].reset_index(drop=True)
            records = []
            for i in range(size):
                rec = {"id": int(i + 1), "embedding": vectors[i]}
                if category_col:
                    rec["category"] = str(sample.iloc[i][category_col])
                rec["content"] = str(sample.iloc[i][text_col])[:6000]
                records.append(rec)

            for index_type in ["FLAT", "HNSW"]:
                collection = f"r3scale_{index_type.lower()}_{size}_{suffix}"
                temp_names.append(collection)
                schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
                if category_col:
                    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=256)
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=6000)
                index_params = client.prepare_index_params()
                if index_type == "HNSW":
                    index_params.add_index(
                        field_name="embedding", index_name="embedding_hnsw",
                        index_type="HNSW", metric_type="COSINE",
                        params={"M": 16, "efConstruction": 200}
                    )
                else:
                    index_params.add_index(
                        field_name="embedding", index_name="embedding_flat",
                        index_type="FLAT", metric_type="COSINE", params={}
                    )
                if category_col:
                    index_params.add_index(
                        field_name="category", index_name="category_inverted",
                        index_type="INVERTED"
                    )
                client.create_collection(
                    collection_name=collection, schema=schema, index_params=index_params
                )
                for start in range(0, size, 500):
                    client.insert(collection_name=collection, data=records[start:start + 500])
                client.load_collection(collection_name=collection)

                # Warm-up each query once, then measure repeated calls.
                latencies = []
                result_ids_by_query = []
                ef = max(64, args.k)
                for qv in queries:
                    params = {"metric_type": "COSINE", "params": ({"ef": ef} if index_type == "HNSW" else {})}
                    client.search(
                        collection_name=collection, data=[qv], anns_field="embedding",
                        limit=args.k, search_params=params, output_fields=[]
                    )
                for qv in queries:
                    q_latencies, last_ids = [], []
                    for _ in range(args.repetitions):
                        params = {"metric_type": "COSINE", "params": ({"ef": ef} if index_type == "HNSW" else {})}
                        t0 = time.perf_counter()
                        response = client.search(
                            collection_name=collection, data=[qv], anns_field="embedding",
                            limit=args.k, search_params=params, output_fields=[]
                        )
                        q_latencies.append((time.perf_counter() - t0) * 1000)
                        hits = response[0] if response else []
                        last_ids = [int(h.get("id", h.get("pk"))) for h in hits]
                    latencies.extend(q_latencies)
                    result_ids_by_query.append(last_ids)

                row = {
                    "corpus_size": size,
                    "index_type": index_type,
                    "k": args.k,
                    "queries_count": len(queries),
                    "repetitions_per_query": args.repetitions,
                    "hnsw_ef": ef if index_type == "HNSW" else "",
                    "mean_latency_ms": round(statistics.mean(latencies), 4),
                    "median_latency_ms": round(statistics.median(latencies), 4),
                    "p95_latency_ms": round(percentile(latencies, 95), 4),
                    "throughput_qps": round(1000 / statistics.mean(latencies), 4),
                    "mean_recall_at_k": "",
                }
                rows.append(row)
                print(f"size={size:>6} index={index_type:<4} mean={row['mean_latency_ms']:.2f} ms median={row['median_latency_ms']:.2f} ms")
                # Store the result IDs temporarily on the row for later recall calculation.
                row["_result_ids"] = result_ids_by_query

        # Compute HNSW recall against FLAT at the same size/query.
        lookup = {(r["corpus_size"], r["index_type"]): r for r in rows}
        for size in sizes:
            flat_ids = lookup[(size, "FLAT")]["_result_ids"]
            hnsw_ids = lookup[(size, "HNSW")]["_result_ids"]
            recalls = []
            for exact, approx in zip(flat_ids, hnsw_ids):
                recalls.append(len(set(exact) & set(approx)) / len(set(exact)) if exact else 1.0)
            lookup[(size, "HNSW")]["mean_recall_at_k"] = round(statistics.mean(recalls), 6)
            lookup[(size, "FLAT")]["mean_recall_at_k"] = 1.0
        clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(clean_rows[0].keys()))
            writer.writeheader()
            writer.writerows(clean_rows)
        summary = {
            "experiment": "Scalability: FLAT vs HNSW across corpus sizes",
            "dataset": str(dataset_path),
            "usable_dataset_rows": len(df),
            "sizes": sizes,
            "k": args.k,
            "repetitions_per_query": args.repetitions,
            "hnsw_query_ef": max(64, args.k),
            "results": clean_rows,
            "notes": "Temporary collections only; original project collections are not modified. Search latency excludes embedding.",
        }
        summary_path = output.with_name(output.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved CSV: {output.resolve()}")
        print(f"Saved summary: {summary_path.resolve()}")
    finally:
        if not args.keep_temp:
            for name in temp_names:
                try:
                    if client.has_collection(collection_name=name):
                        client.drop_collection(collection_name=name)
                except Exception as exc:
                    print(f"WARNING: could not clean temporary collection {name}: {exc}")
            print("Temporary experiment collections cleaned up; existing project collections were left untouched.")


if __name__ == "__main__":
    main()
