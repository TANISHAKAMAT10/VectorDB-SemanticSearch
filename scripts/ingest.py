import argparse
import csv
import time
from pathlib import Path

from app.config import COLLECTION, FLAT_COLLECTION
from app.db import ensure_collections, get_client, insert_documents
from app.embedding import encode_documents


def load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def prepare_records(rows, vectors):
    records = []
    for row, vector in zip(rows, vectors):
        records.append({
            "id": int(row["id"]),
            "text_content": row["text_content"],
            "source_title": row["source_title"],
            "category": row["category"],
            "created_at": int(row.get("created_at", 0)),
            "token_count": len(row["text_content"].split()),
            "embedding": vector,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/semantic_docs.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows")
    ensure_collections(drop_existing=args.drop_existing)
    client = get_client()

    for batch_no, batch in enumerate(chunks(rows, args.batch_size), start=1):
        t0 = time.perf_counter()
        texts = [r["text_content"] for r in batch]
        vectors = encode_documents(texts)
        records = prepare_records(batch, vectors)
        insert_documents(client, COLLECTION, records)
        insert_documents(client, FLAT_COLLECTION, records)
        elapsed = time.perf_counter() - t0
        print(f"batch={batch_no} rows={len(batch)} elapsed={elapsed:.2f}s")

    print(f"Ingested {len(rows)} records into {COLLECTION} and {FLAT_COLLECTION}")


if __name__ == "__main__":
    main()
