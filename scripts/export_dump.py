import argparse
import json
from pathlib import Path
from app.config import COLLECTION
from app.db import get_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--output", default="dump/semantic_docs.json")
    args = parser.parse_args()

    client = get_client()
    rows = client.query(
        collection_name=args.collection,
        filter="",
        output_fields=["*"],
        limit=16384,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(rows)} records to {out}")


if __name__ == "__main__":
    main()
