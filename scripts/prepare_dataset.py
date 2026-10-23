import argparse
import csv
import re
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Prepare a reproducible 20 Newsgroups semantic-search corpus.")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="data/semantic_docs.csv")
    args = parser.parse_args()

    data = fetch_20newsgroups(
        subset="train",
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    rows = []
    for i, (text, target) in enumerate(zip(data.data, data.target)):
        text = clean(text)
        if len(text) < 30:
            continue
        rows.append({
            "id": len(rows),
            "text_content": text[:65000],
            "source_title": "20 Newsgroups",
            "category": data.target_names[target],
            "created_at": 0,
        })
        if len(rows) >= args.limit:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} documents to {out}")


if __name__ == "__main__":
    main()
