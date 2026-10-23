import pandas as pd
from pymilvus import MilvusClient
from app.config import MILVUS_URI, MILVUS_TOKEN

client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
csv_df = pd.read_csv("data/semantic_docs.csv")
csv_ids = set(csv_df["id"].astype(int))

try:
    for name in ["semantic_docs_hnsw", "semantic_docs_flat"]:
        print("\n" + "=" * 55)
        print("Collection:", name)

        all_rows = []
        offset = 0
        batch_size = 5000

        while True:
            rows = client.query(
                collection_name=name,
                filter="",
                output_fields=["id"],
                limit=batch_size,
                offset=offset
            )

            all_rows.extend(rows)

            if len(rows) < batch_size:
                break

            offset += len(rows)

        db_ids = {int(row["id"]) for row in all_rows}

        print("Rows fetched:", len(all_rows))
        print("Unique database IDs:", len(db_ids))
        print("IDs missing from database:", len(csv_ids - db_ids))
        print("IDs not present in CSV:", sorted(db_ids - csv_ids)[:30])
        print("Duplicate IDs returned:", len(all_rows) - len(db_ids))

finally:
    client.close()
