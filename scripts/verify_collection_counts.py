from pymilvus import MilvusClient
from app.config import MILVUS_URI, MILVUS_TOKEN
from collections import Counter

client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

try:
    for name in ["semantic_docs_hnsw", "semantic_docs_flat"]:
        print("\nCollection:", name)
        print("Statistics:", client.get_collection_stats(collection_name=name))

        ids = []
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
            ids.extend(int(row["id"]) for row in rows)
            if len(rows) < batch_size:
                break
            offset += len(rows)

        counts = Counter(ids)
        print("Queried records:", len(ids))
        print("Unique IDs:", len(counts))
        print("Repeated IDs:", [i for i, n in counts.items() if n > 1])

finally:
    client.close()
