import time
import uuid
import pandas as pd
from pymilvus import MilvusClient, DataType
from app.config import MILVUS_URI, MILVUS_TOKEN

client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
collection = "index_timing_" + uuid.uuid4().hex[:8]

try:
    df = pd.read_csv("data/semantic_docs.csv").head(1000)

    print("Creating temporary collection:", collection)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )

    client.create_collection(
        collection_name=collection,
        schema=schema
    )

    # Use real project embeddings so the index is built on actual vectors.
    from app.embedding import encode_documents
    texts = df["text_content"].astype(str).tolist()
    vectors = encode_documents(texts)

    records = [
        {"id": int(row_id), "embedding": vector}
        for row_id, vector in zip(df["id"], vectors)
    ]

    start = time.perf_counter()
    client.insert(collection_name=collection, data=records)
    client.flush(collection_name=collection)
    insert_flush_seconds = time.perf_counter() - start

    start = time.perf_counter()
    client.create_index(
        collection_name=collection,
        index_params=index_params
    )
    client.load_collection(collection_name=collection)

    # Wait until the index is ready before stopping the timer.
    while True:
        indexes = client.list_indexes(collection_name=collection)
        details = [
            client.describe_index(
                collection_name=collection,
                index_name=index_name
            )
            for index_name in indexes
        ]
        if details and all(
            str(detail.get("state", "")).upper() in ("FINISHED", "FINISH")
            for detail in details
        ):
            break
        time.sleep(1)

    index_load_seconds = time.perf_counter() - start

    print("\nRESULTS")
    print("Vectors indexed:", len(records))
    print(f"Insert + flush time: {insert_flush_seconds:.3f} seconds")
    print(f"Index creation + load + readiness time: {index_load_seconds:.3f} seconds")
    print("Note: the second timing includes load/readiness, not pure index-build time.")

finally:
    try:
        if client.has_collection(collection_name=collection):
            client.drop_collection(collection_name=collection)
            print("Temporary collection cleaned up.")
    finally:
        client.close()
