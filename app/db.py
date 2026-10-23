from pymilvus import DataType, MilvusClient
from app.config import (
    COLLECTION,
    EMBEDDING_DIM,
    FLAT_COLLECTION,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    MILVUS_TOKEN,
    MILVUS_URI,
)

FIELDS = ["id", "text_content", "source_title", "category", "created_at", "token_count"]


def get_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


def build_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text_content", DataType.VARCHAR, max_length=65535)
    schema.add_field("source_title", DataType.VARCHAR, max_length=256)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("token_count", DataType.INT64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    return schema


def build_index_params(client: MilvusClient, index_type: str):
    index_params = client.prepare_index_params()
    if index_type == "HNSW":
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_hnsw",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
        )
    elif index_type == "FLAT":
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_flat",
            index_type="FLAT",
            metric_type="COSINE",
            params={},
        )
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index_params.add_index(field_name="category", index_name="category_inverted", index_type="INVERTED")
    return index_params


def create_collection(name: str, index_type: str, drop_existing: bool = False):
    client = get_client()
    if client.has_collection(collection_name=name):
        if drop_existing:
            client.drop_collection(collection_name=name)
        else:
            return client

    schema = build_schema(client)
    index_params = build_index_params(client, index_type)
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    return client


def ensure_collections(drop_existing: bool = False):
    create_collection(COLLECTION, "HNSW", drop_existing)
    create_collection(FLAT_COLLECTION, "FLAT", drop_existing)


def insert_documents(client: MilvusClient, collection_name: str, records: list[dict]):
    return client.insert(collection_name=collection_name, data=records)


def upsert_document(client: MilvusClient, collection_name: str, record: dict):
    return client.upsert(collection_name=collection_name, data=[record])


def get_document(client: MilvusClient, collection_name: str, doc_id: int):
    return client.get(collection_name=collection_name, ids=[doc_id], output_fields=FIELDS)


def delete_document(client: MilvusClient, collection_name: str, doc_id: int):
    return client.delete(collection_name=collection_name, ids=[doc_id])


def search_documents(
    client: MilvusClient,
    collection_name: str,
    vector: list[float],
    limit: int,
    filter_expr: str = "",
    hnsw: bool = True,
):
    params = {"params": {"ef": HNSW_EF_SEARCH}} if hnsw else {"params": {}}
    result = client.search(
        collection_name=collection_name,
        data=[vector],
        anns_field="embedding",
        limit=limit,
        filter=filter_expr,
        output_fields=FIELDS[1:],
        search_params=params,
    )
    return result[0]
