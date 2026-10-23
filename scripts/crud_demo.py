from app.config import COLLECTION
from app.db import delete_document, get_client, get_document, upsert_document
from app.embedding import encode_documents


def main():
    client = get_client()
    doc_id = 999999
    text = "This temporary record is used to demonstrate CRUD operations in Milvus."
    record = {
        "id": doc_id,
        "text_content": text,
        "source_title": "CRUD Demo",
        "category": "demo",
        "created_at": 0,
        "token_count": len(text.split()),
        "embedding": encode_documents([text])[0],
    }

    print("CREATE/UPSERT:", upsert_document(client, COLLECTION, record))
    print("READ:", get_document(client, COLLECTION, doc_id))

    updated_text = "This updated temporary record demonstrates an upsert-based update in Milvus."
    record["text_content"] = updated_text
    record["token_count"] = len(updated_text.split())
    record["embedding"] = encode_documents([updated_text])[0]
    print("UPDATE:", upsert_document(client, COLLECTION, record))
    print("READ AFTER UPDATE:", get_document(client, COLLECTION, doc_id))

    print("DELETE:", delete_document(client, COLLECTION, doc_id))
    print("READ AFTER DELETE:", get_document(client, COLLECTION, doc_id))


if __name__ == "__main__":
    main()
