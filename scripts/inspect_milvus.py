from pymilvus import MilvusClient
from app.config import MILVUS_URI, MILVUS_TOKEN, COLLECTION, FLAT_COLLECTION

client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

print("MILVUS URI:", MILVUS_URI)
print("Configured HNSW collection:", COLLECTION)
print("Configured FLAT collection:", FLAT_COLLECTION)

for name in [COLLECTION, FLAT_COLLECTION]:
    print("\n" + "=" * 60)
    print("COLLECTION:", name)

    try:
        if not client.has_collection(collection_name=name):
            print("Collection does not exist.")
            continue

        print("Collection description:")
        print(client.describe_collection(collection_name=name))

        print("\nCollection statistics:")
        print(client.get_collection_stats(collection_name=name))

        print("\nIndex descriptions:")
        print(client.list_indexes(collection_name=name))

        for index_name in client.list_indexes(collection_name=name):
            try:
                print(f"Details for index {index_name}:")
                print(client.describe_index(
                    collection_name=name,
                    index_name=index_name
                ))
            except Exception as e:
                print("Could not describe index:", e)

    except Exception as e:
        print("Inspection error:", repr(e))

client.close()
