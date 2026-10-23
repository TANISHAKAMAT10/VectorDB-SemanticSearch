from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from app.config import COLLECTION, FLAT_COLLECTION
from app.db import delete_document, get_client, get_document, search_documents, upsert_document
from app.embedding import encode_query, encode_documents
from app.schemas import DocumentCreate, DocumentUpdate, SearchRequest, SearchResult

app = FastAPI(
    title="Vector Database Semantic Search API",
    version="1.0.0",
    description="Review-2 prototype using Sentence-Transformers and Milvus HNSW.",
)

client = get_client()


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse("static/index.html")


def _filter_for_category(category: str | None) -> str:
    if not category:
        return ""
    safe = category.replace('\\', '\\\\').replace('"', '\\"')
    return f'category == "{safe}"'


@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION}


@app.post("/documents", status_code=status.HTTP_201_CREATED)
def create_document(doc: DocumentCreate):
    vector = encode_documents([doc.text_content])[0]
    record = doc.model_dump() | {
        "embedding": vector,
        "token_count": len(doc.text_content.split()),
    }
    result = upsert_document(client, COLLECTION, record)
    return {"message": "document upserted", "result": str(result)}


@app.get("/documents/{doc_id}")
def read_document(doc_id: int):
    rows = get_document(client, COLLECTION, doc_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")
    return rows[0]


@app.put("/documents/{doc_id}")
def update_document(doc_id: int, doc: DocumentUpdate):
    if not get_document(client, COLLECTION, doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    vector = encode_documents([doc.text_content])[0]
    record = {"id": doc_id, **doc.model_dump(), "embedding": vector, "token_count": len(doc.text_content.split())}
    result = upsert_document(client, COLLECTION, record)
    return {"message": "document updated", "result": str(result)}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: int):
    if not get_document(client, COLLECTION, doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    result = delete_document(client, COLLECTION, doc_id)
    return {"message": "document deleted", "result": str(result)}


@app.post("/search", response_model=list[SearchResult])
def semantic_search(request: SearchRequest):
    vector = encode_query(request.query)
    results = search_documents(
        client,
        COLLECTION,
        vector,
        request.top_k,
        _filter_for_category(request.category),
        hnsw=True,
    )
    return [
        SearchResult(
            id=int(r["id"]),
            score=float(r["distance"]),
            text_content=r.get("entity", {}).get("text_content", r.get("text_content", "")),
            source_title=r.get("entity", {}).get("source_title", r.get("source_title", "")),
            category=r.get("entity", {}).get("category", r.get("category", "")),
            created_at=int(r.get("entity", {}).get("created_at", r.get("created_at", 0))),
            token_count=int(r.get("entity", {}).get("token_count", r.get("token_count", 0))),
        )
        for r in results
    ]


@app.post("/search/exact", response_model=list[SearchResult])
def exact_semantic_search(request: SearchRequest):
    """Exact top-k retrieval from the FLAT collection; useful as a reference baseline."""
    vector = encode_query(request.query)
    results = search_documents(
        client,
        FLAT_COLLECTION,
        vector,
        request.top_k,
        _filter_for_category(request.category),
        hnsw=False,
    )
    return [
        SearchResult(
            id=int(r["id"]),
            score=float(r["distance"]),
            text_content=r.get("entity", {}).get("text_content", r.get("text_content", "")),
            source_title=r.get("entity", {}).get("source_title", r.get("source_title", "")),
            category=r.get("entity", {}).get("category", r.get("category", "")),
            created_at=int(r.get("entity", {}).get("created_at", r.get("created_at", 0))),
            token_count=int(r.get("entity", {}).get("token_count", r.get("token_count", 0))),
        )
        for r in results
    ]
