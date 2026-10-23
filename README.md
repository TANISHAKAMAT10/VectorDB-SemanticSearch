# Vector Database for Semantic Search — Review 2 Prototype

## Stack
- Python 3.10+
- Milvus 2.6.x/3.x standalone via Docker
- PyMilvus `MilvusClient`
- Sentence-Transformers `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
- FastAPI + Uvicorn
- scikit-learn 20 Newsgroups corpus

## 1. Start Milvus
Use the official Milvus standalone Docker Compose release file. Then:

```bash
docker compose up -d
docker compose ps
```

Milvus should be available at `http://localhost:19530`.

## 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 3. Prepare the sample corpus

```bash
python scripts/prepare_dataset.py --limit 10000
```

## 4. Build and ingest both indexes

```bash
python scripts/ingest.py --csv data/semantic_docs.csv --drop-existing
```

This creates two identical collections:
- `semantic_docs_hnsw` — approximate HNSW index
- `semantic_docs_flat` — exact FLAT baseline

## 5. Run CRUD demonstration

```bash
python scripts/crud_demo.py
```

## 6. Start API

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI or open `http://127.0.0.1:8000/` for the browser search page.

## 7. Benchmark exact vs ANN

```bash
python scripts/benchmark.py --k 10 --repetitions 3
```

Outputs:
- `results/benchmark.csv`
- `results/summary.json`

The benchmark uses FLAT as the exact reference and calculates Recall@10 for HNSW. Latency is measured after warm-up using `time.perf_counter()`.

## 8. Export database dump

```bash
python scripts/export_dump.py --output dump/semantic_docs.json
```

## API endpoints
- `GET /health`
- `POST /documents`
- `GET /documents/{id}`
- `PUT /documents/{id}`
- `DELETE /documents/{id}`
- `POST /search`
- `POST /search/exact` — FLAT exact-search baseline

Example search:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"space exploration and NASA missions","top_k":10}'
```

Filtered search:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"space exploration","top_k":10,"category":"sci.space"}'
```


## Review 2 vs Review 3 scope
- **Review 2:** database implementation, schema, CRUD, backend APIs, sample dataset, HNSW/FLAT index setup, metadata filtering, working prototype, database dump and documentation.
- **Review 3:** performance evaluation, scalability testing, HNSW parameter tuning, benchmarking, final performance analysis and presentation.

The benchmark script is included now so the final measurements can be produced in Review 3. Review 2 does not require final performance values.
