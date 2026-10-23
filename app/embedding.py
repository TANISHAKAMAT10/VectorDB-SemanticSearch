from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import MODEL_NAME


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def encode_documents(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.astype("float32").tolist()


def encode_query(text: str) -> list[float]:
    model = get_model()
    vector = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return vector.astype("float32").tolist()
