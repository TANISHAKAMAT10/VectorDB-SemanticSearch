from pydantic import BaseModel, Field
from typing import Optional


class DocumentCreate(BaseModel):
    id: int
    text_content: str = Field(min_length=1)
    source_title: str = "20 Newsgroups"
    category: str
    created_at: int


class DocumentUpdate(BaseModel):
    text_content: str = Field(min_length=1)
    source_title: str
    category: str
    created_at: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    category: Optional[str] = None


class SearchResult(BaseModel):
    id: int
    score: float
    text_content: str
    source_title: str
    category: str
    created_at: int
    token_count: int
