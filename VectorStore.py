"""FAISS-backed vector store for the product catalog. Cosine similarity via
L2-normalized embeddings + inner-product index — both the stub and CLIP
embedders already return normalized vectors, so this doesn't re-normalize
(re-normalizing would silently mask a bug in an embedder that forgot to)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CatalogItem:
    item_id: int
    filename: str
    title: str


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.items: list[CatalogItem] = []

    def add(self, embeddings: np.ndarray, items: list[CatalogItem]) -> None:
        if len(embeddings) != len(items):
            raise ValueError(f"Got {len(embeddings)} embeddings but {len(items)} items")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {embeddings.shape[1]} doesn't match index dim {self.dim}")

        self.index.add(embeddings.astype(np.float32))
        self.items.extend(items)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]:
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.items[idx], float(score)))
        return results

    def save(self, index_dir: str | Path, index_name: str) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / f"{index_name}.faiss"))

        metadata = [{"item_id": i.item_id, "filename": i.filename, "title": i.title} for i in self.items]
        with open(index_dir / f"{index_name}.items.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved vector store ({self.index.ntotal} items) to {index_dir}")

    @classmethod
    def load(cls, index_dir: str | Path, index_name: str) -> "VectorStore":
        index_dir = Path(index_dir)
        index_path = index_dir / f"{index_name}.faiss"
        items_path = index_dir / f"{index_name}.items.json"

        if not index_path.exists() or not items_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_dir} with name '{index_name}'. Run indexing first."
            )

        index = faiss.read_index(str(index_path))
        with open(items_path) as f:
            metadata = json.load(f)

        instance = cls(dim=index.d)
        instance.index = index
        instance.items = [CatalogItem(**m) for m in metadata]

        logger.info(f"Loaded vector store ({index.ntotal} items) from {index_dir}")
        return instance