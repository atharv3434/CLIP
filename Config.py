"""Configuration loading for the visual search pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CatalogConfig:
    dir: str = "data/catalog"
    metadata_csv: str = "data/catalog/metadata.csv"


@dataclass
class EmbeddingConfig:
    provider: str = "stub"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"


@dataclass
class IndexConfig:
    index_dir: str = "index"
    index_name: str = "catalog_index"
    top_k: int = 5


@dataclass
class PipelineConfig:
    catalog: CatalogConfig
    embedding: EmbeddingConfig
    index: IndexConfig
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            catalog=CatalogConfig(**raw.get("catalog", {})),
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            index=IndexConfig(**raw.get("index", {})),
            log_level=raw.get("logging", {}).get("level", "INFO"),
        )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )