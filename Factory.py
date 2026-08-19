"""Embedder factory: swap between the offline stub and real CLIP via config."""
from __future__ import annotations

from visual_search.config import EmbeddingConfig
from visual_search.embedders.base import Embedder
from visual_search.embedders.stub_embedder import StubJointEmbedder


def build_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "stub":
        return StubJointEmbedder()

    if config.provider == "clip":
        from visual_search.embedders.clip_embedder import ClipEmbedder
        return ClipEmbedder(model_name=config.clip_model_name, pretrained=config.clip_pretrained)

    raise ValueError(f"Unknown embedding provider: {config.provider}. Available: stub, clip")