"""Real semantic embeddings via CLIP (open_clip). This is the actual
production-quality embedder for this idea — open-ended natural language
queries against real product photos, not a hand-built vocabulary.

Requires the `clip` extra (`pip install -e ".[clip]"`) and internet access
on first use to download pretrained weights from the HuggingFace Hub.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise ImportError(
                "The clip embedding provider requires the 'clip' extra. "
                "Install it with: pip install -e \".[clip]\""
            ) from e

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.dim = self.model.visual.output_dim

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0)
        with self._torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text])
        with self._torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).numpy().astype(np.float32)