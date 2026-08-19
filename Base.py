"""Shared interface every embedding backend implements: mapping both images
and text into the *same* vector space is what makes cross-modal search
(text query -> matching images) possible at all."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    dim: int

    def embed_image(self, image: Image.Image) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...