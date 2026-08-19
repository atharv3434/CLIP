"""A fully offline, no-download joint image/text embedder over a small
color+shape vocabulary.

This is NOT a toy that merely "runs without crashing" — it genuinely derives
image features from pixels (dominant foreground color + silhouette fill-ratio
shape classification, no metadata reading involved) and text features from
keyword matching against the same vocabulary, landing both in one shared
10-dimensional space. That means text queries like "blue square" genuinely
retrieve blue-square catalog images through real vector similarity search,
not a lookup table — the same mechanism CLIP uses, just over a much smaller,
hand-built vocabulary instead of a model trained on hundreds of millions of
image-text pairs.

Use this to develop and test the full pipeline (indexing, search, ranking)
without internet access or a model download. Switch to the `clip` provider
for real product photos and open-ended natural-language queries — see
embedders/clip_embedder.py and the README for that trade-off.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "orange": (230, 140, 40),
    "yellow": (225, 205, 40),
    "green": (50, 160, 70),
    "blue": (40, 90, 220),
    "purple": (130, 60, 180),
}
COLOR_NAMES = list(COLORS.keys())

# Expected filled-area-of-bounding-box ratio for each shape, used to classify
# a silhouette purely from its fill ratio — simple, but genuinely derived
# from the pixels, not from any label.
SHAPE_FILL_RATIOS: dict[str, float] = {
    "square": 1.00,
    "circle": 0.79,   # pi/4
    "triangle": 0.50,
    "star": 0.38,
}
SHAPE_NAMES = list(SHAPE_FILL_RATIOS.keys())

_TEXT_SYNONYMS = {
    "squares": "square", "circles": "circle", "triangles": "triangle", "stars": "star",
    "reds": "red", "oranges": "orange", "yellows": "yellow", "greens": "green",
    "blues": "blue", "purples": "purple", "violet": "purple",
}


class StubJointEmbedder:
    dim = len(COLOR_NAMES) + len(SHAPE_NAMES)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)

        background = _estimate_background_color(arr)
        foreground_mask = _foreground_mask(arr, background)

        color_vec = _color_similarity_vector(arr, foreground_mask)
        shape_vec = _shape_similarity_vector(foreground_mask)

        vec = np.concatenate([color_vec, shape_vec])
        return _l2_normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
        words = [_TEXT_SYNONYMS.get(w, w) for w in text.lower().split()]
        word_set = set(words)

        color_vec = np.array([1.0 if name in word_set else 0.0 for name in COLOR_NAMES], dtype=np.float32)
        shape_vec = np.array([1.0 if name in word_set else 0.0 for name in SHAPE_NAMES], dtype=np.float32)

        vec = np.concatenate([color_vec, shape_vec])
        if not vec.any():
            raise ValueError(
                f"No recognized color or shape words found in '{text}'. "
                f"The offline stub embedder only understands: colors={COLOR_NAMES}, shapes={SHAPE_NAMES}. "
                f"For open-ended natural language queries, switch embedding.provider to 'clip' in the config."
            )
        return _l2_normalize(vec)


def _estimate_background_color(arr: np.ndarray) -> np.ndarray:
    """Sample the four corners and take their median as the background color
    — robust to a single corner accidentally touching the foreground shape."""
    h, w, _ = arr.shape
    corners = np.stack([arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]])
    return np.median(corners, axis=0)


def _foreground_mask(arr: np.ndarray, background: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    distance = np.linalg.norm(arr - background, axis=-1)
    return distance > threshold


def _color_similarity_vector(arr: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    if not foreground_mask.any():
        return np.zeros(len(COLOR_NAMES), dtype=np.float32)

    mean_fg_color = arr[foreground_mask].mean(axis=0)

    # Soft similarity (inverse distance) to each reference color, rather than
    # a hard nearest-neighbor pick — this makes the resulting vectors
    # meaningfully comparable by distance, not just by argmax equality, which
    # matters for nearest-neighbor search quality.
    distances = np.array([np.linalg.norm(mean_fg_color - np.array(rgb)) for rgb in COLORS.values()])
    similarities = 1.0 / (1.0 + distances / 60.0)
    return similarities.astype(np.float32)


def _shape_similarity_vector(foreground_mask: np.ndarray) -> np.ndarray:
    if not foreground_mask.any():
        return np.zeros(len(SHAPE_NAMES), dtype=np.float32)

    ys, xs = np.nonzero(foreground_mask)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    fill_ratio = foreground_mask.sum() / bbox_area

    distances = np.array([abs(fill_ratio - expected) for expected in SHAPE_FILL_RATIOS.values()])
    similarities = 1.0 / (1.0 + distances * 8.0)
    return similarities.astype(np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec