"""Generates a synthetic product catalog: colored shapes on a plain
background, standing in for product photos (a red circle ~ "red plate",
a blue square ~ "blue tile", etc.) so the whole pipeline — indexing,
image-to-image search, and text-to-image search — can be demonstrated and
tested without needing a real photo dataset.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from visual_search.embedders.stub_embedder import COLORS, SHAPE_NAMES

IMG_SIZE = 128
BACKGROUND = (245, 245, 248)

_PRODUCT_NAME_TEMPLATES = {
    "circle": "{color} Ceramic Plate",
    "square": "{color} Coaster Set",
    "triangle": "{color} Wedge Pillow",
    "star": "{color} Star Ornament",
}


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: tuple[int, int, int], margin: int = 18) -> None:
    box = [margin, margin, IMG_SIZE - margin, IMG_SIZE - margin]

    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon([(IMG_SIZE / 2, margin), (margin, IMG_SIZE - margin), (IMG_SIZE - margin, IMG_SIZE - margin)], fill=color)
    elif shape == "star":
        cx, cy = IMG_SIZE / 2, IMG_SIZE / 2
        outer, inner = IMG_SIZE / 2 - margin, (IMG_SIZE / 2 - margin) * 0.45
        points = []
        for i in range(10):
            radius = outer if i % 2 == 0 else inner
            angle = math.radians(i * 36 - 90)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def generate_catalog(catalog_dir: str, metadata_csv: str, seed: int = 42) -> pd.DataFrame:
    """Creates one image per (color, shape) combination — a clean, fully
    covered demo catalog. Returns the metadata DataFrame that was written."""
    rng = random.Random(seed)
    catalog_path = Path(catalog_dir)
    catalog_path.mkdir(parents=True, exist_ok=True)

    rows = []
    item_id = 0
    for color_name, color_rgb in COLORS.items():
        for shape in SHAPE_NAMES:
            jittered = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in color_rgb)

            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BACKGROUND)
            draw = ImageDraw.Draw(image)
            _draw_shape(draw, shape, jittered)

            filename = f"item_{item_id:03d}.png"
            image.save(catalog_path / filename)

            title = _PRODUCT_NAME_TEMPLATES[shape].format(color=color_name.capitalize())
            rows.append({"item_id": item_id, "filename": filename, "title": title, "color": color_name, "shape": shape})
            item_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(metadata_csv, index=False)
    return df


if __name__ == "__main__":
    df = generate_catalog("data/catalog", "data/catalog/metadata.csv")
    print(f"Generated {len(df)} catalog items in data/catalog/")
    print(df[["item_id", "title", "filename"]].to_string(index=False))