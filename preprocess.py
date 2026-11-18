import numpy as np
from PIL import Image, ImageOps

CLASSES = ("NP", "P")
SIZE = (96, 160)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    pixels = np.asarray(rgb.resize(SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    return ((pixels / 255 - MEAN) / STD).transpose(2, 0, 1).copy()

