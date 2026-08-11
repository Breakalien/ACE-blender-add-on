"""
Height/colour image -> tangent-space normal map.

Port of NormalMapGenerator by Mehdi-Antoine (MIT licence):
    https://github.com/Mehdi-Antoine/NormalMapGenerator

The original relies on scipy.ndimage for its convolutions, which Blender does
not ship, so the two convolutions are reimplemented here with numpy alone.
They reproduce scipy.ndimage.convolve exactly:
  - true convolution (the kernel is flipped, unlike a correlation), which
    matters for Sobel since flipping negates the gradient - and therefore the
    direction the surface appears to face;
  - 'reflect' edge handling in scipy's sense, (d c b a | a b c d | d c b a),
    which is numpy's "symmetric" mode, not numpy's "reflect".

Everything else - the greyscale weights, the Sobel kernel, the intensity
maths, the 0.5/0.5 remap and the final green-channel flip - follows the
original step for step.
"""
from __future__ import annotations

import numpy as np

# Original: im_grey = im[...,0]*0.3 + im[...,1]*0.6 + im[...,2]*0.1
_GREY_WEIGHTS = (0.3, 0.6, 0.1)
_SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                     [-2.0, 0.0, 2.0],
                     [-1.0, 0.0, 1.0]])


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """scipy.ndimage.convolve equivalent (flipped kernel, 'reflect' edges)."""
    kernel = np.asarray(kernel, dtype=np.float64)[::-1, ::-1]
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float64), ((ph, ph), (pw, pw)), mode="symmetric")
    height, width = image.shape
    out = np.zeros((height, width), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight:
                out += weight * padded[i:i + height, j:j + width]
    return out


def _smooth_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if not sigma:
        return image.astype(np.float64)
    axis = np.arange(-3 * sigma, 3 * sigma + 1, dtype=np.float64)
    kernel = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    smoothed = _convolve(image, kernel[np.newaxis])
    return _convolve(smoothed, kernel[np.newaxis].T)


def _sobel(image: np.ndarray):
    return _convolve(image, _SOBEL_X), _convolve(image, _SOBEL_X.T)


def _compute_normal_map(gradient_x, gradient_y, intensity: float = 1.0) -> np.ndarray:
    max_value = max(np.max(gradient_x), np.max(gradient_y))
    if max_value == 0:
        max_value = 1.0
    # Original does `intensity = 1/intensity` then
    # `strength = max_value/(max_value*intensity)`, i.e. strength == the
    # intensity the caller passed in; Z is 1/strength.
    strength = intensity if intensity else 1.0

    height, width = gradient_x.shape
    normal = np.zeros((height, width, 3), dtype=np.float64)
    normal[..., 0] = gradient_x / max_value
    normal[..., 1] = gradient_y / max_value
    normal[..., 2] = 1.0 / strength

    norm = np.sqrt((normal ** 2).sum(axis=2))
    norm[norm == 0] = 1.0
    normal /= norm[..., np.newaxis]

    normal *= 0.5
    normal += 0.5
    return normal


def generate_normal_map(rgb: np.ndarray, smooth: float = 0.0, intensity: float = 1.0) -> np.ndarray:
    """rgb: float array (h, w, 3 or 4) in 0..1. Returns float (h, w, 3) in
    0..1, green already flipped like the original's `flipgreen` step."""
    image = rgb.astype(np.float64)
    if image.ndim == 3:
        image = (image[..., 0] * _GREY_WEIGHTS[0]
                 + image[..., 1] * _GREY_WEIGHTS[1]
                 + image[..., 2] * _GREY_WEIGHTS[2])

    smoothed = _smooth_gaussian(image, smooth)
    gradient_x, gradient_y = _sobel(smoothed)
    normal = _compute_normal_map(gradient_x, gradient_y, intensity)
    # `flipgreen()` in the original: the saved PNG gets its green channel
    # inverted, swapping between the two normal-map handedness conventions.
    normal[..., 1] = 1.0 - normal[..., 1]
    return np.clip(normal, 0.0, 1.0)
