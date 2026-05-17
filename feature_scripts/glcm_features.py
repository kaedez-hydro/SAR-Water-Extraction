#!/usr/bin/env python3
"""
Fast GLCM Texture Feature Extraction for SAR Imagery
=====================================================
Computes Gray-Level Co-occurrence Matrix (GLCM) features
directly on grayscale images using an optimized sliding-window approach.

Output features: mean, std, contrast, dissimilarity, homogeneity, ASM, energy, max, entropy
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    """Compute GLCM for all pixels using a sliding window."""
    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape

    bins = np.linspace(mi, ma + 1, nbit + 1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:, 1:], gl1[:, -1:], axis=1)

    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM mean."""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2
    return mean


def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM standard deviation."""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2
    std2 = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i, j] * i - mean) ** 2
    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM contrast."""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i, j] * (i - j) ** 2
    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM dissimilarity."""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i, j] * np.abs(i - j)
    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM homogeneity."""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i, j] / (1. + (i - j) ** 2)
    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM angular second moment and energy."""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm_val = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm_val += glcm[i, j] ** 2
    ene = np.sqrt(asm_val)
    return asm_val, ene


def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM maximum probability."""
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    max_ = np.max(glcm, axis=(0, 1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM entropy."""
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0, 1), dtype='float16') + 1. / ks ** 2
    ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
    return ent


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python glcm_features.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = np.array(Image.open(image_path).convert('L'))

    features = {
        'Homogeneity': fast_glcm_homogeneity(img),
        'ASM': fast_glcm_ASM(img)[0],
        'Entropy': fast_glcm_entropy(img)
    }

    for title, feature in features.items():
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        im = ax.imshow(feature, cmap='jet')
        ax.set_title(title, fontsize=15)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.show()
