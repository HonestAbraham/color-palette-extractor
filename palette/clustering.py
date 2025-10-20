import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .colorspaces import rgb_to_lab_nd, rgb_to_hsv_nd, clamp_uint8_rgb

def compute_kmeans_palette(img_rgb: np.ndarray,
                           k: int = 5,
                           color_space: str = "lab",
                           sample: int = 200_000,
                           random_state: int = 42,
                           mask: np.ndarray | None = None):
    """
    Cluster image colors and return:
      centers_rgb : (k,3) uint8 palette colors (RGB)
      counts      : (k,) pixel counts (on full image) for each cluster
      labels_full : (N,) labels per pixel (flattened full image)
    """
    h, w, _ = img_rgb.shape
    pixels_full = img_rgb.reshape(-1, 3).astype(np.uint8)

    # Apply mask (if provided)
    if mask is not None:
        mask_flat = mask.reshape(-1)
        pixels = pixels_full[mask_flat]
        if pixels.size == 0:
            raise SystemExit("Mask removed all pixels — relax thresholds.")
    else:
        pixels = pixels_full

    # Optional sampling for speed
    if sample and pixels.shape[0] > sample:
        rs = np.random.RandomState(random_state)
        idx = rs.choice(pixels.shape[0], size=sample, replace=False)
        pixels_small = pixels[idx]
    else:
        pixels_small = pixels

    cs = color_space.lower()
    if cs == "lab":
        pixels_small_cs = rgb_to_lab_nd(pixels_small)
        pixels_full_cs  = rgb_to_lab_nd(pixels_full)
    elif cs == "hsv":
        pixels_small_cs = rgb_to_hsv_nd(pixels_small)
        pixels_full_cs  = rgb_to_hsv_nd(pixels_full)
    else:
        pixels_small_cs = pixels_small.astype(np.float32)
        pixels_full_cs  = pixels_full.astype(np.float32)

    # Fit MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=4096)
    kmeans.fit(pixels_small_cs)

    # Predict labels for ALL pixels (for accurate proportions)
    labels_full = kmeans.predict(pixels_full_cs)
    counts = np.bincount(labels_full, minlength=k)

    # Convert centers back to RGB
    centers_cs = kmeans.cluster_centers_.astype(np.float32)
    if cs == "lab":
        centers_rgb = cv2.cvtColor(centers_cs.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    elif cs == "hsv":
        centers_rgb = cv2.cvtColor(centers_cs.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    else:
        centers_rgb = centers_cs
    centers_rgb = clamp_uint8_rgb(centers_rgb)

    return centers_rgb, counts, labels_full

def auto_choose_k(img_rgb: np.ndarray,
                  k_min: int = 3,
                  k_max: int = 9,
                  color_space: str = "lab",
                  sample: int = 50_000,
                  random_state: int = 42,
                  mask: np.ndarray | None = None):
    """Pick K in [k_min, k_max] by maximizing silhouette score on a sample."""
    pixels = img_rgb.reshape(-1, 3).astype(np.uint8)
    if mask is not None:
        mask_flat = mask.reshape(-1)
        pixels = pixels[mask_flat]
    if pixels.size < 2:
        return max(2, k_min), -1.0

    # Sample
    if pixels.shape[0] > sample:
        rs = np.random.RandomState(random_state)
        idx = rs.choice(pixels.shape[0], size=sample, replace=False)
        pixels = pixels[idx]

    cs = color_space.lower()
    if cs == "lab":
        pixels_cs = rgb_to_lab_nd(pixels)
    elif cs == "hsv":
        pixels_cs = rgb_to_hsv_nd(pixels)
    else:
        pixels_cs = pixels.astype(np.float32)

    best_k, best_score = None, -1
    for k in range(max(2, k_min), max(2, k_max) + 1):
        mbk = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=4096)
        labels = mbk.fit_predict(pixels_cs)
        if len(np.unique(labels)) < 2 or len(labels) < 3:
            continue
        score = silhouette_score(pixels_cs, labels, metric="euclidean")
        if score > best_score:
            best_score, best_k = score, k
    if best_k is None:
        best_k = max(2, k_min)
    return best_k, best_score

def sort_palette(centers_rgb: np.ndarray, counts: np.ndarray, mode: str = "percent_desc"):
    """
    Sort palette + proportions for nicer presentation.
    Returns:
      colors_sorted: (k,3) uint8
      perc_sorted  : (k,) float (sum to 1)
    """
    perc = counts / counts.sum()
    if mode == "percent_desc":
        order = np.argsort(-perc)
    elif mode == "luminance":
        r, g, b = centers_rgb[:, 0] / 255.0, centers_rgb[:, 1] / 255.0, centers_rgb[:, 2] / 255.0
        Y = 0.2126 * r + 0.7152 * g + 0.0722 * b  # sRGB relative luminance
        order = np.argsort(Y)  # dark -> light
    elif mode == "chroma":
        r, g, b = centers_rgb[:, 0].astype(float), centers_rgb[:, 1].astype(float), centers_rgb[:, 2].astype(float)
        chroma = np.sqrt((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2)
        order = np.argsort(-chroma)
    else:
        order = np.arange(len(centers_rgb))
    return centers_rgb[order], perc[order]
