import numpy as np
import cv2

def rgb_to_lab_nd(pixels_uint8: np.ndarray) -> np.ndarray:
    """pixels_uint8: (N,3) RGB uint8 -> (N,3) LAB float32 via OpenCV."""
    return cv2.cvtColor(pixels_uint8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

def rgb_to_hsv_nd(pixels_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(pixels_uint8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)

def clamp_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)

def rgb_to_hex(rgb_triplet) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in rgb_triplet])
