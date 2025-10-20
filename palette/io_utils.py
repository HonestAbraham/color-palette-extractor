from pathlib import Path
import cv2

def load_image_bgr(path, max_dim=None):
    """Read image as BGR uint8. Optionally downscale longest side to max_dim for speed."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    if max_dim:
        h, w = img.shape[:2]
        scale = max(h, w) / float(max_dim)
        if scale > 1:
            img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    return img

def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
