import numpy as np

def build_brightness_mask(img_rgb: np.ndarray,
                          ignore_white: bool = True,
                          ignore_black: bool = True,
                          white_thr: int = 245,
                          black_thr: int = 15) -> np.ndarray:
    """
    Return boolean mask of pixels to KEEP (True). Removes near-white/near-black by default.
    """
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    keep = np.ones(r.shape, dtype=bool)
    if ignore_white:
        white = (r >= white_thr) & (g >= white_thr) & (b >= white_thr)
        keep &= ~white
    if ignore_black:
        black = (r <= black_thr) & (g <= black_thr) & (b <= black_thr)
        keep &= ~black
    return keep
