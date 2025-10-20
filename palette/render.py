from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colorspaces import rgb_to_hex

def draw_palette_image(colors_rgb: np.ndarray,
                       perc: np.ndarray,
                       swatch_size=(220, 160),
                       pad=20,
                       font_path=None,
                       show_percent=True,
                       bg=(255, 255, 255)):
    """
    Create a labeled palette PNG as a PIL.Image.
    """
    k = len(colors_rgb)
    W = k * swatch_size[0] + (k + 1) * pad
    H = swatch_size[1] + 2 * pad + 40
    img = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)

    try:
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(font_path, size=18)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    x = pad
    for c, p in zip(colors_rgb, perc):
        c = tuple(int(v) for v in c)
        draw.rectangle([x, pad, x + swatch_size[0], pad + swatch_size[1]], fill=c, outline=(0, 0, 0))
        hexcode = rgb_to_hex(c)
        label = hexcode + (f"  {p * 100:.1f}%" if show_percent else "")
        draw.text((x + 8, pad + swatch_size[1] + 8), label, fill=(0, 0, 0), font=font)
        x += swatch_size[0] + pad

    return img

def save_css_variables(colors_rgb, out_css: Path):
    """Write :root { --c1: #hex; ... } file."""
    hexes = [rgb_to_hex(c) for c in colors_rgb]
    lines = [":root {"]
    for i, hx in enumerate(hexes, 1):
        lines.append(f"  --c{i}: {hx};")
    lines.append("}")
    Path(out_css).write_text("\n".join(lines), encoding="utf-8")

def save_json_palette(colors_rgb, perc, out_json: Path):
    data = [
        {"hex": rgb_to_hex(c), "rgb": [int(v) for v in c], "percent": float(p)}
        for c, p in zip(colors_rgb, perc)
    ]
    Path(out_json).write_text(json.dumps(data, indent=2), encoding="utf-8")

def quantize_to_palette(img_rgb: np.ndarray, colors_rgb: np.ndarray) -> np.ndarray:
    """Map each pixel to nearest palette color (Euclidean in RGB)."""
    h, w, _ = img_rgb.shape
    flat = img_rgb.reshape(-1, 3).astype(np.float32)
    palette = colors_rgb.astype(np.float32)  # (K,3)
    d2 = np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2)  # (N,K)
    nn = np.argmin(d2, axis=1)
    out = palette[nn].reshape(h, w, 3).astype(np.uint8)
    return out
