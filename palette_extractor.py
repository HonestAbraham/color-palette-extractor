#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from palette.io_utils import load_image_bgr, bgr_to_rgb
from palette.mask import build_brightness_mask
from palette.clustering import compute_kmeans_palette, auto_choose_k, sort_palette
from palette.render import draw_palette_image, save_css_variables, save_json_palette, quantize_to_palette

def parse_args():
    ap = argparse.ArgumentParser(description="Extract dominant color palette from an image.")
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("-k", "--colors", type=int, default=5, help="Number of colors (clusters)")
    ap.add_argument("--space", choices=["lab", "rgb", "hsv"], default="lab", help="Color space for clustering")
    ap.add_argument("--max-dim", type=int, default=1024, help="Downscale longest side to this (speed/memory)")
    ap.add_argument("--order", choices=["percent_desc", "luminance", "chroma"], default="percent_desc",
                   help="Sort palette for presentation")
    ap.add_argument("--sample", type=int, default=200_000, help="Pixel sample size for KMeans fit (None for all)")
    ap.add_argument("--out", default=None, help="Output basename (default: <image>_palette)")
    ap.add_argument("--css", action="store_true", help="Also write a CSS file with :root variables")
    ap.add_argument("--json", action="store_true", help="Also write a JSON file with colors and percentages")
    ap.add_argument("--posterize", action="store_true", help="Save an image quantized to the palette")
    ap.add_argument("--show", action="store_true", help="Show quick previews (requires matplotlib)")
    # Mask controls
    ap.add_argument("--no-ignore-white", action="store_true", help="Do NOT ignore near-white background")
    ap.add_argument("--no-ignore-black", action="store_true", help="Do NOT ignore near-black background")
    ap.add_argument("--white-thr", type=int, default=245, help="Threshold for 'near-white' [0..255]")
    ap.add_argument("--black-thr", type=int, default=15, help="Threshold for 'near-black' [0..255]")
    # Auto-K
    ap.add_argument("--auto-k", action="store_true", help="Pick K automatically via silhouette score")
    ap.add_argument("--k-min", type=int, default=3, help="Min K for auto")
    ap.add_argument("--k-max", type=int, default=9, help="Max K for auto")
    return ap.parse_args()

def main():
    args = parse_args()

    in_path = Path(args.image)
    if not in_path.exists():
        raise SystemExit(f"Image not found: {in_path}")

    out_base = args.out or (in_path.stem + "_palette")
    out_png  = Path(out_base + ".png")
    out_css  = Path(out_base + ".css")
    out_json = Path(out_base + ".json")

    # Load & prepare image
    img_bgr = load_image_bgr(in_path, max_dim=args.max_dim)
    img_rgb = bgr_to_rgb(img_bgr)

    # Mask out near-white/near-black if enabled
    mask = build_brightness_mask(
        img_rgb,
        ignore_white=not args.no_ignore_white,
        ignore_black=not args.no_ignore_black,
        white_thr=args.white_thr,
        black_thr=args.black_thr,
    )

    # Auto-select K if requested
    if args.auto_k:
        best_k, s = auto_choose_k(
            img_rgb,
            k_min=args.k_min,
            k_max=args.k_max,
            color_space=args.space,
            sample=50_000,
            mask=mask,
        )
        print(f"[i] Auto-K chose k={best_k} (silhouette={s:.3f})")
        args.colors = best_k

    # Compute palette
    centers_rgb, counts, _ = compute_kmeans_palette(
        img_rgb,
        k=args.colors,
        color_space=args.space,
        sample=args.sample,
        mask=mask,
    )

    # Sort for presentation and save a swatch image
    colors_sorted, perc_sorted = sort_palette(centers_rgb, counts, mode=args.order)
    palette_img = draw_palette_image(colors_sorted, perc_sorted)
    palette_img.save(out_png)
    print(f"[✓] Saved palette image: {out_png}")

    if args.css:
        save_css_variables(colors_sorted, out_css)
        print(f"[✓] Saved CSS variables: {out_css}")

    if args.json:
        save_json_palette(colors_sorted, perc_sorted, out_json)
        print(f"[✓] Saved JSON palette: {out_json}")

    if args.posterize:
        q = quantize_to_palette(img_rgb, colors_sorted)
        q_path = Path(out_base + "_posterized.png")
        from PIL import Image
        Image.fromarray(q).save(q_path)
        print(f"[✓] Saved posterized image: {q_path}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb); plt.title("Original"); plt.axis("off")
        import numpy as np
        plt.figure(figsize=(10, 3))
        sw = np.zeros((80, colors_sorted.shape[0] * 120, 3), dtype=np.uint8)
        s = 0
        for c in colors_sorted:
            sw[:, s:s+120, :] = c
            s += 120
        plt.imshow(sw); plt.title("Palette"); plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
