# inspect_depth_cache.py
import os, argparse
from pathlib import Path
import numpy as np
import cv2

def to_u8(x):
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return (x*255.0 + 0.5).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cache = Path(args.cache_dir); out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    npzs = list(cache.rglob("*.npz"))
    if not npzs:
        print("No se encontraron .npz"); return

    for p in npzs:
        d = np.load(p)["depth01"].astype(np.float32)
        # stats
        mn, mx = float(np.nanmin(d)), float(np.nanmax(d))
        mean = float(np.nanmean(np.nan_to_num(d, nan=0.0)))
        print(f"{p.name}: min={mn:.3f} mean={mean:.3f} max={mx:.3f} NaN={np.isnan(d).any()}")

        # visualize (opcional: colormap)
        vis = to_u8(d)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_BONE)
        cv2.imwrite(str(out / (p.stem.replace(".depth", "") + "_depth.png")), vis)

if __name__ == "__main__":
    main()
