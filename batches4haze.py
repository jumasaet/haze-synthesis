from __future__ import annotations
import os, glob, json, argparse, time, math, random
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image as PILImage

# --- DEPTH BACKEND (usa tu módulo actual) ---
from depth_backends import make_backend, predict_depth

# ----------------- Utilidades de máscara -----------------
def to_binary_01(img):
    if img.dtype == np.uint8:
        return (img > 127).astype(np.uint8)
    else:
        return (img > 0.5).astype(np.uint8)

def morph_kernel(h, w, k_frac=0.01):
    k = max(3, int(round(min(h, w) * k_frac)))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def remove_small_components(bin01, min_area=50):
    bin_uint8 = (bin01 * 255).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_uint8, connectivity=8)
    keep = np.zeros_like(bin01, dtype=np.uint8)
    for i in range(1, num):  # 0 es fondo
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep

def fill_small_holes(bin01, max_hole_area=50):
    inv = 1 - bin01
    inv_uint8 = (inv * 255).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv_uint8, connectivity=8)
    small_holes = np.zeros_like(inv, dtype=np.uint8)
    for i in range(1, num):  # omitir fondo
        if stats[i, cv2.CC_STAT_AREA] < max_hole_area:
            small_holes[labels == i] = 1
    result = bin01.copy()
    result[small_holes == 1] = 1
    return result

def clean_binary_mask(binary_input, h, w,
                      min_blob_frac=0.0005, max_hole_frac=0.0005,
                      k_frac=0.01, open_iters=1, close_iters=1):
    b01 = to_binary_01(binary_input)
    K = morph_kernel(h, w, k_frac)
    if open_iters > 0:
        b01 = cv2.morphologyEx(b01, cv2.MORPH_OPEN, K, iterations=open_iters)
    if close_iters > 0:
        b01 = cv2.morphologyEx(b01, cv2.MORPH_CLOSE, K, iterations=close_iters)

    img_area = float(h * w)
    min_area = max(1, int(round(img_area * min_blob_frac)))
    max_hole_area = max(1, int(round(img_area * max_hole_frac)))

    b01 = remove_small_components(b01, min_area=min_area)
    b01 = fill_small_holes(b01, max_hole_area=max_hole_area)
    return (b01 * 255).astype(np.uint8)

# ----------------- Física de niebla -----------------
def apply_haze_rgb01(J01, depth01, beta, A01_vec3):
    """
    J01: imagen limpia en [0,1], shape (H,W,3)
    depth01: mapa de profundidad normalizado [0,1], shape (H,W)
    beta: float
    A01_vec3: np.array shape (3,) en [0,1]
    """
    H, W = depth01.shape
    t = np.exp(-beta * depth01)                # (H,W)
    t3 = t[..., None]                          # (H,W,1)
    I = J01 * t3 + A01_vec3[None, None, :] * (1 - t3)
    M = 1.0 - t                                # máscara porcentual [0,1]
    return np.clip(I, 0, 1), np.clip(M, 0, 1)

# ----------------- Muestreos de parámetros -----------------
def sample_log_uniform(a, b):
    ua, ub = math.log(a), math.log(b)
    return math.exp(random.uniform(ua, ub))

def sample_A_day():
    base = random.uniform(0.85, 1.00)
    tint = np.clip(np.random.normal(0, 0.04, 3), -0.08, 0.08)
    A = np.clip(base + tint, 0.75, 1.0)
    return A.astype(np.float32)

def sample_A_night():
    base = random.uniform(0.35, 0.60)
    tint = np.clip(np.random.normal(0, 0.06, 3), -0.12, 0.12)
    A = np.clip(base + tint, 0.20, 0.80)
    return A.astype(np.float32)

def target_buckets():
    # (lo, hi, etiqueta)
    return [(0.10, 0.25, "L"), (0.40, 0.55, "M"), (0.65, 0.85, "H")]

# ----------------- Guardado -----------------
def ensure_dirs(base: Path, mode: str):
    # mode in {"day","night"}
    out = {}
    for sub in ["hazy", "masks_float", "masks_bin"]:
        for b in ["L","M","H"]:
            p = base / mode / sub / b
            p.mkdir(parents=True, exist_ok=True)
            out[(sub, b)] = p
    return out

def save_outputs(base_dirs, b_tag, stem, I01, M01, bin_mask, ext="png", save_float_preview=False):
    # I01 [0,1], M01 [0,1], bin_mask uint8(0/255)
    hazy_path = base_dirs[("hazy", b_tag)] / f"{stem}.png"
    float_path = base_dirs[("masks_float", b_tag)] / f"{stem}.npy"
    bin_path = base_dirs[("masks_bin", b_tag)] / f"{stem}.png"

    cv2.imwrite(str(hazy_path), cv2.cvtColor((I01*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    np.save(str(float_path), M01.astype(np.float32))
    cv2.imwrite(str(bin_path), bin_mask)

    if save_float_preview:
        prev = (M01 * 255).astype(np.uint8)
        prev3 = cv2.applyColorMap(prev, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(float_path.with_suffix(".preview.png")), prev)

    return hazy_path, float_path, bin_path

# ----------------- Núcleo por imagen -----------------
def generate_variants_for_image(p_img: Path,
                                backend,
                                mode: str,
                                args,
                                manifest_items: list,
                                base_dirs: dict):
    """
    mode: "day" o "night"
    """
    # 1) Cargar RGB
    pil_img = PILImage.open(p_img).convert("RGB")
    rgb = np.array(pil_img, dtype=np.uint8)
    H, W = rgb.shape[:2]
    rgb01 = rgb.astype(np.float32) / 255.0

    # 2) Profundidad [0,1]
    depth01 = predict_depth(backend=backend, pil_rgb=pil_img)
    if args.invert_depth_for_haze:
        depth01 = 1.0 - depth01
    depth01 = depth01.astype(np.float32)

    # 3) Distribución de variantes por buckets
    buckets = target_buckets()
    per_bucket = max(1, args.variants_per_img // len(buckets))
    leftover = args.variants_per_img - per_bucket * len(buckets)

    # 4) Iterar por bucket
    for (lo, hi, tag) in buckets:
        target_count = per_bucket + (1 if leftover > 0 else 0)
        if leftover > 0:
            leftover -= 1

        got = 0
        trials = 0
        max_trials = args.max_trials

        while got < target_count and trials < max_trials:
            # β según modo
            if mode == "day":
                beta = sample_log_uniform(args.day_beta_min, args.day_beta_max)
            else:
                beta = sample_log_uniform(args.night_beta_min, args.night_beta_max)

            # Estimar M con ese β (A no afecta M)
            M01 = 1.0 - np.exp(-beta * depth01)
            mean_M = float(M01.mean())

            if lo <= mean_M <= hi:
                # A según modo
                A01 = sample_A_day() if mode == "day" else sample_A_night()
                # Reconstrucción final con ese A
                I01, M01 = apply_haze_rgb01(rgb01, depth01, beta, A01)

                # Binarización + limpieza
                thr = args.binary_thres
                bin01 = (M01 > thr).astype(np.uint8)
                bin_mask_clean = clean_binary_mask(
                    (bin01*255).astype(np.uint8), H, W,
                    min_blob_frac=args.min_blob_frac,
                    max_hole_frac=args.max_hole_frac,
                    k_frac=args.k_frac,
                    open_iters=args.open_iters,
                    close_iters=args.close_iters
                )

                # Nombre único
                stem = f"{p_img.stem}_fog-{tag}_b{beta:.3f}_Ar{int(round(A01[0]*255))}_Ag{int(round(A01[1]*255))}_Ab{int(round(A01[2]*255))}"

                # Guardar
                hazy_p, float_p, bin_p = save_outputs(
                    base_dirs, tag, stem, I01, M01, bin_mask_clean,
                    ext=args.ext, save_float_preview=args.save_float_preview
                )

                # Registrar en manifest (rutas relativas al output_root)
                item = {
                    "mode": mode,
                    "original": str(p_img),
                    "generated": str(hazy_p),
                    "mask_float": str(float_p),
                    "mask_bin": str(bin_p),
                    "bucket": tag,                    # L/M/H
                    "beta": round(beta, 6),
                    "A": {
                        "r": round(float(A01[0]), 6),
                        "g": round(float(A01[1]), 6),
                        "b": round(float(A01[2]), 6),
                    },
                    "mean_mask": round(mean_M, 6),
                }
                manifest_items.append(item)
                got += 1

            trials += 1

# ----------------- Main -----------------
def collect_images(folder: Path, ext: str):
    if folder is None:
        return []
    if not folder.exists():
        return []
    return sorted([Path(p) for p in glob.glob(str(folder / f"*.{ext}"))])

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def parse_args():
    ap = argparse.ArgumentParser("Batch haze synthesis with day/night ranges and JSON manifest")
    ap.add_argument("--day_dir", type=str, default="M3FD/day", help="Carpeta con imágenes de día")
    ap.add_argument("--night_dir", type=str, default="M3FD/night", help="Carpeta con imágenes de noche")
    ap.add_argument("--ext", type=str, default="png", help="Extensión (png|jpg|jpeg ...)")

    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--manifest_name", type=str, default="manifest.json")

    # Profundidad
    ap.add_argument("--depth_backend", type=str, default="depth-anything-v2",
                    choices=["monodepth2","depth-anything-v2","zoedepth","midas"])
    ap.add_argument("--invert_depth_for_haze", action="store_true")

    # Variantes y buckets
    ap.add_argument("--variants_per_img", type=int, default=6)
    ap.add_argument("--max_trials", type=int, default=10)

    # Rango β
    ap.add_argument("--day_beta_min", type=float, default=0.8)
    ap.add_argument("--day_beta_max", type=float, default=2.3)
    ap.add_argument("--night_beta_min", type=float, default=0.8)
    ap.add_argument("--night_beta_max", type=float, default=2.3)

    # Binarización y limpieza
    ap.add_argument("--binary_thres", type=float, default=0.55)
    ap.add_argument("--min_blob_frac", type=float, default=0.005)
    ap.add_argument("--max_hole_frac", type=float, default=0.005)
    ap.add_argument("--k_frac", type=float, default=0.01)
    ap.add_argument("--open_iters", type=int, default=1)
    ap.add_argument("--close_iters", type=int, default=1)

    # Batches de imágenes (para memoria/progreso)
    ap.add_argument("--batch_size_images", type=int, default=32)

    # Previews
    ap.add_argument("--save_float_preview", action="store_true")

    # Dispositivo
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--device", type=int, default=0)

    # Semilla
    ap.add_argument("--seed", type=int, default=123)

    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Dispositivo
    use_cuda = torch.cuda.is_available()
    if use_cuda and args.device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.device}")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(args.device)} (ID: {args.device})\n")
    else:
        if use_cuda and args.device >= torch.cuda.device_count():
            print(f"Warning: GPU ID {args.device} not available. Using CPU instead.")
        device = torch.device("cpu")
        print("Using device: CPU")

    # Backend de profundidad
    backend = make_backend(args.depth_backend, device, monodepth2_cfg=None)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Recolectar imágenes
    day_imgs = collect_images(Path(args.day_dir), args.ext) if args.day_dir else []
    night_imgs = collect_images(Path(args.night_dir), args.ext) if args.night_dir else []

    total_imgs = len(day_imgs) + len(night_imgs)
    if total_imgs == 0:
        print("No se encontraron imágenes.")
        return

    # Preparar directorios
    dirs_day = ensure_dirs(out_root, "day") if day_imgs else None
    dirs_night = ensure_dirs(out_root, "night") if night_imgs else None

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "depth_backend": args.depth_backend,
            "invert_depth_for_haze": bool(args.invert_depth_for_haze),
            "variants_per_img": args.variants_per_img,
            "binary_thres": args.binary_thres,
            "day_beta": [args.day_beta_min, args.day_beta_max],
            "night_beta": [args.night_beta_min, args.night_beta_max],
            "day_A_range": [0.85, 1.0],
            "night_A_range": [0.35, 0.60],
            "buckets": {"L":[0.10,0.25], "M":[0.40,0.55], "H":[0.65,0.85]},
        },
        "items": []
    }

    t0 = time.time()

    # Día
    if day_imgs:
        print(f"[DÍA] {len(day_imgs)} imágenes")
        for chunk in chunked(day_imgs, args.batch_size_images):
            for p in chunk:
                generate_variants_for_image(
                    p_img=p, backend=backend, mode="day",
                    args=args, manifest_items=manifest["items"], base_dirs=dirs_day
                )
            print(f"  Progresso día: {min(len(chunk), len(day_imgs))} / {len(day_imgs)} (acum)")

    # Noche
    if night_imgs:
        print(f"[NOCHE] {len(night_imgs)} imágenes")
        for chunk in chunked(night_imgs, args.batch_size_images):
            for p in chunk:
                generate_variants_for_image(
                    p_img=p, backend=backend, mode="night",
                    args=args, manifest_items=manifest["items"], base_dirs=dirs_night
                )
            print(f"  Progresso noche: {min(len(chunk), len(night_imgs))} / {len(night_imgs)} (acum)")

    # Guardar manifest
    manifest_path = out_root / args.manifest_name
    # Convertir rutas a relativas al out_root para portabilidad
    for it in manifest["items"]:
        for k in ["generated", "mask_float", "mask_bin"]:
            it[k] = str(Path(it[k]).resolve().relative_to(out_root.resolve()))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    dt = time.time() - t0
    print(f"\nListo. Items generados: {len(manifest['items'])}")
    print(f"Manifest: {manifest_path}")
    print(f"Tiempo total: {dt/60:.2f} min")

if __name__ == "__main__":
    main()
