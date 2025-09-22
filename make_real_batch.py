#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, cv2, glob, json, argparse, time, random
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

import torch
from tqdm.auto import tqdm

np.seterr(invalid='ignore', over='ignore', divide='ignore')

# ---------------------------
# Helpers de forma
# ---------------------------
def ensure_hw1(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[..., None]
    if x.ndim == 3 and x.shape[2] == 1:
        return x
    if x.ndim == 3 and x.shape[2] > 1:
        return x[..., :1]
    raise ValueError(f"Forma inesperada para HW1: {x.shape}")

def ensure_hwc3(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return np.repeat(x[..., None], 3, axis=2)
    if x.ndim == 3 and x.shape[2] == 3:
        return x
    if x.ndim == 3 and x.shape[2] == 1:
        return np.repeat(x, 3, axis=2)
    if x.ndim == 3 and x.shape[2] > 3:
        return x[..., :3]
    raise ValueError(f"Forma inesperada para HWC3: {x.shape}")

# ---------------------------
# Blur gaussiano en TORCH (GPU/CPU) sobre arrays NumPy (H,W) o (H,W,C)
# ---------------------------
def _torch_gaussian_blur_np(arr: np.ndarray, sigma: float, device: str) -> np.ndarray:
    if sigma <= 0:
        return arr
    dev = torch.device(device)
    a = torch.from_numpy(arr).to(dev).float()
    if a.ndim == 2:
        a = a.unsqueeze(0).unsqueeze(0)  # NCHW: 1x1xH×W
        C = 1
    elif a.ndim == 3:
        a = a.permute(2, 0, 1).unsqueeze(0)  # 1xCxH×W
        C = a.shape[1]
    else:
        raise ValueError("Esperaba 2D o 3D para blur.")

    k = max(3, int(2 * round(3 * sigma) + 1))
    x = torch.arange(k, device=dev) - (k - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma * sigma))
    g = (g / g.sum()).float()

    g1 = g.view(1, 1, k, 1).expand(C, 1, k, 1)
    g2 = g.view(1, 1, 1, k).expand(C, 1, 1, k)

    a = torch.nn.functional.conv2d(a, g1, padding=(k // 2, 0), groups=C)
    a = torch.nn.functional.conv2d(a, g2, padding=(0, k // 2), groups=C)

    if arr.ndim == 2:
        out = a[0, 0].detach().cpu().numpy()
    else:
        out = a[0].permute(1, 2, 0).detach().cpu().numpy()
    return out

# ---------------------------
# Utilidades de imagen
# ---------------------------
def read_rgb(path):
    img = Image.open(path).convert("RGB")
    return (np.array(img).astype(np.float32) / 255.0, img)

def save_rgb(path, img01):
    out = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

def save_gray(path, img01):
    out = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)

# ---------------------------
# FBM / Perlin-like noise
# ---------------------------
def fbm_noise(h, w, octaves=4, base_sigma=0.7, seed=0):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    sigma = max(3, int(min(h, w) * 0.02))
    for o in range(octaves):
        n = rng.normal(loc=0.0, scale=base_sigma, size=(h, w)).astype(np.float32)
        k = int(max(3, (sigma // (2**o)) * 2 + 1))
        n = cv2.GaussianBlur(n, (k, k), 0)
        acc += amp * n
        amp *= 0.5
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(1.0, min(h, w) * 0.01))
    acc = cv2.normalize(acc, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return acc

# ---------------------------
# Sprites desde banco NPZ (premul)
# ---------------------------
def load_sprites_prebaked(bank_dir, n_select, seed=42):
    if bank_dir is None:
        return None
    idx_path = os.path.join(bank_dir, "index.json")
    if not os.path.isfile(idx_path):
        print(f"[WARN] No existe {idx_path}; se continúa sin sprites.")
        return None

    with open(idx_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    rng = random.Random(seed)
    picks = rng.sample(index, k=min(n_select, len(index)))

    sprites = []
    for item in picks:
        npz = np.load(os.path.join(bank_dir, item["file"]))
        rgb_p = npz["rgb_premul"].astype(np.float32)   # (H,W,3) premultiplicado
        a     = npz["alpha"].astype(np.float32)        # (H,W,1)
        eps = 1e-6
        a_safe = np.clip(a, eps, 1.0)
        rgb = rgb_p / a_safe
        rgb = np.clip(rgb, 0.0, 1.0)
        bgra = np.dstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), a])
        sprites.append(bgra)
    if not sprites:
        print("[WARN] No se cargaron sprites del banco; se continúa sin sprites.")
        return None
    return sprites

def place_sprite_alpha(dst_rgb, dst_alpha, sprite_bgra, x, y, scale=1.0, angle_deg=0.0):
    H, W, _ = dst_rgb.shape
    sbgr = sprite_bgra[:, :, :3]
    sa   = sprite_bgra[:, :, 3]

    nh = max(2, int(sbgr.shape[0] * scale))
    nw = max(2, int(sbgr.shape[1] * scale))
    sbgr = cv2.resize(sbgr, (nw, nh), interpolation=cv2.INTER_AREA)
    sa   = cv2.resize(sa,   (nw, nh), interpolation=cv2.INTER_AREA)

    M = cv2.getRotationMatrix2D((nw/2, nh/2), angle_deg, 1.0)
    sbgr = cv2.warpAffine(sbgr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    sa   = cv2.warpAffine(sa,   M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    sa = ensure_hw1(sa)

    x0 = int(x - nw // 2); y0 = int(y - nh // 2)
    x1 = max(0, x0); y1 = max(0, y0)
    x2 = min(W, x0 + nw); y2 = min(H, y0 + nh)
    if x2 <= x1 or y2 <= y1:
        return
    sx1 = x1 - x0; sy1 = y1 - y0; sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)

    patch_rgb = dst_rgb[y1:y2, x1:x2, :]
    patch_a   = dst_alpha[y1:y2, x1:x2, :]

    s_rgb = sbgr[sy1:sy2, sx1:sx2, :]
    s_a   = sa[sy1:sy2, sx1:sx2, :]

    patch_rgb = ensure_hwc3(patch_rgb)
    s_rgb     = ensure_hwc3(s_rgb)
    patch_a   = ensure_hw1(patch_a)
    s_a       = ensure_hw1(s_a)

    patch_rgb = np.nan_to_num(patch_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    s_rgb     = np.nan_to_num(s_rgb,     nan=0.0, posinf=1.0, neginf=0.0)
    patch_a   = np.clip(np.nan_to_num(patch_a, nan=0.0), 0.0, 1.0)
    s_a       = np.clip(np.nan_to_num(s_a,   nan=0.0), 0.0, 1.0)

    out_a = patch_a + s_a * (1.0 - patch_a)
    out_a = np.clip(out_a, 1e-6, 1.0)
    num   = patch_rgb * patch_a + s_rgb * s_a * (1.0 - patch_a)
    out_rgb = num / out_a

    out_rgb = np.nan_to_num(out_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    out_a   = np.nan_to_num(out_a,   nan=0.0, posinf=1.0, neginf=0.0)

    dst_rgb[y1:y2, x1:x2, :]   = np.clip(out_rgb, 0.0, 1.0)
    dst_alpha[y1:y2, x1:x2, :] = np.clip(out_a,   0.0, 1.0)

# ---------------------------
# Núcleo de síntesis (CPU + rutas GPU opcionales)
# ---------------------------
def synthesize_fog(
    rgb, depth01,
    beta=1.2,
    airlight=0.88,
    noise_strength=0.45,
    noise_octaves=4,
    t_min=0.1,
    near_guard=0.08,
    sprites=None,
    n_sprites=12,
    sprite_depth_range=(0.25, 1.0),
    sprite_scale_range=(0.5, 2.0),
    sprite_alpha_gain=0.9,
    seed=42,
    global_fog=1.0,
    gpu_mode="cpu",
    device="cpu"
):
    H, W, _ = rgb.shape
    rng = np.random.default_rng(seed)

    # Parámetros aleatorios por imagen usando sprites
    # beta              = 0
    # airlight          = round(random.uniform(0.45, 0.75), 3)
    # near_guard        = round(random.uniform(0.05, 0.8), 3)
    # n_sprites         = random.randint(20, 30)
    # sprite_alpha_gain = round(random.uniform(0.55, 0.7), 3)
    # global_fog        = round(random.uniform(0.8, 1.2), 3)
    # mask_gamma        = round(random.uniform(0.7, 0.8), 3)   # <1 refuerza medios (más denso)
    # mask_gain         = round(random.uniform(0.9, 1.2), 3)   # >1 sube densidad global
    # shade_strength    = round(random.uniform(0.4, 0.6), 3)   # 0..1 cuánto usar la luminancia del sprite en el color de la niebla

    # Parámetros aleatorios por imagen sin sprites

    prob = round(random.uniform(0.0, 1.0), 1)
    beta              = 0 if prob < 0.5 else round(random.uniform(0.0, 0.5), 3)
    airlight          = round(random.uniform(0.65, 0.95), 3)
    near_guard        = round(random.uniform(0.05, 0.8), 3)
    n_sprites         = random.randint(10, 30)
    sprite_alpha_gain = round(random.uniform(0.4, 0.8), 3)
    global_fog        = round(random.uniform(0.6, 1.2), 3)
    mask_gamma        = round(random.uniform(0.4, 0.8), 3)   # <1 refuerza medios (más denso)
    mask_gain         = round(random.uniform(0.5, 1.2), 3)   # >1 sube densidad global
    shade_strength    = round(random.uniform(0.4, 0.6), 3)   # 0..1 cuánto usar la luminancia del sprite en el color de la niebla

    # 1) Transmisión base
    t_base = np.exp(-beta * np.clip(depth01, 0, 1)).astype(np.float32)

    # 2) Irregularidad
    noise = fbm_noise(H, W, octaves=noise_octaves, seed=seed)
    irregular = noise_strength * (noise - 0.5) * 2.0
    sigma_ir = max(1.0, min(H, W) * 0.006)
    if gpu_mode == "gpu_full" and torch.cuda.is_available() and str(device).startswith("cuda"):
        irregular = _torch_gaussian_blur_np(irregular, sigma_ir, device)
        # print(torch.cuda.get_device_name(device))
    else:
        irregular = cv2.GaussianBlur(irregular, (0, 0), sigmaX=sigma_ir)
    irregular = np.clip(irregular, -0.95, 0.95)

    # 3) Protección primer plano
    guard = (depth01 < near_guard).astype(np.float32)
    guard = cv2.GaussianBlur(guard, (0, 0), sigmaX=1.5)

    # 4) Sprites
    sprites_alpha = np.zeros((H, W, 1), np.float32)
    sprites_rgb   = np.zeros((H, W, 3), np.float32)
    if sprites:
        for _ in range(n_sprites):
            sp = random.choice(sprites)
            z_s = rng.uniform(*sprite_depth_range)
            depth_gate = (depth01 >= z_s).astype(np.float32)[:, :, None]

            x = rng.integers(0, W)
            y = rng.integers(int(H * 0.3), int(H * 0.95))
            sc = rng.uniform(*sprite_scale_range)
            ang = rng.uniform(-5, 5)

            tmp_rgb = np.zeros_like(sprites_rgb)
            tmp_a   = np.zeros_like(sprites_alpha)
            place_sprite_alpha(tmp_rgb, tmp_a, sp, x, y, scale=sc, angle_deg=ang)

            tmp_a *= depth_gate * float(sprite_alpha_gain)
            tmp_a *= (1.0 - guard[:, :, None])

            eps = 1e-6
            sprites_alpha = np.clip(sprites_alpha, 0.0, 1.0)
            tmp_a         = np.clip(tmp_a,         0.0, 1.0)

            out_a = sprites_alpha + tmp_a * (1.0 - sprites_alpha)
            out_a = np.clip(out_a, eps, 1.0)

            num = sprites_rgb * sprites_alpha + tmp_rgb * tmp_a * (1.0 - sprites_alpha)
            out_rgb = num / out_a

            out_rgb = np.nan_to_num(out_rgb, nan=0.0, posinf=1.0, neginf=0.0)
            out_a   = np.nan_to_num(out_a,   nan=0.0, posinf=1.0, neginf=0.0)

            sprites_rgb, sprites_alpha = np.clip(out_rgb, 0, 1), np.clip(out_a, 0, 1)

        if sprites_alpha.max() > 0:
            r = max(1.0, min(H, W) * 0.004)
            if gpu_mode == "gpu_full" and torch.cuda.is_available() and str(device).startswith("cuda"):
                # blur en GPU
                sa2 = _torch_gaussian_blur_np(sprites_alpha.squeeze(-1), r, device)
                sprites_alpha = ensure_hw1(sa2)
                sprites_rgb   = _torch_gaussian_blur_np(sprites_rgb, r, device)
            else:
                sprites_alpha = cv2.GaussianBlur(sprites_alpha, (0, 0), sigmaX=r)
                sprites_rgb   = cv2.GaussianBlur(sprites_rgb,   (0, 0), sigmaX=r)

        sprites_alpha = ensure_hw1(sprites_alpha)
        sprites_rgb   = ensure_hwc3(sprites_rgb)

    # Máscara base de sprites
    M = np.clip(sprites_alpha[..., 0] if sprites_alpha.ndim == 3 else sprites_alpha, 0.0, 1.0)
    M_cont = np.clip((M ** mask_gamma) * mask_gain, 0.0, 1.0)
    t      = 1.0 - M_cont

    # color niebla
    spr_gray = 0.299*sprites_rgb[:, :, 0] + 0.587*sprites_rgb[:, :, 1] + 0.114*sprites_rgb[:, :, 2]
    spr_gray = np.clip(spr_gray, 0.0, 1.0)
    fog_color = ((1.0 - shade_strength) * np.full_like(rgb, airlight)) #+shade_strength * spr_gray[..., None])

    if beta == 0:
        hazy = rgb * t[..., None] + fog_color * (1.0 - t[..., None])
        M_bin = (M_cont >= 0.5).astype(np.float32)
        return hazy, M_cont, M_bin, t, (sprites_alpha[..., 0] if sprites_alpha.ndim == 3 else sprites_alpha), \
               (beta, airlight, near_guard, n_sprites, sprite_alpha_gain, global_fog, mask_gamma, mask_gain, shade_strength)

    # 5) Transmisión final
    alpha_irreg3 = np.clip(irregular, 0.0, 1.0)[..., None]
    alpha_spr3   = np.clip(sprites_alpha, 0.0, 1.0)
    t3 = np.clip(t_base, 0.0, 1.0)[..., None] * (1.0 - alpha_irreg3) * (1.0 - alpha_spr3)
    t3 = np.clip(t3, t_min, 1.0)

    guard3 = (guard[..., None]).astype(np.float32)
    t3 = t3 * (1.0 - guard3) + guard3 * 1.0
    t3 = 1.0 - global_fog * (1.0 - t3)
    t = t3.squeeze(axis=2)

    M_cont = 1.0 - t
    M_bin  = (M_cont >= 0.5).astype(np.float32)

    # 6) Mezcla final (de todos modos la volvemos a hacer en gpu_light/gpu_full)
    A = np.full_like(rgb, airlight, dtype=np.float32)
    hazy = rgb * t[..., None] + A * (1.0 - t[..., None])

    spr_a_out = sprites_alpha[..., 0] if sprites_alpha.ndim == 3 else sprites_alpha
    return hazy, M_cont, M_bin, t, spr_a_out, \
           (beta, airlight, near_guard, n_sprites, sprite_alpha_gain, global_fog, mask_gamma, mask_gain, shade_strength)

# ---------------------------
# Lectura de profundidad
# ---------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEPTH_EXTS = (".png", ".npy", ".npz", ".tif", ".tiff")

def read_depth_01(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        depth = np.load(path).astype(np.float32)
    elif ext == ".npz":
        npz = np.load(path)
        key = "depth" if "depth" in npz.files else npz.files[0]
        depth = npz[key].astype(np.float32)
    else:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"No se pudo leer el depth: {path}")
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        depth = im.astype(np.float32)

    maxv = float(depth.max()) if depth.size else 1.0
    minv = float(depth.min()) if depth.size else 0.0
    if maxv > 1.0 or minv < 0.0:
        if maxv > 255.0:
            depth = depth / 65535.0
        elif maxv > 1.0:
            depth = depth / 255.0
        depth = np.clip(depth, 0.0, 1.0)
    return depth

# ---------------------------
# Emparejar por base name
# ---------------------------
def build_pairs(images_dir: str, depth_dir: str, depth_suffix: Optional[str]=None, recursive: bool=False) -> List[Tuple[str,str,str]]:
    pattern = "**/*" if recursive else "*"
    img_files = []
    for ext in IMG_EXTS:
        img_files += glob.glob(os.path.join(images_dir, f"{pattern}{ext}"), recursive=recursive)

    pairs = []
    for ip in sorted(img_files):
        base = os.path.splitext(os.path.basename(ip))[0]
        cand_list = []
        if depth_suffix:
            for de in DEPTH_EXTS:
                cand = os.path.join(depth_dir, f"{base}{depth_suffix}{de}")
                if os.path.isfile(cand):
                    cand_list.append(cand)
        else:
            for de in DEPTH_EXTS:
                cand = os.path.join(depth_dir, f"{base}{de}")
                if os.path.isfile(cand):
                    cand_list.append(cand)

        if not cand_list:
            print(f"[WARN] No se encontró depth para {base}")
            continue
        depth_path = cand_list[0]
        pairs.append((ip, depth_path, base))
    return pairs

# ---------------------------
# Proceso por imagen (CPU / GPU-light / GPU-full)
# ---------------------------
def process_one(img_path: str, depth_path: str, outdir: str, sprites, args) -> Dict:
    t0_total = time.time()
    rgb01, _ = read_rgb(img_path)
    depth01 = read_depth_01(depth_path)

    if depth01.shape[:2] != rgb01.shape[:2]:
        depth01 = cv2.resize(depth01, (rgb01.shape[1], rgb01.shape[0]), interpolation=cv2.INTER_LINEAR)

    if args.depth_gamma and args.depth_gamma != 1.0:
        depth01 = np.clip(depth01, 0, 1) ** float(args.depth_gamma)

    # Síntesis (CPU con opcionales de blur en GPU dentro si gpu_full)
    hazy, M_cont, M_bin, tmap, spr_a, params = synthesize_fog(
        rgb01, depth01,
        beta=args.beta,
        airlight=args.airlight,
        noise_strength=args.noise_strength,
        noise_octaves=args.noise_octaves,
        t_min=args.t_min,
        near_guard=args.near_guard,
        sprites=sprites,
        n_sprites=args.n_sprites,
        sprite_depth_range=(args.sprite_depth_min, args.sprite_depth_max),
        sprite_scale_range=(args.sprite_scale_min, args.sprite_scale_max),
        sprite_alpha_gain=args.sprite_alpha_gain,
        seed=args.seed,
        global_fog=args.global_fog,
        gpu_mode=args.gpu_mode,
        device=args.device
    )
    synth_time = time.time() - t0_total

    # Mezcla final en GPU (gpu_light / gpu_full)
    if args.gpu_mode in ("gpu_light", "gpu_full") and torch.cuda.is_available() and str(args.device).startswith("cuda"):
        dev = torch.device(args.device)
        with torch.no_grad():
            t = torch.from_numpy(tmap).to(dev).float().clamp(0, 1)
            A = torch.full((rgb01.shape[0], rgb01.shape[1], 3), float(params[1]), device=dev, dtype=torch.float32)
            rgb_t = torch.from_numpy(rgb01).to(dev).float().clamp(0, 1)
            hazy_t = rgb_t * t[..., None] + A * (1.0 - t[..., None])
            hazy = hazy_t.detach().cpu().numpy()

    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(outdir + "/fog/", exist_ok=True)
    os.makedirs(outdir + "/mask/", exist_ok=True)
    save_rgb(os.path.join(outdir, "fog", f"{base}.png"), hazy)
    save_gray(os.path.join(outdir, "mask", f"{base}.png"), M_cont)
    # Opcionales:
    # save_gray(os.path.join(outdir, f"{base}_mask_bin.png"), M_bin)
    # save_gray(os.path.join(outdir, f"{base}_transmission.png"), tmap)
    # save_gray(os.path.join(outdir, f"{base}_sprites_alpha.png"), spr_a)

    return {
        "base"       : base,
        "input_image": img_path,
        "depth"      : depth_path,
        "out_image"  : os.path.join(outdir, "fog", f"{base}.png"),
        "out_mask"   : os.path.join(outdir, "mask", f"{base}.png"),
        "times": {
            "synthesis_sec": round(synth_time, 4)
        },
        "depth_stats": {
            "min": float(np.min(depth01)),
            "mean": float(np.mean(depth01)),
            "max": float(np.max(depth01))
        },
        "params": {
            "beta":              float(params[0]),
            "airlight":          float(params[1]),
            "near_guard":        float(params[2]),
            "n_sprites":         int(params[3]),
            "sprite_alpha_gain": float(params[4]),
            "global_fog":        float(params[5]),
            "mask_gamma":        float(params[6]),
            "mask_gain":         float(params[7]),
            "shade_strength":    float(params[8])
        }
    }

# ---------------------------
# Utilidad para batches
# ---------------------------
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Fog batch with precomputed depth (CPU / GPU)")
    # Entradas/salidas
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--depth_dir",  required=True)
    ap.add_argument("--outdir",     default="out_fog_batch")
    ap.add_argument("--recursive",  action="store_true")
    ap.add_argument("--depth_suffix", type=str, default=None)

    # Apariencia
    ap.add_argument("--beta", type=float, default=1.1)
    ap.add_argument("--airlight", type=float, default=0.1)
    ap.add_argument("--near-guard", type=float, default=0.10)
    ap.add_argument("--t-min", type=float, default=0.10)
    ap.add_argument("--depth_gamma", type=float, default=1.5)

    # Irregularidad / sprites
    ap.add_argument("--noise-strength", type=float, default=0.0)
    ap.add_argument("--noise-octaves", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sprites_dir", type=str, default=None)
    ap.add_argument("--n-sprites", type=int, default=14)
    ap.add_argument("--sprite-depth-min", type=float, default=0.35)
    ap.add_argument("--sprite-depth-max", type=float, default=0.95)
    ap.add_argument("--sprite-scale-min", type=float, default=0.6)
    ap.add_argument("--sprite-scale-max", type=float, default=1.0)
    ap.add_argument("--sprite-alpha-gain", type=float, default=0.9)
    ap.add_argument("--global-fog", type=float, default=1.0)

    # Rendimiento
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="cpu | cuda | cuda:0 | ...")
    ap.add_argument("--gpu_mode", choices=["cpu", "gpu_light", "gpu_full"], default="cpu",
                    help="cpu=todo CPU; gpu_light=solo mezcla final; gpu_full=blur + mezcla en GPU")
    ap.add_argument("--workers", type=int, default=0,
                    help="Procesos en paralelo (CPU). No usar si gpu_mode!=cpu.")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="N imágenes por batch (para progreso anidado). 0 = sin batches")
    ap.add_argument("--manifest", type=str, default="manifest.json")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Compatibilidad con antiguo flag --gpu_light
    if hasattr(args, "gpu_light") and args.gpu_light:
        args.gpu_mode = "gpu_light"

    # Seguridad: no mezclar procesos con CUDA
    if args.gpu_mode != "cpu" and args.workers > 0:
        print("[WARN] gpu_mode != cpu → forzando --workers 0 para evitar conflictos con CUDA.")
        args.workers = 0

    # Cargar sprites
    sprites = load_sprites_prebaked(args.sprites_dir, args.n_sprites, seed=args.seed) if args.sprites_dir else None

    # Emparejar
    pairs = build_pairs(args.images_dir, args.depth_dir, args.depth_suffix, args.recursive)
    if not pairs:
        raise SystemExit("[ERROR] No se encontraron pares RGB+Depth.")

    from functools import partial
    job = partial(process_one, outdir=args.outdir, sprites=sprites, args=args)

    results, errors = [], []
    t0 = time.time()

    # Batches (solo para visualización/organización)
    batches = list(chunked(pairs, args.batch_size)) if args.batch_size and args.batch_size > 0 else [pairs]
    pbar_batches = tqdm(total=len(batches), desc="Batches", unit="batch") if len(batches) > 1 else None

    import concurrent.futures as fut
    for bi, batch in enumerate(batches, 1):
        if args.workers > 0:
            with fut.ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(job, ip, dp) for (ip, dp, _b) in batch]
                pbar_imgs = tqdm(total=len(batch), desc=f"Imágenes (batch {bi}/{len(batches)})", unit="img")
                for f in fut.as_completed(futures):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        errors.append(("?", str(e)))
                    pbar_imgs.update(1)
                pbar_imgs.close()
        else:
            pbar_imgs = tqdm(total=len(batch), desc=f"Imágenes (batch {bi}/{len(batches)})", unit="img")
            for (ip, dp, base) in batch:
                try:
                    results.append(job(ip, dp))
                except Exception as e:
                    errors.append((base, str(e)))
                pbar_imgs.update(1)
            pbar_imgs.close()

        if pbar_batches:
            pbar_batches.update(1)

    if pbar_batches:
        pbar_batches.close()

    manifest = {
        "images_dir": args.images_dir,
        "depth_dir": args.depth_dir,
        "outdir": args.outdir,
        "count": len(results),
        "errors": errors,
        "device": args.device,
        "gpu_mode": args.gpu_mode,
        "timing_sec": round(time.time() - t0, 3),
        "items": results
    }
    with open(os.path.join(args.outdir, args.manifest), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Procesadas {len(results)} imágenes en {manifest['timing_sec']} s. "
          f"Errores: {len(errors)}. Manifest en {os.path.join(args.outdir, args.manifest)}")

if __name__ == "__main__":
    main()
