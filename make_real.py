import os, cv2, math, glob, random, argparse, json
import numpy as np
from PIL import Image

import torch
# Importa tus backends tal cual me pasaste
import depth_backends as DB
import time


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
# Sprites (nubes de niebla) desde imágenes PNG
# ---------------------------
def load_sprites(folder):
    paths = []
    for ext in ("*.png","*.PNG"):
        paths += glob.glob(os.path.join(folder, ext))
    sprites = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # puede ser BGR, BGRA o 1 canal
        if im is None: 
            continue
        im = im.astype(np.float32) / 255.0

        if im.ndim == 2:
            # 1 canal -> usarlo como alpha; color gris
            a = im
            bgr = np.stack([im, im, im], axis=2)
            im = np.dstack([bgr, a])
        elif im.shape[2] == 3:
            # Sin alpha: construimos alpha a partir de luminancia (más claro = más niebla)
            b,g,r = cv2.split(im)
            lum = 0.114*b + 0.587*g + 0.299*r
            a = np.clip(lum, 0, 1)
            im = np.dstack([im, a])
        elif im.shape[2] >= 4:
            # Asegurar solo 4 canales
            im = im[:, :, :4]

        sprites.append(im)
    if not sprites:
        print("[WARN] No se encontraron sprites PNG válidos en", folder)
    return sprites


# ---------------------------
# Sprites (nubes de niebla) desde un formato NPZ preprocesado
# ---------------------------
def load_sprites_prebaked(bank_dir, n_select, seed=42):
    """Carga N variantes NPZ (premultiplied) a float32 SOLO para lo que usarás."""
    idx_path = os.path.join(bank_dir, "index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    rng = random.Random(seed)
    picks = rng.sample(index, k=min(n_select, len(index)))

    sprites = []
    for item in picks:
        npz = np.load(os.path.join(bank_dir, item["file"]))
        rgb_p = npz["rgb_premul"].astype(np.float32)   # (H,W,3) premultiplicado
        a     = npz["alpha"].astype(np.float32)        # (H,W,1)
        # Volver a RGB “no premul” si tu pipeline lo necesita, o mantén premul y cambia la composición.
        # Aquí volvemos a no premul para ser plug&play:
        eps = 1e-6
        a_safe = np.clip(a, eps, 1.0)
        rgb = rgb_p / a_safe
        rgb = np.clip(rgb, 0.0, 1.0)
        bgra = np.dstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), a])
        sprites.append(bgra)
    return sprites


def place_sprite_alpha(dst_rgb, dst_alpha, sprite_bgra, x, y, scale=1.0, angle_deg=0.0):
    H,W,_ = dst_rgb.shape
    sbgr = sprite_bgra[:,:,:3]
    sa   = sprite_bgra[:,:,3]  # <-- 2D por ahora

    # Escala
    nh = max(2, int(sbgr.shape[0]*scale))
    nw = max(2, int(sbgr.shape[1]*scale))
    sbgr = cv2.resize(sbgr, (nw, nh), interpolation=cv2.INTER_AREA)
    sa   = cv2.resize(sa,   (nw, nh), interpolation=cv2.INTER_AREA)

    # Rotación (usar borde transparente)
    M = cv2.getRotationMatrix2D((nw/2, nh/2), angle_deg, 1.0)
    sbgr = cv2.warpAffine(sbgr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    sa   = cv2.warpAffine(sa,   M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Asegurar alfa en 3D
    if sa.ndim == 2:
        sa = sa[..., None]  # (H,W,1)

    # Recorte a la imagen destino
    x0 = int(x - nw//2); y0 = int(y - nh//2)
    x1 = max(0, x0); y1 = max(0, y0)
    x2 = min(W, x0+nw); y2 = min(H, y0+nh)
    if x2<=x1 or y2<=y1: 
        return
    sx1 = x1 - x0; sy1 = y1 - y0; sx2 = sx1 + (x2-x1); sy2 = sy1 + (y2-y1)

    patch_rgb = dst_rgb[y1:y2, x1:x2, :]
    patch_a   = dst_alpha[y1:y2, x1:x2, :]

    s_rgb = sbgr[sy1:sy2, sx1:sx2, :]
    s_a   = sa[sy1:sy2, sx1:sx2, :]   # ahora sí 3D

    eps = 1e-6
    patch_a = np.clip(patch_a, 0.0, 1.0)
    s_a     = np.clip(s_a,     0.0, 1.0)

    # Composición alfa "over"
    out_a = patch_a + s_a*(1.0 - patch_a)
    out_a = np.clip(out_a, eps, 1.0)

    num = patch_rgb*patch_a + s_rgb*s_a*(1.0 - patch_a)
    out_rgb = num / out_a

    # Evitar NaNs/Inf por seguridad
    out_rgb = np.nan_to_num(out_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    out_a   = np.nan_to_num(out_a,   nan=0.0, posinf=1.0, neginf=0.0)

    dst_rgb[y1:y2, x1:x2, :]  = np.clip(out_rgb, 0.0, 1.0)
    dst_alpha[y1:y2, x1:x2, :] = np.clip(out_a,   0.0, 1.0)



# ---------------------------
# Síntesis de niebla (profundidad + irregularidad + sprites)
# ---------------------------
def synthesize_fog(
    rgb, depth01,
    beta=1.2,
    airlight=0.88,
    noise_strength=0.45,
    noise_octaves=4,
    t_min=0.1,
    near_guard=0.08,         # píxeles con profundidad < near_guard quedan sin niebla (t=1)
    sprites=None,
    n_sprites=12,
    sprite_depth_range=(0.25, 1.0),
    sprite_scale_range=(0.5, 2.0),
    sprite_alpha_gain=0.9,
    seed=42,
    global_fog=1.0
):
    
    # Para que los parámetros varíen entre imágenes:
    # beta
    # airlight
    # near_guard
    # n_sprites
    # sprite_alpha_gain
    # global_fog
    H,W,_ = rgb.shape
    rng = np.random.default_rng(seed)

    # 1) Transmisión base (0 cerca, 1 lejos)
    t_base = np.exp(-beta * np.clip(depth01, 0, 1)).astype(np.float32)

    # 2) Irregularidad
    noise = fbm_noise(H, W, octaves=noise_octaves, seed=seed)
    irregular = noise_strength * (noise - 0.5) * 2.0
    irregular = cv2.GaussianBlur(irregular, (0,0), sigmaX=max(1.0, min(H,W)*0.006))
    irregular = np.clip(irregular, -0.95, 0.95)

    # 3) Protección primer plano
    guard = (depth01 < near_guard).astype(np.float32)
    guard = cv2.GaussianBlur(guard, (0,0), sigmaX=1.5)

    # 4) Sprites
    sprites_alpha = np.zeros((H,W,1), np.float32)
    sprites_rgb   = np.zeros((H,W,3), np.float32)
    if sprites:
        for _ in range(n_sprites):
            sp = random.choice(sprites)
            # print(f"  Usando sprite de tamaño {sp.shape}")

            z_s = rng.uniform(*sprite_depth_range)
            depth_gate = (depth01 >= z_s).astype(np.float32)[:,:,None]

            x = rng.integers(0, W)
            y = rng.integers(int(H*0.3), int(H*0.95))
            sc = rng.uniform(*sprite_scale_range)
            ang = rng.uniform(-10, 10)

            tmp_rgb = np.zeros_like(sprites_rgb)
            tmp_a   = np.zeros_like(sprites_alpha)
            place_sprite_alpha(tmp_rgb, tmp_a, sp, x, y, scale=sc, angle_deg=ang)

            tmp_a *= depth_gate * sprite_alpha_gain
            tmp_a *= (1.0 - guard[:,:,None])  # no afecta primer plano

            eps = 1e-6
            sprites_alpha = np.clip(sprites_alpha, 0.0, 1.0)
            tmp_a         = np.clip(tmp_a,         0.0, 1.0)

            out_a = sprites_alpha + tmp_a*(1.0 - sprites_alpha)
            out_a = np.clip(out_a, eps, 1.0)

            num = sprites_rgb*sprites_alpha + tmp_rgb*tmp_a*(1.0 - sprites_alpha)
            out_rgb = num / out_a

            out_rgb = np.nan_to_num(out_rgb, nan=0.0, posinf=1.0, neginf=0.0)
            out_a   = np.nan_to_num(out_a,   nan=0.0, posinf=1.0, neginf=0.0)

            sprites_rgb, sprites_alpha = np.clip(out_rgb,0,1), np.clip(out_a,0,1)

        if sprites_alpha.max() > 0:
            r = max(1.0, min(H,W)*0.004)
            sprites_alpha = cv2.GaussianBlur(sprites_alpha, (0,0), sigmaX=r)
            sprites_rgb   = cv2.GaussianBlur(sprites_rgb,   (0,0), sigmaX=r)
    
    M = np.clip(sprites_alpha.squeeze(), 0.0, 1.0)

    # Perillas rápidas (sin tocar CLI):
    mask_gamma = 0.7     # <1 refuerza medios (más denso)
    mask_gain  = 1.2     # >1 sube densidad global
    shade_strength = .5  # 0..1 cuánto usar la luminancia del sprite en el color de la niebla

    # Densidad final
    M_cont = np.clip((M ** mask_gamma) * mask_gain, 0.0, 1.0)
    t      = 1.0 - M_cont

    # Color de niebla: mezcla entre A y la luminancia del sprite (mantiene “volumen” del PNG)
    spr_gray = 0.299*sprites_rgb[:,:,0] + 0.587*sprites_rgb[:,:,1] + 0.114*sprites_rgb[:,:,2]
    spr_gray = np.clip(spr_gray, 0.0, 1.0)
    fog_color = ((1.0 - shade_strength) * np.full_like(rgb, airlight) +
                shade_strength * spr_gray[..., None])

    if beta == 0:
        # Render final
        hazy = rgb * t[..., None] + fog_color * (1.0 - t[..., None])

        # Salidas (mask_cont = sprites con ajustes)
        M_bin  = (M_cont >= 0.5).astype(np.float32)
        return hazy, M_cont, M_bin, t, sprites_alpha.squeeze()
    
    else:
        # 5) Transmisión final
        alpha_irreg = np.clip((irregular.clip(0,1))[:,:,None], 0, 1)   # (H,W,1)
        alpha_spr   = np.clip(sprites_alpha, 0, 1)                     # (H,W,1)

        # Asegurar mismas dimensiones (H,W,1)
        t_base3      = t_base[..., None]                                # (H,W,1)
        alpha_irreg3 = np.clip(irregular, 0.0, 1.0)[..., None]          # (H,W,1)
        alpha_spr3   = alpha_spr[..., None]                             # (H,W,1)

        t3 = t_base3 * (1.0 - alpha_irreg3) * (1.0 - alpha_spr3)
        t3 = np.clip(t3, t_min, 1.0)

        # Zonas protegidas (primer plano): forzar t=1
        guard3 = guard[..., None]                                       # (H,W,1)
        t3 = t3 * (1.0 - guard3) + guard3 * 1.0
        
        t3 = 1.0 - global_fog * (1.0 - t3)   # escala la densidad total


        # Pasar a 2D
        t = t3.squeeze(2)

        # 6) Máscaras
        M_cont = 1.0 - t
        M_bin  = (M_cont >= 0.5).astype(np.float32)

        # Render final
        A = np.full_like(rgb, airlight, dtype=np.float32)               # (H,W,3)
        hazy = rgb * t[..., None] + A * (1.0 - t[..., None])

        return hazy, M_cont, M_bin, t, sprites_alpha.squeeze()


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Fog single with runtime depth")
    ap.add_argument("--image", required=True, help="Imagen RGB de entrada")
    ap.add_argument("--outdir", default="out_fog", help="Carpeta de salida")
    ap.add_argument("--backend", default="depth-anything-v2",
                    choices=["depth-anything-v2","dav2","danythingv2","zoedepth","zoe","midas","dpt","monodepth2"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--invert-depth", action="store_false",
                    help="Invierte profundidad si tu backend devuelve 1=cerca, 0=lejos")
    # Físicos / apariencia
    ap.add_argument("--beta", type=float, default=1.1)
    ap.add_argument("--airlight", type=float, default=0.88)
    ap.add_argument("--near-guard", type=float, default=0.10)
    ap.add_argument("--t-min", type=float, default=0.10)
    # Irregularidad
    ap.add_argument("--noise-strength", type=float, default=0.50)
    ap.add_argument("--noise-octaves", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # Sprites
    ap.add_argument("--sprites-dir", type=str, default=None, help="Carpeta con PNGs de niebla (RGBA)")
    ap.add_argument("--n-sprites", type=int, default=14)
    ap.add_argument("--sprite-depth-min", type=float, default=0.35)
    ap.add_argument("--sprite-depth-max", type=float, default=0.95)
    ap.add_argument("--sprite-scale-min", type=float, default=0.6)
    ap.add_argument("--sprite-scale-max", type=float, default=1.0)
    ap.add_argument("--sprite-alpha-gain", type=float, default=0.9)
    # Global fog factor (1.0 = normal, 0.0 = no fog)
    ap.add_argument("--global-fog", type=float, default=1.0)


    # Opcional monodepth2 (si lo usas)
    ap.add_argument("--mono-model-dir", type=str, default=None,
                    help="Directorio con encoder.pth y depth.pth (solo monodepth2)")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) RGB
    rgb01, pil_rgb = read_rgb(args.image)
    H,W,_ = rgb01.shape

    # 2) Backend de profundidad
    t0 = time.time()
    if args.backend.lower() in ["monodepth2"]:
        if not args.mono_model_dir:
            raise SystemExit("--mono-model-dir es requerido para monodepth2")
        # Importa clases de tu implementación de monodepth2
        from layers import disp_to_depth  # si lo necesitas
        from networks.resnet_encoder import ResnetEncoder
        from networks.depth_decoder import DepthDecoder
        monodepth2_cfg = dict(model_dir=args.mono_model_dir, device=args.device,
                              encoder_class=ResnetEncoder, depth_decoder_class=DepthDecoder)
        backend = DB.make_backend("monodepth2", device=args.device, monodepth2_cfg=monodepth2_cfg)
        depth01 = DB.predict_depth(backend, pil_rgb, monodepth2_classes=(ResnetEncoder, DepthDecoder))
    else:
        backend = DB.make_backend(args.backend, device=args.device)
        depth01 = DB.predict_depth(backend, pil_rgb)
        gamma_d = 1.5  # 1.2–2.0 comprime ‘lejanos’
        depth01 = np.clip(depth01, 0, 1) ** gamma_d


    if args.invert_depth:
        depth01 = 1.0 - depth01
        
    print(f"[INFO] Tiempo de # 2) Backend de profundidad: {time.time()-t0:.3f} s")

    # 3) Sprites
    t0 = time.time()
    sprites = load_sprites_prebaked(args.sprites_dir, args.n_sprites, seed=args.seed) if args.sprites_dir else None
    print(f"[INFO] # 3) Sprites tiempo: {time.time()-t0:.3f} s")

    # 4) Niebla
    t0 = time.time()
    hazy, M_cont, M_bin, tmap, spr_a = synthesize_fog(
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
        global_fog=args.global_fog
    )
    print(f"[INFO] Tiempo de # 4) Niebla: {time.time()-t0:.3f} s")

    # 5) Guardar
    base = os.path.splitext(os.path.basename(args.image))[0]
    save_rgb(os.path.join(args.outdir, f"{base}_fog.png"), hazy)
    save_gray(os.path.join(args.outdir, f"{base}_depth01.png"), np.clip(depth01,0,1))
    save_gray(os.path.join(args.outdir, f"{base}_mask_cont.png"), M_cont)
    # save_gray(os.path.join(args.outdir, f"{base}_mask_bin.png"), M_bin)
    # save_gray(os.path.join(args.outdir, f"{base}_transmission.png"), tmap)
    # save_gray(os.path.join(args.outdir, f"{base}_sprites_alpha.png"), spr_a)
    print(f"[OK] Salidas en {args.outdir}")
    print("depth01  min/mean/max:", float(depth01.min()), float(depth01.mean()), float(depth01.max()))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[INFO] Tiempo total: {time.time()-t0:.3f} s")
