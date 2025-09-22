# prep_sprites_bank.py
import os, glob, argparse, json, random
import numpy as np
import cv2

def ensure_rgba(img):
    # img puede ser GRAY, BGR, BGRA
    if img is None:
        return None
    if img.ndim == 2:
        a = img.astype(np.float32)/255.0
        bgr = np.dstack([img,img,img]).astype(np.float32)/255.0
        return np.dstack([bgr, a])
    if img.shape[2] == 3:
        b,g,r = cv2.split(img.astype(np.float32)/255.0)
        lum = 0.114*b + 0.587*g + 0.299*r
        a = np.clip(lum,0,1)
        return np.dstack([b,g,r,a])
    if img.shape[2] >= 4:
        img = img[:,:,:4].astype(np.float32)/255.0
        return img
    return None

def resize_max(img, max_dim):
    h,w = img.shape[:2]
    m = max(h,w)
    if m <= max_dim: return img
    s = max_dim / float(m)
    nh, nw = int(round(h*s)), int(round(w*s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def rotate_rgba(img_rgba, angle_deg):
    h,w = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle_deg, 1.0)
    out = cv2.warpAffine(img_rgba, M, (w,h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_TRANSPARENT)
    # volver a 4 canales
    if out.ndim == 2: out = out[...,None]
    if out.shape[2] == 3:
        # si perdió alpha, crear uno nulo (no debería pasar con TRANSPARENT)
        out = np.dstack([out, np.zeros((h,w,1), np.float32)])
    return out

def save_npz(dst_path, rgb_premul, alpha, meta):
    # Guardar en float16 para IO rápido (sin pérdida visible)
    np.savez_compressed(dst_path,
        rgb_premul=rgb_premul.astype(np.float16),
        alpha=alpha.astype(np.float16),
        meta=json.dumps(meta))

def main():
    ap = argparse.ArgumentParser("Pre-bake sprites to a fast bank")
    ap.add_argument("--src", required=True, help="Carpeta con PNGs originales")
    ap.add_argument("--dst", required=True, help="Carpeta de salida (banco)")
    ap.add_argument("--max-dim", type=int, default=2048, help="Máximo lado mayor")
    ap.add_argument("--scales", type=str, default="1.0", help="Escalas separadas por coma, ej: 0.5,0.75,1.0,1.25")
    ap.add_argument("--angles", type=str, default="-20,-10,-5,0,5,10,20", help="Ángulos en grados, ej: -10,-5,0,5,10")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",")]
    angles = [float(a) for a in args.angles.split(",")]

    index = []
    paths = []
    for ext in ("*.png","*.PNG"):
        paths += glob.glob(os.path.join(args.src, ext))
    paths.sort()

    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        rgba0 = ensure_rgba(cv2.imread(p, cv2.IMREAD_UNCHANGED))
        if rgba0 is None: 
            print("[WARN] no se pudo leer:", p); 
            continue

        rgba0 = resize_max(rgba0, args.max_dim)
        h0,w0 = rgba0.shape[:2]
        bgr0, a0 = rgba0[:,:,:3], rgba0[:,:,3:4]
        # premultiplicado
        rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
        base_rgb_premul = rgb0 * a0

        for s in scales:
            # escalar
            if s != 1.0:
                nh, nw = int(round(h0*s)), int(round(w0*s))
                if nh < 2 or nw < 2: continue
                rgb_premul = cv2.resize(base_rgb_premul, (nw, nh), interpolation=cv2.INTER_AREA)
                alpha      = cv2.resize(a0,             (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                rgb_premul = base_rgb_premul.copy()
                alpha = a0.copy()

            for ang in angles:
                if ang != 0:
                    # rotar rgb premul y alpha juntos
                    rgbA = np.dstack([cv2.cvtColor(rgb_premul, cv2.COLOR_RGB2BGR), alpha])
                    rot  = rotate_rgba(rgbA, ang)
                    rot_rgb = cv2.cvtColor(rot[:,:,:3], cv2.COLOR_BGR2RGB)
                    rot_a   = rot[:,:,3:4]
                    rgb_p, a = rot_rgb, rot_a
                else:
                    rgb_p, a = rgb_premul, alpha

                out_name = f"{name}_s{str(s).replace('.','p')}_a{int(ang)}.npz"
                out_path = os.path.join(args.dst, out_name)
                meta = {"src": p, "scale": s, "angle": ang, "h": int(rgb_p.shape[0]), "w": int(rgb_p.shape[1])}
                save_npz(out_path, rgb_p, a, meta)
                index.append({"file": out_name, **meta})

    with open(os.path.join(args.dst, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[OK] Banco listo en {args.dst} con {len(index)} variantes")

if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"[INFO] Tiempo total: {time.time() - t0:.2f} segundos")
