import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

def read_gray(path):
    """Lee imagen en escala de grises (uint8). Lanza error si no existe."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")
    return im

def read_color(path):
    """Lee imagen color BGR (uint8). Lanza error si no existe."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")
    return im

def to_3ch_from_gray(gray, target_shape):
    """Expande una imagen 1 canal a 3 canales y la redimensiona al tamaño target."""
    Ht, Wt = target_shape[:2]
    if gray.shape[:2] != (Ht, Wt):
        gray = cv2.resize(gray, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    return np.dstack([gray, gray, gray])

def resize_like(img, target_shape, interpolation=cv2.INTER_LINEAR):
    """Redimensiona img al tamaño de target_shape (H, W, [C])."""
    Ht, Wt = target_shape[:2]
    if img.shape[:2] != (Ht, Wt):
        img = cv2.resize(img, (Wt, Ht), interpolation=interpolation)
    return img

def choose_mask(base, cont_dir, bin_dir, prefer_cont=True):
    """
    Devuelve (mask_path, tipo) buscando por nombre base en cont_dir y bin_dir.
    tipo ∈ {'cont','bin',None}  (None si no hay máscara)
    """
    cand = []
    if cont_dir:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            p = cont_dir / f"{base}{ext}"
            if p.exists():
                cand.append((p, "cont"))
                break
    if bin_dir:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            p = bin_dir / f"{base}{ext}"
            if p.exists():
                cand.append((p, "bin"))
                break

    if not cand:
        return None, None

    if len(cand) == 2:
        # hay continua y binaria
        if prefer_cont:
            return [c for c in cand if c[1] == "cont"][0]
        else:
            return [c for c in cand if c[1] == "bin"][0]
    else:
        return cand[0]

def fuse_with_mask(ir_img, fus1_img, mask_uint8):
    """
    Aplica I_Fus2 = M*IR + (1-M)*Fus1
    - ir_img: uint8, 1ch o 3ch
    - fus1_img: uint8, 3ch (se tomará como referencia de tamaño y canales)
    - mask_uint8: uint8 1ch [0..255]
    Retorna uint8 3ch (BGR).
    """
    # Asegurar tamaños
    fus1 = fus1_img
    mask = resize_like(mask_uint8, fus1.shape, interpolation=cv2.INTER_LINEAR)

    # IR -> 3 canales y tamaño de fus1
    if ir_img.ndim == 2:
        ir3 = to_3ch_from_gray(ir_img, fus1.shape)
    else:
        ir3 = resize_like(ir_img, fus1.shape, interpolation=cv2.INTER_LINEAR)

    # Preparar en float32
    fus1_f = fus1.astype(np.float32)
    ir3_f  = ir3.astype(np.float32)
    # Normalizar máscara a [0,1] y expandir a (H,W,1)
    M = (mask.astype(np.float32) / 255.0)[..., None]

    # Ecuación
    fus2 = M * ir3_f + (1.0 - M) * fus1_f
    fus2 = np.clip(fus2, 0, 255).astype(np.uint8)
    return fus2

def main():
    ap = argparse.ArgumentParser(description="Aplica I_Fus2 = M*IR + (1-M)*Fus1 usando máscaras 0..255.")
    ap.add_argument("--ir_dir", required=True, type=str, help="Directorio de imágenes IR (grises o color).")
    ap.add_argument("--fus1_dir", required=True, type=str, help="Directorio de imágenes fusionadas (Fus1).")
    ap.add_argument("--mask_cont_dir", type=str, default=None, help="Directorio de máscaras continuas (0..255).")
    ap.add_argument("--mask_bin_dir", type=str, default=None, help="Directorio de máscaras binarias (0/255).")
    ap.add_argument("--prefer", choices=["cont","bin"], default="cont",
                    help="Si existen ambas máscaras para un archivo, cuál priorizar.")
    ap.add_argument("--out_dir", required=True, type=str, help="Directorio de salida.")
    ap.add_argument("--ir_prefix", type=str, default="", help="Prefijo opcional en nombres IR (si aplica).")
    ap.add_argument("--fus1_prefix", type=str, default="", help="Prefijo opcional en nombres Fus1.")
    ap.add_argument("--save_csv", action="store_true", help="Guardar log CSV de archivos procesados.")
    args = ap.parse_args()

    ir_dir = Path(args.ir_dir)
    fus1_dir = Path(args.fus1_dir)
    cont_dir = Path(args.mask_cont_dir) if args.mask_cont_dir else None
    bin_dir  = Path(args.mask_bin_dir) if args.mask_bin_dir else None
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Indexar por base name (sin extensión)
    def index_dir(d: Path, prefix=""):
        idx = {}
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
            for p in d.glob(ext):
                base = p.stem
                if prefix and base.startswith(prefix):
                    base_key = base[len(prefix):]
                else:
                    base_key = base
                idx[base_key] = p
        return idx

    ir_idx   = index_dir(ir_dir, prefix=args.ir_prefix)
    fus1_idx = index_dir(fus1_dir, prefix=args.fus1_prefix)

    common_bases = sorted(set(ir_idx.keys()) & set(fus1_idx.keys()))
    if not common_bases:
        print("No hay nombres en común entre IR y Fus1. Revisa prefijos/archivos.")
        return

    log_rows = []
    print(f"Encontrados {len(common_bases)} pares IR/Fus1. Procesando...")
    for base in tqdm(common_bases):
        ir_path   = ir_idx[base]
        fus1_path = fus1_idx[base]

        mask_path, mtype = choose_mask(
            base, cont_dir, bin_dir, prefer_cont=(args.prefer=="cont")
        )
        if mask_path is None:
            # si no hay máscara, saltar
            log_rows.append([base, str(ir_path), str(fus1_path), "", "NO_MASK"])
            continue

        # Leer
        ir = read_gray(ir_path) if cv2.imread(str(ir_path), cv2.IMREAD_COLOR) is None else read_color(ir_path)
        fus1 = read_color(fus1_path)
        mask = read_gray(mask_path)  # 0..255

        # Aplicar fusión
        fus2 = fuse_with_mask(ir, fus1, mask)

        # Guardar (mantengo extensión de Fus1 para salida)
        out_name = f"{base}{''.join(Path(fus1_path).suffixes)}"
        cv2.imwrite(str(out_dir / out_name), fus2)

        log_rows.append([base, str(ir_path), str(fus1_path), str(mask_path), mtype])

    if args.save_csv:
        csv_path = out_dir / "fusion_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["base","ir_path","fus1_path","mask_path","mask_type"])
            w.writerows(log_rows)
        print(f"Log guardado en: {csv_path}")

    print(f"Listo. Resultados en: {out_dir}")

if __name__ == "__main__":
    main()
