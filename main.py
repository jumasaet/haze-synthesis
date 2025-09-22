from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from tqdm import tqdm

# --- tus imports existentes ---
import networks
from depth_backends import make_backend, predict_depth
from depth_backends import predict_depth_batch as backend_predict_depth_batch


# Warnings de torchvision
warnings.filterwarnings("ignore",
                        message="Using 'weights' as positional parameter*",
                        category=UserWarning)
warnings.filterwarnings("ignore",
                        message="Arguments other than a weight enum*",
                        category=UserWarning)


def simple_collate(batch):
    # batch es una lista de tuplas: [(path, pil, clean_np, name), ...]
    # devolvemos listas por campo
    return list(zip(*batch))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for depth->haze pipeline (batched).')

    # Entradas / salidas
    parser.add_argument('--image_path', type=str, required=True,
                        help='path a imagen o carpeta de imágenes')
    parser.add_argument('--ext', type=str, default="png",
                        help='extensión a buscar dentro de la carpeta')
    parser.add_argument('--output_image_path', type=str, required=True,
                        help='carpeta de salida')

    # Backend de profundidad
    parser.add_argument('--depth_backend', type=str, default="monodepth2",
                        choices=["monodepth2", "depth-anything-v2", "zoedepth", "midas"],
                        help="Backend de estimación de profundidad")
    parser.add_argument('--invert_depth_for_haze', action='store_true',
                        help="Invierte el mapa [0,1] antes de generar la niebla")

    # Compatibilidad monodepth2
    parser.add_argument('--model_name', type=str,
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"],
                        default="mono+stereo_1024x320",
                        help='modelo preentrenado monodepth2 (si usas depth_backend=monodepth2)')
    parser.add_argument("--pred_metric_depth", action='store_true',
                        help='(Monodepth2) stereo-trained KITTI -> metric depth')

    # Dispositivo
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--device", type=int, default=0)

    # Parámetros de niebla y binarización
    parser.add_argument('--beta', type=float, default=1.0, help='grado de niebla (0.5-3.0)')
    parser.add_argument('--airlight', type=float, default=255.0, help='luz atmosférica A')
    parser.add_argument("--thres", type=float, default=0.5, help='umbral [0-1] para binarizar')

    # Limpieza morfológica
    parser.add_argument("--min_blob_frac", type=float, default=0.005,
                        help="fracción mínima del área para conservar componente blanco")
    parser.add_argument("--max_hole_frac", type=float, default=0.005,
                        help="fracción máxima del área para rellenar huecos")
    parser.add_argument("--k_frac", type=float, default=0.01,
                        help="tamaño del kernel como fracción del lado menor")
    parser.add_argument("--open_iters", type=int, default=1, help="iteraciones de apertura")
    parser.add_argument("--close_iters", type=int, default=1, help="iteraciones de cierre")

    # --- NUEVO: batching + IO ---
    parser.add_argument("--batch_size", type=int, default=8, help="tamaño de batch para inferencia")
    parser.add_argument("--num_workers", type=int, default=4, help="workers para DataLoader (I/O)")
    parser.add_argument("--save_workers", type=int, default=4, help="hilos para guardado en disco")

    return parser.parse_args()


# ====================== utilidades de tu pipeline ======================
def gen_haze(clean_img, depth_img, beta=1.0, A=150):
    depth_img_3c = np.repeat(depth_img[:, :, np.newaxis], 3, axis=2)
    norm_depth_img = depth_img_3c / 255.0
    trans = np.exp(-norm_depth_img * beta)
    mask = (1.0 - trans)
    hazy = clean_img * trans + A * (1 - trans)
    hazy = np.array(hazy, dtype=np.uint8)
    mask_uint8 = (mask[:, :, 0] * 255).astype(np.uint8)
    return hazy, mask_uint8

def normalize_mask(mask):
    return mask.astype(np.float32) / 255.0

def binarize_mask(mask, threshold=0.6, save_uint8=True):
    if mask.max() > 1.0:
        mask = normalize_mask(mask)
    binary = (mask > threshold).astype(np.uint8)
    return binary * 255 if save_uint8 else binary

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
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep

def fill_small_holes(bin01, max_hole_area=50):
    inv = 1 - bin01
    inv_uint8 = (inv * 255).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv_uint8, connectivity=8)
    small_holes = np.zeros_like(inv, dtype=np.uint8)
    for i in range(1, num):
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


# ====================== Dataset y helpers de batching ======================
class ImageListDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = pil.open(p).convert('RGB')
        clean_np = np.array(im, dtype=np.uint8)  # para OpenCV/generación
        name = os.path.splitext(os.path.basename(p))[0]
        return p, im, clean_np, name

def predict_depth_batch_fallback(backend, pil_list, monodepth2_classes):
    """
    Fallback seguro: usa la función unitaria existente.
    Devuelve lista de mapas de profundidad [H,W] en [0,1].
    """
    outs = []
    for im in pil_list:
        d01 = predict_depth(
            backend=backend,
            pil_rgb=im,
            monodepth2_classes=monodepth2_classes
        )
        outs.append(d01)
    return outs

def try_predict_depth_batch(backend, pil_list, monodepth2_classes):
    """
    Usa la función batch del backend si está presente;
    si falla por cualquier motivo, cae en unitario uno-a-uno.
    """
    try:
        return backend_predict_depth_batch(backend=backend, pil_list=pil_list, monodepth2_classes=monodepth2_classes)
    except Exception as e:
        # Fallback seguro (sin perder el lote completo)
        outs = []
        for im in pil_list:
            outs.append(predict_depth(backend=backend, pil_rgb=im, monodepth2_classes=monodepth2_classes))
        return outs


# ====================== Pipeline principal (batched) ======================
def test_batched(args):
    # Dispositivo
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if use_cuda and args.device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.device}")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(args.device)} (ID: {args.device})\n")
    else:
        if use_cuda and args.device >= torch.cuda.device_count():
            print(f"Warning: GPU ID {args.device} not available. Using CPU instead.")
        device = torch.device("cpu")
        print("Using device: CPU")

    # Backend
    monodepth2_cfg = None
    monodepth2_classes = None
    if args.depth_backend == "monodepth2":
        model_path = os.path.join("models", args.model_name)
        print("-> Loading monodepth2 from", model_path)
        monodepth2_cfg = {
            "model_dir": model_path,
            "device": device,
            "encoder_class": networks.ResnetEncoder,
            "depth_decoder_class": networks.DepthDecoder
        }
        monodepth2_classes = (networks.ResnetEncoder, networks.DepthDecoder)

    backend = make_backend(args.depth_backend, device, monodepth2_cfg=monodepth2_cfg)

    # Listar imágenes
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, f'*.{args.ext}'))
        print("-> No. of images in folder:", len(paths))
    else:
        raise Exception(f"Can not find args.image_path: {args.image_path}")

    print("-> Predicting on {:d} test images".format(len(paths)))

    # Carpetas de salida
    out_root = args.output_image_path
    os.makedirs(os.path.join(out_root, "img"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "norm"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "binary"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "clean"), exist_ok=True)

    # DataLoader para IO paralelo y collate simple (listas)
    ds = ImageListDataset(paths)
    dl = DataLoader(
        ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=True if (torch.cuda.is_available() and not args.no_cuda) else False,
        collate_fn=simple_collate,          # ✅ función a nivel de módulo
        persistent_workers=True if args.num_workers > 0 else False
    )


    # Pequeño pool para guardado en disco
    saver_pool = ThreadPoolExecutor(max_workers=max(1, args.save_workers))

    save_futures = []
    total = len(paths)

    with torch.no_grad():
        pbar = tqdm(total=total, desc="Processing (batched)", unit="img")
        processed = 0

        for (batch_paths, pil_list, clean_list, names_list) in dl:
            # (1) DEPTH: intenta modo batch
            depth_list_01 = try_predict_depth_batch(
                backend=backend,
                pil_list=pil_list,
                monodepth2_classes=monodepth2_classes
            )

            # Inversión opcional
            if args.invert_depth_for_haze:
                depth_list_01 = [1.0 - d for d in depth_list_01]

            # (2) Por imagen dentro del batch: niebla, máscaras y limpieza
            for clean_img, depth01, name, src_path in zip(clean_list, depth_list_01, names_list, batch_paths):
                if str(src_path).endswith("_disp.jpg"):
                    processed += 1
                    pbar.update(1)
                    continue

                depth_u8 = (depth01 * 255.0).astype(np.uint8)

                hazy, mask = gen_haze(clean_img, depth_u8, beta=args.beta, A=args.airlight)
                normalized_mask = normalize_mask(mask)
                binary_mask = binarize_mask(normalized_mask, threshold=args.thres, save_uint8=True)

                H, W = binary_mask.shape[:2]
                binary_mask_clean = clean_binary_mask(
                    binary_mask, H, W,
                    min_blob_frac=args.min_blob_frac,
                    max_hole_frac=args.max_hole_frac,
                    k_frac=args.k_frac,
                    open_iters=args.open_iters,
                    close_iters=args.close_iters
                )

                # (3) Guardado en paralelo
                out_img = os.path.join(out_root, "img",    f"{name}.{args.ext}")
                out_norm = os.path.join(out_root, "norm",  f"{name}.{args.ext}")
                out_bin = os.path.join(out_root, "binary", f"{name}.{args.ext}")
                out_cln = os.path.join(out_root, "clean",  f"{name}.{args.ext}")

                def _save_all():
                    # cv2.imwrite(out_img,  cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(out_norm, (normalized_mask * 255).astype(np.uint8))
                    # cv2.imwrite(out_bin,  binary_mask)
                    cv2.imwrite(out_cln,  binary_mask_clean)

                save_futures.append(saver_pool.submit(_save_all))

                processed += 1
                pbar.set_postfix(batch=f"{processed}/{total}", file=str(name)[:20] + "...")
                pbar.update(1)

        # Esperar a que terminen los guardados
        for f in as_completed(save_futures):
            _ = f.result()

        pbar.close()

    saver_pool.shutdown(wait=True)
    print(f'-> Done! Find outputs in {out_root}')


if __name__ == '__main__':
    args = parse_args()
    t0 = time.time()
    test_batched(args)
    t1 = time.time()
    print(f"Tiempo total de ejecución: {t1 - t0:.2f} s")
