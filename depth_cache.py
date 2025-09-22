# depth_cache.py
import os, argparse, json, math
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import depth_backends as DB
from tqdm import tqdm
from PIL import Image, ImageOps


def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    root = Path(root)
    return [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]

def save_depth_npz(dst_path, depth01):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, depth01=depth01.astype(np.float16))

def main():
    ap = argparse.ArgumentParser("Batch depth caching")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--cache-dir", required=True, help="Dónde guardar las profundidades .npz")
    ap.add_argument("--backend", default="depth-anything-v2",
                    choices=["depth-anything-v2","dav2","danythingv2","zoedepth","zoe","midas","dpt","monodepth2"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gamma-depth", type=float, default=1.0, help="Opcional: depth01 ** gamma")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--invert-depth", action="store_true", help="Guardar depth como 1 - depth01")
    

    # monodepth2 (opcional)
    ap.add_argument("--mono-model-dir", type=str, default="./models/mono+stereo_1024x320/")
    args = ap.parse_args()

    imgs = list_images(args.images_dir)
    if not imgs:
        print("No se encontraron imágenes.")
        return

    # backend
    if args.backend.lower() == "monodepth2":
        if not args.mono_model_dir:
            raise SystemExit("--mono-model-dir requerido para monodepth2")
        from networks.resnet_encoder import ResnetEncoder
        from networks.depth_decoder import DepthDecoder
        monodepth2_cfg = dict(model_dir=args.mono_model_dir, device=args.device,
                              encoder_class=ResnetEncoder, depth_decoder_class=DepthDecoder)
        backend = DB.make_backend("monodepth2", device=args.device, monodepth2_cfg=monodepth2_cfg)
        monodepth2_classes = (ResnetEncoder, DepthDecoder)
    else:
        backend = DB.make_backend(args.backend, device=args.device)
        monodepth2_classes = None

    # FP16 + autocast (donde aplica)
    use_autocast = (args.device.startswith("cuda"))
    if hasattr(backend.get("model", None), "half") and args.device.startswith("cuda"):
        backend["model"].half()

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.json"
    index = []

    # batching
    B = args.batch_size
    for i in tqdm(range(0, len(imgs), B), desc="Depth batches"):
        batch_paths = imgs[i:i+B]

        todo, pil_list = [], []
        for p in batch_paths:
            rel = Path(p).relative_to(args.images_dir)
            out_npz = cache_dir / rel.with_suffix(".npz")
            if args.overwrite or not out_npz.exists():
                # >>>>>>>>>> EXIF transpose para consistencia
                pil = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
                pil_list.append(pil)
                todo.append((p, out_npz))
        if not todo:
            continue

        # inferencia (tu predict_depth_batch ya parcheado con padding manual)
        with torch.autocast("cuda", enabled=use_autocast, dtype=torch.float16):
            depth_list = DB.predict_depth_batch(backend, pil_list, monodepth2_classes=monodepth2_classes)

        for (src_path, out_npz), depth01 in zip(todo, depth_list):
            # opcional gamma si quieres
            # if args.gamma_depth != 1.0: depth01 = np.clip(depth01,0,1)**args.gamma_depth

            # >>>>>>>>>> invertir si se pide
            if args.invert_depth:
                depth01 = 1.0 - np.clip(depth01, 0, 1)

            save_depth_npz(out_npz, depth01)
            index.append({"image": str(Path(src_path).as_posix()),
                          "depth": str(out_npz.relative_to(cache_dir).as_posix())})
    # guardar índice
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[OK] Cache listo: {len(index)} mapas de profundidad")

if __name__ == "__main__":
    main()
