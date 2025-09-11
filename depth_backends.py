# depth_backends.py
import os
import torch
import numpy as np
import cv2

# --- Utilidad común: asegurar salida [0,1] y del tamaño original ---
def _to_01_and_resize(depth_like, H, W):
    depth = depth_like.astype(np.float32)
    if np.any(np.isinf(depth)) or np.any(np.isnan(depth)):
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    # normalizado robusto por percentiles (evita outliers)
    p2, p98 = np.percentile(depth, [2, 98])
    if p98 > p2:
        depth = np.clip((depth - p2) / (p98 - p2), 0, 1)
    else:
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    # resize si hace falta
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return depth

# -------- Backend: Monodepth2 (tu actual) --------
def load_monodepth2(model_dir, device, encoder_class, depth_decoder_class):
    encoder_path = os.path.join(model_dir, "encoder.pth")
    depth_decoder_path = os.path.join(model_dir, "depth.pth")

    encoder = encoder_class(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_h, feed_w = loaded_dict_enc['height'], loaded_dict_enc['width']
    filtered = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered); encoder.to(device); encoder.eval()

    decoder = depth_decoder_class(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    decoder.load_state_dict(torch.load(depth_decoder_path, map_location=device))
    decoder.to(device); decoder.eval()
    return {"type": "monodepth2", "encoder": encoder, "decoder": decoder, "feed_hw": (feed_h, feed_w), "device": device}

@torch.no_grad()
def predict_monodepth2(backend, pil_rgb):
    encoder, decoder = backend["encoder"], backend["decoder"]
    feed_h, feed_w = backend["feed_hw"]
    device = backend["device"]

    im0_w, im0_h = pil_rgb.size
    inp = pil_rgb.resize((feed_w, feed_h))
    inp = torch.from_numpy(np.array(inp)).float() / 255.0
    inp = inp.permute(2,0,1).unsqueeze(0).to(device)
    feats = encoder(inp)
    out = decoder(feats)[("disp", 0)]  # disparidad relativa
    disp = torch.nn.functional.interpolate(out, (im0_h, im0_w), mode="bilinear", align_corners=False)
    depth_like = disp.squeeze().detach().cpu().numpy()  # relativo/inverso
    # Nota: tu código original invertía con colormap; aquí normalizamos directo
    depth01 = _to_01_and_resize(depth_like, im0_h, im0_w)
    # Dependiendo de tu fórmula de niebla, a veces conviene invertir:
    # near (grande) -> más niebla. Si quieres far (grande) -> invierte:
    # depth01 = 1.0 - depth01
    return depth01

# -------- Backend: Depth Anything V2 (Transformers) --------
# pip install "transformers>=4.43" "torch" "accelerate" "numpy" "opencv-python" "Pillow"
def load_depth_anything_v2(model_id="depth-anything/Depth-Anything-V2-Small-hf", device="cpu"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return {"type":"danythingv2", "processor":processor, "model":model, "device":device}

@torch.no_grad()
def predict_depth_anything_v2(backend, pil_rgb):
    processor, model, device = backend["processor"], backend["model"], backend["device"]
    im0_w, im0_h = pil_rgb.size

    inputs = processor(images=pil_rgb, return_tensors="pt").to(device)
    outputs = model(**inputs)

    pred = outputs.predicted_depth  # puede ser [N,H,W] o [N,1,H,W]

    # 🔧 Asegurar 4D para interpolate
    if pred.dim() == 3:          # [N, H, W]
        pred = pred.unsqueeze(1)  # -> [N, 1, H, W]
    elif pred.dim() != 4:
        raise ValueError(f"predicted_depth con dim inesperada: {pred.shape}")

    pred = torch.nn.functional.interpolate(
        pred, size=(im0_h, im0_w), mode="bicubic", align_corners=False
    )
    depth = pred.squeeze().detach().cpu().numpy()  # -> [H, W]
    depth01 = _to_01_and_resize(depth, im0_h, im0_w)
    return depth01


# -------- Backend: ZoeDepth (Hugging Face Transformers) --------
# pip install "transformers>=4.43"
def load_zoedepth(model_id="Intel/zoedepth-nyu-kitti", device="cpu"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return {"type":"zoedepth", "processor":processor, "model":model, "device":device}

@torch.no_grad()
def predict_zoedepth(backend, pil_rgb):
    processor, model, device = backend["processor"], backend["model"], backend["device"]
    im0_w, im0_h = pil_rgb.size

    inputs = processor(images=pil_rgb, return_tensors="pt").to(device)
    outputs = model(**inputs)

    pred = outputs.predicted_depth

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    elif pred.dim() != 4:
        raise ValueError(f"predicted_depth con dim inesperada: {pred.shape}")

    pred = torch.nn.functional.interpolate(
        pred, size=(im0_h, im0_w), mode="bicubic", align_corners=False
    )
    depth = pred.squeeze().detach().cpu().numpy()
    depth01 = _to_01_and_resize(depth, im0_h, im0_w)
    return depth01


# -------- Backend: MiDaS/DPT (Torch Hub) --------
# pip install timm opencv-python
def load_midas(device="cpu", repo="intel-isl/MiDaS", model_name="DPT_Large"):
    midas = torch.hub.load(repo, model_name, pretrained=True, verbose=False).to(device).eval()
    transforms = torch.hub.load(repo, "transforms").dpt_transform
    return {"type":"midas", "model":midas, "tfm":transforms, "device":device}

@torch.no_grad()
def predict_midas(backend, pil_rgb):
    model, tfm, device = backend["model"], backend["tfm"], backend["device"]
    im0_w, im0_h = pil_rgb.size
    inp = tfm(pil_rgb).to(device)
    pred = model.forward(inp)
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(im0_h, im0_w), mode="bicubic", align_corners=False)
    depth = pred.squeeze().detach().cpu().numpy()
    depth01 = _to_01_and_resize(depth, im0_h, im0_w)
    return depth01

import torch.nn.functional as F

# ===================== PREDICCIÓN POR LOTES =====================

@torch.no_grad()
def predict_monodepth2_batch(backend, pil_list):
    """
    Devuelve lista de depth en [0,1], una por PIL del batch.
    Hace encoder/decoder una sola vez por batch (feed size fijo), y luego
    reescala cada salida al tamaño original de su imagen.
    """
    encoder, decoder = backend["encoder"], backend["decoder"]
    feed_h, feed_w = backend["feed_hw"]
    device = backend["device"]

    # Prepara batch [B,3,Hf,Wf]
    orig_sizes = []
    batch = []
    for im in pil_list:
        w0, h0 = im.size
        orig_sizes.append((h0, w0))
        inp = im.resize((feed_w, feed_h))
        tens = torch.from_numpy(np.array(inp)).float() / 255.0  # [Hf,Wf,3]
        tens = tens.permute(2, 0, 1)                            # [3,Hf,Wf]
        batch.append(tens)
    batch = torch.stack(batch, dim=0).to(device)                # [B,3,Hf,Wf]

    feats = encoder(batch)
    out = decoder(feats)[("disp", 0)]                           # [B,1,Hf,Wf] (relativo)

    outs = []
    for i, (Hi, Wi) in enumerate(orig_sizes):
        disp_i = out[i:i+1]                                     # [1,1,Hf,Wf]
        disp_i = F.interpolate(disp_i, (Hi, Wi), mode="bilinear", align_corners=False)
        depth_like = disp_i.squeeze().detach().cpu().numpy()    # [Hi,Wi]
        depth01 = _to_01_and_resize(depth_like, Hi, Wi)
        outs.append(depth01)
    return outs


@torch.no_grad()
def predict_depth_anything_v2_batch(backend, pil_list):
    processor, model, device = backend["processor"], backend["model"], backend["device"]
    orig_sizes = [(im.size[1], im.size[0]) for im in pil_list]  # (H,W)
    inputs = processor(images=list(pil_list), return_tensors="pt").to(device)
    outputs = model(**inputs)
    pred = outputs.predicted_depth                               # [B,H',W'] o [B,1,H',W']

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)                                 # [B,1,H',W']

    outs = []
    for i, (Hi, Wi) in enumerate(orig_sizes):
        di = F.interpolate(pred[i:i+1], size=(Hi, Wi), mode="bicubic", align_corners=False)
        depth_like = di.squeeze().detach().cpu().numpy()
        depth01 = _to_01_and_resize(depth_like, Hi, Wi)
        outs.append(depth01)
    return outs


@torch.no_grad()
def predict_zoedepth_batch(backend, pil_list):
    # Misma lógica que Depth-Anything v2 (ambos HF)
    return predict_depth_anything_v2_batch(backend, pil_list)


@torch.no_grad()
def predict_midas_batch(backend, pil_list):
    """
    MiDaS/DPT: el transform retorna tensor por imagen; apilamos manualmente.
    """
    model, tfm, device = backend["model"], backend["tfm"], backend["device"]
    tens_list, orig_sizes = [], []
    for im in pil_list:
        H, W = im.size[1], im.size[0]
        orig_sizes.append((H, W))
        t = tfm(im)               # [3,h,w]
        tens_list.append(t)
    batch = torch.stack(tens_list, dim=0).to(device)  # [B,3,h,w]
    pred = model.forward(batch)                       # [B,h',w'] o [B,1,h',w']

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)

    outs = []
    for i, (Hi, Wi) in enumerate(orig_sizes):
        di = F.interpolate(pred[i:i+1], size=(Hi, Wi), mode="bicubic", align_corners=False)
        depth_like = di.squeeze().detach().cpu().numpy()
        depth01 = _to_01_and_resize(depth_like, Hi, Wi)
        outs.append(depth01)
    return outs


# --------- Router batch público (similar a predict_depth unitario) ---------
def predict_depth_batch(backend, pil_list, monodepth2_classes=None):
    t = backend["type"]
    if t == "monodepth2":
        if monodepth2_classes is None:
            raise ValueError("monodepth2_classes requerido para monodepth2")
        return predict_monodepth2_batch(backend, pil_list)
    if t == "danythingv2":
        return predict_depth_anything_v2_batch(backend, pil_list)
    if t == "zoedepth":
        return predict_zoedepth_batch(backend, pil_list)
    if t == "midas":
        return predict_midas_batch(backend, pil_list)
    raise ValueError(f"Tipo backend no soportado (batch): {t}")


# ---------- Fábrica ----------
def make_backend(name, device, monodepth2_cfg=None):
    name = name.lower()
    if name == "monodepth2":
        if monodepth2_cfg is None:
            raise ValueError("monodepth2_cfg requerido")
        return load_monodepth2(**monodepth2_cfg)
    if name in ["depth-anything-v2","danythingv2","dav2","depthanythingv2"]:
        return load_depth_anything_v2(device=device)
    if name in ["zoedepth","zoe"]:
        return load_zoedepth(device=device)
    if name in ["midas","dpt"]:
        return load_midas(device=device)
    raise ValueError(f"Backend de profundidad desconocido: {name}")

def predict_depth(backend, pil_rgb, monodepth2_classes=None):
    t = backend["type"]
    if t == "monodepth2":
        if monodepth2_classes is None:
            raise ValueError("monodepth2_classes requerido")
        return predict_monodepth2(backend, pil_rgb)
    if t == "danythingv2":
        return predict_depth_anything_v2(backend, pil_rgb)
    if t == "zoedepth":
        return predict_zoedepth(backend, pil_rgb)
    if t == "midas":
        return predict_midas(backend, pil_rgb)
    raise ValueError(f"Tipo backend no soportado: {t}")
