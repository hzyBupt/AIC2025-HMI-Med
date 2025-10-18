# ============================================================
# 文件：app2.py
# 功能：Flask Web 推理 + DICOM 动态预览 + 自动生成 GIF
# ============================================================

import os
import glob
import base64
import tempfile
import subprocess
from io import BytesIO
from typing import Optional

import torch
import numpy as np
import torch.nn.functional as F
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pydicom

from builder import build_model
from Data import transforms as TRANS

# ========================= 基础配置 ========================= #
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"dcm"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: "T0/1", 1: "T2A", 2: "T2B", 3: "T3"}

DEFAULT_GIF_FPS = 4.0
SUPPORTED_LANGS = {"zh", "en"}


def normalize_lang(value: Optional[str]) -> str:
    """Normalize language code, defaulting to zh."""
    if not value:
        return "zh"
    value = value.lower()
    return value if value in SUPPORTED_LANGS else "zh"

# ========================= 模型初始化 ========================= #
def init_model(infer_path, epoch):
    global model

    ckpt_path = glob.glob(os.path.join(infer_path, f"Model/Best_*_Epoch_{epoch:04d}.pth"))[0]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]

    cfg.defrost()
    cfg.MODEL.Para.sag_pretrained = None
    cfg.MODEL.Para.cor_pretrained = None
    cfg.MODEL.Para.fuse_pretrained = None
    cfg.MODEL.Para.all_pretrained = None
    cfg.freeze()

    net = build_model(cfg.MODEL)
    net.load_state_dict({
        k.replace("module.", ""): v
        for k, v in ckpt["net_dict"].items()
        if k.replace("module.", "") in net.state_dict()
    })
    net.to(device).eval()
    model = net
    print("✅ 模型初始化完成")


# ========================= GIF 生成函数 ========================= #
def tensor_to_gif(volume: torch.Tensor, fps: Optional[float] = None) -> str:
    """将 (C, T, H, W) 的 4D Tensor 转为 base64 GIF"""
    if volume.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {volume.shape}")

    # 转灰度（取平均通道）
    frames = volume.mean(dim=0) if volume.size(0) > 1 else volume.squeeze(0)
    frames = frames.detach().cpu().clamp(0, 255).byte().numpy()
    images = [Image.fromarray(frame) for frame in frames]
    if not images:
        raise ValueError("No frames to generate GIF")

    target_fps = DEFAULT_GIF_FPS
    if fps is not None:
        try:
            val = float(fps)
            if val > 0:
                target_fps = val
        except Exception:
            pass

    duration = max(1, int(round(1000.0 / target_fps)))

    buffer = BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ========================= DICOM 加载函数 ========================= #
def load_dicom_tensor(path: str, target_frames=8, target_size=256):
    """加载 DICOM 文件并输出 tensor、GIF base64 与帧率"""
    ds = pydicom.dcmread(path)

    # 尝试提取帧率
    def _extract_fps(dataset) -> Optional[float]:
        candidates = []

        # 优先 CineRate / FrameRate
        for attr in ["CineRate", "FrameRate"]:
            val = getattr(dataset, attr, None)
            if val is not None:
                try:
                    fps_val = float(val)
                    if fps_val > 0:
                        candidates.append(fps_val)
                except Exception:
                    continue

        # 若有 FrameTime 信息（ms/帧）
        for attr in ["FrameTime", "FrameTimeVector", "TemporalResolution"]:
            val = getattr(dataset, attr, None)
            if val is None:
                continue
            values = val if isinstance(val, (list, tuple)) else [val]
            for item in values:
                try:
                    frame_time = float(item)
                    if frame_time > 0:
                        candidates.append(1000.0 / frame_time)
                except Exception:
                    continue

        for candidate in candidates:
            if candidate and candidate > 0:
                return candidate
        return None

    frame_fps = _extract_fps(ds) or DEFAULT_GIF_FPS

    # 读取像素数据（带异常处理）
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except RuntimeError:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(["gdcmconv", "--raw", path, tmp_path],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ds = pydicom.dcmread(tmp_path)
            pixel_array = ds.pixel_array.astype(np.float32)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    pixel_array -= pixel_array.min()
    if pixel_array.max() > 0:
        pixel_array = pixel_array / pixel_array.max() * 255.0

    tensor = torch.from_numpy(pixel_array).float()

    # 调整维度
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(1)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(3, 0, 1, 2)
        else:
            raise ValueError(f"Unexpected DICOM shape: {tensor.shape}")
    else:
        raise ValueError(f"Unexpected DICOM shape: {tensor.shape}")

    # 时间采样
    T = tensor.shape[1]
    if T > target_frames:
        idx = torch.linspace(0, T - 1, target_frames).long()
        tensor = tensor[:, idx, :, :]
    elif T < target_frames:
        pad = target_frames - T
        last = tensor[:, -1:, :, :].repeat(1, pad, 1, 1)
        tensor = torch.cat([tensor, last], dim=1)

    # 空间插值
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(target_frames, target_size, target_size),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    preview_tensor = tensor.clone()

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1, 1)

    val_norm = TRANS.TioZNormalization(p=1, div255=True)
    tensor = val_norm(tensor)

    gif_data = tensor_to_gif(preview_tensor, fps=frame_fps)
    return tensor, gif_data, frame_fps


# ========================= Flask 路由 ========================= #
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        cor = request.files.get("cor_dicom")
        lang = normalize_lang(request.form.get("lang", "zh"))
        sag = request.files.get("sag_dicom")
        if not cor or not sag:
            message = {
                "zh": "请上传两个 DICOM 文件",
                "en": "Please upload both DICOM files",
            }
            return message[lang]

        fname1 = secure_filename(cor.filename)
        fname2 = secure_filename(sag.filename)
        path1 = os.path.join(app.config["UPLOAD_FOLDER"], fname1)
        path2 = os.path.join(app.config["UPLOAD_FOLDER"], fname2)
        cor.save(path1)
        sag.save(path2)

        # 读取数据并生成 GIF
        cor_tensor, cor_gif, cor_fps = load_dicom_tensor(path1)
        sag_tensor, sag_gif, sag_fps = load_dicom_tensor(path2)

        cor_t = cor_tensor.unsqueeze(0).to(device)
        sag_t = sag_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model([cor_t, sag_t])

        probs_fuse = F.softmax(logits[0], dim=-1).squeeze().cpu().numpy()
        probs_cor = F.softmax(logits[1], dim=-1).squeeze().cpu().numpy()
        probs_sag = F.softmax(logits[2], dim=-1).squeeze().cpu().numpy()

        pred_fuse = int(np.argmax(probs_fuse))
        pred_cor = int(np.argmax(probs_cor))
        pred_sag = int(np.argmax(probs_sag))

        fuse_info = [(label_map[i], float(probs_fuse[i])) for i in range(len(probs_fuse))]
        cor_info = [(label_map[i], float(probs_cor[i])) for i in range(len(probs_cor))]
        sag_info = [(label_map[i], float(probs_sag[i])) for i in range(len(probs_sag))]

        return render_template(
            "result2.html",
            fuse_info=fuse_info,
            cor_info=cor_info,
            sag_info=sag_info,
            pred_fuse=label_map[pred_fuse],
            pred_cor=label_map[pred_cor],
            pred_sag=label_map[pred_sag],
            gif_cor=cor_gif,
            gif_sag=sag_gif,
            fps_cor=cor_fps,
            fps_sag=sag_fps,
            lang=lang,           
        )
    else:
        lang = normalize_lang(request.args.get("lang", "zh"))
        return render_template("index2.html", lang=lang)


@app.route("/preview_gif", methods=["POST"])
def preview_gif():
    """前端 AJAX 调用预览 GIF"""
    file = request.files.get("dicom")
    if not file or not file.filename.lower().endswith(".dcm"):
        return jsonify({"error": "请上传有效的 DICOM 文件"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        _, gif_data, fps_val = load_dicom_tensor(tmp_path)
        return jsonify({"gif": gif_data, "fps": fps_val})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ========================= 启动 ========================= #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    init_model(args.infer_path, args.epochs)
    app.run(host="0.0.0.0", port=args.port, debug=False)
