"""Run the recovered integer model on the synthetic test image.

Same pipeline as tools/cyclegan_host.py main(), but without the per-layer
.npy reference comparison (those reference vectors were only generated on
the original machine). The recovered weights are already cross-validated
against four independent RTL ROM copies.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cyclegan_host import ArrayCore, GeneratorScheduler  # noqa: E402

model = ROOT / "build/cyclegan_model"
manifest = json.loads((model / "params_manifest.json").read_text())
words = [int(v, 16) for v in (model / "generator_params_q88.mem").read_text().split()]
words = [v - 65536 if v >= 32768 else v for v in words]
lut = [int(v, 16) for v in (ROOT / "rtl/mn_tanh_u8.mem").read_text().split()]

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", type=Path, help="任意尺寸的输入照片；"
                    "保持原图比例：竖幅旋转进入网络后转回，"
                    "非 4:3 横幅等比缩放并用边缘延展补边")
parser.add_argument("--out", type=Path, help="输出 PNG 路径")
parser.add_argument("--final-scale", type=float, default=1.0,
                    help="放大输出层(20427..21602)的 Q8.8 权重，"
                         "拉伸 Tanh 输入 -> 对比度/饱和度增强")
parser.add_argument("--bn-scale", type=float, default=1.0,
                    help="放大全网络 BN 的 scale(a) 参数 -> "
                         "逐层特征风格化强度增强（过大可能过曝）")
args = parser.parse_args()


def scale_word(i, factor):
    words[i] = max(-32768, min(32767, round(words[i] * factor)))


if args.final_scale != 1.0:
    for i in range(20427, 21603):
        scale_word(i, args.final_scale)
if args.bn_scale != 1.0:
    # BN 段为 (a,b) 交错存储，偶数偏移是 scale(a)；收集全部 BN 段。
    segments = []
    for info in manifest["encoder"] + manifest["blocks"]:
        for key in ("dw_bn", "pw_bn"):
            if key in info:
                segments.append((info[key], 2 * info["cout"]))
    for info in manifest["decoder"]:
        segments.append((info["bn_base"], 2 * info["cout"]))
    for base, count in segments:
        for j in range(0, count, 2):
            scale_word(base + j, args.bn_scale)
    print(f"BN scaled x{args.bn_scale} over {len(segments)} segments")

if args.image:
    photo = Image.open(args.image).convert("RGB")
    # 竖幅照片旋转 90 度进入网络（网络固定 4:3 横幅），输出再转回，
    # 原始比例和完整画面都保留；非 4:3 横幅用边缘像素延展补边。
    portrait = photo.height > photo.width
    if portrait:
        photo = photo.transpose(Image.ROTATE_90)
    scale = min(320 / photo.width, 240 / photo.height)
    new_size = (max(1, round(photo.width * scale)),
                max(1, round(photo.height * scale)))
    photo = photo.resize(new_size, Image.LANCZOS)
    canvas = np.full((240, 320, 3), 0, dtype=np.uint8)
    arr = np.asarray(photo, dtype=np.uint8)
    top, left = (240 - arr.shape[0]) // 2, (320 - arr.shape[1]) // 2
    canvas[top:top + arr.shape[0], left:left + arr.shape[1]] = arr
    if top > 0:  # 上下补边用照片边缘行延展，避免生硬黑边
        canvas[:top] = canvas[top]
        canvas[top + arr.shape[0]:] = canvas[top + arr.shape[0] - 1]
    if left > 0:  # 左右补边同理
        canvas[:, :left] = canvas[:, left:left + 1]
        canvas[:, left + arr.shape[1]:] = canvas[:, left + arr.shape[1] - 1:left + arr.shape[1]]
    rgb = canvas
else:
    portrait = False
    rgb = np.frombuffer((model / "vectors/frame_rgbx.bin").read_bytes(),
                        dtype=np.uint8).reshape(240, 320, 4)[:, :, :3]

start = time.time()
core = ArrayCore(words, lut)
scheduler = GeneratorScheduler(core, manifest)
result = scheduler.run(rgb, verify=None)
elapsed = time.time() - start

output = result.transpose(1, 2, 0)
if portrait:  # 转回竖幅，输出与原图同比例
    output = np.rot90(output, k=3)

destination = ROOT / "build/cyclegan_schedule"
destination.mkdir(exist_ok=True)
out_path = args.out or (destination / "output.png")
Image.fromarray(output).save(out_path)
print(f"OUTPUT_SAVED {out_path}")
record = {
    "validation": "recovered weights; instruction-level integer model",
    "weights_source": "rtl ROMs via tools/recover_model_assets.py",
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "counts": dict(core.counts),
    "layers": len(scheduler.layer_results),
    "elapsed_seconds": round(elapsed, 1),
}
(destination / "validation_recovered.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf8")
print("RECOVERED_MODEL_RUN_PASS", json.dumps(record), flush=True)
