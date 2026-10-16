"""数据集加载、图片下载与坐标换算（原 harness.py 的数据集部分）。

数据集：ScreenSpot-Pro（HuggingFace: likaixin/ScreenSpot-Pro）
本地标注：data/annotations/*.json（26 个应用各一份）
图片缓存：data/images/<应用目录>/<文件名>.png（由 scripts/download_data.py 填充）
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import time
import urllib.parse
from pathlib import Path

from . import ANNO, IMGDIR

MIRROR = "https://hf-mirror.com/datasets/likaixin/ScreenSpot-Pro/resolve/main/images/"
TARGET_PIXELS = 2.5e6          # 送图缩放上限（API 对更大图会自行降采样）
NUM4 = re.compile(r"(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)")


def load_all() -> list[dict]:
    """加载全部样例。字段：id / img_filename / bbox / img_size / instruction / application / platform / ui_type / group"""
    out: list[dict] = []
    for f in sorted(ANNO.glob("*.json")):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for x in data:
            b = x.get("bbox")
            sz = x.get("img_size")
            if not b or not sz or len(b) != 4:
                continue
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            x["_file"] = f.stem
            out.append(x)
    return out


def by_id() -> dict[str, dict]:
    return {r["id"]: r for r in load_all()}


def sample(rows: list[dict], n: int, seed: int = 20261014) -> list[dict]:
    rnd = random.Random(seed)
    return rnd.sample(rows, min(n, len(rows)))


ANNO_MIRROR = "https://hf-mirror.com/datasets/likaixin/ScreenSpot-Pro/resolve/main/annotations/"

# 上游 annotations/ 目录下的固定文件名（26 个应用 + 1 个上游就没有数据的占位）。
# ⚠️ 必须写死：图片路径来自标注里的 img_filename，反过来推不出标注该叫什么名字。
ANNO_FILES = [
    "android_studio_macos", "autocad_windows", "blender_windows", "davinci_macos",
    "eviews_windows", "excel_macos", "fruitloops_windows", "illustrator_windows",
    "inventor_windows", "linux_common_linux", "macos_common_macos", "matlab_macos",
    "matlab_windows", "origin_windows", "photoshop_windows", "powerpoint_windows",
    "premiere_windows", "pycharm_macos", "quartus_windows", "solidworks_windows",
    "stata_windows", "unreal_engine_windows", "vivado_windows", "vmware_macos",
    "vscode_macos", "windows_common_windows", "word_macos",
]


def fetch_annotation(stem: str) -> Path | None:
    """按需下载一份标注到 data/annotations/<stem>.json（已存在且可解析则直接返回）。

    ⚠️ 落盘前必须校验能解析成 JSON：上游对不存在的文件返回的是 404 正文
       （`Entry not found`，15 字节），早期就是把它当标注文件存下来了。
    上游确实没有该文件时返回 None。
    """
    p = ANNO / f"{stem}.json"
    if p.exists():
        try:
            if isinstance(json.load(open(p, encoding="utf-8")), list):
                return p
        except Exception:  # noqa: BLE001
            pass
        p.unlink(missing_ok=True)   # 坏文件（如 404 正文 "Entry not found"）删掉重取
    p.parent.mkdir(parents=True, exist_ok=True)
    url = ANNO_MIRROR + urllib.parse.quote(f"{stem}.json")
    tmp = str(p) + ".part"
    for _ in range(3):
        subprocess.run(["curl", "-sL", "--max-time", "180", "-o", tmp, url],
                       capture_output=True, text=True)
        try:
            json.load(open(tmp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            Path(tmp).unlink(missing_ok=True)
            continue
        Path(tmp).replace(p)
        return p
    return None


def fetch_image(name: str) -> Path:
    """按需下载图片到本地缓存（已存在则直接返回）。name 形如 'matlab_mac/xxx.png'。"""
    p = IMGDIR / name
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    url = MIRROR + urllib.parse.quote(name)
    tmp = str(p) + ".part"
    last = ""
    for attempt in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "180", "-o", tmp, url],
                           capture_output=True, text=True)
        if r.returncode == 0 and Path(tmp).exists() and Path(tmp).stat().st_size > 1024:
            Path(tmp).replace(p)
            return p
        last = (r.stderr or "")[-120:]
        time.sleep(2 ** attempt)
    raise RuntimeError(f"下载失败: {name} {last}")


def norm_to_orig(bb, W0: int, H0: int):
    """归一化坐标（0-1000）→ 原图像素坐标。"""
    return [bb[0] / 1000 * W0, bb[1] / 1000 * H0, bb[2] / 1000 * W0, bb[3] / 1000 * H0]


def parse_bbox(text: str):
    m = NUM4.findall(text or "")
    return [float(v) for v in m[-1]] if m else None


def iou(box, gt) -> float:
    x0, y0 = max(box[0], gt[0]), max(box[1], gt[1])
    x1, y1 = min(box[2], gt[2]), min(box[3], gt[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return inter / (a + b - inter)


def target_len(bbox) -> float:
    """目标长边（像素）。"""
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1])
