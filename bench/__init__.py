"""ScreenSpot-Pro 定位 benchmark —— 核心库。

模块：
    dataset   数据集加载、图片下载与缓存、坐标换算
    clients   模型客户端（DeepSeek / Qwen），统一接口
    vision    图像工具（缩放、裁片、画标记）
    metrics   评估指标（命中率 + 落点质量 + 随机基线）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ANNO = DATA / "annotations"
IMGDIR = DATA / "images"
RESULTS = ROOT / "baselines" / "results"        # 原始结果 → baselines/results/raw/
SPLITS = ROOT / "splits"
DESCRIPTIONS = ROOT / "descriptions"

__all__ = ["ROOT", "DATA", "ANNO", "IMGDIR", "RESULTS", "SPLITS", "DESCRIPTIONS"]
