"""图像工具：缩放、裁片、画标记。

所有标记都画在【送出去的图】上，坐标随缩放一起换算——这是 agent 循环的基础。
"""
from __future__ import annotations

from PIL import Image, ImageDraw

# ---------------------------------------------------------------- 缩放
TARGET_PIXELS = 2.5e6


def fit(img: Image.Image, target_px: float = TARGET_PIXELS) -> Image.Image:
    """等比缩放到不超过 target_px 像素（超过才缩）。"""
    w, h = img.size
    if w * h <= target_px:
        return img
    k = (target_px / (w * h)) ** 0.5
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def pad_edge(img: Image.Image, pad: int) -> Image.Image:
    """四周用边缘像素复制扩展 pad 像素，保证任意位置都能取到完整裁片。"""
    w, h = img.size
    out = Image.new("RGB", (w + 2 * pad, h + 2 * pad))
    out.paste(img, (pad, pad))
    out.paste(img.crop((0, 0, w, 1)).resize((w, pad)), (pad, 0))
    out.paste(img.crop((0, h - 1, w, h)).resize((w, pad)), (pad, pad + h))
    out.paste(img.crop((0, 0, 1, h)).resize((pad, h)), (0, pad))
    out.paste(img.crop((w - 1, 0, w, h)).resize((pad, h)), (pad + w, pad))
    for dx, dy, box in [(0, 0, (0, 0, 1, 1)), (pad + w, 0, (w - 1, 0, w, 1)),
                        (0, pad + h, (0, h - 1, 1, h)), (pad + w, pad + h, (w - 1, h - 1, w, h))]:
        out.paste(img.crop(box).resize((pad, pad)), (dx, dy))
    return out


# ---------------------------------------------------------------- 画标记
def draw_box(img: Image.Image, x0, y0, x1, y1, color=(0, 90, 200), width=3) -> Image.Image:
    """画矩形框（蓝）。反向坐标自动归一化——Pillow 10.1 对反向坐标会抛 ValueError。"""
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], outline=color,
                                  width=max(2, int(round(width))))
    return img


def draw_square(img: Image.Image, x: float, y: float, side: float) -> Image.Image:
    """在 (x,y) 画边长 side 的正方形标记（中心落在 x,y）。

    描边式（不填充），保证方块内的目标仍可见：红色内框 + 白色外描边。
    """
    img = img.copy()
    h = float(side) / 2.0
    box = [x - h, y - h, x + h, y + h]
    inner = max(2, int(round(side * 0.10)))
    outer = inner + max(2, int(round(side * 0.08)))
    d = ImageDraw.Draw(img)
    d.rectangle(box, outline=(255, 255, 255), width=outer)
    d.rectangle(box, outline=(220, 30, 30), width=inner)
    return img


# 画标记的别名（历史脚本里叫 draw_cursor）
draw_cursor = draw_square


# ---------------------------------------------------------------- 裁片
def crop_around(img: Image.Image, cx: float, cy: float, side: int, zoom: int = 1):
    """以原图 (cx,cy) 为中心裁 side×side，再放大 zoom 倍（贴边时用边缘像素补齐）。

    返回 (裁片图, 裁片覆盖的原图归一化范围 [x0,y0,x1,y1])。
    """
    half = side // 2
    pad = half + 8
    big = pad_edge(img, pad)
    x0, y0 = int(cx + pad - half), int(cy + pad - half)
    region = big.crop((x0, y0, x0 + side, y0 + side))
    if zoom != 1:
        region = region.resize((side * zoom, side * zoom), Image.LANCZOS)
    W, H = img.size
    bx0, by0 = x0 - pad, y0 - pad
    return region, [bx0 / W * 1000, by0 / H * 1000,
                    (bx0 + side) / W * 1000, (by0 + side) / H * 1000]


def crop_rect(img: Image.Image, box, *, ratio: float = 2.5, min_side: int = 120):
    """按目标 bbox 裁一个正方形区域（边长 = 目标长边 × ratio）。box 是原图像素坐标。"""
    W, H = img.size
    L = max(box[2] - box[0], box[3] - box[1])
    side = max(min_side, int(L * ratio))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return crop_around(img, cx, cy, side)


def map_norm_to_orig(bb, W: int, H: int):
    return [bb[0] / 1000 * W, bb[1] / 1000 * H, bb[2] / 1000 * W, bb[3] / 1000 * H]
