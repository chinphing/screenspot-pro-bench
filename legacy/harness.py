"""ScreenSpot-Pro 两级网格坐标方案 —— 实验 harness。

用法:
    python3 harness.py --n 10 --mode baseline --tag smoke
    python3 harness.py --n 50 --mode baseline
    python3 harness.py --n 50 --mode grid --grid-px 200

输出: results/<tag>.jsonl  (每题一行)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import qwen_client as qc  # noqa: E402

ROOT = Path(__file__).parent
ANNO = ROOT / "data" / "annotations"
IMGDIR = ROOT / "data" / "images"
RESULTS = ROOT / "results"
MIRROR = "https://hf-mirror.com/datasets/likaixin/ScreenSpot-Pro/resolve/main/images/"

TARGET_PIXELS = 2.5e6          # 自己控制缩放上限（API 对更大图会自行降采样）
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


# ---------------------------------------------------------------- 数据
def load_all() -> list[dict]:
    out = []
    for f in sorted(ANNO.glob("*.json")):
        try:
            data = json.load(open(f))
        except Exception:
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


def sample(rows: list[dict], n: int, seed: int = 20261014) -> list[dict]:
    """分层抽样：icon/text 各半，组内尽量均匀。"""
    rng = random.Random(seed)
    buckets: dict[tuple, list] = {}
    for r in rows:
        buckets.setdefault((r.get("ui_type"), r.get("group")), []).append(r)
    for v in buckets.values():
        rng.shuffle(v)
    keys = sorted(buckets, key=lambda k: str(k))
    picked, i = [], 0
    while len(picked) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        i += 1
    return picked[:n]


def fetch_image(name: str) -> Path:
    """用 curl 下载（urllib 不跟随 308，hf-mirror 会 308 跳到被墙的 huggingface.co）。"""
    p = IMGDIR / name
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    url = MIRROR + urllib.parse.quote(name)
    tmp = str(p) + ".part"
    last = ""
    for attempt in range(3):
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "180", "-o", tmp, url],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1024:
            os.replace(tmp, p)
            return p
        last = (r.stderr or "")[-120:]
        time.sleep(2 ** attempt)
    raise RuntimeError(f"下载失败: {name} {last}")


# ---------------------------------------------------------------- 图像处理
def downscale(img: Image.Image, target_px: float = TARGET_PIXELS):
    w, h = img.size
    s = min((target_px / (w * h)) ** 0.5, 1.0)
    if s >= 1.0:
        return img, 1.0
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS), s


def draw_grid(img: Image.Image, scale: float, grid_px: int, label_every: int = 5) -> Image.Image:
    """在已缩放的图上按【原图】grid_px 间距画网格（尺子）。

    线标签用 0-1000 归一化值 —— 与模型输出的坐标格式保持一致。
    """
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    step = grid_px * scale
    if step < 6:
        return img
    try:
        f = ImageFont.truetype(FONT_PATH, 14)
    except Exception:
        f = ImageFont.load_default()
    col = (110, 155, 215)
    col5 = (30, 80, 165)
    i = 0
    x = 0.0
    while x <= W:
        strong = (i % label_every == 0)
        d.line([(x, 0), (x, H)], fill=col5 if strong else col, width=2 if strong else 1)
        if strong and i > 0:
            d.text((x + 4, 4), str(int(round(x / W * 1000))), font=f, fill=col5)
        i += 1
        x += step
    j = 0
    y = 0.0
    while y <= H:
        strong = (j % label_every == 0)
        d.line([(0, y), (W, y)], fill=col5 if strong else col, width=2 if strong else 1)
        if strong and j > 0:
            d.text((4, y + 4), str(int(round(y / H * 1000))), font=f, fill=col5)
        j += 1
        y += step
    return img


# ---------------------------------------------------------------- 解析 / 判分
NUM4 = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def parse_bbox(text: str):
    ms = NUM4.findall(text)
    if not ms:
        return None
    a, b, c, d = (float(v) for v in ms[-1])
    if c < a:
        a, c = c, a
    if d < b:
        b, d = d, b
    return [a, b, c, d]


def iou(box, gt) -> float:
    ix1, iy1 = max(box[0], gt[0]), max(box[1], gt[1])
    ix2, iy2 = min(box[2], gt[2]), min(box[3], gt[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (box[2] - box[0]) * (box[3] - box[1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_prompt(w: int, h: int, instruction: str, mode: str, grid_px: int) -> str:
    if mode == "coarse":
        # 专为第 1 级设计：不要求贴合，只要求框住
        return (f"这张图是 {w} x {h} 像素。\n"
                f"请在界面中找到【{instruction}】这个元素，并给出一个【大致包含】它的矩形区域。\n"
                f"要求：\n"
                f"1. 该区域必须【完整包含】目标元素；\n"
                f"2. 不需要精确贴合边缘，宁可框大一些，建议宽高各留 1~2 倍余量；\n"
                f"3. 坐标归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]，"
                f"0 表示左/上边缘，1000 表示右/下边缘。\n"
                f"只输出这个数组，不要解释。")
    head = f"这张图是 {w} x {h} 像素。"
    if mode == "grid":
        head += (f"图上画了网格线，每隔 {grid_px} 像素一条（粗线每 {grid_px*5} 像素一条，"
                 f"并在边缘标注了它对应的原图坐标）。网格只是参考尺。")
    return (head + f"\n请找出界面中【{instruction}】这个元素，给出它的 bbox。"
            f"\n坐标请归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]，"
            f"其中 0 表示左/上边缘，1000 表示右/下边缘。"
            f"\n只输出这个数组，不要解释。")


def norm_to_orig(bb, W0: int, H0: int):
    """0-1000 归一化 → 原图像素。"""
    return [bb[0] / 1000 * W0, bb[1] / 1000 * H0, bb[2] / 1000 * W0, bb[3] / 1000 * H0]


# ---------------------------------------------------------------- 主循环
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", choices=["baseline", "grid", "coarse"], default="baseline")
    ap.add_argument("--grid-px", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    if args.model:
        qc.MODEL = args.model
    tag = args.tag or f"{args.mode}_{args.n}"
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{tag}.jsonl"

    rows = load_all()
    picked = sample(rows, args.n, args.seed)
    print(f"总标注 {len(rows)} 条 → 抽样 {len(picked)} 条 | mode={args.mode} | model={qc.MODEL}")
    print(f"输出 → {out_path}\n")

    fout = open(out_path, "w", encoding="utf-8")
    ok = err = 0
    for i, r in enumerate(picked, 1):
        rec = {
            "id": r.get("id"), "app": r.get("application"), "platform": r.get("platform"),
            "group": r.get("group"), "ui_type": r.get("ui_type"),
            "instruction": r.get("instruction"), "gt_bbox": r.get("bbox"),
            "gt_img_size": r.get("img_size"), "mode": args.mode,
        }
        try:
            p = fetch_image(r["img_filename"])
            img = Image.open(p).convert("RGB")
            gt = r["bbox"]
            W0, H0 = r["img_size"]
            small, s = downscale(img)
            sw, sh = small.size
            if args.mode == "grid":
                small = draw_grid(small, s, args.grid_px)
            if args.save_images:
                (ROOT / "results" / "imgs").mkdir(parents=True, exist_ok=True)
                small.save(ROOT / "results" / "imgs" / f"{tag}_{i}.png")

            prompt = build_prompt(sw, sh, r["instruction"], args.mode, args.grid_px)
            t0 = time.time()
            txt, usage = qc.ask_image(small, prompt, max_tokens=600)
            dt = time.time() - t0

            bb_s = parse_bbox(txt)
            rec.update({
                "scaled_size": [sw, sh], "scale": s, "latency_s": round(dt, 1),
                "usage": usage, "raw": txt.strip()[:400],
            })
            if bb_s:
                bb = norm_to_orig(bb_s, W0, H0)                  # 0-1000 → 原图像素
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                gx, gy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
                rec.update({
                    "pred_bbox_orig": [round(v, 1) for v in bb],
                    "pred_center": [round(cx, 1), round(cy, 1)],
                    "gt_center": [round(gx, 1), round(gy, 1)],
                    "center_err_px": round(((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5, 1),
                    "hit": bool(gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]),
                    "iou": round(iou(bb, gt), 3),
                    "size_err_px": round(
                        ((bb[2] - bb[0]) - (gt[2] - gt[0])) ** 2
                        + ((bb[3] - bb[1]) - (gt[3] - gt[1])) ** 2, 1) ** 0.5,
                })
                ok += rec["hit"]
                print(f"  [{i:>3}/{len(picked)}] {'✅' if rec['hit'] else '❌'} "
                      f"{rec['ui_type']:<5} err={rec['center_err_px']:>7.1f}px "
                      f"iou={rec['iou']:.2f} {dt:>5.1f}s  {r['instruction'][:40]!r}")
            else:
                rec["error"] = "parse_failed"
                print(f"  [{i:>3}/{len(picked)}] ⚠️  解析失败: {txt[:80]!r}")
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            err += 1
            print(f"  [{i:>3}/{len(picked)}] 💥 {rec['error'][:90]}")
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()

    tot = len(picked)
    print(f"\n{'='*60}")
    print(f"完成 {tot} 题 | 命中 {ok} ({ok/tot*100:.1f}%) | 报错 {err}")
    print(f"结果 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
