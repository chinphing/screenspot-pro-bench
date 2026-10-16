"""17 张图方案：原图 + 4×4 局部放大图，让模型选图 + 给该图内坐标。

流程：
  输入：第 1 张 = 完整原图；第 2~17 张 = 按 4×4 切出的局部图（左→右、上→下）
  模型回答两件事：
    ① 目标在哪一张图里看得最清晰完整
    ② 目标在【该张图】内的归一化 bbox（0-1000）
  代码把该图内坐标换算回原图坐标（换算完全确定，不需要模型参与）

用法：
  python3 ds_17.py --ids a,b,c --desc-file xxx.json --tag t1 \
      [--tile-px 1.7e6] [--reasoning-effort none] [--grid 4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image  # noqa: E402

import harness as H  # noqa: E402
import ds_client as dc  # noqa: E402

NUM4 = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
NUM_IMG = re.compile(r"(?:图号|图片|图)\s*[:：]?\s*(?:第\s*)?(\d+)\s*(?:张)?")


def _fit_up(img, target_px):
    """缩放到目标像素数（可放大也可缩小）。"""
    w, h = img.size
    if w * h == 0:
        return img
    k = (target_px / (w * h)) ** 0.5
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def _desc_lines(d):
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}")


def build_grid(img, grid, tile_px):
    """按 grid×grid 切图，返回 [(图, (x0,y0,x1,y1)), ...]（左→右、上→下）。"""
    W, H = img.size
    out = []
    for r in range(grid):
        for c in range(grid):
            x0, y0 = c * W // grid, r * H // grid
            x1 = W if c == grid - 1 else (c + 1) * W // grid
            y1 = H if r == grid - 1 else (r + 1) * H // grid
            t = img.crop((x0, y0, x1, y1))
            if tile_px:
                t = _fit_up(t, tile_px)
            out.append((t, (x0, y0, x1, y1)))
    return out


def run_one(rid, src, d, *, grid=4, tile_px=1.7e6, full_px=1.7e6,
            reasoning_effort="none", max_tokens=1200, tries=10):
    img = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
    W0, H0 = img.size
    gt = src["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2

    full = _fit_up(img, full_px)
    tiles = build_grid(img, grid, tile_px)

    # ---- 构造 17 张图的内容 ----
    content = [{"type": "text", "text":
        f"下面给你 {1 + len(tiles)} 张图。\n"
        f"· 第 1 张是【完整界面】（{W0} x {H0} 像素，归一化范围 [0,0,1000,1000]）。\n"
        f"· 第 2 ~ {1 + len(tiles)} 张是把完整界面按 {grid}x{grid} 切成的局部图，"
        f"顺序为从左到右、从上到下；每张都比完整界面清晰 {grid} 倍。\n"
        f"每张图前面会标注它的编号和它在完整界面里占的归一化范围。\n"}]
    content.append({"type": "text",
                    "text": f"【第 1 张】完整界面（尺寸 {full.width} x {full.height} 像素，"
                            f"归一化范围 [0,0,1000,1000]）："})
    content.append({"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}})

    for k, (t, box) in enumerate(tiles, start=2):
        r, c = (k - 2) // grid + 1, (k - 2) % grid + 1
        x0 = round(box[0] / W0 * 1000); y0 = round(box[1] / H0 * 1000)
        x1 = round(box[2] / W0 * 1000); y1 = round(box[3] / H0 * 1000)
        content.append({"type": "text",
                        "text": f"【第 {k} 张】第 {r} 行、第 {c} 列的局部图"
                                f"（尺寸 {t.width} x {t.height} 像素），"
                                f"对应完整界面范围 [{x0},{y0},{x1},{y1}]："})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(t)}"}})

    content.append({"type": "text", "text":
        f"\n【任务】\n找出下面描述的元素。\n"
        f"\n【回答要求】\n"
        f"1. 先判断：哪一张图里能【最清晰、最完整】地看到这个目标？\n"
        f"   · 优先选能完整看到目标的局部图（第 2 ~ {1 + len(tiles)} 张）；\n"
        f"   · 如果目标跨在两张局部图的接缝上、任何一张都看不完整，就选第 1 张完整界面。\n"
        f"2. 再给出目标在【你选的那张图】里的 bbox，用该图的归一化坐标（0-1000）。\n"
        f"   注意：坐标是相对你选的那张图，不是相对完整界面。\n"
        f"\n【输出格式】严格按下面两行输出，不要多余文字：\n"
        f"图号: <数字>\n"
        f"坐标: [x1,y1,x2,y2]\n"
        f"\n【目标元素描述】\n{_desc_lines(d)}"})

    t0 = time.time()
    resp, att = dc.chat([{"role": "user", "content": content}],
                        max_tokens=max_tokens, tries=tries,
                        extra={"reasoning_effort": reasoning_effort})
    dt = time.time() - t0
    txt = dc.text_of(resp).strip()
    usage = resp.get("usage", {})

    # ---- 解析 ----
    mi = NUM_IMG.search(txt)
    img_no = int(mi.group(1)) if mi else None
    mb = NUM4.search(txt)
    local = [int(v) for v in mb.groups()] if mb else None

    # 按模型所选图号换算
    final_bbox = None
    region = None
    if img_no is not None and local is not None and 1 <= img_no <= 1 + len(tiles):
        if img_no == 1:
            region = (0, 0, W0, H0)
        else:
            region = tiles[img_no - 2][1]
        rx0, ry0, rx1, ry1 = region
        rw, rh = rx1 - rx0, ry1 - ry0
        final_bbox = [rx0 + local[0] / 1000 * rw, ry0 + local[1] / 1000 * rh,
                      rx0 + local[2] / 1000 * rw, ry0 + local[3] / 1000 * rh]

    # 对照：若模型其实给的是完整界面坐标
    alt_bbox = None
    if local is not None:
        alt_bbox = [local[0] / 1000 * W0, local[1] / 1000 * H0,
                    local[2] / 1000 * W0, local[3] / 1000 * H0]

    def _err(bb):
        if not bb:
            return None
        return (((bb[0] + bb[2]) / 2 - gcx) ** 2 + ((bb[1] + bb[3]) / 2 - gcy) ** 2) ** 0.5

    # 正确切图应该是哪一张（用于诊断）
    tcx, tcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    truth_tile = None
    for k, (_, box) in enumerate(tiles, start=2):
        if box[0] <= tcx <= box[2] and box[1] <= tcy <= box[3]:
            truth_tile = k
            break

    return {"id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
            "grid": grid, "n_images": 1 + len(tiles),
            "tile_size": list(tiles[0][0].size), "full_size": list(full.size),
            "status": "OK" if final_bbox is not None else "PARSE_FAIL",
            "answer": txt[:300], "img_no": img_no, "local_norm": local,
            "truth_tile": truth_tile, "picked_right_tile": (img_no == truth_tile),
            "final_bbox": final_bbox, "final_err": _err(final_bbox),
            "alt_bbox_if_global": alt_bbox, "alt_err_if_global": _err(alt_bbox),
            "usage": usage, "attempts": att, "latency_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--tile-px", type=float, default=1.7e6,
                    help="每张局部图的送图像素数（0=按原分辨率不缩放）")
    ap.add_argument("--full-px", type=float, default=1.7e6)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--tries", type=int, default=10)
    args = ap.parse_args()

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    out = os.path.join("results", "l1", f"g17_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"17 张图方案  {args.grid}x{args.grid} 网格 → {1+args.grid*args.grid} 张图  "
          f"局部图 {args.tile_px/1e6:.2f}Mpx  思考={args.reasoning_effort}")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(args.ids.split(","), 1):
        try:
            rec = run_one(rid, rows[rid], DESC[rid], grid=args.grid,
                          tile_px=args.tile_px, full_px=args.full_px,
                          reasoning_effort=args.reasoning_effort,
                          max_tokens=args.max_tokens, tries=args.tries)
        except Exception as e:  # noqa: BLE001
            rec = {"id": rid, "status": "ERROR", "error": str(e)[:200]}
        recs.append(rec)
        e = rec.get("final_err")
        es = "  无" if e is None else f"{round(e):>4}px"
        pick = ("✅选对图" if rec.get("picked_right_tile")
                else f"❌选图{rec.get('img_no')}(应{rec.get('truth_tile')})")
        print(f"  [{k:>2}] {rid:<26} 误差={es}  {pick:<16} "
              f"{rec.get('latency_s', 0):>5.1f}s  "
              f"{'✅' if (e is not None and e <= 50) else '❌'}", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")


if __name__ == "__main__":
    main()
