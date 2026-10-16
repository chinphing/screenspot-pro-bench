"""按用户原设计端到端实测两级流程。

设计:
  第 1 级: 原图划 200px 网格 → 模型输出一个【最少 200x200 px 的正方形】，要求框住目标
  第 2 级: 把该正方形从【原图】裁出、放大 → 划 10px 网格 → 模型输出目标在窗内的位置
  最终   : 正方形原点 + 窗内位置 → 与原图 GT 比对

用法:
    python3 two_level.py --n 50 --tag tl50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import harness as H
import qwen_client as qc

ROOT = Path(__file__).parent
MIN_SQUARE = 200          # 第一级正方形最小边长（原图像素）
L2_OUT = 800              # 第二级窗口放大到多少像素
L2_GRID_ORIG = 10         # 第二级网格间距（原图像素）
L2_MAX_LINES = 40         # 网格线数量上限（太密模型读不了）


def draw_square_grid(img, square_side, grid_orig):
    """在放大后的窗口上按 grid_orig 原图像素间距画网格，返回(图, 实际间距)。"""
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    step = grid_orig * (W / square_side)
    n = int(W / step) if step > 0 else 0
    if n > L2_MAX_LINES:                      # 太密就降采样
        step = W / L2_MAX_LINES
        n = L2_MAX_LINES
    if step < 8:
        return img, step
    try:
        f = ImageFont.truetype(H.FONT_PATH, 14)
    except Exception:
        f = ImageFont.load_default()
    col, col5 = (120, 165, 220), (35, 85, 170)
    for axis in (0, 1):
        lim = W if axis == 0 else H
        p, i = 0.0, 0
        while p <= lim:
            strong = (i % 5 == 0)
            c = col5 if strong else col
            w = 2 if strong else 1
            if axis == 0:
                d.line([(p, 0), (p, H)], fill=c, width=w)
            else:
                d.line([(0, p), (W, p)], fill=c, width=w)
            i += 1
            p += step
    return img, step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--tag", default="tl50")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-square", type=int, default=MIN_SQUARE)
    ap.add_argument("--l1-grid", type=int, default=200, help="第一级网格间距；0=不画")
    args = ap.parse_args()
    if args.model:
        qc.MODEL = args.model

    out_path = ROOT / "results" / f"{args.tag}.jsonl"
    rows = H.load_all()
    picked = H.sample(rows, args.n, args.seed)
    print(f"两级流程 | n={len(picked)} | model={qc.MODEL} | L1网格={args.l1_grid}px | 最小正方形={args.min_square}px")
    print(f"输出 → {out_path}\n")

    fout = open(out_path, "w", encoding="utf-8")
    l1_ok = l2_ok = 0
    for i, r in enumerate(picked, 1):
        rec = {"id": r.get("id"), "ui_type": r.get("ui_type"), "app": r.get("application"),
               "instruction": r.get("instruction"), "gt_bbox": r.get("bbox"),
               "gt_img_size": r.get("img_size")}
        try:
            img = Image.open(H.fetch_image(r["img_filename"])).convert("RGB")
            W0, H0 = r["img_size"]
            gt = r["bbox"]

            # ---------------- 第 1 级 ----------------
            small, s = H.downscale(img)
            sw, sh = small.size
            if args.l1_grid:
                small = H.draw_grid(small, s, args.l1_grid)
            p1 = (f"这张图是 {sw} x {sh} 像素。"
                  + (f"图上画了网格线，每隔 {args.l1_grid} 像素一条，仅作参考尺。" if args.l1_grid else "")
                  + f"\n请在界面中找到【{r['instruction']}】这个元素，"
                  f"给出一个【完整包含】它的正方形区域。"
                  f"\n要求：边长至少 {args.min_square} 像素；如果元素比这更大，就以能完整装下为准。"
                  f"\n坐标归一化到 0-1000，格式 [x1,y1,x2,y2]（左上、右下）。只输出数组。")
            t1, u1 = qc.ask_image(small, p1, max_tokens=500)
            bb1 = H.parse_bbox(t1)
            if not bb1:
                rec["error"] = "L1解析失败"; rec["raw_l1"] = t1[:200]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                print(f"  [{i:>3}/{len(picked)}] ⚠️ L1解析失败 {t1[:60]!r}")
                continue
            sq = H.norm_to_orig(bb1, W0, H0)
            # 保证最小边长，围绕中心扩
            cx, cy = (sq[0] + sq[2]) / 2, (sq[1] + sq[3]) / 2
            side = max(args.min_square, sq[2] - sq[0], sq[3] - sq[1])
            sq = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
            # 夹到图像范围内
            x1 = max(0, min(sq[0], W0 - side)); y1 = max(0, min(sq[1], H0 - side))
            side = min(side, W0, H0)
            sq = [x1, y1, x1 + side, y1 + side]
            l1_contain = (sq[0] <= gt[0] and gt[2] <= sq[2] and sq[1] <= gt[1] and gt[3] <= sq[3])
            l1_ok += l1_contain
            rec.update({"l1_raw": bb1, "l1_square": [round(v, 1) for v in sq],
                        "l1_side": round(side, 1), "l1_contain": bool(l1_contain)})

            # ---------------- 第 2 级 ----------------
            box = (int(max(0, sq[0])), int(max(0, sq[1])),
                   int(min(W0, sq[2])), int(min(H0, sq[3])))
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                rec["error"] = "裁剪窗过小"
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                print(f"  [{i:>3}/{len(picked)}] ⚠️ 裁剪窗过小")
                continue
            crop = img.crop(box)
            cw = box[2] - box[0]
            up = crop.resize((L2_OUT, L2_OUT), Image.LANCZOS)
            upg, step = draw_square_grid(up, cw, L2_GRID_ORIG)
            p2 = (f"这是一张放大的局部截图（原始区域边长 {cw} 像素，已放大到 {L2_OUT} 像素）。"
                  f"图上画了网格，每隔约 {step:.0f} 像素一条（对应原图 {L2_GRID_ORIG} 像素）。"
                  f"\n请找到其中的【{r['instruction']}】元素，给出它的中心位置。"
                  f"\n坐标用【这张放大图】的归一化 0-1000 表示，格式 [x,y]。只输出数组。")
            t2, u2 = qc.ask_image(upg, p2, max_tokens=400)
            m = re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", t2)
            if not m:
                rec["error"] = "L2解析失败"; rec["raw_l2"] = t2[:200]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                print(f"  [{i:>3}/{len(picked)}] ⚠️ L2解析失败 {t2[:60]!r}")
                continue
            nx, ny = float(m[-1][0]), float(m[-1][1])
            fx = box[0] + nx / 1000 * cw
            fy = box[1] + ny / 1000 * (box[3] - box[1])
            hit = gt[0] <= fx <= gt[2] and gt[1] <= fy <= gt[3]
            l2_ok += hit
            gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
            err = ((fx - gcx) ** 2 + (fy - gcy) ** 2) ** 0.5
            rec.update({"l2_raw": [nx, ny], "final_xy": [round(fx, 1), round(fy, 1)],
                        "center_err_px": round(err, 1), "hit": bool(hit),
                        "l2_step_px": round(step, 1), "usage_l1": u1, "usage_l2": u2})
            print(f"  [{i:>3}/{len(picked)}] {'✅' if hit else '❌'} {r['ui_type']:<5} "
                  f"L1框住={'Y' if l1_contain else 'N'} 边长={side:.0f} err={err:>7.1f}px  {r['instruction'][:34]!r}")
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  [{i:>3}/{len(picked)}] 💥 {rec['error'][:80]}")
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()

    tot = len(picked)
    print(f"\n{'='*64}")
    print(f"第1级 框住目标: {l1_ok}/{tot} = {l1_ok/tot*100:.1f}%")
    print(f"端到端 命中:    {l2_ok}/{tot} = {l2_ok/tot*100:.1f}%")
    print(f"结果 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
