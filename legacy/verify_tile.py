"""在「17 张图选中的那张局部图」上跑校验循环。

流程：
  ① 读 ds_17.py 的结果：模型选中的图号 img_no + 该图内的 bbox
  ② 把那【张局部图】作为底图（不是原图）
  ③ 在它上面跑原有的校验循环：裁 200px → 放大到 1000px → 画方块 → 校验 → 微调
  ④ 循环结束后，把局部图坐标映射回原图坐标

与 verify_loop.py 的差别：
  · 底图 = 选中的局部图（比原图清晰 4 倍），不是原图
  · 起点 = 模型在该局部图里给的 bbox，不是整图 L1
  · 最终坐标需要「局部图 → 原图」一次映射

用法：
  python3 verify_tile.py --ids a,b,c --l1-file results/l1/g17_r1.jsonl \
      --desc-file ... --tag t1 [--grid 4] [--max-iter 3] [--crop-px 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image  # noqa: E402

import harness as H  # noqa: E402
from verify_loop import (draw_cursor, pad_edge, VERIFY_PROMPT, ARROW_DESC,  # noqa: E402
                         parse_verdict, make_ask, make_fit, OUTDIR)

from PIL import ImageDraw  # noqa: E402,F401  (verify_loop 里用到了)


def build_tiles(img, grid):
    """与 ds_17.py 完全一致的切图方式（不缩放，保持原分辨率）。"""
    W, H = img.size
    out = []
    for r in range(grid):
        for c in range(grid):
            x0, y0 = c * W // grid, r * H // grid
            x1 = W if c == grid - 1 else (c + 1) * W // grid
            y1 = H if r == grid - 1 else (r + 1) * H // grid
            out.append((img.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--l1-file", required=True, help="ds_17.py 输出的 jsonl")
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", default="vtile")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--crop-px", type=int, default=200, help="裁片边长（局部图像素）")
    ap.add_argument("--sent-px", type=int, default=1000)
    ap.add_argument("--cursor-px", type=float, default=48.0)
    ap.add_argument("--backend", choices=["qwen", "deepseek"], default="deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--l1-px", type=float, default=None)
    args = ap.parse_args()

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    allrows = {x["id"]: x for x in H.load_all()}

    # ---- 读 17 图结果 ----
    START = {}
    for line in open(args.l1_file, encoding="utf-8"):
        r = json.loads(line)
        if r.get("img_no") and r.get("local_norm"):
            START[r["id"]] = (int(r["img_no"]), [float(v) for v in r["local_norm"]])
    print(f"从 {args.l1_file} 载入 {len(START)} 条起点（图号 + 图内 bbox）")

    global ASK
    ASK = make_ask(args)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{args.tag}.jsonl"
    fout = open(out_path, "w", encoding="utf-8")
    print(f"底图=选中的局部图  裁片={args.crop_px}px(局部图像素)  最大循环={args.max_iter}")
    print(f"输出 → {out_path}\n")

    for n, rid in enumerate(args.ids.split(","), 1):
        src = allrows.get(rid)
        if not src or rid not in START:
            print(f"  [{n:>3}] ⚠️ 跳过 {rid}")
            continue
        t0 = time.time()
        img = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
        W0, H0 = src["img_size"]
        gt = src["bbox"]
        gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        d = DESC[rid]
        _env = d.get("env") or ""
        full_desc = (f"外观：{d['feat']}；控件类型：{d['ctrl']}"
                     + (f"；周边锚点：{_env}" if _env else ""))

        img_no, local = START[rid]
        tiles = build_tiles(img, args.grid)
        if img_no == 1:
            region = (0, 0, W0, H0)
        elif 2 <= img_no <= 1 + len(tiles):
            region = tiles[img_no - 2][1]
        else:
            print(f"  [{n:>3}] ⚠️ 图号越界 {rid}")
            continue
        # 底图 = 局部图【向外扩一圈真实像素】，避免裁片贴边时出现复制出来的假像素
        _mg = args.crop_px // 2 + 8
        rx0 = max(0, region[0] - _mg); ry0 = max(0, region[1] - _mg)
        rx1 = min(W0, region[2] + _mg); ry1 = min(H0, region[3] + _mg)
        base = img.crop((rx0, ry0, rx1, ry1))
        bw, bh = base.size

        # 起点：模型在【原局部图】内给的 bbox → 带 margin 的底图坐标
        tw0, th0 = region[2] - region[0], region[3] - region[1]
        ox, oy = region[0] - rx0, region[1] - ry0
        est = [ox + local[0] / 1000 * tw0, oy + local[1] / 1000 * th0,
               ox + local[2] / 1000 * tw0, oy + local[3] / 1000 * th0]

        rec = {"id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
               "base_img_no": img_no, "base_region": [rx0, ry0, rx1, ry1],
               "base_size": [bw, bh], "start_local_norm": local,
               "start_local_px": [round(v, 1) for v in est],
               "iterations": []}

        status, reason_txt = "MAX_ITER", "循环用尽仍未定位成功"
        for it in range(1, args.max_iter + 1):
            cx, cy = (est[0] + est[2]) / 2, (est[1] + est[3]) / 2
            tw, th = est[2] - est[0], est[3] - est[1]
            cpx = int(max(args.crop_px, round(max(tw, th) * 1.5)))
            cpx = min(cpx, 1600)
            pad = cpx // 2 + 8
            big = pad_edge(base, pad)
            X, Y = cx + pad, cy + pad
            x0, y0 = int(X - cpx / 2), int(Y - cpx / 2)
            region_c = big.crop((x0, y0, x0 + cpx, y0 + cpx))
            gx0, gy0 = X - x0, Y - y0

            sc = args.sent_px / max(region_c.size)
            sw, sh = max(1, round(region_c.width * sc)), max(1, round(region_c.height * sc))
            sent = region_c.resize((sw, sh), Image.LANCZOS)
            cur_w = max(sw, sh) * args.cursor_px / 1000.0
            sent = draw_cursor(sent, gx0 * sc, gy0 * sc, cur_w)
            cpath = OUTDIR / f"{args.tag}_{n:02d}_{rid}_it{it}.png"
            sent.save(cpath)

            vtxt, _, vreason = ASK(sent, VERIFY_PROMPT.format(arrow=ARROW_DESC,
                                                              desc=full_desc), 512)
            v = parse_verdict(vtxt)
            step = {"iter": it, "sent_size": [sw, sh], "crop_px": cpx,
                    "cursor_local": [round(cx, 1), round(cy, 1)],
                    "verdict_raw": vtxt.strip()[:300],
                    "in_image": v["in_image"], "correct": v["correct"],
                    "bbox_in_crop_norm": v["bbox_norm"]}

            if v["in_image"] is False:
                status, reason_txt = "TARGET_NOT_IN_CROP", "目标不在裁片内"
                rec["iterations"].append(step)
                break
            if v["correct"] is True:
                status, reason_txt = "SUCCESS", "方块在目标上，校验通过"
                rec["iterations"].append(step)
                break
            if v["bbox_norm"]:      # 裁片内归一化 → 局部图像素
                est = [x0 - pad + v["bbox_norm"][0] / 1000 * cpx,
                       y0 - pad + v["bbox_norm"][1] / 1000 * cpx,
                       x0 - pad + v["bbox_norm"][2] / 1000 * cpx,
                       y0 - pad + v["bbox_norm"][3] / 1000 * cpx]
                step["est_local_px"] = [round(val, 1) for val in est]
            rec["iterations"].append(step)

        # 局部图像素 → 原图像素
        sx = (rx1 - rx0) / bw
        sy = (ry1 - ry0) / bh
        final = [rx0 + est[0] * sx, ry0 + est[1] * sy,
                 rx0 + est[2] * sx, ry0 + est[3] * sy]
        err = (((final[0] + final[2]) / 2 - gcx) ** 2
               + ((final[1] + final[3]) / 2 - gcy) ** 2) ** 0.5
        rec.update({"status": status, "reason": reason_txt,
                    "final_bbox": [round(v, 1) for v in final],
                    "final_err": round(err, 1),
                    "n_iter": len(rec["iterations"]),
                    "latency_s": round(time.time() - t0, 1)})
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"  [{n:>3}] {rid:<26} 图{img_no:>2}  误差={err:>6.0f}px  "
              f"{status:<20} {rec['latency_s']:>5.1f}s  "
              f"{'✅' if err <= 50 else '❌'}", flush=True)

    fout.close()
    print(f"\n结果 → {out_path}")


if __name__ == "__main__":
    main()
