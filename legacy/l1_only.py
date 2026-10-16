"""第 1 级专项测试 v2 —— 原图像素坐标系。

与 v1 的区别（按用户要求）：
  1. 网格标注 = 【原图真实像素值】，不再是 0-1000 归一化
  2. 模型必须【直接输出原图像素坐标】，跳过归一化
  3. 明确"正方形"= 像素宽高相等
  4. 自动检测模型是否仍在输出归一化值

用法:
    python3 l1_only.py --ids "a,b,c" --grid-px 200 --line-w 3 --tag l1_px_w3
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
OUTDIR = ROOT / "results" / "l1"


def draw_grid_px(img, scale, grid_px, line_w=3, font_size=20):
    """按【原图】grid_px 间距画网格；每条线标注【原图像素坐标】。

    返回 (标注后的图, 是否成功绘制)
    """
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    step = grid_px * scale
    if step < 10:
        return img, False
    try:
        f = ImageFont.truetype(H.FONT_PATH, font_size)
    except Exception:
        f = ImageFont.load_default()
    col, col5 = (255, 60, 60), (220, 0, 0)

    def label(x, y, text):
        """白底红字标签——避免被复杂背景吞掉。"""
        tb = d.textbbox((x, y), text, font=f)
        d.rectangle([tb[0] - 3, tb[1] - 3, tb[2] + 3, tb[3] + 3], fill=(255, 255, 255))
        d.text((x, y), text, font=f, fill=(190, 0, 0))

    lw = int(max(2, font_size * 0.55))
    for axis in (0, 1):
        lim = W if axis == 0 else H
        k = 0
        while k * grid_px * scale <= lim:
            p = k * grid_px * scale
            strong = (k % 5 == 0)
            c = col5 if strong else col
            w = line_w + 1 if strong else line_w
            if axis == 0:
                d.line([(p, 0), (p, H)], fill=c, width=w)
                if k > 0 and p + 90 < W:
                    label(p + 4, 2, f"x={k * grid_px}")
            else:
                d.line([(0, p), (W, p)], fill=c, width=w)
                if k > 0 and p + lw + 34 < H:
                    label(2, p + lw, f"y={k * grid_px}")
            k += 1
    return img, True


def looks_normalized(bb, W0, H0):
    """判断输出是否疑似归一化（值域<=1000 但原图远大于1000）。"""
    if not bb:
        return False
    mx = max(bb)
    return mx <= 1000 and (W0 > 1400 or H0 > 1400)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="逗号分隔的样本 id")
    ap.add_argument("--grid-px", type=int, default=200)
    ap.add_argument("--line-w", type=int, default=3)
    ap.add_argument("--min-square", type=int, default=200)
    ap.add_argument("--tag", default="l1px")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--font", type=int, default=20)
    ap.add_argument("--native", action="store_true", help="送原图，不做任何缩放")
    ap.add_argument("--simple-prompt", action="store_true", help="用旧版单步极简提示词（归一化输出）")
    ap.add_argument("--desc-file", default=None, help="元素描述 JSON，用于增强提示词（归一化输出）")
    ap.add_argument("--no-size", action="store_true", help="提示词里不告诉模型图像尺寸，只输出归一化 0-1000")
    ap.add_argument("--width", type=int, default=0,
                    help="把送图缩放到指定宽度（0=用默认 ~2.2M 像素）")
    args = ap.parse_args()
    if args.model:
        qc.MODEL = args.model

    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    DESC = {}
    if args.desc_file:
        DESC = json.load(open(args.desc_file, encoding="utf-8"))
    allrows = {x["id"]: x for x in H.load_all()}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{args.tag}.jsonl"
    fout = open(out_path, "w", encoding="utf-8")

    print(f"模型={qc.MODEL}  网格={args.grid_px}px  线宽={args.line_w}px  "
          f"坐标系=原图像素  样本={len(ids)}")
    print(f"输出 → {out_path}\n")

    n_contain = n_total = n_suspect = 0
    for n, rid in enumerate(ids, 1):
        src = allrows.get(rid)
        if not src:
            print(f"  [{n:>3}] ⚠️ 找不到 {rid}")
            continue
        rec = {"id": rid, "ui_type": src.get("ui_type"), "app": src.get("application"),
               "instruction": src.get("instruction"), "gt_bbox": src.get("bbox"),
               "gt_img_size": src.get("img_size"), "grid_px": args.grid_px,
               "line_w": args.line_w, "coord_sys": "original_pixel"}
        n_total += 1
        try:
            img = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
            W0, H0 = src["img_size"]
            gt = src["bbox"]
            if args.native:
                small, s = img, 1.0
            elif args.width:
                s = args.width / img.width
                small = img.resize((args.width, max(1, int(round(img.height * s)))),
                                   Image.LANCZOS)
            else:
                small, s = H.downscale(img)
            sw, sh = small.size
            if args.no_grid:
                sent, drew = small, False
            else:
                sent, drew = draw_grid_px(small, s, args.grid_px, args.line_w, args.font)
            imgpath = OUTDIR / f"{args.tag}_{n:02d}_{rid}.png"
            sent.save(imgpath)

            if args.desc_file and rid in DESC:
                d = DESC[rid]
                _hdr = "" if args.no_size else f"这张图是 {sw} x {sh} 像素。"
                _env = (f"· 周边锚点：{d['env']}\n") if d.get('env') else ""
                _txt = (f"· 文字内容：{d['text']}\n") if d.get('text') else ""
                prompt = (f"{_hdr}请找出界面中符合以下描述的元素，给出它的 bbox。\n"
                          f"\n【目标元素描述】\n"
                          f"{_txt}"
                          f"· 外观特征：{d['feat']}\n"
                          f"· 控件类型：{d['ctrl']}\n"
                          f"· 屏幕位置：{d['pos']}\n"
                          f"{_env}"
                          f"\n坐标请归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]，"
                          f"其中 0 表示左/上边缘，1000 表示右/下边缘。"
                          f"\n只输出这个数组，不要解释。")
            elif args.simple_prompt:
                prompt = (f"这张图是 {sw} x {sh} 像素。"
                          f"请找出界面中【{src['instruction']}】这个元素，给出它的 bbox。"
                          f"\n坐标请归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]，"
                          f"其中 0 表示左/上边缘，1000 表示右/下边缘。"
                          f"\n只输出这个数组，不要解释。")
            else:
                prompt = (
                f"这张图是从一张 {W0} x {H0} 像素的原始截图中等比缩小得到的，"
                f"当前显示尺寸 {sw} x {sh} 像素（缩小比例 {s:.4f}）。\n"
                + ("" if args.no_grid else
                   f"图上画了【红色网格线】，相邻两条线的间距是【原图 {args.grid_px} 像素】。"
                   f"每条线旁边标注的红色数字，就是这条线在【原图】中的像素坐标。"
                   f"请直接读取这些标注来确定位置。\n")
                + f"\n请分两步回答，目标是界面中的【{src['instruction']}】元素：\n"
                  f"第一步：给出该元素【本身】的精确 bbox。\n"
                  f"第二步：给出一个【完整包含】该元素的【正方形】，"
                  f"边长 = max({args.min_square}, 第一步 bbox 的最大边长)。\n"
                  f"\n【元素边界的判定标准】——这条最重要，请严格按它执行：\n"
                  f"· 菜单项 / 列表项 / 下拉选项：\n"
                  f"  它的【左右边界 = 所在容器���左右内边界】，不是文字宽度。\n"
                  f"  执行方法：先确定该下拉菜单 / 列表框 / 面板的【左内边界】和【右内边界】，"
                  f"把这���个值直接作为该项的 x1 和 x2；整行的上下边界作为 y1 和 y2。\n"
                  f"· 按钮 / 图标 → 包含图标本体 + 文字标签；\n"
                  f"· 纯文字元素 → 框住文字本身。\n"
                  f"\n坐标一律用【原图像素】表示（x 从 0 到 {W0}，y 从 0 到 {H0}），"
                  f"不要用 0-1000 归一化。\n"
                  f"输出格式（严格两行）：\n"
                  f"BBox: [ax1,ay1,ax2,ay2]\n"
                  f"Square: [x1,y1,x2,y2]")
            t0 = time.time()
            txt, usage, reasoning = qc.ask_image(sent, prompt, max_tokens=2000,
                                                 want_reasoning=True)
            dt = time.time() - t0

            rec.update({"sent_size": [sw, sh], "scale": round(s, 4), "grid_drawn": drew,
                        "prompt": prompt, "answer": txt.strip(), "reasoning": reasoning,
                        "usage": usage, "latency_s": round(dt, 1), "image": str(imgpath)})

            arrs = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                              r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", txt)
            if not arrs:
                rec["error"] = "解析失败"
                print(f"  [{n:>3}] ⚠️ 解析失败 {txt[:60]!r}")
            else:
                bb = [float(v) for v in arrs[0]]
                sq_raw = [float(v) for v in arrs[-1]] if len(arrs) > 1 else bb
                if args.simple_prompt or args.desc_file:    # 归一化 → 原图像素
                    bb = H.norm_to_orig(bb, W0, H0)
                    sq_raw = H.norm_to_orig(sq_raw, W0, H0)
                suspect = looks_normalized(sq_raw, W0, H0)
                if suspect:
                    n_suspect += 1
                rec["parsed"] = bb
                rec["parsed_square_raw"] = sq_raw
                rec["two_step"] = len(arrs) > 1
                rec["suspected_normalized"] = suspect
                sq = list(sq_raw)
                cx, cy = (sq[0] + sq[2]) / 2, (sq[1] + sq[3]) / 2
                side = max(args.min_square, sq[2] - sq[0], sq[3] - sq[1])
                sq = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
                x1 = max(0, min(sq[0], W0 - side)); y1 = max(0, min(sq[1], H0 - side))
                side = min(side, W0, H0)
                sq = [x1, y1, x1 + side, y1 + side]
                contain = (sq[0] <= gt[0] and gt[2] <= sq[2] and sq[1] <= gt[1] and gt[3] <= sq[3])
                n_contain += contain
                gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
                err = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5
                rec.update({"square": [round(v, 1) for v in sq], "side": round(side, 1),
                            "square_center": [round(cx, 1), round(cy, 1)],
                            "gt_center": [round(gcx, 1), round(gcy, 1)],
                            "center_err_px": round(err, 1), "contain": bool(contain)})
                flag = "⚠️疑似归一化" if suspect else ""
                print(f"  [{n:>3}] {'✅框住' if contain else '❌没框住'} "
                      f"err={err:>7.1f}px 边长={side:>5.0f} {dt:>4.0f}s "
                      f"{rid:<28}{flag}")
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  [{n:>3}] 💥 {rec['error'][:90]}")
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()

    print(f"\n{'='*70}")
    print(f"框住目标: {n_contain}/{n_total} = {n_contain/max(n_total,1)*100:.1f}%")
    if n_suspect:
        print(f"⚠️ 疑似仍输出归一化坐标: {n_suspect} 条（需人工核对）")
    print(f"结果 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
