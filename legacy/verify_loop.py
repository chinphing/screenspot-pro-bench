#!/usr/bin/env python3
"""L1 定位 + 光标校验循环（校准定位流程）。

流程：
  1. L1：2.5Mpx 送图 + 元素描述 + 不告知尺寸 → 输出目标【真实大小】bbox（归一化 0-1000）
  2. 校验循环（最多 N 次）：
     a. 在估计位置画鼠标光标（尖端指向该点）
     b. 以该点为中心裁 200px（原图尺度），放大 4× → 800px
     c. 送模型判定：目标在不在图里 / 在不在光标处 / 否则给出目标在这张图上的 bbox
     d. 目标不在图里 → 结束（告知原因）
        在且位置正确   → 结束（成功）
        否则           → 用新 bbox 更新估计，继续循环
     e. 达到上限仍未成功 → 结束（告知定位失败）

坐标系：模型一律输出归一化 0-1000，脚本负责还原到原图像素。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import harness as H  # noqa: E402
import qwen_client as qc  # noqa: E402
try:
    import ds_client as dc  # noqa: E402
except Exception:  # noqa: BLE001
    dc = None

OUTDIR = Path(__file__).parent / "results" / "loop"


# ---------------------------------------------------------------- 光标绘制
def draw_cursor(img: Image.Image, x: float, y: float, side: float) -> Image.Image:
    """在 (x,y) 画边长 side 的正方形标记，【中心】落在 (x,y)。

    描边式（不填充），保证方块内的目标仍可见：红色内框 + 白色外描边。
    """
    img = img.copy()
    d = ImageDraw.Draw(img)
    h = float(side) / 2.0
    box = [x - h, y - h, x + h, y + h]
    inner = max(2, int(round(side * 0.10)))
    outer = inner + max(2, int(round(side * 0.08)))
    d.rectangle(box, outline=(255, 255, 255), width=outer)   # 白色外描边
    d.rectangle(box, outline=(220, 30, 30), width=inner)     # 红色内框
    return img


def pad_edge(img: Image.Image, pad: int) -> Image.Image:
    """四周用边缘像素复制扩展 pad 像素，保证任意位置都能取到完整裁片（光标永远居中）。"""
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


# ---------------------------------------------------------------- 提示词
L1_PROMPT = """请找出界面中符合以下描述的元素，给出它的 bbox。

【目标元素描述】
· 文字内容：{text}
· 外观特征：{feat}
· 控件类型：{ctrl}
· 屏幕位置：{pos}
· 周边锚点：{env}

请给出【刚好框住该元素本身】的 bbox（不要外扩、不要缩小）。
坐标请归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]，其中 0 表示左/上边缘，1000 表示右/下边缘。
只输出这个数组，不要解释。"""

# 箭头（标记物）自身的外观描述 —— 与目标元素描述同一套写法
ARROW_DESC = ("一个正方形方框标记（红色边框、外侧带白色描边），"
              "它是空心的、不填充，方块的中心就是标记的位置")

VERIFY_PROMPT = """这张图是从界面截图中裁出的局部，已放大。

【标记物】{arrow}
【目标元素】{desc}

⚠️ 关于"周边锚点"：它【只是用来区分同名/相似的干扰元素】的辅助线索，
   【不需要逐项核对】。锚点与图中有出入【不算矛盾】，也【不影响】下面的判断。

请严格按以下顺序判断，每一步判断完就照格式输出，不要跳步：

第一步：目标元素是否出现在这张图里？
        · 否 → 立即输出结论并结束
        · 是 → 继续第二步
第二步：这个方块是否落在【你认定的那个目标元素】上面？
        · 只判断【位置关系】：方块与该元素是否重叠。
        · 【不要】因为描述（外观或锚点）有出入而改判——那属于第四行，与位置无关。
        · 是 → 立即输出结论并结束
        · 否 → 继续第三步
第三步：给出目标元素在这张图上的 bbox，归一化到 0-1000。

输出格式（四行，不要写别的）：
在图中：是 或 否 或 不确定
方块在目标上：是 或 否 或 不确定
目标bbox：[x1,y1,x2,y2] 或 none
矛盾点：none 或 一句话说明

说明：
· 第四行【只记录、不参与判断】。
· "在图中"为否时，后三行写"否"、"none"、"none"。
· "方块在目标上"为是时，第三行写 none。"""


# ---------------------------------------------------------------- 解析
def parse_verdict(text: str) -> dict:
    """解析三行裁决结果：在图中 / 箭头在目标上 / 目标bbox。"""
    import re
    t = text.replace("：", ":")
    in_img = None
    correct = None
    if re.search(r"在图中\s*:?\s*是", t):
        in_img = True
    elif re.search(r"在图中\s*:?\s*否", t):
        in_img = False
    elif re.search(r"在图中\s*:?\s*不确定", t):
        in_img = None
    if re.search(r"方块在目标上\s*:?\s*不确定", t):
        correct = None
    elif re.search(r"方块在目标上\s*:?\s*是", t):
        correct = True
    elif re.search(r"方块在目标上\s*:?\s*否", t):
        correct = False
    bb = None
    if not re.search(r"目标bbox\s*:?\s*none", t, re.I):
        ms = H.NUM4.findall(text)
        if ms:
            bb = [float(v) for v in ms[-1]]
    contra = None
    m = re.search(r"矛盾点\s*:?\s*(.+)", t)
    if m and m.group(1).strip().lower() not in ("none", "无", ""):
        contra = m.group(1).strip()[:120]
    return {"in_image": in_img, "correct": correct, "bbox_norm": bb,
            "contradiction": contra}


def make_ask(args):
    """返回 ask(img, prompt, max_tokens) -> (文本, usage, 思考)"""
    if args.backend == "deepseek":
        def ask(img, prompt, max_tokens=512):
            t, u, r, _ = dc.ask_image(img, prompt, max_tokens=max_tokens,
                                      reasoning_effort=args.reasoning_effort)
            return t, u, r
        return ask

    def ask(img, prompt, max_tokens=512):
        return qc.ask_image(img, prompt, model=args.model,
                            max_tokens=max_tokens, want_reasoning=True)
    return ask


def make_fit(args):
    """返回 fit(img) -> (缩好的图, 比例)。两个模型各用自己能吃满的上限。"""
    target = args.l1_px or (1.3e6 if args.backend == "deepseek" else H.TARGET_PIXELS)

    def fit(img):
        w, h = img.size
        if w * h <= target:
            return img, 1.0
        k = (target / (w * h)) ** 0.5
        return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS), k
    return fit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="逗号分隔的样本 id")
    ap.add_argument("--desc-file", required=True, help="元素描述 JSON")
    ap.add_argument("--tag", default="loop")
    ap.add_argument("--max-iter", type=int, default=3, help="校验循环最多执行几次")
    ap.add_argument("--crop-px", type=int, default=200, help="起始裁剪边长（原图像素）")
    ap.add_argument("--crop-ladder", default="200,400,800,1600",
                    help="裁片阶梯：判『不在图中』时逐级放大重试")
    ap.add_argument("--sent-px", type=int, default=1000,
                    help="送模型的图固定缩放到这个边长（箭头按此尺度定大小）")
    ap.add_argument("--zoom", type=int, default=4, help="裁剪后放大倍数")
    ap.add_argument("--cursor-px", type=float, default=24.0,
                    help="光标在『原图缩到 1000px 宽』时的像素高度；原图尺寸=该值*W/1000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", choices=["qwen", "deepseek"], default="qwen")
    ap.add_argument("--reasoning-effort", default="none",
                    help="DeepSeek 思考档位；none=关闭（实测又快又准）")
    ap.add_argument("--l1-file", default=None,
                    help="从 jsonl 读取 L1 结果（字段 final_bbox + gt_img_size），跳过模型 L1 调用")
    ap.add_argument("--l1-px", type=float, default=None,
                    help="L1 送图最大像素；默认 qwen=2.5e6, deepseek=1.3e6")
    args = ap.parse_args()

    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    allrows = {x["id"]: x for x in H.load_all()}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{args.tag}.jsonl"
    fout = open(out_path, "w", encoding="utf-8")

    L1FIX = None
    if args.l1_file:
        L1FIX = {}
        for line in open(args.l1_file, encoding="utf-8"):
            r = json.loads(line)
            bb, sz = r.get("final_bbox"), r.get("gt_img_size")
            if not bb or not sz:
                continue
            _iw, _ih = sz
            L1FIX[r["id"]] = [bb[0] / _iw * 1000, bb[1] / _ih * 1000,
                              bb[2] / _iw * 1000, bb[3] / _ih * 1000]
        print(f"从 {args.l1_file} 载入 {len(L1FIX)} 条 L1 结果（作为起点，跳过模型 L1）")

    global ASK, FIT
    ASK = make_ask(args)
    FIT = make_fit(args)
    _m = "deepseek-flash" if args.backend == "deepseek" else qc.MODEL
    print(f"后端={args.backend} 模型={_m}  L1送图上限={args.l1_px or (1.3e6 if args.backend=='deepseek' else H.TARGET_PIXELS):.2e}px"
          f"  裁剪={args.crop_px}px  最大循环={args.max_iter}")
    print(f"输出 → {out_path}\n")

    for n, rid in enumerate(ids, 1):
        src = allrows.get(rid)
        if not src:
            print(f"  [{n:>3}] ⚠️ 找不到 {rid}")
            continue
        t0 = time.time()
        img = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
        W0, H0 = src["img_size"]
        gt = src["bbox"]
        gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2

        rec = {"id": rid, "app": src.get("application"), "ui_type": src.get("ui_type"),
               "instruction": src.get("instruction"), "gt_bbox": gt,
               "gt_img_size": [W0, H0], "gt_center": [gcx, gcy],
               "max_iter": args.max_iter, "crop_px": args.crop_px, "zoom": args.zoom,
               "cursor_size_in_crop": round(args.cursor_px * args.crop_px / 1000, 1),
               "iterations": []}

        try:
            # ---------- 1) L1：真实大小 bbox ----------
            small, s = FIT(img)
            sw, sh = small.size
            d = DESC[rid]
            # 锚点【只给 L1】——验证模型看到锚点会当成硬约束，产生假矛盾
            # 外观+类型 + 锚点（锚点仅用于区分相似元素，提示词里已声明不用逐项核对）
            _env = d.get('env') or ""
            full_desc = (f"外观：{d['feat']}；控件类型：{d['ctrl']}"
                         + (f"；周边锚点：{_env}" if _env else ""))
            l1_prompt = L1_PROMPT.format(env=d.get('env', ''), text=d.get('text', ''),
                                         **{k: v for k, v in d.items() if k not in ('env', 'text')})
            if L1FIX is not None and rid in L1FIX:
                _b = L1FIX[rid]
                txt, usage, reason = f"[{_b[0]:.0f},{_b[1]:.0f},{_b[2]:.0f},{_b[3]:.0f}]", {}, ""
            else:
                txt, usage, reason = ASK(small, l1_prompt, 512)
            ms = H.NUM4.findall(txt)
            if not ms:
                rec.update({"status": "L1_NO_OUTPUT", "reason": "L1 未输出坐标",
                            "l1_answer": txt.strip()[:200]})
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"  [{n:>3}] ❌ L1 未输出坐标  {rid}")
                continue
            est = H.norm_to_orig([float(v) for v in ms[-1]], W0, H0)
            l1_err = ((est[0] + est[2]) / 2 - gcx) ** 2 + ((est[1] + est[3]) / 2 - gcy) ** 2
            rec["l1_answer"] = txt.strip()[:200]
            rec["l1_bbox"] = [round(v, 1) for v in est]
            rec["l1_err"] = round(l1_err ** 0.5, 1)
            rec["l1_reasoning"] = (reason or "")[-600:]
            print(f"  [{n:>3}] {rid}  L1误差={rec['l1_err']:.0f}px")

            # ---------- 2) 校验循环 ----------
            # 小目标(≤crop_px)：裁 200px → 缩到 1000px
            # 大目标(>crop_px)：整张原图缩到 1000px（不裁、不放大）
            # 两种情况的箭头都是 48px @1000px
            status, reason_txt = "MAX_ITER", "循环用尽仍未定位成功"
            for it in range(1, args.max_iter + 1):
                cx, cy = (est[0] + est[2]) / 2, (est[1] + est[3]) / 2
                tw, th = est[2] - est[0], est[3] - est[1]

                # 统一走裁片：边长 = max(crop_px, 目标长边 × 1.5)
                cpx = int(max(args.crop_px, round(max(tw, th) * 1.5)))
                cpx = min(cpx, 1600)
                mode = "crop"
                pad = cpx // 2 + 8
                big = pad_edge(img, pad)
                X, Y = cx + pad, cy + pad
                x0, y0 = int(X - cpx / 2), int(Y - cpx / 2)
                region = big.crop((x0, y0, x0 + cpx, y0 + cpx))
                box = [x0 - pad, y0 - pad, cpx, cpx]
                gx0, gy0 = X - x0, Y - y0          # 光标在 region 内的位置

                # ① 先缩放：最长边 → sent_px
                sc = args.sent_px / max(region.size)
                sw, sh = max(1, round(region.width * sc)), max(1, round(region.height * sc))
                sent = region.resize((sw, sh), Image.LANCZOS)
                # ② 缩放后再画光标：宽度 = 缩放后最长边 × cursor_px/1000
                cur_w = max(sw, sh) * args.cursor_px / 1000.0
                sent = draw_cursor(sent, gx0 * sc, gy0 * sc, cur_w)

                cpath = OUTDIR / f"{args.tag}_{n:02d}_{rid}_it{it}_{mode}.png"
                sent.save(cpath)

                vtxt, vusage, vreason = ASK(
                    sent, VERIFY_PROMPT.format(arrow=ARROW_DESC, desc=full_desc), 512)
                v = parse_verdict(vtxt)
                step = {"iter": it, "mode": mode, "sent_size": [sw, sh],
                        "cursor_width_in_sent": round(cur_w, 1),
                        "cursor_orig": [round(cx, 1), round(cy, 1)],
                        "crop_origin": [box[0], box[1]], "crop_px": box[2],
                        "crop_path": str(cpath),
                        "verdict_raw": vtxt.strip()[:300],
                        "verdict_reasoning": (vreason or "")[-1500:],
                        "in_image": v["in_image"], "correct": v["correct"],
                        "contradiction": v.get("contradiction"),
                        "bbox_in_crop_norm": v["bbox_norm"]}
                if v["in_image"] is False:
                    status, reason_txt = "TARGET_NOT_IN_CROP", "目标不在图内"
                    step["err"] = round(((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5, 1)
                    rec["iterations"].append(step)
                    break
                if v["correct"] is True:
                    status, reason_txt = "SUCCESS", "目标在光标位置，校验通过"
                    step["err"] = round(((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5, 1)
                    rec["iterations"].append(step)
                    break
                if v["bbox_norm"]:          # 归一化 → 裁片内 → 原图
                    ox0, oy0 = box[0], box[1]
                    est = [ox0 + v["bbox_norm"][0] / 1000 * box[2],
                           oy0 + v["bbox_norm"][1] / 1000 * box[2],
                           ox0 + v["bbox_norm"][2] / 1000 * box[2],
                           oy0 + v["bbox_norm"][3] / 1000 * box[2]]
                    step["mapped_bbox"] = [round(val, 1) for val in est]
                step["err"] = round(
                    (((est[0] + est[2]) / 2 - gcx) ** 2
                     + ((est[1] + est[3]) / 2 - gcy) ** 2) ** 0.5, 1)
                rec["iterations"].append(step)
            ecx, ecy = (est[0] + est[2]) / 2, (est[1] + est[3]) / 2
            rec.update({"status": status, "reason": reason_txt,
                        "final_bbox": [round(v, 1) for v in est],
                        "final_center": [round(ecx, 1), round(ecy, 1)],
                        "final_err": round(((ecx - gcx) ** 2 + (ecy - gcy) ** 2) ** 0.5, 1),
                        "n_iter": len(rec["iterations"]),
                        "latency_s": round(time.time() - t0, 1)})
            tag = {"SUCCESS": "✅", "TARGET_NOT_IN_CROP": "🚫", "MAX_ITER": "❌"}[status]
            print(f"  [{n:>3}] {tag} {status:<20} L1={rec['l1_err']:>7.0f}px → "
                  f"终={rec['final_err']:>7.0f}px  循环{rec['n_iter']}次  {rid}")
        except Exception as e:  # noqa: BLE001
            rec.update({"status": "ERROR", "reason": f"{type(e).__name__}: {e}"})
            print(f"  [{n:>3}] 💥 {type(e).__name__}: {e}  {rid}")
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"\n结果 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
