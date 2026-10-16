#!/usr/bin/env python3
"""B2 —— 单图 + 脚本校验循环。

机制：
    1) L1：整图 + 短提示 → 初定位 bbox
    2) 循环（默认 3 轮）：以当前 bbox 中心裁一块（边长 = max(200, 目标长边×1.5)，上限 1600）
       → 缩放到 1000px → 在【裁片中心】画一个方块标记 → 问模型「方块在目标上吗」
       · 在目标上 → 结束
       · 目标不在图内 → 结束（记 TARGET_NOT_IN_CROP）
       · 不在目标上 → 用模型在裁片里报的 bbox 更新位置，继续下一轮

与 B3/B4 的区别：这里是【脚本】决定下一步，模型只回答"对/不对"。
    python3 baselines/b2_verify_loop.py --split M1 --backend qwen
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from bench import dataset as D  # noqa: E402
from bench import prompts as P  # noqa: E402
from bench import runner as R  # noqa: E402
from bench import vision as V  # noqa: E402

NAME = "b2_verify_loop"

CROP_PX = 200      # 起始裁片边长（原图像素）
SENT_PX = 1000     # 送模型的图固定缩放到这个边长
MARK_PX = 24.0     # 标记在"1000px 宽"尺度下的像素高度


def parse_verdict(text: str) -> dict:
    """解析四行裁决：在图中 / 方块在目标上 / 目标bbox / 矛盾点。"""
    t = (text or "").replace("：", ":")
    in_img = None
    if re.search(r"在图中\s*:?\s*是", t):
        in_img = True
    elif re.search(r"在图中\s*:?\s*否", t):
        in_img = False
    correct = None
    if re.search(r"方块在目标上\s*:?\s*不确定", t):
        correct = None
    elif re.search(r"方块在目标上\s*:?\s*是", t):
        correct = True
    elif re.search(r"方块在目标上\s*:?\s*否", t):
        correct = False
    m = re.search(r"目标bbox\s*:?\s*\[([^\]]+)\]", t)
    bbox = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))][:4] if m else None
    if bbox and len(bbox) < 4:
        bbox = None
    return {"in_image": in_img, "correct": correct, "bbox_norm": bbox}


def run_one(rec, d, client, a):
    img = Image.open(D.fetch_image(rec["img_filename"])).convert("RGB")
    W, H = rec["img_size"]
    gt = rec["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    l1_px = 1.3e6 if a.backend == "deepseek" else D.TARGET_PIXELS

    # ---- 1) L1 初定位 ----
    small = V.fit(img, l1_px)
    txt = client.ask_image(small, P.L1_PROMPT.format(desc=P.desc_lines(d)),
                           max_tokens=512, reasoning_effort=a.reasoning_effort)
    bb = D.parse_bbox(txt)
    out = {"l1_answer": (txt or "").strip()[:200], "iterations": []}
    if not bb:
        out.update({"status": "L1_NO_OUTPUT", "final_px": None, "n_move": 0})
        return out
    est = D.norm_to_orig(bb, W, H)
    out["l1_bbox"] = [round(v, 1) for v in est]
    out["l1_err"] = round(((est[0] + est[2]) / 2 - gcx) ** 2
                          + ((est[1] + est[3]) / 2 - gcy) ** 2, 1) ** 0.5

    # ---- 2) 校验循环 ----
    # 锚点只给 L1：验证模型看到锚点会当硬约束，产生假矛盾（见 docs/ARCHIVE4.md）
    vdesc = f"外观：{d['feat']}；控件类型：{d['ctrl']}"
    if d.get("env"):
        vdesc += f"；周边锚点：{d['env']}"

    status, n_move = "MAX_ITER", 0
    for it in range(1, a.max_turns + 1):
        cx, cy = (est[0] + est[2]) / 2, (est[1] + est[3]) / 2
        tlen = max(est[2] - est[0], est[3] - est[1])
        cpx = int(min(max(CROP_PX, round(tlen * 1.5)), 1600))
        region, _ = V.crop_around(img, cx, cy, cpx, zoom=1)
        # ① 先缩放 ② 再画标记（标记尺寸按缩放后的尺度定）
        sc = SENT_PX / max(region.size)
        sent = region.resize((max(1, round(region.width * sc)),
                              max(1, round(region.height * sc))), Image.LANCZOS)
        cur_w = max(sent.size) * MARK_PX / 1000.0
        sent = V.draw_square(sent, sent.size[0] / 2, sent.size[1] / 2, cur_w)

        vtxt = client.ask_image(sent, P.VERIFY_PROMPT.format(mark=P.MARK_DESC_SQUARE, desc=vdesc),
                                max_tokens=512, reasoning_effort=a.reasoning_effort)
        v = parse_verdict(vtxt)
        n_move += 1
        out["iterations"].append({"iter": it, "crop_px": cpx, "sent_size": list(sent.size),
                                  "cursor_orig": [round(cx, 1), round(cy, 1)],
                                  "verdict": v["correct"], "in_image": v["in_image"],
                                  "raw": (vtxt or "").strip()[:200]})
        if v["in_image"] is False:
            status = "TARGET_NOT_IN_CROP"
            break
        if v["correct"]:
            status = "OK"
            break
        if v["bbox_norm"]:
            # 裁片归一化 → 原图坐标：裁片左上角 + 归一化 × 裁片边长
            x0, y0 = cx - cpx / 2, cy - cpx / 2
            b = v["bbox_norm"]
            est = [x0 + b[0] / 1000 * cpx, y0 + b[1] / 1000 * cpx,
                   x0 + b[2] / 1000 * cpx, y0 + b[3] / 1000 * cpx]
    out.update({
        "status": status, "n_move": n_move,
        "final_bbox": [round(v, 1) for v in est],
        "final_px": [round((est[0] + est[2]) / 2, 1), round((est[1] + est[3]) / 2, 1)],
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_common_args(ap, default_max_turns=3)
    ap.add_argument("--max-iter", type=int, default=0, help="校验循环次数（覆盖 --max-turns）")
    a = ap.parse_args()
    if a.max_iter:
        a.max_turns = a.max_iter
    return R.run(NAME, run_one, a)


if __name__ == "__main__":
    raise SystemExit(main())
