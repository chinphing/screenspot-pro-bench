"""用 DeepSeek 跑 L1（单步定位），与 Qwen 对比。

提示词与 l1_only.py --desc-file 分支完全一致，只是换成 DeepSeek 后端。
输出 jsonl，字段与 results/l1/*.jsonl 兼容（id / l1_bbox / l1_err / latency_s ...）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import ds_client as dc  # noqa: E402

NUM4 = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def _fit(img, target_px):
    """缩放到不超过 target_px 像素（等比）。"""
    w, h = img.size
    if w * h <= target_px:
        return img, 1.0
    k = (target_px / (w * h)) ** 0.5
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    return img.resize((nw, nh), Image.LANCZOS), k


def parse_bbox(txt):
    m = NUM4.search(txt or "")
    if not m:
        return None
    return [int(x) for x in m.groups()]


def build_prompt(d, sw, sh, no_size=False):
    hdr = "" if no_size else f"这张图是 {sw} x {sh} 像素。"
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{hdr}请找出界面中符合以下描述的元素，给出它的 bbox。\n"
            f"\n【目标元素描述】\n"
            f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}"
            f"\n坐标请归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]，"
            f"其中 0 表示左/上边缘，1000 表示右/下边缘。"
            f"\n只输出这个数组，不要解释。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--detail", default=None, help="low / original / 不传")
    ap.add_argument("--reasoning-effort", default=None, help="none / low / high / max")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--tries", type=int, default=120)
    ap.add_argument("--no-size", action="store_true")
    ap.add_argument("--sent-px", type=float, default=1.3e6,
                    help="送图最大像素数（DeepSeek 处理上限约 1300×1300=1.69Mpx，送更多是浪费）")
    args = ap.parse_args()

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    out = os.path.join("results", "l1", f"ds_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"模型=deepseek-flash  detail={args.detail}  reasoning_effort={args.reasoning_effort}")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(args.ids, 1):
        src = rows[rid]
        d = DESC[rid]
        img = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
        sent, s = _fit(img, args.sent_px)
        sw, sh = sent.size
        prompt = build_prompt(d, sw, sh, args.no_size)
        t0 = time.time()
        try:
            txt, usage, reason, att = dc.ask_image(
                sent, prompt, max_tokens=args.max_tokens, tries=args.tries,
                detail=args.detail, reasoning_effort=args.reasoning_effort)
            dt = time.time() - t0
            nb = parse_bbox(txt)
            gt = src["bbox"]
            if nb:
                px = [nb[0] / 1000 * src["img_size"][0], nb[1] / 1000 * src["img_size"][1],
                      nb[2] / 1000 * src["img_size"][0], nb[3] / 1000 * src["img_size"][1]]
                ecx = (px[0] + px[2]) / 2 - (gt[0] + gt[2]) / 2
                ecy = (px[1] + px[3]) / 2 - (gt[1] + gt[3]) / 2
                err = (ecx ** 2 + ecy ** 2) ** 0.5
            else:
                px, err = None, None
            rec = {"id": rid, "gt_bbox": gt, "gt_img_size": src["img_size"],
                   "sent_size": [sw, sh], "l1_norm": nb, "l1_bbox": px,
                   "l1_err": err, "answer": txt, "reasoning": reason,
                   "usage": usage, "attempts": att, "latency_s": round(dt, 1),
                   "detail": args.detail, "reasoning_effort": args.reasoning_effort}
            recs.append(rec)
            flag = "✅" if (err is not None and err <= 50) else "❌"
            es = "  无" if err is None else f"{round(err):>4}px"
            print(f"  [{k:>2}] {rid:<26} 误差={es}  重试{att:>3}次  {dt:>5.1f}s  {flag}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k:>2}] {rid:<26} ❌ {str(e)[:70]}")
            recs.append({"id": rid, "status": "ERROR", "error": str(e)[:200],
                         "latency_s": round(time.time() - t0, 1)})
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")
    ok = sum(1 for r in recs if r.get("l1_err") is not None and r["l1_err"] <= 50)
    print(f"命中(≤50px) {ok}/{len(recs)}")


if __name__ == "__main__":
    main()
