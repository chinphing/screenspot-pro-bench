"""L1 用「原图 + 3×3 九宫格切图」共 10 张图送模型。

动机：DeepSeek 每张图 token 上限 1008（约等于 1300×1300），送整张 4K 截图时
目标被缩得很小。改成同时送整图（拿全局上下文）+ 九张切图（每张覆盖原图
1/3×1/3 区域，等效分辨率提升约 3 倍），让小目标在切图里变大。

用法：
  python3 ds_tiles.py --ids a,b,c --desc-file xxx.json --tag t1 [--backend deepseek|qwen]
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

TASK_HEAD = """你收到 {n} 张图，用于在界面里定位一个目标元素。

【图片构成与用法】
· 第 1 张是【完整原图】——用它判断目标大致在界面的哪个区域。
· 第 2～{n} 张是把原图按 3×3 切成的九块切图，每张覆盖原图 1/9 的区域。
  切图是【原分辨率】的，比完整原图清晰约 3 倍，用来【看清细节、精确定位】。

【推荐做法】
1. 先在【完整原图】里找到目标的大致位置，判断它落在哪个 1/3 × 1/3 区块；
2. 再到【对应那一张切图】里确认目标的确切位置；
3. 最后输出目标在【完整原图】里的归一化 bbox。

【坐标约定】
· 所有坐标一律归一化到 0-1000 的整数，0=左/上边缘，1000=右/下边缘。
· 每张图的说明里都标注了【该图的像素尺寸】和【它在完整原图中的归一化范围】。

【换算公式（在切图里量到像素后，换算成整图归一化坐标）】
  X_整图 = 切图范围x0 + (切图内像素x / 切图像素宽) × (切图范围x1 − 切图范围x0)
  Y_整图 = 切图范围y0 + (切图内像素y / 切图像素高) × (切图范围y1 − 切图范围y0)
  例：在第 2 张（尺寸 853×554，范围 [0,0,333,333]）里量到 x=435，
     则 X_整图 = 0 + (435/853) × 333 ≈ 170。

"""

TASK_TAIL = """
【输出要求】
请找出界面中符合以下描述的元素，给出它在【完整原图】里的 bbox。
坐标归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]。
只输出这个数组，不要解释。

【目标元素描述】
{desc}"""


def _desc_lines(d):
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}")


def split9(img):
    """按 3×3 切九块，返回 [(图, (x0,y0,x1,y1)), ...]（左→右、上→下）"""
    w, h = img.size
    out = []
    for r in range(3):
        for c in range(3):
            x0, y0 = c * w // 3, r * h // 3
            x1, y1 = (w if c == 2 else (c + 1) * w // 3), (h if r == 2 else (r + 1) * h // 3)
            out.append((img.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)))
    return out


def fit(img, target_px):
    w, h = img.size
    if w * h <= target_px:
        return img
    k = (target_px / (w * h)) ** 0.5
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def build_content(full, tiles, d, W0, H0):
    """图文交错：每张图前面加一行文字说明。"""
    n = 1 + len(tiles)
    content = [{"type": "text", "text": TASK_HEAD.format(n=n)}]

    content.append({"type": "text",
                    "text": f"【第 1 张】完整原图（图像尺寸 {full.width} × {full.height} 像素，"
                            f"占满整个界面，归一化范围 [0,0,1000,1000]）："})
    content.append({"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}})

    for idx, (t, box) in enumerate(tiles, start=2):
        r, c = (idx - 2) // 3 + 1, (idx - 2) % 3 + 1
        x0 = round(box[0] / W0 * 1000); y0 = round(box[1] / H0 * 1000)
        x1 = round(box[2] / W0 * 1000); y1 = round(box[3] / H0 * 1000)
        content.append({"type": "text",
                        "text": f"【第 {idx} 张】第 {r} 行、第 {c} 列的切图"
                                f"（图像尺寸 {t.width} × {t.height} 像素），"
                                f"它在完整原图中的归一化范围是 [{x0},{y0},{x1},{y1}]。"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(t)}"}})

    content.append({"type": "text",
                    "text": TASK_TAIL.format(desc=_desc_lines(d))})
    return content


def build_prompt(d, W0, H0, no_size=False):
    hdr = "" if no_size else f"这张图（第 1 张）是 {W0} x {H0} 像素的原始截图。"
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{hdr}\n{TILE_LAYOUT}\n\n"
            f"请找出界面中符合以下描述的元素，给出它在【第 1 张完整界面】里的 bbox。\n"
            f"\n【目标元素描述】\n{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}"
            f"\n坐标请归一化到 0-1000 的整数范围（相对第 1 张完整界面），"
            f"格式 [x1,y1,x2,y2]，其中 0 表示左/上边缘，1000 表示右/下边缘。"
            f"\n只输出这个数组，不要解释。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", choices=["deepseek", "qwen"], default="deepseek")
    ap.add_argument("--px", type=float, default=2.5e6, help="每张图的最大像素数")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--no-size", action="store_true")
    args = ap.parse_args()

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    out = os.path.join("results", "l1", f"tiles_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"后端={args.backend}  模式=原图+9切图(10张)  每张≤{args.px/1e6:.1f}Mpx")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(args.ids.split(","), 1):
        src = rows[rid]
        d = DESC[rid]
        img = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
        W0, H0 = img.size
        full = fit(img, args.px)
        tiles = [(fit(t, args.px), box) for t, box in split9(img)]
        content = build_content(full, tiles, d, W0, H0)
        t0 = time.time()
        try:
            if args.backend == "deepseek":
                resp, att = dc.chat([{"role": "user", "content": content}],
                                    max_tokens=args.max_tokens, tries=args.tries,
                                    extra={"reasoning_effort": args.reasoning_effort})
                txt = dc.text_of(resp)
                usage = resp.get("usage", {})
            else:
                resp = dc.chat([{"role": "user", "content": content}],
                               max_tokens=args.max_tokens)
                txt = resp["choices"][0]["message"].get("content") or ""
                usage = resp.get("usage", {})
                att = 1
            dt = time.time() - t0
            m = NUM4.search(txt)
            gt = src["bbox"]
            if m:
                nb = [int(v) for v in m.groups()]
                px = [nb[0] / 1000 * W0, nb[1] / 1000 * H0,
                      nb[2] / 1000 * W0, nb[3] / 1000 * H0]
                err = (((px[0] + px[2]) / 2 - (gt[0] + gt[2]) / 2) ** 2
                       + ((px[1] + px[3]) / 2 - (gt[1] + gt[3]) / 2) ** 2) ** 0.5
            else:
                nb, px, err = None, None, None
            rec = {"id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
                   "mode": "full+9tiles", "full_size": list(full.size),
                   "tile_sizes": [list(t.size) for t, _ in tiles],
                   "l1_norm": nb, "l1_bbox": px, "l1_err": err,
                   "answer": txt.strip()[:300], "usage": usage,
                   "attempts": att, "latency_s": round(dt, 1)}
            recs.append(rec)
            es = "  无" if err is None else f"{round(err):>4}px"
            print(f"  [{k:>2}] {rid:<26} 误差={es}  {dt:>5.1f}s  "
                  f"{'✅' if (err is not None and err <= 50) else '❌'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k:>2}] {rid:<26} ❌ {str(e)[:70]}", flush=True)
            recs.append({"id": rid, "status": "ERROR", "error": str(e)[:200],
                         "latency_s": round(time.time() - t0, 1)})
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")


if __name__ == "__main__":
    main()
