#!/usr/bin/env python3
"""L1 分块 + 块内给框 探针（3×3 一次请求）。

与 probe_tiles.py 的区别：
  · 上一版只问「目标在第几块」——模型必须把「屏幕位置：右侧/中下」映射成块号，
    而它不知道任何尺度，于是猜测图像尺寸、自我怀疑，把思考耗在建立坐标系上
    （轨迹实录：「不，每块宽度看起来是1440？不对…」）。
  · 这一版改问「块号 + 该块内的归一化框(1-1000)」——模型只需看块内，
    不需要知道原图多大；块号由框中心反推，harness 负责映射回原图。

不接入管线，只出数据。结果落 --out（默认 /tmp/probe_box.jsonl）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from bench import dataset as H  # noqa: E402
from bench.compat import qwen_client as qc  # noqa: E402

GRID = 3
OVERLAP = 0.12
SINGLE_PX = 2.5e6


def tiles_for(W, Hh, n=GRID, ov=OVERLAP):
    """按阅读顺序返回 [(编号, x0, y0, x1, y1)]，编号 1..n*n。"""
    tw, th = W / n, Hh / n
    out = []
    for r in range(n):
        for c in range(n):
            x0, y0 = c * tw, r * th
            dx, dy = tw * ov, th * ov
            out.append((r * n + c + 1,
                        max(0, int(x0 - dx)), max(0, int(y0 - dy)),
                        min(W, int(x0 + tw + dx)), min(Hh, int(y0 + th + dy))))
    return out


def cover(box, gt):
    """真值框被 box 覆盖的比例。"""
    ix = max(0, min(box[2], gt[2]) - max(box[0], gt[0]))
    iy = max(0, min(box[3], gt[3]) - max(box[1], gt[1]))
    ga = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return (ix * iy / ga) if ga > 0 else 0.0


def center_in(box, gt):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def err_px(box, gt):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    gx, gy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return math.hypot(cx - gx, cy - gy)


def ask(orig, ts, desc, reasoning_effort="high"):
    head = (f"下面按【阅读顺序】给出同一张截图的 {len(ts)} 个分块，编号 1-{len(ts)}。\n"
            "这张截图被切成 3×3 共 9 块，位置关系像拼图一样相邻：\n"
            "  第1块 第2块 第3块   ← 屏幕最上一行\n"
            "  第4块 第5块 第6块   ← 屏幕中间一行\n"
            "  第7块 第8块 第9块   ← 屏幕最下一行\n"
            "（第1块=左上角，第9块=右下角；相邻块之间有少量重叠）\n")
    content = [{"type": "text", "text": head}]
    for idx, x0, y0, x1, y1 in ts:
        content.append({"type": "text", "text": f"【第 {idx} 块】"})
        region = orig.crop((x0, y0, x1, y1))
        k = min(1.0, (SINGLE_PX / (region.width * region.height)) ** 0.5)
        if k < 1.0:
            region = region.resize((max(1, int(region.width * k)),
                                    max(1, int(region.height * k))), Image.LANCZOS)
        buf = io.BytesIO()
        region.save(buf, format="JPEG", quality=85)
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
    content.append({"type": "text", "text":
                    f"\n【目标元素描述】\n{desc}\n"
                    "请判断目标元素在哪一块，并给出它在【那一块内】的精确位置。\n"
                    "坐标用【块内】归一化值：1 = 该块最左/最上边缘，1000 = 该块最右/最下边缘。\n"
                    "你不需要知道原图多大，只按你看到的那一块来报。\n"
                    "只回一个 JSON：\n"
                    "{\"tile\": <编号>, \"x0\": <左>, \"y0\": <上>, \"x1\": <右>, \"y1\": <下>,"
                    " \"reason\": \"一句话\"}"})
    _mt = 200 if reasoning_effort in (None, "", "none") else 8000
    r = qc.chat([{"role": "user", "content": content}], max_tokens=_mt,
                timeout=600, retries=4, extra={"reasoning_effort": reasoning_effort})
    msg = r["choices"][0]["message"]
    txt = (msg.get("content") or "")
    th = (msg.get("reasoning_content") or "").strip()
    got = {}
    for k in ("tile", "x0", "y0", "x1", "y1"):
        m = re.search(rf'["\']?{k}["\']?\s*:\s*(-?\d+)', txt)
        if m:
            got[k] = int(m.group(1))
    return got, txt.strip()[:240], th[:3000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="M3")
    ap.add_argument("--ids-file", default="/tmp/ab100_ids.txt")
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--out", default="/tmp/probe_box.jsonl")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reasoning-effort", default="high")
    a = ap.parse_args()

    ids = [i for i in re.split(r"[,\s]+", open(a.ids_file, encoding="utf-8").read()) if i]
    rows = {r["id"]: r for r in H.load_all()}
    DESC = json.load(open(ROOT / "descriptions" / a.desc_version / f"{a.split}.json",
                          encoding="utf-8"))
    l1 = {}
    for f in (ROOT / "baselines" / "results" / "raw").glob("b3_ab100_ctrl_M3_v2_q38_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for t in (r.get("trace") or []):
                if t.get("action") == "move" and t.get("ref") == "整图(L1)":
                    l1[r["id"]] = t["bbox_orig"]
                    break

    def one(rid):
        src, d = rows[rid], DESC[rid]
        orig = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
        W, Hh = orig.size
        ts = tiles_for(W, Hh)
        desc = (f"· 外观特征：{d['feat']}\n· 控件类型：{d['ctrl']}\n"
                f"· 屏幕位置：{d['pos']}\n")
        try:
            got, raw, th = ask(orig, ts, desc, a.reasoning_effort)
        except Exception as e:  # noqa: BLE001
            return {"id": rid, "status": "ERROR", "error": str(e)[:150]}
        if not all(k in got for k in ("tile", "x0", "y0", "x1", "y1")):
            return {"id": rid, "status": "PARSE_FAIL", "got": got, "raw": raw, "think": th}
        t = next((x for x in ts if x[0] == got["tile"]), None)
        if t is None:
            return {"id": rid, "status": "BAD_TILE", "got": got, "raw": raw, "think": th}
        # 块内归一化 → 原图
        x0, x1 = sorted((got["x0"], got["x1"]))
        y0, y1 = sorted((got["y0"], got["y1"]))
        tx0, ty0, tx1, ty1 = t[1], t[2], t[3], t[4]
        box = [tx0 + x0 / 1000.0 * (tx1 - tx0), ty0 + y0 / 1000.0 * (ty1 - ty0),
               tx0 + x1 / 1000.0 * (tx1 - tx0), ty0 + y1 / 1000.0 * (ty1 - ty0)]
        gt = src["bbox"]
        rec = {"id": rid, "status": "OK", "got": got, "box": [round(v) for v in box],
               "hit": center_in(box, gt), "cov": round(cover(box, gt), 3),
               "err": round(err_px(box, gt), 1),
               "bw": round(box[2] - box[0], 1), "bh": round(box[3] - box[1], 1),
               "gtw": gt[2] - gt[0], "gth": gt[3] - gt[1],
               "raw": raw, "think": th}
        if rid in l1:
            b = [l1[rid][0] / 1000 * W, l1[rid][1] / 1000 * Hh,
                 l1[rid][2] / 1000 * W, l1[rid][3] / 1000 * Hh]
            rec.update({"l1_hit": center_in(b, gt), "l1_cov": round(cover(b, gt), 3),
                        "l1_err": round(err_px(b, gt), 1)})
        return rec

    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(one, ids), 1):
            out.append(rec)
            if i % 10 == 0:
                print(f"  …{i}/{len(ids)}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in out if r.get("status") == "OK")
    print(f"\n  完成 {ok}/{len(ids)}  →  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
