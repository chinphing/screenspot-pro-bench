#!/usr/bin/env python3
"""L1 分块可行性探针：3×3 切块、**一次请求全部送入**，测「块选择准确率」。

为什么这样测：
  服务端单图上限 ≈2.5Mpx（实测：6016/4211/3008 三种输入给出相同 image_tokens=2481）。
  分块能不能突破这个上限，取决于上限是【每张图】还是【每次请求】——
  所以这里一次请求送 9 张，直接看块选择准不准。

不接入管线，只出数据。结果落 --out（默认 /tmp/probe_tiles.jsonl）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from bench import dataset as H  # noqa: E402
from bench.compat import ds_client as dc  # noqa: E402
from bench.compat import qwen_client as qc  # noqa: E402

GRID = 3
OVERLAP = 0.12          # 每块向四周外扩的比例，避免目标正好跨边界
SINGLE_PX = 2.5e6       # 单块送出上限（与服务端实测上限一致）


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


def ovl(tile, gt):
    """真值框被该块覆盖的面积比例。"""
    ix = max(0, min(tile[2], gt[2]) - max(tile[0], gt[0]))
    iy = max(0, min(tile[3], gt[3]) - max(tile[1], gt[1]))
    ga = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return (ix * iy / ga) if ga > 0 else 0.0


def ask(orig, ts, desc, reasoning_effort="high", top_k=1, backend="qwen"):
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
    if top_k > 1:
        q = (f"请按可能性从高到低，给出你认为最可能的 {top_k} 个分块。\n"
             "只回一个 JSON：{\"rank\": [<第1名>, <第2名>, <第3名>], \"reason\": \"一句话\"}")
    else:
        q = ("请判断：目标元素出现在哪一个分块里？\n"
             "只回一个 JSON：{\"tile\": <编号>, \"reason\": \"一句话\"}")
    content.append({"type": "text", "text": f"\n【目标元素描述】\n{desc}\n" + q})
    _mt = 200 if reasoning_effort in (None, "", "none") else 8000
    if backend == "deepseek":
        r, _ = dc.chat([{"role": "user", "content": content}], max_tokens=_mt,
                       timeout=600, extra={"reasoning_effort": reasoning_effort})
    else:
        r = qc.chat([{"role": "user", "content": content}], max_tokens=_mt,
                    timeout=600, retries=4, extra={"reasoning_effort": reasoning_effort})
    msg = r["choices"][0]["message"]
    txt = (msg.get("content") or "")
    th = (msg.get("reasoning_content") or "").strip()
    mr = re.search(r'"rank"\s*:\s*\[([^\]]*)\]', txt)
    if mr:
        rank = [int(x) for x in re.findall(r"\d+", mr.group(1))][:top_k]
    else:
        m1 = re.search(r'"tile"\s*:\s*(\d+)', txt) or re.search(r"第?\s*([1-9])\s*块", txt)
        rank = [int(m1.group(1))] if m1 else []
    return rank, txt.strip()[:240], th[:3000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="M3")
    ap.add_argument("--ids-file", default="/tmp/ab100_ids.txt")
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--out", default="/tmp/probe_tiles.jsonl")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reasoning-effort", default="high",
                    help="思考强度：none / low / medium / high（9 选 1 建议 high）")
    ap.add_argument("--top-k", type=int, default=1,
                    help="要求模型给出前 k 个候选分块（k>1 时输出排序列表）")
    a = ap.parse_args()

    ids = [i for i in re.split(r"[,\s]+", open(a.ids_file, encoding="utf-8").read()) if i]
    rows = {r["id"]: r for r in H.load_all()}
    DESC = json.load(open(ROOT / "descriptions" / a.desc_version / f"{a.split}.json",
                          encoding="utf-8"))
    # 对照臂的 L1 框（用于「现有 L1 的等效块准确率」）
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
            rank, raw, th = ask(orig, ts, desc, a.reasoning_effort, a.top_k)
        except Exception as e:  # noqa: BLE001
            return {"id": rid, "status": "ERROR", "error": str(e)[:150]}
        if not rank:
            return {"id": rid, "status": "PARSE_FAIL", "raw": raw, "think": th}
        gt = src["bbox"]
        covs = {x[0]: ovl((x[1], x[2], x[3], x[4]), gt) for x in ts}
        full = {x[0] for x in ts if covs[x[0]] >= 0.999}
        hits = [1 if t in full else 0 for t in rank]
        l1tile, l1cov = None, None
        if rid in l1:
            bx = [l1[rid][0] / 1000 * W, l1[rid][1] / 1000 * Hh,
                  l1[rid][2] / 1000 * W, l1[rid][3] / 1000 * Hh]
            l1cov = round(ovl(bx, gt), 3)
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            for x in ts:
                if x[1] <= cx <= x[3] and x[2] <= cy <= x[4]:
                    l1tile = x[0]
                    break
        return {"id": rid, "status": "OK", "rank": rank, "hits": hits,
                "cov_top": round(covs.get(rank[0], 0.0), 3),
                "full_tiles": sorted(full), "l1_tile": l1tile, "l1_cov": l1cov,
                "raw": raw, "think": th}

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
