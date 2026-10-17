#!/usr/bin/env python3
"""前置认块 + 双块并行 + 置信度择优。

流程：
  1. 整图切 3×3（12% 重叠），**一次请求送 9 块** → 模型给出前 2 名候选块
  2. 对每个候选块：把块裁出来当「整图」，跑**现有 B3 流程**（run_one）→ 框 + 置信度
  3. 两个框映射回原图，取**置信度高的**那个

为什么这样分工（实验依据）：
  · 3×3 分块 top-2 召回 97%（n=99），top-1 91%，对照 L1 框心所在块 85%
  · 但让模型直接给「块内坐标」精度只有 74px（对照 L1 的 20px）——它会看不会换算
  · 所以：认块交给分块，**指位置仍用现有流程**（单图直接指，精度 20px 级）

结果落 --out（默认 /tmp/tile_pipeline.jsonl）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from bench import dataset as H  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b3 = _load("b3_crop_ladder", "baselines/b3_crop_ladder.py")
pt = _load("probe_tiles", "scripts/probe_tiles.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="M3")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--out-px", type=int, default=1600)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--reasoning-effort", default="none",
                    help="B3 流程内的思考强度（与基线一致默认 none）")
    ap.add_argument("--tile-reasoning", default="none",
                    help="认块那一步的思考强度（实测 none 的 top-2=93%%、32s/100 条；"
                         "high 的 top-2=97%% 但要 13 分钟，不值得）")
    ap.add_argument("--early-exit", type=float, default=90.0,
                    help="rank1 的置信度达到该值就直接采用，其余块不再发（<0 关闭）")
    ap.add_argument("--hedge-ms", type=int, default=12000,
                    help="等 rank1 多久后才开跑其余块（毫秒）；实测 12s 最优")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4, help="并发处理的样例数")
    ap.add_argument("--out", default="/tmp/tile_pipeline.jsonl")
    ap.add_argument("--show", action="store_true", help="打印逐条猫眼表")
    a = ap.parse_args()

    # 让 B3 走带置信度的分支
    b3.BACKEND = a.backend
    b3.CONFIDENCE = True

    ids = [i for i in re.split(r"[,\s]+", open(a.ids_file, encoding="utf-8").read()) if i]
    rows = {r["id"]: r for r in H.load_all()}
    DESC = json.load(open(ROOT / "descriptions" / a.desc_version / f"{a.split}.json",
                          encoding="utf-8"))
    TMP = Path(tempfile.mkdtemp(prefix="tilepipe_"))

    def one_sample(rid):
        src, d = rows[rid], DESC[rid]
        orig = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
        W, Hh = orig.size
        gt = src["bbox"]
        gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        ts = pt.tiles_for(W, Hh)
        desc = (f"· 外观特征：{d['feat']}\n· 控件类型：{d['ctrl']}\n"
                f"· 屏幕位置：{d['pos']}\n")

        t0 = time.time()
        # ① 认块
        try:
            rank, raw, _ = pt.ask(orig, ts, desc, a.tile_reasoning, a.top_k)
        except Exception as e:  # noqa: BLE001
            return {"id": rid, "status": "TILE_ERROR", "error": str(e)[:150]}
        if not rank:
            return {"id": rid, "status": "TILE_PARSE_FAIL", "raw": raw}

        # ② 两块并行跑现有流程
        def run_tile(tid):
            t = next(x for x in ts if x[0] == tid)
            tile = orig.crop((t[1], t[2], t[3], t[4]))
            path = TMP / f"{rid}_{tid}.png"
            tile.save(path)
            gt_tile = [gt[0] - t[1], gt[1] - t[2], gt[2] - t[1], gt[3] - t[2]]
            try:
                r = b3.run_one(rid, {"img_filename": str(path), "bbox": gt_tile}, d,
                               max_turns=a.max_turns, reasoning_effort=a.reasoning_effort,
                               out_px=a.out_px)
            except Exception as e:  # noqa: BLE001
                return {"tile": tid, "error": str(e)[:150]}
            nx, ny = r["final_norm"]
            ox = t[1] + nx / 1000.0 * (t[3] - t[1])
            oy = t[2] + ny / 1000.0 * (t[4] - t[2])
            stop = [x for x in (r.get("trace") or []) if x.get("action") == "stop"]
            return {"tile": tid, "rect": [t[1], t[2], t[3], t[4]],
                    "px": [round(ox), round(oy)],
                    "conf": (stop[-1].get("confidence") if stop else None),
                    "used_done": bool(stop),
                    "hit": gt[0] <= ox <= gt[2] and gt[1] <= oy <= gt[3],
                    "err": round(math.hypot(ox - gcx, oy - gcy), 1),
                    "n_move": r.get("n_move"), "turns": r.get("turns_used"),
                    "lat": r.get("latency_s"),
                    "trace_actions": "".join(
                        {"move": "M", "stop": "S", "widen": "W", "blurry": "B"}.get(x.get("action"), "?")
                        for x in (r.get("trace") or []))}

        # 对冲式早退：先只发 rank1。它在 hedge 窗口内回来且够自信 → 其余块根本不发
        # （少调用、且没有并发争用，rank1 自己也最快）；否则再并行补发其余块。
        _k = min(a.top_k, len(rank))
        ex = ThreadPoolExecutor(max_workers=_k)
        early = False
        try:
            f1 = ex.submit(run_tile, rank[0])
            r1 = None
            try:
                r1 = f1.result(timeout=max(0.001, a.hedge_ms / 1000.0))
            except FuturesTimeoutError:
                r1 = None
            if (r1 and a.early_exit >= 0 and r1.get("conf") is not None
                    and r1["conf"] >= a.early_exit):
                runs = [r1]
                early = True
            else:
                futs = [f1] + [ex.submit(run_tile, t) for t in rank[1:_k]]
                runs = []
                for f in futs:
                    try:
                        runs.append(f.result())
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            ex.shutdown(wait=False)
        ok_runs = [r for r in runs if "error" not in r]

        # ③ 择优：置信度高的胜；没有置信度（未 done）视为 -1
        def key(r):
            return r["conf"] if r.get("conf") is not None else -1
        best = max(ok_runs, key=key) if ok_runs else None

        return {"id": rid, "status": "OK", "rank": rank, "runs": runs, "early": early,
                "best_tile": best["tile"] if best else None,
                "best_hit": best["hit"] if best else None,
                "best_err": best["err"] if best else None,
                "any_hit": any(r["hit"] for r in ok_runs),
                "gt_tile": next((x[0] for x in ts if pt.ovl((x[1], x[2], x[3], x[4]), gt) >= 0.999), None),
                "lat_total": round(time.time() - t0, 1)}

    out = []
    fh = open(a.out, "w", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(one_sample, ids), 1):
            out.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()                      # 逐条落盘：中途取消也不丢已完成的
            print(f"  [{i}/{len(ids)}] {rec['id']}  {rec.get('status')}", flush=True)
    fh.close()

    ok = [r for r in out if r["status"] == "OK"]
    print(f"\n  完成 {len(ok)}/{len(out)}  →  {a.out}")
    if ok:
        print(f"  择优命中 {sum(1 for r in ok if r['best_hit'])}/{len(ok)}"
              f"   （任一块命中 {sum(1 for r in ok if r['any_hit'])}/{len(ok)}）")

    if a.show and ok:
        print("\n" + "=" * 132)
        print(f"{'样例':<28}{'块排序':<14}{'真值块':<8}{'块A: conf/hit/err':<24}"
              f"{'块B: conf/hit/err':<24}{'择优':<8}")
        print("-" * 132)
        for r in ok:
            cells = []
            for k in range(2):
                if k < len(r["runs"]) and "error" not in r["runs"][k]:
                    x = r["runs"][k]
                    c = x["conf"] if x["conf"] is not None else "—"
                    cells.append(f"块{x['tile']}: {c}/{int(x['hit'])}/{x['err']:.0f}px")
                else:
                    cells.append("—")
            while len(cells) < 2:
                cells.append("—")
            mark = "✅" if r["best_hit"] else "❌"
            print(f"{r['id']:<28}{str(r['rank']):<14}{str(r['gt_tile']):<8}"
                  f"{cells[0]:<24}{cells[1]:<24}块{r['best_tile']} {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
