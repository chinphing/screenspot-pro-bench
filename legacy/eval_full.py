#!/usr/bin/env python3
"""ScreenSpot 定位结果统一评估（含"落点质量"第二层指标）。

用法:
    python3 eval_full.py <glob...> [--label NAME]...

示例:
    python3 eval_full.py 'results/l1/cursorZF_qfix[1-5].jsonl' --label 'ZF·Qwen'
    python3 eval_full.py A.jsonl B.jsonl --label X --label Y

指标:
    命中率          落点落在真值框内
    到中心          落点 → 真值框中心 的距离（px / 目标长边百分比）
    距边/半边长     落点 → 最近框边 的距离 ÷ 半边长   ← 落点在框内有多居中
    擦边率          距边/半边长 < 25% 的比例（差一点就掉出去）
    vs 随机基线     随机基线 = 距边中位 29%、擦边率 44%（20 万次蒙特卡洛）
    失败距离        未命中样本离框中心多远（判断"差一点"还是"差很远"）
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import statistics
import sys

# 随机基线：落点在真值框内均匀分布（20 万次蒙特卡洛，见 ARCHIVE4 §36.2）
BASE_EDGE_MED = 0.293     # 距边/半边长 中位
BASE_MARGINAL = 0.44      # 擦边率


def load(pattern: str) -> list[dict]:
    rows: list[dict] = []
    files = sorted(_glob.glob(pattern)) or ([pattern] if pattern else [])
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except FileNotFoundError:
            pass
    return rows


def _final(r: dict):
    if "final_px" in r:
        return r["final_px"]
    if "final_center" in r:
        return r["final_center"]
    return None


def _hit(r: dict, fc) -> bool:
    if "hit" in r:
        return bool(r["hit"])
    gb = r.get("gt_bbox")
    return bool(fc and gb and gb[0] <= fc[0] <= gb[2] and gb[1] <= fc[1] <= gb[3])


def evaluate(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") != "ERROR"]
    errs = [r for r in rows if r.get("status") == "ERROR"]
    d_center, d_rel, d_edge, d_edge_frac, margins = [], [], [], [], []
    miss_dist = []
    for r in ok:
        gb = r.get("gt_bbox")
        fc = _final(r)
        if not gb or not fc:
            continue
        gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
        length = max(gb[2] - gb[0], gb[3] - gb[1])
        half = length / 2
        d = ((fc[0] - gcx) ** 2 + (fc[1] - gcy) ** 2) ** 0.5
        if not _hit(r, fc):
            miss_dist.append(d)
            continue
        d_center.append(d)
        d_rel.append(d / length if length else 0.0)
        e = min(fc[0] - gb[0], gb[2] - fc[0], fc[1] - gb[1], gb[3] - fc[1])
        d_edge.append(e)
        d_edge_frac.append(e / half if half else 0.0)
        margins.append(e / half if half else 0.0)
    lat = [r["latency_s"] for r in ok if "latency_s" in r]
    calls = [r.get("n_move", 0) + 1 for r in ok]
    return {
        "n": len(ok), "err": len(errs),
        "hit": len(d_center), "rate": len(d_center) / len(ok) if ok else 0.0,
        "d_center": statistics.median(d_center) if d_center else None,
        "d_rel": statistics.median(d_rel) if d_rel else None,
        "d_p75": sorted(d_center)[int(len(d_center) * 0.75)] if d_center else None,
        "edge_med": statistics.median(d_edge_frac) if d_edge_frac else None,
        "marginal": (sum(1 for x in margins if x < 0.25) / len(margins)) if margins else None,
        "miss_med": statistics.median(miss_dist) if miss_dist else None,
        "miss_n": len(miss_dist),
        "lat": statistics.mean(lat) if lat else None,
        "lat_med": statistics.median(lat) if lat else None,
        "calls": statistics.mean(calls) if calls else None,
    }


def report(label: str, m: dict) -> None:
    if not m["n"]:
        print(f"  {label:<24} （无数据）")
        return
    edge = m["edge_med"]
    marg = m["marginal"]
    edge_vs = (edge - BASE_EDGE_MED) / BASE_EDGE_MED * 100 if edge is not None else None
    marg_vs = (marg - BASE_MARGINAL) / BASE_MARGINAL * 100 if marg is not None else None
    verdict = "—"
    if edge is not None and marg is not None:
        if marg < BASE_MARGINAL and edge > BASE_EDGE_MED:
            verdict = "优于随机"
        elif marg > BASE_MARGINAL and edge < BASE_EDGE_MED:
            verdict = "≈随机(命中靠运气)"
        else:
            verdict = "略优于随机"
    print(f"  {label:<24} 命中 {m['hit']:>3}/{m['n']:<3} = {m['rate']*100:>3.0f}%   ERROR {m['err']:<2} "
          f"单次 {m['lat']:>4.1f}s  调用 {m['calls']:>4.2f}   "
          f"到中心 {m['d_center']:>4.0f}px({m['d_rel']*100:>3.0f}%) P75 {m['d_p75']:>4.0f}px   "
          f"距边 {edge*100:>3.0f}%({edge_vs:>+3.0f}%)  擦边 {marg*100:>3.0f}%({marg_vs:>+3.0f}%)  "
          f"{verdict}")
    if m["miss_med"] is not None:
        print(f"  {'':<24}   ↳ 失败 {m['miss_n']} 条，离中心中位 {m['miss_med']:.0f}px")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--labels", default=None,
                    help="逗号分隔，与 patterns 一一对应，如 --labels 'ZF·Qwen,Z6·Qwen'")
    args = ap.parse_args()
    labels = args.labels.split(",") if args.labels else []
    while len(labels) < len(args.patterns):
        labels.append(None)
    print("=" * 150)
    print(f'【统一评估】随机基线：距边中位 {BASE_EDGE_MED*100:.0f}%、擦边率 {BASE_MARGINAL*100:.0f}%'
          f'（落点在框内均匀分布）')
    print("=" * 150)
    for pat, lab in zip(args.patterns, labels):
        rows = load(pat)
        if not rows:
            print(f"  {(lab or pat):<24} （文件不存在或无数据）")
            continue
        report(lab or pat, evaluate(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
