"""评估指标：命中率 + 落点质量 + 随机基线。

只看命中率会漏掉一层：**命中只表示落点在真值框内，不表示落得准**。
所以这里额外给出"落点在框内有多居中"的指标，并对照随机基线。

随机基线（落点在真值框内均匀分布，20 万次蒙特卡洛）：
    距边/半边长 中位 = 29%
    擦边率(<25% 半边长) = 44%
→ 低于这个基线，说明"命中"基本靠运气。

指标定义见 docs/METRICS.md。
"""
from __future__ import annotations

import glob as _glob
import json
import statistics

BASE_EDGE_MED = 0.293
BASE_MARGINAL = 0.44


def load(pattern: str) -> list[dict]:
    """读取一个 glob（或具体文件）下的全部 jsonl 记录，并补齐样例元信息。"""
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
    return enrich(rows)


_META: dict | None = None


def _meta() -> dict:
    global _META
    if _META is None:
        from . import dataset as D
        _META = {r["id"]: r for r in D.load_all()}
    return _META


def enrich(rows: list[dict]) -> list[dict]:
    """按 id 从数据集补齐样例元信息。

    ⚠️ 为什么需要：baselines/ 下 b2/b3 是从 legacy 原样拷贝的，它们的 run_one 逐字未动，
       产出的结果行【没有】 app/ui_type/platform/group/target_len 字段（只有新写的
       bench/runner.py 才会写）。为了不改基线就能做分组分析，这里在指标层补齐。
    """
    meta = _meta()
    for r in rows:
        src = meta.get(r.get("id"))
        if not src:
            continue
        for dst, key in (("app", "application"), ("ui_type", "ui_type"),
                         ("platform", "platform"), ("group", "group")):
            if not r.get(dst):
                r[dst] = src.get(key)
        if not r.get("target_len"):
            gb = r.get("gt_bbox") or src.get("bbox")
            if gb:
                r["target_len"] = round(max(gb[2] - gb[0], gb[3] - gb[1]), 1)
        if not r.get("gt_bbox") and src.get("bbox"):
            r["gt_bbox"] = src["bbox"]
    return rows


def size_bucket(r: dict) -> str:
    """目标长边分档（与 splits 的分层口径一致）。"""
    t = r.get("target_len") or 0
    return "小(<40px)" if t < 40 else ("中(40-120px)" if t <= 120 else "大(>120px)")


def final_point(r: dict):
    """兼容各种结果格式：final_px（新） / final_center（verify_loop）。"""
    if "final_px" in r:
        return r["final_px"]
    if "final_center" in r:
        return r["final_center"]
    return None


def is_hit(r: dict, fc) -> bool:
    if "hit" in r:
        return bool(r["hit"])
    gb = r.get("gt_bbox")
    return bool(fc and gb and gb[0] <= fc[0] <= gb[2] and gb[1] <= fc[1] <= gb[3])


def final_box(r: dict):
    """最终报出的那个【框】，统一还原成原图像素坐标。

    ⚠️ 两条来源、坐标系不同，必须分别处理：
      · B1 走 bench/runner.py，直接写 final_bbox（已是原图像素）
      · B2/B3 是从 legacy 逐字拷贝的，只有 trace；框在最后一个 move 的
        bbox_orig 里——**名字叫 orig，实际是归一化 0-1000**，要按图尺寸还原。
    """
    if r.get("final_bbox"):
        return [float(v) for v in r["final_bbox"]]
    for t in reversed(r.get("trace") or []):
        if t.get("action") == "move" and t.get("bbox_orig"):
            b = t["bbox_orig"]
            sz = r.get("gt_img_size") or r.get("img_size")
            if not sz:
                return None
            W, H = sz
            return [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H]
    return None


def box_cover(pred, gt) -> tuple[int, float]:
    """框的包裹程度（口径见 docs/ARCHIVE4.md 第 19 节）。

    宏观 macro：预测框【完整包住】真值框 → 1，否则 0   （完整包裹率）
    微观 micro：min(1, 交集面积 / 真值面积)              （平均覆盖率）
    """
    inter = (max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0]))
             * max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1])))
    garea = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    macro = 1 if (pred[0] <= gt[0] and pred[1] <= gt[1]
                  and pred[2] >= gt[2] and pred[3] >= gt[3]) else 0
    return macro, (min(1.0, inter / garea) if garea > 0 else 0.0)


def evaluate(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") != "ERROR"]
    errs = [r for r in rows if r.get("status") == "ERROR"]
    d_center, d_rel, edge_frac, miss = [], [], [], []
    for r in ok:
        gb = r.get("gt_bbox")
        fc = final_point(r)
        if not gb or not fc:
            continue
        gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
        length = max(gb[2] - gb[0], gb[3] - gb[1])
        half = length / 2
        d = ((fc[0] - gcx) ** 2 + (fc[1] - gcy) ** 2) ** 0.5
        if not is_hit(r, fc):
            miss.append(d)
            continue
        d_center.append(d)
        d_rel.append(d / length if length else 0.0)
        e = min(fc[0] - gb[0], gb[2] - fc[0], fc[1] - gb[1], gb[3] - fc[1])
        edge_frac.append(e / half if half else 0.0)

    # 宏微观覆盖率：与落点独立的一遍，纯几何
    macro, micro = [], []
    for r in ok:
        gb = r.get("gt_bbox")
        pb = final_box(r)
        if not gb or not pb:
            continue
        m, u = box_cover(pb, gb)
        macro.append(m)
        micro.append(u)

    lat = [r["latency_s"] for r in ok if "latency_s" in r]
    calls = []
    for r in ok:
        if "n_move" in r:
            calls.append(r["n_move"] + 1)
        elif r.get("iterations"):          # 历史 verify_loop 结果：L1 + 每轮校验
            calls.append(len(r["iterations"]) + 1)
    return {
        "n": len(ok), "err": len(errs),
        "hit": len(d_center), "rate": len(d_center) / len(ok) if ok else 0.0,
        "d_center": statistics.median(d_center) if d_center else None,
        "d_rel": statistics.median(d_rel) if d_rel else None,
        "d_p75": sorted(d_center)[int(len(d_center) * 0.75)] if d_center else None,
        "edge_med": statistics.median(edge_frac) if edge_frac else None,
        "marginal": (sum(1 for x in edge_frac if x < 0.25) / len(edge_frac)) if edge_frac else None,
        "miss_med": statistics.median(miss) if miss else None,
        "miss_n": len(miss),
        "macro": (sum(macro) / len(macro)) if macro else None,
        "micro": statistics.mean(micro) if micro else None,
        "micro_med": statistics.median(micro) if micro else None,
        "box_n": len(macro),
        "lat": statistics.mean(lat) if lat else None,
        "lat_med": statistics.median(lat) if lat else None,
        "calls": statistics.mean(calls) if calls else None,
    }


def verdict(edge: float | None, marg: float | None) -> str:
    if edge is None or marg is None:
        return "—"
    if marg < BASE_MARGINAL and edge > BASE_EDGE_MED:
        return "优于随机"
    if marg > BASE_MARGINAL and edge < BASE_EDGE_MED:
        return "≈随机(命中靠运气)"
    return "略优于随机"


def format_line(label: str, m: dict) -> str:
    if not m["n"]:
        return f"  {label:<24} （无数据）"
    edge, marg = m["edge_med"], m["marginal"]
    ev = (edge - BASE_EDGE_MED) / BASE_EDGE_MED * 100 if edge is not None else 0
    mv = (marg - BASE_MARGINAL) / BASE_MARGINAL * 100 if marg is not None else 0
    out = (f"  {label:<24} 命中 {m['hit']:>3}/{m['n']:<3} = {m['rate']*100:>3.0f}%   ERROR {m['err']:<2} "
           f"单次 {m['lat']:>4.1f}s  调用 {m['calls']:>4.2f}   "
           f"到中心 {m['d_center']:>4.0f}px({m['d_rel']*100:>3.0f}%) P75 {m['d_p75']:>4.0f}px   "
           f"距边 {edge*100:>3.0f}%({ev:>+3.0f}%)  擦边 {marg*100:>3.0f}%({mv:>+3.0f}%)  "
           f"{verdict(edge, marg)}")
    if m.get("macro") is not None:
        out += (f"\n  {'':<24}   ↳ 框：宏观完整包裹 {m['macro']*100:>4.1f}%   "
                f"微观平均覆盖 {m['micro']*100:>4.1f}% (中位 {m['micro_med']*100:>4.1f}%)"
                f"   n={m['box_n']}")
    if m["miss_med"] is not None:
        out += f"\n  {'':<24}   ↳ 失败 {m['miss_n']} 条，离中心中位 {m['miss_med']:.0f}px"
    return out


def header() -> str:
    return ("=" * 150 + "\n"
            f"【统一评估】随机基线：距边中位 {BASE_EDGE_MED*100:.0f}%、"
            f"擦边率 {BASE_MARGINAL*100:.0f}%（落点在框内均匀分布）\n" + "=" * 150)
