#!/usr/bin/env python3
"""扫描 baselines/results/raw/ 生成汇总表 baselines/RESULTS.md。

文件命名约定：<baseline>_<backend>_<split>[_r<N>].jsonl
同名多份（r1/r2/…）会自动合并统计。

    python3 scripts/make_report.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import RESULTS  # noqa: E402
from bench import metrics as M  # noqa: E402

PAT = re.compile(r"^(?P<baseline>.+?)_(?P<backend>qwen|deepseek)_(?P<split>M[123])"
                 r"_(?P<descver>v\d+)_(?P<model>[a-z][a-z0-9]*)(?:_[rs]\d+)?$")


def main() -> int:
    raw = RESULTS / "raw"
    files = sorted(raw.glob("*.jsonl"))
    if not files:
        print(f"  ⚠️ {raw} 下没有结果文件")
        return 1

    groups: dict[tuple[str, str, str, str, str], list[Path]] = defaultdict(list)
    for f in files:
        m = PAT.match(f.stem)
        if not m:
            print(f"  ⚠️ 命名不符约定（需 <基线>_<后端>_<划分>_<描述版本>_<模型>），已跳过：{f.name}")
            continue
        groups[(m["baseline"], m["backend"], m["split"], m["descver"], m["model"])].append(f)

    lines = ["# 基线成绩汇总（自动生成，勿手改）", "",
             "> 生成命令：`python3 scripts/make_report.py`",
             "> 指标定义见 [`../docs/METRICS.md`](../docs/METRICS.md)；"
             "评测协议见 [`../docs/PROTOCOL.md`](../docs/PROTOCOL.md)", "",
             "随机基线：距边中位 **29%**、擦边率 **44%**（落点在框内均匀分布）", ""]

    for split in ("M1", "M2", "M3"):
        keys = [k for k in sorted(groups) if k[2] == split]
        if not keys:
            continue
        lines += [f"## {split}", "",
                  "| 基线 | 后端 | 模型 | 描述版本 | 命中 | ERROR | 单次 | 调用 | 到中心 | 距边 | 擦边 | 宏观包裹 | 微观覆盖 | 失败距离 | 判读 | 文件数 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for base, backend, _, ver, mdl in keys:
            rows = M.load(str(raw / f"{base}_{backend}_{split}_{ver}_{mdl}*.jsonl"))
            m = M.evaluate(rows)
            if not m["n"]:
                continue
            ev = (m["edge_med"] - M.BASE_EDGE_MED) / M.BASE_EDGE_MED * 100
            mv = (m["marginal"] - M.BASE_MARGINAL) / M.BASE_MARGINAL * 100
            mac = f"{m['macro']*100:.1f}%" if m.get("macro") is not None else "—"
            mic = (f"{m['micro']*100:.1f}% ({m['micro_med']*100:.1f}%)"
                   if m.get("micro") is not None else "—")
            lines.append(
                f"| `{base}` | {backend} | `{mdl}` | `{ver}` "
                f"| **{m['rate']*100:.0f}%** ({m['hit']}/{m['n']}) "
                f"| {m['err']} | {m['lat']:.1f}s | {m['calls']:.2f} "
                f"| {m['d_center']:.0f}px ({m['d_rel']*100:.0f}%) "
                f"| {m['edge_med']*100:.0f}% ({ev:+.0f}%) "
                f"| {m['marginal']*100:.0f}% ({mv:+.0f}%) "
                f"| {mac} | {mic} "
                f"| {m['miss_med']:.0f}px | {M.verdict(m['edge_med'], m['marginal'])} "
                f"| {len(groups[(base, backend, split, ver, mdl)])} |")
        lines.append("")

    out = RESULTS.parent / "RESULTS.md"      # 汇总表放 baselines/ 根下
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out}（{len(groups)} 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
