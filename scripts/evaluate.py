#!/usr/bin/env python3
"""统一评估：命中率 + 落点质量 + 随机基线对照。

    python3 scripts/evaluate.py baselines/results/raw/b4_crop_ladder_qwen_M1.jsonl
    python3 scripts/evaluate.py 'baselines/results/raw/*_M1*.jsonl' --labels 'B1,B2,B3,B4'

⚠️ 只看命中率会漏掉一层：命中只表示落点在真值框内，不表示落得准。
   本脚本同时报告"落点在框内有多居中"，并对照随机基线（距边中位 29%、擦边率 44%）。
   低于基线 = "命中"基本靠运气。指标定义见 docs/METRICS.md。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import metrics as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+", help="结果文件 glob 或路径")
    ap.add_argument("--labels", default="", help="逗号分隔的显示名（与 patterns 一一对应）")
    ap.add_argument("--by", default="", choices=["", "app", "ui_type", "platform", "group", "size"],
                    help="额外按某维度分组（size = 目标长边档）")
    a = ap.parse_args()

    labels = [x.strip() for x in a.labels.split(",") if x.strip()] if a.labels else []
    print(M.header())
    all_rows: list[tuple[str, list[dict]]] = []
    for i, pat in enumerate(a.patterns):
        rows = M.load(pat)
        label = labels[i] if i < len(labels) else Path(pat).stem
        all_rows.append((label, rows))
        print(M.format_line(label, M.evaluate(rows)))
    print("=" * 150)

    if a.by:
        for label, rows in all_rows:
            groups: dict[str, list[dict]] = {}
            for r in rows:
                key = M.size_bucket(r) if a.by == "size" else (r.get(a.by) or "?")
                groups.setdefault(str(key), []).append(r)
            print(f"\n【{label}】按 {a.by} 分组")
            for k in sorted(groups, key=lambda x: -len(groups[x])):
                print(M.format_line(f"  {k}", M.evaluate(groups[k])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
