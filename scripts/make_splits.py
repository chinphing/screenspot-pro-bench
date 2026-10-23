#!/usr/bin/env python3
"""生成 M1 / M2 / M3 划分（可复现）。

M1 = 20 条开发集（前序实验用的固定集合，硬编码在本脚本里）
M2 = 100 条受控验证集（= experiments/ab100 的冻结集合：50 条原失败 + 50 条原成功）
M3 = 其余全部（留出集）

    python3 scripts/make_splits.py            # 写入 splits/
    python3 scripts/make_splits.py --verify   # 只校验已有划分与算法一致

⚠️ 划分一旦用于定成绩就不许再改；改种子等于换评测集。

为什么 M2 换成 ab100（2026-10-23）
    原来 M2 是按「应用 × 目标尺寸档」分层抽样的 300 条，问题是对比实验跑不动：
    300 条 × 每条约 30s，两臂就要 5 小时，迭代一轮太慢。
    ab100 是早就冻结的受控集（`experiments/ab100/ids.txt`，seed 20261019），
    50% 原失败 + 50% 原成功，命中率对方案差异更敏感——刻意放大了信号。
    代价是它的绝对命中率不能代表自然分布（M3 真实失败率约 17%，ab100 是 50%），
    所以最终成绩仍以 M3 为准，M2 只用来快速筛方案。

    原 M2 的 300 条没有丢，全部并入 M3（M3 = 原M2 + 原M3 − ab100）。
    旧划分备份在 `splits/_legacy/`。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import SPLITS  # noqa: E402
from bench import dataset as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# M2 的冻结来源。100 个 id 逗号分隔一行，由 experiments/ab100 的实验产生并锁定。
AB100_IDS = ROOT / "experiments" / "ab100" / "ids.txt"
N_M3_LEGACY_M2 = 300  # 仅用于 --verify 时核对并集规模

M1 = [
    "powerpoint_windows_56", "origin_windows_1", "vscode_macos_45", "matlab_macos_64",
    "pycharm_macos_1", "linux_common_linux_42", "solidworks_windows_73", "android_studio_macos_69",
    "fruitloops_windows_50", "autocad_windows_10", "premiere_windows_27", "vmware_macos_14",
    "windows_common_windows_48", "vivado_windows_77", "fruitloops_windows_54", "origin_windows_60",
    "vivado_windows_79", "word_macos_39", "pycharm_macos_42", "excel_macos_14",
]


def size_bucket(r: dict) -> str:
    L = D.target_len(r["bbox"])
    return "小(<40)" if L < 40 else ("中(40-120)" if L <= 120 else "大(>120)")


def load_ab100() -> list[str]:
    """读冻结的 ab100 集合（M2 的来源）。"""
    if not AB100_IDS.exists():
        raise SystemExit(f"找不到 {AB100_IDS}——M2 依赖它，缺失就无法复现划分")
    ids = [x.strip() for x in AB100_IDS.read_text(encoding="utf-8").replace("\n", ",").split(",") if x.strip()]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"ab100 有重复 id：{len(ids)} 条中只有 {len(set(ids))} 个唯一值")
    return ids


def make(rows: list[dict]) -> dict:
    m1 = set(M1)
    m2 = load_ab100()

    overlap = m1 & set(m2)
    if overlap:
        raise SystemExit(f"ab100 与 M1 有交集，划分会互相污染：{sorted(overlap)[:5]}")
    known = {r["id"] for r in rows}
    missing = [i for i in m2 if i not in known]
    if missing:
        raise SystemExit(f"ab100 有 {len(missing)} 个 id 不在数据集里：{missing[:5]}")

    # M3 = 其余全部（含原 M2 那 300 条，它们没有被丢掉，只是降级为留出集）
    m2set = set(m2)
    m3 = [r["id"] for r in rows if r["id"] not in m1 and r["id"] not in m2set]
    return {"M1": list(M1), "M2": m2, "M3": m3}


def report(rows: list[dict], sp: dict) -> None:
    idx = {r["id"]: r for r in rows}
    print(f"  全量 {len(rows)}   M1 {len(sp['M1'])}   M2 {len(sp['M2'])}   M3 {len(sp['M3'])}"
          f"   合计 {len(sp['M1'])+len(sp['M2'])+len(sp['M3'])}")
    print(f"  互斥：M1∩M2={len(set(sp['M1'])&set(sp['M2']))}  "
          f"M2∩M3={len(set(sp['M2'])&set(sp['M3']))}  M1∩M3={len(set(sp['M1'])&set(sp['M3']))}")
    print()
    full, m2 = [idx[i] for i in sp["M2"]], None  # noqa: F841
    print(f"  {'维度':<16}{'全量':>14}{'M2':>14}")
    for label, fn in [("目标尺寸", size_bucket),
                      ("平台", lambda r: r.get("platform")),
                      ("任务组", lambda r: r.get("group")),
                      ("UI 类型", lambda r: r.get("ui_type"))]:
        cf, cm = Counter(fn(r) for r in rows), Counter(fn(idx[i]) for i in sp["M2"])
        for k in sorted(cf):
            print(f"  {label+'·'+str(k):<16}"
                  f"{cf[k]:>6}({cf[k]/len(rows)*100:>4.1f}%)"
                  f"{cm[k]:>7}({cm[k]/max(1,len(sp['M2']))*100:>4.1f}%)")
    print(f"  {'应用数':<16}{len({r['application'] for r in rows}):>14}"
          f"{len({idx[i]['application'] for i in sp['M2']}):>14}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="只校验已有划分与算法一致，不写盘")
    a = ap.parse_args()

    rows = D.load_all()
    sp = make(rows)
    report(rows, sp)

    if a.verify:
        bad = 0
        for k in ("M1", "M2", "M3"):
            f = SPLITS / f"{k}.json"
            if not f.exists():
                print(f"  ⚠️ 缺 {f}")
                bad += 1
                continue
            old = json.load(open(f, encoding="utf-8"))
            if old != sp[k]:
                print(f"  ❌ {k} 与算法结果不一致（文件 {len(old)} 条，算法 {len(sp[k])} 条）")
                bad += 1
        print("\n  ✅ 划分可复现" if not bad else f"\n  ❌ {bad} 处不一致")
        return 0 if not bad else 1

    SPLITS.mkdir(parents=True, exist_ok=True)
    for k in ("M1", "M2", "M3"):
        f = SPLITS / f"{k}.json"
        json.dump(sp[k], open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  → {f}  ({len(sp[k])} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
