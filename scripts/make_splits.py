#!/usr/bin/env python3
"""生成 M1 / M2 / M3 划分（可复现）。

M1 = 20 条开发集（前序实验用的固定集合，硬编码在本脚本里）
M2 = 300 条验证集（分层抽样：一级=应用，二级=目标尺寸档；种子 20261016）
M3 = 其余全部（留出集）

    python3 scripts/make_splits.py            # 写入 splits/
    python3 scripts/make_splits.py --verify   # 只校验已有划分与算法一致

⚠️ 划分一旦用于定成绩就不许再改；改种子等于换评测集。
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

SEED = 20261016
N_M2 = 300

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


def make(rows: list[dict]) -> dict:
    m1 = set(M1)
    pool = [r for r in rows if r["id"] not in m1]

    # 一级分层：应用（按占比等比分配名额）
    by_app: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_app[r["application"]].append(r)
    alloc = {app: max(1, round(N_M2 * len(rs) / len(pool))) for app, rs in by_app.items()}
    diff = N_M2 - sum(alloc.values())
    while diff != 0:
        k = (max if diff < 0 else min)(
            alloc, key=lambda a: alloc[a] if diff < 0 else len(by_app[a]) - alloc[a])
        if diff < 0:
            alloc[k] -= 1
            diff += 1
        elif alloc[k] < len(by_app[k]):
            alloc[k] += 1
            diff -= 1
        else:
            break

    # 二级分层：应用内按目标尺寸档占比分配
    rnd = random.Random(SEED)
    m2: list[str] = []
    for app, rs in sorted(by_app.items()):
        n = alloc[app]
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            buckets[size_bucket(r)].append(r)
        pick: list[dict] = []
        for b in sorted(buckets):
            nb = min(max(1, round(n * len(buckets[b]) / len(rs))) if n >= len(buckets) else 0,
                     len(buckets[b]))
            pick += rnd.sample(buckets[b], nb)
        if len(pick) < n:
            rest = [r for r in rs if r not in pick]
            pick += rnd.sample(rest, min(n - len(pick), len(rest)))
        m2 += [r["id"] for r in pick[:n]]

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
