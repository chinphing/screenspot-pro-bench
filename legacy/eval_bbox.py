#!/usr/bin/env python3
"""bbox 包裹成功率评估

两个指标（都基于**最终报出的那个框**）：
· 宏观 macro：框【完整包住】真值框 → 1，否则 0。均值 = 完整包裹率
· 微观 micro：min(1, 框∩真值 面积 / 真值面积)。均值 = 平均覆盖率

对照指标：
· hit：最终落点（框中心）是否在真值框内（原有口径）

用法：
  python3 eval_bbox.py results/l1/cursorZ4_full[1-5].jsonl
  python3 eval_bbox.py --label "Z4 qwen3.8" results/l1/cursorZ4_full[1-5].jsonl
"""
import argparse
import glob
import json
import statistics
import sys


def inter_area(a, b):
    """两个 bbox 的交集面积。bbox = [x0,y0,x1,y1]"""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def last_box(rec):
    """取最终报出的框（原图像素坐标）。没有则返回 None。"""
    boxes = [t.get("bbox_orig") for t in rec.get("trace", [])
             if t.get("action") == "move" and t.get("bbox_orig")]
    if not boxes:
        return None
    b = boxes[-1]
    W, H = rec["gt_img_size"]
    return [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H]


def load(patterns):
    recs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for line in open(f):
                r = json.loads(line)
                if r.get("status") != "ERROR":
                    recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--label", default="")
    ap.add_argument("--per-sample", action="store_true")
    args = ap.parse_args()

    recs = load(args.paths)
    if not recs:
        print("无有效记录")
        return 1

    macro, micro, hits, nobox = [], [], [], 0
    for r in recs:
        gt = r["gt_bbox"]
        gt_a = (gt[2] - gt[0]) * (gt[3] - gt[1])
        b = last_box(r)
        if b is None or gt_a <= 0:
            nobox += 1
            macro.append(0.0)
            micro.append(0.0)
            continue
        cov = min(1.0, inter_area(b, gt) / gt_a)
        micro.append(cov)
        macro.append(1.0 if cov >= 0.999 else 0.0)
        hits.append(1 if r.get("hit") else 0)

    n = len(recs)
    lab = args.label or args.paths[0]
    print(f"\n{'='*88}")
    print(f"【bbox 包裹成功率】{lab}")
    print(f"{'='*88}")
    print(f"  运行次数            {n}   （无框 {nobox} 次）")
    print(f"  ★ 宏观 完整包裹率    {sum(macro):.0f}/{n} = {sum(macro)/n*100:.1f}%")
    print(f"  ★ 微观 平均覆盖率    {statistics.mean(micro)*100:.1f}%   "
          f"（中位 {statistics.median(micro)*100:.1f}%）")
    print(f"    落点命中率(旧口径) {sum(hits)}/{len(hits)} = "
          f"{(sum(hits)/len(hits)*100) if hits else 0:.1f}%")

    if args.per_sample:
        D = {}
        for r in recs:
            D.setdefault(r["id"], []).append(r)
        print(f"\n  {'样例':<28}{'宏观':>9}{'微观':>9}{'落点':>9}{'真值尺寸':>12}")
        for i in sorted(D):
            v = D[i]
            ms, cs, hs = [], [], []
            for r in v:
                gt = r["gt_bbox"]
                gt_a = (gt[2] - gt[0]) * (gt[3] - gt[1])
                b = last_box(r)
                cov = min(1.0, inter_area(b, gt) / gt_a) if (b and gt_a > 0) else 0.0
                ms.append(1.0 if cov >= 0.999 else 0.0)
                cs.append(cov)
                hs.append(1 if r.get("hit") else 0)
            gt = v[0]["gt_bbox"]
            print(f"  {i:<28}{sum(ms):>5.0f}/{len(ms):<3}{statistics.mean(cs)*100:>8.0f}%"
                  f"{sum(hs):>5.0f}/{len(hs):<3}"
                  f"{f'{gt[2]-gt[0]:.0f}x{gt[3]-gt[1]:.0f}':>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
