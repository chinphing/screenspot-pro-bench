#!/usr/bin/env python3
"""下载 ScreenSpot-Pro 数据集到本地（标注 + 全量图片）。

数据源：HuggingFace `likaixin/ScreenSpot-Pro`（经 hf-mirror.com 镜像）
落盘：data/annotations/<应用>_<平台>.json   —— 上游原件，27 份（含 1 份上游本就没有数据的占位）
      data/images/<应用目录>/<文件名>.png   —— 约 1581 张，3.1GB

已存在的文件会跳过，所以可以反复跑来补齐失败项。

    python3 scripts/download_data.py                    # 标注 + 图片，8 并发
    python3 scripts/download_data.py 4                  # 图片并发降到 4（镜像限流时）
    python3 scripts/download_data.py --only annotations
    python3 scripts/download_data.py --only images

⚠️ 图片镜像会瞬时限流：8 并发首轮可能有 ~8% 失败，降到 2-4 并发重跑即可补齐。
⚠️ 标注必须先取——图片的文件名来自标注里的 img_filename，标注没下来就没法知道要下哪些图。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import ANNO, IMGDIR  # noqa: E402
from bench import dataset as D  # noqa: E402


def get_annotations() -> int:
    """取上游标注。返回「上游缺失」的份数（不算失败）。"""
    print(f"【标注】上游 {len(D.ANNO_FILES)} 份 → {ANNO}", flush=True)
    new = skip = 0
    missing: list[str] = []
    for stem in D.ANNO_FILES:
        existed = (ANNO / f"{stem}.json").exists()
        p = D.fetch_annotation(stem)
        if p is None:
            missing.append(stem)
        elif existed:
            skip += 1
        else:
            new += 1
            print(f"  + {stem}.json", flush=True)
    have = len(list(ANNO.glob("*.json")))
    print(f"  完成：新增 {new} / 已存在 {skip} / 上游无此文件 {len(missing)}   本地现有 {have} 份")
    if missing:
        print(f"  （上游本就没有，属正常：{', '.join(missing)}）")
    return len(missing)


def get_images(workers: int) -> int:
    """取全量图片。返回下载失败的张数。"""
    rows = D.load_all()
    names = sorted({r["img_filename"] for r in rows})
    have = {str(p.relative_to(IMGDIR)) for p in IMGDIR.rglob("*") if p.is_file()}
    todo = [n for n in names if n not in have]
    print(f"\n【图片】数据集 {len(rows)} 条 / 唯一图 {len(names)} 张 / "
          f"已缓存 {len(names)-len(todo)} / 待下载 {len(todo)}  并发 {workers}", flush=True)
    if not todo:
        print("  ✅ 已全部就绪")
        return 0

    ok = fail = 0
    errs: list[str] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(D.fetch_image, n): n for n in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                fut.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                if len(errs) < 20:
                    errs.append(f"{futs[fut]}: {type(e).__name__}: {str(e)[:70]}")
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el else 0
                print(f"  [{i:>4}/{len(todo)}] ok={ok} fail={fail} {el/60:.1f}min "
                      f"{rate:.1f}张/s ETA {(len(todo)-i)/rate/60 if rate else 0:.1f}min", flush=True)

    size = sum(p.stat().st_size for p in IMGDIR.rglob("*") if p.is_file())
    n_have = len([p for p in IMGDIR.rglob("*") if p.is_file()])
    print(f"\n完成: 成功 {ok} / 失败 {fail}   用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"缓存 {IMGDIR}  共 {n_have} 个文件, {size/1024/1024/1024:.2f} GB")
    if errs:
        print("  失败样例（重跑即可补齐）：")
        for e in errs[:10]:
            print(f"    {e}")
    return fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workers", nargs="?", type=int, default=8,
                    help="图片并发数（默认 8；镜像限流时降到 2-4）")
    ap.add_argument("--only", choices=["all", "annotations", "images"], default="all",
                    help="只取其中一类（默认 all = 标注 + 图片）")
    a = ap.parse_args()

    if a.only in ("all", "annotations"):
        get_annotations()
    if a.only in ("all", "images"):
        fail = get_images(a.workers)
        return 1 if fail else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
