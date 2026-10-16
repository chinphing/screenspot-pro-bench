#!/usr/bin/env python3
"""并发下载 ScreenSpot-Pro 全量图片到本地缓存（已存在则跳过）。

用法: python3 dl_all.py [并发数, 默认 8]
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harness as H  # noqa: E402


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = H.load_all()
    names = sorted({r["img_filename"] for r in rows})
    have = {str(p.relative_to(H.IMGDIR)) for p in H.IMGDIR.rglob("*") if p.is_file()}
    todo = [n for n in names if n not in have]
    print(f"数据集 {len(rows)} 条 / 唯一图 {len(names)} 张 / 已缓存 {len(names)-len(todo)} / 待下载 {len(todo)}",
          flush=True)

    ok = fail = 0
    errs: list[str] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(H.fetch_image, n): n for n in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            name = futs[fut]
            try:
                fut.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                if len(errs) < 20:
                    errs.append(f"{name}: {type(e).__name__}: {str(e)[:70]}")
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f"  [{i:>4}/{len(todo)}] ok={ok} fail={fail} "
                      f"{el/60:.1f}min  {rate:.1f}张/s  ETA {eta:.1f}min", flush=True)

    size = sum(p.stat().st_size for p in H.IMGDIR.rglob("*") if p.is_file())
    print(f"\n完成: 成功 {ok} / 失败 {fail}   用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"缓存目录 {H.IMGDIR}  共 {len(list(H.IMGDIR.glob('*')))} 个文件, {size/1024/1024/1024:.2f} GB")
    for e in errs:
        print(f"  ⚠️ {e}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
