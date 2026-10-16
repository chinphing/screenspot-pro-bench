#!/usr/bin/env python3
"""并发分片驱动：把一次基线运行拆成 N 个分片并行执行。

**为什么不改基线脚本**：baselines/ 下的脚本是从 legacy 原样拷贝的（只改了 import/路径/CLI），
循环体逐字未动——并发如果做进脚本里就等于又改了一次行为。所以并发放在这一层：
每个分片是一个独立的基线进程，跑一部分样例，互不干扰。

分片方式：**轮转（round-robin）**。M2/M3 的 id 是按应用顺序排列的，
轮转能让每个分片都覆盖到各个应用，避免某个分片全是难样例。

    python3 scripts/run_parallel.py --baseline b3_crop_ladder --split M2 \\
        --backend qwen --desc-version v2 --shards 5

结果：baselines/results/raw/<tag>_s1.jsonl … _s5.jsonl（make_report.py 会按同一组聚合）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import clients as C  # noqa: E402


def model_tag(backend: str) -> str:
    """模型短标签（结果文件名必须能看出用的哪个模型）。

    实现在 bench.clients（单一来源，bench/runner.py 用同一个）。
    """
    return C.model_tag(C.resolve_model(backend))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="baselines/ 下的脚本名（不含 .py）")
    ap.add_argument("--split", default="M2")
    ap.add_argument("--backend", default="qwen", choices=["qwen", "deepseek"])
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--tag", default="", help="默认 <baseline>_<backend>_<split>_<版本>")
    ap.add_argument("--extra", default="", help="透传给基线脚本的额外参数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试）")
    ap.add_argument("--poll", type=int, default=20, help="进度轮询间隔（秒）")
    a = ap.parse_args()

    script = ROOT / "baselines" / f"{a.baseline}.py"
    if not script.exists():
        raise SystemExit(f"找不到 {script}")
    ids = json.load(open(ROOT / "splits" / f"{a.split}.json", encoding="utf-8"))
    if a.limit:
        ids = ids[: a.limit]
    tag = a.tag or (f"{a.baseline}_{a.backend}_{a.split}_{a.desc_version}"
                    f"_{model_tag(a.backend)}")

    # 轮转分片：保证每个分片覆盖到各个应用
    shards = [ids[i::a.shards] for i in range(a.shards)]
    shards = [s for s in shards if s]
    raw = ROOT / "baselines" / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print(f"{'='*100}")
    print(f"【并发分片】{a.baseline}  划分={a.split}({len(ids)}条)  后端={a.backend}  "
          f"描述版本={a.desc_version}  分片={len(shards)}")
    print(f"{'='*100}")
    for i, s in enumerate(shards, 1):
        print(f"  分片 {i}: {len(s):>3} 条   {s[0]} … {s[-1]}")
    print()

    procs, logs = [], []
    t0 = time.time()
    for i, s in enumerate(shards, 1):
        out = raw / f"{tag}_s{i}.jsonl"
        log = Path(f"/tmp/{tag}_s{i}.log")
        cmd = [sys.executable, "-u", str(script),
               "--ids", ",".join(s), "--split", a.split, "--backend", a.backend,
               "--desc-version", a.desc_version, "--tag", f"{tag}_s{i}"]
        if a.extra:
            cmd += a.extra.split()
        fh = open(log, "w")
        procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT)))
        logs.append((log, out))
        print(f"  启动分片 {i} (PID {procs[-1].pid})  日志 {log}", flush=True)

    # 轮询进度
    while any(p.poll() is None for p in procs):
        time.sleep(a.poll)
        done = sum(sum(1 for _ in open(o, encoding="utf-8")) if o.exists() else 0
                   for _, o in logs)
        alive = sum(1 for p in procs if p.poll() is None)
        el = time.time() - t0
        rate = done / el if el else 0
        eta = (len(ids) - done) / rate / 60 if rate else 0
        print(f"  …{done:>3}/{len(ids)} 条  存活 {alive}/{len(shards)} 分片  "
              f"{el/60:>4.1f}min  {rate*60:.1f}条/min  ETA {eta:.1f}min", flush=True)

    # 汇总
    print(f"\n{'='*100}")
    bad = 0
    for i, (log, out) in enumerate(logs, 1):
        rc = procs[i-1].returncode
        n = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0
        mark = "✅" if rc == 0 else "❌"
        print(f"  {mark} 分片 {i}: exit={rc}  {n} 条  → {out.name}")
        if rc != 0:
            bad += 1
            print(f"      日志尾部：")
            for line in Path(log).read_text(errors="replace").strip().splitlines()[-3:]:
                print(f"        {line}")
    print(f"\n  用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  汇总：python3 scripts/evaluate.py 'baselines/results/raw/{tag}_s*.jsonl' "
          f"--labels '{a.baseline}·{a.backend}·{a.split}'")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
