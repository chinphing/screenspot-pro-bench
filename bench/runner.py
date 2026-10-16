"""基线脚本共用的 CLI 驱动（让每个基线只写"机制"本身）。

标准参数：
    --ids        逗号分隔的样例 id；省略则用 --split
    --split      M1 / M2 / M3（读 splits/<split>.json）
    --desc       描述文件（默认 descriptions/<split>.json）
    --backend    deepseek / qwen
    --model      覆盖默认模型
    --tag        结果文件名前缀，输出到 baselines/results/raw/<tag>.jsonl
    --runs       重复次数（默认 1）
    --limit      只跑前 N 条（调试用）
    --dry-run    只打印不调用模型
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import DESCRIPTIONS, RESULTS, SPLITS
from . import clients as C
from . import dataset as D


def add_common_args(ap: argparse.ArgumentParser, default_max_turns: int = 5) -> None:
    ap.add_argument("--ids", default="", help="逗号分隔样例 id")
    ap.add_argument("--split", default="M1", help="M1 / M2 / M3")
    ap.add_argument("--desc-version", default="v2", help="描述版本（descriptions/<版本>/）")
    ap.add_argument("--desc", default="", help="描述文件路径（覆盖 --desc-version；默认 descriptions/<版本>/<split>.json）")
    ap.add_argument("--backend", default="qwen", choices=list(C.BACKENDS))
    ap.add_argument("--model", default="", help="覆盖默认模型")
    ap.add_argument("--tag", default="", help="结果名（默认 <脚本名>_<后端>_<split>）")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=default_max_turns)
    ap.add_argument("--reasoning-effort", default="none",
                    help="思考强度；none=关闭（⚠️ 不显式关会让 Qwen 慢数十倍）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="", help="覆盖输出目录（默认 baselines/results/raw）")


def resolve_ids(a) -> list[str]:
    if a.ids:
        return [x.strip() for x in a.ids.split(",") if x.strip()]
    f = SPLITS / f"{a.split}.json"
    if not f.exists():
        raise SystemExit(f"找不到划分文件 {f}；先用 scripts/make_splits.py 生成")
    return json.load(open(f, encoding="utf-8"))


def resolve_desc(a) -> dict:
    f = Path(a.desc) if a.desc else DESCRIPTIONS / a.desc_version / f"{a.split}.json"
    if not f.exists():
        raise SystemExit(f"找不到描述文件 {f}；"
                         f"先用 scripts/gen_descriptions.py --split {a.split} 生成")
    return json.load(open(f, encoding="utf-8"))


def run(baseline: str, run_one, a) -> int:
    """跑一个基线：run_one(rec, desc, client, a) → 结果 dict（须含 final_px）。"""
    ids = resolve_ids(a)
    desc = resolve_desc(a)
    rows = D.by_id()
    missing = [i for i in ids if i not in rows]
    if missing:
        print(f"  ⚠️ {len(missing)} 个 id 不在数据集里，已跳过：{missing[:3]}", flush=True)
    ids = [i for i in ids if i in rows]
    no_desc = [i for i in ids if i not in desc]
    if no_desc:
        print(f"  ⚠️ {len(no_desc)} 个 id 没有描述，已跳过：{no_desc[:3]}", flush=True)
    ids = [i for i in ids if i in desc]
    if a.limit:
        ids = ids[: a.limit]

    tag = a.tag or (f"{baseline}_{a.backend}_{a.split}_{a.desc_version}"
                    f"_{C.model_tag(C.resolve_model(a.backend, a.model))}")
    out_dir = Path(a.out_dir) if a.out_dir else RESULTS / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = a.model or None
    client = None if a.dry_run else C.get_client(a.backend, model)

    print(f"{'='*100}\n【{baseline}】后端={a.backend} 模型={model or '默认'} "
          f"描述版本={a.desc_version} 样例={len(ids)} 轮次上限={a.max_turns} "
          f"重复={a.runs}\n{'='*100}", flush=True)
    t_all = time.time()
    for run_i in range(1, a.runs + 1):
        suffix = f"_r{run_i}" if a.runs > 1 else ""
        out = out_dir / f"{tag}{suffix}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for n, rid in enumerate(ids, 1):
                rec = rows[rid]
                t0 = time.time()
                try:
                    r = run_one(rec, desc[rid], client, a)
                    r["status"] = r.get("status", "OK")
                except Exception as e:  # noqa: BLE001
                    r = {"status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    if not a.dry_run:
                        traceback.print_exc()
                r["id"] = rid
                r["latency_s"] = round(time.time() - t0, 2)
                gb = rec["bbox"]
                r["gt_bbox"] = gb
                r.setdefault("img_size", rec["img_size"])
                # 样例元信息（供 scripts/evaluate.py --by 分组分析）
                r["app"] = rec.get("application")
                r["ui_type"] = rec.get("ui_type")
                r["platform"] = rec.get("platform")
                r["group"] = rec.get("group")
                r["target_len"] = round(D.target_len(gb), 1)
                r["backend"] = a.backend
                r["model"] = client.model if client else None
                r["desc_version"] = a.desc_version
                fp = r.get("final_px")
                r["hit"] = bool(fp and gb[0] <= fp[0] <= gb[2] and gb[1] <= fp[1] <= gb[3])
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                mark = "✅" if r["hit"] else ("❌" if r["status"] == "OK" else "⚠️")
                print(f"  [{run_i}.{n:>3}/{len(ids)}] {mark} {rid:<28} "
                      f"{r['latency_s']:>5.1f}s  调用 {r.get('n_move', 0)}"
                      + (f"  {r['error'][:60]}" if r["status"] == "ERROR" else ""), flush=True)
        print(f"\n  结果 → {out}", flush=True)
    print(f"\n全部完成，用时 {(time.time()-t_all)/60:.1f} 分钟", flush=True)
    return 0


def load_desc_or_exit(a):
    return resolve_desc(a)
