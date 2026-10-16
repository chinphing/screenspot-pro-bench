#!/usr/bin/env python3
"""B1 —— 单图直出（最简基线，无循环）。

机制：整图缩放到 <2.5Mpx → 短提示要 bbox → 一次调用结束。
用途：作为「零智能体」参照；后续基线相对它提升多少，就是机制的价值。

    python3 baselines/b1_direct.py --split M1 --backend qwen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from bench import dataset as D  # noqa: E402
from bench import runner as R  # noqa: E402
from bench import vision as V  # noqa: E402
from bench.compat.verify_loop import L1_PROMPT  # noqa: E402  ← legacy 原文，保证 B1 == B3 阶段1

NAME = "b1_direct"


def run_one(rec, d, client, a):
    img = Image.open(D.fetch_image(rec["img_filename"])).convert("RGB")
    W, H = rec["img_size"]
    l1_px = 1.3e6 if a.backend == "deepseek" else D.TARGET_PIXELS
    small = V.fit(img, l1_px)
    # ⚠️ 用 legacy L1_PROMPT 原文 + 与 ZF 完全相同的字段填充方式，
    #    使 B1 严格等于 B3 的阶段 1（否则「B1 vs B3 = 循环的价值」不成立）
    prompt = L1_PROMPT.format(env=d.get("env", ""), text=d.get("text", ""),
                              **{k: v for k, v in d.items() if k not in ("env", "text")})
    txt = client.ask_image(small, prompt, max_tokens=512,
                           reasoning_effort=a.reasoning_effort)
    bb = D.parse_bbox(txt)
    out = {"answer": (txt or "").strip()[:200], "n_move": 0, "sent_size": list(small.size)}
    if not bb:
        out["status"] = "NO_OUTPUT"
        out["final_px"] = None
        return out
    px = D.norm_to_orig(bb, W, H)
    out["final_bbox"] = [round(v, 1) for v in px]
    out["final_px"] = [round((px[0] + px[2]) / 2, 1), round((px[1] + px[3]) / 2, 1)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_common_args(ap, default_max_turns=1)
    a = ap.parse_args()
    return R.run(NAME, run_one, a)


if __name__ == "__main__":
    raise SystemExit(main())
