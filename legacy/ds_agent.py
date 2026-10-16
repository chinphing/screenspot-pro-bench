"""DeepSeek 多轮工具调用定位智能体。

思路：DeepSeek 单图 token 上限只有 1008（约 1300×1300），小目标被缩得看不清；
但单轮只要 ~0.7 秒，所以允许它「多次处理」——自己决定要不要放大再看。

流程：
  第 1 轮：送完整原图
  每一轮模型二选一：
    A. 调用 zoom_in(x1,y1,x2,y2) —— 说"这块看不清，放大给我"
       → 该区域按原分辨率裁出（必要时缩到 ≤px），追加进对话，进入下一轮
    B. 直接输出 bbox 数组 —— 说"我看清了"
       → 结束
  上限 max_turns 轮；用尽仍未给 bbox → 失败，但交付已尝试的路径摘要。

坐标一律用【原图归一化 0-1000】。

用法：
  python3 ds_agent.py --ids a,b,c --desc-file xxx.json --tag t1 \
      [--reasoning-effort none|low|high|max] [--max-turns 5] [--px 1.7e6]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image  # noqa: E402

import harness as H  # noqa: E402
import ds_client as dc  # noqa: E402

NUM4 = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

TOOL_ZOOM = {
    "type": "function",
    "function": {
        "name": "zoom_in",
        "description": (
            "放大查看原图中的某个区域。当你觉得目标太小、细节看不清、或者无法确定"
            "目标到底是哪一个相似元素时，调用本工具放大该区域再看一次。"
            "区域坐标用【原图归一化 0-1000】：x1,y1 是左上角，x2,y2 是右下角。"
            "区域不要太小（建议至少覆盖原图 15% 的宽度），否则会失去上下文。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x1": {"type": "integer", "description": "区域左边界（0-1000）"},
                "y1": {"type": "integer", "description": "区域上边界（0-1000）"},
                "x2": {"type": "integer", "description": "区域右边界（0-1000）"},
                "y2": {"type": "integer", "description": "区域下边界（0-1000）"},
                "reason": {"type": "string", "description": "为什么需要放大这块区域（一句话）"},
            },
            "required": ["x1", "y1", "x2", "y2"],
        },
    },
}

PROMPT = """你是一个界面元素定位助手。第 1 张图是完整的界面截图（{W0} x {H0} 像素）。

【任务】
找出下面描述的元素，给出它在【完整原图】里的 bbox。

【你有两种做法，每轮二选一】
· 做法 A —— 看不清就放大：调用 zoom_in(x1,y1,x2,y2) 工具，
  告诉我你想放大原图的哪一块（归一化 0-1000）。我会把那一块按原分辨率给你看。
  当你觉得目标太小、或者分不清哪个是目标（有多个相似元素）时，用这个方法。
· 做法 B —— 看清了就直接答：直接输出 bbox 数组，格式 [x1,y1,x2,y2]（归一化 0-1000），
  只输出这个数组，不要别的文字。

【关于 bbox 的要求】
· 框必须【完整包住目标元素】——上、下、左、右四个边界都要贴着元素的外沿，
  既不要把元素切掉一部分，也不要留出大片空白。
· 不要把框画得比元素小（那会导致点击落空），也不要把框画得过大（那会框进旁边的元素）。
· 先想清楚元素的外沿在哪里，再写四个数字。

【限制】
· 最多只能放大 {MAXT} 次。
· 每次放大都会追加到对话里（之前的图仍然可见）。
· 无论你在哪张放大图里看到目标，最终答案都必须是【第 1 张完整原图】的归一化坐标。

【目标元素描述】
{desc}"""


def _to_long_side(img, target):
    """把图缩放到长边 = target 像素（小图放大、大图缩小）。"""
    w, h = img.size
    if max(w, h) == target:
        return img
    k = target / max(w, h)
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def _fit(img, target_px):
    w, h = img.size
    if w * h <= target_px:
        return img
    k = (target_px / (w * h)) ** 0.5
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def _desc_lines(d):
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}")


def _parse_bbox_from_msg(msg):
    """从 assistant 消息里提取 bbox（优先 content，其次 reasoning 里的最后一段）"""
    txt = msg.get("content") or ""
    m = NUM4.findall(txt)
    if m:
        return [int(v) for v in m[-1]]
    return None


def run_one(rid, src, d, *, max_turns=5, px=1.7e6, crop_side=1200,
            reasoning_effort="none", max_tokens=1200, tries=10):
    img = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
    W0, H0 = img.size
    gt = src["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2

    full = _fit(img, px)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}},
            {"type": "text",
             "text": PROMPT.format(W0=W0, H0=H0, MAXT=max_turns - 1,
                                   desc=_desc_lines(d))},
        ],
    }]

    trace = []            # 每轮的简要记录
    final_bbox = None
    t0 = time.time()

    for turn in range(1, max_turns + 1):
        extra = {"tools": [TOOL_ZOOM], "tool_choice": "auto"}
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        resp, att = dc.chat(messages, max_tokens=max_tokens, tries=tries, extra=extra)
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []

        if not tcs:
            # 做法 B：直接输出 bbox
            nb = _parse_bbox_from_msg(msg)
            trace.append({"turn": turn, "action": "answer",
                          "text": (msg.get("content") or "").strip()[:200]})
            if nb:
                final_bbox = [nb[0] / 1000 * W0, nb[1] / 1000 * H0,
                              nb[2] / 1000 * W0, nb[3] / 1000 * H0]
            break

        # 做法 A：调用 zoom_in
        tc = tcs[0]
        try:
            a = json.loads(tc["function"]["arguments"])
        except Exception:  # noqa: BLE001
            a = {}
        try:
            x1, y1, x2, y2 = int(a["x1"]), int(a["y1"]), int(a["x2"]), int(a["y2"])
        except Exception:  # noqa: BLE001
            trace.append({"turn": turn, "action": "zoom", "error": "参数解析失败",
                          "raw": str(a)[:120]})
            break
        x1, x2 = sorted((max(0, min(1000, x1)), max(0, min(1000, x2))))
        y1, y2 = sorted((max(0, min(1000, y1)), max(0, min(1000, y2))))
        if x2 - x1 < 10 or y2 - y1 < 10:
            x1, y1, x2, y2 = 0, 0, 1000, 1000     # 区域过小 → 给整图
            note = "（请求区域过小，已改为整图）"
        else:
            note = ""

        px0, py0 = int(x1 / 1000 * W0), int(y1 / 1000 * H0)
        px1, py1 = int(x2 / 1000 * W0), int(y2 / 1000 * H0)
        region = img.crop((px0, py0, px1, py1))
        sent = _to_long_side(region, crop_side)
        trace.append({"turn": turn, "action": "zoom", "norm": [x1, y1, x2, y2],
                      "pix": [px0, py0, px1, py1], "sent": list(sent.size),
                      "reason": (a.get("reason") or "")[:80]})

        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tcs})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": f"已放大区域 [{x1},{y1},{x2},{y2}]（归一化）{note}。"})
        # 图片走 user 消息（官方文档：图片只支持出现在 user 消息里）
        messages.append({"role": "user", "content": [
            {"type": "text",
             "text": f"这是你请求放大的区域 [{x1},{y1},{x2},{y2}]（归一化坐标），"
                     f"按原分辨率显示，尺寸 {sent.width} x {sent.height} 像素。"
                     f"请继续：还看不清就再调 zoom_in，看清了就输出完整原图的归一化 bbox。"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(sent)}"}},
        ]})

    dt = time.time() - t0
    err = None
    if final_bbox is not None:
        err = (((final_bbox[0] + final_bbox[2]) / 2 - gcx) ** 2
               + ((final_bbox[1] + final_bbox[3]) / 2 - gcy) ** 2) ** 0.5
    return {"id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
            "status": "OK" if final_bbox is not None else "NO_ANSWER",
            "final_bbox": final_bbox, "final_err": err,
            "turns_used": len(trace), "trace": trace,
            "reasoning_effort": reasoning_effort, "max_turns": max_turns,
            "latency_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--px", type=float, default=1.7e6,
                    help="第 1 张完整图的送图上限（像素）")
    ap.add_argument("--crop-side", type=int, default=1200,
                    help="放大裁片统一缩放到长边 = 该值（像素）")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--tries", type=int, default=10)
    args = ap.parse_args()

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    out = os.path.join("results", "l1", f"agent_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"DeepSeek 多轮智能体  思考={args.reasoning_effort}  最多 {args.max_turns} 轮  "
          f"整图上限 {args.px/1e6:.2f}Mpx  裁片长边固定 {args.crop_side}px")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(args.ids.split(","), 1):
        t0 = time.time()
        try:
            rec = run_one(rid, rows[rid], DESC[rid], max_turns=args.max_turns,
                          px=args.px, crop_side=args.crop_side,
                          reasoning_effort=args.reasoning_effort,
                          max_tokens=args.max_tokens, tries=args.tries)
        except Exception as e:  # noqa: BLE001
            rec = {"id": rid, "status": "ERROR", "error": str(e)[:200],
                   "latency_s": round(time.time() - t0, 1)}
        recs.append(rec)
        e = rec.get("final_err")
        zooms = [t for t in rec.get("trace", []) if t.get("action") == "zoom"]
        es = "  无" if e is None else f"{round(e):>4}px"
        print(f"  [{k:>2}] {rid:<26} 误差={es}  放大{len(zooms)}次  "
              f"{rec['latency_s']:>5.1f}s  "
              f"{'✅' if (e is not None and e <= 50) else '❌'}", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")


if __name__ == "__main__":
    main()
