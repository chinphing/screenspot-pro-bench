"""鼠标移动定位智能体：让模型通过「移动鼠标」来定位目标。

每轮给模型两张图：
  ① 完整原图 + 画在其上的鼠标光标
  ② 以光标为中心、从【干净原图】裁 200px 再放大 4× 的图（再画上光标）
     ⚠️ 裁片是从干净原图裁的，不是从"已带光标的原图"放大来的
     （否则光标会被一起放大变糊）

模型只有一个工具 move(x, y)：
  · 调用它 → 光标移到新位置，进入下一轮
  · 不调用 → 表示"光标已经在目标上了"，结束
鼠标的最终位置就是答案。

用法：
  python3 ds_cursor.py --ids a,b,c --desc-file xxx.json \
      --tag t1 [--max-turns 5] [--crop-px 200] [--zoom 4] \
      [--cursor-px 24] [--reasoning-effort none] [--start 500,500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image  # noqa: E402

import harness as H  # noqa: E402
import ds_client as dc  # noqa: E402
from verify_loop import draw_cursor, pad_edge  # noqa: E402

TOOL_MOVE = {
    "type": "function",
    "function": {
        "name": "move",
        "description": (
            "把鼠标光标移动一下。你只需要回答【目标相对光标偏了多少】，"
            "不用计算绝对坐标，我会自己换算。"
            "当你觉得光标已经在目标元素上时，不要调用任何工具，直接结束。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "enum": ["原图", "放大图"],
                    "description": (
                        "你是在哪张图上量出这个偏移的："
                        "「原图」= 完整界面那张；「放大图」= 局部放大那张。"
                        "在放大图上量得更准，推荐用「放大图」。"
                    ),
                },
                "x": {
                    "type": "integer",
                    "description": (
                        "水平偏移量。用【你选的那张图】的归一化单位："
                        "1000 = 那张图的整幅宽度。"
                        "正数=目标在光标右侧，负数=在左侧。"
                    ),
                },
                "y": {
                    "type": "integer",
                    "description": (
                        "垂直偏移量。同样用该图的归一化单位：1000 = 那张图的整幅高度。"
                        "正数=目标在光标下方，负数=在上方。"
                    ),
                },
                "reason": {"type": "string", "description": "你看到的情况（一句话）"},
            },
            "required": ["ref", "x", "y"],
        },
    },
}

PROMPT = """你是一个界面元素定位助手。我会每轮给你两张图：

· 【原图】：完整的界面 + 当前光标（红色方块）。整张界面都在这一张里。
· 【放大图】：以光标为中心的局部放大 + 同一个光标（它在正中）。
  这一张只覆盖完整界面的一小块，但清楚 {ZOOM} 倍。

【任务】
找出下面描述的元素，把光标【移动到该元素上】。

【你要做的判断】
看一眼光标现在压在什么位置，再看目标元素在哪，然后告诉我：
  · 目标相对光标，往右偏多少、往下偏多少。
你不用算绝对坐标，也不用做任何换算——只要按你看到的说就行。

【用哪张图量偏移】
· 优先用【放大图】：它放大 {ZOOM} 倍，小图标、小按钮看得最清楚，偏移量也量得最准。
  此时 ref 填「放大图」。
· 只有在【放大图里看不到目标】时，才用【原图】估计方位，此时 ref 填「原图」。

【偏移量的单位】
用你选的那张图的归一化单位：
  · ref=「放大图」时：1000 = 放大图的整幅宽度/高度
  · ref=「原图」时：1000 = 完整界面的整幅宽度/高度
正负号：x 正 = 目标在光标右侧，x 负 = 在左侧；y 正 = 在下方，y 负 = 在上方。

【举例】
· 在放大图里看到目标在光标右边一点点（约占图宽 3%）→ ref=放大图, x=30, y=0
· 在放大图里看到目标在光标正下方半张图处 → ref=放大图, x=0, y=500
· 在原图里看到目标在光标左上方很远处 → ref=原图, x=-400, y=-350

【什么时候结束】
当你认为光标已经落在目标元素上时，【不要调用任何工具】，直接结束。

【目标元素描述】
{desc}"""


def _desc_lines(d):
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}")


def run_one(rid, src, d, *, max_turns=5, crop_px=200, zoom=4, cursor_px=48.0,
            reasoning_effort="none", start=(500, 500), max_tokens=1500, tries=10,
            full_px=1.7e6):
    orig = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
    W0, H0 = orig.size
    # 完整图按 token 上限缩放（DeepSeek 1008 token ≈ 1.7Mpx，再大也没用）
    _fw, _fh = orig.size
    _k = min(1.0, (full_px / (_fw * _fh)) ** 0.5)
    full_base = (orig if _k >= 1.0 else
                 orig.resize((max(1, round(_fw * _k)), max(1, round(_fh * _k))),
                             Image.LANCZOS))
    FS = _k
    gt = src["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2

    def render_pair(nx, ny):
        """返回 (完整图+光标, 裁片放大图+光标)，以及裁片信息。"""
        # ① 完整原图（已缩放）+ 光标
        full = full_base.copy()
        cw_full = max(full.size) * cursor_px / 1000.0
        cx, cy = nx / 1000 * W0, ny / 1000 * H0
        full = draw_cursor(full, cx * FS, cy * FS, cw_full)

        # ② 裁片：从【干净原图】裁 200px，再放大 zoom 倍，最后画光标
        half = crop_px // 2
        pad = half + 8
        big = pad_edge(orig, pad)
        X, Y = cx + pad, cy + pad
        x0, y0 = int(X - half), int(Y - half)
        region = big.crop((x0, y0, x0 + crop_px, y0 + crop_px))
        sw = sh = int(crop_px * zoom)
        crop = region.resize((sw, sh), Image.LANCZOS)
        cw_crop = max(sw, sh) * cursor_px / 1000.0
        crop = draw_cursor(crop, (X - x0) * zoom, (Y - y0) * zoom, cw_crop)
        # 裁片在【原图】里的归一化范围（可能略超出 0-1000，因为贴边时有 8px 外扩）
        bx0, by0 = x0 - pad, y0 - pad
        nr = [round(bx0 / W0 * 1000), round(by0 / H0 * 1000),
              round((bx0 + crop_px) / W0 * 1000), round((by0 + crop_px) / H0 * 1000)]
        return full, crop, nr

    nx, ny = start
    cx, cy = nx / 1000 * W0, ny / 1000 * H0
    err0 = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5

    full, crop, cbox = render_pair(nx, ny)
    messages = [{"role": "user", "content": [
        {"type": "text", "text":
            f"【原图】完整界面（原尺寸 {W0} x {H0} 像素，此处缩放到 {full.width} x {full.height}），光标当前在归一化位置 "
            f"({nx},{ny})，即像素 ({cx:.0f},{cy:.0f})。"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}},
        {"type": "text", "text":
            f"【放大图】以光标为中心裁出的 {crop_px}px 局部，放大 {zoom} 倍"
            f"（{crop.width} x {crop.height} 像素），光标在正中。"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
        {"type": "text", "text": PROMPT.format(ZOOM=zoom, desc=_desc_lines(d))},
    ]}]

    trace = [{"turn": 0, "action": "init", "norm": [nx, ny],
              "px": [round(cx), round(cy)], "err": round(err0)}]
    t0 = time.time()
    n_move = 0

    for turn in range(1, max_turns + 1):
        extra = {"reasoning_effort": reasoning_effort,
                 "tools": [TOOL_MOVE], "tool_choice": "auto"}
        resp, _ = dc.chat(messages, max_tokens=max_tokens, tries=tries, extra=extra)
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []

        if not tcs:
            trace.append({"turn": turn, "action": "stop",
                          "text": (msg.get("content") or "").strip()[:150]})
            break

        tc = tcs[0]
        try:
            a = json.loads(tc["function"]["arguments"] or "{}")
        except Exception:  # noqa: BLE001
            a = {}
        try:
            ref = str(a.get("ref", "放大图")); dx = int(a.get("x", 0)); dy = int(a.get("y", 0))
        except Exception:  # noqa: BLE001
            ref, dx, dy = "放大图", 0, 0
        if ref not in ("原图", "放大图"):
            ref = "放大图"
        # 脚本按参考图换算：把「该图内的偏移」换成「原图归一化偏移」
        if ref == "原图":
            ndx, ndy = dx, dy                       # 原图：归一化单位 = 原图归一化单位
        else:
            ndx = dx * crop_px / W0 * 1.0           # 放大图覆盖 crop_px 原图像素
            ndy = dy * crop_px / H0 * 1.0
        tx = int(round(nx + ndx)); ty = int(round(ny + ndy))
        tx = max(0, min(1000, tx)); ty = max(0, min(1000, ty))
        if tx == nx and ty == ny:
            trace.append({"turn": turn, "action": "move_same", "norm": [tx, ty],
                          "ref": ref, "delta_raw": [dx, dy]})
            break
        nx, ny = tx, ty
        n_move += 1
        px, py = nx / 1000 * W0, ny / 1000 * H0
        err = ((px - gcx) ** 2 + (py - gcy) ** 2) ** 0.5
        trace.append({"turn": turn, "action": "move", "norm": [nx, ny],
                      "px": [round(px), round(py)], "err": round(err),
                      "ref": ref, "delta_raw": [dx, dy],
                      "delta_norm": [round(ndx, 2), round(ndy, 2)],
                      "reason": (a.get("reason") or "")[:70]})

        full, crop, cbox = render_pair(nx, ny)
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tcs})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": f"光标已移动到归一化 ({nx},{ny})。"})
        messages.append({"role": "user", "content": [
            {"type": "text", "text":
                f"【原图】完整界面，光标现在在归一化 ({nx},{ny})。"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}},
            {"type": "text", "text":
                f"【放大图】以光标为中心裁出的 {crop_px}px 局部，放大 {zoom} 倍，光标在正中。"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
            {"type": "text", "text":
                "请判断：光标是否已经落在目标元素上？"
                "是 → 不要调用工具，直接结束；否 → 调用 move(x, y) 调整。"},
        ]})

    fx, fy = nx / 1000 * W0, ny / 1000 * H0
    final_err = ((fx - gcx) ** 2 + (fy - gcy) ** 2) ** 0.5
    return {"id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
            "start_norm": list(start), "start_err": round(err0),
            "final_norm": [nx, ny], "final_px": [round(fx), round(fy)],
            "final_err": round(final_err, 1),
            "hit": gt[0] <= fx <= gt[2] and gt[1] <= fy <= gt[3],
            "n_move": n_move, "turns_used": len(trace) - 1,
            "trace": trace, "reasoning_effort": reasoning_effort,
            "max_turns": max_turns, "crop_px": crop_px, "zoom": zoom,
            "latency_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--crop-px", type=int, default=200)
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--cursor-px", type=float, default=48.0,
                    help="光标宽度 = 图像长边 × 该值/1000")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--start", default="500,500", help="光标起始位置（原图归一化）")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--tries", type=int, default=10)
    args = ap.parse_args()

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    sx, sy = (int(v) for v in args.start.split(","))
    out = os.path.join("results", "l1", f"cursorE_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"鼠标移动智能体  最多 {args.max_turns} 轮  裁片 {args.crop_px}px×{args.zoom}  "
          f"起点 ({sx},{sy})  思考={args.reasoning_effort}")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(args.ids.split(","), 1):
        try:
            rec = run_one(rid, rows[rid], DESC[rid], max_turns=args.max_turns,
                          crop_px=args.crop_px, zoom=args.zoom,
                          cursor_px=args.cursor_px,
                          reasoning_effort=args.reasoning_effort,
                          start=(sx, sy), max_tokens=args.max_tokens,
                          tries=args.tries)
        except Exception as e:  # noqa: BLE001
            rec = {"id": rid, "status": "ERROR", "error": str(e)[:200]}
        recs.append(rec)
        e = rec.get("final_err")
        mv = "".join({"move": "M", "stop": "S", "init": "·",
                      "move_same": "="}.get(t["action"], "?") for t in rec.get("trace", []))
        print(f"  [{k:>2}] {rid:<26} 起点{rec.get('start_err','?'):>5}px → "
              f"终点{(round(e) if e is not None else '?'):>5}px  轨迹={mv:<7}"
              f"{rec.get('latency_s',0):>5.1f}s  {'✅' if rec.get('hit') else '❌'}", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")


if __name__ == "__main__":
    main()
