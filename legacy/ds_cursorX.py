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
from PIL import Image, ImageDraw  # noqa: E402

import harness as H  # noqa: E402
import ds_client as dc  # noqa: E402
import qwen_client as qc  # noqa: E402
from verify_loop import draw_cursor, pad_edge  # noqa: E402

BACKEND = "deepseek"


# ── 与项目 crates/pointer-core/src/agents/computer/vision/screen_overlay.rs 一致 ──
# POINTER_CURSOR_SIZE = 32, POINTER_HOTSPOT = (3, 2)
# 顶点来自 assets/cursor-pointer.svg 的 path：M3 2 L3 21 L8 16 L11 23 L14 22 L11 15 L18 15 Z
CURSOR_SIZE = 32
CURSOR_HOTSPOT = (3, 2)
# 规则：图片长边 1000px 时，箭头高 32px（与项目 screen_overlay 的 32px 原生同量级）
CURSOR_SCALE_PER_1000 = 48.0
CURSOR_POLY = [(3, 2), (3, 21), (8, 16), (11, 23), (14, 22), (11, 15), (18, 15)]


def draw_pointer(img, x, y, scale=1.0):
    """按项目规格画鼠标指针：白填充 + 黑描边，【热点 (3,2)】精确落在 (x, y)。

    scale=1.0 → 32x32 原生像素（与项目一致）。
    放大图里先按原生尺寸画、再 4x 放大，等价于 scale=4.0。
    """
    pts = [(x + (px - CURSOR_HOTSPOT[0]) * scale,
            y + (py - CURSOR_HOTSPOT[1]) * scale) for px, py in CURSOR_POLY]
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.polygon(pts, fill=(255, 255, 255))
    lw = max(1, int(round(scale)))
    for i in range(len(pts)):
        d.line([pts[i], pts[(i + 1) % len(pts)]], fill=(0, 0, 0), width=lw)
    return img


TOOL_MOVE = {
    "type": "function",
    "function": {
        "name": "locate",
        "description": (
            "在图上框出目标元素的位置。用【你选的那张图】的归一化坐标（1-1000 的整数），"
            "报出目标的外接矩形（框住整个元素）。"
            "如果这一轮在图上找不到目标，就不要调用这个工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "enum": ["原图", "放大图"],
                    "description": "你在哪张图上框的：「放大图」= 局部放大那张（更清楚，推荐）；「原图」= 完整界面那张。",
                },
                "x0": {"type": "integer", "description": "目标框【左边缘】的归一化坐标（1-1000，1000=该图整幅宽）"},
                "y0": {"type": "integer", "description": "目标框【上边缘】的归一化坐标（1-1000，1000=该图整幅高）"},
                "x1": {"type": "integer", "description": "目标框【右边缘】的归一化坐标（1-1000）"},
                "y1": {"type": "integer", "description": "目标框【下边缘】的归一化坐标（1-1000）"},
                "reason": {"type": "string", "description": "你看到的情况（一句话）"},
            },
            "required": ["ref", "x0", "y0", "x1", "y1"],
        },
    },
}

PROMPT = """你是一个界面元素定位助手。每轮我会给你两张图。

· 【原图】：完整的界面 + 当前鼠标光标。整张界面都在这一张里。
· 【放大图】：以光标为中心的局部放大 + 同一个光标（它在正中）。
  这一张只覆盖完整界面的一小块，但清楚 {ZOOM} 倍。

⚠️ 鼠标光标是一个【白色箭头】。箭头【尖端】所指的那一点，就是光标的实际位置。

【任务】
找出下面描述的元素，并【在图上把它框出来】。

【每轮你要做的事】
在图上找到目标元素，用 locate 工具报出它的**外接矩形**（框住整个元素）。
坐标用【你选的那张图】的归一化值（1 到 1000 的整数），
  x0,y0 = 矩形左上角，x1,y1 = 矩形右下角。

我会把你报的框换算成实际位置，把鼠标移过去，然后重新给你两张新图。
你不需要一次到位，每轮只要报出【这一轮看到的框】就行。

【用哪张图】
· 优先用【放大图】——它放大 {ZOOM} 倍，小图标、小按钮看得最清楚，
  框也框得最准，此时 ref 填「放大图」。
· 只有在【放大图里看不到目标】时，才用【原图】找，此时 ref 填「原图」。

【放大图 = 放大镜】
【放大图】永远以【当前光标】为中心。所以你可以把光标当成放大镜来用：
如果目标太小、太模糊、或者你只能确定它大概在哪一片区域，
可以【先把光标移到那片区域】（哪怕还没对准），下一轮的放大图就会移到那里，
你就能看清细节了。

【框怎么画】
· 框要**完整包住**目标元素——不要只框住它的一部分。
· 元素很小（可能只有十几到几十像素），请尽量贴着它的边缘画。
· 如果这一轮在图上找不到目标，就不要调用工具，直接结束。

【什么时候结束】
当你看到【光标尖端已经在目标元素里面】时，就不要调用工具，直接结束。

【目标元素描述】
{desc}
"""


def _desc_lines(d):
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}")


def run_one(rid, src, d, *, max_turns=5, crop_px=200, zoom=4,
            first_crop_px=None, first_zoom=None, ladder=None, cursor_px=48.0,
            keep_only_latest=True,
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

    def render_pair(nx, ny, _cp, _zm):
        """返回 (完整图+光标, 裁片放大图+光标)，以及裁片信息。"""
        # ① 完整原图（已缩放）+ 光标
        full = full_base.copy()
        cx, cy = nx / 1000 * W0, ny / 1000 * H0
        full = draw_pointer(full, cx * FS, cy * FS,
                            scale=max(full.size) * CURSOR_SCALE_PER_1000 / 1000.0 / CURSOR_SIZE)

        # ② 裁片：从【干净原图】裁 200px，再放大 zoom 倍，最后画光标
        half = _cp // 2
        pad = half + 8
        big = pad_edge(orig, pad)
        X, Y = cx + pad, cy + pad
        x0, y0 = int(X - half), int(Y - half)
        region = big.crop((x0, y0, x0 + _cp, y0 + _cp))
        sw = sh = int(_cp * _zm)
        crop = region.resize((sw, sh), Image.LANCZOS)
        crop = draw_pointer(crop, (X - x0) * _zm, (Y - y0) * _zm,
                            scale=max(crop.size) * CURSOR_SCALE_PER_1000 / 1000.0 / CURSOR_SIZE)
        # 裁片在【原图】里的归一化范围（可能略超出 0-1000，因为贴边时有 8px 外扩）
        bx0, by0 = x0 - pad, y0 - pad
        nr = [round(bx0 / W0 * 1000), round(by0 / H0 * 1000),
              round((bx0 + _cp) / W0 * 1000), round((by0 + _cp) / H0 * 1000)]
        return full, crop, nr

    nx, ny = start
    cx, cy = nx / 1000 * W0, ny / 1000 * H0
    err0 = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5

    LAD = ladder or [(first_crop_px or crop_px, first_zoom or zoom)] + [(crop_px, zoom)] * (max_turns - 1)
    def _stage(t):
        return LAD[min(t - 1, len(LAD) - 1)]
    CUR_CP, CUR_ZM = _stage(1)
    full, crop, cbox = render_pair(nx, ny, CUR_CP, CUR_ZM)
    messages = [{"role": "user", "content": [
        {"type": "text", "text":
            f"【原图】完整界面（原尺寸 {W0} x {H0} 像素，此处缩放到 {full.width} x {full.height}），光标当前在 "
            f"({nx},{ny})。"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}},
        {"type": "text", "text":
            f"【放大图】以光标为中心裁出的 {CUR_CP}px 局部，放大 {CUR_ZM} 倍"
            f"（{crop.width} x {crop.height} 像素），光标尖端在正中。"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
        {"type": "text", "text": PROMPT.format(ZOOM=CUR_ZM, desc=_desc_lines(d))},
    ]}]

    trace = [{"turn": 0, "action": "init", "norm": [nx, ny],
              "px": [round(cx), round(cy)], "err": round(err0)}]
    t0 = time.time()
    n_move = 0

    for turn in range(1, max_turns + 1):
        if turn >= 2 and CUR_CP != crop_px:
            CUR_CP, CUR_ZM = crop_px, zoom
            full, crop, cbox = render_pair(nx, ny, CUR_CP, CUR_ZM)
            messages[-1]["content"][2]["text"] = (
                f"【放大图】以光标为中心裁出的 {CUR_CP}px 局部，放大 {CUR_ZM} 倍"
                f"（{crop.width} x {crop.height} 像素），光标尖端在正中。")
            messages[-1]["content"][3]["image_url"]["url"] = (
                f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}")
            messages[-1]["content"][4]["text"] = PROMPT.format(ZOOM=CUR_ZM, desc=_desc_lines(d))
        _mt = max_tokens if reasoning_effort in (None, "", "none") else max(max_tokens, 8000)
        if BACKEND == "qwen":
            extra = {"tools": [TOOL_MOVE], "tool_choice": "auto"}
            if reasoning_effort in (None, "", "none"):
                extra["reasoning_effort"] = "none"     # 显式关闭 Qwen 思考
            resp = qc.chat(messages, max_tokens=_mt, extra=extra)
        else:
            extra = {"reasoning_effort": reasoning_effort,
                     "tools": [TOOL_MOVE], "tool_choice": "auto"}
            resp, _ = dc.chat(messages, max_tokens=_mt, tries=tries, extra=extra)
        _ch = resp["choices"][0]
        msg = _ch["message"]
        _finish = _ch.get("finish_reason")
        _think = (msg.get("reasoning_content") or "").strip()
        if _finish == "length":
            print(f"      ⚠️ 第{turn}轮 被 max_tokens 截断（thinking {len(_think)} 字符）", flush=True)
        tcs = msg.get("tool_calls") or []

        if not tcs:
            trace.append({"turn": turn, "action": "stop",
                          "text": (msg.get("content") or "").strip()[:150],
                          "think": _think, "finish": _finish})
            break

        tc = tcs[0]
        try:
            a = json.loads(tc["function"]["arguments"] or "{}")
        except Exception:  # noqa: BLE001
            a = {}
        try:
            ref = str(a.get("ref", "放大图"))
            x0 = int(a.get("x0", 0)); y0 = int(a.get("y0", 0))
            x1 = int(a.get("x1", 0)); y1 = int(a.get("y1", 0))
        except Exception:  # noqa: BLE001
            ref, x0, y0, x1, y1 = "放大图", 0, 0, 0, 0
        if ref not in ("原图", "放大图"):
            ref = "放大图"
        # 该图覆盖的原图归一化范围
        gx0, gy0, gx1, gy1 = (0.0, 0.0, 1000.0, 1000.0) if ref == "原图" else tuple(cbox)
        # 框换算到原图归一化
        bx0 = gx0 + x0 / 1000.0 * (gx1 - gx0); bx1 = gx0 + x1 / 1000.0 * (gx1 - gx0)
        by0 = gy0 + y0 / 1000.0 * (gy1 - gy0); by1 = gy0 + y1 / 1000.0 * (gy1 - gy0)
        tx = int(round((bx0 + bx1) / 2)); ty = int(round((by0 + by1) / 2))
        tx = max(0, min(1000, tx)); ty = max(0, min(1000, ty))
        _same = (tx == nx and ty == ny)
        nx, ny = tx, ty
        if not _same:
            n_move += 1
        px, py = nx / 1000 * W0, ny / 1000 * H0
        err = ((px - gcx) ** 2 + (py - gcy) ** 2) ** 0.5
        trace.append({"turn": turn, "action": "move", "norm": [nx, ny],
                      "px": [round(px), round(py)], "err": round(err),
                      "ref": ref, "bbox_raw": [x0, y0, x1, y1],
                      "bbox_orig": [round(bx0), round(by0), round(bx1), round(by1)],
                      "reason": (a.get("reason") or "")[:70],
                      "think": _think, "finish": _finish})

        # ★ 只保留最新一轮的两张图：把历史里的 image_url 换成文字占位
        if keep_only_latest:
            for _m in messages:
                _c = _m.get("content")
                if isinstance(_c, list):
                    for _b in _c:
                        if isinstance(_b, dict) and _b.get("type") == "image_url":
                            _b.clear()
                            _b.update({"type": "text", "text": "（此处是上一轮的图，已省略）"})

        full, crop, cbox = render_pair(nx, ny, CUR_CP, CUR_ZM)
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
                f"【放大图】以光标为中心裁出的 {CUR_CP}px 局部，放大 {CUR_ZM} 倍，光标尖端在正中。"},
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
            "think_chars": sum(len(t.get("think") or "") for t in trace),
            "max_turns": max_turns, "crop_px": crop_px, "zoom": zoom,
            "first_crop_px": first_crop_px, "first_zoom": first_zoom, "ladder": LAD,
            "latency_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--crop-px", type=int, default=200)
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--first-crop-px", type=int, default=None)
    ap.add_argument("--first-zoom", type=int, default=None)
    ap.add_argument("--ladder", default=None, help="如 600x2,400x3,200x4")
    ap.add_argument("--cursor-px", type=float, default=48.0,
                    help="光标宽度 = 图像长边 × 该值/1000")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--backend", choices=["deepseek", "qwen"], default="deepseek")
    ap.add_argument("--start", default="500,500", help="光标起始位置（原图归一化）")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--tries", type=int, default=10)
    args = ap.parse_args()
    global BACKEND
    BACKEND = args.backend

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    sx, sy = (int(v) for v in args.start.split(","))
    out = os.path.join("results", "l1", f"cursorX_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"鼠标移动智能体  最多 {args.max_turns} 轮  裁片 {args.crop_px}px×{args.zoom}  "
          f"起点 ({sx},{sy})  思考={args.reasoning_effort}")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(args.ids.split(","), 1):
        try:
            rec = run_one(rid, rows[rid], DESC[rid], max_turns=args.max_turns,
                          crop_px=args.crop_px, zoom=args.zoom,
                          first_crop_px=args.first_crop_px, first_zoom=args.first_zoom,
                          ladder=([tuple(int(v) for v in p.split("x")) for p in args.ladder.split(",")]
                                  if args.ladder else None),
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
