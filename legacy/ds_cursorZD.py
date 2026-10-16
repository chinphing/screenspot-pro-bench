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
from verify_loop import draw_cursor, pad_edge, L1_PROMPT  # noqa: E402

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


def draw_box(img, x0, y0, x1, y1, color=(0, 90, 200), width=3):
    """画上一轮的方框（蓝色实线）。"""
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(round(width))))
    return img


TOOL_DONE = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "确认：当前图上画的蓝框【已经框住】了目标元素，任务完成，结束。",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "一句话说明你确认的理由"},
            },
        },
    },
}


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

· 【原图】：完整的界面。整张界面都在这一张里。
· 【放大图】：以【你上一轮报的框】为中心的局部放大。
  这一张只覆盖完整界面的一小块，但清楚 {ZOOM} 倍。

【图上画的蓝框】
图上会画出【你上一轮报的那个框】（蓝色矩形）。
每一轮你要做的第一件事，就是判断：

    【这个蓝框有没有框住目标元素？】

· 【每一轮都必须用 locate 报一个框】——即使只差一点点，也要报出来。
· 只有当这一轮报的框和上一轮【几乎完全一样（误差 1-2 个归一化单位以内）】时，
  才调用 done() 结束。

⚠️ 这是一次"逐轮收紧"的过程：每一轮你都应该比上一轮更贴近目标边缘。
   不要因为"框已经差不多"就提前结束——请把框再收紧一轮。

【已有蓝框】
图上已经画好了【你上一轮报的那个框】。直接判断它有没有框住目标即可。

【框怎么画】
用【你选的那张图】的归一化坐标（1 到 1000 的整数）：
  x0, y0 = 框的左上角；x1, y1 = 框的右下角。
  1000 代表你选的那张图的整幅宽度 / 高度。

· 优先用【放大图】——它放大 {ZOOM} 倍，小图标、小按钮看得最清楚，框也框得最准，
  此时 ref 填「放大图」。
· 只有在【放大图里看不到目标】时，才用【原图】找，此时 ref 填「原图」。

【框的要求】
· 框要【完整包住】目标元素——不要只框住它的一部分。
· 元素很小（可能只有十几到几十像素），请尽量贴着它的边缘画。
· 但框大一点也没关系，只要目标整个在框里就行。

【目标元素描述】
{desc}
"""


PROMPT2 = PROMPT


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

    def render_pair(box, _cp, _zm):
        """box = 上一轮报的框（原图归一化 [x0,y0,x1,y1]）或 None。
        返回 (完整图+蓝框, 裁片放大图+蓝框) 以及裁片范围。"""
        # ① 完整原图（已缩放）+ 蓝框
        full = full_base.copy()
        if box:
            full = draw_box(full,
                            box[0] / 1000 * W0 * FS, box[1] / 1000 * H0 * FS,
                            box[2] / 1000 * W0 * FS, box[3] / 1000 * H0 * FS)
            cx, cy = (box[0] + box[2]) / 2 / 1000 * W0, (box[1] + box[3]) / 2 / 1000 * H0
        else:
            cx, cy = W0 / 2, H0 / 2

        # ② 裁片：从【干净原图】裁 _cp px，再放大 _zm 倍，最后画蓝框
        half = _cp // 2
        pad = half + 8
        big = pad_edge(orig, pad)
        X, Y = cx + pad, cy + pad
        x0, y0 = int(X - half), int(Y - half)
        region = big.crop((x0, y0, x0 + _cp, y0 + _cp))
        sw = sh = int(_cp * _zm)
        crop = region.resize((sw, sh), Image.LANCZOS)
        if box:
            crop = draw_box(crop,
                            (box[0] / 1000 * W0 + pad - x0) * _zm,
                            (box[1] / 1000 * H0 + pad - y0) * _zm,
                            (box[2] / 1000 * W0 + pad - x0) * _zm,
                            (box[3] / 1000 * H0 + pad - y0) * _zm)
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
    # ── 阶段1：L1 短提示（单图 + 纯文本）拿粗定位 ──────────────────
    _l1p = L1_PROMPT.format(
        env=d.get("env", ""), text=d.get("text", ""),
        **{k: v for k, v in d.items() if k not in ("env", "text")})
    _t1, _, _, _ = dc.ask_image(full_base, _l1p, max_tokens=512,
                                reasoning_effort=reasoning_effort)
    _m1 = H.NUM4.findall(_t1)
    cur_box = [float(v) for v in _m1[-1]] if _m1 else None
    if cur_box:
        nx = int(round((cur_box[0] + cur_box[2]) / 2))
        ny = int(round((cur_box[1] + cur_box[3]) / 2))
    full, crop, cbox = render_pair(cur_box, CUR_CP, CUR_ZM)
    messages = [{"role": "user", "content": [
        {"type": "text", "text":
            f"【原图】完整界面（原尺寸 {W0} x {H0} 像素，此处缩放到 {full.width} x {full.height}）。"
            f"蓝框是你上一轮报的框。"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}},
        {"type": "text", "text":
            f"【放大图】以蓝框为中心裁出的 {CUR_CP}px 局部，放大 {CUR_ZM} 倍"
            f"（{crop.width} x {crop.height} 像素）。"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
        {"type": "text", "text": PROMPT2.format(ZOOM=CUR_ZM, desc=_desc_lines(d))},
    ]}]

    trace = [{"turn": 0, "action": "init", "norm": [nx, ny],
              "px": [round(cx), round(cy)], "err": round(err0),
              "text": "第一轮无框"}]
    t0 = time.time()
    n_move = 0
    if cur_box:
        _px, _py = nx / 1000 * W0, ny / 1000 * H0
        trace.append({"turn": 0, "action": "move", "norm": [nx, ny],
                      "px": [round(_px), round(_py)],
                      "err": round(((_px - gcx) ** 2 + (_py - gcy) ** 2) ** 0.5),
                      "ref": "原图", "bbox_raw": [round(v) for v in cur_box],
                      "bbox_orig": [round(v) for v in cur_box],
                      "reason": "L1短提示首轮", "think": "", "finish": "stop"})
        n_move = 1

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
            extra = {"tools": [TOOL_MOVE, TOOL_DONE], "tool_choice": "auto"}
            if reasoning_effort in (None, "", "none"):
                extra["reasoning_effort"] = "none"     # 显式关闭 Qwen 思考
            resp = qc.chat(messages, max_tokens=_mt, extra=extra)
        else:
            extra = {"reasoning_effort": reasoning_effort,
                     "tools": [TOOL_MOVE, TOOL_DONE], "tool_choice": "auto"}
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
        if (tc["function"].get("name") or "").lower() == "done":
            if cur_box is None or n_move < 2:
                # 还没报过框，或报框次数不足 → done() 无效，要求继续收紧
                _why = ("你还没有报过任何框，无法确认。" if cur_box is None
                        else f"你只报了 {n_move} 次框，还没有完成收紧。")
                messages.append({"role": "assistant", "content": msg.get("content") or "",
                                 "tool_calls": tcs})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": _why + "请再用 locate 报一个【比上一轮更贴近目标边缘】的框。"})
                continue
            trace.append({"turn": turn, "action": "stop",
                          "text": "调用 done() 确认结束",
                          "think": _think, "finish": _finish})
            break
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
        bx0 = max(0.0, min(1000.0, bx0)); bx1 = max(0.0, min(1000.0, bx1))
        by0 = max(0.0, min(1000.0, by0)); by1 = max(0.0, min(1000.0, by1))
        cur_box = [bx0, by0, bx1, by1]
        nx = int(round((bx0 + bx1) / 2)); ny = int(round((by0 + by1) / 2))
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

        full, crop, cbox = render_pair(cur_box, CUR_CP, CUR_ZM)
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tcs})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": f"已记录你报的框（原图归一化 {[round(v) for v in cur_box]}）。"})
        messages.append({"role": "user", "content": [
            {"type": "text", "text":
                "【原图】完整界面，蓝框是你上一轮报的框。"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(full)}"}},
            {"type": "text", "text":
                f"【放大图】以蓝框为中心裁出的 {CUR_CP}px 局部，放大 {CUR_ZM} 倍。"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
            {"type": "text", "text":
                "请判断：蓝框有没有框住目标元素？"
                "框住了 → 调用 done() 结束；没框住 → 调用 locate 报一个新的框。"},
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
    ap.add_argument("--full-px", type=float, default=1.7e6,
                    help="送整图时缩放到多少像素（DS 归档方案用 2.5e6）")
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
    out = os.path.join("results", "l1", f"cursorZD_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"[ZD 分阶段+强制2轮] 首轮=L1短提示  鼠标移动智能体  最多 {args.max_turns} 轮  裁片 {args.crop_px}px×{args.zoom}  "
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
                          tries=args.tries, full_px=args.full_px)
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
