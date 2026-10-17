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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw  # noqa: E402

from bench import dataset as H  # noqa: E402
from bench.compat import ds_client as dc  # noqa: E402
from bench.compat import qwen_client as qc  # noqa: E402
from bench.compat.verify_loop import draw_cursor, pad_edge, L1_PROMPT  # noqa: E402

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
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
    """画上一轮的方框（蓝色实线）。"""
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(round(width))))
    return img



TOOL_LOC = {
    "type": "function",
    "function": {
        "name": "locate",
        "description": ("在【这张放大图】上框出目标元素的位置，用【这张放大图】的归一化坐标"
                        "（1-1000 的整数）。注意：坐标是相对这一张小图，不是整个屏幕。"),
        "parameters": {
            "type": "object",
            "properties": {
                "x0": {"type": "integer", "description": "目标框左边缘（1-1000，1000=这张图的右边缘）"},
                "y0": {"type": "integer", "description": "目标框上边缘（1-1000）"},
                "x1": {"type": "integer", "description": "目标框右边缘（1-1000）"},
                "y1": {"type": "integer", "description": "目标框下边缘（1-1000）"},
                "reason": {"type": "string", "description": "你看到的情况（一句话）"},
            },
            "required": ["x0", "y0", "x1", "y1"],
        },
    },
}
TOOL_NOTVIS = {
    "type": "function",
    "function": {
        "name": "not_visible",
        "description": "目标元素【不在这张放大图里】时调用。调用后我会把视野扩大一圈再给你看。",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "你看到的情况（一句话）"}}, "required": []},
    },
}
TOOL_DONE = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "蓝框已经框住目标元素时调用，结束。",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "一句话"}}, "required": []},
    },
}
TOOL_DONE_CONF = {
    "type": "function",
    "function": {
        "name": "done",
        "description": ("蓝框已经框住目标元素时调用，结束。"
                        "必须同时给出你对「蓝框正好框住目标」的把握（0-100 整数）："
                        "100=完全确定，60=比较确定，30=只能猜。"),
        "parameters": {"type": "object", "properties": {
            "confidence": {"type": "integer", "description": "把握 0-100"},
            "reason": {"type": "string", "description": "一句话"}},
            "required": ["confidence"]},
    },
}
TOOL_BLURRY = {
    "type": "function",
    "function": {
        "name": "blurry",
        "description": ("目标【可能】在这张图里，但你【看不清、拿不准】是不是它时调用。"
                        "报出它【大概】在的位置（粗略框即可）——"
                        "我会把视野中心移过去、并放大范围给你看，之后再精修。"),
        "parameters": {
            "type": "object",
            "properties": {
                "x0": {"type": "integer", "description": "大概位置左边缘（1-1000）"},
                "y0": {"type": "integer", "description": "大概位置上边缘（1-1000）"},
                "x1": {"type": "integer", "description": "大概位置右边缘（1-1000）"},
                "y1": {"type": "integer", "description": "大概位置下边缘（1-1000）"},
                "reason": {"type": "string", "description": "你看到的模糊情况（一句话）"},
            },
            "required": ["x0", "y0", "x1", "y1"],
        },
    },
}

PROMPT = """你是一个界面元素定位助手。每一轮我只给你【一张图】：
【放大图】——界面的一小块，放大 {ZOOM} 倍。你看不到屏幕的其他部分。

【图上画的蓝框】
图上画着一个蓝色矩形，那是【你上一轮报的框】。

请【严格按顺序】判断：

  第一步：目标元素是否出现在这张放大图里？
          · 否 → 【调用 not_visible()】，我会把视野扩大一圈再给你看
          · 是 → 继续第二步
  第二步：蓝框是否落在【你认定的那个目标元素】上面？
          · 【只判断位置关系】：蓝框与该元素是否重叠。
          · 【不要】因为描述（外观、文字）有出入而改判——那与位置无关。
          · 是 → 【调用 done() 结束】
          · 否 → 继续第三步
  第三步：用 locate 报出目标元素的位置。

⚠️ 判断的标准是"目标在不在框里"，不是"框准不准"。
   只要目标在框内就算成功，不要因为"框不够精确"而重新框。

【坐标怎么给】
用【这张放大图】的归一化坐标（1 到 1000 的整数）：
  x0, y0 = 框的左上角；x1, y1 = 框的右下角。
  1000 代表【这张放大图】的右边缘 / 下边缘。
  ⚠️ 不要报屏幕坐标，就报你在这一小块里看到的相对位置。

【框的要求】
· 框要【完整包住】目标元素——不要只框住它的一部分。
· 元素可能很小，请尽量贴着它的边缘画。
· 框大一点也没关系，只要目标整个在框里就行。

【目标元素描述】
{desc}
"""

# ── 变体：先描述再动作（--describe-first）────────────────────────────────────
# 动机（M3 失败集分析，n=1261）：
#   · 成功样例的 reason 都是「我的框在哪 vs 目标在哪 + 往哪移」；
#     失败样例只说「未覆盖/未找到」不说方位。
#   · 多轮时成功样例 65% 越调越好；失败样例 57% 越调越坏。
# 做法：在三步判定之前插一个「第零步」，强制先陈述当前位置关系，并把 reason 设为必填。
# ⚠️ 默认关闭 —— 关闭时 PROMPT / 工具集与 legacy 逐字一致。
DESCRIBE_FIRST = False
L1_BOX_OVERRIDE = None   # 受控 A/B 用：给定归一化起点框则跳过 L1 调用

_ANCHOR = "请【严格按顺序】判断：\n"
_DESCRIBE_BLOCK = (
    "\n  第零步【必须先做】：用一句话说清【当前位置关系】，写进 reason 字段——\n"
    "          · 蓝框现在盖在什么上？（描述你看到的那个元素）\n"
    "          · 目标元素相对蓝框在【左 / 右 / 上 / 下】哪个方向？大概差多远（占这张图几分之几）？\n"
    "          ⚠️ 没写清位置关系的动作无效，会被打回。\n"
)
assert _ANCHOR in PROMPT, "变体锚点失效：PROMPT 已被改动"
PROMPT_DESCRIBE_FIRST = PROMPT.replace(_ANCHOR, _ANCHOR + _DESCRIBE_BLOCK)

# ── 变体：强化「目标是否出现在这块里」的判据（--strict-notvis）────────────────
# 动机（受控实验，n=100）：对照臂 83/100 条【视野切换 0 次】——
#   模型几乎从不怀疑"目标不在这块裁片里"，于是在错误位置挑一个最像的元素，
#   再被第二步的"不要因描述有出入而改判"锁死 → 自信答错。
# 做法：给第一步可执行的判据（说不出哪个元素符合外观 = 否），
#       并把第二步的免责句限定为"边界不贴合"，允许因外观不符退回第一步。
# ⚠️ 默认关闭 —— 关闭时 PROMPT 与 legacy 逐字一致。
STRICT_NOTVIS = False

_A_STEP1 = ("  第一步：目标元素是否出现在这张放大图里？\n"
            "          · 否 → 【调用 not_visible()】，我会把视野扩大一圈再给你看\n"
            "          · 是 → 继续第二步\n")
_S_STEP1 = ("  第一步：目标元素是否出现在这张放大图里？\n"
            "          ⚠️ 目标【很可能不在这块里】——这块只是上一轮框的附近区域，\n"
            "             上一轮框偏了的话，目标就在画面之外。\n"
            "          · 先在这张图里逐个看：哪个元素符合描述里的【外观特征】？\n"
            "          · 一个都说不出来 → 【调用 not_visible()】，我会换一档视野给你看。\n"
            "          · 找到了符合的那个 → 继续第二步。\n"
            "          ⚠️ 不要因为这里有个位置差不多的元素就认定它是目标——\n"
            "             位置接近【不等于】是目标；外观不符合，就【不是】目标。\n")
_A_STEP2 = "          · 【不要】因为描述（外观、文字）有出入而改判——那与位置无关。\n"
_S_STEP2 = ("          · 【不要】因为框的边界不够贴合而改判——那与位置无关。\n"
            "            （但若该元素外观根本不符合描述，说明第一步找错了，退回第一步）\n")
assert _A_STEP1 in PROMPT, "strict 变体锚点1失效：PROMPT 已被改动"
assert _A_STEP2 in PROMPT, "strict 变体锚点2失效：PROMPT 已被改动"
PROMPT_STRICT_NOTVIS = PROMPT.replace(_A_STEP1, _S_STEP1).replace(_A_STEP2, _S_STEP2)

# ── 变体：目标模糊时给粗略定位（--blurry-jump）──────────────────────────────
# 动机（受控实验，n=100）：
#   · not_visible 只扩大半径、【不移动中心】；离线重算 287 次放大，19.5% 完全没盖住目标
#   · 模型只有「确定」与「看不见」两个出口，缺「看到了但拿不准」这个中间态
# 做法：新增 blurry(粗略框) —— 把裁片中心移到模型给的位置，给一屏 1600px 大视野，
#       之后回到常规 200px 阶梯（li 归零）。
# ⚠️ 默认关闭 —— 关闭时提示词与工具集与 legacy 逐字一致。
BLURRY_JUMP = False
BLURRY_CROP = 1600

# ── 变体：done() 时要求模型自报置信度（--confidence）────────────────────────
# 用途：认块并行方案需要「两个候选块各跑一遍流程，取置信度高的那个」。
# ⚠️ 默认关闭 —— 关闭时工具集与 legacy 逐字一致。
CONFIDENCE = False

_B_STEP1 = ("  第一步：目标元素是否出现在这张放大图里？\n"
            "          · 否 → 【调用 not_visible()】，我会把视野扩大一圈再给你看\n"
            "          · 是 → 继续第二步\n")
_S_BLURRY = ("  第一步：目标元素是否出现在这张放大图里？\n"
             "          · 完全看不到 → 【调用 not_visible()】，我会把视野扩大一圈再给你看\n"
             "          · 看到了但【拿不准】——好像在这里，却看不清、或不确定是不是它\n"
             "            → 【调用 blurry()】，报出它【大概】在的位置（粗略框即可）。\n"
             "              我会把视野中心移过去、并放大范围给你看，之后再精修。\n"
             "          · 看清楚了 → 继续第二步\n")
assert _B_STEP1 in PROMPT, "blurry 变体锚点失效：PROMPT 已被改动"
PROMPT_BLURRY = PROMPT.replace(_B_STEP1, _S_BLURRY)


def _tools_now():
    """当前该用哪套工具。变体按需加工具 / 把 reason 设为必填。"""
    _done = TOOL_DONE_CONF if CONFIDENCE else TOOL_DONE
    base = [TOOL_LOC, TOOL_NOTVIS, _done] + ([TOOL_BLURRY] if BLURRY_JUMP else [])
    if not (DESCRIBE_FIRST or STRICT_NOTVIS):
        return base
    out = []
    for _t in base:
        _t2 = json.loads(json.dumps(_t))          # deepcopy（不额外引 import）
        _req = list(_t2["function"]["parameters"].get("required") or [])
        if "reason" not in _req:
            _req.append("reason")
        _t2["function"]["parameters"]["required"] = _req
        out.append(_t2)
    return out


def _i(v, default=0):
    """健壮取整：模型可能返回 null / '' / '12' / 12.7。"""
    if v is None:
        return default
    try:
        return int(round(float(str(v).strip())))
    except Exception:  # noqa: BLE001
        return default


def _desc_lines(d):
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n")


def run_one(rid, src, d, *, max_turns=5, ladder=(200, 400, 800, 1600), zoom=4,
            reasoning_effort="none", start=(500, 500), max_tokens=1500, tries=10,
            full_px=2.5e6, out_px=0):
    orig = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
    W0, H0 = orig.size
    _k = min(1.0, (full_px / (W0 * H0)) ** 0.5)
    full_base = (orig if _k >= 1.0 else
                 orig.resize((max(1, round(W0 * _k)), max(1, round(H0 * _k))), Image.LANCZOS))
    gt = src["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2

    def render_crop(box, cp, zm):
        """box = 上一轮报的框（原图归一化）或 None。返回 (裁片图, 裁片在原图的归一化范围)。"""
        if box:
            cx = (box[0] + box[2]) / 2 / 1000 * W0
            cy = (box[1] + box[3]) / 2 / 1000 * H0
        else:
            cx, cy = W0 / 2, H0 / 2
        half = cp // 2
        pad = half + 8
        big = pad_edge(orig, pad)
        X, Y = cx + pad, cy + pad
        x0, y0 = int(X - half), int(Y - half)
        region = big.crop((x0, y0, x0 + cp, y0 + cp))
        sw = sh = int(cp * zm)
        crop = region.resize((sw, sh), Image.LANCZOS)
        if box:
            crop = draw_box(crop,
                            (box[0] / 1000 * W0 + pad - x0) * zm,
                            (box[1] / 1000 * H0 + pad - y0) * zm,
                            (box[2] / 1000 * W0 + pad - x0) * zm,
                            (box[3] / 1000 * H0 + pad - y0) * zm)
        bx0, by0 = x0 - pad, y0 - pad
        nr = [bx0 / W0 * 1000, by0 / H0 * 1000,
              (bx0 + cp) / W0 * 1000, (by0 + cp) / H0 * 1000]
        return crop, nr

    t0 = time.time()
    # ── 阶段1：L1 短提示（整图 + 纯文本）拿粗定位 ──
    _ov = (L1_BOX_OVERRIDE.get(rid) if isinstance(L1_BOX_OVERRIDE, dict)
           else L1_BOX_OVERRIDE)
    if _ov is not None:
        # 受控 A/B：跳过 L1 调用，两臂共享同一个起点框（消掉 L1 的随机噪声）
        cur_box = [float(v) for v in _ov]
    else:
        _l1p = L1_PROMPT.format(env=d.get("env", ""), text=d.get("text", ""),
                                **{k: v for k, v in d.items() if k not in ("env", "text")})
        if BACKEND == "qwen":
            _r = qc.chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url":
                    f"data:image/png;base64,{qc._b64_png(full_base)}"}},
                {"type": "text", "text": _l1p}]}],
                max_tokens=512, timeout=300, retries=4,
                extra={"reasoning_effort": reasoning_effort or "none"})
            _t1 = _r["choices"][0]["message"].get("content") or ""
        else:
            _t1, _, _, _ = dc.ask_image(full_base, _l1p, max_tokens=512,
                                        reasoning_effort=reasoning_effort)
        _m1 = H.NUM4.findall(_t1)
        cur_box = [float(v) for v in _m1[-1]] if _m1 else None
    nx, ny = start
    if cur_box:
        nx = int(round((cur_box[0] + cur_box[2]) / 2))
        ny = int(round((cur_box[1] + cur_box[3]) / 2))
    cx, cy = nx / 1000 * W0, ny / 1000 * H0
    err0 = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5
    trace = [{"turn": 0, "action": "init", "norm": [nx, ny],
              "px": [round(cx), round(cy)], "err": round(err0), "text": "第一轮无框"}]
    n_move = 0
    if cur_box:
        trace.append({"turn": 0, "action": "move", "norm": [nx, ny],
                      "px": [round(cx), round(cy)], "err": round(err0),
                      "ref": "整图(L1)", "bbox_orig": [round(v) for v in cur_box],
                      "reason": "L1短提示首轮", "think": "", "finish": "stop"})
        n_move = 1

    li = 0                      # 裁片阶梯下标
    CP = ladder[li]
    # 放大倍率：固定输出尺寸时由 out_px 决定（render_crop 内部即 int(cp*zm)），否则沿用固定倍率
    ZM = (out_px / CP) if out_px else zoom
    crop, cbox = render_crop(cur_box, CP, ZM)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"【放大图】界面的一小块，边长约 {CP}px，放大 {ZM} 倍"
                                 f"（{crop.width} x {crop.height} 像素）。你看不到屏幕其他部分。"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
        {"type": "text", "text": PROMPT.format(ZOOM=ZM, desc=_desc_lines(d))},
    ]}]

    for turn in range(1, max_turns + 1):
        _mt = max_tokens if reasoning_effort in (None, "", "none") else max(max_tokens, 8000)
        extra = {"reasoning_effort": reasoning_effort,
                 "tools": _tools_now(), "tool_choice": "auto"}
        if BACKEND == "qwen":
            resp = qc.chat(messages, max_tokens=_mt, extra=extra,
                           timeout=300, retries=4)
        else:
            resp, _ = dc.chat(messages, max_tokens=_mt, tries=tries, extra=extra)
        _ch = resp["choices"][0]
        msg = _ch["message"]
        _think = (msg.get("reasoning_content") or "").strip()
        tcs = msg.get("tool_calls") or []
        if not tcs:
            trace.append({"turn": turn, "action": "stop",
                          "text": (msg.get("content") or "").strip()[:150],
                          "think": _think, "finish": _ch.get("finish_reason")})
            break
        tc = tcs[0]
        name = (tc["function"].get("name") or "").lower()
        try:
            a = json.loads(tc["function"]["arguments"] or "{}")
        except Exception:  # noqa: BLE001
            a = {}

        if name == "done":
            if cur_box is None:
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": "你还没有报过任何框。请先用 locate 报框。"})
                continue
            trace.append({"turn": turn, "action": "stop", "text": "done()",
                          "reason": a.get("reason"), "confidence": a.get("confidence"),
                          "think": _think, "finish": _ch.get("finish_reason")})
            break

        if name == "not_visible":
            if li < len(ladder) - 1:
                li += 1
            CP = ladder[li]
            ZM = (out_px / CP) if out_px else zoom
            crop, cbox = render_crop(cur_box, CP, ZM)
            trace.append({"turn": turn, "action": "widen",
                          "text": f"目标不在裁片 → 视野扩到 {CP}px",
                          "think": _think, "finish": _ch.get("finish_reason")})
            for _m in messages:
                _c = _m.get("content")
                if isinstance(_c, list):
                    for _b in _c:
                        if isinstance(_b, dict) and _b.get("type") == "image_url":
                            _b.clear(); _b.update({"type": "text", "text": "（上一轮的图，已省略）"})
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": f"已把视野扩大到 {CP}px（放大 {ZM} 倍）。请看新图。"})
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"【放大图】扩大后的视野，边长约 {CP}px，放大 {ZM} 倍"
                                         f"（{crop.width} x {crop.height} 像素）。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
                {"type": "text", "text": PROMPT.format(ZOOM=ZM, desc=_desc_lines(d))},
            ]})
            continue

        if name == "blurry":
            # 目标模糊：把裁片中心【移到模型给的粗略位置】，给一屏 BLURRY_CROP(1600)px 大视野，
            # 之后回到常规 200px 阶梯（li 归零）
            x0, x1 = _i(a.get("x0")), _i(a.get("x1"))
            y0, y1 = _i(a.get("y0")), _i(a.get("y1"))
            x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
            y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
            gx0, gy0, gx1, gy1 = cbox
            cur_box = [gx0 + x0 / 1000.0 * (gx1 - gx0), gy0 + y0 / 1000.0 * (gy1 - gy0),
                       gx0 + x1 / 1000.0 * (gx1 - gx0), gy0 + y1 / 1000.0 * (gy1 - gy0)]
            cur_box = [max(0.0, min(1000.0, v)) for v in cur_box]
            trace.append({"turn": turn, "action": "blurry",
                          "bbox_raw": [x0, y0, x1, y1],
                          "bbox_orig": [round(v) for v in cur_box],
                          "reason": (a.get("reason") or "")[:70], "think": _think,
                          "finish": _ch.get("finish_reason")})
            li = 0
            CP = BLURRY_CROP
            ZM = (out_px / CP) if out_px else zoom
            crop, cbox = render_crop(cur_box, CP, ZM)
            for _m in messages:
                _c = _m.get("content")
                if isinstance(_c, list):
                    for _b in _c:
                        if isinstance(_b, dict) and _b.get("type") == "image_url":
                            _b.clear(); _b.update({"type": "text", "text": "（上一轮的图，已省略）"})
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": f"已把视野中心移到你报的位置，并扩大到 {CP}px（放大 {ZM} 倍）。请看新图。"})
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"【放大图】以你报的粗略位置为中心，边长约 {CP}px，"
                                         f"放大 {ZM} 倍（{crop.width} x {crop.height} 像素）。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
                {"type": "text", "text": PROMPT.format(ZOOM=ZM, desc=_desc_lines(d))},
            ]})
            continue

        # locate：坐标是【裁片内】归一化 → 映射回原图
        x0, x1 = _i(a.get("x0")), _i(a.get("x1"))
        y0, y1 = _i(a.get("y0")), _i(a.get("y1"))
        x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
        y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
        if x1 - x0 < 1:
            x1 = x0 + 1
        if y1 - y0 < 1:
            y1 = y0 + 1
        gx0, gy0, gx1, gy1 = cbox
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
                      "ref": f"裁片{CP}", "bbox_raw": [x0, y0, x1, y1],
                      "bbox_orig": [round(bx0), round(by0), round(bx1), round(by1)],
                      "reason": (a.get("reason") or "")[:70],
                      "think": _think, "finish": _ch.get("finish_reason")})

        for _m in messages:
            _c = _m.get("content")
            if isinstance(_c, list):
                for _b in _c:
                    if isinstance(_b, dict) and _b.get("type") == "image_url":
                        _b.clear(); _b.update({"type": "text", "text": "（上一轮的图，已省略）"})
        crop, cbox = render_crop(cur_box, CP, ZM)
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": f"已记录（原图归一化 {[round(v) for v in cur_box]}）。"})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"【放大图】以你刚报的框为中心，边长约 {CP}px，放大 {ZM} 倍"
                                     f"（{crop.width} x {crop.height} 像素）。蓝框是你刚报的框。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(crop)}"}},
            {"type": "text", "text": PROMPT.format(ZOOM=ZM, desc=_desc_lines(d))},
        ]})

    fx, fy = nx / 1000 * W0, ny / 1000 * H0
    final_err = ((fx - gcx) ** 2 + (fy - gcy) ** 2) ** 0.5
    return {"id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
            "start_norm": list(start), "start_err": round(err0),
            "final_norm": [nx, ny], "final_px": [round(fx), round(fy)],
            "final_err": round(final_err, 1),
            "hit": gt[0] <= fx <= gt[2] and gt[1] <= fy <= gt[3],
            "n_move": n_move, "turns_used": len(trace) - 1, "trace": trace,
            "reasoning_effort": reasoning_effort,
            "think_chars": sum(len(t.get("think") or "") for t in trace),
            "max_turns": max_turns, "ladder": list(ladder), "zoom": zoom,
            "latency_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="逗号分隔；省略则读 splits/<split>.json")
    ap.add_argument("--split", default="M1")
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--desc-file", default="", help="覆盖 --desc-version")
    ap.add_argument("--tag", default="", help="默认 b3_crop_ladder_<backend>_<split>_<版本>")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--ladder", default="200,400,800,1600")
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--out-px", type=int, default=0,
                    help="固定裁片【输出边长】（像素）。0 = 沿用 --zoom 固定倍率（legacy 行为）。"
                         "例：--out-px 1600 时 200px 档=8×、1600px 档=1×")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--backend", choices=["deepseek", "qwen"], default="deepseek")
    ap.add_argument("--start", default="500,500")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--full-px", type=float, default=2.5e6)
    ap.add_argument("--describe-first", action="store_true",
                    help="变体：强制模型先陈述位置关系再动作（默认关闭；关闭时与 legacy 逐字一致）")
    ap.add_argument("--l1-box", default="",
                    help="跳过 L1 调用，直接用给定归一化框 x0,y0,x1,y1（0-1000）。受控 A/B 用")
    ap.add_argument("--l1-box-file", default="",
                    help="按 id 指定起点框的 JSON（{id:[x0,y0,x1,y1]}），优先级高于 --l1-box")
    ap.add_argument("--strict-notvis", action="store_true",
                    help="变体：强化「目标是否在这块裁片里」的判据（默认关闭；关闭时与 legacy 逐字一致）")
    ap.add_argument("--blurry-jump", action="store_true",
                    help="变体：新增 blurry() 工具——目标模糊时报粗略位置，中心移过去并给 1600px 大视野")
    ap.add_argument("--confidence", action="store_true",
                    help="变体：done() 时要求模型自报 0-100 的把握（认块并行方案用于择优）")
    args = ap.parse_args()
    global BACKEND, DESCRIBE_FIRST, PROMPT, L1_BOX_OVERRIDE, STRICT_NOTVIS, BLURRY_JUMP, CONFIDENCE
    BACKEND = args.backend
    if args.describe_first:
        DESCRIBE_FIRST = True
        PROMPT = PROMPT_DESCRIBE_FIRST
    if args.strict_notvis:
        STRICT_NOTVIS = True
        PROMPT = PROMPT_STRICT_NOTVIS
    if args.blurry_jump:
        BLURRY_JUMP = True
        PROMPT = PROMPT_BLURRY
    if args.confidence:
        CONFIDENCE = True
    if args.l1_box:
        L1_BOX_OVERRIDE = tuple(int(v) for v in args.l1_box.split(","))
    if args.l1_box_file:
        L1_BOX_OVERRIDE = {k: tuple(int(x) for x in v) for k, v in
                           json.load(open(args.l1_box_file, encoding="utf-8")).items()}
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ids = args.ids or ",".join(json.load(
        open(os.path.join(_root, "splits", f"{args.split}.json"), encoding="utf-8")))
    _desc_file = args.desc_file or os.path.join(
        _root, "descriptions", args.desc_version, f"{args.split}.json")
    _tag = args.tag or (f"b3_crop_ladder_{args.backend}_{args.split}_{args.desc_version}"
                        + ("_df" if args.describe_first else "")
                        + ("_sn" if args.strict_notvis else "")
                        + ("_bj" if args.blurry_jump else "")
                        + ("_cf" if args.confidence else ""))
    DESC = json.load(open(_desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    sx, sy = (int(v) for v in args.start.split(","))
    out = os.path.join(_root, "baselines", "results", "raw", f"{_tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lad = tuple(int(v) for v in args.ladder.split(","))
    print(f"[ZF 只送裁片+阶梯 {lad}]  最多 {args.max_turns} 轮  思考={args.reasoning_effort}")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(_ids.split(","), 1):
        try:
            rec = run_one(rid, rows[rid], DESC[rid], max_turns=args.max_turns,
                          ladder=lad, zoom=args.zoom,
                          reasoning_effort=args.reasoning_effort, start=(sx, sy),
                          max_tokens=args.max_tokens, tries=args.tries, full_px=args.full_px,
                          out_px=args.out_px)
        except Exception as e:  # noqa: BLE001
            rec = {"id": rid, "status": "ERROR", "error": str(e)[:200]}
        recs.append(rec)
        e = rec.get("final_err")
        mv = "".join({"move": "M", "stop": "S", "init": "·", "widen": "W"}.get(t["action"], "?")
                     for t in rec.get("trace", []))
        print(f"  [{k:>2}] {rid:<26} 起点{rec.get('start_err','?'):>5}px → "
              f"终点{(round(e) if e is not None else '?'):>5}px  轨迹={mv:<9}"
              f"{rec.get('latency_s',0):>5.1f}s  {'✅' if rec.get('hit') else '❌'}", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")


if __name__ == "__main__":
    main()
