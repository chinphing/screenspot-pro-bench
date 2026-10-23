"""B4：智能体内外层循环（无点击）。

外层：看整屏（带当前蓝框）→ locate(x,y) 或 finish
内层：b3 的裁片阶梯（多轮校验纠偏）→ 返回框
外层再确认：蓝框是目标就 finish，否则继续 locate
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PIL import Image  # noqa: E402

from bench import dataset as H  # noqa: E402
from bench.compat import qwen_client as qc  # noqa: E402
from bench.compat.verify_loop import L1_PROMPT, pad_edge  # noqa: E402
from baselines import b3_crop_ladder as b3  # noqa: E402

TOOL_XY = {
    "type": "function",
    "function": {
        "name": "decide",
        "description": ("你必须先描述【放大图里蓝框框住的是什么】，再判断它和目标是否一致。"
                        "seen 写你实际看到的（文字/图标/控件），不要写目标应该是什么。"
                        "然后 verdict=correct 表示已框住目标；verdict=wrong 表示没框住，需给新 bbox。"),
        "parameters": {
            "type": "object",
            "properties": {
                "seen": {
                    "type": "string",
                    "description": "【必填】① 蓝框里实际是什么 ② 蓝框外紧邻的是什么（左/右/上/下），只写看到的",
                },
                "verdict": {
                    "type": "string",
                    "enum": ["correct", "wrong"],
                    "description": "correct=蓝框已经框住目标；wrong=没有框住（偏了/框错东西/还没有框）",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "verdict=wrong 时必填：新的目标框 [x1,y1,x2,y2]，0-1000 整数",
                },
                "reason": {"type": "string", "description": "一句话说明你的判断依据"},
            },
            "required": ["seen", "verdict"],
        },
    },
}

TOOL_BBOX = {
    "type": "function",
    "function": {
        "name": "locate",
        "description": ("给出目标元素在截图上的 bbox，归一化到 0-1000 整数。"
                        "要能框住该元素本身，不要外扩太多、也不要只框住一角。"),
        "parameters": {"type": "object", "properties": {
            "bbox": {"type": "array", "items": {"type": "integer"},
                     "minItems": 4, "maxItems": 4,
                     "description": "目标框 [x1, y1, x2, y2]，四个 0-1000 的整数"}},
            "required": ["bbox"]},
    },
}

TOOL_SEE = {
    "type": "function",
    "function": {
        "name": "see",
        "description": "只描述放大图里蓝框框住的东西是什么。不做判断、不给坐标、不评价。",
        "parameters": {"type": "object", "properties": {
            "seen": {"type": "string",
                     "description": "① 蓝框里是什么 ② 蓝框外紧邻的是什么（左/右/上/下），只写看到的"}},
            "required": ["seen"]},
    },
}

TOOL_FIN = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": ("蓝框已经框住目标时调用，结束。"
                        "调用前必须确认：蓝框里的是【描述中的那个元素】，不是旁边长得相似的。"
                        "只框住了一个东西但不确定是它，就不要调用。"),
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "一句话"}}, "required": []},
    },
}

_L1_DESC = (L1_PROMPT.split("请给出【刚好框住")[0]
            .replace("给出它的 bbox。", "给出它的位置。"))

PROMPT_FIRST = _L1_DESC + """请调用 locate(bbox=[x1, y1, x2, y2]) 给出【刚好框住该元素本身】的框
（不要外扩、不要缩小）。坐标归一化到 0-1000 的整数范围，0 表示左/上边缘，1000 表示右/下边缘。

**不需要像素级精确**：框的中心落在元素上、四条边没有明显超出元素轮廓，就算合格。
"""

FIRST_DECIDE = False     # True=第1步也用 decide 工具（B4v 行为）
PROMPT_SEE = _L1_DESC + """**这一步只做一件事：描述【放大图】里蓝框框住的东西，以及它周围紧邻的东西。**

调用 see，seen 字段必须写两部分：
① 蓝框【里面】是什么（文字/图标/控件，写你看到的）
② 蓝框【外面紧邻】的是什么（左边是什么、右边是什么、上面/下面是什么）

不要判断对不对，不要给坐标，不要提目标描述。
"""

TWO_PHASE = False        # True=每次确认前先单独做一次「看」的调用
OUTER_EFFORT = "none"   # 只影响外层，内层仍用 reasoning_effort

PROMPT_OUTER = _L1_DESC + """**你要做的第一件事：描述放大图里蓝框框住的是什么，再判断它对不对**
· 已经框住目标 → **调用 finish() 结束**，不要再 locate
· 还没框住（偏了、框错了、或者还没有框）→ 才调用 locate

**locate 的输出结构（严格按这个来）**：参数 bbox 是一个四元整数数组：

    locate(bbox=[x1, y1, x2, y2], reason="一句话")

例：目标在整屏左上角，大约占横向 24-35、纵向 86-107 → `locate(bbox=[24, 86, 35, 107], reason="...")`

四个数都是 0-1000 的整数：0 = 左/上边缘，1000 = 右/下边缘。
不要写成键值对象（如 x1/y1/x2/y2 分开写），也不要塞坐标对 [[x,y],[x,y]]。

**不需要像素级精确**：框的中心落在元素上、四条边没有明显超出元素轮廓，就算合格。

{state}

**看图说明**：
· 第 2 张【放大图】：**首选**。以蓝框为中心放大 4 倍，判断内层结论对不对，主要看这张
· 第 1 张【整屏】：**兜底用**。只有目标不在放大图里时，才用它重新定位

{zoom_help}

**判断顺序（必须按这个来）**：
1. 先在【放大图】里找目标
   · 目标在放大图里、蓝框正好框住它 → 调用 finish()
   · 目标在放大图里、但蓝框偏了 → 用放大图换算新的框 → locate(bbox=[...])
2. 目标【不在】放大图里 → 才看【整屏】，重新给一个框 → locate(bbox=[...])

**换算新框时用放大图来算，不要用整屏硬猜**：
看目标四条边在放大图里的位置（横向/纵向各占百分之几），按上面的范围换算回整屏 0-1000。
例：目标左边缘在放大图横向 40% 处 → 整屏 x1 ≈ 左边界 + 0.4 ×（右边界 − 左边界）。
"""


def _parse_box(a):
    """从工具参数里取出 [x1,y1,x2,y2]，兼容模型各种写法。"""
    if not isinstance(a, dict):
        return None
    for key in ("bbox", "box", "rect", "coords", "coordinates", "region"):
        v = a.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            return [_num(v[0]), _num(v[1]), _num(v[2]), _num(v[3])]
        if isinstance(v, str):                      # 模型把整个数组塞成字符串
            _ns = re.findall(r"-?\d+(?:\.\d+)?", v)
            if len(_ns) >= 4:
                return [_num(x) for x in _ns[:4]]
    # x1/x2 各是 [x, y] 坐标对：{"x1":[27,93], "x2":[34,108]}
    _a, _b = a.get("x1"), a.get("x2")
    if (isinstance(_a, (list, tuple)) and len(_a) == 2
            and isinstance(_b, (list, tuple)) and len(_b) == 2):
        return [_num(_a[0]), _num(_a[1]), _num(_b[0]), _num(_b[1])]

    # x1/y1/x2/y2、x0/y0/x1/y1、left/top/right/bottom 三种命名
    for ks in (("x1", "y1", "x2", "y2"), ("x0", "y0", "x1", "y1"),
               ("left", "top", "right", "bottom")):
        if all(a.get(k) is not None for k in ks):
            vals = [a.get(k) for k in ks]
            # 单元素数组解包：{"x1": [27], "y1": 93}
            vals = [v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v for v in vals]
            if any(isinstance(v, (list, tuple)) for v in vals):
                for v in vals:
                    if isinstance(v, (list, tuple)) and len(v) >= 4:
                        return [_num(v[0]), _num(v[1]), _num(v[2]), _num(v[3])]
                continue
            return [_num(v) for v in vals]
    # 任意字段里塞了整个四元数组 / 字符串
    for v in a.values():
        if isinstance(v, str):
            _ns = re.findall(r"-?\d+(?:\.\d+)?", v)
            if len(_ns) >= 4:
                return [_num(x) for x in _ns[:4]]
        if (isinstance(v, (list, tuple)) and len(v) >= 4
                and all(isinstance(x, (int, float)) for x in v)):
            return [_num(v[0]), _num(v[1]), _num(v[2]), _num(v[3])]
    return None


def _center_box(px, py, W0, H0, side_px=40):
    hw = side_px / 2 / W0 * 1000
    hh = side_px / 2 / H0 * 1000
    return [max(0.0, px - hw), max(0.0, py - hh), min(1000.0, px + hw), min(1000.0, py + hh)]


def _render_verify(orig, box, cp=400, out_px=1600):
    """以 box 为中心渲染一张放大图，并在上面画出 box。"""
    W0, H0 = orig.size
    cx = (box[0] + box[2]) / 2 / 1000 * W0
    cy = (box[1] + box[3]) / 2 / 1000 * H0
    half = cp // 2
    pad = half + 8
    big = pad_edge(orig, pad)
    X, Y = cx + pad, cy + pad
    x0, y0 = int(X - half), int(Y - half)
    region = big.crop((x0, y0, x0 + cp, y0 + cp))
    zm = out_px / cp
    crop = region.resize((int(cp * zm), int(cp * zm)), Image.LANCZOS)
    bx0 = (box[0] / 1000 * W0 + pad - x0) * zm
    bx1 = (box[2] / 1000 * W0 + pad - x0) * zm
    by0 = (box[1] / 1000 * H0 + pad - y0) * zm
    by1 = (box[3] / 1000 * H0 + pad - y0) * zm
    img = b3.draw_box(crop, bx0, by0, bx1, by1)
    # 这张放大图覆盖整屏的哪一块（0-1000 归一化）
    bounds = ((x0 - pad) / W0 * 1000, (y0 - pad) / H0 * 1000,
              (x0 - pad + cp) / W0 * 1000, (y0 - pad + cp) / H0 * 1000)
    return img, bounds


def _num(v, default=500):
    """把模型可能返回的 list/str/float 统一成 0-1000 的整数。"""
    while isinstance(v, (list, tuple)):
        if not v:
            return default
        v = v[0]
    try:
        return max(0, min(1000, int(round(float(v)))))
    except Exception:  # noqa: BLE001
        return default


def run_one_agent(rid, src, d, *, max_outer=4, max_turns=5, out_px=1600,
                  backend="qwen", reasoning_effort="none", abort_widen=2):
    orig = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
    W0, H0 = orig.size
    k = min(1.0, (2.5e6 / (W0 * H0)) ** 0.5)
    full_base = (orig if k >= 1.0 else
                 orig.resize((max(1, round(W0 * k)), max(1, round(H0 * k))), Image.LANCZOS))
    gt = src["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    desc = b3._desc_lines(d)
    b3.BACKEND = backend
    b3.ABORT_WIDEN = abort_widen

    cur_box = None
    _confirmed = None
    outer = []
    t0 = time.time()
    finished = False
    _parse_retries = 0
    _notext = 0
    _history = []
    _note = ""        # 需要传给下一轮的提示（重试说明 / 内层反馈）
    messages = []          # 累积对话，和真实智能体一致

    for step in range(1, max_outer + 1):
        img = full_base
        if cur_box:
            img = b3.draw_box(img,
                              cur_box[0] / 1000 * img.width, cur_box[1] / 1000 * img.height,
                              cur_box[2] / 1000 * img.width, cur_box[3] / 1000 * img.height)
        if cur_box:
            state = (f"当前蓝框（整屏归一化）= [{cur_box[0]:.0f}, {cur_box[1]:.0f}, "
                     f"{cur_box[2]:.0f}, {cur_box[3]:.0f}]，见图中蓝色矩形。")
        else:
            state = "目前还没有蓝框。"
        if _history:
            state += "\n\n**你已经尝试过的**（从上到下）：\n" + "\n".join(_history[-3:])
        if _note:
            state += "\n\n**上一轮的结果**：\n" + _note
            _note = ""
        # 每轮重建：把历史压成文字摘要，避免模型延续上一轮的 locate 动作
        messages = []
        _parts = [{"type": "text", "text": "【整屏】"},
                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{qc._b64_png(img)}"}}]
        _zoom_help = "（本次没有放大图，直接看整屏）"
        if cur_box:
            _vc, _bd = _render_verify(orig, cur_box)
            _oob = any(v < 0 or v > 1000 for v in _bd)
            _zoom_help = (f"【放大图】覆盖整屏的横向 {_bd[0]:.0f}–{_bd[2]:.0f}、"
                          f"纵向 {_bd[1]:.0f}–{_bd[3]:.0f}（0-1000 归一化）。"
                          f"放大图里横向 0% 对应整屏 {_bd[0]:.0f}，横向 100% 对应 {_bd[2]:.0f}；"
                          f"纵向 0% 对应 {_bd[1]:.0f}，纵向 100% 对应 {_bd[3]:.0f}。")
            if _oob:
                _zoom_help += "（出现负数或超过 1000，表示这块已经到屏幕边缘、图里有留白，属正常。）"
            _parts += [{"type": "text", "text": "【放大图】以你当前的框为中心，放大 4 倍"},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{qc._b64_png(_vc)}"}}]
        _fields = {k: (d.get(k) or "") for k in ("text", "feat", "ctrl", "pos", "env")}
        if cur_box is None and not FIRST_DECIDE:
            # 第1步是纯定位任务：用 B3 的 L1 原文，不掺判断和换算
            _prompt = PROMPT_FIRST.format(**_fields)
        else:
            _prompt = PROMPT_OUTER.format(state=state, zoom_help=_zoom_help, **_fields)
        if cur_box is not None:
            _prompt += ("\n\n**第一步：先调用 see，只描述蓝框里是什么。**"
                        "不要判断，不要给坐标。")
        _parts.append({"type": "text", "text": _prompt})
        if cur_box is not None and TWO_PHASE:
            _see_parts = _parts[:-1] + [{"type": "text", "text": PROMPT_SEE.format(**_fields)}]
            _seen = ""
            try:
                _r1 = qc.chat([{"role": "user", "content": _see_parts}], max_tokens=400,
                              timeout=300, retries=4,
                              extra={"reasoning_effort": OUTER_EFFORT,
                                     "tools": [TOOL_SEE], "tool_choice": "auto"})
                _m1 = _r1["choices"][0]["message"]
                _tc1 = (_m1.get("tool_calls") or [None])[0]
                if _tc1:
                    _a1 = json.loads(_tc1["function"].get("arguments") or "{}")
                    _seen = (_a1.get("seen") or "").strip()
            except Exception:  # noqa: BLE001
                _seen = ""
            if _seen:
                _parts[-1] = {"type": "text", "text":
                              _prompt + f"\n\n**你刚才看蓝框，看到的是**：{_seen}\n\n"
                                        f"现在判断：蓝框里的东西，是不是目标描述里的元素？\n\n"
                                        f"**判定规则（必须遵守）**：\n"
                                        f"1. 外观特征必须对上。\n"
                                        f"2. 周边锚点必须在【放大图里真的看得见】才算证据。"
                                        f"如果锚点不在放大图范围内，就写「锚点不在视野内」，"
                                        f"**不能假定它匹配**。\n"
                                        f"3. 如果外观像 A、锚点却指向 B（互相矛盾），"
                                        f"**不要用「上一轮已经验证过」来消除矛盾**——"
                                        f"矛盾本身就说明这个框很可能是错的，应当判 wrong 并重新定位。\n"
                                        f"4. 屏幕位置只作粗筛参考，精判以放大图为准。\n\n"
                                        f"三样都对上才算 correct；有任何一样对不上或看不到，就是 wrong。"}
                _note = f"（上一轮你先看了框里，写的是：{_seen[:120]}）"
        messages.append({"role": "user", "content": _parts})
        try:
            resp = qc.chat(messages, max_tokens=800, timeout=300, retries=4,
                           extra={"reasoning_effort": OUTER_EFFORT,
                                  "tools": ([TOOL_BBOX] if (cur_box is None and not FIRST_DECIDE)
                                            else [TOOL_XY]),
                                  "tool_choice": "auto"})
        except Exception as e:  # noqa: BLE001
            outer.append({"step": step, "action": "error", "error": str(e)[:120]})
            break
        _ch = resp["choices"][0]
        msg = _ch["message"]
        _think = ((msg.get("reasoning_content") or _ch.get("reasoning_content") or "")).strip()
        tcs = msg.get("tool_calls") or []
        if not tcs:
            _notext += 1
            if _notext > 3:
                outer.append({"step": step, "action": "notext",
                              "text": (msg.get("content") or "")[:100]})
                break
            step -= 1                      # 重试，不消耗外层步数
            _note = "你上一轮没有调用工具。必须调用 decide：框住了给 verdict=correct，没框住给 verdict=wrong + bbox。"
            continue
        tc = tcs[0]
        name = (tc["function"].get("name") or "").lower()
        try:
            a = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:  # noqa: BLE001
            a = {}

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})

        if name == "finish":
            if cur_box is None:
                outer.append({"step": step, "action": "finish_early"})
                continue
            outer.append({"step": step, "action": "finish",
                          "reason": (a.get("reason") or "")[:300],
                          "say": (msg.get("content") or "").strip()[:300]})
            finished = True
            break

        _verdict = (a.get("verdict") or "").strip().lower()
        _seen = (a.get("seen") or "").strip()
        if _verdict == "correct" and not _seen:
            # 没有描述框里是什么，不允许判正确
            _note = ("你判了 correct，但没有填写 seen。"
                     "必须先在 seen 里写出【放大图里蓝框实际框住的是什么】，再判断。")
            step -= 1
            continue
        if _verdict == "correct":
            if cur_box is None:
                outer.append({"step": step, "action": "verdict_no_box"})
                _note = "现在还没有蓝框，不可能判 correct。请给 bbox 并设 verdict=wrong。"
                step -= 1
                continue
            outer.append({"step": step, "action": "finish", "via": "verdict",
                          "seen": (a.get("seen") or "")[:200],
                          "reason": (a.get("reason") or "")[:300],
                          "think": _think[:1500],
                          "say": (msg.get("content") or "").strip()[:300]})
            finished = True
            break

        _in_box = _parse_box(a)
        if _in_box is None:
            # 解析不出来就明确报错让模型重发，绝不静默当成屏幕中心
            outer.append({"step": step, "action": "parse_error",
                          "raw": (tc["function"].get("arguments") or "")[:200]})
            _parse_retries += 1
            if _parse_retries > 3:
                break
            step -= 1                      # 解析失败不消耗外层步数
            _note = ("你上一轮的参数无法解析。bbox 必须直接是长度为 4 的整数数组，"
                     "例如 bbox=[500, 73, 504, 81]。不要把数组和 reason 拼成一个字符串。")
            continue
        _in_box = [min(_in_box[0], _in_box[2]), min(_in_box[1], _in_box[3]),
                   max(_in_box[0], _in_box[2]), max(_in_box[1], _in_box[3])]
        x = (_in_box[0] + _in_box[2]) / 2
        y = (_in_box[1] + _in_box[3]) / 2
        # 注意：b3 对 dict 形式按 rid 取值，这里必须传裸列表才会生效
        b3.L1_BOX_OVERRIDE = _in_box
        try:
            r = b3.run_one(rid, src, d, max_turns=max_turns, out_px=out_px,
                           reasoning_effort=reasoning_effort)
        except Exception as e:  # noqa: BLE001
            outer.append({"step": step, "action": "locate_error", "error": str(e)[:120]})
            break
        # 只认有面积的框；内层没找到东西时 trace 里残留的是外层传进去的点，不能当结果
        box = None
        for t in (r.get("trace") or []):
            _b = t.get("bbox_orig")
            if _b and (_b[2] - _b[0]) > 0 and (_b[3] - _b[1]) > 0:
                box = _b
        cur_box = box
        _inner_ok = bool(r.get("trace") and r["trace"][-1].get("action") == "stop")
        if box and _inner_ok:
            _confirmed = box          # 内层确认过的框，优先作为最终答案
        outer.append({"step": step, "action": "locate", "xy": [x, y], "box": box,
                      "reason": (a.get("reason") or "")[:300],
                      "think": _think[:1500],
                      "say": (msg.get("content") or "").strip()[:300],
                      "inner_done": bool(r.get("trace") and r["trace"][-1].get("action") == "stop"),
                      "inner_turns": r.get("turns_used"), "inner_moves": r.get("n_move")})
        # —— 与真实智能体一致的反馈模板（agent.py）——
        _tr = r.get("trace") or []
        _last = _tr[-1] if _tr else {}
        _found = _last.get("action") == "stop"
        _tried, _reasons = [], []
        for _t in _tr:
            _b = _t.get("bbox_orig")
            if _b and _t.get("action") in ("move", "stop"):
                _pt = (round((_b[0] + _b[2]) / 2), round((_b[1] + _b[3]) / 2))
                if _tried and _tried[-1] == _pt:
                    continue
                _tried.append(_pt)
                _reasons.append((_pt[0], _pt[1], (_t.get("reason") or "").strip()))
        _widen_n = sum(1 for _t in _tr if _t.get("action") == "widen")

        def _to_px(p):
            return (round(p[0] / 1000 * W0), round(p[1] / 1000 * H0))

        _hist_txt = (f"· 第{step}轮：框=[{_in_box[0]:.0f},{_in_box[1]:.0f},"
                     f"{_in_box[2]:.0f},{_in_box[3]:.0f}]，"
                     f"内层校验：{'框住了' if _found else '没确认'}")
        _history.append(_hist_txt)

        _noop = (box is not None and _in_box is not None and
                 all(abs(a - b) <= 1 for a, b in zip(box, _in_box)))
        _px = _to_px((x, y))
        if _noop:
            _content = (f"内层没有给出新的框：它在{_px}附近放大查找后，认为目标不在那里。"
                        f"你给的坐标没有被修正。")
            if _widen_n:
                _content += (f"它连续 {_widen_n} 轮说看不到目标，视野已经放大到最大范围。"
                             f"请重新看整张截图，判断这个目标是否真的在屏幕上、"
                             f"或者在什么别的位置。")
            _note = _content
            continue
        if _found:
            _content = f"鼠标成功移动到{_px}，已验证对应位置正好是目标。"
            _rs = [q[2] for q in _reasons if q[2]]
            if _rs:
                _content += f"你最后一轮的判断依据：{_rs[-1]}"
        else:
            _pts = "、".join(str(_to_px(q)) for q in _tried) or "（无）"
            _content = (f"鼠标移动失败，你给的坐标{_px}"
                        f"以及我纠偏的坐标{_pts}经验证都不在目标上面。")
            _rs = [q for q in _reasons if q[2]]
            if _rs:
                _content += "各轮的判断依据："
                for _rx, _ry, _rr in _rs:
                    _content += f"\n  {_to_px((_rx, _ry))}：{_rr}"
            if _widen_n:
                _content += (f"另外你连续 {_widen_n} 轮说看不到目标，"
                             f"视野已经放大到最大范围仍然没找到。"
                             f"请重新看整张截图，判断这个目标是否真的在屏幕上。")
        _note = _content

    _final = _confirmed or cur_box
    if _final:
        fx, fy = (_final[0] + _final[2]) / 2 / 1000 * W0, (_final[1] + _final[3]) / 2 / 1000 * H0
    else:
        fx, fy = -1.0, -1.0
    return {
        "id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
        "final_px": [round(fx), round(fy)],
        "final_err": round(((fx - gcx) ** 2 + (fy - gcy) ** 2) ** 0.5, 1),
        "hit": gt[0] <= fx <= gt[2] and gt[1] <= fy <= gt[3],
        "finished": finished, "outer_steps": len(outer), "outer": outer,
        "latency_s": round(time.time() - t0, 1),
        "model": (b3.qc.MODEL if b3.BACKEND == "qwen" else b3.dc.MODEL),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--split", default="M3")
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--max-outer", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--abort-widen", type=int, default=2)
    ap.add_argument("--out-px", type=int, default=1600)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--first-decide", action="store_true")
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--outer-effort", default="none",
                    help="外层思考强度：none/low/medium/high")
    ap.add_argument("--tag", default="b4")
    args = ap.parse_args()

    b3.ABORT_WIDEN = args.abort_widen
    global OUTER_EFFORT, FIRST_DECIDE, TWO_PHASE
    OUTER_EFFORT = args.outer_effort
    FIRST_DECIDE = args.first_decide
    TWO_PHASE = args.two_phase
    rows = {r["id"]: r for r in H.load_all()}
    DESC = json.load(open(os.path.join(_ROOT, "descriptions", args.desc_version,
                                       f"{args.split}.json"), encoding="utf-8"))
    ids = args.ids or ",".join(json.load(open(os.path.join(_ROOT, "splits", f"{args.split}.json"))))
    out = os.path.join(_ROOT, "baselines", "results", "raw", f"{args.tag}.jsonl")
    print(f"B4 内外层循环  外层最多 {args.max_outer} 步  内层最多 {args.max_turns} 轮")
    print(f"输出 → {out}\n")

    recs = []
    for k, rid in enumerate(ids.split(","), 1):
        try:
            rec = run_one_agent(rid, rows[rid], DESC[rid], max_outer=args.max_outer,
                                max_turns=args.max_turns, out_px=args.out_px,
                                backend=args.backend, reasoning_effort=args.reasoning_effort)
        except Exception as e:  # noqa: BLE001
            rec = {"id": rid, "status": "ERROR", "error": str(e)[:200]}
        rec["tag"] = args.tag
        recs.append(rec)
        acts = "".join({"locate": "L", "finish": "F", "finish_early": "E",
                        "notext": "N", "error": "X"}.get(o.get("action"), "?")
                       for o in rec.get("outer") or [])
        print(f"  [{k:>3}] {rid:<26} 外层={acts:<6} 终点{rec.get('final_err','?'):>6}px "
              f"{rec.get('latency_s',0):>5.1f}s  {'✅' if rec.get('hit') else '❌'}", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")


if __name__ == "__main__":
    main()
