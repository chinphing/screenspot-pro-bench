"""B7：B5 的框定位 + B6 的「判中心」校验

动机
  B5（框）精度好但假确认 15%；B6（点）假确认只有 8% 但精度差 3 倍。
  差异来自校验器问的问题：B5 问「框里是什么」，B6 问「尖端落在什么上」。
  评测口径判的正是「框中心是否落在 GT 内」——所以校验就该判中心点。

做法
  定位器仍然输出 bbox（保留尺寸约束带来的精度）；
  校验器看到的裁剪图上，除了蓝框，还在【框中心】画一个光标箭头，
  让它回答「光标尖端落在什么上面」，与评测口径对齐。

其余（控制器、坐标换算、裁剪自适应）与 B5 相同。

原始 B5 说明：
B5：定位与校验彻底解耦的智能体循环（独立实现，不修改 B4）

设计：三个组件，职责单一
  ① 定位器 locator  —— 只看一张图，给出坐标。没有 verdict、没有 finish。
  ② 校验器 verifier —— 只看裁剪图（不给整屏），输出结构化结论。
  ③ 控制器          —— 纯代码，确定性决策（停不停、扩大还是缩小视野）。

坐标模式（--coord-mode）：
  rel —— 模型只报【当前这张图】内的 0-1000 坐标，脚本换算回整屏（模型不做乘除）
  abs —— 模型直接报【整屏】的 0-1000 坐标（DeepSeek 偏好绝对坐标）

用法：
  python3 -m baselines.b5_decoupled --ids "id1,id2" --split M3 --coord-mode rel --tag b5_rel
  python3 -m baselines.b5_decoupled --ids "id1,id2" --split M3 --coord-mode abs --tag b5_abs
"""
import argparse
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import dataset as H                      # noqa: E402
from bench.compat import qwen_client as qc          # noqa: E402
from bench.compat import ds_client as dc            # noqa: E402
from baselines import b3_crop_ladder as b3          # noqa: E402

FULL_MAX_PX = 2.5e6
OUT_PX = 1600
CROP_MULT = 3.0          # 裁剪边长 = 目标框长边 × 3
CROP_MIN = 240.0
CROP_MAX = 1600.0

COORD_MODE = "rel"
REASONING = "none"
BACKEND = "qwen"


# ---------------------------------------------------------------- 图片

def _fit(img, max_px=FULL_MAX_PX):
    k = min(1.0, (max_px / (img.width * img.height)) ** 0.5)
    if k >= 1.0:
        return img
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def _norm2px(box, W, H):
    return [box[0] / 1000 * W, box[1] / 1000 * H, box[2] / 1000 * W, box[3] / 1000 * H]


def _clamp_box(box):
    x1, y1, x2, y2 = [max(0.0, min(1000.0, float(v))) for v in box]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def render_full(orig, box=None):
    """整屏（可画框）。坐标一律 0-1000 归一化。"""
    img = _fit(orig)
    if box:
        b = _norm2px(box, img.width, img.height)
        img = b3.draw_box(img, *b)
    return img


def crop_side_px(orig, box, mult=CROP_MULT):
    """裁剪边长（原图像素）：目标框长边 × mult，钳制在 [240, 1600]。"""
    b = _norm2px(box, orig.width, orig.height)
    side = max(b[2] - b[0], b[3] - b[1]) * mult
    return max(CROP_MIN, min(CROP_MAX, side))


CURSOR_PX = 48.0


def _draw_cursor(img, x, y, size=CURSOR_PX):
    """在 (x, y) 画鼠标光标：箭头尖端正好在 (x, y)。白填充+黑描边，不加额外标记。"""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    s = float(size)
    pts = [(0.00, 0.00), (0.00, 0.62), (0.17, 0.50), (0.30, 0.82),
           (0.42, 0.76), (0.29, 0.45), (0.50, 0.45)]
    poly = [(x + px * s, y + py * s) for px, py in pts]
    d.polygon(poly, fill=(255, 255, 255), outline=(0, 0, 0))
    d.line(poly + [poly[0]], fill=(0, 0, 0), width=max(1, int(s * 0.04)))
    return img


def render_crop(orig, box, side):
    """以框中心裁 side×side，放大到 OUT_PX，画框。

    返回 (img, bounds_px)。bounds 可能超出图像范围（PIL 会补黑边），属正常。
    """
    b = _norm2px(box, orig.width, orig.height)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = side / 2.0
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    img = orig.crop((round(x0), round(y0), round(x1), round(y1)))
    img = img.resize((OUT_PX, OUT_PX), Image.LANCZOS)
    sx, sy = OUT_PX / (x1 - x0), OUT_PX / (y1 - y0)
    img = b3.draw_box(img, (b[0] - x0) * sx, (b[1] - y0) * sy, (b[2] - x0) * sx, (b[3] - y0) * sy)
    # 框中心画光标：校验器要判的就是「这个点落在什么上面」
    img = _draw_cursor(img, (cx - x0) * sx, (cy - y0) * sy)
    return img, (x0, y0, x1, y1)


def bounds_norm(bounds, W0, H0):
    """裁剪范围换算成整屏 0-1000 归一化。"""
    x0, y0, x1, y1 = bounds
    return (x0 / W0 * 1000, y0 / H0 * 1000, x1 / W0 * 1000, y1 / H0 * 1000)


# ---------------------------------------------------------------- 坐标换算

def rel_to_norm(rel_box, bounds, W0, H0):
    """裁剪图内 0-1000 → 整屏 0-1000。纯计算，可单测。"""
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    return _clamp_box([
        (x0 + rel_box[0] / 1000 * w) / W0 * 1000,
        (y0 + rel_box[1] / 1000 * h) / H0 * 1000,
        (x0 + rel_box[2] / 1000 * w) / W0 * 1000,
        (y0 + rel_box[3] / 1000 * h) / H0 * 1000,
    ])


def _num(v, default=None):
    """从任意嵌套的标量/数组里取出一个数（模型常写成 {"x1":[27],"y1":[[93]]}）。"""
    depth = 0
    while isinstance(v, (list, tuple)):
        if not v or depth > 5:
            return default
        v = v[0]
        depth += 1
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return default


def _parse_box(a):
    """从工具参数里取出 bbox，容忍几种常见畸形。"""
    v = a.get("bbox", a.get("box"))
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            import re
            nums = re.findall(r"-?\d+(?:\.\d+)?", v)
            v = [float(x) for x in nums[:4]] if len(nums) >= 4 else None
    if isinstance(v, dict):
        try:
            v = [v["x1"], v["y1"], v["x2"], v["y2"]]
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    b = [_num(x) for x in v]
    if any(x is None for x in b):
        return None
    return _clamp_box(b)


# ---------------------------------------------------------------- 工具定义

TOOL_LOCATE = {
    "type": "function",
    "function": {
        "name": "locate",
        "description": "给出目标在图中的位置，bbox 为 4 个 0-1000 的整数。",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox": {"type": "array", "items": {"type": "integer"},
                         "minItems": 4, "maxItems": 4,
                         "description": "目标框 [x1, y1, x2, y2]"},
            },
            "required": ["bbox"],
        },
    },
}

TOOL_REPORT = {
    "type": "function",
    "function": {
        "name": "report",
        "description": "报告校验结论。先描述蓝框里是什么，再判断它是不是目标。",
        "parameters": {
            "type": "object",
            "properties": {
                "in_box": {"type": "string",
                           "description": "蓝框里实际是什么？只描述看到的文字/图标/控件"},
                "on_cursor": {"type": "string",
                              "description": "光标尖端（蓝框中心那个箭头）正落在什么上面？只描述看到的"},
                "anchor_seen": {"type": "string",
                                "description": ("目标描述的「周边锚点」，在这张图里能看到吗？"
                                                "看到就写它是什么；看不到就写「看不到」。必填")},
                "center_ok": {"type": "string", "enum": ["yes", "no", "na"],
                              "description": ("光标尖端是否落在目标的【中心区域】？"
                                              "压在边缘/角落/某一半边 → no；目标都没找到 → na")},
                "match": {"type": "string", "enum": ["yes", "no", "unclear"],
                          "description": "蓝框里的东西是不是目标描述的元素"},
                "why": {"type": "string", "description": "判定依据，一句话"},
                "unclear_kind": {
                    "type": "string",
                    "enum": ["", "anchor_out_of_view", "too_small", "too_blurry", "ambiguous"],
                    "description": ("match=unclear 时必填。"
                                    "anchor_out_of_view=锚点不在图内无法核对；"
                                    "too_small=目标太小看不清；"
                                    "too_blurry=图像模糊/被遮挡；"
                                    "ambiguous=描述自相矛盾")},
                "elsewhere_dir": {"type": "string",
                                  "enum": ["", "left", "right", "up", "down", "inside"],
                                  "description": "match=no 且目标在图内别处时，填相对蓝框的方向"},
                "elsewhere_desc": {"type": "string",
                                   "description": "那个位置看起来是什么"},
            },
            "required": ["in_box", "on_cursor", "center_ok", "anchor_seen", "match", "why"],
        },
    },
}


# ---------------------------------------------------------------- 提示词

def _desc_block(d):
    """完整描述块（含周边锚点 env —— 这是区分同类元素的关键，不能丢）。"""
    out = []
    if d.get("text"):
        out.append(f"· 文字内容：{d['text']}")
    for k, label in (("feat", "外观特征"), ("ctrl", "控件类型"),
                     ("pos", "屏幕位置"), ("env", "周边锚点")):
        if d.get(k):
            out.append(f"· {label}：{d[k]}")
    return "\n".join(out) + "\n"


def prompt_locate(d, mode, bounds_n, is_full, hint=""):
    head = f"""请找出界面中符合以下描述的元素，给出它的位置。

【目标元素描述】
{_desc_block(d)}"""
    if mode == "rel":
        coord = """【坐标要求】只按【这张图】给坐标：横向 0 = 这张图左边缘，1000 = 右边缘；纵向 0 = 上边缘，1000 = 下边缘。
不需要知道整张屏幕有多大。"""
    else:
        if is_full:
            coord = """【坐标要求】按【整屏】给坐标：横向 0 = 整屏左边缘，1000 = 右边缘；纵向 0 = 上边缘，1000 = 下边缘。"""
        else:
            bx0, by0, bx1, by1 = bounds_n
            coord = (f"""【坐标要求】按【整屏】给坐标：横向 0 = 整屏左边缘，1000 = 右边缘；纵向 0 = 上边缘，1000 = 下边缘。
这张图是整屏的一块，覆盖整屏横向 {bx0:.0f}–{bx1:.0f}、纵向 {by0:.0f}–{by1:.0f}。
请把图内的位置换算成整屏坐标再报。""")
    tail = """框住的是【像素上可见的那块图形本身】，**紧贴它的边缘，四周不留多余空白**：
- 目标是一个图标 → 只框图标，**不要把图标旁边的文字标签框进来**
- 目标是一段文字 → 只框那段文字的像素
- 描述里提到的「周边锚点」是参照物，不属于目标，不要框进来

不需要像素级精确：框的中心落在元素上、四条边没有明显超出轮廓，就算合格。

请调用 locate(bbox=[x1, y1, x2, y2])。"""
    if hint:
        tail = f"【上一轮的情况】{hint}\n\n" + tail
    return head + "\n" + coord + "\n" + tail


PROMPT_VERIFY = """这张图是从屏幕上裁下来的一块（不是整屏）。
图中有两样东西：
- **蓝框**：候选区域
- **光标箭头**：画在蓝框的【正中心】，箭头尖端就是准备点击的位置

【目标元素描述】
{desc}
【任务】
① 看光标尖端正落在什么上面，填 on_cursor（只描述你看到的文字/图标/控件）。
② 看蓝框整体框住了什么，填 in_box。
③ 判断：**光标尖端落的东西**，是不是上面描述的目标元素。

【判定规则】
1. **以光标尖端为准**——它是实际点击位置。
   **尖端必须落在目标的【中心区域】**。如果尖端压在元素的边缘、角落、或只落在它的某一半边
   （例如"落在图标的右半部分"、"压在圆形左边缘"、"落在文字末尾几个字母上"），
   即使元素本身找对了，也要判 **no**，并在 why 里写明「尖端偏离中心」。
   原因：最终落点就是尖端，压在边缘等于没点中。
2. 外观特征要对上。
3. **必须核对周边锚点**：先回答 anchor_seen。
   - 看得到：写出它是什么，并核对相对位置是否一致；对不上判 no。
   - 看不到：填「看不到」，判 unclear + anchor_out_of_view，**不能假定它匹配**。
4. 外观像 A、锚点却指向 B（互相矛盾）→ 判 **no**（矛盾本身就说明位置可疑）。
5. **注意：蓝框在图中居中、光标在框中心，这是裁剪方式决定的，不能作为位置证据。**
6. 目标太小看不清 → unclear + too_small；图像模糊 → unclear + too_blurry。

【如果判 no】
- 目标就在这张图里、只是光标没点在它上面 → 填 elsewhere_dir（left/right/up/down/inside）和 elsewhere_desc。
- 这张图里根本看不到目标 → elsewhere_dir 留空。

请调用 report。
"""


# ---------------------------------------------------------------- 调用

def _b64(img):
    return qc._b64_png(img) if BACKEND == "qwen" else dc.b64_jpeg(img)


# DeepSeek 的思考 token 也算在 max_tokens 里，给少了会 finish_reason=length、工具调用被截断
_MT = {"qwen": 700, "deepseek": 6000}


CALL_TIMEOUT = 90        # 单次 HTTP 超时（原 300s，配合 retries=4 最坏 20 分钟）
CALL_RETRIES = 2         # 重试次数（最坏 3×90s = 4.5 分钟）


def _chat(parts, tools, max_tokens=None, retries=None):
    msgs = [{"role": "user", "content": parts}]
    mt = max_tokens or _MT.get(BACKEND, 1500)
    rt = CALL_RETRIES if retries is None else retries
    extra = {} if not tools else {"tools": tools, "tool_choice": "auto"}
    if BACKEND == "qwen":
        ex = dict(extra)
        ex["reasoning_effort"] = REASONING
        return qc.chat(msgs, max_tokens=mt, timeout=CALL_TIMEOUT, retries=rt, extra=ex)
    resp, _tries = dc.chat(msgs, max_tokens=mt, extra=extra or None)
    return resp


def _box_from_text(txt):
    """没有 tool_call 时的兜底：从自然语言里抠出 4 个数字。"""
    import re
    if not txt:
        return None
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
                  r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", txt)
    if not m:
        return None
    return _clamp_box([float(x) for x in m.groups()])


def _first_call(resp):
    ch = resp["choices"][0]
    msg = ch["message"]
    think = ((msg.get("reasoning_content") or ch.get("reasoning_content") or "")).strip()
    body = (msg.get("content") or "")
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None, {"_body": body}, think
    try:
        args = json.loads(tcs[0]["function"].get("arguments") or "{}")
    except Exception:  # noqa: BLE001
        args = {}
    return (tcs[0]["function"].get("name") or "").lower(), args, think


def locator(img, d, mode, *, bounds=None, W0=1, H0=1, is_full=True, hint="", retries=3):
    """返回 (整屏归一化 box 或 None, think)。"""
    bn = bounds_norm(bounds, W0, H0) if bounds else (0, 0, 1000, 1000)
    parts = [{"type": "text", "text": "【图】"},
             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}},
             {"type": "text", "text": prompt_locate(d, mode, bn, is_full, hint)}]
    for _ in range(retries):
        try:
            name, a, think = _first_call(_chat(parts, [TOOL_LOCATE]))
        except Exception:  # noqa: BLE001
            continue
        if name != "locate":
            continue
        raw = _parse_box(a)
        if raw is None:
            continue
        if mode == "rel" and bounds is not None:
            return rel_to_norm(raw, bounds, W0, H0), think
        return raw, think

    # 兜底：不发工具，让它用纯文本给数组（DeepSeek 偶尔不返回 tool_call）
    try:
        name, a, think = _first_call(_chat(
            parts + [{"type": "text", "text": "请只输出一个 JSON 数组 [x1, y1, x2, y2]，不要其他内容。"}],
            None))
        raw = _box_from_text(a.get("_body") or "")
        if raw is not None:
            if mode == "rel" and bounds is not None:
                return rel_to_norm(raw, bounds, W0, H0), think
            return raw, think
    except Exception:  # noqa: BLE001
        pass
    return None, ""


def verifier(img, d):
    """返回 (dict, think)。dict 含 match / in_box / unclear_kind / elsewhere_dir。"""
    parts = [{"type": "text", "text": "【裁剪图】"},
             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}},
             {"type": "text", "text": PROMPT_VERIFY.format(desc=_desc_block(d))}]
    for _ in range(3):
        try:
            name, a, think = _first_call(_chat(parts, [TOOL_REPORT]))
        except Exception:  # noqa: BLE001
            continue
        if name != "report":
            continue
        m = (a.get("match") or "").strip().lower()
        if m not in ("yes", "no", "unclear"):
            continue
        # 尖端偏离中心 → 一律按 no 处理（落点就是尖端，压在边缘等于没点中）
        if (a.get("center_ok") or "").strip().lower() == "no":
            m = "no"
        if not (a.get("in_box") or "").strip() or not (a.get("on_cursor") or "").strip():
            continue        # 没描述框里/中心是什么，结论不可信，重试
        return {
            "match": m,
            "in_box": (a.get("in_box") or "").strip()[:200],
            "on_cursor": (a.get("on_cursor") or "").strip()[:200],
            "center_ok": (a.get("center_ok") or "").strip().lower(),
            "anchor_seen": (a.get("anchor_seen") or "").strip()[:200],
            "why": (a.get("why") or "").strip()[:200],
            "unclear_kind": (a.get("unclear_kind") or "").strip().lower(),
            "elsewhere_dir": (a.get("elsewhere_dir") or "").strip().lower(),
            "elsewhere_desc": (a.get("elsewhere_desc") or "").strip()[:150],
        }, think
    return {"match": "unclear", "in_box": "", "why": "校验器无有效输出",
            "unclear_kind": "too_blurry", "elsewhere_dir": "", "elsewhere_desc": ""}, ""


# ---------------------------------------------------------------- 主循环

def run_one(rid, src, d, *, max_locate=3, max_verify_retry=2, mode=None):
    mode = mode or COORD_MODE
    orig = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
    W0, H0 = orig.size
    gt = src["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    t0 = time.time()

    trace = []
    box = None
    confidence = "low"
    stop_reason = "rounds_exhausted"

    # —— ① 首次定位：整屏 ——
    img_full = render_full(orig)
    box, think = locator(img_full, d, mode, is_full=True)
    trace.append({"stage": "locate", "round": 1, "where": "full", "box": box, "think": think[:1200]})
    if box is None:
        stop_reason = "locate_failed"

    # —— ② 校验 → ③ 决策 循环 ——
    locate_rounds = 0
    while box is not None and locate_rounds < max_locate:
        locate_rounds += 1
        side = crop_side_px(orig, box)
        v_retry = 0
        v = None
        crop_img = bounds = None
        while True:
            crop_img, bounds = render_crop(orig, box, side)
            v, vthink = verifier(crop_img, d)
            trace.append({"stage": "verify", "round": locate_rounds, "crop_side": round(side),
                          "verdict": v, "think": vthink[:1200]})
            if v["match"] == "yes":
                confidence, stop_reason = "high", "verified"
                break
            if v["match"] == "unclear":
                kind = v["unclear_kind"]
                if kind == "anchor_out_of_view" and v_retry < max_verify_retry and side < CROP_MAX * 2:
                    side = min(side * 2, max(orig.width, orig.height) * 1.5)
                    v_retry += 1
                    continue
                if kind == "too_small" and v_retry < max_verify_retry and side > CROP_MIN / 2:
                    side = max(side / 2, 120.0)
                    v_retry += 1
                    continue
                if kind == "too_blurry":
                    confidence, stop_reason = "low", "too_blurry"
                    break
                # ambiguous 或重试用尽 → 按 no 处理
            break
        if stop_reason in ("verified", "too_blurry"):
            break

        # —— 判 no：决定在哪儿重新定位 ——
        if v.get("center_ok") == "no" and v["match"] == "no":
            hint = (f"元素基本找对了（尖端落在【{v['on_cursor']}】），"
                    f"但**尖端偏离了元素中心**（压在边缘/半边）。"
                    f"请把框收紧并居中到该元素的正中心，不要偏向任何一侧。")
        else:
            hint = f"你上次给的框里实际是【{v['in_box']}】，不是目标。"
        if v.get("elsewhere_dir"):
            hint += f"目标可能在框的 {v['elsewhere_dir']} 方向，{v.get('elsewhere_desc') or ''}。"
        else:
            hint += "那一带也看不到目标。"

        if v.get("elsewhere_dir") and crop_img is not None:
            # 有界纠偏：在裁剪图内重新定位
            new_box, think = locator(crop_img, d, mode, bounds=bounds, W0=W0, H0=H0,
                                     is_full=False, hint=hint)
            where = "crop"
        else:
            # 无界：回整屏
            new_box, think = locator(render_full(orig, box), d, mode, is_full=True, hint=hint)
            where = "full"

        trace.append({"stage": "locate", "round": locate_rounds + 1, "where": where,
                      "box": new_box, "think": think[:1200]})
        if new_box is None:
            stop_reason = "locate_failed"
            break
        # 无进展防护：几乎没动就升级到整屏重定位
        if all(abs(a - b) <= 10 for a, b in zip(new_box, box)) and where == "crop":
            new_box, think = locator(render_full(orig, box), d, mode, is_full=True, hint=hint)
            trace.append({"stage": "locate", "round": locate_rounds + 1, "where": "full_escalated",
                          "box": new_box, "think": think[:1200]})
            if new_box is None:
                stop_reason = "locate_failed"
                break
        box = new_box

    if box is None:
        fx = fy = -1.0
    else:
        fx = (box[0] + box[2]) / 2 / 1000 * W0
        fy = (box[1] + box[3]) / 2 / 1000 * H0

    return {
        "id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
        "final_px": [round(fx), round(fy)],
        "final_err": round(((fx - gcx) ** 2 + (fy - gcy) ** 2) ** 0.5, 1),
        "hit": gt[0] <= fx <= gt[2] and gt[1] <= fy <= gt[3],
        "box": box, "confidence": confidence, "stop_reason": stop_reason,
        "coord_mode": mode, "locate_rounds": locate_rounds,
        "latency_s": round(time.time() - t0, 1),
        "model": (qc.MODEL if BACKEND == "qwen" else dc.MODEL),
        "trace": trace,
    }


# ---------------------------------------------------------------- CLI

def main():
    global COORD_MODE, REASONING, BACKEND, CROP_MULT, CROP_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--split", default="M3")
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--coord-mode", default="rel", choices=["rel", "abs"])
    ap.add_argument("--max-locate", type=int, default=3)
    ap.add_argument("--max-verify-retry", type=int, default=2)
    ap.add_argument("--crop-mult", type=float, default=CROP_MULT)
    ap.add_argument("--crop-min", type=float, default=CROP_MIN)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--tag", default="b7")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    COORD_MODE, REASONING, BACKEND = args.coord_mode, args.reasoning_effort, args.backend
    CROP_MULT, CROP_MIN = args.crop_mult, args.crop_min
    b3.BACKEND = args.backend

    rows = {r["id"]: r for r in H.load_all()}
    DESC = json.load(open(f"descriptions/{args.desc_version}/{args.split}.json", encoding="utf-8"))
    ids = [x for x in (args.ids or "").split(",") if x] or list(rows)
    if args.limit:
        ids = ids[:args.limit]

    out = f"baselines/results/raw/{args.tag}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[B7] coord={args.coord_mode}  样本 {len(ids)}  max_locate={args.max_locate}  输出 {out}")
    with open(out, "w", encoding="utf-8") as f:
        for i, rid in enumerate(ids, 1):
            if rid not in rows or rid not in DESC:
                continue
            try:
                rec = run_one(rid, rows[rid], DESC[rid], max_locate=args.max_locate,
                              max_verify_retry=args.max_verify_retry)
            except Exception as e:  # noqa: BLE001
                rec = {"id": rid, "hit": False, "final_err": -1, "stop_reason": "error",
                       "error": str(e)[:200], "coord_mode": args.coord_mode}
            rec["tag"] = args.tag
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            flag = "✅" if rec.get("hit") else "❌"
            print(f"  [{i:>3}] {rid:<30} {rec.get('stop_reason','?'):<18} "
                  f"{rec.get('final_err','?')}px {rec.get('latency_s','?')}s {flag}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
