"""B6：光标点方案（定位与校验解耦，输出从「框」改成「点」）

动机
  评测口径是「预测点是否落在 GT 框内」。B5 让模型输出 bbox，再用框中心当落点——
  中间隔了一层，且实测命中样本的框 IoU 中位只有 0.66，框本来就没框准。
  B6 直接让模型回答「鼠标该点在哪里」，图里画的是鼠标光标（尖端=落点）。

与 B5 的差异（流程完全一致）
  - 工具：locate(x, y)，两个 0-1000 整数
  - 图上画的是光标箭头（白箭头+黑描边），不是矩形框
  - 校验器判断「光标尖端落在什么上面」，与评测口径一致
  - 裁剪边长不再依赖框尺寸：从 400px 起，由校验器反馈自适应（太大→缩小，锚点看不见→扩大）

用法
  python3 -m baselines.b6_cursor --ids "id1,id2" --split M3 --coord-mode rel --tag b6_rel
"""
import argparse
import json
import os
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import dataset as H                      # noqa: E402
from bench.compat import qwen_client as qc          # noqa: E402
from bench.compat import ds_client as dc            # noqa: E402

FULL_MAX_PX = 2.5e6
OUT_PX = 1600
CURSOR_PX = 48.0         # 裁剪图(1600px)里光标的高度
CROP_START = 400.0       # 起始裁剪边长（原图像素）
CROP_MIN = 120.0
CROP_MAX_MULT = 2.0      # 相对整屏短边的上限倍数

COORD_MODE = "rel"
REASONING = "none"
BACKEND = "qwen"


# ---------------------------------------------------------------- 图片

def _fit(img, max_px=None):
    max_px = FULL_MAX_PX if max_px is None else max_px
    k = min(1.0, (max_px / (img.width * img.height)) ** 0.5)
    if k >= 1.0:
        return img
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def _draw_cursor(img, x, y, size):
    """在 (x, y) 画鼠标光标：箭头尖端正好在 (x, y)。不加额外标记，让模型识别光标本身。

    size = 箭头总高度（像素）。箭头朝右下展开，尖端在左上角。
    """
    d = ImageDraw.Draw(img)
    s = float(size)
    # 经典箭头轮廓（相对尖端，单位 = s）
    pts = [(0.00, 0.00), (0.00, 0.62), (0.17, 0.50), (0.30, 0.82),
           (0.42, 0.76), (0.29, 0.45), (0.50, 0.45)]
    poly = [(x + px * s, y + py * s) for px, py in pts]
    d.polygon(poly, fill=(255, 255, 255), outline=(0, 0, 0))
    # 描边加粗：再画一次线
    d.line(poly + [poly[0]], fill=(0, 0, 0), width=max(1, int(s * 0.04)))
    return img


def render_full(orig, pt=None):
    img = _fit(orig)
    if pt is not None:
        x = pt[0] / 1000 * img.width
        y = pt[1] / 1000 * img.height
        img = _draw_cursor(img, x, y, size=max(24.0, min(64.0, img.width * 0.021)))
    return img


def render_crop(orig, pt, side):
    """以落点为中心裁 side×side，放大到 OUT_PX，画光标。返回 (img, bounds_px)。"""
    cx = pt[0] / 1000 * orig.width
    cy = pt[1] / 1000 * orig.height
    half = side / 2.0
    x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    img = orig.crop((round(x0), round(y0), round(x1), round(y1)))
    img = img.resize((OUT_PX, OUT_PX), Image.LANCZOS)
    sx, sy = OUT_PX / (x1 - x0), OUT_PX / (y1 - y0)
    img = _draw_cursor(img, (cx - x0) * sx, (cy - y0) * sy, size=CURSOR_PX)
    return img, (x0, y0, x1, y1)


def bounds_norm(bounds, W0, H0):
    x0, y0, x1, y1 = bounds
    return (x0 / W0 * 1000, y0 / H0 * 1000, x1 / W0 * 1000, y1 / H0 * 1000)


# ---------------------------------------------------------------- 坐标换算

def _clamp_pt(x, y):
    return [max(0.0, min(1000.0, float(x))), max(0.0, min(1000.0, float(y)))]


def rel_to_norm(rel_pt, bounds, W0, H0):
    """裁剪图内 0-1000 的点 → 整屏 0-1000。纯计算，可单测。"""
    x0, y0, x1, y1 = bounds
    fx = x0 + rel_pt[0] / 1000 * (x1 - x0)
    fy = y0 + rel_pt[1] / 1000 * (y1 - y0)
    return _clamp_pt(fx / W0 * 1000, fy / H0 * 1000)


def _num(v, default=None):
    """从任意嵌套的标量/数组里取出一个数。模型常写成 {"x":[495],"y":[[442,442]]}。"""
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


def _parse_point(a):
    """从工具参数里取出点，容忍几种常见畸形。"""
    for k in ("x", "px"):
        if k in a:
            nx, ny = _num(a.get("x")), _num(a.get("y"))
            if nx is None or ny is None:
                return None
            return _clamp_pt(nx, ny)
    v = a.get("point", a.get("xy"))
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            import re
            nums = re.findall(r"-?\d+(?:\.\d+)?", v)
            v = [float(n) for n in nums[:2]] if len(nums) >= 2 else None
    if isinstance(v, dict):
        nx, ny = _num(v.get("x")), _num(v.get("y"))
        return None if nx is None or ny is None else _clamp_pt(nx, ny)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        nx, ny = _num(v[0]), _num(v[1])
        return None if nx is None or ny is None else _clamp_pt(nx, ny)
    return None


def _point_from_text(txt):
    import re
    if not txt:
        return None
    m = re.search(r'"?x"?\s*[:=]\s*(-?\d+(?:\.\d+)?)[,\s]+"?y"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', txt)
    if m:
        return _clamp_pt(float(m.group(1)), float(m.group(2)))
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", txt)
    if m:
        return _clamp_pt(float(m.group(1)), float(m.group(2)))
    return None


# ---------------------------------------------------------------- 工具

TOOL_POINT = {
    "type": "function",
    "function": {
        "name": "locate",
        "description": "给出鼠标该点的位置，x/y 为 0-1000 的整数。",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "横向 0-1000"},
                "y": {"type": "integer", "description": "纵向 0-1000"},
            },
            "required": ["x", "y"],
        },
    },
}

TOOL_REPORT = {
    "type": "function",
    "function": {
        "name": "report",
        "description": "报告校验结论：先描述光标尖端落在什么上面，再判断它是不是目标。",
        "parameters": {
            "type": "object",
            "properties": {
                "on_cursor": {"type": "string",
                              "description": "光标尖端正落在什么上面？只描述看到的"},
                "anchor_seen": {"type": "string",
                                "description": ("目标描述的「周边锚点」在这张图里能看到吗？"
                                                "看到写它是什么，看不到写「看不到」。必填")},
                "match": {"type": "string", "enum": ["yes", "no", "unclear"]},
                "why": {"type": "string", "description": "判定依据，一句话"},
                "unclear_kind": {
                    "type": "string",
                    "enum": ["", "anchor_out_of_view", "too_small", "too_large",
                             "too_blurry", "ambiguous"],
                    "description": ("match=unclear 时必填。anchor_out_of_view=锚点不在图内；"
                                    "too_small=目标太小看不清；too_large=目标比这张图还大；"
                                    "too_blurry=图像模糊；ambiguous=描述自相矛盾")},
                "elsewhere_dir": {"type": "string",
                                  "enum": ["", "left", "right", "up", "down", "inside"],
                                  "description": "match=no 且目标在图内别处时，填相对光标的方向"},
                "elsewhere_desc": {"type": "string"},
            },
            "required": ["on_cursor", "anchor_seen", "match", "why"],
        },
    },
}


# ---------------------------------------------------------------- 提示词

def _desc_block(d):
    out = []
    if d.get("text"):
        out.append(f"· 文字内容：{d['text']}")
    for k, label in (("feat", "外观特征"), ("ctrl", "控件类型"),
                     ("pos", "屏幕位置"), ("env", "周边锚点")):
        if d.get(k):
            out.append(f"· {label}：{d[k]}")
    return "\n".join(out) + "\n"


def prompt_locate_simple(d, mode, bounds_n, is_full):
    """简化版：去掉位置/锚点等可能干扰的信息，只给外观，用于首次定位失败后的重试。"""
    feat = " ".join(x for x in (d.get("feat"), d.get("ctrl")) if x)
    if d.get("text"):
        feat = f"文字「{d['text']}」，{feat}"
    if mode == "rel":
        coord = "只按【这张图】给坐标：横向 0=左边缘，1000=右边缘；纵向同理。"
    else:
        coord = "按【整屏】给坐标：横向 0=左边缘，1000=右边缘；纵向同理。"
    return (f"在图中找到这个元素：{feat}\n{coord}\n"
            f"给出它中心的一个点，调用 locate(x=..., y=...)。")


def prompt_locate(d, mode, bounds_n, is_full, hint=""):
    head = f"""请找出界面中符合以下描述的元素，并给出【鼠标应该点在哪里】。

【目标元素描述】
{_desc_block(d)}"""
    if mode == "rel":
        coord = """【坐标要求】只按【这张图】给坐标：横向 0 = 这张图左边缘，1000 = 右边缘；纵向 0 = 上边缘，1000 = 下边缘。
不需要知道整张屏幕有多大。"""
    else:
        if is_full:
            coord = "【坐标要求】按【整屏】给坐标：横向 0 = 左边缘，1000 = 右边缘；纵向 0 = 上边缘，1000 = 下边缘。"
        else:
            bx0, by0, bx1, by1 = bounds_n
            coord = (f"【坐标要求】按【整屏】给坐标。这张图覆盖整屏横向 {bx0:.0f}–{bx1:.0f}、"
                     f"纵向 {by0:.0f}–{by1:.0f}，请换算成整屏坐标再报。")
    tail = """【要点】给出的是【一个点】，不是框。
**必须点在元素的几何中心**：元素有多宽、多高，就点它的正中间。
- 元素是「图标+文字」的组合（比如按钮）→ 点在图标和文字整体的中心，不要只点文字
- 元素是细长条 → 点在它的中点，不要偏向一端
- 元素很小 → 更要点准，点它的正中心

请调用 locate(x=..., y=...)。"""
    if hint:
        tail = f"【上一轮的情况】{hint}\n\n" + tail
    return head + "\n" + coord + "\n" + tail


PROMPT_VERIFY = """这张图是从屏幕上裁下来的一块（不是整屏）。
图中有一个鼠标光标：**白色箭头、黑色描边**。箭头【尖端】（箭头最顶端的尖角）就是准备点击的位置。

【目标元素描述】
{desc}
【任务】
① 看光标尖端正落在什么上面，填 on_cursor（只描述你看到的文字/图标/控件）。
② 判断光标尖端落的东西是不是上面描述的目标元素。

【判定规则】
1. 外观特征要对上。
2. 必须核对周边锚点：先填 anchor_seen。
   - 看得到 → 写出它是什么，并核对相对位置是否一致；对不上判 no。
   - 看不到 → 填「看不到」并判 unclear + anchor_out_of_view，不能假定它匹配。
3. 外观像 A、锚点却指向 B（矛盾）→ unclear + ambiguous。
4. 目标比这张图还大、看不全 → unclear + too_large。
5. 目标太小看不清 → unclear + too_small；图像模糊 → unclear + too_blurry。

【如果判 no】
- 目标就在这张图里、只是光标没点在它上面 → 填 elsewhere_dir（left/right/up/down/inside）和 elsewhere_desc。
- 这张图里根本看不到目标 → elsewhere_dir 留空。

请调用 report。
"""


# ---------------------------------------------------------------- 调用

_MT = {"qwen": 700, "deepseek": 6000}


def _b64(img):
    return qc._b64_png(img) if BACKEND == "qwen" else dc.b64_jpeg(img)


def _chat(parts, tools, max_tokens=None, retries=4):
    msgs = [{"role": "user", "content": parts}]
    mt = max_tokens or _MT.get(BACKEND, 1500)
    extra = {} if not tools else {"tools": tools, "tool_choice": "auto"}
    if BACKEND == "qwen":
        ex = dict(extra)
        ex["reasoning_effort"] = REASONING
        return qc.chat(msgs, max_tokens=mt, timeout=300, retries=retries, extra=ex)
    resp, _tries = dc.chat(msgs, max_tokens=mt, extra=extra or None)
    return resp


def _first_call(resp):
    ch = resp["choices"][0]
    msg = ch["message"]
    think = ((msg.get("reasoning_content") or ch.get("reasoning_content") or "")).strip()
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None, {"_body": (msg.get("content") or "")}, think
    try:
        args = json.loads(tcs[0]["function"].get("arguments") or "{}")
    except Exception:  # noqa: BLE001
        args = {}
    return (tcs[0]["function"].get("name") or "").lower(), args, think


def locator(img, d, mode, *, bounds=None, W0=1, H0=1, is_full=True, hint="", retries=3):
    bn = bounds_norm(bounds, W0, H0) if bounds else (0, 0, 1000, 1000)
    parts = [{"type": "text", "text": "【图】"},
             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}},
             {"type": "text", "text": prompt_locate(d, mode, bn, is_full, hint)}]
    for _ in range(retries):
        try:
            name, a, think = _first_call(_chat(parts, [TOOL_POINT]))
        except Exception:  # noqa: BLE001
            continue
        if name != "locate":
            continue
        raw = _parse_point(a)
        if raw is None:
            continue
        if mode == "rel" and bounds is not None:
            return rel_to_norm(raw, bounds, W0, H0), think
        return raw, think
    # 兜底一：换简化提示词（去掉可能干扰的 pos/env），再试一轮
    try:
        simple = [{"type": "text", "text": "【图】"},
                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}},
                  {"type": "text", "text": prompt_locate_simple(d, mode, bn, is_full)}]
        for _ in range(2):
            name, a, think = _first_call(_chat(simple, [TOOL_POINT]))
            if name != "locate":
                continue
            raw = _parse_point(a)
            if raw is None:
                continue
            if mode == "rel" and bounds is not None:
                return rel_to_norm(raw, bounds, W0, H0), think
            return raw, think
    except Exception:  # noqa: BLE001
        pass

    # 兜底二：不发工具，纯文本给坐标
    try:
        name, a, think = _first_call(_chat(
            parts + [{"type": "text", "text": "请只输出 JSON：{\"x\": 数字, \"y\": 数字}"}], None))
        raw = _point_from_text(a.get("_body") or "")
        if raw is not None:
            if mode == "rel" and bounds is not None:
                return rel_to_norm(raw, bounds, W0, H0), think
            return raw, think
    except Exception:  # noqa: BLE001
        pass
    return None, ""


def verifier(img, d):
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
        if not (a.get("on_cursor") or "").strip():
            continue
        return {
            "match": m,
            "on_cursor": (a.get("on_cursor") or "").strip()[:200],
            "anchor_seen": (a.get("anchor_seen") or "").strip()[:200],
            "why": (a.get("why") or "").strip()[:200],
            "unclear_kind": (a.get("unclear_kind") or "").strip().lower(),
            "elsewhere_dir": (a.get("elsewhere_dir") or "").strip().lower(),
            "elsewhere_desc": (a.get("elsewhere_desc") or "").strip()[:150],
        }, think
    return {"match": "unclear", "on_cursor": "", "anchor_seen": "", "why": "校验器无有效输出",
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
    confidence, stop_reason = "low", "rounds_exhausted"
    cands = []          # [(score, idx, pt)]，score 越高越可信
    pt = None
    side = CROP_START

    img_full = render_full(orig)
    pt, think = locator(img_full, d, mode, is_full=True)
    trace.append({"stage": "locate", "round": 1, "where": "full", "pt": pt, "think": think[:1200]})
    if pt is None:
        stop_reason = "locate_failed"

    rounds = 0
    while pt is not None and rounds < max_locate:
        rounds += 1
        v_retry = 0
        crop_img = bounds = None
        v = None
        while True:
            crop_img, bounds = render_crop(orig, pt, side)
            v, vthink = verifier(crop_img, d)
            trace.append({"stage": "verify", "round": rounds, "crop_side": round(side),
                          "verdict": v, "think": vthink[:1200]})
            if v["match"] == "yes":
                confidence, stop_reason = "high", "verified"
                break
            if v["match"] == "unclear":
                kind = v["unclear_kind"]
                if kind in ("anchor_out_of_view", "too_large") and v_retry < max_verify_retry:
                    side = min(side * 2, max(W0, H0) * CROP_MAX_MULT)
                    v_retry += 1
                    continue
                if kind == "too_small" and v_retry < max_verify_retry and side > CROP_MIN / 2:
                    side = max(side / 2, CROP_MIN / 2)
                    v_retry += 1
                    continue
                if kind == "too_blurry":
                    confidence, stop_reason = "low", "too_blurry"
                    break
            break
        if stop_reason in ("verified", "too_blurry"):
            break

        # 候选分级：说 no 但目标在图内可见(3) > 说 no(2) > unclear(1)
        _sc = 3 if (v["match"] == "no" and v.get("elsewhere_dir")) else \
              (2 if v["match"] == "no" else 1)
        cands.append((_sc, len(cands), pt))
        hint = f"你上次给的点落在【{v['on_cursor']}】上，不是目标。"
        if v.get("elsewhere_dir"):
            hint += f"目标可能在光标的 {v['elsewhere_dir']} 方向，{v.get('elsewhere_desc') or ''}。"
        else:
            hint += "那一带也看不到目标。"

        if v.get("elsewhere_dir") and crop_img is not None:
            new_pt, think = locator(crop_img, d, mode, bounds=bounds, W0=W0, H0=H0,
                                    is_full=False, hint=hint)
            where = "crop"
        else:
            new_pt, think = locator(render_full(orig, pt), d, mode, is_full=True, hint=hint)
            where = "full"
        trace.append({"stage": "locate", "round": rounds + 1, "where": where,
                      "pt": new_pt, "think": think[:1200]})
        if new_pt is None:
            stop_reason = "locate_failed"
            break
        _stuck = all(abs(a - b) <= 10 for a, b in zip(new_pt, pt))
        # 震荡检测：新点与更早的任一点几乎重合，说明在原地打转
        _hist = [t["pt"] for t in trace if t["stage"] == "locate" and t.get("pt")]
        _osc = any(all(abs(a - b) <= 10 for a, b in zip(new_pt, h)) for h in _hist[:-1])
        if (_stuck or _osc) and where == "crop":
            new_pt, think = locator(render_full(orig, pt), d, mode, is_full=True, hint=hint)
            trace.append({"stage": "locate", "round": rounds + 1, "where": "full_escalated",
                          "pt": new_pt, "think": think[:1200]})
            if new_pt is None:
                stop_reason = "locate_failed"
                break
            if _osc and all(abs(a - b) <= 10 for a, b in zip(new_pt, pt)):
                stop_reason = "oscillating"
                break
        pt = new_pt

    if stop_reason != "verified" and cands:
        # 分级最高者优先；同分取【最新】的（重定位整体在改进）
        cands.sort(key=lambda c: (-c[0], -c[1]))
        pt = cands[0][2]
        stop_reason += f"_best{cands[0][0]}"
    if pt is None:
        _valid = [t["pt"] for t in trace if t["stage"] == "locate" and t.get("pt")]
        pt = _valid[-1] if _valid else None
    fx, fy = (-1.0, -1.0) if pt is None else (pt[0] / 1000 * W0, pt[1] / 1000 * H0)
    return {
        "id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
        "final_px": [round(fx), round(fy)],
        "final_err": round(((fx - gcx) ** 2 + (fy - gcy) ** 2) ** 0.5, 1),
        "hit": gt[0] <= fx <= gt[2] and gt[1] <= fy <= gt[3],
        "pt": pt, "confidence": confidence, "stop_reason": stop_reason,
        "coord_mode": mode, "locate_rounds": rounds,
        "latency_s": round(time.time() - t0, 1),
        "model": (qc.MODEL if BACKEND == "qwen" else dc.MODEL),
        "trace": trace,
    }


# ---------------------------------------------------------------- CLI

def main():
    global COORD_MODE, REASONING, BACKEND, CROP_START
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--split", default="M3")
    ap.add_argument("--desc-version", default="v2")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--coord-mode", default="rel", choices=["rel", "abs"])
    ap.add_argument("--max-locate", type=int, default=3)
    ap.add_argument("--max-verify-retry", type=int, default=2)
    ap.add_argument("--crop-start", type=float, default=CROP_START)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--tag", default="b6")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    COORD_MODE, REASONING, BACKEND = args.coord_mode, args.reasoning_effort, args.backend
    CROP_START = args.crop_start

    rows = {r["id"]: r for r in H.load_all()}
    DESC = json.load(open(f"descriptions/{args.desc_version}/{args.split}.json", encoding="utf-8"))
    ids = [x for x in (args.ids or "").split(",") if x] or list(rows)
    if args.limit:
        ids = ids[:args.limit]

    out = f"baselines/results/raw/{args.tag}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[B6] coord={args.coord_mode} 样本 {len(ids)} crop_start={CROP_START} → {out}")
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
