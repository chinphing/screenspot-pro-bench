"""DeepSeek 多轮定位智能体 v2 —— 5 方向离散缩放 + 回退。

相对 v1 的改动（按用户确认的架构）：

  ① 缩放离散化：模型不能自由给框，只能从 上/下/左/右/中 五选一，
     每次裁【当前画面的 1/2 区域】（每边取一半 → 面积 1/4 → 放大 2×）。
     目的：修掉 v1 中"模型要的区域中位只占原图 0.4%、没有上下文"的问题。

  ② 逐层深入：每次缩放基于【上一张图】的坐标系，不是原图。
     模型永远只面对"当前这一层"，不用在脑子里维护多个坐标系。

  ③ 新增回退工具 back()：走错方向可以退回上一层。
     历史对话【不动】（保留完整轨迹），只是把上一层的图重新放到最新位置。

坐标换算【完全交给代码】：每层记录它在原图中的归一化范围，模型只需输出
【当前层】的归一化 bbox，逐层反推回原图。

所有层（含原图）统一缩放到长边 = crop_side，保证每步严格放大 2×。

用法：
  python3 ds_agent2.py --ids a,b,c --desc-file xxx.json --tag t1 \
      [--max-turns 5] [--crop-side 1200] [--reasoning-effort none] [--force-answer-last]
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

DIRECTIONS = ["上", "下", "左", "右", "中"]

TOOL_ZOOM = {
    "type": "function",
    "function": {
        "name": "zoom",
        "description": (
            "放大查看【当前画面】的某个位置。你给出一个坐标点，我会以该点为中心"
            "裁一块区域放大给你看。"
            "当你觉得目标太小看不清、或者不确定哪个是目标时使用。"
            "坐标用【当前画面】的归一化值（0-1000，0=当前画面左/上边，1000=右/下边）。"
            "如果你已经大致知道目标在哪，就填那个位置；如果只是想整体看细一点，填 500,500。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "要放大的中心点 x（当前画面归一化 0-1000）"},
                "y": {"type": "integer", "description": "要放大的中心点 y（当前画面归一化 0-1000）"},
                "reason": {"type": "string", "description": "为什么要放大这里（一句话）"},
            },
            "required": ["x", "y"],
        },
    },
}

TOOL_BACK = {
    "type": "function",
    "function": {
        "name": "back",
        "description": (
            "回退到上一层画面（撤销上一次放大）。"
            "当你发现放大方向走错了、目标并不在当前画面里时使用。"
            "回退后你会重新看到上一层画面。"
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "为什么要回退"}},
            "required": [],
        },
    },
}

PROMPT = """你是一个界面元素定位助手。下面第 1 张图是【完整原图】的当前视图。

【任务】
找出下面描述的元素，给出它在【当前画面】里的 bbox。

【你可以做三件事，每轮选一件】
· 放大（zoom）：给一个坐标点 (x, y)，我会以该点为中心裁一块放大给你看。
  坐标用【当前画面】的归一化值（0-1000）。最多放大 {MAXZ} 次。
· 回退（back）：撤销上一次放大，回到上一层画面。走错方向时用。
· 直接作答：如果已经看清了，直接输出 bbox 数组，格式 [x1,y1,x2,y2]，
  只输出这个数组，不要别的文字。

【重要】
· 输出的 bbox 用【当前画面】的归一化坐标（0-1000）。
  不需要换算成原图坐标，换算我会自己做。
· 每轮我都会告诉你当前画面对应原图的哪个范围。
· 放大时请把你认为目标所在的位置填进去（越准越好），
  如果你只是想让整体更清楚一点，填 500,500 即可。

【目标元素描述】
{desc}"""

def _to_long_side(img, target):
    w, h = img.size
    if max(w, h) == target:
        return img
    k = target / max(w, h)
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def _crop_around(cur, px, py, disp_w, disp_h, cap=600):
    """以当前画面归一化坐标 (px,py) 为中心裁一块。

    尺寸 = 当前画面长宽的 1/2，但每边不超过 cap 像素（否则放大后看不出效果）。
    超出当前画面时贴边修正。返回原图归一化范围。
    """
    cw = min(0.5 * disp_w, cap)
    ch = min(0.5 * disp_h, cap)
    cx, cy = px / 1000 * disp_w, py / 1000 * disp_h
    x0, x1 = cx - cw / 2, cx + cw / 2
    y0, y1 = cy - ch / 2, cy + ch / 2
    if x0 < 0:
        x1 -= x0; x0 = 0.0
    if x1 > disp_w:
        x0 -= (x1 - disp_w); x1 = float(disp_w)
    if y0 < 0:
        y1 -= y0; y0 = 0.0
    if y1 > disp_h:
        y0 -= (y1 - disp_h); y1 = float(disp_h)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    nw, nh = cur[2] - cur[0], cur[3] - cur[1]
    return [cur[0] + x0 / disp_w * nw, cur[1] + y0 / disp_h * nh,
            cur[0] + x1 / disp_w * nw, cur[1] + y1 / disp_h * nh]


def _desc_lines(d):
    env = f"· 周边锚点：{d['env']}\n" if d.get("env") else ""
    txt = f"· 文字内容：{d['text']}\n" if d.get("text") else ""
    return (f"{txt}"
            f"· 外观特征：{d['feat']}\n"
            f"· 控件类型：{d['ctrl']}\n"
            f"· 屏幕位置：{d['pos']}\n"
            f"{env}")


def _img_block(img):
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{dc.b64_jpeg(img)}"}}


def run_one(rid, src, d, *, max_turns=5, crop_side=1200, reasoning_effort="none",
            force_answer_last=False, max_tokens=4000, tries=10, crop_cap=600):
    orig = Image.open(H.fetch_image(src["img_filename"])).convert("RGB")
    W0, H0 = orig.size
    gt = src["bbox"]
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2

    def render(region):
        """按原图归一化区域裁片（从原图取样，避免逐层插值损失），再统一缩到 crop_side。"""
        px0 = int(region[0] / 1000 * W0); py0 = int(region[1] / 1000 * H0)
        px1 = int(region[2] / 1000 * W0); py1 = int(region[3] / 1000 * H0)
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(W0, max(px0 + 1, px1)), min(H0, max(py0 + 1, py1))
        return _to_long_side(orig.crop((px0, py0, px1, py1)), crop_side)

    stack = [[0.0, 0.0, 1000.0, 1000.0]]          # 各层在原图中的归一化范围
    img0 = render(stack[-1])
    messages = [{"role": "user", "content": [
        {"type": "text",
         "text": f"【当前画面】完整原图，尺寸 {img0.width} x {img0.height} 像素，"
                 f"对应原图范围 [0,0,1000,1000]（即整张原图）。"},
        _img_block(img0),
        {"type": "text",
         "text": PROMPT.format(MAXZ=max_turns - 1, desc=_desc_lines(d))},
    ]}]

    trace = []
    final_bbox = None
    t0 = time.time()
    n_zoom = 0

    for turn in range(1, max_turns + 1):
        last = (turn == max_turns)
        extra = {"reasoning_effort": reasoning_effort}
        if last and force_answer_last:
            # 最后一轮：不给工具，并追加一条强制作答指令
            messages.append({"role": "user", "content": [{"type": "text", "text":
                "【最后一轮】不能再放大了，也不能再回退了。"
                "请立刻给出你的答案：直接输出 bbox 数组 [x1,y1,x2,y2]，"
                "用当前画面的归一化坐标（0-1000）。"
                "只输出这个数组，不要写任何解释、不要写任何其他文字。"
                "即使不确定，也给出你认为最可能的位置。"}]})
        else:
            extra["tools"] = [TOOL_ZOOM, TOOL_BACK]
            extra["tool_choice"] = "auto"
        resp, _ = dc.chat(messages, max_tokens=max_tokens, tries=tries, extra=extra)
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []

        if not tcs:
            txt = (msg.get("content") or "").strip()
            m = NUM4.findall(txt)
            trace.append({"turn": turn, "action": "answer", "text": txt[:160]})
            if m:
                nb = [int(v) for v in m[-1]]
                cur = stack[-1]
                cw, ch = cur[2] - cur[0], cur[3] - cur[1]
                nx1 = cur[0] + nb[0] / 1000 * cw
                ny1 = cur[1] + nb[1] / 1000 * ch
                nx2 = cur[0] + nb[2] / 1000 * cw
                ny2 = cur[1] + nb[3] / 1000 * ch
                final_bbox = [nx1 / 1000 * W0, ny1 / 1000 * H0,
                              nx2 / 1000 * W0, ny2 / 1000 * H0]
                trace[-1]["local_norm"] = nb
                trace[-1]["global_norm"] = [round(v) for v in (nx1, ny1, nx2, ny2)]
            break

        tc = tcs[0]
        name = tc["function"]["name"]
        try:
            a = json.loads(tc["function"]["arguments"] or "{}")
        except Exception:  # noqa: BLE001
            a = {}
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tcs})

        if name == "back":
            if len(stack) > 1:
                stack.pop()
                note = f"已回退到上一层（第 {len(stack)} 层）。"
            else:
                note = "已经在最底层（完整原图），无法再回退。"
            trace.append({"turn": turn, "action": "back", "level": len(stack),
                          "reason": (a.get("reason") or "")[:60]})
        elif name == "zoom":
            try:
                zx, zy = int(a.get("x", 500)), int(a.get("y", 500))
            except Exception:  # noqa: BLE001
                zx, zy = 500, 500
            zx = max(0, min(1000, zx)); zy = max(0, min(1000, zy))
            if n_zoom >= max_turns - 1:
                note = "已经用完放大次数，请直接作答。"
                trace.append({"turn": turn, "action": "zoom_rejected", "point": [zx, zy]})
            else:
                cur_img = render(stack[-1])
                stack.append(_crop_around(stack[-1], zx, zy,
                                          cur_img.width, cur_img.height, crop_cap))
                n_zoom += 1
                note = f"已放大以 ({zx},{zy}) 为中心的区域，当前第 {len(stack)} 层。"
                trace.append({"turn": turn, "action": "zoom", "point": [zx, zy],
                              "level": len(stack),
                              "region": [round(v) for v in stack[-1]],
                              "reason": (a.get("reason") or "")[:60]})
        else:
            note = f"未知工具 {name}。"
            trace.append({"turn": turn, "action": "unknown", "name": name})

        cur = stack[-1]
        img = render(cur)
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": note})
        messages.append({"role": "user", "content": [
            {"type": "text",
             "text": f"【当前画面】第 {len(stack)} 层，放大 {2 ** (len(stack) - 1)} 倍，"
                     f"尺寸 {img.width} x {img.height} 像素，"
                     f"对应原图范围 [{round(cur[0])},{round(cur[1])},"
                     f"{round(cur[2])},{round(cur[3])}]。"
                     f"请继续：看不清就 zoom，走错了就 back，看清了就输出当前画面的 bbox。"},
            _img_block(img),
        ]})

    dt = time.time() - t0
    err = None
    if final_bbox is not None:
        err = (((final_bbox[0] + final_bbox[2]) / 2 - gcx) ** 2
               + ((final_bbox[1] + final_bbox[3]) / 2 - gcy) ** 2) ** 0.5
    return {"id": rid, "gt_bbox": gt, "gt_img_size": [W0, H0],
            "status": "OK" if final_bbox is not None else "NO_ANSWER",
            "final_bbox": final_bbox, "final_err": err,
            "turns_used": len(trace), "n_zoom": n_zoom,
            "final_level": len(stack), "trace": trace,
            "reasoning_effort": reasoning_effort, "max_turns": max_turns,
            "force_answer_last": force_answer_last,
            "latency_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--desc-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--crop-side", type=int, default=1700)
    ap.add_argument("--crop-cap", type=int, default=600,
                    help="裁片每边不超过该像素数（否则放大后看不出效果）")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--force-answer-last", action="store_true",
                    help="最后一轮不提供工具，强制作答")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--tries", type=int, default=10)
    args = ap.parse_args()

    DESC = json.load(open(args.desc_file, encoding="utf-8"))
    rows = {r["id"]: r for r in H.load_all()}
    out = os.path.join("results", "l1", f"agent3_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"DeepSeek 多轮智能体 v3（坐标定位+回退）  最多 {args.max_turns} 轮  "
          f"每层长边 {args.crop_side}px  裁片每边≤{args.crop_cap}px  思考={args.reasoning_effort}  "
          f"末轮强制作答={args.force_answer_last}")
    print(f"输出 → {out}\n")
    recs = []
    for k, rid in enumerate(args.ids.split(","), 1):
        t0 = time.time()
        try:
            rec = run_one(rid, rows[rid], DESC[rid], max_turns=args.max_turns,
                          crop_side=args.crop_side,
                          reasoning_effort=args.reasoning_effort,
                          force_answer_last=args.force_answer_last,
                          max_tokens=args.max_tokens, tries=args.tries,
                          crop_cap=args.crop_cap)
        except Exception as e:  # noqa: BLE001
            rec = {"id": rid, "status": "ERROR", "error": str(e)[:200],
                   "latency_s": round(time.time() - t0, 1)}
        recs.append(rec)
        e = rec.get("final_err")
        acts = "".join({"zoom": "Z", "back": "B", "answer": "A",
                        "zoom_rejected": "x"}.get(t.get("action"), "?")
                       for t in rec.get("trace", []))
        es = "  无" if e is None else f"{round(e):>4}px"
        print(f"  [{k:>2}] {rid:<26} 误差={es}  轨迹={acts:<6} "
              f"层={rec.get('final_level', '?')}  {rec['latency_s']:>5.1f}s  "
              f"{'✅' if (e is not None and e <= 50) else '❌'}", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果 → {out}")


if __name__ == "__main__":
    main()
