#!/usr/bin/env python3
"""通用描述生成 v5 —— 外观 + 类型 + 5×5 方位 + 锚点 + 排除项。

设计原则（用户要求）：
  · 精简优先，有区分度即可
  · 周边只写 1-2 个最能定位的紧邻锚点，不写上下左右四方向
  · 不单列排除项——区分度由「外观 + 方位 + 锚点」三件套承担
  · 方位从 GT bbox 纯计算，不调用模型（零成本、零方差、可复现）

两次视觉调用：
  ① 紧裁片   → 外观 + 类型
  ② 上下文裁片 → 最有区分度的周边锚点
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import qwen_client as qc  # noqa: E402

ROWS = {r["id"]: r for r in H.load_all()}

# ---------------------------------------------------------------- 方位（纯计算）

def _band(v: float, names: list[str], ths=(0.10, 0.30, 0.70, 0.90)) -> str:
    for i, t in enumerate(ths):
        if v < t:
            return names[i]
    return names[4]


def pos_5x5(cx: float, cy: float, w: int, h: int) -> tuple[str, str]:
    """5×5 网格：水平 5 档 × 垂直 5 档。"""
    return (_band(cx / w, ["左侧", "中左", "中部", "中右", "右侧"]),
            _band(cy / h, ["上部", "中上", "正中部", "中下", "下部"]))


# ---------------------------------------------------------------- 裁片

def tight_crop(img: Image.Image, gt, side_ratio: float = 2.5) -> Image.Image:
    """紧裁片：目标占裁片宽 12-40%。红框标出目标。"""
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    side = int(max(120, max(gw, gh) * side_ratio))
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    w, h = img.size
    x0 = int(max(0, min(w - side, cx - side / 2)))
    y0 = int(max(0, min(h - side, cy - side / 2)))
    c = img.crop((x0, y0, x0 + side, y0 + side)).copy()
    d = ImageDraw.Draw(c)
    d.rectangle([gt[0] - x0, gt[1] - y0, gt[2] - x0, gt[3] - y0],
                outline=(255, 0, 0), width=max(3, side // 80))
    return c


def band_crop(img: Image.Image, gt, pad_y: int = 70) -> Image.Image:
    """目标所在的横向条带（整幅宽 × 目标上下各 pad_y），【原分辨率】不缩放，
    用于让模型读清左右相邻元素的文字。红框标出目标。"""
    y0 = int(max(0, gt[1] - pad_y))
    y1 = int(min(img.height, gt[3] + pad_y))
    c = img.crop((0, y0, img.width, y1)).copy()
    d = ImageDraw.Draw(c)
    d.rectangle([gt[0], gt[1] - y0, gt[2], gt[3] - y0], outline=(255, 0, 0), width=3)
    return c


def ctx_crop(img: Image.Image, gt, pad: int = 260) -> Image.Image:
    """上下文裁片：GT ± pad。红框标出目标。"""
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    w, h = img.size
    x0 = int(max(0, min(w - 2 * pad, cx - pad)))
    y0 = int(max(0, min(h - 2 * pad, cy - pad)))
    c = img.crop((x0, y0, x0 + 2 * pad, y0 + 2 * pad)).copy()
    d = ImageDraw.Draw(c)
    d.rectangle([gt[0] - x0, gt[1] - y0, gt[2] - x0, gt[3] - y0],
                outline=(255, 0, 0), width=4)
    return c


# ---------------------------------------------------------------- 提示词

P_APPEAR = """注意：图中那个【红色矩形边框】是标注框，用来指出目标位置，
【它不是元素本身的一部分】，千万不要把它当成元素的特征。

请只描述红色标注框【内部】的那个元素本身：
① 形状与颜色（指元素自身的边框、圆圈、底色——不是那个红色标注框）
② 上面有没有文字或符号？有就写出来，没有写"无"
要求：不超过 40 个字；如果元素上有文字，把文字写完整。
只输出描述本身，不要解释、不要编号、不要提红色标注框。"""

P_ANCHOR = """目标元素的外观是：「{feat}」——你的锚点必须和这个外观【指向同一个元素】。

第一张图是【完整界面】，第二张是【目标元素】的放大图，
第三张是【目标所在那一整行的原始分辨率条带】——第三张图没有缩放，
文字最清晰，请用它来辨认左右相邻元素的【文字内容】。

注意：图里的【红色矩形边框】是标注框，不是元素的一部分，不要写进锚点。


请按以下三步流程，为这个目标元素生成一句"周边锚点"：

第 1 步 —— 找相似元素：在完整界面里找出所有与目标元素【外观相似】的元素
        （形状、颜色、图标内容相近的都算，可能没有、也可能有好几个）。

第 2 步 —— 找区别：把目标元素与这些相似元素的【周边环境】逐一对比，
        找出一个【只在目标身上成立、相似元素身上都不成立】的周边特征。

第 3 步 —— 输出（三选一）：
        · 找到了区别 → 就用这个区别当锚点
        · 界面里根本没有相似元素 → 输出目标最明显的 1-2 个紧邻特征
        · 有相似元素、但周边环境也没区别 → 只输出"无"

要求：
· 不超过 30 个字，只写这一句锚点本身，不要写推理过程、不要编号。
· 不要描述元素自身的外观。

⚠️【句式硬性要求】禁止使用"目标在 X 的左侧/右侧"这类说法——它会被反向解读。
   必须改写成"**X 的某个部位**"这种只有一种解读的说法：

   ✗ 禁止：右侧紧邻"utils.py"标签页
   ✗ 禁止：位于 generate.py 标签页右侧
   ✗ 禁止：左侧紧邻 testing_utils.py 标签页
   ✓ 正确：utils.py 标签文字右端的关闭按钮
   ✓ 正确：generate.py 标签右侧那个标签的关闭按钮
   ✓ 正确：工具栏中剪刀图标右边第一个图标

   判断标准：说"目标在X的某侧"时，读者可能理解成"目标自己就在X的那一侧"；
   而说"X 的某部位"时只有一种解读。必须用后者。

· 关键：这句话要能让人【排除掉那些相似元素】。"""


P_ANCHOR_CHECK = """第一张图是【完整界面】，第二张是【目标元素】的放大图（红框内）。
注意：第二张图里那个红色矩形边框是标注框，不是元素的一部分。

有人为这个目标元素写了一句"周边锚点"：
「{anchor}」

请核实这句话是否与画面相符，逐条检查：

① 锚点里提到的相邻元素，在图中【确实存在】吗？
② 锚点描述的【左右/上下位置关系】正确吗？（目标真的在它说的那一侧吗）
③ 有没有【多个元素】都符合这句锚点？（即：锚点能不能唯一确定目标）

输出格式（三行，不要写别的）：
判定：正确 或 部分错误 或 错误
问题：一句话说明哪里不对；没有就写 none
修正后的锚点：如果原锚点有错，给出修正版（不超过 30 字，只写锚点本身）；否则写 none"""


P_CONSIST = """下面两句话描述的是【同一个界面元素】吗？

外观描述：{feat}
锚点描述：{anchor}

判断标准：两句话指向的元素类型、颜色、形状必须对得上。
例如"白色矩形图标带蓝色画笔"和"Book1标题栏右端的关闭按钮"就是【不一致】的。

只回答一行：
一致：是 或 否"""


def _field(txt: str, key: str) -> str:
    """从多行输出里取 `key：value` 的 value。"""
    for line in (txt or "").splitlines():
        line = line.strip().lstrip("-*· ")
        if line.startswith(key):
            return line.split("：", 1)[-1].split(":", 1)[-1].strip()
    return ""


def _clean(txt: str, cap: int) -> str:
    """取首行、去编号；超长时在分句处截断，避免把文字名切成一半。"""
    # 多行输出先合并成一行（模型常分多行写）
    t = " ".join(x.strip() for x in (txt or "").strip().splitlines() if x.strip())
    t = re.sub(r"^[\s\d①-⑩\.、)）:：-]+", "", t).strip()
    t = t.strip('"“”「」 ')
    if len(t) <= cap:
        return t
    # 在 cap 之后找第一个分句边界，避免把文字名切成一半
    for sep in ("。", "；", "，", "、"):
        i = t.find(sep, cap)
        if i != -1 and i < cap * 2:
            return t[:i].strip()
    return t[:cap]


def gen_one(rid: str, *, model: str, verify_model: str = 'qwen3.8-max',
            do_verify: bool = True, quiet: bool = False) -> dict:
    row = ROWS[rid]
    img = Image.open(H.fetch_image(row["img_filename"])).convert("RGB")
    w, h = img.size
    gt = row["bbox"]
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    hz, vt = pos_5x5(cx, cy, w, h)

    t0 = time.time()
    # ① 外观 + 类型
    t_app, _, _ = qc.ask_image(tight_crop(img, gt), P_APPEAR,
                               model=model, max_tokens=200, want_reasoning=True)
    # ② 锚点（两步式：先扫全图找相似元素，再找只在目标身上成立的周边特征）
    small, _ = H.downscale(img)
    ctx = ctx_crop(img, gt)
    band = band_crop(img, gt)
    _appear_tmp = _clean(t_app, 40)          # 外观（先算出来喂给锚点生成）
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{qc._b64_png(small)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{qc._b64_png(ctx)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{qc._b64_png(band)}"}},
        {"type": "text", "text": P_ANCHOR.format(feat=_appear_tmp)},
    ]

    def _gen_anchor() -> str:
        return _clean(qc.text_of(qc.chat([{"role": "user", "content": content}],
                                         model=model, max_tokens=300)), 30)

    def _consistent(feat: str, anc: str) -> bool:
        """纯文本自洽性检查：锚点和外观是不是同一个元素。"""
        if not anc or anc in ("无", "无。"):
            return True
        try:
            r = qc.text_of(qc.chat([{"role": "user",
                                     "content": P_CONSIST.format(feat=feat, anchor=anc)}],
                                   model=model, max_tokens=40))
            return "否" not in r[:20]
        except Exception:  # noqa: BLE001
            return True                       # 检查失败不阻塞

    t_anc = _gen_anchor()
    if not _consistent(_appear_tmp, t_anc):
        t_anc = _gen_anchor()                 # 自相矛盾 → 重生成一次

    # ②b 用更强的模型（默认 qwen3.8-max）校验锚点
    anchor_raw = _clean(t_anc, 30)
    anchor_fix, anchor_verdict, anchor_issue = "", "", ""
    if do_verify and anchor_raw and anchor_raw not in ("无", "无。"):
        tight_v = tight_crop(img, gt)
        vcontent = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{qc._b64_png(small)}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{qc._b64_png(tight_v)}"}},
            {"type": "text", "text": P_ANCHOR_CHECK.format(anchor=anchor_raw)},
        ]
        try:
            vtxt = qc.text_of(qc.chat([{"role": "user", "content": vcontent}],
                                      model=verify_model, max_tokens=400))
            anchor_verdict = _field(vtxt, "判定")
            anchor_issue = _field(vtxt, "问题")
            fix = _field(vtxt, "修正后的锚点")
            if fix and fix.lower() not in ("none", "无", "无。") and anchor_verdict in ("部分错误", "错误"):
                anchor_fix = _clean(fix, 30)
        except Exception as e:  # noqa: BLE001
            anchor_issue = f"校验调用失败: {type(e).__name__}"
    anchor = anchor_fix or anchor_raw

    appear = _appear_tmp
    desc = {
        "feat": appear,
        "ctrl": "按钮" if any(k in appear for k in ("按钮", "图标", "按键")) else "元素",
        "pos": f"屏幕水平【{hz}】、垂直【{vt}】",
        "env": anchor_fix or anchor,
        "env_raw": anchor_raw,
        "env_verdict": anchor_verdict,
        "env_issue": anchor_issue,
    }

    if not quiet:
        if anchor_verdict:
            print(f"    锚点校验[{verify_model}]：{anchor_verdict}"
                  + (f"  问题={anchor_issue}" if anchor_issue and anchor_issue != "none" else "")
                  + (f"  → 修正为：{anchor_fix}" if anchor_fix else ""))
        print(f"  【{rid}】{w}x{h}  目标{gt[2]-gt[0]:.0f}x{gt[3]-gt[1]:.0f}px  用时{time.time()-t0:.1f}s")
        print(f"    外观  : {desc['feat']}")
        print(f"    方位  : {desc['pos']}   (归一化 cx={cx/w*1000:.0f} cy={cy/h*1000:.0f})")
        print(f"    锚点  : {desc['env']}")
    return desc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="逗号分隔")
    ap.add_argument("--out", default="results/l1/tightcrop/descriptions_v5.json")
    ap.add_argument("--model", default="qwen3.8-flash")
    ap.add_argument("--verify-model", default="qwen3.8-max", help="锚点校验用的模型")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过锚点校验（已验证：校验会把正确锚点改错，故默认应跳过）")
    ap.add_argument("--merge", action="store_true",
                    help="与已有 out 文件合并（只覆盖本次 ids）")
    args = ap.parse_args()

    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    out = Path(args.out)
    data = {}
    if args.merge and out.exists():
        data = json.load(open(out, encoding="utf-8"))

    print(f"模型={args.model}  样例 {len(ids)} 条  输出 → {out}\n")
    for i, rid in enumerate(ids, 1):
        print(f"[{i:>2}/{len(ids)}] {rid}")
        try:
            data[rid] = gen_one(rid, model=args.model, verify_model=args.verify_model,
                                do_verify=not args.no_verify)
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ 失败: {type(e).__name__}: {e}")
        print()

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"结果 → {out}  （{len(data)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
