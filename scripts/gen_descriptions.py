#!/usr/bin/env python3
"""为每个样例生成元素描述（feat/ctrl/pos/env）。

描述是整套评测的输入——定位提示词里的"目标元素描述"就来自这里。
生成是【一次性离线任务】，结果落盘到 descriptions/<split>.json 后复用到所有基线。

三步：
    ① 外观：紧裁片（红框标目标）→ 形状/颜色/文字
    ② 锚点：整图 + 上下文裁片 + 原始分辨率条带 → 一句"只在目标身上成立"的周边特征
    ③ 方位：按目标中心算 5×5 网格档位（纯几何，不用模型）

⚠️ 锚点校验（--verify）默认【关闭】：实测校验会把正确锚点改错（见 docs/ARCHIVE4.md）。
   默认不传 reasoning_effort——描述生成需要思考，关掉反而变差（与 bbox 任务相反）。

    python3 scripts/gen_descriptions.py --split M1 --workers 8
    python3 scripts/gen_descriptions.py --split M2 --workers 8 --out descriptions/M2.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from bench import DESCRIPTIONS  # noqa: E402
from bench import clients as C  # noqa: E402
from bench import dataset as D  # noqa: E402
from bench import vision as V  # noqa: E402

# ---------------------------------------------------------------- 方位
def _band(v: float, names: list[str], ths=(0.10, 0.30, 0.70, 0.90)) -> str:
    for i, t in enumerate(ths):
        if v < t:
            return names[i]
    return names[4]


def pos_5x5(cx: float, cy: float, w: int, h: int) -> tuple[str, str]:
    return (_band(cx / w, ["左侧", "中左", "中部", "中右", "右侧"]),
            _band(cy / h, ["上部", "中上", "正中部", "中下", "下部"]))


# ---------------------------------------------------------------- 裁片
def _mark(img: Image.Image, gt, x0: int, y0: int, width: int) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([gt[0] - x0, gt[1] - y0, gt[2] - x0, gt[3] - y0],
                outline=(255, 0, 0), width=width)
    return img


def tight_crop(img: Image.Image, gt, side_ratio: float = 2.5) -> Image.Image:
    """紧裁片：目标占裁片宽 12-40%，红框标出目标。"""
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    side = int(max(120, max(gw, gh) * side_ratio))
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    w, h = img.size
    x0 = int(max(0, min(w - side, cx - side / 2)))
    y0 = int(max(0, min(h - side, cy - side / 2)))
    return _mark(img.crop((x0, y0, x0 + side, y0 + side)).copy(), gt, x0, y0, max(3, side // 80))


def band_crop(img: Image.Image, gt, pad_y: int = 70) -> Image.Image:
    """目标所在的横向条带（整幅宽 × 目标上下各 pad_y），原分辨率不缩放——文字最清晰。"""
    y0 = int(max(0, gt[1] - pad_y))
    y1 = int(min(img.height, gt[3] + pad_y))
    return _mark(img.crop((0, y0, img.width, y1)).copy(), gt, 0, y0, 3)


def ctx_crop(img: Image.Image, gt, pad: int = 260) -> Image.Image:
    """上下文裁片：GT ± pad。"""
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    w, h = img.size
    x0 = int(max(0, min(w - 2 * pad, cx - pad)))
    y0 = int(max(0, min(h - 2 * pad, cy - pad)))
    return _mark(img.crop((x0, y0, x0 + 2 * pad, y0 + 2 * pad)).copy(), gt, x0, y0, 4)


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
    for line in (txt or "").splitlines():
        line = line.strip().lstrip("-*· ")
        if line.startswith(key):
            return line.split("：", 1)[-1].split(":", 1)[-1].strip()
    return ""


def _clean(txt: str, cap: int) -> str:
    """取首行、去编号；超长时在分句处截断，避免把文字名切成一半。"""
    t = " ".join(x.strip() for x in (txt or "").strip().splitlines() if x.strip())
    t = re.sub(r"^[\s\d①-⑩\.、)）:：-]+", "", t).strip().strip('"“”「」 ')
    if len(t) <= cap:
        return t
    for sep in ("。", "；", "，", "、"):
        i = t.find(sep, cap)
        if i != -1 and i < cap * 2:
            return t[:i].strip()
    return t[:cap]


# ---------------------------------------------------------------- 单条生成
_LOCK = threading.Lock()


def gen_one(rid: str, row: dict, client, *, do_verify: bool = False,
            verify_model: str = "qwen3.8-max", quiet: bool = True) -> dict:
    img = Image.open(D.fetch_image(row["img_filename"])).convert("RGB")
    w, h = img.size
    gt = row["bbox"]
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    hz, vt = pos_5x5(cx, cy, w, h)
    t0 = time.time()

    # ① 外观
    appear = _clean(client.ask_image(tight_crop(img, gt), P_APPEAR, max_tokens=200,
                                     fmt="png"), 40)
    # ② 锚点（整图 + 上下文 + 原始分辨率条带）
    small = V.fit(img)
    content = [C.image_block(small, fmt="png"),
               C.image_block(ctx_crop(img, gt), fmt="png"),
               C.image_block(band_crop(img, gt), fmt="png"),
               {"type": "text", "text": P_ANCHOR.format(feat=appear)}]

    def _gen() -> str:
        return _clean(client.chat([{"role": "user", "content": content}],
                                  max_tokens=300)["content"], 30)

    def _consistent(feat: str, anc: str) -> bool:
        if not anc or anc in ("无", "无。"):
            return True
        try:
            r = client.chat([{"role": "user", "content": P_CONSIST.format(feat=feat, anchor=anc)}],
                            max_tokens=40)["content"]
            return "否" not in (r or "")[:20]
        except Exception:  # noqa: BLE001
            return True

    anchor_raw = _gen()
    if not _consistent(appear, anchor_raw):
        anchor_raw = _gen()

    # ③ 可选：强模型校验锚点（默认关闭——实测会把正确锚点改错）
    anchor_fix, verdict, issue = "", "", ""
    if do_verify and anchor_raw and anchor_raw not in ("无", "无。"):
        try:
            vtxt = client.chat([{"role": "user", "content": [
                C.image_block(small, fmt="png"),
                C.image_block(tight_crop(img, gt), fmt="png"),
                {"type": "text", "text": P_ANCHOR_CHECK.format(anchor=anchor_raw)}]}],
                max_tokens=400)["content"]
            verdict = _field(vtxt, "判定")
            issue = _field(vtxt, "问题")
            fix = _field(vtxt, "修正后的锚点")
            if fix and fix.lower() not in ("none", "无", "无。") and verdict in ("部分错误", "错误"):
                anchor_fix = _clean(fix, 30)
        except Exception as e:  # noqa: BLE001
            issue = f"校验调用失败: {type(e).__name__}"

    desc = {
        "feat": appear,
        "ctrl": "按钮" if any(k in appear for k in ("按钮", "图标", "按键")) else "元素",
        "pos": f"屏幕水平【{hz}】、垂直【{vt}】",
        "env": anchor_fix or anchor_raw,
        "env_raw": anchor_raw,
        "env_verdict": verdict,
        "env_issue": issue,
    }
    if not quiet:
        with _LOCK:
            print(f"  【{rid}】{w}x{h} 目标{gt[2]-gt[0]:.0f}x{gt[3]-gt[1]:.0f}px "
                  f"用时{time.time()-t0:.1f}s")
            print(f"    外观: {desc['feat']}")
            print(f"    方位: {desc['pos']}")
            print(f"    锚点: {desc['env']}", flush=True)
    return desc


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default="", help="逗号分隔样例 id")
    ap.add_argument("--split", default="M1", help="M1 / M2 / M3（读 splits/<split>.json）")
    ap.add_argument("--desc-version", default="v2", help="描述版本目录（descriptions/<版本>/）")
    ap.add_argument("--out", default="", help="输出路径（覆盖 --desc-version；默认 descriptions/<版本>/<split>.json）")
    ap.add_argument("--model", default="", help="生成模型（默认后端默认模型）")
    ap.add_argument("--backend", default="qwen", choices=list(C.BACKENDS))
    ap.add_argument("--verify-model", default="qwen3.8-max", help="锚点校验模型")
    ap.add_argument("--verify", action="store_true", help="启用锚点校验（默认关闭）")
    ap.add_argument("--workers", type=int, default=8, help="并发数（实测 8 并发约 8.6s/条）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="与已有输出合并（只覆盖本次 id）")
    a = ap.parse_args()

    if a.ids:
        ids = [x.strip() for x in a.ids.split(",") if x.strip()]
    else:
        f = Path(__file__).resolve().parent.parent / "splits" / f"{a.split}.json"
        ids = json.load(open(f, encoding="utf-8"))
    if a.limit:
        ids = ids[: a.limit]

    rows = D.by_id()
    ids = [i for i in ids if i in rows]
    out = Path(a.out) if a.out else DESCRIPTIONS / a.desc_version / f"{a.split}.json"
    data: dict = {}
    if a.merge and out.exists():
        data = json.load(open(out, encoding="utf-8"))

    client = C.get_client(a.backend, a.model or None)
    todo = [i for i in ids if i not in data]
    print(f"后端={a.backend} 模型={client.model}  样例 {len(ids)} 条（待生成 {len(todo)}）"
          f"  并发 {a.workers}  校验={'开' if a.verify else '关'}")
    print(f"输出 → {out}\n", flush=True)

    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(gen_one, rid, rows[rid], client,
                          do_verify=a.verify, verify_model=a.verify_model, quiet=False): rid
                for rid in todo}
        for fut in cf.as_completed(futs):
            rid = futs[fut]
            try:
                data[rid] = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ❌ {rid}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            done += 1
            if done % 20 == 0:
                el = time.time() - t0
                print(f"  …{done}/{len(todo)}  {el/60:.1f}min  "
                      f"ETA {(len(todo)-done)/(done/el)/60:.1f}min", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n结果 → {out}（{len(data)} 条）  用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
