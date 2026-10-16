"""兼容垫片：legacy/verify_loop.py 里被 ZF/Z6 复用的三个符号（原样提取）。

⚠️ 逐字拷贝，不重构——重构会改变行为。
"""
from __future__ import annotations

from PIL import Image, ImageDraw


def draw_cursor(img: Image.Image, x: float, y: float, side: float) -> Image.Image:
    """在 (x,y) 画边长 side 的正方形标记，【中心】落在 (x,y)。

    描边式（不填充），保证方块内的目标仍可见：红色内框 + 白色外描边。
    """
    img = img.copy()
    d = ImageDraw.Draw(img)
    h = float(side) / 2.0
    box = [x - h, y - h, x + h, y + h]
    inner = max(2, int(round(side * 0.10)))
    outer = inner + max(2, int(round(side * 0.08)))
    d.rectangle(box, outline=(255, 255, 255), width=outer)   # 白色外描边
    d.rectangle(box, outline=(220, 30, 30), width=inner)     # 红色内框
    return img



def pad_edge(img: Image.Image, pad: int) -> Image.Image:
    """四周用边缘像素复制扩展 pad 像素，保证任意位置都能取到完整裁片（光标永远居中）。"""
    w, h = img.size
    out = Image.new("RGB", (w + 2 * pad, h + 2 * pad))
    out.paste(img, (pad, pad))
    out.paste(img.crop((0, 0, w, 1)).resize((w, pad)), (pad, 0))
    out.paste(img.crop((0, h - 1, w, h)).resize((w, pad)), (pad, pad + h))
    out.paste(img.crop((0, 0, 1, h)).resize((pad, h)), (0, pad))
    out.paste(img.crop((w - 1, 0, w, h)).resize((pad, h)), (pad + w, pad))
    for dx, dy, box in [(0, 0, (0, 0, 1, 1)), (pad + w, 0, (w - 1, 0, w, 1)),
                        (0, pad + h, (0, h - 1, 1, h)), (pad + w, pad + h, (w - 1, h - 1, w, h))]:
        out.paste(img.crop(box).resize((pad, pad)), (dx, dy))
    return out


# ---------------------------------------------------------------- 提示词

L1_PROMPT = """请找出界面中符合以下描述的元素，给出它的 bbox。

【目标元素描述】
· 文字内容：{text}
· 外观特征：{feat}
· 控件类型：{ctrl}
· 屏幕位置：{pos}
· 周边锚点：{env}

请给出【刚好框住该元素本身】的 bbox（不要外扩、不要缩小）。
坐标请归一化到 0-1000 的整数范围，格式 [x1,y1,x2,y2]，其中 0 表示左/上边缘，1000 表示右/下边缘。
只输出这个数组，不要解释。"""

# 箭头（标记物）自身的外观描述 —— 与目标元素描述同一套写法
ARROW_DESC = ("一个正方形方框标记（红色边框、外侧带白色描边），"
              "它是空心的、不填充，方块的中心就是标记的位置")

