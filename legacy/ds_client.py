"""DeepSeek 客户端（用 curl 子进程 + 耐心重试）。

背景：api.deepseek.com 的 /chat/completions 会在「可用/不可用」之间快速翻转
（好窗口 30s–2min），失败形态为 HTTP 401（被路由到 alibabacloud 网关）、
HTTP 503（Service is too busy）、或直接挂起。与客户端/UA/HTTP版本/请求体大小均无关。

因此这里用「长窗口耐心重试」兜过去：默认最多重试 120 次 × 2 秒 ≈ 4 分钟。

凭据从环境变量读取：
  DS_BASE  默认 https://api.deepseek.com/v1
  DS_KEY   API Key
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import time

BASE = os.environ.get("DS_BASE", "https://api.deepseek.com/v1").rstrip("/")
KEY = os.environ.get("DS_KEY", "")
MODEL = os.environ.get("DS_MODEL", "deepseek-flash")


def b64_jpeg(img, quality: int = 80) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def chat(messages, *, model=None, max_tokens=2000, tries=120, gap=2.0,
         timeout=120, extra=None, verbose=False):
    """发一次 chat/completions，失败则耐心重试。

    返回 (响应 dict, 实际尝试次数)。全部失败抛 RuntimeError。
    """
    if not KEY:
        raise RuntimeError("缺少 DS_KEY 环境变量")
    payload = {"model": model or MODEL, "messages": messages, "max_tokens": max_tokens}
    if extra:
        payload.update(extra)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    last = None
    try:
        for attempt in range(1, tries + 1):
            proc = subprocess.run(
                ["curl", "-s", "-m", str(timeout), f"{BASE}/chat/completions",
                 "-H", f"Authorization: Bearer {KEY}",
                 "-H", "Content-Type: application/json",
                 "-H", "User-Agent: pointer-app",
                 "-d", f"@{path}"],
                capture_output=True, text=True,
            )
            try:
                data = json.loads(proc.stdout)
                if "choices" in data:
                    return data, attempt
                last = (data.get("error") or {}).get("message", "")[:60]
            except Exception:  # noqa: BLE001
                last = "空响应/挂起"
            if verbose and attempt % 10 == 0:
                print(f"      …重试 {attempt}/{tries}（{last}）", flush=True)
            time.sleep(gap)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    raise RuntimeError(f"重试 {tries} 次仍失败：{last}")


def text_of(resp) -> str:
    return (resp["choices"][0]["message"].get("content") or "").strip()


def reasoning_of(resp) -> str:
    return (resp["choices"][0]["message"].get("reasoning_content") or "").strip()


def ask_image(img, prompt, *, model=None, max_tokens=2000, tries=120,
              detail=None, reasoning_effort=None, quality=80, verbose=False):
    """单图 + 文本提示，返回 (文本, usage, 思考, 尝试次数)"""
    iu = {"url": f"data:image/jpeg;base64,{b64_jpeg(img, quality)}"}
    if detail:
        iu["detail"] = detail
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else None
    resp, att = chat(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": iu},
            {"type": "text", "text": prompt},
        ]}],
        model=model, max_tokens=max_tokens, tries=tries, extra=extra, verbose=verbose,
    )
    return text_of(resp), resp.get("usage", {}), reasoning_of(resp), att
