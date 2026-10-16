"""极简 OpenAI 兼容客户端（只用标准库，避免依赖问题）。

凭据从环境变量读取：
  QW_BASE  例如 https://xxx/compatible-mode/v1
  QW_KEY   API Key
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

BASE = os.environ.get("QW_BASE", "").rstrip("/")
KEY = os.environ.get("QW_KEY", "")
MODEL = os.environ.get("QW_MODEL", "qwen3.8-flash")


def _b64_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def chat(messages, *, model=None, max_tokens=512, temperature=0.0,
         timeout=75, retries=2, extra=None):
    """messages: [{'role','content': str | list}]，图片用 {'type':'image_url',...}"""
    if not BASE or not KEY:
        raise RuntimeError("缺少 QW_BASE / QW_KEY 环境变量")
    payload = {
        "model": model or MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra:
        payload.update(extra)
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{BASE}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:600]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last) from None
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    raise RuntimeError(last or "unknown error")


def ask_image(img, prompt, *, model=None, max_tokens=512, timeout=75, want_reasoning=False):
    """单图 + 文本提示，返回 (文本, usage)；want_reasoning=True 时返回 (文本, usage, 思考)"""
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64_png(img)}"}},
        {"type": "text", "text": prompt},
    ]
    resp = chat([{"role": "user", "content": content}],
                model=model, max_tokens=max_tokens, timeout=timeout)
    msg = resp["choices"][0]["message"]
    txt = msg.get("content") or ""
    if want_reasoning:
        return txt, resp.get("usage", {}), (msg.get("reasoning_content") or "")
    return txt, resp.get("usage", {})


def text_of(resp) -> str:
    return resp["choices"][0]["message"]["content"]
