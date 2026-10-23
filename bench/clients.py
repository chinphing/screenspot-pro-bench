"""模型客户端（统一接口）。

后端：
    deepseek   —— 走 curl 子进程 + 长窗口耐心重试（api.deepseek.com 会快速翻转可用性）
    qwen       —— 走 urllib（阿里云 MaaS 兼容模式）

统一接口：
    c = get_client("qwen")
    text = c.ask_image(img, prompt, max_tokens=512, reasoning_effort="none")
    msg  = c.chat(messages, max_tokens=1500, tools=[...], reasoning_effort="none")
    # msg: {"content", "reasoning_content", "tool_calls", "finish_reason"}

⚠️ 重要（踩过的坑，见 docs/ARCHIVE4.md §32–34）：
   1. 超时参数要成组设置——ask_image 与 chat 两条路径都要给足（默认 300s）
   2. reasoning_effort 要成组设置——Qwen 求 bbox 若不显式关思考会慢 38 倍
   3. 图片编码：DeepSeek 用 JPEG q80；Qwen 历史上 ask_image 用 PNG（两条路径不一致）
      这里统一为 JPEG q80（实测体积/格式对耗时无影响），如需复现历史结果可传 fmt="png"
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# 密钥放本目录 .env（已被 .gitignore 排除），格式见 .env.example。
# 用 python-dotenv 而不是靠 shell export：一次填好所有基线脚本都读得到，
# 也不会因为 /tmp 被系统清理而失效（/tmp/qw.env 就被清过一次）。
# 缺 python-dotenv 时静默跳过，仍可用环境变量。
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

DEFAULT_TIMEOUT = 300
DEFAULT_RETRIES = 4

BACKENDS = ("deepseek", "qwen")


def b64(img, fmt: str = "jpeg", quality: int = 80) -> str:
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
        mime = "png"
    else:
        img.save(buf, format="JPEG", quality=quality)
        mime = "jpeg"
    return mime, base64.b64encode(buf.getvalue()).decode()


def image_block(img, fmt: str = "jpeg", quality: int = 80) -> dict:
    mime, data = b64(img, fmt, quality)
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}}


class BaseClient:
    name = "base"
    default_model = ""

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    # 子类实现 ------------------------------------------------------------
    def chat(self, messages, *, max_tokens=1500, tools=None, tool_choice="auto",
             reasoning_effort=None, timeout=DEFAULT_TIMEOUT,
             retries=DEFAULT_RETRIES) -> dict:
        raise NotImplementedError

    # 公共 ----------------------------------------------------------------
    def ask_image(self, img, prompt, *, max_tokens=512, reasoning_effort=None,
                  fmt="jpeg", timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES) -> str:
        """单图 + 文本提示 → 文本回复。"""
        msg = self.chat([{"role": "user", "content": [
            image_block(img, fmt),
            {"type": "text", "text": prompt},
        ]}], max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            timeout=timeout, retries=retries)
        return msg.get("content") or ""

    def first_tool_call(self, msg) -> tuple[str, dict]:
        """从回复里取第一个工具调用 → (名字, 参数字典)。没有则返回 ("", {})。"""
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return "", {}
        tc = tcs[0]
        name = (tc["function"].get("name") or "").lower()
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:  # noqa: BLE001
            args = {}
        return name, args


class DeepSeekClient(BaseClient):
    name = "deepseek"
    default_model = os.environ.get("DS_MODEL", "deepseek-flash")

    def __init__(self, model: str | None = None):
        super().__init__(model)
        self.base = os.environ.get("DS_BASE", "https://api.deepseek.com/v1").rstrip("/")
        self.key = os.environ.get("DS_KEY", "")

    def chat(self, messages, *, max_tokens=1500, tools=None, tool_choice="auto",
             reasoning_effort=None, timeout=DEFAULT_TIMEOUT,
             retries=DEFAULT_RETRIES) -> dict:
        if not self.key:
            raise RuntimeError("缺少 DS_KEY 环境变量")
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        last = None
        try:
            for _ in range(retries):
                proc = subprocess.run(
                    ["curl", "-s", "-m", str(timeout), f"{self.base}/chat/completions",
                     "-H", f"Authorization: Bearer {self.key}",
                     "-H", "Content-Type: application/json",
                     "-H", "User-Agent: pointer-app",
                     "-d", f"@{path}"],
                    capture_output=True, text=True,
                )
                try:
                    data = json.loads(proc.stdout)
                    if "choices" in data:
                        return _normalize(data)
                    last = (data.get("error") or {}).get("message", "")[:80]
                except Exception:  # noqa: BLE001
                    last = "空响应/挂起"
                time.sleep(2.0)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise RuntimeError(f"DeepSeek 重试 {retries} 次仍失败：{last}")


class QwenClient(BaseClient):
    name = "qwen"
    default_model = os.environ.get("QW_MODEL", "qwen3.8-flash")

    def __init__(self, model: str | None = None):
        super().__init__(model)
        self.base = os.environ.get("QW_BASE", "").rstrip("/")
        self.key = os.environ.get("QW_KEY", "")

    def chat(self, messages, *, max_tokens=1500, tools=None, tool_choice="auto",
             reasoning_effort=None, timeout=DEFAULT_TIMEOUT,
             retries=DEFAULT_RETRIES) -> dict:
        if not self.base or not self.key:
            raise RuntimeError("缺少 QW_BASE / QW_KEY 环境变量")
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": 0.0}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        body = json.dumps(payload).encode()
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(
                f"{self.base}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"},
                method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return _normalize(json.loads(r.read().decode()))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode()[:200]}"
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last) from None
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        raise RuntimeError(last or "unknown error")


def _normalize(resp: dict) -> dict:
    """把 OpenAI 风格响应压成 {content, reasoning_content, tool_calls, finish_reason}。"""
    ch = resp["choices"][0]
    m = ch["message"]
    return {
        "content": m.get("content") or "",
        "reasoning_content": m.get("reasoning_content") or "",
        "tool_calls": m.get("tool_calls") or [],
        "finish_reason": ch.get("finish_reason"),
        "usage": resp.get("usage", {}),
    }


def get_client(backend: str, model: str | None = None) -> BaseClient:
    if backend == "deepseek":
        return DeepSeekClient(model)
    if backend == "qwen":
        return QwenClient(model)
    raise ValueError(f"未知后端: {backend}（可选 {BACKENDS}）")


# ── 模型名 → 结果文件名短标签 ──
# 为什么需要：B2/B3 的 run_one 是从 legacy 逐字拷贝的，写不进结果行的 model 字段，
# 模型只能体现在文件名里——否则 qwen3.7 与 qwen3.8 的成绩会在汇总时撞名合并。
MODEL_ALIAS = {
    "qwen3.8-flash": "q38",
    "qwen3.7-flash": "q37",
    "qwen3.7-max": "q37max",
    "qwen-flash": "qflash",
    "deepseek-flash": "dsflash",
}


def resolve_model(backend: str, explicit: str | None = None) -> str:
    """该后端实际会用到的模型名（显式指定 > 环境变量 > 内置默认）。"""
    cls = DeepSeekClient if backend == "deepseek" else QwenClient
    return explicit or getattr(cls, "default_model", "")


def model_tag(model: str) -> str:
    """模型名 → 文件名短标签（qwen3.8-flash → q38）；未登记的模型做净化截断。"""
    return MODEL_ALIAS.get(model) or "".join(c for c in model.lower() if c.isalnum())[:12]
