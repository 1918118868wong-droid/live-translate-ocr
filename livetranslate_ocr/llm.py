"""极简的 OpenAI 兼容 Chat Completions 客户端。

为什么不用 openai 官方 SDK：SDK 有过 v1 → v2 → v3 的大版本变动，每次都可能
改掉签名或默认行为，而我们要的只是「POST 一个 JSON、按 SSE 读回来」这一件事。
协议本身从 2023 年至今没变过，自己写反而是最稳的，顺便少一个依赖。

支持所有 OpenAI 兼容端点：DeepSeek、OpenAI、Moonshot、通义兼容模式、
Ollama、LM Studio、vLLM、one-api 等等。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "host.docker.internal"}


class LlmError(RuntimeError):
    """调用 LLM 失败。message 里带服务端返回的正文，方便直接看出是 Key 错还是余额不足。"""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# 服务端临时故障：过载、限流、网关抖动。这类等一会儿重试通常就好了，
# 和"Key 错了 / 余额不足 / 模型名错了"不是一回事——后者重试多少次都没用。
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})


def is_transient(exc: BaseException) -> bool:
    """判断这个失败值不值得重试。

    没有状态码的 LlmError（响应格式异常、网络中断）也算——多半是链路抖动。
    """
    if isinstance(exc, LlmError):
        return exc.status is None or exc.status in TRANSIENT_STATUS
    return isinstance(exc, httpx.HTTPError)


def _endpoint(base_url: str) -> str:
    """把 base_url 补成完整的 chat/completions 地址。"""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise LlmError("未配置 LLM base_url")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 12.0,
        extra_headers: dict[str, str] | None = None,
    ):
        self.url = _endpoint(base_url)
        self.model_base = base_url
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        # 本地端点（Ollama / LM Studio / vLLM）必须绕开系统代理。
        # 否则机器上开着 Clash 之类的东西时，请求会被代理转发出去、
        # 再由代理去连 127.0.0.1 —— 结果是一个莫名其妙的 502。
        host = (urlsplit(self.url).hostname or "").lower()
        local = host in _LOCAL_HOSTS or host.endswith(".local")
        self._client = httpx.Client(
            headers=headers,
            trust_env=not local,
            # 流式场景下 read 是"两块之间的最长间隔"，正好用来兜住卡死的连接
            timeout=httpx.Timeout(connect=6.0, read=timeout_s, write=15.0, pool=6.0),
        )

    def close(self) -> None:
        self._client.close()

    # ---- 内部 ----
    def _payload(self, model: str, messages: list[dict], stream: bool, **kw) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        for key, value in kw.items():
            if value is not None:
                body[key] = value
        return body

    @staticmethod
    def _error_from(response: httpx.Response) -> LlmError:
        text = ""
        try:
            text = response.text[:400]
        except Exception:
            pass
        hint = ""
        if response.status_code == 401:
            hint = "（API Key 无效或未生效）"
        elif response.status_code == 402 or "insufficient" in text.lower():
            hint = "（账户余额不足）"
        elif response.status_code == 404:
            hint = "（base_url 或模型名不对）"
        elif response.status_code == 429:
            hint = "（触发限流）"
        return LlmError(f"HTTP {response.status_code}{hint}：{text}", status=response.status_code)

    # ---- 非流式 ----
    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 64,
        **extra,
    ) -> str:
        try:
            resp = self._client.post(
                self.url,
                json=self._payload(model, messages, False, temperature=temperature,
                                   max_tokens=max_tokens, **extra),
            )
        except httpx.HTTPError as exc:
            raise LlmError(f"网络错误：{exc}") from exc
        if resp.status_code != 200:
            raise self._error_from(resp)
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmError(f"响应格式异常：{resp.text[:300]}") from exc

    # ---- 流式 ----
    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.35,
        max_tokens: int = 512,
        **extra,
    ) -> Iterator[str]:
        """逐段产出增量文本。只吐 content，忽略 reasoning_content 之类的字段。"""
        payload = self._payload(
            model, messages, True, temperature=temperature, max_tokens=max_tokens, **extra
        )
        try:
            with self._client.stream(
                "POST", self.url, json=payload, headers={"Accept": "text/event-stream"}
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise self._error_from(resp)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        if line == "[DONE]":
                            return
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise LlmError(f"网络中断：{exc}") from exc

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
