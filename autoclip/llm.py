from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

from .config import ModelConfig
from .utils import extract_json, log


def _data_url(path: str | Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


_req_ids = iter(range(1, 1 << 62))
HEARTBEAT = 60.0


def _heartbeat(stop: threading.Event, tag: str, t0: float, limit: float, state: dict) -> None:
    while not stop.wait(HEARTBEAT):
        phase = "等待并发名额" if state.get("queued") else "等待模型响应"
        log.info("%s %s… 已 %ds / 超时 %ds", tag, phase, time.time() - t0, limit)


def _is_outage(e: Exception) -> bool:
    """超时/连接失败/限流/服务端错误：不值得在主服务上重试。"""
    if isinstance(e, APIConnectionError):  # 含 APITimeoutError
        return True
    return isinstance(e, APIStatusError) and (e.status_code in (401, 402, 403, 429) or e.status_code >= 500)


class LLMClient:
    def __init__(self, cfg: ModelConfig):
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"环境变量 {cfg.api_key_env} 未设置")
        self.cfg = cfg
        headers = dict(cfg.headers)
        if cfg.session_header:
            headers[cfg.session_header] = str(uuid.uuid4())
        self.fallback = LLMClient(ModelConfig(**cfg.fallback)) if cfg.fallback else None
        self.client = OpenAI(api_key=key, base_url=cfg.base_url or None, timeout=cfg.timeout,
                             max_retries=0 if self.fallback else 2, default_headers=headers or None)
        self._sem = threading.Semaphore(cfg.max_concurrency)
        self._down_until = 0.0  # 在此时间前直接走备用服务

    def chat_json(self, system: str, user: str, images: list[str] | None = None,
                  temperature: float | None = None, retries: int = 3, timeout: float | None = None) -> dict:
        if self.fallback and time.time() < self._down_until:
            return self.fallback.chat_json(system, user, images, temperature, retries, timeout)
        try:
            return self._chat_json(system, user, images, temperature, retries, timeout)
        except Exception as e:
            if not self.fallback:
                raise
            cause = e.__cause__ or e
            if isinstance(cause, APIStatusError) and cause.status_code in (401, 402, 403):
                self._down_until = float("inf")
            elif _is_outage(cause):
                self._down_until = time.time() + self.cfg.cooldown
            log.warning("[%s] 切换到备用模型 %s: %s", self.cfg.model, self.fallback.cfg.model, str(e)[:120])
            return self.fallback.chat_json(system, user, images, temperature, retries, timeout)

    def _chat_json(self, system: str, user: str, images: list[str] | None,
                   temperature: float | None, retries: int, timeout: float | None) -> dict:
        content: str | list = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": _data_url(p), "detail": "low"}} for p in images
            ]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        kwargs = {"response_format": {"type": "json_object"}} if self.cfg.json_mode else {}
        last: Exception | None = None
        limit = timeout or self.cfg.timeout
        for attempt in range(retries):
            tag = f"[{self.cfg.model} #{next(_req_ids)}]"
            log.info("%s 发送请求（第 %d/%d 次）：输入 %d 字%s，超时 %ds", tag, attempt + 1, retries,
                     len(system) + len(user), f"、图片 {len(images)} 张" if images else "", limit)
            t0, stop, state = time.time(), threading.Event(), {"queued": True}
            threading.Thread(target=_heartbeat, args=(stop, tag, t0, limit, state), daemon=True).start()
            try:
                with self._sem:
                    state["queued"] = False
                    resp = self.client.chat.completions.create(
                        model=self.cfg.model, messages=messages,
                        temperature=self.cfg.temperature if temperature is None else temperature,
                        timeout=limit, **kwargs,
                    )
                text = resp.choices[0].message.content or ""
                log.info("%s 收到响应：%.0fs，输出 %d 字", tag, time.time() - t0, len(text))
                data = extract_json(text)
                if not isinstance(data, dict):
                    raise ValueError("模型输出不是 JSON 对象")
                return data
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("%s 调用失败 (%s/%s, %.0fs): %s", tag, attempt + 1, retries, time.time() - t0, str(e)[:200])
            finally:
                stop.set()
            if self.fallback and _is_outage(last):
                break
            time.sleep(2 ** attempt * 2)
        raise RuntimeError(f"LLM 调用多次失败: {last}") from last
