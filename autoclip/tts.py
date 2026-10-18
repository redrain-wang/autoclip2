"""阶段5：配音。"""
from __future__ import annotations

import asyncio
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Settings
from .utils import duration, log, read_json, short_hash, write_json

ATTEMPTS = 6      # 单段最多试几次（含首次）
RETRY_ROUND_WAIT = 30   # 整批补做前的等待秒数


async def _edge(text: str, voice: str, rate: str, pitch: str, path: Path) -> None:
    import aiohttp
    import edge_tts
    import edge_tts.communicate as ec
    from edge_tts.constants import TRUSTED_CLIENT_TOKEN, WSS_URL

    # 2026-10 起旧域名对多数地区返回 403（rany2/edge-tts#490），新域名沿用旧令牌即可；上游修复后可删
    hosts = [f"wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1?Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}", WSS_URL]
    for i, url in enumerate(hosts):
        ec.WSS_URL = url
        try:
            await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(path))
            return
        except aiohttp.WSServerHandshakeError as e:
            if e.status != 403 or i == len(hosts) - 1:
                raise
            log.warning("[tts] %s 返回 403，改用旧域名", url.split("?")[0])


def _openai(text: str, s: Settings, path: Path) -> None:
    from openai import OpenAI

    t = s.tts
    client = OpenAI(api_key=os.environ[t.api_key_env], base_url=t.base_url or None)
    kw = {"instructions": t.instructions} if t.instructions else {}
    with client.audio.speech.with_streaming_response.create(
            model=t.model, voice=t.voice, input=text, response_format="mp3", **kw) as r:
        r.stream_to_file(path)


def synthesize(script: dict, out_dir: Path, s: Settings, pronounce: dict | None = None) -> dict[int, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = s.tts
    words = {**t.pronounce, **(pronounce or {})}

    def one(i: int, text: str) -> tuple[int, dict]:
        for k, v in words.items():
            text = text.replace(k, v)
        p, meta = out_dir / f"n{i:03d}.mp3", out_dir / f"n{i:03d}.json"
        h = short_hash([text, t.backend, t.voice, t.rate, t.pitch, t.model, t.instructions])
        if p.exists() and meta.exists() and read_json(meta).get("hash") == h:
            return i, read_json(meta)
        for attempt in range(ATTEMPTS):
            try:
                if t.backend == "openai":
                    _openai(text, s, p)
                else:
                    asyncio.run(_edge(text, t.voice, t.rate, t.pitch, p))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == ATTEMPTS - 1:
                    raise
                # 并发重试会凑成"握手风暴"，所以退避加抖动；429 是明确的限流，单独多等一会儿
                wait = min(2 ** attempt * 2, 30) + random.uniform(0, 2)
                if "429" in str(e):
                    wait = max(wait, 25) + random.uniform(0, 10)
                log.warning("[tts] 第 %d 段第 %d/%d 次失败，%.0fs 后重试: %s",
                            i, attempt + 1, ATTEMPTS, wait, str(e)[:70])
                time.sleep(wait)
        m = {"path": str(p), "duration": duration(p), "hash": h}
        write_json(meta, m)
        return i, m

    jobs = [(i, seg["text"]) for i, seg in enumerate(script["segments"]) if seg["type"] == "narration"]
    log.info("[tts] 合成 %d 段配音（并发 %d）", len(jobs), t.max_concurrency)
    done: dict[int, dict] = {}
    failed: list[tuple[int, str]] = []
    with ThreadPoolExecutor(t.max_concurrency) as ex:
        futs = {ex.submit(one, *job): job for job in jobs}
        for f in as_completed(futs):
            try:
                i, m = f.result()
                done[i] = m
            except Exception as e:  # noqa: BLE001  单段失败不立刻判死，最后再补一轮
                failed.append(futs[f])
                log.warning("[tts] 第 %d 段多次重试仍失败: %s", futs[f][0], str(e)[:80])
    if failed:
        log.warning("[tts] %d 段失败，等 %d 秒后串行补一轮", len(failed), RETRY_ROUND_WAIT)
        time.sleep(RETRY_ROUND_WAIT)
        for i, text in sorted(failed):
            done[i] = one(i, text)[1]
    return done
