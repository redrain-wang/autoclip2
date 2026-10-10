"""阶段5：配音。"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Settings
from .utils import duration, log, read_json, short_hash, write_json


async def _edge(text: str, voice: str, rate: str, pitch: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(path))


def _openai(text: str, s: Settings, path: Path) -> None:
    from openai import OpenAI

    t = s.tts
    client = OpenAI(api_key=os.environ[t.api_key_env], base_url=t.base_url or None)
    kw = {"instructions": t.instructions} if t.instructions else {}
    with client.audio.speech.with_streaming_response.create(
            model=t.model, voice=t.voice, input=text, response_format="mp3", **kw) as r:
        r.stream_to_file(path)


def synthesize(script: dict, out_dir: Path, s: Settings) -> dict[int, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = s.tts

    def one(i: int, text: str) -> tuple[int, dict]:
        for k, v in t.pronounce.items():
            text = text.replace(k, v)
        p, meta = out_dir / f"n{i:03d}.mp3", out_dir / f"n{i:03d}.json"
        h = short_hash([text, t.backend, t.voice, t.rate, t.pitch, t.model, t.instructions])
        if p.exists() and meta.exists() and read_json(meta).get("hash") == h:
            return i, read_json(meta)
        for attempt in range(3):
            try:
                if t.backend == "openai":
                    _openai(text, s, p)
                else:
                    asyncio.run(_edge(text, t.voice, t.rate, t.pitch, p))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                log.warning("[tts] 重试 %d: %s", i, e)
        m = {"path": str(p), "duration": duration(p), "hash": h}
        write_json(meta, m)
        return i, m

    jobs = [(i, seg["text"]) for i, seg in enumerate(script["segments"]) if seg["type"] == "narration"]
    log.info("[tts] 合成 %d 段配音", len(jobs))
    with ThreadPoolExecutor(t.max_concurrency) as ex:
        return dict(ex.map(lambda x: one(*x), jobs))
