"""阶段2：内容理解 —— 逐段落多模态分析打分。"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import prompts
from .ingest import energy_peak
from .llm import LLMClient
from .utils import log, read_json, subs_in, ts, write_json

EMPTY = {"summary": "（分析失败）", "characters": [], "emotion": "", "scores": {}, "quotable_line": "", "is_filler": False}


def analyze_beats(ing: dict, mdir: Path, vlm: LLMClient) -> list[dict]:
    path = mdir / "beats_analysis.json"
    done: dict = read_json(path) if path.exists() else {}
    energy = ing["energy"]
    median = float(np.median(energy)) if energy else 0.0
    todo = [b for b in ing["beats"] if str(b["idx"]) not in done]
    if todo:
        log.info("[analyze] %s: 分析 %d 个段落", Path(ing["file"]).name, len(todo))
        lock = threading.Lock()

        def work(b: dict) -> None:
            res = _describe(b, ing, median, vlm)
            with lock:
                done[str(b["idx"])] = res
                if len(done) % 10 == 0:
                    write_json(path, done)

        with ThreadPoolExecutor(vlm.cfg.max_concurrency) as ex:
            for f in as_completed([ex.submit(work, b) for b in todo]):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001  单段失败不阻塞，下次运行会重试
                    log.warning("[analyze] 段落分析失败: %s", e)
        write_json(path, done)

    out = []
    for b in ing["beats"]:
        a = done.get(str(b["idx"]), EMPTY)
        out.append({**b, **a, "energy_peak": energy_peak(energy, b["start"], b["end"], median)})
    return out


def _describe(b: dict, ing: dict, median: float, vlm: LLMClient) -> dict:
    lines = subs_in(ing["subs"], b["start"], b["end"])
    sub_txt = "\n".join(f"[{ts(x['start'])}] {x['text']}" for x in lines) or "（无台词）"
    peak = energy_peak(ing["energy"], b["start"], b["end"], median)
    user = (f"片段时间：{ts(b['start'])} - {ts(b['end'])}（{b['end'] - b['start']:.0f}秒）\n"
            f"音量峰值：比全片中位数高 {peak:.0f} dB\n"
            f"台词：\n{sub_txt}\n\n关键帧 {len(b['frames'])} 张，按时间顺序。")
    r = vlm.chat_json(prompts.BEAT_SYSTEM, user, images=b["frames"])
    raw_scores = r.get("scores") or {}
    scores = {}
    for k in prompts.SCORE_KEYS:
        try:
            scores[k] = float(raw_scores.get(k, 0) or 0)
        except (TypeError, ValueError):
            scores[k] = 0.0
    return {
        "summary": str(r.get("summary", "")),
        "characters": r.get("characters") or [],
        "emotion": str(r.get("emotion", "")),
        "scores": scores,
        "quotable_line": str(r.get("quotable_line") or ""),
        "is_filler": bool(r.get("is_filler", False)),
    }
