from __future__ import annotations

import shutil
import threading
from pathlib import Path

from . import director, render, series, tts
from .config import Job, Settings
from .ingest import ingest
from .llm import LLMClient
from .understand import analyze_beats
from .utils import cached, file_key, log, short_hash, slug, write_json

STAGES = ["ingest", "analyze", "bible", "episode", "plan", "script", "tts", "render"]

_guard = threading.Lock()
_media_locks: dict[str, threading.Lock] = {}


def _media_lock(key: str) -> threading.Lock:
    with _guard:
        return _media_locks.setdefault(key, threading.Lock())


class Context:
    def __init__(self, settings: Settings):
        self.s = settings
        self._llm: LLMClient | None = None
        self._vlm: LLMClient | None = None
        self._lock = threading.Lock()

    @property
    def llm(self) -> LLMClient:
        with self._lock:
            self._llm = self._llm or LLMClient(self.s.llm)
            return self._llm

    @property
    def vlm(self) -> LLMClient:
        with self._lock:
            self._vlm = self._vlm or LLMClient(self.s.vlm)
            return self._vlm


def invalidate(job: Job, jdir: Path, s: Settings, stage: str) -> None:
    """删除 stage 及其之后所有阶段的缓存（剧集资料文件由 run_job 备份后重建）。"""
    idx = STAGES.index(stage)
    targets: list[Path] = []
    for f in job.files:
        mdir = s.workdir / "media" / file_key(f)
        if idx <= 0:
            targets += [mdir / "ingest.json", mdir / "cuts.json", mdir / "frames"]
        if idx <= 1:
            targets.append(mdir / "beats_analysis.json")
        if idx <= 3:
            targets += list(mdir.glob("episode_*.json"))
    if idx <= 4:
        targets += [jdir / "plan.json", *jdir.glob("v*/")]
    vdirs = list(jdir.glob("v*/"))
    if idx == 5:
        targets += [p for v in vdirs for p in (v / "script.json", v / "script.md")]
    if idx <= 6:
        targets += [v / "tts" for v in vdirs]
    targets += [v / "render" for v in vdirs]
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            t.unlink()
    log.info("[%s] 已清除 %s 及之后阶段的缓存", job.name, stage)


def _ingest_all(files: list[Path], s: Settings) -> list[dict]:
    out = []
    for f in files:
        key = file_key(f)
        with _media_lock(key):
            out.append(ingest(f, s.workdir / "media" / key, s))
    return out


def run_job(job: Job, ctx: Context, redo: str | None = None, stop_after: str | None = None) -> list[Path]:
    s = ctx.s
    jdir = s.workdir / "jobs" / slug(job.name)
    jdir.mkdir(parents=True, exist_ok=True)
    write_json(jdir / "job.json", {"name": job.name, "files": [str(f) for f in job.files], "episodes": job.episodes,
                                   "style": job.style, "target_duration": job.target_duration,
                                   "count": job.count, "render": job.render})
    if redo:
        invalidate(job, jdir, s, redo)

    def stop(stage: str) -> bool:
        return stop_after is not None and STAGES.index(stage) >= STAGES.index(stop_after)

    ingests, eps = [], []
    for f in job.files:
        key = file_key(f)
        mdir = s.workdir / "media" / key
        with _media_lock(key):
            ing = ingest(f, mdir, s)
            beats = [] if stop("ingest") else analyze_beats(ing, mdir, ctx.vlm)
        ingests.append(ing)
        eps.append(beats)
    if stop("analyze"):
        return []

    with _media_lock(f"bible:{job.series}"):
        bible = series.load_bible(job.series, _ingest_all(job.series_files or job.files, s), ctx.llm,
                                  force=redo == "bible")
    series.apply_name_fixes(ingests, series.name_fixes(bible))
    if stop("bible"):
        return []

    bkey = short_hash({k: bible.get(k) for k in ("intro", "characters", "episodes", "asr_fixes")})
    analyses = []
    for f, ing, beats, no in zip(job.files, ingests, eps, job.episodes):
        key = file_key(f)
        with _media_lock(f"episode:{key}"):
            analyses.append(series.analyze_episode(ing, beats, bible, no, ctx.llm,
                                                   s.workdir / "media" / key / f"episode_{bkey}.json"))
        series.apply_text_fixes(ing, analyses[-1])
    doc = series.assemble(ingests, analyses, job.episodes)
    if stop("episode"):
        return []

    plan = cached(jdir / "plan.json", lambda: director.make_plan(job, bible, doc, ctx.llm))
    if stop("plan"):
        return []

    outs = []
    videos = plan["videos"]
    for vi, v in enumerate(videos, 1):
        log.info("[%s] ━━ 成片 %d/%d：%s", job.name, vi, len(videos), v["title"])
        vdir = jdir / f"v{vi}"
        script = cached(vdir / "script.json", lambda: director.write_script(job, bible, doc, v, ctx.llm, s))
        _write_script_md(vdir / "script.md", script)
        if stop("script"):
            continue
        voices = tts.synthesize(script, vdir / "tts", s)
        if stop("tts"):
            continue
        name = slug(job.name) if len(videos) == 1 else f"{slug(job.name)}-{vi}"
        outs.append(render.render(job, script, voices, ingests, vdir / "render", s, name))
    return outs


def _write_script_md(path: Path, script: dict) -> None:
    out = [f"# {script.get('title', '')}", "", f"看点：{script.get('angle', '')}", ""]
    for i, seg in enumerate(script["segments"]):
        if seg["type"] == "narration":
            out.append(f"{i:02d}. 【解说】{seg['text']}  _({seg['beat_id']} @{seg['src_start']}s)_")
        else:
            out.append(f"{i:02d}. 【原声】{seg.get('quote', '')}  _({seg['beat_id']} {seg['src_start']}→{seg['src_end']}s)_")
    out += ["", f"文案：{script.get('description', '')}", f"标签：{' '.join(script.get('tags', []))}"]
    path.write_text("\n".join(out) + "\n", "utf-8")
