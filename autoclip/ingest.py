"""阶段1：素材预处理 —— 音频、字幕/ASR、镜头切分、音量曲线、剧情段落(beat)切分、关键帧。"""
from __future__ import annotations

import math
import os
import re
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import Settings
from .utils import cached, duration, log, media_info, read_json, run, write_json

_whisper_lock = threading.Lock()
_whisper_model = None


def ingest(video: Path, mdir: Path, s: Settings) -> dict:
    mdir.mkdir(parents=True, exist_ok=True)
    out = mdir / "ingest.json"
    if out.exists():
        data = read_json(out)
        _simplify(data["subs"])
        return data
    log.info("[ingest] %s", video.name)
    info = media_info(video)
    wav = mdir / "audio.wav"
    if info["has_audio"] and not wav.exists():
        run(["ffmpeg", "-y", "-v", "error", "-i", video, "-map", "0:a:0", "-vn",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
    subs = load_subtitles(video, wav if info["has_audio"] else None, mdir, s)
    _simplify(subs)
    cuts = cached(mdir / "cuts.json", lambda: detect_cuts(video, s.analysis.scene_threshold))
    energy = audio_energy(wav) if info["has_audio"] else []
    beats = build_beats(info["duration"], cuts, subs, s)
    extract_frames(video, beats, mdir / "frames", s)
    data = {"file": str(video), **info, "subs": subs, "cuts": cuts, "energy": energy, "beats": beats}
    write_json(out, data)
    log.info("[ingest] %s: 时长 %.0fs, 台词 %d 条, 镜头 %d, 段落 %d",
             video.name, info["duration"], len(subs), len(cuts), len(beats))
    return data


# ---------------- 字幕 / ASR ----------------

_t2s = None


def _simplify(subs: list[dict]) -> None:
    """Whisper 偶尔输出繁体，统一转简体。"""
    global _t2s
    if _t2s is None:
        from opencc import OpenCC
        _t2s = OpenCC("t2s")
    for x in subs:
        x["text"] = _t2s.convert(x["text"])


def apply_fixes(text: str, fixes: dict[str, str]) -> str:
    for wrong in sorted(fixes, key=len, reverse=True):
        text = text.replace(wrong, fixes[wrong])
    return text

def find_sidecar_srt(video: Path) -> Path | None:
    cands = sorted(video.parent.glob(f"{glob_escape(video.stem)}*.srt"))
    zh = [c for c in cands if re.search(r"(zh|chs|chi|sc|简)", c.name, re.I)]
    return (zh or cands or [None])[0]


def glob_escape(s: str) -> str:
    return re.sub(r"([*?\[\]])", r"[\1]", s)


def parse_srt(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    pat = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    subs = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = pat.search(line)
            if not m:
                continue
            g = [int(x) for x in m.groups()]
            st = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 10 ** len(m.group(4))
            en = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 10 ** len(m.group(8))
            body = " ".join(lines[i + 1:])
            body = re.sub(r"<[^>]+>|\{[^}]*\}", "", body).strip()
            if body:
                subs.append({"start": round(st, 2), "end": round(en, 2), "text": body})
            break
    return sorted(subs, key=lambda x: x["start"])


def load_subtitles(video: Path, wav: Path | None, mdir: Path, s: Settings) -> list[dict]:
    p = mdir / "subs.json"
    if p.exists():
        return read_json(p)
    srt = find_sidecar_srt(video) if s.asr.use_sidecar_srt else None
    if srt:
        log.info("[asr] 使用外挂字幕 %s", srt.name)
        subs = parse_srt(srt)
    elif wav is None:
        subs = []
    elif s.asr.backend == "openai":
        subs = _asr_openai(wav, mdir, s)
    elif s.asr.backend == "mlx":
        subs = _asr_mlx(wav, s)
    else:
        subs = _asr_local(wav, s)
    write_json(p, subs)
    return subs


def _asr_local(wav: Path, s: Settings) -> list[dict]:
    global _whisper_model
    from faster_whisper import WhisperModel

    with _whisper_lock:
        if _whisper_model is None:
            log.info("[asr] 加载 faster-whisper %s", s.asr.model)
            _whisper_model = WhisperModel(s.asr.model, device="auto", compute_type="int8")
        log.info("[asr] 本地识别 %s", wav)
        prompt = "以下是普通话的句子，使用简体中文和标点。" if s.asr.language.startswith("zh") else None
        segs, _ = _whisper_model.transcribe(str(wav), language=s.asr.language, vad_filter=True,
                                            initial_prompt=prompt)
        return [{"start": round(x.start, 2), "end": round(x.end, 2), "text": x.text.strip()}
                for x in segs if x.text.strip()]


def _asr_mlx(wav: Path, s: Settings) -> list[dict]:
    import mlx_whisper

    with _whisper_lock:
        log.info("[asr] mlx-whisper 识别 %s", wav)
        prompt = "以下是普通话的句子，使用简体中文和标点。" if s.asr.language.startswith("zh") else None
        r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=s.asr.model, language=s.asr.language,
                                   initial_prompt=prompt, condition_on_previous_text=False)
    return [{"start": round(x["start"], 2), "end": round(x["end"], 2), "text": x["text"].strip()}
            for x in r["segments"] if x["text"].strip()]


def _asr_openai(wav: Path, mdir: Path, s: Settings) -> list[dict]:
    from openai import OpenAI

    cdir = mdir / "asr_chunks"
    cdir.mkdir(exist_ok=True)
    if not any(cdir.glob("c*.mp3")):
        run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-f", "segment", "-segment_time", "600",
             "-c:a", "libmp3lame", "-b:a", "32k", cdir / "c%04d.mp3"])
    client = OpenAI(api_key=os.environ[s.asr.api_key_env], base_url=s.asr.api_base_url or None)
    out, offset = [], 0.0
    for f in sorted(cdir.glob("c*.mp3")):
        log.info("[asr] API 识别 %s", f.name)
        with open(f, "rb") as fh:
            r = client.audio.transcriptions.create(
                model=s.asr.api_model, file=fh, language=s.asr.language[:2],
                response_format="verbose_json", timestamp_granularities=["segment"])
        for seg in getattr(r, "segments", None) or []:
            text = (getattr(seg, "text", "") or "").strip()
            if text:
                out.append({"start": round(seg.start + offset, 2), "end": round(seg.end + offset, 2), "text": text})
        offset += duration(f)
    return out


# ---------------- 画面 / 音频信号 ----------------

def detect_cuts(video: Path, threshold: float) -> list[float]:
    log.info("[shots] 镜头检测 %s", video.name)
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-hwaccel", "auto", "-i", video, "-map", "0:v:0", "-an", "-sn",
             "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"])
    return sorted(round(float(x), 2) for x in re.findall(r"pts_time:([\d.]+)", r.stderr))


def audio_energy(wav: Path) -> list[float]:
    """每秒 RMS 音量 (dBFS)。笑声、争吵、爆炸等高光往往伴随音量峰值。"""
    out = []
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        while frames := w.readframes(sr):
            a = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
            out.append(round(float(20 * np.log10(np.sqrt(np.mean(a * a)) + 1e-9)), 1))
    return out


def build_beats(total: float, cuts: list[float], subs: list[dict], s: Settings) -> list[dict]:
    """把镜头合并成 15~50 秒的“剧情段落”，尽量不在台词中间切。"""
    a = s.analysis
    start, end = a.skip_head, total - a.skip_tail

    def in_sub(t: float) -> bool:
        return any(x["start"] + 0.2 < t < x["end"] - 0.2 for x in subs)

    spans, cur = [], start
    for c in [c for c in cuts if start < c < end] + [end]:
        length = c - cur
        if c != end and (length < a.beat_min or (in_sub(c) and length < a.beat_max)):
            continue
        n = max(1, round(length / a.beat_max)) if length > a.beat_max * 1.3 else 1
        step = length / n
        spans += [(cur + k * step, cur + (k + 1) * step) for k in range(n)]
        cur = c
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < a.beat_min / 2:
        spans[-2:] = [(spans[-2][0], spans[-1][1])]
    return [{"idx": i, "start": round(st, 2), "end": round(en, 2)} for i, (st, en) in enumerate(spans)]


def extract_frames(video: Path, beats: list[dict], fdir: Path, s: Settings) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    n = s.analysis.frames_per_beat
    tasks = []
    for b in beats:
        b["frames"] = []
        for k in range(n):
            t = b["start"] + (k + 0.5) / n * (b["end"] - b["start"])
            p = fdir / f"b{b['idx']:04d}_{k}.jpg"
            b["frames"].append(str(p))
            if not p.exists():
                tasks.append((t, p))

    def grab(t: float, p: Path) -> None:
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
             "-vf", f"scale={s.analysis.frame_width}:-2", "-q:v", "4", p])

    if tasks:
        log.info("[frames] 抽取关键帧 %d 张", len(tasks))
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda x: grab(*x), tasks))
    for b in beats:
        b["frames"] = [f for f in b["frames"] if Path(f).exists()]


def energy_peak(energy: list[float], start: float, end: float, median: float) -> float:
    seg = energy[int(start): int(math.ceil(end))]
    return round(max(seg) - median, 1) if seg else 0.0
