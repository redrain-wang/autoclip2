from __future__ import annotations

import glob as globlib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml

from .utils import natural_key

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".webm", ".rmvb", ".wmv"}


@dataclass
class ModelConfig:
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_concurrency: int = 4
    json_mode: bool = True
    headers: dict = field(default_factory=dict)
    session_header: str | None = None  # 每个进程生成稳定会话 ID，如 OpenCode 的 x-opencode-session
    fallback: dict | None = None       # 主服务失败/额度用尽时使用的备用模型配置
    timeout: float = 240.0             # 单次调用总时长上限
    stall_timeout: float = 90.0        # 流式输出中相邻数据包的最长间隔，超过视为卡住
    cooldown: float = 120.0            # 主服务连续失败后改走备用的秒数


@dataclass
class ASRConfig:
    backend: str = "faster-whisper"  # faster-whisper | mlx | openai
    model: str = "large-v3"
    language: str = "zh"
    use_sidecar_srt: bool = True
    api_model: str = "whisper-1"
    api_base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class TTSConfig:
    backend: str = "edge"  # edge | openai
    voice: str = "zh-CN-YunxiNeural"
    rate: str = "+12%"
    pitch: str = "+0Hz"
    model: str = "gpt-4o-mini-tts"
    instructions: str = ""
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_concurrency: int = 2   # edge 的 WebSocket 并发一高就吃 429，2 路实测最稳
    pronounce: dict = field(default_factory=dict)  # 仅用于配音的替换，纠正多音字（字幕不变）


@dataclass
class AnalysisConfig:
    scene_threshold: float = 0.3
    beat_min: float = 15.0
    beat_max: float = 50.0
    frames_per_beat: int = 4
    frame_width: int = 512
    skip_head: float = 0.0
    skip_tail: float = 0.0


@dataclass
class RenderConfig:
    layout: str = "vertical"  # vertical | horizontal
    fps: int = 30
    font: str = "PingFang SC"
    duck_volume: float = 0.12
    bgm: str | None = None
    bgm_volume: float = 0.06
    crf: int = 20
    preset: str = "medium"
    show_title: bool = True
    show_dialogue_subs: bool = True
    narration_speed: float = 4.5  # 字/秒，用于估算解说字数预算
    narration_ratio: float = 0.65  # 解说时长占比，其余为原声
    workers: int = 3
    brand: str = "爱大片儿"           # 账号名，用于生成水印和封面
    watermark: str | None = "assets/brand/logo.png"  # 不存在时按 brand 自动生成；null 关闭
    watermark_opacity: float = 0.85
    watermark_scale: float = 0.22   # 竖屏水印宽度占画面宽度比例
    watermark_margin: int = 90      # 竖屏水印距右边缘像素；抖音右侧会被界面遮挡/裁切
    cover: bool = True                # 同时导出封面图 <成片名>.jpg


@dataclass
class Settings:
    llm: ModelConfig
    vlm: ModelConfig
    workdir: Path = Path("work")
    output_dir: Path = Path("output")
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    presets: dict = field(default_factory=dict)


def load_settings(path: str | Path) -> Settings:
    raw = yaml.safe_load(Path(path).read_text("utf-8")) or {}
    llm = ModelConfig(**raw["llm"])
    return Settings(
        llm=llm,
        vlm=ModelConfig(**raw["vlm"]) if raw.get("vlm") else llm,
        workdir=Path(raw.get("workdir", "work")).resolve(),
        output_dir=Path(raw.get("output_dir", "output")).resolve(),
        asr=ASRConfig(**raw.get("asr", {})),
        tts=TTSConfig(**raw.get("tts", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
        render=RenderConfig(**raw.get("render", {})),
        presets=raw.get("presets") or {},
    )


RENDER_FIELDS = {f.name for f in fields(RenderConfig)}


@dataclass
class Job:
    name: str
    files: list[Path]
    genre: str = "auto"
    style: str = ""  # 额外风格要求，如“毒舌一点”“女性视角”
    target_duration: int = 180
    count: int | str = 1  # 每个任务出几条成片；"auto" 由导演根据素材决定
    max_count: int = 6    # count=auto 时的上限
    render: dict = field(default_factory=dict)  # 覆盖全局 render 配置
    series: str = ""      # 剧名，用于定位 series/<剧名>.yaml
    series_files: list[Path] = field(default_factory=list)  # 全剧所有集，用于生成剧集资料
    episodes: list[int] = field(default_factory=list)       # files 对应的集号（1 起）


def render_cfg(s: Settings, job: Job) -> RenderConfig:
    return replace(s.render, **job.render)


def expand_inputs(spec: str | list[str]) -> list[Path]:
    items = spec if isinstance(spec, list) else [spec]
    files: list[Path] = []
    for s in items:
        p = Path(s).expanduser()
        if p.is_dir():
            files += sorted((f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT and not f.name.startswith(".")), key=natural_key)
        elif any(c in str(s) for c in "*?["):
            files += sorted((Path(f) for f in globlib.glob(str(p))), key=natural_key)
        else:
            files.append(p)
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"找不到文件: {missing}")
    if not files:
        raise FileNotFoundError(f"没有匹配到视频: {spec}")
    return [f.resolve() for f in files]


def parse_episodes(spec) -> set[int] | None:
    """"1,3,5-8" / [1, 2] → {1,3,5,6,7,8}；None 表示全部。"""
    if spec in (None, "", "all"):
        return None
    out: set[int] = set()
    for part in (spec if isinstance(spec, list) else str(spec).split(",")):
        a, _, b = str(part).strip().partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def make_jobs(name: str, inputs, mode: str = "whole", episodes=None, series: str = "", **kw) -> list[Job]:
    """mode=whole: 所选文件作为一个作品出一条视频；per_episode: 每集出一条。"""
    files = expand_inputs(inputs)
    keep = parse_episodes(episodes)
    indexed = [(i + 1, f) for i, f in enumerate(files) if keep is None or i + 1 in keep]
    if not indexed:
        raise ValueError(f"episodes={episodes} 没有匹配到任何集")
    kw.update(series=series or name, series_files=files)
    if mode == "per_episode":
        return [Job(name=f"{name}_{i:02d}", files=[f], episodes=[i], **kw) for i, f in indexed]
    return [Job(name=name, files=[f for _, f in indexed], episodes=[i for i, _ in indexed], **kw)]


def build_jobs(name: str, inputs, s: Settings, presets: list[str] | None = None, **over) -> list[Job]:
    """每个预设生成一组任务，任务名追加预设名；显式参数覆盖预设。"""
    over = {k: v for k, v in over.items() if v is not None}
    jobs: list[Job] = []
    for p in presets or [None]:
        if p is not None and p not in s.presets:
            raise ValueError(f"未定义的预设: {p}，可选: {list(s.presets)}")
        cfg = {**(s.presets.get(p) or {}), **over}
        render = {k: cfg.pop(k) for k in list(cfg) if k in RENDER_FIELDS}
        jobs += make_jobs(f"{name}_{p}" if p else name, inputs, render=render, series=name, **cfg)
    return jobs


def load_jobs(path: str | Path, s: Settings) -> list[Job]:
    raw = yaml.safe_load(Path(path).read_text("utf-8")) or {}
    jobs: list[Job] = []
    for j in raw.get("jobs", []):
        j = dict(j)
        presets = j.pop("presets", None)
        if isinstance(presets, str):
            presets = [presets]
        jobs += build_jobs(j.pop("name"), j.pop("input"), s, presets, **j)
    return jobs
