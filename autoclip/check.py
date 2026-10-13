"""成片自动验收：结果写进成片旁的 .json 的 check 字段，发布前按 ok 判断。"""
from __future__ import annotations

import re
from pathlib import Path

from .config import Job
from .utils import log, media_info, read_json, run, write_json

BAD_WORDS = re.compile(r"钩子|这条视频|开头那句|片头|点赞|关注我|订阅")
MIN_RATIO, MAX_RATIO = 0.4, 1.6  # 相对目标时长
MIN_DB = -45.0  # 平均音量低于此值视为几乎无声


def _mean_volume(path: Path) -> float | None:
    err = run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"]).stderr
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", err)
    return float(m.group(1)) if m else None


def check_script(script: dict) -> list[str]:
    """解说稿的文字问题，配音和渲染之前就能查。"""
    issues = [f"解说出现制作腔“{m.group()}”：{seg['text'][:20]}" for seg in script["segments"]
              if seg["type"] == "narration" and (m := BAD_WORDS.search(seg["text"]))]
    if not str(script.get("title") or "").strip():
        issues.append("标题为空")
    return issues


def verify(out: Path, script: dict, job: Job) -> dict:
    issues = check_script(script)
    if not out.exists() or out.stat().st_size == 0:
        issues.append("成片文件缺失或为空")
    else:
        info = media_info(out)
        if not info["has_audio"]:
            issues.append("成片没有音轨")
        elif (db := _mean_volume(out)) is None or db < MIN_DB:
            issues.append(f"成片几乎无声（平均 {db} dB）")
        if not MIN_RATIO * job.target_duration <= info["duration"] <= MAX_RATIO * job.target_duration:
            issues.append(f"时长 {info['duration']:.0f}s 偏离目标 {job.target_duration}s")
    res = {"ok": not issues, "issues": issues}
    meta = out.with_suffix(".json")
    if meta.exists():
        write_json(meta, {**read_json(meta), "check": res})
    (log.info if res["ok"] else log.warning)("[check] %s %s", out.name, "通过" if res["ok"] else issues)
    return res
