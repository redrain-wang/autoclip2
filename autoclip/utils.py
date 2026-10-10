from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("autoclip")


def run(cmd: list, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    log.debug("$ %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=cwd,
                       stdin=subprocess.DEVNULL)  # 后台运行时子进程读取终端会被挂起
    if r.returncode != 0:
        raise RuntimeError(f"命令失败 ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-3000:]}")
    return r


def ffprobe(path: str | Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]).stdout
    return json.loads(out)


def media_info(path: str | Path) -> dict:
    p = ffprobe(path)
    v = next((s for s in p["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in p["streams"] if s["codec_type"] == "audio"), None)
    return {
        "duration": float(p["format"]["duration"]),
        "width": int(v["width"]) if v else 0,
        "height": int(v["height"]) if v else 0,
        "has_audio": a is not None,
    }


def duration(path: str | Path) -> float:
    return float(ffprobe(path)["format"]["duration"])


def file_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def short_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def cached(path: str | Path, fn: Callable[[], Any]) -> Any:
    path = Path(path)
    if path.exists():
        return read_json(path)
    data = fn()
    write_json(path, data)
    return data


def ts(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}"


def slug(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_")[:80] or "job"


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def subs_in(subs: list[dict], start: float, end: float) -> list[dict]:
    return [s for s in subs if s["end"] > start and s["start"] < end]


def extract_json(text: str) -> Any:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    end = max(text.rfind("}"), text.rfind("]"))
    if starts and end > min(starts):
        return json.loads(text[min(starts): end + 1])
    raise ValueError(f"无法从模型输出中解析 JSON: {text[:300]}")
