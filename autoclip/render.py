"""阶段6：剪辑合成 —— 逐段渲染（解说段压低原声叠配音），拼接，烧录字幕/标题，混 BGM。"""
from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import brand
from .config import Job, RenderConfig, Settings, render_cfg
from .utils import file_key, log, read_json, run, short_hash, slug, subs_in, write_json

NARR_PAD = 0.35  # 配音前后留白
NARR_DELAY = 0.15


def black_ranges(file: str, cache: Path) -> list[list[float]]:
    """黑场（含黑底白字的集数卡）区间，解说画面要避开。"""
    if cache.exists():
        return read_json(cache)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", file, "-an", "-vf",
                        "scale=320:-2,blackdetect=d=0.3:pix_th=0.05:pic_th=0.95", "-f", "null", "-"],
                       capture_output=True, text=True)
    out = [[float(a), float(b)] for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", p.stderr)]
    write_json(cache, out)
    return out


def _skip_black(st: float, dur: float, black: list[list[float]], limit: float) -> float:
    for _ in range(len(black)):
        hit = next((b for b in black if st < b[1] and st + dur > b[0]), None)
        if not hit or hit[1] + 0.1 + dur > limit:
            break
        st = hit[1] + 0.1
    return st


def build_timeline(script: dict, voices: dict[int, dict], ingests: list[dict],
                   blacks: list[list[list[float]]] | None = None) -> list[dict]:
    items, t, prev = [], 0.0, None
    for i, seg in enumerate(script["segments"]):
        ing = ingests[seg["ep"]]
        item = {"type": seg["type"], "beat_id": seg["beat_id"], "file": ing["file"], "has_audio": ing["has_audio"]}
        st = seg["src_start"]
        if seg["type"] == "narration":
            v = voices[i]
            dur = v["duration"] + NARR_PAD
            # 同一段落内连续播放，避免画面回跳重复
            if prev and prev["beat_id"] == seg["beat_id"] and st < prev["src_end"] - 0.5:
                st = prev["src_end"]
            st = max(0.0, min(st, ing["duration"] - dur - 0.1))
            if span := seg.get("scene_span"):
                # 解说画面不跑出本场戏（否则会插进别的场景）
                st = max(span[0], min(st, span[1] - dur))
            if blacks:
                st = _skip_black(st, dur, blacks[seg["ep"]], ing["duration"] - 0.1)
            item.update(audio=v["path"], audio_hash=v["hash"], tts_dur=v["duration"], text=seg["text"])
        else:
            dur = seg["src_end"] - st
            item["subs"] = subs_in(ing["subs"], st, st + dur)
        item.update(src_start=round(st, 3), dur=round(dur, 3), src_end=round(st + dur, 3), offset=round(t, 3))
        t += dur
        prev = item
        items.append(item)
    return items


def _video_chain(r: RenderConfig, W: int, H: int) -> str:
    pre = f"[0:v:0]scale=iw*sar:ih,setsar=1,fps={r.fps}"
    if r.layout == "vertical":
        return (f"{pre},split[s1][s2];"
                f"[s1]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
                f"boxblur=10:2,scale={W}:{H},eq=brightness=-0.12[bg];"
                f"[s2]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v]")
    return (f"{pre},scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v]")


def render_segment(item: dict, out: Path, r: RenderConfig, W: int, H: int) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{item['src_start']:.3f}", "-t", f"{item['dur']:.3f}", "-i", item["file"]]
    fmt = "aresample=48000:async=1:first_pts=0,aformat=sample_fmts=fltp:channel_layouts=stereo"
    orig = f"[0:a:0]{fmt}" if item["has_audio"] else f"anullsrc=r=48000:cl=stereo,atrim=duration={item['dur']:.3f}"
    norm = "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"
    fc = _video_chain(r, W, H) + ";"
    if item["type"] == "narration":
        cmd += ["-i", item["audio"]]
        d = int(NARR_DELAY * 1000)
        fc += (f"{orig},volume={r.duck_volume}[o];[1:a]{fmt},adelay={d}|{d},apad[n];"
               f"[o][n]amix=inputs=2:duration=first:normalize=0,{norm}[a]")
    else:
        fc += f"{orig},{norm}[a]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{item['dur']:.3f}",
            "-c:v", "libx264", "-preset", r.preset, "-crf", str(r.crf), "-r", str(r.fps),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out]
    run(cmd)


# ---------------- 字幕 ----------------

def _ass_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _ass_text(s: str) -> str:
    return s.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ")


def split_caption(text: str, maxlen: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[，。！？；、,.!?;：:…])", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        while len(p) > maxlen:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:maxlen])
            p = p[maxlen:]
        if len(cur) + len(p) <= maxlen:
            cur += p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    strip = "，。、,；;：: "
    return [c.strip(strip) for c in chunks if c.strip(strip)]


def build_ass(items: list[dict], title: str, W: int, H: int, fg_h: int, r: RenderConfig) -> str:
    vertical = r.layout == "vertical"
    narr_sz, dlg_sz, title_sz = (68, 56, 84) if vertical else (56, 48, 52)
    maxlen = 14 if vertical else 22
    if vertical:
        sub_pos = f"{{\\an8\\pos({W // 2},{(H + fg_h) // 2 + 50})}}"
        title_pos = f"{{\\an2\\pos({W // 2},{(H - fg_h) // 2 - 60})}}"
    else:
        sub_pos, title_pos = "", f"{{\\an8\\pos({W // 2},30)}}"
    f = r.font
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Narr,{f},{narr_sz},&H0000E6FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,5,2,2,40,40,70,1",
        f"Style: Dlg,{f},{dlg_sz},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,4,2,2,40,40,70,1",
        f"Style: Title,{f},{title_sz},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,3,0,1,6,0,2,40,40,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    total = items[-1]["offset"] + items[-1]["dur"] if items else 0
    if r.show_title and title:
        t = "\\N".join(_ass_text(c) for c in split_caption(title, 12 if vertical else 30)) or _ass_text(title)
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(total)},Title,,0,0,0,,{title_pos}{t}")

    def add(style: str, st: float, en: float, text: str) -> None:
        if en - st > 0.05:
            lines.append(f"Dialogue: 0,{_ass_time(st)},{_ass_time(en)},{style},,0,0,0,,{sub_pos}{_ass_text(text)}")

    for it in items:
        if it["type"] == "narration":
            chunks = split_caption(it["text"], maxlen)
            weights = [max(len(c), 2) for c in chunks]
            t0, span = it["offset"] + NARR_DELAY, it["tts_dur"]
            acc = 0
            for c, w in zip(chunks, weights):
                st = t0 + span * acc / sum(weights)
                acc += w
                add("Narr", st, t0 + span * acc / sum(weights), c)
        elif r.show_dialogue_subs:
            for x in it["subs"]:
                st = it["offset"] + max(0.0, x["start"] - it["src_start"])
                en = it["offset"] + min(it["dur"], x["end"] - it["src_start"])
                add("Dlg", st, en, x["text"])
    return "\n".join(lines) + "\n"


# ---------------- 主流程 ----------------

def make_cover(job: Job, script: dict, items: list[dict], logo: Path | None, out: Path, r: RenderConfig) -> Path:
    eps = sorted({job.episodes[seg["ep"]] for seg in script["segments"]}) if job.episodes else []
    tag = f"《{job.series}》" if job.series else ""
    if len(eps) == 1:
        tag += f"第{eps[0]}集"
    frame = brand.pick_frame(items)
    brand.make_cover(frame, script.get("cover") or script.get("title", ""), tag, logo, out,
                     r.layout == "vertical", r.font)
    log.info("[render] 封面 → %s", out)
    return out

def render(job: Job, script: dict, voices: dict[int, dict], ingests: list[dict], rdir: Path, s: Settings,
           out_name: str) -> Path:
    r = render_cfg(s, job)
    rdir.mkdir(parents=True, exist_ok=True)
    W, H = (1080, 1920) if r.layout == "vertical" else (1920, 1080)
    src_w, src_h = ingests[0]["width"] or 16, ingests[0]["height"] or 9
    fg_h = min(H, round(W * src_h / src_w / 2) * 2)
    items = build_timeline(script, voices, ingests,
                           [black_ranges(i["file"], s.workdir / "media" / file_key(Path(i["file"])) / "black.json")
                            for i in ingests])

    seg_files = []
    for i, it in enumerate(items):
        key = short_hash([it["file"], it["src_start"], it["dur"], it["type"], it.get("audio_hash"),
                          r.layout, r.fps, r.crf, r.duck_volume])
        seg_files.append(rdir / f"seg_{i:03d}_{key}.mp4")
    todo = [(it, f) for it, f in zip(items, seg_files) if not f.exists()]
    log.info("[render] 共 %d 段，需渲染 %d 段，总时长 %.0fs", len(items), len(todo), sum(i["dur"] for i in items))

    def work(x):
        it, f = x
        tmp = f.with_name("tmp_" + f.name)
        render_segment(it, tmp, r, W, H)
        tmp.replace(f)

    with ThreadPoolExecutor(r.workers) as ex:
        list(ex.map(work, todo))

    (rdir / "concat.txt").write_text("".join(f"file '{f.name}'\n" for f in seg_files), "utf-8")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", "joined.mp4"], cwd=rdir)
    (rdir / "subs.ass").write_text(build_ass(items, script.get("title", ""), W, H, fg_h, r), "utf-8")

    s.output_dir.mkdir(parents=True, exist_ok=True)
    out = s.output_dir / f"{out_name}.mp4"
    logo = brand.ensure_logo(r.watermark, r.brand, r.font) if r.watermark else None
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", "joined.mp4"]
    fc = "[0:v]ass=subs.ass[v]"
    if logo:
        vertical = r.layout == "vertical"
        lw = round(W * (r.watermark_scale if vertical else 0.16) / 2) * 2
        mx = r.watermark_margin if vertical else 30
        y = (H - fg_h) // 2 + 24 if vertical else 36
        cmd += ["-i", str(logo)]
        fc = (f"[0:v]ass=subs.ass[s];[1:v]scale={lw}:-1,format=rgba,colorchannelmixer=aa={r.watermark_opacity}[wm];"
              f"[s][wm]overlay=W-w-{mx}:{y}[v]")
    if r.bgm:
        cmd += ["-stream_loop", "-1", "-i", str(Path(r.bgm).expanduser().resolve())]
        fc += f";[{2 if logo else 1}:a]aresample=48000,volume={r.bgm_volume}[b];[0:a][b]amix=inputs=2:duration=first:normalize=0[a]"
        amap = ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    else:
        amap = ["-map", "0:a", "-c:a", "copy"]
    # 先写临时文件再改名，避免播放器打开渲染中的半成品
    tmp = out.with_name(f".{out.stem}.tmp.mp4")
    cmd += ["-filter_complex", fc, "-map", "[v]", *amap, "-c:v", "libx264", "-preset", r.preset,
            "-crf", str(r.crf), "-movflags", "+faststart", tmp]
    run(cmd, cwd=rdir)
    tmp.replace(out)
    if r.cover:
        make_cover(job, script, items, logo, out.with_suffix(".jpg"), r)

    write_json(out.with_suffix(".json"), {k: script.get(k) for k in ("title", "cover", "description", "tags", "angle", "genre")}
               | {"timeline": [{k: v for k, v in it.items() if k != "subs"} for it in items]})
    log.info("[render] 完成 → %s", out)
    return out
