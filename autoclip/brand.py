"""账号品牌：台标/水印图、头像、成片封面。"""
from __future__ import annotations

import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

RED, ORANGE, YELLOW = (255, 45, 85), (255, 149, 0), (255, 222, 0)


@lru_cache(maxsize=None)
def _font_file(family: str, weight: str) -> tuple[str, int]:
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}|%{index}", f"{family}:weight={weight}"],
                             capture_output=True, text=True, check=True).stdout
        file, idx = out.split("|")
        return file, int(idx or 0)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "/System/Library/Fonts/STHeiti Medium.ttc", 0


def _font(size: int, family: str = "PingFang SC", weight: str = "heavy") -> ImageFont.FreeTypeFont:
    file, idx = _font_file(family, weight)
    return ImageFont.truetype(file, size, index=idx)


def _gradient(w: int, h: int, c1: tuple, c2: tuple) -> Image.Image:
    lin = Image.linear_gradient("L")
    g = Image.blend(lin.resize((w, h)), lin.rotate(90).resize((w, h)), 0.5)
    return Image.composite(Image.new("RGB", (w, h), c2), Image.new("RGB", (w, h), c1), g)


def _badge(d: int) -> Image.Image:
    """圆形渐变底 + 白色播放三角。"""
    s = d * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, s - 1, s - 1), fill=255)
    img.paste(_gradient(s, s, RED, ORANGE), (0, 0), mask)
    ImageDraw.Draw(img).polygon([(s * 0.38, s * 0.27), (s * 0.38, s * 0.73), (s * 0.76, s * 0.5)], fill="white")
    return img.resize((d, d), Image.LANCZOS)


def make_logo(path: Path, name: str, family: str = "PingFang SC") -> Path:
    """透明底横版水印：圆形播放徽标 + 描边文字。"""
    h = 160
    font = _font(int(h * 0.62), family)
    tw = int(font.getlength(name))
    pad, gap, stroke = 16, 22, 7
    w = pad * 2 + h + gap + tw + stroke * 2
    img = Image.new("RGBA", (w, h + pad * 2), (0, 0, 0, 0))
    img.alpha_composite(_badge(h), (pad, pad))
    text = Image.new("RGBA", img.size, (0, 0, 0, 0))
    y = pad + h // 2
    ImageDraw.Draw(text).text((pad + h + gap + stroke, y), name, font=font, fill="white", anchor="lm",
                              stroke_width=stroke, stroke_fill=(0, 0, 0, 200))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(text.getchannel("A").point(lambda a: a * 0.5).filter(ImageFilter.GaussianBlur(6)))
    img.alpha_composite(shadow, (3, 4))
    img.alpha_composite(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def make_avatar(path: Path, name: str, size: int = 1024, family: str = "PingFang SC") -> Path:
    """方形头像/台标：渐变底 + 播放徽标 + 账号名。"""
    img = _gradient(size, size, RED, ORANGE).convert("RGBA")
    d = int(size * 0.36)
    ring = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((0, 0, d - 1, d - 1), fill="white")
    tri = [(d * 0.38, d * 0.27), (d * 0.38, d * 0.73), (d * 0.76, d * 0.5)]
    ImageDraw.Draw(ring).polygon(tri, fill=RED)
    img.alpha_composite(ring, ((size - d) // 2, int(size * 0.14)))
    font = _font(int(size * 0.2), family)
    ImageDraw.Draw(img).text((size // 2, int(size * 0.76)), name, font=font, fill="white", anchor="mm",
                             stroke_width=int(size * 0.008), stroke_fill=(150, 0, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=95)
    return path


def ensure_logo(path: str | Path, name: str, family: str = "PingFang SC") -> Path:
    p = Path(path).expanduser().resolve()
    return p if p.exists() else make_logo(p, name, family)


# ---------------- 封面 ----------------

def _grab(file: str, t: float, out: Path) -> Image.Image | None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", file, "-frames:v", "1",
                    "-vf", "scale=iw*sar:ih,setsar=1", str(out)], check=False)
    return Image.open(out).convert("RGB") if out.exists() else None


def _score(img: Image.Image) -> float:
    st = ImageStat.Stat(img.convert("L").resize((160, 90)))
    edges = ImageStat.Stat(img.convert("L").resize((320, 180)).filter(ImageFilter.FIND_EDGES))
    bright = st.mean[0]
    return st.stddev[0] + edges.mean[0] * 2 - (40 if bright < 45 or bright > 210 else 0)


def pick_frame(items: list[dict], n: int = 8) -> Image.Image:
    """在原声对白段里抽候选帧，挑对比度/清晰度最高的一张。"""
    pool = [it for it in items if it["type"] != "narration"] or items
    step = max(1, len(pool) // n)
    cands = pool[::step][:n]
    best, best_score = None, -1e9
    with tempfile.TemporaryDirectory() as td:
        for i, it in enumerate(cands):
            img = _grab(it["file"], it["src_start"] + it["dur"] * 0.5, Path(td) / f"{i}.jpg")
            if img is not None and (sc := _score(img)) > best_score:
                best, best_score = img, sc
    return best or Image.new("RGB", (1920, 1080), (20, 20, 20))


def _cover_lines(title: str | list[str], maxlen: int) -> list[str]:
    if isinstance(title, list):
        return title[:3]
    parts = [p for p in re.split(r"[，,。！!？?；;：:、\s]+", title) if p]
    lines = []
    for p in parts:
        n = -(-len(p) // maxlen)
        size = -(-len(p) // n)
        lines += [p[i:i + size] for i in range(0, len(p), size)]
    return lines[:3] or [title[:maxlen]]


def _fill_width(lines: list[str], max_w: int, max_size: int, family: str) -> ImageFont.FreeTypeFont:
    size = max_size
    while size > 40 and max(_font(size, family).getlength(t) for t in lines) > max_w:
        size -= 4
    return _font(size, family)


def make_cover(frame: Image.Image, title: str | list[str], tag: str, logo: Path | None, out: Path, vertical: bool,
               family: str = "PingFang SC") -> Path:
    W, H = (1080, 1920) if vertical else (1920, 1080)
    fw, fh = frame.size
    k = max(W / fw, H / fh)
    bg = frame.resize((int(fw * k) + 1, int(fh * k) + 1))
    bx, by = (bg.width - W) // 2, (bg.height - H) // 2
    bg = bg.crop((bx, by, bx + W, by + H))
    if vertical:
        bg = Image.blend(bg.filter(ImageFilter.GaussianBlur(28)), Image.new("RGB", (W, H), (0, 0, 0)), 0.45)
    canvas = bg.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    if vertical:
        # 主画面放大到 4:3 居中，标题放在 3:4 安全区内（主页网格会裁成 3:4）
        fg_h = 760
        fk = max(W / fw, fg_h / fh)
        fg = frame.resize((int(fw * fk) + 1, int(fh * fk) + 1))
        cx = (fg.width - W) // 2
        fg = fg.crop((cx, 0, cx + W, fg_h))
        top = 480
        canvas.paste(fg, (0, top))
        lines = _cover_lines(title, 12)
        font = _fill_width(lines, W - 100, 150, family)
        lh = int(font.size * 1.2)
        y = top + fg_h + 30 + lh // 2
        if y + lh * (len(lines) - 1) > 1640:
            y = 1640 - lh * (len(lines) - 1)
        title_xy = [(W // 2, y + i * lh) for i in range(len(lines))]
        anchor = "mm"
        tag_xy, tag_anchor = (W // 2, top - 60), "mm"
        logo_w = 380
    else:
        shade = Image.linear_gradient("L").resize((W, H)).point(lambda v: int(v * 0.85))
        canvas.alpha_composite(Image.merge("RGBA", (*Image.new("RGB", (W, H)).split(), shade)))
        lines = _cover_lines(title, 10)
        font = _fill_width(lines, W - 200, 150, family)
        lh = int(font.size * 1.18)
        y0 = H - 80 - lh * len(lines) + lh // 2
        title_xy = [(90, y0 + i * lh) for i in range(len(lines))]
        anchor = "lm"
        tag_xy, tag_anchor = (90, y0 - lh // 2 - 50), "lm"
        logo_w = 420

    colors = [YELLOW if i % 2 == 0 else (255, 255, 255) for i in range(len(lines))]
    for (x, yy), t, c in zip(title_xy, lines, colors):
        draw.text((x + 6, yy + 8), t, font=font, fill=(0, 0, 0, 160), anchor=anchor)
        draw.text((x, yy), t, font=font, fill=c, anchor=anchor, stroke_width=max(6, font.size // 14),
                  stroke_fill="black")

    if tag:
        tf = _font(54, family, "bold")
        l, t_, r, b = draw.textbbox(tag_xy, tag, font=tf, anchor=tag_anchor)
        draw.rounded_rectangle((l - 28, t_ - 16, r + 28, b + 16), radius=(b - t_) // 2 + 16, fill=RED)
        draw.text(tag_xy, tag, font=tf, fill="white", anchor=tag_anchor)

    if logo and logo.exists():
        lg = Image.open(logo).convert("RGBA")
        lg = lg.resize((logo_w, int(lg.height * logo_w / lg.width)), Image.LANCZOS)
        canvas.alpha_composite(lg, ((W - lg.width) // 2, 270) if vertical else (W - lg.width - 50, 40))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=92)
    return out
