"""阶段3：导演 —— 按故事线策划成片；阶段4：编剧 —— 写说书式解说并换算成时间轴。"""
from __future__ import annotations

from . import prompts
from .config import Job, Settings, render_cfg
from .llm import LLMClient
from .series import bible_text, line_text, mmss
from .utils import log

HOOK_MAX = 12.0
ORIG_MAX = 18.0


def _scene_header(sc: dict) -> str:
    return (f"### {sc['id']} [{mmss(sc['t0'])}-{mmss(sc['t1'])}，L{sc['l0']}-L{sc['l1']}] {sc.get('place', '')}"
            f" 出场：{'、'.join(sc.get('people') or [])} 精彩度：{sc.get('fun', 0):.0f}\n剧情：{sc.get('summary', '')}")


def _scene_block(doc: dict, sc: dict) -> str:
    return _scene_header(sc) + "\n" + "\n".join(line_text(doc, i) for i in range(sc["l0"], sc["l1"] + 1))


def make_plan(job: Job, bible: dict, doc: dict, llm: LLMClient) -> dict:
    auto = str(job.count).lower() == "auto"
    count = job.max_count if auto else max(1, int(job.count))
    count_desc = f"若干条（由你判断素材值得做几条，最多 {job.max_count} 条）" if auto else f"恰好 {count} 条"
    lines = "\n".join(f"- {s.get('title', '')}：{'、'.join(s['scenes'])} ｜{s.get('summary', '')}" for s in doc["storylines"])
    user = (f"{bible_text(bible, job.episodes)}\n\n【本集剧情】\n{doc['synopsis']}\n\n【故事线】\n{lines}\n\n"
            f"【额外要求】{job.style or '无'}\n\n【场次与台词（行首为说话人）】\n"
            + "\n\n".join(_scene_block(doc, sc) for sc in doc["scenes"]))
    system = prompts.PLAN_SYSTEM.format(duration=job.target_duration, count_desc=count_desc)
    log.info("[plan] %d 场戏、%d 条故事线，数量=%s", len(doc["scenes"]), len(doc["storylines"]), job.count)
    r = llm.chat_json(system, user, temperature=0.6)

    order = {sc["id"]: i for i, sc in enumerate(doc["scenes"])}
    videos = []
    for v in (r.get("videos") or [])[:count]:
        scenes = sorted({x for x in v.get("scenes") or [] if x in order}, key=order.get)
        if scenes:
            videos.append({"title": v.get("title") or job.name, "angle": v.get("angle", ""), "scenes": scenes,
                           "hook": _hook_lines(doc, v.get("hook"))})
    if not videos:
        raise RuntimeError("策划没有返回有效的场次")
    log.info("[plan] 策划 %d 条成片: %s", len(videos), [v["title"] for v in videos])
    return {"videos": videos}


def _hook_lines(doc: dict, hook) -> list[int] | None:
    try:
        a, b = (int(str(x).lstrip("L")) for x in hook[:2])
    except (TypeError, ValueError, IndexError):
        return None
    a, b = min(a, b), max(a, b)
    if not (0 <= a and b < len(doc["lines"])) or doc["lines"][a]["ep"] != doc["lines"][b]["ep"]:
        return None
    while b > a and doc["lines"][b]["end"] - doc["lines"][a]["start"] > HOOK_MAX:
        b -= 1
    return [a, b]


def _scene_of(doc: dict, i: int) -> str:
    return next((sc["id"] for sc in doc["scenes"] if sc["l0"] <= i <= sc["l1"]), "")


def original_seg(doc: dict, a: int, b: int, max_len: float = ORIG_MAX) -> dict | None:
    """台词行 a~b → 原声片段（超长时从尾部按整句裁掉）。"""
    L = doc["lines"]
    if not (0 <= a <= b < len(L)) or L[a]["ep"] != L[b]["ep"]:
        return None
    while b > a and L[b]["end"] - L[a]["start"] > max_len:
        b -= 1
    st = max(0.0, L[a]["start"] - 0.1)
    en = L[b]["end"] + 0.3
    if b + 1 < len(L) and L[b + 1]["ep"] == L[b]["ep"]:
        en = min(en, max(L[b]["end"], L[b + 1]["start"]))
    return {"type": "original", "beat_id": _scene_of(doc, a), "ep": L[a]["ep"], "src_start": round(st, 2),
            "src_end": round(en, 2), "quote": " / ".join(f"{L[i]['spk'] or '?'}：{L[i]['text']}" for i in range(a, b + 1))}


def write_script(job: Job, bible: dict, doc: dict, video: dict, llm: LLMClient, s: Settings) -> dict:
    r_cfg = render_cfg(s, job)
    narr_chars = int(job.target_duration * r_cfg.narration_ratio * r_cfg.narration_speed)
    intro = ("这是全剧第一集，可以先用一两句话介绍这部剧（年代、题材、主角），再进入故事" if 1 in job.episodes
             else "不需要介绍这部剧，必要时用一句话交代前情")
    system = prompts.SCRIPT_SYSTEM.format(extra=job.style or "无", intro_hint=intro, example=prompts.STYLE_EXAMPLE,
                                          orig_pct=round((1 - r_cfg.narration_ratio) * 100), narr_chars=narr_chars)
    hook = video.get("hook")
    hook_txt = "\n".join(line_text(doc, i) for i in range(hook[0], hook[1] + 1)) if hook else "（无，直接从解说开始）"
    by_id = {sc["id"]: sc for sc in doc["scenes"]}
    user = (f"{bible_text(bible, job.episodes)}\n\n【本集剧情】\n{doc['synopsis']}\n\n"
            f"【本条标题】{video['title']}\n【看点】{video['angle']}\n\n【钩子（已在开场播放）】\n{hook_txt}\n\n"
            "【选定的场戏（按时间顺序，行首为说话人）】\n\n"
            + "\n\n".join(_scene_block(doc, by_id[x]) for x in video["scenes"]))
    log.info("[script] 生成解说稿")
    r = llm.chat_json(system, user, temperature=0.8)

    # 钩子单独标记，避免后续同场戏的画面被“防回跳”逻辑推到钩子之后
    segs = [{**x, "beat_id": "HOOK"} for x in [original_seg(doc, *hook, HOOK_MAX)] if x] if hook else []
    n = len(doc["lines"])
    for seg in r.get("segments") or []:
        try:
            if seg.get("type") == "original":
                a, b = int(seg["from"]), int(seg["to"])
                if (o := original_seg(doc, min(a, b), max(a, b))):
                    segs.append(o)
            elif seg.get("type") == "narration" and str(seg.get("text", "")).strip():
                i = min(max(int(seg["at"]), 0), n - 1)
                x = doc["lines"][i]
                segs.append({"type": "narration", "beat_id": _scene_of(doc, i), "ep": x["ep"],
                             "src_start": round(max(0.0, x["start"] - 0.3), 2), "text": str(seg["text"]).strip()})
        except (KeyError, TypeError, ValueError):
            continue
    if not any(x["type"] == "narration" for x in segs):
        raise RuntimeError("解说稿为空")
    return {"title": r.get("title") or video["title"], "description": r.get("description", ""),
            "cover": [str(x).strip() for x in r.get("cover") or [] if str(x).strip()][:3],
            "tags": r.get("tags", []), "angle": video["angle"], "segments": segs}
