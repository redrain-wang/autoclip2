"""阶段3：导演 —— 按故事线策划成片；阶段4：编剧 —— 写说书式解说并换算成时间轴。"""
from __future__ import annotations

import re

from . import prompts
from .config import Job, Settings, render_cfg
from .llm import LLMClient
from .series import bible_text, line_text, mmss
from .utils import log

HOOK_MAX = 12.0
ORIG_MAX = 25.0
TURN_GAP = 2.0   # 同一人两句间隔小于此视为一口气说完，不在中间切
TURN_EXTRA = 8.0 # 为说完一口气允许超出上限的秒数


def _scene_header(sc: dict) -> str:
    return (f"### {sc['id']} [{mmss(sc['t0'])}-{mmss(sc['t1'])}，L{sc['l0']}-L{sc['l1']}] {sc.get('place', '')}"
            f" 出场：{'、'.join(sc.get('people') or [])} 精彩度：{sc.get('fun', 0):.0f}\n剧情：{sc.get('summary', '')}")


def _scene_block(doc: dict, sc: dict) -> str:
    return _scene_header(sc) + "\n" + "\n".join(line_text(doc, i) for i in range(sc["l0"], sc["l1"] + 1))


def _leads(bible: dict) -> list[str]:
    # 资料里人物按重要性排序，未单独指定 leads 时取前两位
    return list(bible.get("leads") or [c.get("name") for c in (bible.get("characters") or [])[:2]])


def _lead_hook(doc: dict, scenes: list[str], leads: list[str]) -> list[int] | None:
    """在所选场戏里找主角台词最密的一段作为钩子（同数量时取精彩度高的场）。"""
    L, by_id = doc["lines"], {sc["id"]: sc for sc in doc["scenes"]}
    best, best_key = None, (0, 0.0)
    for sid in scenes:
        sc = by_id[sid]
        for a in range(sc["l0"], sc["l1"] + 1):
            if L[a].get("spk") not in leads:
                continue
            b = a
            while b + 1 <= sc["l1"] and L[b + 1]["end"] - L[a]["start"] <= HOOK_MAX:
                b += 1
            n = sum(L[i].get("spk") in leads for i in range(a, b + 1))
            if L[b]["end"] - L[a]["start"] >= 3 and (n, sc.get("fun", 0)) > best_key:
                best, best_key = [a, b], (n, sc.get("fun", 0))
    return best


def make_plan(job: Job, bible: dict, doc: dict, llm: LLMClient) -> dict:
    auto = str(job.count).lower() == "auto"
    leads = _leads(bible)
    count = job.max_count if auto else max(1, int(job.count))
    count_desc = f"0~{job.max_count} 条（由你按本集实际看点决定，精彩故事线不漏掉，平淡集可为 0 条）" if auto else f"恰好 {count} 条"
    lines = "\n".join(f"- {s.get('title', '')}：{'、'.join(s['scenes'])} ｜{s.get('summary', '')}" for s in doc["storylines"])
    user = (f"{bible_text(bible, job.episodes)}\n\n【主角】{'、'.join(map(str, leads))}\n\n【本集剧情】\n{doc['synopsis']}\n\n【故事线】\n{lines}\n\n"
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
            hook = _hook_lines(doc, v.get("hook"))
            if leads and not (hook and any(doc["lines"][i].get("spk") in leads for i in range(hook[0], hook[1] + 1))):
                hook = _lead_hook(doc, scenes, leads) or hook
            videos.append({"title": v.get("title") or job.name, "angle": v.get("angle", ""), "scenes": scenes,
                           "hook": hook})
    if not videos:
        if auto:
            log.info("[plan] 本集无足够看点，策划 0 条成片")
            return {"videos": []}
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
    while (b + 1 < len(L) and L[b + 1]["ep"] == L[b]["ep"] and L[b]["spk"] and L[b + 1]["spk"] == L[b]["spk"]
           and L[b + 1]["start"] - L[b]["end"] < TURN_GAP and L[b + 1]["end"] - L[a]["start"] <= max_len + TURN_EXTRA):
        b += 1
    st = max(0.0, L[a]["start"] - 0.1)
    en = L[b]["end"] + 0.3
    if b + 1 < len(L) and L[b + 1]["ep"] == L[b]["ep"]:
        # 识别的起始时间常偏晚，提前收尾避免带出下一句的开头
        en = min(en, max(L[b]["end"], L[b + 1]["start"] - 0.25))
    return {"type": "original", "beat_id": _scene_of(doc, a), "ep": L[a]["ep"], "src_start": round(st, 2),
            "src_end": round(en, 2), "quote": " / ".join(f"{L[i]['spk'] or '?'}：{L[i]['text']}" for i in range(a, b + 1))}


def _seg_lines(doc: dict, o: dict) -> list[int]:
    return [i for i, x in enumerate(doc["lines"])
            if x["ep"] == o["ep"] and o["src_start"] <= x["start"] < o["src_end"]]


def _clean_text(text) -> str:
    # 模型会抄回稿子里的“【解说】”或场次标签，配音会把它念出来
    return re.sub(r"^(\s*(?:【[^】]{0,6}】|\[[A-Z0-9-]{2,8}\]))+", "", str(text or "")).strip()


def _review(bible: dict, job: Job, scene_txt: str, segs: list[dict], llm: LLMClient) -> list[dict]:
    """事实核查：对照台词删改编造、张冠李戴的解说。"""
    narr = [i for i, x in enumerate(segs) if x["type"] == "narration"]
    draft = "\n".join(f"{n}. " + (f"[{x.get('beat_id', '')}] 【解说】{x['text']}" if x["type"] == "narration"
                                  else f"【原声】{x['quote']}")
                      for n, x in enumerate(segs, 1))
    user = f"{bible_text(bible, job.episodes)}\n\n【场次与台词】\n{scene_txt}\n\n【解说稿】\n{draft}"
    try:
        r = llm.chat_json(prompts.REVIEW_SYSTEM, user, temperature=0.2)
    except Exception as e:  # noqa: BLE001  核查失败不阻塞出片
        log.warning("[review] 核查失败，保留原稿: %s", e)
        return segs
    drop = set()
    for f in r.get("fixes") or []:
        try:
            i = int(f["n"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if i not in narr:
            continue
        text = _clean_text(f.get("text"))
        log.info("[review] #%d %s：%s → %s", i + 1, f.get("problem", ""), segs[i]["text"], text or "（删除）")
        if text:
            segs[i] = {**segs[i], "text": text}
        else:
            drop.add(i)
    return [x for i, x in enumerate(segs) if i not in drop]


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
    scene_txt = "\n\n".join(_scene_block(doc, by_id[x]) for x in video["scenes"])
    user = (f"{bible_text(bible, job.episodes)}\n\n【本集剧情】\n{doc['synopsis']}\n\n"
            f"【本条标题】{video['title']}\n【看点】{video['angle']}\n\n【钩子（已在开场播放）】\n{hook_txt}\n\n"
            "【选定的场戏（按时间顺序，行首为说话人）】\n\n" + scene_txt)
    log.info("[script] 生成解说稿")
    r = llm.chat_json(system, user, temperature=0.8)

    # 钩子单独标记，避免后续同场戏的画面被“防回跳”逻辑推到钩子之后
    segs = [{**x, "beat_id": "HOOK"} for x in [original_seg(doc, *hook, HOOK_MAX)] if x] if hook else []
    n = len(doc["lines"])
    used: set[int] = set()  # 正文已放过的台词行；钩子允许在正文里再放一次，故不计入
    for seg in r.get("segments") or []:
        try:
            if seg.get("type") == "original":
                a, b = sorted((int(seg["from"]), int(seg["to"])))
                while a <= b and a in used:
                    a += 1
                while b >= a and b in used:
                    b -= 1
                if a <= b and not used.intersection(range(a, b + 1)) and (o := original_seg(doc, a, b)):
                    used.update(_seg_lines(doc, o))
                    segs.append(o)
                else:
                    log.info("[script] 原声 %s~%s 与已播内容重复，跳过", seg["from"], seg["to"])
            elif seg.get("type") == "narration" and _clean_text(seg.get("text")):
                i = min(max(int(seg["at"]), 0), n - 1)
                x, sc = doc["lines"][i], by_id.get(_scene_of(doc, i)) or {}
                segs.append({"type": "narration", "beat_id": sc.get("id", ""), "ep": x["ep"],
                             "src_start": round(max(0.0, x["start"] - 0.3), 2), "text": _clean_text(seg["text"]),
                             "scene_span": [sc.get("t0", 0.0), sc.get("t1", x["end"])]})
        except (KeyError, TypeError, ValueError):
            continue
    segs = _review(bible, job, scene_txt, segs, llm)
    if not any(x["type"] == "narration" for x in segs):
        raise RuntimeError("解说稿为空")
    return {"title": r.get("title") or video["title"], "description": r.get("description", ""),
            "cover": [str(x).strip() for x in r.get("cover") or [] if str(x).strip()][:3],
            "tags": r.get("tags", []), "angle": video["angle"], "segments": segs}
