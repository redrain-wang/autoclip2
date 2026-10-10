"""剧集资料（series/<剧名>.yaml，可人工编辑）与单集梳理（场次、说话人、故事线）。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import prompts
from .ingest import apply_fixes
from .llm import LLMClient
from .utils import cached, log, slug

SERIES_DIR = Path("series")
LONG_TIMEOUT = 1200.0  # 整剧/整集长上下文调用


def mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def bible_path(name: str) -> Path:
    return SERIES_DIR / f"{slug(name)}.yaml"


def _compact(ing: dict) -> str:
    out, minute = [], -1
    for x in ing["subs"]:
        m = int(x["start"] // 60)
        if m != minute:
            out.append(f"\n[{m}分]")
            minute = m
        out.append(x["text"])
    return " ".join(out)


def load_bible(name: str, ingests: list[dict], llm: LLMClient, force: bool = False) -> dict:
    """已有 characters 的资料文件直接使用（人工修改优先）；否则用全剧台词 + notes 生成。"""
    path = bible_path(name)
    data = (yaml.safe_load(path.read_text("utf-8")) if path.exists() else None) or {}
    if data.get("characters") and not force:
        return data
    notes = data.get("notes", "")
    log.info("[bible] 用 %d 集台词生成剧集资料 → %s", len(ingests), path)
    body = "\n\n".join(f"## 第{i}集\n{_compact(ing)}" for i, ing in enumerate(ingests, 1))
    r = llm.chat_json(prompts.BIBLE_SYSTEM, f"作品：{name}\n\n已知资料：\n{notes or '无'}\n\n{body}",
                      temperature=0.2, timeout=LONG_TIMEOUT)
    data = {"notes": notes, **{k: r.get(k) for k in ("intro", "synopsis", "characters", "episodes", "asr_fixes")}}
    if path.exists():
        path.replace(path.with_suffix(".yaml.bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200), "utf-8")
    return data


def name_fixes(bible: dict) -> dict[str, str]:
    """只保留指向人名/剧中地名的长词校正，避免误伤正常台词。"""
    names = [c["name"] for c in bible.get("characters") or [] if len(c.get("name", "")) >= 2]
    known = f"{bible.get('intro', '')}{bible.get('synopsis', '')}"
    return {k: v for k, v in (bible.get("asr_fixes") or {}).items()
            if isinstance(k, str) and isinstance(v, str) and len(k) >= 3 and k != v
            and (any(n in v for n in names) or v in known)}


def apply_name_fixes(ingests: list[dict], fixes: dict[str, str]) -> None:
    if not fixes:
        return
    log.info("[fix] 校正识别错字: %s", fixes)
    for ing in ingests:
        for x in ing["subs"]:
            x["text"] = apply_fixes(x["text"], fixes)


def bible_text(bible: dict, eps: list[int]) -> str:
    chars = "\n".join(
        f"- {c.get('name')}：{c.get('identity', '')}；关系：{c.get('relations', '')}；"
        f"别称：{'、'.join(c.get('aliases') or []) or '无'}；{c.get('traits', '')}"
        for c in bible.get("characters") or [])
    near = {e + d for e in eps for d in (-1, 0, 1)}
    syn = "\n".join(f"第{e.get('ep')}集：{e.get('synopsis', '')}"
                    for e in bible.get("episodes") or [] if e.get("ep") in near)
    return f"【作品介绍】{bible.get('intro', '')}\n\n【人物】\n{chars}\n\n【前后剧集梗概】\n{syn}"


def analyze_episode(ing: dict, beats: list[dict], bible: dict, ep: int, llm: LLMClient, path: Path) -> dict:
    subs = ing["subs"]

    def scenes() -> dict:
        vis = "\n".join(f"[{mmss(b['start'])}-{mmss(b['end'])}] {b['summary']}" for b in beats if not b.get("is_filler"))
        lines = "\n".join(f"L{i} [{mmss(x['start'])}] {x['text']}" for i, x in enumerate(subs))
        user = f"{bible_text(bible, [ep])}\n\n当前：第{ep}集\n\n【画面摘要】\n{vis}\n\n【台词】\n{lines}"
        log.info("[episode] 第 %d 集：梳理场次（%d 行台词）", ep, len(subs))
        return _clean_episode(llm.chat_json(prompts.EPISODE_SYSTEM, user, temperature=0.2, timeout=LONG_TIMEOUT), ing)

    def work() -> dict:
        data = cached(path.with_name(path.stem + "_scenes.json"), scenes)
        log.info("[episode] 第 %d 集：标注说话人（%d 场）", ep, len(data["scenes"]))
        with ThreadPoolExecutor(llm.cfg.max_concurrency) as ex:
            parts = list(ex.map(lambda sc: _speakers(sc, subs, bible, llm), data["scenes"]))
        return {**data, "speakers": [x for p in parts for x in p]}
    return cached(path, work)


def _speakers(sc: dict, subs: list[dict], bible: dict, llm: LLMClient) -> list[str]:
    n = sc["l1"] - sc["l0"] + 1
    if n <= 0:
        return []
    chars = "\n".join(f"- {c.get('name')}：{c.get('identity', '')}；{c.get('relations', '')}" for c in bible.get("characters") or [])
    lines = "\n".join(f"{k + 1}. {subs[i]['text']}" for k, i in enumerate(range(sc["l0"], sc["l1"] + 1)))
    user = (f"【人物表】\n{chars}\n\n【本场】{sc.get('place', '')}，在场：{'、'.join(sc.get('people') or [])}\n"
            f"剧情：{sc.get('summary', '')}\n\n【台词（共 {n} 行）】\n{lines}")
    try:
        out = [str(x) for x in llm.chat_json(prompts.SPEAKER_SYSTEM, user, temperature=0.1).get("speakers") or []]
    except Exception as e:  # noqa: BLE001  单场失败只丢说话人，不阻塞
        log.warning("[episode] %s 说话人标注失败: %s", sc["id"], e)
        out = []
    return ["" if x == "?" else x for x in (out + [""] * n)[:n]]


def _clean_episode(r: dict, ing: dict) -> dict:
    subs, n = ing["subs"], len(ing["subs"])
    raw = []
    for sc in r.get("scenes") or []:
        try:
            raw.append({**sc, "start": min(max(int(sc["start"]), 0), n - 1)})
        except (KeyError, TypeError, ValueError):
            continue
    raw.sort(key=lambda x: x["start"])
    scenes, seen = [], set()
    for sc in raw:
        if sc["start"] in seen:
            continue
        seen.add(sc["start"])
        scenes.append(sc)
    if not scenes:
        raise RuntimeError("单集梳理没有返回有效场次")
    scenes[0]["start"] = 0
    # 模型返回的场次可能乱序/重复，重新编号后把故事线引用映射过来
    remap = {}
    for i, sc in enumerate(scenes):
        nxt = scenes[i + 1]["start"] if i + 1 < len(scenes) else n
        l0, l1 = sc["start"], nxt - 1
        t0 = 0.0 if i == 0 else max(subs[l0 - 1]["end"], subs[l0]["start"] - 3.0)
        remap[str(sc.get("id"))] = f"S{i + 1:02d}"
        sc.update(id=f"S{i + 1:02d}", l0=l0, l1=l1, t0=round(t0, 2), fun=_num(sc.get("fun")))
        if i:
            scenes[i - 1]["t1"] = sc["t0"]
    scenes[-1]["t1"] = ing["duration"]
    return {"synopsis": r.get("synopsis", ""), "scenes": scenes,
            "storylines": [{**s, "scenes": [remap.get(x, x) for x in s.get("scenes") or []]}
                           for s in r.get("storylines") or []]}


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def assemble(ingests: list[dict], episodes: list[dict], ep_nos: list[int]) -> dict:
    """把一个任务的多集合并成全局行号/场次编号。"""
    multi = len(ingests) > 1
    lines, scenes, synopsis, storylines = [], [], [], []
    for k, (ing, epi, no) in enumerate(zip(ingests, episodes, ep_nos)):
        base = len(lines)
        pre = f"E{no:02d}-" if multi else ""
        for x, sp in zip(ing["subs"], epi["speakers"]):
            lines.append({"ep": k, "start": x["start"], "end": x["end"], "text": x["text"], "spk": sp})
        for sc in epi["scenes"]:
            scenes.append({**sc, "id": pre + sc["id"], "ep": k, "l0": sc["l0"] + base, "l1": sc["l1"] + base})
        synopsis.append(f"第{no}集：{epi['synopsis']}")
        storylines += [{**s, "scenes": [pre + x for x in s["scenes"]]} for s in epi["storylines"]]
    return {"lines": lines, "scenes": scenes, "synopsis": "\n".join(synopsis), "storylines": storylines}


def line_text(doc: dict, i: int) -> str:
    x = doc["lines"][i]
    return f"L{i} [{mmss(x['start'])}] {x['spk'] or '?'}：{x['text']}"
