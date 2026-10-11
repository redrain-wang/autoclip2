"""剧集资料（series/<剧名>.yaml，可人工编辑）与单集梳理（场次、说话人、故事线）。"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from . import prompts, research
from .ingest import apply_fixes
from .llm import LLMClient
from .utils import cached, log, short_hash, slug

SERIES_DIR = Path("series")
LONG_TIMEOUT = 1200.0  # 整剧/整集长上下文调用
BIBLE_CHARS = 150_000  # 单次生成剧集资料的台词上限（约 15 集），超出则分段再合并
GENERATED = ("intro", "synopsis", "characters", "episodes", "asr_fixes")


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


def _fingerprint(ingests: list[dict]) -> str:
    return short_hash(sorted(Path(i["file"]).name for i in ingests))


def _same_show(data: dict, ingests: list[dict]) -> bool:
    """素材变了时判断是不是同一部剧：已有主要人物有一半以上出现在新台词里。"""
    text = "".join(x["text"] for ing in ingests for x in ing["subs"])
    names = [c.get("name", "") for c in (data.get("characters") or [])[:10] if len(c.get("name", "")) >= 2]
    return bool(names) and sum(n in text for n in names) * 2 >= len(names)


def _save(path: Path, data: dict) -> None:
    if path.exists():
        path.replace(path.with_suffix(".yaml.bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200), "utf-8")


def load_bible(name: str, ingests: list[dict], llm: LLMClient, force: bool = False) -> dict:
    """剧集资料：已有且素材未变则直接用（人工修改优先）；新剧/换剧时自动检索资料 → 生成 → 用台词审校。"""
    path = bible_path(name)
    data = (yaml.safe_load(path.read_text("utf-8")) if path.exists() else None) or {}
    fp = _fingerprint(ingests)
    if data.get("characters") and not force:
        if data.get("source") == fp:
            return data
        if data.get("source") is None or _same_show(data, ingests):
            if data.get("source") is not None:
                log.info("[bible] 素材有变化但仍是同一部剧，沿用剧集资料")
            data["source"] = fp
            _save(path, data)
            return data
        log.warning("[bible] 检测到换剧：%s 里的人物在新素材中基本没有出现，重新建立资料（旧文件备份为 .yaml.bak）", path)
        data = {}
    elif not data.get("characters"):
        log.info("[bible] 新剧《%s》，自动建立剧集资料", name)

    notes = data.get("notes", "")
    sample = "\n\n".join(_compact(ing) for ing in ingests[:2])[:20000]
    found = research.research(name, sample, llm)
    log.info("[bible] 用 %d 集台词生成剧集资料 → %s", len(ingests), path)
    parts = [f"## 第{i}集\n{_compact(ing)}" for i, ing in enumerate(ingests, 1)]
    chunks: list[list[str]] = [[]]
    for p in parts:
        if chunks[-1] and sum(map(len, chunks[-1])) + len(p) > BIBLE_CHARS:
            chunks.append([])
        chunks[-1].append(p)
    head = (f"作品：{name}\n\n已知资料（人工确认，可信度最高）：\n{notes or '无'}\n\n"
            f"公开资料（自动检索，可信度次之，与台词冲突时以台词为准）：\n{found or '无'}\n\n")
    if len(chunks) == 1:
        r = llm.chat_json(prompts.BIBLE_SYSTEM, head + "\n\n".join(parts), temperature=0.2, timeout=LONG_TIMEOUT)
    else:
        log.info("[bible] 台词较长，分 %d 段生成后合并", len(chunks))
        with ThreadPoolExecutor(len(chunks)) as ex:
            subs = list(ex.map(lambda c: llm.chat_json(prompts.BIBLE_SYSTEM, head + "\n\n".join(c),
                                                       temperature=0.2, timeout=LONG_TIMEOUT), chunks))
        brief = [{k: s.get(k) for k in ("intro", "synopsis", "characters", "asr_fixes")} for s in subs]
        r = llm.chat_json(prompts.BIBLE_MERGE_SYSTEM, head + json.dumps(brief, ensure_ascii=False),
                          temperature=0.2, timeout=LONG_TIMEOUT)
        r["episodes"] = [e for s in subs for e in s.get("episodes") or []]
    data = {"notes": notes, "research": found,
            **{k: v for k, v in data.items() if k not in GENERATED and k not in ("notes", "research", "source")},
            **{k: r.get(k) for k in GENERATED}, "source": fp}
    _audit(data, ingests, llm)
    _save(path, data)
    return data


def _audit(data: dict, ingests: list[dict], llm: LLMClient) -> None:
    """用全剧台词里检索到的证据核对人物关系和错字表，原地修正。"""
    lines = [x["text"] for ing in ingests for x in ing["subs"]]

    def evidence(words: list[str], k: int) -> list[str]:
        hits = [i for i, t in enumerate(lines) if any(w in t for w in words)]
        step = max(1, len(hits) // k)
        return [" / ".join(lines[max(0, i - 1):i + 2]) for i in hits[::step][:k]]

    fixes = data.get("asr_fixes") or {}
    chars = data.get("characters") or []
    ev = []
    for c in chars:
        words = [c.get("name", "")] + [a for a in c.get("aliases") or [] if a not in fixes]
        found = evidence([w for w in words if len(w) >= 2], 6)
        ev.append(f"### {c.get('name')}（出现 {len(found)} 处样本）\n" + ("\n".join(found) or "（台词中未出现）"))
    for k, v in fixes.items():
        found = evidence([k], 2)
        ev.append(f"### 错字校正 {k}→{v}\n" + ("\n".join(found) or "（台词中未出现）"))
    user = (f"【已知资料】\n{data.get('notes') or '无'}\n{data.get('research') or ''}\n\n"
            f"【人物表】\n{json.dumps(chars, ensure_ascii=False)}\n\n【错字校正】\n{json.dumps(fixes, ensure_ascii=False)}\n\n"
            "【台词证据】\n" + "\n\n".join(ev))
    try:
        r = llm.chat_json(prompts.BIBLE_AUDIT_SYSTEM, user, temperature=0.1, timeout=LONG_TIMEOUT)
    except Exception as e:  # noqa: BLE001  审校失败不阻塞，保留生成结果
        log.warning("[bible] 审校失败，保留原结果: %s", e)
        return
    new = [c for c in r.get("characters") or [] if isinstance(c, dict) and c.get("name")]
    if len(new) * 2 >= len(chars):
        data["characters"] = new
    leads = {c.get("name") for c in chars[:5]}
    for k in r.get("drop_fixes") or []:
        # 主角名字的错写出现最多、最该纠正，不允许审校删掉
        if fixes.get(k) not in leads:
            fixes.pop(k, None)
    for x in r.get("changes") or []:
        log.info("[bible] 审校：%s", x)


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
    # 识别错字不是真别称，给模型看会把“牛头”之类普通词当成人名
    typos = set(bible.get("asr_fixes") or {})
    chars = "\n".join(
        f"- {c.get('name')}：{c.get('identity', '')}；关系：{c.get('relations', '')}；"
        f"别称：{'、'.join(a for a in c.get('aliases') or [] if a not in typos) or '无'}；{c.get('traits', '')}"
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
            parts = list(ex.map(lambda k: _speakers(data["scenes"], k, subs, beats, bible, llm),
                                range(len(data["scenes"]))))
        return {**data, "speakers": [x for p, _ in parts for x in p],
                "fixes": {k: v for _, f in parts for k, v in f.items()}}
    return cached(path, work)


def _speakers(scenes: list[dict], k: int, subs: list[dict], beats: list[dict], bible: dict,
              llm: LLMClient) -> tuple[list[str], dict[str, str]]:
    sc = scenes[k]
    n = sc["l1"] - sc["l0"] + 1
    if n <= 0:
        return [], {}
    chars = "\n".join(f"- {c.get('name')}：{c.get('identity', '')}；{c.get('relations', '')}" for c in bible.get("characters") or [])
    lines = "\n".join(f"{j + 1}. {subs[i]['text']}" for j, i in enumerate(range(sc["l0"], sc["l1"] + 1)))
    vis = "\n".join(f"[{mmss(b['start'])}] {b['summary']}" for b in beats
                    if b["end"] > sc["t0"] and b["start"] < sc["t1"] and b.get("summary")) or "无"
    prev = scenes[k - 1] if k else None
    prev_txt = f"{'、'.join(prev.get('people') or [])}：{prev.get('summary', '')}" if prev else "无"
    user = (f"【人物表】\n{chars}\n\n【上一场】{prev_txt}\n\n【本场】{sc.get('place', '')}，初步判断在场："
            f"{'、'.join(sc.get('people') or [])}\n剧情：{sc.get('summary', '')}\n画面：\n{vis}\n\n"
            f"【台词（共 {n} 行）】\n{lines}")
    try:
        r = llm.chat_json(prompts.SPEAKER_SYSTEM, user, temperature=0.1)
    except Exception as e:  # noqa: BLE001  单场失败只丢说话人，不阻塞
        log.warning("[episode] %s 说话人标注失败: %s", sc["id"], e)
        r = {}
    out = [str(x) for x in r.get("speakers") or []]
    raw = r.get("fixes") if isinstance(r.get("fixes"), dict) else {}
    fixes = {}
    for k, v in raw.items():
        try:
            i = sc["l0"] + int(str(k).lstrip("L")) - 1
        except ValueError:
            continue
        old, new = subs[i]["text"] if sc["l0"] <= i <= sc["l1"] else "", str(v).strip()
        # 只接受小幅修正，防止模型改写台词
        if old and new and new != old and SequenceMatcher(None, old, new).ratio() >= 0.6:
            fixes[str(i)] = new
    return ["" if x == "?" else x for x in (out + [""] * n)[:n]], fixes


def apply_text_fixes(ing: dict, epi: dict) -> None:
    for k, v in (epi.get("fixes") or {}).items():
        ing["subs"][int(k)]["text"] = v


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
