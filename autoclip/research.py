"""换剧时自动补资料：公开百科 + 模型自身知识，并用台词核对是不是同一部剧。"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from . import prompts
from .llm import LLMClient
from .utils import log

BAIKE_API = ("https://baike.baidu.com/api/openapi/BaikeLemmaCardApi"
             "?scope=103&format=json&appid=379020&bk_length=3000&bk_key=")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def baike(name: str) -> str:
    try:
        req = urllib.request.Request(BAIKE_API + urllib.parse.quote(name), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        log.warning("[research] 百科查询失败: %s", e)
        return ""
    card = "\n".join(f"{c.get('name')}：{'、'.join(re.sub(r'<[^>]+>', '', str(v)) for v in c.get('value') or [])}"
                     for c in d.get("card") or [])
    return "\n".join(x for x in (d.get("desc", ""), d.get("abstract", ""), card) if x).strip()


def research(name: str, sample: str, llm: LLMClient) -> str:
    """返回整理好的资料文本；资料与台词对不上（重名作品等）时返回空串。"""
    web = baike(name)
    log.info("[research] %s：百科资料 %d 字", name, len(web))
    user = f"剧名：{name}\n\n【公开百科资料】\n{web or '无'}\n\n【开头几集台词节选（语音识别，有错字）】\n{sample}"
    r = llm.chat_json(prompts.RESEARCH_SYSTEM, user, temperature=0.2)
    if r.get("match") is False:
        log.warning("[research] 资料与台词对不上，弃用：%s", r.get("reason", ""))
        return ""
    return str(r.get("notes") or "").strip()
