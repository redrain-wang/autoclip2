"""本地声纹提取与说话人校验（CAM++ 16kHz 中文声纹模型，纯 PyTorch 实现）。"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import math
import re
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .utils import log, read_json, short_hash, slug, write_json

_lock = threading.Lock()
_model: CAMPPlus | None = None
VP_VER = "campp-v2"


def _get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    seq = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            seq.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            seq.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            seq.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            seq.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
    return seq


class _StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)


class _TDNNLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        if padding < 0:
            padding = (k - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_c, out_c, k, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = _get_nonlinear(config_str, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class _CAMLayer(nn.Module):
    def __init__(self, bn_c: int, out_c: int, k: int, stride: int, padding: int,
                 dilation: int, bias: bool, reduction: int = 2):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_c, out_c, k, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_c, bn_c // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_c // reduction, out_c, 1)
        self.sigmoid = nn.Sigmoid()

    def _seg_pool(self, x: torch.Tensor, seg_len: int = 100) -> torch.Tensor:
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        sh = seg.shape
        return seg.unsqueeze(-1).expand(*sh, seg_len).reshape(*sh[:-1], -1)[..., :x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        ctx = x.mean(-1, keepdim=True) + self._seg_pool(x)
        return y * self.sigmoid(self.linear2(self.relu(self.linear1(ctx))))


class _CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, bn_c: int, k: int, stride: int = 1,
                 dilation: int = 1, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        padding = (k - 1) // 2 * dilation
        self.nonlinear1 = _get_nonlinear(config_str, in_c)
        self.linear1 = nn.Conv1d(in_c, bn_c, 1, bias=False)
        self.nonlinear2 = _get_nonlinear(config_str, bn_c)
        self.cam_layer = _CAMLayer(bn_c, out_c, k, stride, padding, dilation, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(x))))


class _CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers: int, in_c: int, out_c: int, bn_c: int, k: int,
                 stride: int = 1, dilation: int = 1, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"tdnnd{i + 1}", _CAMDenseTDNNLayer(in_c + i * out_c, out_c, bn_c, k,
                                                                 stride, dilation, bias, config_str))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class _TransitLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, bias: bool = True, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.nonlinear = _get_nonlinear(config_str, in_c)
        self.linear = nn.Conv1d(in_c, out_c, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(x))


class _DenseLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_c, out_c, 1, bias=bias)
        self.nonlinear = _get_nonlinear(config_str, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x.unsqueeze(-1)).squeeze(-1) if x.ndim == 2 else self.linear(x)
        return self.nonlinear(x)


class _BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_p: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_p, planes, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_p != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_p, planes, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class _FCM(nn.Module):
    def __init__(self, m_channels: int = 32, feat_dim: int = 80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, 2, stride=2)
        self.layer2 = self._make_layer(m_channels, 2, stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(_BasicResBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        out = F.relu(self.bn2(self.conv2(self.layer2(self.layer1(out)))))
        return out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim: int = 80, embedding_size: int = 192, growth_rate: int = 32,
                 bn_size: int = 4, init_channels: int = 128, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.head = _FCM(feat_dim=feat_dim)
        c = self.head.out_channels
        self.xvector = nn.Sequential(OrderedDict([
            ("tdnn", _TDNNLayer(c, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str)),
        ]))
        c = init_channels
        for i, (n, k, d) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            self.xvector.add_module(f"block{i + 1}", _CAMDenseTDNNBlock(n, c, growth_rate, bn_size * growth_rate,
                                                                         k, dilation=d, config_str=config_str))
            c += n * growth_rate
            self.xvector.add_module(f"transit{i + 1}", _TransitLayer(c, c // 2, bias=False, config_str=config_str))
            c //= 2
        self.xvector.add_module("out_nonlinear", _get_nonlinear(config_str, c))
        self.xvector.add_module("stats", _StatsPool())
        self.xvector.add_module("dense", _DenseLayer(c * 2, embedding_size, config_str="batchnorm_"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xvector(self.head(x.permute(0, 2, 1)))


def _get_model() -> CAMPPlus:
    global _model
    with _lock:
        if _model is None:
            from huggingface_hub import hf_hub_download
            try:
                w = hf_hub_download("funasr/campplus", "campplus_cn_common.bin", local_files_only=True)
            except Exception:  # noqa: BLE001
                w = hf_hub_download("funasr/campplus", "campplus_cn_common.bin")
            m = CAMPPlus()
            m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
            m.eval()
            _model = m
        return _model


def kaldi_fbank(wav: torch.Tensor, sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    wav = wav * 32768.0
    frame_len, frame_shift, n_fft = 400, 160, 512
    if wav.numel() < frame_len:
        wav = F.pad(wav, (0, frame_len - wav.numel()))
    frames = wav.unfold(0, frame_len, frame_shift).clone()
    frames -= frames.mean(dim=-1, keepdim=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1]
    frames[:, 0] *= 0.03
    win = (0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(frame_len, device=wav.device) / (frame_len - 1))).pow(0.85)
    spec = torch.fft.rfft(frames * win, n=n_fft).abs().pow(2)
    low_mel = 1127.0 * math.log(1.0 + 20.0 / 700.0)
    high_mel = 1127.0 * math.log(1.0 + (sr / 2.0) / 700.0)
    mels = torch.linspace(low_mel, high_mel, n_mels + 2, device=wav.device)
    freqs = torch.arange(n_fft // 2 + 1, device=wav.device) * (sr / n_fft)
    fft_mels = 1127.0 * torch.log(1.0 + freqs / 700.0)
    up = (fft_mels.unsqueeze(0) - mels[:-2].unsqueeze(1)) / (mels[1:-1] - mels[:-2]).unsqueeze(1)
    down = (mels[2:].unsqueeze(1) - fft_mels.unsqueeze(0)) / (mels[2:] - mels[1:-1]).unsqueeze(1)
    fb = torch.clamp(torch.minimum(up, down), min=0.0)
    feat = torch.log(torch.clamp(spec @ fb.T, min=1e-10))
    return feat - feat.mean(dim=0, keepdim=True)


def _embed_pcm(pcm: np.ndarray) -> np.ndarray | None:
    if len(pcm) < 6400:
        return None
    feat = kaldi_fbank(torch.from_numpy(pcm)).unsqueeze(0)
    with torch.no_grad():
        v = _get_model()(feat).squeeze(0).numpy()
    nrm = float(np.linalg.norm(v))
    return (v / nrm).astype(np.float32) if nrm > 0 else None


def _load_pcm(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path)) as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def line_embeddings(mdir: Path, subs: list[dict], timing: str | None = None) -> np.ndarray:
    """提取或读取单集每行台词的 192 维声纹向量。"""
    npy_path, meta_path = mdir / "voiceprints.npy", mdir / "voiceprints_meta.json"
    sig = {"n": len(subs), "timing": timing, "ver": VP_VER}
    if npy_path.exists() and meta_path.exists() and read_json(meta_path) == sig:
        arr = np.load(npy_path)
        if arr.shape == (len(subs), 192):
            return arr
    wav_path = mdir / "audio.wav"
    embs = np.zeros((len(subs), 192), dtype=np.float32)
    if not wav_path.exists():
        return embs
    log.info("[voiceprint] 提取台词声纹 (%d 行) → %s", len(subs), mdir.name)
    pcm, sr = _load_pcm(wav_path)
    for i, s in enumerate(subs):
        dur = s["end"] - s["start"]
        if dur < 0.45:
            continue
        a = int(max(0.0, s["start"] - 0.05) * sr)
        b = int(min(len(pcm) / sr, s["end"] + 0.05) * sr)
        if (v := _embed_pcm(pcm[a:b])) is not None:
            embs[i] = v
    np.save(npy_path, embs)
    write_json(meta_path, sig)
    return embs


def _alias_map(bible: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    amap: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    for c in bible.get("characters") or []:
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        amap[name] = name
        alist = [name]
        for a in c.get("aliases") or []:
            a = str(a).strip()
            if a:
                amap[a] = name
                alist.append(a)
        aliases[name] = sorted(set(alist), key=len, reverse=True)
    return amap, aliases


def _scene_spk_emb(idxs: list[int], subs: list[dict], durs: np.ndarray,
                   embs: np.ndarray, pcm: np.ndarray, sr: int) -> tuple[np.ndarray | None, float]:
    valid = [i for i in idxs if durs[i] >= 0.5 and float(np.linalg.norm(embs[i])) > 0.5]
    if not valid:
        return None, 0.0
    if len(valid) >= 3:
        mat = embs[valid]
        sim = mat @ mat.T
        w = np.minimum(durs[valid], 4.0)
        med = int(np.argmax(sim @ w))
        keep = [valid[k] for k in range(len(valid)) if sim[med, k] >= 0.25] or valid
    else:
        keep = valid
    chunks, tot = [], 0.0
    for i in keep:
        s = subs[i]
        a = int(max(0.0, s["start"]) * sr)
        b = int(min(len(pcm) / sr, s["end"]) * sr)
        if b - a >= int(0.4 * sr):
            chunks.append(pcm[a:b])
            tot += (b - a) / sr
            if tot >= 18.0:
                break
    if tot < 1.5:
        return None, tot
    return _embed_pcm(np.concatenate(chunks)), tot


def _swap_text(text: str, swaps: list[tuple[str, str]], aliases: dict[str, list[str]]) -> str:
    if not text or not swaps:
        return text
    placeholders: list[tuple[str, str]] = []
    for idx, (old_c, new_c) in enumerate(swaps):
        ph = f"__SPK_SWAP_{idx}__"
        placeholders.append((ph, new_c))
        for tok in aliases.get(old_c, [old_c]):
            if len(tok) >= 2:
                text = text.replace(tok, ph)
    for ph, new_c in placeholders:
        text = text.replace(ph, new_c)
    return text


def _is_vocative(cname: str, other_lines: list[str], aliases: dict[str, list[str]]) -> bool:
    """判断本场其他人的台词里是否当面称呼了 cname（排除“让老书记…”“把郝运来…”等第三人称提及）。"""
    toks = [re.escape(t) for t in aliases.get(cname, [cname]) if len(t) >= 2]
    if not toks:
        return False
    group = "|".join(toks)
    pat = re.compile(r"(?:^|[，。！？\s]|不过|那个|哎|我说)(" + group + r")")
    third = re.compile(r"[让找替给向把被对叫请由跟和与](" + group + r")|(" + group + r")(?:同志|德高)")
    txt = "\n".join(other_lines)
    return bool(pat.search(txt)) and not bool(third.search(txt))


def _split_if_mixed(idxs: list[int], subs: list[dict], durs: np.ndarray,
                    embs: np.ndarray, pcm: np.ndarray, sr: int) -> list[list[int]]:
    """若同一标签在场内相隔 >=12 行的两个子段声纹明显不同（如 S05 首尾两处误标），拆开分别核验。"""
    groups: list[list[int]] = []
    for i in idxs:
        if groups and i - groups[-1][-1] >= 12:
            groups.append([i])
        else:
            if not groups:
                groups.append([])
            groups[-1].append(i)
    if len(groups) <= 1:
        return [idxs]
    merged: list[list[int]] = []
    for g in groups:
        if merged and (sum(durs[i] for i in g) < 2.5 or sum(durs[i] for i in merged[-1]) < 2.5):
            merged[-1].extend(g)
        else:
            merged.append(g)
    if len(merged) <= 1:
        return [idxs]
    vecs = [_scene_spk_emb(g, subs, durs, embs, pcm, sr)[0] for g in merged]
    if any(v is None for v in vecs):
        return [idxs]
    min_sim = min(float(vecs[a] @ vecs[b]) for a in range(len(vecs)) for b in range(a + 1, len(vecs)))
    return merged if min_sim < 0.42 else [idxs]


def _adj_turns(idxs1: list[int], idxs2: list[int]) -> int:
    s2 = set(idxs2)
    return sum((i - 1 in s2) or (i + 1 in s2) for i in idxs1)


def _load_anchor_vecs(bible: dict, media_root: Path) -> dict[str, np.ndarray]:
    """根据 series/<剧名>.yaml 中的 voice_anchors 提取角色标准声纹，并导出样本音频与截图供人工核查。"""
    anchors = bible.get("voice_anchors")
    if not isinstance(anchors, dict) or not anchors:
        return {}
    sig = short_hash({"ver": VP_VER, "anchors": anchors})
    bdir = media_root.parent / "voice_bank"
    bdir.mkdir(parents=True, exist_ok=True)
    cache_path = bdir / f"anchors_{sig}.json"
    if cache_path.exists():
        raw = read_json(cache_path)
        return {k: np.array(v, dtype=np.float32) for k, v in (raw.get("vecs") or {}).items()}

    ep_map: dict[int, tuple[Path, str]] = {}
    for p in sorted(media_root.glob("*/ingest.json")):
        try:
            d = read_json(p)
            m = re.search(r"(\d+)", Path(d.get("file", "")).stem)
            if m and (p.parent / "audio.wav").exists():
                ep_map[int(m.group(1))] = (p.parent / "audio.wav", d.get("file", ""))
        except Exception:  # noqa: BLE001
            continue

    pcm_cache: dict[int, tuple[np.ndarray, int]] = {}
    out: dict[str, np.ndarray] = {}
    samples_dir = bdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for idx, (cname, spans) in enumerate(anchors.items(), 1):
        if not isinstance(spans, list):
            continue
        chunks: list[np.ndarray] = []
        first_span: tuple[int, float, float] | None = None
        sr_out = 16000
        for item in spans:
            try:
                ep_no, t0, t1 = int(item[0]), float(item[1]), float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            if ep_no not in ep_map:
                continue
            if ep_no not in pcm_cache:
                pcm_cache[ep_no] = _load_pcm(ep_map[ep_no][0])
            pcm, sr = pcm_cache[ep_no]
            sr_out = sr
            a, b = int(max(0.0, t0) * sr), int(min(len(pcm) / sr, t1) * sr)
            if b - a >= int(0.4 * sr):
                chunks.append(pcm[a:b])
                if first_span is None:
                    first_span = (ep_no, t0, t1)
        if not chunks:
            continue
        cat = np.concatenate(chunks)
        v = _embed_pcm(cat)
        if v is None:
            continue
        out[cname] = v
        # 导出样本 wav 与说话时刻画面截图，方便人工核验
        stem = f"{idx:02d}_{slug(cname)}"
        wav_out = samples_dir / f"{stem}.wav"
        jpg_out = samples_dir / f"{stem}.jpg"
        if not wav_out.exists():
            with wave.open(str(wav_out), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr_out)
                wf.writeframes((np.clip(cat[:sr_out * 25], -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        if first_span and not jpg_out.exists():
            ep_no, t0, t1 = first_span
            vfile = ep_map[ep_no][1]
            if vfile and Path(vfile).exists():
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{(t0 + t1) / 2:.1f}",
                                "-i", vfile, "-frames:v", "1", "-vf", "scale=480:-2", str(jpg_out)], check=False)

    if out:
        write_json(cache_path, {"ver": VP_VER, "vecs": {k: [round(float(x), 5) for x in v] for k, v in out.items()}})
        log.info("[voiceprint] 已加载 %d 位角色的标准声纹锚点（样本见 %s）", len(out), samples_dir)
    return out


def verify_episode(ing: dict, epi: dict, bible: dict, mdir: Path,
                   bank_path: Path | None = None) -> dict:
    """用 CAM++ 声纹核对说话人，修正张冠李戴并清理未通过声纹核验的臆测人名。"""
    wav_path = mdir / "audio.wav"
    subs = ing["subs"]
    if not wav_path.exists() or not subs or not epi.get("scenes"):
        return epi
    amap, aliases = _alias_map(bible)
    bible_chars = set(aliases.keys())
    if not bible_chars:
        return epi

    embs = line_embeddings(mdir, subs, ing.get("timing"))
    pcm, sr = _load_pcm(wav_path)
    durs = np.array([s["end"] - s["start"] for s in subs], dtype=np.float32)
    anchor_vecs = _load_anchor_vecs(bible, mdir.parent)

    by_char_scenes: dict[str, list[tuple[str, np.ndarray, float, bool]]] = defaultdict(list)
    sc_by_id = {sc["id"]: sc for sc in epi["scenes"]}

    for sc in epi["scenes"]:
        by_spk: dict[str, list[int]] = defaultdict(list)
        for i in range(sc["l0"], sc["l1"] + 1):
            spk = epi["speakers"][i]
            if spk:
                by_spk[amap.get(spk, spk)].append(i)
        people_c = {amap.get(p, p) for p in (sc.get("people") or [])}
        for cname, idxs in by_spk.items():
            v, tot = _scene_spk_emb(idxs, subs, durs, embs, pcm, sr)
            if v is not None and cname in bible_chars:
                by_char_scenes[cname].append((sc["id"], v, tot, cname in people_c))

    bank: dict[str, tuple[np.ndarray, float]] = {}
    if bank_path and bank_path.exists():
        try:
            raw = read_json(bank_path)
            if raw.get("ver") == VP_VER:
                for k, item in (raw.get("chars") or {}).items():
                    v = np.array(item["vec"], dtype=np.float32)
                    nrm = float(np.linalg.norm(v))
                    if nrm > 0:
                        bank[k] = (v / nrm, float(item.get("dur", 10.0)))
        except Exception:  # noqa: BLE001
            bank = {}

    ep_chars = set(by_char_scenes.keys())
    for sc in epi["scenes"]:
        for p in sc.get("people") or []:
            if amap.get(p, p) in bible_chars:
                ep_chars.add(amap.get(p, p))

    def build_protos(exclude_sc: str | None = None) -> dict[str, np.ndarray]:
        protos: dict[str, np.ndarray] = {}
        for cname in ep_chars:
            items = by_char_scenes.get(cname) or []
            avec = anchor_vecs.get(cname)
            if avec is not None:
                inliers = [x for x in items if x[0] != exclude_sc and float(x[1] @ avec) >= 0.55]
                p = avec * 40.0
                for _, v, tot, _ in inliers:
                    p = p + v * min(tot, 12.0)
                protos[cname] = p / np.linalg.norm(p)
            elif len(items) >= 2 or sum(x[2] for x in items) >= 15.0 or cname in bank:
                pool = ([x for x in items if x[0] != exclude_sc and x[3] and x[2] >= 2.5]
                        or [x for x in items if x[0] != exclude_sc and x[3]]
                        or [x for x in items if x[0] != exclude_sc])
                if pool:
                    mat = np.stack([x[1] for x in pool])
                    w = np.array([min(x[2], 20.0) for x in pool], dtype=np.float32)
                    sim = mat @ mat.T
                    med = int(np.argmax((sim >= 0.45) @ w + 0.3 * (sim @ w)))
                    keep = [k for k in range(len(pool)) if sim[med, k] >= 0.42]
                    p = (mat[keep] * w[keep, None]).sum(axis=0)
                    if cname in bank:
                        bv, bdur = bank[cname]
                        if float((p / np.linalg.norm(p)) @ bv) >= 0.45:
                            p = p + bv * min(bdur, 20.0)
                    protos[cname] = p / np.linalg.norm(p)
                elif cname in bank:
                    protos[cname] = bank[cname][0]
        return protos

    swaps_by_sc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    scenes_list = epi["scenes"]
    for si, sc in enumerate(scenes_list):
        sc_id = sc["id"]
        protos = build_protos(exclude_sc=sc_id)
        if not protos:
            continue
        by_spk: dict[str, list[int]] = defaultdict(list)
        for i in range(sc["l0"], sc["l1"] + 1):
            spk = epi["speakers"][i]
            if spk:
                by_spk[amap.get(spk, spk)].append(i)
        people_c = {amap.get(p, p) for p in (sc.get("people") or [])}
        sc_items: list[tuple[str, str, np.ndarray, float, list[int], bool]] = []
        for cname, idxs in by_spk.items():
            subs_g = _split_if_mixed(idxs, subs, durs, embs, pcm, sr) if cname in bible_chars else [idxs]
            for gi, g in enumerate(subs_g):
                vg, totg = _scene_spk_emb(g, subs, durs, embs, pcm, sr)
                if vg is not None:
                    key = cname if len(subs_g) == 1 else f"{cname}#{gi + 1}"
                    sc_items.append((key, cname, vg, totg, g, cname in people_c))

        near_people: set[str] = set()
        for d in (-1, 0, 1):
            if 0 <= si + d < len(scenes_list):
                for p in scenes_list[si + d].get("people") or []:
                    near_people.add(amap.get(p, p))
                for i in range(scenes_list[si + d]["l0"], scenes_list[si + d]["l1"] + 1):
                    near_people.add(amap.get(epi["speakers"][i], epi["speakers"][i]))

        # 第一遍：锁定本场已确认的说话人（声纹排第一或当面被称呼且声纹吻合）
        assigned: dict[str, tuple[np.ndarray, list[int]]] = {}
        for key, cname, v, tot, idxs, in_p in sc_items:
            if cname in protos:
                self_s = float(v @ protos[cname])
                top_s = max(float(v @ rv) for rv in protos.values())
                other_lines = [subs[i]["text"] for i in range(sc["l0"], sc["l1"] + 1) if i not in set(idxs)]
                voc_anchored = in_p and self_s >= 0.35 and _is_vocative(cname, other_lines, aliases)
                if voc_anchored or (self_s >= 0.42 and top_s - self_s <= 0.06):
                    assigned[cname] = (v, idxs)

        # 第二遍：按最高匹配置信度从高到低更正张冠李戴
        for key, cname, v, tot, idxs, in_p in sorted(sc_items, key=lambda x: -max(float(x[2] @ rv) for rv in protos.values())):
            if cname in assigned and key == cname:
                continue
            is_named = cname in bible_chars
            self_s = float(v @ protos[cname]) if (is_named and cname in protos) else -1.0

            if is_named and not in_p and cname in protos and self_s < 0.43:
                used_raw = {epi["speakers"][i] for i in range(sc["l0"], sc["l1"] + 1) if epi["speakers"][i]}
                fallback = next((p for p in (sc.get("people") or [])
                                 if amap.get(p, p) not in bible_chars and p not in used_raw), "")
                log.info("[voiceprint] %s 清理未在场角色 %s(self=%.2f) → %s", sc_id, key, self_s, fallback or "?")
                for i in idxs:
                    epi["speakers"][i] = fallback
                continue

            if tot < 2.5 or (is_named and cname not in protos):
                continue

            cands = [
                (k, float(v @ rv)) for k, rv in protos.items()
                if (is_named or k in near_people)
                and (k not in assigned or (_adj_turns(idxs, assigned[k][1]) == 0 and float(v @ assigned[k][0]) >= 0.68))
            ]
            if not cands:
                continue
            scores = sorted(cands, key=lambda x: -x[1])
            best_c, best_s = scores[0]
            second_s = scores[1][1] if len(scores) > 1 else 0.0
            if is_named:
                same_room_merge = best_c in assigned and float(v @ assigned[best_c][0]) >= 0.68
                ok = (best_c != cname and best_s >= 0.58 and best_s - max(self_s, 0.35) >= 0.12
                      and (self_s < 0.44 or best_s - self_s >= 0.25 or same_room_merge))
            else:
                ok = best_s >= 0.60 and best_s - second_s >= 0.06
            if not ok:
                if is_named and self_s >= 0.38:
                    assigned[cname] = (v, idxs)
                continue

            log.info("[voiceprint] %s 说话人更正 %s → %s (dur=%.1fs, best=%.2f, self=%.2f)",
                     sc_id, key, best_c, tot, best_s, self_s)
            assigned[best_c] = (v, idxs)
            for i in idxs:
                epi["speakers"][i] = best_c
            if best_c not in sc.get("people", []):
                sc.setdefault("people", []).append(best_c)
            if key == cname:
                swaps_by_sc[sc_id].append((cname, best_c))

    # 同步修正场次 people / place / summary（仅当原名未在该场台词文字里作为话题出现时才替换 summary）
    for sc_id, swaps in swaps_by_sc.items():
        sc = sc_by_id[sc_id]
        sc_text = " ".join(subs[i]["text"] for i in range(sc["l0"], sc["l1"] + 1))
        people = list(sc.get("people") or [])
        for old_c, new_c in swaps:
            old_set = set(aliases.get(old_c, [old_c]))
            people = [new_c if (p in old_set or amap.get(p, p) == old_c) else p for p in people]
            if new_c not in {amap.get(p, p) for p in people}:
                people.append(new_c)
        sc["people"] = list(dict.fromkeys(people))
        safe_swaps = [(o, n) for o, n in swaps if not any(tok in sc_text for tok in aliases.get(o, [o]) if len(tok) >= 2)]
        if safe_swaps:
            sc["place"] = _swap_text(sc.get("place", ""), safe_swaps, aliases)
            sc["summary"] = _swap_text(sc.get("summary", ""), safe_swaps, aliases)

    # 更新跨集声纹库
    if bank_path:
        final_protos = build_protos(exclude_sc=None)
        merged = {k: {"vec": [round(float(x), 5) for x in v], "dur": d} for k, (v, d) in bank.items()}
        for k, v in final_protos.items():
            dur = float(sum(x[2] for x in by_char_scenes.get(k, [])))
            if dur >= 5.0:
                merged[k] = {"vec": [round(float(x), 5) for x in v], "dur": round(min(dur, 60.0), 1)}
        write_json(bank_path, {"ver": VP_VER, "chars": merged})
    return epi

