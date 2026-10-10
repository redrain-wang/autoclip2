# autoclip — 全自动影视解说短视频流水线

## 方法论：像人类解说博主一样“看 → 懂 → 选 → 写 → 配 → 剪”

```mermaid
flowchart LR
  A[原片] --> B[1 预处理<br/>ASR字幕/镜头/音量/段落/关键帧]
  B --> C[2 段落理解<br/>VLM看图+台词 打6维分]
  C --> D[3 剧情梳理<br/>逐集→全剧 人物/主线/转折]
  D --> E[4 导演选段<br/>规则预筛+LLM编排故事线]
  E --> F[5 编剧写稿<br/>解说段/原声段交替]
  F --> G[6 配音 TTS]
  G --> H[7 剪辑合成<br/>压低原声/字幕/标题/BGM]
```

1. **切成剧情段落（beat）**：镜头检测 + 台词边界，把原片合并成 15~50 秒、不在台词中间断开的段落，作为最小“叙事单元”。
2. **多模态打分**：每段 4 张关键帧 + 台词 + 音量峰值交给视觉模型，输出客观摘要和 6 维分数（笑点/冲突/情感/悬念/奇观/台词）。打分标准对所有作品统一，所以可以批量生产。
3. **先懂全剧再选段**：逐集梳理剧情、统一人物称呼、标出转折点；多集作品再汇总成全剧剧情，避免“只见高光不见剧情”。
4. **导演选段（规则加 LLM）**：按类型加权打分（喜剧看重笑点，悬疑看重悬念），预筛候选，再由 LLM 按方法论编排：
   - 黄金 3 秒钩子（允许倒叙闪回）
   - 只走一条主线，除钩子外按时间顺序，保证剧情连贯
   - 每 15~25 秒一个新信息/笑点/反转
   - 选定一个解说角度（吐槽反差/命运共情/反转揭秘……）
5. **解说段与原声段交替**：解说负责交代前因、补信息差、递梗和点评；金句、笑点、情绪爆发保留原声，让观众自己听。原声段自动对齐到完整台词边界。
6. **画面跟着配音走**：解说段的画面时长等于配音时长，同一段落内画面连续推进、不回跳。
7. **每个阶段都有缓存**：中途失败可以断点续跑；可以人工改 `script.json` 后只重跑配音和剪辑；同一素材被多个任务复用时不会重复花钱。

## 安装

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[mlx]"   # Apple Silicon；其他平台用 .[local-asr]
cp config.example.yaml config.yaml && cp .env.example .env  # 填入 API Key
```

## 使用

```bash
# 单集
.venv/bin/autoclip run 第01集.mp4 --name 一乡之长01 --genre comedy --duration 180
# 一个目录 = 一部剧；per_episode 每集出一条，whole 全剧出一条
.venv/bin/autoclip run /path/合集 --name 一乡之长 --mode per_episode
# 批量
.venv/bin/autoclip batch jobs.yaml --workers 2

# 规格预设（config.yaml 的 presets）：一次出多种规格，只做部分集
.venv/bin/autoclip run /path/合集 --name 一乡之长 --preset short,bili,recap --episodes 1-3
# 每个任务出几条：固定数字，或 auto（导演按素材判断，每条独立主线）
.venv/bin/autoclip run 第01集.mp4 --name 一乡之长01 --count auto

# 人工审稿：写完稿就停 → 编辑 work/jobs/<name>/script.json → 再运行一次即可出片
.venv/bin/autoclip run ... --stop-after script
# 换个角度重写
.venv/bin/autoclip run ... --redo plan
```

成片在 `output/<name>.mp4`（一个任务出多条时为 `<name>-1.mp4`、`<name>-2.mp4`…），同名 `.json` 里是标题、发布文案和标签。可读版的稿子在 `work/jobs/<name>/v1/script.md`。

模型配置支持 `fallback`：主服务额度用尽或调用失败时，自动切到备用服务。

## 调优入口

- 解说风格和方法论都在 [autoclip/prompts.py](autoclip/prompts.py)，效果主要靠打磨这里。
- 各类型的选段权重：`GENRE_WEIGHTS`。
- 写稿用最强的模型，逐段打分用便宜的多模态模型（调用量最大）。
