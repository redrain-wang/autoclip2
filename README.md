# autoclip — 全自动影视解说短视频流水线

## 方法论：说书式解说，“先懂全剧 → 再拆一集 → 一条讲一段完整故事”

```mermaid
flowchart LR
  A[原片] --> B[ingest<br/>ASR字幕/镜头/关键帧]
  B --> C[analyze<br/>VLM看画面写摘要]
  C --> D[bible<br/>全剧资料：人物/关系/梗概/错字表]
  D --> E[episode<br/>单集拆场次+故事线<br/>逐场标说话人、校错字]
  E --> F[plan<br/>每集按故事线策划 N 条]
  F --> G[script<br/>钩子+解说/原声交替]
  G --> H[tts 配音]
  H --> I[render<br/>竖屏合成/字幕/水印/封面]
```

1. **剧集资料（bible）是唯一事实来源**：`series/<剧名>.yaml`，用全剧台词生成人物、关系、各集梗概和识别错字表。`notes` 写人工确认的事实，优先级最高；人工修改后用 `--redo episode` 让下游按新资料重做。
2. **单集拆场次**：把一集切成 20~40 场连续的戏，归纳 2~5 条故事线；再逐场判断每句台词是谁说的，顺带校正同音错字（字幕也随之修正）。
3. **一条成片 = 一段完整故事**：导演按故事线策划，只选同一条线的场戏，按时间顺序讲，有起因、经过、结果。
4. **结构固定**：开场 4~12 秒原声钩子 → 解说接住钩子进入故事 → 解说与原声交替（解说约 60%，原声约 40%；笑点、金句、吵架一律留原声）→ 结尾留悬念。
5. **原声剪辑规则**：按整句切，同一人一口气说的话不从中间切断；收尾比下一句提前 0.25 秒，避免带出下一句的开头。
6. **包装**：右上角“爱大片儿”水印；每条成片自动生成同名封面图（剧名集数标签、精选画面、2~3 行封面大字）。
7. **全程缓存、断点续跑**：任一阶段失败重跑即可从断点继续；改了哪一环就 `--redo` 哪一环。

## 安装

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[mlx]"   # Apple Silicon；其他平台用 .[local-asr]
cp config.example.yaml config.yaml && cp .env.example .env  # 填入 API Key
```

## 标准流程

```bash
DIR="/Volumes/ExternalSSD/Data/videos/原片修复/一乡之长/合集"   # 一个目录 = 一部剧，文件名含集号

# ① 试做一集审片（新剧会自动建立剧集资料，见下文）
.venv/bin/autoclip run "$DIR" --name 一乡之长 --preset short --episodes 1

# ② 批量：一条命令出全部成片 + 封面（--workers 为并行集数）
.venv/bin/autoclip run "$DIR" --name 一乡之长 --preset short --episodes 2-18 --workers 2
```

产物都在 `output/`：

| 文件 | 内容 |
|---|---|
| `一乡之长_short_02-1.mp4` | 第 2 集第 1 条成片（竖屏 1080×1920，带水印） |
| `一乡之长_short_02-1.jpg` | 配套封面 |
| `一乡之长_short_02-1.json` | 标题、封面字、发布文案、话题标签、时间轴 |

可读版解说稿：`work/jobs/一乡之长_short_02/v1/script.md`。

## 换一部剧

只需把素材放进一个目录（文件名能按集数自然排序，如 `第01集.mp4`、`EP02.mkv`；有同名 `.srt` 会跳过语音识别），用新的 `--name` 运行上面的命令。剧集资料全自动建立：

1. **换剧检测**：`series/<剧名>.yaml` 不存在 → 新剧；存在但素材变了、且里面的主要人物在新台词里基本没出现 → 换剧（旧文件备份为 `.yaml.bak`）；只是增减了几集 → 沿用。
2. **自动补资料**：查公开百科（年代、题材、主演、剧情简介），结合模型自身对这部剧的了解，再拿开头几集台词核对是不是同一部作品（重名作品对不上就弃用），写入 `research`。
3. **生成剧集资料**：用全剧台词 + 上述资料生成人物、关系、各集梢概、识别错字表；台词超过约 15 集时自动分段生成再合并。
4. **自动审校**：从全剧台词里为每个人物、每条错字校正检索原句作证据，修正与证据矛盾的关系、删掉没有依据的人物和会误伤普通词的错字条目（日志里 `[bible] 审校：` 记录每处改动）。

之后的单集梳理、策划、写稿、事实核查、配音、合成都与剧无关。

**可选**：发现错误时也可以直接在 yaml 里加人工资料，优先级最高：
```yaml
notes: |            # 人工确认的事实
style: 悬疑正剧，解说克制、多留悬念      # 解说口吻
pronounce: {行长: 航长}                 # 本剧特有的多音字（只改配音）
```
改完后 `--redo episode` 重做单集梳理及之后；`--redo bible` 则重新检索并生成整份资料（保留 notes/style/pronounce）。

## 返工

```bash
# 改了 series/<剧名>.yaml（人物关系、错字表）后，重做单集梳理及之后
.venv/bin/autoclip run "$DIR" --name 一乡之长 --preset short --episodes 3 --redo episode
# 换个策划 / 只重写解说稿 / 只重新配音合成
... --redo plan    |    ... --redo script    |    ... --redo tts
# 人工审稿：写完稿就停 → 编辑 work/jobs/<name>/v*/script.json → 再运行一次出片
... --stop-after script
```

## 其他用法

```bash
# 规格预设（config.yaml 的 presets）：short 抖音竖屏 / bili 横屏 / recap 全剧浓缩
.venv/bin/autoclip run "$DIR" --name 一乡之长 --preset short,bili --episodes 1-3
# 按 jobs.yaml 批量处理多部剧
.venv/bin/autoclip batch jobs.yaml --workers 2
```

模型配置支持 `fallback`：主服务额度用尽或调用失败时，自动切到备用服务。

## 调优入口

- 解说方法论都在 [autoclip/prompts.py](autoclip/prompts.py)。
- 解说/原声占比、目标时长：`config.yaml` 的 `presets.short`（`narration_ratio`、`target_duration`）。
- 配音多音字：`config.yaml` 的 `tts.pronounce`（只改配音，不改字幕）。
- 水印与封面：`config.yaml` 的 `render.brand / watermark / cover`；图形设计在 [autoclip/brand.py](autoclip/brand.py)，删掉 `assets/brand/logo.png` 会按当前设计重新生成。
