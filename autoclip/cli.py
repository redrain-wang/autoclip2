from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from .config import build_jobs, load_jobs, load_settings
from .pipeline import STAGES, Context, run_job
from .utils import log

MAX_FAIL_STREAK = 3  # 连续这么多个任务失败或无产出，视为系统性问题并停止


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="autoclip", description="全自动影视解说短视频生产")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--redo", choices=STAGES, help="从该阶段起重新生成（清除缓存）")
    common.add_argument("--stop-after", choices=STAGES, help="执行到该阶段后停止（便于人工审稿）")

    r = sub.add_parser("run", parents=[common], help="处理单个作品")
    r.add_argument("inputs", nargs="+", help="视频文件 / 目录 / 通配符")
    r.add_argument("--name", required=True)
    r.add_argument("--preset", help="config.yaml 中的规格预设，多个用逗号分隔，如 short,bili,recap")
    r.add_argument("--episodes", help="只处理部分集，如 1 或 1-3,5")
    r.add_argument("--mode", choices=["whole", "per_episode"])
    r.add_argument("--genre")
    r.add_argument("--style")
    r.add_argument("--duration", type=int, help="目标时长（秒）")
    r.add_argument("--count", help="每个任务出几条成片：数字或 auto")
    r.add_argument("--layout", choices=["vertical", "horizontal"])
    r.add_argument("--workers", type=int, default=1, help="并行处理的集数")

    b = sub.add_parser("batch", parents=[common], help="按 jobs.yaml 批量生产")
    b.add_argument("jobs")
    b.add_argument("--workers", type=int, default=1, help="并行作品数")

    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ctx = Context(load_settings(a.config))
    if a.cmd == "run":
        inputs = a.inputs if len(a.inputs) > 1 else a.inputs[0]
        presets = [p.strip() for p in a.preset.split(",")] if a.preset else None
        jobs = build_jobs(a.name, inputs, ctx.s, presets, mode=a.mode, episodes=a.episodes, genre=a.genre,
                          style=a.style, target_duration=a.duration, layout=a.layout, count=a.count)
    else:
        jobs = load_jobs(a.jobs, ctx.s)
    workers = a.workers
    log.info("共 %d 个任务", len(jobs))

    failed = []
    streak = 0
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(run_job, j, ctx, a.redo, a.stop_after): j for j in jobs}
        for f in as_completed(futs):
            if f.cancelled():
                continue
            j = futs[f]
            try:
                out = f.result()
                log.info("✔ %s %s", j.name, "\n  ".join(map(str, out or [])))
                streak = streak + 1 if not out and not a.stop_after else 0
            except Exception as e:  # noqa: BLE001
                failed.append(j.name)
                streak += 1
                log.exception("✘ %s: %s", j.name, e)
            if streak >= MAX_FAIL_STREAK:
                log.error("连续 %d 个任务没有产出，停止剩余任务", streak)
                failed.append("(熔断)")
                for g in futs:
                    g.cancel()
    if failed:
        log.error("失败任务: %s", failed)
        sys.exit(1)
