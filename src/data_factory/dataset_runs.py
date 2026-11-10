"""What every table dataset (Olist, credit, holidays) does the same way around its stages.

`run_stages` runs the stages in order, times each one and writes the run manifest whether the
run succeeds or fails. `manifest_header` is the common head of that manifest: the run, the code
and the inputs. Each dataset adds only what is its own (source, counts, outputs) (D72).
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from data_factory.manifest import build_code_inputs, utc_now

log = logging.getLogger(__name__)


def run_stages(context, dataset: str, stage_names, stages: dict[str, Callable], record_run: Callable) -> None:
    record = context.record
    try:
        for name in stage_names:
            log.info("%s stage %s: start", dataset, name)
            started = time.perf_counter()
            record.stages.append(name)
            stages[name](context)
            record.stage_seconds[name] = round(time.perf_counter() - started, 1)
    except Exception as error:
        record_run(context, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    record_run(context, status="succeeded")


def manifest_header(context, dataset: str, status: str, error: str | None, snapshot: str) -> dict:
    config, record = context.config, context.record
    inputs = build_code_inputs(config.path)
    return {
        "dataset": dataset, "status": status, "error": error,
        "run": {"run_id": record.run_id, "started_at_utc": record.started_at.isoformat(),
                "finished_at_utc": utc_now().isoformat(),
                "elapsed_seconds": round(time.perf_counter() - record.started_clock, 1),
                "stages": record.stages, "stage_seconds": record.stage_seconds},
        "code": inputs["code"],
        "inputs": {"snapshot": snapshot, "config_file": config.path.name,
                   "config_sha256": inputs["config_sha256"], "git_commit": inputs["git_commit"]},
    }
