"""Shared helpers for the pipeline stages that still shell out to a sibling
repo's own uv env (IsaacSim, TRELLIS.2) -- everything else in the pipeline
runs in-process (see genrecon/pipeline/stages.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from genrecon.utils.logger import logger


def log_debug_config(
    log_path: Path,
    name: str,
    program: str,
    cwd: str,
    args: list[str],
    *,
    python: str = "${workspaceFolder}/.venv/bin/python",
) -> None:
    """Append a VS Code debugpy launch.json config entry describing a stage's
    exact invocation, so it can be re-run under the debugger on the same
    data. Inlined equivalent of scripts/append_debug_launch_config.py --
    only used by run_external_step (surviving subprocess stages); in-process
    stages have no separate "program" to replay this way.
    """
    config = {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": program,
        "console": "integratedTerminal",
        "justMyCode": False,
        "cwd": cwd,
        "python": python,
        "args": list(args),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(config, indent=4) + ",\n")


class LogMirror:
    """Appends whatever's newly written to a subprocess's own log file into
    the unified pipeline.log. Needed because external subprocesses (Isaac,
    TRELLIS.2, and segmentation/main_light.py's own internal subprocess
    dispatch) don't log through the shared loguru logger, so their per-stage
    log files would otherwise never make it into pipeline.log.

    Mirrors bash's `mirror_log` + `declare -A LOG_LINE_OFFSET` (see
    run_full_pipeline.sh's mirror_log()): tracks how many lines of each log
    file have already been mirrored, so repeated calls only append the delta.
    """

    def __init__(self, pipeline_log: Path) -> None:
        self._pipeline_log = pipeline_log
        self._offsets: dict[Path, int] = {}

    def mirror(self, log_file: Path) -> None:
        if not log_file.exists():
            return
        lines = log_file.read_text(errors="replace").splitlines(keepends=True)
        total = len(lines)
        prev = self._offsets.get(log_file, 0)
        if total > prev:
            with open(self._pipeline_log, "a") as f:
                f.writelines(lines[prev:total])
        self._offsets[log_file] = total


def run_external_step(
    name: str,
    script: Path,
    cwd: Path,
    args: list[str],
    *,
    log_file: Path,
    append: bool = False,
    runner: list[str] | None = None,
    debug_config_log: Path | None = None,
    log_mirror: LogMirror | None = None,
) -> None:
    """Runs a stage that lives in a sibling repo's own uv env (IsaacSim,
    TRELLIS.2) as a real subprocess, logging a replayable debugpy launch
    config first and mirroring its output into the unified pipeline log
    afterwards. Direct port of run_full_pipeline.sh's run_external_step().

    Raises subprocess.CalledProcessError on a nonzero exit -- this is what
    bash's `set -euo pipefail` got for free; here it's explicit via
    check=True.
    """
    runner = runner or ["uv", "run", script.name]
    if debug_config_log is not None:
        log_debug_config(debug_config_log, name, str(script), str(cwd), args)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    logger.info(f"[{name}] running: {' '.join(runner + args)} (cwd={cwd})")
    try:
        with open(log_file, mode) as f:
            subprocess.run(runner + args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)
    finally:
        if log_mirror is not None:
            log_mirror.mirror(log_file)


@contextmanager
def redirect_fd_to_file(log_file: Path, append: bool = False):
    """Redirects OS-level stdout/stderr (fd 1/2) to `log_file` for the duration of the block.

    Needed for in-process stages that (a) call `print()` directly (plain-print scripts like
    segmentation/main_light.py, mv_sam3d/mvsam3d_scripts/*.py have no loguru sink of their own)
    and/or (b) themselves spawn subprocesses that inherit the real fd 1/2 (main_light.py's own
    dispatch to detect_and_segment.py/segment_pointcloud.py) -- plain `contextlib.redirect_stdout`
    only patches `sys.stdout` in this process and would miss both cases.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(log_file, mode) as f:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


@contextmanager
def stage(name: str):
    """Logs a stage's start, and its elapsed time on the way out (including
    on failure, via `finally`, so a failed stage still gets an elapsed-time
    log line before the exception propagates). Replaces bash's
    stage_start/stage_end pairing.
    """
    t0 = time.monotonic()
    logger.info(name)
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        minutes, seconds = divmod(int(elapsed), 60)
        logger.info(f"{name} done (elapsed {minutes}m{seconds:02d}s)")
