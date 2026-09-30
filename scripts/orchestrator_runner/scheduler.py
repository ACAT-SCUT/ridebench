"""Run the experiment manifest with one tmux worker per requested GPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from task_manifest import REGULAR_GROUPS, Task, tasks_for_suite

ROOT_DIR = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = Path(__file__).resolve()
DEFAULT_UV_CACHE_DIR = ROOT_DIR / ".uv-cache"


def parse_gpu_list(raw: str) -> list[str]:
    result = [item.strip() for item in raw.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one GPU id is required.")
    return result


def parse_groups(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    groups = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [group for group in groups if group not in REGULAR_GROUPS]
    if unknown:
        raise ValueError(f"Unknown regular groups: {unknown}. Valid groups: {sorted(REGULAR_GROUPS)}")
    return groups


def write_task_list(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for number, task in enumerate(tasks, 1):
            command = " ".join(shlex.quote(part) for part in task.command)
            output.write(f"{number:04d}\t{task.label}\t{command}\n")


def build_state(tasks: list[Task], suite: str, gpus: list[str], runner: str) -> dict[str, Any]:
    entries = []
    for number, task in enumerate(tasks, start=1):
        entries.append({
            "index": number,
            "group": task.group,
            "label": task.label,
            "command": list(task.command),
            "status": "pending",
            "worker_id": None,
            "gpu_id": None,
            "returncode": None,
            "started_at": None,
            "finished_at": None,
        })
    return {
        "suite": suite,
        "runner": runner,
        "gpus": gpus,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": entries,
    }


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def with_locked_state(lock_path: Path, fn: Callable[[], Any]):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def claim_next_task(*, state_path: Path, lock_path: Path, worker_id: int,
                    gpu_id: str) -> dict[str, Any] | None:
    def claim() -> dict[str, Any] | None:
        state = load_json(state_path)
        for task in state["tasks"]:
            if task["status"] != "pending":
                continue
            task.update({
                "status": "running",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            })
            save_json_atomic(state_path, state)
            return task
        return None

    return with_locked_state(lock_path, claim)


def finish_task(*, state_path: Path, lock_path: Path, task_index: int, returncode: int) -> None:
    def finish() -> None:
        state = load_json(state_path)
        for task in state["tasks"]:
            if task["index"] != task_index:
                continue
            task["status"] = "done" if returncode == 0 else "failed"
            task["returncode"] = returncode
            task["finished_at"] = datetime.now().isoformat(timespec="seconds")
            save_json_atomic(state_path, state)
            return
        raise RuntimeError(f"Task index not found in state file: {task_index}")

    with_locked_state(lock_path, finish)


def _command_text(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_task(*, task: Task, gpu_id: str, worker_id: int, log_path: Path,
             runner: str, uv_cache_dir: Path) -> int:
    del worker_id  # retained in the public signature for worker compatibility
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": gpu_id, "BENCH_RUNNER": runner})
    environment.setdefault("UV_CACHE_DIR", str(uv_cache_dir))

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[RUN] {datetime.now():%F %T} {task.label}\n")
        log.write(f"[CMD] {_command_text(task.command)}\n")
        log.flush()
        completed = subprocess.run(
            task.command,
            cwd=ROOT_DIR,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status = "DONE" if completed.returncode == 0 else "FAIL"
        log.write(f"[{status}] {datetime.now():%F %T} {task.label} exit={completed.returncode}\n")
    return completed.returncode


def tmux_session_alive(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def worker_main(args: argparse.Namespace) -> int:
    state_path = Path(args.state_file)
    lock_path = Path(args.lock_file)
    log_dir = Path(args.log_dir)
    uv_cache_dir = Path(args.uv_cache_dir)
    worker_log = log_dir / f"worker{args.worker_id}_gpu{args.gpu_id}.log"
    failures: list[tuple[str, int]] = []

    while (claimed := claim_next_task(
        state_path=state_path, lock_path=lock_path,
        worker_id=args.worker_id, gpu_id=args.gpu_id,
    )) is not None:
        task = Task(group=claimed["group"], label=claimed["label"], command=tuple(claimed["command"]))
        print(f"[GPU {args.gpu_id}] RUN {task.label}", flush=True)
        result = run_task(
            task=task, gpu_id=args.gpu_id, worker_id=args.worker_id, log_path=worker_log,
            runner=args.runner, uv_cache_dir=uv_cache_dir,
        )
        finish_task(state_path=state_path, lock_path=lock_path,
                    task_index=claimed["index"], returncode=result)
        if result == 0:
            print(f"[GPU {args.gpu_id}] DONE {task.label}", flush=True)
        else:
            print(f"[GPU {args.gpu_id}] FAIL {task.label} exit={result}", flush=True)
            failures.append((task.label, result))

    if failures:
        print("Worker completed with failures:", flush=True)
        for label, result in failures:
            print(f" - {label}: exit={result}", flush=True)
        return 1
    print(f"[GPU {args.gpu_id}] worker idle", flush=True)
    return 0


def launch_tmux_workers(*, suite: str, tasks: list[Task], gpus: list[str], log_dir: Path,
                        runner: str, state_path: Path, lock_path: Path,
                        uv_cache_dir: Path) -> list[str]:
    del tasks  # task definitions are already persisted in state_path
    sessions: list[str] = []
    for worker_id, gpu_id in enumerate(gpus):
        session = f"orchestrator_{suite}_gpu{gpu_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        worker_log = log_dir / f"worker{worker_id}_gpu{gpu_id}.log"
        worker_command = (
            f"cd {shlex.quote(str(ROOT_DIR))} && "
            f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)} "
            f"BENCH_RUNNER={shlex.quote(runner)} "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(SCHEDULER_PATH))} worker "
            f"--state-file {shlex.quote(str(state_path))} "
            f"--lock-file {shlex.quote(str(lock_path))} "
            f"--log-dir {shlex.quote(str(log_dir))} "
            f"--gpu-id {shlex.quote(gpu_id)} "
            f"--worker-id {worker_id} "
            f"--runner {shlex.quote(runner)} "
            f">> {shlex.quote(str(worker_log))} 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, "bash", "-lc", worker_command],
            check=True,
        )
        sessions.append(session)
        print(f"[STARTED] {session} gpu={gpu_id} log={worker_log}")
    return sessions


def wait_for_tmux_sessions(session_names: list[str]) -> None:
    remaining = set(session_names)
    while remaining:
        remaining.difference_update({name for name in remaining if not tmux_session_alive(name)})
        if remaining:
            time.sleep(2)


def summarize_state(state_path: Path) -> list[tuple[str, int]]:
    state = load_json(state_path)
    failures: list[tuple[str, int]] = []
    for task in state["tasks"]:
        if task["status"] in {"failed", "pending", "running"}:
            returncode = task["returncode"]
            failures.append((task["label"], -1 if returncode is None else int(returncode)))
    return failures


def run_tasks(*, suite: str, tasks: list[Task], gpus: list[str], log_dir: Path,
              runner: str, dry_run: bool) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    write_task_list(log_dir / "task_queue.tsv", tasks)
    state_path = log_dir / "task_state.json"
    lock_path = log_dir / "task_state.lock"
    cache_dir = Path(os.environ.get("UV_CACHE_DIR", str(DEFAULT_UV_CACHE_DIR)))
    save_json_atomic(state_path, build_state(tasks, suite, gpus, runner))

    print(f"Task count: {len(tasks)}")
    print(f"GPU workers: {','.join(gpus)}")
    print(f"Logs: {log_dir}")
    if dry_run:
        for number, task in enumerate(tasks, 1):
            print(f"{number:04d} {task.label}: {_command_text(task.command)}")
        return 0

    sessions = launch_tmux_workers(
        suite=suite, tasks=tasks, gpus=gpus, log_dir=log_dir, runner=runner,
        state_path=state_path, lock_path=lock_path, uv_cache_dir=cache_dir,
    )
    print(f"Attached tmux sessions: {', '.join(sessions)}")
    print(f"Attach with: tmux attach -t {sessions[0]}")
    wait_for_tmux_sessions(sessions)
    failures = summarize_state(state_path)
    if failures:
        print("Completed with failures:")
        for label, returncode in failures:
            print(f" - {label}: exit={returncode}")
        return 1
    print("All tasks completed.")
    return 0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if argv[:1] == ["worker"]:
        parser = argparse.ArgumentParser(description="tmux worker loop")
        parser.add_argument("mode")
        parser.add_argument("--state-file", required=True)
        parser.add_argument("--lock-file", required=True)
        parser.add_argument("--log-dir", required=True)
        parser.add_argument("--gpu-id", required=True)
        parser.add_argument("--worker-id", type=int, required=True)
        parser.add_argument("--runner", default=os.environ.get("BENCH_RUNNER", "auto"))
        parser.add_argument("--uv-cache-dir", default=str(DEFAULT_UV_CACHE_DIR))
        args = parser.parse_args(argv)
        args.mode = "worker"
        return args

    parser = argparse.ArgumentParser(description="Schedule benchmark script tasks across available GPUs.")
    parser.add_argument("suite", choices=["regular", "ultra_long", "loss", "area", "lookback", "all"])
    parser.add_argument("--gpus", default=os.environ.get("TMUX_GPU_LIST", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--groups", help="Comma-separated regular groups for regular/ultra_long suites.")
    parser.add_argument("--runner", default=os.environ.get("BENCH_RUNNER", "auto"))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.mode = "suite"
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "worker":
        return worker_main(args)
    groups = parse_groups(args.groups)
    tasks = tasks_for_suite(args.suite, groups)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else ROOT_DIR / "logs" / f"orchestrator_{args.suite}_{timestamp}"
    return run_tasks(
        suite=args.suite, tasks=tasks, gpus=parse_gpu_list(args.gpus), log_dir=log_dir,
        runner=args.runner, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
