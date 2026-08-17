#!/usr/bin/env python3
"""Exercise Docker-in-Docker through the real TerminalBench reward path."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import stat
import sys
from pathlib import Path


DOCKERFILE = """FROM python:3.11-slim
RUN apt-get update \\
    && apt-get install -y --no-install-recommends asciinema tmux \\
    && pip install --no-cache-dir pytest \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN touch /app/terminalbench-smoke-ready
"""

DOCKER_COMPOSE = """services:
  client:
    build:
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: ["sh", "-c", "sleep infinity"]
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
"""

TASK_YAML = """instruction: Verify the TerminalBench Docker runtime.
author_name: runtime-smoke
author_email: unknown
difficulty: easy
category: infrastructure
tags:
  - smoke
parser_name: pytest
max_agent_timeout_sec: 120.0
max_test_timeout_sec: 120.0
run_tests_in_same_shell: false
"""

TEST_FILE = """from pathlib import Path


def test_holder_runtime():
    assert Path('/app/terminalbench-smoke-ready').is_file()
"""


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rllm-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.rllm_root.resolve()
    sys.path[:0] = [str((root / "vendor").resolve()), str(root)]

    from src.tbench_rllm.docker_env import ContainerConfig, DockerIsolatedEnv, TaskConfig
    from src.tbench_rllm.rewards.test_reward import calculate_test_score

    work_root = args.work_root.resolve()
    task_root = work_root / "task" / "dind-smoke"
    output_root = work_root / "output"
    shutil.rmtree(work_root, ignore_errors=True)
    write(task_root / "Dockerfile", DOCKERFILE)
    write(task_root / "docker-compose.yaml", DOCKER_COMPOSE)
    write(task_root / "task.yaml", TASK_YAML)
    write(task_root / "solution.sh", "#!/bin/sh\nexit 0\n", executable=True)
    write(
        task_root / "run-tests.sh",
        "#!/bin/sh\nset -eu\npython -m pytest \"$TEST_DIR/test_outputs.py\" -rA\n",
        executable=True,
    )
    write(task_root / "tests" / "test_outputs.py", TEST_FILE)

    env = DockerIsolatedEnv(
        task_config=TaskConfig(
            task_name="dind-smoke",
            task_path=str(task_root),
            instruction="Verify the TerminalBench Docker runtime.",
            test_weights={"test_holder_runtime": 1.0},
            dockerfile_contents=DOCKERFILE,
            py_test_file_contents=TEST_FILE,
            max_test_timeout_sec=120.0,
        ),
        container_config=ContainerConfig(no_rebuild=False, timeout=120),
        env_id="dind-smoke",
    )

    try:
        observation, _ = env.reset()
        assert observation["status"] == "success"
        trial = env.trial_handler
        terminal = env.terminal
        assert trial is not None and terminal is not None
        trial.trial_paths.post_agent_pane_path.write_text("", encoding="utf-8")
        score = asyncio.run(
            calculate_test_score(
                terminal=terminal,
                trial_handler=trial,
                task_name="dind-smoke",
                test_weights={"test_holder_runtime": 1.0},
                max_test_timeout_sec=120.0,
                rollout_id="dind-smoke",
            )
        )
        assert score == 1.0, score
        print("TerminalBench Docker-in-Docker smoke passed: reward=1.0")
    finally:
        env.close()


if __name__ == "__main__":
    main()
