#!/usr/bin/env python3
"""Materialize Terminal-Bench-RL tasks and verl parquet files deterministically."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import stat
import uuid
from pathlib import Path

import pandas as pd
import yaml

DOCKER_COMPOSE = """services:
  client:
    build:
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: [\"sh\", \"-c\", \"sleep infinity\"]
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
"""

RUN_TESTS = """#!/bin/bash
set -e
source \"$TEST_DIR/setup-uv-pytest.sh\"
bash \"$TEST_DIR/run-uv-pytest.sh\"
"""

SETUP_TESTS = """#!/bin/bash
set -e
if ! command -v curl >/dev/null 2>&1; then
  apt-get update
  apt-get install -y curl
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source \"$HOME/.local/bin/env\"
fi
if [ ! -f pyproject.toml ]; then uv init --no-readme; fi
uv add pytest >/dev/null
"""

RUN_PYTEST = """#!/bin/bash
set -e
uv run pytest \"$TEST_DIR/test_outputs.py\" -rA
"""

SYSTEM_PROMPT = (
    Path(__file__).resolve().parents[1] / "src" / "agent_core" / "system_prompt.md"
).read_text(encoding="utf-8").strip()


def _write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _json_field(row: dict[str, str], key: str, default):
    value = row.get(key, "").strip()
    return json.loads(value) if value else default


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe additional-file path: {value!r}")
    return path


def _materialize_task(
    row: dict[str, str], task_root: Path, published_task_root: Path, index: int
) -> dict:
    task_id = row["task_id"].strip()
    task_dir = task_root / task_id
    task_dir.mkdir(parents=True, exist_ok=False)

    difficulty = row.get("difficulty", "hard").strip().replace("extremely_hard", "hard")
    tags = [item.strip() for item in row.get("tags", "").split("|") if item.strip()]
    task_yaml = {
        "instruction": row["prompt"],
        "author_name": "Terminal-Bench-RL",
        "author_email": "unknown",
        "difficulty": difficulty,
        "category": row.get("category", "terminal"),
        "tags": tags,
        "parser_name": "pytest",
        "max_agent_timeout_sec": 5400.0,
        "max_test_timeout_sec": 900.0,
        "run_tests_in_same_shell": False,
    }

    _write(task_dir / "task.yaml", yaml.safe_dump(task_yaml, sort_keys=False))
    _write(task_dir / "Dockerfile", row["dockerfile"].rstrip() + "\n")
    _write(task_dir / "docker-compose.yaml", DOCKER_COMPOSE)
    _write(task_dir / "run-tests.sh", RUN_TESTS, executable=True)
    _write(task_dir / "solution.sh", "#!/bin/bash\n# Intentionally empty during training.\n", executable=True)
    _write(task_dir / "tests" / "setup-uv-pytest.sh", SETUP_TESTS, executable=True)
    _write(task_dir / "tests" / "run-uv-pytest.sh", RUN_PYTEST, executable=True)
    _write(task_dir / "tests" / "test_outputs.py", row["test_functions"])

    test_weights = _json_field(row, "test_weights", {})
    _write(task_dir / "test_weights.json", json.dumps(test_weights, indent=2, sort_keys=True) + "\n")
    for name, content in _json_field(row, "additional_files", {}).items():
        _write(task_dir / _safe_relative(name), str(content))

    published_task_dir = published_task_root / task_id
    extra_info = {
        "index": index,
        "task_name": task_id,
        "task_path": str(published_task_dir),
        "instruction": row["prompt"],
        "test_weights": json.dumps(test_weights, sort_keys=True),
        "dockerfile_contents": row["dockerfile"],
        "py_test_file_contents": row["test_functions"],
        "max_test_timeout_sec": 900.0,
    }
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ],
        "task_name": task_id,
        "task_path": str(published_task_dir),
        "instruction": row["prompt"],
        "data_source": "terminal_bench",
        "extra_info": extra_info,
    }


def prepare(csv_path: Path, output_root: Path, val_count: int, seed: int) -> None:
    marker = output_root / ".complete.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        print(f"TerminalBench data ready: {metadata}")
        return
    if output_root.exists():
        raise RuntimeError(f"refusing incomplete existing output: {output_root}")

    pending = output_root.with_name(f".{output_root.name}.pending-{uuid.uuid4().hex[:8]}")
    task_root = pending / "tasks"
    task_root.mkdir(parents=True)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"task_id", "prompt", "dockerfile", "test_functions", "test_weights"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError("TerminalBench CSV is empty or lacks required columns")

        rng = random.Random(seed)
        rng.shuffle(rows)
        if val_count <= 0 or val_count >= len(rows):
            raise ValueError(f"val_count must be in [1, {len(rows) - 1}]")

        published_task_root = output_root / "tasks"
        records = [
            _materialize_task(row, task_root, published_task_root, index)
            for index, row in enumerate(rows)
        ]
        val_records = records[:val_count]
        train_records = records[val_count:]
        pd.DataFrame(train_records).to_parquet(pending / "train.parquet", index=False)
        pd.DataFrame(val_records).to_parquet(pending / "val.parquet", index=False)

        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        metadata = {
            "source_sha256": digest,
            "train_examples": len(train_records),
            "val_examples": len(val_records),
            "seed": seed,
        }
        _write(pending / ".complete.json", json.dumps(metadata, sort_keys=True) + "\n")
        pending.rename(output_root)
        print(f"TerminalBench data prepared: {metadata}")
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prepare(args.csv.resolve(), args.output_root.resolve(), args.val_count, args.seed)


if __name__ == "__main__":
    main()
