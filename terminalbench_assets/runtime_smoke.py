#!/usr/bin/env python3
"""Verify the staged rLLM/verl/TerminalBench runtime before GPU launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-root", type=Path, required=True)
    parser.add_argument("--rllm-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()

    sys.path[:0] = [
        str(args.asr_root.resolve()),
        str((args.rllm_root / "vendor").resolve()),
        str(args.rllm_root.resolve()),
    ]

    from verl.block_sparse_attention.parity_production import (
        attach_async_selection_payloads,
        parity_replay_requested,
    )
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

    import verl
    from rllm.engine.rollout.verl_engine import VerlEngine
    from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
    from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer

    assert verl.__version__.startswith("0.5"), verl.__version__
    assert "terminal_bench" in ENV_CLASS_MAPPING
    assert "terminal_bench_agent" in AGENT_CLASS_MAPPING
    assert AgentPPOTrainer is not None
    assert VerlEngine is not None
    assert AsyncActorRolloutRefWorker is not None
    assert attach_async_selection_payloads is not None
    assert callable(parity_replay_requested)

    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    assert model_config["max_position_embeddings"] == 40960
    print(
        "runtime smoke passed:",
        {
            "verl": verl.__version__,
            "max_context": model_config["max_position_embeddings"],
            "terminal_env": ENV_CLASS_MAPPING["terminal_bench"].__name__,
            "terminal_agent": AGENT_CLASS_MAPPING["terminal_bench_agent"].__name__,
        },
    )


if __name__ == "__main__":
    main()
