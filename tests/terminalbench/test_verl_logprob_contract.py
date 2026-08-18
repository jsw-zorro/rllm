import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout import verl_engine as verl_engine_module
from rllm.engine.rollout.verl_engine import VerlEngine
from verl.workers.rollout.replica import TokenOutput


class _Tokenizer:
    def encode(self, prompt, add_special_tokens=False):
        return [11, 12]

    def decode(self, token_ids, skip_special_tokens=True):
        return "completion"


class _Parser:
    def parse(self, messages, **kwargs):
        return "prompt"

    def parse_completion(self, token_ids):
        return {"content": "completion", "reasoning": None, "tool_calls": []}


class _ServerManager:
    def __init__(self, output):
        self.output = output
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def _config():
    return OmegaConf.create(
        {
            "data": {"max_prompt_length": 4096, "max_response_length": 8192},
            "rllm": {"disable_thinking": False},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "sglang",
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_k": -1,
                    "top_p": 0.95,
                    "calculate_log_probs": True,
                    "max_model_len": 40960,
                    "val_kwargs": {
                        "do_sample": False,
                        "temperature": 0.0,
                        "top_k": -1,
                        "top_p": 1.0,
                    },
                }
            },
        }
    )


def _engine(monkeypatch, output):
    parser = _Parser()
    monkeypatch.setattr(
        verl_engine_module.ChatTemplateParser,
        "get_parser",
        lambda *args, **kwargs: parser,
    )
    engine = VerlEngine(
        config=_config(),
        rollout_manager=SimpleNamespace(server_handles=[]),
        tokenizer=_Tokenizer(),
    )
    engine.server_manager = _ServerManager(output)
    return engine


def test_verl_engine_requests_and_preserves_real_logprobs(monkeypatch):
    engine = _engine(
        monkeypatch,
        TokenOutput(token_ids=[21, 22], log_probs=[-0.25, -1.5]),
    )

    output = asyncio.run(
        engine.get_model_response(
            [], application_id="request-1", max_tokens=7, max_new_tokens=11
        )
    )

    assert output.logprobs == [-0.25, -1.5]
    assert engine.server_manager.calls[0]["sampling_params"]["logprobs"] is True
    assert engine.server_manager.calls[0]["sampling_params"]["max_new_tokens"] == 7
    assert engine.train_sampling_params["logprobs"] is True


def test_verl_engine_rejects_missing_requested_logprobs(monkeypatch):
    engine = _engine(
        monkeypatch,
        TokenOutput(token_ids=[21, 22], log_probs=None),
    )

    with pytest.raises(RuntimeError, match="SGLang returned none"):
        asyncio.run(engine.get_model_response([], application_id="request-2"))


def test_trajectory_assembly_preserves_real_logprobs_and_rejects_zero_vector():
    execution = AgentExecutionEngine.__new__(AgentExecutionEngine)
    execution.rollout_engine = SimpleNamespace(calculate_log_probs=True)
    execution.config = OmegaConf.create(
        {"rllm": {"filter_token_mismatch": True}}
    )
    step = {
        "prompt_ids": [11, 12],
        "completion_ids": [21, 22],
        "logprobs": [-0.25, -1.5],
    }

    _, _, _, logprobs, valid = execution.assemble_steps([step])

    assert valid is True
    assert torch.equal(logprobs, torch.tensor([-0.25, -1.5]))

    step["logprobs"] = [0.0, 0.0]
    with pytest.raises(RuntimeError, match="identically-zero"):
        execution.assemble_steps([step])


@pytest.mark.parametrize(
    ("logprobs", "message"),
    [
        ([float("nan"), -1.0], "finite"),
        ([0.1, -1.0], "non-positive"),
        ([-1.0], "lengths disagree"),
    ],
)
def test_trajectory_assembly_rejects_invalid_logprob_vectors(logprobs, message):
    execution = AgentExecutionEngine.__new__(AgentExecutionEngine)
    execution.rollout_engine = SimpleNamespace(calculate_log_probs=True)
    execution.config = OmegaConf.create(
        {"rllm": {"filter_token_mismatch": True}}
    )
    step = {
        "prompt_ids": [11, 12],
        "completion_ids": [21, 22],
        "logprobs": logprobs,
    }

    with pytest.raises(RuntimeError, match=message):
        execution.assemble_steps([step])


def test_verl_engine_rejects_nonpositive_model_budget(monkeypatch):
    engine = _engine(
        monkeypatch,
        TokenOutput(token_ids=[21], log_probs=[-0.25]),
    )
    engine.max_model_len = 2

    with pytest.raises(RuntimeError, match="no positive model response budget"):
        asyncio.run(engine.get_model_response([], application_id="request-3"))


def test_timeline_config_is_optional():
    from rllm.trainer.verl.train_agent_ppo import _timeline_json_file

    assert _timeline_json_file(OmegaConf.create({"trainer": {}})) is None
    assert (
        _timeline_json_file(
            OmegaConf.create({"ray_init": {"timeline_json_file": "trace.json"}})
        )
        == "trace.json"
    )
