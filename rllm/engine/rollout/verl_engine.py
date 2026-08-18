import asyncio
import uuid

import ray
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=rollout_manager.server_handles)
        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.max_model_len = int(config.actor_rollout_ref.rollout.max_model_len)
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)
        self.calculate_log_probs = bool(
            config.actor_rollout_ref.rollout.calculate_log_probs
        )

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
            logprobs=self.calculate_log_probs,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
            logprobs=self.calculate_log_probs,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        prompt_ids_override = kwargs.pop("prompt_ids_override", None)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = sampling_params.pop(
                "max_new_tokens", self.max_response_length
            )
        else:
            # The OpenAI-style name takes precedence when both are supplied.
            sampling_params.pop("max_new_tokens", None)
        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        canonical_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        request_prompt_ids = (
            list(prompt_ids_override)
            if prompt_ids_override is not None
            else canonical_prompt_ids
        )
        available_model_tokens = self.max_model_len - len(request_prompt_ids) - 1
        max_tokens = min(
            int(max_tokens), self.max_response_length, available_model_tokens
        )
        if max_tokens <= 0:
            raise RuntimeError(
                "no positive model response budget remains for this prompt"
            )
        sampling_params["max_new_tokens"] = max_tokens

        if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:
            image_data = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
            model_inputs = self.processor(text=[prompt], images=image_data)
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)
        else:
            image_data = None
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        token_output: TokenOutput = await self.server_manager.generate(request_id=application_id, prompt_ids=request_prompt_ids, image_data=image_data, sampling_params=sampling_params)  # type: ignore
        completion_ids: list[int] = token_output.token_ids
        completion_logprobs = token_output.log_probs
        if self.calculate_log_probs and completion_logprobs is None:
            raise RuntimeError(
                "rollout requested logprobs but SGLang returned none"
            )

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]
            if completion_logprobs is not None:
                completion_logprobs = completion_logprobs[:max_tokens]

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=completion_logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])

    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])

    async def begin_parity_evidence(self, request_count: int, response_steps: int, validate: bool) -> bool:
        """Arm exact sparse-selection capture for one multi-turn rollout batch."""
        from verl.block_sparse_attention.parity_production import (
            parity_replay_requested,
            selection_shard_export_enabled,
        )

        active = parity_replay_requested() and not validate
        if not active:
            return False
        if selection_shard_export_enabled():
            raise RuntimeError("rLLM async parity requires PARITY_SELECTION_SHARDS=0")
        max_snapshots = max(1, int(request_count) * int(response_steps) * 64)
        await asyncio.to_thread(
            ray.get,
            [
                server.begin_parity_evidence.remote(max_snapshots)
                for server in self.rollout_manager.server_handles
            ],
        )
        return True

    async def finish_parity_evidence(self, active: bool):
        if not active:
            return None
        return await asyncio.to_thread(
            ray.get,
            [
                server.finish_parity_evidence.remote()
                for server in self.rollout_manager.server_handles
            ],
        )
