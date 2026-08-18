import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_max_context_and_matched_batch_contract():
    script = (ROOT / "scripts/terminalbench/run_terminalbench_obx.sh").read_text()
    assert "TERMINALBENCH_MAX_PROMPT_LENGTH:-8192" in script
    assert "TERMINALBENCH_MAX_RESPONSE_LENGTH:-32767" in script
    assert "TERMINALBENCH_MAX_MODEL_LEN:-40960" in script
    assert '"data.max_prompt_length=${MAX_PROMPT_LENGTH}"' in script
    assert '"data.max_response_length=${MAX_RESPONSE_LENGTH}"' in script
    assert '"actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"' in script
    assert "TERMINALBENCH_TRAIN_BATCH_SIZE:-4" in script
    assert "TERMINALBENCH_VAL_BATCH_SIZE:-8" in script
    assert '"data.train_batch_size=${TRAIN_BATCH_SIZE}"' in script
    assert '"data.val_batch_size=${VAL_BATCH_SIZE}"' in script
    assert 'TERMINALBENCH_VAL_BEFORE_TRAIN:-True' in script
    assert 'TERMINALBENCH_AGENT_MAX_STEPS:-50' in script
    assert 'TERMINALBENCH_TRAJECTORY_TIMEOUT:-5400' in script
    assert 'TERMINALBENCH_PARITY_MULTI_TURN:-1' in script
    assert 'unset PARITY_MULTI_TURN' in script
    assert 'PARITY_ATTN_FAMILY:-sp_fp32p_split_kv_n8_w8s1' in script
    assert "TERMINALBENCH_ROLLOUT_N:-8" in script
    assert '"actor_rollout_ref.rollout.n=${ROLLOUT_N}"' in script
    assert '"trainer.nnodes=${NUM_NODES}"' in script
    assert "TERMINALBENCH_TOTAL_TRAINING_STEPS:-null" in script
    assert '"trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"' in script
    assert "TerminalBench exact parity forbids rollout-logprob reuse" in script
    assert 'mkdir -p "$(dirname "${DEPS_MARKER}")"' in script
    assert 'TIR_MODEL_STORE="${MODEL_STORE}"' in script
    assert 'pushd "${ASR_ROOT}"' in script
    assert '${ASR_ROOT}/sglang/python' in script
    assert '${ASR_ROOT}/vortex_torch' in script
    assert 'PARITY_TMPDIR="${PARITY_DISK_SCRATCH}"' in script
    assert 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC' in script


def test_sparse_bridge_is_fail_closed_and_uses_stable_request_ids():
    engine = (ROOT / "rllm/engine/agent_execution_engine.py").read_text()
    rollout = (ROOT / "rllm/engine/rollout/verl_engine.py").read_text()
    trainer = (ROOT / "rllm/trainer/verl/agent_ppo_trainer.py").read_text()
    entrypoint = (ROOT / "rllm/trainer/verl/train_agent_ppo.py").read_text()
    assert '"parity_request_id": application_id' in engine
    assert '"response_logprobs": response_logprobs' in engine
    assert '"completion_ids": model_output.completion_ids' in engine
    assert '"response_ids": model_output.completion_ids' not in engine
    assert 'kwargs["prompt_ids_override"] = next_prompt_ids' in engine
    assert "rollout producer promised token logprobs but returned none" in engine
    assert "rollout returned an identically-zero token logprob vector" in engine
    assert 'list(model_output.prompt_ids)' in engine
    assert 'assistant_msg_tokens = list(model_output.completion_ids)' in engine
    assert 'prompt_ids_override = kwargs.pop("prompt_ids_override", None)' in rollout
    assert "logprobs=self.calculate_log_probs" in rollout
    assert 'sampling_params["max_new_tokens"] = max_tokens' in rollout
    assert "rollout requested logprobs but SGLang returned none" in rollout
    assert "begin_parity_evidence.remote" in rollout
    assert "finish_parity_evidence.remote" in rollout
    assert "attach_async_selection_payloads" in trainer
    assert "assert_exact_rollout_actor_logprob_parity" in trainer
    assert "strict sparse rLLM rollout did not provide rollout_log_probs" in trainer
    assert 'config.get("ray_init")' in entrypoint


def test_holder_starts_docker_and_keeps_sleep_infinity():
    holder = (ROOT / "scripts/terminalbench/dind_holder_entrypoint.sh").read_text()
    assert "dockerd" in holder
    assert "docker compose version" in holder
    assert 'subprocess.Popen(["sleep", "infinity"])' in holder
    assert "os.wait()" in holder


def test_detached_starter_is_hash_guarded_and_idempotent():
    starter = (
        ROOT / "scripts/terminalbench/start_terminalbench_obx_once.sh"
    ).read_text()
    assert "TERMINALBENCH_RLLM_SHA256" in starter
    assert "TERMINALBENCH_ASR_SHA256" in starter
    assert 'if ! mkdir "${LOCK_DIR}"' in starter
    assert "docker info" in starter
    assert 'nohup setsid bash "${WRAPPER}"' in starter
    assert "BOOT_STARTED" in starter
    assert "TRAINER_EXITED" in starter


def test_dataset_materialization(tmp_path):
    module_path = ROOT / "terminalbench_assets/prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_terminalbench", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    csv_path = tmp_path / "tasks.csv"
    fields = [
        "task_id",
        "difficulty",
        "category",
        "tags",
        "prompt",
        "dockerfile",
        "test_functions",
        "test_weights",
        "additional_files",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "task_id": f"task-{index}",
                    "difficulty": "easy",
                    "category": "files",
                    "tags": "python|files",
                    "prompt": f"Create output {index}",
                    "dockerfile": "FROM python:3.11-slim\nWORKDIR /app",
                    "test_functions": "def test_ok():\n    assert True\n",
                    "test_weights": json.dumps({"test_ok": 1.0}),
                    "additional_files": json.dumps({"seed.txt": str(index)}),
                }
            )

    output = tmp_path / "prepared"
    module.prepare(csv_path, output, val_count=1, seed=7)
    metadata = json.loads((output / ".complete.json").read_text())
    assert metadata["train_examples"] == 1
    assert metadata["val_examples"] == 1
    assert (output / "train.parquet").is_file()
    assert (output / "val.parquet").is_file()
    assert len(list((output / "tasks").iterdir())) == 2
    records = pd.concat(
        [
            pd.read_parquet(output / "train.parquet"),
            pd.read_parquet(output / "val.parquet"),
        ]
    )
    indices = []
    for record in records.to_dict(orient="records"):
        assert Path(record["task_path"]).is_dir()
        extra_info = record["extra_info"]
        indices.append(extra_info["index"])
        assert Path(extra_info["task_path"]).is_dir()
        assert isinstance(json.loads(extra_info["test_weights"]), dict)
        assert [message["role"] for message in record["prompt"]] == ["system", "user"]
        assert record["prompt"][0]["content"] == (
            ROOT / "src/agent_core/system_prompt.md"
        ).read_text(encoding="utf-8").strip()
    assert sorted(indices) == list(range(len(records)))
