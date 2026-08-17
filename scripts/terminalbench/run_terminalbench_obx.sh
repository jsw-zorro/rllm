#!/usr/bin/env bash
set -euo pipefail

ARM="${TERMINALBENCH_ATTENTION_MODE:?set dense or sparse}"
case "${ARM}" in dense|sparse) ;; *) echo "unsupported arm: ${ARM}" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASR_ROOT="${ASR_ROOT:?set ASR_ROOT to the AdaptiveSparseRoll overlay}"
RUN_ID="${TERMINALBENCH_RUN_ID:?set TERMINALBENCH_RUN_ID}"
JOB_NAME="${TERMINALBENCH_JOB_NAME:?set TERMINALBENCH_JOB_NAME}"
NODE_RANK="${NODE_RANK:-${HOSTNAME##*-}}"
NUM_NODES="${NUM_NODES:-4}"
RUN_ROOT="${TERMINALBENCH_RUN_ROOT:-/shared/dev/shuowei/terminalbench/runs/${RUN_ID}}"
LOCAL_ROOT="${TERMINALBENCH_LOCAL_ROOT:-/tmp/instance_storage/terminalbench-runs/${RUN_ID}}"
DATA_ROOT="${TERMINALBENCH_DATA_ROOT:-/shared/dev/shuowei/terminalbench/data/v1}"
MODEL_SOURCE="${TERMINALBENCH_MODEL_SOURCE:-/shared/models/Qwen3-4B}"
MODEL_PATH="${TERMINALBENCH_MODEL_LOCAL_PATH:-/tmp/instance_storage/models/Qwen3-4B}"
MODEL_STORE="${TERMINALBENCH_MODEL_STORE:-/mnt/nvme/terminalbench-model-objects}"

mkdir -p "${RUN_ROOT}/checkpoints" "${LOCAL_ROOT}/scratch" "${LOCAL_ROOT}/home"
export HOME="${LOCAL_ROOT}/home"
export HF_HOME="${HOME}/hf"
export XDG_CACHE_HOME="${HOME}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${HOME}/torchinductor"
export TRITON_CACHE_DIR="${HOME}/triton"
export PYTHONPATH="${ASR_ROOT}:${ROOT}/vendor:${ROOT}:${PYTHONPATH:-}"
export PATH="/opt/amazon/efa/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ulimit -l unlimited
ulimit -n 1048576

# Install only the runtime dependencies not baked into the training image.
DEPS_MARKER=/tmp/instance_storage/terminalbench-python-deps.complete
mkdir -p "$(dirname "${DEPS_MARKER}")"
if [[ ! -f "${DEPS_MARKER}" ]]; then
    python3 -m pip install --no-cache-dir \
        'docker==7.1.0' 'ruamel.yaml==0.18.10' 'polars==1.32.3'
    touch "${DEPS_MARKER}"
fi

# Stage the model once per physical node.
TIR_MODEL_NAME=Qwen3-4B \
TIR_MODEL_SOURCE="${MODEL_SOURCE}" \
TIR_MODEL_LOCAL_PATH="${MODEL_PATH}" \
TIR_MODEL_STORE="${MODEL_STORE}" \
    bash "${ASR_ROOT}/scripts_gl/stage_tir_model_local.sh"

# Both experiments consume the same deterministic task materialization.
DATA_MARKER="${DATA_ROOT}/.complete.json"
DATA_LOCK="${DATA_ROOT}.prepare.lock"
if [[ ! -f "${DATA_MARKER}" ]]; then
    mkdir -p "$(dirname "${DATA_ROOT}")"
    if mkdir "${DATA_LOCK}" 2>/dev/null; then
        trap 'rmdir "${DATA_LOCK}" 2>/dev/null || true' EXIT
        python3 "${ROOT}/terminalbench_assets/prepare_dataset.py" \
            --csv "${ROOT}/terminalbench_assets/latest_verified.csv" \
            --output-root "${DATA_ROOT}" --val-count 8 --seed 7
        rmdir "${DATA_LOCK}"
        trap - EXIT
    else
        for _ in $(seq 1 900); do
            [[ -f "${DATA_MARKER}" ]] && break
            sleep 2
        done
    fi
fi
[[ -f "${DATA_MARKER}" ]] || { echo "TerminalBench data preparation did not complete" >&2; exit 2; }

export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export NCCL_DEBUG=WARN
export N_GPUS_PER_NODE=8
export MASTER_ADDR="${MASTER_ADDR:-${JOB_NAME}-worker-0.${JOB_NAME}}"
export RAY_HEAD_PORT="${RAY_HEAD_PORT:-6387}"
export RAY_ADDRESS="${MASTER_ADDR}:${RAY_HEAD_PORT}"
export TMPDIR="/dev/shm/terminalbench_${NODE_RANK}"
export RAY_TMPDIR="${TMPDIR}/ray"
export RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"${LOCAL_ROOT}/scratch/ray_spill\"}}"
mkdir -p "${TMPDIR}" "${LOCAL_ROOT}/scratch/ray_spill"

export WANDB_MODE=online
export WANDB_ENTITY=niletron
export WANDB_TEAM=niletron
export WANDB_PROJECT="${WANDB_PROJECT:-TerminalBench-Qwen3-4B-RL}"
export WANDB_RUN_ID="${WANDB_RUN_ID:?set WANDB_RUN_ID}"
export WANDB_RESUME=never
export WANDB_DIR="${LOCAL_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

PARITY_ARGS=()
if [[ "${ARM}" == sparse ]]; then
    export MODEL_PATH
    export ROLLOUT_N=8
    export MAX_PROMPT_BS64=8192
    export MAX_RESP_BS64=32767
    export PPO_TOK_BS64=40960
    export LOGP_TOK_BS64=40960
    export MNBT_BS64=40960
    export TRAIN_BSZ_BS64=4
    export MINI_BSZ_BS64=4
    export MAX_NUM_SEQS=64
    export VORTEX_POLICY=qwen3-4b-fixed128@bs64
    export PARITY_FULL_BS64=1
    export PARITY_MULTI_TURN=1
    export PARITY_SELECTION_SHARDS=0
    export PARITY_CAPTURE_LOGICAL_RING=1
    export PARITY_ASYNC_RAY_ROW_REFS=1
    export PARITY_REUSE_ROLLOUT_LOGPROBS=0
    export PARITY_ATTN_FAMILY=sp_fp32p_split_kv_n8_w8s1
    export PARITY_FAST_BACKWARD_GEMM=1
    export PARITY_CONFIG_ONLY=1
    # shellcheck disable=SC1091
    pushd "${ASR_ROOT}" >/dev/null
    source "${ASR_ROOT}/scripts_gl/qwen1.7b_sparse_train_n8_h200_parity_gl.sh"
    popd >/dev/null
    unset PARITY_CONFIG_ONLY
    PARITY_ARGS=("${PARITY_EXTRA_HYDRA[@]}")
fi

COMMON_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=False"
    "algorithm.use_kl_in_reward=False"
    "data.train_files=${DATA_ROOT}/train.parquet"
    "data.val_files=${DATA_ROOT}/val.parquet"
    "data.train_batch_size=4"
    "data.val_batch_size=1"
    "data.max_prompt_length=8192"
    "data.max_response_length=32767"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_multi_modal_inputs=False"
    "rllm.agent.name=terminal_bench_agent"
    "rllm.agent.max_steps=50"
    "rllm.agent.trajectory_timeout=5400"
    "rllm.agent.overlong_filter=False"
    "rllm.env.name=terminal_bench"
    "+rllm.env.env_args.no_rebuild=False"
    "+rllm.env.env_args.timeout=5400"
    "rllm.stepwise_advantage.enable=False"
    "rllm.rejection_sample.enable=False"
    "rllm.filter_token_mismatch=True"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_shm=False"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.ppo_mini_batch_size=4"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.name=sglang"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
    "actor_rollout_ref.rollout.max_num_seqs=64"
    "actor_rollout_ref.rollout.n=8"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=0.95"
    "actor_rollout_ref.rollout.max_model_len=40960"
    "actor_rollout_ref.rollout.max_num_batched_tokens=40960"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.calculate_log_probs=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.agent.num_workers=32"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.fsdp_config.param_offload=False"
    "trainer.logger=[console,wandb]"
    "trainer.project_name=${WANDB_PROJECT}"
    "trainer.experiment_name=${TERMINALBENCH_EXPERIMENT_NAME:-${RUN_ID}}"
    "trainer.default_local_dir=${RUN_ROOT}/checkpoints"
    "trainer.n_gpus_per_node=8"
    "trainer.nnodes=${NUM_NODES}"
    "trainer.val_before_train=True"
    "trainer.test_freq=20"
    "trainer.save_freq=20"
    "trainer.total_epochs=1"
    "trainer.max_actor_ckpt_to_keep=2"
    "trainer.max_critic_ckpt_to_keep=0"
)

if [[ "${ARM}" == dense ]]; then
    ATTENTION_ARGS=(
        "actor_rollout_ref.rollout.sparse_rollout=False"
        "actor_rollout_ref.actor.use_sparse_training=False"
        "actor_rollout_ref.ref.use_sparse_training=False"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_vortex_sparsity=False"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False"
    )
else
    COUNTS="{counts: 0, count_kind: schedule_policy, policy_name: ${VORTEX_POLICY}, dense_prompt_sparse_response: false, collect_evidence: false}"
    ATTENTION_ARGS=(
        "actor_rollout_ref.rollout.sparse_rollout=True"
        "actor_rollout_ref.actor.use_sparse_training=True"
        "actor_rollout_ref.ref.use_sparse_training=True"
        "actor_rollout_ref.actor.bsa_kwargs_nsa_block_size=64"
        "actor_rollout_ref.ref.bsa_kwargs_nsa_block_size=64"
        "actor_rollout_ref.actor.bsa_kwargs_nsa_block_counts=${COUNTS}"
        "actor_rollout_ref.ref.bsa_kwargs_nsa_block_counts=${COUNTS}"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_module_name=block_sparse_attention"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_topk_val=45"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.page_size=64"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_block_size=64"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_vortex_sparsity=True"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_block_reserved_bos=1"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_block_reserved_eos=2"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_layers_skip=[0,1]"
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_schedule_policy=${VORTEX_POLICY}"
    )
fi

ray stop --force >/dev/null 2>&1 || true
sleep 3
if [[ "${NODE_RANK}" == 0 ]]; then
    cleanup_ray() { ray stop --force >/dev/null 2>&1 || true; }
    trap cleanup_ray EXIT
    ray start --head --port="${RAY_HEAD_PORT}" \
        --num-gpus=8 --dashboard-host=0.0.0.0 --disable-usage-stats
    export RAY_ADDRESS="127.0.0.1:${RAY_HEAD_PORT}"
    expected=$((NUM_NODES * 8))
    for _ in $(seq 1 180); do
        actual=$(python3 -c "import ray; ray.init(address='auto'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
        echo "registered GPUs: ${actual}/${expected}"
        [[ "${actual}" -ge "${expected}" ]] && break
        sleep 10
    done
    [[ "${actual}" -ge "${expected}" ]] || { echo "Ray gang did not register" >&2; exit 2; }
    cd "${ROOT}"
    python3 -u -m rllm.trainer.verl.train_agent_ppo \
        "${PARITY_ARGS[@]}" "${COMMON_ARGS[@]}" "${ATTENTION_ARGS[@]}"
    trap - EXIT
    cleanup_ray
else
    for _ in $(seq 1 180); do
        python3 -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('${MASTER_ADDR}', ${RAY_HEAD_PORT})); s.close()" 2>/dev/null && break
        sleep 10
    done
    ray start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" --num-gpus=8 --disable-usage-stats
    while python3 -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('${MASTER_ADDR}', ${RAY_HEAD_PORT})); s.close()" 2>/dev/null; do
        sleep 30
    done
    ray stop --force >/dev/null 2>&1 || true
fi
