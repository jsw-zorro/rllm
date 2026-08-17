#!/usr/bin/env bash
set -Eeuo pipefail

ARM="${TERMINALBENCH_ATTENTION_MODE:?set dense or sparse}"
case "${ARM}" in dense|sparse) ;; *) echo "unsupported arm: ${ARM}" >&2; exit 2 ;; esac

RUN_ID="${TERMINALBENCH_RUN_ID:?set TERMINALBENCH_RUN_ID}"
SOURCE_ROOT="${TERMINALBENCH_SOURCE_ROOT:?set TERMINALBENCH_SOURCE_ROOT}"
CLOSURE_ID="${TERMINALBENCH_CLOSURE_ID:?set TERMINALBENCH_CLOSURE_ID}"
RLLM_ARCHIVE="${SOURCE_ROOT}/${TERMINALBENCH_RLLM_ARCHIVE:?set TERMINALBENCH_RLLM_ARCHIVE}"
ASR_ARCHIVE="${SOURCE_ROOT}/${TERMINALBENCH_ASR_ARCHIVE:?set TERMINALBENCH_ASR_ARCHIVE}"
RLLM_SHA256="${TERMINALBENCH_RLLM_SHA256:?set TERMINALBENCH_RLLM_SHA256}"
ASR_SHA256="${TERMINALBENCH_ASR_SHA256:?set TERMINALBENCH_ASR_SHA256}"
RUN_ROOT="${TERMINALBENCH_RUN_ROOT:-/shared/dev/shuowei/terminalbench/runs/${RUN_ID}}"
LOCAL_CODE_ROOT="${TERMINALBENCH_LOCAL_CODE_ROOT:-/tmp/instance_storage/terminalbench-closures/${CLOSURE_ID}}"
LOCK_ROOT="${TERMINALBENCH_LOCK_ROOT:-/tmp/instance_storage/terminalbench-start-locks}"
LOCK_DIR="${LOCK_ROOT}/${RUN_ID}"
HOST_ID="${HOSTNAME:?HOSTNAME is unset}"
RECEIPT="${RUN_ROOT}/boot/${HOST_ID}.receipt"
LOG_FILE="${RUN_ROOT}/logs/${HOST_ID}.log"

: "${TERMINALBENCH_JOB_NAME:?set TERMINALBENCH_JOB_NAME}"
: "${WANDB_API_KEY:?WANDB_API_KEY is unset}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is unset}"

mkdir -p "${RUN_ROOT}/boot" "${RUN_ROOT}/logs" "${LOCK_ROOT}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "TerminalBench starter already claimed for ${RUN_ID} on ${HOST_ID}"
    exit 0
fi

write_receipt() {
    local state=$1
    local detail=$2
    local tmp="${RECEIPT}.tmp.$$"
    {
        printf 'state=%s\n' "${state}"
        printf 'detail=%s\n' "${detail}"
        printf 'run_id=%s\n' "${RUN_ID}"
        printf 'attention_mode=%s\n' "${ARM}"
        printf 'host=%s\n' "${HOST_ID}"
        printf 'closure_id=%s\n' "${CLOSURE_ID}"
        printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"${tmp}"
    mv "${tmp}" "${RECEIPT}"
}

on_error() {
    local status=$?
    write_receipt BOOT_FAILED "starter_exit_${status}"
    exit "${status}"
}
trap on_error ERR

verify_archive() {
    local archive=$1
    local expected=$2
    local actual
    actual=$(sha256sum "${archive}")
    actual=${actual%% *}
    [[ "${actual}" == "${expected}" ]]
}

verify_archive "${RLLM_ARCHIVE}" "${RLLM_SHA256}"
verify_archive "${ASR_ARCHIVE}" "${ASR_SHA256}"
docker info >/dev/null

if [[ ! -f "${LOCAL_CODE_ROOT}/.complete" ]]; then
    staging="${LOCAL_CODE_ROOT}.tmp.$$"
    rm -rf -- "${staging}"
    mkdir -p "${staging}/rllm" "${staging}/adaptive-sparse-roll"
    tar -xzf "${RLLM_ARCHIVE}" -C "${staging}/rllm"
    tar -xzf "${ASR_ARCHIVE}" -C "${staging}/adaptive-sparse-roll"
    printf '%s\n' "${CLOSURE_ID}" >"${staging}/.complete"
    if [[ -e "${LOCAL_CODE_ROOT}" ]]; then
        rm -rf -- "${LOCAL_CODE_ROOT}"
    fi
    mkdir -p "$(dirname "${LOCAL_CODE_ROOT}")"
    mv "${staging}" "${LOCAL_CODE_ROOT}"
fi

RLLM_ROOT="${LOCAL_CODE_ROOT}/rllm"
ASR_ROOT="${LOCAL_CODE_ROOT}/adaptive-sparse-roll"
RUN_SCRIPT="${RLLM_ROOT}/scripts/terminalbench/run_terminalbench_obx.sh"
[[ -f "${RUN_SCRIPT}" ]]
[[ -f "${ASR_ROOT}/scripts_gl/qwen1.7b_sparse_train_n8_h200_parity_gl.sh" ]]

WRAPPER="${LOCK_DIR}/run-wrapper.sh"
cat >"${WRAPPER}" <<'EOF'
#!/usr/bin/env bash
set +e
bash "${RLLM_ROOT}/scripts/terminalbench/run_terminalbench_obx.sh"
status=$?
tmp="${TERMINALBENCH_RECEIPT}.tmp.${BASHPID}"
{
    printf 'state=TRAINER_EXITED\n'
    printf 'detail=trainer_exit_%s\n' "${status}"
    printf 'run_id=%s\n' "${TERMINALBENCH_RUN_ID}"
    printf 'attention_mode=%s\n' "${TERMINALBENCH_ATTENTION_MODE}"
    printf 'host=%s\n' "${HOSTNAME}"
    printf 'closure_id=%s\n' "${TERMINALBENCH_CLOSURE_ID}"
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${tmp}"
mv "${tmp}" "${TERMINALBENCH_RECEIPT}"
exit "${status}"
EOF
chmod 700 "${WRAPPER}"

export RLLM_ROOT ASR_ROOT
export TERMINALBENCH_RUN_ROOT="${RUN_ROOT}"
export TERMINALBENCH_RECEIPT="${RECEIPT}"
nohup setsid bash "${WRAPPER}" >>"${LOG_FILE}" 2>&1 </dev/null &
trainer_pid=$!
sleep 1
kill -0 "${trainer_pid}"
write_receipt BOOT_STARTED "trainer_pid_${trainer_pid}"
trap - ERR

echo "TerminalBench ${ARM} launch started on ${HOST_ID} with pid ${trainer_pid}"
