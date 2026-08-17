#!/usr/bin/env sh
set -eu

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends ca-certificates curl docker.io tmux

plugin_dir=/usr/local/lib/docker/cli-plugins
compose_bin=${plugin_dir}/docker-compose
mkdir -p "${plugin_dir}"
if [ ! -x "${compose_bin}" ]; then
    curl -L --fail --retry 4 \
        https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 \
        -o "${compose_bin}"
    chmod 755 "${compose_bin}"
fi

docker_root=${TERMINALBENCH_DOCKER_ROOT:-/mnt/nvme/terminalbench-docker}
mkdir -p /etc/docker "${docker_root}" /mnt/nvme/terminalbench-logs /run/docker
cat >/etc/docker/daemon.json <<'EOF'
{
  "default-address-pools": [
    {"base": "172.17.0.0/16", "size": 24},
    {"base": "192.168.0.0/16", "size": 24}
  ],
  "max-concurrent-downloads": 16,
  "max-concurrent-uploads": 8
}
EOF

dockerd \
    --host=unix:///var/run/docker.sock \
    --data-root="${docker_root}" \
    --exec-root=/run/docker \
    --pidfile=/run/docker.pid \
    --storage-driver=overlay2 \
    >/mnt/nvme/terminalbench-logs/dockerd.log 2>&1 &
export DOCKERD_PID=$!

for _ in $(seq 1 120); do
    docker info >/dev/null 2>&1 && break
    sleep 1
done
docker info >/dev/null
docker compose version
echo "TerminalBench privileged Docker holder ready on ${HOSTNAME}"

exec python3 -c '
import os
import subprocess
import sys

dockerd_pid = int(os.environ["DOCKERD_PID"])
sleeper = subprocess.Popen(["sleep", "infinity"])
while True:
    try:
        pid, status = os.wait()
    except InterruptedError:
        continue
    if pid == dockerd_pid:
        sleeper.terminate()
        sys.exit(70)
    if pid == sleeper.pid:
        if os.WIFEXITED(status):
            sys.exit(os.WEXITSTATUS(status))
        sys.exit(128 + os.WTERMSIG(status))
'
