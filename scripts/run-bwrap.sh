#!/bin/sh
# Run the same release the container runs, under bubblewrap instead of Docker.
#
# The artifact is not rebuilt here. The image's root filesystem is exported once
# and bubblewrap is pointed at it, so a difference between the two runs is a
# difference in the sandbox rather than in the build.
#
# Bubblewrap needs unprivileged user namespaces, so this is Linux only. On
# Windows it runs inside WSL, and `bwrap --version` is the check.
set -eu

IMAGE="${IMAGE:-weftspun/service-livebook:dev}"
ROOTFS="${ROOTFS:-${XDG_CACHE_HOME:-$HOME/.cache}/service-livebook/rootfs}"
DATA="${DATA:-${XDG_DATA_HOME:-$HOME/.local/share}/service-livebook/data}"
PORT="${LIVEBOOK_PORT:-8080}"

command -v bwrap >/dev/null || { echo "bwrap not found; this needs bubblewrap" >&2; exit 1; }

if [ ! -d "$ROOTFS/app" ]; then
  echo "exporting $IMAGE to $ROOTFS"
  mkdir -p "$ROOTFS"
  cid=$(docker create "$IMAGE")
  trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
  docker export "$cid" | tar -x -C "$ROOTFS"
  docker rm -f "$cid" >/dev/null
  trap - EXIT
fi

mkdir -p "$DATA"

# --unshare-all drops every namespace, then --share-net puts the network back,
# because the whole point is a reachable port. What stays dropped is pid, ipc,
# uts, cgroup and the user namespace, so the release cannot see or signal
# anything else on the host.
exec bwrap \
  --bind "$ROOTFS" / \
  --bind "$DATA" /data \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --unshare-all \
  --share-net \
  --die-with-parent \
  --new-session \
  --hostname livebook-sandbox \
  --setenv LIVEBOOK_DATA_PATH /data \
  --setenv LIVEBOOK_IP 0.0.0.0 \
  --setenv LIVEBOOK_PORT "$PORT" \
  --setenv HOME /data   --setenv LANG C.UTF-8 \
  --chdir /app \
  /app/bin/service_livebook start
