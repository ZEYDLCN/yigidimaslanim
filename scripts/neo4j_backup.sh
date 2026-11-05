#!/usr/bin/env bash
# Offline dump for Neo4j Community (online dump is Enterprise-only).
# The database must be stopped for the dump, so this stops the container, dumps, and restarts it.
# Usage: scripts/neo4j_backup.sh <container> <output-dir>
set -euo pipefail
CONTAINER="${1:?container name}"
OUT_DIR="$(mkdir -p "${2:?output dir}" && cd "$2" && pwd)"
IMAGE="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
DB="${NEO4J_DATABASE:-neo4j}"

# The dump runs as the image's neo4j user (uid 7474), which must be able to write the output.
if [ "$(id -u)" = "0" ]; then
  chown 7474:7474 "$OUT_DIR"
elif [ ! -w "$OUT_DIR" ]; then
  echo "output dir $OUT_DIR is not writable by uid 7474; run as root or chown it" >&2
  exit 1
fi

echo "stopping $CONTAINER for an offline dump"
docker stop "$CONTAINER" >/dev/null
trap 'docker start "$CONTAINER" >/dev/null' EXIT
docker run --rm --volumes-from "$CONTAINER" -v "$OUT_DIR":/backups "$IMAGE" \
  neo4j-admin database dump "$DB" --to-path=/backups --overwrite-destination=true
ls -l "$OUT_DIR"/"$DB".dump
