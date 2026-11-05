#!/usr/bin/env bash
# Restore a dump into an existing (stopped) container's data volume, then start it.
# Usage: scripts/neo4j_restore.sh <container> <dump-dir>
set -euo pipefail
CONTAINER="${1:?target container}"
DUMP_DIR="$(cd "${2:?dump dir}" && pwd)"
IMAGE="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
DB="${NEO4J_DATABASE:-neo4j}"

docker stop "$CONTAINER" >/dev/null 2>&1 || true
docker run --rm --volumes-from "$CONTAINER" -v "$DUMP_DIR":/backups "$IMAGE" \
  neo4j-admin database load "$DB" --from-path=/backups --overwrite-destination=true
docker start "$CONTAINER" >/dev/null
echo "restored $DB into $CONTAINER"
