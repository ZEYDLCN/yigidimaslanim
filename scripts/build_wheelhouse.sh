#!/usr/bin/env bash
# Prepares an offline build context for the Docker image, from the committed uv.lock.
# Dependencies and the application wheel are built on the host, so the image build needs no
# network access and does not depend on TLS interception settings inside the container.
set -euo pipefail
cd "$(dirname "$0")/.."
uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.lock.txt >/dev/null
rm -rf wheelhouse && mkdir -p wheelhouse
uv build --wheel -o wheelhouse >/dev/null
# Target the image's interpreter and platform, not the host's (the host may run another Python).
python3 -m pip download --quiet -r requirements.lock.txt -d wheelhouse \
  --only-binary=:all: --python-version 3.12 --implementation cp --abi cp312 \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
  --platform manylinux2014_x86_64 --platform any
echo "wheelhouse ready: $(ls wheelhouse | wc -l) files"
