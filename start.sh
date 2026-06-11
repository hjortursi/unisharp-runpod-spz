#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd /workspace
exec python -u /workspace/handler.py
