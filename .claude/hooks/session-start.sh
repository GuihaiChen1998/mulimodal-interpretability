#!/bin/bash
# Installs runpodctl for Claude Code on the web sessions.
# The Runpod MCP server is configured in .mcp.json and reads RUNPOD_API_KEY
# from the environment; set that key in the cloud environment settings.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

RUNPODCTL_VERSION="v2.14.0"

if ! command -v runpodctl >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64) arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) echo "runpodctl: unsupported arch $(uname -m)" >&2; exit 0 ;;
  esac
  tmp="$(mktemp)"
  if curl -fsSL --retry 3 -o "$tmp" \
      "https://github.com/runpod/runpodctl/releases/download/${RUNPODCTL_VERSION}/runpodctl-linux-${arch}"; then
    install -m 755 "$tmp" /usr/local/bin/runpodctl
  else
    echo "runpodctl: download failed, skipping" >&2
  fi
  rm -f "$tmp"
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "RUNPOD_API_KEY is not set; runpodctl and the Runpod MCP server will be unauthenticated." >&2
fi
