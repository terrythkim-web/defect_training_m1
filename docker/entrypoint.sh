#!/usr/bin/env bash
# Entry point of the compose services (compose.yaml). On the first start it puts the project's
# Claude Code memory (docker/claude-memory, copied from the Mac) where Claude Code looks for the
# memory of /workspace; later starts keep whatever Claude has learned since (claude-home volume).
set -e
MEMORY_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/projects/-workspace/memory"
if [ -d /workspace/docker/claude-memory ] && [ ! -f "$MEMORY_DIR/MEMORY.md" ]; then
  mkdir -p "$MEMORY_DIR"
  # a failed copy must not keep the container from starting
  if cp -n /workspace/docker/claude-memory/*.md "$MEMORY_DIR/"; then
    echo "Claude Code 메모리를 넣었습니다: $MEMORY_DIR"
  else
    echo "Claude Code 메모리를 넣지 못했습니다: $MEMORY_DIR" >&2
  fi
fi
if [ "$#" -eq 0 ]; then
  exec bash
fi
exec "$@"
