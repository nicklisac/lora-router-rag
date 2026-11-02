#!/usr/bin/env bash
# Stop the EG2 and Clef llama.cpp servers started by start_servers.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
for name in eg2 clef; do
  pidfile="$HERE/results/$name-server.pid"
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile")"
    if kill "$pid" 2>/dev/null; then
      echo "stopped $name (pid $pid)"
    else
      echo "$name pid $pid not running"
    fi
    rm -f "$pidfile"
  else
    echo "no pidfile for $name"
  fi
done