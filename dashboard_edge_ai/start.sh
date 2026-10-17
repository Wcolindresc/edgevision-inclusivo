#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_ROOT="$(dirname "$PROJECT_DIR")"

if [[ ! -f "$EDGE_ROOT/venv/bin/activate" ]]; then
  echo "No se encontró el entorno virtual en $EDGE_ROOT/venv"
  exit 1
fi

source "$EDGE_ROOT/venv/bin/activate"
cd "$PROJECT_DIR"
exec python app.py
