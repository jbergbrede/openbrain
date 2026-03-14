#!/bin/bash
set -e

echo "Installing Claude CLI..."
curl -fsSL https://claude.ai/install.sh | bash

echo "Starting Claude authentication..."
claude auth login

echo "Authentication complete. Starting app..."
exec uv run --no-dev openbrain --mode "$MODE"
