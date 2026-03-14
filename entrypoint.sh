#!/bin/sh
set -e

echo "Starting Claude authentication..."
claude auth login

echo "Authentication complete. Starting app..."
exec uv run openbrain --mode "$MODE"
