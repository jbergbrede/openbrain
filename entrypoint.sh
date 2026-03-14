#!/bin/bash
set -e

exec uv run --no-dev openbrain --mode "$MODE"
