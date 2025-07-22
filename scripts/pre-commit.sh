#!/bin/bash

projects=(
    "../fastmcp-ai-agent-bridge-pydantic-ai"
)
cd ..
uv sync -U --active
uv run ruff check --fix
uv run ruff format

for project in "${projects[@]}"; do
    cd $project
    uv venv
    . .venv/bin/activate
    uv run --active ruff format
    uv sync -U --active
    uv run --active ruff check --fix
    uv run --active basedpyright
    deactivate
    cd -
done