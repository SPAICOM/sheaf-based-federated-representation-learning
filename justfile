# Setup the repo .venv via uv
setup:
    uv sync

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
clean: format check

# Run experiment with specific config
run_experiment config="timm_agents_experiment" *args="":
    uv run scripts/experiment.py --config-name {{config}} {{args}}
