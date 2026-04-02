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

# Run wandb leet for experiment trackin
leet:
    # Run wandb leet for experiment trackin
    uv run wandb beta leet run wandb

# Run experiment with specific config
run_experiment config="timm_agents_experiment" *args="":
    uv run scripts/experiment.py --config-name {{config}} {{args}}

# Install test dependencies
test_setup:
    uv pip install pytest pytest-cov

# Run all tests
test:
    PYTHONPATH=. uv run pytest tests/ -v

# Run tests with coverage
test_coverage:
    PYTHONPATH=. uv run pytest tests/ --cov=src --cov-report=term-missing -v

# Run tests matching a pattern
test_pattern pattern="":
    PYTHONPATH=. uv run pytest tests/ -v -k "{{pattern}}"

# Run tests excluding slow tests
test_fast:
    PYTHONPATH=. uv run pytest tests/ -v -m "not slow"

# Run only slow tests
test_slow:
    PYTHONPATH=. uv run pytest tests/ -v -m "slow"

# Run tests for specific module
test_module module="agents":
    PYTHONPATH=. uv run pytest tests/{{module}}/ -v
