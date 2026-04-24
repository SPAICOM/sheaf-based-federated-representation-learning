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
experiment config="cnn_agents_experiment" *args="":
    uv run scripts/experiment.py --config-name {{config}} {{args}}

# Run multimodal (DeepSense) experiment from sheaf_frl onwards
multimodal_dec *args="":
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=sheaf_frl {{args}}
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=sheaf_fmtl {{args}}
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=fedper {{args}}
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=heterofl {{args}}

# Run multimodal (DeepSense) experiment for all supported orchestrators
deepsense *args="":
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=sheaf_frl {{args}}
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=comfed {{args}}
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=fedproto {{args}}
    uv run scripts/deepsense_experiment.py --config-name deepsense_experiment orchestrator=fedmuscle {{args}}

mhealth *args="":
    uv run scripts/mhealth_experiment.py --config-name mhealth_experiment orchestrator=fedproto {{args}}
    uv run scripts/mhealth_experiment.py --config-name mhealth_experiment orchestrator=fedmuscle {{args}}
    uv run scripts/mhealth_experiment.py --config-name mhealth_experiment orchestrator=comfed {{args}}
    uv run scripts/mhealth_experiment.py --config-name mhealth_experiment orchestrator=sheaf_fmtl {{args}}

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
