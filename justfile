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

# Launch (or attach to) the `sfrl` tmux session running both multi-agent
# experiments — one window per config: hetero_multi_agent (default config)
# and homo_multi_agent (hetero_rate_multiagent_mnist_homo).
sfrl:
    #!/usr/bin/env bash
    set -euo pipefail
    session="sfrl"
    root="{{justfile_directory()}}"
    if ! tmux has-session -t "$session" 2>/dev/null; then
        # Window 1: hetero setup (script default config).
        tmux new-session -d -s "$session" -n hetero_multi_agent -c "$root"
        tmux send-keys -t "$session:hetero_multi_agent" \
            'uv run scripts/multi_agent_experiment.py' C-m
        # Window 2: homo setup.
        tmux new-window -t "$session" -n homo_multi_agent -c "$root"
        tmux send-keys -t "$session:homo_multi_agent" \
            'uv run scripts/multi_agent_experiment.py --config-name hetero_rate_multiagent_mnist_homo' C-m
        tmux select-window -t "$session:hetero_multi_agent"
    fi
    # Attach, or switch if we're already inside tmux.
    if [ -n "${TMUX:-}" ]; then
        tmux switch-client -t "$session"
    else
        tmux attach-session -t "$session"
    fi

# Run multi-agent experiment with hetero config (default)
multiagent-hetero *args="":
    uv run scripts/multi_agent_experiment.py {{args}}

# Run multi-agent experiment with homo config
multiagent-homo *args="":
    uv run scripts/multi_agent_experiment.py --config-name hetero_rate_multiagent_mnist_homo {{args}}

# Plot the sfrl_bottleneck sweep (comm task perf vs latent dim)
plot-bottleneck *args="":
    uv run scripts/plot_bottleneck_metrics.py --project sfrl_bottleneck {{args}}

# Plot the multi_hetero_agents_true project (comm-vs-shift, training curves, tables)
plot-hetero *args="":
    uv run scripts/plot_multiagent_metrics.py --project multi_hetero_agents_true {{args}}

# Plot the multi_homo_agents_true project (own out_dir so hetero plots aren't overwritten)
plot-homo *args="":
    uv run scripts/plot_multiagent_metrics.py --project multi_homo_agents_true --out_dir results/multi_agent/plots_homo {{args}}
