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

# Run multi-agent experiment with hetero config (default)
multiagent-shift-hetero *args="":
    uv run scripts/multi_agent_experiment.py --config-name multiagent_mnist_shift_overlap {{args}}

# Previous dispatcher: disjoint grouped_non_iid partition (kept for reproducing old runs)
multiagent-shift-hetero-disjoint *args="":
    uv run scripts/multi_agent_experiment.py --config-name multiagent_mnist_shift_distr {{args}}

# ── Pilot-budget sweep (config: multiagent_mnist_shift_overlap) ────────────────
# How many shared pilots does alignment actually need? Under overlapping_shift
# each agent draws its training rows independently of the pilot carve, so the
# agents' data is IDENTICAL at every point on the curve — the only thing that
# varies is |pilots|.
#
# shift_strength is pinned (the config's own sweeper would otherwise cross it
# with 5 shift values); pilot_num_samples is absolute, so the x-axis does not
# move with test_split. Budgets stay in [256, 3072]: below ~256 the per-step
# penalty falls under ~1.5x the latent dim (d~107) once the edge class filter
# applies, and above 3072 the pilot set stops being smaller than one agent's
# 3,600 training rows. pilot_batch_size self-clamps to the pool.
#
# Pilot-budget sweep: 5 pilot counts x {sheaf_frl, non_cooperative} = 10 jobs
pilot-sweep *args="":
    uv run scripts/multi_agent_experiment.py --config-name=multiagent_mnist_shift_overlap --multirun \
        'orchestrator=sheaf_frl,non_cooperative' \
        'dataset.pilot_num_samples=256,512,1024,2048,3072' \
        'dataset.shift_strength=0.9' \
        logger.project=pilot_sweep {{args}}

# Run multi-agent experiment with homo config
multiagent-homo *args="":
    uv run scripts/multi_agent_experiment.py --config-name hetero_rate_multiagent_mnist_homo {{args}}

multiagent-trial *args="":
    uv run scripts/multi_agent_experiment.py --config-name multiagent_mnist_trial {{args}}

simple-trial *args="":
    uv run scripts/multi_agent_experiment.py --config-name 2agents_mnist_trial {{args}}

hetero-bottleneck *args="":
    uv run scripts/multi_agent_experiment.py --config-name hetero_rate_2agents_mnist_hetero_bottleneck {{args}}

# Plot the sfrl_bottleneck sweep (comm task perf vs latent dim)
plot-bottleneck *args="":
    uv run scripts/plot_bottleneck_metrics.py --project sfrl_bottleneck {{args}}

plot-hetero-bottleneck *args="":
    uv run scripts/plot_hetero_bottleneck_metrics.py --project sfrl_hetero_bottleneck {{args}}

# Plot the multi_hetero_agents_true project (comm-vs-shift, training curves, tables)
plot-hetero *args="":
    uv run scripts/plot_multiagent_metrics.py --project shift_distr {{args}}

# Plot the per-agent degree distribution behind just plot-hetero's weighted-mean estimator
plot-degree *args="":
    uv run scripts/plot_degree_distribution.py {{args}}

# Plot the netowrk_analysis_sfrl sweep (comm/private accuracy vs graph density)
plot-network *args="":
    uv run scripts/plot_network_density_metrics.py --project netowrk_analysis_sfrl {{args}}

# Plot the multi_homo_agents_true project (own out_dir so hetero plots aren't overwritten)
plot-homo *args="":
    uv run scripts/plot_multiagent_metrics.py --project multi_homo --out_dir results/multi_agent/plots_homo {{args}}

# Network complete ablation study
network-ablation exp_args="" plot_args="":
    uv run scripts/multi_agent_experiment.py --config-name=multiagent_mnist_network_analyisis {{exp_args}}
    just plot-network {{plot_args}}

# Plot the comm_ablation sweep (comm accuracy vs comm fraction, one curve per local_reg)
plot-comm-ablation *args="":
    uv run scripts/plot_comm_ablation_metrics.py --project comm_ablation {{args}}

# ── Communication-efficiency ablation (config: multiagent_mnist_comm_ablation) ──
# Sweeps are CLI-driven here, not hard-coded in the config.

# (i) CESheafFRL grid: comm_percentage × local_reg (edit the lists as needed)
comm-ablation-ce *args="":
    uv run scripts/multi_agent_experiment.py --config-name=multiagent_mnist_comm_ablation --multirun \
        orchestrator=ce_sheaf_frl \
        'orchestrator.comm_percentage=2,10,30,60,90' \
        'orchestrator.local_reg=true' {{args}}

# (ii) Baselines: non_cooperative (0% comm) and sheaf_frl (communicates every step)
comm-ablation-baselines *args="":
    uv run scripts/multi_agent_experiment.py --config-name=multiagent_mnist_comm_ablation --multirun \
        'orchestrator=non_cooperative,sheaf_frl' {{args}}

# (iii) Build the comparison table from the wandb comm_ablation project
comm-ablation-table *args="":
    uv run scripts/comm_ablation_table.py --project comm_ablation {{args}}

# Run everything end to end: CE grid, then baselines, then the table
comm-ablation-all:
    just comm-ablation-ce
    just comm-ablation-baselines
    just comm-ablation-table

# ── 5-agent ring (config: 5agents_shift_overlap) ──────────────────────────────
# Five agents built from the 2-agent toy's TWO architectures (d=128 / d=224),
# differing only in which classes they over-represent. Target classes slide
# around a cycle, so adjacent agents share 3 of 5 classes and opposite ones
# share 1 — a spread of overlap strengths inside one graph.
#
# lambda: the 2-agent optimum was max_lmb=1e-2 with a cosine ramp, at degree 1.
# The penalty is SUMMED over edges, so it must be divided by mean degree.
# max_edge_frac 0.5 gives 5/10 edges => degree 2.0 => 5e-3 (the config default).
# If you change max_edge_frac, RESCALE max_lmb by the new mean degree.
RING_LMB := "5e-3"

# Single run (override orchestrator=non_cooperative for the baseline)
ring *args="":
    uv run scripts/multi_agent_experiment.py --config-name 5agents_shift_overlap {{args}}

# Distribution-shift sweep: paired baseline + sheaf at each shift level
ring-shift *args="":
    uv run scripts/multi_agent_experiment.py --config-name=5agents_shift_overlap --multirun \
        'orchestrator=non_cooperative,sheaf_frl' \
        'dataset.shift_strength=0.3,0.5,0.7,0.9' \
        logger.project=ring_shift {{args}}

# (a) PILOT ablation — how do the two methods scale with the number of pilots
# the alignment maps are fit on? pilot_batch_size self-clamps to the pool, and
# pilot_num_samples is absolute so the agents' training rows never move.
ring-pilots *args="":
    uv run scripts/multi_agent_experiment.py --config-name=5agents_shift_overlap --multirun \
        'orchestrator=non_cooperative,sheaf_frl' \
        'dataset.pilot_num_samples=128,256,512,1024,2048' \
        'dataset.shift_strength=0.7' \
        logger.project=ring_pilots {{args}}

# (b) COMMUNICATION ablation — how often must we talk during an epoch?
# CESheafFRL's comm_percentage is the share of epochs that are collaborative;
# non_cooperative (one post-hoc exchange) and sheaf_frl (every step) bracket it.
ring-comm *args="":
    uv run scripts/multi_agent_experiment.py --config-name=5agents_shift_overlap --multirun \
        'orchestrator=ce_sheaf_frl' \
        'orchestrator.comm_percentage=2,10,30,60,90' \
        'orchestrator.lambda_schedule=cosine' \
        'orchestrator.max_lmb={{RING_LMB}}' \
        'dataset.shift_strength=0.7' \
        logger.project=ring_comm {{args}}

# (c) TOPOLOGY ablation — graph DENSITY. NOTE: class_overlap_neighbors always
# keeps a maximum-weight spanning tree, so the edge count never drops below
# n-1 = 4 however small max_edge_frac is. With 5 agents: 0.4->4, 0.5->5,
# 0.7->7, 1.0->10 edges, i.e. mean degree 1.6/2.0/2.8/4.0 — so max_lmb must be
# rescaled per point, which a single multirun cannot do. Run the points you
# want individually with the matching lambda, e.g.:
#   just ring-density 0.4 6.25e-3
#   just ring-density 1.0 2.5e-3
ring-density frac lmb *args="":
    uv run scripts/multi_agent_experiment.py --config-name=5agents_shift_overlap --multirun \
        'orchestrator=non_cooperative,sheaf_frl' \
        graph.max_edge_frac={{frac}} orchestrator.max_lmb={{lmb}} \
        'dataset.shift_strength=0.7' \
        logger.project=ring_topology {{args}}
