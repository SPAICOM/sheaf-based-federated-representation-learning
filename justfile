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

# ── OLS-vs-Procrustes restriction-map ablation (config: hetero_rate_2agents_mnist_ols_ablation)
# SheafFRL on the 2-agent d=32 bottleneck setup, same fixed λ for both arms:
# {Procrustes, OLS} x seeds, logged to the sfrl_ols_ablation wandb project.
#   just ols-ablation                          # seeds 1-5, both arms (10 runs)
#   just ols-ablation "1,2,3" orchestrator.max_lmb=1e-2
ols-ablation seeds="1,2,3,4,5" *args="":
    uv run scripts/multi_agent_experiment.py --config-name hetero_rate_2agents_mnist_ols_ablation --multirun \
        'orchestrator.use_general_maps=false,true' \
        'seed={{seeds}}' {{args}}

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

# ── CIFAR-10 study (overlapping draws, official 10k test split for every agent) ──
# Step 1: single-agent ResNet-10, no shift (target ~0.9 test accuracy)
cifar10-single *args="":
    uv run scripts/multi_agent_experiment.py --config-name cifar10_single_agent {{args}}

# Step 2: single agent under shift (train size shrinks naturally with s)
cifar10-single-shift *args="":
    uv run scripts/multi_agent_experiment.py --config-name cifar10_single_agent \
        'dataset.shift_strength=0.2,0.5,0.8' {{args}}

# Step 3: 4 agents, {sheaf_frl, non_cooperative} x shift {0.2, 0.8} x pilots {128, 256, 2048}
# (SheafFRL jobs run a 10-point lambda grid on [1e-4, 1] first, ranked by network-wide comm acc)
cifar10-4agents *args="":
    uv run scripts/multi_agent_experiment.py --config-name cifar10_4agents_shift {{args}}

# ── CIFAR-10, 2 heterogeneous agents (guide: README, "Two heterogeneous agents on CIFAR-10") ──
# setup = grayblur | grayblur_harsh | crops | shift   (config/hydra/cifar10_2agents_<setup>.yaml)
# Non-coop: one training run with all intersection pilots; prints the wandb run id.
two-agents-noncoop setup *args="":
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run scripts/multi_agent_experiment.py \
        --config-name cifar10_2agents_{{setup}} orchestrator=non_cooperative {{args}}

# Non-coop comm accuracy vs pilot budget, encoders fixed (run_id from two-agents-noncoop).
two-agents-sweep setup run_id *args="":
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run scripts/pilot_count_sweep.py \
        --config-name cifar10_2agents_{{setup}} --run {{run_id}}:::non_cooperative \
        --n 8 32 128 512 2048 100000 --repeats 2 \
        --csv results/cifar10_2agents_{{setup}}/noncoop_sweep.csv {{args}}

# SheafFRL at one pilot budget; whitening = sheaf_frl (SWBN) | sheaf_frl_zca (closed-form ZCA).
two-agents-sheaf setup whitening pilots lmb *args="":
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run scripts/multi_agent_experiment.py \
        --config-name cifar10_2agents_{{setup}} orchestrator={{whitening}} \
        dataset.pilot_num_samples={{pilots}} orchestrator.max_lmb={{lmb}} {{args}}

# ── Tracked launch: run in a detached tmux session with a live dashboard ───────
# Usage:  just launch <sweep-name> <n-jobs|0> "<command>"
#   just launch grayblur_lmb 10 "uv run scripts/multi_agent_experiment.py -m --config-name cifar10_2agents_grayblur_harsh orchestrator.max_lmb=0.01,0.03,..."
# Left pane: the run (log also teed to logs/<sweep>.log). Right pane: dashboard.
# Survives closing Claude/your terminal (tmux server owns it). Attach: just attach <sweep>
launch sweep total cmd:
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{justfile_directory()}}"
    s="{{sweep}}"
    if tmux has-session -t "$s" 2>/dev/null; then echo "tmux session '$s' exists"; exit 1; fi
    mkdir -p "$root/logs/status/$s" "$root/logs/console"
    printf '{"total": %s, "cmd": %s, "started": "%s", "commit": "%s"}\n' \
        "{{total}}" "$(printf '%s' '{{cmd}}' | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" \
        "$(date -Is)" "$(git -C "$root" rev-parse --short HEAD)$(git -C "$root" diff --quiet || echo -dirty)" \
        > "$root/logs/status/$s/_sweep.json"
    tmux new-session -d -s "$s" -n run -c "$root"
    tmux send-keys -t "$s:run" "export SWEEP=$s; {{cmd}} 2>&1 | tee logs/console/$s.log; echo '[sweep finished]'" C-m
    tmux split-window -h -t "$s:run" -c "$root" "uv run python scripts/runs_dashboard.py $s"
    tmux select-pane -t "$s:run.0"
    echo "launched '$s' → attach: just attach $s   |   overview: just dash"

# Attach to (or switch to) a launched sweep's tmux session
attach sweep:
    @if [ -n "${TMUX:-}" ]; then tmux switch-client -t {{sweep}}; else tmux attach -t {{sweep}}; fi

# Dashboard over all sweeps (add a name to filter); --once prints a snapshot
dash *args="":
    uv run python scripts/runs_dashboard.py {{args}}

# Smoke-run an experiment config: no wandb, 1 epoch, a few batches, no λ study.
#   just smoke cifar10_2agents_grayblur_6v6 orchestrator=di_sheaf_frl
smoke config *args="":
    WANDB_MODE=disabled uv run python scripts/multi_agent_experiment.py \
        --config-name {{config}} trainer.max_epochs=1 ++trainer.limit_train_batches=3 \
        ++trainer.limit_val_batches=2 ++trainer.limit_test_batches=2 \
        ++lmb_study.enabled=false {{args}}

# Score wandb runs against state/objectives/<objective>.yaml and append to its .tsv
#   just score 6v6_cnorm <run_id> [<run_id> ...] --note "what changed"
score objective *args="":
    uv run python scripts/score_run.py {{objective}} {{args}}

# Table + curves of a finished sweep from wandb → tmp/<date>_<project>_<slug>/ (see scripts/sweep_report.py)
report project *args:
    uv run python scripts/sweep_report.py {{project}} {{args}}
