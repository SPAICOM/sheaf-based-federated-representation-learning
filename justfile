# ── Environment and checks ─────────────────────────────────────────────────────

# Set up the repo .venv via uv
setup:
    uv sync

# Lint (fixing what can be fixed) and format
check:
    uv run --locked ruff check . --fix
    uv run --locked ruff format .

# Run the test suite (add -m "not slow" to skip the slow tests)
test *args="":
    PYTHONPATH=. uv run pytest tests/ {{args}}

# ── Paper results ──────────────────────────────────────────────────────────────
# One pair of routines per result of the paper: `train-*` runs the experiment
# (Hydra multirun, metrics and checkpoints to wandb and logs/), the matching
# table/figure routine rebuilds the result from wandb and the checkpoints into
# results/. README.md lists the pairing. Extra arguments are Hydra overrides,
# e.g. `just train-ablation-comm-epochs seed=42`; long runs go through
# `just launch <name> <n_jobs> "<routine>"`.

# Lambda grid on seed 42, best test comm into tuned_lmb, then seeds 43/44 (scripts/tune_and_run.py);
# resumes from wandb. Options: methods list, --jobs N, --grid 1e-2,1e-1, --dry-run.
# Tab. main + Tab. geom, train one benchmark (mnist | fashionmnist | femnist | cifar10): 4 agents, 8 methods
train-benchmark dataset *args="":
    uv run python scripts/tune_and_run.py {{dataset}} {{args}}

# Re-scores every run of the four benchmarks from its checkpoint with one post-hoc protocol
# (cached in results/<project>/rescore/), then ranks the methods as in the paper.
# Tab. main + Tab. geom → results/paper/tab_{main,geom}.{tex,md}
tables-benchmarks:
    for p in cifar10_4agents_full femnist_4agents_shift mnist_4agents_shift fashionmnist_4agents_shift; do uv run python scripts/rescore_hetero.py $p; done
    uv run python scripts/table_main.py

# Every method at the tuned_lmb of the 4-agent FEMNIST benchmark (no tuning), 3 seeds.
# Options: methods list, --jobs N, --dry-run.
# Fig. compression, train: FEMNIST, 2 agents with latent d / d/2, d = 8, 16, 32, 64
train-compression *args="":
    uv run python scripts/tune_and_run.py femnist --study bottleneck {{args}}

# ComFed logs no class split in training: its checkpoints are re-scored first (cached).
# Fig. compression → results/femnist_2agents_bottleneck/bottleneck_split.{pdf,png} + bottleneck.md
figure-compression:
    uv run python scripts/rescore_hetero.py femnist_2agents_bottleneck --groups comfed
    uv run python scripts/plot_bottleneck.py femnist

# Every run lands in the wandb project and group its figure script reads. Non-coop and CenterNorm
# DiSheafFRL at K = 4096 are the `..._check_..._p4096` runs, shared with the lambda and topology
# studies and with rows 0% / 100% of Tab. comm-epochs.
# Fig. ablation-pilots, train: MNIST, K = 384..4096, lambda 0.01, 3 seeds: non-coop, CenterNorm, ZCA, OLS, SWBN
train-ablation-pilots *args="":
    #!/usr/bin/env bash
    set -euo pipefail
    run() { uv run python scripts/multi_agent_experiment.py -m "$@" {{args}}; }
    run --config-name mnist_4agents_shift orchestrator=non_cooperative,di_sheaf_frl \
        dataset.pilot_num_samples=384,768,1536,2048 seed=42,43,44 \
        'study_name=mnist4_di_clean_bs256_pilots_${hydra:runtime.choices.orchestrator}_p${dataset.pilot_num_samples}'
    run --config-name mnist_4agents_shift orchestrator=non_cooperative,di_sheaf_frl \
        dataset.pilot_num_samples=4096 seed=42,43,44 \
        'study_name=mnist4_di_clean_bs256_check_${hydra:runtime.choices.orchestrator}_p4096'
    for m in zca:mnist_4agents_shift_zca ols:mnist_4agents_shift_ols swbn:mnist_4agents_mapstudy_swbn; do
        for k in 384 768 1536 2048 4096; do
            run --config-name "${m#*:}" orchestrator=di_sheaf_frl dataset.pilot_num_samples=$k \
                orchestrator.max_lmb=0.01 seed=42,43,44 "study_name=${m%%:*}_p${k}_l.01"
        done
    done

# Redraws from the retained class-scope scores (results/mnist_4agents_shift/swbn_scopes/);
# `just figure-ablation-pilots rescore` re-scores every checkpoint first (CUDA).
# Fig. ablation-pilots → results/mnist_4agents_shift/maps/pilot_study{,_shared}.{pdf,png}
figure-ablation-pilots mode="":
    uv run python scripts/plot_mnist4_swbn_scopes.py {{ if mode == "rescore" { "" } else { "--plot-only" } }}

# The lambda = 0.01 points are the K = 4096 runs of train-ablation-pilots.
# Fig. ablation-lambda, train: MNIST, K = 4096, lambda = 0.003..1, 3 seeds: CenterNorm, ZCA, OLS, SWBN
train-ablation-lambda *args="":
    #!/usr/bin/env bash
    set -euo pipefail
    run() { uv run python scripts/multi_agent_experiment.py -m "$@" {{args}}; }
    run --config-name mnist_4agents_shift orchestrator=di_sheaf_frl \
        orchestrator.max_lmb=0.003,0.03,0.1,0.3,1 seed=42,43,44 \
        'study_name=mnist4_di_clean_bs256_lambda4096_${orchestrator.max_lmb}'
    for m in zca:mnist_4agents_shift_zca ols:mnist_4agents_shift_ols swbn:mnist_4agents_mapstudy_swbn; do
        for l in 0.003:.003 0.03:.03 0.1:.1 0.3:.3 1:1; do
            run --config-name "${m#*:}" orchestrator=di_sheaf_frl dataset.pilot_num_samples=4096 \
                orchestrator.max_lmb=${l%%:*} seed=42,43,44 "study_name=${m%%:*}_p4096_l${l#*:}"
        done
    done

# `just figure-ablation-lambda rescore` re-scores every checkpoint first (CUDA).
# Fig. ablation-lambda → results/mnist_4agents_shift/maps/lambda_study{,_shared}.{pdf,png}
figure-ablation-lambda mode="":
    uv run python scripts/plot_mnist4_swbn_scopes.py --study lambda {{ if mode == "rescore" { "" } else { "--plot-only" } }}

# Pilot / lambda / map summary tables and run provenance (results/mnist_4agents_shift/{pilots,lambda,maps}/),
# the inputs of the rescore mode of the two figures above (re-scores checkpoints: CUDA).
# Fig. ablation-pilots + ablation-lambda, summary tables
tables-ablation-maps:
    uv run python scripts/plot_mnist4_pilots.py
    uv run python scripts/plot_mnist4_lambda.py
    uv run python scripts/plot_mnist4_maps.py

# Star (hub = agent 0) and ring (0-1-2-3-0), wandb projects mnist_4agents_shift_{star,ring}; the fully
# connected arm is the K = 4096 check runs of train-ablation-pilots.
# Fig. ablation-topology, train: MNIST, non-coop and DiSheafFRL on a star and a ring, 3 seeds
train-ablation-topology *args="":
    for t in star ring; do uv run python scripts/multi_agent_experiment.py -m --config-name mnist_4agents_shift_$t \
        orchestrator=non_cooperative,di_sheaf_frl seed=42,43,44 \
        'study_name=${hydra:runtime.choices.orchestrator}_p${dataset.pilot_num_samples}' {{args}}; done

# Redraws from results/mnist_4agents_shift/topology/per_run.csv; `rescore` re-scores the checkpoints (CUDA).
# Fig. ablation-topology → results/mnist_4agents_shift/topology/topology_gap{,_shared}.{pdf,png}
figure-ablation-topology mode="":
    uv run python scripts/plot_mnist4_topology.py {{ if mode == "rescore" { "" } else { "--plot-only" } }}

# CEDiSheafFRL, lambda 0.01; rho = 0% / 100% are the non-coop / DiSheafFRL runs of Tab. main.
# Tab. comm-epochs, train: MNIST, DiSheafFRL communicating on rho = 2..80% of the 100 epochs, 3 seeds
train-ablation-comm-epochs *args="":
    uv run python scripts/multi_agent_experiment.py -m --config-name mnist_4agents_shift \
        orchestrator=ce_di_sheaf_frl orchestrator.comm_percentage=2,5,10,20,40,60,80 seed=42,43,44 \
        'study_name=mnist4_ce_di_comm${orchestrator.comm_percentage}' {{args}}

# Tab. comm-epochs → results/mnist_4agents_shift/comm_epochs/{rows.tex,table.md,runs.csv}
table-comm-epochs:
    uv run python scripts/rescore_hetero.py mnist_4agents_shift --groups mnist4_ce_di_comm2,mnist4_ce_di_comm5,mnist4_ce_di_comm10,mnist4_ce_di_comm20,mnist4_ce_di_comm40,mnist4_ce_di_comm60,mnist4_ce_di_comm80
    uv run python scripts/table_comm_epochs.py

#   just sheaf-fmtl-gamma cifar10_4agents_full fashionmnist_4agents_shift
# Experimental details, Sheaf-FMTL compression factor gamma: RAM of its maps per gamma, per config
sheaf-fmtl-gamma *configs:
    uv run python scripts/sheaf_fmtl_memory.py {{configs}}

# ── New experiments (not in the paper yet) ─────────────────────────────────────

# DiSheafFRL with K per-edge anchors taken from a 4096-pilot pool by k-means prototypes (warm-started
# refits for K >= 768) or by class-stratified sampling (70% from C_i Δ C_j), and the original K-pilot
# runs at K = 128 (K = 384..1536 are the pilot-ablation runs); lambda 0.01, 3 seeds.
# Anchor selection, train: MNIST, K = 128..1536
train-anchors *args="":
    #!/usr/bin/env bash
    set -euo pipefail
    run() { uv run python scripts/multi_agent_experiment.py -m "$@" {{args}}; }
    run --config-name mnist_4agents_shift_anchors_kmeans orchestrator=di_sheaf_frl \
        orchestrator.num_anchors=128,384 seed=42,43,44 'study_name=K${orchestrator.num_anchors}'
    run --config-name mnist_4agents_shift_anchors_kmeans orchestrator=di_sheaf_frl \
        orchestrator.num_anchors=768,1536 orchestrator.kmeans_warm_start=true seed=42,43,44 \
        'study_name=K${orchestrator.num_anchors}_warm'
    run --config-name mnist_4agents_shift_anchors_strat orchestrator=di_sheaf_frl \
        orchestrator.num_anchors=128,384,768,1536 seed=42,43,44 'study_name=K${orchestrator.num_anchors}'
    run --config-name mnist_4agents_shift orchestrator=di_sheaf_frl dataset.pilot_num_samples=128 \
        orchestrator.max_lmb=0.01 seed=42,43,44 study_name=cnorm_p128_l.01
    run --config-name mnist_4agents_shift orchestrator=non_cooperative dataset.pilot_num_samples=128 \
        seed=42,43,44 study_name=mnist4_di_clean_bs256_pilots_non_cooperative_p128

# Plots the cached class-scope scores; `just figure-anchors rescore` first scores the new checkpoints
# (resumable, CUDA; non-coop rows re-score the K = 4096 check runs under every arm).
# Anchor selection → results/mnist_4agents_shift/anchors/anchors_study{,_shared}.{pdf,png} + summary.md
figure-anchors mode="":
    {{ if mode == "rescore" { "uv run python scripts/score_mnist4_anchors.py" } else { "true" } }}
    uv run python scripts/plot_mnist4_anchors.py

# Agent 0 learns the 10 digits (d = 40), agent 1 only even vs odd (d = 8), same images; non-coop and
# DiSheafFRL at lambda = 1e-3, 4e-3, 1e-2, 3e-2, 3 seeds (scripts/mnist_parity_knn.md). Test latents are
# cached in results/mnist_2agents_parity_d40x8/latents/; finished cells are skipped.
# MNIST parity, train: two agents, 10 digits vs even/odd, with the 10-class k-NN probe of agent 1
train-parity *args="":
    uv run python scripts/mnist_parity_knn.py --seeds 42 43 44 --lmb 1e-3 4e-3 1e-2 3e-2 {{args}}

# From the cached latents, no training: 10-class, within-parity and few-label k-NN probes.
# MNIST parity → results/mnist_2agents_parity_d40x8/{knn_vs_k,probes}.{pdf,png}
figure-parity:
    uv run python scripts/mnist_parity_knn.py --plot-only
    uv run python scripts/mnist_parity_probes.py

# DomainNet, 6 agents x 6 domains

# DomainNet data: the 20 best-covered classes of all six domains at 64 px → data/domainnet/dn64_k20_d6/
domainnet-data:
    uv run python scripts/prepare_domainnet.py

# Non-coop and DiSheafFRL at lambda = 0.01, 0.03, 0.1, 0.3, seeds 42/43/44 (wandb groups
# non_cooperative, di_sheaf_frl_lambda).
# DomainNet, train one setup: domainnet_6agents (no label shift) | domainnet_6agents_shift
train-domainnet setup="domainnet_6agents" *args="":
    uv run python scripts/multi_agent_experiment.py -m --config-name {{setup}} \
        orchestrator=non_cooperative seed=42,43,44 study_name=non_cooperative {{args}}
    uv run python scripts/multi_agent_experiment.py -m --config-name {{setup}} \
        orchestrator=di_sheaf_frl orchestrator.max_lmb=0.01,0.03,0.1,0.3 seed=42,43,44 \
        study_name=di_sheaf_frl_lambda {{args}}

# Per-domain scores of every run (cached) → results/<setup>/domain_scope_report.md + domain_scope.csv.
# DomainNet, both setups: per-domain report + figures → results/domainnet_6agents_figures/
report-domainnet:
    uv run python scripts/domain_scope.py domainnet_6agents
    uv run python scripts/domain_scope.py domainnet_6agents_shift
    uv run python scripts/plot_domain_scope.py domainnet_6agents domainnet_6agents_shift \
        --names "No label shift" "Label shift (7/10 classes)" --out results/domainnet_6agents_figures

# ── Running experiments ───────────────────────────────────────────────────────

#   just smoke domainnet_6agents orchestrator=di_sheaf_frl
# Smoke-run a config without wandb: 1 epoch, a few batches
smoke config *args="":
    WANDB_MODE=disabled uv run python scripts/multi_agent_experiment.py \
        --config-name {{config}} trainer.max_epochs=1 ++trainer.limit_train_batches=3 \
        ++trainer.limit_val_batches=2 ++trainer.limit_test_batches=2 \
        ++lmb_study.enabled=false {{args}}

# MNIST parity smoke run: no wandb, 1 epoch, 2k rows, CPU, output in tmp/mnist_parity_smoke
smoke-parity *args="":
    WANDB_MODE=disabled uv run python scripts/mnist_parity_knn.py --epochs 1 --pilots 128 --train-samples 2000 \
        --accelerator cpu --out tmp/mnist_parity_smoke {{args}}

# Left pane: the run, teed to logs/console/<name>.log; right pane: live dashboard. Survives
# closing the terminal; `just attach <name>` to follow it.
#   just launch comm_epochs 21 "just train-ablation-comm-epochs"
# Run a routine detached in tmux
launch name total cmd:
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{justfile_directory()}}"
    s="{{name}}"
    if tmux has-session -t "$s" 2>/dev/null; then echo "tmux session '$s' exists"; exit 1; fi
    mkdir -p "$root/logs/status/$s" "$root/logs/console"
    printf '%s\n' '{{cmd}}' > "$root/logs/status/$s/cmd.sh"
    printf '{"total": %s, "cmd": %s, "started": "%s", "commit": "%s"}\n' \
        "{{total}}" "$(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read().strip()))' "$root/logs/status/$s/cmd.sh")" \
        "$(date +%Y-%m-%dT%H:%M:%S%z)" "$(git -C "$root" rev-parse --short HEAD)$(git -C "$root" diff --quiet || echo -dirty)" \
        > "$root/logs/status/$s/_sweep.json"
    tmux new-session -d -s "$s" -n run -c "$root"
    # The command runs from a file: macOS caps a line typed into a fresh shell at 1024 bytes.
    # tmux shells also inherit launchd's 256 open-file limit, too low for DataLoader workers.
    tmux send-keys -t "$s:run" "ulimit -n 65536 2>/dev/null; export SWEEP=$s; bash logs/status/$s/cmd.sh 2>&1 | tee logs/console/$s.log; echo '[sweep finished]'" C-m
    tmux split-window -h -t "$s:run" -c "$root" "uv run python scripts/runs_dashboard.py $s"
    tmux select-pane -t "$s:run.{left}"   # not .0: pane-base-index may be 1
    echo "launched '$s' → attach: just attach $s   |   overview: just dash"

# Attach to (or switch to) a launched routine's tmux session
attach name:
    @if [ -n "${TMUX:-}" ]; then tmux switch-client -t {{name}}; else tmux attach -t {{name}}; fi

# Dashboard over all launched routines (add a name to filter); --once prints a snapshot
dash *args="":
    uv run python scripts/runs_dashboard.py {{args}}
