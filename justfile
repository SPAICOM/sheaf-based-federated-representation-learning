# ── Environment and checks ─────────────────────────────────────────────────────

# Setup the repo .venv via uv
setup:
    uv sync

# Run static analysis and automatically fix issues where possible
check:
    uv run --locked ruff check . --fix

# Check lint rules without modifying source files
lint-check:
    uv run --locked ruff check . --no-fix

# Format code according to project style
format:
    uv run --locked ruff format .

# Run formatting and linting (CI-style target)
clean: format check

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

# ── Paper experiments ──────────────────────────────────────────────────────────
# Each recipe is one Hydra multirun over the methods and seeds listed in the
# config's hydra.sweeper.params; extra arguments are Hydra overrides, e.g.
#   just hetero mnist orchestrator=di_sheaf_frl seed=42
# Long runs go through `just launch`, e.g. just launch hetero_mnist 24 "just hetero mnist".

# Accuracy vs bottleneck dimension: 2 CIFAR-10 agents, latent d / d/2, d = 8..256.
bottleneck *args="":
    uv run python scripts/multi_agent_experiment.py -m --config-name cifar10_2agents_bottleneck {{args}}

# Heterogeneity table, one dataset (mnist | fashionmnist | femnist | cifar10): 4 agents, shift 1.0.
hetero dataset *args="":
    uv run python scripts/multi_agent_experiment.py -m --config-name {{ if dataset == "cifar10" { "cifar10_4agents_full" } else { dataset + "_4agents_shift" } }} {{args}}

#   just hetero-pipeline cifar10 [fedproto,fedre] [--jobs N, default 1] [--grid 1e-2,1e-1] [--dry-run]
# Heterogeneity table with lambda tuning, all methods of the config unless listed: grid on seed 42 → best comm into tuned_lmb → other seeds; non-coop/FedRE: all seeds. Resumes from wandb.
hetero-pipeline dataset *args="":
    uv run python scripts/tune_and_run.py {{dataset}} {{args}}

#   just bottleneck-pipeline femnist [non_cooperative,fedre] [--jobs N, default 1] [--dry-run]
# Accuracy vs bottleneck dimension (femnist | cifar10), 2 agents with latent d / d/2, every method at its tuned_lmb
# (the 4-agent pipeline's pick, no tuning): <dataset>_2agents_bottleneck, method x d x seed, first seed of every d first. Resumes from wandb.
# Paper run: just launch femnist_bneck_pipeline 126 "just bottleneck-pipeline femnist non_cooperative,sheaf_frl,di_sheaf_frl,comfed,fedproto,fedmuscle,fedre --jobs 4"
bottleneck-pipeline dataset *args="":
    uv run python scripts/tune_and_run.py {{dataset}} --study bottleneck {{args}}

# SheafFMTL RAM per gamma (its P_ij matrices scale with the parameter count): pick sheaf_fmtl_gamma in the config
#   just sheaf-fmtl-memory cifar10_4agents_full fashionmnist_4agents_shift
sheaf-fmtl-memory *configs:
    uv run python scripts/sheaf_fmtl_memory.py {{configs}}

# MNIST ablation: pilot budget, non-coop vs DiSheafFRL (lambda 0.01).
mnist-ablation-pilots *args="":
    uv run python scripts/multi_agent_experiment.py -m --config-name mnist_4agents_shift \
        orchestrator=non_cooperative,di_sheaf_frl dataset.pilot_num_samples=384,768,1536,2048,4096 \
        'study_name=mnist4_di_clean_bs256_pilots_${hydra:runtime.choices.orchestrator}_p${dataset.pilot_num_samples}' {{args}}

# MNIST ablation: penalty weight of DiSheafFRL (4096 pilots).
mnist-ablation-lambda *args="":
    uv run python scripts/multi_agent_experiment.py -m --config-name mnist_4agents_shift \
        orchestrator=di_sheaf_frl orchestrator.max_lmb=0.003,0.01,0.03,0.1,0.3,1 \
        'study_name=mnist4_di_clean_bs256_lambda4096_${orchestrator.max_lmb}' {{args}}

# MNIST ablation: restriction maps (method = zca | ols) over the pilot budget (lambda sweep: see README).
mnist-ablation-maps method *args="":
    uv run python scripts/multi_agent_experiment.py -m --config-name mnist_4agents_shift_{{method}} \
        dataset.pilot_num_samples=384,768,1536,2048,4096 \
        'study_name={{method}}_p${dataset.pilot_num_samples}_l${orchestrator.max_lmb}' {{args}}

# ── Paper figures and tables (read the wandb remote) ───────────────────────────

# Accuracy vs bottleneck dimension (femnist | cifar10) → results/<dataset>_2agents_bottleneck/bottleneck{,_rx_scope}.{pdf,png} + bottleneck.md
# Re-scores new ComFed checkpoints first (cached): ComFed logs no seen/unseen-class split in training.
plot-bottleneck dataset="femnist":
    uv run python scripts/rescore_hetero.py {{dataset}}_2agents_bottleneck --groups comfed
    uv run python scripts/plot_bottleneck.py {{dataset}}

# Heterogeneity table over MNIST, FMNIST, FEMNIST, CIFAR-10 (+ seen/unseen split) → results/paper/hetero_table{,_scope}.{md,tex}
table-hetero:
    uv run python scripts/table_hetero.py

# Seen/unseen-class breakdown of one experiment at network, agent and edge level → results/<project>/class_scope/
#   just class-scope cifar10_4agents_full
#   just class-scope cifar10_2agents_bottleneck --x latent_d
class-scope project *args="":
    uv run python scripts/report_class_scope.py {{project}} {{args}}

# Add the seen/unseen metrics to finished runs that predate them, from their checkpoints (dry run unless --write)
#   just rescore-class-scope cifar10_4agents_full --groups non_cooperative,sheaf_frl,di_sheaf_frl --write
rescore-class-scope project *args="":
    uv run python scripts/rescore_class_scope.py {{project}} {{args}}

# Re-score finished heterogeneity runs from their checkpoints with one post-hoc protocol for all methods:
# private (+seen), comm (all / both seen / only RX seen), MRR, misalignment, sheaf TV + λ_min → results/<project>/rescore/
#   just rescore-hetero cifar10_4agents_full [--groups comfed,sheaf_frl] [--force]
rescore-hetero project *args="":
    uv run python scripts/rescore_hetero.py {{project}} {{args}}

# MNIST pilot ablation: re-scores the 30 saved final checkpoints (needs CUDA)
mnist-pilots-summary:
    uv run python scripts/plot_mnist4_pilots.py

# MNIST lambda ablation
mnist-lambda-summary:
    uv run python scripts/plot_mnist4_lambda.py

# MNIST map ablation (CenterNorm vs ZCA vs OLS); reads the pilot/lambda aggregates
mnist-maps-summary:
    uv run python scripts/plot_mnist4_maps.py

# FashionMNIST pilot sweep, non-coop vs DiSheafFRL (retained runs)
fmnist-pilots-summary *args="":
    uv run python scripts/fashionmnist_4agents_shift_summary.py {{args}}

# ── Tracked launch: run in a detached tmux session with a live dashboard ───────
# Usage:  just launch <sweep-name> <n-jobs|0> "<command>"
#   just launch hetero_mnist 24 "just hetero mnist"
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
#   just smoke femnist_4agents_shift orchestrator=di_sheaf_frl
smoke config *args="":
    WANDB_MODE=disabled uv run python scripts/multi_agent_experiment.py \
        --config-name {{config}} trainer.max_epochs=1 ++trainer.limit_train_batches=3 \
        ++trainer.limit_val_batches=2 ++trainer.limit_test_batches=2 \
        ++lmb_study.enabled=false {{args}}

# Score wandb runs against state/objectives/<objective>.yaml and append to its .tsv
#   just score 6v6_cnorm <run_id> [<run_id> ...] --note "what changed"
score objective *args="":
    uv run python scripts/score_run.py {{objective}} {{args}}

# Table + curves of a finished sweep from wandb → tmp/<date>_<project>_<slug>/
report project *args:
    uv run python scripts/sweep_report.py {{project}} {{args}}

# MNIST two-agent 7-vs-7, CenterNorm and exclusive pool pilots, clean inputs.
mnist-7v7 *args="":
    uv run python scripts/multi_agent_experiment.py --config-name mnist_2agents_7v7 {{args}}

# MNIST 7-vs-7 with native grayscale vs CIFAR-matched Gaussian blur.
mnist-grayblur-7v7 *args="":
    uv run python scripts/multi_agent_experiment.py --config-name mnist_2agents_grayblur_7v7 {{args}}

# MNIST four-agents 7-vs-7 (CNNs); setup = clean | grayblur.
mnist-4agents setup="clean" *args="":
    uv run python scripts/multi_agent_experiment.py --config-name {{if setup == "clean" {"mnist_4agents_shift"} else {"mnist_4agents_grayblur_shift"}}} {{args}}

# Run one map-ablation experiment (method = zca | ols), with Hydra overrides.
mnist4-mapstudy method *args="":
    uv run python scripts/multi_agent_experiment.py --config-name mnist_4agents_mapstudy_{{method}} {{args}}

# Regenerate the retained two-agent summary from remote W&B.
mnist-7v7-summary:
    uv run python scripts/plot_mnist_7v7.py

# Re-score saved four-agent checkpoints and regenerate the pilot summary.
mnist4-pilot-summary:
    uv run python scripts/plot_mnist4_pilots.py

# Regenerate the four-agent lambda summary from remote W&B.
mnist4-lambda-summary:
    uv run python scripts/plot_mnist4_lambda.py

# Regenerate combined pilot/lambda map figures using remote W&B and retained summaries.
mnist4-map-summary:
    uv run python scripts/plot_mnist4_maps.py

# Redraw the topology gap figure (DiSheafFRL - Non-coop, star/ring/full) from the retained rescore; pass "" to rescore checkpoints.
mnist4-topology-summary *args="--plot-only":
    uv run python scripts/plot_mnist4_topology.py {{args}}

# Redraw the pilot study figure (maps/pilot_study) from the retained class-scope rescore; pass "" to rescore checkpoints (CUDA).
mnist4-swbn-scopes-summary *args="--plot-only":
    uv run python scripts/plot_mnist4_swbn_scopes.py {{args}}

# Lambda study by class scope (private / sender-unseen receiver-seen / receiver-seen) -> maps/lambda_study.*; pass "" to rescore checkpoints (CUDA).
mnist4-lambda-scopes-summary *args="--plot-only":
    uv run python scripts/plot_mnist4_swbn_scopes.py --study lambda {{args}}

# Queued MNIST4 pilot-BN-update .
mnist4-bn-update:
    just launch mnist4_bn_update 3 "bash tmp/2026-11-10_mnist4_bn_update.sh"

# Resume saved OLS-map lambda or pilot evaluation; --plan audits remote coverage.
mnist4-ols-learned study *args="":
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 uv run python tmp/2026-11-12_mnist4_ols_learned.py {{study}} {{args}}

# Refresh OLS learned-map overlays from finished W&B evaluations.
mnist4-ols-overlay:
    uv run python tmp/2026-11-12_mnist4_ols_overlay.py

# Audit posthoc jitter and optionally publish the missing lambda.3 saved-map evaluation.
mnist4-ols-sanity-audit *args="":
    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 WANDB_DIR="{{justfile_directory()}}/logs/wandb" uv run python tmp/2026-11-12_ols_sanity_audit.py {{args}}

# Topology study (ring + star, noncoop vs DiSheafFRL, 3 seeds); launcher tmp/2026-11-12_mnist4_topology.sh.
mnist4-topology:
    just launch mnist4_topology 12 "bash tmp/2026-11-12_mnist4_topology.sh"

# Topology study report: rescoring of saved checkpoints (comm / sender-unseen / receiver-seen) -> table + plot in tmp/.
mnist4-topology-report *args="":
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" uv run python tmp/2026-11-12_mnist4_topology_report.py {{args}}

# SWBN-vs-ZCA pilot ablation (15 runs: 5 pilot counts x 3 seeds, learned eval); launcher tmp/2026-11-12_mnist4_swbn.sh.
mnist4-swbn:
    just launch mnist4_swbn 15 "bash tmp/2026-11-12_mnist4_swbn.sh"

# Method summary (lambda + pilot map study) rescored by class scope from saved checkpoints on GPU0 -> tmp/2026-11-13_mnist4_method_scopes/.
mnist4-method-scopes *args="":
    just launch mnist4_method_scopes 1 "CUDA_VISIBLE_DEVICES=0 uv run python tmp/2026-11-13_mnist4_method_scopes.py {{args}}"

# SWBN lambda ablation (15 runs: lambda .003/.03/.1/.3/1 x 3 seeds at 4096 pilots; .01 reused); launcher tmp/2026-11-14_mnist4_swbn_lambda.sh.
mnist4-swbn-lambda:
    just launch mnist4_swbn_lambda 15 "bash tmp/2026-11-14_mnist4_swbn_lambda.sh"
