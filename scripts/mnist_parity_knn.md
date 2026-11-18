# MNIST: 10 classes vs even/odd, k-NN probing

**Question.** Two agents see the same MNIST images but at different label
granularity: agent 0 learns the 10 digits, agent 1 only learns even vs odd.
Does digit identity, which agent 1 is never asked to keep, survive in agent 1's
latent space? And does cooperation through the sheaf (DiSheafFRL) put more of it
there than independent training does?

## Files

| File | Role |
|---|---|
| `config/hydra/mnist_2agents_parity_base.yaml` | Shared setting: data split, training, orchestrator defaults (not run directly) |
| `config/hydra/mnist_2agents_parity_d40x8.yaml` | Setup: latent sizes d = 40 / 8; wandb project `mnist_2agents_parity_d40x8` |
| `src/utils/knn_probes.py` (+ `tests/utils/test_knn_probes.py`) | The cosine k-NN probes: 10-class, within-parity, few-label |
| `scripts/mnist_parity_knn.py` | Trains each (method, λ, seed) cell, runs the 10-class k-NN probe, caches the test latents, plots |
| `scripts/mnist_parity_probes.py` | No training: reads the cached latents and runs the within-parity and few-label probes |

Recipes: `just train-parity` (the study), `just figure-parity` (figures from the cache), `just smoke-parity`.

## Setup

| | Agent 0 ("fine") | Agent 1 ("parity") |
|---|---|---|
| Task | 10 digit classes | even vs odd (`agent_label_map: y -> y % 2`) |
| Encoder | CNN `[32, 64, 128, 40]`, d = 40 | CNN `[32, 64, 8]`, d = 8 |
| Head | MLP `[256]` | MLP `[120]` (10 outputs, only targets 0/1 are trained) |
| Latent | CenterNorm `(z - mean) / ‖z - mean‖`, no whitening | same |

- **Data:** `overlapping_shift` with `shift_strength: 0`. Each agent draws
  20 000 training rows uniformly from the official train split, and the two
  draws overlap.
- **Pilots:** 512 rows of the train split. Neither agent trains on them, and no
  method sees their labels. Both agents get every pilot (`shared_global_pilots`).
- **Test:** the full official 10k test split, the same for both agents, with the
  true 10-class labels.
- **Graph:** 2 agents, fully connected (one edge).
- **Training:** `optimizer: sgd_cosine` (SGD, Nesterov momentum 0.9, weight decay
  5e-4, cosine lr schedule over the epochs) with the base lr overridden to
  1e-2, 100 epochs, batch 64, grad clip 1.0, deterministic.

> The original branch (`origin/leo-mnist-parity`) referenced a constant-lr
> `optimizer/sgd` config that no longer exists in the repo; this setup uses
> `sgd_cosine` instead, so its numbers are not comparable to runs made on that
> branch.

**Methods**

| Tag | Orchestrator | Notes |
|---|---|---|
| `noncoop` | `non_cooperative` | Agents train independently. At test time a post-hoc Procrustes map is fitted on the pilots |
| `disheaf` | `di_sheaf_frl` | Directed sheaf penalty on the shared unlabelled pilots, weight `max_lmb` = λ (default 4e-3, fixed) |

Post-hoc alignment only changes the comm accuracy. It never changes agent 1's
own latents, so the k-NN probes on agent 1 do not depend on it.

> The private/comm numbers that W&B logs for agent 1 from the orchestrator use
> the 10-class labels, so they mean nothing. Use the `probe/*` metrics the
> script adds to each run's summary.

## Metrics

All probes run on the **test** latents with cosine distance, because the
CenterNorm latents lie on the unit sphere.

From `mnist_parity_knn.py`, per run (wandb summary `probe/<name>`, and
recomputed from the cache for the tables and plot):

- `knn_parity_agent_k<k>`: 10-class k-NN accuracy on agent 1's latents. The test
  split is cut into two stratified halves. The probe is fitted on one half and
  scored on the other, then the halves swap and the two scores are averaged.
- `knn_fine_agent_k<k>`: the same probe on agent 0's latents (reference).
- `parity_acc_agent1`: agent 1's even/odd accuracy.
- `fine_acc_agent0`: agent 0's 10-class accuracy.
- `comm_acc_1_to_0`: agent 1's test latents mapped into agent 0's space and
  decoded by agent 0's 10-class head.
- `gate_mean`: the DiSheafFRL edge gate, from `train/gate_mean_0_1` (NaN for noncoop).

From `mnist_parity_probes.py` (printed and plotted, not logged):

- **Within-parity k-NN:** separate the 5 even digits among the even test images,
  and the 5 odd digits among the odd ones (chance 0.20). Parity is what agent 1
  was trained on, so this probe isolates the digit information it was never asked
  to keep.
- **Few-label k-NN:** the 10-class probe with only *n* labelled references per
  class, averaged over `--repeats` random draws. Cells with k > n are skipped.

## Running

Everything runs from the repo root. Real runs log to wandb project
`mnist_2agents_parity_d40x8` (group = orchestrator name); the smoke recipe sets
`WANDB_MODE=disabled`.

```bash
# 1. Smoke check (wandb off, 1 epoch, 2k rows, CPU, output in tmp/mnist_parity_smoke)
just smoke-parity --seeds 1
uv run python scripts/mnist_parity_probes.py --out tmp/mnist_parity_smoke --k 1 5 --n 5 10 --repeats 2

# 2. The study: noncoop + DiSheafFRL at λ = 1e-3, 4e-3, 1e-2, 3e-2, 3 seeds
#    (= mnist_parity_knn.py --seeds 42 43 44 --lmb 1e-3 4e-3 1e-2 3e-2; λ = 4e-3 alone is the default)
just train-parity

# 3. Figures from the cached latents, no training: the k-NN plot, then the within-parity and
#    few-label probes (defaults); other probe settings:
just figure-parity
uv run python scripts/mnist_parity_probes.py --k 1 5 10 20 --n 1 5 10 50 100 --repeats 10
```

Useful flags for `mnist_parity_knn.py`:

| Flag | Default | |
|---|---|---|
| `--config` | `mnist_2agents_parity_d40x8` | Setup config (new latent sizes = new child config of the base) |
| `--methods` | `noncoop disheaf` | Run only one method |
| `--lmb` | `4e-3` | DiSheafFRL λ values (ignored by noncoop) |
| `--seeds` | `42` | |
| `--k` | `1 3 5 10 20 50 100` | k values for the probe |
| `--epochs` / `--pilots` / `--train-samples` | config (`100`) / `512` / `20000` | `--epochs` only overrides `trainer.max_epochs` when given |
| `--out` | `results/<config>` | |
| `--overrides` | — | Extra Hydra overrides, e.g. `model.agents.1.encoder_hidden_dims=[32,64,16]` |

**Resuming.** Each finished run leaves `<out>/latents/<run>.npz`. A cell whose
npz exists for the same setup (hash of the composed config, minus method, λ and
seed) is skipped, so an interrupted sweep restarts with the same command.
Changing epochs, pilots, train samples or any override changes the hash: use a
new `--out` for a new setup (the tables refuse to mix setups).

## Outputs (`results/mnist_2agents_parity_d40x8/`, git-ignored)

Numbers live in wandb; locally only the cache and the figures:

| File | Content |
|---|---|
| `latents/<method>[_lmb<λ>]_s<seed>.npz` | `z_fine`, `z_parity`, `y` (test split), the scalar metrics and the setup hash: the offline re-scoring cache |
| `knn_vs_k.{png,pdf}` | 10-class k-NN on agent 1 against k, with agent 0 as the dashed reference. With several λ, a second panel against λ |
| `probes.{png,pdf}` | Within-parity k-NN against k, and few-label k-NN against references per class |

## Reading the results

- Agent 1's k-NN 10-class accuracy sits between chance (0.10) and agent 0's
  reference curve. The gap to the reference shows how much digit identity the
  parity task throws away.
- The within-parity probe is the cleanest measure. Above 0.20, agent 1 still
  separates digits that its loss treats as the same class.
- The main comparison is `disheaf` against `noncoop` on these probes: does the
  sheaf coupling on the pilots move digit structure into agent 1's latents?
  Check alongside that `parity_acc_agent1` does not drop as λ grows.
- One seed is a hint, not a result (noise floor ~±1.5 pt): use ≥ 3 seeds, and
  add a LEDGER row once a study is run.
