# MNIST: 10 classes vs even/odd, k-NN probing

**Question.** Two agents see the same MNIST images but at different label
granularity: agent 0 learns the 10 digits, agent 1 only learns even vs odd.
Does digit identity, which agent 1 is never asked to keep, survive in agent 1's
latent space? And does cooperation through the sheaf (DiSheafFRL) put more of it
there than independent training does?

## Files

| File | Role |
|---|---|
| `config/hydra/mnist_2agents_parity.yaml` | Hydra config: data split, both agents, orchestrator defaults |
| `scripts/mnist_parity_knn.py` | Trains each (method, λ, seed) cell, runs the 10-class k-NN probe, saves the test latents, plots |
| `scripts/mnist_parity_probes.py` | No training: reads the saved latents and runs the within-parity and few-label probes |

## Setup (`mnist_2agents_parity.yaml`)

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
- **Training:** SGD with lr 1e-2, 20 epochs, batch 64, grad clip 1.0, deterministic.

**Methods**

| Tag | Orchestrator | Notes |
|---|---|---|
| `noncoop` | `non_cooperative` | Agents train independently. At test time a post-hoc Procrustes map is fitted on the pilots |
| `disheaf` | `di_sheaf_frl` | Directed sheaf penalty on the shared unlabelled pilots, weight `max_lmb` = λ (default 4e-3, fixed) |

Post-hoc alignment only changes the comm accuracy. It never changes agent 1's
own latents, so the k-NN probes on agent 1 do not depend on it.

> The private/comm numbers that W&B logs for agent 1 use the 10-class labels, so
> they mean nothing. Use the metrics computed by the scripts below.

## Metrics

All probes run on the **test** latents with cosine distance, because the
CenterNorm latents lie on the unit sphere.

From `mnist_parity_knn.py`, per run:

- `knn_parity_agent`: 10-class k-NN accuracy on agent 1's latents. The test
  split is cut into two stratified halves. The probe is fitted on one half and
  scored on the other, then the halves swap and the two scores are averaged.
- `knn_fine_agent`: the same probe on agent 0's latents (reference).
- `parity_acc_agent1`: agent 1's even/odd accuracy.
- `fine_acc_agent0`: agent 0's 10-class accuracy.
- `comm_acc_1_to_0`: agent 1's test latents mapped into agent 0's space and
  decoded by agent 0's 10-class head.
- `gate_mean`: the DiSheafFRL edge gate, from `train/gate_mean_0_1` (NaN for noncoop).

From `mnist_parity_probes.py`:

- **Within-parity k-NN:** separate the 5 even digits among the even test images,
  and the 5 odd digits among the odd ones (chance 0.20). Parity is what agent 1
  was trained on, so this probe isolates the digit information it was never asked
  to keep.
- **Few-label k-NN:** the 10-class probe with only *n* labelled references per
  class, averaged over `--repeats` random draws. Cells with k > n are skipped.

## Running

Everything runs from the repo root. `--accelerator auto` picks MPS on Apple
silicon, and `cpu` also works.

```bash
# 1. Quick end-to-end check (2 epochs, separate output folder)
uv run python scripts/mnist_parity_knn.py --epochs 2 --out results/tmp_smoke

# 2. Main run: noncoop + DiSheafFRL at λ = 4e-3, 3 seeds
uv run python scripts/mnist_parity_knn.py --seeds 1 2 3

# 3. λ sweep (adds a second panel: k-NN and comm accuracy against λ)
uv run python scripts/mnist_parity_knn.py --seeds 1 2 3 --lmb 1e-3 4e-3 1e-2 3e-2

# 4. Sharper probes on the saved latents (no training)
uv run python scripts/mnist_parity_probes.py
uv run python scripts/mnist_parity_probes.py --k 1 5 10 20 --n 1 5 10 50 100 --repeats 10

# Redraw the plot from runs.csv only
uv run python scripts/mnist_parity_knn.py --plot-only
```

Useful flags for `mnist_parity_knn.py`:

| Flag | Default | |
|---|---|---|
| `--methods` | `noncoop disheaf` | Run only one method |
| `--lmb` | `4e-3` | DiSheafFRL λ values (ignored by noncoop) |
| `--seeds` | `42` | |
| `--k` | `1 3 5 10 20 50 100` | k values for the probe |
| `--epochs` / `--pilots` / `--train-samples` | `20` / `512` / `20000` | |
| `--wandb` | off | Log to W&B project `mnist_2agents_parity` |
| `--out` | `results/mnist_2agents_parity` | |
| `--overrides` | — | Extra Hydra overrides, e.g. `model.agents.1.encoder_hidden_dims=[32,64,16]` |

**Changing the latent sizes.** With `--overrides model.agents.0.encoder_hidden_dims=[...]`
(or `agents.1`), the script reads the dims from the composed config and tags every
output with them (e.g. `d40x8`). Runs at different sizes therefore sit side by side
in the same folder. `mnist_parity_probes.py --dims 40x8` picks one size. Without
`--dims` it uses the size of the most recent run.

**Resuming.** Rows are appended to `runs.csv` as each run finishes. A
(method, λ, seed, epochs, pilots, dims) cell that already has rows and saved
latents is skipped, so an interrupted sweep can be restarted with the same
command.

## Outputs (`results/mnist_2agents_parity/`, git-ignored)

| File | Content |
|---|---|
| `runs.csv` | One row per (run, k) with all the metrics above |
| `latents/<method>[_lmb<λ>]_s<seed>_e<epochs>_p<pilots>_d<dims>.npz` | `z_fine`, `z_parity`, `y` (test split) |
| `knn_vs_k_d<dims>.{png,pdf}` | 10-class k-NN on agent 1 against k, with agent 0 as the dashed reference. With several λ, a second panel against λ |
| `probe_within_parity_d<dims>.csv`, `probe_few_label_d<dims>.csv` | Probe tables |
| `probes_d<dims>.{png,pdf}` | Within-parity k-NN against k, and few-label k-NN against references per class |

Files without a `_d<dims>` suffix come from older runs with d = 320/64.

## Reading the results

- Agent 1's k-NN 10-class accuracy sits between chance (0.10) and agent 0's
  reference curve. The gap to the reference shows how much digit identity the
  parity task throws away.
- The within-parity probe is the cleanest measure. Above 0.20, agent 1 still
  separates digits that its loss treats as the same class.
- The main comparison is `disheaf` against `noncoop` on these probes: does the
  sheaf coupling on the pilots move digit structure into agent 1's latents?
  Check alongside that `parity_acc_agent1` does not drop as λ grows.
