# Reconstruction: matched inpainting protocol (inpainting_v2)

The default `masked_cifar10_vae` configuration compares SheafFRL with independent
training plus a post-hoc Procrustes map. Both agents receive the same local pilot
supervision. `orchestrator.comm_task_coeff=0.0`: no cross-reconstruction objective
is optimized. The optional cross term is full-image MSE only, without a second KL.
It measures decoder compatibility, not fusion with a receiver observation.

## Data and masks

Private train/validation/test images are disjoint across agents. The pilot pool
is separate from those splits and shared by sample ID. Complete images are
supervised targets, including pixels missing from the input. This is supervised
inpainting, not learning from exclusively observed pixel targets.

Each of the two agents sees 65% of the image through 2x2 blocks. With a 32x32
image this gives a 16x16 grid: 166 of 256 blocks are visible after rounding,
or 64.84375% of the pixels. `constant_shared_visible_probability` is the
fraction of this fixed visible budget shared by both agents. Remaining visible
blocks are private and disjoint across agents. Changing sharedness therefore
changes the intersection and union while keeping each agent's budget fixed.

The sweep in `results/reconstruction/shared_sweeps/20260915_122825_596513`
used sharedness values 0.5, 0.6, 0.7, 0.8, 0.9 and 1.0. Exact block fractions
were:

| Sharedness | Common image | Private per agent | Two-agent union |
|---:|---:|---:|---:|
| 0.5 | 32.421875% | 32.421875% | 97.265625% |
| 0.6 | 39.062500% | 25.781250% | 90.625000% |
| 0.7 | 45.312500% | 19.531250% | 84.375000% |
| 0.8 | 51.953125% | 12.890625% | 77.734375% |
| 0.9 | 58.203125% | 6.640625% | 71.484375% |
| 1.0 | 64.843750% | 0.000000% | 64.843750% |

Training masks are reproducible from sample ID and epoch and are resampled at
every epoch (`mask_augmentation.resample_each_epoch=true`). Validation and test
use fixed masks. Pilot masks switch to fixed views for evaluation and back to
the training epoch for optimization. Mask state uses shared memory for loader
workers.

`dataset.private_train_fraction` subsamples the private training split only,
after the common split is established; pilots and evaluation sets stay fixed.
Small budgets must still supply a full batch per agent (training uses drop_last).

## VAE and objective

Input is RGB plus the visible-mask channel; the supervised target is the full
RGB image. The deterministic autoencoder uses latent dimension 256, hidden
dimensions `[128, 256, 512]`, posterior sampling disabled and `beta=0.0`.
`sigma_vae`, L1, SSIM, MMD, free bits and explicit model weight decay are
disabled. Local private and pilot objectives use full-image MSE plus the frozen
VGG perceptual loss with weight 0.01. `pilot_loss_weight=1.0` gives the matched
pilot supervision its full local-loss weight. The optional post-communication
training loss remains disabled (`comm_task_coeff=0.0`).

Sheaf-FRL uses learned SWBN whitening (`learn_whitening=true`) updated from
rotating pilot batches. It aligns all matched pilot anchors with an orthogonal
Procrustes/Stiefel map (`anchor_selection=all`, `use_general_maps=false`) after
10 warmup epochs. The Sheaf coefficient follows a cosine schedule up to 0.5.
Non-cooperative training uses no communication, then fits closed-form ZCA on
the final local training latents and a Procrustes map on matched pilots before
test. FedAvg uses synchronized model spaces and identity transport, without
post-hoc whitening or alignment.

## Run and ablate

From the repository root (activate the environment or use `.venv/bin/python`):

```bash
.venv/bin/python scripts/reconstruction_experiment.py
.venv/bin/python scripts/reconstruction_experiment.py orchestrator=non_cooperative
.venv/bin/python scripts/reconstruction_experiment.py -m orchestrator=sheaf_frl,non_cooperative seed=42,43,44 dataset.constant_shared_visible_probability=0.5,0.7,0.9
```

The default is already deterministic (`model.sample_posterior=false`,
`model.beta=0`). Bottleneck ablation: `model.latent_dim=64,128` in a multirun. Private-data ablation:
`dataset.private_train_fraction=0.1,0.25,1.0`. Map-family ablation: apply `orchestrator.use_general_maps=true
orchestrator.alignment_method=general` to both arms (the baseline ignores the
Sheaf-only map flag). Keep data/model settings matched and analyze map families
separately. These sweeps are commands, not pre-run results.

## Results

Parquet files under `results/reconstruction` include the full resolved YAML,
protocol version and comparison ID. Existing Hydra outputs are preserved.
Primary private metric: `private_mse_missing`. Communication metrics:
`comm_mse` (full image) and `comm_mse_sender_missing` (pixels missing to sender).
Legacy internal communication aliases still denote full-image MSE.

```bash
.venv/bin/python scripts/plot_reconstruction_overlap_metrics.py
```

The plotter selects inpainting_v2 runs and refuses to mix different comparison
IDs; use `--comparison-id ID` when several setups exist. It keeps the newest run
per method/sharedness/seed/agent, averages agents within each seed, then computes
mean and standard deviation across seeds. `--seed 42` selects a single seed (no
between-seed uncertainty). Test data are not monitored during fitting by default;
no reported test-based selection or performance improvement is implied.

## Independent loss ablation (30 epochs)

Prepare or launch the five runs (at most two concurrently on the same GPU) from the current configuration:

```bash
python scripts/run_reconstruction_loss_ablation.py --dry-run
python scripts/run_reconstruction_loss_ablation.py --gpu 1
```

The runs are baseline AE, VAE (`beta=0.0001`, posterior sampling enabled),
AE plus perceptual loss (`perceptual_weight=0.01`), AE with decoder
log-MSE calibration (`sigma_vae=true`), and baseline AE with `latent_dim=256`. These are separate variants, not their
Cartesian product. The VAE keeps the current KL normalization and warmup; the
other three loss/sampling settings are explicitly reset for each variant.
All other model/data settings come from the current configuration. Every run
uses non-cooperative training, cross-MSE zero, 30 epochs, and final testing.
The perceptual loss uses frozen pretrained early VGG16 features and compares
full images, so the perceptual variant is not supervised only on missing pixels.
The proposed coefficients are starting points, not validated optima.

W&B defaults to online with one shared group and variant-specific run names.
Use `--wandb-mode offline` to store runs for later synchronization. Final test
reconstructions are logged regardless of the periodic validation image cadence,
with masked input, mask, target, raw prediction and completed image.

The launcher prints its output directory under
`results/reconstruction/loss_ablations/`. It contains frozen composed configs,
a manifest with commands/status/result paths, one console log per run,
`per_agent.csv`, `per_agent.parquet`, and `summary.csv` (means across agents).
Individual experiment Parquet files remain under `results/reconstruction/`.
The comparison script for mask sweeps is intentionally not used here, because
these runs have different loss configurations.

Optional arguments include `--beta`, `--perceptual-weight`,
`--variants baseline perceptual`, and repeated common Hydra settings such as
`--override dataset.num_workers=2` or `--override seed=43`.
Use `--max-parallel 1` for sequential execution; the default is 2.
All children inherit the same GPU selection; a free slot is filled as soon as a
run finishes. Failed runs are recorded while the remaining queue continues.
The script prepares the configuration snapshots before starting the first run;
subsequent edits to the source YAML do not alter that sweep.

## FedAvg reconstruction control

### Shared-visibility sweep with three methods

```bash
.venv/bin/python scripts/run_reconstruction_shared_sweep.py --dry-run
.venv/bin/python scripts/run_reconstruction_shared_sweep.py --gpu 0
# Optional shorter run and multiple independent seeds:
.venv/bin/python scripts/run_reconstruction_shared_sweep.py --gpu 0 --epochs 30 \
  --shared-values 0.5 0.6 0.7 0.8 0.9 1.0 --seeds 42 43 44
```

The launcher freezes the current `config/hydra/masked_cifar10_vae.yaml` for
Sheaf-FRL, non-cooperative and federated. It preserves model, loss, visibility,
block size and epoch settings unless overridden (`--override KEY=VALUE` is
repeatable), and enables final testing. Each GPU runs one process by default.
Explicit GPU selection forces one device per training process. W&B defaults to
offline; use `--wandb-mode online` to upload.

```bash
# One run on each GPU concurrently (two total):
.venv/bin/python scripts/run_reconstruction_shared_sweep.py --gpu 0 1
# Two runs on each GPU concurrently (four total):
.venv/bin/python scripts/run_reconstruction_shared_sweep.py --gpu 0 1 --max-parallel 2
# Two runs on GPU 0 and one on GPU 1 (three total):
.venv/bin/python scripts/run_reconstruction_shared_sweep.py --gpu 0 1 --max-parallel 2 1
# Two runs sharing GPU 0:
.venv/bin/python scripts/run_reconstruction_shared_sweep.py --gpu 0 --max-parallel 2
```

`--max-parallel 1 2` reverses the asymmetric allocation. GPU indices refer to
physical devices via each child's `CUDA_VISIBLE_DEVICES`; inside a child the
selected device is CUDA 0. A common queue fills each slot as soon as it becomes
free, regardless of method. GPU assignments and process status are recorded in
the manifest. `--dry-run` prints capacities and saves configs without training.

The default shared fractions are 0.5 through 1.0, compatible with the current
two-agent visibility of 0.65. Sharedness is a fraction of each agent's **total
visible budget**, including shared and agent-exclusive blocks. At visibility
`v` and sharedness `s`, intersection is approximately `v*s` of the image and
the two-agent union is `v*(2-s)`. Thus `v=0.65, s=0.5` gives about 32.5%
intersection and 97.5% union. Exact counts use rounded blocks. Feasibility
requires approximately `s >= 2 - 1/v` for two agents (about 0.4615 here);
the launcher checks the actual block counts before launching any run.

Each timestamped folder in `results/reconstruction/shared_sweeps/` contains
frozen configs, a manifest, logs, per-agent results, `summary.csv`, and PNG/PDF
plots for `private_mse_full`, `private_mse_visible`, and `comm_mse` against
sharedness, plus `private_lpips_full` and `comm_lpips_full`. Agents are averaged within each seed; bands show standard deviation
across seeds when multiple seeds are provided. Only this sweep's result files
are collected. Failed runs remain recorded and cause a nonzero exit; any plots
produced from completed runs are therefore partial. FedAvg uses identity
transport, while the non-cooperative baseline retains post-hoc alignment.
This intentional difference gives FedAvg a different comparison ID, so this
launcher plots its manifest-selected results directly.

With reconstruction diagnostics enabled (the default), final test images are
logged to W&B as `test/masked_reconstruction_examples_agent_i` for private
reconstruction and `test/communication_reconstruction_sender_i_receiver_j`
for each directed communication edge. Communication grids use the exact
receiver predictions evaluated by the metrics, while post-hoc maps are still
active. Each row shows sender input, sender mask, target, receiver prediction,
and a display-only completion using sender-visible pixels. The raw prediction
is used for full-image MSE and LPIPS. Non-cooperative post-hoc evaluation also
computes private and communication LPIPS.

Use `--wandb-mode online` for immediate W&B upload. The default offline mode
stores W&B media locally for subsequent synchronization.

```bash
python scripts/reconstruction_experiment.py --config-name masked_cifar10_vae \
  orchestrator=federated logger.group=fedavg-reconstruction
```

This uses the current model, masks, loss and epoch count, including the same local
pilot supervision as non-cooperative and Sheaf training. Masked and full-image
reconstruction losses both work. The reconstruction entrypoint sets the effective
FedAvg `alignment_method` to null, overriding the shared config's Procrustes value;
this effective setting is saved in W&B and the result's `config_yaml`.

Before the first training update, all clients receive the first agent's initial
state. This synchronization is skipped when resuming after nonzero training
progress. Each epoch performs one round of uniform neighbor-plus-self averaging,
including BatchNorm buffers. With the default complete graph, all clients have
the same model after each aggregation. Test inference then transfers raw latents
with the identity map; no whitening or post-hoc map fitting is performed.
Identical architecture alone is not sufficient for identity transport: on a sparse
graph, neighborhood averages can differ. During local updates, clients can also
diverge until the next aggregation. Validation during fitting evaluates local
client models; the final test evaluates the final aggregated models.

The aggregation remains uniform, not sample-count weighted. The default balanced
private partition gives clients essentially equal sample counts. Adam/momentum
state is cleared for participating clients after each aggregation while retaining
learning rates; `orchestrator.reset_optimizer_on_aggregation=false` opts into
persistent local optimizer state. `orchestrator.synchronize_initial_weights=false`
opts out of the initial broadcast (not recommended for the FedAvg control).

Final reconstructions and Parquet metrics use the same entrypoint callbacks as
other methods. Under identical model states, receiver communication MSE equals
the sender's self-reconstruction MSE on the same examples, up to numerical error;
it need not equal the receiver's private MSE, evaluated on different images.

### LPIPS di test

La valutazione finale misura anche LPIPS sulla ricostruzione RGB completa del
 decoder (senza sostituire i pixel visibili), privata e dopo comunicazione.
Si usa il pacchetto ufficiale https://github.com/richzhang/PerceptualSimilarity,
`lpips==0.1.4`, backbone AlexNet e calibrazione LPIPS `version='0.1'`, congelati
in modalità eval. Le immagini mantengono la risoluzione originale; `normalize=True`
converte i valori da [0,1] a [-1,1]. Valori inferiori indicano maggiore somiglianza.

La metrica è indipendente da `perceptual_weight` e non entra nella loss, negli
optimizer o in FedAvg. Il primo utilizzo può scaricare i pesi AlexNet pretrained.
Non viene calcolata durante il monitoraggio della comunicazione a ogni epoca.
W&B registra `test/private_lpips_full_agent_i`, `test/comm_lpips_full_agent_i`,
`test/avg_private_lpips_full` e `test/avg_comm_lpips_full`. Ogni valore privato
è la media sulle immagini; quello comunicato è la media sui mittenti vicini.
I parquet e i riepiloghi delle ablation includono `private_lpips_full` e
`comm_lpips_full`; i risultati precedenti privi della metrica restano NaN.
