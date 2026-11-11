# Retained MNIST four-agent studies

Multi-seed results: seeds 42/43/44; four heterogeneous CNNs, shift strength 1,
100 epochs, training batch 256, pilot batch 128, exclusive all-class pool
pilots. Communication accuracy pools twelve directed edges; errors are sample
standard deviations across seeds. Evaluation fits posthoc Procrustes maps.

- [Pilot scope table](pilots/table.md): 384/768/1536/2048/4096 pilots;
  noncooperative baseline versus CenterNorm DiSheafFRL at lambda .01.
- [Lambda aggregates](lambda/summary.csv): 4096 pilots, lambda
  .003/.01/.03/.1/.3/1, private and communication scores by agent and average.
- [Combined pilot comparison](maps/pilot_summary.md) and
  [lambda comparison](maps/lambda_summary.md): noncoop, CenterNorm, ZCA, OLS.
- [ZCA provenance](maps/zca_provenance.md) and
  [OLS provenance](maps/ols_provenance.md): remote run IDs and seeds.
- Each study directory includes PDF and PNG figures. Poor completed map
  settings remain in the curves; selecting only high scores would bias them.

## Sources and regeneration

| Promoted script | Original successful script | Recipe |
|---|---|---|
| `scripts/plot_mnist4_pilots.py` | `tmp/2026-11-08_mnist4_table.py` | `just mnist4-pilot-summary` |
| `scripts/plot_mnist4_lambda.py` | `tmp/2026-11-09_mnist4_extension_report.py` | `just mnist4-lambda-summary` |
| `scripts/plot_mnist4_maps.py` | `tmp/2026-11-11_mnist4_ols_overlay.py` (without p128) | `just mnist4-map-summary` |

The pilot summary is the retained remote-verified checkpoint rescore; rerunning
it requires the 30 saved final checkpoints and CUDA. Its overall totals were
verified against W&B to within .002 percentage points. The lambda and map
recipes refresh remote W&B metrics; map plots read the retained baseline
aggregates. The lambda baseline is now read directly from W&B instead of
reusing the rounded checkpoint-derived pilot totals.

Training configs: `mnist_2agents_7v7.yaml` (shared inherited protocol),
`mnist_4agents_shift.yaml`, and `mnist_4agents_mapstudy_{base,zca,ols}.yaml`.
Training entry point: `scripts/multi_agent_experiment.py`; use the direct
`just mnist-4agents` / `just mnist4-mapstudy` commands documented in the root
README. No parallel queue or shell wrapper is required.

The CenterNorm lambda sweep (18 trained settings/seeds plus three baseline
runs) and both 30-run map sweeps were rechecked remotely on 2026-11-11.
The original ledger's OLS entry was stale at 26/30; all 30 are now finished.
Pilot evidence is recorded in the valid completed pilot and 1536 rows.
The separate two-agent study has only two seeds and remains preliminary.
Historical implementation revisions were dirty (7972e4b / 552cb0e).

Not included: p128, SimCLR, shift .8/.9, BatchNorm ablation, learned-map
rescoring, failed/cancelled 1408/3072 trials, or the abandoned batch-128 runs.
