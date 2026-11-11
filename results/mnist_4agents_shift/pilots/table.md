Mean ± sample SD (%), seeds 42/43/44; shift=1, batch=256, 100 epochs.

| Metric | 384 Non-coop | 384 DiSheafFRL | 768 Non-coop | 768 DiSheafFRL | 1536 Non-coop | 1536 DiSheafFRL | 2048 Non-coop | 2048 DiSheafFRL | 4096 Non-coop | 4096 DiSheafFRL |
|---|---|---|---|---|---|---|---|---|---|---|
| Private | 69.56 ± 0.01 | 69.56 ± 0.03 | 69.57 ± 0.01 | 69.58 ± 0.01 | 69.57 ± 0.01 | 69.58 ± 0.00 | 69.56 ± 0.02 | 69.58 ± 0.01 | 69.56 ± 0.01 | 69.57 ± 0.01 |
| Private seen | 99.58 ± 0.01 | 99.58 ± 0.04 | 99.59 ± 0.01 | 99.61 ± 0.02 | 99.60 ± 0.01 | 99.61 ± 0.00 | 99.59 ± 0.03 | 99.61 ± 0.02 | 99.58 ± 0.02 | 99.60 ± 0.01 |
| Private unseen | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Comm | 64.78 ± 0.24 | 68.79 ± 0.21 | 65.21 ± 0.17 | 69.15 ± 0.01 | 65.60 ± 0.38 | 69.27 ± 0.01 | 65.58 ± 0.44 | 69.28 ± 0.03 | 65.78 ± 0.29 | 69.29 ± 0.03 |
| Comm sender unseen / receiver seen | 82.19 ± 1.09 | 96.95 ± 0.75 | 83.62 ± 0.68 | 98.36 ± 0.04 | 85.01 ± 1.41 | 98.81 ± 0.05 | 84.95 ± 1.57 | 98.91 ± 0.02 | 85.70 ± 1.05 | 98.99 ± 0.07 |
| Comm sender seen / receiver unseen | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

Scopes pool correct predictions and sample counts across four agents or twelve directed edges within each seed. Uncertainty is across seeds, not edges.

All 30 checkpoint totals match remote W&B test metrics within 0.002 percentage points.

Run IDs: x9u62pd7, dsnopa19, c8ix7vn9, 7x9j9uyl, cndm7lcm, 7zvxszuc, d08ea2jl, cs2jq075, ql07v6js, qlgjhu1p, 0qqr67v1, 2qa8ryfj, zkm2chcx, 81atmtj6, 3jh19vzq, i7c3d3s0, pqj5vh38, py0kxr69, ofnckann, x56k3su1, t7kwcuje, vwrpvki1, yswqrnqg, fftod1dm, 0344yx97, 3jm8q65s, kqxi7mvx, 90dq9z2e, 77f2a026, y42wgwuz
