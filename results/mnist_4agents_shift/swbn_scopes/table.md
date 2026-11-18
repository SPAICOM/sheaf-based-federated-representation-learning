Test accuracy (%), mean ± sample SD over seeds 42/43/44; lambda .01, batch 256, 100 epochs, shift 1.
Scopes pool the 12 directed edges within a seed. Eval: posthoc Procrustes for non-coop / CenterNorm / ZCA (as reported remotely); learned SWBN + saved Stiefel maps for SWBN, plus SWBN posthoc as reference.
CenterNorm (learned): same checkpoints as CenterNorm through the Stiefel maps saved in training (offline, no remote reference). CenterNorm + OLS: saved OLS maps; Comm from the remote saved-map eval runs (results/mnist_4agents_shift/maps/pilot_summary.csv), seen scopes rescored from the checkpoints.

### Private

| Pilots | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS (posthoc) | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|---|
| 384 | 69.56 ± 0.01 | 69.56 ± 0.03 | 67.25 ± 4.00 | 69.57 ± 0.01 | 69.57 ± 0.01 | 69.59 ± 0.00 | 69.59 ± 0.00 |
| 768 | 69.57 ± 0.01 | 69.58 ± 0.01 | 65.97 ± 3.11 | 69.58 ± 0.02 | 69.58 ± 0.02 | 69.58 ± 0.02 | 69.58 ± 0.02 |
| 1536 | 69.57 ± 0.01 | 69.58 ± 0.00 | 69.15 ± 0.27 | 69.56 ± 0.02 | 69.56 ± 0.02 | 69.59 ± 0.01 | 69.59 ± 0.01 |
| 2048 | 69.56 ± 0.02 | 69.58 ± 0.01 | 69.20 ± 0.31 | 69.57 ± 0.02 | 69.57 ± 0.02 | 69.57 ± 0.00 | 69.57 ± 0.00 |
| 4096 | 69.56 ± 0.01 | 69.57 ± 0.01 | 69.28 ± 0.28 | 69.56 ± 0.01 | 69.56 ± 0.01 | 69.59 ± 0.01 | 69.59 ± 0.01 |

### Private target classes

| Pilots | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS (posthoc) | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|---|
| 384 | 99.58 ± 0.01 | 99.58 ± 0.04 | 96.28 ± 5.73 | 99.59 ± 0.02 | 99.59 ± 0.02 | 99.62 ± 0.00 | 99.62 ± 0.00 |
| 768 | 99.59 ± 0.01 | 99.61 ± 0.02 | 94.44 ± 4.46 | 99.61 ± 0.03 | 99.61 ± 0.03 | 99.61 ± 0.03 | 99.61 ± 0.03 |
| 1536 | 99.60 ± 0.01 | 99.61 ± 0.00 | 98.99 ± 0.39 | 99.58 ± 0.03 | 99.58 ± 0.03 | 99.62 ± 0.02 | 99.62 ± 0.02 |
| 2048 | 99.59 ± 0.03 | 99.61 ± 0.02 | 99.07 ± 0.45 | 99.60 ± 0.03 | 99.60 ± 0.03 | 99.60 ± 0.01 | 99.60 ± 0.01 |
| 4096 | 99.58 ± 0.02 | 99.60 ± 0.01 | 99.18 ± 0.40 | 99.58 ± 0.02 | 99.58 ± 0.02 | 99.62 ± 0.01 | 99.62 ± 0.01 |

### Comm

| Pilots | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS (posthoc) | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|---|
| 384 | 64.78 ± 0.24 | 68.79 ± 0.21 | 62.82 ± 2.33 | 67.60 ± 0.08 | 52.08 ± 0.87 | 68.70 ± 0.14 | 65.67 ± 0.99 |
| 768 | 65.21 ± 0.17 | 69.15 ± 0.01 | 59.75 ± 3.13 | 68.45 ± 0.09 | 58.85 ± 0.13 | 69.11 ± 0.02 | 67.89 ± 0.05 |
| 1536 | 65.60 ± 0.38 | 69.27 ± 0.01 | 60.48 ± 1.06 | 68.91 ± 0.03 | 62.36 ± 5.32 | 68.37 ± 1.12 | 68.54 ± 0.10 |
| 2048 | 65.58 ± 0.44 | 69.28 ± 0.03 | 62.26 ± 1.23 | 68.95 ± 0.03 | 59.31 ± 0.02 | 69.15 ± 0.19 | 68.27 ± 0.46 |
| 4096 | 65.78 ± 0.29 | 69.29 ± 0.03 | 60.89 ± 4.68 | 69.05 ± 0.03 | 65.49 ± 4.65 | 68.21 ± 0.87 | 68.26 ± 0.56 |

### Comm sender unseen / receiver seen

| Pilots | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS (posthoc) | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|---|
| 384 | 82.19 ± 1.09 | 96.95 ± 0.75 | 78.31 ± 5.96 | 92.71 ± 0.25 | 72.02 ± 1.80 | 97.06 ± 0.40 | 87.43 ± 1.98 |
| 768 | 83.62 ± 0.68 | 98.36 ± 0.04 | 72.57 ± 4.81 | 95.83 ± 0.19 | 84.93 ± 0.27 | 98.24 ± 0.06 | 94.05 ± 0.32 |
| 1536 | 85.01 ± 1.41 | 98.81 ± 0.05 | 74.18 ± 1.68 | 97.56 ± 0.15 | 89.97 ± 6.54 | 97.14 ± 1.87 | 96.21 ± 0.40 |
| 2048 | 84.95 ± 1.57 | 98.91 ± 0.02 | 77.23 ± 0.71 | 97.71 ± 0.11 | 86.32 ± 0.08 | 98.54 ± 0.31 | 95.87 ± 1.53 |
| 4096 | 85.70 ± 1.05 | 98.99 ± 0.07 | 75.33 ± 6.86 | 98.05 ± 0.13 | 93.02 ± 8.28 | 96.77 ± 1.59 | 95.89 ± 1.62 |

### Comm receiver seen

| Pilots | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS (posthoc) | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|---|
| 384 | 92.75 ± 0.35 | 98.48 ± 0.31 | 89.93 ± 3.33 | 96.78 ± 0.11 | 74.54 ± 1.23 | 98.35 ± 0.20 | 94.01 ± 1.42 |
| 768 | 93.35 ± 0.24 | 99.00 ± 0.01 | 85.53 ± 4.49 | 98.00 ± 0.13 | 84.26 ± 0.18 | 98.94 ± 0.03 | 97.19 ± 0.07 |
| 1536 | 93.92 ± 0.54 | 99.17 ± 0.02 | 86.59 ± 1.52 | 98.66 ± 0.05 | 89.28 ± 7.62 | 97.89 ± 1.60 | 98.12 ± 0.15 |
| 2048 | 93.89 ± 0.63 | 99.18 ± 0.04 | 89.14 ± 1.76 | 98.71 ± 0.04 | 84.90 ± 0.03 | 99.00 ± 0.28 | 97.74 ± 0.66 |
| 4096 | 94.17 ± 0.41 | 99.20 ± 0.04 | 87.17 ± 6.71 | 98.86 ± 0.05 | 93.76 ± 6.66 | 97.66 ± 1.25 | 97.72 ± 0.81 |

### Comm sender seen / receiver seen

| Pilots | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS (posthoc) | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|---|
| 384 | 99.29 ± 0.11 | 99.43 ± 0.03 | 97.13 ± 1.73 | 99.30 ± 0.16 | 76.10 ± 0.88 | 99.15 ± 0.09 | 98.09 ± 1.14 |
| 768 | 99.38 ± 0.06 | 99.39 ± 0.01 | 93.56 ± 5.24 | 99.34 ± 0.10 | 83.84 ± 0.12 | 99.38 ± 0.02 | 99.13 ± 0.09 |
| 1536 | 99.44 ± 0.05 | 99.39 ± 0.00 | 94.28 ± 1.72 | 99.33 ± 0.02 | 88.84 ± 8.28 | 98.35 ± 1.43 | 99.30 ± 0.02 |
| 2048 | 99.43 ± 0.05 | 99.35 ± 0.05 | 96.51 ± 2.95 | 99.34 ± 0.01 | 84.03 ± 0.01 | 99.29 ± 0.26 | 98.90 ± 0.40 |
| 4096 | 99.42 ± 0.03 | 99.34 ± 0.03 | 94.50 ± 6.63 | 99.36 ± 0.01 | 94.22 ± 5.66 | 98.21 ± 1.04 | 98.86 ± 0.31 |

Run IDs: DiSheafFRL + ZCA/384.0/43.0=yg8ocrdq, DiSheafFRL + ZCA/384.0/42.0=iie82r8o, DiSheafFRL + ZCA/384.0/44.0=punj852j, DiSheafFRL + ZCA/768.0/43.0=lqpyr37p, DiSheafFRL + ZCA/768.0/44.0=2afhs6z9, DiSheafFRL + ZCA/768.0/42.0=2gqe8i8g, DiSheafFRL + ZCA/1536.0/42.0=0hddrnny, DiSheafFRL + ZCA/1536.0/43.0=l4csg60y, DiSheafFRL + ZCA/1536.0/44.0=0ddl333l, DiSheafFRL + ZCA/2048.0/42.0=czlg5hul, DiSheafFRL + ZCA/2048.0/43.0=ifvlsss4, DiSheafFRL + ZCA/2048.0/44.0=pfqx2rnn, DiSheafFRL + ZCA/4096.0/42.0=206rghx6, Non-coop/384.0/44.0=d08ea2jl, Non-coop/384.0/43.0=0qqr67v1, Non-coop/384.0/42.0=qlgjhu1p, Non-coop/768.0/42.0=zkm2chcx, Non-coop/768.0/44.0=81atmtj6, Non-coop/768.0/43.0=3jh19vzq, Non-coop/1536.0/42.0=0344yx97, Non-coop/1536.0/44.0=3jm8q65s, Non-coop/1536.0/43.0=kqxi7mvx, Non-coop/2048.0/43.0=ofnckann, Non-coop/2048.0/42.0=x56k3su1, Non-coop/2048.0/44.0=t7kwcuje, Non-coop/4096.0/43.0=x9u62pd7, Non-coop/4096.0/44.0=dsnopa19, Non-coop/4096.0/42.0=c8ix7vn9, DiSheafFRL + CenterNorm/384.0/44.0=2qa8ryfj, DiSheafFRL + CenterNorm/384.0/42.0=cs2jq075, DiSheafFRL + CenterNorm/384.0/43.0=ql07v6js, DiSheafFRL + CenterNorm/768.0/43.0=i7c3d3s0, DiSheafFRL + CenterNorm/768.0/42.0=pqj5vh38, DiSheafFRL + CenterNorm/768.0/44.0=py0kxr69, DiSheafFRL + CenterNorm/1536.0/42.0=90dq9z2e, DiSheafFRL + CenterNorm/1536.0/43.0=77f2a026, DiSheafFRL + CenterNorm/1536.0/44.0=y42wgwuz, DiSheafFRL + CenterNorm/2048.0/43.0=vwrpvki1, DiSheafFRL + CenterNorm/2048.0/42.0=yswqrnqg, DiSheafFRL + CenterNorm/2048.0/44.0=fftod1dm, DiSheafFRL + CenterNorm/4096.0/43.0=7x9j9uyl, DiSheafFRL + CenterNorm/4096.0/44.0=7zvxszuc, DiSheafFRL + CenterNorm/4096.0/42.0=cndm7lcm, DiSheafFRL + ZCA/4096.0/43.0=94j0envu, DiSheafFRL + ZCA/4096.0/44.0=rxkkjdmi, DiSheafFRL + SWBN/384.0/42.0=6fvz2uhq, DiSheafFRL + SWBN (posthoc)/384.0/42.0=6fvz2uhq, DiSheafFRL + SWBN/384.0/44.0=l2y3q5av, DiSheafFRL + SWBN (posthoc)/384.0/44.0=l2y3q5av, DiSheafFRL + SWBN/384.0/43.0=yky7whz2, DiSheafFRL + SWBN (posthoc)/384.0/43.0=yky7whz2, DiSheafFRL + SWBN/768.0/42.0=y9ju88dv, DiSheafFRL + SWBN (posthoc)/768.0/42.0=y9ju88dv, DiSheafFRL + SWBN/768.0/43.0=bkg1w8mh, DiSheafFRL + SWBN (posthoc)/768.0/43.0=bkg1w8mh, DiSheafFRL + SWBN/768.0/44.0=svtbhhpp, DiSheafFRL + SWBN (posthoc)/768.0/44.0=svtbhhpp, DiSheafFRL + SWBN/1536.0/42.0=ze5kezu1, DiSheafFRL + SWBN (posthoc)/1536.0/42.0=ze5kezu1, DiSheafFRL + SWBN/1536.0/43.0=ztilv7h0, DiSheafFRL + SWBN (posthoc)/1536.0/43.0=ztilv7h0, DiSheafFRL + SWBN/1536.0/44.0=6kdnwv44, DiSheafFRL + SWBN (posthoc)/1536.0/44.0=6kdnwv44, DiSheafFRL + SWBN/2048.0/42.0=2a6uo96b, DiSheafFRL + SWBN (posthoc)/2048.0/42.0=2a6uo96b, DiSheafFRL + SWBN/2048.0/43.0=gtcrvpvq, DiSheafFRL + SWBN (posthoc)/2048.0/43.0=gtcrvpvq, DiSheafFRL + SWBN/2048.0/44.0=7yz1sqml, DiSheafFRL + SWBN (posthoc)/2048.0/44.0=7yz1sqml, DiSheafFRL + SWBN/4096.0/42.0=3n1cirhj, DiSheafFRL + SWBN (posthoc)/4096.0/42.0=3n1cirhj, DiSheafFRL + SWBN/4096.0/43.0=y1ui2qws, DiSheafFRL + SWBN (posthoc)/4096.0/43.0=y1ui2qws, DiSheafFRL + SWBN/4096.0/44.0=ahi7288n, DiSheafFRL + SWBN (posthoc)/4096.0/44.0=ahi7288n, DiSheafFRL + CenterNorm + OLS (posthoc)/384.0/42.0=pwjvaqx1, DiSheafFRL + CenterNorm + OLS (posthoc)/384.0/43.0=q8y3rv4j, DiSheafFRL + CenterNorm + OLS (posthoc)/384.0/44.0=zp4ea8r3, DiSheafFRL + CenterNorm + OLS (posthoc)/768.0/42.0=b5lx8f49, DiSheafFRL + CenterNorm + OLS (posthoc)/768.0/43.0=0w8f5548, DiSheafFRL + CenterNorm + OLS (posthoc)/768.0/44.0=0rikzqub, DiSheafFRL + CenterNorm + OLS (posthoc)/1536.0/42.0=tdkvyn6t, DiSheafFRL + CenterNorm + OLS (posthoc)/1536.0/43.0=64a1u8qe, DiSheafFRL + CenterNorm + OLS (posthoc)/1536.0/44.0=qco12x4u, DiSheafFRL + CenterNorm + OLS (posthoc)/2048.0/42.0=zacu0kzy, DiSheafFRL + CenterNorm + OLS (posthoc)/2048.0/43.0=w5u5zidx, DiSheafFRL + CenterNorm + OLS (posthoc)/2048.0/44.0=gvtx4lm1, DiSheafFRL + CenterNorm + OLS (posthoc)/4096.0/42.0=hjx4w7x9, DiSheafFRL + CenterNorm + OLS (posthoc)/4096.0/43.0=5dpijm43, DiSheafFRL + CenterNorm + OLS (posthoc)/4096.0/44.0=jm3v8o5g, DiSheafFRL + CenterNorm + OLS/384.0/42.0=pwjvaqx1, DiSheafFRL + CenterNorm + OLS/384.0/43.0=q8y3rv4j, DiSheafFRL + CenterNorm + OLS/384.0/44.0=zp4ea8r3, DiSheafFRL + CenterNorm + OLS/768.0/42.0=b5lx8f49, DiSheafFRL + CenterNorm + OLS/768.0/43.0=0w8f5548, DiSheafFRL + CenterNorm + OLS/768.0/44.0=0rikzqub, DiSheafFRL + CenterNorm + OLS/1536.0/42.0=tdkvyn6t, DiSheafFRL + CenterNorm + OLS/1536.0/43.0=64a1u8qe, DiSheafFRL + CenterNorm + OLS/1536.0/44.0=qco12x4u, DiSheafFRL + CenterNorm + OLS/2048.0/42.0=zacu0kzy, DiSheafFRL + CenterNorm + OLS/2048.0/43.0=w5u5zidx, DiSheafFRL + CenterNorm + OLS/2048.0/44.0=gvtx4lm1, DiSheafFRL + CenterNorm + OLS/4096.0/42.0=hjx4w7x9, DiSheafFRL + CenterNorm + OLS/4096.0/43.0=5dpijm43, DiSheafFRL + CenterNorm + OLS/4096.0/44.0=jm3v8o5g
