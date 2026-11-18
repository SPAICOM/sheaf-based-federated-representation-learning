Test accuracy (%), mean ± sample SD over seeds 42/43/44; 4096 pilots (non-coop lambda-free, repeated), batch 256, 100 epochs, shift 1.
Scopes pool the 12 directed edges within a seed. Eval: posthoc Procrustes for non-coop / CenterNorm / ZCA (as reported remotely); learned SWBN + saved Stiefel maps for SWBN, plus SWBN posthoc as reference.
CenterNorm (learned): same checkpoints as CenterNorm through the Stiefel maps saved in training (offline, no remote reference). CenterNorm + OLS: saved OLS maps; Comm from the remote saved-map eval runs (results/mnist_4agents_shift/maps/lambda_summary.csv), seen scopes rescored from the checkpoints.

### Private

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|
| 0.003 | 69.56 ± 0.01 | 69.59 ± 0.02 | 69.41 ± 0.25 | 69.57 ± 0.03 | 69.59 ± 0.01 | 69.59 ± 0.01 |
| 0.01 | 69.56 ± 0.01 | 69.57 ± 0.01 | 69.28 ± 0.28 | 69.56 ± 0.01 | 69.59 ± 0.01 | 69.59 ± 0.01 |
| 0.03 | 69.56 ± 0.01 | 69.53 ± 0.03 | 68.81 ± 0.60 | 69.51 ± 0.03 | 69.50 ± 0.09 | 69.50 ± 0.09 |
| 0.1 | 69.56 ± 0.01 | 69.08 ± 0.46 | 64.34 ± 8.22 | 65.89 ± 4.90 | 69.21 ± 0.08 | 69.21 ± 0.08 |
| 0.3 | 69.56 ± 0.01 | 63.39 ± 2.24 | 66.66 ± 1.19 | 69.04 ± 0.48 | 66.31 ± 0.36 | 66.31 ± 0.36 |
| 1.0 | 69.56 ± 0.01 | 60.14 ± 2.97 | 66.65 ± 2.32 | 68.77 ± 0.36 | 56.78 ± 0.50 | 56.78 ± 0.50 |

### Private target classes

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|
| 0.003 | 99.58 ± 0.02 | 99.62 ± 0.02 | 99.37 ± 0.36 | 99.60 ± 0.04 | 99.63 ± 0.01 | 99.63 ± 0.01 |
| 0.01 | 99.58 ± 0.02 | 99.60 ± 0.01 | 99.18 ± 0.40 | 99.58 ± 0.02 | 99.62 ± 0.01 | 99.62 ± 0.01 |
| 0.03 | 99.58 ± 0.02 | 99.54 ± 0.04 | 98.51 ± 0.86 | 99.51 ± 0.04 | 99.50 ± 0.13 | 99.50 ± 0.13 |
| 0.1 | 99.58 ± 0.02 | 98.90 ± 0.66 | 92.11 ± 11.77 | 94.32 ± 7.02 | 99.09 ± 0.11 | 99.09 ± 0.11 |
| 0.3 | 99.58 ± 0.02 | 90.75 ± 3.21 | 95.43 ± 1.71 | 98.84 ± 0.69 | 94.94 ± 0.51 | 94.94 ± 0.51 |
| 1.0 | 99.58 ± 0.02 | 86.10 ± 4.25 | 95.42 ± 3.32 | 98.46 ± 0.52 | 81.28 ± 0.71 | 81.28 ± 0.71 |

### Comm

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|
| 0.003 | 65.78 ± 0.29 | 69.19 ± 0.03 | 63.28 ± 1.31 | 65.24 ± 5.13 | 69.19 ± 0.02 | 68.00 ± 0.29 |
| 0.01 | 65.78 ± 0.29 | 69.29 ± 0.03 | 60.89 ± 4.68 | 65.49 ± 4.65 | 68.21 ± 0.87 | 68.26 ± 0.56 |
| 0.03 | 65.78 ± 0.29 | 69.31 ± 0.03 | 61.80 ± 0.69 | 64.98 ± 4.89 | 65.85 ± 2.11 | 67.18 ± 0.60 |
| 0.1 | 65.78 ± 0.29 | 68.83 ± 0.39 | 36.11 ± 7.45 | 49.02 ± 7.46 | 64.89 ± 6.28 | 65.08 ± 1.04 |
| 0.3 | 65.78 ± 0.29 | 60.28 ± 2.16 | 40.08 ± 7.34 | 44.08 ± 5.64 | 62.81 ± 1.10 | 58.87 ± 1.66 |
| 1.0 | 65.78 ± 0.29 | 53.88 ± 3.44 | 38.24 ± 10.25 | 50.39 ± 10.47 | 47.59 ± 2.76 | 46.15 ± 2.71 |

### Comm sender unseen / receiver seen

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|
| 0.003 | 85.70 ± 1.05 | 98.46 ± 0.10 | 79.70 ± 1.95 | 93.62 ± 6.34 | 98.49 ± 0.12 | 94.75 ± 0.59 |
| 0.01 | 85.70 ± 1.05 | 98.99 ± 0.07 | 75.33 ± 6.86 | 93.02 ± 8.28 | 96.77 ± 1.59 | 95.89 ± 1.62 |
| 0.03 | 85.70 ± 1.05 | 99.17 ± 0.07 | 77.55 ± 2.10 | 93.57 ± 6.03 | 93.57 ± 2.76 | 95.19 ± 0.72 |
| 0.1 | 85.70 ± 1.05 | 98.26 ± 0.80 | 42.87 ± 6.60 | 70.78 ± 10.36 | 92.52 ± 8.60 | 92.11 ± 1.12 |
| 0.3 | 85.70 ± 1.05 | 81.22 ± 2.87 | 44.69 ± 14.26 | 59.88 ± 5.77 | 89.00 ± 2.16 | 81.55 ± 2.27 |
| 1.0 | 85.70 ± 1.05 | 68.49 ± 5.51 | 47.77 ± 12.36 | 71.58 ± 15.59 | 66.37 ± 4.38 | 67.51 ± 4.23 |

### Comm receiver seen

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|
| 0.003 | 94.17 ± 0.41 | 99.05 ± 0.04 | 90.59 ± 1.88 | 93.39 ± 7.35 | 99.06 ± 0.03 | 97.35 ± 0.41 |
| 0.01 | 94.17 ± 0.41 | 99.20 ± 0.04 | 87.17 ± 6.71 | 93.76 ± 6.66 | 97.66 ± 1.25 | 97.72 ± 0.81 |
| 0.03 | 94.17 ± 0.41 | 99.22 ± 0.04 | 88.47 ± 0.98 | 93.03 ± 7.01 | 94.28 ± 3.03 | 96.17 ± 0.87 |
| 0.1 | 94.17 ± 0.41 | 98.54 ± 0.56 | 51.69 ± 10.66 | 70.19 ± 10.67 | 92.90 ± 8.99 | 93.17 ± 1.49 |
| 0.3 | 94.17 ± 0.41 | 86.29 ± 3.09 | 57.38 ± 10.51 | 63.10 ± 8.08 | 89.92 ± 1.57 | 84.28 ± 2.38 |
| 1.0 | 94.17 ± 0.41 | 77.14 ± 4.92 | 54.74 ± 14.67 | 72.13 ± 15.00 | 68.14 ± 3.95 | 66.07 ± 3.88 |

### Comm sender seen / receiver seen

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS | DiSheafFRL + SWBN | DiSheafFRL + SWBN (posthoc) |
|---|---|---|---|---|---|---|
| 0.003 | 99.42 ± 0.03 | 99.42 ± 0.02 | 97.33 ± 2.12 | 93.25 ± 7.98 | 99.41 ± 0.03 | 98.96 ± 0.31 |
| 0.01 | 99.42 ± 0.03 | 99.34 ± 0.03 | 94.50 ± 6.63 | 94.22 ± 5.66 | 98.21 ± 1.04 | 98.86 ± 0.31 |
| 0.03 | 99.42 ± 0.03 | 99.25 ± 0.02 | 95.23 ± 0.44 | 92.70 ± 7.62 | 94.72 ± 3.19 | 96.78 ± 1.08 |
| 0.1 | 99.42 ± 0.03 | 98.71 ± 0.41 | 57.16 ± 13.73 | 69.82 ± 10.87 | 93.13 ± 9.24 | 93.83 ± 1.72 |
| 0.3 | 99.42 ± 0.03 | 89.44 ± 3.28 | 65.24 ± 8.20 | 65.10 ± 9.52 | 90.49 ± 1.22 | 85.98 ± 3.43 |
| 1.0 | 99.42 ± 0.03 | 82.49 ± 4.58 | 59.06 ± 16.76 | 72.47 ± 14.65 | 69.23 ± 3.71 | 65.18 ± 3.81 |

Run IDs: Non-coop/nan/43.0=x9u62pd7, Non-coop/nan/44.0=dsnopa19, Non-coop/nan/42.0=c8ix7vn9, DiSheafFRL + CenterNorm/0.01/43.0=7x9j9uyl, DiSheafFRL + CenterNorm/0.01/44.0=7zvxszuc, DiSheafFRL + CenterNorm/0.01/42.0=cndm7lcm, DiSheafFRL + CenterNorm/0.003/44.0=w9sqx14x, DiSheafFRL + CenterNorm/0.003/42.0=pg3drg2e, DiSheafFRL + CenterNorm/0.03/43.0=id358klz, DiSheafFRL + CenterNorm/0.003/43.0=dv9wdwou, DiSheafFRL + CenterNorm/0.03/42.0=dwmlanm8, DiSheafFRL + CenterNorm/0.03/44.0=4oft73b5, DiSheafFRL + CenterNorm/0.1/42.0=3o8lowo2, DiSheafFRL + CenterNorm/0.3/43.0=kglyukwl, DiSheafFRL + CenterNorm/0.1/44.0=ghj8sjya, DiSheafFRL + CenterNorm/0.3/42.0=xcns8p91, DiSheafFRL + CenterNorm/0.1/43.0=fb45eucn, DiSheafFRL + CenterNorm/0.3/44.0=hkt2jzt0, DiSheafFRL + CenterNorm/1.0/44.0=4ajshdm6, DiSheafFRL + CenterNorm/1.0/42.0=vdf9zm0u, DiSheafFRL + CenterNorm/1.0/43.0=lew4oqwm, DiSheafFRL + ZCA/0.01/42.0=206rghx6, DiSheafFRL + ZCA/0.01/43.0=94j0envu, DiSheafFRL + ZCA/0.01/44.0=rxkkjdmi, DiSheafFRL + ZCA/0.003/42.0=rwckbvlu, DiSheafFRL + ZCA/0.003/43.0=h2qtxmwl, DiSheafFRL + ZCA/0.003/44.0=l34igjyy, DiSheafFRL + ZCA/0.03/42.0=9ib5zwlm, DiSheafFRL + ZCA/0.03/43.0=indu3dl4, DiSheafFRL + ZCA/0.03/44.0=xl1gbb62, DiSheafFRL + ZCA/0.1/42.0=mjbmxbo1, DiSheafFRL + ZCA/0.1/43.0=racb0zwv, DiSheafFRL + ZCA/0.1/44.0=xbpsllm1, DiSheafFRL + ZCA/0.3/42.0=yt3jlx94, DiSheafFRL + ZCA/0.3/43.0=95qhgcye, DiSheafFRL + ZCA/0.3/44.0=0u8y8uu0, DiSheafFRL + ZCA/1.0/42.0=khjatck2, DiSheafFRL + ZCA/1.0/43.0=s46w78eu, DiSheafFRL + ZCA/1.0/44.0=0jzlmvyn, DiSheafFRL + CenterNorm + OLS/0.01/42.0=hjx4w7x9, DiSheafFRL + CenterNorm + OLS/0.01/43.0=5dpijm43, DiSheafFRL + CenterNorm + OLS/0.01/44.0=jm3v8o5g, DiSheafFRL + CenterNorm + OLS/0.003/42.0=avulh2rm, DiSheafFRL + CenterNorm + OLS/0.003/43.0=w6ocus3u, DiSheafFRL + CenterNorm + OLS/0.003/44.0=ysiok61f, DiSheafFRL + CenterNorm + OLS/0.03/42.0=y4gr8omv, DiSheafFRL + CenterNorm + OLS/0.03/43.0=stiemhx6, DiSheafFRL + CenterNorm + OLS/0.03/44.0=s7njait6, DiSheafFRL + CenterNorm + OLS/0.1/42.0=acc2dmeg, DiSheafFRL + CenterNorm + OLS/0.1/43.0=jbi0y1r7, DiSheafFRL + CenterNorm + OLS/0.1/44.0=u7azm1x5, DiSheafFRL + CenterNorm + OLS/0.3/42.0=xyge13wc, DiSheafFRL + CenterNorm + OLS/0.3/43.0=q3dppdpb, DiSheafFRL + CenterNorm + OLS/0.3/44.0=eki926tm, DiSheafFRL + CenterNorm + OLS/1.0/42.0=peepk9xw, DiSheafFRL + CenterNorm + OLS/1.0/43.0=omyaum06, DiSheafFRL + CenterNorm + OLS/1.0/44.0=w8gbejot, DiSheafFRL + SWBN/0.01/42.0=3n1cirhj, DiSheafFRL + SWBN (posthoc)/0.01/42.0=3n1cirhj, DiSheafFRL + SWBN/0.01/43.0=y1ui2qws, DiSheafFRL + SWBN (posthoc)/0.01/43.0=y1ui2qws, DiSheafFRL + SWBN/0.01/44.0=ahi7288n, DiSheafFRL + SWBN (posthoc)/0.01/44.0=ahi7288n, DiSheafFRL + SWBN/0.03/42.0=31k2ax63, DiSheafFRL + SWBN (posthoc)/0.03/42.0=31k2ax63, DiSheafFRL + SWBN/0.003/44.0=4ewdrour, DiSheafFRL + SWBN (posthoc)/0.003/44.0=4ewdrour, DiSheafFRL + SWBN/0.003/42.0=5owu8oob, DiSheafFRL + SWBN (posthoc)/0.003/42.0=5owu8oob, DiSheafFRL + SWBN/0.1/43.0=dl677b70, DiSheafFRL + SWBN (posthoc)/0.1/43.0=dl677b70, DiSheafFRL + SWBN/0.003/43.0=dnchyhej, DiSheafFRL + SWBN (posthoc)/0.003/43.0=dnchyhej, DiSheafFRL + SWBN/0.1/42.0=ykc01lwx, DiSheafFRL + SWBN (posthoc)/0.1/42.0=ykc01lwx, DiSheafFRL + SWBN/0.03/44.0=v3krxnyi, DiSheafFRL + SWBN (posthoc)/0.03/44.0=v3krxnyi, DiSheafFRL + SWBN/0.03/43.0=pxsoygwy, DiSheafFRL + SWBN (posthoc)/0.03/43.0=pxsoygwy, DiSheafFRL + SWBN/0.1/44.0=fklk5jum, DiSheafFRL + SWBN (posthoc)/0.1/44.0=fklk5jum, DiSheafFRL + SWBN/0.3/43.0=2eut5tk2, DiSheafFRL + SWBN (posthoc)/0.3/43.0=2eut5tk2, DiSheafFRL + SWBN/0.3/42.0=sxhnjpbg, DiSheafFRL + SWBN (posthoc)/0.3/42.0=sxhnjpbg, DiSheafFRL + SWBN/0.3/44.0=mtb2l4hr, DiSheafFRL + SWBN (posthoc)/0.3/44.0=mtb2l4hr, DiSheafFRL + SWBN/1.0/42.0=4wpd5ryj, DiSheafFRL + SWBN (posthoc)/1.0/42.0=4wpd5ryj, DiSheafFRL + SWBN/1.0/43.0=pqqb8400, DiSheafFRL + SWBN (posthoc)/1.0/43.0=pqqb8400, DiSheafFRL + SWBN/1.0/44.0=azu6d0sx, DiSheafFRL + SWBN (posthoc)/1.0/44.0=azu6d0sx
