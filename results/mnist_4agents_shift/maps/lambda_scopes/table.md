Test accuracy (%), mean ± sample SD over seeds 42/43/44; 4096 pilots (non-coop lambda-free, repeated), batch 256, 100 epochs, shift 1.
Scopes pool the 12 directed edges within a seed. Eval: posthoc Procrustes for non-coop / CenterNorm / ZCA (as reported remotely).
CenterNorm + OLS: saved OLS maps; Comm from the remote saved-map eval runs (results/mnist_4agents_shift/maps/lambda_summary.csv), seen scopes rescored from the checkpoints.

### Private

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS |
|---|---|---|---|---|
| 0.003 | 69.56 ± 0.01 | 69.59 ± 0.02 | 69.41 ± 0.25 | 69.57 ± 0.03 |
| 0.01 | 69.56 ± 0.01 | 69.57 ± 0.01 | 69.28 ± 0.28 | 69.56 ± 0.01 |
| 0.03 | 69.56 ± 0.01 | 69.53 ± 0.03 | 68.81 ± 0.60 | 69.51 ± 0.03 |
| 0.1 | 69.56 ± 0.01 | 69.08 ± 0.46 | 64.34 ± 8.22 | 65.89 ± 4.90 |
| 0.3 | 69.56 ± 0.01 | 63.39 ± 2.24 | 66.66 ± 1.19 | 69.04 ± 0.48 |
| 1.0 | 69.56 ± 0.01 | 60.14 ± 2.97 | 66.65 ± 2.32 | 68.77 ± 0.36 |

### Private target classes

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS |
|---|---|---|---|---|
| 0.003 | 99.58 ± 0.02 | 99.62 ± 0.02 | 99.37 ± 0.36 | 99.60 ± 0.04 |
| 0.01 | 99.58 ± 0.02 | 99.60 ± 0.01 | 99.18 ± 0.40 | 99.58 ± 0.02 |
| 0.03 | 99.58 ± 0.02 | 99.54 ± 0.04 | 98.51 ± 0.86 | 99.51 ± 0.04 |
| 0.1 | 99.58 ± 0.02 | 98.90 ± 0.66 | 92.11 ± 11.77 | 94.32 ± 7.02 |
| 0.3 | 99.58 ± 0.02 | 90.75 ± 3.21 | 95.43 ± 1.71 | 98.84 ± 0.69 |
| 1.0 | 99.58 ± 0.02 | 86.10 ± 4.25 | 95.42 ± 3.32 | 98.46 ± 0.52 |

### Comm

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS |
|---|---|---|---|---|
| 0.003 | 65.78 ± 0.29 | 69.19 ± 0.03 | 63.28 ± 1.31 | 65.24 ± 5.13 |
| 0.01 | 65.78 ± 0.29 | 69.29 ± 0.03 | 60.89 ± 4.68 | 65.49 ± 4.65 |
| 0.03 | 65.78 ± 0.29 | 69.31 ± 0.03 | 61.80 ± 0.69 | 64.98 ± 4.89 |
| 0.1 | 65.78 ± 0.29 | 68.83 ± 0.39 | 36.11 ± 7.45 | 49.02 ± 7.46 |
| 0.3 | 65.78 ± 0.29 | 60.28 ± 2.16 | 40.08 ± 7.34 | 44.08 ± 5.64 |
| 1.0 | 65.78 ± 0.29 | 53.88 ± 3.44 | 38.24 ± 10.25 | 50.39 ± 10.47 |

### Comm sender unseen / receiver seen

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS |
|---|---|---|---|---|
| 0.003 | 85.70 ± 1.05 | 98.46 ± 0.10 | 79.70 ± 1.95 | 93.62 ± 6.34 |
| 0.01 | 85.70 ± 1.05 | 98.99 ± 0.07 | 75.33 ± 6.86 | 93.02 ± 8.28 |
| 0.03 | 85.70 ± 1.05 | 99.17 ± 0.07 | 77.55 ± 2.10 | 93.57 ± 6.03 |
| 0.1 | 85.70 ± 1.05 | 98.26 ± 0.80 | 42.87 ± 6.60 | 70.78 ± 10.36 |
| 0.3 | 85.70 ± 1.05 | 81.22 ± 2.87 | 44.69 ± 14.26 | 59.88 ± 5.77 |
| 1.0 | 85.70 ± 1.05 | 68.49 ± 5.51 | 47.77 ± 12.36 | 71.58 ± 15.59 |

### Comm receiver seen

| Lambda | Non-coop | DiSheafFRL + CenterNorm | DiSheafFRL + ZCA | DiSheafFRL + CenterNorm + OLS |
|---|---|---|---|---|
| 0.003 | 94.17 ± 0.41 | 99.05 ± 0.04 | 90.59 ± 1.88 | 93.39 ± 7.35 |
| 0.01 | 94.17 ± 0.41 | 99.20 ± 0.04 | 87.17 ± 6.71 | 93.76 ± 6.66 |
| 0.03 | 94.17 ± 0.41 | 99.22 ± 0.04 | 88.47 ± 0.98 | 93.03 ± 7.01 |
| 0.1 | 94.17 ± 0.41 | 98.54 ± 0.56 | 51.69 ± 10.66 | 70.19 ± 10.67 |
| 0.3 | 94.17 ± 0.41 | 86.29 ± 3.09 | 57.38 ± 10.51 | 63.10 ± 8.08 |
| 1.0 | 94.17 ± 0.41 | 77.14 ± 4.92 | 54.74 ± 14.67 | 72.13 ± 15.00 |

Run IDs: Non-coop/nan/43.0=x9u62pd7, Non-coop/nan/44.0=dsnopa19, Non-coop/nan/42.0=c8ix7vn9, DiSheafFRL + CenterNorm/0.01/43.0=7x9j9uyl, DiSheafFRL + CenterNorm/0.01/44.0=7zvxszuc, DiSheafFRL + CenterNorm/0.01/42.0=cndm7lcm, DiSheafFRL + CenterNorm/0.003/44.0=w9sqx14x, DiSheafFRL + CenterNorm/0.003/42.0=pg3drg2e, DiSheafFRL + CenterNorm/0.03/43.0=id358klz, DiSheafFRL + CenterNorm/0.003/43.0=dv9wdwou, DiSheafFRL + CenterNorm/0.03/42.0=dwmlanm8, DiSheafFRL + CenterNorm/0.03/44.0=4oft73b5, DiSheafFRL + CenterNorm/0.1/42.0=3o8lowo2, DiSheafFRL + CenterNorm/0.3/43.0=kglyukwl, DiSheafFRL + CenterNorm/0.1/44.0=ghj8sjya, DiSheafFRL + CenterNorm/0.3/42.0=xcns8p91, DiSheafFRL + CenterNorm/0.1/43.0=fb45eucn, DiSheafFRL + CenterNorm/0.3/44.0=hkt2jzt0, DiSheafFRL + CenterNorm/1.0/44.0=4ajshdm6, DiSheafFRL + CenterNorm/1.0/42.0=vdf9zm0u, DiSheafFRL + CenterNorm/1.0/43.0=lew4oqwm, DiSheafFRL + ZCA/0.01/42.0=206rghx6, DiSheafFRL + ZCA/0.01/43.0=94j0envu, DiSheafFRL + ZCA/0.01/44.0=rxkkjdmi, DiSheafFRL + ZCA/0.003/42.0=rwckbvlu, DiSheafFRL + ZCA/0.003/43.0=h2qtxmwl, DiSheafFRL + ZCA/0.003/44.0=l34igjyy, DiSheafFRL + ZCA/0.03/42.0=9ib5zwlm, DiSheafFRL + ZCA/0.03/43.0=indu3dl4, DiSheafFRL + ZCA/0.03/44.0=xl1gbb62, DiSheafFRL + ZCA/0.1/42.0=mjbmxbo1, DiSheafFRL + ZCA/0.1/43.0=racb0zwv, DiSheafFRL + ZCA/0.1/44.0=xbpsllm1, DiSheafFRL + ZCA/0.3/42.0=yt3jlx94, DiSheafFRL + ZCA/0.3/43.0=95qhgcye, DiSheafFRL + ZCA/0.3/44.0=0u8y8uu0, DiSheafFRL + ZCA/1.0/42.0=khjatck2, DiSheafFRL + ZCA/1.0/43.0=s46w78eu, DiSheafFRL + ZCA/1.0/44.0=0jzlmvyn, DiSheafFRL + CenterNorm + OLS/0.01/42.0=hjx4w7x9, DiSheafFRL + CenterNorm + OLS/0.01/43.0=5dpijm43, DiSheafFRL + CenterNorm + OLS/0.01/44.0=jm3v8o5g, DiSheafFRL + CenterNorm + OLS/0.003/42.0=avulh2rm, DiSheafFRL + CenterNorm + OLS/0.003/43.0=w6ocus3u, DiSheafFRL + CenterNorm + OLS/0.003/44.0=ysiok61f, DiSheafFRL + CenterNorm + OLS/0.03/42.0=y4gr8omv, DiSheafFRL + CenterNorm + OLS/0.03/43.0=stiemhx6, DiSheafFRL + CenterNorm + OLS/0.03/44.0=s7njait6, DiSheafFRL + CenterNorm + OLS/0.1/42.0=acc2dmeg, DiSheafFRL + CenterNorm + OLS/0.1/43.0=jbi0y1r7, DiSheafFRL + CenterNorm + OLS/0.1/44.0=u7azm1x5, DiSheafFRL + CenterNorm + OLS/0.3/42.0=xyge13wc, DiSheafFRL + CenterNorm + OLS/0.3/43.0=q3dppdpb, DiSheafFRL + CenterNorm + OLS/0.3/44.0=eki926tm, DiSheafFRL + CenterNorm + OLS/1.0/42.0=peepk9xw, DiSheafFRL + CenterNorm + OLS/1.0/43.0=omyaum06, DiSheafFRL + CenterNorm + OLS/1.0/44.0=w8gbejot
