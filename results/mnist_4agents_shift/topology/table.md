Test accuracy (%), mean ± sample SD over seeds 42/43/44. 4096 pilots, lambda .01, batch 256, 100 epochs, CenterNorm, no whitening.
Communication pooled over the directed edges of each topology (star 6, ring 8, fully connected 12) within a seed; uncertainty across seeds only.

| Scope | Method | Star | Ring | Fully connected |
|---|---|---|---|---|
| Private accuracy (all classes) | Non-coop | 69.56 ± 0.01 | 69.56 ± 0.01 | 69.56 ± 0.01 |
| Private accuracy (all classes) | DiSheafFRL | 69.55 ± 0.02 | 69.57 ± 0.02 | 69.57 ± 0.01 |
| Private accuracy (target classes) | Non-coop | 99.58 ± 0.02 | 99.58 ± 0.02 | 99.58 ± 0.02 |
| Private accuracy (target classes) | DiSheafFRL | 99.58 ± 0.03 | 99.60 ± 0.03 | 99.60 ± 0.01 |
| Communication, all classes | Non-coop | 65.19 ± 0.36 | 65.68 ± 0.22 | 65.78 ± 0.29 |
| Communication, all classes | DiSheafFRL | 69.19 ± 0.03 | 69.24 ± 0.02 | 69.29 ± 0.03 |
| Sender unseen / receiver seen | Non-coop | 82.88 ± 1.45 | 84.35 ± 0.84 | 85.68 ± 1.06 |
| Sender unseen / receiver seen | DiSheafFRL | 98.55 ± 0.03 | 98.79 ± 0.05 | 98.99 ± 0.07 |
| Receiver seen (its target classes) | Non-coop | 93.32 ± 0.51 | 94.02 ± 0.31 | 94.16 ± 0.41 |
| Receiver seen (its target classes) | DiSheafFRL | 99.05 ± 0.04 | 99.13 ± 0.03 | 99.20 ± 0.04 |
| Sender seen / receiver seen (shared classes) | Non-coop | 99.56 ± 0.03 | 99.41 ± 0.04 | 99.42 ± 0.03 |
| Sender seen / receiver seen (shared classes) | DiSheafFRL | 99.36 ± 0.08 | 99.31 ± 0.02 | 99.33 ± 0.03 |

Run IDs: star/Non-coop/42=thp6ckyw, star/Non-coop/43=l2mafcc9, star/Non-coop/44=a3ntx21c, star/DiSheafFRL/43=hl2sgdl4, star/DiSheafFRL/44=qz4yl8xs, star/DiSheafFRL/42=5n8efnf2, ring/Non-coop/42=x6dcfb37, ring/Non-coop/43=cvrgr5x5, ring/Non-coop/44=dp5ygmlu, ring/DiSheafFRL/42=ojzodjwz, ring/DiSheafFRL/43=bbj6ergm, ring/DiSheafFRL/44=jwiv895r, fully_connected/Non-coop/42=c8ix7vn9, fully_connected/Non-coop/43=x9u62pd7, fully_connected/Non-coop/44=dsnopa19, fully_connected/DiSheafFRL/42=cndm7lcm, fully_connected/DiSheafFRL/43=7x9j9uyl, fully_connected/DiSheafFRL/44=7zvxszuc
