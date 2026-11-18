Fashion-MNIST, 4 agents, shift 1.0, 40 epochs. Seeds 42/43/44 at the tuned lambda (chosen on seed-42 test accuracy); %. gain = comm minus non-coop (pts), paired by seed, SD over seeds. Scope columns: both_seen = both agents trained the class; txU_rxS = sender unseen, receiver seen; priv_target = private accuracy on the agent's own target classes; MB = training communication.

```
                        lambda  comm_sd    comm  gain_vs_noncoop  gain_sd  both_seen  txU_rxS  private  priv_target    MRR  misal        MB
method                                                                                                                                     
DiSheafFRL                0.10    0.082  55.449           12.707    0.812     94.316   90.515   57.168       95.279  0.076  0.079  5012.000
SheafFRL                  0.01    0.077  54.683           11.941    0.689     94.325   87.952   57.162       95.269  0.075  0.081  5012.000
FedMuscle                 1.00    0.672  46.219            3.477    0.354     88.854   65.209   57.134       95.224  0.046  0.503  3247.000
NonCooperativeLearning    0.00    0.743  42.742            0.000    0.000     86.192   56.281   57.196       95.326  0.018  0.446     7.000
FedProto                  0.01    0.312  42.486           -0.256    0.457     85.810   55.808   57.184       95.307  0.018  0.443   260.125
FedRE                     0.00    0.430  42.206           -0.536    0.759     87.019   53.666   57.251       95.418  0.013  0.434    50.572
ComFed                    0.01    0.618  29.952          -12.790    1.182        NaN      NaN   57.197          NaN    NaN    NaN   759.375
```

Lambda grid, seed 42, comm acc %:

```
lambda                  0.000  0.001  0.003  0.010  0.030  0.100  0.300  1.000
method                                                                        
ComFed                    NaN  20.48  28.33  30.48  28.31  30.43  28.43  24.76
DiSheafFRL                NaN  50.60  53.58  55.00  55.46  55.54  55.50  55.50
FedMuscle                 NaN  41.62  42.04  41.79  42.45  42.14  44.74  45.67
FedProto                  NaN  41.53  41.73  42.13  42.05  41.62  41.35  41.31
SheafFRL                  NaN  51.47  53.63  54.64  54.34  51.73  49.48  40.82
FedRE                   42.23    NaN    NaN    NaN    NaN    NaN    NaN    NaN
NonCooperativeLearning  41.92    NaN    NaN    NaN    NaN    NaN    NaN    NaN
```

Run IDs (seed:run): DiSheafFRL: 42:elaalpwu 43:aulug7vx 44:qa3ge59b; SheafFRL: 42:lamz4dcs 43:ghefim5n 44:x6z6tc5f; FedMuscle: 42:avksljum 43:f3acq57l 44:gelk00po; NonCooperativeLearning: 42:sdgzql7q 43:g2ff66vb 44:m5gbsehy; FedProto: 42:u4f3rt33 43:f6ii1gjc 44:sg49r72l; FedRE: 42:qwhbje7w 43:d0w9fygz 44:d1s88ftl; ComFed: 42:fy8uea0u 43:jdodnde0 44:zi8lddjb
