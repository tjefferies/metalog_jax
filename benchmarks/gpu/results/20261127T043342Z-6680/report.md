## NVIDIA A100-SXM4-80GB

| Datasets (10 terms) | Time | Per dataset | Peak memory |
|---|---|---|---|
| 1,024 | 4.34 s | 4.24 ms | 0.07 GB |
| 4,096 | 4.20 s | 1.03 ms | 0.27 GB |
| 32,768 | 4.76 s | 0.15 ms | 2.15 GB |
| 131,072 | 6.56 s | 50.0 µs | 8.27 GB |
| 524,288 | 18.4 s | 35.0 µs | 32.63 GB |
| 1,048,576 (2 chunks of 524,288) | 38.0 s | 36.2 µs | 32.74 GB |

2,097,152 datasets (feasible): oom

At 32,768 datasets:

| Terms | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Per dataset | 4.5 µs | 27.9 µs | 84.9 µs | 0.14 ms | 0.24 ms |

All 5 term counts in one grid: 0.49 ms per dataset.

## NVIDIA H100 80GB HBM3

| Datasets (10 terms) | Time | Per dataset | Peak memory |
|---|---|---|---|
| 1,024 | 2.66 s | 2.60 ms | 0.07 GB |
| 4,096 | 3.94 s | 0.96 ms | 0.27 GB |
| 32,768 | 3.56 s | 0.11 ms | 2.15 GB |
| 131,072 | 3.77 s | 28.7 µs | 8.24 GB |
| 524,288 | 10.5 s | 20.1 µs | 32.63 GB |
| 2,097,152 (4 chunks of 524,288) | 42.6 s | 20.3 µs | 32.85 GB |

2,097,152 datasets (feasible): oom

At 32,768 datasets:

| Terms | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Per dataset | 3.6 µs | 18.3 µs | 61.7 µs | 0.11 ms | 0.19 ms |

All 5 term counts in one grid: 0.38 ms per dataset.

## Across devices

| Device | Feasible fits per hour | OLS fits per hour | Largest Feasible batch | Cost per million Feasible fits |
|---|---|---|---|---|
| NVIDIA A100-SXM4-80GB | 99.4 million | 568.2 million | 524,288 | $0.02 |
| NVIDIA H100 80GB HBM3 | 177.2 million | 1,144.7 million | 524,288 | $0.02 |

| Device | Jitted ppf(u) | rvs, jax.random | rvs, HDR | cdf | pdf(cdf(x)) |
|---|---|---|---|---|---|
| NVIDIA A100-SXM4-80GB | 44,507.5M/s at 1e+09 | 1,139.3M/s at 1e+07 | 1,456.1M/s at 1e+07 | 56.9M/s at 1e+07 | 57.3M/s at 1e+07 |
| NVIDIA H100 80GB HBM3 | 99,832.7M/s at 1e+09 | 1,914.8M/s at 1e+07 | 1,767.0M/s at 1e+07 | 97.0M/s at 1e+07 | 96.4M/s at 1e+07 |
