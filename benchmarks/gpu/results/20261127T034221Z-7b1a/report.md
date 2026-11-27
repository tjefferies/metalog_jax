## NVIDIA H200

| Datasets (10 terms) | Time | Per dataset | Peak memory |
|---|---|---|---|
| 1,024 | 2.22 s | 2.17 ms | 0.07 GB |
| 4,096 | 2.22 s | 0.54 ms | 0.27 GB |
| 32,768 | 3.27 s | 99.7 µs | 2.15 GB |
| 131,072 | 3.09 s | 23.6 µs | 8.27 GB |
| 524,288 | 8.13 s | 15.5 µs | 32.63 GB |
| 2,621,440 (5 chunks of 524,288) | 41.0 s | 15.7 µs | 32.85 GB |

2,097,152 datasets (feasible): skipped

At 32,768 datasets:

| Terms | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Per dataset | 3.3 µs | 17.9 µs | 48.2 µs | 81.9 µs | 0.13 ms |

All 5 term counts in one grid: 0.27 ms per dataset.

## Across devices

| Device | Feasible fits per hour | OLS fits per hour | Largest Feasible batch | Cost per million Feasible fits |
|---|---|---|---|---|
| NVIDIA H200 | 230.0 million | 1,288.6 million | 524,288 | $0.02 |

| Device | Jitted ppf(u) | rvs, jax.random | rvs, HDR | cdf | pdf(cdf(x)) |
|---|---|---|---|---|---|
| NVIDIA H200 | 102,202.4M/s at 1e+09 | 2,576.7M/s at 1e+07 | 2,435.7M/s at 1e+07 | 120.7M/s at 1e+07 | 121.1M/s at 1e+07 |
