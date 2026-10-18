| System | 66,635: All@5 | R@5 | ms | 500,002: All@5 | R@5 | ms | ΔAll@5 [95% CI] |
|---|---|---|---|---|---|---|---|
| Dense (BGE-small) | 59.6 | 77.8 | 6 | 47.4 | 70.5 | 12 | -12.2 [-14.3, -10.2] |
| TitleLink only | 71.3 | 82.6 | 80 | 46.5 | 64.5 | 365 | -24.8 [-27.5, -22.2] |
| Two-hop | 67.4 | 81.0 | 106 | 50.3 | 69.0 | 438 | -17.1 [-19.6, -14.6] |
| Two-hop + gate | 71.7 | 83.2 | 80 | 57.4 | 73.2 | 295 | -14.3 [-16.6, -12.1] |

Paired bootstrap, All@5 gain over Dense (10,000 resamples):

| Pool | System | Δ [95% CI] | p |
|---|---|---|---|
| 66,635 | TitleLink only | +11.7 [+9.0, +14.4] | 0.0001 |
| 66,635 | Two-hop | +7.8 [+4.5, +11.1] | 0.0001 |
| 66,635 | Two-hop + gate | +12.1 [+9.3, +15.0] | 0.0001 |
| 500,002 | TitleLink only | -0.9 [-4.4, +2.7] | 0.6226 |
| 500,002 | Two-hop | +2.9 [-0.6, +6.4] | 0.1048 |
| 500,002 | Two-hop + gate | +10.0 [+7.2, +12.8] | 0.0001 |

Blind-title control (All@5):

| Pool | System | Real | Blind | Δ [95% CI] |
|---|---|---|---|---|

Large pool cost: dense index 9.9 s, graph build 34.4 s, peak RSS 4746 MB, embeddings 768.0 MB, graph nnz 8,507,361.
