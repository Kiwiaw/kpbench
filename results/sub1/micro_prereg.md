# SUB1 prereg v2 scorer on micro (393 queries, 393 flat; uniform cube0.5 vs strong sd, rule 3 also vs grid)

| metric | dAUROC | dPR-AUC | rule 1 | d (cube0.5 - sd) | z | rule 2 | within vs sd n / share / p | within vs grid n / share / p | rule 3 |
|---|---|---|---|---|---|---|---|---|---|
| hull | +0.0054 | +0.0209 | no | +0.0265 | 1.54 | no | 71 / 0.7324 / 0.00011 | 54 / 0.6226 / 0.098 | no |
| reproj_med | +0.1053 | +0.1411 | yes | +1.1234 | 2.56 | yes | 71 / 0.8028 / 2.7e-07 | 54 / 0.7037 / 0.0038 | yes |
| rep5 | +0.0149 | +0.0041 | no | -0.0115 | -0.51 | no | 71 / 0.6667 / 0.0092 | 54 / 0.5455 / 0.65 | no |
| crlb | -0.0318 | -0.0246 | no | -0.0508 | -1.25 | no | 71 / 0.6056 / 0.096 | 54 / 0.5185 / 0.89 | no |
| inl_ratio | +0.0058 | +0.0029 | no | -0.0600 | -3.10 | no | 71 / 0.6056 / 0.096 | 54 / 0.6667 / 0.02 | no |
| cells | missing | | | | | | | | |
| ent | missing | | | | | | | | |
| offplane | missing | | | | | | | | |

**no candidate passes all three rules -> SUB1 stays query-level (prereg kill)**
