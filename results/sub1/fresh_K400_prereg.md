# SUB1 prereg v2 scorer on fresh, K 400 (13440 queries, 2075 flat; uniform dpp0.5 vs strong cube0.5; one pair only)

| metric | dAUROC | dPR-AUC | rule 1 | d (dpp0.5 - cube0.5) | z | rule 2 | within vs cube0.5 n / share / p | within vs - n / share / p | rule 3 |
|---|---|---|---|---|---|---|---|---|---|
| hull | -0.0114 | +0.0172 | no | +0.0191 | 11.82 | yes | 31 / 0.4194 / 0.47 | - | no |
| reproj_med | -0.0392 | +0.0350 | no | -0.0008 | -0.03 | no | 31 / 0.6129 / 0.28 | - | no |
| rep5 | -0.0065 | -0.0420 | no | +0.0778 | 37.88 | yes | 31 / 0.4444 / 0.7 | - | no |
| crlb | -0.0162 | -0.0349 | no | +0.0672 | 24.24 | yes | 31 / 0.3226 / 0.071 | - | no |
| inl_ratio | +0.0055 | +0.0571 | no | +0.0286 | 24.83 | yes | 31 / 0.7097 / 0.029 | - | yes |
| cells | -0.0510 | +0.0190 | no | +0.0133 | 7.85 | yes | 31 / 0.4138 / 0.46 | - | no |
| ent | -0.0316 | +0.0304 | no | +0.0109 | 10.36 | yes | 31 / 0.3871 / 0.28 | - | no |
| offplane | +0.0163 | +0.1564 | yes | +0.0175 | 14.93 | yes | 31 / 0.6452 / 0.15 | - | no |
| sel_logdet | -0.0045 | -0.0087 | no | +0.0405 | 11.05 | yes | 31 / 0.3871 / 0.28 | - | no |
| sel_hull | -0.0061 | -0.0132 | no | -0.0091 | -10.10 | no | 31 / 0.3871 / 0.28 | - | no |
| pcov | -0.0002 | -0.0161 | no | +0.0152 | 10.28 | yes | 31 / 0.5161 / 1 | - | no |

**no candidate passes all three rules -> SUB1 stays query-level (prereg kill) [K 400, one pair only: K-dependence read-out, not the prereg verdict]**
