# Euler login-node audit, 8 Nov 2026 (read out 11 Nov, SUB2 engineer step)

Source: ~/kpbench/logs/audit_login_2026-11-08.log on Euler (`python -m kpbench.audit --only C1,C2,C3,C23,C25,C38,C39`, CPU, login node). 6 PASS / 1 WARN / 0 FAIL.

| check | status | what | counts quoted from the log |
|---|---|---|---|
| C1 | PASS | counts | total 18571 flat 1813 very flat 1058; mismatches: none |
| C2 | PASS | K = 100 | rows with n_sel != 100 per run: {'raco_sd_K100_same_lg': 0, 'raco_cube0.5_K100_same_lg': 0, 'raco_grid_K100_same_lg': 0, 'raco_sd_K100_ret5_lg': 0, 'raco_cube0.5_K100_ret5_lg': 0, 'raco_grid_K100_ret5_lg': 0} |
| C3 | PASS | keyframe rule | issues none; keyframes per query on oldtown/P001: {} |
| C23 | PASS | FRESH-1 disjoint from the 31 flights | 14 fresh flights vs 31 reference flights; DEV flat 936, FRESH flat 2075 / 13440 queries; closest same-length same-env pair hospital_easy/P025 vs hospital/P045 max /dpos/ 15.58 m |
| C25 | WARN | DEV gate verdicts recomputed from rows | 30 recomputed: cube0.359 -0.75 [-2.64, +1.23] 5/2 REVERT n_flat 936; deco +4.06 [+2.04, +6.29] 1/6 REVERT n_flat 936; dpp0.317a1.6 +0.85 [-0.53, +2.38] 1/5 REVERT n_flat 936; dpp0.317a1.6 +1.39 [-0.00, +2.97] 1/5 REVERT n_flat 936; dpp0.368a1.09 +0.00 [-1.10, +1.17] 3/3 REVERT n_flat 936; dpp0.5 -1. |
| C38 | PASS | FRESH / HELD_OUT gate verdicts recomputed from rows | 10 recomputed: cube0.5 -2.55 [-4.80, -0.34] 5/3 KEEP n_flat 2075; cube0.5 -0.34 [-1.00, +0.40] 5/2 REVERT n_flat 2075; dpp0.317a1.6 +0.48 [-0.29, +1.25] 3/5 REVERT n_flat 2075; dpp0.601a0.283 -0.14 [-0.71, +0.34] 2/3 REVERT n_flat 2075; dpp2d120a1 +2.75 [+1.08, +4.90] 3/5 REVERT n_flat 2075; dppr0.3 |
| C39 | PASS | FRESH-1 labels in the repo | 14 flights, 13440 queries, 2075 flat from repo labels; 9 FRESH gate files, n_flat [2075]; 18 mirrored FRESH runs all 13440 rows; runs with picks 0, without 103 (per-pick tables refused on those by gate.require_picks) |

C25 WARN = pre-sync gate files (937 flat) with the same verdict; 30 DEV gates recomputed, 0 verdict changes.

## Addendum 11 Nov 2026 (ssh up 13:10)

- Euler `kpbench.audit --only C38` rerun today: the K400 gate cube0.5 vs sd recomputes from rows as -4.39 [-7.01, -2.01], 6/0, KEEP, n_flat 2075 (= gate json). Status WARN only because the cube0.5 K400 dir was rewritten by job 20220547 on 8 Nov (node eu-a2p-394), so the older cube0.5 vs dpp0.5 K400 gate now sees two nodes per flight (gate device guard).
- Laptop `kpbench.audit --only C38,C39` today, after mirroring raco_sd_K400_same_lg_g-cube0.5-fresh-K400 (13440 rows): the mirror image. cube0.5 vs dpp0.5 K400 recomputes -0.34 [-1.00, +0.40] 5/2 REVERT (laptop cube rows = 1 Nov copy), cube0.5 vs sd K400 trips the device guard. C39 PASS: 14 flights, 13440 queries, 2075 flat, 10 FRESH gate files, 19 mirrored FRESH runs all 13440 rows.
- The two cube0.5 K400 copies (1 Nov laptop, 8 Nov Euler) are equal row for row: 0 of 13440 rows differ; only meta date and device differ. So both K400 gates are row-audited, one on each side, on the same cube rows.
- 148 = 148: results/audit/r29_148_verdict_2026-11-11.md (rule 3 fired: different frame sets, Jaccard 0.4231; no row reuse; recounts agree at 148 and 148).
