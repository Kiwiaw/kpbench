# Picks dump format dry run for the SUB1 sep scorer (SUB3, engineer 2026-11-21, dispatch 13:15)

`python tools/picks_smoke.py --sep` on hospital/P041 (laptop cache, PLACEHOLDER poses: local_data is gone, so no real MICRO poses). Format only: no off / sep value and no fail comparison was computed or printed; the one blind read is unspent.

```
raco_dpp0.5_K100_same_lg_picks: has_picks True, rows 530, onplane entries 530 (5-field 530, picks per entry 100..100), flat read by pickarms 153 of 153, off/offn/dspan filled [153, 153, 153] -> OK
raco_sd_K100_same_lg_picks: has_picks True, rows 530, onplane entries 530 (5-field 530, picks per entry 100..100), flat read by pickarms 153 of 153, off/offn/dspan filled [153, 153, 153] -> OK
raco_cube0.5_K100_same_lg_picks: has_picks True, rows 530, onplane entries 530 (5-field 530, picks per entry 100..100), flat read by pickarms 153 of 153, off/offn/dspan filled [153, 153, 153] -> OK
paired read-out vs dpp0.5: paired n per field [[153, 153, 153], [153, 153, 153]], md table rows 3 -> OK
format OK: 3 arms, 1 flights, 530 queries, 530 with off in all arms (0 missing), has_picks True; no value computed
picks smoke (hospital/P041, 589 frames, placeholder poses, K100, lg): PASS
```

Result: PASS. Rows carry has_picks True, the off-plane column is present on 530 / 530 queries in all three arms, kpbench.sepscore --check reads the dump without error. Left: the same check on the Euler login node with real poses (VPN), then the 12-flight DEV dump (Kinga's yes).
