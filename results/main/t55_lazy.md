# T55 opt-in lazy dpp path: identity and timing (MAIN N3 follow-up, engineer 2026-11-24, dispatch 2026-11-24T13:15) [P]

Gist: `dpplazy0.5` (kpbench.pickers.pick_dpp_lazy / dpp_greedy_lazy) returns the same ordered picks as the benchmark `dpp0.5`
on every laptop query frame and is 10x faster at K100, but its cost still grows with K: 2.2x cube0.5 at K100, 25x at K400.
The benchmark picker `pick_dpp` is unchanged (lazy path added beside it, not used by any registered run).

- Identity (tools/t55_lazy.py check, every query frame, each K run on its own): **0 of 3980** frames differ at K50 / K100 / K200 / K400
  (6 flights; table: t55_lazy_identity.md). Unit test tests/test_lazy.py 2/2 pass (synthetic frames, n 30 / 129 / 1500 / 2000).
- Time (tools/t55_lazy.py time, 1328 query frames = every 3rd, 6.6x the 201 frames of 23 Nov; table: t55_lazy_time.md, raw t55_lazy_time_raw.npz on the laptop, gitignored):

| K | dpplazy0.5 ms (median) | cube0.5 ms | lazy / cube0.5 | 23 Nov (201 frames) |
|---|---|---|---|---|
| 50 | 1.30 | 1.82 | 0.7x | 0.95 |
| 100 | 3.96 (benchmark path 40.55, 10.3x slower) | 1.83 | 2.2x | 2.8 vs 1.1 (2.5x) |
| 200 | 13.30 | 1.84 | 7.2x | 11.2 |
| 400 | 46.65 | 1.86 | 25.1x | 32.3 |

- Absolute ms are ~1.4x the 23 Nov run on the same laptop (cube0.5 1.83 vs 1.1 ms: machine load / power state), ratios are stable
  (K100 2.2x vs 2.5x). Quote ratios, not ms.
- Kill read: identity kill NOT met (0 differ). K400 ratio 25x > 20x: kill met above K100, so N3 stays a limitation above K100;
  only the K100 line (about 2x cube0.5, 10x faster than the repo code) can carry a timing sentence.
