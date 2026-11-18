# DEV / HELD_OUT raw-frame inventory on Euler (2026-11-18 13:35, T27)

`bash tools/dev_frame_check.sh dev|held_out` and `python tools/restore_frames.py --split dev|held_out` (dry run) on the login node.

- DEV: 0 of 12 flights block a run (was 9/12 complete on 2 Nov: hospital/P044 384/577, hospital/P047 401/533, factory/P008 617/920 depth frames).
- HELD_OUT: 0 of 19 flights block a run (was 0/19 on 2 Nov).
- Restore dry run: 0 links to make on every flight; poses match the project copy (max |d| <= 3.9e-08).

So DEV 12/12 and HELD_OUT 19/19 are on disk; the restore step is no longer needed before a gate. Not done here: who restored them (no --apply in this run).
