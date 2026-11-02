# DEV / HELD_OUT raw-frame inventory on Euler (2026-11-02 13:20, T27 step 2)

`bash tools/dev_frame_check.sh dev|held_out|fresh` on the login node plus a per-flight count (poses, depth files, images, cache frames, label queries). Root /cluster/scratch/kkaczmarzyk; scratch purges untouched files, so depth and image frames vanish while pose_left.txt, the candidate caches and the labels stay.

| flight | split | poses | depth | images | cache | labels |
|---|---|---|---|---|---|---|
| hospital/P044 | dev | 577 | 384 | 13 | 577 | 519 |
| hospital/P047 | dev | 533 | 401 | 1 | 533 | 479 |
| factory/P008 | dev | 920 | 617 | 1 | 920 | 828 |
| 9 other DEV flights | dev | = depth = images = cache | | | | |
| all 19 HELD_OUT flights | held_out | 308-1215 | 0 | 0 | = poses | = queries |
| all 14 FRESH-1 flights | fresh | complete (symlinks into the project copy) | | | | |

DEV complete 9 of 12 (the 31 Oct count of 2 short flights missed factory/P008), HELD_OUT 0 of 19, FRESH-1 14 of 14. kpbench.run refuses a flight whose cache frames != flight frames, so every DEV gate and tune fails on these 3 flights until restored.

## Restore source (verified)

/cluster/project/cvg/data/processed_data/processed_tartanair/<env>/Hard/<traj> (hospital, abandonedfactory, oldtown; the FRESH flights are linked from the same copy by kpbench.fresh.link). For every short flight the source frame count equals the pose count, and 21 sampled poses (from <i>_cam.npz through kpbench.fresh.pose_row) match pose_left.txt to 3.3e-8 in rotation and translation (a plain quaternion compare shows 1.9 on some flights: the q / -q sign, not a mismatch).

`python tools/restore_frames.py --split dev|held_out [--apply]`: dry run by default; checks count and poses per flight, then symlinks every missing depth_left / image_left file into the source (nothing deleted or overwritten; symlinks are purge-proof, FRESH-1 has held since 18 Oct). Dry run 2 Nov: DEV 757 + 664 + 1222 links, HELD_OUT 19 flights, all pose-matched. Applying it is Kinga's call (data decision, board/decision.md).
