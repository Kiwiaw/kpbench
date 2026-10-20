# FRESH-1 freeze record (T14, 2026-10-20)

FRESH-1 = 14 TartanAir trajectories outside the 31 benchmark flights (results/gates/FRESH_set.json, created 19 Oct):
oldtown_easy P000 P001 P002 P004 P005 P007, hospital_easy P004 P006 P019 P025, factory_hard P009 P010 P011, factory_easy P005.
13440 queries, 2075 flat (oldtown 900, hospital 693, factory 482); oldtown_easy/P004 has 0 flat, so paired tests use 13 flights.
Hashes: splits_fresh.json sha256 2ba540e1..., fresh label manifest sha256 0b165c7c... (files on Euler; C23 disjointness PASS 19 Oct).

Use so far (near-miss retest rule, Kinga 19 Oct): both DEV near-misses retested on one node and CLOSED:

| candidate vs dpp0.5 | DEV flat diff pt | FRESH-1 flat fail % | FRESH-1 flat diff pt [95 %] | flights better/worse/tied |
|---|---|---|---|---|
| dppr0.486a0.453b1.02t0.9 | -1.18 [-2.53, +0.11] | 3.04 vs 2.99 | +0.05 [-0.45, +0.61] | 3/5/5 |
| win23+dpp0.666a0.526 | -1.18 [-3.09, +0.80] | 3.95 vs 2.99 | +0.96 [-0.22, +2.26] | 3/5/5 |

Status: regime / transfer set, NOT a gate set. Base rate 2.99 % (about 62 flat fails) vs DEV 14.42 %: the 10 % relative rule
is a -0.30 pt shift there, while the smallest detectable shift (power 0.74, one-sided 2.5 %) is 0.71 pt = 23.6 % relative.
A future FRESH-2 gate set needs DEV-like difficulty (pre-registered hard-frame filter, euler/fresh_order.json order).
Never mixed into splits.json, DEV or HELD_OUT.
