# FRESH-2 extension: candidate flights and reachable flat counts (MAIN, engineer 2026-11-20)

Script: `python tools/fresh2_candidates.py --pool <agent repo>/kpbench/fresh_pool.json`. No picker row and no FRESH-2 frame was read; nothing ran on Euler.

## Calibration of the flat estimate (14 FRESH-1 flights, the only ones with estimate and truth)
- true flat 2075 vs depth-only estimate 1830 -> ratio 1.134, flight-bootstrap 95 % interval [0.952, 1.393].
- single flights: true / estimate from 0.75 to 2.41 (flights with estimate >= 20); largest miss oldtown/Easy/P001 (338 true vs 140).

## A. Registered pool: same environments, never run (42 flights)
- queries 27439, flat estimate 2774; with the calibration ratio: 3146 [2640, 3865].
- same-environment flights beyond these: 0 (every oldtown / hospital / abandonedfactory trajectory is in DEV, HELD_OUT, FRESH-1 or this pool).

| flight | queries | flat est |
|---|---|---|
| abandonedfactory/Easy/P000 | 1958 | 185 |
| abandonedfactory/Easy/P001 | 390 | 69 |
| abandonedfactory/Easy/P002 | 834 | 130 |
| abandonedfactory/Easy/P004 | 1221 | 122 |
| abandonedfactory/Easy/P006 | 657 | 40 |
| abandonedfactory/Easy/P008 | 1490 | 18 |
| abandonedfactory/Easy/P009 | 305 | 37 |
| abandonedfactory/Easy/P010 | 3345 | 181 |
| abandonedfactory/Easy/P011 | 2775 | 278 |
| hospital/Easy/P000 | 506 | 141 |
| hospital/Easy/P001 | 432 | 28 |
| hospital/Easy/P002 | 346 | 60 |
| hospital/Easy/P003 | 421 | 1 |
| hospital/Easy/P005 | 444 | 164 |
| hospital/Easy/P007 | 897 | 136 |
| hospital/Easy/P008 | 327 | 38 |
| hospital/Easy/P009 | 365 | 32 |
| hospital/Easy/P010 | 291 | 2 |
| hospital/Easy/P011 | 296 | 33 |
| hospital/Easy/P012 | 271 | 40 |
| hospital/Easy/P013 | 306 | 6 |
| hospital/Easy/P014 | 276 | 11 |
| hospital/Easy/P015 | 517 | 35 |
| hospital/Easy/P016 | 440 | 94 |
| hospital/Easy/P017 | 303 | 9 |
| hospital/Easy/P018 | 1288 | 72 |
| hospital/Easy/P020 | 347 | 78 |
| hospital/Easy/P021 | 439 | 26 |
| hospital/Easy/P022 | 324 | 17 |
| hospital/Easy/P023 | 348 | 4 |
| hospital/Easy/P024 | 337 | 44 |
| hospital/Easy/P026 | 402 | 32 |
| hospital/Easy/P027 | 336 | 15 |
| hospital/Easy/P028 | 331 | 11 |
| hospital/Easy/P029 | 579 | 89 |
| hospital/Easy/P030 | 522 | 7 |
| hospital/Easy/P031 | 398 | 106 |
| hospital/Easy/P032 | 906 | 156 |
| hospital/Easy/P033 | 306 | 23 |
| hospital/Easy/P034 | 513 | 177 |
| hospital/Easy/P035 | 276 | 7 |
| hospital/Easy/P036 | 374 | 22 |

## B. Extension candidates: new environments (282 flights in 30 environment/difficulty groups; no labels, no depth estimate)
Flat is bracketed at 9.76 % (share on the 31 board flights) and 15.4 % (share on FRESH-1) of the queries.

| group | flights | queries | flat at 9.76 % | flat at 15.4 % | flights (queries) |
|---|---|---|---|---|---|
| abandonedfactory_night/Easy | 13 | 13064 | 1275 | 2012 | P001(753) P002(410) P003(972) P004(1010) P005(574) P006(891) P007(1411) P008(1648) P009(332) P010(539) P011(1719) P012(274) P013(2531) |
| abandonedfactory_night/Hard | 14 | 9137 | 892 | 1407 | P000(1221) P001(417) P002(288) P003(767) P005(539) P006(352) P007(936) P008(738) P009(281) P010(472) P011(300) P012(1242) P013(1170) P014(414) |
| amusement/Easy | 7 | 6971 | 680 | 1074 | P001(660) P002(1222) P003(1412) P004(1224) P006(1395) P007(478) P008(580) |
| amusement/Hard | 8 | 5257 | 513 | 810 | P000(460) P001(483) P002(805) P003(631) P004(938) P005(716) P006(729) P007(495) |
| carwelding/Easy | 6 | 3392 | 331 | 522 | P001(878) P002(1182) P004(352) P005(386) P006(273) P007(321) |
| carwelding/Hard | 4 | 1926 | 188 | 297 | P000(406) P001(748) P002(456) P003(316) |
| endofworld/Easy | 10 | 7575 | 739 | 1167 | P000(1393) P001(419) P002(708) P003(538) P004(884) P005(1155) P006(972) P007(487) P008(531) P009(488) |
| endofworld/Hard | 5 | 2591 | 253 | 399 | P000(827) P001(568) P002(451) P005(434) P006(311) |
| gascola/Easy | 7 | 11876 | 1159 | 1829 | P001(343) P003(1388) P004(2669) P005(1523) P006(2446) P007(2205) P008(1302) |
| gascola/Hard | 10 | 9473 | 925 | 1459 | P000(310) P001(337) P002(1271) P003(603) P004(1663) P005(1265) P006(1378) P007(639) P008(1320) P009(687) |
| japanesealley/Easy | 6 | 3974 | 388 | 612 | P001(952) P002(587) P003(756) P004(699) P005(521) P007(459) |
| japanesealley/Hard | 6 | 2431 | 237 | 374 | P000(357) P001(587) P002(351) P003(278) P004(325) P005(533) |
| neighborhood/Easy | 20 | 25850 | 2523 | 3981 | P000(3783) P001(2535) P002(470) P003(1823) P004(993) P005(651) P007(783) P008(1317) P009(981) P010(930) P012(806) P013(2374) P014(1466) P015(328) P016(1231) P017(757) P018(867) P019(1496) P020(1279) P021(980) |
| neighborhood/Hard | 18 | 16160 | 1577 | 2489 | P000(2084) P001(1792) P002(958) P003(528) P004(1149) P005(772) P006(520) P007(479) P008(1020) P009(270) P010(1393) P011(885) P012(1094) P013(447) P014(498) P015(553) P016(926) P017(792) |
| ocean/Easy | 12 | 6976 | 681 | 1074 | P000(1184) P001(432) P002(473) P004(999) P005(396) P006(286) P008(1094) P009(335) P010(275) P011(497) P012(470) P013(535) |
| ocean/Hard | 10 | 5239 | 511 | 807 | P000(897) P001(468) P002(565) P003(282) P004(441) P005(348) P006(663) P007(607) P008(507) P009(461) |
| office/Easy | 7 | 8880 | 867 | 1368 | P000(1277) P001(1255) P002(1461) P003(1640) P004(1423) P005(863) P006(961) |
| office/Hard | 8 | 5099 | 498 | 785 | P000(690) P001(586) P002(893) P003(270) P004(613) P005(806) P006(446) P007(795) |
| office2/Easy | 10 | 7676 | 749 | 1182 | P000(340) P003(550) P004(1528) P005(487) P006(824) P007(279) P008(1323) P009(1434) P010(273) P011(638) |
| office2/Hard | 11 | 6872 | 671 | 1058 | P000(277) P001(315) P002(633) P003(1359) P004(310) P005(1050) P006(319) P007(425) P008(747) P009(935) P010(502) |
| seasidetown/Easy | 10 | 5877 | 574 | 905 | P000(504) P001(362) P002(303) P003(301) P004(456) P005(567) P006(922) P007(851) P008(787) P009(824) |
| seasidetown/Hard | 5 | 2538 | 248 | 391 | P000(282) P001(280) P002(803) P003(330) P004(843) |
| seasonsforest/Easy | 10 | 4163 | 406 | 641 | P001(287) P002(270) P003(468) P004(576) P005(340) P007(413) P008(396) P009(650) P010(413) P011(350) |
| seasonsforest/Hard | 5 | 1779 | 174 | 274 | P001(495) P002(288) P004(291) P005(435) P006(270) |
| seasonsforest_winter/Easy | 10 | 9271 | 905 | 1428 | P000(2096) P001(762) P002(880) P003(946) P004(870) P005(1575) P006(297) P007(1071) P008(359) P009(415) |
| seasonsforest_winter/Hard | 9 | 6632 | 647 | 1021 | P010(1419) P011(1278) P012(618) P013(476) P014(432) P015(360) P016(291) P017(1098) P018(660) |
| soulcity/Easy | 13 | 11901 | 1162 | 1833 | P000(1192) P001(974) P002(496) P003(324) P004(967) P005(957) P006(1753) P007(447) P008(1215) P009(582) P010(1323) P011(1014) P012(657) |
| soulcity/Hard | 8 | 4285 | 418 | 660 | P000(582) P001(434) P002(504) P003(708) P004(517) P005(713) P008(504) P009(323) |
| westerndesert/Easy | 12 | 5676 | 554 | 874 | P001(289) P002(351) P004(968) P005(323) P006(484) P007(411) P008(387) P009(543) P010(429) P011(294) P012(306) P013(891) |
| westerndesert/Hard | 8 | 3824 | 373 | 589 | P000(833) P001(276) P002(751) P003(459) P004(415) P005(531) P006(287) P007(272) |
| **total** | **282** | **216365** | **21117** | **33320** | |

## Reachability
| target | from pool A alone (est 2774, calibrated low 2640) | extra flat needed (est / calibrated low) | new-env queries needed at 9.76 % (est / low) | share of the 216365 new-env queries |
|---|---|---|---|---|
| 3000 flat | NO | 226 / 360 | 2310 / 3687 | 1.1 % / 1.7 % |
| 3500 flat | NO | 726 / 860 | 7433 / 8810 | 3.4 % / 4.1 % |
| 4150 flat | NO | 1376 / 1510 | 14093 / 15470 | 6.5 % / 7.1 % |
| >= 20 flights | yes (42) | 0 | 0 | 0 |

All three flat targets are reachable only by adding NEW-environment flights (0 same-environment flights are left); the new-environment pool holds 21117-33320 flat by the brackets, 14x-22x the largest need.

## Reading (dispatch kill: "< 3000 flat reachable with unused flights -> no extension")
- Kill NOT met, with a caveat: 3000 / 3500 / 4150 flat are reachable, but only with new-environment flights, whose flat counts are brackets (no labels yet), not counts. The exact per-flight flat count needs the picker-free label pass on Euler (euler/fresh_labels.sbatch, depth and plane fit only), which needs VPN and Kinga's go.
- The registered pool itself may already hold more than 2774: the same estimate read 1830 on FRESH-1 where the truth was 2075 (ratio 1.134), giving 3146 [2640, 3865]. So pool A alone reaches 3000 in the central case but not at the low end, and 3500 only in the upper part of the interval.
- New-environment flights change what the endpoint says: from "holds on unseen flights of the three known environments" to "holds on unseen flights including new environments". Flat frames there may behave differently (outdoor ground planes vs indoor walls); the power note (results/main/power_fresh2.md) assumed flights like the ones already run.

## Draft addendum to fresh2_k100_endpoint_prereg.md (UNSIGNED: effective only with Kinga's yes, before any FRESH-2 frame is read by a picker)
- Unchanged: arms (dpp0.5 vs cube0.5, K100, lg, ap3p, one commit), unit = flight, statistic d_f, exact one-sided sign-flip, alpha 0.05, one look, no flights added after the first read.
- Step 1 (picker-free): run the label pass on the 42 pool-A flights. If their true flat count is >= 3500, FRESH-2 = pool A, no extension.
- Step 2 (only if step 1 gives < 3500): add new-environment flights in a fixed order until the total is >= 3500 flat: groups in the order of section B (alphabetical, Easy before Hard), one flight per group per round, flights inside a group in P-number order; the label pass runs on each added flight before the next is taken; a flight enters whole.
- The final list is frozen in splits_fresh2.json (with the label manifest hash) before any picker row exists; the list and its flat count are audited (kpbench.audit) before the run.
- Reported with the endpoint: the same statistic on pool A only and on the added flights only (descriptive, no test).
- Not chosen here (Kinga's decision): target 3500 (80 % power if the gap is as seen) vs 4150 (the old 5a floor; about 88-90 %) and whether new environments may enter at all (option b) or the test runs on pool A as registered (option a).
