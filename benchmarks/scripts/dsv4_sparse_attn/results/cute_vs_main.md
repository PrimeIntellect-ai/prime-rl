- `main`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git 6d5cf0180
- `cute`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git 8af8d58fb (dirty)
- corpus hash b5b289171983c8c6; synthetic corpus: random-weight CSA picks are near-uniform, while a
  trained indexer favors recent and neighboring entries, so CSA gather locality here is pessimistic.

Op-boundary time per call in µs (lower is better): median over rounds, p20-p80 across rounds.
`/TL` is this time divided by tilelang's in the same run.

| item | backend | fwd µs | fwd p20-p80 | fwd /TL | f+b µs | f+b p20-p80 | f+b /TL |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 1197 | 1180-1209 | 1.00 | 3186 | 3183-3216 | 1.00 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 384.0 | 381.9-387.6 | 0.32 | - | - | - |
| single-2048-csa-cp1 | tilelang@cute | 1213 | 1197-1219 | 1.00 | 3261 | 3257-3275 | 1.00 |
| single-2048-csa-cp1 | cute@cute | 608.8 | 606.2-609.3 | 0.50 | 2627 | 2619-2635 | 0.81 |
| single-2048-csa-cp1 | cute_ws@cute | 303.3 | 302.5-304.6 | 0.25 | 2469 | 2463-2474 | 0.76 |
| single-2048-csa-cp1 | flashmla_fwd_ref@cute | 389.7 | 387.6-392.5 | 0.32 | - | - | - |
| single-2048-csa-cp8r0 | tilelang@main | 733.5 | 729.6-769.4 | 1.00 | 2001 | 1987-2043 | 1.00 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 182.5 | 181.2-196.9 | 0.25 | - | - | - |
| single-2048-csa-cp8r0 | tilelang@cute | 740.2 | 730.6-756.8 | 1.00 | 2069 | 2062-2153 | 1.00 |
| single-2048-csa-cp8r0 | cute@cute | 177.8 | 173.9-181.9 | 0.24 | 1420 | 1410-1427 | 0.69 |
| single-2048-csa-cp8r0 | cute_ws@cute | 86.3 | 83.8-87.9 | 0.12 | 1273 | 1263-1295 | 0.62 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 189.8 | 185.6-193.5 | 0.26 | - | - | - |
| single-2048-csa-cp8r4 | tilelang@main | 763.3 | 755.8-769.4 | 1.00 | 2028 | 2010-2035 | 1.00 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 197.7 | 194.4-198.5 | 0.26 | - | - | - |
| single-2048-csa-cp8r4 | tilelang@cute | 781.4 | 764.3-798.5 | 1.00 | 2093 | 2076-2103 | 1.00 |
| single-2048-csa-cp8r4 | cute@cute | 202.8 | 196.8-205.5 | 0.26 | 1459 | 1440-1461 | 0.70 |
| single-2048-csa-cp8r4 | cute_ws@cute | 98.8 | 98.5-101.2 | 0.13 | 1292 | 1284-1303 | 0.62 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 208.9 | 201.8-211.1 | 0.27 | - | - | - |
| single-2048-csa-cp8r7 | tilelang@main | 774.3 | 771.8-795.5 | 1.00 | 2025 | 2024-2030 | 1.00 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 202.8 | 198.5-208.2 | 0.26 | - | - | - |
| single-2048-csa-cp8r7 | tilelang@cute | 785.6 | 776.1-789.2 | 1.00 | 2087 | 2072-2099 | 1.00 |
| single-2048-csa-cp8r7 | cute@cute | 217.8 | 215.5-219.4 | 0.28 | 1439 | 1434-1442 | 0.69 |
| single-2048-csa-cp8r7 | cute_ws@cute | 104.7 | 103.8-105.3 | 0.13 | 1295 | 1283-1319 | 0.62 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 213.0 | 208.2-215.0 | 0.27 | - | - | - |
| single-2048-hca-cp1 | tilelang@main | 1085 | 1075-1099 | 1.00 | 2504 | 2477-2531 | 1.00 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 341.1 | 339.6-343.2 | 0.31 | - | - | - |
| single-2048-hca-cp1 | tilelang@cute | 1093 | 1081-1094 | 1.00 | 2564 | 2549-2570 | 1.00 |
| single-2048-hca-cp1 | cute@cute | 484.4 | 481.4-488.4 | 0.44 | 1911 | 1902-1918 | 0.75 |
| single-2048-hca-cp1 | cute_ws@cute | 223.6 | 223.4-226.3 | 0.20 | 1718 | 1713-1733 | 0.67 |
| single-2048-hca-cp1 | flashmla_fwd_ref@cute | 351.7 | 350.7-359.0 | 0.32 | - | - | - |
| single-2048-hca-cp8r0 | tilelang@main | 763.9 | 761.9-769.2 | 1.00 | 2152 | 2142-2403 | 1.00 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 188.7 | 185.0-194.9 | 0.25 | - | - | - |
| single-2048-hca-cp8r0 | tilelang@cute | 782.9 | 768.3-788.1 | 1.00 | 2175 | 2159-2183 | 1.00 |
| single-2048-hca-cp8r0 | cute@cute | 203.4 | 200.4-204.9 | 0.26 | 1539 | 1530-1557 | 0.71 |
| single-2048-hca-cp8r0 | cute_ws@cute | 77.0 | 76.6-77.5 | 0.10 | 1335 | 1326-1370 | 0.61 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 198.3 | 193.1-200.3 | 0.25 | - | - | - |
| single-2048-hca-cp8r4 | tilelang@main | 793.0 | 784.8-810.2 | 1.00 | 2158 | 2111-2193 | 1.00 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 193.9 | 191.6-196.0 | 0.24 | - | - | - |
| single-2048-hca-cp8r4 | tilelang@cute | 791.5 | 783.1-798.4 | 1.00 | 2216 | 2199-2223 | 1.00 |
| single-2048-hca-cp8r4 | cute@cute | 208.9 | 206.5-211.8 | 0.26 | 1546 | 1543-1566 | 0.70 |
| single-2048-hca-cp8r4 | cute_ws@cute | 82.2 | 81.0-82.8 | 0.10 | 1331 | 1329-1348 | 0.60 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 200.4 | 196.1-203.7 | 0.25 | - | - | - |
| single-2048-hca-cp8r7 | tilelang@main | 806.9 | 791.4-813.7 | 1.00 | 2178 | 2146-2204 | 1.00 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 193.5 | 189.0-195.7 | 0.24 | - | - | - |
| single-2048-hca-cp8r7 | tilelang@cute | 793.4 | 785.0-801.1 | 1.00 | 2207 | 2161-2334 | 1.00 |
| single-2048-hca-cp8r7 | cute@cute | 205.3 | 202.4-208.1 | 0.26 | 1538 | 1531-1564 | 0.70 |
| single-2048-hca-cp8r7 | cute_ws@cute | 80.6 | 80.0-80.8 | 0.10 | 1326 | 1320-1331 | 0.60 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 198.6 | 194.2-204.6 | 0.25 | - | - | - |
| single-2048-sliding-cp1 | tilelang@main | 1016 | 1002-1021 | 1.00 | 2288 | 2278-2306 | 1.00 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 298.4 | 293.8-303.0 | 0.29 | - | - | - |
| single-2048-sliding-cp1 | tilelang@cute | 992.0 | 986.6-1013 | 1.00 | 2361 | 2348-2385 | 1.00 |
| single-2048-sliding-cp1 | cute@cute | 415.4 | 412.8-418.5 | 0.42 | 1767 | 1728-1768 | 0.75 |
| single-2048-sliding-cp1 | cute_ws@cute | 172.5 | 171.8-172.8 | 0.17 | 1576 | 1572-1596 | 0.67 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@cute | 303.2 | 301.3-304.2 | 0.31 | - | - | - |
| single-2048-sliding-cp8r0 | tilelang@main | 732.5 | 723.8-761.5 | 1.00 | 2047 | 2013-2148 | 1.00 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 179.7 | 174.7-186.8 | 0.25 | - | - | - |
| single-2048-sliding-cp8r0 | tilelang@cute | 745.9 | 735.7-752.8 | 1.00 | 2070 | 2061-2077 | 1.00 |
| single-2048-sliding-cp8r0 | cute@cute | 171.2 | 169.8-175.4 | 0.23 | 1434 | 1423-1546 | 0.69 |
| single-2048-sliding-cp8r0 | cute_ws@cute | 74.1 | 73.1-75.3 | 0.10 | 1272 | 1264-1275 | 0.61 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 186.2 | 183.2-192.2 | 0.25 | - | - | - |
| single-2048-sliding-cp8r4 | tilelang@main | 739.5 | 732.5-778.8 | 1.00 | 2018 | 2006-2039 | 1.00 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 179.0 | 174.0-188.2 | 0.24 | - | - | - |
| single-2048-sliding-cp8r4 | tilelang@cute | 733.5 | 731.2-781.4 | 1.00 | 2111 | 2096-2116 | 1.00 |
| single-2048-sliding-cp8r4 | cute@cute | 175.1 | 173.7-177.1 | 0.24 | 1461 | 1448-1469 | 0.69 |
| single-2048-sliding-cp8r4 | cute_ws@cute | 74.1 | 73.6-75.4 | 0.10 | 1284 | 1276-1290 | 0.61 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 188.3 | 186.6-189.3 | 0.26 | - | - | - |
| single-2048-sliding-cp8r7 | tilelang@main | 739.5 | 735.6-746.3 | 1.00 | 2034 | 2020-2067 | 1.00 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 180.9 | 179.1-190.1 | 0.24 | - | - | - |
| single-2048-sliding-cp8r7 | tilelang@cute | 730.1 | 725.4-733.9 | 1.00 | 2112 | 2101-2133 | 1.00 |
| single-2048-sliding-cp8r7 | cute@cute | 175.5 | 172.0-179.4 | 0.24 | 1492 | 1457-1538 | 0.71 |
| single-2048-sliding-cp8r7 | cute_ws@cute | 73.9 | 73.8-75.4 | 0.10 | 1290 | 1272-1311 | 0.61 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 187.0 | 184.5-190.4 | 0.26 | - | - | - |
| short-2048-csa-cp1 | tilelang@main | 1132 | 1121-1135 | 1.00 | 2756 | 2748-2783 | 1.00 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 368.0 | 365.2-371.5 | 0.33 | - | - | - |
| short-2048-csa-cp1 | tilelang@cute | 1125 | 1119-1132 | 1.00 | 2841 | 2823-2847 | 1.00 |
| short-2048-csa-cp1 | cute@cute | 535.0 | 533.7-536.3 | 0.48 | 2190 | 2188-2201 | 0.77 |
| short-2048-csa-cp1 | cute_ws@cute | 260.4 | 260.1-262.9 | 0.23 | 2058 | 2049-2063 | 0.72 |
| short-2048-csa-cp1 | flashmla_fwd_ref@cute | 375.4 | 373.0-378.5 | 0.33 | - | - | - |
| short-2048-csa-cp8r0 | tilelang@main | 753.1 | 737.6-770.5 | 1.00 | 2023 | 2006-2054 | 1.00 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 187.7 | 179.4-191.9 | 0.25 | - | - | - |
| short-2048-csa-cp8r0 | tilelang@cute | 742.1 | 737.1-758.0 | 1.00 | 2052 | 2049-2056 | 1.00 |
| short-2048-csa-cp8r0 | cute@cute | 179.1 | 175.1-179.8 | 0.24 | 1432 | 1421-1437 | 0.70 |
| short-2048-csa-cp8r0 | cute_ws@cute | 86.4 | 85.6-87.1 | 0.12 | 1264 | 1260-1283 | 0.62 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 193.2 | 190.7-200.4 | 0.26 | - | - | - |
| short-2048-csa-cp8r4 | tilelang@main | 742.5 | 736.4-755.8 | 1.00 | 2033 | 2020-2063 | 1.00 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 187.7 | 183.3-192.1 | 0.25 | - | - | - |
| short-2048-csa-cp8r4 | tilelang@cute | 745.8 | 743.4-757.4 | 1.00 | 2090 | 2083-2106 | 1.00 |
| short-2048-csa-cp8r4 | cute@cute | 184.9 | 179.3-187.1 | 0.25 | 1429 | 1412-1444 | 0.68 |
| short-2048-csa-cp8r4 | cute_ws@cute | 90.6 | 90.2-93.0 | 0.12 | 1286 | 1273-1302 | 0.62 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 197.2 | 191.4-199.3 | 0.26 | - | - | - |
| short-2048-csa-cp8r7 | tilelang@main | 762.2 | 749.4-771.5 | 1.00 | 2037 | 2025-2095 | 1.00 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 191.5 | 186.9-192.3 | 0.25 | - | - | - |
| short-2048-csa-cp8r7 | tilelang@cute | 766.4 | 760.0-785.9 | 1.00 | 2058 | 2048-2078 | 1.00 |
| short-2048-csa-cp8r7 | cute@cute | 194.4 | 193.2-198.7 | 0.25 | 1433 | 1424-1442 | 0.70 |
| short-2048-csa-cp8r7 | cute_ws@cute | 93.8 | 92.4-94.6 | 0.12 | 1291 | 1275-1318 | 0.63 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 199.2 | 195.3-202.2 | 0.26 | - | - | - |
| short-2048-hca-cp1 | tilelang@main | 1079 | 1074-1103 | 1.00 | 2489 | 2485-2573 | 1.00 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 340.4 | 337.8-344.8 | 0.32 | - | - | - |
| short-2048-hca-cp1 | tilelang@cute | 1080 | 1078-1082 | 1.00 | 2524 | 2523-2533 | 1.00 |
| short-2048-hca-cp1 | cute@cute | 476.8 | 475.9-477.6 | 0.44 | 1884 | 1859-1889 | 0.75 |
| short-2048-hca-cp1 | cute_ws@cute | 220.3 | 220.0-220.7 | 0.20 | 1680 | 1679-1685 | 0.67 |
| short-2048-hca-cp1 | flashmla_fwd_ref@cute | 345.7 | 344.1-349.2 | 0.32 | - | - | - |
| short-2048-hca-cp8r0 | tilelang@main | 787.4 | 779.9-790.5 | 1.00 | 2154 | 2142-2191 | 1.00 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 191.5 | 187.6-192.1 | 0.24 | - | - | - |
| short-2048-hca-cp8r0 | tilelang@cute | 777.4 | 776.2-797.8 | 1.00 | 2326 | 2163-2350 | 1.00 |
| short-2048-hca-cp8r0 | cute@cute | 201.8 | 200.5-211.1 | 0.26 | 1736 | 1576-1743 | 0.75 |
| short-2048-hca-cp8r0 | cute_ws@cute | 78.3 | 77.2-83.2 | 0.10 | 1337 | 1334-1564 | 0.57 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 196.2 | 188.9-212.1 | 0.25 | - | - | - |
| short-2048-hca-cp8r4 | tilelang@main | 795.0 | 779.4-801.7 | 1.00 | 2162 | 2154-2181 | 1.00 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 193.3 | 190.6-194.5 | 0.24 | - | - | - |
| short-2048-hca-cp8r4 | tilelang@cute | 781.3 | 776.0-790.8 | 1.00 | 2386 | 2370-2393 | 1.00 |
| short-2048-hca-cp8r4 | cute@cute | 200.0 | 199.3-201.3 | 0.26 | 1770 | 1762-1776 | 0.74 |
| short-2048-hca-cp8r4 | cute_ws@cute | 77.1 | 76.5-79.3 | 0.10 | 1577 | 1575-1599 | 0.66 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 197.2 | 196.6-203.4 | 0.25 | - | - | - |
| short-2048-hca-cp8r7 | tilelang@main | 792.8 | 779.8-798.3 | 1.00 | 2193 | 2161-2199 | 1.00 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 188.9 | 188.3-192.7 | 0.24 | - | - | - |
| short-2048-hca-cp8r7 | tilelang@cute | 782.0 | 776.4-784.1 | 1.00 | 2164 | 2147-2253 | 1.00 |
| short-2048-hca-cp8r7 | cute@cute | 207.3 | 204.1-208.7 | 0.27 | 1545 | 1536-1572 | 0.71 |
| short-2048-hca-cp8r7 | cute_ws@cute | 80.0 | 79.7-82.2 | 0.10 | 1337 | 1331-1342 | 0.62 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 195.9 | 192.8-198.6 | 0.25 | - | - | - |
| short-2048-sliding-cp1 | tilelang@main | 1003 | 994.3-1018 | 1.00 | 2330 | 2322-2398 | 1.00 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 293.3 | 291.6-301.8 | 0.29 | - | - | - |
| short-2048-sliding-cp1 | tilelang@cute | 990.1 | 973.9-1004 | 1.00 | 2319 | 2314-2330 | 1.00 |
| short-2048-sliding-cp1 | cute@cute | 407.5 | 406.2-410.1 | 0.41 | 1705 | 1684-1707 | 0.74 |
| short-2048-sliding-cp1 | cute_ws@cute | 170.6 | 168.3-171.7 | 0.17 | 1544 | 1540-1560 | 0.67 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@cute | 298.0 | 293.6-301.4 | 0.30 | - | - | - |
| short-2048-sliding-cp8r0 | tilelang@main | 746.8 | 734.3-752.0 | 1.00 | 2042 | 2039-2056 | 1.00 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 181.9 | 179.7-187.0 | 0.24 | - | - | - |
| short-2048-sliding-cp8r0 | tilelang@cute | 759.6 | 744.6-767.3 | 1.00 | 2057 | 2047-2068 | 1.00 |
| short-2048-sliding-cp8r0 | cute@cute | 171.9 | 169.9-173.3 | 0.23 | 1425 | 1418-1436 | 0.69 |
| short-2048-sliding-cp8r0 | cute_ws@cute | 74.2 | 73.3-78.0 | 0.10 | 1293 | 1260-1351 | 0.63 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 189.4 | 188.5-197.2 | 0.25 | - | - | - |
| short-2048-sliding-cp8r4 | tilelang@main | 756.9 | 751.3-773.4 | 1.00 | 2022 | 2012-2028 | 1.00 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 189.8 | 187.4-194.1 | 0.25 | - | - | - |
| short-2048-sliding-cp8r4 | tilelang@cute | 751.5 | 751.1-788.4 | 1.00 | 2109 | 2100-2216 | 1.00 |
| short-2048-sliding-cp8r4 | cute@cute | 175.3 | 172.0-177.7 | 0.23 | 1449 | 1431-1476 | 0.69 |
| short-2048-sliding-cp8r4 | cute_ws@cute | 74.9 | 74.2-76.4 | 0.10 | 1298 | 1274-1322 | 0.62 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 199.2 | 188.8-202.7 | 0.27 | - | - | - |
| short-2048-sliding-cp8r7 | tilelang@main | 736.4 | 731.7-751.1 | 1.00 | 2044 | 2038-2069 | 1.00 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 185.1 | 179.8-188.5 | 0.25 | - | - | - |
| short-2048-sliding-cp8r7 | tilelang@cute | 763.5 | 760.5-776.6 | 1.00 | 2094 | 2085-2117 | 1.00 |
| short-2048-sliding-cp8r7 | cute@cute | 179.6 | 173.6-179.9 | 0.24 | 1437 | 1431-1448 | 0.69 |
| short-2048-sliding-cp8r7 | cute_ws@cute | 77.0 | 76.0-78.4 | 0.10 | 1289 | 1272-1291 | 0.62 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 193.5 | 184.4-198.2 | 0.25 | - | - | - |
| heavy-2048-csa-cp1 | tilelang@main | 1171 | 1161-1186 | 1.00 | 2955 | 2937-2956 | 1.00 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 389.7 | 383.9-390.4 | 0.33 | - | - | - |
| heavy-2048-csa-cp1 | tilelang@cute | 1167 | 1163-1180 | 1.00 | 3023 | 2999-3139 | 1.00 |
| heavy-2048-csa-cp1 | cute@cute | 570.7 | 567.8-573.4 | 0.49 | 2379 | 2354-2527 | 0.79 |
| heavy-2048-csa-cp1 | cute_ws@cute | 280.0 | 278.6-280.8 | 0.24 | 2230 | 2215-2370 | 0.74 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@cute | 394.6 | 393.1-401.5 | 0.34 | - | - | - |
| heavy-2048-csa-cp8r0 | tilelang@main | 747.5 | 737.9-759.0 | 1.00 | 2043 | 2029-2055 | 1.00 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 189.1 | 186.1-190.5 | 0.25 | - | - | - |
| heavy-2048-csa-cp8r0 | tilelang@cute | 749.7 | 737.5-753.1 | 1.00 | 2081 | 2068-2089 | 1.00 |
| heavy-2048-csa-cp8r0 | cute@cute | 182.4 | 179.4-184.1 | 0.24 | 1458 | 1437-1521 | 0.70 |
| heavy-2048-csa-cp8r0 | cute_ws@cute | 87.4 | 86.8-89.0 | 0.12 | 1301 | 1287-1436 | 0.63 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 207.1 | 196.8-212.6 | 0.28 | - | - | - |
| heavy-2048-csa-cp8r4 | tilelang@main | 777.6 | 755.6-789.9 | 1.00 | 2065 | 2036-2070 | 1.00 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 200.7 | 189.6-207.8 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r4 | tilelang@cute | 764.7 | 750.2-773.8 | 1.00 | 2074 | 2067-2085 | 1.00 |
| heavy-2048-csa-cp8r4 | cute@cute | 194.3 | 194.0-198.4 | 0.25 | 1440 | 1436-1452 | 0.69 |
| heavy-2048-csa-cp8r4 | cute_ws@cute | 93.3 | 93.0-95.6 | 0.12 | 1287 | 1275-1295 | 0.62 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 199.4 | 194.7-201.5 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r7 | tilelang@main | 793.2 | 765.5-859.1 | 1.00 | 2034 | 2029-2075 | 1.00 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 209.7 | 204.5-224.9 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r7 | tilelang@cute | 784.4 | 775.7-804.3 | 1.00 | 2086 | 2077-2099 | 1.00 |
| heavy-2048-csa-cp8r7 | cute@cute | 210.5 | 208.7-217.6 | 0.27 | 1434 | 1424-1445 | 0.69 |
| heavy-2048-csa-cp8r7 | cute_ws@cute | 103.8 | 101.9-104.9 | 0.13 | 1285 | 1278-1297 | 0.62 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 209.8 | 205.7-214.1 | 0.27 | - | - | - |
| heavy-2048-hca-cp1 | tilelang@main | 1074 | 1069-1080 | 1.00 | 2464 | 2458-2470 | 1.00 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 335.2 | 329.4-337.8 | 0.31 | - | - | - |
| heavy-2048-hca-cp1 | tilelang@cute | 1070 | 1064-1082 | 1.00 | 2500 | 2491-2511 | 1.00 |
| heavy-2048-hca-cp1 | cute@cute | 471.2 | 468.2-477.0 | 0.44 | 1862 | 1845-1874 | 0.74 |
| heavy-2048-hca-cp1 | cute_ws@cute | 213.0 | 212.3-214.1 | 0.20 | 1673 | 1656-1685 | 0.67 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@cute | 340.7 | 332.2-345.2 | 0.32 | - | - | - |
| heavy-2048-hca-cp8r0 | tilelang@main | 774.2 | 771.3-778.2 | 1.00 | 2135 | 2131-2143 | 1.00 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 189.3 | 188.5-195.6 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r0 | tilelang@cute | 789.9 | 774.5-801.1 | 1.00 | 2194 | 2185-2217 | 1.00 |
| heavy-2048-hca-cp8r0 | cute@cute | 198.6 | 196.5-202.9 | 0.25 | 1566 | 1552-1583 | 0.71 |
| heavy-2048-hca-cp8r0 | cute_ws@cute | 77.4 | 75.4-80.2 | 0.10 | 1367 | 1358-1373 | 0.62 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 194.5 | 190.4-206.7 | 0.25 | - | - | - |
| heavy-2048-hca-cp8r4 | tilelang@main | 782.0 | 771.9-791.9 | 1.00 | 2122 | 2120-2137 | 1.00 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 189.5 | 183.3-193.8 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r4 | tilelang@cute | 817.1 | 803.7-852.1 | 1.00 | 2234 | 2218-2270 | 1.00 |
| heavy-2048-hca-cp8r4 | cute@cute | 215.2 | 210.1-221.6 | 0.26 | 1587 | 1578-1624 | 0.71 |
| heavy-2048-hca-cp8r4 | cute_ws@cute | 83.3 | 81.6-85.4 | 0.10 | 1378 | 1369-1393 | 0.62 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 207.9 | 201.6-215.4 | 0.25 | - | - | - |
| heavy-2048-hca-cp8r7 | tilelang@main | 786.8 | 778.1-795.3 | 1.00 | 2157 | 2147-2182 | 1.00 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 189.8 | 187.0-192.2 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r7 | tilelang@cute | 790.8 | 785.8-807.5 | 1.00 | 2218 | 2198-2225 | 1.00 |
| heavy-2048-hca-cp8r7 | cute@cute | 207.6 | 206.2-209.6 | 0.26 | 1553 | 1546-1570 | 0.70 |
| heavy-2048-hca-cp8r7 | cute_ws@cute | 81.5 | 80.5-82.9 | 0.10 | 1371 | 1340-1469 | 0.62 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 199.8 | 195.5-203.1 | 0.25 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang@main | 1000 | 986.0-1002 | 1.00 | 2294 | 2292-2303 | 1.00 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 291.8 | 289.1-294.8 | 0.29 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang@cute | 1002 | 992.3-1006 | 1.00 | 2347 | 2335-2355 | 1.00 |
| heavy-2048-sliding-cp1 | cute@cute | 407.0 | 405.7-408.1 | 0.41 | 1710 | 1708-1714 | 0.73 |
| heavy-2048-sliding-cp1 | cute_ws@cute | 169.2 | 168.5-172.9 | 0.17 | 1561 | 1543-1574 | 0.67 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@cute | 296.7 | 292.8-301.8 | 0.30 | - | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@main | 736.0 | 727.2-829.0 | 1.00 | 2039 | 2024-2043 | 1.00 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 180.0 | 176.3-183.7 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@cute | 742.4 | 738.0-784.6 | 1.00 | 2100 | 2087-2115 | 1.00 |
| heavy-2048-sliding-cp8r0 | cute@cute | 174.4 | 173.0-174.8 | 0.23 | 1458 | 1451-1488 | 0.69 |
| heavy-2048-sliding-cp8r0 | cute_ws@cute | 75.0 | 72.5-77.1 | 0.10 | 1290 | 1285-1295 | 0.61 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 192.1 | 184.3-193.4 | 0.26 | - | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@main | 742.3 | 731.2-750.7 | 1.00 | 2036 | 2019-2052 | 1.00 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 180.7 | 175.9-183.0 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@cute | 732.4 | 728.8-742.1 | 1.00 | 2118 | 2107-2127 | 1.00 |
| heavy-2048-sliding-cp8r4 | cute@cute | 172.3 | 168.9-172.6 | 0.24 | 1442 | 1437-1449 | 0.68 |
| heavy-2048-sliding-cp8r4 | cute_ws@cute | 73.7 | 73.7-75.6 | 0.10 | 1290 | 1278-1296 | 0.61 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 190.4 | 183.3-192.5 | 0.26 | - | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@main | 743.6 | 736.9-744.8 | 1.00 | 2071 | 2052-2104 | 1.00 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 182.7 | 175.3-184.9 | 0.25 | - | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@cute | 752.7 | 735.8-768.5 | 1.00 | 2108 | 2102-2142 | 1.00 |
| heavy-2048-sliding-cp8r7 | cute@cute | 174.4 | 173.3-177.8 | 0.23 | 1476 | 1445-1546 | 0.70 |
| heavy-2048-sliding-cp8r7 | cute_ws@cute | 74.9 | 73.6-76.4 | 0.10 | 1297 | 1288-1310 | 0.62 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 192.5 | 188.5-193.3 | 0.26 | - | - | - |
| tiny-2048-csa-cp1 | tilelang@main | 1043 | 1040-1056 | 1.00 | 2359 | 2333-2369 | 1.00 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 333.9 | 329.9-336.8 | 0.32 | - | - | - |
| tiny-2048-csa-cp1 | tilelang@cute | 1041 | 1036-1067 | 1.00 | 2398 | 2385-2407 | 1.00 |
| tiny-2048-csa-cp1 | cute@cute | 452.4 | 451.7-454.9 | 0.43 | 1750 | 1735-1756 | 0.73 |
| tiny-2048-csa-cp1 | cute_ws@cute | 228.6 | 227.8-230.7 | 0.22 | 1610 | 1591-1615 | 0.67 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@cute | 340.5 | 334.5-344.4 | 0.33 | - | - | - |
| tiny-2048-csa-cp8r0 | tilelang@main | 753.9 | 747.6-789.7 | 1.00 | 2056 | 2032-2068 | 1.00 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 192.5 | 185.3-201.4 | 0.26 | - | - | - |
| tiny-2048-csa-cp8r0 | tilelang@cute | 761.5 | 757.3-787.6 | 1.00 | 2114 | 2108-2145 | 1.00 |
| tiny-2048-csa-cp8r0 | cute@cute | 181.9 | 175.2-183.1 | 0.24 | 1463 | 1447-1473 | 0.69 |
| tiny-2048-csa-cp8r0 | cute_ws@cute | 87.3 | 85.9-88.4 | 0.11 | 1294 | 1292-1315 | 0.61 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 192.8 | 187.7-197.3 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r4 | tilelang@main | 748.5 | 741.9-755.9 | 1.00 | 2029 | 2015-2055 | 1.00 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 187.9 | 184.4-189.9 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r4 | tilelang@cute | 760.3 | 754.3-760.6 | 1.00 | 2097 | 2095-2115 | 1.00 |
| tiny-2048-csa-cp8r4 | cute@cute | 179.1 | 177.5-180.5 | 0.24 | 1447 | 1440-1460 | 0.69 |
| tiny-2048-csa-cp8r4 | cute_ws@cute | 85.7 | 85.0-86.3 | 0.11 | 1304 | 1296-1308 | 0.62 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 189.9 | 185.3-195.3 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r7 | tilelang@main | 761.0 | 757.8-769.9 | 1.00 | 2040 | 2032-2079 | 1.00 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 189.9 | 186.1-198.6 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r7 | tilelang@cute | 766.0 | 753.8-778.0 | 1.00 | 2123 | 2094-2129 | 1.00 |
| tiny-2048-csa-cp8r7 | cute@cute | 181.3 | 177.6-185.5 | 0.24 | 1465 | 1455-1483 | 0.69 |
| tiny-2048-csa-cp8r7 | cute_ws@cute | 87.5 | 86.3-89.4 | 0.11 | 1295 | 1285-1311 | 0.61 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 194.7 | 192.6-199.2 | 0.25 | - | - | - |
| tiny-2048-hca-cp1 | tilelang@main | 962.7 | 956.2-975.0 | 1.00 | 2093 | 2080-2128 | 1.00 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 292.2 | 290.6-305.2 | 0.30 | - | - | - |
| tiny-2048-hca-cp1 | tilelang@cute | 959.3 | 945.0-982.0 | 1.00 | 2183 | 2154-2294 | 1.00 |
| tiny-2048-hca-cp1 | cute@cute | 372.5 | 370.6-383.4 | 0.39 | 1516 | 1510-1694 | 0.69 |
| tiny-2048-hca-cp1 | cute_ws@cute | 168.5 | 167.2-173.6 | 0.18 | 1367 | 1357-1508 | 0.63 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@cute | 297.1 | 295.7-304.1 | 0.31 | - | - | - |
| tiny-2048-hca-cp8r0 | tilelang@main | 736.5 | 733.4-737.7 | 1.00 | 2030 | 2013-2052 | 1.00 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 182.6 | 176.8-187.7 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r0 | tilelang@cute | 754.1 | 737.7-765.9 | 1.00 | 2096 | 2091-2112 | 1.00 |
| tiny-2048-hca-cp8r0 | cute@cute | 171.6 | 170.5-175.7 | 0.23 | 1458 | 1449-1466 | 0.70 |
| tiny-2048-hca-cp8r0 | cute_ws@cute | 75.1 | 73.8-75.6 | 0.10 | 1288 | 1273-1320 | 0.61 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 191.1 | 187.9-197.8 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r4 | tilelang@main | 735.0 | 723.8-756.2 | 1.00 | 2049 | 2011-2074 | 1.00 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 180.9 | 179.2-183.4 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r4 | tilelang@cute | 748.4 | 740.4-762.2 | 1.00 | 2093 | 2076-2185 | 1.00 |
| tiny-2048-hca-cp8r4 | cute@cute | 171.5 | 168.2-174.9 | 0.23 | 1442 | 1434-1464 | 0.69 |
| tiny-2048-hca-cp8r4 | cute_ws@cute | 74.8 | 73.6-76.0 | 0.10 | 1277 | 1264-1290 | 0.61 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 185.7 | 183.0-194.3 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r7 | tilelang@main | 751.2 | 738.8-819.8 | 1.00 | 2099 | 2044-2164 | 1.00 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 190.0 | 184.2-195.8 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r7 | tilelang@cute | 745.5 | 737.9-763.9 | 1.00 | 2053 | 2049-2061 | 1.00 |
| tiny-2048-hca-cp8r7 | cute@cute | 170.1 | 168.2-172.7 | 0.23 | 1421 | 1410-1428 | 0.69 |
| tiny-2048-hca-cp8r7 | cute_ws@cute | 75.2 | 74.3-75.9 | 0.10 | 1252 | 1247-1270 | 0.61 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 189.2 | 180.2-190.2 | 0.25 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang@main | 957.8 | 952.0-962.9 | 1.00 | 2108 | 2103-2118 | 1.00 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 290.5 | 284.1-294.3 | 0.30 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang@cute | 952.0 | 940.3-952.7 | 1.00 | 2139 | 2125-2143 | 1.00 |
| tiny-2048-sliding-cp1 | cute@cute | 377.0 | 375.8-379.6 | 0.40 | 1493 | 1478-1510 | 0.70 |
| tiny-2048-sliding-cp1 | cute_ws@cute | 169.7 | 168.4-173.2 | 0.18 | 1357 | 1336-1361 | 0.63 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@cute | 299.4 | 296.0-302.0 | 0.31 | - | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@main | 731.7 | 730.4-739.6 | 1.00 | 2136 | 2058-2216 | 1.00 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 179.8 | 173.2-180.5 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@cute | 753.5 | 743.3-761.2 | 1.00 | 2071 | 2059-2094 | 1.00 |
| tiny-2048-sliding-cp8r0 | cute@cute | 169.0 | 166.0-171.2 | 0.22 | 1442 | 1435-1445 | 0.70 |
| tiny-2048-sliding-cp8r0 | cute_ws@cute | 74.6 | 73.9-75.0 | 0.10 | 1266 | 1264-1267 | 0.61 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 188.2 | 184.9-194.3 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@main | 738.5 | 727.1-742.4 | 1.00 | 2024 | 2010-2050 | 1.00 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 178.4 | 171.8-182.8 | 0.24 | - | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@cute | 724.0 | 716.1-740.4 | 1.00 | 2101 | 2088-2118 | 1.00 |
| tiny-2048-sliding-cp8r4 | cute@cute | 170.1 | 169.7-174.9 | 0.23 | 1443 | 1430-1460 | 0.69 |
| tiny-2048-sliding-cp8r4 | cute_ws@cute | 76.1 | 74.7-77.2 | 0.11 | 1274 | 1272-1281 | 0.61 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 189.0 | 183.6-194.7 | 0.26 | - | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@main | 758.1 | 748.7-774.6 | 1.00 | 2049 | 2038-2151 | 1.00 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 186.8 | 182.1-192.1 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@cute | 749.7 | 746.0-760.1 | 1.00 | 2090 | 2081-2106 | 1.00 |
| tiny-2048-sliding-cp8r7 | cute@cute | 174.4 | 173.6-179.5 | 0.23 | 1448 | 1440-1456 | 0.69 |
| tiny-2048-sliding-cp8r7 | cute_ws@cute | 75.8 | 74.8-76.3 | 0.10 | 1279 | 1276-1287 | 0.61 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 191.2 | 189.1-193.0 | 0.26 | - | - | - |
| single-4096-csa-cp1 | tilelang@main | 1890 | 1888-1954 | 1.00 | 5972 | 5965-5976 | 1.00 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 715.4 | 711.7-718.7 | 0.38 | - | - | - |
| single-4096-csa-cp1 | tilelang@cute | 1885 | 1877-1898 | 1.00 | 6017 | 6015-6031 | 1.00 |
| single-4096-csa-cp1 | cute@cute | 1273 | 1269-1275 | 0.68 | 5337 | 5330-5344 | 0.89 |
| single-4096-csa-cp1 | cute_ws@cute | 606.1 | 605.3-607.6 | 0.32 | 4694 | 4680-4699 | 0.78 |
| single-4096-csa-cp1 | flashmla_fwd_ref@cute | 723.3 | 720.3-725.4 | 0.38 | - | - | - |
| single-4096-csa-cp8r0 | tilelang@main | 795.7 | 791.3-798.4 | 1.00 | 2037 | 2027-2047 | 1.00 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 206.3 | 203.2-212.2 | 0.26 | - | - | - |
| single-4096-csa-cp8r0 | tilelang@cute | 798.1 | 792.1-801.7 | 1.00 | 2065 | 2045-2075 | 1.00 |
| single-4096-csa-cp8r0 | cute@cute | 226.6 | 223.9-227.4 | 0.28 | 1429 | 1408-1440 | 0.69 |
| single-4096-csa-cp8r0 | cute_ws@cute | 107.5 | 106.4-108.6 | 0.13 | 1260 | 1250-1262 | 0.61 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 219.3 | 214.0-222.0 | 0.27 | - | - | - |
| single-4096-csa-cp8r4 | tilelang@main | 869.0 | 862.6-883.1 | 1.00 | 2333 | 2321-2341 | 1.00 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 248.4 | 245.0-250.7 | 0.29 | - | - | - |
| single-4096-csa-cp8r4 | tilelang@cute | 880.2 | 873.9-887.3 | 1.00 | 2375 | 2365-2395 | 1.00 |
| single-4096-csa-cp8r4 | cute@cute | 305.6 | 301.2-306.0 | 0.35 | 1730 | 1725-1750 | 0.73 |
| single-4096-csa-cp8r4 | cute_ws@cute | 148.2 | 146.1-148.5 | 0.17 | 1582 | 1570-1585 | 0.67 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 253.0 | 250.6-255.9 | 0.29 | - | - | - |
| single-4096-csa-cp8r7 | tilelang@main | 880.8 | 872.3-893.7 | 1.00 | 2349 | 2340-2358 | 1.00 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 252.0 | 246.9-253.2 | 0.29 | - | - | - |
| single-4096-csa-cp8r7 | tilelang@cute | 890.1 | 865.4-906.7 | 1.00 | 2368 | 2365-2380 | 1.00 |
| single-4096-csa-cp8r7 | cute@cute | 304.7 | 301.5-311.4 | 0.34 | 1730 | 1715-1749 | 0.73 |
| single-4096-csa-cp8r7 | cute_ws@cute | 148.9 | 148.7-149.8 | 0.17 | 1580 | 1576-1591 | 0.67 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 261.5 | 255.3-266.8 | 0.29 | - | - | - |
| single-4096-hca-cp1 | tilelang@main | 1432 | 1411-1475 | 1.00 | 3134 | 3126-3142 | 1.00 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 505.5 | 499.0-532.2 | 0.35 | - | - | - |
| single-4096-hca-cp1 | tilelang@cute | 1418 | 1402-1427 | 1.00 | 3161 | 3153-3187 | 1.00 |
| single-4096-hca-cp1 | cute@cute | 789.1 | 786.2-790.1 | 0.56 | 2523 | 2522-2523 | 0.80 |
| single-4096-hca-cp1 | cute_ws@cute | 372.6 | 371.0-373.2 | 0.26 | 2326 | 2321-2333 | 0.74 |
| single-4096-hca-cp1 | flashmla_fwd_ref@cute | 512.9 | 507.4-517.8 | 0.36 | - | - | - |
| single-4096-hca-cp8r0 | tilelang@main | 828.2 | 816.4-834.6 | 1.00 | 2126 | 2122-2148 | 1.00 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 217.1 | 213.7-217.8 | 0.26 | - | - | - |
| single-4096-hca-cp8r0 | tilelang@cute | 832.5 | 823.3-844.6 | 1.00 | 2225 | 2214-2320 | 1.00 |
| single-4096-hca-cp8r0 | cute@cute | 248.0 | 245.5-258.8 | 0.30 | 1563 | 1549-1579 | 0.70 |
| single-4096-hca-cp8r0 | cute_ws@cute | 101.3 | 100.6-105.9 | 0.12 | 1362 | 1334-1375 | 0.61 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 226.8 | 219.8-233.2 | 0.27 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 824.2 | 816.0-832.5 | 1.00 | 2143 | 2120-2164 | 1.00 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 216.8 | 214.5-219.7 | 0.26 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@cute | 840.5 | 836.9-843.1 | 1.00 | 2192 | 2189-2214 | 1.00 |
| single-4096-hca-cp8r4 | cute@cute | 247.5 | 243.9-252.0 | 0.29 | 1555 | 1547-1558 | 0.71 |
| single-4096-hca-cp8r4 | cute_ws@cute | 102.3 | 101.5-105.7 | 0.12 | 1337 | 1331-1350 | 0.61 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 222.4 | 214.7-223.7 | 0.26 | - | - | - |
| single-4096-hca-cp8r7 | tilelang@main | 827.1 | 813.1-838.7 | 1.00 | 2159 | 2139-2187 | 1.00 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 217.2 | 212.4-222.6 | 0.26 | - | - | - |
| single-4096-hca-cp8r7 | tilelang@cute | 832.8 | 828.7-838.8 | 1.00 | 2172 | 2156-2194 | 1.00 |
| single-4096-hca-cp8r7 | cute@cute | 250.2 | 245.8-251.3 | 0.30 | 1531 | 1526-1538 | 0.70 |
| single-4096-hca-cp8r7 | cute_ws@cute | 103.4 | 102.2-104.1 | 0.12 | 1334 | 1322-1351 | 0.61 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 224.0 | 221.7-227.0 | 0.27 | - | - | - |
| single-4096-sliding-cp1 | tilelang@main | 1297 | 1285-1352 | 1.00 | 2859 | 2858-2862 | 1.00 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 420.2 | 412.8-423.1 | 0.32 | - | - | - |
| single-4096-sliding-cp1 | tilelang@cute | 1285 | 1283-1299 | 1.00 | 2892 | 2876-2921 | 1.00 |
| single-4096-sliding-cp1 | cute@cute | 679.2 | 674.9-681.1 | 0.53 | 2258 | 2251-2263 | 0.78 |
| single-4096-sliding-cp1 | cute_ws@cute | 272.9 | 271.2-273.4 | 0.21 | 2117 | 2106-2123 | 0.73 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@cute | 424.8 | 420.7-427.1 | 0.33 | - | - | - |
| single-4096-sliding-cp8r0 | tilelang@main | 790.1 | 775.7-812.3 | 1.00 | 2034 | 2027-2045 | 1.00 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 199.3 | 197.8-208.9 | 0.25 | - | - | - |
| single-4096-sliding-cp8r0 | tilelang@cute | 786.9 | 771.5-793.1 | 1.00 | 2072 | 2062-2077 | 1.00 |
| single-4096-sliding-cp8r0 | cute@cute | 207.5 | 206.6-210.5 | 0.26 | 1439 | 1436-1443 | 0.69 |
| single-4096-sliding-cp8r0 | cute_ws@cute | 89.2 | 88.1-90.5 | 0.11 | 1278 | 1269-1282 | 0.62 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 209.2 | 203.5-211.3 | 0.27 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 791.1 | 779.7-892.0 | 1.00 | 2057 | 2032-2070 | 1.00 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 197.7-217.5 | 0.26 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@cute | 797.5 | 779.5-815.7 | 1.00 | 2114 | 2104-2118 | 1.00 |
| single-4096-sliding-cp8r4 | cute@cute | 208.3 | 206.3-209.9 | 0.26 | 1453 | 1446-1464 | 0.69 |
| single-4096-sliding-cp8r4 | cute_ws@cute | 88.7 | 88.2-89.0 | 0.11 | 1283 | 1281-1334 | 0.61 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 205.9 | 200.2-208.8 | 0.26 | - | - | - |
| single-4096-sliding-cp8r7 | tilelang@main | 785.2 | 769.9-787.5 | 1.00 | 2065 | 2049-2079 | 1.00 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 194.0 | 192.7-204.3 | 0.25 | - | - | - |
| single-4096-sliding-cp8r7 | tilelang@cute | 793.7 | 784.5-803.0 | 1.00 | 2106 | 2081-2126 | 1.00 |
| single-4096-sliding-cp8r7 | cute@cute | 212.4 | 209.4-212.9 | 0.27 | 1456 | 1446-1508 | 0.69 |
| single-4096-sliding-cp8r7 | cute_ws@cute | 89.8 | 88.0-90.2 | 0.11 | 1288 | 1279-1306 | 0.61 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 210.1 | 200.9-214.1 | 0.26 | - | - | - |
| short-4096-csa-cp1 | tilelang@main | 1523 | 1513-1536 | 1.00 | 3873 | 3871-3875 | 1.00 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 552.9 | 546.9-555.2 | 0.36 | - | - | - |
| short-4096-csa-cp1 | tilelang@cute | 1512 | 1509-1538 | 1.00 | 3893 | 3884-3904 | 1.00 |
| short-4096-csa-cp1 | cute@cute | 902.5 | 898.6-908.5 | 0.60 | 3234 | 3229-3235 | 0.83 |
| short-4096-csa-cp1 | cute_ws@cute | 427.7 | 424.8-429.8 | 0.28 | 2964 | 2955-2975 | 0.76 |
| short-4096-csa-cp1 | flashmla_fwd_ref@cute | 550.3 | 548.8-567.0 | 0.36 | - | - | - |
| short-4096-csa-cp8r0 | tilelang@main | 827.7 | 802.6-859.4 | 1.00 | 2054 | 2046-2075 | 1.00 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 222.4 | 213.5-232.8 | 0.27 | - | - | - |
| short-4096-csa-cp8r0 | tilelang@cute | 797.2 | 784.4-803.7 | 1.00 | 2073 | 2061-2084 | 1.00 |
| short-4096-csa-cp8r0 | cute@cute | 224.9 | 220.4-227.2 | 0.28 | 1436 | 1431-1447 | 0.69 |
| short-4096-csa-cp8r0 | cute_ws@cute | 106.7 | 106.3-107.5 | 0.13 | 1287 | 1274-1289 | 0.62 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 213.2 | 209.9-217.4 | 0.27 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 876.9 | 821.4-901.4 | 1.00 | 2145 | 2070-2654 | 1.00 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 227.1 | 220.9-229.2 | 0.26 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@cute | 793.0 | 788.5-800.6 | 1.00 | 2079 | 2069-2089 | 1.00 |
| short-4096-csa-cp8r4 | cute@cute | 224.9 | 223.5-230.2 | 0.28 | 1445 | 1436-1469 | 0.69 |
| short-4096-csa-cp8r4 | cute_ws@cute | 110.5 | 109.6-113.3 | 0.14 | 1301 | 1280-1313 | 0.63 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 217.8 | 215.5-223.3 | 0.27 | - | - | - |
| short-4096-csa-cp8r7 | tilelang@main | 818.1 | 806.9-826.3 | 1.00 | 2060 | 2047-2075 | 1.00 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 210.7 | 208.5-216.6 | 0.26 | - | - | - |
| short-4096-csa-cp8r7 | tilelang@cute | 799.6 | 791.1-804.8 | 1.00 | 2064 | 2047-2086 | 1.00 |
| short-4096-csa-cp8r7 | cute@cute | 227.6 | 227.1-228.4 | 0.28 | 1439 | 1418-1450 | 0.70 |
| short-4096-csa-cp8r7 | cute_ws@cute | 109.6 | 108.0-110.8 | 0.14 | 1270 | 1257-1274 | 0.62 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 218.5 | 212.5-221.8 | 0.27 | - | - | - |
| short-4096-hca-cp1 | tilelang@main | 1411 | 1404-1416 | 1.00 | 3089 | 3082-3102 | 1.00 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 500.1 | 497.7-502.9 | 0.35 | - | - | - |
| short-4096-hca-cp1 | tilelang@cute | 1388 | 1379-1398 | 1.00 | 3082 | 3078-3110 | 1.00 |
| short-4096-hca-cp1 | cute@cute | 762.4 | 762.2-769.7 | 0.55 | 2463 | 2458-2472 | 0.80 |
| short-4096-hca-cp1 | cute_ws@cute | 354.8 | 353.7-358.1 | 0.26 | 2269 | 2264-2274 | 0.74 |
| short-4096-hca-cp1 | flashmla_fwd_ref@cute | 499.9 | 497.5-504.2 | 0.36 | - | - | - |
| short-4096-hca-cp8r0 | tilelang@main | 844.5 | 832.2-863.5 | 1.00 | 2180 | 2165-2244 | 1.00 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 216.0 | 211.8-217.3 | 0.26 | - | - | - |
| short-4096-hca-cp8r0 | tilelang@cute | 825.2 | 815.0-837.7 | 1.00 | 2204 | 2174-2221 | 1.00 |
| short-4096-hca-cp8r0 | cute@cute | 243.9 | 242.4-244.8 | 0.30 | 1554 | 1548-1569 | 0.71 |
| short-4096-hca-cp8r0 | cute_ws@cute | 101.1 | 99.4-102.0 | 0.12 | 1328 | 1325-1335 | 0.60 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 221.0 | 217.1-222.9 | 0.27 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 826.5 | 818.9-845.1 | 1.00 | 2157 | 2154-2171 | 1.00 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 208.2 | 206.1-217.1 | 0.25 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@cute | 820.5 | 818.2-828.1 | 1.00 | 2182 | 2162-2191 | 1.00 |
| short-4096-hca-cp8r4 | cute@cute | 239.2 | 237.4-241.6 | 0.29 | 1544 | 1529-1554 | 0.71 |
| short-4096-hca-cp8r4 | cute_ws@cute | 98.1 | 97.7-100.3 | 0.12 | 1342 | 1334-1347 | 0.62 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 214.8 | 213.5-216.3 | 0.26 | - | - | - |
| short-4096-hca-cp8r7 | tilelang@main | 831.1 | 816.4-835.0 | 1.00 | 2197 | 2167-2202 | 1.00 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 214.2 | 209.4-216.6 | 0.26 | - | - | - |
| short-4096-hca-cp8r7 | tilelang@cute | 826.7 | 817.6-836.7 | 1.00 | 2198 | 2193-2205 | 1.00 |
| short-4096-hca-cp8r7 | cute@cute | 243.1 | 241.6-249.9 | 0.29 | 1551 | 1539-1564 | 0.71 |
| short-4096-hca-cp8r7 | cute_ws@cute | 101.2 | 99.0-102.1 | 0.12 | 1344 | 1330-1370 | 0.61 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 219.1 | 214.9-223.0 | 0.27 | - | - | - |
| short-4096-sliding-cp1 | tilelang@main | 1312 | 1272-1315 | 1.00 | 2842 | 2832-2846 | 1.00 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 423.2 | 414.6-434.4 | 0.32 | - | - | - |
| short-4096-sliding-cp1 | tilelang@cute | 1284 | 1280-1291 | 1.00 | 2886 | 2880-2909 | 1.00 |
| short-4096-sliding-cp1 | cute@cute | 663.8 | 660.5-670.0 | 0.52 | 2220 | 2210-2222 | 0.77 |
| short-4096-sliding-cp1 | cute_ws@cute | 267.9 | 265.9-269.2 | 0.21 | 2074 | 2070-2082 | 0.72 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@cute | 425.5 | 417.3-430.2 | 0.33 | - | - | - |
| short-4096-sliding-cp8r0 | tilelang@main | 793.8 | 787.3-805.0 | 1.00 | 2036 | 2030-2048 | 1.00 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 197.4 | 192.0-199.4 | 0.25 | - | - | - |
| short-4096-sliding-cp8r0 | tilelang@cute | 789.4 | 775.8-814.3 | 1.00 | 2123 | 2104-2153 | 1.00 |
| short-4096-sliding-cp8r0 | cute@cute | 210.1 | 208.0-212.7 | 0.27 | 1458 | 1440-1549 | 0.69 |
| short-4096-sliding-cp8r0 | cute_ws@cute | 89.2 | 88.5-90.4 | 0.11 | 1295 | 1274-1383 | 0.61 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 201.3 | 200.7-206.7 | 0.26 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 794.0 | 780.5-815.6 | 1.00 | 2041 | 2033-2066 | 1.00 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 197.7 | 193.1-216.8 | 0.25 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@cute | 779.6 | 774.4-827.2 | 1.00 | 2094 | 2077-2112 | 1.00 |
| short-4096-sliding-cp8r4 | cute@cute | 210.2 | 207.8-211.1 | 0.27 | 1444 | 1429-1456 | 0.69 |
| short-4096-sliding-cp8r4 | cute_ws@cute | 90.2 | 88.8-91.2 | 0.12 | 1281 | 1272-1297 | 0.61 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 204.8 | 198.9-209.0 | 0.26 | - | - | - |
| short-4096-sliding-cp8r7 | tilelang@main | 774.8 | 766.8-785.9 | 1.00 | 2086 | 2079-2120 | 1.00 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 195.7 | 192.6-198.1 | 0.25 | - | - | - |
| short-4096-sliding-cp8r7 | tilelang@cute | 790.7 | 786.1-807.7 | 1.00 | 2076 | 2050-2085 | 1.00 |
| short-4096-sliding-cp8r7 | cute@cute | 208.6 | 207.7-216.3 | 0.26 | 1428 | 1408-1438 | 0.69 |
| short-4096-sliding-cp8r7 | cute_ws@cute | 90.0 | 89.1-92.1 | 0.11 | 1269 | 1256-1278 | 0.61 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 206.8 | 205.1-215.9 | 0.26 | - | - | - |
| heavy-4096-csa-cp1 | tilelang@main | 1462 | 1446-1472 | 1.00 | 3545 | 3535-3555 | 1.00 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 524.9 | 521.9-530.8 | 0.36 | - | - | - |
| heavy-4096-csa-cp1 | tilelang@cute | 1467 | 1449-1477 | 1.00 | 3553 | 3543-3572 | 1.00 |
| heavy-4096-csa-cp1 | cute@cute | 840.2 | 836.4-845.4 | 0.57 | 2904 | 2898-2907 | 0.82 |
| heavy-4096-csa-cp1 | cute_ws@cute | 407.4 | 406.8-408.4 | 0.28 | 2691 | 2683-2718 | 0.76 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@cute | 530.8 | 528.3-535.0 | 0.36 | - | - | - |
| heavy-4096-csa-cp8r0 | tilelang@main | 794.4 | 789.7-815.2 | 1.00 | 2033 | 2023-2050 | 1.00 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 208.3 | 203.7-215.7 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r0 | tilelang@cute | 808.2 | 799.5-817.4 | 1.00 | 2107 | 2096-2109 | 1.00 |
| heavy-4096-csa-cp8r0 | cute@cute | 229.9 | 229.3-230.9 | 0.28 | 1455 | 1453-1461 | 0.69 |
| heavy-4096-csa-cp8r0 | cute_ws@cute | 109.1 | 108.2-109.6 | 0.13 | 1280 | 1276-1289 | 0.61 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 219.8 | 215.7-223.8 | 0.27 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 817.9 | 805.9-831.3 | 1.00 | 2049 | 2038-2194 | 1.00 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 217.3 | 211.7-218.4 | 0.27 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@cute | 808.3 | 800.5-815.4 | 1.00 | 2132 | 2108-2176 | 1.00 |
| heavy-4096-csa-cp8r4 | cute@cute | 230.3 | 226.1-231.4 | 0.28 | 1454 | 1433-1475 | 0.68 |
| heavy-4096-csa-cp8r4 | cute_ws@cute | 111.3 | 110.2-111.9 | 0.14 | 1276 | 1270-1290 | 0.60 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 223.3 | 215.3-224.7 | 0.28 | - | - | - |
| heavy-4096-csa-cp8r7 | tilelang@main | 849.9 | 811.7-856.0 | 1.00 | 2089 | 2057-2341 | 1.00 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 215.8 | 214.5-218.0 | 0.25 | - | - | - |
| heavy-4096-csa-cp8r7 | tilelang@cute | 810.5 | 801.3-821.0 | 1.00 | 2124 | 2116-2146 | 1.00 |
| heavy-4096-csa-cp8r7 | cute@cute | 225.6 | 222.3-228.5 | 0.28 | 1444 | 1442-1454 | 0.68 |
| heavy-4096-csa-cp8r7 | cute_ws@cute | 109.9 | 108.3-111.3 | 0.14 | 1287 | 1275-1319 | 0.61 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 219.7 | 214.5-221.6 | 0.27 | - | - | - |
| heavy-4096-hca-cp1 | tilelang@main | 1375 | 1372-1386 | 1.00 | 2982 | 2967-2988 | 1.00 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 482.7 | 480.6-489.0 | 0.35 | - | - | - |
| heavy-4096-hca-cp1 | tilelang@cute | 1383 | 1371-1418 | 1.00 | 3019 | 3007-3023 | 1.00 |
| heavy-4096-hca-cp1 | cute@cute | 746.9 | 744.8-753.9 | 0.54 | 2373 | 2366-2384 | 0.79 |
| heavy-4096-hca-cp1 | cute_ws@cute | 341.1 | 340.1-343.6 | 0.25 | 2178 | 2174-2183 | 0.72 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@cute | 491.6 | 486.7-495.9 | 0.36 | - | - | - |
| heavy-4096-hca-cp8r0 | tilelang@main | 835.5 | 828.0-844.7 | 1.00 | 2153 | 2137-2162 | 1.00 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 214.5 | 211.6-218.6 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r0 | tilelang@cute | 832.8 | 817.8-853.3 | 1.00 | 2208 | 2197-2218 | 1.00 |
| heavy-4096-hca-cp8r0 | cute@cute | 245.9 | 242.2-250.9 | 0.30 | 1553 | 1540-1562 | 0.70 |
| heavy-4096-hca-cp8r0 | cute_ws@cute | 100.4 | 100.0-101.2 | 0.12 | 1334 | 1330-1346 | 0.60 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 222.2 | 217.3-224.6 | 0.27 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 827.9 | 820.9-840.8 | 1.00 | 2160 | 2157-2177 | 1.00 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 207.4 | 205.8-212.8 | 0.25 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@cute | 832.7 | 825.5-834.9 | 1.00 | 2218 | 2211-2248 | 1.00 |
| heavy-4096-hca-cp8r4 | cute@cute | 241.6 | 238.2-245.4 | 0.29 | 1568 | 1551-1585 | 0.71 |
| heavy-4096-hca-cp8r4 | cute_ws@cute | 98.7 | 97.6-98.8 | 0.12 | 1356 | 1350-1379 | 0.61 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 212.8 | 211.6-221.3 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r7 | tilelang@main | 838.2 | 821.0-847.4 | 1.00 | 2180 | 2177-2181 | 1.00 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 215.7 | 212.8-219.9 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r7 | tilelang@cute | 844.7 | 832.7-856.0 | 1.00 | 2230 | 2201-2251 | 1.00 |
| heavy-4096-hca-cp8r7 | cute@cute | 240.3 | 239.2-250.1 | 0.28 | 1571 | 1540-1581 | 0.70 |
| heavy-4096-hca-cp8r7 | cute_ws@cute | 96.3 | 94.4-97.4 | 0.11 | 1346 | 1326-1361 | 0.60 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 218.4 | 213.2-224.0 | 0.26 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang@main | 1283 | 1276-1286 | 1.00 | 2911 | 2843-2955 | 1.00 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 418.8 | 418.5-423.0 | 0.33 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang@cute | 1311 | 1296-1341 | 1.00 | 2819 | 2815-2840 | 1.00 |
| heavy-4096-sliding-cp1 | cute@cute | 671.8 | 664.6-673.1 | 0.51 | 2162 | 2160-2166 | 0.77 |
| heavy-4096-sliding-cp1 | cute_ws@cute | 273.9 | 272.0-276.1 | 0.21 | 2014 | 2014-2028 | 0.71 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@cute | 430.6 | 422.9-435.5 | 0.33 | - | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@main | 793.9 | 787.1-802.0 | 1.00 | 2083 | 2065-2088 | 1.00 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 198.7 | 195.6-205.1 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@cute | 788.0 | 777.1-799.4 | 1.00 | 2105 | 2102-2110 | 1.00 |
| heavy-4096-sliding-cp8r0 | cute@cute | 207.0 | 206.4-212.8 | 0.26 | 1449 | 1445-1459 | 0.69 |
| heavy-4096-sliding-cp8r0 | cute_ws@cute | 88.9 | 88.3-89.7 | 0.11 | 1302 | 1292-1306 | 0.62 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 210.1 | 205.0-212.6 | 0.27 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 780.6 | 767.1-784.0 | 1.00 | 2050 | 2037-2053 | 1.00 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 194.3 | 190.0-200.3 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@cute | 788.5 | 784.4-797.6 | 1.00 | 2114 | 2102-2188 | 1.00 |
| heavy-4096-sliding-cp8r4 | cute@cute | 207.9 | 205.0-210.0 | 0.26 | 1438 | 1427-1447 | 0.68 |
| heavy-4096-sliding-cp8r4 | cute_ws@cute | 89.8 | 88.1-90.9 | 0.11 | 1287 | 1271-1299 | 0.61 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 207.8 | 202.5-208.5 | 0.26 | - | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@main | 772.0 | 762.3-785.7 | 1.00 | 2099 | 2076-2132 | 1.00 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 197.5 | 194.8-199.0 | 0.26 | - | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@cute | 797.3 | 779.9-810.4 | 1.00 | 2099 | 2075-2111 | 1.00 |
| heavy-4096-sliding-cp8r7 | cute@cute | 207.6 | 204.5-208.6 | 0.26 | 1452 | 1440-1459 | 0.69 |
| heavy-4096-sliding-cp8r7 | cute_ws@cute | 89.6 | 88.6-90.9 | 0.11 | 1290 | 1280-1305 | 0.61 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 202.6 | 199.2-207.3 | 0.25 | - | - | - |
| tiny-4096-csa-cp1 | tilelang@main | 1374 | 1368-1389 | 1.00 | 2919 | 2904-2930 | 1.00 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 501.5 | 498.9-506.4 | 0.37 | - | - | - |
| tiny-4096-csa-cp1 | tilelang@cute | 1375 | 1356-1416 | 1.00 | 2925 | 2921-2944 | 1.00 |
| tiny-4096-csa-cp1 | cute@cute | 750.9 | 748.8-754.8 | 0.55 | 2300 | 2287-2314 | 0.79 |
| tiny-4096-csa-cp1 | cute_ws@cute | 382.2 | 380.9-384.8 | 0.28 | 2142 | 2139-2147 | 0.73 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@cute | 499.0 | 498.7-505.0 | 0.36 | - | - | - |
| tiny-4096-csa-cp8r0 | tilelang@main | 796.5 | 784.8-820.9 | 1.00 | 2020 | 2015-2135 | 1.00 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 210.9 | 210.1-221.9 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r0 | tilelang@cute | 801.5 | 790.9-812.3 | 1.00 | 2134 | 2096-2226 | 1.00 |
| tiny-4096-csa-cp8r0 | cute@cute | 217.5 | 217.1-223.4 | 0.27 | 1456 | 1449-1460 | 0.68 |
| tiny-4096-csa-cp8r0 | cute_ws@cute | 108.8 | 107.7-110.4 | 0.14 | 1284 | 1280-1293 | 0.60 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 213.5 | 212.4-220.5 | 0.27 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 809.3 | 793.1-830.6 | 1.00 | 2034 | 2014-2048 | 1.00 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 209.6 | 207.2-216.1 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@cute | 797.8 | 789.4-813.4 | 1.00 | 2097 | 2087-2108 | 1.00 |
| tiny-4096-csa-cp8r4 | cute@cute | 222.5 | 220.4-224.2 | 0.28 | 1451 | 1449-1457 | 0.69 |
| tiny-4096-csa-cp8r4 | cute_ws@cute | 108.7 | 108.2-110.1 | 0.14 | 1293 | 1279-1305 | 0.62 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 217.0 | 214.4-221.3 | 0.27 | - | - | - |
| tiny-4096-csa-cp8r7 | tilelang@main | 786.0 | 777.7-790.5 | 1.00 | 2043 | 2030-2080 | 1.00 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 210.6 | 206.3-211.8 | 0.27 | - | - | - |
| tiny-4096-csa-cp8r7 | tilelang@cute | 811.5 | 784.5-813.9 | 1.00 | 2105 | 2092-2132 | 1.00 |
| tiny-4096-csa-cp8r7 | cute@cute | 223.6 | 214.6-225.6 | 0.28 | 1465 | 1437-1539 | 0.70 |
| tiny-4096-csa-cp8r7 | cute_ws@cute | 108.1 | 106.6-109.5 | 0.13 | 1320 | 1300-1567 | 0.63 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 215.9 | 211.0-218.4 | 0.27 | - | - | - |
| tiny-4096-hca-cp1 | tilelang@main | 1227 | 1218-1246 | 1.00 | 2409 | 2400-2413 | 1.00 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 412.3 | 408.0-430.9 | 0.34 | - | - | - |
| tiny-4096-hca-cp1 | tilelang@cute | 1234 | 1229-1245 | 1.00 | 2464 | 2457-2471 | 1.00 |
| tiny-4096-hca-cp1 | cute@cute | 613.1 | 612.6-623.9 | 0.50 | 1813 | 1809-1826 | 0.74 |
| tiny-4096-hca-cp1 | cute_ws@cute | 289.4 | 272.9-292.0 | 0.23 | 1675 | 1663-1684 | 0.68 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@cute | 419.9 | 418.4-428.1 | 0.34 | - | - | - |
| tiny-4096-hca-cp8r0 | tilelang@main | 770.4 | 766.3-773.0 | 1.00 | 2033 | 2020-2052 | 1.00 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 194.8 | 192.6-198.4 | 0.25 | - | - | - |
| tiny-4096-hca-cp8r0 | tilelang@cute | 788.2 | 781.1-793.9 | 1.00 | 2109 | 2106-2119 | 1.00 |
| tiny-4096-hca-cp8r0 | cute@cute | 202.1 | 200.0-205.2 | 0.26 | 1447 | 1438-1460 | 0.69 |
| tiny-4096-hca-cp8r0 | cute_ws@cute | 89.3 | 88.1-91.8 | 0.11 | 1282 | 1281-1301 | 0.61 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 205.7 | 202.7-210.8 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 767.4 | 763.6-775.5 | 1.00 | 2023 | 2018-2038 | 1.00 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 196.6 | 192.9-202.8 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@cute | 789.6 | 773.4-791.0 | 1.00 | 2137 | 2130-2190 | 1.00 |
| tiny-4096-hca-cp8r4 | cute@cute | 198.3 | 195.6-199.9 | 0.25 | 1474 | 1467-1488 | 0.69 |
| tiny-4096-hca-cp8r4 | cute_ws@cute | 87.1 | 85.5-87.8 | 0.11 | 1314 | 1293-1352 | 0.61 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 202.4 | 200.7-211.1 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r7 | tilelang@main | 784.1 | 768.8-793.5 | 1.00 | 2053 | 2045-2065 | 1.00 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 201.5 | 196.4-202.1 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r7 | tilelang@cute | 773.2 | 764.0-786.6 | 1.00 | 2099 | 2091-2101 | 1.00 |
| tiny-4096-hca-cp8r7 | cute@cute | 201.2 | 198.4-202.7 | 0.26 | 1453 | 1450-1457 | 0.69 |
| tiny-4096-hca-cp8r7 | cute_ws@cute | 87.8 | 87.4-89.3 | 0.11 | 1298 | 1288-1332 | 0.62 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 206.8 | 199.4-209.7 | 0.27 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang@main | 1226 | 1215-1234 | 1.00 | 2407 | 2386-2415 | 1.00 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 408.8 | 408.3-411.3 | 0.33 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang@cute | 1228 | 1223-1231 | 1.00 | 2467 | 2455-2491 | 1.00 |
| tiny-4096-sliding-cp1 | cute@cute | 618.1 | 615.5-620.4 | 0.50 | 1827 | 1819-1836 | 0.74 |
| tiny-4096-sliding-cp1 | cute_ws@cute | 286.0 | 274.6-289.6 | 0.23 | 1672 | 1668-1674 | 0.68 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@cute | 427.5 | 422.1-433.7 | 0.35 | - | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@main | 775.3 | 771.9-783.8 | 1.00 | 2030 | 2005-2076 | 1.00 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 200.7 | 197.0-203.9 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@cute | 787.8 | 782.5-807.2 | 1.00 | 2111 | 2094-2128 | 1.00 |
| tiny-4096-sliding-cp8r0 | cute@cute | 201.3 | 199.6-206.9 | 0.26 | 1452 | 1443-1460 | 0.69 |
| tiny-4096-sliding-cp8r0 | cute_ws@cute | 90.3 | 89.5-93.2 | 0.11 | 1296 | 1280-1303 | 0.61 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 206.5 | 203.7-213.0 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 781.6 | 770.1-787.6 | 1.00 | 2040 | 2028-2056 | 1.00 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 201.3-214.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@cute | 798.0 | 774.2-809.5 | 1.00 | 2121 | 2111-2137 | 1.00 |
| tiny-4096-sliding-cp8r4 | cute@cute | 204.5 | 201.8-207.7 | 0.26 | 1468 | 1442-1488 | 0.69 |
| tiny-4096-sliding-cp8r4 | cute_ws@cute | 88.6 | 87.3-90.3 | 0.11 | 1277 | 1272-1283 | 0.60 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 206.3 | 199.0-213.0 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@main | 786.4 | 772.9-790.8 | 1.00 | 2081 | 2076-2129 | 1.00 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 202.2 | 194.2-206.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@cute | 775.6 | 774.2-782.1 | 1.00 | 2111 | 2071-2117 | 1.00 |
| tiny-4096-sliding-cp8r7 | cute@cute | 201.2 | 200.7-205.6 | 0.26 | 1454 | 1424-1464 | 0.69 |
| tiny-4096-sliding-cp8r7 | cute_ws@cute | 91.6 | 89.2-92.7 | 0.12 | 1310 | 1278-1313 | 0.62 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 211.1 | 201.7-215.0 | 0.27 | - | - | - |
| single-16384-csa-cp1 | tilelang@main | 5918 | 5886-6013 | 1.00 | 23560 | 23556-23572 | 1.00 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 2570 | 2560-2654 | 0.43 | - | - | - |
| single-16384-csa-cp1 | tilelang@cute | 5884 | 5866-5940 | 1.00 | 23583 | 23576-23593 | 1.00 |
| single-16384-csa-cp1 | cute@cute | 5105 | 5070-5378 | 0.87 | 22716 | 22713-22731 | 0.96 |
| single-16384-csa-cp1 | cute_ws@cute | 2443 | 2426-2658 | 0.42 | 20079 | 20074-20083 | 0.85 |
| single-16384-csa-cp1 | flashmla_fwd_ref@cute | 2620 | 2581-2758 | 0.45 | - | - | - |
| single-16384-csa-cp8r0 | tilelang@main | 1232 | 1216-1235 | 1.00 | 3243 | 3230-3250 | 1.00 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 417.6 | 414.9-421.8 | 0.34 | - | - | - |
| single-16384-csa-cp8r0 | tilelang@cute | 1229 | 1221-1251 | 1.00 | 3289 | 3273-3310 | 1.00 |
| single-16384-csa-cp8r0 | cute@cute | 633.2 | 632.0-633.6 | 0.52 | 2635 | 2625-2660 | 0.80 |
| single-16384-csa-cp8r0 | cute_ws@cute | 307.7 | 305.9-309.0 | 0.25 | 2487 | 2481-2506 | 0.76 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 427.7 | 425.3-431.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r4 | tilelang@main | 1409 | 1392-1436 | 1.00 | 4079 | 4072-4087 | 1.00 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 495.2 | 492.1-501.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r4 | tilelang@cute | 1399 | 1386-1402 | 1.00 | 4145 | 4128-4159 | 1.00 |
| single-16384-csa-cp8r4 | cute@cute | 798.2 | 797.0-805.1 | 0.57 | 3490 | 3481-3549 | 0.84 |
| single-16384-csa-cp8r4 | cute_ws@cute | 382.6 | 380.3-386.1 | 0.27 | 3350 | 3336-3359 | 0.81 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 497.8 | 496.4-498.5 | 0.36 | - | - | - |
| single-16384-csa-cp8r7 | tilelang@main | 1405 | 1396-1424 | 1.00 | 4104 | 4096-4125 | 1.00 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 496.7 | 493.8-498.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r7 | tilelang@cute | 1407 | 1401-1417 | 1.00 | 4105 | 4084-4144 | 1.00 |
| single-16384-csa-cp8r7 | cute@cute | 795.6 | 792.9-799.3 | 0.57 | 3460 | 3453-3471 | 0.84 |
| single-16384-csa-cp8r7 | cute_ws@cute | 382.4 | 381.8-383.2 | 0.27 | 3321 | 3309-3323 | 0.81 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 497.6 | 493.7-500.6 | 0.35 | - | - | - |
| single-16384-hca-cp1 | tilelang@main | 3573 | 3561-3617 | 1.00 | 10905 | 10899-10911 | 1.00 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1490 | 1486-1513 | 0.42 | - | - | - |
| single-16384-hca-cp1 | tilelang@cute | 3540 | 3524-3567 | 1.00 | 10867 | 10844-10890 | 1.00 |
| single-16384-hca-cp1 | cute@cute | 2792 | 2784-2801 | 0.79 | 10051 | 10048-10084 | 0.92 |
| single-16384-hca-cp1 | cute_ws@cute | 1262 | 1254-1265 | 0.36 | 8423 | 8416-8436 | 0.78 |
| single-16384-hca-cp1 | flashmla_fwd_ref@cute | 1490 | 1488-1495 | 0.42 | - | - | - |
| single-16384-hca-cp8r0 | tilelang@main | 1081 | 1067-1109 | 1.00 | 2422 | 2395-2446 | 1.00 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 340.5 | 338.5-348.6 | 0.32 | - | - | - |
| single-16384-hca-cp8r0 | tilelang@cute | 1112 | 1073-1156 | 1.00 | 2468 | 2455-2477 | 1.00 |
| single-16384-hca-cp8r0 | cute@cute | 467.7 | 462.5-470.2 | 0.42 | 1836 | 1832-1849 | 0.74 |
| single-16384-hca-cp8r0 | cute_ws@cute | 224.4 | 223.1-224.5 | 0.20 | 1685 | 1679-1697 | 0.68 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 352.7 | 348.1-359.4 | 0.32 | - | - | - |
| single-16384-hca-cp8r4 | tilelang@main | 1097 | 1094-1101 | 1.00 | 2665 | 2641-2689 | 1.00 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 339.9 | 338.0-344.9 | 0.31 | - | - | - |
| single-16384-hca-cp8r4 | tilelang@cute | 1096 | 1083-1104 | 1.00 | 2721 | 2694-2779 | 1.00 |
| single-16384-hca-cp8r4 | cute@cute | 506.7 | 505.7-511.5 | 0.46 | 2106 | 2071-2128 | 0.77 |
| single-16384-hca-cp8r4 | cute_ws@cute | 221.4 | 220.7-223.0 | 0.20 | 1935 | 1921-1960 | 0.71 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 341.4 | 337.1-344.7 | 0.31 | - | - | - |
| single-16384-hca-cp8r7 | tilelang@main | 1106 | 1097-1108 | 1.00 | 2819 | 2816-2866 | 1.00 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 343.8 | 338.7-344.0 | 0.31 | - | - | - |
| single-16384-hca-cp8r7 | tilelang@cute | 1106 | 1102-1114 | 1.00 | 2830 | 2812-2848 | 1.00 |
| single-16384-hca-cp8r7 | cute@cute | 516.7 | 515.0-521.1 | 0.47 | 2191 | 2172-2219 | 0.77 |
| single-16384-hca-cp8r7 | cute_ws@cute | 226.6 | 223.4-228.8 | 0.20 | 2056 | 2032-2121 | 0.73 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 350.1 | 345.5-357.9 | 0.32 | - | - | - |
| single-16384-sliding-cp1 | tilelang@main | 2973 | 2970-2979 | 1.00 | 8286 | 8275-8288 | 1.00 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 1164 | 1159-1174 | 0.39 | - | - | - |
| single-16384-sliding-cp1 | tilelang@cute | 2979 | 2969-2997 | 1.00 | 8295 | 8273-8340 | 1.00 |
| single-16384-sliding-cp1 | cute@cute | 2224 | 2223-2231 | 0.75 | 7503 | 7491-7522 | 0.90 |
| single-16384-sliding-cp1 | cute_ws@cute | 853.8 | 850.6-856.0 | 0.29 | 6080 | 6074-6097 | 0.73 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1170 | 1168-1172 | 0.39 | - | - | - |
| single-16384-sliding-cp8r0 | tilelang@main | 1019 | 1002-1035 | 1.00 | 2333 | 2317-2348 | 1.00 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 299.8 | 294.8-317.2 | 0.29 | - | - | - |
| single-16384-sliding-cp8r0 | tilelang@cute | 1014 | 1007-1037 | 1.00 | 2373 | 2370-2392 | 1.00 |
| single-16384-sliding-cp8r0 | cute@cute | 414.7 | 411.8-419.6 | 0.41 | 1733 | 1727-1743 | 0.73 |
| single-16384-sliding-cp8r0 | cute_ws@cute | 171.0 | 167.7-173.6 | 0.17 | 1586 | 1581-1599 | 0.67 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 299.6 | 296.3-304.0 | 0.30 | - | - | - |
| single-16384-sliding-cp8r4 | tilelang@main | 1004 | 986.3-1009 | 1.00 | 2390 | 2382-2476 | 1.00 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 296.7 | 289.8-313.0 | 0.30 | - | - | - |
| single-16384-sliding-cp8r4 | tilelang@cute | 1012 | 1008-1014 | 1.00 | 2536 | 2517-2552 | 1.00 |
| single-16384-sliding-cp8r4 | cute@cute | 418.4 | 417.8-423.0 | 0.41 | 1933 | 1924-1941 | 0.76 |
| single-16384-sliding-cp8r4 | cute_ws@cute | 170.1 | 169.8-171.4 | 0.17 | 1806 | 1668-1812 | 0.71 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 301.3 | 297.6-304.3 | 0.30 | - | - | - |
| single-16384-sliding-cp8r7 | tilelang@main | 996.2 | 987.6-1004 | 1.00 | 2389 | 2385-2438 | 1.00 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 294.0 | 292.2-302.3 | 0.30 | - | - | - |
| single-16384-sliding-cp8r7 | tilelang@cute | 1008 | 1004-1024 | 1.00 | 2532 | 2524-2544 | 1.00 |
| single-16384-sliding-cp8r7 | cute@cute | 419.6 | 418.7-422.5 | 0.42 | 1932 | 1927-1936 | 0.76 |
| single-16384-sliding-cp8r7 | cute_ws@cute | 171.5 | 170.9-173.4 | 0.17 | 1793 | 1790-1803 | 0.71 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 301.2 | 300.0-304.5 | 0.30 | - | - | - |
| short-16384-csa-cp1 | tilelang@main | 4509 | 4494-4544 | 1.00 | 15976 | 15975-16005 | 1.00 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 1957 | 1952-1959 | 0.43 | - | - | - |
| short-16384-csa-cp1 | tilelang@cute | 4504 | 4493-4541 | 1.00 | 15914 | 15888-15916 | 1.00 |
| short-16384-csa-cp1 | cute@cute | 3706 | 3699-3872 | 0.82 | 15078 | 15076-15088 | 0.95 |
| short-16384-csa-cp1 | cute_ws@cute | 1784 | 1782-1855 | 0.40 | 13124 | 13123-13138 | 0.82 |
| short-16384-csa-cp1 | flashmla_fwd_ref@cute | 1979 | 1964-1985 | 0.44 | - | - | - |
| short-16384-csa-cp8r0 | tilelang@main | 1175 | 1156-1200 | 1.00 | 2997 | 2984-3003 | 1.00 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 390.1 | 384.5-394.8 | 0.33 | - | - | - |
| short-16384-csa-cp8r0 | tilelang@cute | 1176 | 1163-1179 | 1.00 | 3028 | 3000-3069 | 1.00 |
| short-16384-csa-cp8r0 | cute@cute | 573.8 | 569.6-576.1 | 0.49 | 2368 | 2367-2376 | 0.78 |
| short-16384-csa-cp8r0 | cute_ws@cute | 278.4 | 276.2-279.6 | 0.24 | 2222 | 2215-2252 | 0.73 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 389.4 | 389.2-394.5 | 0.33 | - | - | - |
| short-16384-csa-cp8r4 | tilelang@main | 1249 | 1236-1261 | 1.00 | 3387 | 3384-3406 | 1.00 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 431.6 | 429.3-432.5 | 0.35 | - | - | - |
| short-16384-csa-cp8r4 | tilelang@cute | 1248 | 1246-1254 | 1.00 | 3471 | 3454-3495 | 1.00 |
| short-16384-csa-cp8r4 | cute@cute | 656.3 | 655.4-659.1 | 0.53 | 2815 | 2805-2818 | 0.81 |
| short-16384-csa-cp8r4 | cute_ws@cute | 319.0 | 317.2-320.1 | 0.26 | 2655 | 2651-2664 | 0.76 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 434.2 | 433.6-436.2 | 0.35 | - | - | - |
| short-16384-csa-cp8r7 | tilelang@main | 1211 | 1195-1219 | 1.00 | 3241 | 3236-3255 | 1.00 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 408.0 | 406.4-415.0 | 0.34 | - | - | - |
| short-16384-csa-cp8r7 | tilelang@cute | 1230 | 1219-1247 | 1.00 | 3265 | 3248-3283 | 1.00 |
| short-16384-csa-cp8r7 | cute@cute | 623.6 | 621.4-624.3 | 0.51 | 2614 | 2609-2618 | 0.80 |
| short-16384-csa-cp8r7 | cute_ws@cute | 308.7 | 307.2-309.5 | 0.25 | 2463 | 2456-2501 | 0.75 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 420.9 | 419.8-424.3 | 0.34 | - | - | - |
| short-16384-hca-cp1 | tilelang@main | 3337 | 3333-3349 | 1.00 | 9225 | 9192-9255 | 1.00 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 1463 | 1461-1465 | 0.44 | - | - | - |
| short-16384-hca-cp1 | tilelang@cute | 3327 | 3306-3336 | 1.00 | 9228 | 9206-9238 | 1.00 |
| short-16384-hca-cp1 | cute@cute | 2558 | 2551-2578 | 0.77 | 8372 | 8369-8397 | 0.91 |
| short-16384-hca-cp1 | cute_ws@cute | 1242 | 1238-1246 | 0.37 | 6989 | 6986-7002 | 0.76 |
| short-16384-hca-cp1 | flashmla_fwd_ref@cute | 1462 | 1461-1468 | 0.44 | - | - | - |
| short-16384-hca-cp8r0 | tilelang@main | 1070 | 1063-1086 | 1.00 | 2499 | 2480-2505 | 1.00 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 338.9 | 332.9-342.3 | 0.32 | - | - | - |
| short-16384-hca-cp8r0 | tilelang@cute | 1078 | 1077-1089 | 1.00 | 2535 | 2530-2571 | 1.00 |
| short-16384-hca-cp8r0 | cute@cute | 475.3 | 472.9-476.3 | 0.44 | 1904 | 1896-1932 | 0.75 |
| short-16384-hca-cp8r0 | cute_ws@cute | 220.2 | 218.2-220.7 | 0.20 | 1709 | 1700-1717 | 0.67 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 342.2 | 340.7-345.8 | 0.32 | - | - | - |
| short-16384-hca-cp8r4 | tilelang@main | 1080 | 1076-1091 | 1.00 | 2532 | 2525-2555 | 1.00 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 337.4 | 334.8-343.4 | 0.31 | - | - | - |
| short-16384-hca-cp8r4 | tilelang@cute | 1085 | 1076-1094 | 1.00 | 2608 | 2593-2611 | 1.00 |
| short-16384-hca-cp8r4 | cute@cute | 486.4 | 485.6-489.4 | 0.45 | 1959 | 1950-1964 | 0.75 |
| short-16384-hca-cp8r4 | cute_ws@cute | 222.4 | 221.3-222.8 | 0.20 | 1763 | 1754-1769 | 0.68 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 349.0 | 346.0-350.7 | 0.32 | - | - | - |
| short-16384-hca-cp8r7 | tilelang@main | 1081 | 1069-1086 | 1.00 | 2558 | 2548-2561 | 1.00 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 338.8 | 338.8-342.2 | 0.31 | - | - | - |
| short-16384-hca-cp8r7 | tilelang@cute | 1082 | 1079-1094 | 1.00 | 2571 | 2564-2578 | 1.00 |
| short-16384-hca-cp8r7 | cute@cute | 476.9 | 475.5-479.6 | 0.44 | 1934 | 1916-1943 | 0.75 |
| short-16384-hca-cp8r7 | cute_ws@cute | 218.3 | 216.9-219.2 | 0.20 | 1736 | 1733-1745 | 0.68 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 346.0 | 342.8-347.5 | 0.32 | - | - | - |
| short-16384-sliding-cp1 | tilelang@main | 2985 | 2970-3000 | 1.00 | 8155 | 8151-8192 | 1.00 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 1169 | 1162-1175 | 0.39 | - | - | - |
| short-16384-sliding-cp1 | tilelang@cute | 2959 | 2949-2960 | 1.00 | 8156 | 8149-8174 | 1.00 |
| short-16384-sliding-cp1 | cute@cute | 2215 | 2212-2222 | 0.75 | 7357 | 7349-7365 | 0.90 |
| short-16384-sliding-cp1 | cute_ws@cute | 854.0 | 853.4-859.5 | 0.29 | 5960 | 5948-5980 | 0.73 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1173 | 1172-1176 | 0.40 | - | - | - |
| short-16384-sliding-cp8r0 | tilelang@main | 1001 | 987.1-1006 | 1.00 | 2305 | 2292-2331 | 1.00 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 296.3 | 293.4-300.4 | 0.30 | - | - | - |
| short-16384-sliding-cp8r0 | tilelang@cute | 1018 | 1009-1026 | 1.00 | 2387 | 2385-2397 | 1.00 |
| short-16384-sliding-cp8r0 | cute@cute | 405.6 | 404.1-413.5 | 0.40 | 1774 | 1733-1797 | 0.74 |
| short-16384-sliding-cp8r0 | cute_ws@cute | 170.0 | 168.8-173.2 | 0.17 | 1576 | 1565-1588 | 0.66 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 303.8 | 300.9-306.4 | 0.30 | - | - | - |
| short-16384-sliding-cp8r4 | tilelang@main | 993.9 | 985.0-1070 | 1.00 | 2370 | 2342-2396 | 1.00 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 294.4 | 287.7-309.9 | 0.30 | - | - | - |
| short-16384-sliding-cp8r4 | tilelang@cute | 1078 | 1023-1126 | 1.00 | 2410 | 2401-2422 | 1.00 |
| short-16384-sliding-cp8r4 | cute@cute | 422.2 | 418.5-455.9 | 0.39 | 1822 | 1759-1882 | 0.76 |
| short-16384-sliding-cp8r4 | cute_ws@cute | 172.5 | 171.9-180.5 | 0.16 | 1614 | 1606-1616 | 0.67 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 314.2 | 304.7-318.8 | 0.29 | - | - | - |
| short-16384-sliding-cp8r7 | tilelang@main | 1001 | 987.3-1004 | 1.00 | 2361 | 2358-2371 | 1.00 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 294.5 | 290.2-294.9 | 0.29 | - | - | - |
| short-16384-sliding-cp8r7 | tilelang@cute | 1039 | 1025-1057 | 1.00 | 2389 | 2374-2398 | 1.00 |
| short-16384-sliding-cp8r7 | cute@cute | 419.1 | 418.4-421.4 | 0.40 | 1756 | 1751-1766 | 0.74 |
| short-16384-sliding-cp8r7 | cute_ws@cute | 173.0 | 170.5-173.8 | 0.17 | 1597 | 1589-1605 | 0.67 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 303.8 | 299.3-312.5 | 0.29 | - | - | - |
| heavy-16384-csa-cp1 | tilelang@main | 3969 | 3954-4002 | 1.00 | 12997 | 12956-13003 | 1.00 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1726 | 1725-1731 | 0.43 | - | - | - |
| heavy-16384-csa-cp1 | tilelang@cute | 3960 | 3947-3977 | 1.00 | 12928 | 12920-12934 | 1.00 |
| heavy-16384-csa-cp1 | cute@cute | 3192 | 3180-3201 | 0.81 | 12110 | 12099-12116 | 0.94 |
| heavy-16384-csa-cp1 | cute_ws@cute | 1558 | 1556-1560 | 0.39 | 10426 | 10419-10436 | 0.81 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@cute | 1736 | 1734-1736 | 0.44 | - | - | - |
| heavy-16384-csa-cp8r0 | tilelang@main | 1092 | 1069-1121 | 1.00 | 2575 | 2548-2608 | 1.00 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 346.5 | 345.0-355.2 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r0 | tilelang@cute | 1097 | 1084-1102 | 1.00 | 2610 | 2604-2621 | 1.00 |
| heavy-16384-csa-cp8r0 | cute@cute | 490.1 | 488.8-495.0 | 0.45 | 1973 | 1966-1981 | 0.76 |
| heavy-16384-csa-cp8r0 | cute_ws@cute | 240.2 | 239.8-240.3 | 0.22 | 1821 | 1814-1826 | 0.70 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 354.8 | 350.9-363.1 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r4 | tilelang@main | 1117 | 1112-1126 | 1.00 | 2762 | 2761-2778 | 1.00 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 362.7 | 357.6-370.0 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r4 | tilelang@cute | 1131 | 1114-1138 | 1.00 | 2840 | 2825-2863 | 1.00 |
| heavy-16384-csa-cp8r4 | cute@cute | 531.0 | 528.7-531.9 | 0.47 | 2183 | 2181-2192 | 0.77 |
| heavy-16384-csa-cp8r4 | cute_ws@cute | 256.1 | 254.2-256.9 | 0.23 | 2037 | 2028-2046 | 0.72 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 374.8 | 372.6-381.5 | 0.33 | - | - | - |
| heavy-16384-csa-cp8r7 | tilelang@main | 1264 | 1261-1266 | 1.00 | 3482 | 3467-3491 | 1.00 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 428.5 | 427.4-434.8 | 0.34 | - | - | - |
| heavy-16384-csa-cp8r7 | tilelang@cute | 1291 | 1277-1301 | 1.00 | 3504 | 3488-3522 | 1.00 |
| heavy-16384-csa-cp8r7 | cute@cute | 672.2 | 668.1-675.1 | 0.52 | 2856 | 2845-2861 | 0.81 |
| heavy-16384-csa-cp8r7 | cute_ws@cute | 323.1 | 322.7-326.9 | 0.25 | 2696 | 2682-2710 | 0.77 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 443.6 | 439.6-444.3 | 0.34 | - | - | - |
| heavy-16384-hca-cp1 | tilelang@main | 3252 | 3222-3290 | 1.00 | 8705 | 8691-8740 | 1.00 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 1404 | 1396-1410 | 0.43 | - | - | - |
| heavy-16384-hca-cp1 | tilelang@cute | 3223 | 3206-3238 | 1.00 | 8694 | 8667-8723 | 1.00 |
| heavy-16384-hca-cp1 | cute@cute | 2455 | 2447-2458 | 0.76 | 7853 | 7848-7857 | 0.90 |
| heavy-16384-hca-cp1 | cute_ws@cute | 1163 | 1160-1169 | 0.36 | 6501 | 6498-6507 | 0.75 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@cute | 1407 | 1404-1410 | 0.44 | - | - | - |
| heavy-16384-hca-cp8r0 | tilelang@main | 1068 | 1060-1096 | 1.00 | 2448 | 2445-2477 | 1.00 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 335.5 | 331.0-346.8 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r0 | tilelang@cute | 1072 | 1059-1097 | 1.00 | 2463 | 2456-2478 | 1.00 |
| heavy-16384-hca-cp8r0 | cute@cute | 458.8 | 457.0-461.1 | 0.43 | 1827 | 1817-1841 | 0.74 |
| heavy-16384-hca-cp8r0 | cute_ws@cute | 205.2 | 204.4-207.8 | 0.19 | 1630 | 1612-1638 | 0.66 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 333.6 | 331.6-336.4 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r4 | tilelang@main | 1070 | 1066-1081 | 1.00 | 2488 | 2479-2500 | 1.00 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 341.7 | 338.1-344.5 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r4 | tilelang@cute | 1070 | 1069-1081 | 1.00 | 2529 | 2514-2535 | 1.00 |
| heavy-16384-hca-cp8r4 | cute@cute | 469.6 | 468.7-471.9 | 0.44 | 1870 | 1865-1885 | 0.74 |
| heavy-16384-hca-cp8r4 | cute_ws@cute | 216.3 | 214.2-217.7 | 0.20 | 1684 | 1674-1689 | 0.67 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 344.0 | 339.7-348.4 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r7 | tilelang@main | 1088 | 1082-1106 | 1.00 | 2621 | 2616-2649 | 1.00 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 346.1 | 343.6-346.5 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r7 | tilelang@cute | 1111 | 1110-1118 | 1.00 | 2576 | 2570-2605 | 1.00 |
| heavy-16384-hca-cp8r7 | cute@cute | 494.1 | 484.4-497.2 | 0.44 | 1952 | 1942-1964 | 0.76 |
| heavy-16384-hca-cp8r7 | cute_ws@cute | 228.2 | 225.0-229.3 | 0.21 | 1756 | 1748-1758 | 0.68 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 355.7 | 351.1-360.4 | 0.32 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang@main | 2979 | 2958-3008 | 1.00 | 7887 | 7879-7894 | 1.00 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 1173 | 1173-1183 | 0.39 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang@cute | 2941 | 2926-2952 | 1.00 | 7892 | 7865-7904 | 1.00 |
| heavy-16384-sliding-cp1 | cute@cute | 2199 | 2194-2202 | 0.75 | 7098 | 7092-7112 | 0.90 |
| heavy-16384-sliding-cp1 | cute_ws@cute | 868.4 | 865.0-872.2 | 0.30 | 5730 | 5720-5742 | 0.73 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1176 | 1174-1177 | 0.40 | - | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@main | 1013 | 1002-1040 | 1.00 | 2292 | 2285-2305 | 1.00 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 297.7 | 295.7-301.9 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@cute | 1007 | 997.7-1011 | 1.00 | 2318 | 2314-2327 | 1.00 |
| heavy-16384-sliding-cp8r0 | cute@cute | 402.0 | 399.6-407.4 | 0.40 | 1690 | 1677-1700 | 0.73 |
| heavy-16384-sliding-cp8r0 | cute_ws@cute | 170.0 | 169.5-171.0 | 0.17 | 1533 | 1529-1549 | 0.66 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 299.4 | 297.9-303.0 | 0.30 | - | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@main | 982.7 | 980.0-1006 | 1.00 | 2331 | 2316-2337 | 1.00 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 293.1 | 290.1-298.7 | 0.30 | - | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@cute | 986.3 | 981.8-1012 | 1.00 | 2343 | 2338-2365 | 1.00 |
| heavy-16384-sliding-cp8r4 | cute@cute | 408.8 | 407.8-411.4 | 0.41 | 1708 | 1698-1711 | 0.73 |
| heavy-16384-sliding-cp8r4 | cute_ws@cute | 170.2 | 169.6-170.6 | 0.17 | 1567 | 1564-1572 | 0.67 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 300.1 | 298.8-303.7 | 0.30 | - | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@main | 1022 | 994.6-1025 | 1.00 | 2405 | 2394-2439 | 1.00 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 296.4 | 293.9-310.3 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@cute | 1001 | 995.9-1004 | 1.00 | 2411 | 2405-2424 | 1.00 |
| heavy-16384-sliding-cp8r7 | cute@cute | 418.8 | 414.4-420.3 | 0.42 | 1767 | 1755-1772 | 0.73 |
| heavy-16384-sliding-cp8r7 | cute_ws@cute | 171.1 | 170.3-172.8 | 0.17 | 1612 | 1607-1627 | 0.67 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 298.9 | 296.0-304.2 | 0.30 | - | - | - |
| tiny-16384-csa-cp1 | tilelang@main | 3287 | 3284-3298 | 1.00 | 8726 | 8702-8759 | 1.00 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 1476 | 1471-1478 | 0.45 | - | - | - |
| tiny-16384-csa-cp1 | tilelang@cute | 3315 | 3288-3354 | 1.00 | 8714 | 8707-8745 | 1.00 |
| tiny-16384-csa-cp1 | cute@cute | 2544 | 2542-2549 | 0.77 | 7901 | 7894-7909 | 0.91 |
| tiny-16384-csa-cp1 | cute_ws@cute | 1301 | 1296-1303 | 0.39 | 6614 | 6610-6618 | 0.76 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@cute | 1477 | 1474-1482 | 0.45 | - | - | - |
| tiny-16384-csa-cp8r0 | tilelang@main | 1043 | 1041-1055 | 1.00 | 2376 | 2356-2382 | 1.00 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 334.5 | 334.0-340.0 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r0 | tilelang@cute | 1036 | 1025-1045 | 1.00 | 2392 | 2385-2401 | 1.00 |
| tiny-16384-csa-cp8r0 | cute@cute | 453.2 | 450.0-454.2 | 0.44 | 1765 | 1758-1769 | 0.74 |
| tiny-16384-csa-cp8r0 | cute_ws@cute | 228.4 | 227.3-229.3 | 0.22 | 1614 | 1596-1623 | 0.67 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 343.7 | 340.9-347.4 | 0.33 | - | - | - |
| tiny-16384-csa-cp8r4 | tilelang@main | 1048 | 1040-1052 | 1.00 | 2358 | 2354-2374 | 1.00 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 342.2 | 335.7-346.6 | 0.33 | - | - | - |
| tiny-16384-csa-cp8r4 | tilelang@cute | 1048 | 1035-1076 | 1.00 | 2406 | 2399-2420 | 1.00 |
| tiny-16384-csa-cp8r4 | cute@cute | 457.2 | 454.6-459.5 | 0.44 | 1764 | 1746-1771 | 0.73 |
| tiny-16384-csa-cp8r4 | cute_ws@cute | 228.0 | 226.1-229.6 | 0.22 | 1615 | 1602-1617 | 0.67 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 344.2 | 336.8-345.4 | 0.33 | - | - | - |
| tiny-16384-csa-cp8r7 | tilelang@main | 1032 | 1022-1076 | 1.00 | 2369 | 2363-2372 | 1.00 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 334.5 | 327.7-343.5 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r7 | tilelang@cute | 1057 | 1044-1064 | 1.00 | 2391 | 2378-2399 | 1.00 |
| tiny-16384-csa-cp8r7 | cute@cute | 456.5 | 453.8-460.7 | 0.43 | 1749 | 1744-1752 | 0.73 |
| tiny-16384-csa-cp8r7 | cute_ws@cute | 228.8 | 227.7-230.1 | 0.22 | 1599 | 1595-1604 | 0.67 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 343.3 | 340.6-348.0 | 0.32 | - | - | - |
| tiny-16384-hca-cp1 | tilelang@main | 2707 | 2700-2714 | 1.00 | 6184 | 6177-6195 | 1.00 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 1176 | 1174-1178 | 0.43 | - | - | - |
| tiny-16384-hca-cp1 | tilelang@cute | 2737 | 2727-2741 | 1.00 | 6194 | 6190-6202 | 1.00 |
| tiny-16384-hca-cp1 | cute@cute | 1989 | 1982-1992 | 0.73 | 5413 | 5406-5418 | 0.87 |
| tiny-16384-hca-cp1 | cute_ws@cute | 983.4 | 980.9-985.5 | 0.36 | 4366 | 4362-4368 | 0.70 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@cute | 1191 | 1188-1198 | 0.44 | - | - | - |
| tiny-16384-hca-cp8r0 | tilelang@main | 981.1 | 976.9-1003 | 1.00 | 2144 | 2130-2255 | 1.00 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 296.6 | 289.3-309.1 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r0 | tilelang@cute | 991.8 | 982.9-998.3 | 1.00 | 2178 | 2170-2285 | 1.00 |
| tiny-16384-hca-cp8r0 | cute@cute | 389.7 | 386.1-393.0 | 0.39 | 1513 | 1501-1529 | 0.69 |
| tiny-16384-hca-cp8r0 | cute_ws@cute | 172.0 | 170.5-172.9 | 0.17 | 1359 | 1344-1370 | 0.62 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 308.1 | 306.5-311.5 | 0.31 | - | - | - |
| tiny-16384-hca-cp8r4 | tilelang@main | 968.2 | 959.1-972.4 | 1.00 | 2095 | 2090-2123 | 1.00 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 293.1 | 289.9-295.4 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r4 | tilelang@cute | 982.3 | 972.3-986.3 | 1.00 | 2162 | 2157-2175 | 1.00 |
| tiny-16384-hca-cp8r4 | cute@cute | 381.9 | 378.1-389.8 | 0.39 | 1508 | 1506-1525 | 0.70 |
| tiny-16384-hca-cp8r4 | cute_ws@cute | 167.6 | 166.2-169.7 | 0.17 | 1358 | 1353-1383 | 0.63 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 301.7 | 299.0-302.6 | 0.31 | - | - | - |
| tiny-16384-hca-cp8r7 | tilelang@main | 955.4 | 942.5-968.8 | 1.00 | 2130 | 2120-2172 | 1.00 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 294.3 | 290.2-298.9 | 0.31 | - | - | - |
| tiny-16384-hca-cp8r7 | tilelang@cute | 963.3 | 958.2-968.7 | 1.00 | 2173 | 2157-2211 | 1.00 |
| tiny-16384-hca-cp8r7 | cute@cute | 374.4 | 373.5-377.3 | 0.39 | 1496 | 1491-1513 | 0.69 |
| tiny-16384-hca-cp8r7 | cute_ws@cute | 168.5 | 167.2-171.1 | 0.17 | 1343 | 1330-1349 | 0.62 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 307.8 | 301.7-310.0 | 0.32 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang@main | 2728 | 2711-2754 | 1.00 | 6163 | 6161-6164 | 1.00 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 1181 | 1174-1182 | 0.43 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang@cute | 2732 | 2707-2745 | 1.00 | 6225 | 6207-6229 | 1.00 |
| tiny-16384-sliding-cp1 | cute@cute | 1976 | 1974-1978 | 0.72 | 5415 | 5408-5431 | 0.87 |
| tiny-16384-sliding-cp1 | cute_ws@cute | 982.6 | 974.8-985.7 | 0.36 | 4374 | 4364-4384 | 0.70 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1184 | 1181-1191 | 0.43 | - | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@main | 962.6 | 956.2-977.8 | 1.00 | 2142 | 2126-2157 | 1.00 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 297.0 | 295.1-301.5 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@cute | 1048 | 1002-1055 | 1.00 | 2175 | 2164-2179 | 1.00 |
| tiny-16384-sliding-cp8r0 | cute@cute | 388.9 | 384.1-393.7 | 0.37 | 1527 | 1519-1530 | 0.70 |
| tiny-16384-sliding-cp8r0 | cute_ws@cute | 170.5 | 169.3-171.2 | 0.16 | 1378 | 1371-1384 | 0.63 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 303.8 | 299.4-304.5 | 0.29 | - | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@main | 962.2 | 954.6-977.3 | 1.00 | 2174 | 2113-2294 | 1.00 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 299.2 | 290.4-299.6 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@cute | 969.6 | 960.9-993.4 | 1.00 | 2156 | 2146-2161 | 1.00 |
| tiny-16384-sliding-cp8r4 | cute@cute | 381.0 | 378.5-385.2 | 0.39 | 1499 | 1498-1510 | 0.70 |
| tiny-16384-sliding-cp8r4 | cute_ws@cute | 168.0 | 166.4-169.0 | 0.17 | 1349 | 1342-1359 | 0.63 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 297.1 | 292.2-298.5 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@main | 961.0 | 949.7-972.0 | 1.00 | 2140 | 2129-2174 | 1.00 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 293.2 | 292.1-301.2 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@cute | 963.4 | 952.5-978.1 | 1.00 | 2137 | 2129-2159 | 1.00 |
| tiny-16384-sliding-cp8r7 | cute@cute | 375.9 | 372.7-376.9 | 0.39 | 1523 | 1493-1542 | 0.71 |
| tiny-16384-sliding-cp8r7 | cute_ws@cute | 169.1 | 167.4-172.6 | 0.18 | 1350 | 1344-1358 | 0.63 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 306.9 | 301.9-321.8 | 0.32 | - | - | - |
| single-49208-csa-cp1 | tilelang@main | 16424 | 16369-16661 | 1.00 | 70708 | 70691-70733 | 1.00 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 8309 | 8259-8522 | 0.51 | - | - | - |
| single-49208-csa-cp1 | tilelang@cute | 16390 | 16333-16452 | 1.00 | 70849 | 70825-70863 | 1.00 |
| single-49208-csa-cp1 | cute@cute | 15147 | 15055-15336 | 0.92 | 69465 | 69414-69476 | 0.98 |
| single-49208-csa-cp1 | cute_ws@cute | 7872 | 7731-8104 | 0.48 | 61763 | 61753-61792 | 0.87 |
| single-49208-csa-cp1 | flashmla_fwd_ref@cute | 8626 | 8525-8640 | 0.53 | - | - | - |
| single-49208-csa-cp8r0 | tilelang@main | 2556 | 2547-2584 | 1.00 | 8923 | 8921-8925 | 1.00 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 1026 | 1020-1028 | 0.40 | - | - | - |
| single-49208-csa-cp8r0 | tilelang@cute | 2581 | 2575-2605 | 1.00 | 8958 | 8933-8982 | 1.00 |
| single-49208-csa-cp8r0 | cute@cute | 1888 | 1886-1936 | 0.73 | 8218 | 8205-8223 | 0.92 |
| single-49208-csa-cp8r0 | cute_ws@cute | 905.7 | 903.7-909.1 | 0.35 | 7197 | 7189-7206 | 0.80 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 1032 | 1028-1034 | 0.40 | - | - | - |
| single-49208-csa-cp8r4 | tilelang@main | 2744 | 2725-2754 | 1.00 | 10025 | 10005-10044 | 1.00 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1100 | 1100-1108 | 0.40 | - | - | - |
| single-49208-csa-cp8r4 | tilelang@cute | 2762 | 2738-2779 | 1.00 | 10076 | 10062-10093 | 1.00 |
| single-49208-csa-cp8r4 | cute@cute | 2067 | 2062-2123 | 0.75 | 9321 | 9295-9330 | 0.93 |
| single-49208-csa-cp8r4 | cute_ws@cute | 986.0 | 983.7-1001 | 0.36 | 8162 | 8155-8165 | 0.81 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 1112 | 1107-1115 | 0.40 | - | - | - |
| single-49208-csa-cp8r7 | tilelang@main | 2745 | 2736-2763 | 1.00 | 10361 | 10302-10387 | 1.00 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1105 | 1101-1108 | 0.40 | - | - | - |
| single-49208-csa-cp8r7 | tilelang@cute | 2771 | 2756-2775 | 1.00 | 10390 | 10354-10422 | 1.00 |
| single-49208-csa-cp8r7 | cute@cute | 2068 | 2067-2086 | 0.75 | 9639 | 9620-9644 | 0.93 |
| single-49208-csa-cp8r7 | cute_ws@cute | 988.2 | 987.1-991.3 | 0.36 | 8497 | 8469-8518 | 0.82 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 1116 | 1112-1119 | 0.40 | - | - | - |
| single-49208-hca-cp1 | tilelang@main | 11496 | 11468-11639 | 1.00 | 42559 | 42547-42587 | 1.00 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 5402 | 5373-5651 | 0.47 | - | - | - |
| single-49208-hca-cp1 | tilelang@cute | 11502 | 11443-11512 | 1.00 | 42556 | 42555-42572 | 1.00 |
| single-49208-hca-cp1 | cute@cute | 10306 | 10244-10534 | 0.90 | 41326 | 41319-41330 | 0.97 |
| single-49208-hca-cp1 | cute_ws@cute | 5010 | 4945-5155 | 0.44 | 35893 | 35887-35899 | 0.84 |
| single-49208-hca-cp1 | flashmla_fwd_ref@cute | 5670 | 5614-5692 | 0.49 | - | - | - |
| single-49208-hca-cp8r0 | tilelang@main | 1710 | 1698-1758 | 1.00 | 4247 | 4246-4326 | 1.00 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 670.5 | 667.3-673.7 | 0.39 | - | - | - |
| single-49208-hca-cp8r0 | tilelang@cute | 1707 | 1704-1730 | 1.00 | 4270 | 4265-4318 | 1.00 |
| single-49208-hca-cp8r0 | cute@cute | 1072 | 1066-1081 | 0.63 | 3583 | 3575-3591 | 0.84 |
| single-49208-hca-cp8r0 | cute_ws@cute | 530.7 | 529.1-532.7 | 0.31 | 3035 | 3026-3041 | 0.71 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 675.9 | 672.5-679.9 | 0.40 | - | - | - |
| single-49208-hca-cp8r4 | tilelang@main | 2145 | 2139-2186 | 1.00 | 6565 | 6561-6572 | 1.00 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 801.2 | 799.3-802.6 | 0.37 | - | - | - |
| single-49208-hca-cp8r4 | tilelang@cute | 2155 | 2150-2173 | 1.00 | 6587 | 6582-6591 | 1.00 |
| single-49208-hca-cp8r4 | cute@cute | 1493 | 1488-1497 | 0.69 | 5876 | 5874-5886 | 0.89 |
| single-49208-hca-cp8r4 | cute_ws@cute | 672.6 | 668.4-678.1 | 0.31 | 5015 | 5009-5017 | 0.76 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 809.6 | 808.9-813.7 | 0.38 | - | - | - |
| single-49208-hca-cp8r7 | tilelang@main | 2430 | 2427-2432 | 1.00 | 8244 | 8226-8262 | 1.00 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 944.5 | 939.2-955.3 | 0.39 | - | - | - |
| single-49208-hca-cp8r7 | tilelang@cute | 2453 | 2443-2504 | 1.00 | 8224 | 8219-8224 | 1.00 |
| single-49208-hca-cp8r7 | cute@cute | 1784 | 1778-1795 | 0.73 | 7515 | 7512-7522 | 0.91 |
| single-49208-hca-cp8r7 | cute_ws@cute | 820.5 | 819.5-822.0 | 0.33 | 6518 | 6512-6521 | 0.79 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 953.6 | 948.9-956.8 | 0.39 | - | - | - |
| single-49208-sliding-cp1 | tilelang@main | 7433 | 7429-7462 | 1.00 | 22888 | 22878-22898 | 1.00 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 3125 | 3112-3128 | 0.42 | - | - | - |
| single-49208-sliding-cp1 | tilelang@cute | 7473 | 7457-7493 | 1.00 | 22949 | 22941-22977 | 1.00 |
| single-49208-sliding-cp1 | cute@cute | 6392 | 6372-6541 | 0.86 | 21816 | 21795-21824 | 0.95 |
| single-49208-sliding-cp1 | cute_ws@cute | 2426 | 2421-2439 | 0.32 | 17823 | 17822-17829 | 0.78 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3185 | 3141-3192 | 0.43 | - | - | - |
| single-49208-sliding-cp8r0 | tilelang@main | 1574 | 1566-1595 | 1.00 | 3745 | 3734-3748 | 1.00 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 543.0 | 540.3-547.6 | 0.35 | - | - | - |
| single-49208-sliding-cp8r0 | tilelang@cute | 1581 | 1568-1601 | 1.00 | 3761 | 3747-3771 | 1.00 |
| single-49208-sliding-cp8r0 | cute@cute | 932.8 | 928.6-935.8 | 0.59 | 3072 | 3067-3079 | 0.82 |
| single-49208-sliding-cp8r0 | cute_ws@cute | 368.5 | 366.0-372.3 | 0.23 | 2664 | 2658-2680 | 0.71 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 543.9 | 543.1-547.5 | 0.34 | - | - | - |
| single-49208-sliding-cp8r4 | tilelang@main | 1557 | 1554-1558 | 1.00 | 3783 | 3768-3790 | 1.00 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 539.1 | 537.2-545.8 | 0.35 | - | - | - |
| single-49208-sliding-cp8r4 | tilelang@cute | 1565 | 1559-1587 | 1.00 | 3794 | 3791-3803 | 1.00 |
| single-49208-sliding-cp8r4 | cute@cute | 934.3 | 930.3-949.5 | 0.60 | 3109 | 3106-3112 | 0.82 |
| single-49208-sliding-cp8r4 | cute_ws@cute | 369.7 | 369.1-370.5 | 0.24 | 2707 | 2695-2730 | 0.71 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 547.0 | 543.8-549.8 | 0.35 | - | - | - |
| single-49208-sliding-cp8r7 | tilelang@main | 1572 | 1565-1598 | 1.00 | 3792 | 3777-3798 | 1.00 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 540.1 | 536.8-541.1 | 0.34 | - | - | - |
| single-49208-sliding-cp8r7 | tilelang@cute | 1579 | 1565-1588 | 1.00 | 3807 | 3800-3811 | 1.00 |
| single-49208-sliding-cp8r7 | cute@cute | 937.0 | 935.5-941.0 | 0.59 | 3122 | 3116-3125 | 0.82 |
| single-49208-sliding-cp8r7 | cute_ws@cute | 371.4 | 370.3-372.6 | 0.24 | 2709 | 2707-2727 | 0.71 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 547.6 | 543.3-553.1 | 0.35 | - | - | - |
| short-49208-csa-cp1 | tilelang@main | 12988 | 12944-13072 | 1.00 | 51070 | 51061-51101 | 1.00 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 6227 | 6191-6569 | 0.48 | - | - | - |
| short-49208-csa-cp1 | tilelang@cute | 12987 | 12965-13004 | 1.00 | 51109 | 51105-51116 | 1.00 |
| short-49208-csa-cp1 | cute@cute | 11807 | 11746-12037 | 0.91 | 49847 | 49836-49849 | 0.98 |
| short-49208-csa-cp1 | cute_ws@cute | 6012 | 5740-6181 | 0.46 | 43753 | 43729-43761 | 0.86 |
| short-49208-csa-cp1 | flashmla_fwd_ref@cute | 6615 | 6607-6677 | 0.51 | - | - | - |
| short-49208-csa-cp8r0 | tilelang@main | 2037 | 2013-2042 | 1.00 | 6012 | 6007-6017 | 1.00 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 791.1 | 785.7-795.3 | 0.39 | - | - | - |
| short-49208-csa-cp8r0 | tilelang@cute | 2031 | 2024-2074 | 1.00 | 6036 | 6030-6048 | 1.00 |
| short-49208-csa-cp8r0 | cute@cute | 1386 | 1381-1391 | 0.68 | 5345 | 5341-5348 | 0.89 |
| short-49208-csa-cp8r0 | cute_ws@cute | 659.8 | 659.2-661.4 | 0.32 | 4578 | 4569-4586 | 0.76 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 796.5 | 796.1-805.7 | 0.39 | - | - | - |
| short-49208-csa-cp8r4 | tilelang@main | 2386 | 2373-2409 | 1.00 | 8086 | 8074-8104 | 1.00 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 959.5 | 958.1-965.9 | 0.40 | - | - | - |
| short-49208-csa-cp8r4 | tilelang@cute | 2411 | 2408-2454 | 1.00 | 8103 | 8098-8121 | 1.00 |
| short-49208-csa-cp8r4 | cute@cute | 1748 | 1744-1748 | 0.72 | 7404 | 7395-7426 | 0.91 |
| short-49208-csa-cp8r4 | cute_ws@cute | 845.3 | 844.1-850.5 | 0.35 | 6466 | 6462-6468 | 0.80 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 967.3 | 966.7-969.9 | 0.40 | - | - | - |
| short-49208-csa-cp8r7 | tilelang@main | 2161 | 2150-2181 | 1.00 | 6809 | 6800-6810 | 1.00 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 858.3 | 856.4-860.1 | 0.40 | - | - | - |
| short-49208-csa-cp8r7 | tilelang@cute | 2163 | 2160-2174 | 1.00 | 6789 | 6776-6797 | 1.00 |
| short-49208-csa-cp8r7 | cute@cute | 1511 | 1510-1524 | 0.70 | 6099 | 6095-6110 | 0.90 |
| short-49208-csa-cp8r7 | cute_ws@cute | 733.6 | 730.8-737.4 | 0.34 | 5284 | 5280-5287 | 0.78 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 863.1 | 861.6-865.0 | 0.40 | - | - | - |
| short-49208-hca-cp1 | tilelang@main | 8505 | 8498-8522 | 1.00 | 26175 | 26167-26201 | 1.00 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 4103 | 4093-4107 | 0.48 | - | - | - |
| short-49208-hca-cp1 | tilelang@cute | 8563 | 8520-8568 | 1.00 | 26190 | 26173-26206 | 1.00 |
| short-49208-hca-cp1 | cute@cute | 7414 | 7403-7419 | 0.87 | 25039 | 25028-25042 | 0.96 |
| short-49208-hca-cp1 | cute_ws@cute | 3571 | 3559-3596 | 0.42 | 21167 | 21159-21172 | 0.81 |
| short-49208-hca-cp1 | flashmla_fwd_ref@cute | 4122 | 4110-4127 | 0.48 | - | - | - |
| short-49208-hca-cp8r0 | tilelang@main | 1701 | 1696-1716 | 1.00 | 4068 | 4063-4077 | 1.00 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 646.6 | 644.3-648.9 | 0.38 | - | - | - |
| short-49208-hca-cp8r0 | tilelang@cute | 1728 | 1719-1736 | 1.00 | 4078 | 4068-4087 | 1.00 |
| short-49208-hca-cp8r0 | cute@cute | 1068 | 1066-1074 | 0.62 | 3393 | 3387-3398 | 0.83 |
| short-49208-hca-cp8r0 | cute_ws@cute | 509.1 | 507.7-514.8 | 0.29 | 2871 | 2863-2876 | 0.70 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 660.4 | 659.1-662.7 | 0.38 | - | - | - |
| short-49208-hca-cp8r4 | tilelang@main | 1747 | 1727-1765 | 1.00 | 4187 | 4177-4188 | 1.00 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 665.3 | 661.2-667.1 | 0.38 | - | - | - |
| short-49208-hca-cp8r4 | tilelang@cute | 1730 | 1721-1737 | 1.00 | 4234 | 4226-4240 | 1.00 |
| short-49208-hca-cp8r4 | cute@cute | 1086 | 1082-1090 | 0.63 | 3542 | 3540-3545 | 0.84 |
| short-49208-hca-cp8r4 | cute_ws@cute | 522.6 | 517.7-524.8 | 0.30 | 3004 | 2997-3006 | 0.71 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 668.4 | 666.4-674.0 | 0.39 | - | - | - |
| short-49208-hca-cp8r7 | tilelang@main | 1733 | 1727-1755 | 1.00 | 4169 | 4154-4176 | 1.00 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 664.6 | 659.7-670.8 | 0.38 | - | - | - |
| short-49208-hca-cp8r7 | tilelang@cute | 1741 | 1728-1751 | 1.00 | 4162 | 4161-4167 | 1.00 |
| short-49208-hca-cp8r7 | cute@cute | 1090 | 1080-1095 | 0.63 | 3480 | 3473-3483 | 0.84 |
| short-49208-hca-cp8r7 | cute_ws@cute | 519.4 | 516.3-520.8 | 0.30 | 2939 | 2936-2946 | 0.71 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 669.9 | 665.9-675.0 | 0.38 | - | - | - |
| short-49208-sliding-cp1 | tilelang@main | 7423 | 7418-7435 | 1.00 | 22632 | 22622-22641 | 1.00 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 3123 | 3117-3127 | 0.42 | - | - | - |
| short-49208-sliding-cp1 | tilelang@cute | 7463 | 7450-7470 | 1.00 | 22656 | 22616-22665 | 1.00 |
| short-49208-sliding-cp1 | cute@cute | 6354 | 6344-6542 | 0.85 | 21509 | 21506-21514 | 0.95 |
| short-49208-sliding-cp1 | cute_ws@cute | 2431 | 2425-2432 | 0.33 | 17554 | 17542-17556 | 0.77 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3164 | 3153-3180 | 0.42 | - | - | - |
| short-49208-sliding-cp8r0 | tilelang@main | 1565 | 1552-1569 | 1.00 | 3663 | 3651-3680 | 1.00 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 541.0 | 537.3-548.1 | 0.35 | - | - | - |
| short-49208-sliding-cp8r0 | tilelang@cute | 1552 | 1545-1563 | 1.00 | 3685 | 3675-3702 | 1.00 |
| short-49208-sliding-cp8r0 | cute@cute | 916.9 | 916.3-921.8 | 0.59 | 3004 | 2999-3006 | 0.82 |
| short-49208-sliding-cp8r0 | cute_ws@cute | 369.2 | 367.4-370.7 | 0.24 | 2614 | 2592-2617 | 0.71 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 547.1 | 543.6-548.9 | 0.35 | - | - | - |
| short-49208-sliding-cp8r4 | tilelang@main | 1569 | 1551-1574 | 1.00 | 3739 | 3720-3763 | 1.00 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 545.8 | 542.3-547.2 | 0.35 | - | - | - |
| short-49208-sliding-cp8r4 | tilelang@cute | 1568 | 1555-1570 | 1.00 | 3754 | 3749-3760 | 1.00 |
| short-49208-sliding-cp8r4 | cute@cute | 930.6 | 928.9-937.7 | 0.59 | 3064 | 3063-3071 | 0.82 |
| short-49208-sliding-cp8r4 | cute_ws@cute | 371.3 | 369.5-372.2 | 0.24 | 2663 | 2659-2674 | 0.71 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 550.3 | 544.8-554.9 | 0.35 | - | - | - |
| short-49208-sliding-cp8r7 | tilelang@main | 1556 | 1551-1565 | 1.00 | 3743 | 3735-3746 | 1.00 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 536.9 | 535.8-539.6 | 0.34 | - | - | - |
| short-49208-sliding-cp8r7 | tilelang@cute | 1571 | 1561-1576 | 1.00 | 3731 | 3718-3744 | 1.00 |
| short-49208-sliding-cp8r7 | cute@cute | 934.1 | 931.6-937.6 | 0.59 | 3049 | 3046-3060 | 0.82 |
| short-49208-sliding-cp8r7 | cute_ws@cute | 369.1 | 368.6-369.6 | 0.23 | 2643 | 2635-2648 | 0.71 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 548.5 | 545.8-550.8 | 0.35 | - | - | - |
| heavy-49208-csa-cp1 | tilelang@main | 14914 | 14896-14994 | 1.00 | 61708 | 61688-61710 | 1.00 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 7394 | 7345-7676 | 0.50 | - | - | - |
| heavy-49208-csa-cp1 | tilelang@cute | 14860 | 14784-14883 | 1.00 | 61745 | 61731-61760 | 1.00 |
| heavy-49208-csa-cp1 | cute@cute | 13637 | 13559-13858 | 0.92 | 60436 | 60434-60448 | 0.98 |
| heavy-49208-csa-cp1 | cute_ws@cute | 6950 | 6936-7285 | 0.47 | 53458 | 53457-53483 | 0.87 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@cute | 7663 | 7647-7673 | 0.52 | - | - | - |
| heavy-49208-csa-cp8r0 | tilelang@main | 1925 | 1921-1931 | 1.00 | 5420 | 5416-5427 | 1.00 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 759.8 | 758.7-765.0 | 0.39 | - | - | - |
| heavy-49208-csa-cp8r0 | tilelang@cute | 1925 | 1912-1932 | 1.00 | 5464 | 5462-5482 | 1.00 |
| heavy-49208-csa-cp8r0 | cute@cute | 1280 | 1278-1284 | 0.66 | 4757 | 4750-4759 | 0.87 |
| heavy-49208-csa-cp8r0 | cute_ws@cute | 618.4 | 613.2-620.3 | 0.32 | 4048 | 4044-4050 | 0.74 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 762.8 | 761.3-767.8 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r4 | tilelang@main | 2629 | 2613-2647 | 1.00 | 9369 | 9360-9377 | 1.00 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1058 | 1054-1062 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r4 | tilelang@cute | 2666 | 2630-2690 | 1.00 | 9431 | 9408-9441 | 1.00 |
| heavy-49208-csa-cp8r4 | cute@cute | 1965 | 1961-2024 | 0.74 | 8648 | 8644-8665 | 0.92 |
| heavy-49208-csa-cp8r4 | cute_ws@cute | 939.9 | 938.3-946.6 | 0.35 | 7580 | 7571-7586 | 0.80 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 1062 | 1059-1068 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r7 | tilelang@main | 2477 | 2464-2492 | 1.00 | 8582 | 8578-8592 | 1.00 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 990.5 | 988.4-996.9 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r7 | tilelang@cute | 2504 | 2495-2592 | 1.00 | 8586 | 8565-8628 | 1.00 |
| heavy-49208-csa-cp8r7 | cute@cute | 1840 | 1821-1874 | 0.73 | 7876 | 7869-7884 | 0.92 |
| heavy-49208-csa-cp8r7 | cute_ws@cute | 874.9 | 871.0-878.1 | 0.35 | 6886 | 6877-6887 | 0.80 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 997.6 | 995.3-998.6 | 0.40 | - | - | - |
| heavy-49208-hca-cp1 | tilelang@main | 9089 | 9066-9107 | 1.00 | 29605 | 29574-29619 | 1.00 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4274 | 4269-4284 | 0.47 | - | - | - |
| heavy-49208-hca-cp1 | tilelang@cute | 9101 | 9096-9128 | 1.00 | 29678 | 29667-29705 | 1.00 |
| heavy-49208-hca-cp1 | cute@cute | 7984 | 7977-8079 | 0.88 | 28516 | 28506-28522 | 0.96 |
| heavy-49208-hca-cp1 | cute_ws@cute | 3680 | 3665-3757 | 0.40 | 24220 | 24216-24223 | 0.82 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@cute | 4334 | 4309-4366 | 0.48 | - | - | - |
| heavy-49208-hca-cp8r0 | tilelang@main | 1702 | 1695-1730 | 1.00 | 3912 | 3899-3948 | 1.00 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 642.8 | 634.5-652.1 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r0 | tilelang@cute | 1688 | 1682-1698 | 1.00 | 3922 | 3906-3925 | 1.00 |
| heavy-49208-hca-cp8r0 | cute@cute | 1032 | 1030-1036 | 0.61 | 3241 | 3236-3247 | 0.83 |
| heavy-49208-hca-cp8r0 | cute_ws@cute | 486.4 | 484.6-486.6 | 0.29 | 2760 | 2756-2765 | 0.70 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 642.2 | 638.5-642.5 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r4 | tilelang@main | 1753 | 1742-1768 | 1.00 | 4393 | 4384-4416 | 1.00 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 675.0 | 669.0-676.8 | 0.39 | - | - | - |
| heavy-49208-hca-cp8r4 | tilelang@cute | 1769 | 1760-1774 | 1.00 | 4426 | 4417-4436 | 1.00 |
| heavy-49208-hca-cp8r4 | cute@cute | 1108 | 1106-1113 | 0.63 | 3727 | 3720-3729 | 0.84 |
| heavy-49208-hca-cp8r4 | cute_ws@cute | 528.1 | 527.2-531.4 | 0.30 | 3184 | 3177-3186 | 0.72 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 677.4 | 677.0-681.1 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r7 | tilelang@main | 1845 | 1815-1849 | 1.00 | 4673 | 4665-4690 | 1.00 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 712.1 | 708.3-714.2 | 0.39 | - | - | - |
| heavy-49208-hca-cp8r7 | tilelang@cute | 1829 | 1827-1848 | 1.00 | 4664 | 4662-4670 | 1.00 |
| heavy-49208-hca-cp8r7 | cute@cute | 1170 | 1167-1172 | 0.64 | 3972 | 3968-3975 | 0.85 |
| heavy-49208-hca-cp8r7 | cute_ws@cute | 572.5 | 568.7-573.9 | 0.31 | 3365 | 3360-3383 | 0.72 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 717.8 | 716.8-726.2 | 0.39 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang@main | 7448 | 7405-7474 | 1.00 | 22583 | 22581-22614 | 1.00 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 3135 | 3122-3149 | 0.42 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang@cute | 7465 | 7446-7467 | 1.00 | 22630 | 22612-22645 | 1.00 |
| heavy-49208-sliding-cp1 | cute@cute | 6356 | 6345-6509 | 0.85 | 21496 | 21491-21512 | 0.95 |
| heavy-49208-sliding-cp1 | cute_ws@cute | 2429 | 2423-2448 | 0.33 | 17534 | 17525-17541 | 0.77 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3153 | 3149-3183 | 0.42 | - | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@main | 1572 | 1556-1586 | 1.00 | 3588 | 3579-3605 | 1.00 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 542.2 | 536.7-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@cute | 1572 | 1565-1603 | 1.00 | 3593 | 3592-3605 | 1.00 |
| heavy-49208-sliding-cp8r0 | cute@cute | 922.8 | 920.5-927.0 | 0.59 | 2913 | 2911-2922 | 0.81 |
| heavy-49208-sliding-cp8r0 | cute_ws@cute | 380.6 | 378.5-382.1 | 0.24 | 2533 | 2522-2546 | 0.70 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 554.2 | 549.8-555.4 | 0.35 | - | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@main | 1565 | 1560-1583 | 1.00 | 3761 | 3760-3770 | 1.00 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 538.9 | 537.2-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@cute | 1575 | 1570-1583 | 1.00 | 3837 | 3816-3841 | 1.00 |
| heavy-49208-sliding-cp8r4 | cute@cute | 942.6 | 937.5-943.9 | 0.60 | 3137 | 3130-3142 | 0.82 |
| heavy-49208-sliding-cp8r4 | cute_ws@cute | 369.8 | 369.0-373.5 | 0.23 | 2746 | 2733-2764 | 0.72 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 551.0 | 547.9-552.2 | 0.35 | - | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@main | 1567 | 1558-1572 | 1.00 | 3766 | 3757-3772 | 1.00 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 543.2 | 540.1-547.6 | 0.35 | - | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@cute | 1580 | 1573-1616 | 1.00 | 3775 | 3773-3788 | 1.00 |
| heavy-49208-sliding-cp8r7 | cute@cute | 941.9 | 936.7-943.1 | 0.60 | 3094 | 3089-3099 | 0.82 |
| heavy-49208-sliding-cp8r7 | cute_ws@cute | 371.3 | 370.5-374.0 | 0.24 | 2700 | 2694-2706 | 0.72 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 550.6 | 549.1-555.6 | 0.35 | - | - | - |
| tiny-49208-csa-cp1 | tilelang@main | 8393 | 8360-8427 | 1.00 | 24104 | 24092-24115 | 1.00 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 4295 | 4284-4297 | 0.51 | - | - | - |
| tiny-49208-csa-cp1 | tilelang@cute | 8397 | 8363-8416 | 1.00 | 24155 | 24152-24158 | 1.00 |
| tiny-49208-csa-cp1 | cute@cute | 7365 | 7361-7413 | 0.88 | 23038 | 23030-23041 | 0.95 |
| tiny-49208-csa-cp1 | cute_ws@cute | 3733 | 3714-3747 | 0.44 | 19408 | 19392-19411 | 0.80 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@cute | 4298 | 4287-4308 | 0.51 | - | - | - |
| tiny-49208-csa-cp8r0 | tilelang@main | 1704 | 1673-1720 | 1.00 | 3875 | 3869-3881 | 1.00 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 658.2 | 656.7-659.3 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r0 | tilelang@cute | 1686 | 1681-1696 | 1.00 | 3910 | 3903-3917 | 1.00 |
| tiny-49208-csa-cp8r0 | cute@cute | 1057 | 1054-1059 | 0.63 | 3242 | 3240-3243 | 0.83 |
| tiny-49208-csa-cp8r0 | cute_ws@cute | 533.3 | 533.0-535.9 | 0.32 | 2696 | 2689-2701 | 0.69 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 676.6 | 669.1-679.9 | 0.40 | - | - | - |
| tiny-49208-csa-cp8r4 | tilelang@main | 1680 | 1674-1688 | 1.00 | 3906 | 3895-3913 | 1.00 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 656.4 | 651.7-664.8 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r4 | tilelang@cute | 1700 | 1688-1723 | 1.00 | 3947 | 3936-3958 | 1.00 |
| tiny-49208-csa-cp8r4 | cute@cute | 1061 | 1057-1064 | 0.62 | 3257 | 3255-3259 | 0.83 |
| tiny-49208-csa-cp8r4 | cute_ws@cute | 536.7 | 534.9-539.6 | 0.32 | 2712 | 2704-2715 | 0.69 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 668.2 | 660.2-671.5 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r7 | tilelang@main | 1689 | 1670-1693 | 1.00 | 3912 | 3898-3930 | 1.00 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 659.4 | 655.1-660.4 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r7 | tilelang@cute | 1693 | 1682-1701 | 1.00 | 3926 | 3905-3930 | 1.00 |
| tiny-49208-csa-cp8r7 | cute@cute | 1052 | 1049-1057 | 0.62 | 3240 | 3237-3242 | 0.83 |
| tiny-49208-csa-cp8r7 | cute_ws@cute | 534.4 | 533.4-535.8 | 0.32 | 2696 | 2694-2700 | 0.69 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 663.2 | 661.9-668.0 | 0.39 | - | - | - |
| tiny-49208-hca-cp1 | tilelang@main | 6744 | 6728-6770 | 1.00 | 16786 | 16781-16790 | 1.00 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 3164 | 3162-3181 | 0.47 | - | - | - |
| tiny-49208-hca-cp1 | tilelang@cute | 6721 | 6705-6757 | 1.00 | 16758 | 16756-16764 | 1.00 |
| tiny-49208-hca-cp1 | cute@cute | 5635 | 5621-5641 | 0.84 | 15645 | 15643-15647 | 0.93 |
| tiny-49208-hca-cp1 | cute_ws@cute | 2757 | 2734-2762 | 0.41 | 12711 | 12700-12733 | 0.76 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@cute | 3185 | 3175-3192 | 0.47 | - | - | - |
| tiny-49208-hca-cp8r0 | tilelang@main | 1479 | 1468-1496 | 1.00 | 2952 | 2948-2958 | 1.00 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 541.4 | 536.4-544.8 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r0 | tilelang@cute | 1472 | 1470-1488 | 1.00 | 2965 | 2964-2974 | 1.00 |
| tiny-49208-hca-cp8r0 | cute@cute | 838.8 | 832.1-841.1 | 0.57 | 2287 | 2286-2289 | 0.77 |
| tiny-49208-hca-cp8r0 | cute_ws@cute | 403.3 | 396.9-408.7 | 0.27 | 1980 | 1976-2017 | 0.67 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 549.7 | 545.1-555.5 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r4 | tilelang@main | 1485 | 1472-1546 | 1.00 | 2981 | 2974-2984 | 1.00 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 544.0 | 539.4-559.9 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r4 | tilelang@cute | 1476 | 1473-1505 | 1.00 | 3018 | 3011-3038 | 1.00 |
| tiny-49208-hca-cp8r4 | cute@cute | 849.3 | 845.4-851.2 | 0.58 | 2331 | 2323-2338 | 0.77 |
| tiny-49208-hca-cp8r4 | cute_ws@cute | 402.8 | 389.9-406.3 | 0.27 | 2021 | 2016-2022 | 0.67 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 551.8 | 548.3-554.3 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r7 | tilelang@main | 1474 | 1461-1497 | 1.00 | 2953 | 2945-2961 | 1.00 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 540.6 | 538.1-542.2 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r7 | tilelang@cute | 1487 | 1482-1501 | 1.00 | 2962 | 2955-2972 | 1.00 |
| tiny-49208-hca-cp8r7 | cute@cute | 843.1 | 841.8-849.0 | 0.57 | 2284 | 2281-2289 | 0.77 |
| tiny-49208-hca-cp8r7 | cute_ws@cute | 405.7 | 399.2-410.2 | 0.27 | 1993 | 1968-2023 | 0.67 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 568.6 | 551.8-575.1 | 0.38 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang@main | 6733 | 6721-6755 | 1.00 | 16801 | 16796-16808 | 1.00 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 3173 | 3173-3185 | 0.47 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang@cute | 6726 | 6706-6754 | 1.00 | 16761 | 16755-16763 | 1.00 |
| tiny-49208-sliding-cp1 | cute@cute | 5644 | 5639-5647 | 0.84 | 15653 | 15645-15656 | 0.93 |
| tiny-49208-sliding-cp1 | cute_ws@cute | 2745 | 2734-2768 | 0.41 | 12705 | 12698-12706 | 0.76 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3186 | 3181-3193 | 0.47 | - | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@main | 1463 | 1458-1465 | 1.00 | 2947 | 2942-2964 | 1.00 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 540.3 | 536.1-543.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@cute | 1478 | 1472-1489 | 1.00 | 2978 | 2976-2985 | 1.00 |
| tiny-49208-sliding-cp8r0 | cute@cute | 842.6 | 838.0-844.8 | 0.57 | 2298 | 2297-2302 | 0.77 |
| tiny-49208-sliding-cp8r0 | cute_ws@cute | 406.8 | 400.4-408.6 | 0.28 | 1992 | 1983-1995 | 0.67 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 554.3 | 545.1-556.9 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@main | 1485 | 1470-1510 | 1.00 | 2976 | 2975-2983 | 1.00 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 542.5 | 538.3-548.4 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@cute | 1482 | 1480-1484 | 1.00 | 3022 | 3017-3050 | 1.00 |
| tiny-49208-sliding-cp8r4 | cute@cute | 849.3 | 847.3-855.0 | 0.57 | 2333 | 2326-2335 | 0.77 |
| tiny-49208-sliding-cp8r4 | cute_ws@cute | 404.2 | 390.2-406.8 | 0.27 | 2045 | 2016-2059 | 0.68 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 548.7 | 547.4-552.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@main | 1465 | 1463-1477 | 1.00 | 2951 | 2947-2968 | 1.00 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 542.3 | 536.3-545.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@cute | 1482 | 1476-1494 | 1.00 | 2963 | 2963-2976 | 1.00 |
| tiny-49208-sliding-cp8r7 | cute@cute | 844.1 | 843.1-847.0 | 0.57 | 2278 | 2275-2287 | 0.77 |
| tiny-49208-sliding-cp8r7 | cute_ws@cute | 406.2 | 397.0-407.9 | 0.27 | 1973 | 1971-1984 | 0.67 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 548.1 | 546.5-554.8 | 0.37 | - | - | - |
| single-65536-csa-cp1 | tilelang@main | 21694 | 21617-21724 | 1.00 | 95444 | 95324-95476 | 1.00 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 11385 | 11377-11470 | 0.52 | - | - | - |
| single-65536-csa-cp1 | tilelang@cute | 21603 | 21470-21637 | 1.00 | 95477 | 95402-95502 | 1.00 |
| single-65536-csa-cp1 | cute@cute | 20174 | 20054-20232 | 0.93 | 93897 | 93838-93933 | 0.98 |
| single-65536-csa-cp1 | cute_ws@cute | 10950 | 10683-10955 | 0.51 | 83963 | 83870-84103 | 0.88 |
| single-65536-csa-cp1 | flashmla_fwd_ref@cute | 11473 | 11462-11523 | 0.53 | - | - | - |
| single-65536-csa-cp8r0 | tilelang@main | 3265 | 3257-3300 | 1.00 | 11931 | 11922-11967 | 1.00 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 1325 | 1322-1327 | 0.41 | - | - | - |
| single-65536-csa-cp8r0 | tilelang@cute | 3228 | 3222-3261 | 1.00 | 11969 | 11962-11975 | 1.00 |
| single-65536-csa-cp8r0 | cute@cute | 2536 | 2521-2651 | 0.79 | 11227 | 11213-11228 | 0.94 |
| single-65536-csa-cp8r0 | cute_ws@cute | 1217 | 1208-1237 | 0.38 | 9863 | 9852-9871 | 0.82 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 1337 | 1335-1339 | 0.41 | - | - | - |
| single-65536-csa-cp8r4 | tilelang@main | 3429 | 3394-3504 | 1.00 | 13145 | 13143-13204 | 1.00 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1410 | 1405-1413 | 0.41 | - | - | - |
| single-65536-csa-cp8r4 | tilelang@cute | 3411 | 3400-3472 | 1.00 | 13217 | 13195-13245 | 1.00 |
| single-65536-csa-cp8r4 | cute@cute | 2704 | 2688-2883 | 0.79 | 12458 | 12446-12475 | 0.94 |
| single-65536-csa-cp8r4 | cute_ws@cute | 1293 | 1289-1356 | 0.38 | 11002 | 10986-11005 | 0.83 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 1422 | 1414-1435 | 0.42 | - | - | - |
| single-65536-csa-cp8r7 | tilelang@main | 3384 | 3367-3434 | 1.00 | 13996 | 13970-14040 | 1.00 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1410 | 1408-1414 | 0.42 | - | - | - |
| single-65536-csa-cp8r7 | tilelang@cute | 3432 | 3421-3477 | 1.00 | 14019 | 13980-14029 | 1.00 |
| single-65536-csa-cp8r7 | cute@cute | 2712 | 2694-2848 | 0.79 | 13266 | 13231-13302 | 0.95 |
| single-65536-csa-cp8r7 | cute_ws@cute | 1309 | 1306-1341 | 0.38 | 11790 | 11775-11875 | 0.84 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 1443 | 1426-1445 | 0.42 | - | - | - |
| single-65536-hca-cp1 | tilelang@main | 16503 | 16484-16545 | 1.00 | 64327 | 64312-64341 | 1.00 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 8196 | 7971-8458 | 0.50 | - | - | - |
| single-65536-hca-cp1 | tilelang@cute | 16487 | 16391-16559 | 1.00 | 64333 | 64323-64359 | 1.00 |
| single-65536-hca-cp1 | cute@cute | 15213 | 15086-15275 | 0.92 | 62940 | 62935-62947 | 0.98 |
| single-65536-hca-cp1 | cute_ws@cute | 7655 | 7512-7848 | 0.46 | 55014 | 55007-55025 | 0.86 |
| single-65536-hca-cp1 | flashmla_fwd_ref@cute | 8430 | 8401-8499 | 0.51 | - | - | - |
| single-65536-hca-cp8r0 | tilelang@main | 2074 | 2053-2077 | 1.00 | 5434 | 5425-5441 | 1.00 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 835.2 | 832.6-836.7 | 0.40 | - | - | - |
| single-65536-hca-cp8r0 | tilelang@cute | 2052 | 2049-2079 | 1.00 | 5446 | 5442-5468 | 1.00 |
| single-65536-hca-cp8r0 | cute@cute | 1384 | 1377-1386 | 0.67 | 4744 | 4742-4757 | 0.87 |
| single-65536-hca-cp8r0 | cute_ws@cute | 675.7 | 673.6-677.8 | 0.33 | 3993 | 3985-3997 | 0.73 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 843.8 | 838.1-848.2 | 0.41 | - | - | - |
| single-65536-hca-cp8r4 | tilelang@main | 2797 | 2781-2799 | 1.00 | 9582 | 9559-9586 | 1.00 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1185 | 1183-1190 | 0.42 | - | - | - |
| single-65536-hca-cp8r4 | tilelang@cute | 2813 | 2802-2875 | 1.00 | 9624 | 9613-9634 | 1.00 |
| single-65536-hca-cp8r4 | cute@cute | 2142 | 2133-2148 | 0.76 | 8857 | 8842-8863 | 0.92 |
| single-65536-hca-cp8r4 | cute_ws@cute | 1068 | 1065-1070 | 0.38 | 7729 | 7727-7737 | 0.80 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 1201 | 1195-1205 | 0.43 | - | - | - |
| single-65536-hca-cp8r7 | tilelang@main | 3390 | 3380-3403 | 1.00 | 12689 | 12685-12694 | 1.00 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 1384 | 1378-1390 | 0.41 | - | - | - |
| single-65536-hca-cp8r7 | tilelang@cute | 3436 | 3404-3463 | 1.00 | 12697 | 12689-12700 | 1.00 |
| single-65536-hca-cp8r7 | cute@cute | 2709 | 2698-2728 | 0.79 | 11951 | 11948-11966 | 0.94 |
| single-65536-hca-cp8r7 | cute_ws@cute | 1271 | 1267-1279 | 0.37 | 10482 | 10472-10490 | 0.83 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 1408 | 1398-1416 | 0.41 | - | - | - |
| single-65536-sliding-cp1 | tilelang@main | 9650 | 9637-9666 | 1.00 | 30146 | 30143-30155 | 1.00 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4122 | 4112-4130 | 0.43 | - | - | - |
| single-65536-sliding-cp1 | tilelang@cute | 9686 | 9680-9728 | 1.00 | 30171 | 30144-30180 | 1.00 |
| single-65536-sliding-cp1 | cute@cute | 8434 | 8424-8592 | 0.87 | 28883 | 28874-28896 | 0.96 |
| single-65536-sliding-cp1 | cute_ws@cute | 3217 | 3199-3237 | 0.33 | 23639 | 23635-23650 | 0.78 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4157 | 4142-4211 | 0.43 | - | - | - |
| single-65536-sliding-cp8r0 | tilelang@main | 1871 | 1858-1899 | 1.00 | 4658 | 4655-4673 | 1.00 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 661.3 | 655.0-663.8 | 0.35 | - | - | - |
| single-65536-sliding-cp8r0 | tilelang@cute | 1859 | 1851-1864 | 1.00 | 4693 | 4684-4695 | 1.00 |
| single-65536-sliding-cp8r0 | cute@cute | 1191 | 1187-1197 | 0.64 | 3975 | 3971-3976 | 0.85 |
| single-65536-sliding-cp8r0 | cute_ws@cute | 462.5 | 460.2-465.0 | 0.25 | 3217 | 3208-3226 | 0.69 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 671.6 | 666.4-672.7 | 0.36 | - | - | - |
| single-65536-sliding-cp8r4 | tilelang@main | 1843 | 1830-1855 | 1.00 | 4694 | 4692-4707 | 1.00 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 660.7 | 659.4-664.0 | 0.36 | - | - | - |
| single-65536-sliding-cp8r4 | tilelang@cute | 1865 | 1860-1894 | 1.00 | 4730 | 4725-4745 | 1.00 |
| single-65536-sliding-cp8r4 | cute@cute | 1191 | 1189-1194 | 0.64 | 4025 | 4020-4028 | 0.85 |
| single-65536-sliding-cp8r4 | cute_ws@cute | 463.2 | 460.9-465.1 | 0.25 | 3262 | 3253-3263 | 0.69 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 669.2 | 667.6-672.9 | 0.36 | - | - | - |
| single-65536-sliding-cp8r7 | tilelang@main | 1852 | 1844-1863 | 1.00 | 4716 | 4715-4727 | 1.00 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 662.4 | 658.8-665.8 | 0.36 | - | - | - |
| single-65536-sliding-cp8r7 | tilelang@cute | 1858 | 1850-1922 | 1.00 | 4720 | 4715-4724 | 1.00 |
| single-65536-sliding-cp8r7 | cute@cute | 1193 | 1189-1199 | 0.64 | 4015 | 4013-4019 | 0.85 |
| single-65536-sliding-cp8r7 | cute_ws@cute | 465.4 | 462.8-466.1 | 0.25 | 3277 | 3263-3291 | 0.69 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 679.3 | 665.8-687.4 | 0.37 | - | - | - |
| short-65536-csa-cp1 | tilelang@main | 16242 | 16220-16289 | 1.00 | 63170 | 63160-63183 | 1.00 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 8146 | 8080-8320 | 0.50 | - | - | - |
| short-65536-csa-cp1 | tilelang@cute | 16233 | 16183-16292 | 1.00 | 63182 | 63158-63189 | 1.00 |
| short-65536-csa-cp1 | cute@cute | 14920 | 14819-15065 | 0.92 | 61761 | 61754-61777 | 0.98 |
| short-65536-csa-cp1 | cute_ws@cute | 7586 | 7291-7758 | 0.47 | 54079 | 54075-54103 | 0.86 |
| short-65536-csa-cp1 | flashmla_fwd_ref@cute | 8333 | 8273-8418 | 0.51 | - | - | - |
| short-65536-csa-cp8r0 | tilelang@main | 2287 | 2281-2291 | 1.00 | 6780 | 6771-6782 | 1.00 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 907.8 | 902.6-914.0 | 0.40 | - | - | - |
| short-65536-csa-cp8r0 | tilelang@cute | 2303 | 2299-2313 | 1.00 | 6790 | 6783-6819 | 1.00 |
| short-65536-csa-cp8r0 | cute@cute | 1624 | 1623-1628 | 0.71 | 6092 | 6085-6100 | 0.90 |
| short-65536-csa-cp8r0 | cute_ws@cute | 772.0 | 768.9-773.0 | 0.34 | 5195 | 5188-5201 | 0.77 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 916.8 | 915.7-922.9 | 0.40 | - | - | - |
| short-65536-csa-cp8r4 | tilelang@main | 2862 | 2860-2870 | 1.00 | 10076 | 10047-10086 | 1.00 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1170 | 1169-1173 | 0.41 | - | - | - |
| short-65536-csa-cp8r4 | tilelang@cute | 2891 | 2886-2922 | 1.00 | 10124 | 10116-10129 | 1.00 |
| short-65536-csa-cp8r4 | cute@cute | 2213 | 2208-2245 | 0.77 | 9357 | 9348-9368 | 0.92 |
| short-65536-csa-cp8r4 | cute_ws@cute | 1049 | 1047-1058 | 0.36 | 8113 | 8102-8115 | 0.80 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 1180 | 1177-1186 | 0.41 | - | - | - |
| short-65536-csa-cp8r7 | tilelang@main | 2664 | 2649-2673 | 1.00 | 8942 | 8934-8970 | 1.00 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1088 | 1087-1093 | 0.41 | - | - | - |
| short-65536-csa-cp8r7 | tilelang@cute | 2698 | 2688-2707 | 1.00 | 8944 | 8940-8957 | 1.00 |
| short-65536-csa-cp8r7 | cute@cute | 2023 | 2006-2038 | 0.75 | 8208 | 8195-8221 | 0.92 |
| short-65536-csa-cp8r7 | cute_ws@cute | 959.3 | 958.0-966.1 | 0.36 | 7117 | 7108-7124 | 0.80 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 1105 | 1103-1109 | 0.41 | - | - | - |
| short-65536-hca-cp1 | tilelang@main | 10922 | 10906-10936 | 1.00 | 33532 | 33528-33548 | 1.00 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5378 | 5372-5387 | 0.49 | - | - | - |
| short-65536-hca-cp1 | tilelang@cute | 11008 | 10980-11046 | 1.00 | 33570 | 33556-33582 | 1.00 |
| short-65536-hca-cp1 | cute@cute | 9660 | 9647-9729 | 0.88 | 32249 | 32243-32256 | 0.96 |
| short-65536-hca-cp1 | cute_ws@cute | 4748 | 4739-4777 | 0.43 | 27289 | 27272-27296 | 0.81 |
| short-65536-hca-cp1 | flashmla_fwd_ref@cute | 5408 | 5399-5412 | 0.49 | - | - | - |
| short-65536-hca-cp8r0 | tilelang@main | 2020 | 2012-2043 | 1.00 | 5027 | 5026-5032 | 1.00 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 811.2 | 808.9-814.9 | 0.40 | - | - | - |
| short-65536-hca-cp8r0 | tilelang@cute | 2047 | 2034-2100 | 1.00 | 5056 | 5051-5056 | 1.00 |
| short-65536-hca-cp8r0 | cute@cute | 1353 | 1348-1358 | 0.66 | 4346 | 4340-4349 | 0.86 |
| short-65536-hca-cp8r0 | cute_ws@cute | 647.1 | 643.3-648.9 | 0.32 | 3580 | 3577-3586 | 0.71 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 821.6 | 818.7-843.2 | 0.40 | - | - | - |
| short-65536-hca-cp8r4 | tilelang@main | 2054 | 2049-2060 | 1.00 | 5198 | 5193-5210 | 1.00 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 832.4 | 828.5-836.2 | 0.41 | - | - | - |
| short-65536-hca-cp8r4 | tilelang@cute | 2062 | 2049-2072 | 1.00 | 5248 | 5236-5253 | 1.00 |
| short-65536-hca-cp8r4 | cute@cute | 1385 | 1384-1389 | 0.67 | 4523 | 4522-4529 | 0.86 |
| short-65536-hca-cp8r4 | cute_ws@cute | 663.8 | 661.9-668.1 | 0.32 | 3744 | 3741-3746 | 0.71 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 835.0 | 831.8-846.0 | 0.40 | - | - | - |
| short-65536-hca-cp8r7 | tilelang@main | 2040 | 2035-2045 | 1.00 | 5167 | 5154-5172 | 1.00 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 819.8 | 815.1-823.4 | 0.40 | - | - | - |
| short-65536-hca-cp8r7 | tilelang@cute | 2064 | 2059-2072 | 1.00 | 5158 | 5153-5162 | 1.00 |
| short-65536-hca-cp8r7 | cute@cute | 1379 | 1376-1388 | 0.67 | 4453 | 4450-4461 | 0.86 |
| short-65536-hca-cp8r7 | cute_ws@cute | 656.4 | 654.9-661.8 | 0.32 | 3676 | 3673-3678 | 0.71 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 833.5 | 830.4-838.8 | 0.40 | - | - | - |
| short-65536-sliding-cp1 | tilelang@main | 9612 | 9605-9628 | 1.00 | 29641 | 29634-29651 | 1.00 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4105 | 4097-4107 | 0.43 | - | - | - |
| short-65536-sliding-cp1 | tilelang@cute | 9682 | 9658-9689 | 1.00 | 29716 | 29692-29761 | 1.00 |
| short-65536-sliding-cp1 | cute@cute | 8420 | 8404-8620 | 0.87 | 28408 | 28393-28433 | 0.96 |
| short-65536-sliding-cp1 | cute_ws@cute | 3225 | 3206-3232 | 0.33 | 23196 | 23185-23199 | 0.78 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4192 | 4155-4249 | 0.43 | - | - | - |
| short-65536-sliding-cp8r0 | tilelang@main | 1863 | 1854-1872 | 1.00 | 4557 | 4556-4558 | 1.00 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 665.9 | 661.3-668.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r0 | tilelang@cute | 1837 | 1834-1851 | 1.00 | 4574 | 4572-4588 | 1.00 |
| short-65536-sliding-cp8r0 | cute@cute | 1183 | 1172-1185 | 0.64 | 3865 | 3862-3878 | 0.84 |
| short-65536-sliding-cp8r0 | cute_ws@cute | 465.6 | 462.5-466.3 | 0.25 | 3115 | 3108-3124 | 0.68 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 669.2 | 663.4-674.3 | 0.36 | - | - | - |
| short-65536-sliding-cp8r4 | tilelang@main | 1838 | 1829-1848 | 1.00 | 4624 | 4622-4646 | 1.00 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 660.4 | 656.3-663.9 | 0.36 | - | - | - |
| short-65536-sliding-cp8r4 | tilelang@cute | 1845 | 1834-1856 | 1.00 | 4686 | 4675-4687 | 1.00 |
| short-65536-sliding-cp8r4 | cute@cute | 1183 | 1181-1187 | 0.64 | 3966 | 3963-3971 | 0.85 |
| short-65536-sliding-cp8r4 | cute_ws@cute | 461.6 | 460.5-461.7 | 0.25 | 3211 | 3209-3225 | 0.69 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 669.1 | 668.0-669.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r7 | tilelang@main | 1841 | 1834-1848 | 1.00 | 4633 | 4620-4639 | 1.00 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 662.2 | 661.1-666.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r7 | tilelang@cute | 1847 | 1837-1859 | 1.00 | 4635 | 4632-4637 | 1.00 |
| short-65536-sliding-cp8r7 | cute@cute | 1185 | 1185-1190 | 0.64 | 3918 | 3918-3923 | 0.85 |
| short-65536-sliding-cp8r7 | cute_ws@cute | 464.0 | 463.8-466.0 | 0.25 | 3166 | 3162-3169 | 0.68 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 674.4 | 670.6-676.4 | 0.37 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 17373 | 17342-17419 | 1.00 | 69696 | 69684-69701 | 1.00 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8820 | 8665-9037 | 0.51 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@cute | 17398 | 17260-17430 | 1.00 | 69675 | 69636-69683 | 1.00 |
| heavy-65536-csa-cp1 | cute@cute | 16058 | 15937-16098 | 0.92 | 68243 | 68235-68261 | 0.98 |
| heavy-65536-csa-cp1 | cute_ws@cute | 8450 | 8222-8489 | 0.49 | 60056 | 60050-60081 | 0.86 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@cute | 9006 | 8956-9049 | 0.52 | - | - | - |
| heavy-65536-csa-cp8r0 | tilelang@main | 2238 | 2232-2242 | 1.00 | 6420 | 6417-6435 | 1.00 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 900.9 | 888.9-902.5 | 0.40 | - | - | - |
| heavy-65536-csa-cp8r0 | tilelang@cute | 2237 | 2230-2245 | 1.00 | 6447 | 6445-6455 | 1.00 |
| heavy-65536-csa-cp8r0 | cute@cute | 1579 | 1577-1579 | 0.71 | 5748 | 5746-5756 | 0.89 |
| heavy-65536-csa-cp8r0 | cute_ws@cute | 764.6 | 761.8-767.0 | 0.34 | 4890 | 4887-4892 | 0.76 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 909.5 | 904.5-911.0 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r4 | tilelang@main | 3403 | 3379-3442 | 1.00 | 12973 | 12936-12999 | 1.00 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1402 | 1397-1405 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r4 | tilelang@cute | 3402 | 3393-3439 | 1.00 | 13006 | 13001-13012 | 1.00 |
| heavy-65536-csa-cp8r4 | cute@cute | 2708 | 2693-2880 | 0.80 | 12266 | 12257-12274 | 0.94 |
| heavy-65536-csa-cp8r4 | cute_ws@cute | 1289 | 1286-1360 | 0.38 | 10807 | 10806-10815 | 0.83 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 1419 | 1408-1426 | 0.42 | - | - | - |
| heavy-65536-csa-cp8r7 | tilelang@main | 2550 | 2543-2560 | 1.00 | 8299 | 8297-8303 | 1.00 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1048 | 1046-1052 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r7 | tilelang@cute | 2592 | 2584-2603 | 1.00 | 8316 | 8313-8327 | 1.00 |
| heavy-65536-csa-cp8r7 | cute@cute | 1895 | 1894-1897 | 0.73 | 7600 | 7594-7611 | 0.91 |
| heavy-65536-csa-cp8r7 | cute_ws@cute | 915.3 | 913.4-918.9 | 0.35 | 6573 | 6568-6582 | 0.79 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 1061 | 1060-1066 | 0.41 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 11714 | 11677-11736 | 1.00 | 38272 | 38241-38292 | 1.00 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5682 | 5674-5693 | 0.49 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@cute | 11741 | 11740-11750 | 1.00 | 38300 | 38294-38305 | 1.00 |
| heavy-65536-hca-cp1 | cute@cute | 10580 | 10460-10612 | 0.90 | 36972 | 36970-36993 | 0.97 |
| heavy-65536-hca-cp1 | cute_ws@cute | 4987 | 4955-4997 | 0.42 | 31447 | 31441-31464 | 0.82 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@cute | 5780 | 5723-5831 | 0.49 | - | - | - |
| heavy-65536-hca-cp8r0 | tilelang@main | 1991 | 1967-1996 | 1.00 | 4821 | 4813-4828 | 1.00 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 777.0 | 775.5-781.9 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r0 | tilelang@cute | 1961 | 1957-1973 | 1.00 | 4834 | 4822-4837 | 1.00 |
| heavy-65536-hca-cp8r0 | cute@cute | 1303 | 1302-1306 | 0.66 | 4124 | 4120-4127 | 0.85 |
| heavy-65536-hca-cp8r0 | cute_ws@cute | 607.6 | 606.5-610.8 | 0.31 | 3365 | 3364-3368 | 0.70 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 787.2 | 777.6-795.1 | 0.40 | - | - | - |
| heavy-65536-hca-cp8r4 | tilelang@main | 2378 | 2363-2384 | 1.00 | 7050 | 7045-7053 | 1.00 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 928.0 | 924.2-932.5 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r4 | tilelang@cute | 2361 | 2353-2374 | 1.00 | 7101 | 7098-7106 | 1.00 |
| heavy-65536-hca-cp8r4 | cute@cute | 1685 | 1683-1689 | 0.71 | 6371 | 6368-6373 | 0.90 |
| heavy-65536-hca-cp8r4 | cute_ws@cute | 772.1 | 771.6-772.5 | 0.33 | 5394 | 5381-5395 | 0.76 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 933.4 | 932.8-933.7 | 0.40 | - | - | - |
| heavy-65536-hca-cp8r7 | tilelang@main | 2006 | 2002-2013 | 1.00 | 5010 | 5008-5012 | 1.00 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 799.1 | 797.0-808.2 | 0.40 | - | - | - |
| heavy-65536-hca-cp8r7 | tilelang@cute | 2023 | 2017-2036 | 1.00 | 5013 | 5007-5020 | 1.00 |
| heavy-65536-hca-cp8r7 | cute@cute | 1351 | 1347-1353 | 0.67 | 4302 | 4298-4306 | 0.86 |
| heavy-65536-hca-cp8r7 | cute_ws@cute | 641.4 | 639.1-648.0 | 0.32 | 3530 | 3529-3536 | 0.70 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 811.7 | 809.8-814.6 | 0.40 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 9616 | 9585-9660 | 1.00 | 29282 | 29271-29295 | 1.00 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4111 | 4108-4115 | 0.43 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@cute | 9608 | 9608-9613 | 1.00 | 29283 | 29275-29293 | 1.00 |
| heavy-65536-sliding-cp1 | cute@cute | 8381 | 8362-8570 | 0.87 | 27989 | 27982-27997 | 0.96 |
| heavy-65536-sliding-cp1 | cute_ws@cute | 3214 | 3195-3242 | 0.33 | 22835 | 22827-22838 | 0.78 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4128 | 4106-4197 | 0.43 | - | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@main | 1853 | 1835-1865 | 1.00 | 4405 | 4399-4411 | 1.00 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 669.7 | 664.8-674.5 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@cute | 1821 | 1818-1835 | 1.00 | 4423 | 4419-4439 | 1.00 |
| heavy-65536-sliding-cp8r0 | cute@cute | 1170 | 1168-1171 | 0.64 | 3722 | 3721-3726 | 0.84 |
| heavy-65536-sliding-cp8r0 | cute_ws@cute | 468.4 | 466.0-472.9 | 0.26 | 2986 | 2984-2991 | 0.68 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 675.9 | 671.1-684.6 | 0.37 | - | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@main | 1840 | 1825-1842 | 1.00 | 4678 | 4674-4682 | 1.00 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 655.9 | 655.0-657.4 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@cute | 1881 | 1871-1890 | 1.00 | 4787 | 4724-4836 | 1.00 |
| heavy-65536-sliding-cp8r4 | cute@cute | 1196 | 1190-1197 | 0.64 | 4031 | 4017-4048 | 0.84 |
| heavy-65536-sliding-cp8r4 | cute_ws@cute | 463.8 | 461.4-466.1 | 0.25 | 3353 | 3267-3377 | 0.70 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 669.0 | 664.2-672.9 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@main | 1824 | 1817-1827 | 1.00 | 4524 | 4521-4533 | 1.00 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 665.7 | 659.8-668.6 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@cute | 1862 | 1858-1893 | 1.00 | 4536 | 4530-4538 | 1.00 |
| heavy-65536-sliding-cp8r7 | cute@cute | 1186 | 1184-1187 | 0.64 | 3832 | 3825-3833 | 0.84 |
| heavy-65536-sliding-cp8r7 | cute_ws@cute | 467.2 | 465.6-469.3 | 0.25 | 3084 | 3077-3089 | 0.68 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 677.6 | 672.6-677.8 | 0.36 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 10884 | 10872-10885 | 1.00 | 31800 | 31799-31808 | 1.00 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5659 | 5656-5664 | 0.52 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@cute | 10906 | 10896-10912 | 1.00 | 31847 | 31838-31870 | 1.00 |
| tiny-65536-csa-cp1 | cute@cute | 9773 | 9763-9775 | 0.90 | 30609 | 30590-30615 | 0.96 |
| tiny-65536-csa-cp1 | cute_ws@cute | 4924 | 4919-4934 | 0.45 | 25783 | 25779-25792 | 0.81 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@cute | 5663 | 5656-5664 | 0.52 | - | - | - |
| tiny-65536-csa-cp8r0 | tilelang@main | 2004 | 1988-2021 | 1.00 | 4891 | 4874-4905 | 1.00 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 809.2 | 808.8-818.0 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r0 | tilelang@cute | 2001 | 1996-2020 | 1.00 | 4896 | 4888-4914 | 1.00 |
| tiny-65536-csa-cp8r0 | cute@cute | 1346 | 1344-1350 | 0.67 | 4193 | 4191-4197 | 0.86 |
| tiny-65536-csa-cp8r0 | cute_ws@cute | 680.6 | 680.5-681.0 | 0.34 | 3482 | 3480-3483 | 0.71 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 823.8 | 820.5-826.0 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r4 | tilelang@main | 2008 | 1980-2018 | 1.00 | 4867 | 4862-4872 | 1.00 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 823.2 | 820.0-828.9 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r4 | tilelang@cute | 2002 | 1992-2012 | 1.00 | 4912 | 4909-4918 | 1.00 |
| tiny-65536-csa-cp8r4 | cute@cute | 1347 | 1343-1348 | 0.67 | 4210 | 4208-4211 | 0.86 |
| tiny-65536-csa-cp8r4 | cute_ws@cute | 685.1 | 682.4-690.2 | 0.34 | 3491 | 3484-3494 | 0.71 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 828.7 | 825.2-833.6 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r7 | tilelang@main | 2064 | 2023-2100 | 1.00 | 4878 | 4870-4892 | 1.00 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 831.3 | 809.5-891.6 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r7 | tilelang@cute | 2020 | 2013-2032 | 1.00 | 4912 | 4900-4923 | 1.00 |
| tiny-65536-csa-cp8r7 | cute@cute | 1351 | 1346-1352 | 0.67 | 4211 | 4209-4215 | 0.86 |
| tiny-65536-csa-cp8r7 | cute_ws@cute | 680.5 | 678.1-681.8 | 0.34 | 3489 | 3485-3491 | 0.71 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 823.6 | 818.8-827.4 | 0.41 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 8689 | 8676-8718 | 1.00 | 22022 | 21996-22029 | 1.00 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4173 | 4168-4184 | 0.48 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@cute | 8687 | 8668-8697 | 1.00 | 22031 | 22020-22041 | 1.00 |
| tiny-65536-hca-cp1 | cute@cute | 7441 | 7423-7448 | 0.86 | 20739 | 20735-20753 | 0.94 |
| tiny-65536-hca-cp1 | cute_ws@cute | 3668 | 3659-3672 | 0.42 | 16933 | 16925-16937 | 0.77 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@cute | 4162 | 4158-4172 | 0.48 | - | - | - |
| tiny-65536-hca-cp8r0 | tilelang@main | 1706 | 1702-1728 | 1.00 | 3628 | 3625-3630 | 1.00 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 657.9 | 656.0-673.7 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r0 | tilelang@cute | 1722 | 1717-1727 | 1.00 | 3617 | 3613-3630 | 1.00 |
| tiny-65536-hca-cp8r0 | cute@cute | 1064 | 1058-1068 | 0.62 | 2931 | 2926-2933 | 0.81 |
| tiny-65536-hca-cp8r0 | cute_ws@cute | 514.1 | 505.4-519.5 | 0.30 | 2340 | 2337-2348 | 0.65 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 671.5 | 669.8-674.6 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r4 | tilelang@main | 1732 | 1714-1738 | 1.00 | 3608 | 3597-3618 | 1.00 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 662.3 | 656.5-663.6 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r4 | tilelang@cute | 1752 | 1746-1768 | 1.00 | 3652 | 3644-3659 | 1.00 |
| tiny-65536-hca-cp8r4 | cute@cute | 1072 | 1069-1076 | 0.61 | 2947 | 2942-2949 | 0.81 |
| tiny-65536-hca-cp8r4 | cute_ws@cute | 507.9 | 494.9-520.0 | 0.29 | 2355 | 2349-2368 | 0.64 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 664.6 | 662.9-665.1 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r7 | tilelang@main | 1716 | 1711-1718 | 1.00 | 3631 | 3624-3641 | 1.00 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 666.2 | 661.7-670.2 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r7 | tilelang@cute | 1723 | 1715-1746 | 1.00 | 3632 | 3628-3653 | 1.00 |
| tiny-65536-hca-cp8r7 | cute@cute | 1058 | 1056-1075 | 0.61 | 2931 | 2928-2936 | 0.81 |
| tiny-65536-hca-cp8r7 | cute_ws@cute | 514.7 | 501.9-518.9 | 0.30 | 2341 | 2340-2346 | 0.64 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 668.4 | 660.8-670.4 | 0.39 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 8717 | 8700-8727 | 1.00 | 22036 | 22028-22065 | 1.00 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4178 | 4164-4186 | 0.48 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@cute | 8692 | 8664-8710 | 1.00 | 22029 | 22015-22032 | 1.00 |
| tiny-65536-sliding-cp1 | cute@cute | 7436 | 7430-7452 | 0.86 | 20739 | 20736-20750 | 0.94 |
| tiny-65536-sliding-cp1 | cute_ws@cute | 3674 | 3661-3688 | 0.42 | 16935 | 16924-16942 | 0.77 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4162 | 4156-4179 | 0.48 | - | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@main | 1745 | 1740-1764 | 1.00 | 3631 | 3625-3633 | 1.00 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 668.3 | 661.3-672.0 | 0.38 | - | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@cute | 1724 | 1721-1726 | 1.00 | 3637 | 3631-3641 | 1.00 |
| tiny-65536-sliding-cp8r0 | cute@cute | 1063 | 1060-1070 | 0.62 | 2935 | 2934-2939 | 0.81 |
| tiny-65536-sliding-cp8r0 | cute_ws@cute | 519.0 | 514.1-521.8 | 0.30 | 2347 | 2345-2351 | 0.65 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 670.6 | 664.9-672.3 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@main | 1716 | 1698-1722 | 1.00 | 3613 | 3604-3624 | 1.00 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 662.2 | 654.3-664.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@cute | 1728 | 1720-1763 | 1.00 | 3643 | 3638-3682 | 1.00 |
| tiny-65536-sliding-cp8r4 | cute@cute | 1070 | 1067-1072 | 0.62 | 2940 | 2935-2943 | 0.81 |
| tiny-65536-sliding-cp8r4 | cute_ws@cute | 512.0 | 496.2-514.8 | 0.30 | 2340 | 2339-2347 | 0.64 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 668.0 | 666.0-671.1 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@main | 1705 | 1695-1721 | 1.00 | 3629 | 3621-3632 | 1.00 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 665.0 | 659.9-666.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@cute | 1722 | 1719-1722 | 1.00 | 3625 | 3619-3631 | 1.00 |
| tiny-65536-sliding-cp8r7 | cute@cute | 1065 | 1062-1069 | 0.62 | 2927 | 2926-2931 | 0.81 |
| tiny-65536-sliding-cp8r7 | cute_ws@cute | 510.8 | 500.2-517.9 | 0.30 | 2346 | 2342-2348 | 0.65 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 671.0 | 668.5-675.3 | 0.39 | - | - | - |

GPU busy time per call in µs from profiler traces (lower is better); `host` is op-boundary minus GPU
busy time (launch overhead and gaps); `/TL` divides GPU busy time by tilelang's; `peak MiB` is the
allocation above the inputs during one call, forward+backward where the backend has it, else forward.

| item | backend | fwd gpu µs | fwd host | fwd /TL | f+b gpu µs | f+b host | f+b /TL | peak MiB |
|---|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 539.5 | 657.7 | 1.00 | 2161 | 1025 | 1.00 | 265 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 284.0 | 100.0 | 0.53 | - | - | - | 139 |
| single-2048-csa-cp1 | tilelang@cute | 539.5 | 673.0 | 1.00 | 2160 | 1102 | 1.00 | 265 |
| single-2048-csa-cp1 | cute@cute | 510.2 | 98.6 | 0.95 | 2133 | 494.2 | 0.99 | 265 |
| single-2048-csa-cp1 | cute_ws@cute | 255.6 | 47.7 | 0.47 | 1881 | 587.8 | 0.87 | 265 |
| single-2048-csa-cp1 | flashmla_fwd_ref@cute | 284.7 | 105.0 | 0.53 | - | - | - | 139 |
| single-2048-csa-cp8r0 | tilelang@main | 60.1 | 673.4 | 1.00 | 221.7 | 1780 | 1.00 | 40 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.0 | 138.4 | 0.73 | - | - | - | 17 |
| single-2048-csa-cp8r0 | tilelang@cute | 60.6 | 679.6 | 1.00 | 222.5 | 1846 | 1.00 | 40 |
| single-2048-csa-cp8r0 | cute@cute | 58.0 | 119.8 | 0.96 | 219.1 | 1201 | 0.98 | 40 |
| single-2048-csa-cp8r0 | cute_ws@cute | 36.5 | 49.7 | 0.60 | 198.0 | 1075 | 0.89 | 40 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 44.4 | 145.4 | 0.73 | - | - | - | 17 |
| single-2048-csa-cp8r4 | tilelang@main | 84.4 | 679.0 | 1.00 | 356.6 | 1672 | 1.00 | 40 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 57.8 | 139.9 | 0.68 | - | - | - | 17 |
| single-2048-csa-cp8r4 | tilelang@cute | 83.8 | 697.6 | 1.00 | 357.0 | 1736 | 1.00 | 40 |
| single-2048-csa-cp8r4 | cute@cute | 81.3 | 121.6 | 0.97 | 353.5 | 1106 | 0.99 | 40 |
| single-2048-csa-cp8r4 | cute_ws@cute | 49.2 | 49.6 | 0.59 | 322.4 | 969.7 | 0.90 | 40 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 57.5 | 151.4 | 0.69 | - | - | - | 17 |
| single-2048-csa-cp8r7 | tilelang@main | 101.7 | 672.6 | 1.00 | 455.4 | 1569 | 1.00 | 40 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 64.1 | 138.7 | 0.63 | - | - | - | 17 |
| single-2048-csa-cp8r7 | tilelang@cute | 101.5 | 684.1 | 1.00 | 454.1 | 1633 | 1.00 | 40 |
| single-2048-csa-cp8r7 | cute@cute | 98.5 | 119.3 | 0.97 | 450.3 | 988.8 | 0.99 | 40 |
| single-2048-csa-cp8r7 | cute_ws@cute | 55.5 | 49.2 | 0.55 | 409.4 | 885.9 | 0.90 | 40 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 63.7 | 149.3 | 0.63 | - | - | - | 17 |
| single-2048-hca-cp1 | tilelang@main | 362.9 | 722.3 | 1.00 | 1166 | 1338 | 1.00 | 265 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 199.3 | 141.7 | 0.55 | - | - | - | 133 |
| single-2048-hca-cp1 | tilelang@cute | 362.8 | 730.3 | 1.00 | 1166 | 1398 | 1.00 | 265 |
| single-2048-hca-cp1 | cute@cute | 336.6 | 147.8 | 0.93 | 1140 | 770.9 | 0.98 | 265 |
| single-2048-hca-cp1 | cute_ws@cute | 169.4 | 54.1 | 0.47 | 977.3 | 740.6 | 0.84 | 265 |
| single-2048-hca-cp1 | flashmla_fwd_ref@cute | 201.1 | 150.6 | 0.55 | - | - | - | 133 |
| single-2048-hca-cp8r0 | tilelang@main | 57.6 | 706.3 | 1.00 | 206.3 | 1945 | 1.00 | 38 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 41.6 | 147.1 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r0 | tilelang@cute | 57.4 | 725.5 | 1.00 | 204.8 | 1970 | 1.00 | 38 |
| single-2048-hca-cp8r0 | cute@cute | 55.1 | 148.3 | 0.96 | 203.1 | 1336 | 0.99 | 38 |
| single-2048-hca-cp8r0 | cute_ws@cute | 26.0 | 51.0 | 0.45 | 175.3 | 1160 | 0.86 | 38 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 41.5 | 156.8 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r4 | tilelang@main | 61.4 | 731.7 | 1.00 | 219.3 | 1938 | 1.00 | 38 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 44.9 | 149.0 | 0.73 | - | - | - | 17 |
| single-2048-hca-cp8r4 | tilelang@cute | 61.8 | 729.7 | 1.00 | 220.3 | 1995 | 1.00 | 38 |
| single-2048-hca-cp8r4 | cute@cute | 59.8 | 149.1 | 0.97 | 216.4 | 1329 | 0.98 | 38 |
| single-2048-hca-cp8r4 | cute_ws@cute | 30.1 | 52.1 | 0.49 | 187.4 | 1144 | 0.85 | 38 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 44.5 | 156.0 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r7 | tilelang@main | 62.1 | 744.8 | 1.00 | 219.6 | 1958 | 1.00 | 38 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.2 | 148.3 | 0.73 | - | - | - | 17 |
| single-2048-hca-cp8r7 | tilelang@cute | 62.1 | 731.3 | 1.00 | 220.1 | 1987 | 1.00 | 38 |
| single-2048-hca-cp8r7 | cute@cute | 60.1 | 145.3 | 0.97 | 217.0 | 1322 | 0.99 | 38 |
| single-2048-hca-cp8r7 | cute_ws@cute | 30.0 | 50.6 | 0.48 | 188.8 | 1137 | 0.86 | 38 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 44.8 | 153.7 | 0.72 | - | - | - | 17 |
| single-2048-sliding-cp1 | tilelang@main | 312.4 | 703.5 | 1.00 | 1033 | 1255 | 1.00 | 264 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 150.8 | 147.6 | 0.48 | - | - | - | 131 |
| single-2048-sliding-cp1 | tilelang@cute | 312.0 | 679.9 | 1.00 | 1032 | 1329 | 1.00 | 264 |
| single-2048-sliding-cp1 | cute@cute | 290.4 | 124.9 | 0.93 | 1008 | 758.6 | 0.98 | 264 |
| single-2048-sliding-cp1 | cute_ws@cute | 119.3 | 53.2 | 0.38 | 839.8 | 736.0 | 0.81 | 264 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@cute | 149.3 | 153.8 | 0.48 | - | - | - | 131 |
| single-2048-sliding-cp8r0 | tilelang@main | 51.4 | 681.1 | 1.00 | 190.3 | 1857 | 1.00 | 38 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.6 | 144.1 | 0.69 | - | - | - | 16 |
| single-2048-sliding-cp8r0 | tilelang@cute | 51.8 | 694.1 | 1.00 | 190.2 | 1880 | 1.00 | 38 |
| single-2048-sliding-cp8r0 | cute@cute | 49.8 | 121.4 | 0.96 | 188.4 | 1245 | 0.99 | 38 |
| single-2048-sliding-cp8r0 | cute_ws@cute | 22.3 | 51.8 | 0.43 | 161.9 | 1110 | 0.85 | 38 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 35.6 | 150.7 | 0.69 | - | - | - | 16 |
| single-2048-sliding-cp8r4 | tilelang@main | 53.4 | 686.1 | 1.00 | 195.9 | 1822 | 1.00 | 38 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.5 | 143.5 | 0.66 | - | - | - | 16 |
| single-2048-sliding-cp8r4 | tilelang@cute | 53.1 | 680.4 | 1.00 | 195.6 | 1915 | 1.00 | 38 |
| single-2048-sliding-cp8r4 | cute@cute | 51.2 | 123.9 | 0.96 | 194.0 | 1267 | 0.99 | 38 |
| single-2048-sliding-cp8r4 | cute_ws@cute | 22.4 | 51.7 | 0.42 | 166.2 | 1117 | 0.85 | 38 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 35.9 | 152.5 | 0.68 | - | - | - | 16 |
| single-2048-sliding-cp8r7 | tilelang@main | 53.2 | 686.3 | 1.00 | 195.3 | 1838 | 1.00 | 38 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 36.2 | 144.7 | 0.68 | - | - | - | 16 |
| single-2048-sliding-cp8r7 | tilelang@cute | 53.3 | 676.8 | 1.00 | 195.9 | 1916 | 1.00 | 38 |
| single-2048-sliding-cp8r7 | cute@cute | 51.4 | 124.1 | 0.96 | 193.2 | 1299 | 0.99 | 38 |
| single-2048-sliding-cp8r7 | cute_ws@cute | 22.6 | 51.3 | 0.42 | 165.7 | 1124 | 0.85 | 38 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 36.2 | 150.9 | 0.68 | - | - | - | 16 |
| short-2048-csa-cp1 | tilelang@main | 442.8 | 688.9 | 1.00 | 1638 | 1118 | 1.00 | 265 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 234.8 | 133.2 | 0.53 | - | - | - | 139 |
| short-2048-csa-cp1 | tilelang@cute | 442.2 | 682.5 | 1.00 | 1638 | 1203 | 1.00 | 265 |
| short-2048-csa-cp1 | cute@cute | 415.6 | 119.4 | 0.94 | 1614 | 575.6 | 0.99 | 265 |
| short-2048-csa-cp1 | cute_ws@cute | 210.2 | 50.2 | 0.48 | 1413 | 644.8 | 0.86 | 265 |
| short-2048-csa-cp1 | flashmla_fwd_ref@cute | 233.8 | 141.7 | 0.53 | - | - | - | 139 |
| short-2048-csa-cp8r0 | tilelang@main | 60.1 | 693.0 | 1.00 | 221.8 | 1801 | 1.00 | 40 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.4 | 143.4 | 0.74 | - | - | - | 17 |
| short-2048-csa-cp8r0 | tilelang@cute | 60.6 | 681.5 | 1.00 | 222.3 | 1830 | 1.00 | 40 |
| short-2048-csa-cp8r0 | cute@cute | 58.2 | 120.9 | 0.96 | 219.5 | 1212 | 0.99 | 40 |
| short-2048-csa-cp8r0 | cute_ws@cute | 36.2 | 50.2 | 0.60 | 198.9 | 1065 | 0.89 | 40 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 44.4 | 148.9 | 0.73 | - | - | - | 17 |
| short-2048-csa-cp8r4 | tilelang@main | 67.8 | 674.7 | 1.00 | 262.9 | 1770 | 1.00 | 40 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 48.9 | 138.8 | 0.72 | - | - | - | 17 |
| short-2048-csa-cp8r4 | tilelang@cute | 67.6 | 678.3 | 1.00 | 261.3 | 1829 | 1.00 | 40 |
| short-2048-csa-cp8r4 | cute@cute | 64.9 | 120.0 | 0.96 | 258.9 | 1170 | 0.99 | 40 |
| short-2048-csa-cp8r4 | cute_ws@cute | 41.4 | 49.2 | 0.61 | 236.4 | 1049 | 0.90 | 40 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 48.7 | 148.5 | 0.72 | - | - | - | 17 |
| short-2048-csa-cp8r7 | tilelang@main | 76.4 | 685.8 | 1.00 | 319.2 | 1718 | 1.00 | 40 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 50.9 | 140.6 | 0.67 | - | - | - | 17 |
| short-2048-csa-cp8r7 | tilelang@cute | 76.7 | 689.7 | 1.00 | 320.4 | 1737 | 1.00 | 40 |
| short-2048-csa-cp8r7 | cute@cute | 73.4 | 121.0 | 0.96 | 317.6 | 1116 | 0.99 | 40 |
| short-2048-csa-cp8r7 | cute_ws@cute | 42.7 | 51.1 | 0.56 | 287.3 | 1003 | 0.90 | 40 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 51.3 | 147.9 | 0.67 | - | - | - | 17 |
| short-2048-hca-cp1 | tilelang@main | 354.6 | 724.6 | 1.00 | 1143 | 1346 | 1.00 | 265 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 196.7 | 143.8 | 0.55 | - | - | - | 133 |
| short-2048-hca-cp1 | tilelang@cute | 353.2 | 726.6 | 1.00 | 1141 | 1383 | 1.00 | 265 |
| short-2048-hca-cp1 | cute@cute | 329.8 | 147.0 | 0.93 | 1119 | 765.1 | 0.98 | 265 |
| short-2048-hca-cp1 | cute_ws@cute | 167.0 | 53.3 | 0.47 | 959.8 | 720.5 | 0.84 | 265 |
| short-2048-hca-cp1 | flashmla_fwd_ref@cute | 194.9 | 150.8 | 0.55 | - | - | - | 133 |
| short-2048-hca-cp8r0 | tilelang@main | 57.2 | 730.2 | 1.00 | 205.4 | 1949 | 1.00 | 38 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 42.0 | 149.5 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r0 | tilelang@cute | 57.2 | 720.2 | 1.00 | 205.9 | 2120 | 1.00 | 38 |
| short-2048-hca-cp8r0 | cute@cute | 55.4 | 146.4 | 0.97 | 204.0 | 1532 | 0.99 | 38 |
| short-2048-hca-cp8r0 | cute_ws@cute | 26.0 | 52.3 | 0.45 | 175.6 | 1161 | 0.85 | 38 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 42.0 | 154.1 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r4 | tilelang@main | 58.6 | 736.4 | 1.00 | 211.8 | 1950 | 1.00 | 38 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 43.2 | 150.1 | 0.74 | - | - | - | 17 |
| short-2048-hca-cp8r4 | tilelang@cute | 58.5 | 722.9 | 1.00 | 213.0 | 2173 | 1.00 | 38 |
| short-2048-hca-cp8r4 | cute@cute | 56.4 | 143.6 | 0.96 | 210.5 | 1560 | 0.99 | 38 |
| short-2048-hca-cp8r4 | cute_ws@cute | 26.9 | 50.2 | 0.46 | 181.0 | 1396 | 0.85 | 38 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 43.5 | 153.6 | 0.74 | - | - | - | 17 |
| short-2048-hca-cp8r7 | tilelang@main | 62.0 | 730.7 | 1.00 | 219.8 | 1973 | 1.00 | 38 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.4 | 143.5 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r7 | tilelang@cute | 61.7 | 720.3 | 1.00 | 220.0 | 1944 | 1.00 | 38 |
| short-2048-hca-cp8r7 | cute@cute | 59.5 | 147.8 | 0.97 | 218.0 | 1327 | 0.99 | 38 |
| short-2048-hca-cp8r7 | cute_ws@cute | 30.0 | 50.0 | 0.49 | 187.3 | 1149 | 0.85 | 38 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 45.1 | 150.8 | 0.73 | - | - | - | 17 |
| short-2048-sliding-cp1 | tilelang@main | 307.0 | 695.7 | 1.00 | 1011 | 1319 | 1.00 | 264 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 147.7 | 145.6 | 0.48 | - | - | - | 131 |
| short-2048-sliding-cp1 | tilelang@cute | 308.4 | 681.7 | 1.00 | 1008 | 1312 | 1.00 | 264 |
| short-2048-sliding-cp1 | cute@cute | 284.9 | 122.6 | 0.92 | 984.8 | 720.5 | 0.98 | 264 |
| short-2048-sliding-cp1 | cute_ws@cute | 118.3 | 52.3 | 0.38 | 826.3 | 717.9 | 0.82 | 264 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@cute | 151.0 | 146.9 | 0.49 | - | - | - | 131 |
| short-2048-sliding-cp8r0 | tilelang@main | 51.5 | 695.3 | 1.00 | 190.0 | 1852 | 1.00 | 38 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 36.3 | 145.5 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r0 | tilelang@cute | 51.3 | 708.2 | 1.00 | 190.6 | 1866 | 1.00 | 38 |
| short-2048-sliding-cp8r0 | cute@cute | 49.5 | 122.3 | 0.97 | 188.7 | 1236 | 0.99 | 38 |
| short-2048-sliding-cp8r0 | cute_ws@cute | 22.0 | 52.1 | 0.43 | 162.5 | 1131 | 0.85 | 38 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 35.5 | 153.9 | 0.69 | - | - | - | 16 |
| short-2048-sliding-cp8r4 | tilelang@main | 51.5 | 705.5 | 1.00 | 191.7 | 1831 | 1.00 | 38 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 36.2 | 153.6 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r4 | tilelang@cute | 52.1 | 699.4 | 1.00 | 191.1 | 1918 | 1.00 | 38 |
| short-2048-sliding-cp8r4 | cute@cute | 49.5 | 125.8 | 0.95 | 189.8 | 1260 | 0.99 | 38 |
| short-2048-sliding-cp8r4 | cute_ws@cute | 22.3 | 52.6 | 0.43 | 163.7 | 1135 | 0.86 | 38 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 35.7 | 163.4 | 0.69 | - | - | - | 16 |
| short-2048-sliding-cp8r7 | tilelang@main | 53.4 | 683.0 | 1.00 | 196.2 | 1848 | 1.00 | 38 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.6 | 149.5 | 0.67 | - | - | - | 16 |
| short-2048-sliding-cp8r7 | tilelang@cute | 53.3 | 710.2 | 1.00 | 195.8 | 1898 | 1.00 | 38 |
| short-2048-sliding-cp8r7 | cute@cute | 51.6 | 128.0 | 0.97 | 193.0 | 1244 | 0.99 | 38 |
| short-2048-sliding-cp8r7 | cute_ws@cute | 22.4 | 54.6 | 0.42 | 165.2 | 1124 | 0.84 | 38 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 35.5 | 158.1 | 0.67 | - | - | - | 16 |
| heavy-2048-csa-cp1 | tilelang@main | 476.0 | 694.7 | 1.00 | 1821 | 1134 | 1.00 | 265 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 254.4 | 135.3 | 0.53 | - | - | - | 139 |
| heavy-2048-csa-cp1 | tilelang@cute | 475.3 | 691.5 | 1.00 | 1821 | 1202 | 1.00 | 265 |
| heavy-2048-csa-cp1 | cute@cute | 449.5 | 121.3 | 0.95 | 1794 | 584.2 | 0.99 | 265 |
| heavy-2048-csa-cp1 | cute_ws@cute | 226.9 | 53.1 | 0.48 | 1575 | 654.5 | 0.87 | 265 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@cute | 255.1 | 139.5 | 0.54 | - | - | - | 139 |
| heavy-2048-csa-cp8r0 | tilelang@main | 60.1 | 687.4 | 1.00 | 214.2 | 1829 | 1.00 | 40 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.1 | 145.0 | 0.73 | - | - | - | 17 |
| heavy-2048-csa-cp8r0 | tilelang@cute | 60.1 | 689.5 | 1.00 | 214.6 | 1866 | 1.00 | 40 |
| heavy-2048-csa-cp8r0 | cute@cute | 57.9 | 124.6 | 0.96 | 213.2 | 1245 | 0.99 | 40 |
| heavy-2048-csa-cp8r0 | cute_ws@cute | 36.2 | 51.3 | 0.60 | 192.3 | 1109 | 0.90 | 40 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 44.4 | 162.7 | 0.74 | - | - | - | 17 |
| heavy-2048-csa-cp8r4 | tilelang@main | 75.4 | 702.2 | 1.00 | 312.4 | 1752 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 51.7 | 149.0 | 0.69 | - | - | - | 17 |
| heavy-2048-csa-cp8r4 | tilelang@cute | 75.2 | 689.5 | 1.00 | 312.0 | 1762 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | cute@cute | 72.9 | 121.5 | 0.97 | 309.6 | 1131 | 0.99 | 40 |
| heavy-2048-csa-cp8r4 | cute_ws@cute | 42.8 | 50.5 | 0.57 | 281.8 | 1005 | 0.90 | 40 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 50.8 | 148.6 | 0.68 | - | - | - | 17 |
| heavy-2048-csa-cp8r7 | tilelang@main | 93.5 | 699.7 | 1.00 | 410.4 | 1624 | 1.00 | 40 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 60.9 | 148.8 | 0.65 | - | - | - | 17 |
| heavy-2048-csa-cp8r7 | tilelang@cute | 92.6 | 691.8 | 1.00 | 409.1 | 1677 | 1.00 | 40 |
| heavy-2048-csa-cp8r7 | cute@cute | 89.4 | 121.1 | 0.97 | 405.0 | 1029 | 0.99 | 40 |
| heavy-2048-csa-cp8r7 | cute_ws@cute | 52.2 | 51.6 | 0.56 | 370.3 | 914.2 | 0.91 | 40 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 60.8 | 148.9 | 0.66 | - | - | - | 17 |
| heavy-2048-hca-cp1 | tilelang@main | 344.4 | 729.5 | 1.00 | 1109 | 1355 | 1.00 | 265 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 188.5 | 146.7 | 0.55 | - | - | - | 133 |
| heavy-2048-hca-cp1 | tilelang@cute | 343.7 | 726.6 | 1.00 | 1108 | 1391 | 1.00 | 265 |
| heavy-2048-hca-cp1 | cute@cute | 321.5 | 149.7 | 0.94 | 1088 | 774.1 | 0.98 | 265 |
| heavy-2048-hca-cp1 | cute_ws@cute | 160.5 | 52.5 | 0.47 | 929.7 | 743.0 | 0.84 | 265 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@cute | 187.4 | 153.3 | 0.55 | - | - | - | 133 |
| heavy-2048-hca-cp8r0 | tilelang@main | 54.0 | 720.3 | 1.00 | 187.5 | 1948 | 1.00 | 38 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 41.0 | 148.3 | 0.76 | - | - | - | 17 |
| heavy-2048-hca-cp8r0 | tilelang@cute | 53.4 | 736.5 | 1.00 | 187.9 | 2006 | 1.00 | 38 |
| heavy-2048-hca-cp8r0 | cute@cute | 51.8 | 146.9 | 0.97 | 185.0 | 1381 | 0.98 | 38 |
| heavy-2048-hca-cp8r0 | cute_ws@cute | 25.4 | 52.0 | 0.48 | 160.2 | 1207 | 0.85 | 38 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 40.9 | 153.6 | 0.77 | - | - | - | 17 |
| heavy-2048-hca-cp8r4 | tilelang@main | 61.8 | 720.2 | 1.00 | 219.6 | 1903 | 1.00 | 38 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 45.0 | 144.6 | 0.73 | - | - | - | 17 |
| heavy-2048-hca-cp8r4 | tilelang@cute | 61.8 | 755.3 | 1.00 | 219.3 | 2015 | 1.00 | 38 |
| heavy-2048-hca-cp8r4 | cute@cute | 59.5 | 155.7 | 0.96 | 217.3 | 1369 | 0.99 | 38 |
| heavy-2048-hca-cp8r4 | cute_ws@cute | 29.4 | 53.9 | 0.48 | 187.5 | 1190 | 0.85 | 38 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 44.7 | 163.2 | 0.72 | - | - | - | 17 |
| heavy-2048-hca-cp8r7 | tilelang@main | 61.9 | 725.0 | 1.00 | 219.4 | 1938 | 1.00 | 38 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.1 | 144.7 | 0.73 | - | - | - | 17 |
| heavy-2048-hca-cp8r7 | tilelang@cute | 62.1 | 728.6 | 1.00 | 220.4 | 1997 | 1.00 | 38 |
| heavy-2048-hca-cp8r7 | cute@cute | 59.8 | 147.8 | 0.96 | 217.0 | 1336 | 0.98 | 38 |
| heavy-2048-hca-cp8r7 | cute_ws@cute | 29.5 | 52.0 | 0.48 | 188.5 | 1182 | 0.86 | 38 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 45.0 | 154.8 | 0.72 | - | - | - | 17 |
| heavy-2048-sliding-cp1 | tilelang@main | 305.0 | 695.3 | 1.00 | 1000 | 1294 | 1.00 | 264 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 150.5 | 141.3 | 0.49 | - | - | - | 131 |
| heavy-2048-sliding-cp1 | tilelang@cute | 302.8 | 699.5 | 1.00 | 996.6 | 1350 | 1.00 | 264 |
| heavy-2048-sliding-cp1 | cute@cute | 283.4 | 123.6 | 0.94 | 974.8 | 735.2 | 0.98 | 264 |
| heavy-2048-sliding-cp1 | cute_ws@cute | 118.8 | 50.4 | 0.39 | 821.7 | 739.1 | 0.82 | 264 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@cute | 150.3 | 146.4 | 0.50 | - | - | - | 131 |
| heavy-2048-sliding-cp8r0 | tilelang@main | 49.9 | 686.1 | 1.00 | 180.0 | 1859 | 1.00 | 38 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.3 | 144.8 | 0.71 | - | - | - | 16 |
| heavy-2048-sliding-cp8r0 | tilelang@cute | 50.1 | 692.3 | 1.00 | 180.8 | 1919 | 1.00 | 38 |
| heavy-2048-sliding-cp8r0 | cute@cute | 47.9 | 126.5 | 0.96 | 178.1 | 1280 | 0.99 | 38 |
| heavy-2048-sliding-cp8r0 | cute_ws@cute | 22.0 | 53.0 | 0.44 | 153.2 | 1136 | 0.85 | 38 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 35.6 | 156.5 | 0.71 | - | - | - | 16 |
| heavy-2048-sliding-cp8r4 | tilelang@main | 53.5 | 688.8 | 1.00 | 195.8 | 1841 | 1.00 | 38 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.8 | 144.9 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r4 | tilelang@cute | 52.9 | 679.5 | 1.00 | 197.3 | 1921 | 1.00 | 38 |
| heavy-2048-sliding-cp8r4 | cute@cute | 51.2 | 121.1 | 0.97 | 194.1 | 1248 | 0.98 | 38 |
| heavy-2048-sliding-cp8r4 | cute_ws@cute | 22.4 | 51.4 | 0.42 | 166.3 | 1124 | 0.84 | 38 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 35.7 | 154.7 | 0.68 | - | - | - | 16 |
| heavy-2048-sliding-cp8r7 | tilelang@main | 53.2 | 690.4 | 1.00 | 195.4 | 1876 | 1.00 | 38 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.7 | 147.0 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r7 | tilelang@cute | 53.4 | 699.3 | 1.00 | 196.7 | 1912 | 1.00 | 38 |
| heavy-2048-sliding-cp8r7 | cute@cute | 51.4 | 123.1 | 0.96 | 193.8 | 1282 | 0.99 | 38 |
| heavy-2048-sliding-cp8r7 | cute_ws@cute | 22.2 | 52.7 | 0.41 | 165.9 | 1131 | 0.84 | 38 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 35.9 | 156.6 | 0.67 | - | - | - | 16 |
| tiny-2048-csa-cp1 | tilelang@main | 358.8 | 683.7 | 1.00 | 1100 | 1259 | 1.00 | 265 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 202.3 | 131.6 | 0.56 | - | - | - | 139 |
| tiny-2048-csa-cp1 | tilelang@cute | 359.3 | 681.4 | 1.00 | 1099 | 1298 | 1.00 | 265 |
| tiny-2048-csa-cp1 | cute@cute | 335.6 | 116.8 | 0.93 | 1076 | 674.0 | 0.98 | 265 |
| tiny-2048-csa-cp1 | cute_ws@cute | 178.2 | 50.4 | 0.50 | 920.7 | 689.2 | 0.84 | 265 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@cute | 203.0 | 137.5 | 0.57 | - | - | - | 139 |
| tiny-2048-csa-cp8r0 | tilelang@main | 59.5 | 694.4 | 1.00 | 208.2 | 1848 | 1.00 | 40 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.4 | 148.0 | 0.75 | - | - | - | 17 |
| tiny-2048-csa-cp8r0 | tilelang@cute | 59.6 | 701.9 | 1.00 | 207.5 | 1907 | 1.00 | 40 |
| tiny-2048-csa-cp8r0 | cute@cute | 57.2 | 124.7 | 0.96 | 205.6 | 1258 | 0.99 | 40 |
| tiny-2048-csa-cp8r0 | cute_ws@cute | 36.0 | 51.4 | 0.60 | 185.5 | 1109 | 0.89 | 40 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 44.2 | 148.5 | 0.74 | - | - | - | 17 |
| tiny-2048-csa-cp8r4 | tilelang@main | 59.6 | 688.9 | 1.00 | 206.9 | 1822 | 1.00 | 40 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 43.5 | 144.4 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r4 | tilelang@cute | 59.2 | 701.1 | 1.00 | 206.7 | 1890 | 1.00 | 40 |
| tiny-2048-csa-cp8r4 | cute@cute | 57.1 | 122.1 | 0.96 | 203.9 | 1243 | 0.99 | 40 |
| tiny-2048-csa-cp8r4 | cute_ws@cute | 36.0 | 49.8 | 0.61 | 184.1 | 1120 | 0.89 | 40 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 43.6 | 146.4 | 0.74 | - | - | - | 17 |
| tiny-2048-csa-cp8r7 | tilelang@main | 60.4 | 700.6 | 1.00 | 209.9 | 1830 | 1.00 | 40 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 44.1 | 145.9 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r7 | tilelang@cute | 59.9 | 706.1 | 1.00 | 210.3 | 1913 | 1.00 | 40 |
| tiny-2048-csa-cp8r7 | cute@cute | 58.0 | 123.4 | 0.97 | 207.7 | 1258 | 0.99 | 40 |
| tiny-2048-csa-cp8r7 | cute_ws@cute | 36.2 | 51.3 | 0.60 | 187.7 | 1108 | 0.89 | 40 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 44.0 | 150.7 | 0.73 | - | - | - | 17 |
| tiny-2048-hca-cp1 | tilelang@main | 270.8 | 691.8 | 1.00 | 772.0 | 1321 | 1.00 | 264 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 148.6 | 143.6 | 0.55 | - | - | - | 131 |
| tiny-2048-hca-cp1 | tilelang@cute | 271.1 | 688.2 | 1.00 | 772.5 | 1411 | 1.00 | 264 |
| tiny-2048-hca-cp1 | cute@cute | 251.8 | 120.7 | 0.93 | 753.6 | 762.9 | 0.98 | 264 |
| tiny-2048-hca-cp1 | cute_ws@cute | 116.2 | 52.3 | 0.43 | 630.7 | 736.0 | 0.82 | 264 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@cute | 148.3 | 148.8 | 0.55 | - | - | - | 131 |
| tiny-2048-hca-cp8r0 | tilelang@main | 48.4 | 688.1 | 1.00 | 166.6 | 1863 | 1.00 | 38 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 35.5 | 147.1 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r0 | tilelang@cute | 48.7 | 705.4 | 1.00 | 165.2 | 1931 | 1.00 | 38 |
| tiny-2048-hca-cp8r0 | cute@cute | 46.4 | 125.2 | 0.95 | 163.8 | 1295 | 0.99 | 38 |
| tiny-2048-hca-cp8r0 | cute_ws@cute | 22.3 | 52.8 | 0.46 | 140.1 | 1148 | 0.85 | 38 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 35.3 | 155.9 | 0.72 | - | - | - | 16 |
| tiny-2048-hca-cp8r4 | tilelang@main | 48.4 | 686.5 | 1.00 | 164.1 | 1885 | 1.00 | 38 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 35.5 | 145.4 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r4 | tilelang@cute | 48.6 | 699.8 | 1.00 | 164.2 | 1929 | 1.00 | 38 |
| tiny-2048-hca-cp8r4 | cute@cute | 46.4 | 125.1 | 0.96 | 163.1 | 1278 | 0.99 | 38 |
| tiny-2048-hca-cp8r4 | cute_ws@cute | 22.0 | 52.8 | 0.45 | 138.8 | 1138 | 0.85 | 38 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 35.4 | 150.3 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r7 | tilelang@main | 50.0 | 701.3 | 1.00 | 175.8 | 1923 | 1.00 | 38 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 35.6 | 154.3 | 0.71 | - | - | - | 16 |
| tiny-2048-hca-cp8r7 | tilelang@cute | 50.3 | 695.2 | 1.00 | 175.9 | 1877 | 1.00 | 38 |
| tiny-2048-hca-cp8r7 | cute@cute | 47.9 | 122.2 | 0.95 | 174.2 | 1247 | 0.99 | 38 |
| tiny-2048-hca-cp8r7 | cute_ws@cute | 21.9 | 53.3 | 0.44 | 149.2 | 1102 | 0.85 | 38 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 35.5 | 153.7 | 0.71 | - | - | - | 16 |
| tiny-2048-sliding-cp1 | tilelang@main | 271.0 | 686.8 | 1.00 | 772.5 | 1336 | 1.00 | 264 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 149.0 | 141.5 | 0.55 | - | - | - | 131 |
| tiny-2048-sliding-cp1 | tilelang@cute | 271.0 | 680.9 | 1.00 | 772.7 | 1366 | 1.00 | 264 |
| tiny-2048-sliding-cp1 | cute@cute | 252.1 | 124.8 | 0.93 | 752.8 | 740.1 | 0.97 | 264 |
| tiny-2048-sliding-cp1 | cute_ws@cute | 116.8 | 52.9 | 0.43 | 631.5 | 725.8 | 0.82 | 264 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@cute | 149.1 | 150.3 | 0.55 | - | - | - | 131 |
| tiny-2048-sliding-cp8r0 | tilelang@main | 48.3 | 683.4 | 1.00 | 165.9 | 1970 | 1.00 | 38 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.2 | 144.5 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r0 | tilelang@cute | 48.2 | 705.4 | 1.00 | 165.4 | 1905 | 1.00 | 38 |
| tiny-2048-sliding-cp8r0 | cute@cute | 46.4 | 122.6 | 0.96 | 164.1 | 1278 | 0.99 | 38 |
| tiny-2048-sliding-cp8r0 | cute_ws@cute | 21.8 | 52.8 | 0.45 | 140.9 | 1125 | 0.85 | 38 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 35.7 | 152.5 | 0.74 | - | - | - | 16 |
| tiny-2048-sliding-cp8r4 | tilelang@main | 48.3 | 690.2 | 1.00 | 164.1 | 1860 | 1.00 | 38 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.3 | 143.1 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r4 | tilelang@cute | 48.5 | 675.5 | 1.00 | 164.6 | 1936 | 1.00 | 38 |
| tiny-2048-sliding-cp8r4 | cute@cute | 46.6 | 123.5 | 0.96 | 162.3 | 1281 | 0.99 | 38 |
| tiny-2048-sliding-cp8r4 | cute_ws@cute | 21.7 | 54.4 | 0.45 | 138.3 | 1136 | 0.84 | 38 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 35.4 | 153.6 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r7 | tilelang@main | 50.1 | 708.0 | 1.00 | 175.6 | 1874 | 1.00 | 38 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.6 | 151.2 | 0.71 | - | - | - | 16 |
| tiny-2048-sliding-cp8r7 | tilelang@cute | 50.0 | 699.7 | 1.00 | 175.1 | 1915 | 1.00 | 38 |
| tiny-2048-sliding-cp8r7 | cute@cute | 48.0 | 126.3 | 0.96 | 173.4 | 1275 | 0.99 | 38 |
| tiny-2048-sliding-cp8r7 | cute_ws@cute | 22.0 | 53.8 | 0.44 | 149.3 | 1130 | 0.85 | 38 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 35.6 | 155.6 | 0.71 | - | - | - | 16 |
| single-4096-csa-cp1 | tilelang@main | 1208 | 682.8 | 1.00 | 5145 | 827.0 | 1.00 | 530 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 593.7 | 121.7 | 0.49 | - | - | - | 278 |
| single-4096-csa-cp1 | tilelang@cute | 1218 | 667.5 | 1.00 | 5149 | 867.8 | 1.00 | 530 |
| single-4096-csa-cp1 | cute@cute | 1159 | 113.6 | 0.95 | 5087 | 250.1 | 0.99 | 530 |
| single-4096-csa-cp1 | cute_ws@cute | 553.8 | 52.3 | 0.45 | 4491 | 202.8 | 0.87 | 530 |
| single-4096-csa-cp1 | flashmla_fwd_ref@cute | 599.7 | 123.5 | 0.49 | - | - | - | 278 |
| single-4096-csa-cp8r0 | tilelang@main | 112.0 | 683.6 | 1.00 | 410.0 | 1627 | 1.00 | 79 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.0 | 138.3 | 0.61 | - | - | - | 35 |
| single-4096-csa-cp8r0 | tilelang@cute | 112.8 | 685.4 | 1.00 | 410.7 | 1654 | 1.00 | 79 |
| single-4096-csa-cp8r0 | cute@cute | 106.8 | 119.9 | 0.95 | 404.6 | 1024 | 0.99 | 79 |
| single-4096-csa-cp8r0 | cute_ws@cute | 57.2 | 50.3 | 0.51 | 356.3 | 903.4 | 0.87 | 79 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 68.7 | 150.6 | 0.61 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang@main | 191.2 | 677.8 | 1.00 | 856.0 | 1477 | 1.00 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 109.5 | 138.9 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang@cute | 191.3 | 688.9 | 1.00 | 859.6 | 1516 | 1.00 | 79 |
| single-4096-csa-cp8r4 | cute@cute | 183.4 | 122.2 | 0.96 | 852.5 | 878.0 | 0.99 | 79 |
| single-4096-csa-cp8r4 | cute_ws@cute | 97.3 | 50.9 | 0.51 | 766.3 | 815.8 | 0.89 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 109.5 | 143.5 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r7 | tilelang@main | 191.7 | 689.2 | 1.00 | 859.6 | 1490 | 1.00 | 79 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 109.4 | 142.6 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r7 | tilelang@cute | 190.5 | 699.6 | 1.00 | 858.6 | 1509 | 1.00 | 79 |
| single-4096-csa-cp8r7 | cute@cute | 182.9 | 121.8 | 0.96 | 849.7 | 880.0 | 0.99 | 79 |
| single-4096-csa-cp8r7 | cute_ws@cute | 96.5 | 52.4 | 0.51 | 765.1 | 814.9 | 0.89 | 79 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 109.7 | 151.8 | 0.58 | - | - | - | 35 |
| single-4096-hca-cp1 | tilelang@main | 691.9 | 740.4 | 1.00 | 2233 | 900.3 | 1.00 | 530 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 366.3 | 139.2 | 0.53 | - | - | - | 266 |
| single-4096-hca-cp1 | tilelang@cute | 690.8 | 727.5 | 1.00 | 2235 | 925.8 | 1.00 | 530 |
| single-4096-hca-cp1 | cute@cute | 644.4 | 144.7 | 0.93 | 2183 | 339.6 | 0.98 | 530 |
| single-4096-hca-cp1 | cute_ws@cute | 317.2 | 55.5 | 0.46 | 1868 | 457.9 | 0.84 | 530 |
| single-4096-hca-cp1 | flashmla_fwd_ref@cute | 363.3 | 149.6 | 0.53 | - | - | - | 266 |
| single-4096-hca-cp8r0 | tilelang@main | 102.0 | 726.2 | 1.00 | 349.1 | 1777 | 1.00 | 77 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 66.4 | 150.7 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r0 | tilelang@cute | 101.9 | 730.6 | 1.00 | 349.6 | 1876 | 1.00 | 77 |
| single-4096-hca-cp8r0 | cute@cute | 96.2 | 151.8 | 0.94 | 344.8 | 1218 | 0.99 | 77 |
| single-4096-hca-cp8r0 | cute_ws@cute | 48.6 | 52.7 | 0.48 | 296.1 | 1066 | 0.85 | 77 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 65.0 | 161.8 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang@main | 107.4 | 716.8 | 1.00 | 370.8 | 1772 | 1.00 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 70.2 | 146.6 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang@cute | 106.9 | 733.7 | 1.00 | 370.8 | 1821 | 1.00 | 77 |
| single-4096-hca-cp8r4 | cute@cute | 101.6 | 145.9 | 0.95 | 365.0 | 1190 | 0.98 | 77 |
| single-4096-hca-cp8r4 | cute_ws@cute | 51.4 | 50.9 | 0.48 | 315.5 | 1022 | 0.85 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 68.9 | 153.6 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r7 | tilelang@main | 107.1 | 720.0 | 1.00 | 372.8 | 1786 | 1.00 | 77 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 68.8 | 148.4 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r7 | tilelang@cute | 107.2 | 725.6 | 1.00 | 371.8 | 1800 | 1.00 | 77 |
| single-4096-hca-cp8r7 | cute@cute | 101.4 | 148.8 | 0.95 | 367.3 | 1163 | 0.99 | 77 |
| single-4096-hca-cp8r7 | cute_ws@cute | 51.7 | 51.7 | 0.48 | 317.2 | 1017 | 0.85 | 77 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 69.5 | 154.6 | 0.65 | - | - | - | 33 |
| single-4096-sliding-cp1 | tilelang@main | 594.3 | 703.0 | 1.00 | 1945 | 914.2 | 1.00 | 527 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 273.9 | 146.3 | 0.46 | - | - | - | 262 |
| single-4096-sliding-cp1 | tilelang@cute | 593.8 | 691.1 | 1.00 | 1947 | 944.7 | 1.00 | 527 |
| single-4096-sliding-cp1 | cute@cute | 547.8 | 131.4 | 0.92 | 1904 | 353.9 | 0.98 | 527 |
| single-4096-sliding-cp1 | cute_ws@cute | 218.6 | 54.4 | 0.37 | 1586 | 531.0 | 0.81 | 527 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@cute | 276.0 | 148.8 | 0.46 | - | - | - | 262 |
| single-4096-sliding-cp8r0 | tilelang@main | 90.6 | 699.5 | 1.00 | 319.4 | 1715 | 1.00 | 76 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 53.1 | 146.2 | 0.59 | - | - | - | 33 |
| single-4096-sliding-cp8r0 | tilelang@cute | 90.0 | 697.0 | 1.00 | 318.2 | 1754 | 1.00 | 76 |
| single-4096-sliding-cp8r0 | cute@cute | 84.5 | 123.0 | 0.94 | 312.8 | 1126 | 0.98 | 76 |
| single-4096-sliding-cp8r0 | cute_ws@cute | 37.2 | 52.0 | 0.41 | 265.1 | 1013 | 0.83 | 76 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 53.3 | 155.8 | 0.59 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@main | 92.4 | 698.6 | 1.00 | 327.4 | 1730 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.7 | 149.4 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@cute | 92.4 | 705.1 | 1.00 | 327.1 | 1787 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | cute@cute | 87.3 | 120.9 | 0.94 | 321.8 | 1131 | 0.98 | 76 |
| single-4096-sliding-cp8r4 | cute_ws@cute | 37.2 | 51.5 | 0.40 | 272.8 | 1010 | 0.83 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 53.8 | 152.1 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r7 | tilelang@main | 92.2 | 693.0 | 1.00 | 327.1 | 1738 | 1.00 | 76 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.9 | 140.2 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r7 | tilelang@cute | 92.8 | 700.9 | 1.00 | 326.0 | 1780 | 1.00 | 76 |
| single-4096-sliding-cp8r7 | cute@cute | 87.5 | 125.0 | 0.94 | 322.6 | 1134 | 0.99 | 76 |
| single-4096-sliding-cp8r7 | cute_ws@cute | 37.0 | 52.8 | 0.40 | 272.2 | 1016 | 0.83 | 76 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 53.8 | 156.4 | 0.58 | - | - | - | 33 |
| short-4096-csa-cp1 | tilelang@main | 829.2 | 694.2 | 1.00 | 3054 | 819.3 | 1.00 | 530 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 428.1 | 124.9 | 0.52 | - | - | - | 278 |
| short-4096-csa-cp1 | tilelang@cute | 829.9 | 682.6 | 1.00 | 3054 | 838.9 | 1.00 | 530 |
| short-4096-csa-cp1 | cute@cute | 782.0 | 120.5 | 0.94 | 3004 | 229.6 | 0.98 | 530 |
| short-4096-csa-cp1 | cute_ws@cute | 374.4 | 53.3 | 0.45 | 2606 | 358.0 | 0.85 | 530 |
| short-4096-csa-cp1 | flashmla_fwd_ref@cute | 429.9 | 120.4 | 0.52 | - | - | - | 278 |
| short-4096-csa-cp8r0 | tilelang@main | 112.0 | 715.7 | 1.00 | 410.6 | 1643 | 1.00 | 79 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.3 | 154.1 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r0 | tilelang@cute | 111.8 | 685.4 | 1.00 | 410.7 | 1662 | 1.00 | 79 |
| short-4096-csa-cp8r0 | cute@cute | 106.6 | 118.4 | 0.95 | 405.2 | 1031 | 0.99 | 79 |
| short-4096-csa-cp8r0 | cute_ws@cute | 57.5 | 49.2 | 0.51 | 355.6 | 931.1 | 0.87 | 79 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 68.4 | 144.7 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang@main | 114.7 | 762.2 | 1.00 | 432.7 | 1712 | 1.00 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.3 | 154.8 | 0.63 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang@cute | 113.8 | 679.2 | 1.00 | 430.8 | 1648 | 1.00 | 79 |
| short-4096-csa-cp8r4 | cute@cute | 108.9 | 116.0 | 0.96 | 426.3 | 1018 | 0.99 | 79 |
| short-4096-csa-cp8r4 | cute_ws@cute | 60.5 | 50.0 | 0.53 | 379.5 | 921.6 | 0.88 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 72.6 | 145.1 | 0.64 | - | - | - | 35 |
| short-4096-csa-cp8r7 | tilelang@main | 112.8 | 705.3 | 1.00 | 430.6 | 1629 | 1.00 | 79 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 68.9 | 141.8 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r7 | tilelang@cute | 112.3 | 687.3 | 1.00 | 430.5 | 1634 | 1.00 | 79 |
| short-4096-csa-cp8r7 | cute@cute | 107.0 | 120.6 | 0.95 | 425.8 | 1013 | 0.99 | 79 |
| short-4096-csa-cp8r7 | cute_ws@cute | 58.5 | 51.2 | 0.52 | 378.5 | 891.1 | 0.88 | 79 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 69.1 | 149.4 | 0.62 | - | - | - | 35 |
| short-4096-hca-cp1 | tilelang@main | 666.3 | 744.5 | 1.00 | 2131 | 958.7 | 1.00 | 530 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 359.6 | 140.5 | 0.54 | - | - | - | 266 |
| short-4096-hca-cp1 | tilelang@cute | 664.5 | 723.8 | 1.00 | 2134 | 948.6 | 1.00 | 530 |
| short-4096-hca-cp1 | cute@cute | 621.1 | 141.3 | 0.93 | 2088 | 374.7 | 0.98 | 530 |
| short-4096-hca-cp1 | cute_ws@cute | 302.7 | 52.2 | 0.46 | 1777 | 492.2 | 0.83 | 530 |
| short-4096-hca-cp1 | flashmla_fwd_ref@cute | 360.6 | 139.3 | 0.54 | - | - | - | 266 |
| short-4096-hca-cp8r0 | tilelang@main | 100.2 | 744.3 | 1.00 | 347.4 | 1833 | 1.00 | 77 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 64.7 | 151.3 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r0 | tilelang@cute | 100.2 | 725.0 | 1.00 | 346.9 | 1857 | 1.00 | 77 |
| short-4096-hca-cp8r0 | cute@cute | 94.7 | 149.1 | 0.95 | 341.3 | 1213 | 0.98 | 77 |
| short-4096-hca-cp8r0 | cute_ws@cute | 47.6 | 53.5 | 0.48 | 294.8 | 1033 | 0.85 | 77 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 65.5 | 155.6 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang@main | 101.5 | 725.0 | 1.00 | 354.4 | 1803 | 1.00 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 65.3 | 142.9 | 0.64 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang@cute | 101.1 | 719.4 | 1.00 | 354.7 | 1827 | 1.00 | 77 |
| short-4096-hca-cp8r4 | cute@cute | 95.9 | 143.3 | 0.95 | 348.0 | 1196 | 0.98 | 77 |
| short-4096-hca-cp8r4 | cute_ws@cute | 47.3 | 50.8 | 0.47 | 299.3 | 1043 | 0.84 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 64.6 | 150.1 | 0.64 | - | - | - | 33 |
| short-4096-hca-cp8r7 | tilelang@main | 102.6 | 728.6 | 1.00 | 359.5 | 1837 | 1.00 | 77 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 67.0 | 147.2 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r7 | tilelang@cute | 102.6 | 724.1 | 1.00 | 360.2 | 1837 | 1.00 | 77 |
| short-4096-hca-cp8r7 | cute@cute | 97.6 | 145.5 | 0.95 | 355.2 | 1196 | 0.99 | 77 |
| short-4096-hca-cp8r7 | cute_ws@cute | 47.8 | 53.4 | 0.47 | 305.7 | 1038 | 0.85 | 77 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 67.3 | 151.8 | 0.66 | - | - | - | 33 |
| short-4096-sliding-cp1 | tilelang@main | 584.0 | 727.8 | 1.00 | 1896 | 946.0 | 1.00 | 527 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 271.4 | 151.8 | 0.46 | - | - | - | 262 |
| short-4096-sliding-cp1 | tilelang@cute | 582.7 | 701.6 | 1.00 | 1897 | 989.0 | 1.00 | 527 |
| short-4096-sliding-cp1 | cute@cute | 541.7 | 122.1 | 0.93 | 1855 | 365.8 | 0.98 | 527 |
| short-4096-sliding-cp1 | cute_ws@cute | 215.6 | 52.3 | 0.37 | 1532 | 541.4 | 0.81 | 527 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@cute | 273.0 | 152.5 | 0.47 | - | - | - | 262 |
| short-4096-sliding-cp8r0 | tilelang@main | 89.3 | 704.5 | 1.00 | 317.4 | 1719 | 1.00 | 76 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 53.0 | 144.4 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r0 | tilelang@cute | 89.6 | 699.8 | 1.00 | 316.9 | 1806 | 1.00 | 76 |
| short-4096-sliding-cp8r0 | cute@cute | 84.7 | 125.4 | 0.94 | 310.8 | 1147 | 0.98 | 76 |
| short-4096-sliding-cp8r0 | cute_ws@cute | 37.4 | 51.8 | 0.42 | 264.0 | 1031 | 0.83 | 76 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 53.4 | 147.9 | 0.60 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@main | 89.9 | 704.2 | 1.00 | 319.7 | 1722 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.1 | 144.6 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@cute | 90.4 | 689.2 | 1.00 | 321.4 | 1773 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | cute@cute | 85.3 | 124.9 | 0.94 | 315.0 | 1129 | 0.98 | 76 |
| short-4096-sliding-cp8r4 | cute_ws@cute | 37.4 | 52.9 | 0.41 | 267.6 | 1013 | 0.83 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 53.1 | 151.7 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r7 | tilelang@main | 90.8 | 684.0 | 1.00 | 323.9 | 1762 | 1.00 | 76 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 54.0 | 141.7 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r7 | tilelang@cute | 90.6 | 700.1 | 1.00 | 323.2 | 1753 | 1.00 | 76 |
| short-4096-sliding-cp8r7 | cute@cute | 85.4 | 123.2 | 0.94 | 317.5 | 1111 | 0.98 | 76 |
| short-4096-sliding-cp8r7 | cute_ws@cute | 37.3 | 52.7 | 0.41 | 269.8 | 999.7 | 0.83 | 76 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 53.3 | 153.5 | 0.59 | - | - | - | 33 |
| heavy-4096-csa-cp1 | tilelang@main | 770.8 | 691.6 | 1.00 | 2718 | 827.3 | 1.00 | 530 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 404.9 | 120.0 | 0.53 | - | - | - | 278 |
| heavy-4096-csa-cp1 | tilelang@cute | 772.3 | 695.1 | 1.00 | 2723 | 830.2 | 1.00 | 530 |
| heavy-4096-csa-cp1 | cute@cute | 724.7 | 115.4 | 0.94 | 2673 | 231.8 | 0.98 | 530 |
| heavy-4096-csa-cp1 | cute_ws@cute | 357.4 | 50.0 | 0.46 | 2301 | 390.0 | 0.84 | 530 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@cute | 407.3 | 123.6 | 0.53 | - | - | - | 278 |
| heavy-4096-csa-cp8r0 | tilelang@main | 111.8 | 682.7 | 1.00 | 410.6 | 1622 | 1.00 | 79 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.1 | 140.3 | 0.61 | - | - | - | 35 |
| heavy-4096-csa-cp8r0 | tilelang@cute | 112.2 | 696.0 | 1.00 | 411.7 | 1695 | 1.00 | 79 |
| heavy-4096-csa-cp8r0 | cute@cute | 106.4 | 123.5 | 0.95 | 405.6 | 1050 | 0.99 | 79 |
| heavy-4096-csa-cp8r0 | cute_ws@cute | 56.9 | 52.1 | 0.51 | 356.9 | 922.7 | 0.87 | 79 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 68.3 | 151.6 | 0.61 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang@main | 114.6 | 703.3 | 1.00 | 434.1 | 1615 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.1 | 145.2 | 0.63 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang@cute | 114.2 | 694.1 | 1.00 | 433.5 | 1698 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | cute@cute | 108.9 | 121.5 | 0.95 | 428.1 | 1026 | 0.99 | 79 |
| heavy-4096-csa-cp8r4 | cute_ws@cute | 60.0 | 51.3 | 0.53 | 381.0 | 895.5 | 0.88 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 72.3 | 151.0 | 0.63 | - | - | - | 35 |
| heavy-4096-csa-cp8r7 | tilelang@main | 105.7 | 744.1 | 1.00 | 378.3 | 1711 | 1.00 | 79 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 68.6 | 147.1 | 0.65 | - | - | - | 35 |
| heavy-4096-csa-cp8r7 | tilelang@cute | 105.5 | 704.9 | 1.00 | 379.4 | 1745 | 1.00 | 79 |
| heavy-4096-csa-cp8r7 | cute@cute | 100.4 | 125.3 | 0.95 | 374.7 | 1069 | 0.99 | 79 |
| heavy-4096-csa-cp8r7 | cute_ws@cute | 57.2 | 52.8 | 0.54 | 330.5 | 956.6 | 0.87 | 79 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 68.0 | 151.7 | 0.64 | - | - | - | 35 |
| heavy-4096-hca-cp1 | tilelang@main | 647.0 | 727.5 | 1.00 | 2030 | 952.6 | 1.00 | 530 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 343.9 | 138.8 | 0.53 | - | - | - | 266 |
| heavy-4096-hca-cp1 | tilelang@cute | 646.3 | 736.7 | 1.00 | 2027 | 992.2 | 1.00 | 530 |
| heavy-4096-hca-cp1 | cute@cute | 601.3 | 145.5 | 0.93 | 1983 | 389.5 | 0.98 | 530 |
| heavy-4096-hca-cp1 | cute_ws@cute | 286.7 | 54.4 | 0.44 | 1671 | 506.6 | 0.82 | 530 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@cute | 342.4 | 149.2 | 0.53 | - | - | - | 266 |
| heavy-4096-hca-cp8r0 | tilelang@main | 100.5 | 735.0 | 1.00 | 348.5 | 1804 | 1.00 | 77 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 65.0 | 149.5 | 0.65 | - | - | - | 33 |
| heavy-4096-hca-cp8r0 | tilelang@cute | 100.8 | 732.0 | 1.00 | 350.1 | 1858 | 1.00 | 77 |
| heavy-4096-hca-cp8r0 | cute@cute | 95.5 | 150.3 | 0.95 | 345.3 | 1208 | 0.99 | 77 |
| heavy-4096-hca-cp8r0 | cute_ws@cute | 47.6 | 52.8 | 0.47 | 297.1 | 1036 | 0.85 | 77 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 64.8 | 157.3 | 0.64 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang@main | 102.2 | 725.7 | 1.00 | 357.1 | 1803 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 64.6 | 142.8 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang@cute | 100.6 | 732.1 | 1.00 | 357.4 | 1860 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | cute@cute | 96.4 | 145.2 | 0.96 | 350.8 | 1217 | 0.98 | 77 |
| heavy-4096-hca-cp8r4 | cute_ws@cute | 48.1 | 50.6 | 0.48 | 301.5 | 1055 | 0.84 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 64.5 | 148.3 | 0.64 | - | - | - | 33 |
| heavy-4096-hca-cp8r7 | tilelang@main | 97.6 | 740.6 | 1.00 | 338.4 | 1841 | 1.00 | 77 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 61.8 | 154.0 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r7 | tilelang@cute | 97.8 | 746.9 | 1.00 | 337.9 | 1892 | 1.00 | 77 |
| heavy-4096-hca-cp8r7 | cute@cute | 92.9 | 147.4 | 0.95 | 332.2 | 1239 | 0.98 | 77 |
| heavy-4096-hca-cp8r7 | cute_ws@cute | 44.9 | 51.4 | 0.46 | 284.0 | 1062 | 0.84 | 77 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 61.6 | 156.8 | 0.63 | - | - | - | 33 |
| heavy-4096-sliding-cp1 | tilelang@main | 582.4 | 700.3 | 1.00 | 1840 | 1071 | 1.00 | 527 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 269.1 | 149.7 | 0.46 | - | - | - | 262 |
| heavy-4096-sliding-cp1 | tilelang@cute | 585.1 | 725.8 | 1.00 | 1843 | 975.5 | 1.00 | 527 |
| heavy-4096-sliding-cp1 | cute@cute | 537.1 | 134.8 | 0.92 | 1800 | 362.0 | 0.98 | 527 |
| heavy-4096-sliding-cp1 | cute_ws@cute | 216.2 | 57.7 | 0.37 | 1484 | 530.7 | 0.80 | 527 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@cute | 276.0 | 154.7 | 0.47 | - | - | - | 262 |
| heavy-4096-sliding-cp8r0 | tilelang@main | 88.9 | 704.9 | 1.00 | 318.0 | 1765 | 1.00 | 76 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 52.5 | 146.2 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r0 | tilelang@cute | 89.2 | 698.8 | 1.00 | 317.6 | 1787 | 1.00 | 76 |
| heavy-4096-sliding-cp8r0 | cute@cute | 83.7 | 123.4 | 0.94 | 312.4 | 1136 | 0.98 | 76 |
| heavy-4096-sliding-cp8r0 | cute_ws@cute | 37.4 | 51.5 | 0.42 | 263.7 | 1038 | 0.83 | 76 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 53.2 | 156.9 | 0.60 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@main | 90.6 | 690.0 | 1.00 | 321.4 | 1729 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.9 | 140.4 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@cute | 90.9 | 697.6 | 1.00 | 323.0 | 1791 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | cute@cute | 85.5 | 122.4 | 0.94 | 316.2 | 1122 | 0.98 | 76 |
| heavy-4096-sliding-cp8r4 | cute_ws@cute | 37.5 | 52.3 | 0.41 | 268.0 | 1019 | 0.83 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 53.3 | 154.5 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r7 | tilelang@main | 87.9 | 684.1 | 1.00 | 309.0 | 1790 | 1.00 | 76 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.5 | 144.1 | 0.61 | - | - | - | 33 |
| heavy-4096-sliding-cp8r7 | tilelang@cute | 87.7 | 709.6 | 1.00 | 310.6 | 1789 | 1.00 | 76 |
| heavy-4096-sliding-cp8r7 | cute@cute | 82.4 | 125.2 | 0.94 | 304.6 | 1148 | 0.98 | 76 |
| heavy-4096-sliding-cp8r7 | cute_ws@cute | 37.2 | 52.4 | 0.42 | 259.5 | 1031 | 0.84 | 76 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 52.9 | 149.7 | 0.60 | - | - | - | 33 |
| tiny-4096-csa-cp1 | tilelang@main | 680.3 | 693.4 | 1.00 | 2071 | 848.0 | 1.00 | 530 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 372.6 | 128.9 | 0.55 | - | - | - | 278 |
| tiny-4096-csa-cp1 | tilelang@cute | 680.5 | 694.2 | 1.00 | 2074 | 850.6 | 1.00 | 530 |
| tiny-4096-csa-cp1 | cute@cute | 637.5 | 113.5 | 0.94 | 2029 | 271.5 | 0.98 | 530 |
| tiny-4096-csa-cp1 | cute_ws@cute | 327.3 | 54.8 | 0.48 | 1721 | 421.4 | 0.83 | 530 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@cute | 376.6 | 122.3 | 0.55 | - | - | - | 278 |
| tiny-4096-csa-cp8r0 | tilelang@main | 103.6 | 692.9 | 1.00 | 340.7 | 1680 | 1.00 | 79 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 66.6 | 144.3 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r0 | tilelang@cute | 103.0 | 698.6 | 1.00 | 342.7 | 1792 | 1.00 | 79 |
| tiny-4096-csa-cp8r0 | cute@cute | 98.3 | 119.2 | 0.95 | 337.4 | 1118 | 0.98 | 79 |
| tiny-4096-csa-cp8r0 | cute_ws@cute | 57.3 | 51.5 | 0.56 | 296.7 | 987.8 | 0.87 | 79 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 67.5 | 146.1 | 0.66 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang@main | 104.5 | 704.8 | 1.00 | 345.8 | 1688 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 67.1 | 142.5 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang@cute | 104.5 | 693.3 | 1.00 | 344.8 | 1752 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | cute@cute | 98.7 | 123.8 | 0.94 | 339.9 | 1111 | 0.99 | 79 |
| tiny-4096-csa-cp8r4 | cute_ws@cute | 57.2 | 51.5 | 0.55 | 298.4 | 994.2 | 0.87 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 67.2 | 149.8 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r7 | tilelang@main | 104.5 | 681.6 | 1.00 | 346.0 | 1697 | 1.00 | 79 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 67.8 | 142.8 | 0.65 | - | - | - | 35 |
| tiny-4096-csa-cp8r7 | tilelang@cute | 104.2 | 707.2 | 1.00 | 345.9 | 1759 | 1.00 | 79 |
| tiny-4096-csa-cp8r7 | cute@cute | 99.6 | 124.0 | 0.96 | 340.9 | 1125 | 0.99 | 79 |
| tiny-4096-csa-cp8r7 | cute_ws@cute | 57.2 | 50.9 | 0.55 | 299.6 | 1021 | 0.87 | 79 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 68.2 | 147.7 | 0.65 | - | - | - | 35 |
| tiny-4096-hca-cp1 | tilelang@main | 532.2 | 695.1 | 1.00 | 1436 | 973.1 | 1.00 | 527 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 273.7 | 138.6 | 0.51 | - | - | - | 262 |
| tiny-4096-hca-cp1 | tilelang@cute | 530.8 | 702.9 | 1.00 | 1434 | 1031 | 1.00 | 527 |
| tiny-4096-hca-cp1 | cute@cute | 490.0 | 123.0 | 0.92 | 1394 | 418.4 | 0.97 | 527 |
| tiny-4096-hca-cp1 | cute_ws@cute | 234.8 | 54.7 | 0.44 | 1148 | 527.9 | 0.80 | 527 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@cute | 272.7 | 147.2 | 0.51 | - | - | - | 262 |
| tiny-4096-hca-cp8r0 | tilelang@main | 81.8 | 688.6 | 1.00 | 254.2 | 1778 | 1.00 | 76 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 52.5 | 142.3 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r0 | tilelang@cute | 81.6 | 706.7 | 1.00 | 254.4 | 1854 | 1.00 | 76 |
| tiny-4096-hca-cp8r0 | cute@cute | 76.7 | 125.4 | 0.94 | 247.8 | 1200 | 0.97 | 76 |
| tiny-4096-hca-cp8r0 | cute_ws@cute | 36.9 | 52.4 | 0.45 | 209.2 | 1073 | 0.82 | 76 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 52.9 | 152.9 | 0.65 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang@main | 84.2 | 683.2 | 1.00 | 267.7 | 1755 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 53.3 | 143.4 | 0.63 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang@cute | 84.3 | 705.3 | 1.00 | 267.4 | 1870 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | cute@cute | 79.6 | 118.7 | 0.94 | 261.9 | 1212 | 0.98 | 76 |
| tiny-4096-hca-cp8r4 | cute_ws@cute | 36.6 | 50.4 | 0.43 | 220.6 | 1093 | 0.83 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 52.7 | 149.7 | 0.62 | - | - | - | 33 |
| tiny-4096-hca-cp8r7 | tilelang@main | 82.2 | 701.9 | 1.00 | 260.4 | 1793 | 1.00 | 76 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 52.8 | 148.7 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r7 | tilelang@cute | 82.2 | 691.1 | 1.00 | 260.2 | 1839 | 1.00 | 76 |
| tiny-4096-hca-cp8r7 | cute@cute | 77.7 | 123.5 | 0.95 | 256.4 | 1197 | 0.99 | 76 |
| tiny-4096-hca-cp8r7 | cute_ws@cute | 36.9 | 50.9 | 0.45 | 214.8 | 1083 | 0.83 | 76 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 53.0 | 153.8 | 0.65 | - | - | - | 33 |
| tiny-4096-sliding-cp1 | tilelang@main | 534.3 | 692.2 | 1.00 | 1439 | 968.5 | 1.00 | 527 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 269.6 | 139.2 | 0.50 | - | - | - | 262 |
| tiny-4096-sliding-cp1 | tilelang@cute | 531.1 | 696.5 | 1.00 | 1435 | 1031 | 1.00 | 527 |
| tiny-4096-sliding-cp1 | cute@cute | 490.2 | 127.8 | 0.92 | 1395 | 432.0 | 0.97 | 527 |
| tiny-4096-sliding-cp1 | cute_ws@cute | 233.1 | 52.9 | 0.44 | 1148 | 524.1 | 0.80 | 527 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@cute | 275.4 | 152.2 | 0.52 | - | - | - | 262 |
| tiny-4096-sliding-cp8r0 | tilelang@main | 81.9 | 693.4 | 1.00 | 254.8 | 1776 | 1.00 | 76 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 52.1 | 148.6 | 0.64 | - | - | - | 33 |
| tiny-4096-sliding-cp8r0 | tilelang@cute | 82.1 | 705.7 | 1.00 | 254.5 | 1857 | 1.00 | 76 |
| tiny-4096-sliding-cp8r0 | cute@cute | 76.6 | 124.7 | 0.93 | 250.3 | 1202 | 0.98 | 76 |
| tiny-4096-sliding-cp8r0 | cute_ws@cute | 37.5 | 52.7 | 0.46 | 209.6 | 1086 | 0.82 | 76 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 53.5 | 153.0 | 0.65 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@main | 84.0 | 697.6 | 1.00 | 267.6 | 1772 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 52.9 | 150.1 | 0.63 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@cute | 84.4 | 713.5 | 1.00 | 266.8 | 1854 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | cute@cute | 79.7 | 124.8 | 0.94 | 261.8 | 1206 | 0.98 | 76 |
| tiny-4096-sliding-cp8r4 | cute_ws@cute | 36.6 | 52.0 | 0.43 | 220.4 | 1057 | 0.83 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 52.9 | 153.5 | 0.63 | - | - | - | 33 |
| tiny-4096-sliding-cp8r7 | tilelang@main | 82.0 | 704.4 | 1.00 | 259.8 | 1821 | 1.00 | 76 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.1 | 149.0 | 0.65 | - | - | - | 33 |
| tiny-4096-sliding-cp8r7 | tilelang@cute | 82.3 | 693.3 | 1.00 | 260.6 | 1851 | 1.00 | 76 |
| tiny-4096-sliding-cp8r7 | cute@cute | 77.7 | 123.6 | 0.94 | 257.0 | 1197 | 0.99 | 76 |
| tiny-4096-sliding-cp8r7 | cute_ws@cute | 37.3 | 54.4 | 0.45 | 214.3 | 1096 | 0.82 | 76 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 53.2 | 157.9 | 0.65 | - | - | - | 33 |
| single-16384-csa-cp1 | tilelang@main | 5358 | 559.6 | 1.00 | 22675 | 885.3 | 1.00 | 2120 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 2542 | 28.1 | 0.47 | - | - | - | 1112 |
| single-16384-csa-cp1 | tilelang@cute | 5790 | 94.8 | 1.00 | 22695 | 888.7 | 1.00 | 2120 |
| single-16384-csa-cp1 | cute@cute | 5026 | 79.3 | 0.87 | 22430 | 286.3 | 0.99 | 2120 |
| single-16384-csa-cp1 | cute_ws@cute | 2365 | 77.6 | 0.41 | 19870 | 209.0 | 0.88 | 2120 |
| single-16384-csa-cp1 | flashmla_fwd_ref@cute | 2535 | 84.6 | 0.44 | - | - | - | 1112 |
| single-16384-csa-cp8r0 | tilelang@main | 539.8 | 692.6 | 1.00 | 2182 | 1062 | 1.00 | 318 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 283.6 | 134.0 | 0.53 | - | - | - | 139 |
| single-16384-csa-cp8r0 | tilelang@cute | 535.5 | 693.6 | 1.00 | 2163 | 1126 | 1.00 | 318 |
| single-16384-csa-cp8r0 | cute@cute | 505.9 | 127.3 | 0.94 | 2137 | 498.4 | 0.99 | 318 |
| single-16384-csa-cp8r0 | cute_ws@cute | 253.8 | 54.0 | 0.47 | 1886 | 601.8 | 0.87 | 318 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 282.3 | 145.4 | 0.53 | - | - | - | 139 |
| single-16384-csa-cp8r4 | tilelang@main | 710.9 | 698.0 | 1.00 | 3186 | 893.3 | 1.00 | 318 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 359.6 | 135.6 | 0.51 | - | - | - | 139 |
| single-16384-csa-cp8r4 | tilelang@cute | 710.7 | 688.4 | 1.00 | 3154 | 991.1 | 1.00 | 318 |
| single-16384-csa-cp8r4 | cute@cute | 676.1 | 122.1 | 0.95 | 3123 | 367.2 | 0.99 | 318 |
| single-16384-csa-cp8r4 | cute_ws@cute | 328.8 | 53.8 | 0.46 | 2783 | 567.5 | 0.88 | 318 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 359.3 | 138.5 | 0.51 | - | - | - | 139 |
| single-16384-csa-cp8r7 | tilelang@main | 711.0 | 694.1 | 1.00 | 3201 | 903.7 | 1.00 | 318 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 358.9 | 137.7 | 0.50 | - | - | - | 139 |
| single-16384-csa-cp8r7 | tilelang@cute | 709.0 | 698.2 | 1.00 | 3164 | 941.0 | 1.00 | 318 |
| single-16384-csa-cp8r7 | cute@cute | 675.4 | 120.1 | 0.95 | 3137 | 323.5 | 0.99 | 318 |
| single-16384-csa-cp8r7 | cute_ws@cute | 329.2 | 53.1 | 0.46 | 2783 | 538.2 | 0.88 | 318 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 358.7 | 139.0 | 0.51 | - | - | - | 139 |
| single-16384-hca-cp1 | tilelang@main | 2873 | 700.6 | 1.00 | 10008 | 896.7 | 1.00 | 2108 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1399 | 90.2 | 0.49 | - | - | - | 1064 |
| single-16384-hca-cp1 | tilelang@cute | 2888 | 651.2 | 1.00 | 9965 | 901.7 | 1.00 | 2108 |
| single-16384-hca-cp1 | cute@cute | 2699 | 92.4 | 0.93 | 9764 | 287.7 | 0.98 | 2108 |
| single-16384-hca-cp1 | cute_ws@cute | 1210 | 51.5 | 0.42 | 8295 | 128.4 | 0.83 | 2108 |
| single-16384-hca-cp1 | flashmla_fwd_ref@cute | 1403 | 87.4 | 0.49 | - | - | - | 1064 |
| single-16384-hca-cp8r0 | tilelang@main | 360.1 | 720.7 | 1.00 | 1178 | 1245 | 1.00 | 306 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 196.9 | 143.6 | 0.55 | - | - | - | 133 |
| single-16384-hca-cp8r0 | tilelang@cute | 358.2 | 753.3 | 1.00 | 1174 | 1293 | 1.00 | 306 |
| single-16384-hca-cp8r0 | cute@cute | 332.8 | 134.9 | 0.93 | 1152 | 684.2 | 0.98 | 306 |
| single-16384-hca-cp8r0 | cute_ws@cute | 170.3 | 54.0 | 0.48 | 988.4 | 696.4 | 0.84 | 306 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 195.8 | 156.9 | 0.55 | - | - | - | 133 |
| single-16384-hca-cp8r4 | tilelang@main | 411.5 | 685.4 | 1.00 | 1468 | 1197 | 1.00 | 306 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 200.6 | 139.3 | 0.49 | - | - | - | 133 |
| single-16384-hca-cp8r4 | tilelang@cute | 411.2 | 685.0 | 1.00 | 1469 | 1252 | 1.00 | 306 |
| single-16384-hca-cp8r4 | cute@cute | 386.2 | 120.5 | 0.94 | 1443 | 663.5 | 0.98 | 306 |
| single-16384-hca-cp8r4 | cute_ws@cute | 173.7 | 47.6 | 0.42 | 1232 | 702.9 | 0.84 | 306 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 202.0 | 139.4 | 0.49 | - | - | - | 133 |
| single-16384-hca-cp8r7 | tilelang@main | 416.3 | 690.1 | 1.00 | 1589 | 1230 | 1.00 | 306 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 200.7 | 143.1 | 0.48 | - | - | - | 133 |
| single-16384-hca-cp8r7 | tilelang@cute | 415.4 | 690.4 | 1.00 | 1588 | 1242 | 1.00 | 306 |
| single-16384-hca-cp8r7 | cute@cute | 389.2 | 127.6 | 0.94 | 1565 | 626.0 | 0.99 | 306 |
| single-16384-hca-cp8r7 | cute_ws@cute | 171.5 | 55.2 | 0.41 | 1352 | 704.2 | 0.85 | 306 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 200.9 | 149.3 | 0.48 | - | - | - | 133 |
| single-16384-sliding-cp1 | tilelang@main | 2290 | 682.5 | 1.00 | 7470 | 816.3 | 1.00 | 2108 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 1049 | 115.4 | 0.46 | - | - | - | 1048 |
| single-16384-sliding-cp1 | tilelang@cute | 2292 | 687.5 | 1.00 | 7436 | 858.6 | 1.00 | 2108 |
| single-16384-sliding-cp1 | cute@cute | 2113 | 111.2 | 0.92 | 7261 | 241.2 | 0.98 | 2108 |
| single-16384-sliding-cp1 | cute_ws@cute | 795.8 | 58.0 | 0.35 | 5948 | 131.8 | 0.80 | 2108 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1045 | 124.7 | 0.46 | - | - | - | 1048 |
| single-16384-sliding-cp8r0 | tilelang@main | 310.8 | 707.7 | 1.00 | 1046 | 1288 | 1.00 | 306 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 150.5 | 149.3 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r0 | tilelang@cute | 312.5 | 701.1 | 1.00 | 1040 | 1332 | 1.00 | 306 |
| single-16384-sliding-cp8r0 | cute@cute | 288.7 | 126.0 | 0.92 | 1020 | 712.9 | 0.98 | 306 |
| single-16384-sliding-cp8r0 | cute_ws@cute | 116.1 | 54.9 | 0.37 | 853.8 | 731.9 | 0.82 | 306 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 150.2 | 149.4 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r4 | tilelang@main | 315.2 | 689.1 | 1.00 | 1080 | 1310 | 1.00 | 306 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 150.5 | 146.2 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r4 | tilelang@cute | 314.9 | 697.2 | 1.00 | 1081 | 1455 | 1.00 | 306 |
| single-16384-sliding-cp8r4 | cute@cute | 292.3 | 126.1 | 0.93 | 1059 | 873.3 | 0.98 | 306 |
| single-16384-sliding-cp8r4 | cute_ws@cute | 118.8 | 51.3 | 0.38 | 884.4 | 921.4 | 0.82 | 306 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 150.3 | 151.0 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r7 | tilelang@main | 314.2 | 682.0 | 1.00 | 1084 | 1305 | 1.00 | 306 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 150.6 | 143.4 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r7 | tilelang@cute | 313.2 | 695.0 | 1.00 | 1084 | 1448 | 1.00 | 306 |
| single-16384-sliding-cp8r7 | cute@cute | 292.5 | 127.2 | 0.93 | 1061 | 870.7 | 0.98 | 306 |
| single-16384-sliding-cp8r7 | cute_ws@cute | 118.3 | 53.2 | 0.38 | 889.8 | 903.4 | 0.82 | 306 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 148.6 | 152.6 | 0.47 | - | - | - | 131 |
| short-16384-csa-cp1 | tilelang@main | 3853 | 656.0 | 1.00 | 15075 | 901.5 | 1.00 | 2120 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 1929 | 28.0 | 0.50 | - | - | - | 1112 |
| short-16384-csa-cp1 | tilelang@cute | 3906 | 598.2 | 1.00 | 14981 | 933.5 | 1.00 | 2120 |
| short-16384-csa-cp1 | cute@cute | 3678 | 27.9 | 0.94 | 14781 | 297.3 | 0.99 | 2120 |
| short-16384-csa-cp1 | cute_ws@cute | 1728 | 56.0 | 0.44 | 12931 | 193.3 | 0.86 | 2120 |
| short-16384-csa-cp1 | flashmla_fwd_ref@cute | 1916 | 62.8 | 0.49 | - | - | - | 1112 |
| short-16384-csa-cp8r0 | tilelang@main | 478.1 | 697.2 | 1.00 | 1852 | 1145 | 1.00 | 317 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 254.2 | 135.9 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r0 | tilelang@cute | 474.3 | 701.9 | 1.00 | 1843 | 1185 | 1.00 | 317 |
| short-16384-csa-cp8r0 | cute@cute | 448.5 | 125.3 | 0.95 | 1817 | 550.9 | 0.99 | 317 |
| short-16384-csa-cp8r0 | cute_ws@cute | 223.9 | 54.5 | 0.47 | 1594 | 628.7 | 0.86 | 317 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 253.3 | 136.1 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r4 | tilelang@main | 566.4 | 683.1 | 1.00 | 2358 | 1029 | 1.00 | 317 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 295.5 | 136.1 | 0.52 | - | - | - | 139 |
| short-16384-csa-cp8r4 | tilelang@cute | 565.6 | 682.0 | 1.00 | 2357 | 1114 | 1.00 | 317 |
| short-16384-csa-cp8r4 | cute@cute | 536.0 | 120.3 | 0.95 | 2328 | 486.7 | 0.99 | 317 |
| short-16384-csa-cp8r4 | cute_ws@cute | 264.8 | 54.1 | 0.47 | 2061 | 593.6 | 0.87 | 317 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 295.3 | 138.9 | 0.52 | - | - | - | 139 |
| short-16384-csa-cp8r7 | tilelang@main | 524.8 | 685.9 | 1.00 | 2140 | 1101 | 1.00 | 317 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 276.9 | 131.2 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r7 | tilelang@cute | 524.1 | 705.5 | 1.00 | 2141 | 1124 | 1.00 | 317 |
| short-16384-csa-cp8r7 | cute@cute | 497.0 | 126.6 | 0.95 | 2108 | 506.2 | 0.98 | 317 |
| short-16384-csa-cp8r7 | cute_ws@cute | 252.6 | 56.1 | 0.48 | 1863 | 600.3 | 0.87 | 317 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 276.4 | 144.5 | 0.53 | - | - | - | 139 |
| short-16384-hca-cp1 | tilelang@main | 2599 | 738.1 | 1.00 | 8299 | 926.1 | 1.00 | 2120 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 1361 | 102.2 | 0.52 | - | - | - | 1064 |
| short-16384-hca-cp1 | tilelang@cute | 2606 | 721.0 | 1.00 | 8236 | 991.4 | 1.00 | 2120 |
| short-16384-hca-cp1 | cute@cute | 2433 | 125.0 | 0.93 | 8063 | 308.4 | 0.98 | 2120 |
| short-16384-hca-cp1 | cute_ws@cute | 1194 | 48.1 | 0.46 | 6821 | 168.3 | 0.83 | 2120 |
| short-16384-hca-cp1 | flashmla_fwd_ref@cute | 1373 | 89.6 | 0.53 | - | - | - | 1064 |
| short-16384-hca-cp8r0 | tilelang@main | 349.8 | 720.5 | 1.00 | 1148 | 1351 | 1.00 | 307 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 196.9 | 142.0 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r0 | tilelang@cute | 347.2 | 730.4 | 1.00 | 1150 | 1385 | 1.00 | 307 |
| short-16384-hca-cp8r0 | cute@cute | 326.2 | 149.1 | 0.94 | 1127 | 777.5 | 0.98 | 307 |
| short-16384-hca-cp8r0 | cute_ws@cute | 165.1 | 55.1 | 0.48 | 966.1 | 743.3 | 0.84 | 307 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 195.0 | 147.2 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r4 | tilelang@main | 362.1 | 717.9 | 1.00 | 1211 | 1322 | 1.00 | 307 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 200.7 | 136.7 | 0.55 | - | - | - | 133 |
| short-16384-hca-cp8r4 | tilelang@cute | 364.2 | 720.9 | 1.00 | 1210 | 1397 | 1.00 | 307 |
| short-16384-hca-cp8r4 | cute@cute | 339.6 | 146.8 | 0.93 | 1187 | 771.6 | 0.98 | 307 |
| short-16384-hca-cp8r4 | cute_ws@cute | 169.1 | 53.3 | 0.46 | 1024 | 738.6 | 0.85 | 307 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 198.5 | 150.5 | 0.54 | - | - | - | 133 |
| short-16384-hca-cp8r7 | tilelang@main | 355.1 | 726.0 | 1.00 | 1179 | 1380 | 1.00 | 307 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 198.1 | 140.8 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r7 | tilelang@cute | 354.0 | 727.9 | 1.00 | 1178 | 1393 | 1.00 | 307 |
| short-16384-hca-cp8r7 | cute@cute | 331.2 | 145.7 | 0.94 | 1153 | 781.4 | 0.98 | 307 |
| short-16384-hca-cp8r7 | cute_ws@cute | 166.9 | 51.4 | 0.47 | 992.7 | 743.3 | 0.84 | 307 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 197.8 | 148.2 | 0.56 | - | - | - | 133 |
| short-16384-sliding-cp1 | tilelang@main | 2264 | 720.8 | 1.00 | 7323 | 832.1 | 1.00 | 2108 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 1041 | 128.7 | 0.46 | - | - | - | 1048 |
| short-16384-sliding-cp1 | tilelang@cute | 2269 | 690.1 | 1.00 | 7292 | 863.9 | 1.00 | 2108 |
| short-16384-sliding-cp1 | cute@cute | 2097 | 117.6 | 0.92 | 7126 | 231.4 | 0.98 | 2108 |
| short-16384-sliding-cp1 | cute_ws@cute | 791.2 | 62.8 | 0.35 | 5832 | 127.9 | 0.80 | 2108 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1041 | 132.2 | 0.46 | - | - | - | 1048 |
| short-16384-sliding-cp8r0 | tilelang@main | 303.6 | 697.8 | 1.00 | 1025 | 1281 | 1.00 | 306 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 149.7 | 146.6 | 0.49 | - | - | - | 131 |
| short-16384-sliding-cp8r0 | tilelang@cute | 302.9 | 715.4 | 1.00 | 1025 | 1362 | 1.00 | 306 |
| short-16384-sliding-cp8r0 | cute@cute | 282.2 | 123.4 | 0.93 | 1006 | 768.3 | 0.98 | 306 |
| short-16384-sliding-cp8r0 | cute_ws@cute | 118.6 | 51.5 | 0.39 | 841.6 | 734.9 | 0.82 | 306 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 149.8 | 154.0 | 0.49 | - | - | - | 131 |
| short-16384-sliding-cp8r4 | tilelang@main | 314.2 | 679.7 | 1.00 | 1072 | 1298 | 1.00 | 306 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 147.1 | 147.3 | 0.47 | - | - | - | 131 |
| short-16384-sliding-cp8r4 | tilelang@cute | 314.7 | 763.4 | 1.00 | 1073 | 1337 | 1.00 | 306 |
| short-16384-sliding-cp8r4 | cute@cute | 292.9 | 129.3 | 0.93 | 1045 | 777.9 | 0.97 | 306 |
| short-16384-sliding-cp8r4 | cute_ws@cute | 118.0 | 54.5 | 0.38 | 875.5 | 738.9 | 0.82 | 306 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 149.2 | 165.1 | 0.47 | - | - | - | 131 |
| short-16384-sliding-cp8r7 | tilelang@main | 312.3 | 689.1 | 1.00 | 1051 | 1310 | 1.00 | 306 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 150.9 | 143.6 | 0.48 | - | - | - | 131 |
| short-16384-sliding-cp8r7 | tilelang@cute | 313.5 | 725.3 | 1.00 | 1051 | 1338 | 1.00 | 306 |
| short-16384-sliding-cp8r7 | cute@cute | 291.5 | 127.5 | 0.93 | 1030 | 725.6 | 0.98 | 306 |
| short-16384-sliding-cp8r7 | cute_ws@cute | 117.8 | 55.2 | 0.38 | 861.9 | 735.4 | 0.82 | 306 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 149.9 | 153.9 | 0.48 | - | - | - | 131 |
| heavy-16384-csa-cp1 | tilelang@main | 3305 | 664.5 | 1.00 | 12082 | 914.7 | 1.00 | 2120 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1691 | 34.9 | 0.51 | - | - | - | 1112 |
| heavy-16384-csa-cp1 | tilelang@cute | 3316 | 644.2 | 1.00 | 12011 | 917.7 | 1.00 | 2120 |
| heavy-16384-csa-cp1 | cute@cute | 3118 | 74.5 | 0.94 | 11832 | 277.8 | 0.99 | 2120 |
| heavy-16384-csa-cp1 | cute_ws@cute | 1503 | 54.4 | 0.45 | 10217 | 209.1 | 0.85 | 2120 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@cute | 1690 | 45.9 | 0.51 | - | - | - | 1112 |
| heavy-16384-csa-cp8r0 | tilelang@main | 395.5 | 697.0 | 1.00 | 1364 | 1211 | 1.00 | 317 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 213.5 | 133.0 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r0 | tilelang@cute | 394.9 | 701.8 | 1.00 | 1366 | 1244 | 1.00 | 317 |
| heavy-16384-csa-cp8r0 | cute@cute | 371.1 | 119.0 | 0.94 | 1342 | 630.6 | 0.98 | 317 |
| heavy-16384-csa-cp8r0 | cute_ws@cute | 187.7 | 52.4 | 0.48 | 1160 | 661.6 | 0.85 | 317 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 215.9 | 138.9 | 0.55 | - | - | - | 139 |
| heavy-16384-csa-cp8r4 | tilelang@main | 433.6 | 683.1 | 1.00 | 1614 | 1148 | 1.00 | 317 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 232.8 | 129.9 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r4 | tilelang@cute | 434.4 | 696.6 | 1.00 | 1610 | 1230 | 1.00 | 317 |
| heavy-16384-csa-cp8r4 | cute@cute | 409.5 | 121.5 | 0.94 | 1585 | 597.4 | 0.98 | 317 |
| heavy-16384-csa-cp8r4 | cute_ws@cute | 204.4 | 51.7 | 0.47 | 1384 | 652.4 | 0.86 | 317 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 234.5 | 140.3 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r7 | tilelang@main | 579.5 | 684.9 | 1.00 | 2413 | 1069 | 1.00 | 317 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 301.6 | 126.9 | 0.52 | - | - | - | 139 |
| heavy-16384-csa-cp8r7 | tilelang@cute | 577.1 | 713.7 | 1.00 | 2417 | 1088 | 1.00 | 317 |
| heavy-16384-csa-cp8r7 | cute@cute | 550.4 | 121.8 | 0.95 | 2388 | 467.3 | 0.99 | 317 |
| heavy-16384-csa-cp8r7 | cute_ws@cute | 270.4 | 52.7 | 0.47 | 2107 | 589.4 | 0.87 | 317 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 301.8 | 141.9 | 0.52 | - | - | - | 139 |
| heavy-16384-hca-cp1 | tilelang@main | 2507 | 744.8 | 1.00 | 7784 | 921.8 | 1.00 | 2120 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 1309 | 94.9 | 0.52 | - | - | - | 1064 |
| heavy-16384-hca-cp1 | tilelang@cute | 2493 | 730.3 | 1.00 | 7754 | 939.5 | 1.00 | 2120 |
| heavy-16384-hca-cp1 | cute@cute | 2318 | 136.1 | 0.93 | 7572 | 281.5 | 0.98 | 2120 |
| heavy-16384-hca-cp1 | cute_ws@cute | 1104 | 58.6 | 0.44 | 6370 | 131.1 | 0.82 | 2120 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@cute | 1303 | 104.2 | 0.52 | - | - | - | 1064 |
| heavy-16384-hca-cp8r0 | tilelang@main | 333.5 | 734.1 | 1.00 | 1069 | 1379 | 1.00 | 308 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 184.9 | 150.6 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r0 | tilelang@cute | 333.5 | 738.1 | 1.00 | 1072 | 1391 | 1.00 | 308 |
| heavy-16384-hca-cp8r0 | cute@cute | 311.3 | 147.5 | 0.93 | 1048 | 779.0 | 0.98 | 308 |
| heavy-16384-hca-cp8r0 | cute_ws@cute | 152.2 | 53.1 | 0.46 | 891.7 | 738.4 | 0.83 | 308 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 185.6 | 148.0 | 0.56 | - | - | - | 133 |
| heavy-16384-hca-cp8r4 | tilelang@main | 343.8 | 725.9 | 1.00 | 1127 | 1361 | 1.00 | 308 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 190.5 | 151.2 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r4 | tilelang@cute | 343.8 | 726.1 | 1.00 | 1128 | 1401 | 1.00 | 308 |
| heavy-16384-hca-cp8r4 | cute@cute | 322.6 | 147.0 | 0.94 | 1107 | 762.3 | 0.98 | 308 |
| heavy-16384-hca-cp8r4 | cute_ws@cute | 161.1 | 55.2 | 0.47 | 949.7 | 733.9 | 0.84 | 308 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 192.4 | 151.6 | 0.56 | - | - | - | 133 |
| heavy-16384-hca-cp8r7 | tilelang@main | 366.8 | 721.2 | 1.00 | 1231 | 1390 | 1.00 | 308 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 203.8 | 142.3 | 0.56 | - | - | - | 133 |
| heavy-16384-hca-cp8r7 | tilelang@cute | 367.2 | 743.9 | 1.00 | 1228 | 1349 | 1.00 | 308 |
| heavy-16384-hca-cp8r7 | cute@cute | 342.1 | 152.1 | 0.93 | 1202 | 750.7 | 0.98 | 308 |
| heavy-16384-hca-cp8r7 | cute_ws@cute | 172.2 | 55.9 | 0.47 | 1038 | 717.6 | 0.85 | 308 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 204.5 | 151.3 | 0.56 | - | - | - | 133 |
| heavy-16384-sliding-cp1 | tilelang@main | 2247 | 731.8 | 1.00 | 7046 | 841.1 | 1.00 | 2108 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 1042 | 130.6 | 0.46 | - | - | - | 1048 |
| heavy-16384-sliding-cp1 | tilelang@cute | 2250 | 690.7 | 1.00 | 7030 | 862.2 | 1.00 | 2108 |
| heavy-16384-sliding-cp1 | cute@cute | 2082 | 117.3 | 0.93 | 6845 | 253.4 | 0.97 | 2108 |
| heavy-16384-sliding-cp1 | cute_ws@cute | 801.3 | 67.1 | 0.36 | 5586 | 144.0 | 0.79 | 2108 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1051 | 125.1 | 0.47 | - | - | - | 1048 |
| heavy-16384-sliding-cp8r0 | tilelang@main | 300.8 | 712.1 | 1.00 | 974.6 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 148.3 | 149.4 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r0 | tilelang@cute | 300.0 | 706.8 | 1.00 | 975.1 | 1343 | 1.00 | 306 |
| heavy-16384-sliding-cp8r0 | cute@cute | 277.6 | 124.4 | 0.93 | 953.4 | 736.2 | 0.98 | 306 |
| heavy-16384-sliding-cp8r0 | cute_ws@cute | 117.2 | 52.7 | 0.39 | 793.7 | 738.8 | 0.81 | 306 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 149.0 | 150.4 | 0.50 | - | - | - | 131 |
| heavy-16384-sliding-cp8r4 | tilelang@main | 303.7 | 679.0 | 1.00 | 1014 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 149.5 | 143.6 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r4 | tilelang@cute | 303.1 | 683.1 | 1.00 | 1016 | 1328 | 1.00 | 306 |
| heavy-16384-sliding-cp8r4 | cute@cute | 284.2 | 124.6 | 0.94 | 995.4 | 712.1 | 0.98 | 306 |
| heavy-16384-sliding-cp8r4 | cute_ws@cute | 116.4 | 53.8 | 0.38 | 834.3 | 732.7 | 0.82 | 306 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 150.5 | 149.6 | 0.50 | - | - | - | 131 |
| heavy-16384-sliding-cp8r7 | tilelang@main | 314.8 | 707.2 | 1.00 | 1082 | 1323 | 1.00 | 306 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 147.4 | 149.0 | 0.47 | - | - | - | 131 |
| heavy-16384-sliding-cp8r7 | tilelang@cute | 315.6 | 685.5 | 1.00 | 1085 | 1325 | 1.00 | 306 |
| heavy-16384-sliding-cp8r7 | cute@cute | 294.4 | 124.5 | 0.93 | 1061 | 706.0 | 0.98 | 306 |
| heavy-16384-sliding-cp8r7 | cute_ws@cute | 117.2 | 53.8 | 0.37 | 888.7 | 723.2 | 0.82 | 306 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 151.3 | 147.7 | 0.48 | - | - | - | 131 |
| tiny-16384-csa-cp1 | tilelang@main | 2636 | 650.2 | 1.00 | 7892 | 833.7 | 1.00 | 2120 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 1437 | 39.5 | 0.54 | - | - | - | 1112 |
| tiny-16384-csa-cp1 | tilelang@cute | 2636 | 679.0 | 1.00 | 7886 | 828.0 | 1.00 | 2120 |
| tiny-16384-csa-cp1 | cute@cute | 2466 | 78.0 | 0.94 | 7725 | 175.6 | 0.98 | 2120 |
| tiny-16384-csa-cp1 | cute_ws@cute | 1242 | 59.6 | 0.47 | 6500 | 114.2 | 0.82 | 2120 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@cute | 1441 | 36.2 | 0.55 | - | - | - | 1112 |
| tiny-16384-csa-cp8r0 | tilelang@main | 354.2 | 688.8 | 1.00 | 1109 | 1267 | 1.00 | 317 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 200.1 | 134.4 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r0 | tilelang@cute | 352.6 | 683.3 | 1.00 | 1110 | 1282 | 1.00 | 317 |
| tiny-16384-csa-cp8r0 | cute@cute | 332.0 | 121.1 | 0.94 | 1086 | 678.3 | 0.98 | 317 |
| tiny-16384-csa-cp8r0 | cute_ws@cute | 175.3 | 53.2 | 0.50 | 930.5 | 683.1 | 0.84 | 317 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 199.8 | 143.9 | 0.57 | - | - | - | 139 |
| tiny-16384-csa-cp8r4 | tilelang@main | 356.2 | 692.0 | 1.00 | 1108 | 1250 | 1.00 | 317 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 200.5 | 141.7 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r4 | tilelang@cute | 356.2 | 692.0 | 1.00 | 1106 | 1299 | 1.00 | 317 |
| tiny-16384-csa-cp8r4 | cute@cute | 334.6 | 122.6 | 0.94 | 1085 | 679.1 | 0.98 | 317 |
| tiny-16384-csa-cp8r4 | cute_ws@cute | 175.3 | 52.7 | 0.49 | 927.7 | 687.5 | 0.84 | 317 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 199.7 | 144.4 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r7 | tilelang@main | 356.2 | 675.8 | 1.00 | 1099 | 1270 | 1.00 | 317 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 200.3 | 134.2 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r7 | tilelang@cute | 356.7 | 700.7 | 1.00 | 1097 | 1295 | 1.00 | 317 |
| tiny-16384-csa-cp8r7 | cute@cute | 334.3 | 122.2 | 0.94 | 1076 | 673.9 | 0.98 | 317 |
| tiny-16384-csa-cp8r7 | cute_ws@cute | 176.4 | 52.5 | 0.49 | 917.9 | 681.2 | 0.84 | 317 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 200.5 | 142.8 | 0.56 | - | - | - | 139 |
| tiny-16384-hca-cp1 | tilelang@main | 2024 | 683.5 | 1.00 | 5355 | 828.5 | 1.00 | 2108 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 1052 | 124.0 | 0.52 | - | - | - | 1048 |
| tiny-16384-hca-cp1 | tilelang@cute | 2025 | 712.0 | 1.00 | 5354 | 839.6 | 1.00 | 2108 |
| tiny-16384-hca-cp1 | cute@cute | 1863 | 126.7 | 0.92 | 5192 | 220.5 | 0.97 | 2108 |
| tiny-16384-hca-cp1 | cute_ws@cute | 927.4 | 56.0 | 0.46 | 4258 | 107.8 | 0.80 | 2108 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@cute | 1062 | 129.1 | 0.52 | - | - | - | 1048 |
| tiny-16384-hca-cp8r0 | tilelang@main | 278.4 | 702.7 | 1.00 | 784.0 | 1360 | 1.00 | 306 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 149.1 | 147.5 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r0 | tilelang@cute | 278.8 | 713.0 | 1.00 | 785.0 | 1393 | 1.00 | 306 |
| tiny-16384-hca-cp8r0 | cute@cute | 257.1 | 132.6 | 0.92 | 763.9 | 748.8 | 0.97 | 306 |
| tiny-16384-hca-cp8r0 | cute_ws@cute | 114.8 | 57.2 | 0.41 | 632.2 | 727.1 | 0.81 | 306 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 147.8 | 160.4 | 0.53 | - | - | - | 131 |
| tiny-16384-hca-cp8r4 | tilelang@main | 276.8 | 691.4 | 1.00 | 771.3 | 1324 | 1.00 | 306 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 148.8 | 144.3 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r4 | tilelang@cute | 276.1 | 706.2 | 1.00 | 771.3 | 1391 | 1.00 | 306 |
| tiny-16384-hca-cp8r4 | cute@cute | 255.5 | 126.4 | 0.93 | 750.0 | 758.0 | 0.97 | 306 |
| tiny-16384-hca-cp8r4 | cute_ws@cute | 114.1 | 53.5 | 0.41 | 621.7 | 735.9 | 0.81 | 306 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 149.1 | 152.6 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r7 | tilelang@main | 268.8 | 686.6 | 1.00 | 754.2 | 1376 | 1.00 | 306 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 149.0 | 145.3 | 0.55 | - | - | - | 131 |
| tiny-16384-hca-cp8r7 | tilelang@cute | 269.1 | 694.2 | 1.00 | 752.7 | 1420 | 1.00 | 306 |
| tiny-16384-hca-cp8r7 | cute@cute | 247.8 | 126.7 | 0.92 | 730.0 | 765.5 | 0.97 | 306 |
| tiny-16384-hca-cp8r7 | cute_ws@cute | 114.9 | 53.6 | 0.43 | 610.9 | 732.5 | 0.81 | 306 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 148.5 | 159.2 | 0.55 | - | - | - | 131 |
| tiny-16384-sliding-cp1 | tilelang@main | 2025 | 703.4 | 1.00 | 5346 | 817.0 | 1.00 | 2108 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 1050 | 130.1 | 0.52 | - | - | - | 1048 |
| tiny-16384-sliding-cp1 | tilelang@cute | 2030 | 701.4 | 1.00 | 5343 | 881.2 | 1.00 | 2108 |
| tiny-16384-sliding-cp1 | cute@cute | 1865 | 111.7 | 0.92 | 5184 | 230.2 | 0.97 | 2108 |
| tiny-16384-sliding-cp1 | cute_ws@cute | 934.5 | 48.1 | 0.46 | 4228 | 145.6 | 0.79 | 2108 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@cute | 1051 | 133.0 | 0.52 | - | - | - | 1048 |
| tiny-16384-sliding-cp8r0 | tilelang@main | 278.4 | 684.2 | 1.00 | 784.9 | 1357 | 1.00 | 306 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 147.7 | 149.3 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r0 | tilelang@cute | 278.3 | 769.7 | 1.00 | 785.7 | 1389 | 1.00 | 306 |
| tiny-16384-sliding-cp8r0 | cute@cute | 257.9 | 131.0 | 0.93 | 764.4 | 762.9 | 0.97 | 306 |
| tiny-16384-sliding-cp8r0 | cute_ws@cute | 115.4 | 55.1 | 0.41 | 632.2 | 745.9 | 0.80 | 306 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 148.6 | 155.2 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r4 | tilelang@main | 276.5 | 685.8 | 1.00 | 772.0 | 1402 | 1.00 | 306 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 147.7 | 151.4 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r4 | tilelang@cute | 275.4 | 694.2 | 1.00 | 773.0 | 1383 | 1.00 | 306 |
| tiny-16384-sliding-cp8r4 | cute@cute | 255.5 | 125.5 | 0.93 | 750.9 | 748.3 | 0.97 | 306 |
| tiny-16384-sliding-cp8r4 | cute_ws@cute | 113.5 | 54.5 | 0.41 | 624.0 | 725.2 | 0.81 | 306 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 149.0 | 148.1 | 0.54 | - | - | - | 131 |
| tiny-16384-sliding-cp8r7 | tilelang@main | 267.7 | 693.4 | 1.00 | 754.5 | 1385 | 1.00 | 306 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 148.1 | 145.0 | 0.55 | - | - | - | 131 |
| tiny-16384-sliding-cp8r7 | tilelang@cute | 268.3 | 695.0 | 1.00 | 753.7 | 1383 | 1.00 | 306 |
| tiny-16384-sliding-cp8r7 | cute@cute | 246.1 | 129.7 | 0.92 | 732.3 | 791.2 | 0.97 | 306 |
| tiny-16384-sliding-cp8r7 | cute_ws@cute | 115.0 | 54.1 | 0.43 | 612.6 | 737.0 | 0.81 | 306 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 147.7 | 159.2 | 0.55 | - | - | - | 131 |
| single-49208-csa-cp1 | tilelang@main | 16237 | 187.0 | 1.00 | 69993 | 715.1 | 1.00 | 6368 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 7737 | 571.7 | 0.48 | - | - | - | 3340 |
| single-49208-csa-cp1 | tilelang@cute | 16384 | 6.7 | 1.00 | 70013 | 836.0 | 1.00 | 6368 |
| single-49208-csa-cp1 | cute@cute | 15107 | 39.9 | 0.92 | 69333 | 131.3 | 0.99 | 6368 |
| single-49208-csa-cp1 | cute_ws@cute | 7271 | 600.6 | 0.44 | 61535 | 228.0 | 0.88 | 6368 |
| single-49208-csa-cp1 | flashmla_fwd_ref@cute | 8281 | 345.0 | 0.51 | - | - | - | 3340 |
| single-49208-csa-cp8r0 | tilelang@main | 1866 | 689.5 | 1.00 | 8084 | 839.4 | 1.00 | 954 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 922.1 | 104.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r0 | tilelang@cute | 1873 | 707.9 | 1.00 | 8080 | 877.8 | 1.00 | 954 |
| single-49208-csa-cp8r0 | cute@cute | 1779 | 109.1 | 0.95 | 7972 | 246.5 | 0.99 | 954 |
| single-49208-csa-cp8r0 | cute_ws@cute | 847.8 | 57.9 | 0.45 | 7071 | 125.5 | 0.88 | 954 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 923.5 | 108.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r4 | tilelang@main | 2035 | 709.5 | 1.00 | 9123 | 902.5 | 1.00 | 954 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 999.9 | 100.6 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r4 | tilelang@cute | 2052 | 710.5 | 1.00 | 9128 | 948.1 | 1.00 | 954 |
| single-49208-csa-cp8r4 | cute@cute | 1953 | 114.9 | 0.95 | 9023 | 298.1 | 0.99 | 954 |
| single-49208-csa-cp8r4 | cute_ws@cute | 929.5 | 56.5 | 0.45 | 8012 | 149.6 | 0.88 | 954 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 1001 | 111.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r7 | tilelang@main | 2039 | 705.7 | 1.00 | 9444 | 917.3 | 1.00 | 954 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1003 | 101.6 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r7 | tilelang@cute | 2049 | 722.2 | 1.00 | 9470 | 920.2 | 1.00 | 954 |
| single-49208-csa-cp8r7 | cute@cute | 1950 | 117.9 | 0.95 | 9425 | 214.0 | 1.00 | 954 |
| single-49208-csa-cp8r7 | cute_ws@cute | 925.4 | 62.8 | 0.45 | 8249 | 248.8 | 0.87 | 954 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 1003 | 112.3 | 0.49 | - | - | - | 418 |
| single-49208-hca-cp1 | tilelang@main | 11188 | 307.9 | 1.00 | 41810 | 748.7 | 1.00 | 6334 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 5350 | 51.8 | 0.48 | - | - | - | 3292 |
| single-49208-hca-cp1 | tilelang@cute | 11066 | 436.2 | 1.00 | 41786 | 769.6 | 1.00 | 6334 |
| single-49208-hca-cp1 | cute@cute | 10249 | 56.1 | 0.93 | 41203 | 123.0 | 0.99 | 6334 |
| single-49208-hca-cp1 | cute_ws@cute | 4758 | 252.2 | 0.43 | 35705 | 187.9 | 0.85 | 6334 |
| single-49208-hca-cp1 | flashmla_fwd_ref@cute | 5587 | 83.0 | 0.50 | - | - | - | 3292 |
| single-49208-hca-cp8r0 | tilelang@main | 1029 | 680.6 | 1.00 | 3422 | 825.2 | 1.00 | 919 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 555.3 | 115.3 | 0.54 | - | - | - | 412 |
| single-49208-hca-cp8r0 | tilelang@cute | 1026 | 681.1 | 1.00 | 3434 | 835.8 | 1.00 | 919 |
| single-49208-hca-cp8r0 | cute@cute | 956.4 | 115.6 | 0.93 | 3349 | 233.6 | 0.98 | 919 |
| single-49208-hca-cp8r0 | cute_ws@cute | 472.3 | 58.4 | 0.46 | 2878 | 157.1 | 0.84 | 919 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 553.4 | 122.5 | 0.54 | - | - | - | 412 |
| single-49208-hca-cp8r4 | tilelang@main | 1459 | 686.0 | 1.00 | 5739 | 825.7 | 1.00 | 919 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 691.9 | 109.3 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r4 | tilelang@cute | 1461 | 694.2 | 1.00 | 5740 | 846.7 | 1.00 | 919 |
| single-49208-hca-cp8r4 | cute@cute | 1377 | 116.0 | 0.94 | 5657 | 219.7 | 0.99 | 919 |
| single-49208-hca-cp8r4 | cute_ws@cute | 619.2 | 53.4 | 0.42 | 4882 | 132.3 | 0.85 | 919 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 687.9 | 121.7 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r7 | tilelang@main | 1762 | 668.0 | 1.00 | 7389 | 854.6 | 1.00 | 919 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 831.5 | 113.1 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r7 | tilelang@cute | 1764 | 689.3 | 1.00 | 7376 | 848.3 | 1.00 | 919 |
| single-49208-hca-cp8r7 | cute@cute | 1670 | 114.4 | 0.95 | 7303 | 211.8 | 0.99 | 919 |
| single-49208-hca-cp8r7 | cute_ws@cute | 765.3 | 55.1 | 0.43 | 6396 | 122.4 | 0.87 | 919 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 827.0 | 126.6 | 0.47 | - | - | - | 412 |
| single-49208-sliding-cp1 | tilelang@main | 6768 | 664.9 | 1.00 | 22045 | 843.6 | 1.00 | 6332 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 3083 | 41.8 | 0.46 | - | - | - | 3148 |
| single-49208-sliding-cp1 | tilelang@cute | 6811 | 662.0 | 1.00 | 22057 | 891.7 | 1.00 | 6332 |
| single-49208-sliding-cp1 | cute@cute | 6282 | 109.6 | 0.92 | 21554 | 262.5 | 0.98 | 6332 |
| single-49208-sliding-cp1 | cute_ws@cute | 2372 | 53.8 | 0.35 | 17635 | 188.1 | 0.80 | 6332 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3095 | 90.3 | 0.45 | - | - | - | 3148 |
| single-49208-sliding-cp8r0 | tilelang@main | 873.5 | 700.3 | 1.00 | 2906 | 838.8 | 1.00 | 918 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 402.7 | 140.4 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r0 | tilelang@cute | 874.3 | 706.4 | 1.00 | 2903 | 857.6 | 1.00 | 918 |
| single-49208-sliding-cp8r0 | cute@cute | 807.3 | 125.5 | 0.92 | 2834 | 238.4 | 0.98 | 918 |
| single-49208-sliding-cp8r0 | cute_ws@cute | 312.6 | 55.9 | 0.36 | 2354 | 309.5 | 0.81 | 918 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 403.6 | 140.3 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r4 | tilelang@main | 878.6 | 678.7 | 1.00 | 2949 | 833.9 | 1.00 | 918 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 404.1 | 135.1 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r4 | tilelang@cute | 879.1 | 686.0 | 1.00 | 2937 | 857.2 | 1.00 | 918 |
| single-49208-sliding-cp8r4 | cute@cute | 811.6 | 122.7 | 0.92 | 2879 | 230.7 | 0.98 | 918 |
| single-49208-sliding-cp8r4 | cute_ws@cute | 315.2 | 54.5 | 0.36 | 2385 | 321.9 | 0.81 | 918 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 406.2 | 140.8 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r7 | tilelang@main | 880.4 | 691.7 | 1.00 | 2941 | 850.3 | 1.00 | 918 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 401.2 | 139.0 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r7 | tilelang@cute | 879.5 | 699.1 | 1.00 | 2960 | 846.7 | 1.00 | 918 |
| single-49208-sliding-cp8r7 | cute@cute | 815.3 | 121.7 | 0.93 | 2892 | 230.0 | 0.98 | 918 |
| single-49208-sliding-cp8r7 | cute_ws@cute | 313.3 | 58.1 | 0.36 | 2390 | 319.3 | 0.81 | 918 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 403.4 | 144.2 | 0.46 | - | - | - | 394 |
| short-49208-csa-cp1 | tilelang@main | 12688 | 299.8 | 1.00 | 50333 | 736.6 | 1.00 | 6368 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 6166 | 60.7 | 0.49 | - | - | - | 3340 |
| short-49208-csa-cp1 | tilelang@cute | 12983 | 4.7 | 1.00 | 50329 | 779.3 | 1.00 | 6368 |
| short-49208-csa-cp1 | cute@cute | 11773 | 33.6 | 0.91 | 49721 | 125.3 | 0.99 | 6368 |
| short-49208-csa-cp1 | cute_ws@cute | 5595 | 417.4 | 0.43 | 43569 | 183.6 | 0.87 | 6368 |
| short-49208-csa-cp1 | flashmla_fwd_ref@cute | 6285 | 330.2 | 0.48 | - | - | - | 3340 |
| short-49208-csa-cp8r0 | tilelang@main | 1344 | 692.7 | 1.00 | 5188 | 824.3 | 1.00 | 954 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 686.4 | 104.7 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r0 | tilelang@cute | 1346 | 685.1 | 1.00 | 5195 | 841.1 | 1.00 | 954 |
| short-49208-csa-cp8r0 | cute@cute | 1273 | 113.2 | 0.95 | 5118 | 226.9 | 0.99 | 954 |
| short-49208-csa-cp8r0 | cute_ws@cute | 605.4 | 54.3 | 0.45 | 4449 | 128.8 | 0.86 | 954 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 687.7 | 108.8 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r4 | tilelang@main | 1711 | 674.8 | 1.00 | 7251 | 834.8 | 1.00 | 954 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 852.5 | 107.1 | 0.50 | - | - | - | 418 |
| short-49208-csa-cp8r4 | tilelang@cute | 1721 | 689.8 | 1.00 | 7246 | 857.3 | 1.00 | 954 |
| short-49208-csa-cp8r4 | cute@cute | 1636 | 111.4 | 0.95 | 7161 | 242.8 | 0.99 | 954 |
| short-49208-csa-cp8r4 | cute_ws@cute | 784.8 | 60.4 | 0.46 | 6337 | 129.1 | 0.87 | 954 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 858.0 | 109.3 | 0.50 | - | - | - | 418 |
| short-49208-csa-cp8r7 | tilelang@main | 1484 | 676.8 | 1.00 | 5954 | 855.8 | 1.00 | 954 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 756.8 | 101.5 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r7 | tilelang@cute | 1478 | 684.5 | 1.00 | 5963 | 825.5 | 1.00 | 954 |
| short-49208-csa-cp8r7 | cute@cute | 1399 | 112.2 | 0.95 | 5878 | 221.4 | 0.99 | 954 |
| short-49208-csa-cp8r7 | cute_ws@cute | 667.9 | 65.7 | 0.45 | 5153 | 130.9 | 0.86 | 954 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 753.6 | 109.5 | 0.51 | - | - | - | 418 |
| short-49208-hca-cp1 | tilelang@main | 7876 | 628.9 | 1.00 | 25343 | 831.9 | 1.00 | 6382 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 4089 | 14.0 | 0.52 | - | - | - | 3197 |
| short-49208-hca-cp1 | tilelang@cute | 7869 | 693.6 | 1.00 | 25371 | 819.3 | 1.00 | 6382 |
| short-49208-hca-cp1 | cute@cute | 7350 | 63.5 | 0.93 | 24826 | 213.9 | 0.98 | 6382 |
| short-49208-hca-cp1 | cute_ws@cute | 3497 | 73.7 | 0.44 | 20974 | 193.0 | 0.83 | 6382 |
| short-49208-hca-cp1 | flashmla_fwd_ref@cute | 4091 | 31.2 | 0.52 | - | - | - | 3197 |
| short-49208-hca-cp8r0 | tilelang@main | 989.7 | 710.9 | 1.00 | 3199 | 869.2 | 1.00 | 925 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 521.4 | 125.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r0 | tilelang@cute | 989.9 | 737.9 | 1.00 | 3194 | 883.8 | 1.00 | 925 |
| short-49208-hca-cp8r0 | cute@cute | 925.3 | 143.2 | 0.93 | 3125 | 268.1 | 0.98 | 925 |
| short-49208-hca-cp8r0 | cute_ws@cute | 465.9 | 43.2 | 0.47 | 2670 | 201.7 | 0.84 | 925 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 524.0 | 136.4 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r4 | tilelang@main | 1004 | 742.6 | 1.00 | 3328 | 858.1 | 1.00 | 925 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 529.1 | 136.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r4 | tilelang@cute | 1005 | 724.7 | 1.00 | 3327 | 907.0 | 1.00 | 925 |
| short-49208-hca-cp8r4 | cute@cute | 942.5 | 143.0 | 0.94 | 3261 | 281.3 | 0.98 | 925 |
| short-49208-hca-cp8r4 | cute_ws@cute | 466.9 | 55.7 | 0.46 | 2789 | 214.3 | 0.84 | 925 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 532.7 | 135.7 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r7 | tilelang@main | 1012 | 720.8 | 1.00 | 3277 | 892.2 | 1.00 | 925 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 532.2 | 132.3 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r7 | tilelang@cute | 1012 | 728.8 | 1.00 | 3279 | 883.1 | 1.00 | 925 |
| short-49208-hca-cp8r7 | cute@cute | 941.4 | 148.7 | 0.93 | 3207 | 273.7 | 0.98 | 925 |
| short-49208-hca-cp8r7 | cute_ws@cute | 462.0 | 57.4 | 0.46 | 2727 | 211.9 | 0.83 | 925 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 532.8 | 137.1 | 0.53 | - | - | - | 400 |
| short-49208-sliding-cp1 | tilelang@main | 6741 | 681.3 | 1.00 | 21763 | 869.5 | 1.00 | 6332 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 3084 | 39.0 | 0.46 | - | - | - | 3148 |
| short-49208-sliding-cp1 | tilelang@cute | 6820 | 643.2 | 1.00 | 21766 | 889.8 | 1.00 | 6332 |
| short-49208-sliding-cp1 | cute@cute | 6267 | 87.4 | 0.92 | 21268 | 241.3 | 0.98 | 6332 |
| short-49208-sliding-cp1 | cute_ws@cute | 2352 | 79.6 | 0.34 | 17366 | 188.6 | 0.80 | 6332 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3088 | 76.6 | 0.45 | - | - | - | 3148 |
| short-49208-sliding-cp8r0 | tilelang@main | 864.5 | 700.0 | 1.00 | 2838 | 825.1 | 1.00 | 918 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 397.8 | 143.2 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r0 | tilelang@cute | 860.8 | 691.3 | 1.00 | 2832 | 853.1 | 1.00 | 918 |
| short-49208-sliding-cp8r0 | cute@cute | 794.4 | 122.5 | 0.92 | 2763 | 240.9 | 0.98 | 918 |
| short-49208-sliding-cp8r0 | cute_ws@cute | 318.1 | 51.1 | 0.37 | 2290 | 324.3 | 0.81 | 918 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 401.9 | 145.3 | 0.47 | - | - | - | 394 |
| short-49208-sliding-cp8r4 | tilelang@main | 869.7 | 699.7 | 1.00 | 2893 | 846.4 | 1.00 | 918 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 402.9 | 142.9 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r4 | tilelang@cute | 869.6 | 698.0 | 1.00 | 2895 | 859.0 | 1.00 | 918 |
| short-49208-sliding-cp8r4 | cute@cute | 805.0 | 125.6 | 0.93 | 2831 | 233.4 | 0.98 | 918 |
| short-49208-sliding-cp8r4 | cute_ws@cute | 314.7 | 56.6 | 0.36 | 2343 | 319.2 | 0.81 | 918 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 403.0 | 147.3 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r7 | tilelang@main | 872.3 | 683.9 | 1.00 | 2886 | 857.6 | 1.00 | 918 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 405.0 | 131.9 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r7 | tilelang@cute | 873.2 | 698.1 | 1.00 | 2881 | 850.8 | 1.00 | 918 |
| short-49208-sliding-cp8r7 | cute@cute | 806.4 | 127.7 | 0.92 | 2817 | 232.1 | 0.98 | 918 |
| short-49208-sliding-cp8r7 | cute_ws@cute | 314.2 | 54.9 | 0.36 | 2329 | 313.4 | 0.81 | 918 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 401.6 | 146.8 | 0.46 | - | - | - | 394 |
| heavy-49208-csa-cp1 | tilelang@main | 14881 | 32.9 | 1.00 | 60982 | 725.7 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 7056 | 338.1 | 0.47 | - | - | - | 3340 |
| heavy-49208-csa-cp1 | tilelang@cute | 14869 | -8.8 | 1.00 | 60934 | 810.8 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | cute@cute | 13624 | 13.2 | 0.92 | 60231 | 205.5 | 0.99 | 6368 |
| heavy-49208-csa-cp1 | cute_ws@cute | 6561 | 389.3 | 0.44 | 53264 | 193.5 | 0.87 | 6368 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@cute | 7280 | 382.5 | 0.49 | - | - | - | 3340 |
| heavy-49208-csa-cp8r0 | tilelang@main | 1239 | 685.7 | 1.00 | 4603 | 816.8 | 1.00 | 954 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 650.2 | 109.6 | 0.52 | - | - | - | 418 |
| heavy-49208-csa-cp8r0 | tilelang@cute | 1239 | 685.9 | 1.00 | 4598 | 865.3 | 1.00 | 954 |
| heavy-49208-csa-cp8r0 | cute@cute | 1167 | 113.0 | 0.94 | 4533 | 223.3 | 0.99 | 954 |
| heavy-49208-csa-cp8r0 | cute_ws@cute | 564.6 | 53.8 | 0.46 | 3929 | 119.0 | 0.85 | 954 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 649.2 | 113.5 | 0.52 | - | - | - | 418 |
| heavy-49208-csa-cp8r4 | tilelang@main | 1940 | 688.9 | 1.00 | 8493 | 875.9 | 1.00 | 954 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 953.3 | 104.7 | 0.49 | - | - | - | 418 |
| heavy-49208-csa-cp8r4 | tilelang@cute | 1948 | 717.4 | 1.00 | 8490 | 941.3 | 1.00 | 954 |
| heavy-49208-csa-cp8r4 | cute@cute | 1854 | 110.7 | 0.95 | 8406 | 242.0 | 0.99 | 954 |
| heavy-49208-csa-cp8r4 | cute_ws@cute | 884.4 | 55.5 | 0.45 | 7444 | 135.3 | 0.88 | 954 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 955.3 | 106.8 | 0.49 | - | - | - | 418 |
| heavy-49208-csa-cp8r7 | tilelang@main | 1792 | 685.2 | 1.00 | 7721 | 860.7 | 1.00 | 954 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 889.3 | 101.1 | 0.50 | - | - | - | 418 |
| heavy-49208-csa-cp8r7 | tilelang@cute | 1803 | 701.8 | 1.00 | 7741 | 845.0 | 1.00 | 954 |
| heavy-49208-csa-cp8r7 | cute@cute | 1714 | 125.7 | 0.95 | 7633 | 242.5 | 0.99 | 954 |
| heavy-49208-csa-cp8r7 | cute_ws@cute | 814.1 | 60.8 | 0.45 | 6757 | 128.7 | 0.87 | 954 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 892.7 | 105.0 | 0.50 | - | - | - | 418 |
| heavy-49208-hca-cp1 | tilelang@main | 8488 | 600.9 | 1.00 | 28842 | 762.6 | 1.00 | 6394 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4251 | 22.7 | 0.50 | - | - | - | 3244 |
| heavy-49208-hca-cp1 | tilelang@cute | 8583 | 517.5 | 1.00 | 28873 | 805.0 | 1.00 | 6394 |
| heavy-49208-hca-cp1 | cute@cute | 8002 | -18.9 | 0.93 | 28327 | 189.3 | 0.98 | 6394 |
| heavy-49208-hca-cp1 | cute_ws@cute | 3640 | 40.3 | 0.42 | 24059 | 161.3 | 0.83 | 6394 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@cute | 4279 | 54.9 | 0.50 | - | - | - | 3244 |
| heavy-49208-hca-cp8r0 | tilelang@main | 956.8 | 745.4 | 1.00 | 3055 | 857.0 | 1.00 | 926 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 510.4 | 132.4 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r0 | tilelang@cute | 957.0 | 731.2 | 1.00 | 3039 | 883.6 | 1.00 | 926 |
| heavy-49208-hca-cp8r0 | cute@cute | 891.9 | 140.2 | 0.93 | 2978 | 263.2 | 0.98 | 926 |
| heavy-49208-hca-cp8r0 | cute_ws@cute | 430.6 | 55.8 | 0.45 | 2522 | 238.5 | 0.83 | 926 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 512.3 | 130.0 | 0.54 | - | - | - | 406 |
| heavy-49208-hca-cp8r4 | tilelang@main | 1044 | 709.2 | 1.00 | 3546 | 847.7 | 1.00 | 926 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 550.7 | 124.3 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r4 | tilelang@cute | 1035 | 733.9 | 1.00 | 3509 | 917.9 | 1.00 | 926 |
| heavy-49208-hca-cp8r4 | cute@cute | 965.2 | 142.4 | 0.93 | 3442 | 285.7 | 0.98 | 926 |
| heavy-49208-hca-cp8r4 | cute_ws@cute | 473.8 | 54.4 | 0.46 | 2970 | 213.6 | 0.85 | 926 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 550.1 | 127.2 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r7 | tilelang@main | 1105 | 739.6 | 1.00 | 3792 | 880.6 | 1.00 | 926 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 591.4 | 120.7 | 0.54 | - | - | - | 406 |
| heavy-49208-hca-cp8r7 | tilelang@cute | 1098 | 731.3 | 1.00 | 3771 | 893.0 | 1.00 | 926 |
| heavy-49208-hca-cp8r7 | cute@cute | 1030 | 140.6 | 0.94 | 3698 | 273.9 | 0.98 | 926 |
| heavy-49208-hca-cp8r7 | cute_ws@cute | 516.4 | 56.0 | 0.47 | 3190 | 174.7 | 0.85 | 926 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 586.7 | 131.1 | 0.53 | - | - | - | 406 |
| heavy-49208-sliding-cp1 | tilelang@main | 6750 | 697.8 | 1.00 | 21733 | 849.9 | 1.00 | 6332 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 3069 | 65.7 | 0.45 | - | - | - | 3148 |
| heavy-49208-sliding-cp1 | tilelang@cute | 6859 | 605.8 | 1.00 | 21734 | 896.6 | 1.00 | 6332 |
| heavy-49208-sliding-cp1 | cute@cute | 6261 | 95.4 | 0.91 | 21216 | 280.0 | 0.98 | 6332 |
| heavy-49208-sliding-cp1 | cute_ws@cute | 2385 | 43.6 | 0.35 | 17342 | 192.3 | 0.80 | 6332 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3092 | 60.5 | 0.45 | - | - | - | 3148 |
| heavy-49208-sliding-cp8r0 | tilelang@main | 861.4 | 710.5 | 1.00 | 2755 | 833.2 | 1.00 | 918 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 402.5 | 139.7 | 0.47 | - | - | - | 394 |
| heavy-49208-sliding-cp8r0 | tilelang@cute | 860.6 | 711.3 | 1.00 | 2752 | 840.8 | 1.00 | 918 |
| heavy-49208-sliding-cp8r0 | cute@cute | 794.2 | 128.6 | 0.92 | 2687 | 226.6 | 0.98 | 918 |
| heavy-49208-sliding-cp8r0 | cute_ws@cute | 319.0 | 61.6 | 0.37 | 2223 | 309.6 | 0.81 | 918 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 402.6 | 151.6 | 0.47 | - | - | - | 394 |
| heavy-49208-sliding-cp8r4 | tilelang@main | 877.6 | 687.8 | 1.00 | 2942 | 819.4 | 1.00 | 918 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 403.9 | 135.0 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r4 | tilelang@cute | 877.5 | 697.4 | 1.00 | 2954 | 883.0 | 1.00 | 918 |
| heavy-49208-sliding-cp8r4 | cute@cute | 815.8 | 126.8 | 0.93 | 2895 | 242.7 | 0.98 | 918 |
| heavy-49208-sliding-cp8r4 | cute_ws@cute | 314.9 | 55.0 | 0.36 | 2400 | 346.4 | 0.81 | 918 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 405.6 | 145.4 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r7 | tilelang@main | 882.7 | 684.6 | 1.00 | 2924 | 841.8 | 1.00 | 918 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 402.0 | 141.2 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r7 | tilelang@cute | 883.8 | 695.8 | 1.00 | 2923 | 851.5 | 1.00 | 918 |
| heavy-49208-sliding-cp8r7 | cute@cute | 814.9 | 127.0 | 0.92 | 2861 | 232.7 | 0.98 | 918 |
| heavy-49208-sliding-cp8r7 | cute_ws@cute | 315.2 | 56.1 | 0.36 | 2369 | 331.1 | 0.81 | 918 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 406.9 | 143.7 | 0.46 | - | - | - | 394 |
| tiny-49208-csa-cp1 | tilelang@main | 7840 | 553.1 | 1.00 | 23410 | 693.2 | 1.00 | 6368 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 4272 | 23.0 | 0.54 | - | - | - | 3340 |
| tiny-49208-csa-cp1 | tilelang@cute | 7844 | 552.3 | 1.00 | 23409 | 745.6 | 1.00 | 6368 |
| tiny-49208-csa-cp1 | cute@cute | 7324 | 40.5 | 0.93 | 22890 | 148.2 | 0.98 | 6368 |
| tiny-49208-csa-cp1 | cute_ws@cute | 3644 | 89.0 | 0.46 | 19234 | 174.0 | 0.82 | 6368 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@cute | 4281 | 17.4 | 0.55 | - | - | - | 3340 |
| tiny-49208-csa-cp8r0 | tilelang@main | 998.7 | 705.4 | 1.00 | 3072 | 802.9 | 1.00 | 953 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 556.1 | 102.0 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r0 | tilelang@cute | 1007 | 679.7 | 1.00 | 3086 | 824.2 | 1.00 | 953 |
| tiny-49208-csa-cp8r0 | cute@cute | 943.5 | 113.6 | 0.94 | 3025 | 217.0 | 0.98 | 953 |
| tiny-49208-csa-cp8r0 | cute_ws@cute | 474.8 | 58.5 | 0.47 | 2556 | 139.5 | 0.83 | 953 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 558.3 | 118.3 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r4 | tilelang@main | 1012 | 668.1 | 1.00 | 3097 | 809.2 | 1.00 | 953 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 554.1 | 102.4 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r4 | tilelang@cute | 1011 | 689.2 | 1.00 | 3101 | 846.4 | 1.00 | 953 |
| tiny-49208-csa-cp8r4 | cute@cute | 947.2 | 113.8 | 0.94 | 3036 | 220.9 | 0.98 | 953 |
| tiny-49208-csa-cp8r4 | cute_ws@cute | 484.4 | 52.2 | 0.48 | 2573 | 139.0 | 0.83 | 953 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 556.9 | 111.2 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r7 | tilelang@main | 1007 | 682.4 | 1.00 | 3080 | 832.5 | 1.00 | 953 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 560.0 | 99.4 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r7 | tilelang@cute | 1004 | 688.7 | 1.00 | 3079 | 846.2 | 1.00 | 953 |
| tiny-49208-csa-cp8r7 | cute@cute | 939.9 | 112.0 | 0.94 | 3020 | 219.7 | 0.98 | 953 |
| tiny-49208-csa-cp8r7 | cute_ws@cute | 475.8 | 58.6 | 0.47 | 2563 | 132.9 | 0.83 | 953 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 560.7 | 102.5 | 0.56 | - | - | - | 418 |
| tiny-49208-hca-cp1 | tilelang@main | 6058 | 685.3 | 1.00 | 15953 | 833.4 | 1.00 | 6332 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 3126 | 38.3 | 0.52 | - | - | - | 3148 |
| tiny-49208-hca-cp1 | tilelang@cute | 6036 | 684.8 | 1.00 | 15897 | 861.2 | 1.00 | 6332 |
| tiny-49208-hca-cp1 | cute@cute | 5546 | 89.1 | 0.92 | 15395 | 250.2 | 0.97 | 6332 |
| tiny-49208-hca-cp1 | cute_ws@cute | 2686 | 71.3 | 0.44 | 12551 | 160.5 | 0.79 | 6332 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@cute | 3113 | 72.4 | 0.52 | - | - | - | 3148 |
| tiny-49208-hca-cp8r0 | tilelang@main | 774.9 | 704.2 | 1.00 | 2125 | 827.8 | 1.00 | 918 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 401.6 | 139.8 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r0 | tilelang@cute | 769.8 | 702.1 | 1.00 | 2117 | 848.2 | 1.00 | 918 |
| tiny-49208-hca-cp8r0 | cute@cute | 711.2 | 127.6 | 0.92 | 2058 | 229.4 | 0.97 | 918 |
| tiny-49208-hca-cp8r0 | cute_ws@cute | 351.6 | 51.7 | 0.46 | 1702 | 278.0 | 0.80 | 918 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 404.7 | 145.0 | 0.53 | - | - | - | 394 |
| tiny-49208-hca-cp8r4 | tilelang@main | 783.0 | 701.8 | 1.00 | 2153 | 827.3 | 1.00 | 918 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 404.4 | 139.6 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r4 | tilelang@cute | 786.2 | 690.2 | 1.00 | 2157 | 861.1 | 1.00 | 918 |
| tiny-49208-hca-cp8r4 | cute@cute | 723.9 | 125.4 | 0.92 | 2093 | 237.8 | 0.97 | 918 |
| tiny-49208-hca-cp8r4 | cute_ws@cute | 354.0 | 48.8 | 0.45 | 1719 | 302.4 | 0.80 | 918 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 405.4 | 146.4 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r7 | tilelang@main | 780.8 | 692.8 | 1.00 | 2111 | 841.6 | 1.00 | 918 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 401.6 | 139.0 | 0.51 | - | - | - | 394 |
| tiny-49208-hca-cp8r7 | tilelang@cute | 780.7 | 706.4 | 1.00 | 2107 | 855.0 | 1.00 | 918 |
| tiny-49208-hca-cp8r7 | cute@cute | 717.2 | 125.9 | 0.92 | 2042 | 241.5 | 0.97 | 918 |
| tiny-49208-hca-cp8r7 | cute_ws@cute | 352.8 | 52.9 | 0.45 | 1687 | 305.9 | 0.80 | 918 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 403.3 | 165.3 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp1 | tilelang@main | 6059 | 674.1 | 1.00 | 15952 | 849.2 | 1.00 | 6332 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 3124 | 48.6 | 0.52 | - | - | - | 3148 |
| tiny-49208-sliding-cp1 | tilelang@cute | 6037 | 688.9 | 1.00 | 15884 | 877.5 | 1.00 | 6332 |
| tiny-49208-sliding-cp1 | cute@cute | 5538 | 106.5 | 0.92 | 15380 | 272.6 | 0.97 | 6332 |
| tiny-49208-sliding-cp1 | cute_ws@cute | 2688 | 57.2 | 0.45 | 12546 | 159.3 | 0.79 | 6332 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@cute | 3119 | 67.3 | 0.52 | - | - | - | 3148 |
| tiny-49208-sliding-cp8r0 | tilelang@main | 773.6 | 689.3 | 1.00 | 2123 | 823.5 | 1.00 | 918 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 403.5 | 136.8 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r0 | tilelang@cute | 778.3 | 700.1 | 1.00 | 2128 | 850.4 | 1.00 | 918 |
| tiny-49208-sliding-cp8r0 | cute@cute | 715.1 | 127.5 | 0.92 | 2063 | 235.4 | 0.97 | 918 |
| tiny-49208-sliding-cp8r0 | cute_ws@cute | 354.9 | 51.8 | 0.46 | 1704 | 287.7 | 0.80 | 918 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 402.9 | 151.4 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r4 | tilelang@main | 785.5 | 699.1 | 1.00 | 2156 | 819.1 | 1.00 | 918 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 402.9 | 139.6 | 0.51 | - | - | - | 394 |
| tiny-49208-sliding-cp8r4 | tilelang@cute | 785.2 | 697.2 | 1.00 | 2158 | 864.4 | 1.00 | 918 |
| tiny-49208-sliding-cp8r4 | cute@cute | 724.4 | 124.9 | 0.92 | 2092 | 240.9 | 0.97 | 918 |
| tiny-49208-sliding-cp8r4 | cute_ws@cute | 351.1 | 53.1 | 0.45 | 1725 | 320.2 | 0.80 | 918 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 405.3 | 143.5 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r7 | tilelang@main | 779.6 | 685.1 | 1.00 | 2107 | 843.6 | 1.00 | 918 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 406.0 | 136.3 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r7 | tilelang@cute | 781.2 | 701.2 | 1.00 | 2108 | 855.7 | 1.00 | 918 |
| tiny-49208-sliding-cp8r7 | cute@cute | 716.0 | 128.1 | 0.92 | 2041 | 237.0 | 0.97 | 918 |
| tiny-49208-sliding-cp8r7 | cute_ws@cute | 356.0 | 50.2 | 0.46 | 1684 | 288.7 | 0.80 | 918 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 402.3 | 145.8 | 0.51 | - | - | - | 394 |
| single-65536-csa-cp1 | tilelang@main | 21804 | -109.6 | 1.00 | 94799 | 644.7 | 1.00 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 10352 | 1033 | 0.47 | - | - | - | 4448 |
| single-65536-csa-cp1 | tilelang@cute | 21733 | -129.7 | 1.00 | 94985 | 492.1 | 1.00 | 8480 |
| single-65536-csa-cp1 | cute@cute | 20142 | 31.8 | 0.93 | 93681 | 215.8 | 0.99 | 8480 |
| single-65536-csa-cp1 | cute_ws@cute | 9765 | 1184 | 0.45 | 83591 | 372.1 | 0.88 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref@cute | 11083 | 389.9 | 0.51 | - | - | - | 4448 |
| single-65536-csa-cp8r0 | tilelang@main | 2543 | 722.8 | 1.00 | 11048 | 882.7 | 1.00 | 1270 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 1248 | 76.7 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r0 | tilelang@cute | 2631 | 597.5 | 1.00 | 11032 | 937.1 | 1.00 | 1270 |
| single-65536-csa-cp8r0 | cute@cute | 2437 | 98.2 | 0.93 | 10904 | 322.6 | 0.99 | 1270 |
| single-65536-csa-cp8r0 | cute_ws@cute | 1146 | 71.2 | 0.44 | 9658 | 204.7 | 0.88 | 1270 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 1243 | 94.8 | 0.47 | - | - | - | 556 |
| single-65536-csa-cp8r4 | tilelang@main | 2711 | 717.6 | 1.00 | 12259 | 886.8 | 1.00 | 1270 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1324 | 86.3 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r4 | tilelang@cute | 2889 | 522.3 | 1.00 | 12226 | 991.0 | 1.00 | 1270 |
| single-65536-csa-cp8r4 | cute@cute | 2688 | 15.7 | 0.93 | 12091 | 367.0 | 0.99 | 1270 |
| single-65536-csa-cp8r4 | cute_ws@cute | 1241 | 52.1 | 0.43 | 10760 | 242.1 | 0.88 | 1270 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 1330 | 91.9 | 0.46 | - | - | - | 556 |
| single-65536-csa-cp8r7 | tilelang@main | 2709 | 675.0 | 1.00 | 13123 | 872.4 | 1.00 | 1270 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1331 | 78.8 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r7 | tilelang@cute | 2741 | 690.7 | 1.00 | 13074 | 945.5 | 1.00 | 1270 |
| single-65536-csa-cp8r7 | cute@cute | 2589 | 123.4 | 0.94 | 12984 | 282.6 | 0.99 | 1270 |
| single-65536-csa-cp8r7 | cute_ws@cute | 1240 | 69.3 | 0.45 | 11675 | 115.3 | 0.89 | 1270 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 1332 | 110.2 | 0.49 | - | - | - | 556 |
| single-65536-hca-cp1 | tilelang@main | 16321 | 181.6 | 1.00 | 63704 | 622.9 | 1.00 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 7958 | 238.2 | 0.49 | - | - | - | 4448 |
| single-65536-hca-cp1 | tilelang@cute | 16222 | 264.6 | 1.00 | 63651 | 682.2 | 1.00 | 8434 |
| single-65536-hca-cp1 | cute@cute | 15178 | 35.6 | 0.94 | 62828 | 111.8 | 0.99 | 8434 |
| single-65536-hca-cp1 | cute_ws@cute | 7152 | 503.5 | 0.44 | 54839 | 175.4 | 0.86 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref@cute | 8247 | 183.1 | 0.51 | - | - | - | 4448 |
| single-65536-hca-cp8r0 | tilelang@main | 1365 | 708.7 | 1.00 | 4606 | 828.1 | 1.00 | 1224 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 751.0 | 84.2 | 0.55 | - | - | - | 556 |
| single-65536-hca-cp8r0 | tilelang@cute | 1362 | 689.6 | 1.00 | 4613 | 833.0 | 1.00 | 1224 |
| single-65536-hca-cp8r0 | cute@cute | 1267 | 117.6 | 0.93 | 4517 | 227.7 | 0.98 | 1224 |
| single-65536-hca-cp8r0 | cute_ws@cute | 621.6 | 54.1 | 0.46 | 3860 | 132.8 | 0.84 | 1224 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 749.4 | 94.4 | 0.55 | - | - | - | 556 |
| single-65536-hca-cp8r4 | tilelang@main | 2123 | 673.9 | 1.00 | 8699 | 883.4 | 1.00 | 1224 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1114 | 70.8 | 0.52 | - | - | - | 556 |
| single-65536-hca-cp8r4 | tilelang@cute | 2141 | 672.5 | 1.00 | 8701 | 923.5 | 1.00 | 1224 |
| single-65536-hca-cp8r4 | cute@cute | 2029 | 113.2 | 0.95 | 8588 | 268.9 | 0.99 | 1224 |
| single-65536-hca-cp8r4 | cute_ws@cute | 1008 | 59.4 | 0.47 | 7600 | 129.2 | 0.87 | 1224 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 1112 | 88.8 | 0.52 | - | - | - | 556 |
| single-65536-hca-cp8r7 | tilelang@main | 2715 | 674.6 | 1.00 | 11760 | 929.3 | 1.00 | 1224 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 1305 | 79.3 | 0.48 | - | - | - | 556 |
| single-65536-hca-cp8r7 | tilelang@cute | 2819 | 616.9 | 1.00 | 11755 | 942.1 | 1.00 | 1224 |
| single-65536-hca-cp8r7 | cute@cute | 2635 | 74.2 | 0.93 | 11644 | 306.8 | 0.99 | 1224 |
| single-65536-hca-cp8r7 | cute_ws@cute | 1207 | 63.6 | 0.43 | 10266 | 215.7 | 0.87 | 1224 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 1307 | 101.9 | 0.46 | - | - | - | 556 |
| single-65536-sliding-cp1 | tilelang@main | 8998 | 652.7 | 1.00 | 29308 | 838.2 | 1.00 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4089 | 32.9 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp1 | tilelang@cute | 9047 | 639.5 | 1.00 | 29341 | 829.9 | 1.00 | 8432 |
| single-65536-sliding-cp1 | cute@cute | 8361 | 73.3 | 0.92 | 28630 | 252.7 | 0.98 | 8432 |
| single-65536-sliding-cp1 | cute_ws@cute | 3147 | 70.0 | 0.35 | 23476 | 162.8 | 0.80 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4095 | 62.5 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp8r0 | tilelang@main | 1148 | 723.0 | 1.00 | 3834 | 823.8 | 1.00 | 1222 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 522.4 | 139.0 | 0.45 | - | - | - | 524 |
| single-65536-sliding-cp8r0 | tilelang@cute | 1155 | 703.9 | 1.00 | 3828 | 864.6 | 1.00 | 1222 |
| single-65536-sliding-cp8r0 | cute@cute | 1063 | 127.9 | 0.92 | 3744 | 231.3 | 0.98 | 1222 |
| single-65536-sliding-cp8r0 | cute_ws@cute | 407.3 | 55.2 | 0.35 | 3080 | 136.9 | 0.80 | 1222 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 529.1 | 142.5 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r4 | tilelang@main | 1153 | 690.8 | 1.00 | 3867 | 826.9 | 1.00 | 1222 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 533.2 | 127.5 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r4 | tilelang@cute | 1156 | 709.2 | 1.00 | 3874 | 856.0 | 1.00 | 1222 |
| single-65536-sliding-cp8r4 | cute@cute | 1064 | 127.0 | 0.92 | 3779 | 246.3 | 0.98 | 1222 |
| single-65536-sliding-cp8r4 | cute_ws@cute | 407.2 | 56.0 | 0.35 | 3125 | 137.3 | 0.81 | 1222 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 526.8 | 142.4 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r7 | tilelang@main | 1154 | 698.5 | 1.00 | 3861 | 855.4 | 1.00 | 1222 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 529.4 | 133.0 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r7 | tilelang@cute | 1158 | 700.6 | 1.00 | 3869 | 850.9 | 1.00 | 1222 |
| single-65536-sliding-cp8r7 | cute@cute | 1068 | 125.1 | 0.92 | 3785 | 230.7 | 0.98 | 1222 |
| single-65536-sliding-cp8r7 | cute_ws@cute | 407.2 | 58.2 | 0.35 | 3135 | 142.8 | 0.81 | 1222 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 529.2 | 150.2 | 0.46 | - | - | - | 524 |
| short-65536-csa-cp1 | tilelang@main | 16000 | 242.3 | 1.00 | 62509 | 660.3 | 1.00 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 7830 | 315.8 | 0.49 | - | - | - | 4448 |
| short-65536-csa-cp1 | tilelang@cute | 15992 | 240.9 | 1.00 | 62497 | 684.9 | 1.00 | 8480 |
| short-65536-csa-cp1 | cute@cute | 14892 | 28.0 | 0.93 | 61673 | 87.9 | 0.99 | 8480 |
| short-65536-csa-cp1 | cute_ws@cute | 7066 | 520.1 | 0.44 | 53908 | 170.3 | 0.86 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref@cute | 8166 | 167.2 | 0.51 | - | - | - | 4448 |
| short-65536-csa-cp8r0 | tilelang@main | 1610 | 676.7 | 1.00 | 5968 | 811.5 | 1.00 | 1270 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 825.1 | 82.7 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r0 | tilelang@cute | 1611 | 692.1 | 1.00 | 5958 | 831.9 | 1.00 | 1270 |
| short-65536-csa-cp8r0 | cute@cute | 1516 | 107.7 | 0.94 | 5864 | 227.5 | 0.98 | 1270 |
| short-65536-csa-cp8r0 | cute_ws@cute | 715.8 | 56.2 | 0.44 | 5077 | 117.8 | 0.85 | 1270 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 825.5 | 91.3 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r4 | tilelang@main | 2195 | 666.7 | 1.00 | 9179 | 896.3 | 1.00 | 1270 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1090 | 79.7 | 0.50 | - | - | - | 556 |
| short-65536-csa-cp8r4 | tilelang@cute | 2214 | 677.4 | 1.00 | 9181 | 942.6 | 1.00 | 1270 |
| short-65536-csa-cp8r4 | cute@cute | 2105 | 107.5 | 0.95 | 9070 | 287.3 | 0.99 | 1270 |
| short-65536-csa-cp8r4 | cute_ws@cute | 993.5 | 55.5 | 0.45 | 7985 | 128.0 | 0.87 | 1270 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 1099 | 81.1 | 0.50 | - | - | - | 556 |
| short-65536-csa-cp8r7 | tilelang@main | 1997 | 667.0 | 1.00 | 8087 | 855.0 | 1.00 | 1270 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1012 | 76.5 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r7 | tilelang@cute | 2009 | 688.9 | 1.00 | 8079 | 864.9 | 1.00 | 1270 |
| short-65536-csa-cp8r7 | cute@cute | 1899 | 124.3 | 0.95 | 7973 | 235.7 | 0.99 | 1270 |
| short-65536-csa-cp8r7 | cute_ws@cute | 896.8 | 62.5 | 0.45 | 6996 | 121.7 | 0.87 | 1270 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 1008 | 97.7 | 0.50 | - | - | - | 556 |
| short-65536-hca-cp1 | tilelang@main | 10309 | 612.7 | 1.00 | 32762 | 770.6 | 1.00 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5376 | 1.8 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp1 | tilelang@cute | 10363 | 644.5 | 1.00 | 32752 | 818.3 | 1.00 | 8482 |
| short-65536-hca-cp1 | cute@cute | 9687 | -27.3 | 0.93 | 32072 | 177.0 | 0.98 | 8482 |
| short-65536-hca-cp1 | cute_ws@cute | 4699 | 49.4 | 0.45 | 27122 | 166.8 | 0.83 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref@cute | 5422 | -13.8 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp8r0 | tilelang@main | 1296 | 723.7 | 1.00 | 4168 | 858.9 | 1.00 | 1229 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 687.6 | 123.7 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r0 | tilelang@cute | 1293 | 753.6 | 1.00 | 4160 | 895.8 | 1.00 | 1229 |
| short-65536-hca-cp8r0 | cute@cute | 1207 | 146.8 | 0.93 | 4075 | 271.6 | 0.98 | 1229 |
| short-65536-hca-cp8r0 | cute_ws@cute | 586.6 | 60.5 | 0.45 | 3466 | 114.1 | 0.83 | 1229 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 681.2 | 140.4 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r4 | tilelang@main | 1337 | 717.1 | 1.00 | 4340 | 857.5 | 1.00 | 1229 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 704.0 | 128.4 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r4 | tilelang@cute | 1336 | 725.5 | 1.00 | 4344 | 903.7 | 1.00 | 1229 |
| short-65536-hca-cp8r4 | cute@cute | 1245 | 139.9 | 0.93 | 4248 | 275.7 | 0.98 | 1229 |
| short-65536-hca-cp8r4 | cute_ws@cute | 609.8 | 54.0 | 0.46 | 3610 | 134.0 | 0.83 | 1229 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 705.7 | 129.4 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r7 | tilelang@main | 1320 | 720.0 | 1.00 | 4272 | 894.3 | 1.00 | 1229 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 702.8 | 117.0 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r7 | tilelang@cute | 1320 | 744.1 | 1.00 | 4278 | 880.6 | 1.00 | 1229 |
| short-65536-hca-cp8r7 | cute@cute | 1229 | 150.2 | 0.93 | 4188 | 265.0 | 0.98 | 1229 |
| short-65536-hca-cp8r7 | cute_ws@cute | 596.4 | 60.0 | 0.45 | 3564 | 112.0 | 0.83 | 1229 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 695.7 | 137.9 | 0.53 | - | - | - | 532 |
| short-65536-sliding-cp1 | tilelang@main | 8974 | 637.9 | 1.00 | 28844 | 797.0 | 1.00 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4081 | 23.9 | 0.45 | - | - | - | 4192 |
| short-65536-sliding-cp1 | tilelang@cute | 9020 | 662.0 | 1.00 | 28843 | 873.2 | 1.00 | 8432 |
| short-65536-sliding-cp1 | cute@cute | 8334 | 85.7 | 0.92 | 28143 | 264.6 | 0.98 | 8432 |
| short-65536-sliding-cp1 | cute_ws@cute | 3146 | 79.0 | 0.35 | 23035 | 160.8 | 0.80 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4107 | 84.8 | 0.46 | - | - | - | 4192 |
| short-65536-sliding-cp8r0 | tilelang@main | 1138 | 725.1 | 1.00 | 3714 | 842.5 | 1.00 | 1222 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 524.0 | 141.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r0 | tilelang@cute | 1138 | 699.1 | 1.00 | 3718 | 856.8 | 1.00 | 1222 |
| short-65536-sliding-cp8r0 | cute@cute | 1053 | 129.7 | 0.93 | 3636 | 228.4 | 0.98 | 1222 |
| short-65536-sliding-cp8r0 | cute_ws@cute | 412.5 | 53.1 | 0.36 | 3003 | 111.4 | 0.81 | 1222 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 530.7 | 138.5 | 0.47 | - | - | - | 524 |
| short-65536-sliding-cp8r4 | tilelang@main | 1149 | 688.8 | 1.00 | 3813 | 810.9 | 1.00 | 1222 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 526.8 | 133.6 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r4 | tilelang@cute | 1151 | 693.5 | 1.00 | 3806 | 880.1 | 1.00 | 1222 |
| short-65536-sliding-cp8r4 | cute@cute | 1063 | 119.7 | 0.92 | 3717 | 249.1 | 0.98 | 1222 |
| short-65536-sliding-cp8r4 | cute_ws@cute | 405.5 | 56.0 | 0.35 | 3071 | 140.4 | 0.81 | 1222 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 524.2 | 144.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r7 | tilelang@main | 1153 | 688.0 | 1.00 | 3777 | 855.3 | 1.00 | 1222 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 526.2 | 135.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r7 | tilelang@cute | 1155 | 692.7 | 1.00 | 3781 | 854.0 | 1.00 | 1222 |
| short-65536-sliding-cp8r7 | cute@cute | 1062 | 122.6 | 0.92 | 3686 | 232.1 | 0.97 | 1222 |
| short-65536-sliding-cp8r7 | cute_ws@cute | 406.2 | 57.9 | 0.35 | 3039 | 127.1 | 0.80 | 1222 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 533.2 | 141.2 | 0.46 | - | - | - | 524 |
| heavy-65536-csa-cp1 | tilelang@main | 17414 | -41.5 | 1.00 | 69035 | 660.4 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8439 | 381.8 | 0.48 | - | - | - | 4448 |
| heavy-65536-csa-cp1 | tilelang@cute | 17380 | 17.8 | 1.00 | 68974 | 700.5 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | cute@cute | 16031 | 26.7 | 0.92 | 68138 | 105.6 | 0.99 | 8480 |
| heavy-65536-csa-cp1 | cute_ws@cute | 7753 | 696.7 | 0.45 | 59903 | 153.7 | 0.87 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@cute | 8809 | 197.1 | 0.51 | - | - | - | 4448 |
| heavy-65536-csa-cp8r0 | tilelang@main | 1561 | 676.9 | 1.00 | 5628 | 791.6 | 1.00 | 1270 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 818.9 | 82.0 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r0 | tilelang@cute | 1562 | 675.0 | 1.00 | 5620 | 827.4 | 1.00 | 1270 |
| heavy-65536-csa-cp8r0 | cute@cute | 1473 | 105.5 | 0.94 | 5535 | 212.7 | 0.98 | 1270 |
| heavy-65536-csa-cp8r0 | cute_ws@cute | 709.3 | 55.3 | 0.45 | 4771 | 119.2 | 0.85 | 1270 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 822.9 | 86.6 | 0.53 | - | - | - | 556 |
| heavy-65536-csa-cp8r4 | tilelang@main | 2719 | 683.2 | 1.00 | 12075 | 898.7 | 1.00 | 1270 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1326 | 75.9 | 0.49 | - | - | - | 556 |
| heavy-65536-csa-cp8r4 | tilelang@cute | 2860 | 541.9 | 1.00 | 12057 | 948.8 | 1.00 | 1270 |
| heavy-65536-csa-cp8r4 | cute@cute | 2610 | 98.4 | 0.91 | 11944 | 321.9 | 0.99 | 1270 |
| heavy-65536-csa-cp8r4 | cute_ws@cute | 1229 | 60.7 | 0.43 | 10595 | 211.3 | 0.88 | 1270 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 1324 | 94.8 | 0.46 | - | - | - | 556 |
| heavy-65536-csa-cp8r7 | tilelang@main | 1888 | 661.1 | 1.00 | 7479 | 820.1 | 1.00 | 1270 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 972.7 | 74.8 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r7 | tilelang@cute | 1890 | 702.7 | 1.00 | 7474 | 841.6 | 1.00 | 1270 |
| heavy-65536-csa-cp8r7 | cute@cute | 1788 | 107.0 | 0.95 | 7371 | 228.2 | 0.99 | 1270 |
| heavy-65536-csa-cp8r7 | cute_ws@cute | 858.5 | 56.8 | 0.45 | 6451 | 121.5 | 0.86 | 1270 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 966.9 | 93.8 | 0.51 | - | - | - | 556 |
| heavy-65536-hca-cp1 | tilelang@main | 11197 | 516.8 | 1.00 | 37594 | 678.6 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5693 | -10.7 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp1 | tilelang@cute | 11249 | 491.8 | 1.00 | 37621 | 679.0 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | cute@cute | 10542 | 38.8 | 0.94 | 36891 | 80.9 | 0.98 | 8530 |
| heavy-65536-hca-cp1 | cute_ws@cute | 4909 | 78.1 | 0.44 | 31284 | 162.8 | 0.83 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@cute | 5722 | 58.8 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp8r0 | tilelang@main | 1261 | 730.2 | 1.00 | 3956 | 864.8 | 1.00 | 1235 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 660.8 | 116.2 | 0.52 | - | - | - | 540 |
| heavy-65536-hca-cp8r0 | tilelang@cute | 1257 | 704.3 | 1.00 | 3951 | 883.7 | 1.00 | 1235 |
| heavy-65536-hca-cp8r0 | cute@cute | 1172 | 131.5 | 0.93 | 3871 | 253.0 | 0.98 | 1235 |
| heavy-65536-hca-cp8r0 | cute_ws@cute | 551.8 | 55.8 | 0.44 | 3256 | 108.9 | 0.82 | 1235 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 660.8 | 126.4 | 0.53 | - | - | - | 540 |
| heavy-65536-hca-cp8r4 | tilelang@main | 1655 | 722.9 | 1.00 | 6188 | 861.1 | 1.00 | 1235 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 811.0 | 117.0 | 0.49 | - | - | - | 540 |
| heavy-65536-hca-cp8r4 | tilelang@cute | 1659 | 701.3 | 1.00 | 6194 | 907.0 | 1.00 | 1235 |
| heavy-65536-hca-cp8r4 | cute@cute | 1555 | 130.5 | 0.94 | 6096 | 274.5 | 0.98 | 1235 |
| heavy-65536-hca-cp8r4 | cute_ws@cute | 716.6 | 55.5 | 0.43 | 5264 | 129.5 | 0.85 | 1235 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 813.1 | 120.3 | 0.49 | - | - | - | 540 |
| heavy-65536-hca-cp8r7 | tilelang@main | 1308 | 698.6 | 1.00 | 4132 | 878.0 | 1.00 | 1235 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 693.7 | 105.4 | 0.53 | - | - | - | 540 |
| heavy-65536-hca-cp8r7 | tilelang@cute | 1305 | 717.4 | 1.00 | 4137 | 875.5 | 1.00 | 1235 |
| heavy-65536-hca-cp8r7 | cute@cute | 1216 | 134.5 | 0.93 | 4048 | 253.8 | 0.98 | 1235 |
| heavy-65536-hca-cp8r7 | cute_ws@cute | 584.8 | 56.5 | 0.45 | 3410 | 120.2 | 0.82 | 1235 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 690.9 | 120.8 | 0.53 | - | - | - | 540 |
| heavy-65536-sliding-cp1 | tilelang@main | 8945 | 671.5 | 1.00 | 28442 | 840.0 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4091 | 20.5 | 0.46 | - | - | - | 4192 |
| heavy-65536-sliding-cp1 | tilelang@cute | 9075 | 532.6 | 1.00 | 28460 | 822.8 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | cute@cute | 8304 | 76.7 | 0.92 | 27774 | 214.4 | 0.98 | 8432 |
| heavy-65536-sliding-cp1 | cute_ws@cute | 3172 | 41.9 | 0.35 | 22667 | 167.7 | 0.80 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4104 | 24.2 | 0.45 | - | - | - | 4192 |
| heavy-65536-sliding-cp8r0 | tilelang@main | 1132 | 721.2 | 1.00 | 3582 | 823.1 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 528.4 | 141.3 | 0.47 | - | - | - | 524 |
| heavy-65536-sliding-cp8r0 | tilelang@cute | 1135 | 686.6 | 1.00 | 3589 | 834.3 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r0 | cute@cute | 1047 | 123.7 | 0.92 | 3497 | 225.4 | 0.97 | 1222 |
| heavy-65536-sliding-cp8r0 | cute_ws@cute | 411.9 | 56.5 | 0.36 | 2870 | 116.2 | 0.80 | 1222 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 526.1 | 149.8 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r4 | tilelang@main | 1152 | 688.6 | 1.00 | 3866 | 811.3 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 524.6 | 131.3 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r4 | tilelang@cute | 1153 | 728.6 | 1.00 | 3866 | 920.6 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r4 | cute@cute | 1065 | 130.3 | 0.92 | 3775 | 255.6 | 0.98 | 1222 |
| heavy-65536-sliding-cp8r4 | cute_ws@cute | 408.3 | 55.5 | 0.35 | 3129 | 223.7 | 0.81 | 1222 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 528.4 | 140.6 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r7 | tilelang@main | 1148 | 675.9 | 1.00 | 3690 | 833.4 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 525.3 | 140.4 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r7 | tilelang@cute | 1145 | 716.6 | 1.00 | 3684 | 851.5 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r7 | cute@cute | 1057 | 128.8 | 0.92 | 3595 | 236.7 | 0.98 | 1222 |
| heavy-65536-sliding-cp8r7 | cute_ws@cute | 405.3 | 61.9 | 0.35 | 2957 | 127.1 | 0.80 | 1222 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 523.7 | 153.9 | 0.46 | - | - | - | 524 |
| tiny-65536-csa-cp1 | tilelang@main | 10414 | 470.2 | 1.00 | 31178 | 622.0 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5690 | -30.5 | 0.55 | - | - | - | 4448 |
| tiny-65536-csa-cp1 | tilelang@cute | 10428 | 477.5 | 1.00 | 31177 | 669.5 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | cute@cute | 9744 | 28.5 | 0.93 | 30501 | 107.5 | 0.98 | 8479 |
| tiny-65536-csa-cp1 | cute_ws@cute | 4854 | 69.1 | 0.47 | 25631 | 151.6 | 0.82 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@cute | 5681 | -17.4 | 0.54 | - | - | - | 4448 |
| tiny-65536-csa-cp8r0 | tilelang@main | 1324 | 680.4 | 1.00 | 4061 | 830.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 727.4 | 81.8 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r0 | tilelang@cute | 1324 | 676.2 | 1.00 | 4062 | 834.1 | 1.00 | 1269 |
| tiny-65536-csa-cp8r0 | cute@cute | 1238 | 107.5 | 0.93 | 3976 | 216.9 | 0.98 | 1269 |
| tiny-65536-csa-cp8r0 | cute_ws@cute | 620.1 | 60.5 | 0.47 | 3363 | 119.8 | 0.83 | 1269 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 728.1 | 95.7 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r4 | tilelang@main | 1326 | 681.4 | 1.00 | 4066 | 801.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 731.1 | 92.1 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r4 | tilelang@cute | 1326 | 676.0 | 1.00 | 4066 | 846.1 | 1.00 | 1269 |
| tiny-65536-csa-cp8r4 | cute@cute | 1241 | 105.5 | 0.94 | 3982 | 228.5 | 0.98 | 1269 |
| tiny-65536-csa-cp8r4 | cute_ws@cute | 629.4 | 55.7 | 0.47 | 3365 | 126.3 | 0.83 | 1269 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 732.0 | 96.7 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r7 | tilelang@main | 1330 | 734.8 | 1.00 | 4068 | 810.1 | 1.00 | 1269 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 728.8 | 102.6 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r7 | tilelang@cute | 1330 | 690.1 | 1.00 | 4063 | 849.6 | 1.00 | 1269 |
| tiny-65536-csa-cp8r7 | cute@cute | 1242 | 109.8 | 0.93 | 3978 | 233.2 | 0.98 | 1269 |
| tiny-65536-csa-cp8r7 | cute_ws@cute | 620.2 | 60.3 | 0.47 | 3363 | 125.6 | 0.83 | 1269 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 733.3 | 90.3 | 0.55 | - | - | - | 556 |
| tiny-65536-hca-cp1 | tilelang@main | 8038 | 651.6 | 1.00 | 21189 | 833.7 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4147 | 25.6 | 0.52 | - | - | - | 4192 |
| tiny-65536-hca-cp1 | tilelang@cute | 8037 | 649.6 | 1.00 | 21185 | 845.5 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | cute@cute | 7360 | 81.1 | 0.92 | 20517 | 221.9 | 0.97 | 8432 |
| tiny-65536-hca-cp1 | cute_ws@cute | 3636 | 32.1 | 0.45 | 16751 | 182.4 | 0.79 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@cute | 4130 | 31.8 | 0.51 | - | - | - | 4192 |
| tiny-65536-hca-cp8r0 | tilelang@main | 1023 | 682.2 | 1.00 | 2791 | 837.3 | 1.00 | 1222 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 527.5 | 130.4 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r0 | tilelang@cute | 1024 | 698.5 | 1.00 | 2788 | 829.5 | 1.00 | 1222 |
| tiny-65536-hca-cp8r0 | cute@cute | 942.6 | 121.8 | 0.92 | 2708 | 222.5 | 0.97 | 1222 |
| tiny-65536-hca-cp8r0 | cute_ws@cute | 461.7 | 52.4 | 0.45 | 2240 | 100.1 | 0.80 | 1222 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 523.7 | 147.7 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r4 | tilelang@main | 1031 | 700.4 | 1.00 | 2795 | 813.5 | 1.00 | 1222 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 528.8 | 133.5 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r4 | tilelang@cute | 1028 | 723.2 | 1.00 | 2791 | 861.4 | 1.00 | 1222 |
| tiny-65536-hca-cp8r4 | cute@cute | 946.6 | 125.0 | 0.92 | 2707 | 239.9 | 0.97 | 1222 |
| tiny-65536-hca-cp8r4 | cute_ws@cute | 457.0 | 50.9 | 0.44 | 2212 | 143.4 | 0.79 | 1222 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 525.3 | 139.2 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r7 | tilelang@main | 1025 | 690.8 | 1.00 | 2797 | 833.6 | 1.00 | 1222 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 530.4 | 135.8 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r7 | tilelang@cute | 1019 | 704.7 | 1.00 | 2782 | 849.4 | 1.00 | 1222 |
| tiny-65536-hca-cp8r7 | cute@cute | 938.6 | 119.6 | 0.92 | 2706 | 224.9 | 0.97 | 1222 |
| tiny-65536-hca-cp8r7 | cute_ws@cute | 455.2 | 59.5 | 0.45 | 2230 | 111.4 | 0.80 | 1222 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 530.0 | 138.4 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp1 | tilelang@main | 8031 | 685.8 | 1.00 | 21186 | 850.2 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4116 | 62.0 | 0.51 | - | - | - | 4192 |
| tiny-65536-sliding-cp1 | tilelang@cute | 8027 | 664.7 | 1.00 | 21191 | 838.0 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | cute@cute | 7371 | 64.7 | 0.92 | 20524 | 214.5 | 0.97 | 8432 |
| tiny-65536-sliding-cp1 | cute_ws@cute | 3650 | 24.3 | 0.45 | 16716 | 219.4 | 0.79 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@cute | 4137 | 24.7 | 0.52 | - | - | - | 4192 |
| tiny-65536-sliding-cp8r0 | tilelang@main | 1021 | 723.6 | 1.00 | 2791 | 839.4 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 525.1 | 143.2 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r0 | tilelang@cute | 1020 | 703.9 | 1.00 | 2788 | 848.4 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r0 | cute@cute | 939.9 | 123.4 | 0.92 | 2706 | 229.2 | 0.97 | 1222 |
| tiny-65536-sliding-cp8r0 | cute_ws@cute | 475.6 | 43.4 | 0.47 | 2234 | 112.5 | 0.80 | 1222 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 529.9 | 140.7 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp8r4 | tilelang@main | 1028 | 687.6 | 1.00 | 2791 | 821.8 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 522.8 | 139.4 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r4 | tilelang@cute | 1027 | 701.6 | 1.00 | 2792 | 850.5 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r4 | cute@cute | 947.2 | 122.8 | 0.92 | 2711 | 229.2 | 0.97 | 1222 |
| tiny-65536-sliding-cp8r4 | cute_ws@cute | 463.8 | 48.2 | 0.45 | 2231 | 108.7 | 0.80 | 1222 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 520.5 | 147.5 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r7 | tilelang@main | 1017 | 687.1 | 1.00 | 2786 | 842.6 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 529.2 | 135.9 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp8r7 | tilelang@cute | 1019 | 702.8 | 1.00 | 2786 | 838.7 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r7 | cute@cute | 938.9 | 126.1 | 0.92 | 2703 | 223.5 | 0.97 | 1222 |
| tiny-65536-sliding-cp8r7 | cute_ws@cute | 468.3 | 42.5 | 0.46 | 2236 | 109.8 | 0.80 | 1222 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 525.2 | 145.7 | 0.52 | - | - | - | 524 |

Useful FLOPs count valid slots only (fwd 4HD, bwd 10HD per slot); `exec/useful` counts the slots each
backend's tiles touch, or every padded slot for an arm without tile information. TFLOP/s divide
useful FLOPs by op-boundary time (higher is better);
`% peak` is f+b against 989.5 dense BF16 TFLOP/s (https://www.nvidia.com/en-us/data-center/h200/ (H200 SXM BF16 1,979 TFLOPS with sparsity, halved)).

| item | backend | f+b GFLOP | exec/useful fwd | exec/useful bwd | fwd TFLOP/s | f+b TFLOP/s | % peak |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 356.8 | 1.09 | 1.05 | 85.2 | 112.0 | 11.3 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 356.8 | 1.69 | 1.69 | 265.5 | - | - |
| single-2048-csa-cp1 | tilelang@cute | 356.8 | 1.09 | 1.05 | 84.1 | 109.4 | 11.1 |
| single-2048-csa-cp1 | cute@cute | 356.8 | 1.09 | 1.05 | 167.4 | 135.8 | 13.7 |
| single-2048-csa-cp1 | cute_ws@cute | 356.8 | 1.18 | 1.05 | 336.1 | 144.5 | 14.6 |
| single-2048-csa-cp1 | flashmla_fwd_ref@cute | 356.8 | 1.69 | 1.69 | 261.6 | - | - |
| single-2048-csa-cp8r0 | tilelang@main | 15.0 | 1.49 | 1.36 | 5.9 | 7.5 | 0.8 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 15.0 | 5.00 | 5.00 | 23.5 | - | - |
| single-2048-csa-cp8r0 | tilelang@cute | 15.0 | 1.49 | 1.36 | 5.8 | 7.3 | 0.7 |
| single-2048-csa-cp8r0 | cute@cute | 15.0 | 1.49 | 1.36 | 24.2 | 10.6 | 1.1 |
| single-2048-csa-cp8r0 | cute_ws@cute | 15.0 | 1.99 | 1.36 | 49.8 | 11.8 | 1.2 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 15.0 | 5.00 | 5.00 | 22.6 | - | - |
| single-2048-csa-cp8r4 | tilelang@main | 48.8 | 1.08 | 1.04 | 18.3 | 24.1 | 2.4 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 48.8 | 1.54 | 1.54 | 70.6 | - | - |
| single-2048-csa-cp8r4 | tilelang@cute | 48.8 | 1.08 | 1.04 | 17.9 | 23.3 | 2.4 |
| single-2048-csa-cp8r4 | cute@cute | 48.8 | 1.08 | 1.04 | 68.8 | 33.5 | 3.4 |
| single-2048-csa-cp8r4 | cute_ws@cute | 48.8 | 1.23 | 1.04 | 141.2 | 37.8 | 3.8 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 48.8 | 1.54 | 1.54 | 66.8 | - | - |
| single-2048-csa-cp8r7 | tilelang@main | 71.4 | 1.05 | 1.03 | 26.3 | 35.3 | 3.6 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 71.4 | 1.05 | 1.05 | 100.5 | - | - |
| single-2048-csa-cp8r7 | tilelang@cute | 71.4 | 1.05 | 1.03 | 26.0 | 34.2 | 3.5 |
| single-2048-csa-cp8r7 | cute@cute | 71.4 | 1.05 | 1.03 | 93.6 | 49.6 | 5.0 |
| single-2048-csa-cp8r7 | cute_ws@cute | 71.4 | 1.05 | 1.03 | 194.8 | 55.1 | 5.6 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 71.4 | 1.05 | 1.05 | 95.8 | - | - |
| single-2048-hca-cp1 | tilelang@main | 123.6 | 1.41 | 1.18 | 32.5 | 49.4 | 5.0 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 123.6 | 1.95 | 1.95 | 103.5 | - | - |
| single-2048-hca-cp1 | tilelang@cute | 123.6 | 1.41 | 1.18 | 32.3 | 48.2 | 4.9 |
| single-2048-hca-cp1 | cute@cute | 123.6 | 1.41 | 1.18 | 72.9 | 64.7 | 6.5 |
| single-2048-hca-cp1 | cute_ws@cute | 123.6 | 1.89 | 1.18 | 157.9 | 71.9 | 7.3 |
| single-2048-hca-cp1 | flashmla_fwd_ref@cute | 123.6 | 1.95 | 1.95 | 100.4 | - | - |
| single-2048-hca-cp8r0 | tilelang@main | 11.4 | 1.49 | 1.24 | 4.3 | 5.3 | 0.5 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 11.4 | 2.65 | 2.65 | 17.2 | - | - |
| single-2048-hca-cp8r0 | tilelang@cute | 11.4 | 1.49 | 1.24 | 4.1 | 5.2 | 0.5 |
| single-2048-hca-cp8r0 | cute@cute | 11.4 | 1.49 | 1.24 | 16.0 | 7.4 | 0.7 |
| single-2048-hca-cp8r0 | cute_ws@cute | 11.4 | 1.99 | 1.24 | 42.2 | 8.5 | 0.9 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 11.4 | 2.65 | 2.65 | 16.4 | - | - |
| single-2048-hca-cp8r4 | tilelang@main | 16.0 | 1.41 | 1.17 | 5.8 | 7.4 | 0.8 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 16.0 | 1.88 | 1.88 | 23.6 | - | - |
| single-2048-hca-cp8r4 | tilelang@cute | 16.0 | 1.41 | 1.17 | 5.8 | 7.2 | 0.7 |
| single-2048-hca-cp8r4 | cute@cute | 16.0 | 1.41 | 1.17 | 21.9 | 10.4 | 1.0 |
| single-2048-hca-cp8r4 | cute_ws@cute | 16.0 | 1.88 | 1.17 | 55.7 | 12.0 | 1.2 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 16.0 | 1.88 | 1.88 | 22.9 | - | - |
| single-2048-hca-cp8r7 | tilelang@main | 16.7 | 1.35 | 1.12 | 5.9 | 7.7 | 0.8 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 16.7 | 1.80 | 1.80 | 24.7 | - | - |
| single-2048-hca-cp8r7 | tilelang@cute | 16.7 | 1.35 | 1.12 | 6.0 | 7.6 | 0.8 |
| single-2048-hca-cp8r7 | cute@cute | 16.7 | 1.35 | 1.12 | 23.3 | 10.9 | 1.1 |
| single-2048-hca-cp8r7 | cute_ws@cute | 16.7 | 1.80 | 1.12 | 59.3 | 12.6 | 1.3 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 16.7 | 1.80 | 1.80 | 24.1 | - | - |
| single-2048-sliding-cp1 | tilelang@main | 116.5 | 1.02 | 1.01 | 32.8 | 50.9 | 5.1 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 111.6 | - | - |
| single-2048-sliding-cp1 | tilelang@cute | 116.5 | 1.02 | 1.01 | 33.6 | 49.4 | 5.0 |
| single-2048-sliding-cp1 | cute@cute | 116.5 | 1.02 | 1.01 | 80.2 | 66.0 | 6.7 |
| single-2048-sliding-cp1 | cute_ws@cute | 116.5 | 1.03 | 1.01 | 193.0 | 73.9 | 7.5 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@cute | 116.5 | 1.03 | 1.03 | 109.8 | - | - |
| single-2048-sliding-cp8r0 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.4 | 5.5 | 0.6 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 18.0 | - | - |
| single-2048-sliding-cp8r0 | tilelang@cute | 11.3 | 1.16 | 1.08 | 4.3 | 5.5 | 0.6 |
| single-2048-sliding-cp8r0 | cute@cute | 11.3 | 1.16 | 1.08 | 18.9 | 7.9 | 0.8 |
| single-2048-sliding-cp8r0 | cute_ws@cute | 11.3 | 1.33 | 1.08 | 43.6 | 8.9 | 0.9 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 11.3 | 1.33 | 1.33 | 17.3 | - | - |
| single-2048-sliding-cp8r4 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.8 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 24.0 | - | - |
| single-2048-sliding-cp8r4 | tilelang@cute | 15.0 | 1.00 | 1.00 | 5.9 | 7.1 | 0.7 |
| single-2048-sliding-cp8r4 | cute@cute | 15.0 | 1.00 | 1.00 | 24.5 | 10.3 | 1.0 |
| single-2048-sliding-cp8r4 | cute_ws@cute | 15.0 | 1.00 | 1.00 | 57.9 | 11.7 | 1.2 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 15.0 | 1.00 | 1.00 | 22.8 | - | - |
| single-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.7 | - | - |
| single-2048-sliding-cp8r7 | tilelang@cute | 15.0 | 1.00 | 1.00 | 5.9 | 7.1 | 0.7 |
| single-2048-sliding-cp8r7 | cute@cute | 15.0 | 1.00 | 1.00 | 24.5 | 10.1 | 1.0 |
| single-2048-sliding-cp8r7 | cute_ws@cute | 15.0 | 1.00 | 1.00 | 58.1 | 11.7 | 1.2 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 15.0 | 1.00 | 1.00 | 23.0 | - | - |
| short-2048-csa-cp1 | tilelang@main | 233.1 | 1.16 | 1.10 | 58.8 | 84.6 | 8.5 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 233.1 | 2.58 | 2.58 | 181.0 | - | - |
| short-2048-csa-cp1 | tilelang@cute | 233.1 | 1.16 | 1.10 | 59.2 | 82.0 | 8.3 |
| short-2048-csa-cp1 | cute@cute | 233.1 | 1.16 | 1.10 | 124.5 | 106.5 | 10.8 |
| short-2048-csa-cp1 | cute_ws@cute | 233.1 | 1.30 | 1.10 | 255.8 | 113.3 | 11.4 |
| short-2048-csa-cp1 | flashmla_fwd_ref@cute | 233.1 | 2.58 | 2.58 | 177.4 | - | - |
| short-2048-csa-cp8r0 | tilelang@main | 15.0 | 1.49 | 1.36 | 5.7 | 7.4 | 0.8 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 15.0 | 5.00 | 5.00 | 22.9 | - | - |
| short-2048-csa-cp8r0 | tilelang@cute | 15.0 | 1.49 | 1.36 | 5.8 | 7.3 | 0.7 |
| short-2048-csa-cp8r0 | cute@cute | 15.0 | 1.49 | 1.36 | 24.0 | 10.5 | 1.1 |
| short-2048-csa-cp8r0 | cute_ws@cute | 15.0 | 1.99 | 1.36 | 49.7 | 11.9 | 1.2 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 15.0 | 5.00 | 5.00 | 22.2 | - | - |
| short-2048-csa-cp8r4 | tilelang@main | 19.6 | 1.43 | 1.30 | 7.6 | 9.7 | 1.0 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 19.6 | 3.83 | 3.83 | 29.9 | - | - |
| short-2048-csa-cp8r4 | tilelang@cute | 19.6 | 1.43 | 1.30 | 7.5 | 9.4 | 0.9 |
| short-2048-csa-cp8r4 | cute@cute | 19.6 | 1.43 | 1.30 | 30.3 | 13.7 | 1.4 |
| short-2048-csa-cp8r4 | cute_ws@cute | 19.6 | 1.81 | 1.30 | 61.9 | 15.3 | 1.5 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 19.6 | 3.83 | 3.83 | 28.4 | - | - |
| short-2048-csa-cp8r7 | tilelang@main | 39.9 | 1.09 | 1.05 | 14.9 | 19.6 | 2.0 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 39.9 | 1.89 | 1.89 | 59.5 | - | - |
| short-2048-csa-cp8r7 | tilelang@cute | 39.9 | 1.09 | 1.05 | 14.9 | 19.4 | 2.0 |
| short-2048-csa-cp8r7 | cute@cute | 39.9 | 1.09 | 1.05 | 58.6 | 27.8 | 2.8 |
| short-2048-csa-cp8r7 | cute_ws@cute | 39.9 | 1.13 | 1.05 | 121.5 | 30.9 | 3.1 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 39.9 | 1.89 | 1.89 | 57.2 | - | - |
| short-2048-hca-cp1 | tilelang@main | 116.1 | 1.46 | 1.21 | 30.7 | 46.7 | 4.7 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 116.1 | 2.07 | 2.07 | 97.5 | - | - |
| short-2048-hca-cp1 | tilelang@cute | 116.1 | 1.46 | 1.21 | 30.7 | 46.0 | 4.6 |
| short-2048-hca-cp1 | cute@cute | 116.1 | 1.46 | 1.21 | 69.6 | 61.6 | 6.2 |
| short-2048-hca-cp1 | cute_ws@cute | 116.1 | 1.94 | 1.21 | 150.6 | 69.1 | 7.0 |
| short-2048-hca-cp1 | flashmla_fwd_ref@cute | 116.1 | 2.07 | 2.07 | 96.0 | - | - |
| short-2048-hca-cp8r0 | tilelang@main | 11.4 | 1.49 | 1.24 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 11.4 | 2.65 | 2.65 | 17.0 | - | - |
| short-2048-hca-cp8r0 | tilelang@cute | 11.4 | 1.49 | 1.24 | 4.2 | 4.9 | 0.5 |
| short-2048-hca-cp8r0 | cute@cute | 11.4 | 1.49 | 1.24 | 16.1 | 6.5 | 0.7 |
| short-2048-hca-cp8r0 | cute_ws@cute | 11.4 | 1.99 | 1.24 | 41.5 | 8.5 | 0.9 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 11.4 | 2.65 | 2.65 | 16.5 | - | - |
| short-2048-hca-cp8r4 | tilelang@main | 11.5 | 1.47 | 1.22 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 11.5 | 2.61 | 2.61 | 17.0 | - | - |
| short-2048-hca-cp8r4 | tilelang@cute | 11.5 | 1.47 | 1.22 | 4.2 | 4.8 | 0.5 |
| short-2048-hca-cp8r4 | cute@cute | 11.5 | 1.47 | 1.22 | 16.5 | 6.5 | 0.7 |
| short-2048-hca-cp8r4 | cute_ws@cute | 11.5 | 1.96 | 1.22 | 42.7 | 7.3 | 0.7 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 11.5 | 2.61 | 2.61 | 16.7 | - | - |
| short-2048-hca-cp8r7 | tilelang@main | 15.8 | 1.43 | 1.19 | 5.7 | 7.2 | 0.7 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 15.8 | 1.91 | 1.91 | 23.8 | - | - |
| short-2048-hca-cp8r7 | tilelang@cute | 15.8 | 1.43 | 1.19 | 5.8 | 7.3 | 0.7 |
| short-2048-hca-cp8r7 | cute@cute | 15.8 | 1.43 | 1.19 | 21.7 | 10.2 | 1.0 |
| short-2048-hca-cp8r7 | cute_ws@cute | 15.8 | 1.91 | 1.19 | 56.2 | 11.8 | 1.2 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 15.8 | 1.91 | 1.91 | 23.0 | - | - |
| short-2048-sliding-cp1 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.1 | 48.4 | 4.9 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 109.9 | - | - |
| short-2048-sliding-cp1 | tilelang@cute | 112.8 | 1.03 | 1.02 | 32.6 | 48.6 | 4.9 |
| short-2048-sliding-cp1 | cute@cute | 112.8 | 1.03 | 1.02 | 79.1 | 66.1 | 6.7 |
| short-2048-sliding-cp1 | cute_ws@cute | 112.8 | 1.07 | 1.02 | 189.0 | 73.0 | 7.4 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@cute | 112.8 | 1.07 | 1.07 | 108.2 | - | - |
| short-2048-sliding-cp8r0 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.3 | 5.5 | 0.6 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 17.8 | - | - |
| short-2048-sliding-cp8r0 | tilelang@cute | 11.3 | 1.16 | 1.08 | 4.3 | 5.5 | 0.6 |
| short-2048-sliding-cp8r0 | cute@cute | 11.3 | 1.16 | 1.08 | 18.8 | 7.9 | 0.8 |
| short-2048-sliding-cp8r0 | cute_ws@cute | 11.3 | 1.33 | 1.08 | 43.5 | 8.7 | 0.9 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 11.3 | 1.33 | 1.33 | 17.1 | - | - |
| short-2048-sliding-cp8r4 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.3 | 5.6 | 0.6 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 17.0 | - | - |
| short-2048-sliding-cp8r4 | tilelang@cute | 11.3 | 1.16 | 1.08 | 4.3 | 5.4 | 0.5 |
| short-2048-sliding-cp8r4 | cute@cute | 11.3 | 1.16 | 1.08 | 18.4 | 7.8 | 0.8 |
| short-2048-sliding-cp8r4 | cute_ws@cute | 11.3 | 1.33 | 1.08 | 43.1 | 8.7 | 0.9 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 11.3 | 1.33 | 1.33 | 16.2 | - | - |
| short-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.2 | - | - |
| short-2048-sliding-cp8r7 | tilelang@cute | 15.0 | 1.00 | 1.00 | 5.6 | 7.2 | 0.7 |
| short-2048-sliding-cp8r7 | cute@cute | 15.0 | 1.00 | 1.00 | 23.9 | 10.5 | 1.1 |
| short-2048-sliding-cp8r7 | cute_ws@cute | 15.0 | 1.00 | 1.00 | 55.8 | 11.7 | 1.2 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 15.0 | 1.00 | 1.00 | 22.2 | - | - |
| heavy-2048-csa-cp1 | tilelang@main | 272.5 | 1.16 | 1.09 | 66.5 | 92.2 | 9.3 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 272.5 | 2.21 | 2.21 | 199.8 | - | - |
| heavy-2048-csa-cp1 | tilelang@cute | 272.5 | 1.16 | 1.09 | 66.7 | 90.2 | 9.1 |
| heavy-2048-csa-cp1 | cute@cute | 272.5 | 1.16 | 1.09 | 136.4 | 114.6 | 11.6 |
| heavy-2048-csa-cp1 | cute_ws@cute | 272.5 | 1.29 | 1.09 | 278.1 | 122.2 | 12.4 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@cute | 272.5 | 2.21 | 2.21 | 197.3 | - | - |
| heavy-2048-csa-cp8r0 | tilelang@main | 10.3 | 2.16 | 1.86 | 3.9 | 5.0 | 0.5 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 10.3 | 7.32 | 7.32 | 15.5 | - | - |
| heavy-2048-csa-cp8r0 | tilelang@cute | 10.3 | 2.16 | 1.86 | 3.9 | 4.9 | 0.5 |
| heavy-2048-csa-cp8r0 | cute@cute | 10.3 | 2.16 | 1.86 | 16.1 | 7.0 | 0.7 |
| heavy-2048-csa-cp8r0 | cute_ws@cute | 10.3 | 2.89 | 1.86 | 33.6 | 7.9 | 0.8 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 10.3 | 7.32 | 7.32 | 14.2 | - | - |
| heavy-2048-csa-cp8r4 | tilelang@main | 37.7 | 1.10 | 1.05 | 13.8 | 18.2 | 1.8 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 37.7 | 2.00 | 2.00 | 53.6 | - | - |
| heavy-2048-csa-cp8r4 | tilelang@cute | 37.7 | 1.10 | 1.05 | 14.1 | 18.2 | 1.8 |
| heavy-2048-csa-cp8r4 | cute@cute | 37.7 | 1.10 | 1.05 | 55.4 | 26.2 | 2.6 |
| heavy-2048-csa-cp8r4 | cute_ws@cute | 37.7 | 1.20 | 1.05 | 115.3 | 29.3 | 3.0 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 37.7 | 2.00 | 2.00 | 54.0 | - | - |
| heavy-2048-csa-cp8r7 | tilelang@main | 60.2 | 1.06 | 1.03 | 21.7 | 29.6 | 3.0 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 60.2 | 1.25 | 1.25 | 82.1 | - | - |
| heavy-2048-csa-cp8r7 | tilelang@cute | 60.2 | 1.06 | 1.03 | 21.9 | 28.9 | 2.9 |
| heavy-2048-csa-cp8r7 | cute@cute | 60.2 | 1.06 | 1.03 | 81.7 | 42.0 | 4.2 |
| heavy-2048-csa-cp8r7 | cute_ws@cute | 60.2 | 1.12 | 1.03 | 165.8 | 46.9 | 4.7 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 60.2 | 1.25 | 1.25 | 82.0 | - | - |
| heavy-2048-hca-cp1 | tilelang@main | 113.7 | 1.44 | 1.20 | 30.3 | 46.2 | 4.7 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 113.7 | 2.11 | 2.11 | 97.0 | - | - |
| heavy-2048-hca-cp1 | tilelang@cute | 113.7 | 1.44 | 1.20 | 30.4 | 45.5 | 4.6 |
| heavy-2048-hca-cp1 | cute@cute | 113.7 | 1.44 | 1.20 | 69.0 | 61.1 | 6.2 |
| heavy-2048-hca-cp1 | cute_ws@cute | 113.7 | 1.92 | 1.20 | 152.6 | 68.0 | 6.9 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@cute | 113.7 | 2.11 | 2.11 | 95.4 | - | - |
| heavy-2048-hca-cp8r0 | tilelang@main | 8.2 | 1.57 | 1.28 | 3.0 | 3.8 | 0.4 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 8.2 | 3.68 | 3.68 | 12.3 | - | - |
| heavy-2048-hca-cp8r0 | tilelang@cute | 8.2 | 1.57 | 1.28 | 3.0 | 3.7 | 0.4 |
| heavy-2048-hca-cp8r0 | cute@cute | 8.2 | 1.57 | 1.28 | 11.7 | 5.2 | 0.5 |
| heavy-2048-hca-cp8r0 | cute_ws@cute | 8.2 | 2.21 | 1.28 | 30.1 | 6.0 | 0.6 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 8.2 | 3.68 | 3.68 | 12.0 | - | - |
| heavy-2048-hca-cp8r4 | tilelang@main | 15.7 | 1.44 | 1.20 | 5.7 | 7.4 | 0.7 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 15.7 | 1.92 | 1.92 | 23.6 | - | - |
| heavy-2048-hca-cp8r4 | tilelang@cute | 15.7 | 1.44 | 1.20 | 5.5 | 7.0 | 0.7 |
| heavy-2048-hca-cp8r4 | cute@cute | 15.7 | 1.44 | 1.20 | 20.8 | 9.9 | 1.0 |
| heavy-2048-hca-cp8r4 | cute_ws@cute | 15.7 | 1.92 | 1.20 | 53.8 | 11.4 | 1.2 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 15.7 | 1.92 | 1.92 | 21.6 | - | - |
| heavy-2048-hca-cp8r7 | tilelang@main | 16.4 | 1.38 | 1.15 | 6.0 | 7.6 | 0.8 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 16.4 | 1.83 | 1.83 | 24.7 | - | - |
| heavy-2048-hca-cp8r7 | tilelang@cute | 16.4 | 1.38 | 1.15 | 5.9 | 7.4 | 0.7 |
| heavy-2048-hca-cp8r7 | cute@cute | 16.4 | 1.38 | 1.15 | 22.6 | 10.5 | 1.1 |
| heavy-2048-hca-cp8r7 | cute_ws@cute | 16.4 | 1.83 | 1.15 | 57.4 | 12.0 | 1.2 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 16.4 | 1.83 | 1.83 | 23.4 | - | - |
| heavy-2048-sliding-cp1 | tilelang@main | 109.1 | 1.05 | 1.03 | 31.2 | 47.5 | 4.8 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 109.1 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-2048-sliding-cp1 | tilelang@cute | 109.1 | 1.05 | 1.03 | 31.1 | 46.5 | 4.7 |
| heavy-2048-sliding-cp1 | cute@cute | 109.1 | 1.05 | 1.03 | 76.6 | 63.8 | 6.4 |
| heavy-2048-sliding-cp1 | cute_ws@cute | 109.1 | 1.10 | 1.03 | 184.2 | 69.9 | 7.1 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@cute | 109.1 | 1.10 | 1.10 | 105.0 | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@main | 8.1 | 1.39 | 1.19 | 3.2 | 4.0 | 0.4 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 8.1 | 1.85 | 1.85 | 12.9 | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@cute | 8.1 | 1.39 | 1.19 | 3.1 | 3.9 | 0.4 |
| heavy-2048-sliding-cp8r0 | cute@cute | 8.1 | 1.39 | 1.19 | 13.3 | 5.6 | 0.6 |
| heavy-2048-sliding-cp8r0 | cute_ws@cute | 8.1 | 1.85 | 1.19 | 31.0 | 6.3 | 0.6 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 8.1 | 1.85 | 1.85 | 12.1 | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.8 | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@cute | 15.0 | 1.00 | 1.00 | 5.9 | 7.1 | 0.7 |
| heavy-2048-sliding-cp8r4 | cute@cute | 15.0 | 1.00 | 1.00 | 24.9 | 10.4 | 1.1 |
| heavy-2048-sliding-cp8r4 | cute_ws@cute | 15.0 | 1.00 | 1.00 | 58.2 | 11.7 | 1.2 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 15.0 | 1.00 | 1.00 | 22.6 | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.3 | 0.7 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.5 | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@cute | 15.0 | 1.00 | 1.00 | 5.7 | 7.1 | 0.7 |
| heavy-2048-sliding-cp8r7 | cute@cute | 15.0 | 1.00 | 1.00 | 24.6 | 10.2 | 1.0 |
| heavy-2048-sliding-cp8r7 | cute_ws@cute | 15.0 | 1.00 | 1.00 | 57.4 | 11.6 | 1.2 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 15.0 | 1.00 | 1.00 | 22.3 | - | - |
| tiny-2048-csa-cp1 | tilelang@main | 53.6 | 3.28 | 2.72 | 14.7 | 22.7 | 2.3 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 53.6 | 11.22 | 11.22 | 45.9 | - | - |
| tiny-2048-csa-cp1 | tilelang@cute | 53.6 | 3.28 | 2.72 | 14.7 | 22.3 | 2.3 |
| tiny-2048-csa-cp1 | cute@cute | 53.6 | 3.28 | 2.72 | 33.8 | 30.6 | 3.1 |
| tiny-2048-csa-cp1 | cute_ws@cute | 53.6 | 4.40 | 2.72 | 67.0 | 33.3 | 3.4 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@cute | 53.6 | 11.22 | 11.22 | 45.0 | - | - |
| tiny-2048-csa-cp8r0 | tilelang@main | 6.6 | 3.31 | 2.74 | 2.5 | 3.2 | 0.3 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 6.6 | 11.38 | 11.38 | 9.8 | - | - |
| tiny-2048-csa-cp8r0 | tilelang@cute | 6.6 | 3.31 | 2.74 | 2.5 | 3.1 | 0.3 |
| tiny-2048-csa-cp8r0 | cute@cute | 6.6 | 3.31 | 2.74 | 10.4 | 4.5 | 0.5 |
| tiny-2048-csa-cp8r0 | cute_ws@cute | 6.6 | 4.44 | 2.74 | 21.6 | 5.1 | 0.5 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@cute | 6.6 | 11.38 | 11.38 | 9.8 | - | - |
| tiny-2048-csa-cp8r4 | tilelang@main | 4.7 | 4.58 | 3.79 | 1.8 | 2.3 | 0.2 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 4.7 | 15.89 | 15.89 | 7.2 | - | - |
| tiny-2048-csa-cp8r4 | tilelang@cute | 4.7 | 4.58 | 3.79 | 1.8 | 2.3 | 0.2 |
| tiny-2048-csa-cp8r4 | cute@cute | 4.7 | 4.58 | 3.79 | 7.5 | 3.3 | 0.3 |
| tiny-2048-csa-cp8r4 | cute_ws@cute | 4.7 | 6.17 | 3.79 | 15.8 | 3.6 | 0.4 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@cute | 4.7 | 15.89 | 15.89 | 7.1 | - | - |
| tiny-2048-csa-cp8r7 | tilelang@main | 8.6 | 2.59 | 2.15 | 3.2 | 4.2 | 0.4 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 8.6 | 8.76 | 8.76 | 12.9 | - | - |
| tiny-2048-csa-cp8r7 | tilelang@cute | 8.6 | 2.59 | 2.15 | 3.2 | 4.0 | 0.4 |
| tiny-2048-csa-cp8r7 | cute@cute | 8.6 | 2.59 | 2.15 | 13.5 | 5.9 | 0.6 |
| tiny-2048-csa-cp8r7 | cute_ws@cute | 8.6 | 3.46 | 2.15 | 28.0 | 6.6 | 0.7 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@cute | 8.6 | 8.76 | 8.76 | 12.6 | - | - |
| tiny-2048-hca-cp1 | tilelang@main | 43.2 | 1.79 | 1.36 | 12.8 | 20.6 | 2.1 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 43.2 | 2.79 | 2.79 | 42.2 | - | - |
| tiny-2048-hca-cp1 | tilelang@cute | 43.2 | 1.79 | 1.36 | 12.9 | 19.8 | 2.0 |
| tiny-2048-hca-cp1 | cute@cute | 43.2 | 1.79 | 1.36 | 33.1 | 28.5 | 2.9 |
| tiny-2048-hca-cp1 | cute_ws@cute | 43.2 | 2.79 | 1.36 | 73.2 | 31.6 | 3.2 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@cute | 43.2 | 2.79 | 2.79 | 41.5 | - | - |
| tiny-2048-hca-cp8r0 | tilelang@main | 5.3 | 1.76 | 1.37 | 2.1 | 2.6 | 0.3 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 5.3 | 2.83 | 2.83 | 8.3 | - | - |
| tiny-2048-hca-cp8r0 | tilelang@cute | 5.3 | 1.76 | 1.37 | 2.0 | 2.5 | 0.3 |
| tiny-2048-hca-cp8r0 | cute@cute | 5.3 | 1.76 | 1.37 | 8.9 | 3.6 | 0.4 |
| tiny-2048-hca-cp8r0 | cute_ws@cute | 5.3 | 2.83 | 1.37 | 20.2 | 4.1 | 0.4 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@cute | 5.3 | 2.83 | 2.83 | 8.0 | - | - |
| tiny-2048-hca-cp8r4 | tilelang@main | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 3.8 | 3.94 | 3.94 | 6.0 | - | - |
| tiny-2048-hca-cp8r4 | tilelang@cute | 3.8 | 2.23 | 1.52 | 1.5 | 1.8 | 0.2 |
| tiny-2048-hca-cp8r4 | cute@cute | 3.8 | 2.23 | 1.52 | 6.4 | 2.6 | 0.3 |
| tiny-2048-hca-cp8r4 | cute_ws@cute | 3.8 | 3.94 | 1.52 | 14.6 | 3.0 | 0.3 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@cute | 3.8 | 3.94 | 3.94 | 5.9 | - | - |
| tiny-2048-hca-cp8r7 | tilelang@main | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 6.9 | 2.18 | 2.18 | 10.4 | - | - |
| tiny-2048-hca-cp8r7 | tilelang@cute | 6.9 | 1.60 | 1.28 | 2.6 | 3.4 | 0.3 |
| tiny-2048-hca-cp8r7 | cute@cute | 6.9 | 1.60 | 1.28 | 11.6 | 4.9 | 0.5 |
| tiny-2048-hca-cp8r7 | cute_ws@cute | 6.9 | 2.18 | 1.28 | 26.2 | 5.5 | 0.6 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@cute | 6.9 | 2.18 | 2.18 | 10.4 | - | - |
| tiny-2048-sliding-cp1 | tilelang@main | 43.2 | 1.79 | 1.36 | 12.9 | 20.5 | 2.1 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 43.2 | 2.79 | 2.79 | 42.4 | - | - |
| tiny-2048-sliding-cp1 | tilelang@cute | 43.2 | 1.79 | 1.36 | 13.0 | 20.2 | 2.0 |
| tiny-2048-sliding-cp1 | cute@cute | 43.2 | 1.79 | 1.36 | 32.7 | 28.9 | 2.9 |
| tiny-2048-sliding-cp1 | cute_ws@cute | 43.2 | 2.79 | 1.36 | 72.7 | 31.8 | 3.2 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@cute | 43.2 | 2.79 | 2.79 | 41.2 | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@main | 5.3 | 1.76 | 1.37 | 2.1 | 2.5 | 0.3 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 5.3 | 2.83 | 2.83 | 8.5 | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@cute | 5.3 | 1.76 | 1.37 | 2.0 | 2.6 | 0.3 |
| tiny-2048-sliding-cp8r0 | cute@cute | 5.3 | 1.76 | 1.37 | 9.0 | 3.7 | 0.4 |
| tiny-2048-sliding-cp8r0 | cute_ws@cute | 5.3 | 2.83 | 1.37 | 20.4 | 4.2 | 0.4 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@cute | 5.3 | 2.83 | 2.83 | 8.1 | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@main | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 3.8 | 3.94 | 3.94 | 6.1 | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@cute | 3.8 | 2.23 | 1.52 | 1.5 | 1.8 | 0.2 |
| tiny-2048-sliding-cp8r4 | cute@cute | 3.8 | 2.23 | 1.52 | 6.4 | 2.6 | 0.3 |
| tiny-2048-sliding-cp8r4 | cute_ws@cute | 3.8 | 3.94 | 1.52 | 14.3 | 3.0 | 0.3 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@cute | 3.8 | 3.94 | 3.94 | 5.8 | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@main | 6.9 | 1.60 | 1.28 | 2.6 | 3.4 | 0.3 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 6.9 | 2.18 | 2.18 | 10.6 | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@cute | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-sliding-cp8r7 | cute@cute | 6.9 | 1.60 | 1.28 | 11.3 | 4.8 | 0.5 |
| tiny-2048-sliding-cp8r7 | cute_ws@cute | 6.9 | 2.18 | 1.28 | 26.0 | 5.4 | 0.5 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@cute | 6.9 | 2.18 | 2.18 | 10.3 | - | - |
| single-4096-csa-cp1 | tilelang@main | 958.1 | 1.03 | 1.02 | 144.8 | 160.4 | 16.2 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 958.1 | 1.26 | 1.26 | 382.7 | - | - |
| single-4096-csa-cp1 | tilelang@cute | 958.1 | 1.03 | 1.02 | 145.2 | 159.2 | 16.1 |
| single-4096-csa-cp1 | cute@cute | 958.1 | 1.03 | 1.02 | 215.0 | 179.5 | 18.1 |
| single-4096-csa-cp1 | cute_ws@cute | 958.1 | 1.07 | 1.02 | 451.7 | 204.1 | 20.6 |
| single-4096-csa-cp1 | flashmla_fwd_ref@cute | 958.1 | 1.26 | 1.26 | 378.5 | - | - |
| single-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.8 | 20.3 | 2.0 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 57.2 | - | - |
| single-4096-csa-cp8r0 | tilelang@cute | 41.3 | 1.27 | 1.18 | 14.8 | 20.0 | 2.0 |
| single-4096-csa-cp8r0 | cute@cute | 41.3 | 1.27 | 1.18 | 52.1 | 28.9 | 2.9 |
| single-4096-csa-cp8r0 | cute_ws@cute | 41.3 | 1.45 | 1.18 | 109.8 | 32.8 | 3.3 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 41.3 | 3.64 | 3.64 | 53.8 | - | - |
| single-4096-csa-cp8r4 | tilelang@main | 150.3 | 1.00 | 1.00 | 49.4 | 64.4 | 6.5 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 150.3 | 1.00 | 1.00 | 172.9 | - | - |
| single-4096-csa-cp8r4 | tilelang@cute | 150.3 | 1.00 | 1.00 | 48.8 | 63.3 | 6.4 |
| single-4096-csa-cp8r4 | cute@cute | 150.3 | 1.00 | 1.00 | 140.5 | 86.9 | 8.8 |
| single-4096-csa-cp8r4 | cute_ws@cute | 150.3 | 1.00 | 1.00 | 289.8 | 95.0 | 9.6 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 150.3 | 1.00 | 1.00 | 169.8 | - | - |
| single-4096-csa-cp8r7 | tilelang@main | 150.3 | 1.00 | 1.00 | 48.8 | 64.0 | 6.5 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 150.3 | 1.00 | 1.00 | 170.5 | - | - |
| single-4096-csa-cp8r7 | tilelang@cute | 150.3 | 1.00 | 1.00 | 48.3 | 63.5 | 6.4 |
| single-4096-csa-cp8r7 | cute@cute | 150.3 | 1.00 | 1.00 | 141.0 | 86.9 | 8.8 |
| single-4096-csa-cp8r7 | cute_ws@cute | 150.3 | 1.00 | 1.00 | 288.4 | 95.1 | 9.6 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 150.3 | 1.00 | 1.00 | 164.3 | - | - |
| single-4096-hca-cp1 | tilelang@main | 265.9 | 1.34 | 1.11 | 53.0 | 84.9 | 8.6 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 265.9 | 1.81 | 1.81 | 150.3 | - | - |
| single-4096-hca-cp1 | tilelang@cute | 265.9 | 1.34 | 1.11 | 53.6 | 84.1 | 8.5 |
| single-4096-hca-cp1 | cute@cute | 265.9 | 1.34 | 1.11 | 96.3 | 105.4 | 10.7 |
| single-4096-hca-cp1 | cute_ws@cute | 265.9 | 1.78 | 1.11 | 203.9 | 114.3 | 11.6 |
| single-4096-hca-cp1 | flashmla_fwd_ref@cute | 265.9 | 1.81 | 1.81 | 148.1 | - | - |
| single-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.2 | 12.6 | 1.3 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.1 | - | - |
| single-4096-hca-cp8r0 | tilelang@cute | 26.7 | 1.48 | 1.23 | 9.2 | 12.0 | 1.2 |
| single-4096-hca-cp8r0 | cute@cute | 26.7 | 1.48 | 1.23 | 30.8 | 17.1 | 1.7 |
| single-4096-hca-cp8r0 | cute_ws@cute | 26.7 | 1.97 | 1.23 | 75.3 | 19.6 | 2.0 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 26.7 | 2.25 | 2.25 | 33.6 | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 34.2 | 1.32 | 1.10 | 11.8 | 16.0 | 1.6 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 34.2 | 1.76 | 1.76 | 45.0 | - | - |
| single-4096-hca-cp8r4 | tilelang@cute | 34.2 | 1.32 | 1.10 | 11.6 | 15.6 | 1.6 |
| single-4096-hca-cp8r4 | cute@cute | 34.2 | 1.32 | 1.10 | 39.5 | 22.0 | 2.2 |
| single-4096-hca-cp8r4 | cute_ws@cute | 34.2 | 1.76 | 1.10 | 95.5 | 25.6 | 2.6 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 34.2 | 1.76 | 1.76 | 43.9 | - | - |
| single-4096-hca-cp8r7 | tilelang@main | 37.0 | 1.22 | 1.02 | 12.8 | 17.1 | 1.7 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 37.0 | 1.63 | 1.63 | 48.7 | - | - |
| single-4096-hca-cp8r7 | tilelang@cute | 37.0 | 1.22 | 1.02 | 12.7 | 17.0 | 1.7 |
| single-4096-hca-cp8r7 | cute@cute | 37.0 | 1.22 | 1.02 | 42.3 | 24.2 | 2.4 |
| single-4096-hca-cp8r7 | cute_ws@cute | 37.0 | 1.63 | 1.02 | 102.2 | 27.7 | 2.8 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 37.0 | 1.63 | 1.63 | 47.2 | - | - |
| single-4096-sliding-cp1 | tilelang@main | 236.8 | 1.01 | 1.00 | 52.2 | 82.8 | 8.4 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 236.8 | 1.02 | 1.02 | 161.0 | - | - |
| single-4096-sliding-cp1 | tilelang@cute | 236.8 | 1.01 | 1.00 | 52.7 | 81.9 | 8.3 |
| single-4096-sliding-cp1 | cute@cute | 236.8 | 1.01 | 1.00 | 99.6 | 104.9 | 10.6 |
| single-4096-sliding-cp1 | cute_ws@cute | 236.8 | 1.02 | 1.00 | 247.9 | 111.9 | 11.3 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@cute | 236.8 | 1.02 | 1.02 | 159.2 | - | - |
| single-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 37.8 | - | - |
| single-4096-sliding-cp8r0 | tilelang@cute | 26.3 | 1.07 | 1.03 | 9.6 | 12.7 | 1.3 |
| single-4096-sliding-cp8r0 | cute@cute | 26.3 | 1.07 | 1.03 | 36.3 | 18.3 | 1.8 |
| single-4096-sliding-cp8r0 | cute_ws@cute | 26.3 | 1.14 | 1.03 | 84.4 | 20.6 | 2.1 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 26.3 | 1.14 | 1.14 | 36.0 | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 30.1 | 1.00 | 1.00 | 42.3 | - | - |
| single-4096-sliding-cp8r4 | tilelang@cute | 30.1 | 1.00 | 1.00 | 10.8 | 14.2 | 1.4 |
| single-4096-sliding-cp8r4 | cute@cute | 30.1 | 1.00 | 1.00 | 41.2 | 20.7 | 2.1 |
| single-4096-sliding-cp8r4 | cute_ws@cute | 30.1 | 1.00 | 1.00 | 96.8 | 23.4 | 2.4 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 30.1 | 1.00 | 1.00 | 41.7 | - | - |
| single-4096-sliding-cp8r7 | tilelang@main | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 30.1 | 1.00 | 1.00 | 44.3 | - | - |
| single-4096-sliding-cp8r7 | tilelang@cute | 30.1 | 1.00 | 1.00 | 10.8 | 14.3 | 1.4 |
| single-4096-sliding-cp8r7 | cute@cute | 30.1 | 1.00 | 1.00 | 40.4 | 20.6 | 2.1 |
| single-4096-sliding-cp8r7 | cute_ws@cute | 30.1 | 1.00 | 1.00 | 95.6 | 23.3 | 2.4 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 30.1 | 1.00 | 1.00 | 40.9 | - | - |
| short-4096-csa-cp1 | tilelang@main | 449.8 | 1.18 | 1.11 | 84.4 | 116.1 | 11.7 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 449.8 | 2.67 | 2.67 | 232.4 | - | - |
| short-4096-csa-cp1 | tilelang@cute | 449.8 | 1.18 | 1.11 | 85.0 | 115.6 | 11.7 |
| short-4096-csa-cp1 | cute@cute | 449.8 | 1.18 | 1.11 | 142.4 | 139.1 | 14.1 |
| short-4096-csa-cp1 | cute_ws@cute | 449.8 | 1.33 | 1.11 | 300.5 | 151.8 | 15.3 |
| short-4096-csa-cp1 | flashmla_fwd_ref@cute | 449.8 | 2.67 | 2.67 | 233.5 | - | - |
| short-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.3 | 20.1 | 2.0 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 53.1 | - | - |
| short-4096-csa-cp8r0 | tilelang@cute | 41.3 | 1.27 | 1.18 | 14.8 | 19.9 | 2.0 |
| short-4096-csa-cp8r0 | cute@cute | 41.3 | 1.27 | 1.18 | 52.5 | 28.8 | 2.9 |
| short-4096-csa-cp8r0 | cute_ws@cute | 41.3 | 1.45 | 1.18 | 110.6 | 32.1 | 3.2 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 41.3 | 3.64 | 3.64 | 55.4 | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 44.6 | 1.21 | 1.13 | 14.5 | 20.8 | 2.1 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 44.6 | 3.37 | 3.37 | 56.1 | - | - |
| short-4096-csa-cp8r4 | tilelang@cute | 44.6 | 1.21 | 1.13 | 16.1 | 21.5 | 2.2 |
| short-4096-csa-cp8r4 | cute@cute | 44.6 | 1.21 | 1.13 | 56.7 | 30.9 | 3.1 |
| short-4096-csa-cp8r4 | cute_ws@cute | 44.6 | 1.38 | 1.13 | 115.4 | 34.3 | 3.5 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 44.6 | 3.37 | 3.37 | 58.5 | - | - |
| short-4096-csa-cp8r7 | tilelang@main | 42.3 | 1.26 | 1.17 | 14.8 | 20.6 | 2.1 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 42.3 | 3.55 | 3.55 | 57.4 | - | - |
| short-4096-csa-cp8r7 | tilelang@cute | 42.3 | 1.26 | 1.17 | 15.1 | 20.5 | 2.1 |
| short-4096-csa-cp8r7 | cute@cute | 42.3 | 1.26 | 1.17 | 53.1 | 29.4 | 3.0 |
| short-4096-csa-cp8r7 | cute_ws@cute | 42.3 | 1.46 | 1.17 | 110.3 | 33.3 | 3.4 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 42.3 | 3.55 | 3.55 | 55.4 | - | - |
| short-4096-hca-cp1 | tilelang@main | 228.1 | 1.46 | 1.22 | 46.2 | 73.8 | 7.5 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 228.1 | 2.11 | 2.11 | 130.3 | - | - |
| short-4096-hca-cp1 | tilelang@cute | 228.1 | 1.46 | 1.22 | 46.9 | 74.0 | 7.5 |
| short-4096-hca-cp1 | cute@cute | 228.1 | 1.46 | 1.22 | 85.5 | 92.6 | 9.4 |
| short-4096-hca-cp1 | cute_ws@cute | 228.1 | 1.95 | 1.22 | 183.7 | 100.5 | 10.2 |
| short-4096-hca-cp1 | flashmla_fwd_ref@cute | 228.1 | 2.11 | 2.11 | 130.4 | - | - |
| short-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.0 | 12.2 | 1.2 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.3 | - | - |
| short-4096-hca-cp8r0 | tilelang@cute | 26.7 | 1.48 | 1.23 | 9.2 | 12.1 | 1.2 |
| short-4096-hca-cp8r0 | cute@cute | 26.7 | 1.48 | 1.23 | 31.3 | 17.2 | 1.7 |
| short-4096-hca-cp8r0 | cute_ws@cute | 26.7 | 1.97 | 1.23 | 75.4 | 20.1 | 2.0 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 26.7 | 2.25 | 2.25 | 34.5 | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 28.3 | 1.46 | 1.23 | 9.8 | 13.1 | 1.3 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.3 | 2.13 | 2.13 | 38.8 | - | - |
| short-4096-hca-cp8r4 | tilelang@cute | 28.3 | 1.46 | 1.23 | 9.9 | 13.0 | 1.3 |
| short-4096-hca-cp8r4 | cute@cute | 28.3 | 1.46 | 1.23 | 33.8 | 18.3 | 1.9 |
| short-4096-hca-cp8r4 | cute_ws@cute | 28.3 | 1.92 | 1.23 | 82.4 | 21.1 | 2.1 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 28.3 | 2.13 | 2.13 | 37.6 | - | - |
| short-4096-hca-cp8r7 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.2 | 12.2 | 1.2 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.7 | - | - |
| short-4096-hca-cp8r7 | tilelang@cute | 26.7 | 1.48 | 1.23 | 9.2 | 12.2 | 1.2 |
| short-4096-hca-cp8r7 | cute@cute | 26.7 | 1.48 | 1.23 | 31.4 | 17.2 | 1.7 |
| short-4096-hca-cp8r7 | cute_ws@cute | 26.7 | 1.97 | 1.23 | 75.4 | 19.9 | 2.0 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 26.7 | 2.25 | 2.25 | 34.9 | - | - |
| short-4096-sliding-cp1 | tilelang@main | 221.9 | 1.04 | 1.02 | 48.3 | 78.1 | 7.9 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 221.9 | 1.08 | 1.08 | 149.8 | - | - |
| short-4096-sliding-cp1 | tilelang@cute | 221.9 | 1.04 | 1.02 | 49.4 | 76.9 | 7.8 |
| short-4096-sliding-cp1 | cute@cute | 221.9 | 1.04 | 1.02 | 95.5 | 99.9 | 10.1 |
| short-4096-sliding-cp1 | cute_ws@cute | 221.9 | 1.08 | 1.02 | 236.6 | 107.0 | 10.8 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@cute | 221.9 | 1.08 | 1.08 | 149.0 | - | - |
| short-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 38.1 | - | - |
| short-4096-sliding-cp8r0 | tilelang@cute | 26.3 | 1.07 | 1.03 | 9.5 | 12.4 | 1.3 |
| short-4096-sliding-cp8r0 | cute@cute | 26.3 | 1.07 | 1.03 | 35.8 | 18.1 | 1.8 |
| short-4096-sliding-cp8r0 | cute_ws@cute | 26.3 | 1.14 | 1.03 | 84.3 | 20.3 | 2.1 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 26.3 | 1.14 | 1.14 | 37.4 | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 27.9 | 1.04 | 1.02 | 10.0 | 13.7 | 1.4 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 27.9 | 1.08 | 1.08 | 40.3 | - | - |
| short-4096-sliding-cp8r4 | tilelang@cute | 27.9 | 1.04 | 1.02 | 10.2 | 13.3 | 1.3 |
| short-4096-sliding-cp8r4 | cute@cute | 27.9 | 1.04 | 1.02 | 37.9 | 19.3 | 2.0 |
| short-4096-sliding-cp8r4 | cute_ws@cute | 27.9 | 1.08 | 1.02 | 88.3 | 21.8 | 2.2 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 27.9 | 1.08 | 1.08 | 38.9 | - | - |
| short-4096-sliding-cp8r7 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.7 | 12.6 | 1.3 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 38.4 | - | - |
| short-4096-sliding-cp8r7 | tilelang@cute | 26.3 | 1.07 | 1.03 | 9.5 | 12.7 | 1.3 |
| short-4096-sliding-cp8r7 | cute@cute | 26.3 | 1.07 | 1.03 | 36.1 | 18.4 | 1.9 |
| short-4096-sliding-cp8r7 | cute_ws@cute | 26.3 | 1.14 | 1.03 | 83.6 | 20.7 | 2.1 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 26.3 | 1.14 | 1.14 | 36.4 | - | - |
| heavy-4096-csa-cp1 | tilelang@main | 356.7 | 1.29 | 1.19 | 69.7 | 100.6 | 10.2 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 356.7 | 3.37 | 3.37 | 194.2 | - | - |
| heavy-4096-csa-cp1 | tilelang@cute | 356.7 | 1.29 | 1.19 | 69.5 | 100.4 | 10.1 |
| heavy-4096-csa-cp1 | cute@cute | 356.7 | 1.29 | 1.19 | 121.3 | 122.8 | 12.4 |
| heavy-4096-csa-cp1 | cute_ws@cute | 356.7 | 1.52 | 1.19 | 250.1 | 132.5 | 13.4 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@cute | 356.7 | 3.37 | 3.37 | 192.0 | - | - |
| heavy-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.9 | 20.3 | 2.1 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 56.7 | - | - |
| heavy-4096-csa-cp8r0 | tilelang@cute | 41.3 | 1.27 | 1.18 | 14.6 | 19.6 | 2.0 |
| heavy-4096-csa-cp8r0 | cute@cute | 41.3 | 1.27 | 1.18 | 51.3 | 28.4 | 2.9 |
| heavy-4096-csa-cp8r0 | cute_ws@cute | 41.3 | 1.45 | 1.18 | 108.2 | 32.3 | 3.3 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 41.3 | 3.64 | 3.64 | 53.7 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 45.5 | 1.20 | 1.12 | 15.9 | 22.2 | 2.2 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 45.5 | 3.30 | 3.30 | 59.9 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@cute | 45.5 | 1.20 | 1.12 | 16.1 | 21.4 | 2.2 |
| heavy-4096-csa-cp8r4 | cute@cute | 45.5 | 1.20 | 1.12 | 56.5 | 31.3 | 3.2 |
| heavy-4096-csa-cp8r4 | cute_ws@cute | 45.5 | 1.37 | 1.12 | 116.9 | 35.7 | 3.6 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 45.5 | 3.30 | 3.30 | 58.3 | - | - |
| heavy-4096-csa-cp8r7 | tilelang@main | 29.6 | 1.51 | 1.38 | 9.9 | 14.2 | 1.4 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 29.6 | 5.09 | 5.09 | 39.1 | - | - |
| heavy-4096-csa-cp8r7 | tilelang@cute | 29.6 | 1.51 | 1.38 | 10.4 | 13.9 | 1.4 |
| heavy-4096-csa-cp8r7 | cute@cute | 29.6 | 1.51 | 1.38 | 37.4 | 20.5 | 2.1 |
| heavy-4096-csa-cp8r7 | cute_ws@cute | 29.6 | 2.02 | 1.38 | 76.8 | 23.0 | 2.3 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 29.6 | 5.09 | 5.09 | 38.4 | - | - |
| heavy-4096-hca-cp1 | tilelang@main | 207.2 | 1.47 | 1.23 | 43.1 | 69.5 | 7.0 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 207.2 | 2.32 | 2.32 | 122.6 | - | - |
| heavy-4096-hca-cp1 | tilelang@cute | 207.2 | 1.47 | 1.23 | 42.8 | 68.6 | 6.9 |
| heavy-4096-hca-cp1 | cute@cute | 207.2 | 1.47 | 1.23 | 79.3 | 87.3 | 8.8 |
| heavy-4096-hca-cp1 | cute_ws@cute | 207.2 | 1.96 | 1.23 | 173.5 | 95.1 | 9.6 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@cute | 207.2 | 2.32 | 2.32 | 120.4 | - | - |
| heavy-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.1 | 12.4 | 1.3 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.5 | - | - |
| heavy-4096-hca-cp8r0 | tilelang@cute | 26.7 | 1.48 | 1.23 | 9.2 | 12.1 | 1.2 |
| heavy-4096-hca-cp8r0 | cute@cute | 26.7 | 1.48 | 1.23 | 31.0 | 17.2 | 1.7 |
| heavy-4096-hca-cp8r0 | cute_ws@cute | 26.7 | 1.97 | 1.23 | 76.0 | 20.0 | 2.0 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 26.7 | 2.25 | 2.25 | 34.3 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 28.7 | 1.46 | 1.22 | 9.9 | 13.3 | 1.3 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.7 | 2.10 | 2.10 | 39.5 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@cute | 28.7 | 1.46 | 1.22 | 9.8 | 12.9 | 1.3 |
| heavy-4096-hca-cp8r4 | cute@cute | 28.7 | 1.46 | 1.22 | 33.9 | 18.3 | 1.8 |
| heavy-4096-hca-cp8r4 | cute_ws@cute | 28.7 | 1.92 | 1.22 | 83.1 | 21.2 | 2.1 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 28.7 | 2.10 | 2.10 | 38.5 | - | - |
| heavy-4096-hca-cp8r7 | tilelang@main | 22.7 | 1.49 | 1.24 | 7.7 | 10.4 | 1.1 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 22.7 | 2.65 | 2.65 | 30.1 | - | - |
| heavy-4096-hca-cp8r7 | tilelang@cute | 22.7 | 1.49 | 1.24 | 7.7 | 10.2 | 1.0 |
| heavy-4096-hca-cp8r7 | cute@cute | 22.7 | 1.49 | 1.24 | 27.0 | 14.5 | 1.5 |
| heavy-4096-hca-cp8r7 | cute_ws@cute | 22.7 | 1.99 | 1.24 | 67.4 | 16.9 | 1.7 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 22.7 | 2.65 | 2.65 | 29.7 | - | - |
| heavy-4096-sliding-cp1 | tilelang@main | 203.2 | 1.09 | 1.04 | 45.3 | 69.8 | 7.1 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 203.2 | 1.18 | 1.18 | 138.7 | - | - |
| heavy-4096-sliding-cp1 | tilelang@cute | 203.2 | 1.09 | 1.04 | 44.3 | 72.1 | 7.3 |
| heavy-4096-sliding-cp1 | cute@cute | 203.2 | 1.09 | 1.04 | 86.4 | 94.0 | 9.5 |
| heavy-4096-sliding-cp1 | cute_ws@cute | 203.2 | 1.18 | 1.04 | 212.0 | 100.9 | 10.2 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@cute | 203.2 | 1.18 | 1.18 | 134.8 | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.6 | 1.3 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 37.9 | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@cute | 26.3 | 1.07 | 1.03 | 9.5 | 12.5 | 1.3 |
| heavy-4096-sliding-cp8r0 | cute@cute | 26.3 | 1.07 | 1.03 | 36.3 | 18.2 | 1.8 |
| heavy-4096-sliding-cp8r0 | cute_ws@cute | 26.3 | 1.14 | 1.03 | 84.6 | 20.2 | 2.0 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 26.3 | 1.14 | 1.14 | 35.8 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 28.3 | 1.04 | 1.02 | 10.3 | 13.8 | 1.4 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 28.3 | 1.06 | 1.06 | 41.6 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@cute | 28.3 | 1.04 | 1.02 | 10.2 | 13.4 | 1.4 |
| heavy-4096-sliding-cp8r4 | cute@cute | 28.3 | 1.04 | 1.02 | 38.8 | 19.7 | 2.0 |
| heavy-4096-sliding-cp8r4 | cute_ws@cute | 28.3 | 1.06 | 1.02 | 89.9 | 22.0 | 2.2 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 28.3 | 1.06 | 1.06 | 38.9 | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@main | 22.6 | 1.16 | 1.08 | 8.4 | 10.8 | 1.1 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 22.6 | 1.33 | 1.33 | 32.7 | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@cute | 22.6 | 1.16 | 1.08 | 8.1 | 10.8 | 1.1 |
| heavy-4096-sliding-cp8r7 | cute@cute | 22.6 | 1.16 | 1.08 | 31.1 | 15.6 | 1.6 |
| heavy-4096-sliding-cp8r7 | cute_ws@cute | 22.6 | 1.33 | 1.08 | 72.1 | 17.5 | 1.8 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 22.6 | 1.33 | 1.33 | 31.9 | - | - |
| tiny-4096-csa-cp1 | tilelang@main | 104.6 | 3.35 | 2.77 | 21.8 | 35.8 | 3.6 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 104.6 | 11.49 | 11.49 | 59.6 | - | - |
| tiny-4096-csa-cp1 | tilelang@cute | 104.6 | 3.35 | 2.77 | 21.7 | 35.8 | 3.6 |
| tiny-4096-csa-cp1 | cute@cute | 104.6 | 3.35 | 2.77 | 39.8 | 45.5 | 4.6 |
| tiny-4096-csa-cp1 | cute_ws@cute | 104.6 | 4.50 | 2.77 | 78.2 | 48.8 | 4.9 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@cute | 104.6 | 11.49 | 11.49 | 59.9 | - | - |
| tiny-4096-csa-cp8r0 | tilelang@main | 11.9 | 3.67 | 3.04 | 4.3 | 5.9 | 0.6 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 11.9 | 12.64 | 12.64 | 16.1 | - | - |
| tiny-4096-csa-cp8r0 | tilelang@cute | 11.9 | 3.67 | 3.04 | 4.2 | 5.6 | 0.6 |
| tiny-4096-csa-cp8r0 | cute@cute | 11.9 | 3.67 | 3.04 | 15.6 | 8.2 | 0.8 |
| tiny-4096-csa-cp8r0 | cute_ws@cute | 11.9 | 4.94 | 3.04 | 31.2 | 9.3 | 0.9 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@cute | 11.9 | 12.64 | 12.64 | 15.9 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 15.1 | 2.92 | 2.42 | 5.3 | 7.4 | 0.7 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 15.1 | 9.98 | 9.98 | 20.5 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@cute | 15.1 | 2.92 | 2.42 | 5.4 | 7.2 | 0.7 |
| tiny-4096-csa-cp8r4 | cute@cute | 15.1 | 2.92 | 2.42 | 19.4 | 10.4 | 1.0 |
| tiny-4096-csa-cp8r4 | cute_ws@cute | 15.1 | 3.92 | 2.42 | 39.6 | 11.7 | 1.2 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@cute | 15.1 | 9.98 | 9.98 | 19.8 | - | - |
| tiny-4096-csa-cp8r7 | tilelang@main | 14.5 | 3.04 | 2.52 | 5.3 | 7.1 | 0.7 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 14.5 | 10.36 | 10.36 | 19.7 | - | - |
| tiny-4096-csa-cp8r7 | tilelang@cute | 14.5 | 3.04 | 2.52 | 5.1 | 6.9 | 0.7 |
| tiny-4096-csa-cp8r7 | cute@cute | 14.5 | 3.04 | 2.52 | 18.5 | 9.9 | 1.0 |
| tiny-4096-csa-cp8r7 | cute_ws@cute | 14.5 | 4.07 | 2.52 | 38.4 | 11.0 | 1.1 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@cute | 14.5 | 10.36 | 10.36 | 19.2 | - | - |
| tiny-4096-hca-cp1 | tilelang@main | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 84.3 | 2.85 | 2.85 | 58.4 | - | - |
| tiny-4096-hca-cp1 | tilelang@cute | 84.3 | 1.81 | 1.37 | 19.5 | 34.2 | 3.5 |
| tiny-4096-hca-cp1 | cute@cute | 84.3 | 1.81 | 1.37 | 39.3 | 46.5 | 4.7 |
| tiny-4096-hca-cp1 | cute_ws@cute | 84.3 | 2.85 | 1.37 | 83.2 | 50.3 | 5.1 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@cute | 84.3 | 2.85 | 2.85 | 57.3 | - | - |
| tiny-4096-hca-cp8r0 | tilelang@main | 9.6 | 1.91 | 1.41 | 3.6 | 4.7 | 0.5 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 9.6 | 3.14 | 3.14 | 14.1 | - | - |
| tiny-4096-hca-cp8r0 | tilelang@cute | 9.6 | 1.91 | 1.41 | 3.5 | 4.5 | 0.5 |
| tiny-4096-hca-cp8r0 | cute@cute | 9.6 | 1.91 | 1.41 | 13.6 | 6.6 | 0.7 |
| tiny-4096-hca-cp8r0 | cute_ws@cute | 9.6 | 3.14 | 1.41 | 30.7 | 7.5 | 0.8 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@cute | 9.6 | 3.14 | 3.14 | 13.3 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.5 | 6.0 | 0.6 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.6 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@cute | 12.1 | 1.66 | 1.32 | 4.4 | 5.7 | 0.6 |
| tiny-4096-hca-cp8r4 | cute@cute | 12.1 | 1.66 | 1.32 | 17.5 | 8.2 | 0.8 |
| tiny-4096-hca-cp8r4 | cute_ws@cute | 12.1 | 2.48 | 1.32 | 39.8 | 9.2 | 0.9 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@cute | 12.1 | 2.48 | 2.48 | 17.1 | - | - |
| tiny-4096-hca-cp8r7 | tilelang@main | 11.7 | 1.70 | 1.33 | 4.3 | 5.7 | 0.6 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 11.7 | 2.57 | 2.57 | 16.6 | - | - |
| tiny-4096-hca-cp8r7 | tilelang@cute | 11.7 | 1.70 | 1.33 | 4.3 | 5.6 | 0.6 |
| tiny-4096-hca-cp8r7 | cute@cute | 11.7 | 1.70 | 1.33 | 16.6 | 8.0 | 0.8 |
| tiny-4096-hca-cp8r7 | cute_ws@cute | 11.7 | 2.57 | 1.33 | 38.0 | 9.0 | 0.9 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@cute | 11.7 | 2.57 | 2.57 | 16.1 | - | - |
| tiny-4096-sliding-cp1 | tilelang@main | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 84.3 | 2.85 | 2.85 | 58.9 | - | - |
| tiny-4096-sliding-cp1 | tilelang@cute | 84.3 | 1.81 | 1.37 | 19.6 | 34.2 | 3.5 |
| tiny-4096-sliding-cp1 | cute@cute | 84.3 | 1.81 | 1.37 | 39.0 | 46.1 | 4.7 |
| tiny-4096-sliding-cp1 | cute_ws@cute | 84.3 | 2.85 | 1.37 | 84.2 | 50.4 | 5.1 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@cute | 84.3 | 2.85 | 2.85 | 56.3 | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@main | 9.6 | 1.91 | 1.41 | 3.5 | 4.7 | 0.5 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 9.6 | 3.14 | 3.14 | 13.6 | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@cute | 9.6 | 1.91 | 1.41 | 3.5 | 4.5 | 0.5 |
| tiny-4096-sliding-cp8r0 | cute@cute | 9.6 | 1.91 | 1.41 | 13.6 | 6.6 | 0.7 |
| tiny-4096-sliding-cp8r0 | cute_ws@cute | 9.6 | 3.14 | 1.41 | 30.3 | 7.4 | 0.7 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@cute | 9.6 | 3.14 | 3.14 | 13.3 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.4 | 5.9 | 0.6 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.1 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@cute | 12.1 | 1.66 | 1.32 | 4.3 | 5.7 | 0.6 |
| tiny-4096-sliding-cp8r4 | cute@cute | 12.1 | 1.66 | 1.32 | 16.9 | 8.3 | 0.8 |
| tiny-4096-sliding-cp8r4 | cute_ws@cute | 12.1 | 2.48 | 1.32 | 39.1 | 9.5 | 1.0 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@cute | 12.1 | 2.48 | 2.48 | 16.8 | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@main | 11.7 | 1.70 | 1.33 | 4.2 | 5.6 | 0.6 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 11.7 | 2.57 | 2.57 | 16.5 | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@cute | 11.7 | 1.70 | 1.33 | 4.3 | 5.5 | 0.6 |
| tiny-4096-sliding-cp8r7 | cute@cute | 11.7 | 1.70 | 1.33 | 16.6 | 8.0 | 0.8 |
| tiny-4096-sliding-cp8r7 | cute_ws@cute | 11.7 | 2.57 | 1.33 | 36.4 | 8.9 | 0.9 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@cute | 11.7 | 2.57 | 2.57 | 15.8 | - | - |
| single-16384-csa-cp1 | tilelang@main | 4565.9 | 1.01 | 1.00 | 220.5 | 193.8 | 19.6 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 4565.9 | 1.05 | 1.05 | 507.6 | - | - |
| single-16384-csa-cp1 | tilelang@cute | 4565.9 | 1.01 | 1.00 | 221.7 | 193.6 | 19.6 |
| single-16384-csa-cp1 | cute@cute | 4565.9 | 1.01 | 1.00 | 255.5 | 201.0 | 20.3 |
| single-16384-csa-cp1 | cute_ws@cute | 4565.9 | 1.01 | 1.00 | 534.0 | 227.4 | 23.0 |
| single-16384-csa-cp1 | flashmla_fwd_ref@cute | 4565.9 | 1.05 | 1.05 | 498.0 | - | - |
| single-16384-csa-cp8r0 | tilelang@main | 356.8 | 1.09 | 1.05 | 82.7 | 110.0 | 11.1 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 356.8 | 1.69 | 1.69 | 244.1 | - | - |
| single-16384-csa-cp8r0 | tilelang@cute | 356.8 | 1.09 | 1.05 | 82.9 | 108.5 | 11.0 |
| single-16384-csa-cp8r0 | cute@cute | 356.8 | 1.09 | 1.05 | 161.0 | 135.4 | 13.7 |
| single-16384-csa-cp8r0 | cute_ws@cute | 356.8 | 1.18 | 1.05 | 331.3 | 143.5 | 14.5 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 356.8 | 1.69 | 1.69 | 238.3 | - | - |
| single-16384-csa-cp8r4 | tilelang@main | 601.3 | 1.00 | 1.00 | 121.9 | 147.4 | 14.9 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 601.3 | 1.00 | 1.00 | 347.0 | - | - |
| single-16384-csa-cp8r4 | tilelang@cute | 601.3 | 1.00 | 1.00 | 122.8 | 145.1 | 14.7 |
| single-16384-csa-cp8r4 | cute@cute | 601.3 | 1.00 | 1.00 | 215.2 | 172.3 | 17.4 |
| single-16384-csa-cp8r4 | cute_ws@cute | 601.3 | 1.00 | 1.00 | 449.0 | 179.5 | 18.1 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 601.3 | 1.00 | 1.00 | 345.1 | - | - |
| single-16384-csa-cp8r7 | tilelang@main | 601.3 | 1.00 | 1.00 | 122.3 | 146.5 | 14.8 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 601.3 | 1.00 | 1.00 | 345.9 | - | - |
| single-16384-csa-cp8r7 | tilelang@cute | 601.3 | 1.00 | 1.00 | 122.1 | 146.5 | 14.8 |
| single-16384-csa-cp8r7 | cute@cute | 601.3 | 1.00 | 1.00 | 215.9 | 173.8 | 17.6 |
| single-16384-csa-cp8r7 | cute_ws@cute | 601.3 | 1.00 | 1.00 | 449.3 | 181.1 | 18.3 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 601.3 | 1.00 | 1.00 | 345.2 | - | - |
| single-16384-hca-cp1 | tilelang@main | 1435.7 | 1.17 | 1.08 | 114.8 | 131.7 | 13.3 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1435.7 | 1.34 | 1.34 | 275.4 | - | - |
| single-16384-hca-cp1 | tilelang@cute | 1435.7 | 1.17 | 1.08 | 115.9 | 132.1 | 13.4 |
| single-16384-hca-cp1 | cute@cute | 1435.7 | 1.17 | 1.08 | 146.9 | 142.8 | 14.4 |
| single-16384-hca-cp1 | cute_ws@cute | 1435.7 | 1.34 | 1.08 | 325.2 | 170.5 | 17.2 |
| single-16384-hca-cp1 | flashmla_fwd_ref@cute | 1435.7 | 1.34 | 1.34 | 275.3 | - | - |
| single-16384-hca-cp8r0 | tilelang@main | 123.6 | 1.41 | 1.18 | 32.7 | 51.0 | 5.2 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 123.6 | 1.95 | 1.95 | 103.7 | - | - |
| single-16384-hca-cp8r0 | tilelang@cute | 123.6 | 1.41 | 1.18 | 31.8 | 50.1 | 5.1 |
| single-16384-hca-cp8r0 | cute@cute | 123.6 | 1.41 | 1.18 | 75.5 | 67.3 | 6.8 |
| single-16384-hca-cp8r0 | cute_ws@cute | 123.6 | 1.89 | 1.18 | 157.4 | 73.4 | 7.4 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 123.6 | 1.95 | 1.95 | 100.1 | - | - |
| single-16384-hca-cp8r4 | tilelang@main | 187.4 | 1.26 | 1.11 | 48.8 | 70.3 | 7.1 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 187.4 | 1.28 | 1.28 | 157.6 | - | - |
| single-16384-hca-cp8r4 | tilelang@cute | 187.4 | 1.26 | 1.11 | 48.9 | 68.9 | 7.0 |
| single-16384-hca-cp8r4 | cute@cute | 187.4 | 1.26 | 1.11 | 105.7 | 89.0 | 9.0 |
| single-16384-hca-cp8r4 | cute_ws@cute | 187.4 | 1.28 | 1.11 | 241.9 | 96.9 | 9.8 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 187.4 | 1.28 | 1.28 | 156.9 | - | - |
| single-16384-hca-cp8r7 | tilelang@main | 232.5 | 1.03 | 1.03 | 60.1 | 82.5 | 8.3 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 232.5 | 1.03 | 1.03 | 193.2 | - | - |
| single-16384-hca-cp8r7 | tilelang@cute | 232.5 | 1.03 | 1.03 | 60.1 | 82.2 | 8.3 |
| single-16384-hca-cp8r7 | cute@cute | 232.5 | 1.03 | 1.03 | 128.6 | 106.1 | 10.7 |
| single-16384-hca-cp8r7 | cute_ws@cute | 232.5 | 1.03 | 1.03 | 293.2 | 113.1 | 11.4 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 232.5 | 1.03 | 1.03 | 189.8 | - | - |
| single-16384-sliding-cp1 | tilelang@main | 958.3 | 1.00 | 1.00 | 92.1 | 115.7 | 11.7 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 958.3 | 1.00 | 1.00 | 235.2 | - | - |
| single-16384-sliding-cp1 | tilelang@cute | 958.3 | 1.00 | 1.00 | 91.9 | 115.5 | 11.7 |
| single-16384-sliding-cp1 | cute@cute | 958.3 | 1.00 | 1.00 | 123.1 | 127.7 | 12.9 |
| single-16384-sliding-cp1 | cute_ws@cute | 958.3 | 1.00 | 1.00 | 320.7 | 157.6 | 15.9 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@cute | 958.3 | 1.00 | 1.00 | 234.1 | - | - |
| single-16384-sliding-cp8r0 | tilelang@main | 116.5 | 1.02 | 1.01 | 32.7 | 49.9 | 5.0 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 111.1 | - | - |
| single-16384-sliding-cp8r0 | tilelang@cute | 116.5 | 1.02 | 1.01 | 32.8 | 49.1 | 5.0 |
| single-16384-sliding-cp8r0 | cute@cute | 116.5 | 1.02 | 1.01 | 80.3 | 67.2 | 6.8 |
| single-16384-sliding-cp8r0 | cute_ws@cute | 116.5 | 1.03 | 1.01 | 194.7 | 73.5 | 7.4 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 116.5 | 1.03 | 1.03 | 111.1 | - | - |
| single-16384-sliding-cp8r4 | tilelang@main | 120.3 | 1.00 | 1.00 | 34.2 | 50.3 | 5.1 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 115.8 | - | - |
| single-16384-sliding-cp8r4 | tilelang@cute | 120.3 | 1.00 | 1.00 | 34.0 | 47.4 | 4.8 |
| single-16384-sliding-cp8r4 | cute@cute | 120.3 | 1.00 | 1.00 | 82.1 | 62.2 | 6.3 |
| single-16384-sliding-cp8r4 | cute_ws@cute | 120.3 | 1.00 | 1.00 | 202.0 | 66.6 | 6.7 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 120.3 | 1.00 | 1.00 | 114.0 | - | - |
| single-16384-sliding-cp8r7 | tilelang@main | 120.3 | 1.00 | 1.00 | 34.5 | 50.3 | 5.1 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 116.9 | - | - |
| single-16384-sliding-cp8r7 | tilelang@cute | 120.3 | 1.00 | 1.00 | 34.1 | 47.5 | 4.8 |
| single-16384-sliding-cp8r7 | cute@cute | 120.3 | 1.00 | 1.00 | 81.9 | 62.2 | 6.3 |
| single-16384-sliding-cp8r7 | cute_ws@cute | 120.3 | 1.00 | 1.00 | 200.3 | 67.1 | 6.8 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 120.3 | 1.00 | 1.00 | 114.1 | - | - |
| short-16384-csa-cp1 | tilelang@main | 2610.6 | 1.11 | 1.06 | 165.4 | 163.4 | 16.5 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 2610.6 | 1.84 | 1.84 | 381.2 | - | - |
| short-16384-csa-cp1 | tilelang@cute | 2610.6 | 1.11 | 1.06 | 165.6 | 164.0 | 16.6 |
| short-16384-csa-cp1 | cute@cute | 2610.6 | 1.11 | 1.06 | 201.3 | 173.1 | 17.5 |
| short-16384-csa-cp1 | cute_ws@cute | 2610.6 | 1.20 | 1.06 | 418.0 | 198.9 | 20.1 |
| short-16384-csa-cp1 | flashmla_fwd_ref@cute | 2610.6 | 1.84 | 1.84 | 377.0 | - | - |
| short-16384-csa-cp8r0 | tilelang@main | 280.2 | 1.14 | 1.08 | 68.1 | 93.5 | 9.5 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 280.2 | 2.15 | 2.15 | 205.3 | - | - |
| short-16384-csa-cp8r0 | tilelang@cute | 280.2 | 1.14 | 1.08 | 68.1 | 92.6 | 9.4 |
| short-16384-csa-cp8r0 | cute@cute | 280.2 | 1.14 | 1.08 | 139.5 | 118.4 | 12.0 |
| short-16384-csa-cp8r0 | cute_ws@cute | 280.2 | 1.26 | 1.08 | 287.6 | 126.1 | 12.7 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 280.2 | 2.15 | 2.15 | 205.6 | - | - |
| short-16384-csa-cp8r4 | tilelang@main | 410.5 | 1.07 | 1.04 | 93.9 | 121.2 | 12.3 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 410.5 | 1.46 | 1.46 | 271.8 | - | - |
| short-16384-csa-cp8r4 | tilelang@cute | 410.5 | 1.07 | 1.04 | 94.0 | 118.3 | 12.0 |
| short-16384-csa-cp8r4 | cute@cute | 410.5 | 1.07 | 1.04 | 178.7 | 145.9 | 14.7 |
| short-16384-csa-cp8r4 | cute_ws@cute | 410.5 | 1.12 | 1.04 | 367.8 | 154.7 | 15.6 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 410.5 | 1.46 | 1.46 | 270.2 | - | - |
| short-16384-csa-cp8r7 | tilelang@main | 352.9 | 1.10 | 1.06 | 83.3 | 108.9 | 11.0 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 352.9 | 1.70 | 1.70 | 247.1 | - | - |
| short-16384-csa-cp8r7 | tilelang@cute | 352.9 | 1.10 | 1.06 | 82.0 | 108.1 | 10.9 |
| short-16384-csa-cp8r7 | cute@cute | 352.9 | 1.10 | 1.06 | 161.7 | 135.0 | 13.6 |
| short-16384-csa-cp8r7 | cute_ws@cute | 352.9 | 1.20 | 1.06 | 326.6 | 143.3 | 14.5 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 352.9 | 1.70 | 1.70 | 239.6 | - | - |
| short-16384-hca-cp1 | tilelang@main | 963.6 | 1.42 | 1.18 | 82.5 | 104.5 | 10.6 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 963.6 | 2.00 | 2.00 | 188.2 | - | - |
| short-16384-hca-cp1 | tilelang@cute | 963.6 | 1.42 | 1.18 | 82.8 | 104.4 | 10.6 |
| short-16384-hca-cp1 | cute@cute | 963.6 | 1.42 | 1.18 | 107.6 | 115.1 | 11.6 |
| short-16384-hca-cp1 | cute_ws@cute | 963.6 | 1.90 | 1.18 | 221.7 | 137.9 | 13.9 |
| short-16384-hca-cp1 | flashmla_fwd_ref@cute | 963.6 | 2.00 | 2.00 | 188.3 | - | - |
| short-16384-hca-cp8r0 | tilelang@main | 117.6 | 1.44 | 1.20 | 31.4 | 47.0 | 4.8 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 117.6 | 2.05 | 2.05 | 99.1 | - | - |
| short-16384-hca-cp8r0 | tilelang@cute | 117.6 | 1.44 | 1.20 | 31.2 | 46.4 | 4.7 |
| short-16384-hca-cp8r0 | cute@cute | 117.6 | 1.44 | 1.20 | 70.7 | 61.7 | 6.2 |
| short-16384-hca-cp8r0 | cute_ws@cute | 117.6 | 1.92 | 1.20 | 152.6 | 68.8 | 7.0 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 117.6 | 2.05 | 2.05 | 98.2 | - | - |
| short-16384-hca-cp8r4 | tilelang@main | 125.5 | 1.39 | 1.16 | 33.2 | 49.5 | 5.0 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 125.5 | 1.92 | 1.92 | 106.2 | - | - |
| short-16384-hca-cp8r4 | tilelang@cute | 125.5 | 1.39 | 1.16 | 33.0 | 48.1 | 4.9 |
| short-16384-hca-cp8r4 | cute@cute | 125.5 | 1.39 | 1.16 | 73.7 | 64.1 | 6.5 |
| short-16384-hca-cp8r4 | cute_ws@cute | 125.5 | 1.86 | 1.16 | 161.2 | 71.2 | 7.2 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 125.5 | 1.92 | 1.92 | 102.7 | - | - |
| short-16384-hca-cp8r7 | tilelang@main | 119.9 | 1.41 | 1.18 | 31.7 | 46.9 | 4.7 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 119.9 | 2.01 | 2.01 | 101.1 | - | - |
| short-16384-hca-cp8r7 | tilelang@cute | 119.9 | 1.41 | 1.18 | 31.7 | 46.6 | 4.7 |
| short-16384-hca-cp8r7 | cute@cute | 119.9 | 1.41 | 1.18 | 71.8 | 62.0 | 6.3 |
| short-16384-hca-cp8r7 | cute_ws@cute | 119.9 | 1.88 | 1.18 | 156.9 | 69.1 | 7.0 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 119.9 | 2.01 | 2.01 | 99.0 | - | - |
| short-16384-sliding-cp1 | tilelang@main | 913.6 | 1.03 | 1.01 | 87.4 | 112.0 | 11.3 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 913.6 | 1.05 | 1.05 | 223.3 | - | - |
| short-16384-sliding-cp1 | tilelang@cute | 913.6 | 1.03 | 1.01 | 88.2 | 112.0 | 11.3 |
| short-16384-sliding-cp1 | cute@cute | 913.6 | 1.03 | 1.01 | 117.9 | 124.2 | 12.5 |
| short-16384-sliding-cp1 | cute_ws@cute | 913.6 | 1.05 | 1.01 | 305.7 | 153.3 | 15.5 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@cute | 913.6 | 1.05 | 1.05 | 222.6 | - | - |
| short-16384-sliding-cp8r0 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.2 | 48.9 | 4.9 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 108.8 | - | - |
| short-16384-sliding-cp8r0 | tilelang@cute | 112.8 | 1.03 | 1.02 | 31.6 | 47.3 | 4.8 |
| short-16384-sliding-cp8r0 | cute@cute | 112.8 | 1.03 | 1.02 | 79.5 | 63.6 | 6.4 |
| short-16384-sliding-cp8r0 | cute_ws@cute | 112.8 | 1.07 | 1.02 | 189.6 | 71.6 | 7.2 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 112.8 | 1.07 | 1.07 | 106.1 | - | - |
| short-16384-sliding-cp8r4 | tilelang@main | 116.5 | 1.02 | 1.01 | 33.5 | 49.2 | 5.0 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 113.1 | - | - |
| short-16384-sliding-cp8r4 | tilelang@cute | 116.5 | 1.02 | 1.01 | 30.9 | 48.4 | 4.9 |
| short-16384-sliding-cp8r4 | cute@cute | 116.5 | 1.02 | 1.01 | 78.9 | 63.9 | 6.5 |
| short-16384-sliding-cp8r4 | cute_ws@cute | 116.5 | 1.03 | 1.01 | 193.0 | 72.2 | 7.3 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 116.5 | 1.03 | 1.03 | 106.0 | - | - |
| short-16384-sliding-cp8r7 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.2 | 47.8 | 4.8 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 109.4 | - | - |
| short-16384-sliding-cp8r7 | tilelang@cute | 112.8 | 1.03 | 1.02 | 31.0 | 47.2 | 4.8 |
| short-16384-sliding-cp8r7 | cute@cute | 112.8 | 1.03 | 1.02 | 76.9 | 64.3 | 6.5 |
| short-16384-sliding-cp8r7 | cute_ws@cute | 112.8 | 1.07 | 1.02 | 186.3 | 70.6 | 7.1 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 112.8 | 1.07 | 1.07 | 106.1 | - | - |
| heavy-16384-csa-cp1 | tilelang@main | 1795.9 | 1.22 | 1.14 | 129.3 | 138.2 | 14.0 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1795.9 | 2.68 | 2.68 | 297.3 | - | - |
| heavy-16384-csa-cp1 | tilelang@cute | 1795.9 | 1.22 | 1.14 | 129.6 | 138.9 | 14.0 |
| heavy-16384-csa-cp1 | cute@cute | 1795.9 | 1.22 | 1.14 | 160.7 | 148.3 | 15.0 |
| heavy-16384-csa-cp1 | cute_ws@cute | 1795.9 | 1.40 | 1.14 | 329.4 | 172.3 | 17.4 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@cute | 1795.9 | 2.68 | 2.68 | 295.6 | - | - |
| heavy-16384-csa-cp8r0 | tilelang@main | 154.2 | 1.36 | 1.24 | 40.3 | 59.9 | 6.1 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 154.2 | 3.90 | 3.90 | 127.1 | - | - |
| heavy-16384-csa-cp8r0 | tilelang@cute | 154.2 | 1.36 | 1.24 | 40.2 | 59.1 | 6.0 |
| heavy-16384-csa-cp8r0 | cute@cute | 154.2 | 1.36 | 1.24 | 89.9 | 78.1 | 7.9 |
| heavy-16384-csa-cp8r0 | cute_ws@cute | 154.2 | 1.65 | 1.24 | 183.4 | 84.7 | 8.6 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 154.2 | 3.90 | 3.90 | 124.2 | - | - |
| heavy-16384-csa-cp8r4 | tilelang@main | 219.5 | 1.19 | 1.12 | 56.2 | 79.5 | 8.0 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 219.5 | 2.74 | 2.74 | 172.9 | - | - |
| heavy-16384-csa-cp8r4 | tilelang@cute | 219.5 | 1.19 | 1.12 | 55.5 | 77.3 | 7.8 |
| heavy-16384-csa-cp8r4 | cute@cute | 219.5 | 1.19 | 1.12 | 118.1 | 100.6 | 10.2 |
| heavy-16384-csa-cp8r4 | cute_ws@cute | 219.5 | 1.35 | 1.12 | 244.9 | 107.8 | 10.9 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 219.5 | 2.74 | 2.74 | 167.3 | - | - |
| heavy-16384-csa-cp8r7 | tilelang@main | 410.5 | 1.06 | 1.03 | 92.8 | 117.9 | 11.9 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 410.5 | 1.46 | 1.46 | 273.7 | - | - |
| heavy-16384-csa-cp8r7 | tilelang@cute | 410.5 | 1.06 | 1.03 | 90.9 | 117.1 | 11.8 |
| heavy-16384-csa-cp8r7 | cute@cute | 410.5 | 1.06 | 1.03 | 174.5 | 143.8 | 14.5 |
| heavy-16384-csa-cp8r7 | cute_ws@cute | 410.5 | 1.12 | 1.03 | 363.1 | 152.2 | 15.4 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 410.5 | 1.46 | 1.46 | 264.4 | - | - |
| heavy-16384-hca-cp1 | tilelang@main | 847.5 | 1.45 | 1.21 | 74.5 | 97.4 | 9.8 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 847.5 | 2.27 | 2.27 | 172.5 | - | - |
| heavy-16384-hca-cp1 | tilelang@cute | 847.5 | 1.45 | 1.21 | 75.1 | 97.5 | 9.9 |
| heavy-16384-hca-cp1 | cute@cute | 847.5 | 1.45 | 1.21 | 98.7 | 107.9 | 10.9 |
| heavy-16384-hca-cp1 | cute_ws@cute | 847.5 | 1.94 | 1.21 | 208.3 | 130.4 | 13.2 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@cute | 847.5 | 2.27 | 2.27 | 172.1 | - | - |
| heavy-16384-hca-cp8r0 | tilelang@main | 99.2 | 1.48 | 1.23 | 26.6 | 40.5 | 4.1 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 99.2 | 2.42 | 2.42 | 84.5 | - | - |
| heavy-16384-hca-cp8r0 | tilelang@cute | 99.2 | 1.48 | 1.23 | 26.5 | 40.3 | 4.1 |
| heavy-16384-hca-cp8r0 | cute@cute | 99.2 | 1.48 | 1.23 | 61.8 | 54.3 | 5.5 |
| heavy-16384-hca-cp8r0 | cute_ws@cute | 99.2 | 1.97 | 1.23 | 138.1 | 60.9 | 6.2 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 99.2 | 2.42 | 2.42 | 85.0 | - | - |
| heavy-16384-hca-cp8r4 | tilelang@main | 112.5 | 1.46 | 1.22 | 30.1 | 45.2 | 4.6 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 112.5 | 2.14 | 2.14 | 94.1 | - | - |
| heavy-16384-hca-cp8r4 | tilelang@cute | 112.5 | 1.46 | 1.22 | 30.0 | 44.5 | 4.5 |
| heavy-16384-hca-cp8r4 | cute@cute | 112.5 | 1.46 | 1.22 | 68.5 | 60.2 | 6.1 |
| heavy-16384-hca-cp8r4 | cute_ws@cute | 112.5 | 1.94 | 1.22 | 148.7 | 66.8 | 6.8 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 112.5 | 2.14 | 2.14 | 93.5 | - | - |
| heavy-16384-hca-cp8r7 | tilelang@main | 129.0 | 1.40 | 1.17 | 33.9 | 49.2 | 5.0 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 129.0 | 1.86 | 1.86 | 106.5 | - | - |
| heavy-16384-hca-cp8r7 | tilelang@cute | 129.0 | 1.40 | 1.17 | 33.2 | 50.1 | 5.1 |
| heavy-16384-hca-cp8r7 | cute@cute | 129.0 | 1.40 | 1.17 | 74.6 | 66.1 | 6.7 |
| heavy-16384-hca-cp8r7 | cute_ws@cute | 129.0 | 1.86 | 1.17 | 161.5 | 73.5 | 7.4 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 129.0 | 1.86 | 1.86 | 103.6 | - | - |
| heavy-16384-sliding-cp1 | tilelang@main | 820.4 | 1.09 | 1.04 | 78.7 | 104.0 | 10.5 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 820.4 | 1.17 | 1.17 | 199.8 | - | - |
| heavy-16384-sliding-cp1 | tilelang@cute | 820.4 | 1.09 | 1.04 | 79.7 | 104.0 | 10.5 |
| heavy-16384-sliding-cp1 | cute@cute | 820.4 | 1.09 | 1.04 | 106.6 | 115.6 | 11.7 |
| heavy-16384-sliding-cp1 | cute_ws@cute | 820.4 | 1.17 | 1.04 | 269.9 | 143.2 | 14.5 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@cute | 820.4 | 1.17 | 1.17 | 199.3 | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@main | 97.9 | 1.11 | 1.06 | 27.6 | 42.7 | 4.3 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 97.9 | 1.23 | 1.23 | 93.9 | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@cute | 97.9 | 1.11 | 1.06 | 27.8 | 42.2 | 4.3 |
| heavy-16384-sliding-cp8r0 | cute@cute | 97.9 | 1.11 | 1.06 | 69.6 | 57.9 | 5.9 |
| heavy-16384-sliding-cp8r0 | cute_ws@cute | 97.9 | 1.23 | 1.06 | 164.6 | 63.9 | 6.5 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 97.9 | 1.23 | 1.23 | 93.4 | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@main | 109.5 | 1.05 | 1.02 | 31.8 | 47.0 | 4.7 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 109.5 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@cute | 109.5 | 1.05 | 1.02 | 31.7 | 46.7 | 4.7 |
| heavy-16384-sliding-cp8r4 | cute@cute | 109.5 | 1.05 | 1.02 | 76.6 | 64.1 | 6.5 |
| heavy-16384-sliding-cp8r4 | cute_ws@cute | 109.5 | 1.10 | 1.02 | 183.9 | 69.9 | 7.1 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 109.5 | 1.10 | 1.10 | 104.3 | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@main | 120.3 | 1.00 | 1.00 | 33.6 | 50.0 | 5.1 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 115.9 | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@cute | 120.3 | 1.00 | 1.00 | 34.3 | 49.9 | 5.0 |
| heavy-16384-sliding-cp8r7 | cute@cute | 120.3 | 1.00 | 1.00 | 82.0 | 68.1 | 6.9 |
| heavy-16384-sliding-cp8r7 | cute_ws@cute | 120.3 | 1.00 | 1.00 | 200.9 | 74.6 | 7.5 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 120.3 | 1.00 | 1.00 | 114.9 | - | - |
| tiny-16384-csa-cp1 | tilelang@main | 391.4 | 3.56 | 2.95 | 34.0 | 44.9 | 4.5 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 391.4 | 12.29 | 12.29 | 75.8 | - | - |
| tiny-16384-csa-cp1 | tilelang@cute | 391.4 | 3.56 | 2.95 | 33.7 | 44.9 | 4.5 |
| tiny-16384-csa-cp1 | cute@cute | 391.4 | 3.56 | 2.95 | 44.0 | 49.5 | 5.0 |
| tiny-16384-csa-cp1 | cute_ws@cute | 391.4 | 4.79 | 2.95 | 85.9 | 59.2 | 6.0 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@cute | 391.4 | 12.29 | 12.29 | 75.7 | - | - |
| tiny-16384-csa-cp8r0 | tilelang@main | 50.6 | 3.45 | 2.86 | 13.9 | 21.3 | 2.2 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 50.6 | 11.89 | 11.89 | 43.2 | - | - |
| tiny-16384-csa-cp8r0 | tilelang@cute | 50.6 | 3.45 | 2.86 | 13.9 | 21.1 | 2.1 |
| tiny-16384-csa-cp8r0 | cute@cute | 50.6 | 3.45 | 2.86 | 31.9 | 28.7 | 2.9 |
| tiny-16384-csa-cp8r0 | cute_ws@cute | 50.6 | 4.64 | 2.86 | 63.3 | 31.3 | 3.2 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@cute | 50.6 | 11.89 | 11.89 | 42.0 | - | - |
| tiny-16384-csa-cp8r4 | tilelang@main | 48.5 | 3.60 | 2.98 | 13.2 | 20.6 | 2.1 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 48.5 | 12.39 | 12.39 | 40.5 | - | - |
| tiny-16384-csa-cp8r4 | tilelang@cute | 48.5 | 3.60 | 2.98 | 13.2 | 20.2 | 2.0 |
| tiny-16384-csa-cp8r4 | cute@cute | 48.5 | 3.60 | 2.98 | 30.3 | 27.5 | 2.8 |
| tiny-16384-csa-cp8r4 | cute_ws@cute | 48.5 | 4.84 | 2.98 | 60.8 | 30.1 | 3.0 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@cute | 48.5 | 12.39 | 12.39 | 40.3 | - | - |
| tiny-16384-csa-cp8r7 | tilelang@main | 43.9 | 3.96 | 3.27 | 12.1 | 18.5 | 1.9 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 43.9 | 13.71 | 13.71 | 37.5 | - | - |
| tiny-16384-csa-cp8r7 | tilelang@cute | 43.9 | 3.96 | 3.27 | 11.9 | 18.3 | 1.9 |
| tiny-16384-csa-cp8r7 | cute@cute | 43.9 | 3.96 | 3.27 | 27.5 | 25.1 | 2.5 |
| tiny-16384-csa-cp8r7 | cute_ws@cute | 43.9 | 5.33 | 3.27 | 54.8 | 27.4 | 2.8 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@cute | 43.9 | 13.71 | 13.71 | 36.5 | - | - |
| tiny-16384-hca-cp1 | tilelang@main | 315.4 | 1.89 | 1.40 | 33.3 | 51.0 | 5.2 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 315.4 | 3.05 | 3.05 | 76.6 | - | - |
| tiny-16384-hca-cp1 | tilelang@cute | 315.4 | 1.89 | 1.40 | 32.9 | 50.9 | 5.1 |
| tiny-16384-hca-cp1 | cute@cute | 315.4 | 1.89 | 1.40 | 45.3 | 58.3 | 5.9 |
| tiny-16384-hca-cp1 | cute_ws@cute | 315.4 | 3.05 | 1.40 | 91.6 | 72.2 | 7.3 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@cute | 315.4 | 3.05 | 3.05 | 75.7 | - | - |
| tiny-16384-hca-cp8r0 | tilelang@main | 40.7 | 1.86 | 1.39 | 11.9 | 19.0 | 1.9 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-hca-cp8r0 | tilelang@cute | 40.7 | 1.86 | 1.39 | 11.7 | 18.7 | 1.9 |
| tiny-16384-hca-cp8r0 | cute@cute | 40.7 | 1.86 | 1.39 | 29.9 | 26.9 | 2.7 |
| tiny-16384-hca-cp8r0 | cute_ws@cute | 40.7 | 2.95 | 1.39 | 67.7 | 30.0 | 3.0 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@cute | 40.7 | 2.95 | 2.95 | 37.8 | - | - |
| tiny-16384-hca-cp8r4 | tilelang@main | 39.1 | 1.88 | 1.40 | 11.5 | 18.7 | 1.9 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 39.1 | 3.07 | 3.07 | 38.1 | - | - |
| tiny-16384-hca-cp8r4 | tilelang@cute | 39.1 | 1.88 | 1.40 | 11.4 | 18.1 | 1.8 |
| tiny-16384-hca-cp8r4 | cute@cute | 39.1 | 1.88 | 1.40 | 29.3 | 25.9 | 2.6 |
| tiny-16384-hca-cp8r4 | cute_ws@cute | 39.1 | 3.07 | 1.40 | 66.7 | 28.8 | 2.9 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@cute | 39.1 | 3.07 | 3.07 | 37.0 | - | - |
| tiny-16384-hca-cp8r7 | tilelang@main | 35.4 | 2.01 | 1.45 | 10.6 | 16.6 | 1.7 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 35.4 | 3.40 | 3.40 | 34.3 | - | - |
| tiny-16384-hca-cp8r7 | tilelang@cute | 35.4 | 2.01 | 1.45 | 10.5 | 16.3 | 1.6 |
| tiny-16384-hca-cp8r7 | cute@cute | 35.4 | 2.01 | 1.45 | 27.0 | 23.7 | 2.4 |
| tiny-16384-hca-cp8r7 | cute_ws@cute | 35.4 | 3.40 | 1.45 | 60.0 | 26.3 | 2.7 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@cute | 35.4 | 3.40 | 3.40 | 32.8 | - | - |
| tiny-16384-sliding-cp1 | tilelang@main | 315.4 | 1.89 | 1.40 | 33.0 | 51.2 | 5.2 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 315.4 | 3.05 | 3.05 | 76.3 | - | - |
| tiny-16384-sliding-cp1 | tilelang@cute | 315.4 | 1.89 | 1.40 | 33.0 | 50.7 | 5.1 |
| tiny-16384-sliding-cp1 | cute@cute | 315.4 | 1.89 | 1.40 | 45.6 | 58.2 | 5.9 |
| tiny-16384-sliding-cp1 | cute_ws@cute | 315.4 | 3.05 | 1.40 | 91.7 | 72.1 | 7.3 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@cute | 315.4 | 3.05 | 3.05 | 76.1 | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@main | 40.7 | 1.86 | 1.39 | 12.1 | 19.0 | 1.9 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@cute | 40.7 | 1.86 | 1.39 | 11.1 | 18.7 | 1.9 |
| tiny-16384-sliding-cp8r0 | cute@cute | 40.7 | 1.86 | 1.39 | 29.9 | 26.7 | 2.7 |
| tiny-16384-sliding-cp8r0 | cute_ws@cute | 40.7 | 2.95 | 1.39 | 68.3 | 29.6 | 3.0 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@cute | 40.7 | 2.95 | 2.95 | 38.3 | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@main | 39.1 | 1.88 | 1.40 | 11.6 | 18.0 | 1.8 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 39.1 | 3.07 | 3.07 | 37.4 | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@cute | 39.1 | 1.88 | 1.40 | 11.5 | 18.1 | 1.8 |
| tiny-16384-sliding-cp8r4 | cute@cute | 39.1 | 1.88 | 1.40 | 29.3 | 26.1 | 2.6 |
| tiny-16384-sliding-cp8r4 | cute_ws@cute | 39.1 | 3.07 | 1.40 | 66.5 | 29.0 | 2.9 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@cute | 39.1 | 3.07 | 3.07 | 37.6 | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@main | 35.4 | 2.01 | 1.45 | 10.5 | 16.5 | 1.7 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 35.4 | 3.40 | 3.40 | 34.5 | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@cute | 35.4 | 2.01 | 1.45 | 10.5 | 16.6 | 1.7 |
| tiny-16384-sliding-cp8r7 | cute@cute | 35.4 | 2.01 | 1.45 | 26.9 | 23.2 | 2.3 |
| tiny-16384-sliding-cp8r7 | cute_ws@cute | 35.4 | 3.40 | 1.45 | 59.8 | 26.2 | 2.6 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@cute | 35.4 | 3.40 | 3.40 | 32.9 | - | - |
| single-49208-csa-cp1 | tilelang@main | 14203.1 | 1.00 | 1.00 | 247.1 | 200.9 | 20.3 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 14203.1 | 1.02 | 1.02 | 488.4 | - | - |
| single-49208-csa-cp1 | tilelang@cute | 14203.1 | 1.00 | 1.00 | 247.6 | 200.5 | 20.3 |
| single-49208-csa-cp1 | cute@cute | 14203.1 | 1.00 | 1.00 | 267.9 | 204.5 | 20.7 |
| single-49208-csa-cp1 | cute_ws@cute | 14203.1 | 1.00 | 1.00 | 515.5 | 230.0 | 23.2 |
| single-49208-csa-cp1 | flashmla_fwd_ref@cute | 14203.1 | 1.02 | 1.02 | 470.4 | - | - |
| single-49208-csa-cp8r0 | tilelang@main | 1561.5 | 1.02 | 1.01 | 174.6 | 175.0 | 17.7 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 1561.5 | 1.16 | 1.16 | 434.8 | - | - |
| single-49208-csa-cp8r0 | tilelang@cute | 1561.5 | 1.02 | 1.01 | 172.9 | 174.3 | 17.6 |
| single-49208-csa-cp8r0 | cute@cute | 1561.5 | 1.02 | 1.01 | 236.3 | 190.0 | 19.2 |
| single-49208-csa-cp8r0 | cute_ws@cute | 1561.5 | 1.04 | 1.01 | 492.6 | 217.0 | 21.9 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 1561.5 | 1.16 | 1.16 | 432.5 | - | - |
| single-49208-csa-cp8r4 | tilelang@main | 1805.9 | 1.00 | 1.00 | 188.0 | 180.1 | 18.2 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1805.9 | 1.00 | 1.00 | 468.9 | - | - |
| single-49208-csa-cp8r4 | tilelang@cute | 1805.9 | 1.00 | 1.00 | 186.8 | 179.2 | 18.1 |
| single-49208-csa-cp8r4 | cute@cute | 1805.9 | 1.00 | 1.00 | 249.6 | 193.7 | 19.6 |
| single-49208-csa-cp8r4 | cute_ws@cute | 1805.9 | 1.00 | 1.00 | 523.3 | 221.3 | 22.4 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 1805.9 | 1.00 | 1.00 | 464.2 | - | - |
| single-49208-csa-cp8r7 | tilelang@main | 1805.9 | 1.00 | 1.00 | 188.0 | 174.3 | 17.6 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1805.9 | 1.00 | 1.00 | 467.1 | - | - |
| single-49208-csa-cp8r7 | tilelang@cute | 1805.9 | 1.00 | 1.00 | 186.2 | 173.8 | 17.6 |
| single-49208-csa-cp8r7 | cute@cute | 1805.9 | 1.00 | 1.00 | 249.5 | 187.4 | 18.9 |
| single-49208-csa-cp8r7 | cute_ws@cute | 1805.9 | 1.00 | 1.00 | 522.1 | 212.5 | 21.5 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 1805.9 | 1.00 | 1.00 | 462.5 | - | - |
| single-49208-hca-cp1 | tilelang@main | 7213.9 | 1.10 | 1.05 | 179.3 | 169.5 | 17.1 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 7213.9 | 1.60 | 1.60 | 381.5 | - | - |
| single-49208-hca-cp1 | tilelang@cute | 7213.9 | 1.10 | 1.05 | 179.2 | 169.5 | 17.1 |
| single-49208-hca-cp1 | cute@cute | 7213.9 | 1.10 | 1.05 | 200.0 | 174.6 | 17.6 |
| single-49208-hca-cp1 | cute_ws@cute | 7213.9 | 1.20 | 1.05 | 411.4 | 201.0 | 20.3 |
| single-49208-hca-cp1 | flashmla_fwd_ref@cute | 7213.9 | 1.60 | 1.60 | 363.5 | - | - |
| single-49208-hca-cp8r0 | tilelang@main | 423.9 | 1.26 | 1.12 | 70.8 | 99.8 | 10.1 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 423.9 | 3.41 | 3.41 | 180.6 | - | - |
| single-49208-hca-cp8r0 | tilelang@cute | 423.9 | 1.26 | 1.12 | 70.9 | 99.3 | 10.0 |
| single-49208-hca-cp8r0 | cute@cute | 423.9 | 1.26 | 1.12 | 113.0 | 118.3 | 12.0 |
| single-49208-hca-cp8r0 | cute_ws@cute | 423.9 | 1.69 | 1.12 | 228.2 | 139.6 | 14.1 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 423.9 | 3.41 | 3.41 | 179.2 | - | - |
| single-49208-hca-cp8r4 | tilelang@main | 970.0 | 1.11 | 1.05 | 129.2 | 147.8 | 14.9 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 970.0 | 1.49 | 1.49 | 345.9 | - | - |
| single-49208-hca-cp8r4 | tilelang@cute | 970.0 | 1.11 | 1.05 | 128.6 | 147.3 | 14.9 |
| single-49208-hca-cp8r4 | cute@cute | 970.0 | 1.11 | 1.05 | 185.7 | 165.1 | 16.7 |
| single-49208-hca-cp8r4 | cute_ws@cute | 970.0 | 1.12 | 1.05 | 412.1 | 193.4 | 19.5 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 970.0 | 1.49 | 1.49 | 342.3 | - | - |
| single-49208-hca-cp8r7 | tilelang@main | 1376.8 | 1.05 | 1.03 | 161.9 | 167.0 | 16.9 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 1376.8 | 1.05 | 1.05 | 416.5 | - | - |
| single-49208-hca-cp8r7 | tilelang@cute | 1376.8 | 1.05 | 1.03 | 160.4 | 167.4 | 16.9 |
| single-49208-hca-cp8r7 | cute@cute | 1376.8 | 1.05 | 1.03 | 220.5 | 183.2 | 18.5 |
| single-49208-hca-cp8r7 | cute_ws@cute | 1376.8 | 1.05 | 1.03 | 479.4 | 211.2 | 21.3 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 1376.8 | 1.05 | 1.05 | 412.5 | - | - |
| single-49208-sliding-cp1 | tilelang@main | 2885.8 | 1.00 | 1.00 | 110.9 | 126.1 | 12.7 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 2885.8 | 1.00 | 1.00 | 263.8 | - | - |
| single-49208-sliding-cp1 | tilelang@cute | 2885.8 | 1.00 | 1.00 | 110.3 | 125.7 | 12.7 |
| single-49208-sliding-cp1 | cute@cute | 2885.8 | 1.00 | 1.00 | 129.0 | 132.3 | 13.4 |
| single-49208-sliding-cp1 | cute_ws@cute | 2885.8 | 1.00 | 1.00 | 339.9 | 161.9 | 16.4 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@cute | 2885.8 | 1.00 | 1.00 | 258.9 | - | - |
| single-49208-sliding-cp8r0 | tilelang@main | 357.5 | 1.01 | 1.00 | 64.9 | 95.5 | 9.6 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 357.5 | 1.01 | 1.01 | 188.1 | - | - |
| single-49208-sliding-cp8r0 | tilelang@cute | 357.5 | 1.01 | 1.00 | 64.6 | 95.1 | 9.6 |
| single-49208-sliding-cp8r0 | cute@cute | 357.5 | 1.01 | 1.00 | 109.5 | 116.4 | 11.8 |
| single-49208-sliding-cp8r0 | cute_ws@cute | 357.5 | 1.01 | 1.00 | 277.2 | 134.2 | 13.6 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 357.5 | 1.01 | 1.01 | 187.8 | - | - |
| single-49208-sliding-cp8r4 | tilelang@main | 361.2 | 1.00 | 1.00 | 66.3 | 95.5 | 9.6 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.4 | - | - |
| single-49208-sliding-cp8r4 | tilelang@cute | 361.2 | 1.00 | 1.00 | 65.9 | 95.2 | 9.6 |
| single-49208-sliding-cp8r4 | cute@cute | 361.2 | 1.00 | 1.00 | 110.5 | 116.2 | 11.7 |
| single-49208-sliding-cp8r4 | cute_ws@cute | 361.2 | 1.00 | 1.00 | 279.1 | 133.4 | 13.5 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 361.2 | 1.00 | 1.00 | 188.7 | - | - |
| single-49208-sliding-cp8r7 | tilelang@main | 361.2 | 1.00 | 1.00 | 65.6 | 95.3 | 9.6 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.1 | - | - |
| single-49208-sliding-cp8r7 | tilelang@cute | 361.2 | 1.00 | 1.00 | 65.4 | 94.9 | 9.6 |
| single-49208-sliding-cp8r7 | cute@cute | 361.2 | 1.00 | 1.00 | 110.1 | 115.7 | 11.7 |
| single-49208-sliding-cp8r7 | cute_ws@cute | 361.2 | 1.00 | 1.00 | 277.8 | 133.3 | 13.5 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 361.2 | 1.00 | 1.00 | 188.5 | - | - |
| short-49208-csa-cp1 | tilelang@main | 9266.5 | 1.07 | 1.04 | 203.8 | 181.4 | 18.3 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 9266.5 | 1.56 | 1.56 | 425.2 | - | - |
| short-49208-csa-cp1 | tilelang@cute | 9266.5 | 1.07 | 1.04 | 203.9 | 181.3 | 18.3 |
| short-49208-csa-cp1 | cute@cute | 9266.5 | 1.07 | 1.04 | 224.2 | 185.9 | 18.8 |
| short-49208-csa-cp1 | cute_ws@cute | 9266.5 | 1.13 | 1.04 | 440.3 | 211.8 | 21.4 |
| short-49208-csa-cp1 | flashmla_fwd_ref@cute | 9266.5 | 1.56 | 1.56 | 400.3 | - | - |
| short-49208-csa-cp8r0 | tilelang@main | 832.8 | 1.14 | 1.08 | 116.8 | 138.5 | 14.0 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 832.8 | 2.17 | 2.17 | 300.8 | - | - |
| short-49208-csa-cp8r0 | tilelang@cute | 832.8 | 1.14 | 1.08 | 117.2 | 138.0 | 13.9 |
| short-49208-csa-cp8r0 | cute@cute | 832.8 | 1.14 | 1.08 | 171.7 | 155.8 | 15.7 |
| short-49208-csa-cp8r0 | cute_ws@cute | 832.8 | 1.26 | 1.08 | 360.7 | 181.9 | 18.4 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 832.8 | 2.17 | 2.17 | 298.8 | - | - |
| short-49208-csa-cp8r4 | tilelang@main | 1351.2 | 1.04 | 1.02 | 161.8 | 167.1 | 16.9 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1351.2 | 1.34 | 1.34 | 402.3 | - | - |
| short-49208-csa-cp8r4 | tilelang@cute | 1351.2 | 1.04 | 1.02 | 160.1 | 166.8 | 16.9 |
| short-49208-csa-cp8r4 | cute@cute | 1351.2 | 1.04 | 1.02 | 220.9 | 182.5 | 18.4 |
| short-49208-csa-cp8r4 | cute_ws@cute | 1351.2 | 1.08 | 1.02 | 456.7 | 209.0 | 21.1 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 1351.2 | 1.34 | 1.34 | 399.1 | - | - |
| short-49208-csa-cp8r7 | tilelang@main | 1017.3 | 1.09 | 1.05 | 134.5 | 149.4 | 15.1 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1017.3 | 1.78 | 1.78 | 338.6 | - | - |
| short-49208-csa-cp8r7 | tilelang@cute | 1017.3 | 1.09 | 1.05 | 134.4 | 149.8 | 15.1 |
| short-49208-csa-cp8r7 | cute@cute | 1017.3 | 1.09 | 1.05 | 192.3 | 166.8 | 16.9 |
| short-49208-csa-cp8r7 | cute_ws@cute | 1017.3 | 1.18 | 1.05 | 396.2 | 192.5 | 19.5 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 1017.3 | 1.78 | 1.78 | 336.7 | - | - |
| short-49208-hca-cp1 | tilelang@main | 3131.2 | 1.36 | 1.16 | 105.2 | 119.6 | 12.1 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 3131.2 | 1.85 | 1.85 | 218.0 | - | - |
| short-49208-hca-cp1 | tilelang@cute | 3131.2 | 1.36 | 1.16 | 104.5 | 119.6 | 12.1 |
| short-49208-hca-cp1 | cute@cute | 3131.2 | 1.36 | 1.16 | 120.7 | 125.1 | 12.6 |
| short-49208-hca-cp1 | cute_ws@cute | 3131.2 | 1.78 | 1.16 | 250.6 | 147.9 | 15.0 |
| short-49208-hca-cp1 | flashmla_fwd_ref@cute | 3131.2 | 1.85 | 1.85 | 217.0 | - | - |
| short-49208-hca-cp8r0 | tilelang@main | 352.9 | 1.44 | 1.20 | 59.3 | 86.7 | 8.8 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 352.9 | 2.05 | 2.05 | 155.9 | - | - |
| short-49208-hca-cp8r0 | tilelang@cute | 352.9 | 1.44 | 1.20 | 58.4 | 86.5 | 8.7 |
| short-49208-hca-cp8r0 | cute@cute | 352.9 | 1.44 | 1.20 | 94.4 | 104.0 | 10.5 |
| short-49208-hca-cp8r0 | cute_ws@cute | 352.9 | 1.92 | 1.20 | 198.1 | 122.9 | 12.4 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 352.9 | 2.05 | 2.05 | 152.7 | - | - |
| short-49208-hca-cp8r4 | tilelang@main | 398.0 | 1.33 | 1.14 | 65.1 | 95.1 | 9.6 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 398.0 | 1.82 | 1.82 | 170.9 | - | - |
| short-49208-hca-cp8r4 | tilelang@cute | 398.0 | 1.33 | 1.14 | 65.7 | 94.0 | 9.5 |
| short-49208-hca-cp8r4 | cute@cute | 398.0 | 1.33 | 1.14 | 104.8 | 112.4 | 11.4 |
| short-49208-hca-cp8r4 | cute_ws@cute | 398.0 | 1.78 | 1.14 | 217.6 | 132.5 | 13.4 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 398.0 | 1.82 | 1.82 | 170.1 | - | - |
| short-49208-hca-cp8r7 | tilelang@main | 369.7 | 1.42 | 1.18 | 61.0 | 88.7 | 9.0 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 369.7 | 1.95 | 1.95 | 159.0 | - | - |
| short-49208-hca-cp8r7 | tilelang@cute | 369.7 | 1.42 | 1.18 | 60.7 | 88.8 | 9.0 |
| short-49208-hca-cp8r7 | cute@cute | 369.7 | 1.42 | 1.18 | 96.9 | 106.2 | 10.7 |
| short-49208-hca-cp8r7 | cute_ws@cute | 369.7 | 1.89 | 1.18 | 203.4 | 125.8 | 12.7 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 369.7 | 1.95 | 1.95 | 157.7 | - | - |
| short-49208-sliding-cp1 | tilelang@main | 2785.1 | 1.02 | 1.01 | 107.2 | 123.1 | 12.4 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 2785.1 | 1.04 | 1.04 | 254.8 | - | - |
| short-49208-sliding-cp1 | tilelang@cute | 2785.1 | 1.02 | 1.01 | 106.6 | 122.9 | 12.4 |
| short-49208-sliding-cp1 | cute@cute | 2785.1 | 1.02 | 1.01 | 125.2 | 129.5 | 13.1 |
| short-49208-sliding-cp1 | cute_ws@cute | 2785.1 | 1.04 | 1.01 | 327.3 | 158.7 | 16.0 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@cute | 2785.1 | 1.04 | 1.04 | 251.5 | - | - |
| short-49208-sliding-cp8r0 | tilelang@main | 338.8 | 1.03 | 1.02 | 61.9 | 92.5 | 9.3 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 338.8 | 1.07 | 1.07 | 178.9 | - | - |
| short-49208-sliding-cp8r0 | tilelang@cute | 338.8 | 1.03 | 1.02 | 62.4 | 91.9 | 9.3 |
| short-49208-sliding-cp8r0 | cute@cute | 338.8 | 1.03 | 1.02 | 105.6 | 112.8 | 11.4 |
| short-49208-sliding-cp8r0 | cute_ws@cute | 338.8 | 1.07 | 1.02 | 262.2 | 129.6 | 13.1 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 338.8 | 1.07 | 1.07 | 176.9 | - | - |
| short-49208-sliding-cp8r4 | tilelang@main | 353.7 | 1.01 | 1.01 | 64.4 | 94.6 | 9.6 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 353.7 | 1.02 | 1.02 | 185.2 | - | - |
| short-49208-sliding-cp8r4 | tilelang@cute | 353.7 | 1.01 | 1.01 | 64.5 | 94.2 | 9.5 |
| short-49208-sliding-cp8r4 | cute@cute | 353.7 | 1.01 | 1.01 | 108.6 | 115.4 | 11.7 |
| short-49208-sliding-cp8r4 | cute_ws@cute | 353.7 | 1.02 | 1.01 | 272.2 | 132.9 | 13.4 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 353.7 | 1.02 | 1.02 | 183.6 | - | - |
| short-49208-sliding-cp8r7 | tilelang@main | 350.0 | 1.02 | 1.01 | 64.3 | 93.5 | 9.4 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 350.0 | 1.03 | 1.03 | 186.3 | - | - |
| short-49208-sliding-cp8r7 | tilelang@cute | 350.0 | 1.02 | 1.01 | 63.6 | 93.8 | 9.5 |
| short-49208-sliding-cp8r7 | cute@cute | 350.0 | 1.02 | 1.01 | 107.1 | 114.8 | 11.6 |
| short-49208-sliding-cp8r7 | cute_ws@cute | 350.0 | 1.03 | 1.01 | 271.0 | 132.4 | 13.4 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 350.0 | 1.03 | 1.03 | 182.3 | - | - |
| heavy-49208-csa-cp1 | tilelang@main | 11959.6 | 1.03 | 1.02 | 229.1 | 193.8 | 19.6 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 11959.6 | 1.21 | 1.21 | 462.2 | - | - |
| heavy-49208-csa-cp1 | tilelang@cute | 11959.6 | 1.03 | 1.02 | 229.9 | 193.7 | 19.6 |
| heavy-49208-csa-cp1 | cute@cute | 11959.6 | 1.03 | 1.02 | 250.6 | 197.9 | 20.0 |
| heavy-49208-csa-cp1 | cute_ws@cute | 11959.6 | 1.05 | 1.02 | 491.6 | 223.7 | 22.6 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@cute | 11959.6 | 1.21 | 1.21 | 445.9 | - | - |
| heavy-49208-csa-cp8r0 | tilelang@main | 663.7 | 1.22 | 1.14 | 98.5 | 122.4 | 12.4 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 663.7 | 2.72 | 2.72 | 249.6 | - | - |
| heavy-49208-csa-cp8r0 | tilelang@cute | 663.7 | 1.22 | 1.14 | 98.5 | 121.5 | 12.3 |
| heavy-49208-csa-cp8r0 | cute@cute | 663.7 | 1.22 | 1.14 | 148.1 | 139.5 | 14.1 |
| heavy-49208-csa-cp8r0 | cute_ws@cute | 663.7 | 1.41 | 1.14 | 306.6 | 163.9 | 16.6 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 663.7 | 2.72 | 2.72 | 248.6 | - | - |
| heavy-49208-csa-cp8r4 | tilelang@main | 1663.9 | 1.01 | 1.01 | 180.8 | 177.6 | 17.9 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1663.9 | 1.09 | 1.09 | 449.3 | - | - |
| heavy-49208-csa-cp8r4 | tilelang@cute | 1663.9 | 1.01 | 1.01 | 178.3 | 176.4 | 17.8 |
| heavy-49208-csa-cp8r4 | cute@cute | 1663.9 | 1.01 | 1.01 | 242.0 | 192.4 | 19.4 |
| heavy-49208-csa-cp8r4 | cute_ws@cute | 1663.9 | 1.03 | 1.01 | 505.8 | 219.5 | 22.2 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 1663.9 | 1.09 | 1.09 | 447.6 | - | - |
| heavy-49208-csa-cp8r7 | tilelang@main | 1454.9 | 1.03 | 1.02 | 167.8 | 169.5 | 17.1 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1454.9 | 1.24 | 1.24 | 419.7 | - | - |
| heavy-49208-csa-cp8r7 | tilelang@cute | 1454.9 | 1.03 | 1.02 | 166.0 | 169.5 | 17.1 |
| heavy-49208-csa-cp8r7 | cute@cute | 1454.9 | 1.03 | 1.02 | 226.0 | 184.7 | 18.7 |
| heavy-49208-csa-cp8r7 | cute_ws@cute | 1454.9 | 1.06 | 1.02 | 475.1 | 211.3 | 21.4 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 1454.9 | 1.24 | 1.24 | 416.7 | - | - |
| heavy-49208-hca-cp1 | tilelang@main | 4077.2 | 1.21 | 1.09 | 128.2 | 137.7 | 13.9 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4077.2 | 2.13 | 2.13 | 272.6 | - | - |
| heavy-49208-hca-cp1 | tilelang@cute | 4077.2 | 1.21 | 1.09 | 128.0 | 137.4 | 13.9 |
| heavy-49208-hca-cp1 | cute@cute | 4077.2 | 1.21 | 1.09 | 145.9 | 143.0 | 14.4 |
| heavy-49208-hca-cp1 | cute_ws@cute | 4077.2 | 1.43 | 1.09 | 316.5 | 168.3 | 17.0 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@cute | 4077.2 | 2.13 | 2.13 | 268.8 | - | - |
| heavy-49208-hca-cp8r0 | tilelang@main | 318.8 | 1.45 | 1.21 | 53.5 | 81.5 | 8.2 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 318.8 | 3.40 | 3.40 | 141.7 | - | - |
| heavy-49208-hca-cp8r0 | tilelang@cute | 318.8 | 1.45 | 1.21 | 53.9 | 81.3 | 8.2 |
| heavy-49208-hca-cp8r0 | cute@cute | 318.8 | 1.45 | 1.21 | 88.2 | 98.4 | 9.9 |
| heavy-49208-hca-cp8r0 | cute_ws@cute | 318.8 | 1.94 | 1.21 | 187.3 | 115.5 | 11.7 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 318.8 | 3.40 | 3.40 | 141.8 | - | - |
| heavy-49208-hca-cp8r4 | tilelang@main | 438.1 | 1.24 | 1.11 | 71.4 | 99.7 | 10.1 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 438.1 | 2.47 | 2.47 | 185.4 | - | - |
| heavy-49208-hca-cp8r4 | tilelang@cute | 438.1 | 1.24 | 1.11 | 70.8 | 99.0 | 10.0 |
| heavy-49208-hca-cp8r4 | cute@cute | 438.1 | 1.24 | 1.11 | 113.0 | 117.5 | 11.9 |
| heavy-49208-hca-cp8r4 | cute_ws@cute | 438.1 | 1.65 | 1.11 | 237.0 | 137.6 | 13.9 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 438.1 | 2.47 | 2.47 | 184.8 | - | - |
| heavy-49208-hca-cp8r7 | tilelang@main | 507.2 | 1.25 | 1.08 | 78.6 | 108.5 | 11.0 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 507.2 | 2.14 | 2.14 | 203.5 | - | - |
| heavy-49208-hca-cp8r7 | tilelang@cute | 507.2 | 1.25 | 1.08 | 79.2 | 108.7 | 11.0 |
| heavy-49208-hca-cp8r7 | cute@cute | 507.2 | 1.25 | 1.08 | 123.8 | 127.7 | 12.9 |
| heavy-49208-hca-cp8r7 | cute_ws@cute | 507.2 | 1.60 | 1.08 | 253.1 | 150.7 | 15.2 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 507.2 | 2.14 | 2.14 | 201.9 | - | - |
| heavy-49208-sliding-cp1 | tilelang@main | 2781.4 | 1.02 | 1.01 | 106.7 | 123.2 | 12.4 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 2781.4 | 1.04 | 1.04 | 253.5 | - | - |
| heavy-49208-sliding-cp1 | tilelang@cute | 2781.4 | 1.02 | 1.01 | 106.5 | 122.9 | 12.4 |
| heavy-49208-sliding-cp1 | cute@cute | 2781.4 | 1.02 | 1.01 | 125.0 | 129.4 | 13.1 |
| heavy-49208-sliding-cp1 | cute_ws@cute | 2781.4 | 1.04 | 1.01 | 327.2 | 158.6 | 16.0 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@cute | 2781.4 | 1.04 | 1.04 | 252.1 | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@main | 309.0 | 1.08 | 1.04 | 56.2 | 86.1 | 8.7 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 309.0 | 1.17 | 1.17 | 162.8 | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@cute | 309.0 | 1.08 | 1.04 | 56.2 | 86.0 | 8.7 |
| heavy-49208-sliding-cp8r0 | cute@cute | 309.0 | 1.08 | 1.04 | 95.7 | 106.1 | 10.7 |
| heavy-49208-sliding-cp8r0 | cute_ws@cute | 309.0 | 1.17 | 1.04 | 232.0 | 122.0 | 12.3 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 309.0 | 1.17 | 1.17 | 159.3 | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@main | 361.2 | 1.00 | 1.00 | 65.9 | 96.0 | 9.7 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.5 | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@cute | 361.2 | 1.00 | 1.00 | 65.5 | 94.1 | 9.5 |
| heavy-49208-sliding-cp8r4 | cute@cute | 361.2 | 1.00 | 1.00 | 109.5 | 115.1 | 11.6 |
| heavy-49208-sliding-cp8r4 | cute_ws@cute | 361.2 | 1.00 | 1.00 | 279.0 | 131.5 | 13.3 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 361.2 | 1.00 | 1.00 | 187.3 | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@main | 353.7 | 1.01 | 1.01 | 64.5 | 93.9 | 9.5 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 353.7 | 1.02 | 1.02 | 186.1 | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@cute | 353.7 | 1.01 | 1.01 | 64.0 | 93.7 | 9.5 |
| heavy-49208-sliding-cp8r7 | cute@cute | 353.7 | 1.01 | 1.01 | 107.3 | 114.3 | 11.6 |
| heavy-49208-sliding-cp8r7 | cute_ws@cute | 353.7 | 1.02 | 1.01 | 272.2 | 131.0 | 13.2 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 353.7 | 1.02 | 1.02 | 183.5 | - | - |
| tiny-49208-csa-cp1 | tilelang@main | 1202.7 | 3.49 | 2.89 | 40.9 | 49.9 | 5.0 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 1202.7 | 12.01 | 12.01 | 80.0 | - | - |
| tiny-49208-csa-cp1 | tilelang@cute | 1202.7 | 3.49 | 2.89 | 40.9 | 49.8 | 5.0 |
| tiny-49208-csa-cp1 | cute@cute | 1202.7 | 3.49 | 2.89 | 46.7 | 52.2 | 5.3 |
| tiny-49208-csa-cp1 | cute_ws@cute | 1202.7 | 4.69 | 2.89 | 92.0 | 62.0 | 6.3 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@cute | 1202.7 | 12.01 | 12.01 | 79.9 | - | - |
| tiny-49208-csa-cp8r0 | tilelang@main | 150.2 | 3.50 | 2.89 | 25.2 | 38.8 | 3.9 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 150.2 | 12.03 | 12.03 | 65.2 | - | - |
| tiny-49208-csa-cp8r0 | tilelang@cute | 150.2 | 3.50 | 2.89 | 25.4 | 38.4 | 3.9 |
| tiny-49208-csa-cp8r0 | cute@cute | 150.2 | 3.50 | 2.89 | 40.6 | 46.3 | 4.7 |
| tiny-49208-csa-cp8r0 | cute_ws@cute | 150.2 | 4.70 | 2.89 | 80.4 | 55.7 | 5.6 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@cute | 150.2 | 12.03 | 12.03 | 63.4 | - | - |
| tiny-49208-csa-cp8r4 | tilelang@main | 156.7 | 3.35 | 2.78 | 26.6 | 40.1 | 4.1 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 156.7 | 11.53 | 11.53 | 68.2 | - | - |
| tiny-49208-csa-cp8r4 | tilelang@cute | 156.7 | 3.35 | 2.78 | 26.3 | 39.7 | 4.0 |
| tiny-49208-csa-cp8r4 | cute@cute | 156.7 | 3.35 | 2.78 | 42.2 | 48.1 | 4.9 |
| tiny-49208-csa-cp8r4 | cute_ws@cute | 156.7 | 4.50 | 2.78 | 83.4 | 57.8 | 5.8 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@cute | 156.7 | 11.53 | 11.53 | 67.0 | - | - |
| tiny-49208-csa-cp8r7 | tilelang@main | 144.3 | 3.63 | 3.01 | 24.4 | 36.9 | 3.7 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 144.3 | 12.51 | 12.51 | 62.5 | - | - |
| tiny-49208-csa-cp8r7 | tilelang@cute | 144.3 | 3.63 | 3.01 | 24.4 | 36.8 | 3.7 |
| tiny-49208-csa-cp8r7 | cute@cute | 144.3 | 3.63 | 3.01 | 39.2 | 44.5 | 4.5 |
| tiny-49208-csa-cp8r7 | cute_ws@cute | 144.3 | 4.89 | 3.01 | 77.2 | 53.5 | 5.4 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@cute | 144.3 | 12.51 | 12.51 | 62.2 | - | - |
| tiny-49208-hca-cp1 | tilelang@main | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 969.0 | 2.98 | 2.98 | 87.5 | - | - |
| tiny-49208-hca-cp1 | tilelang@cute | 969.0 | 1.86 | 1.39 | 41.2 | 57.8 | 5.8 |
| tiny-49208-hca-cp1 | cute@cute | 969.0 | 1.86 | 1.39 | 49.1 | 61.9 | 6.3 |
| tiny-49208-hca-cp1 | cute_ws@cute | 969.0 | 2.98 | 1.39 | 100.4 | 76.2 | 7.7 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@cute | 969.0 | 2.98 | 2.98 | 86.9 | - | - |
| tiny-49208-hca-cp8r0 | tilelang@main | 121.0 | 1.86 | 1.39 | 23.4 | 41.0 | 4.1 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 121.0 | 2.99 | 2.99 | 63.8 | - | - |
| tiny-49208-hca-cp8r0 | tilelang@cute | 121.0 | 1.86 | 1.39 | 23.5 | 40.8 | 4.1 |
| tiny-49208-hca-cp8r0 | cute@cute | 121.0 | 1.86 | 1.39 | 41.2 | 52.9 | 5.3 |
| tiny-49208-hca-cp8r0 | cute_ws@cute | 121.0 | 2.99 | 1.39 | 85.7 | 61.1 | 6.2 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@cute | 121.0 | 2.99 | 2.99 | 62.9 | - | - |
| tiny-49208-hca-cp8r4 | tilelang@main | 126.2 | 1.83 | 1.38 | 24.3 | 42.3 | 4.3 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 126.2 | 2.86 | 2.86 | 66.3 | - | - |
| tiny-49208-hca-cp8r4 | tilelang@cute | 126.2 | 1.83 | 1.38 | 24.4 | 41.8 | 4.2 |
| tiny-49208-hca-cp8r4 | cute@cute | 126.2 | 1.83 | 1.38 | 42.4 | 54.1 | 5.5 |
| tiny-49208-hca-cp8r4 | cute_ws@cute | 126.2 | 2.86 | 1.38 | 89.5 | 62.4 | 6.3 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@cute | 126.2 | 2.86 | 2.86 | 65.3 | - | - |
| tiny-49208-hca-cp8r7 | tilelang@main | 116.3 | 1.90 | 1.41 | 22.6 | 39.4 | 4.0 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 116.3 | 3.11 | 3.11 | 61.5 | - | - |
| tiny-49208-hca-cp8r7 | tilelang@cute | 116.3 | 1.90 | 1.41 | 22.3 | 39.3 | 4.0 |
| tiny-49208-hca-cp8r7 | cute@cute | 116.3 | 1.90 | 1.41 | 39.4 | 50.9 | 5.1 |
| tiny-49208-hca-cp8r7 | cute_ws@cute | 116.3 | 3.11 | 1.41 | 81.9 | 58.4 | 5.9 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@cute | 116.3 | 3.11 | 3.11 | 58.4 | - | - |
| tiny-49208-sliding-cp1 | tilelang@main | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 969.0 | 2.98 | 2.98 | 87.2 | - | - |
| tiny-49208-sliding-cp1 | tilelang@cute | 969.0 | 1.86 | 1.39 | 41.2 | 57.8 | 5.8 |
| tiny-49208-sliding-cp1 | cute@cute | 969.0 | 1.86 | 1.39 | 49.0 | 61.9 | 6.3 |
| tiny-49208-sliding-cp1 | cute_ws@cute | 969.0 | 2.98 | 1.39 | 100.8 | 76.3 | 7.7 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@cute | 969.0 | 2.98 | 2.98 | 86.9 | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@main | 121.0 | 1.86 | 1.39 | 23.6 | 41.1 | 4.1 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 121.0 | 2.99 | 2.99 | 64.0 | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@cute | 121.0 | 1.86 | 1.39 | 23.4 | 40.6 | 4.1 |
| tiny-49208-sliding-cp8r0 | cute@cute | 121.0 | 1.86 | 1.39 | 41.0 | 52.6 | 5.3 |
| tiny-49208-sliding-cp8r0 | cute_ws@cute | 121.0 | 2.99 | 1.39 | 85.0 | 60.7 | 6.1 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@cute | 121.0 | 2.99 | 2.99 | 62.4 | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@main | 126.2 | 1.83 | 1.38 | 24.3 | 42.4 | 4.3 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 126.2 | 2.86 | 2.86 | 66.5 | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@cute | 126.2 | 1.83 | 1.38 | 24.3 | 41.8 | 4.2 |
| tiny-49208-sliding-cp8r4 | cute@cute | 126.2 | 1.83 | 1.38 | 42.4 | 54.1 | 5.5 |
| tiny-49208-sliding-cp8r4 | cute_ws@cute | 126.2 | 2.86 | 1.38 | 89.2 | 61.7 | 6.2 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@cute | 126.2 | 2.86 | 2.86 | 65.7 | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@main | 116.3 | 1.90 | 1.41 | 22.7 | 39.4 | 4.0 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 116.3 | 3.11 | 3.11 | 61.3 | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@cute | 116.3 | 1.90 | 1.41 | 22.4 | 39.3 | 4.0 |
| tiny-49208-sliding-cp8r7 | cute@cute | 116.3 | 1.90 | 1.41 | 39.4 | 51.1 | 5.2 |
| tiny-49208-sliding-cp8r7 | cute_ws@cute | 116.3 | 3.11 | 1.41 | 81.8 | 58.9 | 6.0 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@cute | 116.3 | 3.11 | 3.11 | 60.6 | - | - |
| single-65536-csa-cp1 | tilelang@main | 18997.0 | 1.00 | 1.00 | 250.2 | 199.0 | 20.1 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 18997.0 | 1.01 | 1.01 | 476.7 | - | - |
| single-65536-csa-cp1 | tilelang@cute | 18997.0 | 1.00 | 1.00 | 251.2 | 199.0 | 20.1 |
| single-65536-csa-cp1 | cute@cute | 18997.0 | 1.00 | 1.00 | 269.0 | 202.3 | 20.4 |
| single-65536-csa-cp1 | cute_ws@cute | 18997.0 | 1.00 | 1.00 | 495.7 | 226.3 | 22.9 |
| single-65536-csa-cp1 | flashmla_fwd_ref@cute | 18997.0 | 1.01 | 1.01 | 473.1 | - | - |
| single-65536-csa-cp8r0 | tilelang@main | 2160.7 | 1.02 | 1.01 | 189.1 | 181.1 | 18.3 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 2160.7 | 1.11 | 1.11 | 465.9 | - | - |
| single-65536-csa-cp8r0 | tilelang@cute | 2160.7 | 1.02 | 1.01 | 191.2 | 180.5 | 18.2 |
| single-65536-csa-cp8r0 | cute@cute | 2160.7 | 1.02 | 1.01 | 243.5 | 192.5 | 19.5 |
| single-65536-csa-cp8r0 | cute_ws@cute | 2160.7 | 1.03 | 1.01 | 507.2 | 219.1 | 22.1 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 2160.7 | 1.11 | 1.11 | 461.6 | - | - |
| single-65536-csa-cp8r4 | tilelang@main | 2405.2 | 1.00 | 1.00 | 200.4 | 183.0 | 18.5 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 487.2 | - | - |
| single-65536-csa-cp8r4 | tilelang@cute | 2405.2 | 1.00 | 1.00 | 201.4 | 182.0 | 18.4 |
| single-65536-csa-cp8r4 | cute@cute | 2405.2 | 1.00 | 1.00 | 254.2 | 193.1 | 19.5 |
| single-65536-csa-cp8r4 | cute_ws@cute | 2405.2 | 1.00 | 1.00 | 531.4 | 218.6 | 22.1 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 2405.2 | 1.00 | 1.00 | 483.2 | - | - |
| single-65536-csa-cp8r7 | tilelang@main | 2405.2 | 1.00 | 1.00 | 203.1 | 171.9 | 17.4 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 487.3 | - | - |
| single-65536-csa-cp8r7 | tilelang@cute | 2405.2 | 1.00 | 1.00 | 200.3 | 171.6 | 17.3 |
| single-65536-csa-cp8r7 | cute@cute | 2405.2 | 1.00 | 1.00 | 253.4 | 181.3 | 18.3 |
| single-65536-csa-cp8r7 | cute_ws@cute | 2405.2 | 1.00 | 1.00 | 525.0 | 204.0 | 20.6 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 2405.2 | 1.00 | 1.00 | 476.4 | - | - |
| single-65536-hca-cp1 | tilelang@main | 11526.3 | 1.08 | 1.04 | 199.6 | 179.2 | 18.1 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 11526.3 | 1.67 | 1.67 | 401.8 | - | - |
| single-65536-hca-cp1 | tilelang@cute | 11526.3 | 1.08 | 1.04 | 199.7 | 179.2 | 18.1 |
| single-65536-hca-cp1 | cute@cute | 11526.3 | 1.08 | 1.04 | 216.5 | 183.1 | 18.5 |
| single-65536-hca-cp1 | cute_ws@cute | 11526.3 | 1.17 | 1.04 | 430.2 | 209.5 | 21.2 |
| single-65536-hca-cp1 | flashmla_fwd_ref@cute | 11526.3 | 1.67 | 1.67 | 390.7 | - | - |
| single-65536-hca-cp8r0 | tilelang@main | 595.7 | 1.20 | 1.10 | 82.1 | 109.6 | 11.1 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 595.7 | 4.04 | 4.04 | 203.8 | - | - |
| single-65536-hca-cp8r0 | tilelang@cute | 595.7 | 1.20 | 1.10 | 83.0 | 109.4 | 11.1 |
| single-65536-hca-cp8r0 | cute@cute | 595.7 | 1.20 | 1.10 | 122.9 | 125.6 | 12.7 |
| single-65536-hca-cp8r0 | cute_ws@cute | 595.7 | 1.60 | 1.10 | 251.9 | 149.2 | 15.1 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 595.7 | 4.04 | 4.04 | 201.7 | - | - |
| single-65536-hca-cp8r4 | tilelang@main | 1561.5 | 1.08 | 1.04 | 159.5 | 163.0 | 16.5 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1561.5 | 1.54 | 1.54 | 376.5 | - | - |
| single-65536-hca-cp8r4 | tilelang@cute | 1561.5 | 1.08 | 1.04 | 158.6 | 162.3 | 16.4 |
| single-65536-hca-cp8r4 | cute@cute | 1561.5 | 1.08 | 1.04 | 208.2 | 176.3 | 17.8 |
| single-65536-hca-cp8r4 | cute_ws@cute | 1561.5 | 1.23 | 1.04 | 417.9 | 202.0 | 20.4 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 1561.5 | 1.54 | 1.54 | 371.4 | - | - |
| single-65536-hca-cp8r7 | tilelang@main | 2283.1 | 1.05 | 1.03 | 192.4 | 179.9 | 18.2 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 2283.1 | 1.05 | 1.05 | 471.2 | - | - |
| single-65536-hca-cp8r7 | tilelang@cute | 2283.1 | 1.05 | 1.03 | 189.8 | 179.8 | 18.2 |
| single-65536-hca-cp8r7 | cute@cute | 2283.1 | 1.05 | 1.03 | 240.8 | 191.0 | 19.3 |
| single-65536-hca-cp8r7 | cute_ws@cute | 2283.1 | 1.05 | 1.03 | 513.4 | 217.8 | 22.0 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 2283.1 | 1.05 | 1.05 | 463.1 | - | - |
| single-65536-sliding-cp1 | tilelang@main | 3844.6 | 1.00 | 1.00 | 113.8 | 127.5 | 12.9 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 3844.6 | 1.00 | 1.00 | 266.5 | - | - |
| single-65536-sliding-cp1 | tilelang@cute | 3844.6 | 1.00 | 1.00 | 113.4 | 127.4 | 12.9 |
| single-65536-sliding-cp1 | cute@cute | 3844.6 | 1.00 | 1.00 | 130.2 | 133.1 | 13.5 |
| single-65536-sliding-cp1 | cute_ws@cute | 3844.6 | 1.00 | 1.00 | 341.4 | 162.6 | 16.4 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@cute | 3844.6 | 1.00 | 1.00 | 264.2 | - | - |
| single-65536-sliding-cp8r0 | tilelang@main | 477.3 | 1.00 | 1.00 | 72.9 | 102.5 | 10.4 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 477.3 | 1.01 | 1.01 | 206.2 | - | - |
| single-65536-sliding-cp8r0 | tilelang@cute | 477.3 | 1.00 | 1.00 | 73.3 | 101.7 | 10.3 |
| single-65536-sliding-cp8r0 | cute@cute | 477.3 | 1.00 | 1.00 | 114.5 | 120.1 | 12.1 |
| single-65536-sliding-cp8r0 | cute_ws@cute | 477.3 | 1.01 | 1.00 | 294.9 | 148.4 | 15.0 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 477.3 | 1.01 | 1.01 | 203.1 | - | - |
| single-65536-sliding-cp8r4 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.6 | 102.5 | 10.4 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 208.0 | - | - |
| single-65536-sliding-cp8r4 | tilelang@cute | 481.0 | 1.00 | 1.00 | 73.7 | 101.7 | 10.3 |
| single-65536-sliding-cp8r4 | cute@cute | 481.0 | 1.00 | 1.00 | 115.4 | 119.5 | 12.1 |
| single-65536-sliding-cp8r4 | cute_ws@cute | 481.0 | 1.00 | 1.00 | 296.7 | 147.5 | 14.9 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 481.0 | 1.00 | 1.00 | 205.4 | - | - |
| single-65536-sliding-cp8r7 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.2 | 102.0 | 10.3 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 207.5 | - | - |
| single-65536-sliding-cp8r7 | tilelang@cute | 481.0 | 1.00 | 1.00 | 74.0 | 101.9 | 10.3 |
| single-65536-sliding-cp8r7 | cute@cute | 481.0 | 1.00 | 1.00 | 115.2 | 119.8 | 12.1 |
| single-65536-sliding-cp8r7 | cute_ws@cute | 481.0 | 1.00 | 1.00 | 295.3 | 146.8 | 14.8 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 481.0 | 1.00 | 1.00 | 202.3 | - | - |
| short-65536-csa-cp1 | tilelang@main | 11209.9 | 1.08 | 1.05 | 197.2 | 177.5 | 17.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 11209.9 | 1.72 | 1.72 | 393.2 | - | - |
| short-65536-csa-cp1 | tilelang@cute | 11209.9 | 1.08 | 1.05 | 197.3 | 177.4 | 17.9 |
| short-65536-csa-cp1 | cute@cute | 11209.9 | 1.08 | 1.05 | 214.7 | 181.5 | 18.3 |
| short-65536-csa-cp1 | cute_ws@cute | 11209.9 | 1.16 | 1.05 | 422.2 | 207.3 | 20.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref@cute | 11209.9 | 1.72 | 1.72 | 384.3 | - | - |
| short-65536-csa-cp8r0 | tilelang@main | 885.0 | 1.18 | 1.11 | 110.6 | 130.5 | 13.2 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 885.0 | 2.72 | 2.72 | 278.5 | - | - |
| short-65536-csa-cp8r0 | tilelang@cute | 885.0 | 1.18 | 1.11 | 109.8 | 130.3 | 13.2 |
| short-65536-csa-cp8r0 | cute@cute | 885.0 | 1.18 | 1.11 | 155.7 | 145.3 | 14.7 |
| short-65536-csa-cp8r0 | cute_ws@cute | 885.0 | 1.34 | 1.11 | 327.5 | 170.4 | 17.2 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 885.0 | 2.72 | 2.72 | 275.8 | - | - |
| short-65536-csa-cp8r4 | tilelang@main | 1689.8 | 1.05 | 1.03 | 168.7 | 167.7 | 16.9 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1689.8 | 1.42 | 1.42 | 412.8 | - | - |
| short-65536-csa-cp8r4 | tilelang@cute | 1689.8 | 1.05 | 1.03 | 167.0 | 166.9 | 16.9 |
| short-65536-csa-cp8r4 | cute@cute | 1689.8 | 1.05 | 1.03 | 218.2 | 180.6 | 18.3 |
| short-65536-csa-cp8r4 | cute_ws@cute | 1689.8 | 1.10 | 1.03 | 460.3 | 208.3 | 21.1 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 1689.8 | 1.42 | 1.42 | 409.3 | - | - |
| short-65536-csa-cp8r7 | tilelang@main | 1414.7 | 1.08 | 1.05 | 151.7 | 158.2 | 16.0 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1414.7 | 1.70 | 1.70 | 371.4 | - | - |
| short-65536-csa-cp8r7 | tilelang@cute | 1414.7 | 1.08 | 1.05 | 149.8 | 158.2 | 16.0 |
| short-65536-csa-cp8r7 | cute@cute | 1414.7 | 1.08 | 1.05 | 199.8 | 172.4 | 17.4 |
| short-65536-csa-cp8r7 | cute_ws@cute | 1414.7 | 1.16 | 1.05 | 421.3 | 198.8 | 20.1 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 1414.7 | 1.70 | 1.70 | 365.7 | - | - |
| short-65536-hca-cp1 | tilelang@main | 3930.2 | 1.40 | 1.17 | 102.8 | 117.2 | 11.8 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 3930.2 | 1.96 | 1.96 | 208.8 | - | - |
| short-65536-hca-cp1 | tilelang@cute | 3930.2 | 1.40 | 1.17 | 102.0 | 117.1 | 11.8 |
| short-65536-hca-cp1 | cute@cute | 3930.2 | 1.40 | 1.17 | 116.2 | 121.9 | 12.3 |
| short-65536-hca-cp1 | cute_ws@cute | 3930.2 | 1.87 | 1.17 | 236.5 | 144.0 | 14.6 |
| short-65536-hca-cp1 | flashmla_fwd_ref@cute | 3930.2 | 1.96 | 1.96 | 207.6 | - | - |
| short-65536-hca-cp8r0 | tilelang@main | 455.7 | 1.46 | 1.22 | 64.5 | 90.7 | 9.2 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 455.7 | 2.11 | 2.11 | 160.5 | - | - |
| short-65536-hca-cp8r0 | tilelang@cute | 455.7 | 1.46 | 1.22 | 63.6 | 90.1 | 9.1 |
| short-65536-hca-cp8r0 | cute@cute | 455.7 | 1.46 | 1.22 | 96.2 | 104.9 | 10.6 |
| short-65536-hca-cp8r0 | cute_ws@cute | 455.7 | 1.95 | 1.22 | 201.2 | 127.3 | 12.9 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 455.7 | 2.11 | 2.11 | 158.5 | - | - |
| short-65536-hca-cp8r4 | tilelang@main | 514.5 | 1.37 | 1.14 | 71.6 | 99.0 | 10.0 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 514.5 | 1.87 | 1.87 | 176.6 | - | - |
| short-65536-hca-cp8r4 | tilelang@cute | 514.5 | 1.37 | 1.14 | 71.3 | 98.1 | 9.9 |
| short-65536-hca-cp8r4 | cute@cute | 514.5 | 1.37 | 1.14 | 106.2 | 113.7 | 11.5 |
| short-65536-hca-cp8r4 | cute_ws@cute | 514.5 | 1.83 | 1.14 | 221.5 | 137.4 | 13.9 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 514.5 | 1.87 | 1.87 | 176.1 | - | - |
| short-65536-hca-cp8r7 | tilelang@main | 491.0 | 1.40 | 1.17 | 68.7 | 95.0 | 9.6 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 491.0 | 1.96 | 1.96 | 171.1 | - | - |
| short-65536-hca-cp8r7 | tilelang@cute | 491.0 | 1.40 | 1.17 | 68.0 | 95.2 | 9.6 |
| short-65536-hca-cp8r7 | cute@cute | 491.0 | 1.40 | 1.17 | 101.7 | 110.3 | 11.1 |
| short-65536-hca-cp8r7 | cute_ws@cute | 491.0 | 1.87 | 1.17 | 213.7 | 133.5 | 13.5 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 491.0 | 1.96 | 1.96 | 168.3 | - | - |
| short-65536-sliding-cp1 | tilelang@main | 3673.0 | 1.02 | 1.01 | 109.2 | 123.9 | 12.5 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 3673.0 | 1.05 | 1.05 | 255.6 | - | - |
| short-65536-sliding-cp1 | tilelang@cute | 3673.0 | 1.02 | 1.01 | 108.4 | 123.6 | 12.5 |
| short-65536-sliding-cp1 | cute@cute | 3673.0 | 1.02 | 1.01 | 124.6 | 129.3 | 13.1 |
| short-65536-sliding-cp1 | cute_ws@cute | 3673.0 | 1.05 | 1.01 | 325.4 | 158.3 | 16.0 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@cute | 3673.0 | 1.05 | 1.05 | 250.3 | - | - |
| short-65536-sliding-cp8r0 | tilelang@main | 443.7 | 1.04 | 1.02 | 68.1 | 97.4 | 9.8 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 443.7 | 1.08 | 1.08 | 190.4 | - | - |
| short-65536-sliding-cp8r0 | tilelang@cute | 443.7 | 1.04 | 1.02 | 69.0 | 97.0 | 9.8 |
| short-65536-sliding-cp8r0 | cute@cute | 443.7 | 1.04 | 1.02 | 107.2 | 114.8 | 11.6 |
| short-65536-sliding-cp8r0 | cute_ws@cute | 443.7 | 1.08 | 1.02 | 272.3 | 142.5 | 14.4 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 443.7 | 1.08 | 1.08 | 189.5 | - | - |
| short-65536-sliding-cp8r4 | tilelang@main | 469.9 | 1.01 | 1.01 | 73.0 | 101.6 | 10.3 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 469.9 | 1.02 | 1.02 | 203.3 | - | - |
| short-65536-sliding-cp8r4 | tilelang@cute | 469.9 | 1.01 | 1.01 | 72.8 | 100.3 | 10.1 |
| short-65536-sliding-cp8r4 | cute@cute | 469.9 | 1.01 | 1.01 | 113.5 | 118.5 | 12.0 |
| short-65536-sliding-cp8r4 | cute_ws@cute | 469.9 | 1.02 | 1.01 | 290.9 | 146.3 | 14.8 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 469.9 | 1.02 | 1.02 | 200.6 | - | - |
| short-65536-sliding-cp8r7 | tilelang@main | 458.7 | 1.02 | 1.01 | 71.2 | 99.0 | 10.0 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 458.7 | 1.05 | 1.05 | 197.9 | - | - |
| short-65536-sliding-cp8r7 | tilelang@cute | 458.7 | 1.02 | 1.01 | 70.9 | 98.9 | 10.0 |
| short-65536-sliding-cp8r7 | cute@cute | 458.7 | 1.02 | 1.01 | 110.6 | 117.1 | 11.8 |
| short-65536-sliding-cp8r7 | cute_ws@cute | 458.7 | 1.05 | 1.01 | 282.4 | 144.9 | 14.6 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 458.7 | 1.05 | 1.05 | 194.3 | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 12785.2 | 1.07 | 1.04 | 210.3 | 183.4 | 18.5 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 12785.2 | 1.50 | 1.50 | 414.1 | - | - |
| heavy-65536-csa-cp1 | tilelang@cute | 12785.2 | 1.07 | 1.04 | 210.0 | 183.5 | 18.5 |
| heavy-65536-csa-cp1 | cute@cute | 12785.2 | 1.07 | 1.04 | 227.5 | 187.3 | 18.9 |
| heavy-65536-csa-cp1 | cute_ws@cute | 12785.2 | 1.12 | 1.04 | 432.3 | 212.9 | 21.5 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@cute | 12785.2 | 1.50 | 1.50 | 405.6 | - | - |
| heavy-65536-csa-cp8r0 | tilelang@main | 771.9 | 1.27 | 1.18 | 98.5 | 120.2 | 12.2 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 771.9 | 3.12 | 3.12 | 244.8 | - | - |
| heavy-65536-csa-cp8r0 | tilelang@cute | 771.9 | 1.27 | 1.18 | 98.6 | 119.7 | 12.1 |
| heavy-65536-csa-cp8r0 | cute@cute | 771.9 | 1.27 | 1.18 | 139.7 | 134.3 | 13.6 |
| heavy-65536-csa-cp8r0 | cute_ws@cute | 771.9 | 1.49 | 1.18 | 288.4 | 157.9 | 16.0 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 771.9 | 3.12 | 3.12 | 242.5 | - | - |
| heavy-65536-csa-cp8r4 | tilelang@main | 2405.2 | 1.00 | 1.00 | 202.0 | 185.4 | 18.7 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 490.2 | - | - |
| heavy-65536-csa-cp8r4 | tilelang@cute | 2405.2 | 1.00 | 1.00 | 202.0 | 184.9 | 18.7 |
| heavy-65536-csa-cp8r4 | cute@cute | 2405.2 | 1.00 | 1.00 | 253.8 | 196.1 | 19.8 |
| heavy-65536-csa-cp8r4 | cute_ws@cute | 2405.2 | 1.00 | 1.00 | 532.9 | 222.6 | 22.5 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 2405.2 | 1.00 | 1.00 | 484.3 | - | - |
| heavy-65536-csa-cp8r7 | tilelang@main | 1250.3 | 1.12 | 1.08 | 140.1 | 150.7 | 15.2 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1250.3 | 1.92 | 1.92 | 341.0 | - | - |
| heavy-65536-csa-cp8r7 | tilelang@cute | 1250.3 | 1.12 | 1.08 | 137.8 | 150.4 | 15.2 |
| heavy-65536-csa-cp8r7 | cute@cute | 1250.3 | 1.12 | 1.08 | 188.5 | 164.5 | 16.6 |
| heavy-65536-csa-cp8r7 | cute_ws@cute | 1250.3 | 1.23 | 1.08 | 390.3 | 190.2 | 19.2 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 1250.3 | 1.92 | 1.92 | 336.8 | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 5141.1 | 1.25 | 1.11 | 125.4 | 134.3 | 13.6 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5141.1 | 2.25 | 2.25 | 258.5 | - | - |
| heavy-65536-hca-cp1 | tilelang@cute | 5141.1 | 1.25 | 1.11 | 125.1 | 134.2 | 13.6 |
| heavy-65536-hca-cp1 | cute@cute | 5141.1 | 1.25 | 1.11 | 138.8 | 139.1 | 14.1 |
| heavy-65536-hca-cp1 | cute_ws@cute | 5141.1 | 1.52 | 1.11 | 294.5 | 163.5 | 16.5 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@cute | 5141.1 | 2.25 | 2.25 | 254.1 | - | - |
| heavy-65536-hca-cp8r0 | tilelang@main | 408.9 | 1.46 | 1.22 | 58.7 | 84.8 | 8.6 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 408.9 | 3.53 | 3.53 | 150.4 | - | - |
| heavy-65536-hca-cp8r0 | tilelang@cute | 408.9 | 1.46 | 1.22 | 59.6 | 84.6 | 8.5 |
| heavy-65536-hca-cp8r0 | cute@cute | 408.9 | 1.46 | 1.22 | 89.6 | 99.2 | 10.0 |
| heavy-65536-hca-cp8r0 | cute_ws@cute | 408.9 | 1.95 | 1.22 | 192.3 | 121.5 | 12.3 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 408.9 | 3.53 | 3.53 | 148.4 | - | - |
| heavy-65536-hca-cp8r4 | tilelang@main | 965.4 | 1.12 | 1.06 | 116.0 | 136.9 | 13.8 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 965.4 | 1.49 | 1.49 | 297.2 | - | - |
| heavy-65536-hca-cp8r4 | tilelang@cute | 965.4 | 1.12 | 1.06 | 116.8 | 135.9 | 13.7 |
| heavy-65536-hca-cp8r4 | cute@cute | 965.4 | 1.12 | 1.06 | 163.7 | 151.5 | 15.3 |
| heavy-65536-hca-cp8r4 | cute_ws@cute | 965.4 | 1.25 | 1.06 | 357.2 | 179.0 | 18.1 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 965.4 | 1.49 | 1.49 | 295.5 | - | - |
| heavy-65536-hca-cp8r7 | tilelang@main | 458.3 | 1.40 | 1.17 | 65.3 | 91.5 | 9.2 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 458.3 | 3.15 | 3.15 | 163.9 | - | - |
| heavy-65536-hca-cp8r7 | tilelang@cute | 458.3 | 1.40 | 1.17 | 64.7 | 91.4 | 9.2 |
| heavy-65536-hca-cp8r7 | cute@cute | 458.3 | 1.40 | 1.17 | 96.9 | 106.5 | 10.8 |
| heavy-65536-hca-cp8r7 | cute_ws@cute | 458.3 | 1.87 | 1.17 | 204.1 | 129.8 | 13.1 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 458.3 | 3.15 | 3.15 | 161.3 | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 3542.5 | 1.04 | 1.02 | 105.3 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 3542.5 | 1.09 | 1.09 | 246.2 | - | - |
| heavy-65536-sliding-cp1 | tilelang@cute | 3542.5 | 1.04 | 1.02 | 105.3 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | cute@cute | 3542.5 | 1.04 | 1.02 | 120.8 | 126.6 | 12.8 |
| heavy-65536-sliding-cp1 | cute_ws@cute | 3542.5 | 1.09 | 1.02 | 314.9 | 155.1 | 15.7 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@cute | 3542.5 | 1.09 | 1.09 | 245.2 | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@main | 399.0 | 1.10 | 1.05 | 61.5 | 90.6 | 9.2 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 399.0 | 1.21 | 1.21 | 170.2 | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@cute | 399.0 | 1.10 | 1.05 | 62.6 | 90.2 | 9.1 |
| heavy-65536-sliding-cp8r0 | cute@cute | 399.0 | 1.10 | 1.05 | 97.4 | 107.2 | 10.8 |
| heavy-65536-sliding-cp8r0 | cute_ws@cute | 399.0 | 1.21 | 1.05 | 243.4 | 133.6 | 13.5 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 399.0 | 1.21 | 1.21 | 168.7 | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.7 | 102.8 | 10.4 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 209.5 | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@cute | 481.0 | 1.00 | 1.00 | 73.1 | 100.5 | 10.2 |
| heavy-65536-sliding-cp8r4 | cute@cute | 481.0 | 1.00 | 1.00 | 115.0 | 119.3 | 12.1 |
| heavy-65536-sliding-cp8r4 | cute_ws@cute | 481.0 | 1.00 | 1.00 | 296.3 | 143.5 | 14.5 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 481.0 | 1.00 | 1.00 | 205.4 | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@main | 428.8 | 1.06 | 1.03 | 67.2 | 94.8 | 9.6 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 428.8 | 1.12 | 1.12 | 184.1 | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@cute | 428.8 | 1.06 | 1.03 | 65.8 | 94.5 | 9.6 |
| heavy-65536-sliding-cp8r7 | cute@cute | 428.8 | 1.06 | 1.03 | 103.3 | 111.9 | 11.3 |
| heavy-65536-sliding-cp8r7 | cute_ws@cute | 428.8 | 1.12 | 1.03 | 262.2 | 139.0 | 14.1 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 428.8 | 1.12 | 1.12 | 180.8 | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 1621.2 | 3.45 | 2.86 | 42.6 | 51.0 | 5.2 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 1621.2 | 11.87 | 11.87 | 81.8 | - | - |
| tiny-65536-csa-cp1 | tilelang@cute | 1621.2 | 3.45 | 2.86 | 42.5 | 50.9 | 5.1 |
| tiny-65536-csa-cp1 | cute@cute | 1621.2 | 3.45 | 2.86 | 47.4 | 53.0 | 5.4 |
| tiny-65536-csa-cp1 | cute_ws@cute | 1621.2 | 4.64 | 2.86 | 94.1 | 62.9 | 6.4 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@cute | 1621.2 | 11.87 | 11.87 | 81.8 | - | - |
| tiny-65536-csa-cp8r0 | tilelang@main | 202.8 | 3.45 | 2.86 | 28.9 | 41.5 | 4.2 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 202.8 | 11.86 | 11.86 | 71.6 | - | - |
| tiny-65536-csa-cp8r0 | tilelang@cute | 202.8 | 3.45 | 2.86 | 29.0 | 41.4 | 4.2 |
| tiny-65536-csa-cp8r0 | cute@cute | 202.8 | 3.45 | 2.86 | 43.1 | 48.4 | 4.9 |
| tiny-65536-csa-cp8r0 | cute_ws@cute | 202.8 | 4.63 | 2.86 | 85.1 | 58.2 | 5.9 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@cute | 202.8 | 11.86 | 11.86 | 70.3 | - | - |
| tiny-65536-csa-cp8r4 | tilelang@main | 202.7 | 3.45 | 2.86 | 28.8 | 41.6 | 4.2 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 202.7 | 11.87 | 11.87 | 70.3 | - | - |
| tiny-65536-csa-cp8r4 | tilelang@cute | 202.7 | 3.45 | 2.86 | 28.9 | 41.3 | 4.2 |
| tiny-65536-csa-cp8r4 | cute@cute | 202.7 | 3.45 | 2.86 | 43.0 | 48.1 | 4.9 |
| tiny-65536-csa-cp8r4 | cute_ws@cute | 202.7 | 4.64 | 2.86 | 84.5 | 58.1 | 5.9 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@cute | 202.7 | 11.87 | 11.87 | 69.9 | - | - |
| tiny-65536-csa-cp8r7 | tilelang@main | 201.4 | 3.47 | 2.88 | 27.9 | 41.3 | 4.2 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 201.4 | 11.94 | 11.94 | 69.2 | - | - |
| tiny-65536-csa-cp8r7 | tilelang@cute | 201.4 | 3.47 | 2.88 | 28.5 | 41.0 | 4.1 |
| tiny-65536-csa-cp8r7 | cute@cute | 201.4 | 3.47 | 2.88 | 42.6 | 47.8 | 4.8 |
| tiny-65536-csa-cp8r7 | cute_ws@cute | 201.4 | 4.67 | 2.88 | 84.6 | 57.7 | 5.8 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@cute | 201.4 | 11.94 | 11.94 | 69.9 | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.4 | - | - |
| tiny-65536-hca-cp1 | tilelang@cute | 1306.0 | 1.85 | 1.39 | 43.0 | 59.3 | 6.0 |
| tiny-65536-hca-cp1 | cute@cute | 1306.0 | 1.85 | 1.39 | 50.1 | 63.0 | 6.4 |
| tiny-65536-hca-cp1 | cute_ws@cute | 1306.0 | 2.95 | 1.39 | 101.7 | 77.1 | 7.8 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@cute | 1306.0 | 2.95 | 2.95 | 89.7 | - | - |
| tiny-65536-hca-cp8r0 | tilelang@main | 163.4 | 1.85 | 1.39 | 27.4 | 45.0 | 4.6 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 163.4 | 2.94 | 2.94 | 71.0 | - | - |
| tiny-65536-hca-cp8r0 | tilelang@cute | 163.4 | 1.85 | 1.39 | 27.1 | 45.2 | 4.6 |
| tiny-65536-hca-cp8r0 | cute@cute | 163.4 | 1.85 | 1.39 | 43.9 | 55.7 | 5.6 |
| tiny-65536-hca-cp8r0 | cute_ws@cute | 163.4 | 2.94 | 1.39 | 90.8 | 69.8 | 7.1 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@cute | 163.4 | 2.94 | 2.94 | 69.5 | - | - |
| tiny-65536-hca-cp8r4 | tilelang@main | 163.3 | 1.85 | 1.39 | 26.9 | 45.2 | 4.6 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-hca-cp8r4 | tilelang@cute | 163.3 | 1.85 | 1.39 | 26.6 | 44.7 | 4.5 |
| tiny-65536-hca-cp8r4 | cute@cute | 163.3 | 1.85 | 1.39 | 43.5 | 55.4 | 5.6 |
| tiny-65536-hca-cp8r4 | cute_ws@cute | 163.3 | 2.95 | 1.39 | 91.9 | 69.3 | 7.0 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@cute | 163.3 | 2.95 | 2.95 | 70.2 | - | - |
| tiny-65536-hca-cp8r7 | tilelang@main | 162.3 | 1.86 | 1.39 | 27.0 | 44.7 | 4.5 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 162.3 | 2.96 | 2.96 | 69.6 | - | - |
| tiny-65536-hca-cp8r7 | tilelang@cute | 162.3 | 1.86 | 1.39 | 26.9 | 44.7 | 4.5 |
| tiny-65536-hca-cp8r7 | cute@cute | 162.3 | 1.86 | 1.39 | 43.8 | 55.4 | 5.6 |
| tiny-65536-hca-cp8r7 | cute_ws@cute | 162.3 | 2.96 | 1.39 | 90.1 | 69.3 | 7.0 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@cute | 162.3 | 2.96 | 2.96 | 69.4 | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.8 | 59.3 | 6.0 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.3 | - | - |
| tiny-65536-sliding-cp1 | tilelang@cute | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-sliding-cp1 | cute@cute | 1306.0 | 1.85 | 1.39 | 50.2 | 63.0 | 6.4 |
| tiny-65536-sliding-cp1 | cute_ws@cute | 1306.0 | 2.95 | 1.39 | 101.6 | 77.1 | 7.8 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@cute | 1306.0 | 2.95 | 2.95 | 89.7 | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@main | 163.4 | 1.85 | 1.39 | 26.7 | 45.0 | 4.5 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 163.4 | 2.94 | 2.94 | 69.8 | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@cute | 163.4 | 1.85 | 1.39 | 27.1 | 44.9 | 4.5 |
| tiny-65536-sliding-cp8r0 | cute@cute | 163.4 | 1.85 | 1.39 | 43.9 | 55.7 | 5.6 |
| tiny-65536-sliding-cp8r0 | cute_ws@cute | 163.4 | 2.94 | 1.39 | 89.9 | 69.6 | 7.0 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@cute | 163.4 | 2.94 | 2.94 | 69.6 | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@main | 163.3 | 1.85 | 1.39 | 27.2 | 45.2 | 4.6 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@cute | 163.3 | 1.85 | 1.39 | 27.0 | 44.8 | 4.5 |
| tiny-65536-sliding-cp8r4 | cute@cute | 163.3 | 1.85 | 1.39 | 43.6 | 55.5 | 5.6 |
| tiny-65536-sliding-cp8r4 | cute_ws@cute | 163.3 | 2.95 | 1.39 | 91.1 | 69.8 | 7.1 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@cute | 163.3 | 2.95 | 2.95 | 69.8 | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@main | 162.3 | 1.86 | 1.39 | 27.2 | 44.7 | 4.5 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 162.3 | 2.96 | 2.96 | 69.7 | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@cute | 162.3 | 1.86 | 1.39 | 26.9 | 44.8 | 4.5 |
| tiny-65536-sliding-cp8r7 | cute@cute | 162.3 | 1.86 | 1.39 | 43.5 | 55.4 | 5.6 |
| tiny-65536-sliding-cp8r7 | cute_ws@cute | 162.3 | 2.96 | 1.39 | 90.8 | 69.2 | 7.0 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@cute | 162.3 | 2.96 | 2.96 | 69.1 | - | - |

Correctness failures (excluded from timing): 0
