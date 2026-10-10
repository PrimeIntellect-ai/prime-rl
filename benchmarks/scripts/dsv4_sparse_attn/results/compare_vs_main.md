- `main`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git 6d5cf0180
- `compare`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git dfb6b9bd3
- corpus hash b5b289171983c8c6; synthetic corpus: random-weight CSA picks are near-uniform, while a
  trained indexer favors recent and neighboring entries, so CSA gather locality here is pessimistic.

Op-boundary time per call in µs (lower is better): median over rounds, p20-p80 across rounds.
`/TL` is this time divided by tilelang's in the same run.

| item | backend | fwd µs | fwd p20-p80 | fwd /TL | f+b µs | f+b p20-p80 | f+b /TL |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 1197 | 1180-1209 | 1.00 | 3186 | 3183-3216 | 1.00 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 384.0 | 381.9-387.6 | 0.32 | - | - | - |
| single-2048-csa-cp1 | tilelang@compare | 1210 | 1204-1219 | 1.00 | 3266 | 3260-3271 | 1.00 |
| single-2048-csa-cp1 | cudnn_flashmla@compare | 437.6 | 434.8-446.6 | 0.36 | 1940 | 1933-1959 | 0.59 |
| single-2048-csa-cp1 | cute@compare | 607.5 | 605.1-608.9 | 0.50 | 2630 | 2620-2639 | 0.81 |
| single-2048-csa-cp1 | cute_ws@compare | 304.3 | 302.4-306.0 | 0.25 | 2486 | 2477-2504 | 0.76 |
| single-2048-csa-cp1 | flashmla_fwd_ref@compare | 381.5 | 378.3-384.0 | 0.32 | - | - | - |
| single-2048-csa-cp8r0 | tilelang@main | 733.5 | 729.6-769.4 | 1.00 | 2001 | 1987-2043 | 1.00 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 182.5 | 181.2-196.9 | 0.25 | - | - | - |
| single-2048-csa-cp8r0 | tilelang@compare | 760.8 | 755.9-771.9 | 1.00 | 2116 | 2113-2124 | 1.00 |
| single-2048-csa-cp8r0 | cudnn_flashmla@compare | 319.0 | 317.2-321.6 | 0.42 | 1472 | 1461-1474 | 0.70 |
| single-2048-csa-cp8r0 | cute@compare | 182.2 | 175.7-182.6 | 0.24 | 1461 | 1451-1462 | 0.69 |
| single-2048-csa-cp8r0 | cute_ws@compare | 86.0 | 85.8-87.0 | 0.11 | 1303 | 1287-1332 | 0.62 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 195.4 | 192.2-196.9 | 0.26 | - | - | - |
| single-2048-csa-cp8r4 | tilelang@main | 763.3 | 755.8-769.4 | 1.00 | 2028 | 2010-2035 | 1.00 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 197.7 | 194.4-198.5 | 0.26 | - | - | - |
| single-2048-csa-cp8r4 | tilelang@compare | 770.3 | 757.9-781.5 | 1.00 | 2127 | 2090-2132 | 1.00 |
| single-2048-csa-cp8r4 | cudnn_flashmla@compare | 310.7 | 309.6-312.4 | 0.40 | 1475 | 1469-1485 | 0.69 |
| single-2048-csa-cp8r4 | cute@compare | 200.2 | 198.5-202.7 | 0.26 | 1450 | 1439-1453 | 0.68 |
| single-2048-csa-cp8r4 | cute_ws@compare | 98.6 | 98.1-102.0 | 0.13 | 1292 | 1289-1306 | 0.61 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 202.6 | 199.7-204.5 | 0.26 | - | - | - |
| single-2048-csa-cp8r7 | tilelang@main | 774.3 | 771.8-795.5 | 1.00 | 2025 | 2024-2030 | 1.00 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 202.8 | 198.5-208.2 | 0.26 | - | - | - |
| single-2048-csa-cp8r7 | tilelang@compare | 789.8 | 788.1-799.9 | 1.00 | 2116 | 2102-2134 | 1.00 |
| single-2048-csa-cp8r7 | cudnn_flashmla@compare | 315.4 | 313.5-319.5 | 0.40 | 1470 | 1463-1489 | 0.69 |
| single-2048-csa-cp8r7 | cute@compare | 216.2 | 215.6-217.8 | 0.27 | 1461 | 1443-1503 | 0.69 |
| single-2048-csa-cp8r7 | cute_ws@compare | 105.1 | 104.9-106.6 | 0.13 | 1298 | 1296-1310 | 0.61 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 208.3 | 206.7-212.1 | 0.26 | - | - | - |
| single-2048-hca-cp1 | tilelang@main | 1085 | 1075-1099 | 1.00 | 2504 | 2477-2531 | 1.00 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 341.1 | 339.6-343.2 | 0.31 | - | - | - |
| single-2048-hca-cp1 | tilelang@compare | 1102 | 1095-1109 | 1.00 | 2568 | 2563-2577 | 1.00 |
| single-2048-hca-cp1 | cudnn_flashmla@compare | 403.9 | 401.9-408.0 | 0.37 | 1585 | 1576-1593 | 0.62 |
| single-2048-hca-cp1 | cute@compare | 483.7 | 482.9-484.2 | 0.44 | 1919 | 1906-1938 | 0.75 |
| single-2048-hca-cp1 | cute_ws@compare | 222.3 | 221.6-224.7 | 0.20 | 1732 | 1723-1737 | 0.67 |
| single-2048-hca-cp1 | flashmla_fwd_ref@compare | 342.3 | 341.5-349.9 | 0.31 | - | - | - |
| single-2048-hca-cp8r0 | tilelang@main | 763.9 | 761.9-769.2 | 1.00 | 2152 | 2142-2403 | 1.00 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 188.7 | 185.0-194.9 | 0.25 | - | - | - |
| single-2048-hca-cp8r0 | tilelang@compare | 785.1 | 778.3-807.3 | 1.00 | 2210 | 2203-2255 | 1.00 |
| single-2048-hca-cp8r0 | cudnn_flashmla@compare | 322.0 | 315.1-323.9 | 0.41 | 1494 | 1476-1525 | 0.68 |
| single-2048-hca-cp8r0 | cute@compare | 197.6 | 195.6-199.1 | 0.25 | 1574 | 1556-1608 | 0.71 |
| single-2048-hca-cp8r0 | cute_ws@compare | 76.6 | 75.3-77.0 | 0.10 | 1359 | 1350-1369 | 0.62 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 189.4 | 185.8-191.7 | 0.24 | - | - | - |
| single-2048-hca-cp8r4 | tilelang@main | 793.0 | 784.8-810.2 | 1.00 | 2158 | 2111-2193 | 1.00 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 193.9 | 191.6-196.0 | 0.24 | - | - | - |
| single-2048-hca-cp8r4 | tilelang@compare | 806.7 | 796.6-823.3 | 1.00 | 2221 | 2195-2329 | 1.00 |
| single-2048-hca-cp8r4 | cudnn_flashmla@compare | 316.7 | 310.8-336.7 | 0.39 | 1485 | 1478-1487 | 0.67 |
| single-2048-hca-cp8r4 | cute@compare | 204.4 | 200.8-205.2 | 0.25 | 1549 | 1544-1578 | 0.70 |
| single-2048-hca-cp8r4 | cute_ws@compare | 80.7 | 80.2-84.4 | 0.10 | 1406 | 1345-1448 | 0.63 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 195.2 | 192.9-197.7 | 0.24 | - | - | - |
| single-2048-hca-cp8r7 | tilelang@main | 806.9 | 791.4-813.7 | 1.00 | 2178 | 2146-2204 | 1.00 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 193.5 | 189.0-195.7 | 0.24 | - | - | - |
| single-2048-hca-cp8r7 | tilelang@compare | 784.0 | 772.7-785.0 | 1.00 | 2221 | 2202-2234 | 1.00 |
| single-2048-hca-cp8r7 | cudnn_flashmla@compare | 310.6 | 306.3-316.0 | 0.40 | 1459 | 1450-1480 | 0.66 |
| single-2048-hca-cp8r7 | cute@compare | 200.4 | 198.1-204.1 | 0.26 | 1562 | 1537-1569 | 0.70 |
| single-2048-hca-cp8r7 | cute_ws@compare | 80.3 | 79.0-83.0 | 0.10 | 1359 | 1324-1376 | 0.61 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 190.6 | 187.2-196.2 | 0.24 | - | - | - |
| single-2048-sliding-cp1 | tilelang@main | 1016 | 1002-1021 | 1.00 | 2288 | 2278-2306 | 1.00 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 298.4 | 293.8-303.0 | 0.29 | - | - | - |
| single-2048-sliding-cp1 | tilelang@compare | 1028 | 1006-1030 | 1.00 | 2390 | 2378-2404 | 1.00 |
| single-2048-sliding-cp1 | cudnn_flashmla@compare | 362.0 | 361.4-374.1 | 0.35 | 1530 | 1517-1536 | 0.64 |
| single-2048-sliding-cp1 | cute@compare | 410.5 | 409.7-420.3 | 0.40 | 1754 | 1727-1765 | 0.73 |
| single-2048-sliding-cp1 | cute_ws@compare | 171.3 | 170.3-174.1 | 0.17 | 1598 | 1591-1613 | 0.67 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@compare | 293.8 | 289.1-299.8 | 0.29 | - | - | - |
| single-2048-sliding-cp8r0 | tilelang@main | 732.5 | 723.8-761.5 | 1.00 | 2047 | 2013-2148 | 1.00 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 179.7 | 174.7-186.8 | 0.25 | - | - | - |
| single-2048-sliding-cp8r0 | tilelang@compare | 758.9 | 745.9-767.5 | 1.00 | 2128 | 2114-2151 | 1.00 |
| single-2048-sliding-cp8r0 | cudnn_flashmla@compare | 312.6 | 304.9-331.1 | 0.41 | 1472 | 1459-1484 | 0.69 |
| single-2048-sliding-cp8r0 | cute@compare | 171.7 | 168.8-174.8 | 0.23 | 1488 | 1465-1504 | 0.70 |
| single-2048-sliding-cp8r0 | cute_ws@compare | 72.1 | 71.3-73.4 | 0.10 | 1310 | 1287-1322 | 0.62 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 179.0 | 175.9-187.7 | 0.24 | - | - | - |
| single-2048-sliding-cp8r4 | tilelang@main | 739.5 | 732.5-778.8 | 1.00 | 2018 | 2006-2039 | 1.00 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 179.0 | 174.0-188.2 | 0.24 | - | - | - |
| single-2048-sliding-cp8r4 | tilelang@compare | 748.1 | 741.6-760.6 | 1.00 | 2112 | 2101-2123 | 1.00 |
| single-2048-sliding-cp8r4 | cudnn_flashmla@compare | 311.9 | 306.0-313.3 | 0.42 | 1476 | 1470-1480 | 0.70 |
| single-2048-sliding-cp8r4 | cute@compare | 169.4 | 169.0-172.4 | 0.23 | 1470 | 1467-1581 | 0.70 |
| single-2048-sliding-cp8r4 | cute_ws@compare | 75.1 | 74.6-75.9 | 0.10 | 1304 | 1292-1309 | 0.62 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 180.1 | 178.2-189.0 | 0.24 | - | - | - |
| single-2048-sliding-cp8r7 | tilelang@main | 739.5 | 735.6-746.3 | 1.00 | 2034 | 2020-2067 | 1.00 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 180.9 | 179.1-190.1 | 0.24 | - | - | - |
| single-2048-sliding-cp8r7 | tilelang@compare | 760.0 | 744.2-822.9 | 1.00 | 2109 | 2101-2154 | 1.00 |
| single-2048-sliding-cp8r7 | cudnn_flashmla@compare | 318.1 | 312.8-333.7 | 0.42 | 1468 | 1455-1494 | 0.70 |
| single-2048-sliding-cp8r7 | cute@compare | 172.7 | 172.0-181.0 | 0.23 | 1456 | 1451-1465 | 0.69 |
| single-2048-sliding-cp8r7 | cute_ws@compare | 74.3 | 73.9-78.2 | 0.10 | 1287 | 1284-1292 | 0.61 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 187.3 | 185.6-189.9 | 0.25 | - | - | - |
| short-2048-csa-cp1 | tilelang@main | 1132 | 1121-1135 | 1.00 | 2756 | 2748-2783 | 1.00 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 368.0 | 365.2-371.5 | 0.33 | - | - | - |
| short-2048-csa-cp1 | tilelang@compare | 1129 | 1120-1139 | 1.00 | 2874 | 2869-2971 | 1.00 |
| short-2048-csa-cp1 | cudnn_flashmla@compare | 430.2 | 428.3-433.3 | 0.38 | 1746 | 1742-1758 | 0.61 |
| short-2048-csa-cp1 | cute@compare | 536.7 | 533.9-537.6 | 0.48 | 2222 | 2201-2239 | 0.77 |
| short-2048-csa-cp1 | cute_ws@compare | 261.3 | 260.7-262.7 | 0.23 | 2069 | 2054-2080 | 0.72 |
| short-2048-csa-cp1 | flashmla_fwd_ref@compare | 367.3 | 365.1-371.7 | 0.33 | - | - | - |
| short-2048-csa-cp8r0 | tilelang@main | 753.1 | 737.6-770.5 | 1.00 | 2023 | 2006-2054 | 1.00 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 187.7 | 179.4-191.9 | 0.25 | - | - | - |
| short-2048-csa-cp8r0 | tilelang@compare | 760.5 | 756.5-779.8 | 1.00 | 2116 | 2104-2144 | 1.00 |
| short-2048-csa-cp8r0 | cudnn_flashmla@compare | 315.6 | 310.6-322.0 | 0.42 | 1489 | 1474-1493 | 0.70 |
| short-2048-csa-cp8r0 | cute@compare | 179.2 | 174.4-181.7 | 0.24 | 1461 | 1444-1467 | 0.69 |
| short-2048-csa-cp8r0 | cute_ws@compare | 87.7 | 86.1-88.4 | 0.12 | 1308 | 1301-1309 | 0.62 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 189.0 | 187.2-193.4 | 0.25 | - | - | - |
| short-2048-csa-cp8r4 | tilelang@main | 742.5 | 736.4-755.8 | 1.00 | 2033 | 2020-2063 | 1.00 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 187.7 | 183.3-192.1 | 0.25 | - | - | - |
| short-2048-csa-cp8r4 | tilelang@compare | 756.0 | 750.5-786.3 | 1.00 | 2088 | 2074-2115 | 1.00 |
| short-2048-csa-cp8r4 | cudnn_flashmla@compare | 308.5 | 307.3-310.9 | 0.41 | 1445 | 1436-1466 | 0.69 |
| short-2048-csa-cp8r4 | cute@compare | 185.2 | 185.0-187.0 | 0.25 | 1434 | 1411-1455 | 0.69 |
| short-2048-csa-cp8r4 | cute_ws@compare | 89.3 | 88.6-91.6 | 0.12 | 1278 | 1263-1293 | 0.61 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 197.6 | 192.5-199.1 | 0.26 | - | - | - |
| short-2048-csa-cp8r7 | tilelang@main | 762.2 | 749.4-771.5 | 1.00 | 2037 | 2025-2095 | 1.00 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 191.5 | 186.9-192.3 | 0.25 | - | - | - |
| short-2048-csa-cp8r7 | tilelang@compare | 764.4 | 755.1-788.5 | 1.00 | 2085 | 2063-2099 | 1.00 |
| short-2048-csa-cp8r7 | cudnn_flashmla@compare | 307.6 | 303.7-320.8 | 0.40 | 1461 | 1449-1525 | 0.70 |
| short-2048-csa-cp8r7 | cute@compare | 198.1 | 188.1-201.4 | 0.26 | 1426 | 1409-1484 | 0.68 |
| short-2048-csa-cp8r7 | cute_ws@compare | 91.2 | 90.2-92.0 | 0.12 | 1275 | 1258-1279 | 0.61 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 192.4 | 188.7-193.5 | 0.25 | - | - | - |
| short-2048-hca-cp1 | tilelang@main | 1079 | 1074-1103 | 1.00 | 2489 | 2485-2573 | 1.00 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 340.4 | 337.8-344.8 | 0.32 | - | - | - |
| short-2048-hca-cp1 | tilelang@compare | 1072 | 1065-1084 | 1.00 | 2513 | 2493-2528 | 1.00 |
| short-2048-hca-cp1 | cudnn_flashmla@compare | 402.7 | 397.6-404.2 | 0.38 | 1560 | 1554-1565 | 0.62 |
| short-2048-hca-cp1 | cute@compare | 474.2 | 472.2-476.7 | 0.44 | 1886 | 1853-1898 | 0.75 |
| short-2048-hca-cp1 | cute_ws@compare | 219.5 | 218.6-220.7 | 0.20 | 1673 | 1668-1675 | 0.67 |
| short-2048-hca-cp1 | flashmla_fwd_ref@compare | 340.9 | 339.3-342.2 | 0.32 | - | - | - |
| short-2048-hca-cp8r0 | tilelang@main | 787.4 | 779.9-790.5 | 1.00 | 2154 | 2142-2191 | 1.00 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 191.5 | 187.6-192.1 | 0.24 | - | - | - |
| short-2048-hca-cp8r0 | tilelang@compare | 782.6 | 774.8-801.8 | 1.00 | 2200 | 2179-2214 | 1.00 |
| short-2048-hca-cp8r0 | cudnn_flashmla@compare | 316.1 | 309.5-318.4 | 0.40 | 1464 | 1453-1478 | 0.67 |
| short-2048-hca-cp8r0 | cute@compare | 198.7 | 198.2-210.7 | 0.25 | 1549 | 1533-1560 | 0.70 |
| short-2048-hca-cp8r0 | cute_ws@compare | 78.3 | 77.0-83.2 | 0.10 | 1340 | 1331-1346 | 0.61 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 195.3 | 189.3-201.4 | 0.25 | - | - | - |
| short-2048-hca-cp8r4 | tilelang@main | 795.0 | 779.4-801.7 | 1.00 | 2162 | 2154-2181 | 1.00 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 193.3 | 190.6-194.5 | 0.24 | - | - | - |
| short-2048-hca-cp8r4 | tilelang@compare | 777.8 | 767.6-816.3 | 1.00 | 2215 | 2204-2225 | 1.00 |
| short-2048-hca-cp8r4 | cudnn_flashmla@compare | 304.0 | 300.3-314.5 | 0.39 | 1471 | 1465-1475 | 0.66 |
| short-2048-hca-cp8r4 | cute@compare | 195.2 | 192.0-198.1 | 0.25 | 1554 | 1545-1569 | 0.70 |
| short-2048-hca-cp8r4 | cute_ws@compare | 78.4 | 77.4-79.8 | 0.10 | 1367 | 1357-1394 | 0.62 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 189.9 | 189.7-197.2 | 0.24 | - | - | - |
| short-2048-hca-cp8r7 | tilelang@main | 792.8 | 779.8-798.3 | 1.00 | 2193 | 2161-2199 | 1.00 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 188.9 | 188.3-192.7 | 0.24 | - | - | - |
| short-2048-hca-cp8r7 | tilelang@compare | 792.4 | 781.4-796.2 | 1.00 | 2220 | 2193-2232 | 1.00 |
| short-2048-hca-cp8r7 | cudnn_flashmla@compare | 311.2 | 307.3-313.1 | 0.39 | 1470 | 1466-1485 | 0.66 |
| short-2048-hca-cp8r7 | cute@compare | 200.4 | 198.0-202.7 | 0.25 | 1546 | 1536-1561 | 0.70 |
| short-2048-hca-cp8r7 | cute_ws@compare | 80.4 | 78.9-83.0 | 0.10 | 1371 | 1361-1415 | 0.62 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 193.8 | 187.0-198.2 | 0.24 | - | - | - |
| short-2048-sliding-cp1 | tilelang@main | 1003 | 994.3-1018 | 1.00 | 2330 | 2322-2398 | 1.00 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 293.3 | 291.6-301.8 | 0.29 | - | - | - |
| short-2048-sliding-cp1 | tilelang@compare | 998.3 | 995.6-1100 | 1.00 | 2355 | 2343-2378 | 1.00 |
| short-2048-sliding-cp1 | cudnn_flashmla@compare | 355.0 | 352.8-357.2 | 0.36 | 1530 | 1521-1535 | 0.65 |
| short-2048-sliding-cp1 | cute@compare | 405.0 | 402.8-409.0 | 0.41 | 1723 | 1715-1735 | 0.73 |
| short-2048-sliding-cp1 | cute_ws@compare | 169.4 | 168.3-171.0 | 0.17 | 1579 | 1576-1586 | 0.67 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@compare | 290.8 | 287.8-293.6 | 0.29 | - | - | - |
| short-2048-sliding-cp8r0 | tilelang@main | 746.8 | 734.3-752.0 | 1.00 | 2042 | 2039-2056 | 1.00 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 181.9 | 179.7-187.0 | 0.24 | - | - | - |
| short-2048-sliding-cp8r0 | tilelang@compare | 751.2 | 742.9-755.2 | 1.00 | 2126 | 2104-2142 | 1.00 |
| short-2048-sliding-cp8r0 | cudnn_flashmla@compare | 303.6 | 301.6-306.4 | 0.40 | 1495 | 1474-1499 | 0.70 |
| short-2048-sliding-cp8r0 | cute@compare | 170.8 | 168.2-172.3 | 0.23 | 1478 | 1460-1595 | 0.70 |
| short-2048-sliding-cp8r0 | cute_ws@compare | 73.4 | 72.4-76.4 | 0.10 | 1304 | 1284-1306 | 0.61 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 182.2 | 178.9-183.2 | 0.24 | - | - | - |
| short-2048-sliding-cp8r4 | tilelang@main | 756.9 | 751.3-773.4 | 1.00 | 2022 | 2012-2028 | 1.00 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 189.8 | 187.4-194.1 | 0.25 | - | - | - |
| short-2048-sliding-cp8r4 | tilelang@compare | 731.3 | 721.0-746.3 | 1.00 | 2120 | 2109-2130 | 1.00 |
| short-2048-sliding-cp8r4 | cudnn_flashmla@compare | 306.2 | 303.4-313.4 | 0.42 | 1464 | 1459-1482 | 0.69 |
| short-2048-sliding-cp8r4 | cute@compare | 168.4 | 167.5-172.7 | 0.23 | 1456 | 1448-1474 | 0.69 |
| short-2048-sliding-cp8r4 | cute_ws@compare | 74.4 | 71.6-75.5 | 0.10 | 1287 | 1282-1312 | 0.61 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 183.5 | 178.6-188.4 | 0.25 | - | - | - |
| short-2048-sliding-cp8r7 | tilelang@main | 736.4 | 731.7-751.1 | 1.00 | 2044 | 2038-2069 | 1.00 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 185.1 | 179.8-188.5 | 0.25 | - | - | - |
| short-2048-sliding-cp8r7 | tilelang@compare | 739.0 | 737.4-767.2 | 1.00 | 2098 | 2090-2105 | 1.00 |
| short-2048-sliding-cp8r7 | cudnn_flashmla@compare | 314.9 | 307.1-319.0 | 0.43 | 1470 | 1461-1485 | 0.70 |
| short-2048-sliding-cp8r7 | cute@compare | 171.3 | 168.7-173.8 | 0.23 | 1458 | 1448-1467 | 0.69 |
| short-2048-sliding-cp8r7 | cute_ws@compare | 74.4 | 73.5-75.8 | 0.10 | 1284 | 1276-1292 | 0.61 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 179.8 | 176.6-181.1 | 0.24 | - | - | - |
| heavy-2048-csa-cp1 | tilelang@main | 1171 | 1161-1186 | 1.00 | 2955 | 2937-2956 | 1.00 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 389.7 | 383.9-390.4 | 0.33 | - | - | - |
| heavy-2048-csa-cp1 | tilelang@compare | 1153 | 1151-1165 | 1.00 | 3021 | 2985-3028 | 1.00 |
| heavy-2048-csa-cp1 | cudnn_flashmla@compare | 455.1 | 452.7-459.5 | 0.39 | 1811 | 1806-1823 | 0.60 |
| heavy-2048-csa-cp1 | cute@compare | 569.0 | 565.5-570.5 | 0.49 | 2364 | 2356-2378 | 0.78 |
| heavy-2048-csa-cp1 | cute_ws@compare | 277.8 | 276.3-279.5 | 0.24 | 2215 | 2206-2224 | 0.73 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@compare | 390.1 | 387.8-392.9 | 0.34 | - | - | - |
| heavy-2048-csa-cp8r0 | tilelang@main | 747.5 | 737.9-759.0 | 1.00 | 2043 | 2029-2055 | 1.00 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 189.1 | 186.1-190.5 | 0.25 | - | - | - |
| heavy-2048-csa-cp8r0 | tilelang@compare | 764.1 | 761.6-770.4 | 1.00 | 2136 | 2125-2167 | 1.00 |
| heavy-2048-csa-cp8r0 | cudnn_flashmla@compare | 311.2 | 309.9-321.8 | 0.41 | 1490 | 1477-1543 | 0.70 |
| heavy-2048-csa-cp8r0 | cute@compare | 178.8 | 177.9-179.0 | 0.23 | 1473 | 1458-1505 | 0.69 |
| heavy-2048-csa-cp8r0 | cute_ws@compare | 86.3 | 85.7-88.4 | 0.11 | 1324 | 1306-1432 | 0.62 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 188.6 | 183.3-192.5 | 0.25 | - | - | - |
| heavy-2048-csa-cp8r4 | tilelang@main | 777.6 | 755.6-789.9 | 1.00 | 2065 | 2036-2070 | 1.00 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 200.7 | 189.6-207.8 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r4 | tilelang@compare | 764.7 | 756.1-772.0 | 1.00 | 2114 | 2096-2200 | 1.00 |
| heavy-2048-csa-cp8r4 | cudnn_flashmla@compare | 309.9 | 305.5-313.9 | 0.41 | 1461 | 1458-1468 | 0.69 |
| heavy-2048-csa-cp8r4 | cute@compare | 191.0 | 188.6-191.9 | 0.25 | 1464 | 1433-1471 | 0.69 |
| heavy-2048-csa-cp8r4 | cute_ws@compare | 93.5 | 90.3-94.8 | 0.12 | 1310 | 1280-1371 | 0.62 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 197.2 | 193.0-208.7 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r7 | tilelang@main | 793.2 | 765.5-859.1 | 1.00 | 2034 | 2029-2075 | 1.00 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 209.7 | 204.5-224.9 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r7 | tilelang@compare | 809.0 | 798.8-810.9 | 1.00 | 2141 | 2128-2155 | 1.00 |
| heavy-2048-csa-cp8r7 | cudnn_flashmla@compare | 309.9 | 306.5-318.9 | 0.38 | 1486 | 1480-1496 | 0.69 |
| heavy-2048-csa-cp8r7 | cute@compare | 207.7 | 206.5-211.4 | 0.26 | 1474 | 1430-1480 | 0.69 |
| heavy-2048-csa-cp8r7 | cute_ws@compare | 103.3 | 102.5-104.4 | 0.13 | 1300 | 1295-1323 | 0.61 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 205.1 | 204.5-213.3 | 0.25 | - | - | - |
| heavy-2048-hca-cp1 | tilelang@main | 1074 | 1069-1080 | 1.00 | 2464 | 2458-2470 | 1.00 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 335.2 | 329.4-337.8 | 0.31 | - | - | - |
| heavy-2048-hca-cp1 | tilelang@compare | 1081 | 1073-1086 | 1.00 | 2497 | 2474-2510 | 1.00 |
| heavy-2048-hca-cp1 | cudnn_flashmla@compare | 398.5 | 393.6-402.3 | 0.37 | 1545 | 1535-1549 | 0.62 |
| heavy-2048-hca-cp1 | cute@compare | 474.2 | 470.8-476.4 | 0.44 | 1850 | 1845-1866 | 0.74 |
| heavy-2048-hca-cp1 | cute_ws@compare | 214.2 | 213.8-215.1 | 0.20 | 1674 | 1660-1708 | 0.67 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@compare | 338.2 | 331.7-342.1 | 0.31 | - | - | - |
| heavy-2048-hca-cp8r0 | tilelang@main | 774.2 | 771.3-778.2 | 1.00 | 2135 | 2131-2143 | 1.00 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 189.3 | 188.5-195.6 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r0 | tilelang@compare | 775.8 | 773.3-801.1 | 1.00 | 2228 | 2220-2233 | 1.00 |
| heavy-2048-hca-cp8r0 | cudnn_flashmla@compare | 317.9 | 312.1-320.7 | 0.41 | 1466 | 1460-1471 | 0.66 |
| heavy-2048-hca-cp8r0 | cute@compare | 193.4 | 190.6-195.7 | 0.25 | 1560 | 1556-1564 | 0.70 |
| heavy-2048-hca-cp8r0 | cute_ws@compare | 76.9 | 76.5-78.0 | 0.10 | 1352 | 1339-1375 | 0.61 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 188.1 | 187.5-194.9 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r4 | tilelang@main | 782.0 | 771.9-791.9 | 1.00 | 2122 | 2120-2137 | 1.00 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 189.5 | 183.3-193.8 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r4 | tilelang@compare | 811.3 | 795.3-873.3 | 1.00 | 2234 | 2195-2333 | 1.00 |
| heavy-2048-hca-cp8r4 | cudnn_flashmla@compare | 321.4 | 320.5-330.5 | 0.40 | 1483 | 1474-1620 | 0.66 |
| heavy-2048-hca-cp8r4 | cute@compare | 204.4 | 204.2-206.7 | 0.25 | 1554 | 1540-1566 | 0.70 |
| heavy-2048-hca-cp8r4 | cute_ws@compare | 82.5 | 81.6-82.7 | 0.10 | 1351 | 1342-1363 | 0.60 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 195.0 | 192.8-199.5 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r7 | tilelang@main | 786.8 | 778.1-795.3 | 1.00 | 2157 | 2147-2182 | 1.00 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 189.8 | 187.0-192.2 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r7 | tilelang@compare | 790.1 | 776.2-798.3 | 1.00 | 2243 | 2219-2259 | 1.00 |
| heavy-2048-hca-cp8r7 | cudnn_flashmla@compare | 321.5 | 312.7-324.1 | 0.41 | 1476 | 1467-1490 | 0.66 |
| heavy-2048-hca-cp8r7 | cute@compare | 203.4 | 196.8-206.6 | 0.26 | 1563 | 1553-1572 | 0.70 |
| heavy-2048-hca-cp8r7 | cute_ws@compare | 83.1 | 80.8-85.5 | 0.11 | 1338 | 1326-1352 | 0.60 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 195.4 | 193.4-206.9 | 0.25 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang@main | 1000 | 986.0-1002 | 1.00 | 2294 | 2292-2303 | 1.00 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 291.8 | 289.1-294.8 | 0.29 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang@compare | 1005 | 1002-1011 | 1.00 | 2346 | 2343-2356 | 1.00 |
| heavy-2048-sliding-cp1 | cudnn_flashmla@compare | 354.6 | 350.7-358.6 | 0.35 | 1518 | 1514-1521 | 0.65 |
| heavy-2048-sliding-cp1 | cute@compare | 403.7 | 401.0-405.1 | 0.40 | 1706 | 1698-1707 | 0.73 |
| heavy-2048-sliding-cp1 | cute_ws@compare | 168.3 | 168.0-172.8 | 0.17 | 1556 | 1547-1577 | 0.66 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@compare | 290.1 | 285.5-293.8 | 0.29 | - | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@main | 736.0 | 727.2-829.0 | 1.00 | 2039 | 2024-2043 | 1.00 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 180.0 | 176.3-183.7 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@compare | 755.0 | 744.8-775.8 | 1.00 | 2127 | 2120-2174 | 1.00 |
| heavy-2048-sliding-cp8r0 | cudnn_flashmla@compare | 317.6 | 315.6-319.1 | 0.42 | 1474 | 1455-1510 | 0.69 |
| heavy-2048-sliding-cp8r0 | cute@compare | 170.2 | 167.6-172.0 | 0.23 | 1460 | 1439-1468 | 0.69 |
| heavy-2048-sliding-cp8r0 | cute_ws@compare | 75.0 | 74.1-77.2 | 0.10 | 1295 | 1281-1300 | 0.61 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 184.6 | 183.6-188.4 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@main | 742.3 | 731.2-750.7 | 1.00 | 2036 | 2019-2052 | 1.00 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 180.7 | 175.9-183.0 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@compare | 756.5 | 746.1-758.8 | 1.00 | 2120 | 2113-2182 | 1.00 |
| heavy-2048-sliding-cp8r4 | cudnn_flashmla@compare | 315.8 | 311.3-319.1 | 0.42 | 1457 | 1455-1467 | 0.69 |
| heavy-2048-sliding-cp8r4 | cute@compare | 175.3 | 173.6-176.6 | 0.23 | 1456 | 1451-1465 | 0.69 |
| heavy-2048-sliding-cp8r4 | cute_ws@compare | 76.0 | 73.7-78.1 | 0.10 | 1297 | 1285-1327 | 0.61 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 189.0 | 181.4-189.5 | 0.25 | - | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@main | 743.6 | 736.9-744.8 | 1.00 | 2071 | 2052-2104 | 1.00 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 182.7 | 175.3-184.9 | 0.25 | - | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@compare | 763.4 | 750.6-770.8 | 1.00 | 2120 | 2115-2124 | 1.00 |
| heavy-2048-sliding-cp8r7 | cudnn_flashmla@compare | 320.9 | 318.2-326.4 | 0.42 | 1462 | 1455-1473 | 0.69 |
| heavy-2048-sliding-cp8r7 | cute@compare | 174.0 | 173.3-179.9 | 0.23 | 1475 | 1462-1534 | 0.70 |
| heavy-2048-sliding-cp8r7 | cute_ws@compare | 76.7 | 75.6-76.8 | 0.10 | 1296 | 1291-1301 | 0.61 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 187.6 | 184.3-196.5 | 0.25 | - | - | - |
| tiny-2048-csa-cp1 | tilelang@main | 1043 | 1040-1056 | 1.00 | 2359 | 2333-2369 | 1.00 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 333.9 | 329.9-336.8 | 0.32 | - | - | - |
| tiny-2048-csa-cp1 | tilelang@compare | 1061 | 1049-1064 | 1.00 | 2420 | 2403-2434 | 1.00 |
| tiny-2048-csa-cp1 | cudnn_flashmla@compare | 395.2 | 393.4-398.7 | 0.37 | 1526 | 1522-1555 | 0.63 |
| tiny-2048-csa-cp1 | cute@compare | 451.6 | 451.0-453.9 | 0.43 | 1764 | 1744-1774 | 0.73 |
| tiny-2048-csa-cp1 | cute_ws@compare | 228.8 | 227.4-229.9 | 0.22 | 1594 | 1586-1621 | 0.66 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@compare | 336.6 | 330.9-337.8 | 0.32 | - | - | - |
| tiny-2048-csa-cp8r0 | tilelang@main | 753.9 | 747.6-789.7 | 1.00 | 2056 | 2032-2068 | 1.00 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 192.5 | 185.3-201.4 | 0.26 | - | - | - |
| tiny-2048-csa-cp8r0 | tilelang@compare | 766.9 | 749.6-778.1 | 1.00 | 2158 | 2121-2175 | 1.00 |
| tiny-2048-csa-cp8r0 | cudnn_flashmla@compare | 315.7 | 313.1-335.9 | 0.41 | 1475 | 1447-1476 | 0.68 |
| tiny-2048-csa-cp8r0 | cute@compare | 176.9 | 174.4-185.8 | 0.23 | 1448 | 1445-1459 | 0.67 |
| tiny-2048-csa-cp8r0 | cute_ws@compare | 88.3 | 86.5-92.5 | 0.12 | 1288 | 1286-1298 | 0.60 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 191.1 | 186.9-204.9 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r4 | tilelang@main | 748.5 | 741.9-755.9 | 1.00 | 2029 | 2015-2055 | 1.00 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 187.9 | 184.4-189.9 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r4 | tilelang@compare | 755.6 | 748.4-769.1 | 1.00 | 2128 | 2120-2213 | 1.00 |
| tiny-2048-csa-cp8r4 | cudnn_flashmla@compare | 313.4 | 310.6-318.5 | 0.41 | 1478 | 1468-1496 | 0.69 |
| tiny-2048-csa-cp8r4 | cute@compare | 176.3 | 174.8-177.9 | 0.23 | 1459 | 1452-1483 | 0.69 |
| tiny-2048-csa-cp8r4 | cute_ws@compare | 86.3 | 85.4-86.6 | 0.11 | 1294 | 1288-1318 | 0.61 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 192.1 | 188.2-194.6 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r7 | tilelang@main | 761.0 | 757.8-769.9 | 1.00 | 2040 | 2032-2079 | 1.00 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 189.9 | 186.1-198.6 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r7 | tilelang@compare | 762.1 | 750.5-790.1 | 1.00 | 2127 | 2113-2139 | 1.00 |
| tiny-2048-csa-cp8r7 | cudnn_flashmla@compare | 312.4 | 307.7-315.9 | 0.41 | 1478 | 1472-1481 | 0.69 |
| tiny-2048-csa-cp8r7 | cute@compare | 176.0 | 175.5-178.5 | 0.23 | 1462 | 1457-1464 | 0.69 |
| tiny-2048-csa-cp8r7 | cute_ws@compare | 86.4 | 85.1-88.7 | 0.11 | 1297 | 1292-1301 | 0.61 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 190.8 | 181.4-197.3 | 0.25 | - | - | - |
| tiny-2048-hca-cp1 | tilelang@main | 962.7 | 956.2-975.0 | 1.00 | 2093 | 2080-2128 | 1.00 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 292.2 | 290.6-305.2 | 0.30 | - | - | - |
| tiny-2048-hca-cp1 | tilelang@compare | 964.7 | 958.0-975.7 | 1.00 | 2147 | 2132-2159 | 1.00 |
| tiny-2048-hca-cp1 | cudnn_flashmla@compare | 363.7 | 356.8-366.8 | 0.38 | 1491 | 1485-1492 | 0.69 |
| tiny-2048-hca-cp1 | cute@compare | 375.1 | 369.7-379.8 | 0.39 | 1512 | 1495-1513 | 0.70 |
| tiny-2048-hca-cp1 | cute_ws@compare | 167.9 | 166.1-169.5 | 0.17 | 1358 | 1350-1373 | 0.63 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@compare | 291.2 | 286.9-293.4 | 0.30 | - | - | - |
| tiny-2048-hca-cp8r0 | tilelang@main | 736.5 | 733.4-737.7 | 1.00 | 2030 | 2013-2052 | 1.00 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 182.6 | 176.8-187.7 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r0 | tilelang@compare | 737.7 | 733.4-757.9 | 1.00 | 2118 | 2097-2137 | 1.00 |
| tiny-2048-hca-cp8r0 | cudnn_flashmla@compare | 313.4 | 310.0-316.6 | 0.42 | 1466 | 1442-1476 | 0.69 |
| tiny-2048-hca-cp8r0 | cute@compare | 168.0 | 166.1-170.9 | 0.23 | 1456 | 1453-1473 | 0.69 |
| tiny-2048-hca-cp8r0 | cute_ws@compare | 73.1 | 72.9-74.2 | 0.10 | 1293 | 1276-1296 | 0.61 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 181.8 | 174.2-188.8 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r4 | tilelang@main | 735.0 | 723.8-756.2 | 1.00 | 2049 | 2011-2074 | 1.00 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 180.9 | 179.2-183.4 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r4 | tilelang@compare | 742.6 | 737.0-749.5 | 1.00 | 2151 | 2139-2168 | 1.00 |
| tiny-2048-hca-cp8r4 | cudnn_flashmla@compare | 312.6 | 308.9-318.9 | 0.42 | 1499 | 1484-1508 | 0.70 |
| tiny-2048-hca-cp8r4 | cute@compare | 169.6 | 166.8-172.3 | 0.23 | 1477 | 1456-1493 | 0.69 |
| tiny-2048-hca-cp8r4 | cute_ws@compare | 75.3 | 73.9-78.4 | 0.10 | 1312 | 1291-1315 | 0.61 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 184.4 | 184.3-186.5 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r7 | tilelang@main | 751.2 | 738.8-819.8 | 1.00 | 2099 | 2044-2164 | 1.00 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 190.0 | 184.2-195.8 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r7 | tilelang@compare | 749.3 | 737.2-752.4 | 1.00 | 2108 | 2103-2127 | 1.00 |
| tiny-2048-hca-cp8r7 | cudnn_flashmla@compare | 319.4 | 316.2-323.0 | 0.43 | 1499 | 1459-1578 | 0.71 |
| tiny-2048-hca-cp8r7 | cute@compare | 172.0 | 171.1-176.1 | 0.23 | 1473 | 1467-1554 | 0.70 |
| tiny-2048-hca-cp8r7 | cute_ws@compare | 76.8 | 74.2-78.6 | 0.10 | 1297 | 1294-1304 | 0.62 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 184.4 | 183.0-188.8 | 0.25 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang@main | 957.8 | 952.0-962.9 | 1.00 | 2108 | 2103-2118 | 1.00 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 290.5 | 284.1-294.3 | 0.30 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang@compare | 963.8 | 949.2-969.2 | 1.00 | 2211 | 2175-2248 | 1.00 |
| tiny-2048-sliding-cp1 | cudnn_flashmla@compare | 359.1 | 354.2-360.4 | 0.37 | 1519 | 1496-1535 | 0.69 |
| tiny-2048-sliding-cp1 | cute@compare | 378.3 | 374.6-380.9 | 0.39 | 1543 | 1537-1551 | 0.70 |
| tiny-2048-sliding-cp1 | cute_ws@compare | 167.8 | 167.4-169.6 | 0.17 | 1384 | 1366-1435 | 0.63 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@compare | 295.9 | 292.2-298.8 | 0.31 | - | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@main | 731.7 | 730.4-739.6 | 1.00 | 2136 | 2058-2216 | 1.00 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 179.8 | 173.2-180.5 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@compare | 750.1 | 744.8-766.3 | 1.00 | 2175 | 2159-2227 | 1.00 |
| tiny-2048-sliding-cp8r0 | cudnn_flashmla@compare | 316.8 | 314.3-324.7 | 0.42 | 1507 | 1487-1545 | 0.69 |
| tiny-2048-sliding-cp8r0 | cute@compare | 168.2 | 165.3-169.2 | 0.22 | 1535 | 1488-1575 | 0.71 |
| tiny-2048-sliding-cp8r0 | cute_ws@compare | 74.2 | 73.3-75.2 | 0.10 | 1341 | 1322-1437 | 0.62 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 184.8 | 179.1-189.3 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@main | 738.5 | 727.1-742.4 | 1.00 | 2024 | 2010-2050 | 1.00 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 178.4 | 171.8-182.8 | 0.24 | - | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@compare | 754.5 | 740.6-852.0 | 1.00 | 2117 | 2112-2125 | 1.00 |
| tiny-2048-sliding-cp8r4 | cudnn_flashmla@compare | 312.7 | 310.7-317.7 | 0.41 | 1459 | 1449-1476 | 0.69 |
| tiny-2048-sliding-cp8r4 | cute@compare | 163.7 | 162.1-172.1 | 0.22 | 1456 | 1438-1459 | 0.69 |
| tiny-2048-sliding-cp8r4 | cute_ws@compare | 72.0 | 70.8-77.8 | 0.10 | 1290 | 1278-1324 | 0.61 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 177.0 | 173.9-185.8 | 0.23 | - | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@main | 758.1 | 748.7-774.6 | 1.00 | 2049 | 2038-2151 | 1.00 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 186.8 | 182.1-192.1 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@compare | 749.5 | 744.0-754.3 | 1.00 | 2114 | 2109-2124 | 1.00 |
| tiny-2048-sliding-cp8r7 | cudnn_flashmla@compare | 314.4 | 312.6-321.7 | 0.42 | 1464 | 1454-1466 | 0.69 |
| tiny-2048-sliding-cp8r7 | cute@compare | 170.4 | 169.8-172.0 | 0.23 | 1447 | 1433-1476 | 0.68 |
| tiny-2048-sliding-cp8r7 | cute_ws@compare | 74.0 | 73.2-75.6 | 0.10 | 1294 | 1286-1443 | 0.61 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 182.9 | 180.5-186.1 | 0.24 | - | - | - |
| single-4096-csa-cp1 | tilelang@main | 1890 | 1888-1954 | 1.00 | 5972 | 5965-5976 | 1.00 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 715.4 | 711.7-718.7 | 0.38 | - | - | - |
| single-4096-csa-cp1 | tilelang@compare | 1896 | 1885-1906 | 1.00 | 6015 | 6009-6026 | 1.00 |
| single-4096-csa-cp1 | cudnn_flashmla@compare | 781.5 | 778.6-785.9 | 0.41 | 3139 | 3129-3150 | 0.52 |
| single-4096-csa-cp1 | cute@compare | 1272 | 1265-1276 | 0.67 | 5331 | 5323-5334 | 0.89 |
| single-4096-csa-cp1 | cute_ws@compare | 604.4 | 602.2-609.7 | 0.32 | 4728 | 4710-4742 | 0.79 |
| single-4096-csa-cp1 | flashmla_fwd_ref@compare | 714.6 | 711.5-716.0 | 0.38 | - | - | - |
| single-4096-csa-cp8r0 | tilelang@main | 795.7 | 791.3-798.4 | 1.00 | 2037 | 2027-2047 | 1.00 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 206.3 | 203.2-212.2 | 0.26 | - | - | - |
| single-4096-csa-cp8r0 | tilelang@compare | 801.5 | 795.1-808.6 | 1.00 | 2106 | 2097-2133 | 1.00 |
| single-4096-csa-cp8r0 | cudnn_flashmla@compare | 313.2 | 307.6-327.9 | 0.39 | 1484 | 1468-1492 | 0.70 |
| single-4096-csa-cp8r0 | cute@compare | 224.1 | 223.3-228.9 | 0.28 | 1445 | 1442-1456 | 0.69 |
| single-4096-csa-cp8r0 | cute_ws@compare | 107.4 | 106.6-108.7 | 0.13 | 1285 | 1282-1292 | 0.61 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 211.1 | 207.9-216.0 | 0.26 | - | - | - |
| single-4096-csa-cp8r4 | tilelang@main | 869.0 | 862.6-883.1 | 1.00 | 2333 | 2321-2341 | 1.00 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 248.4 | 245.0-250.7 | 0.29 | - | - | - |
| single-4096-csa-cp8r4 | tilelang@compare | 890.8 | 873.8-899.3 | 1.00 | 2396 | 2385-2404 | 1.00 |
| single-4096-csa-cp8r4 | cudnn_flashmla@compare | 313.0 | 311.1-313.8 | 0.35 | 1475 | 1448-1475 | 0.62 |
| single-4096-csa-cp8r4 | cute@compare | 304.0 | 303.4-308.1 | 0.34 | 1758 | 1740-1770 | 0.73 |
| single-4096-csa-cp8r4 | cute_ws@compare | 149.5 | 147.6-150.1 | 0.17 | 1593 | 1585-1601 | 0.66 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 253.2 | 249.6-255.8 | 0.28 | - | - | - |
| single-4096-csa-cp8r7 | tilelang@main | 880.8 | 872.3-893.7 | 1.00 | 2349 | 2340-2358 | 1.00 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 252.0 | 246.9-253.2 | 0.29 | - | - | - |
| single-4096-csa-cp8r7 | tilelang@compare | 877.2 | 874.0-884.6 | 1.00 | 2435 | 2429-2449 | 1.00 |
| single-4096-csa-cp8r7 | cudnn_flashmla@compare | 311.9 | 310.8-317.9 | 0.36 | 1489 | 1475-1498 | 0.61 |
| single-4096-csa-cp8r7 | cute@compare | 302.8 | 301.3-303.0 | 0.35 | 1767 | 1755-1769 | 0.73 |
| single-4096-csa-cp8r7 | cute_ws@compare | 146.8 | 146.1-151.0 | 0.17 | 1610 | 1605-1624 | 0.66 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 252.4 | 250.1-254.6 | 0.29 | - | - | - |
| single-4096-hca-cp1 | tilelang@main | 1432 | 1411-1475 | 1.00 | 3134 | 3126-3142 | 1.00 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 505.5 | 499.0-532.2 | 0.35 | - | - | - |
| single-4096-hca-cp1 | tilelang@compare | 1428 | 1426-1452 | 1.00 | 3195 | 3176-3218 | 1.00 |
| single-4096-hca-cp1 | cudnn_flashmla@compare | 570.2 | 567.6-574.4 | 0.40 | 2114 | 2100-2120 | 0.66 |
| single-4096-hca-cp1 | cute@compare | 791.2 | 788.1-793.1 | 0.55 | 2551 | 2549-2576 | 0.80 |
| single-4096-hca-cp1 | cute_ws@compare | 372.0 | 370.8-374.1 | 0.26 | 2358 | 2348-2366 | 0.74 |
| single-4096-hca-cp1 | flashmla_fwd_ref@compare | 505.8 | 504.5-508.2 | 0.35 | - | - | - |
| single-4096-hca-cp8r0 | tilelang@main | 828.2 | 816.4-834.6 | 1.00 | 2126 | 2122-2148 | 1.00 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 217.1 | 213.7-217.8 | 0.26 | - | - | - |
| single-4096-hca-cp8r0 | tilelang@compare | 843.0 | 836.2-849.6 | 1.00 | 2245 | 2232-2260 | 1.00 |
| single-4096-hca-cp8r0 | cudnn_flashmla@compare | 323.6 | 321.2-329.9 | 0.38 | 1486 | 1481-1504 | 0.66 |
| single-4096-hca-cp8r0 | cute@compare | 242.0 | 240.1-244.8 | 0.29 | 1582 | 1579-1591 | 0.70 |
| single-4096-hca-cp8r0 | cute_ws@compare | 100.4 | 99.6-101.7 | 0.12 | 1370 | 1360-1380 | 0.61 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 221.2 | 212.8-224.1 | 0.26 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 824.2 | 816.0-832.5 | 1.00 | 2143 | 2120-2164 | 1.00 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 216.8 | 214.5-219.7 | 0.26 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@compare | 914.6 | 885.6-949.6 | 1.00 | 2251 | 2247-2257 | 1.00 |
| single-4096-hca-cp8r4 | cudnn_flashmla@compare | 328.7 | 319.0-338.8 | 0.36 | 1489 | 1483-1492 | 0.66 |
| single-4096-hca-cp8r4 | cute@compare | 252.8 | 248.1-254.3 | 0.28 | 1571 | 1567-1588 | 0.70 |
| single-4096-hca-cp8r4 | cute_ws@compare | 105.5 | 102.7-107.6 | 0.12 | 1355 | 1350-1370 | 0.60 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 226.9 | 218.6-228.8 | 0.25 | - | - | - |
| single-4096-hca-cp8r7 | tilelang@main | 827.1 | 813.1-838.7 | 1.00 | 2159 | 2139-2187 | 1.00 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 217.2 | 212.4-222.6 | 0.26 | - | - | - |
| single-4096-hca-cp8r7 | tilelang@compare | 845.0 | 841.8-854.4 | 1.00 | 2312 | 2219-2746 | 1.00 |
| single-4096-hca-cp8r7 | cudnn_flashmla@compare | 320.2 | 311.7-340.7 | 0.38 | 1525 | 1466-1727 | 0.66 |
| single-4096-hca-cp8r7 | cute@compare | 245.7 | 245.5-248.1 | 0.29 | 1587 | 1562-1802 | 0.69 |
| single-4096-hca-cp8r7 | cute_ws@compare | 103.5 | 102.6-105.5 | 0.12 | 1349 | 1339-1564 | 0.58 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 219.4 | 213.6-222.7 | 0.26 | - | - | - |
| single-4096-sliding-cp1 | tilelang@main | 1297 | 1285-1352 | 1.00 | 2859 | 2858-2862 | 1.00 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 420.2 | 412.8-423.1 | 0.32 | - | - | - |
| single-4096-sliding-cp1 | tilelang@compare | 1338 | 1299-1366 | 1.00 | 2939 | 2930-2946 | 1.00 |
| single-4096-sliding-cp1 | cudnn_flashmla@compare | 484.8 | 478.2-492.7 | 0.36 | 1993 | 1990-2003 | 0.68 |
| single-4096-sliding-cp1 | cute@compare | 671.5 | 670.7-674.1 | 0.50 | 2278 | 2258-2307 | 0.78 |
| single-4096-sliding-cp1 | cute_ws@compare | 269.7 | 268.3-276.4 | 0.20 | 2116 | 2106-2136 | 0.72 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@compare | 422.1 | 417.8-429.1 | 0.32 | - | - | - |
| single-4096-sliding-cp8r0 | tilelang@main | 790.1 | 775.7-812.3 | 1.00 | 2034 | 2027-2045 | 1.00 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 199.3 | 197.8-208.9 | 0.25 | - | - | - |
| single-4096-sliding-cp8r0 | tilelang@compare | 796.2 | 783.5-810.4 | 1.00 | 2135 | 2128-2148 | 1.00 |
| single-4096-sliding-cp8r0 | cudnn_flashmla@compare | 308.4 | 307.1-318.7 | 0.39 | 1478 | 1474-1492 | 0.69 |
| single-4096-sliding-cp8r0 | cute@compare | 207.8 | 206.0-209.1 | 0.26 | 1478 | 1454-1503 | 0.69 |
| single-4096-sliding-cp8r0 | cute_ws@compare | 88.8 | 88.1-89.1 | 0.11 | 1301 | 1292-1321 | 0.61 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 205.0 | 200.0-207.3 | 0.26 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 791.1 | 779.7-892.0 | 1.00 | 2057 | 2032-2070 | 1.00 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 197.7-217.5 | 0.26 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@compare | 796.9 | 785.8-818.8 | 1.00 | 2119 | 2101-2129 | 1.00 |
| single-4096-sliding-cp8r4 | cudnn_flashmla@compare | 316.4 | 310.6-317.2 | 0.40 | 1461 | 1454-1507 | 0.69 |
| single-4096-sliding-cp8r4 | cute@compare | 208.5 | 205.9-209.4 | 0.26 | 1460 | 1437-1465 | 0.69 |
| single-4096-sliding-cp8r4 | cute_ws@compare | 87.3 | 86.6-88.6 | 0.11 | 1278 | 1272-1285 | 0.60 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 200.1 | 192.3-205.9 | 0.25 | - | - | - |
| single-4096-sliding-cp8r7 | tilelang@main | 785.2 | 769.9-787.5 | 1.00 | 2065 | 2049-2079 | 1.00 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 194.0 | 192.7-204.3 | 0.25 | - | - | - |
| single-4096-sliding-cp8r7 | tilelang@compare | 778.7 | 776.4-792.1 | 1.00 | 2116 | 2107-2126 | 1.00 |
| single-4096-sliding-cp8r7 | cudnn_flashmla@compare | 312.4 | 305.2-319.1 | 0.40 | 1462 | 1454-1465 | 0.69 |
| single-4096-sliding-cp8r7 | cute@compare | 207.6 | 206.8-213.6 | 0.27 | 1452 | 1448-1466 | 0.69 |
| single-4096-sliding-cp8r7 | cute_ws@compare | 88.3 | 87.4-89.3 | 0.11 | 1285 | 1276-1296 | 0.61 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 200.0 | 194.8-203.7 | 0.26 | - | - | - |
| short-4096-csa-cp1 | tilelang@main | 1523 | 1513-1536 | 1.00 | 3873 | 3871-3875 | 1.00 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 552.9 | 546.9-555.2 | 0.36 | - | - | - |
| short-4096-csa-cp1 | tilelang@compare | 1510 | 1505-1511 | 1.00 | 3902 | 3893-3908 | 1.00 |
| short-4096-csa-cp1 | cudnn_flashmla@compare | 614.0 | 608.1-616.3 | 0.41 | 2363 | 2360-2371 | 0.61 |
| short-4096-csa-cp1 | cute@compare | 898.8 | 893.9-908.8 | 0.60 | 3228 | 3228-3231 | 0.83 |
| short-4096-csa-cp1 | cute_ws@compare | 428.0 | 426.6-429.3 | 0.28 | 2997 | 2979-2999 | 0.77 |
| short-4096-csa-cp1 | flashmla_fwd_ref@compare | 555.1 | 551.3-559.8 | 0.37 | - | - | - |
| short-4096-csa-cp8r0 | tilelang@main | 827.7 | 802.6-859.4 | 1.00 | 2054 | 2046-2075 | 1.00 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 222.4 | 213.5-232.8 | 0.27 | - | - | - |
| short-4096-csa-cp8r0 | tilelang@compare | 808.2 | 802.1-838.8 | 1.00 | 2096 | 2085-2122 | 1.00 |
| short-4096-csa-cp8r0 | cudnn_flashmla@compare | 316.5 | 310.2-322.3 | 0.39 | 1456 | 1455-1472 | 0.69 |
| short-4096-csa-cp8r0 | cute@compare | 228.3 | 225.0-229.5 | 0.28 | 1439 | 1430-1463 | 0.69 |
| short-4096-csa-cp8r0 | cute_ws@compare | 108.7 | 107.1-109.2 | 0.13 | 1281 | 1278-1299 | 0.61 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 211.1 | 209.3-224.7 | 0.26 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 876.9 | 821.4-901.4 | 1.00 | 2145 | 2070-2654 | 1.00 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 227.1 | 220.9-229.2 | 0.26 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@compare | 827.9 | 806.3-831.9 | 1.00 | 2114 | 2088-2169 | 1.00 |
| short-4096-csa-cp8r4 | cudnn_flashmla@compare | 307.4 | 305.3-316.3 | 0.37 | 1466 | 1450-1475 | 0.69 |
| short-4096-csa-cp8r4 | cute@compare | 228.4 | 225.2-230.0 | 0.28 | 1451 | 1431-1471 | 0.69 |
| short-4096-csa-cp8r4 | cute_ws@compare | 110.8 | 109.1-113.2 | 0.13 | 1288 | 1270-1295 | 0.61 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 215.3 | 210.8-216.8 | 0.26 | - | - | - |
| short-4096-csa-cp8r7 | tilelang@main | 818.1 | 806.9-826.3 | 1.00 | 2060 | 2047-2075 | 1.00 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 210.7 | 208.5-216.6 | 0.26 | - | - | - |
| short-4096-csa-cp8r7 | tilelang@compare | 795.3 | 790.4-808.4 | 1.00 | 2094 | 2088-2101 | 1.00 |
| short-4096-csa-cp8r7 | cudnn_flashmla@compare | 305.3 | 304.3-314.0 | 0.38 | 1469 | 1463-1472 | 0.70 |
| short-4096-csa-cp8r7 | cute@compare | 226.5 | 222.5-228.9 | 0.28 | 1447 | 1433-1467 | 0.69 |
| short-4096-csa-cp8r7 | cute_ws@compare | 108.4 | 107.7-110.3 | 0.14 | 1299 | 1283-1303 | 0.62 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 210.7 | 208.4-214.2 | 0.26 | - | - | - |
| short-4096-hca-cp1 | tilelang@main | 1411 | 1404-1416 | 1.00 | 3089 | 3082-3102 | 1.00 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 500.1 | 497.7-502.9 | 0.35 | - | - | - |
| short-4096-hca-cp1 | tilelang@compare | 1395 | 1386-1423 | 1.00 | 3088 | 3071-3523 | 1.00 |
| short-4096-hca-cp1 | cudnn_flashmla@compare | 562.9 | 556.3-566.3 | 0.40 | 2040 | 2014-2087 | 0.66 |
| short-4096-hca-cp1 | cute@compare | 764.2 | 763.0-771.3 | 0.55 | 2456 | 2446-2634 | 0.80 |
| short-4096-hca-cp1 | cute_ws@compare | 358.7 | 356.6-359.7 | 0.26 | 2271 | 2263-2427 | 0.74 |
| short-4096-hca-cp1 | flashmla_fwd_ref@compare | 501.4 | 495.4-502.9 | 0.36 | - | - | - |
| short-4096-hca-cp8r0 | tilelang@main | 844.5 | 832.2-863.5 | 1.00 | 2180 | 2165-2244 | 1.00 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 216.0 | 211.8-217.3 | 0.26 | - | - | - |
| short-4096-hca-cp8r0 | tilelang@compare | 833.6 | 822.4-844.8 | 1.00 | 2210 | 2191-2215 | 1.00 |
| short-4096-hca-cp8r0 | cudnn_flashmla@compare | 317.8 | 313.6-322.3 | 0.38 | 1475 | 1461-1506 | 0.67 |
| short-4096-hca-cp8r0 | cute@compare | 242.2 | 240.1-245.1 | 0.29 | 1551 | 1544-1566 | 0.70 |
| short-4096-hca-cp8r0 | cute_ws@compare | 99.6 | 98.9-100.1 | 0.12 | 1344 | 1338-1351 | 0.61 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 217.3 | 213.3-221.2 | 0.26 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 826.5 | 818.9-845.1 | 1.00 | 2157 | 2154-2171 | 1.00 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 208.2 | 206.1-217.1 | 0.25 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@compare | 828.4 | 813.7-839.3 | 1.00 | 2208 | 2196-2222 | 1.00 |
| short-4096-hca-cp8r4 | cudnn_flashmla@compare | 316.3 | 306.5-317.3 | 0.38 | 1476 | 1465-1479 | 0.67 |
| short-4096-hca-cp8r4 | cute@compare | 239.9 | 236.7-244.6 | 0.29 | 1547 | 1543-1558 | 0.70 |
| short-4096-hca-cp8r4 | cute_ws@compare | 99.2 | 98.3-100.7 | 0.12 | 1327 | 1319-1332 | 0.60 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 213.7 | 209.5-218.3 | 0.26 | - | - | - |
| short-4096-hca-cp8r7 | tilelang@main | 831.1 | 816.4-835.0 | 1.00 | 2197 | 2167-2202 | 1.00 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 214.2 | 209.4-216.6 | 0.26 | - | - | - |
| short-4096-hca-cp8r7 | tilelang@compare | 829.7 | 824.0-854.8 | 1.00 | 2197 | 2185-2203 | 1.00 |
| short-4096-hca-cp8r7 | cudnn_flashmla@compare | 319.8 | 312.7-324.2 | 0.39 | 1453 | 1448-1470 | 0.66 |
| short-4096-hca-cp8r7 | cute@compare | 241.9 | 239.2-244.9 | 0.29 | 1551 | 1528-1557 | 0.71 |
| short-4096-hca-cp8r7 | cute_ws@compare | 99.9 | 99.4-100.9 | 0.12 | 1351 | 1333-1405 | 0.61 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 220.3 | 210.8-221.3 | 0.27 | - | - | - |
| short-4096-sliding-cp1 | tilelang@main | 1312 | 1272-1315 | 1.00 | 2842 | 2832-2846 | 1.00 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 423.2 | 414.6-434.4 | 0.32 | - | - | - |
| short-4096-sliding-cp1 | tilelang@compare | 1288 | 1279-1296 | 1.00 | 2871 | 2862-2878 | 1.00 |
| short-4096-sliding-cp1 | cudnn_flashmla@compare | 484.4 | 477.9-486.9 | 0.38 | 1983 | 1970-1987 | 0.69 |
| short-4096-sliding-cp1 | cute@compare | 665.7 | 665.0-667.5 | 0.52 | 2229 | 2223-2234 | 0.78 |
| short-4096-sliding-cp1 | cute_ws@compare | 272.0 | 270.8-272.3 | 0.21 | 2074 | 2065-2079 | 0.72 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@compare | 414.5 | 413.9-418.6 | 0.32 | - | - | - |
| short-4096-sliding-cp8r0 | tilelang@main | 793.8 | 787.3-805.0 | 1.00 | 2036 | 2030-2048 | 1.00 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 197.4 | 192.0-199.4 | 0.25 | - | - | - |
| short-4096-sliding-cp8r0 | tilelang@compare | 789.3 | 784.1-796.7 | 1.00 | 2130 | 2115-2164 | 1.00 |
| short-4096-sliding-cp8r0 | cudnn_flashmla@compare | 311.4 | 308.9-323.1 | 0.39 | 1483 | 1470-1488 | 0.70 |
| short-4096-sliding-cp8r0 | cute@compare | 205.7 | 200.4-206.6 | 0.26 | 1462 | 1460-1464 | 0.69 |
| short-4096-sliding-cp8r0 | cute_ws@compare | 89.0 | 87.3-90.0 | 0.11 | 1299 | 1287-1306 | 0.61 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 203.4 | 199.6-205.7 | 0.26 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 794.0 | 780.5-815.6 | 1.00 | 2041 | 2033-2066 | 1.00 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 197.7 | 193.1-216.8 | 0.25 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@compare | 794.4 | 785.9-802.2 | 1.00 | 2122 | 2116-2147 | 1.00 |
| short-4096-sliding-cp8r4 | cudnn_flashmla@compare | 315.4 | 312.5-319.1 | 0.40 | 1486 | 1469-1498 | 0.70 |
| short-4096-sliding-cp8r4 | cute@compare | 209.9 | 207.6-211.8 | 0.26 | 1470 | 1462-1471 | 0.69 |
| short-4096-sliding-cp8r4 | cute_ws@compare | 90.4 | 89.4-91.9 | 0.11 | 1307 | 1297-1317 | 0.62 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 203.5 | 201.0-206.5 | 0.26 | - | - | - |
| short-4096-sliding-cp8r7 | tilelang@main | 774.8 | 766.8-785.9 | 1.00 | 2086 | 2079-2120 | 1.00 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 195.7 | 192.6-198.1 | 0.25 | - | - | - |
| short-4096-sliding-cp8r7 | tilelang@compare | 797.7 | 787.0-809.3 | 1.00 | 2138 | 2116-2163 | 1.00 |
| short-4096-sliding-cp8r7 | cudnn_flashmla@compare | 317.1 | 307.9-319.5 | 0.40 | 1472 | 1457-1493 | 0.69 |
| short-4096-sliding-cp8r7 | cute@compare | 208.0 | 205.9-208.7 | 0.26 | 1468 | 1447-1472 | 0.69 |
| short-4096-sliding-cp8r7 | cute_ws@compare | 88.6 | 86.8-91.2 | 0.11 | 1296 | 1294-1406 | 0.61 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 198.0 | 197.2-203.9 | 0.25 | - | - | - |
| heavy-4096-csa-cp1 | tilelang@main | 1462 | 1446-1472 | 1.00 | 3545 | 3535-3555 | 1.00 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 524.9 | 521.9-530.8 | 0.36 | - | - | - |
| heavy-4096-csa-cp1 | tilelang@compare | 1475 | 1466-1494 | 1.00 | 3600 | 3576-3606 | 1.00 |
| heavy-4096-csa-cp1 | cudnn_flashmla@compare | 597.8 | 596.8-602.1 | 0.41 | 2283 | 2262-2309 | 0.63 |
| heavy-4096-csa-cp1 | cute@compare | 843.5 | 842.1-845.0 | 0.57 | 2903 | 2898-2910 | 0.81 |
| heavy-4096-csa-cp1 | cute_ws@compare | 410.9 | 409.4-413.3 | 0.28 | 2724 | 2721-2739 | 0.76 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@compare | 530.5 | 527.6-532.4 | 0.36 | - | - | - |
| heavy-4096-csa-cp8r0 | tilelang@main | 794.4 | 789.7-815.2 | 1.00 | 2033 | 2023-2050 | 1.00 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 208.3 | 203.7-215.7 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r0 | tilelang@compare | 811.4 | 804.5-827.8 | 1.00 | 2129 | 2127-2163 | 1.00 |
| heavy-4096-csa-cp8r0 | cudnn_flashmla@compare | 317.2 | 306.7-326.2 | 0.39 | 1488 | 1483-1493 | 0.70 |
| heavy-4096-csa-cp8r0 | cute@compare | 227.3 | 223.1-227.5 | 0.28 | 1453 | 1444-1475 | 0.68 |
| heavy-4096-csa-cp8r0 | cute_ws@compare | 108.1 | 106.8-108.9 | 0.13 | 1314 | 1290-1368 | 0.62 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 211.7 | 208.9-218.6 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 817.9 | 805.9-831.3 | 1.00 | 2049 | 2038-2194 | 1.00 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 217.3 | 211.7-218.4 | 0.27 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@compare | 813.5 | 806.3-816.2 | 1.00 | 2116 | 2106-2132 | 1.00 |
| heavy-4096-csa-cp8r4 | cudnn_flashmla@compare | 315.0 | 310.1-336.0 | 0.39 | 1484 | 1473-1493 | 0.70 |
| heavy-4096-csa-cp8r4 | cute@compare | 228.7 | 224.6-236.6 | 0.28 | 1458 | 1453-1577 | 0.69 |
| heavy-4096-csa-cp8r4 | cute_ws@compare | 110.9 | 109.5-115.8 | 0.14 | 1303 | 1295-1317 | 0.62 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 218.3 | 213.4-219.6 | 0.27 | - | - | - |
| heavy-4096-csa-cp8r7 | tilelang@main | 849.9 | 811.7-856.0 | 1.00 | 2089 | 2057-2341 | 1.00 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 215.8 | 214.5-218.0 | 0.25 | - | - | - |
| heavy-4096-csa-cp8r7 | tilelang@compare | 799.3 | 795.7-887.2 | 1.00 | 2138 | 2112-2155 | 1.00 |
| heavy-4096-csa-cp8r7 | cudnn_flashmla@compare | 319.4 | 305.5-338.8 | 0.40 | 1505 | 1502-1509 | 0.70 |
| heavy-4096-csa-cp8r7 | cute@compare | 221.2 | 215.7-225.5 | 0.28 | 1477 | 1468-1486 | 0.69 |
| heavy-4096-csa-cp8r7 | cute_ws@compare | 108.7 | 107.7-109.4 | 0.14 | 1310 | 1306-1317 | 0.61 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 211.9 | 206.5-216.4 | 0.27 | - | - | - |
| heavy-4096-hca-cp1 | tilelang@main | 1375 | 1372-1386 | 1.00 | 2982 | 2967-2988 | 1.00 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 482.7 | 480.6-489.0 | 0.35 | - | - | - |
| heavy-4096-hca-cp1 | tilelang@compare | 1417 | 1392-1443 | 1.00 | 3026 | 3016-3045 | 1.00 |
| heavy-4096-hca-cp1 | cudnn_flashmla@compare | 552.5 | 548.1-566.9 | 0.39 | 2022 | 2020-2030 | 0.67 |
| heavy-4096-hca-cp1 | cute@compare | 749.8 | 747.0-751.6 | 0.53 | 2387 | 2379-2391 | 0.79 |
| heavy-4096-hca-cp1 | cute_ws@compare | 343.6 | 342.2-344.4 | 0.24 | 2193 | 2181-2197 | 0.72 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@compare | 485.0 | 484.1-488.4 | 0.34 | - | - | - |
| heavy-4096-hca-cp8r0 | tilelang@main | 835.5 | 828.0-844.7 | 1.00 | 2153 | 2137-2162 | 1.00 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 214.5 | 211.6-218.6 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r0 | tilelang@compare | 839.7 | 835.0-850.1 | 1.00 | 2243 | 2235-2261 | 1.00 |
| heavy-4096-hca-cp8r0 | cudnn_flashmla@compare | 324.4 | 317.6-325.8 | 0.39 | 1497 | 1497-1499 | 0.67 |
| heavy-4096-hca-cp8r0 | cute@compare | 242.1 | 240.1-245.7 | 0.29 | 1570 | 1561-1574 | 0.70 |
| heavy-4096-hca-cp8r0 | cute_ws@compare | 97.9 | 97.4-99.0 | 0.12 | 1359 | 1340-1370 | 0.61 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 216.6 | 211.4-220.0 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 827.9 | 820.9-840.8 | 1.00 | 2160 | 2157-2177 | 1.00 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 207.4 | 205.8-212.8 | 0.25 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@compare | 821.1 | 815.5-823.2 | 1.00 | 2251 | 2234-2255 | 1.00 |
| heavy-4096-hca-cp8r4 | cudnn_flashmla@compare | 321.7 | 317.4-324.6 | 0.39 | 1504 | 1498-1526 | 0.67 |
| heavy-4096-hca-cp8r4 | cute@compare | 245.0 | 240.2-246.6 | 0.30 | 1592 | 1568-1597 | 0.71 |
| heavy-4096-hca-cp8r4 | cute_ws@compare | 99.4 | 99.1-100.4 | 0.12 | 1365 | 1349-1407 | 0.61 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 214.3 | 209.2-217.3 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r7 | tilelang@main | 838.2 | 821.0-847.4 | 1.00 | 2180 | 2177-2181 | 1.00 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 215.7 | 212.8-219.9 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r7 | tilelang@compare | 820.8 | 816.8-837.8 | 1.00 | 2207 | 2202-2247 | 1.00 |
| heavy-4096-hca-cp8r7 | cudnn_flashmla@compare | 314.4 | 313.9-322.4 | 0.38 | 1482 | 1481-1485 | 0.67 |
| heavy-4096-hca-cp8r7 | cute@compare | 236.2 | 233.3-240.1 | 0.29 | 1564 | 1560-1567 | 0.71 |
| heavy-4096-hca-cp8r7 | cute_ws@compare | 95.7 | 94.2-96.3 | 0.12 | 1355 | 1338-1377 | 0.61 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 209.5 | 208.6-220.6 | 0.26 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang@main | 1283 | 1276-1286 | 1.00 | 2911 | 2843-2955 | 1.00 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 418.8 | 418.5-423.0 | 0.33 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang@compare | 1277 | 1268-1286 | 1.00 | 2808 | 2802-2816 | 1.00 |
| heavy-4096-sliding-cp1 | cudnn_flashmla@compare | 484.0 | 482.0-488.6 | 0.38 | 1932 | 1925-1946 | 0.69 |
| heavy-4096-sliding-cp1 | cute@compare | 660.7 | 659.2-666.8 | 0.52 | 2158 | 2148-2161 | 0.77 |
| heavy-4096-sliding-cp1 | cute_ws@compare | 271.8 | 270.1-274.5 | 0.21 | 2008 | 2002-2035 | 0.71 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@compare | 417.6 | 416.3-422.2 | 0.33 | - | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@main | 793.9 | 787.1-802.0 | 1.00 | 2083 | 2065-2088 | 1.00 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 198.7 | 195.6-205.1 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@compare | 779.3 | 763.8-791.7 | 1.00 | 2122 | 2105-2132 | 1.00 |
| heavy-4096-sliding-cp8r0 | cudnn_flashmla@compare | 309.9 | 307.1-313.4 | 0.40 | 1462 | 1459-1474 | 0.69 |
| heavy-4096-sliding-cp8r0 | cute@compare | 203.8 | 203.2-205.5 | 0.26 | 1447 | 1443-1449 | 0.68 |
| heavy-4096-sliding-cp8r0 | cute_ws@compare | 88.7 | 86.7-89.3 | 0.11 | 1287 | 1282-1299 | 0.61 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 199.4 | 196.7-201.2 | 0.26 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 780.6 | 767.1-784.0 | 1.00 | 2050 | 2037-2053 | 1.00 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 194.3 | 190.0-200.3 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@compare | 780.7 | 772.1-782.1 | 1.00 | 2088 | 2076-2104 | 1.00 |
| heavy-4096-sliding-cp8r4 | cudnn_flashmla@compare | 310.5 | 304.7-318.8 | 0.40 | 1443 | 1437-1462 | 0.69 |
| heavy-4096-sliding-cp8r4 | cute@compare | 205.2 | 201.5-206.7 | 0.26 | 1429 | 1420-1458 | 0.68 |
| heavy-4096-sliding-cp8r4 | cute_ws@compare | 86.6 | 85.5-87.2 | 0.11 | 1297 | 1282-1302 | 0.62 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 196.1 | 192.3-197.5 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@main | 772.0 | 762.3-785.7 | 1.00 | 2099 | 2076-2132 | 1.00 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 197.5 | 194.8-199.0 | 0.26 | - | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@compare | 787.1 | 779.0-791.5 | 1.00 | 2108 | 2080-2119 | 1.00 |
| heavy-4096-sliding-cp8r7 | cudnn_flashmla@compare | 314.5 | 307.8-321.6 | 0.40 | 1456 | 1447-1462 | 0.69 |
| heavy-4096-sliding-cp8r7 | cute@compare | 205.0 | 201.6-208.5 | 0.26 | 1462 | 1441-1474 | 0.69 |
| heavy-4096-sliding-cp8r7 | cute_ws@compare | 90.1 | 87.8-91.2 | 0.11 | 1280 | 1276-1295 | 0.61 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 201.5 | 197.7-206.0 | 0.26 | - | - | - |
| tiny-4096-csa-cp1 | tilelang@main | 1374 | 1368-1389 | 1.00 | 2919 | 2904-2930 | 1.00 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 501.5 | 498.9-506.4 | 0.37 | - | - | - |
| tiny-4096-csa-cp1 | tilelang@compare | 1354 | 1350-1373 | 1.00 | 2961 | 2941-2970 | 1.00 |
| tiny-4096-csa-cp1 | cudnn_flashmla@compare | 552.9 | 547.5-562.3 | 0.41 | 1975 | 1962-1981 | 0.67 |
| tiny-4096-csa-cp1 | cute@compare | 749.7 | 746.4-754.3 | 0.55 | 2301 | 2291-2324 | 0.78 |
| tiny-4096-csa-cp1 | cute_ws@compare | 380.8 | 379.0-382.7 | 0.28 | 2165 | 2149-2171 | 0.73 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@compare | 490.6 | 488.1-495.4 | 0.36 | - | - | - |
| tiny-4096-csa-cp8r0 | tilelang@main | 796.5 | 784.8-820.9 | 1.00 | 2020 | 2015-2135 | 1.00 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 210.9 | 210.1-221.9 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r0 | tilelang@compare | 802.1 | 793.9-810.7 | 1.00 | 2101 | 2090-2124 | 1.00 |
| tiny-4096-csa-cp8r0 | cudnn_flashmla@compare | 309.6 | 306.5-310.8 | 0.39 | 1472 | 1470-1484 | 0.70 |
| tiny-4096-csa-cp8r0 | cute@compare | 217.5 | 216.3-220.4 | 0.27 | 1449 | 1439-1464 | 0.69 |
| tiny-4096-csa-cp8r0 | cute_ws@compare | 108.1 | 107.3-108.8 | 0.13 | 1289 | 1286-1307 | 0.61 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 210.1 | 209.3-214.5 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 809.3 | 793.1-830.6 | 1.00 | 2034 | 2014-2048 | 1.00 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 209.6 | 207.2-216.1 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@compare | 811.5 | 800.0-818.7 | 1.00 | 2103 | 2083-2115 | 1.00 |
| tiny-4096-csa-cp8r4 | cudnn_flashmla@compare | 306.9 | 306.0-311.9 | 0.38 | 1461 | 1455-1471 | 0.69 |
| tiny-4096-csa-cp8r4 | cute@compare | 218.8 | 213.9-220.9 | 0.27 | 1435 | 1431-1453 | 0.68 |
| tiny-4096-csa-cp8r4 | cute_ws@compare | 107.5 | 107.0-108.6 | 0.13 | 1289 | 1280-1298 | 0.61 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 211.4 | 207.6-215.4 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r7 | tilelang@main | 786.0 | 777.7-790.5 | 1.00 | 2043 | 2030-2080 | 1.00 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 210.6 | 206.3-211.8 | 0.27 | - | - | - |
| tiny-4096-csa-cp8r7 | tilelang@compare | 799.8 | 796.3-802.7 | 1.00 | 2140 | 2106-2165 | 1.00 |
| tiny-4096-csa-cp8r7 | cudnn_flashmla@compare | 308.9 | 307.2-315.5 | 0.39 | 1460 | 1454-1487 | 0.68 |
| tiny-4096-csa-cp8r7 | cute@compare | 218.7 | 216.6-220.7 | 0.27 | 1458 | 1439-1469 | 0.68 |
| tiny-4096-csa-cp8r7 | cute_ws@compare | 107.0 | 106.5-108.0 | 0.13 | 1302 | 1295-1413 | 0.61 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 208.6 | 204.6-214.0 | 0.26 | - | - | - |
| tiny-4096-hca-cp1 | tilelang@main | 1227 | 1218-1246 | 1.00 | 2409 | 2400-2413 | 1.00 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 412.3 | 408.0-430.9 | 0.34 | - | - | - |
| tiny-4096-hca-cp1 | tilelang@compare | 1226 | 1216-1239 | 1.00 | 2444 | 2418-2452 | 1.00 |
| tiny-4096-hca-cp1 | cudnn_flashmla@compare | 478.0 | 473.2-483.6 | 0.39 | 1767 | 1763-1776 | 0.72 |
| tiny-4096-hca-cp1 | cute@compare | 610.3 | 609.7-615.3 | 0.50 | 1812 | 1807-1817 | 0.74 |
| tiny-4096-hca-cp1 | cute_ws@compare | 280.8 | 270.5-292.8 | 0.23 | 1652 | 1651-1664 | 0.68 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@compare | 420.0 | 414.8-423.4 | 0.34 | - | - | - |
| tiny-4096-hca-cp8r0 | tilelang@main | 770.4 | 766.3-773.0 | 1.00 | 2033 | 2020-2052 | 1.00 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 194.8 | 192.6-198.4 | 0.25 | - | - | - |
| tiny-4096-hca-cp8r0 | tilelang@compare | 776.3 | 765.8-789.5 | 1.00 | 2118 | 2117-2121 | 1.00 |
| tiny-4096-hca-cp8r0 | cudnn_flashmla@compare | 312.0 | 305.2-315.2 | 0.40 | 1460 | 1459-1465 | 0.69 |
| tiny-4096-hca-cp8r0 | cute@compare | 195.7 | 193.5-196.4 | 0.25 | 1464 | 1445-1477 | 0.69 |
| tiny-4096-hca-cp8r0 | cute_ws@compare | 87.1 | 86.5-89.0 | 0.11 | 1297 | 1284-1302 | 0.61 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 195.1 | 192.5-199.7 | 0.25 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 767.4 | 763.6-775.5 | 1.00 | 2023 | 2018-2038 | 1.00 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 196.6 | 192.9-202.8 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@compare | 781.5 | 771.2-793.9 | 1.00 | 2116 | 2101-2130 | 1.00 |
| tiny-4096-hca-cp8r4 | cudnn_flashmla@compare | 311.9 | 308.0-314.6 | 0.40 | 1471 | 1465-1476 | 0.70 |
| tiny-4096-hca-cp8r4 | cute@compare | 198.8 | 197.7-203.7 | 0.25 | 1459 | 1449-1472 | 0.69 |
| tiny-4096-hca-cp8r4 | cute_ws@compare | 87.7 | 86.6-88.5 | 0.11 | 1294 | 1290-1299 | 0.61 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 200.2 | 192.5-200.8 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r7 | tilelang@main | 784.1 | 768.8-793.5 | 1.00 | 2053 | 2045-2065 | 1.00 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 201.5 | 196.4-202.1 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r7 | tilelang@compare | 774.1 | 764.4-777.2 | 1.00 | 2102 | 2097-2103 | 1.00 |
| tiny-4096-hca-cp8r7 | cudnn_flashmla@compare | 312.4 | 308.7-316.9 | 0.40 | 1459 | 1447-1469 | 0.69 |
| tiny-4096-hca-cp8r7 | cute@compare | 202.4 | 198.6-202.9 | 0.26 | 1452 | 1438-1463 | 0.69 |
| tiny-4096-hca-cp8r7 | cute_ws@compare | 87.2 | 87.0-88.7 | 0.11 | 1279 | 1274-1291 | 0.61 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 199.6 | 195.5-204.3 | 0.26 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang@main | 1226 | 1215-1234 | 1.00 | 2407 | 2386-2415 | 1.00 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 408.8 | 408.3-411.3 | 0.33 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang@compare | 1236 | 1232-1255 | 1.00 | 2468 | 2457-2479 | 1.00 |
| tiny-4096-sliding-cp1 | cudnn_flashmla@compare | 480.5 | 479.3-484.0 | 0.39 | 1781 | 1768-1795 | 0.72 |
| tiny-4096-sliding-cp1 | cute@compare | 613.1 | 611.6-615.9 | 0.50 | 1814 | 1812-1820 | 0.74 |
| tiny-4096-sliding-cp1 | cute_ws@compare | 286.2 | 274.2-288.6 | 0.23 | 1675 | 1666-1682 | 0.68 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@compare | 414.7 | 408.4-417.5 | 0.34 | - | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@main | 775.3 | 771.9-783.8 | 1.00 | 2030 | 2005-2076 | 1.00 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 200.7 | 197.0-203.9 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@compare | 781.6 | 763.6-791.2 | 1.00 | 2119 | 2114-2187 | 1.00 |
| tiny-4096-sliding-cp8r0 | cudnn_flashmla@compare | 308.4 | 301.1-314.0 | 0.39 | 1468 | 1460-1476 | 0.69 |
| tiny-4096-sliding-cp8r0 | cute@compare | 196.0 | 192.9-199.3 | 0.25 | 1468 | 1453-1472 | 0.69 |
| tiny-4096-sliding-cp8r0 | cute_ws@compare | 87.1 | 86.1-87.6 | 0.11 | 1296 | 1290-1305 | 0.61 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 198.7 | 190.9-200.3 | 0.25 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 781.6 | 770.1-787.6 | 1.00 | 2040 | 2028-2056 | 1.00 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 201.3-214.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@compare | 781.9 | 775.5-799.7 | 1.00 | 2112 | 2105-2127 | 1.00 |
| tiny-4096-sliding-cp8r4 | cudnn_flashmla@compare | 308.2 | 305.1-311.8 | 0.39 | 1466 | 1458-1481 | 0.69 |
| tiny-4096-sliding-cp8r4 | cute@compare | 198.8 | 198.0-201.8 | 0.25 | 1456 | 1453-1472 | 0.69 |
| tiny-4096-sliding-cp8r4 | cute_ws@compare | 88.4 | 86.0-89.8 | 0.11 | 1304 | 1292-1310 | 0.62 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 198.9 | 192.5-204.3 | 0.25 | - | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@main | 786.4 | 772.9-790.8 | 1.00 | 2081 | 2076-2129 | 1.00 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 202.2 | 194.2-206.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@compare | 781.3 | 777.2-786.0 | 1.00 | 2128 | 2105-2169 | 1.00 |
| tiny-4096-sliding-cp8r7 | cudnn_flashmla@compare | 311.2 | 305.4-312.1 | 0.40 | 1478 | 1466-1541 | 0.69 |
| tiny-4096-sliding-cp8r7 | cute@compare | 196.0 | 193.8-198.0 | 0.25 | 1475 | 1459-1511 | 0.69 |
| tiny-4096-sliding-cp8r7 | cute_ws@compare | 87.2 | 86.7-88.4 | 0.11 | 1300 | 1292-1375 | 0.61 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 201.7 | 195.8-202.4 | 0.26 | - | - | - |
| single-16384-csa-cp1 | tilelang@main | 5918 | 5886-6013 | 1.00 | 23560 | 23556-23572 | 1.00 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 2570 | 2560-2654 | 0.43 | - | - | - |
| single-16384-csa-cp1 | tilelang@compare | 5897 | 5879-5996 | 1.00 | 23714 | 23620-23756 | 1.00 |
| single-16384-csa-cp1 | cudnn_flashmla@compare | 2633 | 2630-2636 | 0.45 | 12686 | 12612-12703 | 0.53 |
| single-16384-csa-cp1 | cute@compare | 5165 | 5138-5320 | 0.88 | 22853 | 22699-22879 | 0.96 |
| single-16384-csa-cp1 | cute_ws@compare | 2433 | 2424-2643 | 0.41 | 20127 | 20065-20157 | 0.85 |
| single-16384-csa-cp1 | flashmla_fwd_ref@compare | 2761 | 2715-2789 | 0.47 | - | - | - |
| single-16384-csa-cp8r0 | tilelang@main | 1232 | 1216-1235 | 1.00 | 3243 | 3230-3250 | 1.00 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 417.6 | 414.9-421.8 | 0.34 | - | - | - |
| single-16384-csa-cp8r0 | tilelang@compare | 1234 | 1226-1238 | 1.00 | 3276 | 3266-3277 | 1.00 |
| single-16384-csa-cp8r0 | cudnn_flashmla@compare | 480.4 | 476.7-481.7 | 0.39 | 1949 | 1937-1954 | 0.59 |
| single-16384-csa-cp8r0 | cute@compare | 629.3 | 627.4-631.7 | 0.51 | 2630 | 2627-2645 | 0.80 |
| single-16384-csa-cp8r0 | cute_ws@compare | 307.5 | 304.8-307.9 | 0.25 | 2488 | 2485-2493 | 0.76 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 419.8 | 416.0-423.1 | 0.34 | - | - | - |
| single-16384-csa-cp8r4 | tilelang@main | 1409 | 1392-1436 | 1.00 | 4079 | 4072-4087 | 1.00 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 495.2 | 492.1-501.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r4 | tilelang@compare | 1420 | 1405-1425 | 1.00 | 4115 | 4101-4125 | 1.00 |
| single-16384-csa-cp8r4 | cudnn_flashmla@compare | 562.9 | 558.7-570.3 | 0.40 | 2325 | 2319-2376 | 0.57 |
| single-16384-csa-cp8r4 | cute@compare | 798.2 | 797.7-802.7 | 0.56 | 3467 | 3461-3472 | 0.84 |
| single-16384-csa-cp8r4 | cute_ws@compare | 384.0 | 383.4-384.8 | 0.27 | 3321 | 3318-3328 | 0.81 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 491.5 | 490.5-496.9 | 0.35 | - | - | - |
| single-16384-csa-cp8r7 | tilelang@main | 1405 | 1396-1424 | 1.00 | 4104 | 4096-4125 | 1.00 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 496.7 | 493.8-498.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r7 | tilelang@compare | 1409 | 1402-1413 | 1.00 | 4135 | 4112-4147 | 1.00 |
| single-16384-csa-cp8r7 | cudnn_flashmla@compare | 551.6 | 548.1-554.9 | 0.39 | 2337 | 2327-2347 | 0.57 |
| single-16384-csa-cp8r7 | cute@compare | 795.3 | 793.5-800.2 | 0.56 | 3476 | 3458-3495 | 0.84 |
| single-16384-csa-cp8r7 | cute_ws@compare | 383.0 | 382.0-385.1 | 0.27 | 3321 | 3306-3333 | 0.80 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 494.5 | 489.6-503.1 | 0.35 | - | - | - |
| single-16384-hca-cp1 | tilelang@main | 3573 | 3561-3617 | 1.00 | 10905 | 10899-10911 | 1.00 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1490 | 1486-1513 | 0.42 | - | - | - |
| single-16384-hca-cp1 | tilelang@compare | 3536 | 3524-3547 | 1.00 | 10875 | 10870-10886 | 1.00 |
| single-16384-hca-cp1 | cudnn_flashmla@compare | 1558 | 1554-1568 | 0.44 | 6534 | 6532-6565 | 0.60 |
| single-16384-hca-cp1 | cute@compare | 2773 | 2771-2777 | 0.78 | 10050 | 10037-10059 | 0.92 |
| single-16384-hca-cp1 | cute_ws@compare | 1249 | 1246-1253 | 0.35 | 8413 | 8402-8418 | 0.77 |
| single-16384-hca-cp1 | flashmla_fwd_ref@compare | 1486 | 1469-1487 | 0.42 | - | - | - |
| single-16384-hca-cp8r0 | tilelang@main | 1081 | 1067-1109 | 1.00 | 2422 | 2395-2446 | 1.00 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 340.5 | 338.5-348.6 | 0.32 | - | - | - |
| single-16384-hca-cp8r0 | tilelang@compare | 1072 | 1063-1075 | 1.00 | 2473 | 2463-2480 | 1.00 |
| single-16384-hca-cp8r0 | cudnn_flashmla@compare | 401.3 | 400.7-407.3 | 0.37 | 1588 | 1581-1606 | 0.64 |
| single-16384-hca-cp8r0 | cute@compare | 458.8 | 457.8-462.0 | 0.43 | 1823 | 1810-1826 | 0.74 |
| single-16384-hca-cp8r0 | cute_ws@compare | 222.9 | 222.3-223.8 | 0.21 | 1672 | 1662-1677 | 0.68 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 342.3 | 339.8-347.9 | 0.32 | - | - | - |
| single-16384-hca-cp8r4 | tilelang@main | 1097 | 1094-1101 | 1.00 | 2665 | 2641-2689 | 1.00 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 339.9 | 338.0-344.9 | 0.31 | - | - | - |
| single-16384-hca-cp8r4 | tilelang@compare | 1101 | 1097-1144 | 1.00 | 2731 | 2730-2737 | 1.00 |
| single-16384-hca-cp8r4 | cudnn_flashmla@compare | 404.3 | 403.4-422.0 | 0.37 | 1720 | 1712-1730 | 0.63 |
| single-16384-hca-cp8r4 | cute@compare | 511.6 | 507.3-518.5 | 0.46 | 2058 | 2053-2076 | 0.75 |
| single-16384-hca-cp8r4 | cute_ws@compare | 225.1 | 223.2-226.7 | 0.20 | 1903 | 1899-1921 | 0.70 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 341.5 | 339.8-344.5 | 0.31 | - | - | - |
| single-16384-hca-cp8r7 | tilelang@main | 1106 | 1097-1108 | 1.00 | 2819 | 2816-2866 | 1.00 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 343.8 | 338.7-344.0 | 0.31 | - | - | - |
| single-16384-hca-cp8r7 | tilelang@compare | 1124 | 1110-1132 | 1.00 | 2884 | 2856-2897 | 1.00 |
| single-16384-hca-cp8r7 | cudnn_flashmla@compare | 413.8 | 402.0-415.2 | 0.37 | 1750 | 1734-1775 | 0.61 |
| single-16384-hca-cp8r7 | cute@compare | 517.7 | 513.4-518.4 | 0.46 | 2172 | 2170-2196 | 0.75 |
| single-16384-hca-cp8r7 | cute_ws@compare | 225.6 | 223.9-227.1 | 0.20 | 2034 | 2018-2061 | 0.71 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 342.8 | 338.7-350.3 | 0.30 | - | - | - |
| single-16384-sliding-cp1 | tilelang@main | 2973 | 2970-2979 | 1.00 | 8286 | 8275-8288 | 1.00 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 1164 | 1159-1174 | 0.39 | - | - | - |
| single-16384-sliding-cp1 | tilelang@compare | 2969 | 2967-2994 | 1.00 | 8315 | 8294-8335 | 1.00 |
| single-16384-sliding-cp1 | cudnn_flashmla@compare | 1244 | 1243-1251 | 0.42 | 5340 | 5329-5347 | 0.64 |
| single-16384-sliding-cp1 | cute@compare | 2214 | 2212-2220 | 0.75 | 7501 | 7483-7506 | 0.90 |
| single-16384-sliding-cp1 | cute_ws@compare | 850.0 | 849.0-855.6 | 0.29 | 6072 | 6067-6077 | 0.73 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1159 | 1157-1164 | 0.39 | - | - | - |
| single-16384-sliding-cp8r0 | tilelang@main | 1019 | 1002-1035 | 1.00 | 2333 | 2317-2348 | 1.00 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 299.8 | 294.8-317.2 | 0.29 | - | - | - |
| single-16384-sliding-cp8r0 | tilelang@compare | 1008 | 993.3-1023 | 1.00 | 2406 | 2397-2419 | 1.00 |
| single-16384-sliding-cp8r0 | cudnn_flashmla@compare | 361.1 | 355.0-373.9 | 0.36 | 1561 | 1560-1579 | 0.65 |
| single-16384-sliding-cp8r0 | cute@compare | 412.5 | 407.4-416.5 | 0.41 | 1738 | 1733-1755 | 0.72 |
| single-16384-sliding-cp8r0 | cute_ws@compare | 171.5 | 169.4-172.1 | 0.17 | 1595 | 1584-1601 | 0.66 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 292.7 | 290.0-295.4 | 0.29 | - | - | - |
| single-16384-sliding-cp8r4 | tilelang@main | 1004 | 986.3-1009 | 1.00 | 2390 | 2382-2476 | 1.00 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 296.7 | 289.8-313.0 | 0.30 | - | - | - |
| single-16384-sliding-cp8r4 | tilelang@compare | 1015 | 1007-1028 | 1.00 | 2417 | 2408-2434 | 1.00 |
| single-16384-sliding-cp8r4 | cudnn_flashmla@compare | 361.2 | 359.9-361.9 | 0.36 | 1557 | 1553-1566 | 0.64 |
| single-16384-sliding-cp8r4 | cute@compare | 420.2 | 417.4-424.3 | 0.41 | 1777 | 1763-1792 | 0.73 |
| single-16384-sliding-cp8r4 | cute_ws@compare | 171.0 | 169.9-174.5 | 0.17 | 1627 | 1617-1633 | 0.67 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 299.3 | 295.0-300.8 | 0.29 | - | - | - |
| single-16384-sliding-cp8r7 | tilelang@main | 996.2 | 987.6-1004 | 1.00 | 2389 | 2385-2438 | 1.00 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 294.0 | 292.2-302.3 | 0.30 | - | - | - |
| single-16384-sliding-cp8r7 | tilelang@compare | 1021 | 1011-1026 | 1.00 | 2455 | 2430-2459 | 1.00 |
| single-16384-sliding-cp8r7 | cudnn_flashmla@compare | 358.5 | 358.0-365.1 | 0.35 | 1562 | 1554-1571 | 0.64 |
| single-16384-sliding-cp8r7 | cute@compare | 419.2 | 413.8-423.9 | 0.41 | 1785 | 1782-1790 | 0.73 |
| single-16384-sliding-cp8r7 | cute_ws@compare | 170.9 | 170.4-171.7 | 0.17 | 1638 | 1623-1652 | 0.67 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 295.5 | 295.1-302.6 | 0.29 | - | - | - |
| short-16384-csa-cp1 | tilelang@main | 4509 | 4494-4544 | 1.00 | 15976 | 15975-16005 | 1.00 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 1957 | 1952-1959 | 0.43 | - | - | - |
| short-16384-csa-cp1 | tilelang@compare | 4497 | 4490-4500 | 1.00 | 15994 | 15975-16022 | 1.00 |
| short-16384-csa-cp1 | cudnn_flashmla@compare | 2023 | 2009-2028 | 0.45 | 8842 | 8840-8872 | 0.55 |
| short-16384-csa-cp1 | cute@compare | 3694 | 3689-3819 | 0.82 | 15148 | 15061-15169 | 0.95 |
| short-16384-csa-cp1 | cute_ws@compare | 1792 | 1790-1859 | 0.40 | 13188 | 13123-13200 | 0.82 |
| short-16384-csa-cp1 | flashmla_fwd_ref@compare | 1985 | 1957-1998 | 0.44 | - | - | - |
| short-16384-csa-cp8r0 | tilelang@main | 1175 | 1156-1200 | 1.00 | 2997 | 2984-3003 | 1.00 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 390.1 | 384.5-394.8 | 0.33 | - | - | - |
| short-16384-csa-cp8r0 | tilelang@compare | 1190 | 1173-1191 | 1.00 | 3013 | 3004-3031 | 1.00 |
| short-16384-csa-cp8r0 | cudnn_flashmla@compare | 451.6 | 450.6-474.2 | 0.38 | 1828 | 1822-1840 | 0.61 |
| short-16384-csa-cp8r0 | cute@compare | 570.5 | 569.5-573.7 | 0.48 | 2368 | 2362-2385 | 0.79 |
| short-16384-csa-cp8r0 | cute_ws@compare | 278.1 | 275.9-279.3 | 0.23 | 2214 | 2206-2225 | 0.73 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 385.5 | 382.2-387.3 | 0.32 | - | - | - |
| short-16384-csa-cp8r4 | tilelang@main | 1249 | 1236-1261 | 1.00 | 3387 | 3384-3406 | 1.00 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 431.6 | 429.3-432.5 | 0.35 | - | - | - |
| short-16384-csa-cp8r4 | tilelang@compare | 1270 | 1263-1278 | 1.00 | 3462 | 3445-3467 | 1.00 |
| short-16384-csa-cp8r4 | cudnn_flashmla@compare | 491.7 | 487.2-500.3 | 0.39 | 2026 | 2011-2032 | 0.59 |
| short-16384-csa-cp8r4 | cute@compare | 652.5 | 649.4-659.0 | 0.51 | 2810 | 2800-2812 | 0.81 |
| short-16384-csa-cp8r4 | cute_ws@compare | 318.6 | 316.2-319.6 | 0.25 | 2654 | 2648-2662 | 0.77 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 431.8 | 426.2-434.6 | 0.34 | - | - | - |
| short-16384-csa-cp8r7 | tilelang@main | 1211 | 1195-1219 | 1.00 | 3241 | 3236-3255 | 1.00 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 408.0 | 406.4-415.0 | 0.34 | - | - | - |
| short-16384-csa-cp8r7 | tilelang@compare | 1205 | 1197-1213 | 1.00 | 3319 | 3287-3331 | 1.00 |
| short-16384-csa-cp8r7 | cudnn_flashmla@compare | 473.7 | 468.8-479.6 | 0.39 | 1965 | 1962-1967 | 0.59 |
| short-16384-csa-cp8r7 | cute@compare | 614.9 | 614.0-617.4 | 0.51 | 2641 | 2634-2646 | 0.80 |
| short-16384-csa-cp8r7 | cute_ws@compare | 305.1 | 304.5-307.6 | 0.25 | 2486 | 2479-2499 | 0.75 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 412.3 | 408.3-414.0 | 0.34 | - | - | - |
| short-16384-hca-cp1 | tilelang@main | 3337 | 3333-3349 | 1.00 | 9225 | 9192-9255 | 1.00 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 1463 | 1461-1465 | 0.44 | - | - | - |
| short-16384-hca-cp1 | tilelang@compare | 3324 | 3313-3334 | 1.00 | 9140 | 9116-9154 | 1.00 |
| short-16384-hca-cp1 | cudnn_flashmla@compare | 1547 | 1536-1548 | 0.47 | 5881 | 5877-5891 | 0.64 |
| short-16384-hca-cp1 | cute@compare | 2544 | 2537-2554 | 0.77 | 8313 | 8310-8321 | 0.91 |
| short-16384-hca-cp1 | cute_ws@compare | 1231 | 1230-1234 | 0.37 | 6943 | 6935-6950 | 0.76 |
| short-16384-hca-cp1 | flashmla_fwd_ref@compare | 1458 | 1456-1463 | 0.44 | - | - | - |
| short-16384-hca-cp8r0 | tilelang@main | 1070 | 1063-1086 | 1.00 | 2499 | 2480-2505 | 1.00 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 338.9 | 332.9-342.3 | 0.32 | - | - | - |
| short-16384-hca-cp8r0 | tilelang@compare | 1093 | 1075-1105 | 1.00 | 2549 | 2541-2551 | 1.00 |
| short-16384-hca-cp8r0 | cudnn_flashmla@compare | 408.3 | 399.5-409.3 | 0.37 | 1595 | 1593-1601 | 0.63 |
| short-16384-hca-cp8r0 | cute@compare | 469.5 | 468.5-478.7 | 0.43 | 1911 | 1896-1921 | 0.75 |
| short-16384-hca-cp8r0 | cute_ws@compare | 217.8 | 217.4-218.8 | 0.20 | 1702 | 1696-1714 | 0.67 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 342.3 | 338.4-344.3 | 0.31 | - | - | - |
| short-16384-hca-cp8r4 | tilelang@main | 1080 | 1076-1091 | 1.00 | 2532 | 2525-2555 | 1.00 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 337.4 | 334.8-343.4 | 0.31 | - | - | - |
| short-16384-hca-cp8r4 | tilelang@compare | 1101 | 1083-1104 | 1.00 | 2639 | 2623-2646 | 1.00 |
| short-16384-hca-cp8r4 | cudnn_flashmla@compare | 407.5 | 402.2-409.0 | 0.37 | 1630 | 1628-1632 | 0.62 |
| short-16384-hca-cp8r4 | cute@compare | 483.8 | 480.4-484.6 | 0.44 | 1959 | 1953-1990 | 0.74 |
| short-16384-hca-cp8r4 | cute_ws@compare | 221.0 | 220.6-222.4 | 0.20 | 1763 | 1753-1776 | 0.67 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 344.7 | 339.6-350.4 | 0.31 | - | - | - |
| short-16384-hca-cp8r7 | tilelang@main | 1081 | 1069-1086 | 1.00 | 2558 | 2548-2561 | 1.00 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 338.8 | 338.8-342.2 | 0.31 | - | - | - |
| short-16384-hca-cp8r7 | tilelang@compare | 1094 | 1088-1107 | 1.00 | 2571 | 2562-2587 | 1.00 |
| short-16384-hca-cp8r7 | cudnn_flashmla@compare | 414.3 | 408.7-423.5 | 0.38 | 1608 | 1596-1610 | 0.63 |
| short-16384-hca-cp8r7 | cute@compare | 480.7 | 474.7-490.1 | 0.44 | 1932 | 1929-1938 | 0.75 |
| short-16384-hca-cp8r7 | cute_ws@compare | 220.1 | 218.7-222.0 | 0.20 | 1723 | 1719-1747 | 0.67 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 343.1 | 339.5-348.3 | 0.31 | - | - | - |
| short-16384-sliding-cp1 | tilelang@main | 2985 | 2970-3000 | 1.00 | 8155 | 8151-8192 | 1.00 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 1169 | 1162-1175 | 0.39 | - | - | - |
| short-16384-sliding-cp1 | tilelang@compare | 2956 | 2952-2968 | 1.00 | 8123 | 8120-8128 | 1.00 |
| short-16384-sliding-cp1 | cudnn_flashmla@compare | 1247 | 1243-1252 | 0.42 | 5258 | 5254-5261 | 0.65 |
| short-16384-sliding-cp1 | cute@compare | 2203 | 2202-2209 | 0.75 | 7336 | 7330-7342 | 0.90 |
| short-16384-sliding-cp1 | cute_ws@compare | 850.8 | 848.8-855.8 | 0.29 | 5942 | 5935-5947 | 0.73 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1166 | 1162-1168 | 0.39 | - | - | - |
| short-16384-sliding-cp8r0 | tilelang@main | 1001 | 987.1-1006 | 1.00 | 2305 | 2292-2331 | 1.00 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 296.3 | 293.4-300.4 | 0.30 | - | - | - |
| short-16384-sliding-cp8r0 | tilelang@compare | 1003 | 995.4-1005 | 1.00 | 2414 | 2396-2421 | 1.00 |
| short-16384-sliding-cp8r0 | cudnn_flashmla@compare | 364.4 | 356.4-365.2 | 0.36 | 1559 | 1557-1564 | 0.65 |
| short-16384-sliding-cp8r0 | cute@compare | 409.3 | 407.1-409.8 | 0.41 | 1743 | 1734-1773 | 0.72 |
| short-16384-sliding-cp8r0 | cute_ws@compare | 169.3 | 169.1-172.5 | 0.17 | 1590 | 1584-1598 | 0.66 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 295.5 | 290.8-300.3 | 0.29 | - | - | - |
| short-16384-sliding-cp8r4 | tilelang@main | 993.9 | 985.0-1070 | 1.00 | 2370 | 2342-2396 | 1.00 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 294.4 | 287.7-309.9 | 0.30 | - | - | - |
| short-16384-sliding-cp8r4 | tilelang@compare | 1015 | 1010-1036 | 1.00 | 2403 | 2384-2424 | 1.00 |
| short-16384-sliding-cp8r4 | cudnn_flashmla@compare | 364.6 | 363.1-365.5 | 0.36 | 1545 | 1541-1559 | 0.64 |
| short-16384-sliding-cp8r4 | cute@compare | 420.4 | 416.9-422.0 | 0.41 | 1757 | 1754-1764 | 0.73 |
| short-16384-sliding-cp8r4 | cute_ws@compare | 170.8 | 170.1-172.2 | 0.17 | 1621 | 1614-1685 | 0.67 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 297.3 | 297.1-297.7 | 0.29 | - | - | - |
| short-16384-sliding-cp8r7 | tilelang@main | 1001 | 987.3-1004 | 1.00 | 2361 | 2358-2371 | 1.00 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 294.5 | 290.2-294.9 | 0.29 | - | - | - |
| short-16384-sliding-cp8r7 | tilelang@compare | 1042 | 1013-1083 | 1.00 | 2401 | 2396-2420 | 1.00 |
| short-16384-sliding-cp8r7 | cudnn_flashmla@compare | 361.9 | 358.7-365.1 | 0.35 | 1558 | 1549-1562 | 0.65 |
| short-16384-sliding-cp8r7 | cute@compare | 415.4 | 413.9-418.5 | 0.40 | 1762 | 1749-1778 | 0.73 |
| short-16384-sliding-cp8r7 | cute_ws@compare | 170.6 | 169.0-171.8 | 0.16 | 1608 | 1598-1611 | 0.67 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 294.3 | 292.4-299.0 | 0.28 | - | - | - |
| heavy-16384-csa-cp1 | tilelang@main | 3969 | 3954-4002 | 1.00 | 12997 | 12956-13003 | 1.00 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1726 | 1725-1731 | 0.43 | - | - | - |
| heavy-16384-csa-cp1 | tilelang@compare | 3972 | 3954-4006 | 1.00 | 12916 | 12907-12939 | 1.00 |
| heavy-16384-csa-cp1 | cudnn_flashmla@compare | 1790 | 1786-1790 | 0.45 | 7453 | 7432-7464 | 0.58 |
| heavy-16384-csa-cp1 | cute@compare | 3191 | 3183-3199 | 0.80 | 12107 | 12094-12127 | 0.94 |
| heavy-16384-csa-cp1 | cute_ws@compare | 1557 | 1554-1563 | 0.39 | 10427 | 10419-10433 | 0.81 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@compare | 1733 | 1721-1739 | 0.44 | - | - | - |
| heavy-16384-csa-cp8r0 | tilelang@main | 1092 | 1069-1121 | 1.00 | 2575 | 2548-2608 | 1.00 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 346.5 | 345.0-355.2 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r0 | tilelang@compare | 1089 | 1087-1116 | 1.00 | 2607 | 2605-2639 | 1.00 |
| heavy-16384-csa-cp8r0 | cudnn_flashmla@compare | 412.0 | 406.1-421.9 | 0.38 | 1650 | 1647-1654 | 0.63 |
| heavy-16384-csa-cp8r0 | cute@compare | 489.1 | 487.5-491.1 | 0.45 | 1976 | 1973-1985 | 0.76 |
| heavy-16384-csa-cp8r0 | cute_ws@compare | 238.0 | 237.5-239.9 | 0.22 | 1829 | 1819-1838 | 0.70 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 350.4 | 344.0-352.8 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r4 | tilelang@main | 1117 | 1112-1126 | 1.00 | 2762 | 2761-2778 | 1.00 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 362.7 | 357.6-370.0 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r4 | tilelang@compare | 1136 | 1125-1154 | 1.00 | 2878 | 2853-2886 | 1.00 |
| heavy-16384-csa-cp8r4 | cudnn_flashmla@compare | 442.1 | 431.1-451.2 | 0.39 | 1761 | 1753-1772 | 0.61 |
| heavy-16384-csa-cp8r4 | cute@compare | 527.2 | 526.2-532.2 | 0.46 | 2203 | 2190-2213 | 0.77 |
| heavy-16384-csa-cp8r4 | cute_ws@compare | 257.1 | 255.4-258.2 | 0.23 | 2041 | 2039-2044 | 0.71 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 366.7 | 361.9-371.1 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r7 | tilelang@main | 1264 | 1261-1266 | 1.00 | 3482 | 3467-3491 | 1.00 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 428.5 | 427.4-434.8 | 0.34 | - | - | - |
| heavy-16384-csa-cp8r7 | tilelang@compare | 1273 | 1265-1277 | 1.00 | 3519 | 3514-3556 | 1.00 |
| heavy-16384-csa-cp8r7 | cudnn_flashmla@compare | 502.7 | 498.6-515.8 | 0.39 | 2070 | 2062-2081 | 0.59 |
| heavy-16384-csa-cp8r7 | cute@compare | 670.4 | 670.2-674.6 | 0.53 | 2865 | 2839-2877 | 0.81 |
| heavy-16384-csa-cp8r7 | cute_ws@compare | 325.9 | 324.8-326.4 | 0.26 | 2709 | 2688-2740 | 0.77 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 440.0 | 437.0-441.3 | 0.35 | - | - | - |
| heavy-16384-hca-cp1 | tilelang@main | 3252 | 3222-3290 | 1.00 | 8705 | 8691-8740 | 1.00 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 1404 | 1396-1410 | 0.43 | - | - | - |
| heavy-16384-hca-cp1 | tilelang@compare | 3212 | 3203-3229 | 1.00 | 8609 | 8604-8616 | 1.00 |
| heavy-16384-hca-cp1 | cudnn_flashmla@compare | 1475 | 1470-1491 | 0.46 | 5628 | 5622-5633 | 0.65 |
| heavy-16384-hca-cp1 | cute@compare | 2448 | 2446-2454 | 0.76 | 7828 | 7820-7834 | 0.91 |
| heavy-16384-hca-cp1 | cute_ws@compare | 1163 | 1160-1169 | 0.36 | 6478 | 6470-6483 | 0.75 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@compare | 1397 | 1393-1398 | 0.43 | - | - | - |
| heavy-16384-hca-cp8r0 | tilelang@main | 1068 | 1060-1096 | 1.00 | 2448 | 2445-2477 | 1.00 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 335.5 | 331.0-346.8 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r0 | tilelang@compare | 1073 | 1067-1077 | 1.00 | 2511 | 2494-2525 | 1.00 |
| heavy-16384-hca-cp8r0 | cudnn_flashmla@compare | 398.2 | 395.6-404.9 | 0.37 | 1565 | 1558-1569 | 0.62 |
| heavy-16384-hca-cp8r0 | cute@compare | 458.5 | 455.0-458.9 | 0.43 | 1851 | 1838-1863 | 0.74 |
| heavy-16384-hca-cp8r0 | cute_ws@compare | 206.9 | 205.9-207.6 | 0.19 | 1652 | 1648-1660 | 0.66 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 333.7 | 328.2-337.2 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r4 | tilelang@main | 1070 | 1066-1081 | 1.00 | 2488 | 2479-2500 | 1.00 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 341.7 | 338.1-344.5 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r4 | tilelang@compare | 1064 | 1055-1083 | 1.00 | 2573 | 2570-2630 | 1.00 |
| heavy-16384-hca-cp8r4 | cudnn_flashmla@compare | 404.4 | 395.1-405.8 | 0.38 | 1605 | 1595-1636 | 0.62 |
| heavy-16384-hca-cp8r4 | cute@compare | 469.8 | 466.3-471.4 | 0.44 | 1911 | 1896-1982 | 0.74 |
| heavy-16384-hca-cp8r4 | cute_ws@compare | 212.0 | 210.6-216.0 | 0.20 | 1710 | 1702-1784 | 0.66 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 338.8 | 336.8-342.0 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r7 | tilelang@main | 1088 | 1082-1106 | 1.00 | 2621 | 2616-2649 | 1.00 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 346.1 | 343.6-346.5 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r7 | tilelang@compare | 1107 | 1098-1120 | 1.00 | 2649 | 2624-2652 | 1.00 |
| heavy-16384-hca-cp8r7 | cudnn_flashmla@compare | 406.0 | 404.6-412.6 | 0.37 | 1644 | 1641-1652 | 0.62 |
| heavy-16384-hca-cp8r7 | cute@compare | 487.0 | 485.1-490.0 | 0.44 | 1977 | 1974-1991 | 0.75 |
| heavy-16384-hca-cp8r7 | cute_ws@compare | 225.1 | 223.0-227.3 | 0.20 | 1778 | 1773-1799 | 0.67 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 347.9 | 343.4-349.2 | 0.31 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang@main | 2979 | 2958-3008 | 1.00 | 7887 | 7879-7894 | 1.00 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 1173 | 1173-1183 | 0.39 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang@compare | 2959 | 2947-2966 | 1.00 | 7864 | 7853-7924 | 1.00 |
| heavy-16384-sliding-cp1 | cudnn_flashmla@compare | 1251 | 1250-1253 | 0.42 | 5149 | 5142-5165 | 0.65 |
| heavy-16384-sliding-cp1 | cute@compare | 2197 | 2194-2201 | 0.74 | 7072 | 7068-7098 | 0.90 |
| heavy-16384-sliding-cp1 | cute_ws@compare | 870.8 | 867.9-874.1 | 0.29 | 5702 | 5695-5735 | 0.73 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1174 | 1169-1177 | 0.40 | - | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@main | 1013 | 1002-1040 | 1.00 | 2292 | 2285-2305 | 1.00 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 297.7 | 295.7-301.9 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@compare | 1013 | 986.5-1024 | 1.00 | 2362 | 2355-2371 | 1.00 |
| heavy-16384-sliding-cp8r0 | cudnn_flashmla@compare | 360.8 | 359.5-363.7 | 0.36 | 1534 | 1524-1554 | 0.65 |
| heavy-16384-sliding-cp8r0 | cute@compare | 402.8 | 400.8-404.4 | 0.40 | 1702 | 1677-1754 | 0.72 |
| heavy-16384-sliding-cp8r0 | cute_ws@compare | 170.3 | 169.0-170.6 | 0.17 | 1556 | 1554-1607 | 0.66 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 297.6 | 292.8-303.0 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@main | 982.7 | 980.0-1006 | 1.00 | 2331 | 2316-2337 | 1.00 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 293.1 | 290.1-298.7 | 0.30 | - | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@compare | 1007 | 1003-1014 | 1.00 | 2375 | 2371-2401 | 1.00 |
| heavy-16384-sliding-cp8r4 | cudnn_flashmla@compare | 362.5 | 360.4-364.1 | 0.36 | 1550 | 1541-1570 | 0.65 |
| heavy-16384-sliding-cp8r4 | cute@compare | 411.0 | 409.6-412.9 | 0.41 | 1732 | 1722-1743 | 0.73 |
| heavy-16384-sliding-cp8r4 | cute_ws@compare | 170.6 | 169.2-171.4 | 0.17 | 1589 | 1576-1608 | 0.67 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 295.8 | 293.9-298.7 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@main | 1022 | 994.6-1025 | 1.00 | 2405 | 2394-2439 | 1.00 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 296.4 | 293.9-310.3 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@compare | 1027 | 1019-1078 | 1.00 | 2475 | 2462-2567 | 1.00 |
| heavy-16384-sliding-cp8r7 | cudnn_flashmla@compare | 362.9 | 361.9-372.1 | 0.35 | 1565 | 1558-1566 | 0.63 |
| heavy-16384-sliding-cp8r7 | cute@compare | 425.4 | 423.3-427.8 | 0.41 | 1789 | 1776-1803 | 0.72 |
| heavy-16384-sliding-cp8r7 | cute_ws@compare | 173.0 | 170.9-173.5 | 0.17 | 1636 | 1636-1644 | 0.66 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 304.6 | 299.2-311.3 | 0.30 | - | - | - |
| tiny-16384-csa-cp1 | tilelang@main | 3287 | 3284-3298 | 1.00 | 8726 | 8702-8759 | 1.00 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 1476 | 1471-1478 | 0.45 | - | - | - |
| tiny-16384-csa-cp1 | tilelang@compare | 3296 | 3279-3318 | 1.00 | 8778 | 8735-8784 | 1.00 |
| tiny-16384-csa-cp1 | cudnn_flashmla@compare | 1536 | 1535-1549 | 0.47 | 5556 | 5552-5565 | 0.63 |
| tiny-16384-csa-cp1 | cute@compare | 2543 | 2542-2549 | 0.77 | 7903 | 7898-7913 | 0.90 |
| tiny-16384-csa-cp1 | cute_ws@compare | 1300 | 1299-1302 | 0.39 | 6611 | 6610-6623 | 0.75 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@compare | 1477 | 1475-1478 | 0.45 | - | - | - |
| tiny-16384-csa-cp8r0 | tilelang@main | 1043 | 1041-1055 | 1.00 | 2376 | 2356-2382 | 1.00 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 334.5 | 334.0-340.0 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r0 | tilelang@compare | 1064 | 1049-1097 | 1.00 | 2442 | 2409-2469 | 1.00 |
| tiny-16384-csa-cp8r0 | cudnn_flashmla@compare | 407.5 | 401.9-412.9 | 0.38 | 1552 | 1547-1556 | 0.64 |
| tiny-16384-csa-cp8r0 | cute@compare | 454.3 | 453.5-460.7 | 0.43 | 1773 | 1767-1792 | 0.73 |
| tiny-16384-csa-cp8r0 | cute_ws@compare | 227.4 | 226.7-229.4 | 0.21 | 1612 | 1604-1620 | 0.66 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 340.5 | 334.7-343.5 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r4 | tilelang@main | 1048 | 1040-1052 | 1.00 | 2358 | 2354-2374 | 1.00 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 342.2 | 335.7-346.6 | 0.33 | - | - | - |
| tiny-16384-csa-cp8r4 | tilelang@compare | 1040 | 1036-1047 | 1.00 | 2420 | 2411-2425 | 1.00 |
| tiny-16384-csa-cp8r4 | cudnn_flashmla@compare | 401.4 | 396.8-406.5 | 0.39 | 1545 | 1538-1547 | 0.64 |
| tiny-16384-csa-cp8r4 | cute@compare | 451.5 | 449.9-452.2 | 0.43 | 1765 | 1758-1779 | 0.73 |
| tiny-16384-csa-cp8r4 | cute_ws@compare | 227.7 | 225.9-228.2 | 0.22 | 1618 | 1607-1625 | 0.67 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 335.0 | 329.7-344.6 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r7 | tilelang@main | 1032 | 1022-1076 | 1.00 | 2369 | 2363-2372 | 1.00 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 334.5 | 327.7-343.5 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r7 | tilelang@compare | 1044 | 1032-1049 | 1.00 | 2438 | 2427-2501 | 1.00 |
| tiny-16384-csa-cp8r7 | cudnn_flashmla@compare | 394.6 | 393.7-400.1 | 0.38 | 1572 | 1545-1581 | 0.64 |
| tiny-16384-csa-cp8r7 | cute@compare | 451.9 | 451.3-452.8 | 0.43 | 1779 | 1760-1793 | 0.73 |
| tiny-16384-csa-cp8r7 | cute_ws@compare | 228.2 | 225.6-229.3 | 0.22 | 1610 | 1600-1634 | 0.66 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 338.2 | 332.5-344.9 | 0.32 | - | - | - |
| tiny-16384-hca-cp1 | tilelang@main | 2707 | 2700-2714 | 1.00 | 6184 | 6177-6195 | 1.00 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 1176 | 1174-1178 | 0.43 | - | - | - |
| tiny-16384-hca-cp1 | tilelang@compare | 2718 | 2710-2742 | 1.00 | 6179 | 6177-6187 | 1.00 |
| tiny-16384-hca-cp1 | cudnn_flashmla@compare | 1256 | 1251-1258 | 0.46 | 4476 | 4468-4478 | 0.72 |
| tiny-16384-hca-cp1 | cute@compare | 1975 | 1972-1983 | 0.73 | 5404 | 5395-5408 | 0.87 |
| tiny-16384-hca-cp1 | cute_ws@compare | 975.2 | 973.8-983.2 | 0.36 | 4355 | 4347-4356 | 0.70 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@compare | 1181 | 1174-1186 | 0.43 | - | - | - |
| tiny-16384-hca-cp8r0 | tilelang@main | 981.1 | 976.9-1003 | 1.00 | 2144 | 2130-2255 | 1.00 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 296.6 | 289.3-309.1 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r0 | tilelang@compare | 995.6 | 979.6-999.7 | 1.00 | 2176 | 2171-2184 | 1.00 |
| tiny-16384-hca-cp8r0 | cudnn_flashmla@compare | 360.9 | 359.4-362.4 | 0.36 | 1519 | 1499-1520 | 0.70 |
| tiny-16384-hca-cp8r0 | cute@compare | 382.8 | 381.9-385.4 | 0.38 | 1524 | 1518-1539 | 0.70 |
| tiny-16384-hca-cp8r0 | cute_ws@compare | 167.2 | 166.1-169.6 | 0.17 | 1375 | 1362-1384 | 0.63 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 300.2 | 297.1-301.4 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r4 | tilelang@main | 968.2 | 959.1-972.4 | 1.00 | 2095 | 2090-2123 | 1.00 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 293.1 | 289.9-295.4 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r4 | tilelang@compare | 977.0 | 969.5-979.3 | 1.00 | 2183 | 2177-2195 | 1.00 |
| tiny-16384-hca-cp8r4 | cudnn_flashmla@compare | 361.0 | 359.0-368.0 | 0.37 | 1501 | 1497-1514 | 0.69 |
| tiny-16384-hca-cp8r4 | cute@compare | 384.1 | 382.0-388.2 | 0.39 | 1520 | 1501-1531 | 0.70 |
| tiny-16384-hca-cp8r4 | cute_ws@compare | 167.7 | 166.9-170.2 | 0.17 | 1361 | 1349-1377 | 0.62 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 296.5 | 293.7-307.0 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r7 | tilelang@main | 955.4 | 942.5-968.8 | 1.00 | 2130 | 2120-2172 | 1.00 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 294.3 | 290.2-298.9 | 0.31 | - | - | - |
| tiny-16384-hca-cp8r7 | tilelang@compare | 972.3 | 961.6-981.8 | 1.00 | 2149 | 2139-2159 | 1.00 |
| tiny-16384-hca-cp8r7 | cudnn_flashmla@compare | 361.8 | 358.4-364.7 | 0.37 | 1496 | 1495-1497 | 0.70 |
| tiny-16384-hca-cp8r7 | cute@compare | 371.6 | 368.3-377.4 | 0.38 | 1493 | 1488-1503 | 0.69 |
| tiny-16384-hca-cp8r7 | cute_ws@compare | 168.1 | 166.1-169.3 | 0.17 | 1334 | 1329-1348 | 0.62 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 294.7 | 292.9-300.4 | 0.30 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang@main | 2728 | 2711-2754 | 1.00 | 6163 | 6161-6164 | 1.00 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 1181 | 1174-1182 | 0.43 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang@compare | 2732 | 2719-2743 | 1.00 | 6196 | 6189-6204 | 1.00 |
| tiny-16384-sliding-cp1 | cudnn_flashmla@compare | 1259 | 1256-1262 | 0.46 | 4472 | 4467-4475 | 0.72 |
| tiny-16384-sliding-cp1 | cute@compare | 1976 | 1970-1978 | 0.72 | 5405 | 5396-5410 | 0.87 |
| tiny-16384-sliding-cp1 | cute_ws@compare | 980.0 | 977.7-981.2 | 0.36 | 4360 | 4357-4368 | 0.70 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1180 | 1178-1186 | 0.43 | - | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@main | 962.6 | 956.2-977.8 | 1.00 | 2142 | 2126-2157 | 1.00 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 297.0 | 295.1-301.5 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@compare | 965.5 | 956.9-980.1 | 1.00 | 2223 | 2215-2235 | 1.00 |
| tiny-16384-sliding-cp8r0 | cudnn_flashmla@compare | 359.1 | 354.0-360.4 | 0.37 | 1566 | 1538-1588 | 0.70 |
| tiny-16384-sliding-cp8r0 | cute@compare | 381.3 | 380.6-383.6 | 0.39 | 1560 | 1545-1565 | 0.70 |
| tiny-16384-sliding-cp8r0 | cute_ws@compare | 166.9 | 165.8-168.6 | 0.17 | 1408 | 1387-1423 | 0.63 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 299.0 | 292.4-299.2 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@main | 962.2 | 954.6-977.3 | 1.00 | 2174 | 2113-2294 | 1.00 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 299.2 | 290.4-299.6 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@compare | 979.9 | 973.5-991.1 | 1.00 | 2183 | 2175-2223 | 1.00 |
| tiny-16384-sliding-cp8r4 | cudnn_flashmla@compare | 366.4 | 360.9-377.0 | 0.37 | 1523 | 1508-1540 | 0.70 |
| tiny-16384-sliding-cp8r4 | cute@compare | 384.2 | 382.8-394.4 | 0.39 | 1520 | 1510-1562 | 0.70 |
| tiny-16384-sliding-cp8r4 | cute_ws@compare | 169.8 | 168.5-171.1 | 0.17 | 1371 | 1366-1397 | 0.63 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 296.9 | 296.1-302.6 | 0.30 | - | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@main | 961.0 | 949.7-972.0 | 1.00 | 2140 | 2129-2174 | 1.00 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 293.2 | 292.1-301.2 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@compare | 973.4 | 970.6-985.2 | 1.00 | 2178 | 2162-2203 | 1.00 |
| tiny-16384-sliding-cp8r7 | cudnn_flashmla@compare | 362.7 | 361.7-368.4 | 0.37 | 1519 | 1513-1529 | 0.70 |
| tiny-16384-sliding-cp8r7 | cute@compare | 377.1 | 375.4-379.7 | 0.39 | 1504 | 1496-1525 | 0.69 |
| tiny-16384-sliding-cp8r7 | cute_ws@compare | 170.4 | 169.2-171.6 | 0.18 | 1352 | 1344-1366 | 0.62 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 298.4 | 296.9-302.5 | 0.31 | - | - | - |
| single-49208-csa-cp1 | tilelang@main | 16424 | 16369-16661 | 1.00 | 70708 | 70691-70733 | 1.00 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 8309 | 8259-8522 | 0.51 | - | - | - |
| single-49208-csa-cp1 | tilelang@compare | 16493 | 16396-16601 | 1.00 | 70854 | 70763-71076 | 1.00 |
| single-49208-csa-cp1 | cudnn_flashmla@compare | 8300 | 8120-8361 | 0.50 | 39489 | 39223-39525 | 0.56 |
| single-49208-csa-cp1 | cute@compare | 15339 | 15328-15379 | 0.93 | 69675 | 69432-69691 | 0.98 |
| single-49208-csa-cp1 | cute_ws@compare | 7885 | 7815-8102 | 0.48 | 61743 | 61698-61771 | 0.87 |
| single-49208-csa-cp1 | flashmla_fwd_ref@compare | 8533 | 8532-8604 | 0.52 | - | - | - |
| single-49208-csa-cp8r0 | tilelang@main | 2556 | 2547-2584 | 1.00 | 8923 | 8921-8925 | 1.00 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 1026 | 1020-1028 | 0.40 | - | - | - |
| single-49208-csa-cp8r0 | tilelang@compare | 2564 | 2556-2597 | 1.00 | 8977 | 8944-9000 | 1.00 |
| single-49208-csa-cp8r0 | cudnn_flashmla@compare | 1101 | 1098-1105 | 0.43 | 4733 | 4723-4740 | 0.53 |
| single-49208-csa-cp8r0 | cute@compare | 1891 | 1883-1954 | 0.74 | 8258 | 8222-8259 | 0.92 |
| single-49208-csa-cp8r0 | cute_ws@compare | 909.2 | 906.4-910.9 | 0.35 | 7215 | 7191-7218 | 0.80 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 1025 | 1024-1038 | 0.40 | - | - | - |
| single-49208-csa-cp8r4 | tilelang@main | 2744 | 2725-2754 | 1.00 | 10025 | 10005-10044 | 1.00 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1100 | 1100-1108 | 0.40 | - | - | - |
| single-49208-csa-cp8r4 | tilelang@compare | 2795 | 2778-2808 | 1.00 | 10134 | 10124-10198 | 1.00 |
| single-49208-csa-cp8r4 | cudnn_flashmla@compare | 1195 | 1186-1198 | 0.43 | 5230 | 5216-5233 | 0.52 |
| single-49208-csa-cp8r4 | cute@compare | 2089 | 2075-2129 | 0.75 | 9377 | 9352-9411 | 0.93 |
| single-49208-csa-cp8r4 | cute_ws@compare | 992.8 | 986.7-997.3 | 0.36 | 8211 | 8196-8217 | 0.81 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 1117 | 1113-1119 | 0.40 | - | - | - |
| single-49208-csa-cp8r7 | tilelang@main | 2745 | 2736-2763 | 1.00 | 10361 | 10302-10387 | 1.00 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1105 | 1101-1108 | 0.40 | - | - | - |
| single-49208-csa-cp8r7 | tilelang@compare | 2752 | 2744-2776 | 1.00 | 10465 | 10453-10498 | 1.00 |
| single-49208-csa-cp8r7 | cudnn_flashmla@compare | 1178 | 1174-1187 | 0.43 | 5333 | 5315-5336 | 0.51 |
| single-49208-csa-cp8r7 | cute@compare | 2069 | 2059-2126 | 0.75 | 9698 | 9639-9755 | 0.93 |
| single-49208-csa-cp8r7 | cute_ws@compare | 988.8 | 987.9-1003 | 0.36 | 8526 | 8507-8570 | 0.81 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 1111 | 1108-1113 | 0.40 | - | - | - |
| single-49208-hca-cp1 | tilelang@main | 11496 | 11468-11639 | 1.00 | 42559 | 42547-42587 | 1.00 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 5402 | 5373-5651 | 0.47 | - | - | - |
| single-49208-hca-cp1 | tilelang@compare | 11537 | 11488-11608 | 1.00 | 42644 | 42592-42838 | 1.00 |
| single-49208-hca-cp1 | cudnn_flashmla@compare | 5450 | 5446-5458 | 0.47 | 25013 | 24995-25097 | 0.59 |
| single-49208-hca-cp1 | cute@compare | 10436 | 10422-10470 | 0.90 | 41533 | 41349-41602 | 0.97 |
| single-49208-hca-cp1 | cute_ws@compare | 5009 | 4890-5172 | 0.43 | 35916 | 35907-35919 | 0.84 |
| single-49208-hca-cp1 | flashmla_fwd_ref@compare | 5635 | 5598-5711 | 0.49 | - | - | - |
| single-49208-hca-cp8r0 | tilelang@main | 1710 | 1698-1758 | 1.00 | 4247 | 4246-4326 | 1.00 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 670.5 | 667.3-673.7 | 0.39 | - | - | - |
| single-49208-hca-cp8r0 | tilelang@compare | 1732 | 1722-1745 | 1.00 | 4294 | 4287-4303 | 1.00 |
| single-49208-hca-cp8r0 | cudnn_flashmla@compare | 749.4 | 741.6-749.7 | 0.43 | 2747 | 2743-2755 | 0.64 |
| single-49208-hca-cp8r0 | cute@compare | 1074 | 1068-1076 | 0.62 | 3588 | 3580-3597 | 0.84 |
| single-49208-hca-cp8r0 | cute_ws@compare | 527.7 | 524.6-528.5 | 0.30 | 3064 | 3057-3069 | 0.71 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 678.6 | 673.0-679.7 | 0.39 | - | - | - |
| single-49208-hca-cp8r4 | tilelang@main | 2145 | 2139-2186 | 1.00 | 6565 | 6561-6572 | 1.00 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 801.2 | 799.3-802.6 | 0.37 | - | - | - |
| single-49208-hca-cp8r4 | tilelang@compare | 2155 | 2149-2163 | 1.00 | 6614 | 6594-6622 | 1.00 |
| single-49208-hca-cp8r4 | cudnn_flashmla@compare | 873.0 | 868.2-883.0 | 0.41 | 3621 | 3612-3627 | 0.55 |
| single-49208-hca-cp8r4 | cute@compare | 1497 | 1496-1499 | 0.69 | 5890 | 5882-5897 | 0.89 |
| single-49208-hca-cp8r4 | cute_ws@compare | 670.1 | 668.6-670.7 | 0.31 | 5020 | 5018-5023 | 0.76 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 806.1 | 804.2-814.1 | 0.37 | - | - | - |
| single-49208-hca-cp8r7 | tilelang@main | 2430 | 2427-2432 | 1.00 | 8244 | 8226-8262 | 1.00 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 944.5 | 939.2-955.3 | 0.39 | - | - | - |
| single-49208-hca-cp8r7 | tilelang@compare | 2454 | 2445-2468 | 1.00 | 8238 | 8229-8258 | 1.00 |
| single-49208-hca-cp8r7 | cudnn_flashmla@compare | 1020 | 1012-1025 | 0.42 | 4363 | 4357-4399 | 0.53 |
| single-49208-hca-cp8r7 | cute@compare | 1783 | 1773-1788 | 0.73 | 7524 | 7510-7579 | 0.91 |
| single-49208-hca-cp8r7 | cute_ws@compare | 819.7 | 818.4-824.2 | 0.33 | 6522 | 6516-6549 | 0.79 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 946.4 | 944.2-949.7 | 0.39 | - | - | - |
| single-49208-sliding-cp1 | tilelang@main | 7433 | 7429-7462 | 1.00 | 22888 | 22878-22898 | 1.00 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 3125 | 3112-3128 | 0.42 | - | - | - |
| single-49208-sliding-cp1 | tilelang@compare | 7463 | 7461-7479 | 1.00 | 22928 | 22922-22936 | 1.00 |
| single-49208-sliding-cp1 | cudnn_flashmla@compare | 3237 | 3233-3248 | 0.43 | 15384 | 15379-15390 | 0.67 |
| single-49208-sliding-cp1 | cute@compare | 6364 | 6356-6491 | 0.85 | 21820 | 21809-21830 | 0.95 |
| single-49208-sliding-cp1 | cute_ws@compare | 2427 | 2416-2435 | 0.33 | 17827 | 17827-17831 | 0.78 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3151 | 3134-3161 | 0.42 | - | - | - |
| single-49208-sliding-cp8r0 | tilelang@main | 1574 | 1566-1595 | 1.00 | 3745 | 3734-3748 | 1.00 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 543.0 | 540.3-547.6 | 0.35 | - | - | - |
| single-49208-sliding-cp8r0 | tilelang@compare | 1572 | 1564-1602 | 1.00 | 3769 | 3766-3781 | 1.00 |
| single-49208-sliding-cp8r0 | cudnn_flashmla@compare | 611.2 | 606.1-617.4 | 0.39 | 2559 | 2551-2567 | 0.68 |
| single-49208-sliding-cp8r0 | cute@compare | 931.3 | 930.1-932.9 | 0.59 | 3067 | 3066-3069 | 0.81 |
| single-49208-sliding-cp8r0 | cute_ws@compare | 368.3 | 368.0-371.1 | 0.23 | 2682 | 2674-2695 | 0.71 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 548.8 | 540.6-550.8 | 0.35 | - | - | - |
| single-49208-sliding-cp8r4 | tilelang@main | 1557 | 1554-1558 | 1.00 | 3783 | 3768-3790 | 1.00 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 539.1 | 537.2-545.8 | 0.35 | - | - | - |
| single-49208-sliding-cp8r4 | tilelang@compare | 1585 | 1576-1591 | 1.00 | 3814 | 3805-3825 | 1.00 |
| single-49208-sliding-cp8r4 | cudnn_flashmla@compare | 613.7 | 609.8-619.1 | 0.39 | 2555 | 2550-2560 | 0.67 |
| single-49208-sliding-cp8r4 | cute@compare | 941.2 | 935.0-942.6 | 0.59 | 3114 | 3111-3116 | 0.82 |
| single-49208-sliding-cp8r4 | cute_ws@compare | 369.2 | 367.4-370.9 | 0.23 | 2724 | 2715-2730 | 0.71 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 537.6 | 537.0-544.7 | 0.34 | - | - | - |
| single-49208-sliding-cp8r7 | tilelang@main | 1572 | 1565-1598 | 1.00 | 3792 | 3777-3798 | 1.00 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 540.1 | 536.8-541.1 | 0.34 | - | - | - |
| single-49208-sliding-cp8r7 | tilelang@compare | 1582 | 1574-1604 | 1.00 | 3813 | 3803-3824 | 1.00 |
| single-49208-sliding-cp8r7 | cudnn_flashmla@compare | 625.2 | 618.2-633.0 | 0.40 | 2573 | 2568-2602 | 0.67 |
| single-49208-sliding-cp8r7 | cute@compare | 939.1 | 933.7-942.3 | 0.59 | 3118 | 3112-3119 | 0.82 |
| single-49208-sliding-cp8r7 | cute_ws@compare | 370.5 | 368.4-371.7 | 0.23 | 2747 | 2737-2795 | 0.72 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 547.4 | 545.9-555.0 | 0.35 | - | - | - |
| short-49208-csa-cp1 | tilelang@main | 12988 | 12944-13072 | 1.00 | 51070 | 51061-51101 | 1.00 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 6227 | 6191-6569 | 0.48 | - | - | - |
| short-49208-csa-cp1 | tilelang@compare | 13128 | 13024-13223 | 1.00 | 51168 | 51139-51415 | 1.00 |
| short-49208-csa-cp1 | cudnn_flashmla@compare | 6301 | 6262-6353 | 0.48 | 29425 | 29359-29523 | 0.58 |
| short-49208-csa-cp1 | cute@compare | 11948 | 11932-12108 | 0.91 | 50177 | 49858-50192 | 0.98 |
| short-49208-csa-cp1 | cute_ws@compare | 5820 | 5772-6101 | 0.44 | 43755 | 43747-43774 | 0.86 |
| short-49208-csa-cp1 | flashmla_fwd_ref@compare | 6675 | 6640-6693 | 0.51 | - | - | - |
| short-49208-csa-cp8r0 | tilelang@main | 2037 | 2013-2042 | 1.00 | 6012 | 6007-6017 | 1.00 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 791.1 | 785.7-795.3 | 0.39 | - | - | - |
| short-49208-csa-cp8r0 | tilelang@compare | 2053 | 2052-2067 | 1.00 | 6038 | 6035-6059 | 1.00 |
| short-49208-csa-cp8r0 | cudnn_flashmla@compare | 865.1 | 858.7-867.6 | 0.42 | 3391 | 3387-3395 | 0.56 |
| short-49208-csa-cp8r0 | cute@compare | 1384 | 1379-1388 | 0.67 | 5330 | 5329-5332 | 0.88 |
| short-49208-csa-cp8r0 | cute_ws@compare | 661.1 | 659.7-663.1 | 0.32 | 4575 | 4570-4577 | 0.76 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 794.6 | 789.1-797.5 | 0.39 | - | - | - |
| short-49208-csa-cp8r4 | tilelang@main | 2386 | 2373-2409 | 1.00 | 8086 | 8074-8104 | 1.00 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 959.5 | 958.1-965.9 | 0.40 | - | - | - |
| short-49208-csa-cp8r4 | tilelang@compare | 2414 | 2410-2429 | 1.00 | 8112 | 8107-8135 | 1.00 |
| short-49208-csa-cp8r4 | cudnn_flashmla@compare | 1035 | 1031-1052 | 0.43 | 4340 | 4331-4344 | 0.53 |
| short-49208-csa-cp8r4 | cute@compare | 1749 | 1746-1758 | 0.72 | 7401 | 7397-7450 | 0.91 |
| short-49208-csa-cp8r4 | cute_ws@compare | 848.8 | 845.0-850.0 | 0.35 | 6465 | 6461-6501 | 0.80 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 966.8 | 963.7-972.4 | 0.40 | - | - | - |
| short-49208-csa-cp8r7 | tilelang@main | 2161 | 2150-2181 | 1.00 | 6809 | 6800-6810 | 1.00 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 858.3 | 856.4-860.1 | 0.40 | - | - | - |
| short-49208-csa-cp8r7 | tilelang@compare | 2168 | 2160-2196 | 1.00 | 6812 | 6810-6834 | 1.00 |
| short-49208-csa-cp8r7 | cudnn_flashmla@compare | 925.4 | 923.2-928.9 | 0.43 | 3749 | 3745-3759 | 0.55 |
| short-49208-csa-cp8r7 | cute@compare | 1513 | 1507-1518 | 0.70 | 6101 | 6099-6108 | 0.90 |
| short-49208-csa-cp8r7 | cute_ws@compare | 734.8 | 731.6-735.7 | 0.34 | 5278 | 5275-5281 | 0.77 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 857.9 | 855.0-862.4 | 0.40 | - | - | - |
| short-49208-hca-cp1 | tilelang@main | 8505 | 8498-8522 | 1.00 | 26175 | 26167-26201 | 1.00 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 4103 | 4093-4107 | 0.48 | - | - | - |
| short-49208-hca-cp1 | tilelang@compare | 8529 | 8488-8567 | 1.00 | 26194 | 26178-26197 | 1.00 |
| short-49208-hca-cp1 | cudnn_flashmla@compare | 4197 | 4191-4210 | 0.49 | 17407 | 17392-17417 | 0.66 |
| short-49208-hca-cp1 | cute@compare | 7376 | 7365-7419 | 0.86 | 25044 | 25041-25048 | 0.96 |
| short-49208-hca-cp1 | cute_ws@compare | 3572 | 3563-3593 | 0.42 | 21179 | 21174-21180 | 0.81 |
| short-49208-hca-cp1 | flashmla_fwd_ref@compare | 4115 | 4106-4121 | 0.48 | - | - | - |
| short-49208-hca-cp8r0 | tilelang@main | 1701 | 1696-1716 | 1.00 | 4068 | 4063-4077 | 1.00 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 646.6 | 644.3-648.9 | 0.38 | - | - | - |
| short-49208-hca-cp8r0 | tilelang@compare | 1742 | 1727-1749 | 1.00 | 4101 | 4098-4115 | 1.00 |
| short-49208-hca-cp8r0 | cudnn_flashmla@compare | 728.5 | 725.7-735.2 | 0.42 | 2642 | 2634-2649 | 0.64 |
| short-49208-hca-cp8r0 | cute@compare | 1072 | 1067-1073 | 0.62 | 3401 | 3401-3407 | 0.83 |
| short-49208-hca-cp8r0 | cute_ws@compare | 511.4 | 509.3-513.0 | 0.29 | 2915 | 2912-2919 | 0.71 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 666.0 | 662.3-666.9 | 0.38 | - | - | - |
| short-49208-hca-cp8r4 | tilelang@main | 1747 | 1727-1765 | 1.00 | 4187 | 4177-4188 | 1.00 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 665.3 | 661.2-667.1 | 0.38 | - | - | - |
| short-49208-hca-cp8r4 | tilelang@compare | 1744 | 1741-1751 | 1.00 | 4252 | 4241-4258 | 1.00 |
| short-49208-hca-cp8r4 | cudnn_flashmla@compare | 736.1 | 735.9-747.8 | 0.42 | 2722 | 2713-2729 | 0.64 |
| short-49208-hca-cp8r4 | cute@compare | 1092 | 1085-1094 | 0.63 | 3540 | 3537-3544 | 0.83 |
| short-49208-hca-cp8r4 | cute_ws@compare | 517.8 | 516.7-518.6 | 0.30 | 3028 | 3020-3051 | 0.71 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 668.2 | 661.8-671.5 | 0.38 | - | - | - |
| short-49208-hca-cp8r7 | tilelang@main | 1733 | 1727-1755 | 1.00 | 4169 | 4154-4176 | 1.00 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 664.6 | 659.7-670.8 | 0.38 | - | - | - |
| short-49208-hca-cp8r7 | tilelang@compare | 1753 | 1745-1770 | 1.00 | 4177 | 4166-4208 | 1.00 |
| short-49208-hca-cp8r7 | cudnn_flashmla@compare | 739.7 | 733.4-740.7 | 0.42 | 2657 | 2643-2707 | 0.64 |
| short-49208-hca-cp8r7 | cute@compare | 1085 | 1082-1088 | 0.62 | 3476 | 3468-3484 | 0.83 |
| short-49208-hca-cp8r7 | cute_ws@compare | 518.1 | 517.4-519.5 | 0.30 | 2952 | 2941-3034 | 0.71 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 669.4 | 665.6-677.1 | 0.38 | - | - | - |
| short-49208-sliding-cp1 | tilelang@main | 7423 | 7418-7435 | 1.00 | 22632 | 22622-22641 | 1.00 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 3123 | 3117-3127 | 0.42 | - | - | - |
| short-49208-sliding-cp1 | tilelang@compare | 7492 | 7481-7498 | 1.00 | 22663 | 22588-22671 | 1.00 |
| short-49208-sliding-cp1 | cudnn_flashmla@compare | 3249 | 3247-3254 | 0.43 | 15215 | 15175-15235 | 0.67 |
| short-49208-sliding-cp1 | cute@compare | 6346 | 6344-6448 | 0.85 | 21519 | 21515-21535 | 0.95 |
| short-49208-sliding-cp1 | cute_ws@compare | 2417 | 2410-2444 | 0.32 | 17570 | 17559-17579 | 0.78 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3153 | 3131-3197 | 0.42 | - | - | - |
| short-49208-sliding-cp8r0 | tilelang@main | 1565 | 1552-1569 | 1.00 | 3663 | 3651-3680 | 1.00 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 541.0 | 537.3-548.1 | 0.35 | - | - | - |
| short-49208-sliding-cp8r0 | tilelang@compare | 1565 | 1556-1575 | 1.00 | 3701 | 3698-3719 | 1.00 |
| short-49208-sliding-cp8r0 | cudnn_flashmla@compare | 611.2 | 610.8-619.1 | 0.39 | 2533 | 2530-2536 | 0.68 |
| short-49208-sliding-cp8r0 | cute@compare | 916.4 | 915.9-920.3 | 0.59 | 2998 | 2995-3000 | 0.81 |
| short-49208-sliding-cp8r0 | cute_ws@compare | 369.6 | 366.5-370.9 | 0.24 | 2649 | 2635-2659 | 0.72 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 543.4 | 539.9-552.3 | 0.35 | - | - | - |
| short-49208-sliding-cp8r4 | tilelang@main | 1569 | 1551-1574 | 1.00 | 3739 | 3720-3763 | 1.00 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 545.8 | 542.3-547.2 | 0.35 | - | - | - |
| short-49208-sliding-cp8r4 | tilelang@compare | 1570 | 1565-1589 | 1.00 | 3755 | 3744-3764 | 1.00 |
| short-49208-sliding-cp8r4 | cudnn_flashmla@compare | 614.9 | 613.2-621.5 | 0.39 | 2529 | 2527-2551 | 0.67 |
| short-49208-sliding-cp8r4 | cute@compare | 933.1 | 928.8-937.2 | 0.59 | 3062 | 3057-3069 | 0.82 |
| short-49208-sliding-cp8r4 | cute_ws@compare | 367.6 | 367.0-371.2 | 0.23 | 2685 | 2676-2689 | 0.71 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 547.6 | 545.2-551.1 | 0.35 | - | - | - |
| short-49208-sliding-cp8r7 | tilelang@main | 1556 | 1551-1565 | 1.00 | 3743 | 3735-3746 | 1.00 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 536.9 | 535.8-539.6 | 0.34 | - | - | - |
| short-49208-sliding-cp8r7 | tilelang@compare | 1585 | 1572-1609 | 1.00 | 3766 | 3757-3781 | 1.00 |
| short-49208-sliding-cp8r7 | cudnn_flashmla@compare | 622.3 | 614.0-626.5 | 0.39 | 2547 | 2537-2560 | 0.68 |
| short-49208-sliding-cp8r7 | cute@compare | 934.3 | 933.0-937.4 | 0.59 | 3049 | 3048-3058 | 0.81 |
| short-49208-sliding-cp8r7 | cute_ws@compare | 369.6 | 367.7-370.2 | 0.23 | 2659 | 2658-2667 | 0.71 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 549.2 | 545.0-556.2 | 0.35 | - | - | - |
| heavy-49208-csa-cp1 | tilelang@main | 14914 | 14896-14994 | 1.00 | 61708 | 61688-61710 | 1.00 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 7394 | 7345-7676 | 0.50 | - | - | - |
| heavy-49208-csa-cp1 | tilelang@compare | 14941 | 14851-15040 | 1.00 | 61745 | 61715-62080 | 1.00 |
| heavy-49208-csa-cp1 | cudnn_flashmla@compare | 7401 | 7370-7431 | 0.50 | 34671 | 34624-34731 | 0.56 |
| heavy-49208-csa-cp1 | cute@compare | 13843 | 13783-13848 | 0.93 | 60823 | 60403-60858 | 0.99 |
| heavy-49208-csa-cp1 | cute_ws@compare | 6875 | 6689-7313 | 0.46 | 53457 | 53444-53470 | 0.87 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@compare | 7726 | 7688-7783 | 0.52 | - | - | - |
| heavy-49208-csa-cp8r0 | tilelang@main | 1925 | 1921-1931 | 1.00 | 5420 | 5416-5427 | 1.00 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 759.8 | 758.7-765.0 | 0.39 | - | - | - |
| heavy-49208-csa-cp8r0 | tilelang@compare | 1938 | 1927-1957 | 1.00 | 5446 | 5440-5456 | 1.00 |
| heavy-49208-csa-cp8r0 | cudnn_flashmla@compare | 828.2 | 822.7-833.4 | 0.43 | 3141 | 3139-3148 | 0.58 |
| heavy-49208-csa-cp8r0 | cute@compare | 1279 | 1276-1283 | 0.66 | 4763 | 4756-4775 | 0.87 |
| heavy-49208-csa-cp8r0 | cute_ws@compare | 620.7 | 616.1-620.8 | 0.32 | 4050 | 4047-4057 | 0.74 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 754.9 | 752.9-769.9 | 0.39 | - | - | - |
| heavy-49208-csa-cp8r4 | tilelang@main | 2629 | 2613-2647 | 1.00 | 9369 | 9360-9377 | 1.00 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1058 | 1054-1062 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r4 | tilelang@compare | 2652 | 2626-2659 | 1.00 | 9444 | 9438-9494 | 1.00 |
| heavy-49208-csa-cp8r4 | cudnn_flashmla@compare | 1145 | 1142-1150 | 0.43 | 4924 | 4913-4936 | 0.52 |
| heavy-49208-csa-cp8r4 | cute@compare | 1963 | 1961-2020 | 0.74 | 8698 | 8686-8732 | 0.92 |
| heavy-49208-csa-cp8r4 | cute_ws@compare | 938.0 | 938.0-946.4 | 0.35 | 7613 | 7588-7619 | 0.81 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 1068 | 1065-1072 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r7 | tilelang@main | 2477 | 2464-2492 | 1.00 | 8582 | 8578-8592 | 1.00 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 990.5 | 988.4-996.9 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r7 | tilelang@compare | 2489 | 2487-2495 | 1.00 | 8650 | 8617-8687 | 1.00 |
| heavy-49208-csa-cp8r7 | cudnn_flashmla@compare | 1066 | 1060-1070 | 0.43 | 4542 | 4539-4550 | 0.53 |
| heavy-49208-csa-cp8r7 | cute@compare | 1830 | 1826-1842 | 0.74 | 7939 | 7888-7944 | 0.92 |
| heavy-49208-csa-cp8r7 | cute_ws@compare | 875.7 | 872.7-878.9 | 0.35 | 6923 | 6892-6933 | 0.80 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 989.8 | 984.7-990.6 | 0.40 | - | - | - |
| heavy-49208-hca-cp1 | tilelang@main | 9089 | 9066-9107 | 1.00 | 29605 | 29574-29619 | 1.00 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4274 | 4269-4284 | 0.47 | - | - | - |
| heavy-49208-hca-cp1 | tilelang@compare | 9105 | 9105-9109 | 1.00 | 29679 | 29660-29687 | 1.00 |
| heavy-49208-hca-cp1 | cudnn_flashmla@compare | 4419 | 4415-4429 | 0.49 | 19011 | 19002-19022 | 0.64 |
| heavy-49208-hca-cp1 | cute@compare | 7981 | 7923-8076 | 0.88 | 28502 | 28495-28519 | 0.96 |
| heavy-49208-hca-cp1 | cute_ws@compare | 3694 | 3680-3755 | 0.41 | 24211 | 24208-24212 | 0.82 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@compare | 4369 | 4346-4384 | 0.48 | - | - | - |
| heavy-49208-hca-cp8r0 | tilelang@main | 1702 | 1695-1730 | 1.00 | 3912 | 3899-3948 | 1.00 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 642.8 | 634.5-652.1 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r0 | tilelang@compare | 1700 | 1697-1702 | 1.00 | 3958 | 3943-3960 | 1.00 |
| heavy-49208-hca-cp8r0 | cudnn_flashmla@compare | 713.3 | 710.0-718.3 | 0.42 | 2588 | 2571-2601 | 0.65 |
| heavy-49208-hca-cp8r0 | cute@compare | 1035 | 1032-1037 | 0.61 | 3240 | 3237-3247 | 0.82 |
| heavy-49208-hca-cp8r0 | cute_ws@compare | 486.7 | 486.2-487.0 | 0.29 | 2788 | 2768-2802 | 0.70 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 645.1 | 640.8-648.6 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r4 | tilelang@main | 1753 | 1742-1768 | 1.00 | 4393 | 4384-4416 | 1.00 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 675.0 | 669.0-676.8 | 0.39 | - | - | - |
| heavy-49208-hca-cp8r4 | tilelang@compare | 1769 | 1765-1778 | 1.00 | 4428 | 4421-4428 | 1.00 |
| heavy-49208-hca-cp8r4 | cudnn_flashmla@compare | 739.4 | 735.1-746.2 | 0.42 | 2767 | 2756-2770 | 0.62 |
| heavy-49208-hca-cp8r4 | cute@compare | 1105 | 1104-1107 | 0.62 | 3715 | 3712-3721 | 0.84 |
| heavy-49208-hca-cp8r4 | cute_ws@compare | 527.8 | 525.3-528.4 | 0.30 | 3181 | 3172-3198 | 0.72 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 670.0 | 669.5-673.8 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r7 | tilelang@main | 1845 | 1815-1849 | 1.00 | 4673 | 4665-4690 | 1.00 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 712.1 | 708.3-714.2 | 0.39 | - | - | - |
| heavy-49208-hca-cp8r7 | tilelang@compare | 1829 | 1821-1836 | 1.00 | 4651 | 4647-4661 | 1.00 |
| heavy-49208-hca-cp8r7 | cudnn_flashmla@compare | 780.0 | 774.2-782.3 | 0.43 | 2845 | 2836-2849 | 0.61 |
| heavy-49208-hca-cp8r7 | cute@compare | 1175 | 1171-1187 | 0.64 | 3960 | 3956-3962 | 0.85 |
| heavy-49208-hca-cp8r7 | cute_ws@compare | 570.2 | 568.6-571.1 | 0.31 | 3348 | 3344-3359 | 0.72 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 708.4 | 706.3-714.9 | 0.39 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang@main | 7448 | 7405-7474 | 1.00 | 22583 | 22581-22614 | 1.00 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 3135 | 3122-3149 | 0.42 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang@compare | 7456 | 7455-7472 | 1.00 | 22622 | 22596-22625 | 1.00 |
| heavy-49208-sliding-cp1 | cudnn_flashmla@compare | 3254 | 3240-3254 | 0.44 | 15231 | 15223-15237 | 0.67 |
| heavy-49208-sliding-cp1 | cute@compare | 6351 | 6345-6437 | 0.85 | 21489 | 21485-21501 | 0.95 |
| heavy-49208-sliding-cp1 | cute_ws@compare | 2432 | 2419-2454 | 0.33 | 17545 | 17544-17550 | 0.78 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3136 | 3119-3140 | 0.42 | - | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@main | 1572 | 1556-1586 | 1.00 | 3588 | 3579-3605 | 1.00 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 542.2 | 536.7-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@compare | 1567 | 1565-1579 | 1.00 | 3620 | 3613-3659 | 1.00 |
| heavy-49208-sliding-cp8r0 | cudnn_flashmla@compare | 613.0 | 611.3-614.0 | 0.39 | 2499 | 2483-2508 | 0.69 |
| heavy-49208-sliding-cp8r0 | cute@compare | 920.4 | 919.2-925.6 | 0.59 | 2911 | 2907-2922 | 0.80 |
| heavy-49208-sliding-cp8r0 | cute_ws@compare | 377.6 | 375.2-378.4 | 0.24 | 2565 | 2549-2585 | 0.71 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 548.5 | 546.0-550.5 | 0.35 | - | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@main | 1565 | 1560-1583 | 1.00 | 3761 | 3760-3770 | 1.00 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 538.9 | 537.2-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@compare | 1603 | 1591-1630 | 1.00 | 3805 | 3801-3837 | 1.00 |
| heavy-49208-sliding-cp8r4 | cudnn_flashmla@compare | 613.7 | 611.8-616.8 | 0.38 | 2553 | 2547-2571 | 0.67 |
| heavy-49208-sliding-cp8r4 | cute@compare | 938.1 | 936.0-946.4 | 0.59 | 3107 | 3104-3117 | 0.82 |
| heavy-49208-sliding-cp8r4 | cute_ws@compare | 368.7 | 367.1-371.5 | 0.23 | 2730 | 2718-2777 | 0.72 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 546.1 | 543.3-548.7 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@main | 1567 | 1558-1572 | 1.00 | 3766 | 3757-3772 | 1.00 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 543.2 | 540.1-547.6 | 0.35 | - | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@compare | 1585 | 1579-1604 | 1.00 | 3799 | 3796-3814 | 1.00 |
| heavy-49208-sliding-cp8r7 | cudnn_flashmla@compare | 617.9 | 616.8-627.0 | 0.39 | 2559 | 2551-2591 | 0.67 |
| heavy-49208-sliding-cp8r7 | cute@compare | 933.4 | 933.1-936.0 | 0.59 | 3083 | 3076-3084 | 0.81 |
| heavy-49208-sliding-cp8r7 | cute_ws@compare | 369.1 | 367.3-370.7 | 0.23 | 2706 | 2699-2711 | 0.71 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 551.9 | 546.3-554.0 | 0.35 | - | - | - |
| tiny-49208-csa-cp1 | tilelang@main | 8393 | 8360-8427 | 1.00 | 24104 | 24092-24115 | 1.00 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 4295 | 4284-4297 | 0.51 | - | - | - |
| tiny-49208-csa-cp1 | tilelang@compare | 8376 | 8371-8381 | 1.00 | 24160 | 24146-24170 | 1.00 |
| tiny-49208-csa-cp1 | cudnn_flashmla@compare | 4337 | 4332-4353 | 0.52 | 16270 | 16260-16277 | 0.67 |
| tiny-49208-csa-cp1 | cute@compare | 7363 | 7359-7363 | 0.88 | 23055 | 23054-23059 | 0.95 |
| tiny-49208-csa-cp1 | cute_ws@compare | 3717 | 3710-3721 | 0.44 | 19429 | 19421-19436 | 0.80 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@compare | 4274 | 4269-4278 | 0.51 | - | - | - |
| tiny-49208-csa-cp8r0 | tilelang@main | 1704 | 1673-1720 | 1.00 | 3875 | 3869-3881 | 1.00 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 658.2 | 656.7-659.3 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r0 | tilelang@compare | 1694 | 1679-1708 | 1.00 | 3896 | 3892-3911 | 1.00 |
| tiny-49208-csa-cp8r0 | cudnn_flashmla@compare | 735.4 | 728.5-745.2 | 0.43 | 2476 | 2465-2495 | 0.64 |
| tiny-49208-csa-cp8r0 | cute@compare | 1045 | 1044-1054 | 0.62 | 3217 | 3214-3219 | 0.83 |
| tiny-49208-csa-cp8r0 | cute_ws@compare | 531.4 | 528.9-532.1 | 0.31 | 2689 | 2682-2696 | 0.69 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 656.1 | 653.6-657.9 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r4 | tilelang@main | 1680 | 1674-1688 | 1.00 | 3906 | 3895-3913 | 1.00 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 656.4 | 651.7-664.8 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r4 | tilelang@compare | 1689 | 1688-1691 | 1.00 | 3924 | 3918-3937 | 1.00 |
| tiny-49208-csa-cp8r4 | cudnn_flashmla@compare | 730.0 | 726.2-732.8 | 0.43 | 2501 | 2498-2506 | 0.64 |
| tiny-49208-csa-cp8r4 | cute@compare | 1045 | 1044-1049 | 0.62 | 3248 | 3243-3258 | 0.83 |
| tiny-49208-csa-cp8r4 | cute_ws@compare | 534.4 | 529.8-536.1 | 0.32 | 2712 | 2711-2719 | 0.69 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 659.2 | 656.8-661.1 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r7 | tilelang@main | 1689 | 1670-1693 | 1.00 | 3912 | 3898-3930 | 1.00 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 659.4 | 655.1-660.4 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r7 | tilelang@compare | 1688 | 1679-1691 | 1.00 | 3925 | 3920-3930 | 1.00 |
| tiny-49208-csa-cp8r7 | cudnn_flashmla@compare | 726.6 | 725.8-729.5 | 0.43 | 2508 | 2494-2561 | 0.64 |
| tiny-49208-csa-cp8r7 | cute@compare | 1047 | 1042-1049 | 0.62 | 3224 | 3217-3233 | 0.82 |
| tiny-49208-csa-cp8r7 | cute_ws@compare | 531.1 | 529.4-532.4 | 0.31 | 2711 | 2709-2790 | 0.69 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 656.4 | 653.1-663.3 | 0.39 | - | - | - |
| tiny-49208-hca-cp1 | tilelang@main | 6744 | 6728-6770 | 1.00 | 16786 | 16781-16790 | 1.00 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 3164 | 3162-3181 | 0.47 | - | - | - |
| tiny-49208-hca-cp1 | tilelang@compare | 6706 | 6702-6730 | 1.00 | 16763 | 16760-16787 | 1.00 |
| tiny-49208-hca-cp1 | cudnn_flashmla@compare | 3285 | 3277-3288 | 0.49 | 12808 | 12787-12812 | 0.76 |
| tiny-49208-hca-cp1 | cute@compare | 5621 | 5617-5632 | 0.84 | 15666 | 15661-15680 | 0.93 |
| tiny-49208-hca-cp1 | cute_ws@compare | 2742 | 2734-2743 | 0.41 | 12738 | 12730-12760 | 0.76 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@compare | 3171 | 3158-3176 | 0.47 | - | - | - |
| tiny-49208-hca-cp8r0 | tilelang@main | 1479 | 1468-1496 | 1.00 | 2952 | 2948-2958 | 1.00 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 541.4 | 536.4-544.8 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r0 | tilelang@compare | 1552 | 1540-1565 | 1.00 | 2967 | 2962-2968 | 1.00 |
| tiny-49208-hca-cp8r0 | cudnn_flashmla@compare | 637.4 | 631.2-638.5 | 0.41 | 2208 | 2197-2212 | 0.74 |
| tiny-49208-hca-cp8r0 | cute@compare | 843.1 | 838.4-851.2 | 0.54 | 2283 | 2279-2293 | 0.77 |
| tiny-49208-hca-cp8r0 | cute_ws@compare | 406.4 | 400.2-410.9 | 0.26 | 1987 | 1983-1998 | 0.67 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 561.5 | 557.1-564.5 | 0.36 | - | - | - |
| tiny-49208-hca-cp8r4 | tilelang@main | 1485 | 1472-1546 | 1.00 | 2981 | 2974-2984 | 1.00 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 544.0 | 539.4-559.9 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r4 | tilelang@compare | 1495 | 1490-1510 | 1.00 | 3011 | 2994-3032 | 1.00 |
| tiny-49208-hca-cp8r4 | cudnn_flashmla@compare | 618.9 | 611.9-629.6 | 0.41 | 2209 | 2206-2227 | 0.73 |
| tiny-49208-hca-cp8r4 | cute@compare | 842.5 | 840.5-847.2 | 0.56 | 2306 | 2304-2309 | 0.77 |
| tiny-49208-hca-cp8r4 | cute_ws@compare | 401.3 | 400.4-403.2 | 0.27 | 2015 | 2002-2029 | 0.67 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 544.6 | 541.5-545.3 | 0.36 | - | - | - |
| tiny-49208-hca-cp8r7 | tilelang@main | 1474 | 1461-1497 | 1.00 | 2953 | 2945-2961 | 1.00 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 540.6 | 538.1-542.2 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r7 | tilelang@compare | 1464 | 1460-1480 | 1.00 | 2960 | 2955-2968 | 1.00 |
| tiny-49208-hca-cp8r7 | cudnn_flashmla@compare | 607.0 | 605.0-607.8 | 0.41 | 2185 | 2176-2198 | 0.74 |
| tiny-49208-hca-cp8r7 | cute@compare | 835.4 | 834.2-836.4 | 0.57 | 2253 | 2252-2262 | 0.76 |
| tiny-49208-hca-cp8r7 | cute_ws@compare | 402.5 | 391.9-403.5 | 0.28 | 1969 | 1959-2017 | 0.67 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 541.1 | 537.7-542.2 | 0.37 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang@main | 6733 | 6721-6755 | 1.00 | 16801 | 16796-16808 | 1.00 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 3173 | 3173-3185 | 0.47 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang@compare | 6720 | 6705-6767 | 1.00 | 16810 | 16808-16816 | 1.00 |
| tiny-49208-sliding-cp1 | cudnn_flashmla@compare | 3282 | 3268-3287 | 0.49 | 12788 | 12787-12793 | 0.76 |
| tiny-49208-sliding-cp1 | cute@compare | 5626 | 5621-5630 | 0.84 | 15667 | 15656-15673 | 0.93 |
| tiny-49208-sliding-cp1 | cute_ws@compare | 2745 | 2740-2761 | 0.41 | 12731 | 12727-12745 | 0.76 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3170 | 3162-3174 | 0.47 | - | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@main | 1463 | 1458-1465 | 1.00 | 2947 | 2942-2964 | 1.00 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 540.3 | 536.1-543.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@compare | 1474 | 1458-1482 | 1.00 | 3026 | 2968-3138 | 1.00 |
| tiny-49208-sliding-cp8r0 | cudnn_flashmla@compare | 609.5 | 605.8-613.2 | 0.41 | 2228 | 2207-2344 | 0.74 |
| tiny-49208-sliding-cp8r0 | cute@compare | 834.4 | 832.6-836.1 | 0.57 | 2286 | 2281-2352 | 0.76 |
| tiny-49208-sliding-cp8r0 | cute_ws@compare | 403.7 | 398.8-405.6 | 0.27 | 2026 | 1999-2162 | 0.67 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 545.2 | 543.4-545.9 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@main | 1485 | 1470-1510 | 1.00 | 2976 | 2975-2983 | 1.00 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 542.5 | 538.3-548.4 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@compare | 1530 | 1477-1561 | 1.00 | 3013 | 3007-3014 | 1.00 |
| tiny-49208-sliding-cp8r4 | cudnn_flashmla@compare | 617.5 | 609.9-630.3 | 0.40 | 2230 | 2223-2236 | 0.74 |
| tiny-49208-sliding-cp8r4 | cute@compare | 852.1 | 842.7-856.1 | 0.56 | 2314 | 2312-2316 | 0.77 |
| tiny-49208-sliding-cp8r4 | cute_ws@compare | 403.5 | 402.2-408.0 | 0.26 | 2015 | 2012-2021 | 0.67 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 545.7 | 543.8-557.4 | 0.36 | - | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@main | 1465 | 1463-1477 | 1.00 | 2951 | 2947-2968 | 1.00 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 542.3 | 536.3-545.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@compare | 1498 | 1473-1506 | 1.00 | 2967 | 2962-2986 | 1.00 |
| tiny-49208-sliding-cp8r7 | cudnn_flashmla@compare | 609.7 | 609.3-616.6 | 0.41 | 2179 | 2177-2194 | 0.73 |
| tiny-49208-sliding-cp8r7 | cute@compare | 837.2 | 833.7-845.5 | 0.56 | 2261 | 2260-2263 | 0.76 |
| tiny-49208-sliding-cp8r7 | cute_ws@compare | 394.6 | 391.0-401.9 | 0.26 | 1971 | 1953-1975 | 0.66 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 551.9 | 538.0-552.2 | 0.37 | - | - | - |
| single-65536-csa-cp1 | tilelang@main | 21694 | 21617-21724 | 1.00 | 95444 | 95324-95476 | 1.00 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 11385 | 11377-11470 | 0.52 | - | - | - |
| single-65536-csa-cp1 | tilelang@compare | 21715 | 21635-21753 | 1.00 | 95508 | 95450-95550 | 1.00 |
| single-65536-csa-cp1 | cudnn_flashmla@compare | 11464 | 11390-11495 | 0.53 | 53338 | 53208-53451 | 0.56 |
| single-65536-csa-cp1 | cute@compare | 20267 | 20197-20288 | 0.93 | 94008 | 93922-94067 | 0.98 |
| single-65536-csa-cp1 | cute_ws@compare | 10945 | 10912-10996 | 0.50 | 84000 | 83819-84054 | 0.88 |
| single-65536-csa-cp1 | flashmla_fwd_ref@compare | 11474 | 11473-11476 | 0.53 | - | - | - |
| single-65536-csa-cp8r0 | tilelang@main | 3265 | 3257-3300 | 1.00 | 11931 | 11922-11967 | 1.00 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 1325 | 1322-1327 | 0.41 | - | - | - |
| single-65536-csa-cp8r0 | tilelang@compare | 3235 | 3224-3269 | 1.00 | 12075 | 12066-12079 | 1.00 |
| single-65536-csa-cp8r0 | cudnn_flashmla@compare | 1400 | 1396-1410 | 0.43 | 6295 | 6287-6310 | 0.52 |
| single-65536-csa-cp8r0 | cute@compare | 2543 | 2536-2638 | 0.79 | 11321 | 11222-11334 | 0.94 |
| single-65536-csa-cp8r0 | cute_ws@compare | 1214 | 1212-1217 | 0.38 | 9924 | 9903-9944 | 0.82 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 1332 | 1330-1342 | 0.41 | - | - | - |
| single-65536-csa-cp8r4 | tilelang@main | 3429 | 3394-3504 | 1.00 | 13145 | 13143-13204 | 1.00 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1410 | 1405-1413 | 0.41 | - | - | - |
| single-65536-csa-cp8r4 | tilelang@compare | 3403 | 3387-3434 | 1.00 | 13246 | 13227-13335 | 1.00 |
| single-65536-csa-cp8r4 | cudnn_flashmla@compare | 1489 | 1475-1507 | 0.44 | 6839 | 6816-6850 | 0.52 |
| single-65536-csa-cp8r4 | cute@compare | 2702 | 2696-2873 | 0.79 | 12505 | 12445-12616 | 0.94 |
| single-65536-csa-cp8r4 | cute_ws@compare | 1294 | 1288-1342 | 0.38 | 11050 | 11028-11105 | 0.83 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 1420 | 1415-1442 | 0.42 | - | - | - |
| single-65536-csa-cp8r7 | tilelang@main | 3384 | 3367-3434 | 1.00 | 13996 | 13970-14040 | 1.00 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1410 | 1408-1414 | 0.42 | - | - | - |
| single-65536-csa-cp8r7 | tilelang@compare | 3431 | 3418-3475 | 1.00 | 14167 | 14130-14201 | 1.00 |
| single-65536-csa-cp8r7 | cudnn_flashmla@compare | 1507 | 1496-1510 | 0.44 | 7211 | 7202-7235 | 0.51 |
| single-65536-csa-cp8r7 | cute@compare | 2696 | 2692-2927 | 0.79 | 13402 | 13311-13450 | 0.95 |
| single-65536-csa-cp8r7 | cute_ws@compare | 1306 | 1304-1373 | 0.38 | 11914 | 11887-11951 | 0.84 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 1432 | 1425-1466 | 0.42 | - | - | - |
| single-65536-hca-cp1 | tilelang@main | 16503 | 16484-16545 | 1.00 | 64327 | 64312-64341 | 1.00 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 8196 | 7971-8458 | 0.50 | - | - | - |
| single-65536-hca-cp1 | tilelang@compare | 16548 | 16506-16583 | 1.00 | 64492 | 64349-64616 | 1.00 |
| single-65536-hca-cp1 | cudnn_flashmla@compare | 8296 | 8197-8322 | 0.50 | 37422 | 37406-37431 | 0.58 |
| single-65536-hca-cp1 | cute@compare | 15304 | 15268-15319 | 0.92 | 63169 | 62990-63241 | 0.98 |
| single-65536-hca-cp1 | cute_ws@compare | 7652 | 7537-7856 | 0.46 | 55060 | 55024-55075 | 0.85 |
| single-65536-hca-cp1 | flashmla_fwd_ref@compare | 8429 | 8403-8468 | 0.51 | - | - | - |
| single-65536-hca-cp8r0 | tilelang@main | 2074 | 2053-2077 | 1.00 | 5434 | 5425-5441 | 1.00 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 835.2 | 832.6-836.7 | 0.40 | - | - | - |
| single-65536-hca-cp8r0 | tilelang@compare | 2068 | 2043-2075 | 1.00 | 5451 | 5445-5464 | 1.00 |
| single-65536-hca-cp8r0 | cudnn_flashmla@compare | 914.6 | 909.0-919.1 | 0.44 | 3432 | 3421-3435 | 0.63 |
| single-65536-hca-cp8r0 | cute@compare | 1377 | 1371-1379 | 0.67 | 4736 | 4726-4739 | 0.87 |
| single-65536-hca-cp8r0 | cute_ws@compare | 674.0 | 673.5-674.3 | 0.33 | 3985 | 3983-3987 | 0.73 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 835.5 | 830.8-848.0 | 0.40 | - | - | - |
| single-65536-hca-cp8r4 | tilelang@main | 2797 | 2781-2799 | 1.00 | 9582 | 9559-9586 | 1.00 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1185 | 1183-1190 | 0.42 | - | - | - |
| single-65536-hca-cp8r4 | tilelang@compare | 2818 | 2808-2851 | 1.00 | 9638 | 9623-9660 | 1.00 |
| single-65536-hca-cp8r4 | cudnn_flashmla@compare | 1274 | 1272-1282 | 0.45 | 5277 | 5254-5287 | 0.55 |
| single-65536-hca-cp8r4 | cute@compare | 2127 | 2121-2139 | 0.75 | 8849 | 8846-8853 | 0.92 |
| single-65536-hca-cp8r4 | cute_ws@compare | 1066 | 1064-1073 | 0.38 | 7720 | 7713-7734 | 0.80 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 1202 | 1194-1204 | 0.43 | - | - | - |
| single-65536-hca-cp8r7 | tilelang@main | 3390 | 3380-3403 | 1.00 | 12689 | 12685-12694 | 1.00 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 1384 | 1378-1390 | 0.41 | - | - | - |
| single-65536-hca-cp8r7 | tilelang@compare | 3411 | 3396-3442 | 1.00 | 12790 | 12780-12815 | 1.00 |
| single-65536-hca-cp8r7 | cudnn_flashmla@compare | 1471 | 1453-1483 | 0.43 | 6568 | 6561-6616 | 0.51 |
| single-65536-hca-cp8r7 | cute@compare | 2703 | 2694-2804 | 0.79 | 12013 | 11942-12054 | 0.94 |
| single-65536-hca-cp8r7 | cute_ws@compare | 1271 | 1266-1278 | 0.37 | 10542 | 10536-10555 | 0.82 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 1393 | 1388-1413 | 0.41 | - | - | - |
| single-65536-sliding-cp1 | tilelang@main | 9650 | 9637-9666 | 1.00 | 30146 | 30143-30155 | 1.00 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4122 | 4112-4130 | 0.43 | - | - | - |
| single-65536-sliding-cp1 | tilelang@compare | 9710 | 9700-9757 | 1.00 | 30154 | 30143-30158 | 1.00 |
| single-65536-sliding-cp1 | cudnn_flashmla@compare | 4233 | 4222-4251 | 0.44 | 20288 | 20268-20298 | 0.67 |
| single-65536-sliding-cp1 | cute@compare | 8422 | 8398-8619 | 0.87 | 28887 | 28863-28900 | 0.96 |
| single-65536-sliding-cp1 | cute_ws@compare | 3223 | 3212-3264 | 0.33 | 23640 | 23635-23659 | 0.78 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4238 | 4172-4302 | 0.44 | - | - | - |
| single-65536-sliding-cp8r0 | tilelang@main | 1871 | 1858-1899 | 1.00 | 4658 | 4655-4673 | 1.00 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 661.3 | 655.0-663.8 | 0.35 | - | - | - |
| single-65536-sliding-cp8r0 | tilelang@compare | 1871 | 1853-1876 | 1.00 | 4718 | 4706-4727 | 1.00 |
| single-65536-sliding-cp8r0 | cudnn_flashmla@compare | 731.5 | 728.4-744.9 | 0.39 | 3082 | 3058-3114 | 0.65 |
| single-65536-sliding-cp8r0 | cute@compare | 1185 | 1185-1190 | 0.63 | 3990 | 3985-3997 | 0.85 |
| single-65536-sliding-cp8r0 | cute_ws@compare | 462.3 | 461.9-465.3 | 0.25 | 3243 | 3239-3256 | 0.69 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 664.2 | 659.6-667.9 | 0.36 | - | - | - |
| single-65536-sliding-cp8r4 | tilelang@main | 1843 | 1830-1855 | 1.00 | 4694 | 4692-4707 | 1.00 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 660.7 | 659.4-664.0 | 0.36 | - | - | - |
| single-65536-sliding-cp8r4 | tilelang@compare | 1860 | 1857-1875 | 1.00 | 4748 | 4741-4754 | 1.00 |
| single-65536-sliding-cp8r4 | cudnn_flashmla@compare | 734.7 | 732.0-736.3 | 0.39 | 3068 | 3064-3098 | 0.65 |
| single-65536-sliding-cp8r4 | cute@compare | 1191 | 1188-1193 | 0.64 | 4027 | 4021-4037 | 0.85 |
| single-65536-sliding-cp8r4 | cute_ws@compare | 462.4 | 461.9-463.8 | 0.25 | 3294 | 3281-3300 | 0.69 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 660.3 | 656.4-663.8 | 0.35 | - | - | - |
| single-65536-sliding-cp8r7 | tilelang@main | 1852 | 1844-1863 | 1.00 | 4716 | 4715-4727 | 1.00 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 662.4 | 658.8-665.8 | 0.36 | - | - | - |
| single-65536-sliding-cp8r7 | tilelang@compare | 1861 | 1846-1862 | 1.00 | 4744 | 4739-4754 | 1.00 |
| single-65536-sliding-cp8r7 | cudnn_flashmla@compare | 732.5 | 730.4-742.1 | 0.39 | 3052 | 3049-3097 | 0.64 |
| single-65536-sliding-cp8r7 | cute@compare | 1192 | 1187-1196 | 0.64 | 4017 | 4014-4021 | 0.85 |
| single-65536-sliding-cp8r7 | cute_ws@compare | 464.6 | 463.7-465.2 | 0.25 | 3275 | 3270-3280 | 0.69 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 661.2 | 658.5-665.8 | 0.36 | - | - | - |
| short-65536-csa-cp1 | tilelang@main | 16242 | 16220-16289 | 1.00 | 63170 | 63160-63183 | 1.00 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 8146 | 8080-8320 | 0.50 | - | - | - |
| short-65536-csa-cp1 | tilelang@compare | 16346 | 16219-16357 | 1.00 | 63215 | 63167-63481 | 1.00 |
| short-65536-csa-cp1 | cudnn_flashmla@compare | 8142 | 8056-8154 | 0.50 | 36742 | 36715-36785 | 0.58 |
| short-65536-csa-cp1 | cute@compare | 14997 | 14970-15025 | 0.92 | 62075 | 61782-62105 | 0.98 |
| short-65536-csa-cp1 | cute_ws@compare | 7663 | 7463-7679 | 0.47 | 54090 | 54079-54097 | 0.86 |
| short-65536-csa-cp1 | flashmla_fwd_ref@compare | 8325 | 8280-8395 | 0.51 | - | - | - |
| short-65536-csa-cp8r0 | tilelang@main | 2287 | 2281-2291 | 1.00 | 6780 | 6771-6782 | 1.00 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 907.8 | 902.6-914.0 | 0.40 | - | - | - |
| short-65536-csa-cp8r0 | tilelang@compare | 2297 | 2284-2337 | 1.00 | 6827 | 6823-6839 | 1.00 |
| short-65536-csa-cp8r0 | cudnn_flashmla@compare | 988.1 | 984.7-993.4 | 0.43 | 3970 | 3970-3980 | 0.58 |
| short-65536-csa-cp8r0 | cute@compare | 1622 | 1618-1628 | 0.71 | 6093 | 6077-6104 | 0.89 |
| short-65536-csa-cp8r0 | cute_ws@compare | 772.2 | 770.7-773.9 | 0.34 | 5197 | 5189-5203 | 0.76 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 911.9 | 910.8-916.0 | 0.40 | - | - | - |
| short-65536-csa-cp8r4 | tilelang@main | 2862 | 2860-2870 | 1.00 | 10076 | 10047-10086 | 1.00 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1170 | 1169-1173 | 0.41 | - | - | - |
| short-65536-csa-cp8r4 | tilelang@compare | 2872 | 2867-2887 | 1.00 | 10171 | 10109-10177 | 1.00 |
| short-65536-csa-cp8r4 | cudnn_flashmla@compare | 1253 | 1247-1256 | 0.44 | 5430 | 5426-5451 | 0.53 |
| short-65536-csa-cp8r4 | cute@compare | 2206 | 2200-2241 | 0.77 | 9347 | 9335-9382 | 0.92 |
| short-65536-csa-cp8r4 | cute_ws@compare | 1051 | 1048-1056 | 0.37 | 8152 | 8140-8164 | 0.80 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 1180 | 1176-1185 | 0.41 | - | - | - |
| short-65536-csa-cp8r7 | tilelang@main | 2664 | 2649-2673 | 1.00 | 8942 | 8934-8970 | 1.00 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1088 | 1087-1093 | 0.41 | - | - | - |
| short-65536-csa-cp8r7 | tilelang@compare | 2706 | 2694-2730 | 1.00 | 8957 | 8942-8978 | 1.00 |
| short-65536-csa-cp8r7 | cudnn_flashmla@compare | 1172 | 1168-1179 | 0.43 | 4923 | 4920-4937 | 0.55 |
| short-65536-csa-cp8r7 | cute@compare | 2007 | 2001-2021 | 0.74 | 8193 | 8191-8203 | 0.91 |
| short-65536-csa-cp8r7 | cute_ws@compare | 962.2 | 957.3-963.5 | 0.36 | 7102 | 7097-7106 | 0.79 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 1099 | 1095-1105 | 0.41 | - | - | - |
| short-65536-hca-cp1 | tilelang@main | 10922 | 10906-10936 | 1.00 | 33532 | 33528-33548 | 1.00 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5378 | 5372-5387 | 0.49 | - | - | - |
| short-65536-hca-cp1 | tilelang@compare | 10955 | 10948-10986 | 1.00 | 33593 | 33587-33605 | 1.00 |
| short-65536-hca-cp1 | cudnn_flashmla@compare | 5485 | 5480-5510 | 0.50 | 22681 | 22676-22685 | 0.68 |
| short-65536-hca-cp1 | cute@compare | 9677 | 9643-9722 | 0.88 | 32261 | 32249-32270 | 0.96 |
| short-65536-hca-cp1 | cute_ws@compare | 4739 | 4729-4796 | 0.43 | 27293 | 27281-27307 | 0.81 |
| short-65536-hca-cp1 | flashmla_fwd_ref@compare | 5402 | 5391-5407 | 0.49 | - | - | - |
| short-65536-hca-cp8r0 | tilelang@main | 2020 | 2012-2043 | 1.00 | 5027 | 5026-5032 | 1.00 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 811.2 | 808.9-814.9 | 0.40 | - | - | - |
| short-65536-hca-cp8r0 | tilelang@compare | 2031 | 2017-2036 | 1.00 | 5051 | 5048-5055 | 1.00 |
| short-65536-hca-cp8r0 | cudnn_flashmla@compare | 871.6 | 868.6-885.9 | 0.43 | 3185 | 3182-3195 | 0.63 |
| short-65536-hca-cp8r0 | cute@compare | 1345 | 1344-1351 | 0.66 | 4345 | 4340-4348 | 0.86 |
| short-65536-hca-cp8r0 | cute_ws@compare | 642.3 | 641.7-644.6 | 0.32 | 3576 | 3576-3579 | 0.71 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 803.7 | 801.6-804.8 | 0.40 | - | - | - |
| short-65536-hca-cp8r4 | tilelang@main | 2054 | 2049-2060 | 1.00 | 5198 | 5193-5210 | 1.00 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 832.4 | 828.5-836.2 | 0.41 | - | - | - |
| short-65536-hca-cp8r4 | tilelang@compare | 2080 | 2078-2082 | 1.00 | 5268 | 5261-5273 | 1.00 |
| short-65536-hca-cp8r4 | cudnn_flashmla@compare | 901.5 | 899.8-909.8 | 0.43 | 3323 | 3316-3351 | 0.63 |
| short-65536-hca-cp8r4 | cute@compare | 1385 | 1383-1387 | 0.67 | 4523 | 4516-4555 | 0.86 |
| short-65536-hca-cp8r4 | cute_ws@compare | 665.5 | 663.2-668.3 | 0.32 | 3734 | 3733-3781 | 0.71 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 838.3 | 827.8-844.6 | 0.40 | - | - | - |
| short-65536-hca-cp8r7 | tilelang@main | 2040 | 2035-2045 | 1.00 | 5167 | 5154-5172 | 1.00 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 819.8 | 815.1-823.4 | 0.40 | - | - | - |
| short-65536-hca-cp8r7 | tilelang@compare | 2051 | 2047-2086 | 1.00 | 5168 | 5162-5179 | 1.00 |
| short-65536-hca-cp8r7 | cudnn_flashmla@compare | 886.5 | 883.4-889.2 | 0.43 | 3261 | 3256-3266 | 0.63 |
| short-65536-hca-cp8r7 | cute@compare | 1370 | 1368-1373 | 0.67 | 4452 | 4445-4458 | 0.86 |
| short-65536-hca-cp8r7 | cute_ws@compare | 656.2 | 655.1-664.1 | 0.32 | 3678 | 3676-3683 | 0.71 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 820.8 | 817.5-829.5 | 0.40 | - | - | - |
| short-65536-sliding-cp1 | tilelang@main | 9612 | 9605-9628 | 1.00 | 29641 | 29634-29651 | 1.00 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4105 | 4097-4107 | 0.43 | - | - | - |
| short-65536-sliding-cp1 | tilelang@compare | 9658 | 9639-9666 | 1.00 | 29676 | 29670-29681 | 1.00 |
| short-65536-sliding-cp1 | cudnn_flashmla@compare | 4222 | 4217-4232 | 0.44 | 20007 | 19998-20020 | 0.67 |
| short-65536-sliding-cp1 | cute@compare | 8368 | 8360-8681 | 0.87 | 28391 | 28385-28402 | 0.96 |
| short-65536-sliding-cp1 | cute_ws@compare | 3231 | 3206-3274 | 0.33 | 23194 | 23190-23202 | 0.78 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4122 | 4111-4143 | 0.43 | - | - | - |
| short-65536-sliding-cp8r0 | tilelang@main | 1863 | 1854-1872 | 1.00 | 4557 | 4556-4558 | 1.00 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 665.9 | 661.3-668.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r0 | tilelang@compare | 1852 | 1845-1866 | 1.00 | 4583 | 4582-4591 | 1.00 |
| short-65536-sliding-cp8r0 | cudnn_flashmla@compare | 735.3 | 732.0-740.6 | 0.40 | 2990 | 2986-2998 | 0.65 |
| short-65536-sliding-cp8r0 | cute@compare | 1175 | 1174-1178 | 0.63 | 3865 | 3857-3871 | 0.84 |
| short-65536-sliding-cp8r0 | cute_ws@compare | 466.3 | 464.5-467.1 | 0.25 | 3144 | 3128-3159 | 0.69 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 664.8 | 661.9-669.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r4 | tilelang@main | 1838 | 1829-1848 | 1.00 | 4624 | 4622-4646 | 1.00 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 660.4 | 656.3-663.9 | 0.36 | - | - | - |
| short-65536-sliding-cp8r4 | tilelang@compare | 1840 | 1837-1859 | 1.00 | 4680 | 4661-4686 | 1.00 |
| short-65536-sliding-cp8r4 | cudnn_flashmla@compare | 737.6 | 735.8-745.3 | 0.40 | 3032 | 3024-3036 | 0.65 |
| short-65536-sliding-cp8r4 | cute@compare | 1186 | 1182-1187 | 0.64 | 3943 | 3940-3946 | 0.84 |
| short-65536-sliding-cp8r4 | cute_ws@compare | 463.6 | 461.9-465.3 | 0.25 | 3204 | 3203-3219 | 0.68 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 666.1 | 659.9-668.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r7 | tilelang@main | 1841 | 1834-1848 | 1.00 | 4633 | 4620-4639 | 1.00 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 662.2 | 661.1-666.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r7 | tilelang@compare | 1833 | 1828-1852 | 1.00 | 4639 | 4629-4647 | 1.00 |
| short-65536-sliding-cp8r7 | cudnn_flashmla@compare | 730.1 | 728.2-740.6 | 0.40 | 3004 | 3001-3015 | 0.65 |
| short-65536-sliding-cp8r7 | cute@compare | 1183 | 1181-1188 | 0.65 | 3915 | 3906-3926 | 0.84 |
| short-65536-sliding-cp8r7 | cute_ws@compare | 465.2 | 462.4-465.9 | 0.25 | 3184 | 3178-3187 | 0.69 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 661.4 | 658.5-665.3 | 0.36 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 17373 | 17342-17419 | 1.00 | 69696 | 69684-69701 | 1.00 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8820 | 8665-9037 | 0.51 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@compare | 17457 | 17369-17461 | 1.00 | 69808 | 69743-69896 | 1.00 |
| heavy-65536-csa-cp1 | cudnn_flashmla@compare | 8811 | 8760-8895 | 0.50 | 40160 | 39926-40294 | 0.58 |
| heavy-65536-csa-cp1 | cute@compare | 16178 | 16112-16186 | 0.93 | 68453 | 68317-68523 | 0.98 |
| heavy-65536-csa-cp1 | cute_ws@compare | 8375 | 8173-8583 | 0.48 | 60104 | 60098-60124 | 0.86 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@compare | 9040 | 9002-9052 | 0.52 | - | - | - |
| heavy-65536-csa-cp8r0 | tilelang@main | 2238 | 2232-2242 | 1.00 | 6420 | 6417-6435 | 1.00 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 900.9 | 888.9-902.5 | 0.40 | - | - | - |
| heavy-65536-csa-cp8r0 | tilelang@compare | 2249 | 2244-2272 | 1.00 | 6475 | 6472-6481 | 1.00 |
| heavy-65536-csa-cp8r0 | cudnn_flashmla@compare | 971.0 | 968.6-977.1 | 0.43 | 3820 | 3810-3825 | 0.59 |
| heavy-65536-csa-cp8r0 | cute@compare | 1579 | 1573-1583 | 0.70 | 5747 | 5745-5754 | 0.89 |
| heavy-65536-csa-cp8r0 | cute_ws@compare | 763.3 | 761.0-765.5 | 0.34 | 4897 | 4884-4898 | 0.76 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 902.7 | 898.6-905.2 | 0.40 | - | - | - |
| heavy-65536-csa-cp8r4 | tilelang@main | 3403 | 3379-3442 | 1.00 | 12973 | 12936-12999 | 1.00 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1402 | 1397-1405 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r4 | tilelang@compare | 3396 | 3390-3429 | 1.00 | 13092 | 13080-13114 | 1.00 |
| heavy-65536-csa-cp8r4 | cudnn_flashmla@compare | 1494 | 1483-1503 | 0.44 | 6779 | 6771-6799 | 0.52 |
| heavy-65536-csa-cp8r4 | cute@compare | 2772 | 2691-2903 | 0.82 | 12346 | 12248-12368 | 0.94 |
| heavy-65536-csa-cp8r4 | cute_ws@compare | 1297 | 1289-1355 | 0.38 | 10878 | 10875-10882 | 0.83 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 1426 | 1414-1435 | 0.42 | - | - | - |
| heavy-65536-csa-cp8r7 | tilelang@main | 2550 | 2543-2560 | 1.00 | 8299 | 8297-8303 | 1.00 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1048 | 1046-1052 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r7 | tilelang@compare | 2595 | 2585-2595 | 1.00 | 8347 | 8340-8435 | 1.00 |
| heavy-65536-csa-cp8r7 | cudnn_flashmla@compare | 1133 | 1131-1150 | 0.44 | 4679 | 4672-4705 | 0.56 |
| heavy-65536-csa-cp8r7 | cute@compare | 1906 | 1904-1907 | 0.73 | 7605 | 7601-7631 | 0.91 |
| heavy-65536-csa-cp8r7 | cute_ws@compare | 917.7 | 916.5-919.1 | 0.35 | 6578 | 6576-6583 | 0.79 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 1062 | 1058-1065 | 0.41 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 11714 | 11677-11736 | 1.00 | 38272 | 38241-38292 | 1.00 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5682 | 5674-5693 | 0.49 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@compare | 11771 | 11715-11788 | 1.00 | 38343 | 38316-38492 | 1.00 |
| heavy-65536-hca-cp1 | cudnn_flashmla@compare | 5860 | 5847-5886 | 0.50 | 24881 | 24877-24900 | 0.65 |
| heavy-65536-hca-cp1 | cute@compare | 10553 | 10480-10616 | 0.90 | 37190 | 37013-37197 | 0.97 |
| heavy-65536-hca-cp1 | cute_ws@compare | 5012 | 4952-5102 | 0.43 | 31453 | 31449-31473 | 0.82 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@compare | 5798 | 5746-5820 | 0.49 | - | - | - |
| heavy-65536-hca-cp8r0 | tilelang@main | 1991 | 1967-1996 | 1.00 | 4821 | 4813-4828 | 1.00 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 777.0 | 775.5-781.9 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r0 | tilelang@compare | 1982 | 1976-2017 | 1.00 | 4855 | 4850-4859 | 1.00 |
| heavy-65536-hca-cp8r0 | cudnn_flashmla@compare | 848.7 | 844.1-857.6 | 0.43 | 3102 | 3097-3106 | 0.64 |
| heavy-65536-hca-cp8r0 | cute@compare | 1304 | 1301-1309 | 0.66 | 4128 | 4125-4142 | 0.85 |
| heavy-65536-hca-cp8r0 | cute_ws@compare | 608.8 | 607.5-610.9 | 0.31 | 3365 | 3363-3377 | 0.69 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 777.8 | 775.0-782.7 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r4 | tilelang@main | 2378 | 2363-2384 | 1.00 | 7050 | 7045-7053 | 1.00 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 928.0 | 924.2-932.5 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r4 | tilelang@compare | 2395 | 2382-2402 | 1.00 | 7108 | 7102-7120 | 1.00 |
| heavy-65536-hca-cp8r4 | cudnn_flashmla@compare | 1003 | 998.0-1005 | 0.42 | 4060 | 4052-4067 | 0.57 |
| heavy-65536-hca-cp8r4 | cute@compare | 1690 | 1686-1696 | 0.71 | 6358 | 6348-6362 | 0.89 |
| heavy-65536-hca-cp8r4 | cute_ws@compare | 771.5 | 771.0-772.7 | 0.32 | 5383 | 5377-5387 | 0.76 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 934.8 | 932.6-939.1 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r7 | tilelang@main | 2006 | 2002-2013 | 1.00 | 5010 | 5008-5012 | 1.00 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 799.1 | 797.0-808.2 | 0.40 | - | - | - |
| heavy-65536-hca-cp8r7 | tilelang@compare | 2040 | 2024-2056 | 1.00 | 5049 | 5026-5051 | 1.00 |
| heavy-65536-hca-cp8r7 | cudnn_flashmla@compare | 875.7 | 870.1-879.2 | 0.43 | 3216 | 3213-3221 | 0.64 |
| heavy-65536-hca-cp8r7 | cute@compare | 1346 | 1343-1348 | 0.66 | 4297 | 4293-4308 | 0.85 |
| heavy-65536-hca-cp8r7 | cute_ws@compare | 640.9 | 637.6-642.2 | 0.31 | 3536 | 3531-3540 | 0.70 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 797.9 | 795.4-800.4 | 0.39 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 9616 | 9585-9660 | 1.00 | 29282 | 29271-29295 | 1.00 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4111 | 4108-4115 | 0.43 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@compare | 9620 | 9610-9625 | 1.00 | 29274 | 29266-29297 | 1.00 |
| heavy-65536-sliding-cp1 | cudnn_flashmla@compare | 4231 | 4225-4240 | 0.44 | 19870 | 19856-19892 | 0.68 |
| heavy-65536-sliding-cp1 | cute@compare | 8333 | 8330-8477 | 0.87 | 27999 | 27993-28013 | 0.96 |
| heavy-65536-sliding-cp1 | cute_ws@compare | 3227 | 3208-3276 | 0.34 | 22836 | 22830-22845 | 0.78 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4163 | 4141-4186 | 0.43 | - | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@main | 1853 | 1835-1865 | 1.00 | 4405 | 4399-4411 | 1.00 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 669.7 | 664.8-674.5 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@compare | 1838 | 1827-1845 | 1.00 | 4450 | 4447-4478 | 1.00 |
| heavy-65536-sliding-cp8r0 | cudnn_flashmla@compare | 726.4 | 724.6-730.3 | 0.40 | 2933 | 2922-2949 | 0.66 |
| heavy-65536-sliding-cp8r0 | cute@compare | 1163 | 1159-1165 | 0.63 | 3738 | 3733-3747 | 0.84 |
| heavy-65536-sliding-cp8r0 | cute_ws@compare | 468.0 | 465.6-471.9 | 0.25 | 3006 | 2996-3010 | 0.68 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 663.8 | 662.3-668.6 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@main | 1840 | 1825-1842 | 1.00 | 4678 | 4674-4682 | 1.00 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 655.9 | 655.0-657.4 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@compare | 1857 | 1852-1867 | 1.00 | 4741 | 4737-4749 | 1.00 |
| heavy-65536-sliding-cp8r4 | cudnn_flashmla@compare | 732.9 | 727.8-737.3 | 0.39 | 3050 | 3047-3053 | 0.64 |
| heavy-65536-sliding-cp8r4 | cute@compare | 1194 | 1192-1196 | 0.64 | 4022 | 4018-4028 | 0.85 |
| heavy-65536-sliding-cp8r4 | cute_ws@compare | 462.7 | 461.2-465.0 | 0.25 | 3273 | 3256-3293 | 0.69 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 664.6 | 661.1-666.2 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@main | 1824 | 1817-1827 | 1.00 | 4524 | 4521-4533 | 1.00 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 665.7 | 659.8-668.6 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@compare | 1858 | 1850-1869 | 1.00 | 4548 | 4544-4556 | 1.00 |
| heavy-65536-sliding-cp8r7 | cudnn_flashmla@compare | 732.5 | 727.7-738.0 | 0.39 | 2984 | 2950-3022 | 0.66 |
| heavy-65536-sliding-cp8r7 | cute@compare | 1182 | 1181-1183 | 0.64 | 3832 | 3824-3838 | 0.84 |
| heavy-65536-sliding-cp8r7 | cute_ws@compare | 466.1 | 464.6-469.9 | 0.25 | 3128 | 3117-3167 | 0.69 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 664.6 | 660.7-668.4 | 0.36 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 10884 | 10872-10885 | 1.00 | 31800 | 31799-31808 | 1.00 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5659 | 5656-5664 | 0.52 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@compare | 10920 | 10904-10932 | 1.00 | 31834 | 31827-31853 | 1.00 |
| tiny-65536-csa-cp1 | cudnn_flashmla@compare | 5733 | 5728-5741 | 0.53 | 21544 | 21536-21561 | 0.68 |
| tiny-65536-csa-cp1 | cute@compare | 9776 | 9769-9784 | 0.90 | 30612 | 30609-30614 | 0.96 |
| tiny-65536-csa-cp1 | cute_ws@compare | 4921 | 4915-4935 | 0.45 | 25796 | 25783-25801 | 0.81 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@compare | 5667 | 5661-5682 | 0.52 | - | - | - |
| tiny-65536-csa-cp8r0 | tilelang@main | 2004 | 1988-2021 | 1.00 | 4891 | 4874-4905 | 1.00 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 809.2 | 808.8-818.0 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r0 | tilelang@compare | 2011 | 2006-2033 | 1.00 | 4927 | 4919-4936 | 1.00 |
| tiny-65536-csa-cp8r0 | cudnn_flashmla@compare | 888.7 | 883.6-892.2 | 0.44 | 3127 | 3124-3129 | 0.63 |
| tiny-65536-csa-cp8r0 | cute@compare | 1348 | 1341-1351 | 0.67 | 4235 | 4229-4239 | 0.86 |
| tiny-65536-csa-cp8r0 | cute_ws@compare | 680.3 | 678.1-683.1 | 0.34 | 3530 | 3528-3532 | 0.72 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 815.5 | 811.5-821.0 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r4 | tilelang@main | 2008 | 1980-2018 | 1.00 | 4867 | 4862-4872 | 1.00 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 823.2 | 820.0-828.9 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r4 | tilelang@compare | 2009 | 1996-2017 | 1.00 | 4908 | 4889-4926 | 1.00 |
| tiny-65536-csa-cp8r4 | cudnn_flashmla@compare | 890.3 | 886.6-894.8 | 0.44 | 3112 | 3103-3136 | 0.63 |
| tiny-65536-csa-cp8r4 | cute@compare | 1352 | 1344-1354 | 0.67 | 4200 | 4184-4232 | 0.86 |
| tiny-65536-csa-cp8r4 | cute_ws@compare | 684.8 | 682.8-690.2 | 0.34 | 3483 | 3478-3523 | 0.71 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 813.0 | 809.4-818.1 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r7 | tilelang@main | 2064 | 2023-2100 | 1.00 | 4878 | 4870-4892 | 1.00 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 831.3 | 809.5-891.6 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r7 | tilelang@compare | 2026 | 2009-2039 | 1.00 | 4915 | 4902-4945 | 1.00 |
| tiny-65536-csa-cp8r7 | cudnn_flashmla@compare | 886.3 | 882.5-889.2 | 0.44 | 3105 | 3097-3112 | 0.63 |
| tiny-65536-csa-cp8r7 | cute@compare | 1354 | 1351-1355 | 0.67 | 4212 | 4195-4218 | 0.86 |
| tiny-65536-csa-cp8r7 | cute_ws@compare | 682.7 | 682.3-684.4 | 0.34 | 3487 | 3473-3492 | 0.71 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 819.2 | 813.1-821.3 | 0.40 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 8689 | 8676-8718 | 1.00 | 22022 | 21996-22029 | 1.00 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4173 | 4168-4184 | 0.48 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@compare | 8689 | 8682-8708 | 1.00 | 22032 | 22025-22034 | 1.00 |
| tiny-65536-hca-cp1 | cudnn_flashmla@compare | 4276 | 4272-4304 | 0.49 | 16847 | 16832-16853 | 0.76 |
| tiny-65536-hca-cp1 | cute@compare | 7433 | 7424-7438 | 0.86 | 20744 | 20739-20746 | 0.94 |
| tiny-65536-hca-cp1 | cute_ws@compare | 3654 | 3648-3679 | 0.42 | 16952 | 16944-16960 | 0.77 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@compare | 4159 | 4156-4162 | 0.48 | - | - | - |
| tiny-65536-hca-cp8r0 | tilelang@main | 1706 | 1702-1728 | 1.00 | 3628 | 3625-3630 | 1.00 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 657.9 | 656.0-673.7 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r0 | tilelang@compare | 1719 | 1715-1731 | 1.00 | 3643 | 3641-3647 | 1.00 |
| tiny-65536-hca-cp8r0 | cudnn_flashmla@compare | 728.6 | 723.7-734.8 | 0.42 | 2578 | 2567-2586 | 0.71 |
| tiny-65536-hca-cp8r0 | cute@compare | 1058 | 1055-1061 | 0.62 | 2927 | 2923-2929 | 0.80 |
| tiny-65536-hca-cp8r0 | cute_ws@compare | 517.4 | 513.5-520.3 | 0.30 | 2352 | 2348-2352 | 0.65 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 660.8 | 658.2-670.1 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r4 | tilelang@main | 1732 | 1714-1738 | 1.00 | 3608 | 3597-3618 | 1.00 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 662.3 | 656.5-663.6 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r4 | tilelang@compare | 1735 | 1721-1741 | 1.00 | 3640 | 3640-3648 | 1.00 |
| tiny-65536-hca-cp8r4 | cudnn_flashmla@compare | 730.5 | 727.5-732.8 | 0.42 | 2599 | 2593-2603 | 0.71 |
| tiny-65536-hca-cp8r4 | cute@compare | 1071 | 1070-1074 | 0.62 | 2934 | 2933-2938 | 0.81 |
| tiny-65536-hca-cp8r4 | cute_ws@compare | 511.8 | 506.4-521.4 | 0.29 | 2340 | 2336-2349 | 0.64 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 667.5 | 662.0-675.2 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r7 | tilelang@main | 1716 | 1711-1718 | 1.00 | 3631 | 3624-3641 | 1.00 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 666.2 | 661.7-670.2 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r7 | tilelang@compare | 1723 | 1717-1742 | 1.00 | 3669 | 3650-3688 | 1.00 |
| tiny-65536-hca-cp8r7 | cudnn_flashmla@compare | 733.4 | 731.3-739.2 | 0.43 | 2604 | 2600-2620 | 0.71 |
| tiny-65536-hca-cp8r7 | cute@compare | 1066 | 1060-1070 | 0.62 | 2942 | 2939-2944 | 0.80 |
| tiny-65536-hca-cp8r7 | cute_ws@compare | 523.0 | 518.5-523.3 | 0.30 | 2354 | 2351-2360 | 0.64 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 666.4 | 663.0-668.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 8717 | 8700-8727 | 1.00 | 22036 | 22028-22065 | 1.00 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4178 | 4164-4186 | 0.48 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@compare | 8690 | 8659-8707 | 1.00 | 22015 | 22009-22025 | 1.00 |
| tiny-65536-sliding-cp1 | cudnn_flashmla@compare | 4271 | 4265-4284 | 0.49 | 16862 | 16854-16869 | 0.77 |
| tiny-65536-sliding-cp1 | cute@compare | 7431 | 7415-7438 | 0.86 | 20745 | 20738-20754 | 0.94 |
| tiny-65536-sliding-cp1 | cute_ws@compare | 3672 | 3655-3682 | 0.42 | 16929 | 16921-16935 | 0.77 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4167 | 4152-4173 | 0.48 | - | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@main | 1745 | 1740-1764 | 1.00 | 3631 | 3625-3633 | 1.00 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 668.3 | 661.3-672.0 | 0.38 | - | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@compare | 1742 | 1725-1763 | 1.00 | 3659 | 3645-3706 | 1.00 |
| tiny-65536-sliding-cp8r0 | cudnn_flashmla@compare | 738.6 | 730.0-743.6 | 0.42 | 2596 | 2578-2619 | 0.71 |
| tiny-65536-sliding-cp8r0 | cute@compare | 1062 | 1058-1063 | 0.61 | 2931 | 2928-2934 | 0.80 |
| tiny-65536-sliding-cp8r0 | cute_ws@compare | 522.8 | 516.6-523.1 | 0.30 | 2350 | 2349-2352 | 0.64 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 670.8 | 662.6-673.2 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@main | 1716 | 1698-1722 | 1.00 | 3613 | 3604-3624 | 1.00 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 662.2 | 654.3-664.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@compare | 1726 | 1721-1764 | 1.00 | 3655 | 3649-3662 | 1.00 |
| tiny-65536-sliding-cp8r4 | cudnn_flashmla@compare | 733.4 | 726.1-737.2 | 0.42 | 2602 | 2597-2605 | 0.71 |
| tiny-65536-sliding-cp8r4 | cute@compare | 1068 | 1066-1074 | 0.62 | 2936 | 2934-2938 | 0.80 |
| tiny-65536-sliding-cp8r4 | cute_ws@compare | 515.0 | 505.5-518.1 | 0.30 | 2344 | 2340-2346 | 0.64 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 663.5 | 660.9-665.7 | 0.38 | - | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@main | 1705 | 1695-1721 | 1.00 | 3629 | 3621-3632 | 1.00 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 665.0 | 659.9-666.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@compare | 1718 | 1711-1727 | 1.00 | 3659 | 3656-3660 | 1.00 |
| tiny-65536-sliding-cp8r7 | cudnn_flashmla@compare | 735.5 | 731.2-740.8 | 0.43 | 2585 | 2577-2601 | 0.71 |
| tiny-65536-sliding-cp8r7 | cute@compare | 1062 | 1060-1069 | 0.62 | 2936 | 2935-2945 | 0.80 |
| tiny-65536-sliding-cp8r7 | cute_ws@compare | 521.6 | 515.7-523.0 | 0.30 | 2357 | 2354-2359 | 0.64 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 663.0 | 659.5-665.0 | 0.39 | - | - | - |

GPU busy time per call in µs from profiler traces (lower is better); `host` is op-boundary minus GPU
busy time (launch overhead and gaps); `/TL` divides GPU busy time by tilelang's; `peak MiB` is the
allocation above the inputs during one call, forward+backward where the backend has it, else forward.

| item | backend | fwd gpu µs | fwd host | fwd /TL | f+b gpu µs | f+b host | f+b /TL | peak MiB |
|---|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 539.5 | 657.7 | 1.00 | 2161 | 1025 | 1.00 | 265 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 284.0 | 100.0 | 0.53 | - | - | - | 139 |
| single-2048-csa-cp1 | tilelang@compare | 536.8 | 673.3 | 1.00 | 2156 | 1110 | 1.00 | 265 |
| single-2048-csa-cp1 | cudnn_flashmla@compare | 291.0 | 146.6 | 0.54 | 1260 | 680.3 | 0.58 | 271 |
| single-2048-csa-cp1 | cute@compare | 508.5 | 99.0 | 0.95 | 2129 | 501.1 | 0.99 | 265 |
| single-2048-csa-cp1 | cute_ws@compare | 253.0 | 51.3 | 0.47 | 1875 | 610.9 | 0.87 | 265 |
| single-2048-csa-cp1 | flashmla_fwd_ref@compare | 281.0 | 100.5 | 0.52 | - | - | - | 139 |
| single-2048-csa-cp8r0 | tilelang@main | 60.1 | 673.4 | 1.00 | 221.7 | 1780 | 1.00 | 40 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.0 | 138.4 | 0.73 | - | - | - | 17 |
| single-2048-csa-cp8r0 | tilelang@compare | 60.1 | 700.7 | 1.00 | 220.8 | 1895 | 1.00 | 40 |
| single-2048-csa-cp8r0 | cudnn_flashmla@compare | 52.2 | 266.9 | 0.87 | 183.5 | 1289 | 0.83 | 40 |
| single-2048-csa-cp8r0 | cute@compare | 58.1 | 124.1 | 0.97 | 218.9 | 1242 | 0.99 | 40 |
| single-2048-csa-cp8r0 | cute_ws@compare | 36.3 | 49.8 | 0.60 | 198.6 | 1104 | 0.90 | 40 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 44.5 | 150.8 | 0.74 | - | - | - | 17 |
| single-2048-csa-cp8r4 | tilelang@main | 84.4 | 679.0 | 1.00 | 356.6 | 1672 | 1.00 | 40 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 57.8 | 139.9 | 0.68 | - | - | - | 17 |
| single-2048-csa-cp8r4 | tilelang@compare | 83.6 | 686.7 | 1.00 | 355.4 | 1771 | 1.00 | 40 |
| single-2048-csa-cp8r4 | cudnn_flashmla@compare | 65.3 | 245.4 | 0.78 | 248.9 | 1226 | 0.70 | 40 |
| single-2048-csa-cp8r4 | cute@compare | 80.5 | 119.7 | 0.96 | 351.1 | 1099 | 0.99 | 40 |
| single-2048-csa-cp8r4 | cute_ws@compare | 48.7 | 49.9 | 0.58 | 321.0 | 971.3 | 0.90 | 40 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 57.2 | 145.4 | 0.68 | - | - | - | 17 |
| single-2048-csa-cp8r7 | tilelang@main | 101.7 | 672.6 | 1.00 | 455.4 | 1569 | 1.00 | 40 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 64.1 | 138.7 | 0.63 | - | - | - | 17 |
| single-2048-csa-cp8r7 | tilelang@compare | 100.8 | 689.0 | 1.00 | 452.7 | 1663 | 1.00 | 40 |
| single-2048-csa-cp8r7 | cudnn_flashmla@compare | 71.5 | 243.9 | 0.71 | 291.9 | 1178 | 0.64 | 40 |
| single-2048-csa-cp8r7 | cute@compare | 97.4 | 118.8 | 0.97 | 448.9 | 1012 | 0.99 | 40 |
| single-2048-csa-cp8r7 | cute_ws@compare | 55.0 | 50.0 | 0.55 | 407.2 | 890.7 | 0.90 | 40 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 63.8 | 144.5 | 0.63 | - | - | - | 17 |
| single-2048-hca-cp1 | tilelang@main | 362.9 | 722.3 | 1.00 | 1166 | 1338 | 1.00 | 265 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 199.3 | 141.7 | 0.55 | - | - | - | 133 |
| single-2048-hca-cp1 | tilelang@compare | 361.6 | 740.8 | 1.00 | 1168 | 1400 | 1.00 | 265 |
| single-2048-hca-cp1 | cudnn_flashmla@compare | 207.8 | 196.1 | 0.57 | 807.7 | 777.6 | 0.69 | 267 |
| single-2048-hca-cp1 | cute@compare | 335.6 | 148.1 | 0.93 | 1145 | 774.3 | 0.98 | 265 |
| single-2048-hca-cp1 | cute_ws@compare | 169.9 | 52.3 | 0.47 | 975.9 | 755.7 | 0.84 | 265 |
| single-2048-hca-cp1 | flashmla_fwd_ref@compare | 198.8 | 143.5 | 0.55 | - | - | - | 133 |
| single-2048-hca-cp8r0 | tilelang@main | 57.6 | 706.3 | 1.00 | 206.3 | 1945 | 1.00 | 38 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 41.6 | 147.1 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r0 | tilelang@compare | 58.0 | 727.1 | 1.00 | 206.7 | 2003 | 1.00 | 38 |
| single-2048-hca-cp8r0 | cudnn_flashmla@compare | 49.4 | 272.6 | 0.85 | 172.7 | 1322 | 0.84 | 39 |
| single-2048-hca-cp8r0 | cute@compare | 55.4 | 142.2 | 0.95 | 203.1 | 1370 | 0.98 | 38 |
| single-2048-hca-cp8r0 | cute_ws@compare | 26.4 | 50.2 | 0.46 | 176.5 | 1182 | 0.85 | 38 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 41.3 | 148.1 | 0.71 | - | - | - | 17 |
| single-2048-hca-cp8r4 | tilelang@main | 61.4 | 731.7 | 1.00 | 219.3 | 1938 | 1.00 | 38 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 44.9 | 149.0 | 0.73 | - | - | - | 17 |
| single-2048-hca-cp8r4 | tilelang@compare | 62.0 | 744.7 | 1.00 | 219.5 | 2002 | 1.00 | 38 |
| single-2048-hca-cp8r4 | cudnn_flashmla@compare | 53.0 | 263.7 | 0.85 | 183.0 | 1302 | 0.83 | 39 |
| single-2048-hca-cp8r4 | cute@compare | 59.9 | 144.5 | 0.97 | 216.9 | 1332 | 0.99 | 38 |
| single-2048-hca-cp8r4 | cute_ws@compare | 29.9 | 50.8 | 0.48 | 188.4 | 1218 | 0.86 | 38 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 44.9 | 150.3 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r7 | tilelang@main | 62.1 | 744.8 | 1.00 | 219.6 | 1958 | 1.00 | 38 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.2 | 148.3 | 0.73 | - | - | - | 17 |
| single-2048-hca-cp8r7 | tilelang@compare | 61.9 | 722.1 | 1.00 | 219.8 | 2001 | 1.00 | 38 |
| single-2048-hca-cp8r7 | cudnn_flashmla@compare | 52.5 | 258.0 | 0.85 | 184.1 | 1275 | 0.84 | 39 |
| single-2048-hca-cp8r7 | cute@compare | 59.6 | 140.8 | 0.96 | 218.2 | 1344 | 0.99 | 38 |
| single-2048-hca-cp8r7 | cute_ws@compare | 30.4 | 49.8 | 0.49 | 189.8 | 1169 | 0.86 | 38 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 45.0 | 145.5 | 0.73 | - | - | - | 17 |
| single-2048-sliding-cp1 | tilelang@main | 312.4 | 703.5 | 1.00 | 1033 | 1255 | 1.00 | 264 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 150.8 | 147.6 | 0.48 | - | - | - | 131 |
| single-2048-sliding-cp1 | tilelang@compare | 312.2 | 715.8 | 1.00 | 1027 | 1362 | 1.00 | 264 |
| single-2048-sliding-cp1 | cudnn_flashmla@compare | 160.9 | 201.1 | 0.52 | 714.0 | 815.7 | 0.69 | 266 |
| single-2048-sliding-cp1 | cute@compare | 290.5 | 120.0 | 0.93 | 1010 | 743.9 | 0.98 | 264 |
| single-2048-sliding-cp1 | cute_ws@compare | 116.9 | 54.4 | 0.37 | 840.2 | 758.1 | 0.82 | 264 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@compare | 148.7 | 145.1 | 0.48 | - | - | - | 131 |
| single-2048-sliding-cp8r0 | tilelang@main | 51.4 | 681.1 | 1.00 | 190.3 | 1857 | 1.00 | 38 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.6 | 144.1 | 0.69 | - | - | - | 16 |
| single-2048-sliding-cp8r0 | tilelang@compare | 51.5 | 707.4 | 1.00 | 191.3 | 1937 | 1.00 | 38 |
| single-2048-sliding-cp8r0 | cudnn_flashmla@compare | 43.7 | 268.8 | 0.85 | 160.5 | 1312 | 0.84 | 38 |
| single-2048-sliding-cp8r0 | cute@compare | 49.5 | 122.2 | 0.96 | 188.4 | 1300 | 0.99 | 38 |
| single-2048-sliding-cp8r0 | cute_ws@compare | 22.1 | 50.0 | 0.43 | 162.5 | 1147 | 0.85 | 38 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 35.9 | 143.1 | 0.70 | - | - | - | 16 |
| single-2048-sliding-cp8r4 | tilelang@main | 53.4 | 686.1 | 1.00 | 195.9 | 1822 | 1.00 | 38 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.5 | 143.5 | 0.66 | - | - | - | 16 |
| single-2048-sliding-cp8r4 | tilelang@compare | 53.4 | 694.7 | 1.00 | 196.3 | 1916 | 1.00 | 38 |
| single-2048-sliding-cp8r4 | cudnn_flashmla@compare | 44.5 | 267.4 | 0.83 | 164.2 | 1312 | 0.84 | 38 |
| single-2048-sliding-cp8r4 | cute@compare | 51.2 | 118.2 | 0.96 | 194.2 | 1276 | 0.99 | 38 |
| single-2048-sliding-cp8r4 | cute_ws@compare | 22.6 | 52.5 | 0.42 | 166.5 | 1137 | 0.85 | 38 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 36.1 | 144.1 | 0.67 | - | - | - | 16 |
| single-2048-sliding-cp8r7 | tilelang@main | 53.2 | 686.3 | 1.00 | 195.3 | 1838 | 1.00 | 38 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 36.2 | 144.7 | 0.68 | - | - | - | 16 |
| single-2048-sliding-cp8r7 | tilelang@compare | 53.3 | 706.8 | 1.00 | 195.5 | 1914 | 1.00 | 38 |
| single-2048-sliding-cp8r7 | cudnn_flashmla@compare | 44.1 | 274.0 | 0.83 | 164.4 | 1303 | 0.84 | 38 |
| single-2048-sliding-cp8r7 | cute@compare | 51.5 | 121.2 | 0.97 | 193.9 | 1262 | 0.99 | 38 |
| single-2048-sliding-cp8r7 | cute_ws@compare | 22.4 | 51.9 | 0.42 | 165.2 | 1122 | 0.85 | 38 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 36.1 | 151.2 | 0.68 | - | - | - | 16 |
| short-2048-csa-cp1 | tilelang@main | 442.8 | 688.9 | 1.00 | 1638 | 1118 | 1.00 | 265 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 234.8 | 133.2 | 0.53 | - | - | - | 139 |
| short-2048-csa-cp1 | tilelang@compare | 441.0 | 687.6 | 1.00 | 1633 | 1241 | 1.00 | 265 |
| short-2048-csa-cp1 | cudnn_flashmla@compare | 241.4 | 188.8 | 0.55 | 1013 | 732.9 | 0.62 | 271 |
| short-2048-csa-cp1 | cute@compare | 415.6 | 121.2 | 0.94 | 1608 | 614.6 | 0.98 | 265 |
| short-2048-csa-cp1 | cute_ws@compare | 208.8 | 52.5 | 0.47 | 1401 | 668.1 | 0.86 | 265 |
| short-2048-csa-cp1 | flashmla_fwd_ref@compare | 234.2 | 133.1 | 0.53 | - | - | - | 139 |
| short-2048-csa-cp8r0 | tilelang@main | 60.1 | 693.0 | 1.00 | 221.8 | 1801 | 1.00 | 40 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.4 | 143.4 | 0.74 | - | - | - | 17 |
| short-2048-csa-cp8r0 | tilelang@compare | 60.6 | 699.9 | 1.00 | 222.1 | 1894 | 1.00 | 40 |
| short-2048-csa-cp8r0 | cudnn_flashmla@compare | 52.4 | 263.2 | 0.86 | 183.6 | 1306 | 0.83 | 40 |
| short-2048-csa-cp8r0 | cute@compare | 58.0 | 121.2 | 0.96 | 219.1 | 1242 | 0.99 | 40 |
| short-2048-csa-cp8r0 | cute_ws@compare | 35.9 | 51.8 | 0.59 | 198.6 | 1109 | 0.89 | 40 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 44.2 | 144.9 | 0.73 | - | - | - | 17 |
| short-2048-csa-cp8r4 | tilelang@main | 67.8 | 674.7 | 1.00 | 262.9 | 1770 | 1.00 | 40 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 48.9 | 138.8 | 0.72 | - | - | - | 17 |
| short-2048-csa-cp8r4 | tilelang@compare | 67.4 | 688.7 | 1.00 | 261.9 | 1826 | 1.00 | 40 |
| short-2048-csa-cp8r4 | cudnn_flashmla@compare | 56.4 | 252.1 | 0.84 | 202.1 | 1243 | 0.77 | 40 |
| short-2048-csa-cp8r4 | cute@compare | 64.8 | 120.4 | 0.96 | 258.6 | 1176 | 0.99 | 40 |
| short-2048-csa-cp8r4 | cute_ws@compare | 41.2 | 48.1 | 0.61 | 236.6 | 1041 | 0.90 | 40 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 48.9 | 148.7 | 0.73 | - | - | - | 17 |
| short-2048-csa-cp8r7 | tilelang@main | 76.4 | 685.8 | 1.00 | 319.2 | 1718 | 1.00 | 40 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 50.9 | 140.6 | 0.67 | - | - | - | 17 |
| short-2048-csa-cp8r7 | tilelang@compare | 76.5 | 688.0 | 1.00 | 318.1 | 1767 | 1.00 | 40 |
| short-2048-csa-cp8r7 | cudnn_flashmla@compare | 58.7 | 248.9 | 0.77 | 224.6 | 1236 | 0.71 | 40 |
| short-2048-csa-cp8r7 | cute@compare | 73.6 | 124.5 | 0.96 | 315.2 | 1111 | 0.99 | 40 |
| short-2048-csa-cp8r7 | cute_ws@compare | 42.6 | 48.6 | 0.56 | 285.9 | 989.5 | 0.90 | 40 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 51.3 | 141.1 | 0.67 | - | - | - | 17 |
| short-2048-hca-cp1 | tilelang@main | 354.6 | 724.6 | 1.00 | 1143 | 1346 | 1.00 | 265 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 196.7 | 143.8 | 0.55 | - | - | - | 133 |
| short-2048-hca-cp1 | tilelang@compare | 354.7 | 717.6 | 1.00 | 1138 | 1375 | 1.00 | 265 |
| short-2048-hca-cp1 | cudnn_flashmla@compare | 205.9 | 196.7 | 0.58 | 789.1 | 770.7 | 0.69 | 267 |
| short-2048-hca-cp1 | cute@compare | 331.4 | 142.8 | 0.93 | 1110 | 776.1 | 0.98 | 265 |
| short-2048-hca-cp1 | cute_ws@compare | 166.4 | 53.2 | 0.47 | 952.6 | 720.0 | 0.84 | 265 |
| short-2048-hca-cp1 | flashmla_fwd_ref@compare | 195.5 | 145.4 | 0.55 | - | - | - | 133 |
| short-2048-hca-cp8r0 | tilelang@main | 57.2 | 730.2 | 1.00 | 205.4 | 1949 | 1.00 | 38 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 42.0 | 149.5 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r0 | tilelang@compare | 57.6 | 725.0 | 1.00 | 206.2 | 1993 | 1.00 | 38 |
| short-2048-hca-cp8r0 | cudnn_flashmla@compare | 49.1 | 267.0 | 0.85 | 170.7 | 1293 | 0.83 | 39 |
| short-2048-hca-cp8r0 | cute@compare | 55.4 | 143.3 | 0.96 | 202.7 | 1346 | 0.98 | 38 |
| short-2048-hca-cp8r0 | cute_ws@compare | 26.0 | 52.3 | 0.45 | 175.4 | 1164 | 0.85 | 38 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 41.2 | 154.1 | 0.71 | - | - | - | 17 |
| short-2048-hca-cp8r4 | tilelang@main | 58.6 | 736.4 | 1.00 | 211.8 | 1950 | 1.00 | 38 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 43.2 | 150.1 | 0.74 | - | - | - | 17 |
| short-2048-hca-cp8r4 | tilelang@compare | 58.3 | 719.5 | 1.00 | 211.4 | 2003 | 1.00 | 38 |
| short-2048-hca-cp8r4 | cudnn_flashmla@compare | 50.8 | 253.2 | 0.87 | 175.4 | 1296 | 0.83 | 39 |
| short-2048-hca-cp8r4 | cute@compare | 56.3 | 138.9 | 0.96 | 209.9 | 1344 | 0.99 | 38 |
| short-2048-hca-cp8r4 | cute_ws@compare | 26.8 | 51.6 | 0.46 | 182.0 | 1185 | 0.86 | 38 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 43.0 | 146.9 | 0.74 | - | - | - | 17 |
| short-2048-hca-cp8r7 | tilelang@main | 62.0 | 730.7 | 1.00 | 219.8 | 1973 | 1.00 | 38 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.4 | 143.5 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r7 | tilelang@compare | 62.1 | 730.3 | 1.00 | 219.0 | 2001 | 1.00 | 38 |
| short-2048-hca-cp8r7 | cudnn_flashmla@compare | 52.6 | 258.6 | 0.85 | 181.8 | 1288 | 0.83 | 39 |
| short-2048-hca-cp8r7 | cute@compare | 59.5 | 141.0 | 0.96 | 216.8 | 1329 | 0.99 | 38 |
| short-2048-hca-cp8r7 | cute_ws@compare | 29.2 | 51.2 | 0.47 | 187.6 | 1183 | 0.86 | 38 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 44.5 | 149.3 | 0.72 | - | - | - | 17 |
| short-2048-sliding-cp1 | tilelang@main | 307.0 | 695.7 | 1.00 | 1011 | 1319 | 1.00 | 264 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 147.7 | 145.6 | 0.48 | - | - | - | 131 |
| short-2048-sliding-cp1 | tilelang@compare | 307.2 | 691.0 | 1.00 | 1010 | 1345 | 1.00 | 264 |
| short-2048-sliding-cp1 | cudnn_flashmla@compare | 159.7 | 195.3 | 0.52 | 700.5 | 829.7 | 0.69 | 266 |
| short-2048-sliding-cp1 | cute@compare | 282.9 | 122.1 | 0.92 | 987.3 | 735.2 | 0.98 | 264 |
| short-2048-sliding-cp1 | cute_ws@compare | 118.5 | 51.0 | 0.39 | 822.2 | 757.2 | 0.81 | 264 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@compare | 150.8 | 140.0 | 0.49 | - | - | - | 131 |
| short-2048-sliding-cp8r0 | tilelang@main | 51.5 | 695.3 | 1.00 | 190.0 | 1852 | 1.00 | 38 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 36.3 | 145.5 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r0 | tilelang@compare | 51.5 | 699.7 | 1.00 | 190.7 | 1935 | 1.00 | 38 |
| short-2048-sliding-cp8r0 | cudnn_flashmla@compare | 43.7 | 259.8 | 0.85 | 162.2 | 1333 | 0.85 | 38 |
| short-2048-sliding-cp8r0 | cute@compare | 49.4 | 121.4 | 0.96 | 187.8 | 1290 | 0.98 | 38 |
| short-2048-sliding-cp8r0 | cute_ws@compare | 22.0 | 51.4 | 0.43 | 162.0 | 1142 | 0.85 | 38 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 36.3 | 145.9 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r4 | tilelang@main | 51.5 | 705.5 | 1.00 | 191.7 | 1831 | 1.00 | 38 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 36.2 | 153.6 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r4 | tilelang@compare | 51.7 | 679.6 | 1.00 | 191.4 | 1928 | 1.00 | 38 |
| short-2048-sliding-cp8r4 | cudnn_flashmla@compare | 44.1 | 262.1 | 0.85 | 161.2 | 1302 | 0.84 | 38 |
| short-2048-sliding-cp8r4 | cute@compare | 49.6 | 118.8 | 0.96 | 189.6 | 1267 | 0.99 | 38 |
| short-2048-sliding-cp8r4 | cute_ws@compare | 22.0 | 52.4 | 0.43 | 163.3 | 1124 | 0.85 | 38 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 35.9 | 147.6 | 0.69 | - | - | - | 16 |
| short-2048-sliding-cp8r7 | tilelang@main | 53.4 | 683.0 | 1.00 | 196.2 | 1848 | 1.00 | 38 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.6 | 149.5 | 0.67 | - | - | - | 16 |
| short-2048-sliding-cp8r7 | tilelang@compare | 53.1 | 685.9 | 1.00 | 196.5 | 1901 | 1.00 | 38 |
| short-2048-sliding-cp8r7 | cudnn_flashmla@compare | 44.1 | 270.8 | 0.83 | 163.9 | 1306 | 0.83 | 38 |
| short-2048-sliding-cp8r7 | cute@compare | 51.0 | 120.3 | 0.96 | 194.0 | 1264 | 0.99 | 38 |
| short-2048-sliding-cp8r7 | cute_ws@compare | 22.7 | 51.7 | 0.43 | 166.0 | 1118 | 0.84 | 38 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 36.2 | 143.6 | 0.68 | - | - | - | 16 |
| heavy-2048-csa-cp1 | tilelang@main | 476.0 | 694.7 | 1.00 | 1821 | 1134 | 1.00 | 265 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 254.4 | 135.3 | 0.53 | - | - | - | 139 |
| heavy-2048-csa-cp1 | tilelang@compare | 473.0 | 679.9 | 1.00 | 1816 | 1205 | 1.00 | 265 |
| heavy-2048-csa-cp1 | cudnn_flashmla@compare | 259.7 | 195.4 | 0.55 | 1095 | 716.7 | 0.60 | 271 |
| heavy-2048-csa-cp1 | cute@compare | 448.0 | 121.0 | 0.95 | 1790 | 573.3 | 0.99 | 265 |
| heavy-2048-csa-cp1 | cute_ws@compare | 223.8 | 54.0 | 0.47 | 1570 | 645.3 | 0.86 | 265 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@compare | 255.1 | 135.0 | 0.54 | - | - | - | 139 |
| heavy-2048-csa-cp8r0 | tilelang@main | 60.1 | 687.4 | 1.00 | 214.2 | 1829 | 1.00 | 40 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.1 | 145.0 | 0.73 | - | - | - | 17 |
| heavy-2048-csa-cp8r0 | tilelang@compare | 60.0 | 704.1 | 1.00 | 214.9 | 1921 | 1.00 | 40 |
| heavy-2048-csa-cp8r0 | cudnn_flashmla@compare | 51.7 | 259.5 | 0.86 | 177.1 | 1313 | 0.82 | 40 |
| heavy-2048-csa-cp8r0 | cute@compare | 57.7 | 121.1 | 0.96 | 212.8 | 1260 | 0.99 | 40 |
| heavy-2048-csa-cp8r0 | cute_ws@compare | 36.1 | 50.2 | 0.60 | 192.1 | 1132 | 0.89 | 40 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 44.3 | 144.3 | 0.74 | - | - | - | 17 |
| heavy-2048-csa-cp8r4 | tilelang@main | 75.4 | 702.2 | 1.00 | 312.4 | 1752 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 51.7 | 149.0 | 0.69 | - | - | - | 17 |
| heavy-2048-csa-cp8r4 | tilelang@compare | 75.4 | 689.4 | 1.00 | 309.6 | 1805 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | cudnn_flashmla@compare | 59.3 | 250.6 | 0.79 | 225.6 | 1235 | 0.73 | 40 |
| heavy-2048-csa-cp8r4 | cute@compare | 72.7 | 118.4 | 0.96 | 309.5 | 1154 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | cute_ws@compare | 42.4 | 51.1 | 0.56 | 279.8 | 1030 | 0.90 | 40 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 51.2 | 146.0 | 0.68 | - | - | - | 17 |
| heavy-2048-csa-cp8r7 | tilelang@main | 93.5 | 699.7 | 1.00 | 410.4 | 1624 | 1.00 | 40 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 60.9 | 148.8 | 0.65 | - | - | - | 17 |
| heavy-2048-csa-cp8r7 | tilelang@compare | 93.0 | 716.0 | 1.00 | 407.8 | 1734 | 1.00 | 40 |
| heavy-2048-csa-cp8r7 | cudnn_flashmla@compare | 68.5 | 241.4 | 0.74 | 271.3 | 1215 | 0.67 | 40 |
| heavy-2048-csa-cp8r7 | cute@compare | 89.2 | 118.5 | 0.96 | 404.2 | 1070 | 0.99 | 40 |
| heavy-2048-csa-cp8r7 | cute_ws@compare | 52.1 | 51.1 | 0.56 | 369.0 | 931.0 | 0.90 | 40 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 60.5 | 144.5 | 0.65 | - | - | - | 17 |
| heavy-2048-hca-cp1 | tilelang@main | 344.4 | 729.5 | 1.00 | 1109 | 1355 | 1.00 | 265 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 188.5 | 146.7 | 0.55 | - | - | - | 133 |
| heavy-2048-hca-cp1 | tilelang@compare | 343.7 | 737.6 | 1.00 | 1107 | 1391 | 1.00 | 265 |
| heavy-2048-hca-cp1 | cudnn_flashmla@compare | 198.1 | 200.4 | 0.58 | 769.7 | 775.7 | 0.70 | 267 |
| heavy-2048-hca-cp1 | cute@compare | 322.0 | 152.2 | 0.94 | 1084 | 765.6 | 0.98 | 265 |
| heavy-2048-hca-cp1 | cute_ws@compare | 159.3 | 54.9 | 0.46 | 924.2 | 749.6 | 0.83 | 265 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@compare | 188.1 | 150.1 | 0.55 | - | - | - | 133 |
| heavy-2048-hca-cp8r0 | tilelang@main | 54.0 | 720.3 | 1.00 | 187.5 | 1948 | 1.00 | 38 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 41.0 | 148.3 | 0.76 | - | - | - | 17 |
| heavy-2048-hca-cp8r0 | tilelang@compare | 54.0 | 721.8 | 1.00 | 187.0 | 2041 | 1.00 | 38 |
| heavy-2048-hca-cp8r0 | cudnn_flashmla@compare | 48.6 | 269.2 | 0.90 | 165.0 | 1301 | 0.88 | 39 |
| heavy-2048-hca-cp8r0 | cute@compare | 51.7 | 141.6 | 0.96 | 185.2 | 1375 | 0.99 | 38 |
| heavy-2048-hca-cp8r0 | cute_ws@compare | 25.2 | 51.7 | 0.47 | 160.2 | 1192 | 0.86 | 38 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 40.7 | 147.5 | 0.75 | - | - | - | 17 |
| heavy-2048-hca-cp8r4 | tilelang@main | 61.8 | 720.2 | 1.00 | 219.6 | 1903 | 1.00 | 38 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 45.0 | 144.6 | 0.73 | - | - | - | 17 |
| heavy-2048-hca-cp8r4 | tilelang@compare | 62.4 | 748.9 | 1.00 | 219.6 | 2015 | 1.00 | 38 |
| heavy-2048-hca-cp8r4 | cudnn_flashmla@compare | 52.9 | 268.5 | 0.85 | 182.8 | 1301 | 0.83 | 39 |
| heavy-2048-hca-cp8r4 | cute@compare | 59.7 | 144.7 | 0.96 | 217.4 | 1336 | 0.99 | 38 |
| heavy-2048-hca-cp8r4 | cute_ws@compare | 29.5 | 53.0 | 0.47 | 187.8 | 1163 | 0.86 | 38 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 45.1 | 149.9 | 0.72 | - | - | - | 17 |
| heavy-2048-hca-cp8r7 | tilelang@main | 61.9 | 725.0 | 1.00 | 219.4 | 1938 | 1.00 | 38 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.1 | 144.7 | 0.73 | - | - | - | 17 |
| heavy-2048-hca-cp8r7 | tilelang@compare | 61.9 | 728.2 | 1.00 | 219.4 | 2024 | 1.00 | 38 |
| heavy-2048-hca-cp8r7 | cudnn_flashmla@compare | 52.8 | 268.6 | 0.85 | 183.5 | 1292 | 0.84 | 39 |
| heavy-2048-hca-cp8r7 | cute@compare | 59.6 | 143.8 | 0.96 | 217.2 | 1345 | 0.99 | 38 |
| heavy-2048-hca-cp8r7 | cute_ws@compare | 29.8 | 53.3 | 0.48 | 188.5 | 1149 | 0.86 | 38 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 44.8 | 150.6 | 0.72 | - | - | - | 17 |
| heavy-2048-sliding-cp1 | tilelang@main | 305.0 | 695.3 | 1.00 | 1000 | 1294 | 1.00 | 264 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 150.5 | 141.3 | 0.49 | - | - | - | 131 |
| heavy-2048-sliding-cp1 | tilelang@compare | 303.2 | 702.0 | 1.00 | 994.5 | 1352 | 1.00 | 264 |
| heavy-2048-sliding-cp1 | cudnn_flashmla@compare | 158.2 | 196.4 | 0.52 | 697.7 | 820.6 | 0.70 | 266 |
| heavy-2048-sliding-cp1 | cute@compare | 282.4 | 121.3 | 0.93 | 974.1 | 731.6 | 0.98 | 264 |
| heavy-2048-sliding-cp1 | cute_ws@compare | 118.0 | 50.3 | 0.39 | 815.7 | 739.9 | 0.82 | 264 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@compare | 150.1 | 140.1 | 0.49 | - | - | - | 131 |
| heavy-2048-sliding-cp8r0 | tilelang@main | 49.9 | 686.1 | 1.00 | 180.0 | 1859 | 1.00 | 38 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.3 | 144.8 | 0.71 | - | - | - | 16 |
| heavy-2048-sliding-cp8r0 | tilelang@compare | 50.0 | 705.0 | 1.00 | 180.3 | 1947 | 1.00 | 38 |
| heavy-2048-sliding-cp8r0 | cudnn_flashmla@compare | 43.5 | 274.1 | 0.87 | 158.1 | 1315 | 0.88 | 38 |
| heavy-2048-sliding-cp8r0 | cute@compare | 48.3 | 121.9 | 0.97 | 177.5 | 1283 | 0.98 | 38 |
| heavy-2048-sliding-cp8r0 | cute_ws@compare | 22.1 | 52.9 | 0.44 | 154.3 | 1141 | 0.86 | 38 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 36.3 | 148.3 | 0.73 | - | - | - | 16 |
| heavy-2048-sliding-cp8r4 | tilelang@main | 53.5 | 688.8 | 1.00 | 195.8 | 1841 | 1.00 | 38 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.8 | 144.9 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r4 | tilelang@compare | 53.2 | 703.3 | 1.00 | 195.8 | 1924 | 1.00 | 38 |
| heavy-2048-sliding-cp8r4 | cudnn_flashmla@compare | 44.3 | 271.6 | 0.83 | 163.3 | 1294 | 0.83 | 38 |
| heavy-2048-sliding-cp8r4 | cute@compare | 51.2 | 124.0 | 0.96 | 194.0 | 1262 | 0.99 | 38 |
| heavy-2048-sliding-cp8r4 | cute_ws@compare | 22.5 | 53.5 | 0.42 | 166.4 | 1130 | 0.85 | 38 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 35.9 | 153.1 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r7 | tilelang@main | 53.2 | 690.4 | 1.00 | 195.4 | 1876 | 1.00 | 38 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.7 | 147.0 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r7 | tilelang@compare | 53.1 | 710.3 | 1.00 | 196.1 | 1923 | 1.00 | 38 |
| heavy-2048-sliding-cp8r7 | cudnn_flashmla@compare | 44.0 | 276.9 | 0.83 | 162.9 | 1299 | 0.83 | 38 |
| heavy-2048-sliding-cp8r7 | cute@compare | 51.4 | 122.6 | 0.97 | 193.6 | 1281 | 0.99 | 38 |
| heavy-2048-sliding-cp8r7 | cute_ws@compare | 22.5 | 54.2 | 0.42 | 166.4 | 1129 | 0.85 | 38 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 36.1 | 151.5 | 0.68 | - | - | - | 16 |
| tiny-2048-csa-cp1 | tilelang@main | 358.8 | 683.7 | 1.00 | 1100 | 1259 | 1.00 | 265 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 202.3 | 131.6 | 0.56 | - | - | - | 139 |
| tiny-2048-csa-cp1 | tilelang@compare | 357.6 | 703.6 | 1.00 | 1096 | 1324 | 1.00 | 265 |
| tiny-2048-csa-cp1 | cudnn_flashmla@compare | 209.6 | 185.6 | 0.59 | 756.3 | 769.6 | 0.69 | 271 |
| tiny-2048-csa-cp1 | cute@compare | 334.3 | 117.3 | 0.93 | 1073 | 691.8 | 0.98 | 265 |
| tiny-2048-csa-cp1 | cute_ws@compare | 174.4 | 54.5 | 0.49 | 916.1 | 677.8 | 0.84 | 265 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@compare | 200.5 | 136.1 | 0.56 | - | - | - | 139 |
| tiny-2048-csa-cp8r0 | tilelang@main | 59.5 | 694.4 | 1.00 | 208.2 | 1848 | 1.00 | 40 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.4 | 148.0 | 0.75 | - | - | - | 17 |
| tiny-2048-csa-cp8r0 | tilelang@compare | 59.4 | 707.6 | 1.00 | 208.7 | 1949 | 1.00 | 40 |
| tiny-2048-csa-cp8r0 | cudnn_flashmla@compare | 51.5 | 264.2 | 0.87 | 176.9 | 1298 | 0.85 | 40 |
| tiny-2048-csa-cp8r0 | cute@compare | 57.3 | 119.6 | 0.96 | 205.9 | 1242 | 0.99 | 40 |
| tiny-2048-csa-cp8r0 | cute_ws@compare | 36.2 | 52.1 | 0.61 | 185.2 | 1103 | 0.89 | 40 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 43.9 | 147.3 | 0.74 | - | - | - | 17 |
| tiny-2048-csa-cp8r4 | tilelang@main | 59.6 | 688.9 | 1.00 | 206.9 | 1822 | 1.00 | 40 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 43.5 | 144.4 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r4 | tilelang@compare | 59.3 | 696.3 | 1.00 | 206.9 | 1921 | 1.00 | 40 |
| tiny-2048-csa-cp8r4 | cudnn_flashmla@compare | 51.4 | 262.0 | 0.87 | 172.2 | 1306 | 0.83 | 40 |
| tiny-2048-csa-cp8r4 | cute@compare | 57.3 | 119.0 | 0.97 | 203.6 | 1255 | 0.98 | 40 |
| tiny-2048-csa-cp8r4 | cute_ws@compare | 35.7 | 50.5 | 0.60 | 184.1 | 1109 | 0.89 | 40 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 43.6 | 148.4 | 0.74 | - | - | - | 17 |
| tiny-2048-csa-cp8r7 | tilelang@main | 60.4 | 700.6 | 1.00 | 209.9 | 1830 | 1.00 | 40 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 44.1 | 145.9 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r7 | tilelang@compare | 60.0 | 702.1 | 1.00 | 210.1 | 1917 | 1.00 | 40 |
| tiny-2048-csa-cp8r7 | cudnn_flashmla@compare | 51.9 | 260.5 | 0.87 | 175.5 | 1303 | 0.84 | 40 |
| tiny-2048-csa-cp8r7 | cute@compare | 58.0 | 118.0 | 0.97 | 209.1 | 1253 | 0.99 | 40 |
| tiny-2048-csa-cp8r7 | cute_ws@compare | 36.2 | 50.2 | 0.60 | 188.3 | 1109 | 0.90 | 40 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 44.2 | 146.6 | 0.74 | - | - | - | 17 |
| tiny-2048-hca-cp1 | tilelang@main | 270.8 | 691.8 | 1.00 | 772.0 | 1321 | 1.00 | 264 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 148.6 | 143.6 | 0.55 | - | - | - | 131 |
| tiny-2048-hca-cp1 | tilelang@compare | 270.5 | 694.3 | 1.00 | 769.3 | 1378 | 1.00 | 264 |
| tiny-2048-hca-cp1 | cudnn_flashmla@compare | 158.3 | 205.4 | 0.59 | 598.6 | 892.2 | 0.78 | 266 |
| tiny-2048-hca-cp1 | cute@compare | 251.1 | 124.0 | 0.93 | 750.7 | 760.9 | 0.98 | 264 |
| tiny-2048-hca-cp1 | cute_ws@compare | 115.8 | 52.1 | 0.43 | 629.2 | 728.7 | 0.82 | 264 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@compare | 147.0 | 144.2 | 0.54 | - | - | - | 131 |
| tiny-2048-hca-cp8r0 | tilelang@main | 48.4 | 688.1 | 1.00 | 166.6 | 1863 | 1.00 | 38 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 35.5 | 147.1 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r0 | tilelang@compare | 48.4 | 689.3 | 1.00 | 165.8 | 1953 | 1.00 | 38 |
| tiny-2048-hca-cp8r0 | cudnn_flashmla@compare | 43.9 | 269.5 | 0.91 | 150.8 | 1315 | 0.91 | 38 |
| tiny-2048-hca-cp8r0 | cute@compare | 46.5 | 121.5 | 0.96 | 163.7 | 1292 | 0.99 | 38 |
| tiny-2048-hca-cp8r0 | cute_ws@compare | 21.8 | 51.3 | 0.45 | 141.0 | 1152 | 0.85 | 38 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 35.7 | 146.0 | 0.74 | - | - | - | 16 |
| tiny-2048-hca-cp8r4 | tilelang@main | 48.4 | 686.5 | 1.00 | 164.1 | 1885 | 1.00 | 38 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 35.5 | 145.4 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r4 | tilelang@compare | 48.3 | 694.3 | 1.00 | 164.5 | 1987 | 1.00 | 38 |
| tiny-2048-hca-cp8r4 | cudnn_flashmla@compare | 43.6 | 269.0 | 0.90 | 149.0 | 1350 | 0.91 | 38 |
| tiny-2048-hca-cp8r4 | cute@compare | 46.7 | 122.9 | 0.97 | 162.5 | 1314 | 0.99 | 38 |
| tiny-2048-hca-cp8r4 | cute_ws@compare | 21.8 | 53.5 | 0.45 | 138.9 | 1173 | 0.84 | 38 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 36.1 | 148.3 | 0.75 | - | - | - | 16 |
| tiny-2048-hca-cp8r7 | tilelang@main | 50.0 | 701.3 | 1.00 | 175.8 | 1923 | 1.00 | 38 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 35.6 | 154.3 | 0.71 | - | - | - | 16 |
| tiny-2048-hca-cp8r7 | tilelang@compare | 50.0 | 699.3 | 1.00 | 176.1 | 1932 | 1.00 | 38 |
| tiny-2048-hca-cp8r7 | cudnn_flashmla@compare | 43.7 | 275.6 | 0.88 | 154.7 | 1345 | 0.88 | 38 |
| tiny-2048-hca-cp8r7 | cute@compare | 48.0 | 124.0 | 0.96 | 173.1 | 1300 | 0.98 | 38 |
| tiny-2048-hca-cp8r7 | cute_ws@compare | 21.9 | 54.9 | 0.44 | 149.4 | 1148 | 0.85 | 38 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 35.5 | 149.0 | 0.71 | - | - | - | 16 |
| tiny-2048-sliding-cp1 | tilelang@main | 271.0 | 686.8 | 1.00 | 772.5 | 1336 | 1.00 | 264 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 149.0 | 141.5 | 0.55 | - | - | - | 131 |
| tiny-2048-sliding-cp1 | tilelang@compare | 269.7 | 694.1 | 1.00 | 770.0 | 1441 | 1.00 | 264 |
| tiny-2048-sliding-cp1 | cudnn_flashmla@compare | 158.0 | 201.1 | 0.59 | 597.7 | 921.3 | 0.78 | 266 |
| tiny-2048-sliding-cp1 | cute@compare | 251.7 | 126.6 | 0.93 | 750.5 | 792.9 | 0.97 | 264 |
| tiny-2048-sliding-cp1 | cute_ws@compare | 115.5 | 52.3 | 0.43 | 627.2 | 756.5 | 0.81 | 264 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@compare | 147.0 | 148.9 | 0.55 | - | - | - | 131 |
| tiny-2048-sliding-cp8r0 | tilelang@main | 48.3 | 683.4 | 1.00 | 165.9 | 1970 | 1.00 | 38 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.2 | 144.5 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r0 | tilelang@compare | 48.5 | 701.6 | 1.00 | 165.4 | 2009 | 1.00 | 38 |
| tiny-2048-sliding-cp8r0 | cudnn_flashmla@compare | 44.1 | 272.7 | 0.91 | 151.4 | 1356 | 0.92 | 38 |
| tiny-2048-sliding-cp8r0 | cute@compare | 45.9 | 122.3 | 0.95 | 163.5 | 1371 | 0.99 | 38 |
| tiny-2048-sliding-cp8r0 | cute_ws@compare | 21.8 | 52.4 | 0.45 | 140.5 | 1200 | 0.85 | 38 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 36.0 | 148.8 | 0.74 | - | - | - | 16 |
| tiny-2048-sliding-cp8r4 | tilelang@main | 48.3 | 690.2 | 1.00 | 164.1 | 1860 | 1.00 | 38 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.3 | 143.1 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r4 | tilelang@compare | 48.4 | 706.0 | 1.00 | 164.0 | 1953 | 1.00 | 38 |
| tiny-2048-sliding-cp8r4 | cudnn_flashmla@compare | 44.2 | 268.5 | 0.91 | 148.0 | 1311 | 0.90 | 38 |
| tiny-2048-sliding-cp8r4 | cute@compare | 46.4 | 117.3 | 0.96 | 162.6 | 1294 | 0.99 | 38 |
| tiny-2048-sliding-cp8r4 | cute_ws@compare | 21.7 | 50.3 | 0.45 | 139.6 | 1150 | 0.85 | 38 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 36.1 | 140.9 | 0.74 | - | - | - | 16 |
| tiny-2048-sliding-cp8r7 | tilelang@main | 50.1 | 708.0 | 1.00 | 175.6 | 1874 | 1.00 | 38 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.6 | 151.2 | 0.71 | - | - | - | 16 |
| tiny-2048-sliding-cp8r7 | tilelang@compare | 50.4 | 699.0 | 1.00 | 175.8 | 1938 | 1.00 | 38 |
| tiny-2048-sliding-cp8r7 | cudnn_flashmla@compare | 43.6 | 270.8 | 0.86 | 155.5 | 1309 | 0.88 | 38 |
| tiny-2048-sliding-cp8r7 | cute@compare | 48.1 | 122.3 | 0.95 | 173.6 | 1273 | 0.99 | 38 |
| tiny-2048-sliding-cp8r7 | cute_ws@compare | 21.8 | 52.3 | 0.43 | 149.5 | 1145 | 0.85 | 38 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 36.0 | 146.8 | 0.71 | - | - | - | 16 |
| single-4096-csa-cp1 | tilelang@main | 1208 | 682.8 | 1.00 | 5145 | 827.0 | 1.00 | 530 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 593.7 | 121.7 | 0.49 | - | - | - | 278 |
| single-4096-csa-cp1 | tilelang@compare | 1207 | 688.6 | 1.00 | 5149 | 866.4 | 1.00 | 530 |
| single-4096-csa-cp1 | cudnn_flashmla@compare | 602.9 | 178.6 | 0.50 | 2786 | 352.7 | 0.54 | 542 |
| single-4096-csa-cp1 | cute@compare | 1154 | 117.8 | 0.96 | 5096 | 235.3 | 0.99 | 530 |
| single-4096-csa-cp1 | cute_ws@compare | 549.0 | 55.3 | 0.45 | 4483 | 245.4 | 0.87 | 530 |
| single-4096-csa-cp1 | flashmla_fwd_ref@compare | 594.3 | 120.4 | 0.49 | - | - | - | 278 |
| single-4096-csa-cp8r0 | tilelang@main | 112.0 | 683.6 | 1.00 | 410.0 | 1627 | 1.00 | 79 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.0 | 138.3 | 0.61 | - | - | - | 35 |
| single-4096-csa-cp8r0 | tilelang@compare | 112.3 | 689.2 | 1.00 | 410.5 | 1695 | 1.00 | 79 |
| single-4096-csa-cp8r0 | cudnn_flashmla@compare | 76.6 | 236.5 | 0.68 | 305.2 | 1179 | 0.74 | 81 |
| single-4096-csa-cp8r0 | cute@compare | 106.8 | 117.3 | 0.95 | 403.8 | 1042 | 0.98 | 79 |
| single-4096-csa-cp8r0 | cute_ws@compare | 57.3 | 50.1 | 0.51 | 355.4 | 929.2 | 0.87 | 79 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 68.5 | 142.6 | 0.61 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang@main | 191.2 | 677.8 | 1.00 | 856.0 | 1477 | 1.00 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 109.5 | 138.9 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang@compare | 191.5 | 699.3 | 1.00 | 858.9 | 1537 | 1.00 | 79 |
| single-4096-csa-cp8r4 | cudnn_flashmla@compare | 117.9 | 195.1 | 0.62 | 511.7 | 963.0 | 0.60 | 81 |
| single-4096-csa-cp8r4 | cute@compare | 183.7 | 120.2 | 0.96 | 850.5 | 907.1 | 0.99 | 79 |
| single-4096-csa-cp8r4 | cute_ws@compare | 97.0 | 52.5 | 0.51 | 767.3 | 825.6 | 0.89 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 109.8 | 143.4 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r7 | tilelang@main | 191.7 | 689.2 | 1.00 | 859.6 | 1490 | 1.00 | 79 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 109.4 | 142.6 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r7 | tilelang@compare | 191.3 | 685.9 | 1.00 | 859.8 | 1575 | 1.00 | 79 |
| single-4096-csa-cp8r7 | cudnn_flashmla@compare | 117.7 | 194.2 | 0.62 | 513.4 | 975.3 | 0.60 | 81 |
| single-4096-csa-cp8r7 | cute@compare | 182.8 | 120.1 | 0.96 | 852.6 | 914.3 | 0.99 | 79 |
| single-4096-csa-cp8r7 | cute_ws@compare | 97.1 | 49.7 | 0.51 | 767.0 | 843.0 | 0.89 | 79 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 109.0 | 143.4 | 0.57 | - | - | - | 35 |
| single-4096-hca-cp1 | tilelang@main | 691.9 | 740.4 | 1.00 | 2233 | 900.3 | 1.00 | 530 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 366.3 | 139.2 | 0.53 | - | - | - | 266 |
| single-4096-hca-cp1 | tilelang@compare | 691.9 | 735.9 | 1.00 | 2225 | 969.6 | 1.00 | 530 |
| single-4096-hca-cp1 | cudnn_flashmla@compare | 373.7 | 196.5 | 0.54 | 1512 | 602.6 | 0.68 | 533 |
| single-4096-hca-cp1 | cute@compare | 644.2 | 147.0 | 0.93 | 2183 | 367.7 | 0.98 | 530 |
| single-4096-hca-cp1 | cute_ws@compare | 316.8 | 55.2 | 0.46 | 1862 | 496.5 | 0.84 | 530 |
| single-4096-hca-cp1 | flashmla_fwd_ref@compare | 362.6 | 143.2 | 0.52 | - | - | - | 266 |
| single-4096-hca-cp8r0 | tilelang@main | 102.0 | 726.2 | 1.00 | 349.1 | 1777 | 1.00 | 77 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 66.4 | 150.7 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r0 | tilelang@compare | 100.9 | 742.0 | 1.00 | 349.7 | 1896 | 1.00 | 77 |
| single-4096-hca-cp8r0 | cudnn_flashmla@compare | 73.4 | 250.2 | 0.73 | 272.7 | 1213 | 0.78 | 77 |
| single-4096-hca-cp8r0 | cute@compare | 95.7 | 146.3 | 0.95 | 344.5 | 1237 | 0.99 | 77 |
| single-4096-hca-cp8r0 | cute_ws@compare | 48.1 | 52.3 | 0.48 | 297.3 | 1072 | 0.85 | 77 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 64.8 | 156.4 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang@main | 107.4 | 716.8 | 1.00 | 370.8 | 1772 | 1.00 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 70.2 | 146.6 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang@compare | 107.5 | 807.2 | 1.00 | 370.1 | 1880 | 1.00 | 77 |
| single-4096-hca-cp8r4 | cudnn_flashmla@compare | 78.0 | 250.6 | 0.73 | 291.3 | 1198 | 0.79 | 77 |
| single-4096-hca-cp8r4 | cute@compare | 101.6 | 151.2 | 0.95 | 365.3 | 1206 | 0.99 | 77 |
| single-4096-hca-cp8r4 | cute_ws@compare | 51.6 | 53.9 | 0.48 | 316.0 | 1039 | 0.85 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 69.1 | 157.8 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r7 | tilelang@main | 107.1 | 720.0 | 1.00 | 372.8 | 1786 | 1.00 | 77 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 68.8 | 148.4 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r7 | tilelang@compare | 107.3 | 737.7 | 1.00 | 372.2 | 1940 | 1.00 | 77 |
| single-4096-hca-cp8r7 | cudnn_flashmla@compare | 77.5 | 242.7 | 0.72 | 293.2 | 1232 | 0.79 | 77 |
| single-4096-hca-cp8r7 | cute@compare | 102.0 | 143.7 | 0.95 | 368.3 | 1218 | 0.99 | 77 |
| single-4096-hca-cp8r7 | cute_ws@compare | 51.3 | 52.2 | 0.48 | 316.7 | 1033 | 0.85 | 77 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 68.8 | 150.7 | 0.64 | - | - | - | 33 |
| single-4096-sliding-cp1 | tilelang@main | 594.3 | 703.0 | 1.00 | 1945 | 914.2 | 1.00 | 527 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 273.9 | 146.3 | 0.46 | - | - | - | 262 |
| single-4096-sliding-cp1 | tilelang@compare | 593.2 | 744.9 | 1.00 | 1945 | 993.9 | 1.00 | 527 |
| single-4096-sliding-cp1 | cudnn_flashmla@compare | 283.4 | 201.4 | 0.48 | 1304 | 689.7 | 0.67 | 531 |
| single-4096-sliding-cp1 | cute@compare | 549.1 | 122.4 | 0.93 | 1902 | 376.4 | 0.98 | 527 |
| single-4096-sliding-cp1 | cute_ws@compare | 217.0 | 52.8 | 0.37 | 1575 | 540.3 | 0.81 | 527 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@compare | 274.5 | 147.6 | 0.46 | - | - | - | 262 |
| single-4096-sliding-cp8r0 | tilelang@main | 90.6 | 699.5 | 1.00 | 319.4 | 1715 | 1.00 | 76 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 53.1 | 146.2 | 0.59 | - | - | - | 33 |
| single-4096-sliding-cp8r0 | tilelang@compare | 89.4 | 706.8 | 1.00 | 318.1 | 1817 | 1.00 | 76 |
| single-4096-sliding-cp8r0 | cudnn_flashmla@compare | 61.5 | 246.9 | 0.69 | 245.3 | 1233 | 0.77 | 77 |
| single-4096-sliding-cp8r0 | cute@compare | 84.3 | 123.6 | 0.94 | 314.4 | 1164 | 0.99 | 76 |
| single-4096-sliding-cp8r0 | cute_ws@compare | 37.5 | 51.2 | 0.42 | 265.4 | 1036 | 0.83 | 76 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 53.7 | 151.3 | 0.60 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@main | 92.4 | 698.6 | 1.00 | 327.4 | 1730 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.7 | 149.4 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@compare | 92.5 | 704.4 | 1.00 | 326.5 | 1792 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | cudnn_flashmla@compare | 62.3 | 254.1 | 0.67 | 254.3 | 1207 | 0.78 | 77 |
| single-4096-sliding-cp8r4 | cute@compare | 87.4 | 121.1 | 0.94 | 321.3 | 1138 | 0.98 | 76 |
| single-4096-sliding-cp8r4 | cute_ws@compare | 37.4 | 49.9 | 0.40 | 271.2 | 1007 | 0.83 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 54.2 | 145.9 | 0.59 | - | - | - | 33 |
| single-4096-sliding-cp8r7 | tilelang@main | 92.2 | 693.0 | 1.00 | 327.1 | 1738 | 1.00 | 76 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.9 | 140.2 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r7 | tilelang@compare | 92.7 | 686.1 | 1.00 | 326.1 | 1790 | 1.00 | 76 |
| single-4096-sliding-cp8r7 | cudnn_flashmla@compare | 62.7 | 249.7 | 0.68 | 254.4 | 1208 | 0.78 | 77 |
| single-4096-sliding-cp8r7 | cute@compare | 87.6 | 120.1 | 0.94 | 322.4 | 1130 | 0.99 | 76 |
| single-4096-sliding-cp8r7 | cute_ws@compare | 37.5 | 50.8 | 0.41 | 271.8 | 1013 | 0.83 | 76 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 54.1 | 145.9 | 0.58 | - | - | - | 33 |
| short-4096-csa-cp1 | tilelang@main | 829.2 | 694.2 | 1.00 | 3054 | 819.3 | 1.00 | 530 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 428.1 | 124.9 | 0.52 | - | - | - | 278 |
| short-4096-csa-cp1 | tilelang@compare | 832.5 | 677.8 | 1.00 | 3054 | 847.6 | 1.00 | 530 |
| short-4096-csa-cp1 | cudnn_flashmla@compare | 437.3 | 176.7 | 0.53 | 1851 | 511.8 | 0.61 | 542 |
| short-4096-csa-cp1 | cute@compare | 782.0 | 116.8 | 0.94 | 3004 | 224.0 | 0.98 | 530 |
| short-4096-csa-cp1 | cute_ws@compare | 372.4 | 55.6 | 0.45 | 2603 | 393.8 | 0.85 | 530 |
| short-4096-csa-cp1 | flashmla_fwd_ref@compare | 425.1 | 130.0 | 0.51 | - | - | - | 278 |
| short-4096-csa-cp8r0 | tilelang@main | 112.0 | 715.7 | 1.00 | 410.6 | 1643 | 1.00 | 79 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.3 | 154.1 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r0 | tilelang@compare | 111.6 | 696.6 | 1.00 | 409.3 | 1687 | 1.00 | 79 |
| short-4096-csa-cp8r0 | cudnn_flashmla@compare | 76.4 | 240.2 | 0.68 | 303.3 | 1153 | 0.74 | 81 |
| short-4096-csa-cp8r0 | cute@compare | 106.6 | 121.7 | 0.96 | 404.0 | 1035 | 0.99 | 79 |
| short-4096-csa-cp8r0 | cute_ws@compare | 57.5 | 51.2 | 0.52 | 356.7 | 924.4 | 0.87 | 79 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 67.9 | 143.2 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang@main | 114.7 | 762.2 | 1.00 | 432.7 | 1712 | 1.00 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.3 | 154.8 | 0.63 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang@compare | 113.8 | 714.1 | 1.00 | 432.7 | 1682 | 1.00 | 79 |
| short-4096-csa-cp8r4 | cudnn_flashmla@compare | 81.1 | 226.3 | 0.71 | 315.3 | 1151 | 0.73 | 81 |
| short-4096-csa-cp8r4 | cute@compare | 108.6 | 119.8 | 0.95 | 426.0 | 1025 | 0.98 | 79 |
| short-4096-csa-cp8r4 | cute_ws@compare | 60.9 | 49.9 | 0.54 | 379.2 | 908.6 | 0.88 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 72.6 | 142.7 | 0.64 | - | - | - | 35 |
| short-4096-csa-cp8r7 | tilelang@main | 112.8 | 705.3 | 1.00 | 430.6 | 1629 | 1.00 | 79 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 68.9 | 141.8 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r7 | tilelang@compare | 112.3 | 683.0 | 1.00 | 432.0 | 1662 | 1.00 | 79 |
| short-4096-csa-cp8r7 | cudnn_flashmla@compare | 77.6 | 227.7 | 0.69 | 309.7 | 1159 | 0.72 | 81 |
| short-4096-csa-cp8r7 | cute@compare | 107.1 | 119.4 | 0.95 | 426.8 | 1020 | 0.99 | 79 |
| short-4096-csa-cp8r7 | cute_ws@compare | 58.9 | 49.4 | 0.52 | 378.9 | 920.5 | 0.88 | 79 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 69.2 | 141.5 | 0.62 | - | - | - | 35 |
| short-4096-hca-cp1 | tilelang@main | 666.3 | 744.5 | 1.00 | 2131 | 958.7 | 1.00 | 530 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 359.6 | 140.5 | 0.54 | - | - | - | 266 |
| short-4096-hca-cp1 | tilelang@compare | 666.0 | 729.2 | 1.00 | 2132 | 956.1 | 1.00 | 530 |
| short-4096-hca-cp1 | cudnn_flashmla@compare | 370.2 | 192.7 | 0.56 | 1438 | 601.7 | 0.67 | 533 |
| short-4096-hca-cp1 | cute@compare | 622.3 | 141.9 | 0.93 | 2090 | 366.1 | 0.98 | 530 |
| short-4096-hca-cp1 | cute_ws@compare | 302.9 | 55.8 | 0.45 | 1773 | 497.6 | 0.83 | 530 |
| short-4096-hca-cp1 | flashmla_fwd_ref@compare | 357.8 | 143.7 | 0.54 | - | - | - | 266 |
| short-4096-hca-cp8r0 | tilelang@main | 100.2 | 744.3 | 1.00 | 347.4 | 1833 | 1.00 | 77 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 64.7 | 151.3 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r0 | tilelang@compare | 100.3 | 733.3 | 1.00 | 348.5 | 1862 | 1.00 | 77 |
| short-4096-hca-cp8r0 | cudnn_flashmla@compare | 73.1 | 244.8 | 0.73 | 271.6 | 1204 | 0.78 | 77 |
| short-4096-hca-cp8r0 | cute@compare | 95.2 | 147.0 | 0.95 | 342.5 | 1208 | 0.98 | 77 |
| short-4096-hca-cp8r0 | cute_ws@compare | 47.6 | 52.1 | 0.47 | 295.1 | 1049 | 0.85 | 77 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 65.2 | 152.1 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang@main | 101.5 | 725.0 | 1.00 | 354.4 | 1803 | 1.00 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 65.3 | 142.9 | 0.64 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang@compare | 101.6 | 726.8 | 1.00 | 354.7 | 1853 | 1.00 | 77 |
| short-4096-hca-cp8r4 | cudnn_flashmla@compare | 72.7 | 243.6 | 0.72 | 274.6 | 1202 | 0.77 | 77 |
| short-4096-hca-cp8r4 | cute@compare | 96.0 | 143.9 | 0.94 | 349.4 | 1198 | 0.98 | 77 |
| short-4096-hca-cp8r4 | cute_ws@compare | 47.1 | 52.1 | 0.46 | 300.8 | 1026 | 0.85 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 64.4 | 149.3 | 0.63 | - | - | - | 33 |
| short-4096-hca-cp8r7 | tilelang@main | 102.6 | 728.6 | 1.00 | 359.5 | 1837 | 1.00 | 77 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 67.0 | 147.2 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r7 | tilelang@compare | 102.9 | 726.8 | 1.00 | 359.7 | 1837 | 1.00 | 77 |
| short-4096-hca-cp8r7 | cudnn_flashmla@compare | 75.3 | 244.5 | 0.73 | 278.9 | 1174 | 0.78 | 77 |
| short-4096-hca-cp8r7 | cute@compare | 97.8 | 144.1 | 0.95 | 354.1 | 1197 | 0.98 | 77 |
| short-4096-hca-cp8r7 | cute_ws@compare | 48.6 | 51.2 | 0.47 | 307.0 | 1044 | 0.85 | 77 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 67.1 | 153.2 | 0.65 | - | - | - | 33 |
| short-4096-sliding-cp1 | tilelang@main | 584.0 | 727.8 | 1.00 | 1896 | 946.0 | 1.00 | 527 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 271.4 | 151.8 | 0.46 | - | - | - | 262 |
| short-4096-sliding-cp1 | tilelang@compare | 585.8 | 702.3 | 1.00 | 1902 | 968.7 | 1.00 | 527 |
| short-4096-sliding-cp1 | cudnn_flashmla@compare | 283.9 | 200.5 | 0.48 | 1290 | 693.0 | 0.68 | 531 |
| short-4096-sliding-cp1 | cute@compare | 541.8 | 123.9 | 0.92 | 1854 | 375.1 | 0.97 | 527 |
| short-4096-sliding-cp1 | cute_ws@compare | 215.6 | 56.4 | 0.37 | 1532 | 541.9 | 0.81 | 527 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@compare | 274.3 | 140.2 | 0.47 | - | - | - | 262 |
| short-4096-sliding-cp8r0 | tilelang@main | 89.3 | 704.5 | 1.00 | 317.4 | 1719 | 1.00 | 76 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 53.0 | 144.4 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r0 | tilelang@compare | 89.8 | 699.5 | 1.00 | 317.4 | 1813 | 1.00 | 76 |
| short-4096-sliding-cp8r0 | cudnn_flashmla@compare | 61.3 | 250.1 | 0.68 | 251.0 | 1232 | 0.79 | 77 |
| short-4096-sliding-cp8r0 | cute@compare | 85.4 | 120.3 | 0.95 | 313.7 | 1148 | 0.99 | 76 |
| short-4096-sliding-cp8r0 | cute_ws@compare | 37.5 | 51.5 | 0.42 | 265.2 | 1034 | 0.84 | 76 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 53.7 | 149.7 | 0.60 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@main | 89.9 | 704.2 | 1.00 | 319.7 | 1722 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.1 | 144.6 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@compare | 90.6 | 703.8 | 1.00 | 320.8 | 1801 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | cudnn_flashmla@compare | 62.2 | 253.2 | 0.69 | 253.6 | 1232 | 0.79 | 77 |
| short-4096-sliding-cp8r4 | cute@compare | 84.9 | 125.0 | 0.94 | 314.4 | 1155 | 0.98 | 76 |
| short-4096-sliding-cp8r4 | cute_ws@compare | 37.5 | 53.0 | 0.41 | 266.5 | 1041 | 0.83 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 54.2 | 149.3 | 0.60 | - | - | - | 33 |
| short-4096-sliding-cp8r7 | tilelang@main | 90.8 | 684.0 | 1.00 | 323.9 | 1762 | 1.00 | 76 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 54.0 | 141.7 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r7 | tilelang@compare | 90.9 | 706.8 | 1.00 | 323.0 | 1815 | 1.00 | 76 |
| short-4096-sliding-cp8r7 | cudnn_flashmla@compare | 61.9 | 255.2 | 0.68 | 248.8 | 1223 | 0.77 | 77 |
| short-4096-sliding-cp8r7 | cute@compare | 86.1 | 121.9 | 0.95 | 317.9 | 1150 | 0.98 | 76 |
| short-4096-sliding-cp8r7 | cute_ws@compare | 37.2 | 51.4 | 0.41 | 270.3 | 1026 | 0.84 | 76 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 54.3 | 143.8 | 0.60 | - | - | - | 33 |
| heavy-4096-csa-cp1 | tilelang@main | 770.8 | 691.6 | 1.00 | 2718 | 827.3 | 1.00 | 530 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 404.9 | 120.0 | 0.53 | - | - | - | 278 |
| heavy-4096-csa-cp1 | tilelang@compare | 771.0 | 703.6 | 1.00 | 2720 | 880.3 | 1.00 | 530 |
| heavy-4096-csa-cp1 | cudnn_flashmla@compare | 414.8 | 183.0 | 0.54 | 1699 | 584.4 | 0.62 | 542 |
| heavy-4096-csa-cp1 | cute@compare | 726.8 | 116.7 | 0.94 | 2675 | 228.7 | 0.98 | 530 |
| heavy-4096-csa-cp1 | cute_ws@compare | 357.3 | 53.6 | 0.46 | 2305 | 418.2 | 0.85 | 530 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@compare | 406.9 | 123.6 | 0.53 | - | - | - | 278 |
| heavy-4096-csa-cp8r0 | tilelang@main | 111.8 | 682.7 | 1.00 | 410.6 | 1622 | 1.00 | 79 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.1 | 140.3 | 0.61 | - | - | - | 35 |
| heavy-4096-csa-cp8r0 | tilelang@compare | 111.6 | 699.8 | 1.00 | 409.9 | 1719 | 1.00 | 79 |
| heavy-4096-csa-cp8r0 | cudnn_flashmla@compare | 76.7 | 240.5 | 0.69 | 303.4 | 1185 | 0.74 | 81 |
| heavy-4096-csa-cp8r0 | cute@compare | 106.5 | 120.8 | 0.95 | 404.7 | 1049 | 0.99 | 79 |
| heavy-4096-csa-cp8r0 | cute_ws@compare | 57.3 | 50.8 | 0.51 | 355.8 | 958.0 | 0.87 | 79 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 68.4 | 143.3 | 0.61 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang@main | 114.6 | 703.3 | 1.00 | 434.1 | 1615 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.1 | 145.2 | 0.63 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang@compare | 114.2 | 699.2 | 1.00 | 435.7 | 1680 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | cudnn_flashmla@compare | 81.0 | 234.0 | 0.71 | 316.7 | 1168 | 0.73 | 81 |
| heavy-4096-csa-cp8r4 | cute@compare | 109.0 | 119.7 | 0.95 | 428.6 | 1030 | 0.98 | 79 |
| heavy-4096-csa-cp8r4 | cute_ws@compare | 60.8 | 50.1 | 0.53 | 380.8 | 922.3 | 0.87 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 72.9 | 145.4 | 0.64 | - | - | - | 35 |
| heavy-4096-csa-cp8r7 | tilelang@main | 105.7 | 744.1 | 1.00 | 378.3 | 1711 | 1.00 | 79 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 68.6 | 147.1 | 0.65 | - | - | - | 35 |
| heavy-4096-csa-cp8r7 | tilelang@compare | 105.4 | 693.9 | 1.00 | 379.7 | 1758 | 1.00 | 79 |
| heavy-4096-csa-cp8r7 | cudnn_flashmla@compare | 76.9 | 242.5 | 0.73 | 287.6 | 1218 | 0.76 | 81 |
| heavy-4096-csa-cp8r7 | cute@compare | 100.3 | 120.9 | 0.95 | 374.4 | 1102 | 0.99 | 79 |
| heavy-4096-csa-cp8r7 | cute_ws@compare | 58.4 | 50.2 | 0.55 | 332.1 | 978.1 | 0.87 | 79 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 68.7 | 143.2 | 0.65 | - | - | - | 35 |
| heavy-4096-hca-cp1 | tilelang@main | 647.0 | 727.5 | 1.00 | 2030 | 952.6 | 1.00 | 530 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 343.9 | 138.8 | 0.53 | - | - | - | 266 |
| heavy-4096-hca-cp1 | tilelang@compare | 646.7 | 770.2 | 1.00 | 2031 | 995.5 | 1.00 | 530 |
| heavy-4096-hca-cp1 | cudnn_flashmla@compare | 352.0 | 200.5 | 0.54 | 1394 | 627.9 | 0.69 | 533 |
| heavy-4096-hca-cp1 | cute@compare | 602.1 | 147.8 | 0.93 | 1977 | 410.0 | 0.97 | 530 |
| heavy-4096-hca-cp1 | cute_ws@compare | 289.4 | 54.2 | 0.45 | 1670 | 522.7 | 0.82 | 530 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@compare | 340.8 | 144.2 | 0.53 | - | - | - | 266 |
| heavy-4096-hca-cp8r0 | tilelang@main | 100.5 | 735.0 | 1.00 | 348.5 | 1804 | 1.00 | 77 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 65.0 | 149.5 | 0.65 | - | - | - | 33 |
| heavy-4096-hca-cp8r0 | tilelang@compare | 100.4 | 739.3 | 1.00 | 349.8 | 1893 | 1.00 | 77 |
| heavy-4096-hca-cp8r0 | cudnn_flashmla@compare | 73.1 | 251.4 | 0.73 | 273.6 | 1224 | 0.78 | 77 |
| heavy-4096-hca-cp8r0 | cute@compare | 95.8 | 146.2 | 0.96 | 345.0 | 1225 | 0.99 | 77 |
| heavy-4096-hca-cp8r0 | cute_ws@compare | 47.6 | 50.3 | 0.47 | 296.1 | 1063 | 0.85 | 77 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 64.5 | 152.1 | 0.64 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang@main | 102.2 | 725.7 | 1.00 | 357.1 | 1803 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 64.6 | 142.8 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang@compare | 101.8 | 719.3 | 1.00 | 356.0 | 1895 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | cudnn_flashmla@compare | 72.5 | 249.3 | 0.71 | 275.3 | 1229 | 0.77 | 77 |
| heavy-4096-hca-cp8r4 | cute@compare | 96.5 | 148.5 | 0.95 | 351.5 | 1240 | 0.99 | 77 |
| heavy-4096-hca-cp8r4 | cute_ws@compare | 47.7 | 51.7 | 0.47 | 302.5 | 1063 | 0.85 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 64.4 | 150.0 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r7 | tilelang@main | 97.6 | 740.6 | 1.00 | 338.4 | 1841 | 1.00 | 77 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 61.8 | 154.0 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r7 | tilelang@compare | 97.5 | 723.2 | 1.00 | 337.0 | 1870 | 1.00 | 77 |
| heavy-4096-hca-cp8r7 | cudnn_flashmla@compare | 70.0 | 244.4 | 0.72 | 261.8 | 1220 | 0.78 | 77 |
| heavy-4096-hca-cp8r7 | cute@compare | 94.1 | 142.1 | 0.96 | 333.0 | 1231 | 0.99 | 77 |
| heavy-4096-hca-cp8r7 | cute_ws@compare | 44.8 | 50.9 | 0.46 | 285.3 | 1069 | 0.85 | 77 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 61.2 | 148.3 | 0.63 | - | - | - | 33 |
| heavy-4096-sliding-cp1 | tilelang@main | 582.4 | 700.3 | 1.00 | 1840 | 1071 | 1.00 | 527 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 269.1 | 149.7 | 0.46 | - | - | - | 262 |
| heavy-4096-sliding-cp1 | tilelang@compare | 584.2 | 692.5 | 1.00 | 1844 | 964.5 | 1.00 | 527 |
| heavy-4096-sliding-cp1 | cudnn_flashmla@compare | 283.7 | 200.3 | 0.49 | 1263 | 669.0 | 0.69 | 531 |
| heavy-4096-sliding-cp1 | cute@compare | 538.7 | 122.0 | 0.92 | 1799 | 359.2 | 0.98 | 527 |
| heavy-4096-sliding-cp1 | cute_ws@compare | 215.7 | 56.1 | 0.37 | 1477 | 530.4 | 0.80 | 527 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@compare | 271.4 | 146.2 | 0.46 | - | - | - | 262 |
| heavy-4096-sliding-cp8r0 | tilelang@main | 88.9 | 704.9 | 1.00 | 318.0 | 1765 | 1.00 | 76 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 52.5 | 146.2 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r0 | tilelang@compare | 89.6 | 689.7 | 1.00 | 315.8 | 1807 | 1.00 | 76 |
| heavy-4096-sliding-cp8r0 | cudnn_flashmla@compare | 61.1 | 248.8 | 0.68 | 250.3 | 1212 | 0.79 | 77 |
| heavy-4096-sliding-cp8r0 | cute@compare | 84.4 | 119.4 | 0.94 | 312.9 | 1134 | 0.99 | 76 |
| heavy-4096-sliding-cp8r0 | cute_ws@compare | 37.4 | 51.3 | 0.42 | 265.6 | 1022 | 0.84 | 76 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 53.7 | 145.7 | 0.60 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@main | 90.6 | 690.0 | 1.00 | 321.4 | 1729 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.9 | 140.4 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@compare | 90.4 | 690.3 | 1.00 | 323.4 | 1765 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | cudnn_flashmla@compare | 62.7 | 247.9 | 0.69 | 254.5 | 1189 | 0.79 | 77 |
| heavy-4096-sliding-cp8r4 | cute@compare | 86.0 | 119.2 | 0.95 | 317.3 | 1112 | 0.98 | 76 |
| heavy-4096-sliding-cp8r4 | cute_ws@compare | 37.0 | 49.7 | 0.41 | 268.4 | 1029 | 0.83 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 53.6 | 142.4 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r7 | tilelang@main | 87.9 | 684.1 | 1.00 | 309.0 | 1790 | 1.00 | 76 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.5 | 144.1 | 0.61 | - | - | - | 33 |
| heavy-4096-sliding-cp8r7 | tilelang@compare | 88.0 | 699.1 | 1.00 | 308.5 | 1799 | 1.00 | 76 |
| heavy-4096-sliding-cp8r7 | cudnn_flashmla@compare | 61.8 | 252.7 | 0.70 | 245.2 | 1211 | 0.80 | 77 |
| heavy-4096-sliding-cp8r7 | cute@compare | 82.9 | 122.0 | 0.94 | 304.1 | 1158 | 0.99 | 76 |
| heavy-4096-sliding-cp8r7 | cute_ws@compare | 37.4 | 52.7 | 0.42 | 259.7 | 1020 | 0.84 | 76 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 53.3 | 148.2 | 0.61 | - | - | - | 33 |
| tiny-4096-csa-cp1 | tilelang@main | 680.3 | 693.4 | 1.00 | 2071 | 848.0 | 1.00 | 530 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 372.6 | 128.9 | 0.55 | - | - | - | 278 |
| tiny-4096-csa-cp1 | tilelang@compare | 680.3 | 673.3 | 1.00 | 2073 | 887.6 | 1.00 | 530 |
| tiny-4096-csa-cp1 | cudnn_flashmla@compare | 381.3 | 171.6 | 0.56 | 1395 | 580.4 | 0.67 | 542 |
| tiny-4096-csa-cp1 | cute@compare | 637.1 | 112.6 | 0.94 | 2030 | 270.6 | 0.98 | 530 |
| tiny-4096-csa-cp1 | cute_ws@compare | 330.1 | 50.7 | 0.49 | 1719 | 445.5 | 0.83 | 530 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@compare | 371.9 | 118.7 | 0.55 | - | - | - | 278 |
| tiny-4096-csa-cp8r0 | tilelang@main | 103.6 | 692.9 | 1.00 | 340.7 | 1680 | 1.00 | 79 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 66.6 | 144.3 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r0 | tilelang@compare | 104.1 | 698.0 | 1.00 | 342.0 | 1759 | 1.00 | 79 |
| tiny-4096-csa-cp8r0 | cudnn_flashmla@compare | 75.6 | 234.0 | 0.73 | 266.9 | 1205 | 0.78 | 81 |
| tiny-4096-csa-cp8r0 | cute@compare | 98.0 | 119.4 | 0.94 | 336.4 | 1112 | 0.98 | 79 |
| tiny-4096-csa-cp8r0 | cute_ws@compare | 57.2 | 50.8 | 0.55 | 296.1 | 992.5 | 0.87 | 79 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 67.1 | 143.0 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang@main | 104.5 | 704.8 | 1.00 | 345.8 | 1688 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 67.1 | 142.5 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang@compare | 104.7 | 706.8 | 1.00 | 345.8 | 1757 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | cudnn_flashmla@compare | 76.2 | 230.7 | 0.73 | 273.0 | 1188 | 0.79 | 81 |
| tiny-4096-csa-cp8r4 | cute@compare | 99.2 | 119.6 | 0.95 | 340.4 | 1094 | 0.98 | 79 |
| tiny-4096-csa-cp8r4 | cute_ws@compare | 57.4 | 50.1 | 0.55 | 298.8 | 989.7 | 0.86 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 67.5 | 143.9 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r7 | tilelang@main | 104.5 | 681.6 | 1.00 | 346.0 | 1697 | 1.00 | 79 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 67.8 | 142.8 | 0.65 | - | - | - | 35 |
| tiny-4096-csa-cp8r7 | tilelang@compare | 104.5 | 695.3 | 1.00 | 345.5 | 1795 | 1.00 | 79 |
| tiny-4096-csa-cp8r7 | cudnn_flashmla@compare | 76.6 | 232.3 | 0.73 | 271.8 | 1189 | 0.79 | 81 |
| tiny-4096-csa-cp8r7 | cute@compare | 99.6 | 119.1 | 0.95 | 341.2 | 1117 | 0.99 | 79 |
| tiny-4096-csa-cp8r7 | cute_ws@compare | 57.8 | 49.2 | 0.55 | 299.6 | 1002 | 0.87 | 79 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 68.2 | 140.4 | 0.65 | - | - | - | 35 |
| tiny-4096-hca-cp1 | tilelang@main | 532.2 | 695.1 | 1.00 | 1436 | 973.1 | 1.00 | 527 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 273.7 | 138.6 | 0.51 | - | - | - | 262 |
| tiny-4096-hca-cp1 | tilelang@compare | 531.5 | 694.4 | 1.00 | 1434 | 1010 | 1.00 | 527 |
| tiny-4096-hca-cp1 | cudnn_flashmla@compare | 284.8 | 193.2 | 0.54 | 1091 | 675.6 | 0.76 | 531 |
| tiny-4096-hca-cp1 | cute@compare | 488.9 | 121.4 | 0.92 | 1395 | 416.3 | 0.97 | 527 |
| tiny-4096-hca-cp1 | cute_ws@compare | 220.4 | 60.3 | 0.41 | 1146 | 505.9 | 0.80 | 527 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@compare | 272.0 | 148.1 | 0.51 | - | - | - | 262 |
| tiny-4096-hca-cp8r0 | tilelang@main | 81.8 | 688.6 | 1.00 | 254.2 | 1778 | 1.00 | 76 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 52.5 | 142.3 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r0 | tilelang@compare | 82.3 | 694.0 | 1.00 | 254.9 | 1863 | 1.00 | 76 |
| tiny-4096-hca-cp8r0 | cudnn_flashmla@compare | 60.7 | 251.2 | 0.74 | 224.1 | 1236 | 0.88 | 77 |
| tiny-4096-hca-cp8r0 | cute@compare | 77.1 | 118.6 | 0.94 | 249.3 | 1215 | 0.98 | 76 |
| tiny-4096-hca-cp8r0 | cute_ws@compare | 36.9 | 50.2 | 0.45 | 209.3 | 1088 | 0.82 | 76 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 52.9 | 142.2 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang@main | 84.2 | 683.2 | 1.00 | 267.7 | 1755 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 53.3 | 143.4 | 0.63 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang@compare | 84.5 | 697.1 | 1.00 | 267.3 | 1849 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | cudnn_flashmla@compare | 61.6 | 250.3 | 0.73 | 228.5 | 1242 | 0.85 | 77 |
| tiny-4096-hca-cp8r4 | cute@compare | 79.6 | 119.2 | 0.94 | 263.0 | 1196 | 0.98 | 76 |
| tiny-4096-hca-cp8r4 | cute_ws@compare | 36.9 | 50.8 | 0.44 | 220.5 | 1074 | 0.82 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 54.2 | 146.0 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r7 | tilelang@main | 82.2 | 701.9 | 1.00 | 260.4 | 1793 | 1.00 | 76 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 52.8 | 148.7 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r7 | tilelang@compare | 81.9 | 692.2 | 1.00 | 260.6 | 1841 | 1.00 | 76 |
| tiny-4096-hca-cp8r7 | cudnn_flashmla@compare | 61.3 | 251.1 | 0.75 | 224.8 | 1235 | 0.86 | 77 |
| tiny-4096-hca-cp8r7 | cute@compare | 77.8 | 124.5 | 0.95 | 255.6 | 1197 | 0.98 | 76 |
| tiny-4096-hca-cp8r7 | cute_ws@compare | 37.0 | 50.2 | 0.45 | 216.6 | 1062 | 0.83 | 76 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 53.8 | 145.8 | 0.66 | - | - | - | 33 |
| tiny-4096-sliding-cp1 | tilelang@main | 534.3 | 692.2 | 1.00 | 1439 | 968.5 | 1.00 | 527 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 269.6 | 139.2 | 0.50 | - | - | - | 262 |
| tiny-4096-sliding-cp1 | tilelang@compare | 530.2 | 705.6 | 1.00 | 1435 | 1033 | 1.00 | 527 |
| tiny-4096-sliding-cp1 | cudnn_flashmla@compare | 282.4 | 198.1 | 0.53 | 1092 | 688.9 | 0.76 | 531 |
| tiny-4096-sliding-cp1 | cute@compare | 488.5 | 124.5 | 0.92 | 1394 | 420.6 | 0.97 | 527 |
| tiny-4096-sliding-cp1 | cute_ws@compare | 227.6 | 58.5 | 0.43 | 1146 | 529.6 | 0.80 | 527 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@compare | 272.2 | 142.5 | 0.51 | - | - | - | 262 |
| tiny-4096-sliding-cp8r0 | tilelang@main | 81.9 | 693.4 | 1.00 | 254.8 | 1776 | 1.00 | 76 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 52.1 | 148.6 | 0.64 | - | - | - | 33 |
| tiny-4096-sliding-cp8r0 | tilelang@compare | 82.1 | 699.4 | 1.00 | 254.0 | 1865 | 1.00 | 76 |
| tiny-4096-sliding-cp8r0 | cudnn_flashmla@compare | 61.5 | 246.9 | 0.75 | 224.2 | 1244 | 0.88 | 77 |
| tiny-4096-sliding-cp8r0 | cute@compare | 76.9 | 119.1 | 0.94 | 249.1 | 1219 | 0.98 | 76 |
| tiny-4096-sliding-cp8r0 | cute_ws@compare | 36.9 | 50.2 | 0.45 | 208.8 | 1087 | 0.82 | 76 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 53.9 | 144.8 | 0.66 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@main | 84.0 | 697.6 | 1.00 | 267.6 | 1772 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 52.9 | 150.1 | 0.63 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@compare | 84.7 | 697.2 | 1.00 | 267.8 | 1845 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | cudnn_flashmla@compare | 61.0 | 247.2 | 0.72 | 227.4 | 1238 | 0.85 | 77 |
| tiny-4096-sliding-cp8r4 | cute@compare | 79.6 | 119.2 | 0.94 | 263.1 | 1193 | 0.98 | 76 |
| tiny-4096-sliding-cp8r4 | cute_ws@compare | 36.7 | 51.6 | 0.43 | 220.8 | 1083 | 0.82 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 54.4 | 144.5 | 0.64 | - | - | - | 33 |
| tiny-4096-sliding-cp8r7 | tilelang@main | 82.0 | 704.4 | 1.00 | 259.8 | 1821 | 1.00 | 76 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.1 | 149.0 | 0.65 | - | - | - | 33 |
| tiny-4096-sliding-cp8r7 | tilelang@compare | 82.3 | 699.0 | 1.00 | 261.1 | 1867 | 1.00 | 76 |
| tiny-4096-sliding-cp8r7 | cudnn_flashmla@compare | 61.5 | 249.7 | 0.75 | 224.5 | 1254 | 0.86 | 77 |
| tiny-4096-sliding-cp8r7 | cute@compare | 77.7 | 118.3 | 0.94 | 255.3 | 1220 | 0.98 | 76 |
| tiny-4096-sliding-cp8r7 | cute_ws@compare | 36.8 | 50.3 | 0.45 | 216.0 | 1084 | 0.83 | 76 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 54.1 | 147.6 | 0.66 | - | - | - | 33 |
| single-16384-csa-cp1 | tilelang@main | 5358 | 559.6 | 1.00 | 22675 | 885.3 | 1.00 | 2120 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 2542 | 28.1 | 0.47 | - | - | - | 1112 |
| single-16384-csa-cp1 | tilelang@compare | 5519 | 378.2 | 1.00 | 22666 | 1048 | 1.00 | 2120 |
| single-16384-csa-cp1 | cudnn_flashmla@compare | 2570 | 62.7 | 0.47 | 12137 | 549.1 | 0.54 | 2168 |
| single-16384-csa-cp1 | cute@compare | 4953 | 211.8 | 0.90 | 22613 | 240.1 | 1.00 | 2120 |
| single-16384-csa-cp1 | cute_ws@compare | 2367 | 65.3 | 0.43 | 19987 | 139.4 | 0.88 | 2120 |
| single-16384-csa-cp1 | flashmla_fwd_ref@compare | 2537 | 224.5 | 0.46 | - | - | - | 1112 |
| single-16384-csa-cp8r0 | tilelang@main | 539.8 | 692.6 | 1.00 | 2182 | 1062 | 1.00 | 318 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 283.6 | 134.0 | 0.53 | - | - | - | 139 |
| single-16384-csa-cp8r0 | tilelang@compare | 534.7 | 698.8 | 1.00 | 2169 | 1107 | 1.00 | 318 |
| single-16384-csa-cp8r0 | cudnn_flashmla@compare | 289.7 | 190.8 | 0.54 | 1283 | 665.9 | 0.59 | 324 |
| single-16384-csa-cp8r0 | cute@compare | 505.8 | 123.5 | 0.95 | 2138 | 492.3 | 0.99 | 318 |
| single-16384-csa-cp8r0 | cute_ws@compare | 250.8 | 56.7 | 0.47 | 1887 | 601.3 | 0.87 | 318 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 282.3 | 137.5 | 0.53 | - | - | - | 139 |
| single-16384-csa-cp8r4 | tilelang@main | 710.9 | 698.0 | 1.00 | 3186 | 893.3 | 1.00 | 318 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 359.6 | 135.6 | 0.51 | - | - | - | 139 |
| single-16384-csa-cp8r4 | tilelang@compare | 712.4 | 707.9 | 1.00 | 3157 | 957.8 | 1.00 | 318 |
| single-16384-csa-cp8r4 | cudnn_flashmla@compare | 366.4 | 196.5 | 0.51 | 1728 | 597.2 | 0.55 | 324 |
| single-16384-csa-cp8r4 | cute@compare | 679.3 | 119.0 | 0.95 | 3124 | 343.5 | 0.99 | 318 |
| single-16384-csa-cp8r4 | cute_ws@compare | 329.6 | 54.4 | 0.46 | 2782 | 538.3 | 0.88 | 318 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 360.4 | 131.1 | 0.51 | - | - | - | 139 |
| single-16384-csa-cp8r7 | tilelang@main | 711.0 | 694.1 | 1.00 | 3201 | 903.7 | 1.00 | 318 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 358.9 | 137.7 | 0.50 | - | - | - | 139 |
| single-16384-csa-cp8r7 | tilelang@compare | 706.6 | 702.6 | 1.00 | 3160 | 974.2 | 1.00 | 318 |
| single-16384-csa-cp8r7 | cudnn_flashmla@compare | 365.2 | 186.4 | 0.52 | 1732 | 605.1 | 0.55 | 324 |
| single-16384-csa-cp8r7 | cute@compare | 674.8 | 120.5 | 0.96 | 3125 | 351.3 | 0.99 | 318 |
| single-16384-csa-cp8r7 | cute_ws@compare | 327.2 | 55.9 | 0.46 | 2784 | 537.0 | 0.88 | 318 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 358.3 | 136.2 | 0.51 | - | - | - | 139 |
| single-16384-hca-cp1 | tilelang@main | 2873 | 700.6 | 1.00 | 10008 | 896.7 | 1.00 | 2108 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1399 | 90.2 | 0.49 | - | - | - | 1064 |
| single-16384-hca-cp1 | tilelang@compare | 2849 | 686.7 | 1.00 | 9949 | 925.7 | 1.00 | 2108 |
| single-16384-hca-cp1 | cudnn_flashmla@compare | 1402 | 155.8 | 0.49 | 6234 | 299.3 | 0.63 | 2132 |
| single-16384-hca-cp1 | cute@compare | 2674 | 98.9 | 0.94 | 9773 | 276.9 | 0.98 | 2108 |
| single-16384-hca-cp1 | cute_ws@compare | 1195 | 53.3 | 0.42 | 8278 | 135.1 | 0.83 | 2108 |
| single-16384-hca-cp1 | flashmla_fwd_ref@compare | 1396 | 89.6 | 0.49 | - | - | - | 1064 |
| single-16384-hca-cp8r0 | tilelang@main | 360.1 | 720.7 | 1.00 | 1178 | 1245 | 1.00 | 306 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 196.9 | 143.6 | 0.55 | - | - | - | 133 |
| single-16384-hca-cp8r0 | tilelang@compare | 356.9 | 714.7 | 1.00 | 1172 | 1301 | 1.00 | 306 |
| single-16384-hca-cp8r0 | cudnn_flashmla@compare | 202.1 | 199.2 | 0.57 | 825.9 | 762.5 | 0.70 | 309 |
| single-16384-hca-cp8r0 | cute@compare | 332.6 | 126.2 | 0.93 | 1143 | 679.9 | 0.98 | 306 |
| single-16384-hca-cp8r0 | cute_ws@compare | 167.4 | 55.5 | 0.47 | 983.7 | 688.7 | 0.84 | 306 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 194.6 | 147.7 | 0.55 | - | - | - | 133 |
| single-16384-hca-cp8r4 | tilelang@main | 411.5 | 685.4 | 1.00 | 1468 | 1197 | 1.00 | 306 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 200.6 | 139.3 | 0.49 | - | - | - | 133 |
| single-16384-hca-cp8r4 | tilelang@compare | 410.3 | 691.1 | 1.00 | 1472 | 1259 | 1.00 | 306 |
| single-16384-hca-cp8r4 | cudnn_flashmla@compare | 210.6 | 193.7 | 0.51 | 939.6 | 780.4 | 0.64 | 309 |
| single-16384-hca-cp8r4 | cute@compare | 385.6 | 126.0 | 0.94 | 1443 | 615.8 | 0.98 | 306 |
| single-16384-hca-cp8r4 | cute_ws@compare | 171.7 | 53.4 | 0.42 | 1229 | 673.6 | 0.83 | 306 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 200.0 | 141.5 | 0.49 | - | - | - | 133 |
| single-16384-hca-cp8r7 | tilelang@main | 416.3 | 690.1 | 1.00 | 1589 | 1230 | 1.00 | 306 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 200.7 | 143.1 | 0.48 | - | - | - | 133 |
| single-16384-hca-cp8r7 | tilelang@compare | 415.7 | 708.4 | 1.00 | 1587 | 1297 | 1.00 | 306 |
| single-16384-hca-cp8r7 | cudnn_flashmla@compare | 211.4 | 202.4 | 0.51 | 975.7 | 774.8 | 0.61 | 309 |
| single-16384-hca-cp8r7 | cute@compare | 389.9 | 127.7 | 0.94 | 1563 | 608.7 | 0.99 | 306 |
| single-16384-hca-cp8r7 | cute_ws@compare | 171.8 | 53.9 | 0.41 | 1347 | 686.4 | 0.85 | 306 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 200.8 | 142.0 | 0.48 | - | - | - | 133 |
| single-16384-sliding-cp1 | tilelang@main | 2290 | 682.5 | 1.00 | 7470 | 816.3 | 1.00 | 2108 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 1049 | 115.4 | 0.46 | - | - | - | 1048 |
| single-16384-sliding-cp1 | tilelang@compare | 2271 | 697.9 | 1.00 | 7437 | 878.0 | 1.00 | 2108 |
| single-16384-sliding-cp1 | cudnn_flashmla@compare | 1060 | 184.0 | 0.47 | 5007 | 333.3 | 0.67 | 2124 |
| single-16384-sliding-cp1 | cute@compare | 2100 | 113.7 | 0.92 | 7262 | 238.9 | 0.98 | 2108 |
| single-16384-sliding-cp1 | cute_ws@compare | 789.9 | 60.1 | 0.35 | 5949 | 122.7 | 0.80 | 2108 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1034 | 124.9 | 0.46 | - | - | - | 1048 |
| single-16384-sliding-cp8r0 | tilelang@main | 310.8 | 707.7 | 1.00 | 1046 | 1288 | 1.00 | 306 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 150.5 | 149.3 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r0 | tilelang@compare | 310.3 | 698.1 | 1.00 | 1046 | 1360 | 1.00 | 306 |
| single-16384-sliding-cp8r0 | cudnn_flashmla@compare | 159.0 | 202.1 | 0.51 | 733.1 | 828.3 | 0.70 | 308 |
| single-16384-sliding-cp8r0 | cute@compare | 287.0 | 125.6 | 0.92 | 1017 | 720.9 | 0.97 | 306 |
| single-16384-sliding-cp8r0 | cute_ws@compare | 115.5 | 56.0 | 0.37 | 850.3 | 744.4 | 0.81 | 306 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 148.1 | 144.7 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r4 | tilelang@main | 315.2 | 689.1 | 1.00 | 1080 | 1310 | 1.00 | 306 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 150.5 | 146.2 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r4 | tilelang@compare | 314.4 | 700.9 | 1.00 | 1081 | 1336 | 1.00 | 306 |
| single-16384-sliding-cp8r4 | cudnn_flashmla@compare | 159.7 | 201.5 | 0.51 | 743.3 | 813.4 | 0.69 | 308 |
| single-16384-sliding-cp8r4 | cute@compare | 292.6 | 127.6 | 0.93 | 1059 | 717.2 | 0.98 | 306 |
| single-16384-sliding-cp8r4 | cute_ws@compare | 118.9 | 52.1 | 0.38 | 889.3 | 737.8 | 0.82 | 306 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 148.1 | 151.2 | 0.47 | - | - | - | 131 |
| single-16384-sliding-cp8r7 | tilelang@main | 314.2 | 682.0 | 1.00 | 1084 | 1305 | 1.00 | 306 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 150.6 | 143.4 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r7 | tilelang@compare | 315.6 | 705.4 | 1.00 | 1085 | 1370 | 1.00 | 306 |
| single-16384-sliding-cp8r7 | cudnn_flashmla@compare | 159.5 | 198.9 | 0.51 | 747.6 | 814.0 | 0.69 | 308 |
| single-16384-sliding-cp8r7 | cute@compare | 292.4 | 126.8 | 0.93 | 1062 | 722.4 | 0.98 | 306 |
| single-16384-sliding-cp8r7 | cute_ws@compare | 116.0 | 54.8 | 0.37 | 890.5 | 747.3 | 0.82 | 306 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 148.8 | 146.6 | 0.47 | - | - | - | 131 |
| short-16384-csa-cp1 | tilelang@main | 3853 | 656.0 | 1.00 | 15075 | 901.5 | 1.00 | 2120 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 1929 | 28.0 | 0.50 | - | - | - | 1112 |
| short-16384-csa-cp1 | tilelang@compare | 3915 | 581.6 | 1.00 | 15005 | 988.7 | 1.00 | 2120 |
| short-16384-csa-cp1 | cudnn_flashmla@compare | 1930 | 92.9 | 0.49 | 8561 | 280.3 | 0.57 | 2168 |
| short-16384-csa-cp1 | cute@compare | 3620 | 73.7 | 0.92 | 14798 | 350.7 | 0.99 | 2120 |
| short-16384-csa-cp1 | cute_ws@compare | 1729 | 63.5 | 0.44 | 12908 | 279.6 | 0.86 | 2120 |
| short-16384-csa-cp1 | flashmla_fwd_ref@compare | 1919 | 66.8 | 0.49 | - | - | - | 1112 |
| short-16384-csa-cp8r0 | tilelang@main | 478.1 | 697.2 | 1.00 | 1852 | 1145 | 1.00 | 317 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 254.2 | 135.9 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r0 | tilelang@compare | 475.6 | 714.4 | 1.00 | 1843 | 1170 | 1.00 | 317 |
| short-16384-csa-cp8r0 | cudnn_flashmla@compare | 261.5 | 190.1 | 0.55 | 1133 | 695.6 | 0.61 | 324 |
| short-16384-csa-cp8r0 | cute@compare | 449.4 | 121.1 | 0.95 | 1816 | 551.2 | 0.99 | 317 |
| short-16384-csa-cp8r0 | cute_ws@compare | 225.8 | 52.4 | 0.47 | 1595 | 618.5 | 0.87 | 317 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 252.8 | 132.7 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r4 | tilelang@main | 566.4 | 683.1 | 1.00 | 2358 | 1029 | 1.00 | 317 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 295.5 | 136.1 | 0.52 | - | - | - | 139 |
| short-16384-csa-cp8r4 | tilelang@compare | 563.0 | 707.1 | 1.00 | 2350 | 1113 | 1.00 | 317 |
| short-16384-csa-cp8r4 | cudnn_flashmla@compare | 301.8 | 189.9 | 0.54 | 1359 | 666.9 | 0.58 | 324 |
| short-16384-csa-cp8r4 | cute@compare | 533.3 | 119.3 | 0.95 | 2319 | 490.4 | 0.99 | 317 |
| short-16384-csa-cp8r4 | cute_ws@compare | 264.2 | 54.4 | 0.47 | 2051 | 602.3 | 0.87 | 317 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 294.6 | 137.2 | 0.52 | - | - | - | 139 |
| short-16384-csa-cp8r7 | tilelang@main | 524.8 | 685.9 | 1.00 | 2140 | 1101 | 1.00 | 317 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 276.9 | 131.2 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r7 | tilelang@compare | 522.1 | 683.2 | 1.00 | 2124 | 1195 | 1.00 | 317 |
| short-16384-csa-cp8r7 | cudnn_flashmla@compare | 284.9 | 188.8 | 0.55 | 1258 | 706.9 | 0.59 | 324 |
| short-16384-csa-cp8r7 | cute@compare | 495.6 | 119.3 | 0.95 | 2101 | 539.9 | 0.99 | 317 |
| short-16384-csa-cp8r7 | cute_ws@compare | 252.3 | 52.9 | 0.48 | 1858 | 627.7 | 0.87 | 317 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 276.3 | 136.1 | 0.53 | - | - | - | 139 |
| short-16384-hca-cp1 | tilelang@main | 2599 | 738.1 | 1.00 | 8299 | 926.1 | 1.00 | 2120 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 1361 | 102.2 | 0.52 | - | - | - | 1064 |
| short-16384-hca-cp1 | tilelang@compare | 2593 | 730.4 | 1.00 | 8244 | 895.7 | 1.00 | 2120 |
| short-16384-hca-cp1 | cudnn_flashmla@compare | 1368 | 178.7 | 0.53 | 5586 | 295.5 | 0.68 | 2132 |
| short-16384-hca-cp1 | cute@compare | 2416 | 127.7 | 0.93 | 8070 | 243.4 | 0.98 | 2120 |
| short-16384-hca-cp1 | cute_ws@compare | 1176 | 54.7 | 0.45 | 6823 | 119.4 | 0.83 | 2120 |
| short-16384-hca-cp1 | flashmla_fwd_ref@compare | 1369 | 89.0 | 0.53 | - | - | - | 1064 |
| short-16384-hca-cp8r0 | tilelang@main | 349.8 | 720.5 | 1.00 | 1148 | 1351 | 1.00 | 307 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 196.9 | 142.0 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r0 | tilelang@compare | 347.8 | 745.2 | 1.00 | 1143 | 1406 | 1.00 | 307 |
| short-16384-hca-cp8r0 | cudnn_flashmla@compare | 204.7 | 203.6 | 0.59 | 814.3 | 780.9 | 0.71 | 309 |
| short-16384-hca-cp8r0 | cute@compare | 326.8 | 142.6 | 0.94 | 1123 | 788.0 | 0.98 | 307 |
| short-16384-hca-cp8r0 | cute_ws@compare | 163.6 | 54.2 | 0.47 | 963.8 | 738.4 | 0.84 | 307 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 193.5 | 148.8 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r4 | tilelang@main | 362.1 | 717.9 | 1.00 | 1211 | 1322 | 1.00 | 307 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 200.7 | 136.7 | 0.55 | - | - | - | 133 |
| short-16384-hca-cp8r4 | tilelang@compare | 362.4 | 738.9 | 1.00 | 1209 | 1430 | 1.00 | 307 |
| short-16384-hca-cp8r4 | cudnn_flashmla@compare | 208.4 | 199.1 | 0.58 | 838.0 | 791.8 | 0.69 | 309 |
| short-16384-hca-cp8r4 | cute@compare | 338.7 | 145.1 | 0.93 | 1184 | 775.4 | 0.98 | 307 |
| short-16384-hca-cp8r4 | cute_ws@compare | 169.1 | 51.9 | 0.47 | 1017 | 745.9 | 0.84 | 307 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 199.0 | 145.7 | 0.55 | - | - | - | 133 |
| short-16384-hca-cp8r7 | tilelang@main | 355.1 | 726.0 | 1.00 | 1179 | 1380 | 1.00 | 307 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 198.1 | 140.8 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r7 | tilelang@compare | 353.3 | 741.1 | 1.00 | 1177 | 1394 | 1.00 | 307 |
| short-16384-hca-cp8r7 | cudnn_flashmla@compare | 205.0 | 209.3 | 0.58 | 827.6 | 780.3 | 0.70 | 309 |
| short-16384-hca-cp8r7 | cute@compare | 331.3 | 149.4 | 0.94 | 1153 | 779.0 | 0.98 | 307 |
| short-16384-hca-cp8r7 | cute_ws@compare | 166.0 | 54.1 | 0.47 | 990.9 | 732.6 | 0.84 | 307 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 197.4 | 145.7 | 0.56 | - | - | - | 133 |
| short-16384-sliding-cp1 | tilelang@main | 2264 | 720.8 | 1.00 | 7323 | 832.1 | 1.00 | 2108 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 1041 | 128.7 | 0.46 | - | - | - | 1048 |
| short-16384-sliding-cp1 | tilelang@compare | 2260 | 696.4 | 1.00 | 7287 | 836.5 | 1.00 | 2108 |
| short-16384-sliding-cp1 | cudnn_flashmla@compare | 1063 | 184.1 | 0.47 | 4933 | 325.1 | 0.68 | 2124 |
| short-16384-sliding-cp1 | cute@compare | 2091 | 112.4 | 0.93 | 7118 | 218.7 | 0.98 | 2108 |
| short-16384-sliding-cp1 | cute_ws@compare | 791.2 | 59.6 | 0.35 | 5823 | 119.5 | 0.80 | 2108 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1037 | 128.9 | 0.46 | - | - | - | 1048 |
| short-16384-sliding-cp8r0 | tilelang@main | 303.6 | 697.8 | 1.00 | 1025 | 1281 | 1.00 | 306 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 149.7 | 146.6 | 0.49 | - | - | - | 131 |
| short-16384-sliding-cp8r0 | tilelang@compare | 305.9 | 697.1 | 1.00 | 1023 | 1390 | 1.00 | 306 |
| short-16384-sliding-cp8r0 | cudnn_flashmla@compare | 159.6 | 204.8 | 0.52 | 725.8 | 833.7 | 0.71 | 308 |
| short-16384-sliding-cp8r0 | cute@compare | 282.1 | 127.2 | 0.92 | 1003 | 740.2 | 0.98 | 306 |
| short-16384-sliding-cp8r0 | cute_ws@compare | 116.5 | 52.8 | 0.38 | 836.4 | 753.5 | 0.82 | 306 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 148.9 | 146.5 | 0.49 | - | - | - | 131 |
| short-16384-sliding-cp8r4 | tilelang@main | 314.2 | 679.7 | 1.00 | 1072 | 1298 | 1.00 | 306 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 147.1 | 147.3 | 0.47 | - | - | - | 131 |
| short-16384-sliding-cp8r4 | tilelang@compare | 315.0 | 700.1 | 1.00 | 1066 | 1337 | 1.00 | 306 |
| short-16384-sliding-cp8r4 | cudnn_flashmla@compare | 160.0 | 204.6 | 0.51 | 739.4 | 805.9 | 0.69 | 308 |
| short-16384-sliding-cp8r4 | cute@compare | 294.0 | 126.4 | 0.93 | 1051 | 705.6 | 0.99 | 306 |
| short-16384-sliding-cp8r4 | cute_ws@compare | 117.5 | 53.3 | 0.37 | 877.5 | 743.8 | 0.82 | 306 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 151.9 | 145.4 | 0.48 | - | - | - | 131 |
| short-16384-sliding-cp8r7 | tilelang@main | 312.3 | 689.1 | 1.00 | 1051 | 1310 | 1.00 | 306 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 150.9 | 143.6 | 0.48 | - | - | - | 131 |
| short-16384-sliding-cp8r7 | tilelang@compare | 312.7 | 729.1 | 1.00 | 1052 | 1350 | 1.00 | 306 |
| short-16384-sliding-cp8r7 | cudnn_flashmla@compare | 159.0 | 202.9 | 0.51 | 731.1 | 826.6 | 0.70 | 308 |
| short-16384-sliding-cp8r7 | cute@compare | 290.0 | 125.3 | 0.93 | 1031 | 731.1 | 0.98 | 306 |
| short-16384-sliding-cp8r7 | cute_ws@compare | 118.0 | 52.6 | 0.38 | 866.5 | 741.6 | 0.82 | 306 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 150.3 | 144.0 | 0.48 | - | - | - | 131 |
| heavy-16384-csa-cp1 | tilelang@main | 3305 | 664.5 | 1.00 | 12082 | 914.7 | 1.00 | 2120 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1691 | 34.9 | 0.51 | - | - | - | 1112 |
| heavy-16384-csa-cp1 | tilelang@compare | 3359 | 612.6 | 1.00 | 12029 | 887.1 | 1.00 | 2120 |
| heavy-16384-csa-cp1 | cudnn_flashmla@compare | 1710 | 79.3 | 0.51 | 7225 | 228.4 | 0.60 | 2168 |
| heavy-16384-csa-cp1 | cute@compare | 3117 | 74.1 | 0.93 | 11820 | 287.2 | 0.98 | 2120 |
| heavy-16384-csa-cp1 | cute_ws@compare | 1505 | 52.1 | 0.45 | 10236 | 191.3 | 0.85 | 2120 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@compare | 1703 | 30.2 | 0.51 | - | - | - | 1112 |
| heavy-16384-csa-cp8r0 | tilelang@main | 395.5 | 697.0 | 1.00 | 1364 | 1211 | 1.00 | 317 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 213.5 | 133.0 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r0 | tilelang@compare | 392.2 | 696.8 | 1.00 | 1363 | 1244 | 1.00 | 317 |
| heavy-16384-csa-cp8r0 | cudnn_flashmla@compare | 220.9 | 191.1 | 0.56 | 906.1 | 744.1 | 0.66 | 323 |
| heavy-16384-csa-cp8r0 | cute@compare | 368.5 | 120.5 | 0.94 | 1341 | 634.6 | 0.98 | 317 |
| heavy-16384-csa-cp8r0 | cute_ws@compare | 185.4 | 52.6 | 0.47 | 1159 | 670.5 | 0.85 | 317 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 213.3 | 137.0 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r4 | tilelang@main | 433.6 | 683.1 | 1.00 | 1614 | 1148 | 1.00 | 317 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 232.8 | 129.9 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r4 | tilelang@compare | 433.1 | 702.8 | 1.00 | 1609 | 1269 | 1.00 | 317 |
| heavy-16384-csa-cp8r4 | cudnn_flashmla@compare | 240.8 | 201.3 | 0.56 | 1018 | 743.0 | 0.63 | 323 |
| heavy-16384-csa-cp8r4 | cute@compare | 406.6 | 120.6 | 0.94 | 1582 | 620.7 | 0.98 | 317 |
| heavy-16384-csa-cp8r4 | cute_ws@compare | 202.0 | 55.1 | 0.47 | 1381 | 660.5 | 0.86 | 317 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 235.6 | 131.1 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r7 | tilelang@main | 579.5 | 684.9 | 1.00 | 2413 | 1069 | 1.00 | 317 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 301.6 | 126.9 | 0.52 | - | - | - | 139 |
| heavy-16384-csa-cp8r7 | tilelang@compare | 575.0 | 698.0 | 1.00 | 2415 | 1104 | 1.00 | 317 |
| heavy-16384-csa-cp8r7 | cudnn_flashmla@compare | 308.9 | 193.8 | 0.54 | 1388 | 682.0 | 0.57 | 323 |
| heavy-16384-csa-cp8r7 | cute@compare | 546.9 | 123.5 | 0.95 | 2380 | 484.6 | 0.99 | 317 |
| heavy-16384-csa-cp8r7 | cute_ws@compare | 269.6 | 56.3 | 0.47 | 2104 | 604.2 | 0.87 | 317 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 300.6 | 139.4 | 0.52 | - | - | - | 139 |
| heavy-16384-hca-cp1 | tilelang@main | 2507 | 744.8 | 1.00 | 7784 | 921.8 | 1.00 | 2120 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 1309 | 94.9 | 0.52 | - | - | - | 1064 |
| heavy-16384-hca-cp1 | tilelang@compare | 2492 | 719.8 | 1.00 | 7758 | 850.4 | 1.00 | 2120 |
| heavy-16384-hca-cp1 | cudnn_flashmla@compare | 1317 | 158.1 | 0.53 | 5329 | 299.4 | 0.69 | 2132 |
| heavy-16384-hca-cp1 | cute@compare | 2320 | 128.3 | 0.93 | 7573 | 255.0 | 0.98 | 2120 |
| heavy-16384-hca-cp1 | cute_ws@compare | 1102 | 61.1 | 0.44 | 6366 | 112.4 | 0.82 | 2120 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@compare | 1304 | 93.6 | 0.52 | - | - | - | 1064 |
| heavy-16384-hca-cp8r0 | tilelang@main | 333.5 | 734.1 | 1.00 | 1069 | 1379 | 1.00 | 308 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 184.9 | 150.6 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r0 | tilelang@compare | 333.9 | 739.2 | 1.00 | 1071 | 1439 | 1.00 | 308 |
| heavy-16384-hca-cp8r0 | cudnn_flashmla@compare | 193.0 | 205.2 | 0.58 | 770.3 | 794.9 | 0.72 | 309 |
| heavy-16384-hca-cp8r0 | cute@compare | 310.0 | 148.5 | 0.93 | 1046 | 805.3 | 0.98 | 308 |
| heavy-16384-hca-cp8r0 | cute_ws@compare | 153.2 | 53.7 | 0.46 | 890.9 | 760.7 | 0.83 | 308 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 183.5 | 150.3 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r4 | tilelang@main | 343.8 | 725.9 | 1.00 | 1127 | 1361 | 1.00 | 308 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 190.5 | 151.2 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r4 | tilelang@compare | 344.7 | 719.5 | 1.00 | 1128 | 1445 | 1.00 | 308 |
| heavy-16384-hca-cp8r4 | cudnn_flashmla@compare | 201.0 | 203.5 | 0.58 | 804.7 | 800.3 | 0.71 | 309 |
| heavy-16384-hca-cp8r4 | cute@compare | 323.8 | 146.0 | 0.94 | 1107 | 803.6 | 0.98 | 308 |
| heavy-16384-hca-cp8r4 | cute_ws@compare | 160.9 | 51.1 | 0.47 | 946.5 | 763.6 | 0.84 | 308 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 190.8 | 147.9 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r7 | tilelang@main | 366.8 | 721.2 | 1.00 | 1231 | 1390 | 1.00 | 308 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 203.8 | 142.3 | 0.56 | - | - | - | 133 |
| heavy-16384-hca-cp8r7 | tilelang@compare | 366.6 | 740.1 | 1.00 | 1224 | 1425 | 1.00 | 308 |
| heavy-16384-hca-cp8r7 | cudnn_flashmla@compare | 211.7 | 194.3 | 0.58 | 847.6 | 796.3 | 0.69 | 309 |
| heavy-16384-hca-cp8r7 | cute@compare | 341.4 | 145.5 | 0.93 | 1205 | 772.2 | 0.98 | 308 |
| heavy-16384-hca-cp8r7 | cute_ws@compare | 170.8 | 54.3 | 0.47 | 1042 | 735.7 | 0.85 | 308 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 200.0 | 147.9 | 0.55 | - | - | - | 133 |
| heavy-16384-sliding-cp1 | tilelang@main | 2247 | 731.8 | 1.00 | 7046 | 841.1 | 1.00 | 2108 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 1042 | 130.6 | 0.46 | - | - | - | 1048 |
| heavy-16384-sliding-cp1 | tilelang@compare | 2255 | 704.2 | 1.00 | 7010 | 854.4 | 1.00 | 2108 |
| heavy-16384-sliding-cp1 | cudnn_flashmla@compare | 1068 | 183.5 | 0.47 | 4814 | 334.4 | 0.69 | 2124 |
| heavy-16384-sliding-cp1 | cute@compare | 2085 | 111.8 | 0.92 | 6854 | 217.9 | 0.98 | 2108 |
| heavy-16384-sliding-cp1 | cute_ws@compare | 812.6 | 58.2 | 0.36 | 5589 | 112.8 | 0.80 | 2108 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1054 | 120.2 | 0.47 | - | - | - | 1048 |
| heavy-16384-sliding-cp8r0 | tilelang@main | 300.8 | 712.1 | 1.00 | 974.6 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 148.3 | 149.4 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r0 | tilelang@compare | 299.2 | 713.4 | 1.00 | 976.6 | 1385 | 1.00 | 306 |
| heavy-16384-sliding-cp8r0 | cudnn_flashmla@compare | 158.8 | 202.0 | 0.53 | 699.0 | 835.0 | 0.72 | 308 |
| heavy-16384-sliding-cp8r0 | cute@compare | 278.9 | 123.9 | 0.93 | 952.6 | 749.2 | 0.98 | 306 |
| heavy-16384-sliding-cp8r0 | cute_ws@compare | 116.5 | 53.8 | 0.39 | 796.4 | 759.7 | 0.82 | 306 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 150.8 | 146.8 | 0.50 | - | - | - | 131 |
| heavy-16384-sliding-cp8r4 | tilelang@main | 303.7 | 679.0 | 1.00 | 1014 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 149.5 | 143.6 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r4 | tilelang@compare | 306.3 | 700.5 | 1.00 | 1017 | 1358 | 1.00 | 306 |
| heavy-16384-sliding-cp8r4 | cudnn_flashmla@compare | 158.6 | 204.0 | 0.52 | 715.9 | 833.8 | 0.70 | 308 |
| heavy-16384-sliding-cp8r4 | cute@compare | 285.0 | 126.0 | 0.93 | 995.4 | 736.3 | 0.98 | 306 |
| heavy-16384-sliding-cp8r4 | cute_ws@compare | 118.3 | 52.2 | 0.39 | 834.7 | 753.9 | 0.82 | 306 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 150.1 | 145.6 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r7 | tilelang@main | 314.8 | 707.2 | 1.00 | 1082 | 1323 | 1.00 | 306 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 147.4 | 149.0 | 0.47 | - | - | - | 131 |
| heavy-16384-sliding-cp8r7 | tilelang@compare | 314.5 | 712.9 | 1.00 | 1084 | 1391 | 1.00 | 306 |
| heavy-16384-sliding-cp8r7 | cudnn_flashmla@compare | 159.7 | 203.1 | 0.51 | 745.2 | 819.5 | 0.69 | 308 |
| heavy-16384-sliding-cp8r7 | cute@compare | 293.2 | 132.2 | 0.93 | 1062 | 727.1 | 0.98 | 306 |
| heavy-16384-sliding-cp8r7 | cute_ws@compare | 118.3 | 54.8 | 0.38 | 890.5 | 745.5 | 0.82 | 306 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 149.6 | 155.0 | 0.48 | - | - | - | 131 |
| tiny-16384-csa-cp1 | tilelang@main | 2636 | 650.2 | 1.00 | 7892 | 833.7 | 1.00 | 2120 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 1437 | 39.5 | 0.54 | - | - | - | 1112 |
| tiny-16384-csa-cp1 | tilelang@compare | 2637 | 658.8 | 1.00 | 7894 | 883.9 | 1.00 | 2120 |
| tiny-16384-csa-cp1 | cudnn_flashmla@compare | 1453 | 83.9 | 0.55 | 5333 | 223.7 | 0.68 | 2168 |
| tiny-16384-csa-cp1 | cute@compare | 2466 | 76.8 | 0.94 | 7716 | 186.8 | 0.98 | 2120 |
| tiny-16384-csa-cp1 | cute_ws@compare | 1241 | 59.7 | 0.47 | 6500 | 110.5 | 0.82 | 2120 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@compare | 1445 | 31.7 | 0.55 | - | - | - | 1112 |
| tiny-16384-csa-cp8r0 | tilelang@main | 354.2 | 688.8 | 1.00 | 1109 | 1267 | 1.00 | 317 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 200.1 | 134.4 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r0 | tilelang@compare | 353.6 | 710.4 | 1.00 | 1107 | 1334 | 1.00 | 317 |
| tiny-16384-csa-cp8r0 | cudnn_flashmla@compare | 209.8 | 197.7 | 0.59 | 778.8 | 772.8 | 0.70 | 323 |
| tiny-16384-csa-cp8r0 | cute@compare | 331.9 | 122.4 | 0.94 | 1087 | 685.7 | 0.98 | 317 |
| tiny-16384-csa-cp8r0 | cute_ws@compare | 174.0 | 53.4 | 0.49 | 931.9 | 680.0 | 0.84 | 317 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 200.5 | 140.1 | 0.57 | - | - | - | 139 |
| tiny-16384-csa-cp8r4 | tilelang@main | 356.2 | 692.0 | 1.00 | 1108 | 1250 | 1.00 | 317 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 200.5 | 141.7 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r4 | tilelang@compare | 356.5 | 683.2 | 1.00 | 1108 | 1312 | 1.00 | 317 |
| tiny-16384-csa-cp8r4 | cudnn_flashmla@compare | 208.4 | 192.9 | 0.58 | 779.1 | 766.1 | 0.70 | 323 |
| tiny-16384-csa-cp8r4 | cute@compare | 333.0 | 118.6 | 0.93 | 1085 | 680.2 | 0.98 | 317 |
| tiny-16384-csa-cp8r4 | cute_ws@compare | 174.8 | 53.0 | 0.49 | 928.8 | 689.4 | 0.84 | 317 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 200.2 | 134.8 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r7 | tilelang@main | 356.2 | 675.8 | 1.00 | 1099 | 1270 | 1.00 | 317 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 200.3 | 134.2 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r7 | tilelang@compare | 355.3 | 688.7 | 1.00 | 1098 | 1340 | 1.00 | 317 |
| tiny-16384-csa-cp8r7 | cudnn_flashmla@compare | 210.4 | 184.3 | 0.59 | 771.8 | 800.2 | 0.70 | 323 |
| tiny-16384-csa-cp8r7 | cute@compare | 332.8 | 119.1 | 0.94 | 1077 | 702.2 | 0.98 | 317 |
| tiny-16384-csa-cp8r7 | cute_ws@compare | 174.5 | 53.6 | 0.49 | 919.8 | 690.4 | 0.84 | 317 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 202.1 | 136.1 | 0.57 | - | - | - | 139 |
| tiny-16384-hca-cp1 | tilelang@main | 2024 | 683.5 | 1.00 | 5355 | 828.5 | 1.00 | 2108 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 1052 | 124.0 | 0.52 | - | - | - | 1048 |
| tiny-16384-hca-cp1 | tilelang@compare | 2029 | 689.2 | 1.00 | 5354 | 825.3 | 1.00 | 2108 |
| tiny-16384-hca-cp1 | cudnn_flashmla@compare | 1080 | 175.4 | 0.53 | 4165 | 310.6 | 0.78 | 2124 |
| tiny-16384-hca-cp1 | cute@compare | 1866 | 109.4 | 0.92 | 5185 | 218.8 | 0.97 | 2108 |
| tiny-16384-hca-cp1 | cute_ws@compare | 922.2 | 53.0 | 0.45 | 4257 | 98.1 | 0.80 | 2108 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@compare | 1055 | 126.1 | 0.52 | - | - | - | 1048 |
| tiny-16384-hca-cp8r0 | tilelang@main | 278.4 | 702.7 | 1.00 | 784.0 | 1360 | 1.00 | 306 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 149.1 | 147.5 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r0 | tilelang@compare | 278.0 | 717.6 | 1.00 | 784.9 | 1391 | 1.00 | 306 |
| tiny-16384-hca-cp8r0 | cudnn_flashmla@compare | 156.9 | 204.1 | 0.56 | 625.1 | 893.6 | 0.80 | 308 |
| tiny-16384-hca-cp8r0 | cute@compare | 257.3 | 125.4 | 0.93 | 763.8 | 760.7 | 0.97 | 306 |
| tiny-16384-hca-cp8r0 | cute_ws@compare | 113.7 | 53.5 | 0.41 | 634.3 | 740.6 | 0.81 | 306 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 148.2 | 152.0 | 0.53 | - | - | - | 131 |
| tiny-16384-hca-cp8r4 | tilelang@main | 276.8 | 691.4 | 1.00 | 771.3 | 1324 | 1.00 | 306 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 148.8 | 144.3 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r4 | tilelang@compare | 275.8 | 701.2 | 1.00 | 772.7 | 1410 | 1.00 | 306 |
| tiny-16384-hca-cp8r4 | cudnn_flashmla@compare | 157.8 | 203.2 | 0.57 | 615.8 | 884.7 | 0.80 | 308 |
| tiny-16384-hca-cp8r4 | cute@compare | 255.9 | 128.1 | 0.93 | 751.0 | 769.2 | 0.97 | 306 |
| tiny-16384-hca-cp8r4 | cute_ws@compare | 115.3 | 52.5 | 0.42 | 623.4 | 737.7 | 0.81 | 306 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 147.8 | 148.7 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r7 | tilelang@main | 268.8 | 686.6 | 1.00 | 754.2 | 1376 | 1.00 | 306 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 149.0 | 145.3 | 0.55 | - | - | - | 131 |
| tiny-16384-hca-cp8r7 | tilelang@compare | 268.1 | 704.2 | 1.00 | 752.1 | 1397 | 1.00 | 306 |
| tiny-16384-hca-cp8r7 | cudnn_flashmla@compare | 156.9 | 204.9 | 0.59 | 609.6 | 886.6 | 0.81 | 308 |
| tiny-16384-hca-cp8r7 | cute@compare | 246.8 | 124.8 | 0.92 | 733.0 | 760.0 | 0.97 | 306 |
| tiny-16384-hca-cp8r7 | cute_ws@compare | 114.7 | 53.4 | 0.43 | 611.0 | 723.3 | 0.81 | 306 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 148.5 | 146.2 | 0.55 | - | - | - | 131 |
| tiny-16384-sliding-cp1 | tilelang@main | 2025 | 703.4 | 1.00 | 5346 | 817.0 | 1.00 | 2108 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 1050 | 130.1 | 0.52 | - | - | - | 1048 |
| tiny-16384-sliding-cp1 | tilelang@compare | 2026 | 705.7 | 1.00 | 5351 | 845.2 | 1.00 | 2108 |
| tiny-16384-sliding-cp1 | cudnn_flashmla@compare | 1076 | 183.0 | 0.53 | 4140 | 332.1 | 0.77 | 2124 |
| tiny-16384-sliding-cp1 | cute@compare | 1859 | 117.1 | 0.92 | 5187 | 218.0 | 0.97 | 2108 |
| tiny-16384-sliding-cp1 | cute_ws@compare | 933.6 | 46.4 | 0.46 | 4256 | 104.0 | 0.80 | 2108 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@compare | 1051 | 129.1 | 0.52 | - | - | - | 1048 |
| tiny-16384-sliding-cp8r0 | tilelang@main | 278.4 | 684.2 | 1.00 | 784.9 | 1357 | 1.00 | 306 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 147.7 | 149.3 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r0 | tilelang@compare | 279.7 | 685.8 | 1.00 | 785.0 | 1438 | 1.00 | 306 |
| tiny-16384-sliding-cp8r0 | cudnn_flashmla@compare | 157.9 | 201.2 | 0.56 | 622.0 | 944.5 | 0.79 | 308 |
| tiny-16384-sliding-cp8r0 | cute@compare | 257.0 | 124.3 | 0.92 | 762.8 | 796.8 | 0.97 | 306 |
| tiny-16384-sliding-cp8r0 | cute_ws@compare | 115.1 | 51.8 | 0.41 | 632.0 | 776.0 | 0.81 | 306 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 148.1 | 150.9 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r4 | tilelang@main | 276.5 | 685.8 | 1.00 | 772.0 | 1402 | 1.00 | 306 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 147.7 | 151.4 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r4 | tilelang@compare | 277.0 | 702.9 | 1.00 | 773.7 | 1410 | 1.00 | 306 |
| tiny-16384-sliding-cp8r4 | cudnn_flashmla@compare | 157.9 | 208.5 | 0.57 | 616.6 | 906.7 | 0.80 | 308 |
| tiny-16384-sliding-cp8r4 | cute@compare | 256.4 | 127.7 | 0.93 | 751.4 | 768.1 | 0.97 | 306 |
| tiny-16384-sliding-cp8r4 | cute_ws@compare | 115.1 | 54.7 | 0.42 | 623.8 | 747.4 | 0.81 | 306 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 148.0 | 149.0 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r7 | tilelang@main | 267.7 | 693.4 | 1.00 | 754.5 | 1385 | 1.00 | 306 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 148.1 | 145.0 | 0.55 | - | - | - | 131 |
| tiny-16384-sliding-cp8r7 | tilelang@compare | 267.9 | 705.5 | 1.00 | 753.0 | 1425 | 1.00 | 306 |
| tiny-16384-sliding-cp8r7 | cudnn_flashmla@compare | 157.6 | 205.0 | 0.59 | 608.9 | 909.7 | 0.81 | 308 |
| tiny-16384-sliding-cp8r7 | cute@compare | 246.4 | 130.6 | 0.92 | 731.5 | 773.0 | 0.97 | 306 |
| tiny-16384-sliding-cp8r7 | cute_ws@compare | 114.1 | 56.3 | 0.43 | 612.5 | 740.0 | 0.81 | 306 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 151.8 | 146.5 | 0.57 | - | - | - | 131 |
| single-49208-csa-cp1 | tilelang@main | 16237 | 187.0 | 1.00 | 69993 | 715.1 | 1.00 | 6368 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 7737 | 571.7 | 0.48 | - | - | - | 3340 |
| single-49208-csa-cp1 | tilelang@compare | 16419 | 74.0 | 1.00 | 70059 | 795.4 | 1.00 | 6368 |
| single-49208-csa-cp1 | cudnn_flashmla@compare | 7762 | 538.4 | 0.47 | 38518 | 970.3 | 0.55 | 6513 |
| single-49208-csa-cp1 | cute@compare | 15561 | -222.1 | 0.95 | 69837 | -162.5 | 1.00 | 6368 |
| single-49208-csa-cp1 | cute_ws@compare | 7271 | 613.9 | 0.44 | 61476 | 267.2 | 0.88 | 6368 |
| single-49208-csa-cp1 | flashmla_fwd_ref@compare | 8405 | 127.7 | 0.51 | - | - | - | 3340 |
| single-49208-csa-cp8r0 | tilelang@main | 1866 | 689.5 | 1.00 | 8084 | 839.4 | 1.00 | 954 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 922.1 | 104.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r0 | tilelang@compare | 1904 | 660.5 | 1.00 | 8067 | 910.5 | 1.00 | 954 |
| single-49208-csa-cp8r0 | cudnn_flashmla@compare | 935.0 | 166.0 | 0.49 | 4404 | 329.1 | 0.55 | 972 |
| single-49208-csa-cp8r0 | cute@compare | 1784 | 106.5 | 0.94 | 7986 | 272.5 | 0.99 | 954 |
| single-49208-csa-cp8r0 | cute_ws@compare | 849.6 | 59.6 | 0.45 | 7071 | 143.7 | 0.88 | 954 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 927.5 | 97.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r4 | tilelang@main | 2035 | 709.5 | 1.00 | 9123 | 902.5 | 1.00 | 954 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 999.9 | 100.6 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r4 | tilelang@compare | 2049 | 746.0 | 1.00 | 9191 | 943.3 | 1.00 | 954 |
| single-49208-csa-cp8r4 | cudnn_flashmla@compare | 1011 | 184.0 | 0.49 | 4896 | 333.7 | 0.53 | 973 |
| single-49208-csa-cp8r4 | cute@compare | 1951 | 137.9 | 0.95 | 9041 | 336.5 | 0.98 | 954 |
| single-49208-csa-cp8r4 | cute_ws@compare | 925.6 | 67.2 | 0.45 | 8009 | 201.1 | 0.87 | 954 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 1001 | 116.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r7 | tilelang@main | 2039 | 705.7 | 1.00 | 9444 | 917.3 | 1.00 | 954 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1003 | 101.6 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r7 | tilelang@compare | 2127 | 625.1 | 1.00 | 9549 | 916.4 | 1.00 | 954 |
| single-49208-csa-cp8r7 | cudnn_flashmla@compare | 1023 | 155.3 | 0.48 | 4977 | 355.9 | 0.52 | 973 |
| single-49208-csa-cp8r7 | cute@compare | 1969 | 100.6 | 0.93 | 9294 | 403.9 | 0.97 | 954 |
| single-49208-csa-cp8r7 | cute_ws@compare | 932.7 | 56.0 | 0.44 | 8319 | 206.1 | 0.87 | 954 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 1007 | 104.7 | 0.47 | - | - | - | 418 |
| single-49208-hca-cp1 | tilelang@main | 11188 | 307.9 | 1.00 | 41810 | 748.7 | 1.00 | 6334 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 5350 | 51.8 | 0.48 | - | - | - | 3292 |
| single-49208-hca-cp1 | tilelang@compare | 11153 | 383.7 | 1.00 | 41825 | 819.3 | 1.00 | 6334 |
| single-49208-hca-cp1 | cudnn_flashmla@compare | 5368 | 82.4 | 0.48 | 24855 | 158.1 | 0.59 | 6454 |
| single-49208-hca-cp1 | cute@compare | 10263 | 172.7 | 0.92 | 41449 | 83.9 | 0.99 | 6334 |
| single-49208-hca-cp1 | cute_ws@compare | 4752 | 256.3 | 0.43 | 35712 | 204.5 | 0.85 | 6334 |
| single-49208-hca-cp1 | flashmla_fwd_ref@compare | 5526 | 109.1 | 0.50 | - | - | - | 3292 |
| single-49208-hca-cp8r0 | tilelang@main | 1029 | 680.6 | 1.00 | 3422 | 825.2 | 1.00 | 919 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 555.3 | 115.3 | 0.54 | - | - | - | 412 |
| single-49208-hca-cp8r0 | tilelang@compare | 1032 | 700.3 | 1.00 | 3436 | 858.3 | 1.00 | 919 |
| single-49208-hca-cp8r0 | cudnn_flashmla@compare | 571.2 | 178.2 | 0.55 | 2338 | 409.3 | 0.68 | 934 |
| single-49208-hca-cp8r0 | cute@compare | 960.6 | 113.0 | 0.93 | 3358 | 230.0 | 0.98 | 919 |
| single-49208-hca-cp8r0 | cute_ws@compare | 469.6 | 58.2 | 0.46 | 2885 | 178.7 | 0.84 | 919 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 554.5 | 124.1 | 0.54 | - | - | - | 412 |
| single-49208-hca-cp8r4 | tilelang@main | 1459 | 686.0 | 1.00 | 5739 | 825.7 | 1.00 | 919 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 691.9 | 109.3 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r4 | tilelang@compare | 1472 | 682.4 | 1.00 | 5729 | 885.5 | 1.00 | 919 |
| single-49208-hca-cp8r4 | cudnn_flashmla@compare | 701.8 | 171.2 | 0.48 | 3293 | 328.2 | 0.57 | 934 |
| single-49208-hca-cp8r4 | cute@compare | 1387 | 109.8 | 0.94 | 5651 | 238.2 | 0.99 | 919 |
| single-49208-hca-cp8r4 | cute_ws@compare | 613.6 | 56.4 | 0.42 | 4893 | 126.5 | 0.85 | 919 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 684.4 | 121.7 | 0.46 | - | - | - | 412 |
| single-49208-hca-cp8r7 | tilelang@main | 1762 | 668.0 | 1.00 | 7389 | 854.6 | 1.00 | 919 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 831.5 | 113.1 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r7 | tilelang@compare | 1751 | 703.3 | 1.00 | 7385 | 852.7 | 1.00 | 919 |
| single-49208-hca-cp8r7 | cudnn_flashmla@compare | 837.4 | 182.4 | 0.48 | 4045 | 317.8 | 0.55 | 934 |
| single-49208-hca-cp8r7 | cute@compare | 1657 | 126.4 | 0.95 | 7293 | 231.4 | 0.99 | 919 |
| single-49208-hca-cp8r7 | cute_ws@compare | 761.8 | 57.8 | 0.44 | 6394 | 128.2 | 0.87 | 919 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 826.7 | 119.8 | 0.47 | - | - | - | 412 |
| single-49208-sliding-cp1 | tilelang@main | 6768 | 664.9 | 1.00 | 22045 | 843.6 | 1.00 | 6332 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 3083 | 41.8 | 0.46 | - | - | - | 3148 |
| single-49208-sliding-cp1 | tilelang@compare | 6896 | 566.3 | 1.00 | 22087 | 840.9 | 1.00 | 6332 |
| single-49208-sliding-cp1 | cudnn_flashmla@compare | 3128 | 108.9 | 0.45 | 15042 | 342.2 | 0.68 | 6380 |
| single-49208-sliding-cp1 | cute@compare | 6276 | 88.1 | 0.91 | 21581 | 238.8 | 0.98 | 6332 |
| single-49208-sliding-cp1 | cute_ws@compare | 2366 | 60.6 | 0.34 | 17666 | 161.6 | 0.80 | 6332 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3079 | 72.0 | 0.45 | - | - | - | 3148 |
| single-49208-sliding-cp8r0 | tilelang@main | 873.5 | 700.3 | 1.00 | 2906 | 838.8 | 1.00 | 918 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 402.7 | 140.4 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r0 | tilelang@compare | 874.1 | 697.6 | 1.00 | 2905 | 864.2 | 1.00 | 918 |
| single-49208-sliding-cp8r0 | cudnn_flashmla@compare | 413.9 | 197.3 | 0.47 | 1996 | 563.6 | 0.69 | 925 |
| single-49208-sliding-cp8r0 | cute@compare | 808.3 | 123.1 | 0.92 | 2831 | 235.4 | 0.97 | 918 |
| single-49208-sliding-cp8r0 | cute_ws@compare | 315.9 | 52.4 | 0.36 | 2343 | 339.2 | 0.81 | 918 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 402.8 | 146.1 | 0.46 | - | - | - | 393 |
| single-49208-sliding-cp8r4 | tilelang@main | 878.6 | 678.7 | 1.00 | 2949 | 833.9 | 1.00 | 918 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 404.1 | 135.1 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r4 | tilelang@compare | 878.7 | 706.7 | 1.00 | 2949 | 864.7 | 1.00 | 918 |
| single-49208-sliding-cp8r4 | cudnn_flashmla@compare | 414.2 | 199.5 | 0.47 | 2004 | 550.9 | 0.68 | 925 |
| single-49208-sliding-cp8r4 | cute@compare | 812.6 | 128.5 | 0.92 | 2885 | 228.7 | 0.98 | 918 |
| single-49208-sliding-cp8r4 | cute_ws@compare | 314.2 | 55.0 | 0.36 | 2392 | 332.2 | 0.81 | 918 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 407.2 | 130.4 | 0.46 | - | - | - | 393 |
| single-49208-sliding-cp8r7 | tilelang@main | 880.4 | 691.7 | 1.00 | 2941 | 850.3 | 1.00 | 918 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 401.2 | 139.0 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r7 | tilelang@compare | 880.1 | 702.2 | 1.00 | 2946 | 866.8 | 1.00 | 918 |
| single-49208-sliding-cp8r7 | cudnn_flashmla@compare | 416.7 | 208.5 | 0.47 | 2004 | 568.9 | 0.68 | 925 |
| single-49208-sliding-cp8r7 | cute@compare | 810.3 | 128.8 | 0.92 | 2880 | 237.7 | 0.98 | 918 |
| single-49208-sliding-cp8r7 | cute_ws@compare | 314.2 | 56.2 | 0.36 | 2387 | 360.3 | 0.81 | 918 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 403.1 | 144.3 | 0.46 | - | - | - | 393 |
| short-49208-csa-cp1 | tilelang@main | 12688 | 299.8 | 1.00 | 50333 | 736.6 | 1.00 | 6368 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 6166 | 60.7 | 0.49 | - | - | - | 3340 |
| short-49208-csa-cp1 | tilelang@compare | 12939 | 188.9 | 1.00 | 50333 | 834.6 | 1.00 | 6368 |
| short-49208-csa-cp1 | cudnn_flashmla@compare | 6169 | 131.7 | 0.48 | 28641 | 784.5 | 0.57 | 6513 |
| short-49208-csa-cp1 | cute@compare | 11893 | 55.1 | 0.92 | 50140 | 37.1 | 1.00 | 6368 |
| short-49208-csa-cp1 | cute_ws@compare | 5611 | 208.8 | 0.43 | 43545 | 210.5 | 0.87 | 6368 |
| short-49208-csa-cp1 | flashmla_fwd_ref@compare | 6238 | 437.1 | 0.48 | - | - | - | 3340 |
| short-49208-csa-cp8r0 | tilelang@main | 1344 | 692.7 | 1.00 | 5188 | 824.3 | 1.00 | 954 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 686.4 | 104.7 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r0 | tilelang@compare | 1346 | 707.0 | 1.00 | 5195 | 842.8 | 1.00 | 954 |
| short-49208-csa-cp8r0 | cudnn_flashmla@compare | 693.2 | 171.9 | 0.51 | 3091 | 299.7 | 0.60 | 972 |
| short-49208-csa-cp8r0 | cute@compare | 1271 | 113.1 | 0.94 | 5110 | 220.0 | 0.98 | 954 |
| short-49208-csa-cp8r0 | cute_ws@compare | 604.6 | 56.4 | 0.45 | 4448 | 127.1 | 0.86 | 954 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 686.3 | 108.3 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r4 | tilelang@main | 1711 | 674.8 | 1.00 | 7251 | 834.8 | 1.00 | 954 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 852.5 | 107.1 | 0.50 | - | - | - | 418 |
| short-49208-csa-cp8r4 | tilelang@compare | 1722 | 691.3 | 1.00 | 7304 | 808.0 | 1.00 | 954 |
| short-49208-csa-cp8r4 | cudnn_flashmla@compare | 864.3 | 170.5 | 0.50 | 4017 | 323.0 | 0.55 | 972 |
| short-49208-csa-cp8r4 | cute@compare | 1637 | 111.9 | 0.95 | 7199 | 201.9 | 0.99 | 954 |
| short-49208-csa-cp8r4 | cute_ws@compare | 788.5 | 60.4 | 0.46 | 6324 | 141.0 | 0.87 | 954 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 862.2 | 104.6 | 0.50 | - | - | - | 418 |
| short-49208-csa-cp8r7 | tilelang@main | 1484 | 676.8 | 1.00 | 5954 | 855.8 | 1.00 | 954 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 756.8 | 101.5 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r7 | tilelang@compare | 1476 | 691.9 | 1.00 | 5951 | 861.2 | 1.00 | 954 |
| short-49208-csa-cp8r7 | cudnn_flashmla@compare | 764.4 | 161.0 | 0.52 | 3438 | 311.2 | 0.58 | 972 |
| short-49208-csa-cp8r7 | cute@compare | 1398 | 114.8 | 0.95 | 5882 | 219.0 | 0.99 | 954 |
| short-49208-csa-cp8r7 | cute_ws@compare | 667.1 | 67.7 | 0.45 | 5157 | 120.2 | 0.87 | 954 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 753.2 | 104.7 | 0.51 | - | - | - | 418 |
| short-49208-hca-cp1 | tilelang@main | 7876 | 628.9 | 1.00 | 25343 | 831.9 | 1.00 | 6382 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 4089 | 14.0 | 0.52 | - | - | - | 3197 |
| short-49208-hca-cp1 | tilelang@compare | 7931 | 598.4 | 1.00 | 25369 | 824.4 | 1.00 | 6382 |
| short-49208-hca-cp1 | cudnn_flashmla@compare | 4144 | 52.9 | 0.52 | 17212 | 195.6 | 0.68 | 6406 |
| short-49208-hca-cp1 | cute@compare | 7394 | -17.4 | 0.93 | 24853 | 190.9 | 0.98 | 6382 |
| short-49208-hca-cp1 | cute_ws@compare | 3519 | 52.3 | 0.44 | 20990 | 188.6 | 0.83 | 6382 |
| short-49208-hca-cp1 | flashmla_fwd_ref@compare | 4112 | 3.2 | 0.52 | - | - | - | 3197 |
| short-49208-hca-cp8r0 | tilelang@main | 989.7 | 710.9 | 1.00 | 3199 | 869.2 | 1.00 | 925 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 521.4 | 125.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r0 | tilelang@compare | 993.2 | 748.3 | 1.00 | 3196 | 905.2 | 1.00 | 925 |
| short-49208-hca-cp8r0 | cudnn_flashmla@compare | 531.4 | 197.1 | 0.54 | 2201 | 441.9 | 0.69 | 928 |
| short-49208-hca-cp8r0 | cute@compare | 924.4 | 147.5 | 0.93 | 3130 | 271.9 | 0.98 | 925 |
| short-49208-hca-cp8r0 | cute_ws@compare | 453.8 | 57.6 | 0.46 | 2674 | 241.6 | 0.84 | 925 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 522.1 | 143.9 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r4 | tilelang@main | 1004 | 742.6 | 1.00 | 3328 | 858.1 | 1.00 | 925 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 529.1 | 136.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r4 | tilelang@compare | 1008 | 735.8 | 1.00 | 3328 | 924.2 | 1.00 | 925 |
| short-49208-hca-cp8r4 | cudnn_flashmla@compare | 537.3 | 198.8 | 0.53 | 2270 | 451.3 | 0.68 | 928 |
| short-49208-hca-cp8r4 | cute@compare | 943.2 | 149.2 | 0.94 | 3269 | 271.2 | 0.98 | 925 |
| short-49208-hca-cp8r4 | cute_ws@compare | 460.1 | 57.8 | 0.46 | 2793 | 235.0 | 0.84 | 925 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 527.8 | 140.3 | 0.52 | - | - | - | 400 |
| short-49208-hca-cp8r7 | tilelang@main | 1012 | 720.8 | 1.00 | 3277 | 892.2 | 1.00 | 925 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 532.2 | 132.3 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r7 | tilelang@compare | 1013 | 740.2 | 1.00 | 3281 | 896.0 | 1.00 | 925 |
| short-49208-hca-cp8r7 | cudnn_flashmla@compare | 545.3 | 194.4 | 0.54 | 2240 | 416.8 | 0.68 | 928 |
| short-49208-hca-cp8r7 | cute@compare | 943.2 | 141.9 | 0.93 | 3210 | 265.6 | 0.98 | 925 |
| short-49208-hca-cp8r7 | cute_ws@compare | 463.8 | 54.3 | 0.46 | 2728 | 224.3 | 0.83 | 925 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 534.1 | 135.3 | 0.53 | - | - | - | 400 |
| short-49208-sliding-cp1 | tilelang@main | 6741 | 681.3 | 1.00 | 21763 | 869.5 | 1.00 | 6332 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 3084 | 39.0 | 0.46 | - | - | - | 3148 |
| short-49208-sliding-cp1 | tilelang@compare | 6951 | 541.2 | 1.00 | 21794 | 868.5 | 1.00 | 6332 |
| short-49208-sliding-cp1 | cudnn_flashmla@compare | 3131 | 117.7 | 0.45 | 14908 | 306.9 | 0.68 | 6380 |
| short-49208-sliding-cp1 | cute@compare | 6273 | 73.6 | 0.90 | 21282 | 236.8 | 0.98 | 6332 |
| short-49208-sliding-cp1 | cute_ws@compare | 2365 | 52.3 | 0.34 | 17384 | 186.0 | 0.80 | 6332 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3095 | 57.7 | 0.45 | - | - | - | 3148 |
| short-49208-sliding-cp8r0 | tilelang@main | 864.5 | 700.0 | 1.00 | 2838 | 825.1 | 1.00 | 918 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 397.8 | 143.2 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r0 | tilelang@compare | 861.3 | 703.9 | 1.00 | 2835 | 866.1 | 1.00 | 918 |
| short-49208-sliding-cp8r0 | cudnn_flashmla@compare | 416.0 | 195.2 | 0.48 | 1959 | 574.2 | 0.69 | 924 |
| short-49208-sliding-cp8r0 | cute@compare | 797.5 | 118.8 | 0.93 | 2761 | 237.0 | 0.97 | 918 |
| short-49208-sliding-cp8r0 | cute_ws@compare | 318.2 | 51.4 | 0.37 | 2283 | 365.7 | 0.81 | 918 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 403.7 | 139.7 | 0.47 | - | - | - | 394 |
| short-49208-sliding-cp8r4 | tilelang@main | 869.7 | 699.7 | 1.00 | 2893 | 846.4 | 1.00 | 918 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 402.9 | 142.9 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r4 | tilelang@compare | 873.5 | 696.9 | 1.00 | 2900 | 854.6 | 1.00 | 918 |
| short-49208-sliding-cp8r4 | cudnn_flashmla@compare | 418.2 | 196.8 | 0.48 | 1981 | 547.8 | 0.68 | 924 |
| short-49208-sliding-cp8r4 | cute@compare | 805.2 | 127.9 | 0.92 | 2833 | 229.2 | 0.98 | 918 |
| short-49208-sliding-cp8r4 | cute_ws@compare | 314.4 | 53.1 | 0.36 | 2344 | 340.7 | 0.81 | 918 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 404.6 | 143.0 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r7 | tilelang@main | 872.3 | 683.9 | 1.00 | 2886 | 857.6 | 1.00 | 918 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 405.0 | 131.9 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r7 | tilelang@compare | 876.3 | 708.8 | 1.00 | 2881 | 884.7 | 1.00 | 918 |
| short-49208-sliding-cp8r7 | cudnn_flashmla@compare | 419.3 | 203.0 | 0.48 | 1983 | 563.9 | 0.69 | 924 |
| short-49208-sliding-cp8r7 | cute@compare | 810.9 | 123.3 | 0.93 | 2817 | 231.2 | 0.98 | 918 |
| short-49208-sliding-cp8r7 | cute_ws@compare | 315.8 | 53.8 | 0.36 | 2327 | 332.0 | 0.81 | 918 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 402.9 | 146.3 | 0.46 | - | - | - | 394 |
| heavy-49208-csa-cp1 | tilelang@main | 14881 | 32.9 | 1.00 | 60982 | 725.7 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 7056 | 338.1 | 0.47 | - | - | - | 3340 |
| heavy-49208-csa-cp1 | tilelang@compare | 14905 | 36.4 | 1.00 | 60936 | 809.1 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | cudnn_flashmla@compare | 7072 | 329.2 | 0.47 | 34232 | 438.3 | 0.56 | 6513 |
| heavy-49208-csa-cp1 | cute@compare | 13741 | 102.5 | 0.92 | 60713 | 109.2 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | cute_ws@compare | 6542 | 332.6 | 0.44 | 53346 | 111.7 | 0.88 | 6368 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@compare | 7273 | 452.2 | 0.49 | - | - | - | 3340 |
| heavy-49208-csa-cp8r0 | tilelang@main | 1239 | 685.7 | 1.00 | 4603 | 816.8 | 1.00 | 954 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 650.2 | 109.6 | 0.52 | - | - | - | 418 |
| heavy-49208-csa-cp8r0 | tilelang@compare | 1241 | 696.8 | 1.00 | 4597 | 848.3 | 1.00 | 954 |
| heavy-49208-csa-cp8r0 | cudnn_flashmla@compare | 654.8 | 173.4 | 0.53 | 2832 | 308.8 | 0.62 | 972 |
| heavy-49208-csa-cp8r0 | cute@compare | 1167 | 112.0 | 0.94 | 4531 | 232.2 | 0.99 | 954 |
| heavy-49208-csa-cp8r0 | cute_ws@compare | 564.5 | 56.2 | 0.45 | 3927 | 123.7 | 0.85 | 954 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 653.0 | 101.9 | 0.53 | - | - | - | 418 |
| heavy-49208-csa-cp8r4 | tilelang@main | 1940 | 688.9 | 1.00 | 8493 | 875.9 | 1.00 | 954 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 953.3 | 104.7 | 0.49 | - | - | - | 418 |
| heavy-49208-csa-cp8r4 | tilelang@compare | 1948 | 703.6 | 1.00 | 8543 | 901.1 | 1.00 | 954 |
| heavy-49208-csa-cp8r4 | cudnn_flashmla@compare | 963.1 | 182.2 | 0.49 | 4607 | 317.5 | 0.54 | 972 |
| heavy-49208-csa-cp8r4 | cute@compare | 1850 | 112.6 | 0.95 | 8460 | 238.6 | 0.99 | 954 |
| heavy-49208-csa-cp8r4 | cute_ws@compare | 884.8 | 53.2 | 0.45 | 7431 | 182.3 | 0.87 | 954 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 955.9 | 112.1 | 0.49 | - | - | - | 418 |
| heavy-49208-csa-cp8r7 | tilelang@main | 1792 | 685.2 | 1.00 | 7721 | 860.7 | 1.00 | 954 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 889.3 | 101.1 | 0.50 | - | - | - | 418 |
| heavy-49208-csa-cp8r7 | tilelang@compare | 1800 | 688.8 | 1.00 | 7710 | 940.0 | 1.00 | 954 |
| heavy-49208-csa-cp8r7 | cudnn_flashmla@compare | 898.6 | 167.2 | 0.50 | 4220 | 322.0 | 0.55 | 972 |
| heavy-49208-csa-cp8r7 | cute@compare | 1713 | 117.2 | 0.95 | 7629 | 310.2 | 0.99 | 954 |
| heavy-49208-csa-cp8r7 | cute_ws@compare | 822.0 | 53.7 | 0.46 | 6746 | 176.2 | 0.88 | 954 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 891.0 | 98.8 | 0.49 | - | - | - | 418 |
| heavy-49208-hca-cp1 | tilelang@main | 8488 | 600.9 | 1.00 | 28842 | 762.6 | 1.00 | 6394 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4251 | 22.7 | 0.50 | - | - | - | 3244 |
| heavy-49208-hca-cp1 | tilelang@compare | 8525 | 579.4 | 1.00 | 28874 | 804.3 | 1.00 | 6394 |
| heavy-49208-hca-cp1 | cudnn_flashmla@compare | 4371 | 47.8 | 0.51 | 18821 | 190.5 | 0.65 | 6454 |
| heavy-49208-hca-cp1 | cute@compare | 7958 | 23.3 | 0.93 | 28337 | 164.3 | 0.98 | 6394 |
| heavy-49208-hca-cp1 | cute_ws@compare | 3625 | 69.7 | 0.43 | 24034 | 177.1 | 0.83 | 6394 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@compare | 4274 | 95.1 | 0.50 | - | - | - | 3244 |
| heavy-49208-hca-cp8r0 | tilelang@main | 956.8 | 745.4 | 1.00 | 3055 | 857.0 | 1.00 | 926 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 510.4 | 132.4 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r0 | tilelang@compare | 959.3 | 740.9 | 1.00 | 3039 | 918.7 | 1.00 | 926 |
| heavy-49208-hca-cp8r0 | cudnn_flashmla@compare | 530.0 | 183.4 | 0.55 | 2150 | 438.6 | 0.71 | 934 |
| heavy-49208-hca-cp8r0 | cute@compare | 892.0 | 142.8 | 0.93 | 2970 | 269.6 | 0.98 | 926 |
| heavy-49208-hca-cp8r0 | cute_ws@compare | 429.5 | 57.1 | 0.45 | 2514 | 274.4 | 0.83 | 926 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 505.7 | 139.4 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r4 | tilelang@main | 1044 | 709.2 | 1.00 | 3546 | 847.7 | 1.00 | 926 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 550.7 | 124.3 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r4 | tilelang@compare | 1042 | 727.1 | 1.00 | 3523 | 905.3 | 1.00 | 926 |
| heavy-49208-hca-cp8r4 | cudnn_flashmla@compare | 571.1 | 168.3 | 0.55 | 2368 | 398.8 | 0.67 | 934 |
| heavy-49208-hca-cp8r4 | cute@compare | 965.3 | 139.9 | 0.93 | 3450 | 265.4 | 0.98 | 926 |
| heavy-49208-hca-cp8r4 | cute_ws@compare | 470.4 | 57.4 | 0.45 | 2951 | 230.5 | 0.84 | 926 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 550.9 | 119.1 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r7 | tilelang@main | 1105 | 739.6 | 1.00 | 3792 | 880.6 | 1.00 | 926 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 591.4 | 120.7 | 0.54 | - | - | - | 406 |
| heavy-49208-hca-cp8r7 | tilelang@compare | 1099 | 729.6 | 1.00 | 3776 | 874.9 | 1.00 | 926 |
| heavy-49208-hca-cp8r7 | cudnn_flashmla@compare | 605.4 | 174.6 | 0.55 | 2510 | 335.4 | 0.66 | 934 |
| heavy-49208-hca-cp8r7 | cute@compare | 1032 | 143.1 | 0.94 | 3696 | 263.5 | 0.98 | 926 |
| heavy-49208-hca-cp8r7 | cute_ws@compare | 511.9 | 58.3 | 0.47 | 3196 | 152.6 | 0.85 | 926 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 582.1 | 126.2 | 0.53 | - | - | - | 406 |
| heavy-49208-sliding-cp1 | tilelang@main | 6750 | 697.8 | 1.00 | 21733 | 849.9 | 1.00 | 6332 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 3069 | 65.7 | 0.45 | - | - | - | 3148 |
| heavy-49208-sliding-cp1 | tilelang@compare | 6775 | 680.8 | 1.00 | 21744 | 878.0 | 1.00 | 6332 |
| heavy-49208-sliding-cp1 | cudnn_flashmla@compare | 3132 | 121.3 | 0.46 | 14875 | 356.1 | 0.68 | 6380 |
| heavy-49208-sliding-cp1 | cute@compare | 6260 | 90.8 | 0.92 | 21250 | 238.4 | 0.98 | 6332 |
| heavy-49208-sliding-cp1 | cute_ws@compare | 2370 | 61.9 | 0.35 | 17364 | 181.4 | 0.80 | 6332 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3086 | 49.5 | 0.46 | - | - | - | 3148 |
| heavy-49208-sliding-cp8r0 | tilelang@main | 861.4 | 710.5 | 1.00 | 2755 | 833.2 | 1.00 | 918 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 402.5 | 139.7 | 0.47 | - | - | - | 394 |
| heavy-49208-sliding-cp8r0 | tilelang@compare | 860.0 | 706.7 | 1.00 | 2745 | 875.0 | 1.00 | 918 |
| heavy-49208-sliding-cp8r0 | cudnn_flashmla@compare | 413.6 | 199.4 | 0.48 | 1924 | 575.6 | 0.70 | 925 |
| heavy-49208-sliding-cp8r0 | cute@compare | 795.2 | 125.2 | 0.92 | 2680 | 231.1 | 0.98 | 918 |
| heavy-49208-sliding-cp8r0 | cute_ws@compare | 319.5 | 58.2 | 0.37 | 2215 | 350.1 | 0.81 | 918 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 403.4 | 145.1 | 0.47 | - | - | - | 393 |
| heavy-49208-sliding-cp8r4 | tilelang@main | 877.6 | 687.8 | 1.00 | 2942 | 819.4 | 1.00 | 918 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 403.9 | 135.0 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r4 | tilelang@compare | 878.1 | 724.8 | 1.00 | 2952 | 853.5 | 1.00 | 918 |
| heavy-49208-sliding-cp8r4 | cudnn_flashmla@compare | 418.7 | 195.0 | 0.48 | 2003 | 550.4 | 0.68 | 925 |
| heavy-49208-sliding-cp8r4 | cute@compare | 811.0 | 127.0 | 0.92 | 2877 | 230.0 | 0.97 | 918 |
| heavy-49208-sliding-cp8r4 | cute_ws@compare | 312.1 | 56.6 | 0.36 | 2393 | 336.6 | 0.81 | 918 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 402.5 | 143.6 | 0.46 | - | - | - | 393 |
| heavy-49208-sliding-cp8r7 | tilelang@main | 882.7 | 684.6 | 1.00 | 2924 | 841.8 | 1.00 | 918 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 402.0 | 141.2 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r7 | tilelang@compare | 875.7 | 709.6 | 1.00 | 2916 | 883.3 | 1.00 | 918 |
| heavy-49208-sliding-cp8r7 | cudnn_flashmla@compare | 414.1 | 203.8 | 0.47 | 2007 | 551.9 | 0.69 | 925 |
| heavy-49208-sliding-cp8r7 | cute@compare | 809.8 | 123.6 | 0.92 | 2847 | 236.3 | 0.98 | 918 |
| heavy-49208-sliding-cp8r7 | cute_ws@compare | 312.0 | 57.1 | 0.36 | 2363 | 343.7 | 0.81 | 918 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 403.7 | 148.2 | 0.46 | - | - | - | 393 |
| tiny-49208-csa-cp1 | tilelang@main | 7840 | 553.1 | 1.00 | 23410 | 693.2 | 1.00 | 6368 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 4272 | 23.0 | 0.54 | - | - | - | 3340 |
| tiny-49208-csa-cp1 | tilelang@compare | 7831 | 544.4 | 1.00 | 23437 | 722.5 | 1.00 | 6368 |
| tiny-49208-csa-cp1 | cudnn_flashmla@compare | 4288 | 48.4 | 0.55 | 16103 | 167.2 | 0.69 | 6512 |
| tiny-49208-csa-cp1 | cute@compare | 7333 | 29.6 | 0.94 | 22934 | 121.1 | 0.98 | 6368 |
| tiny-49208-csa-cp1 | cute_ws@compare | 3649 | 67.9 | 0.47 | 19243 | 185.8 | 0.82 | 6368 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@compare | 4283 | -9.1 | 0.55 | - | - | - | 3340 |
| tiny-49208-csa-cp8r0 | tilelang@main | 998.7 | 705.4 | 1.00 | 3072 | 802.9 | 1.00 | 953 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 556.1 | 102.0 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r0 | tilelang@compare | 999.5 | 694.5 | 1.00 | 3069 | 827.1 | 1.00 | 953 |
| tiny-49208-csa-cp8r0 | cudnn_flashmla@compare | 557.6 | 177.8 | 0.56 | 2131 | 345.3 | 0.69 | 971 |
| tiny-49208-csa-cp8r0 | cute@compare | 935.8 | 109.0 | 0.94 | 3008 | 208.4 | 0.98 | 953 |
| tiny-49208-csa-cp8r0 | cute_ws@compare | 474.2 | 57.2 | 0.47 | 2543 | 146.2 | 0.83 | 953 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 556.2 | 99.9 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r4 | tilelang@main | 1012 | 668.1 | 1.00 | 3097 | 809.2 | 1.00 | 953 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 554.1 | 102.4 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r4 | tilelang@compare | 1004 | 685.3 | 1.00 | 3080 | 843.9 | 1.00 | 953 |
| tiny-49208-csa-cp8r4 | cudnn_flashmla@compare | 557.4 | 172.6 | 0.56 | 2131 | 370.0 | 0.69 | 971 |
| tiny-49208-csa-cp8r4 | cute@compare | 939.4 | 105.9 | 0.94 | 3016 | 231.6 | 0.98 | 953 |
| tiny-49208-csa-cp8r4 | cute_ws@compare | 474.5 | 59.9 | 0.47 | 2557 | 155.0 | 0.83 | 953 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 550.9 | 108.4 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r7 | tilelang@main | 1007 | 682.4 | 1.00 | 3080 | 832.5 | 1.00 | 953 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 560.0 | 99.4 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r7 | tilelang@compare | 997.2 | 690.4 | 1.00 | 3062 | 863.1 | 1.00 | 953 |
| tiny-49208-csa-cp8r7 | cudnn_flashmla@compare | 561.4 | 165.1 | 0.56 | 2131 | 377.1 | 0.70 | 971 |
| tiny-49208-csa-cp8r7 | cute@compare | 934.7 | 111.8 | 0.94 | 3004 | 220.3 | 0.98 | 953 |
| tiny-49208-csa-cp8r7 | cute_ws@compare | 475.1 | 56.0 | 0.48 | 2546 | 165.1 | 0.83 | 953 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 554.9 | 101.5 | 0.56 | - | - | - | 418 |
| tiny-49208-hca-cp1 | tilelang@main | 6058 | 685.3 | 1.00 | 15953 | 833.4 | 1.00 | 6332 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 3126 | 38.3 | 0.52 | - | - | - | 3148 |
| tiny-49208-hca-cp1 | tilelang@compare | 6044 | 662.2 | 1.00 | 15886 | 876.8 | 1.00 | 6332 |
| tiny-49208-hca-cp1 | cudnn_flashmla@compare | 3152 | 133.6 | 0.52 | 12472 | 336.2 | 0.79 | 6380 |
| tiny-49208-hca-cp1 | cute@compare | 5536 | 85.0 | 0.92 | 15391 | 275.4 | 0.97 | 6332 |
| tiny-49208-hca-cp1 | cute_ws@compare | 2706 | 35.5 | 0.45 | 12584 | 153.7 | 0.79 | 6332 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@compare | 3115 | 55.3 | 0.52 | - | - | - | 3148 |
| tiny-49208-hca-cp8r0 | tilelang@main | 774.9 | 704.2 | 1.00 | 2125 | 827.8 | 1.00 | 918 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 401.6 | 139.8 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r0 | tilelang@compare | 774.1 | 777.5 | 1.00 | 2115 | 851.9 | 1.00 | 918 |
| tiny-49208-hca-cp8r0 | cudnn_flashmla@compare | 413.6 | 223.8 | 0.53 | 1668 | 540.3 | 0.79 | 925 |
| tiny-49208-hca-cp8r0 | cute@compare | 716.0 | 127.1 | 0.92 | 2055 | 228.9 | 0.97 | 918 |
| tiny-49208-hca-cp8r0 | cute_ws@compare | 339.0 | 67.4 | 0.44 | 1701 | 285.4 | 0.80 | 918 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 400.6 | 160.9 | 0.52 | - | - | - | 393 |
| tiny-49208-hca-cp8r4 | tilelang@main | 783.0 | 701.8 | 1.00 | 2153 | 827.3 | 1.00 | 918 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 404.4 | 139.6 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r4 | tilelang@compare | 783.7 | 711.1 | 1.00 | 2145 | 865.6 | 1.00 | 918 |
| tiny-49208-hca-cp8r4 | cudnn_flashmla@compare | 413.2 | 205.7 | 0.53 | 1667 | 541.3 | 0.78 | 925 |
| tiny-49208-hca-cp8r4 | cute@compare | 719.9 | 122.6 | 0.92 | 2085 | 220.6 | 0.97 | 918 |
| tiny-49208-hca-cp8r4 | cute_ws@compare | 352.6 | 48.6 | 0.45 | 1712 | 302.2 | 0.80 | 918 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 405.8 | 138.8 | 0.52 | - | - | - | 393 |
| tiny-49208-hca-cp8r7 | tilelang@main | 780.8 | 692.8 | 1.00 | 2111 | 841.6 | 1.00 | 918 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 401.6 | 139.0 | 0.51 | - | - | - | 394 |
| tiny-49208-hca-cp8r7 | tilelang@compare | 775.8 | 687.8 | 1.00 | 2098 | 862.2 | 1.00 | 918 |
| tiny-49208-hca-cp8r7 | cudnn_flashmla@compare | 415.9 | 191.1 | 0.54 | 1651 | 533.4 | 0.79 | 925 |
| tiny-49208-hca-cp8r7 | cute@compare | 713.8 | 121.6 | 0.92 | 2037 | 215.8 | 0.97 | 918 |
| tiny-49208-hca-cp8r7 | cute_ws@compare | 345.6 | 56.9 | 0.45 | 1678 | 291.3 | 0.80 | 918 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 401.2 | 139.8 | 0.52 | - | - | - | 393 |
| tiny-49208-sliding-cp1 | tilelang@main | 6059 | 674.1 | 1.00 | 15952 | 849.2 | 1.00 | 6332 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 3124 | 48.6 | 0.52 | - | - | - | 3148 |
| tiny-49208-sliding-cp1 | tilelang@compare | 6040 | 679.9 | 1.00 | 15903 | 907.6 | 1.00 | 6332 |
| tiny-49208-sliding-cp1 | cudnn_flashmla@compare | 3148 | 134.9 | 0.52 | 12462 | 325.4 | 0.78 | 6380 |
| tiny-49208-sliding-cp1 | cute@compare | 5538 | 87.7 | 0.92 | 15398 | 269.2 | 0.97 | 6332 |
| tiny-49208-sliding-cp1 | cute_ws@compare | 2706 | 38.9 | 0.45 | 12561 | 170.2 | 0.79 | 6332 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@compare | 3110 | 59.8 | 0.51 | - | - | - | 3148 |
| tiny-49208-sliding-cp8r0 | tilelang@main | 773.6 | 689.3 | 1.00 | 2123 | 823.5 | 1.00 | 918 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 403.5 | 136.8 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r0 | tilelang@compare | 769.8 | 704.0 | 1.00 | 2115 | 911.4 | 1.00 | 918 |
| tiny-49208-sliding-cp8r0 | cudnn_flashmla@compare | 413.2 | 196.3 | 0.54 | 1664 | 564.4 | 0.79 | 925 |
| tiny-49208-sliding-cp8r0 | cute@compare | 710.5 | 123.9 | 0.92 | 2055 | 231.7 | 0.97 | 918 |
| tiny-49208-sliding-cp8r0 | cute_ws@compare | 348.7 | 55.0 | 0.45 | 1695 | 330.3 | 0.80 | 918 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 401.4 | 143.7 | 0.52 | - | - | - | 393 |
| tiny-49208-sliding-cp8r4 | tilelang@main | 785.5 | 699.1 | 1.00 | 2156 | 819.1 | 1.00 | 918 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 402.9 | 139.6 | 0.51 | - | - | - | 394 |
| tiny-49208-sliding-cp8r4 | tilelang@compare | 780.4 | 749.6 | 1.00 | 2147 | 866.0 | 1.00 | 918 |
| tiny-49208-sliding-cp8r4 | cudnn_flashmla@compare | 414.1 | 203.4 | 0.53 | 1678 | 551.9 | 0.78 | 925 |
| tiny-49208-sliding-cp8r4 | cute@compare | 720.5 | 131.5 | 0.92 | 2087 | 227.5 | 0.97 | 918 |
| tiny-49208-sliding-cp8r4 | cute_ws@compare | 343.2 | 60.3 | 0.44 | 1720 | 295.6 | 0.80 | 918 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 403.0 | 142.7 | 0.52 | - | - | - | 393 |
| tiny-49208-sliding-cp8r7 | tilelang@main | 779.6 | 685.1 | 1.00 | 2107 | 843.6 | 1.00 | 918 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 406.0 | 136.3 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r7 | tilelang@compare | 777.3 | 721.2 | 1.00 | 2100 | 866.8 | 1.00 | 918 |
| tiny-49208-sliding-cp8r7 | cudnn_flashmla@compare | 415.2 | 194.5 | 0.53 | 1652 | 526.7 | 0.79 | 925 |
| tiny-49208-sliding-cp8r7 | cute@compare | 712.5 | 124.7 | 0.92 | 2032 | 228.9 | 0.97 | 918 |
| tiny-49208-sliding-cp8r7 | cute_ws@compare | 342.5 | 52.1 | 0.44 | 1673 | 297.5 | 0.80 | 918 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 404.0 | 147.9 | 0.52 | - | - | - | 393 |
| single-65536-csa-cp1 | tilelang@main | 21804 | -109.6 | 1.00 | 94799 | 644.7 | 1.00 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 10352 | 1033 | 0.47 | - | - | - | 4448 |
| single-65536-csa-cp1 | tilelang@compare | 21694 | 21.1 | 1.00 | 94645 | 863.3 | 1.00 | 8480 |
| single-65536-csa-cp1 | cudnn_flashmla@compare | 10392 | 1072 | 0.48 | 53021 | 317.0 | 0.56 | 8672 |
| single-65536-csa-cp1 | cute@compare | 20298 | -31.2 | 0.94 | 94448 | -440.1 | 1.00 | 8480 |
| single-65536-csa-cp1 | cute_ws@compare | 9774 | 1172 | 0.45 | 84187 | -187.7 | 0.89 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref@compare | 11427 | 47.3 | 0.53 | - | - | - | 4448 |
| single-65536-csa-cp8r0 | tilelang@main | 2543 | 722.8 | 1.00 | 11048 | 882.7 | 1.00 | 1270 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 1248 | 76.7 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r0 | tilelang@compare | 2641 | 593.7 | 1.00 | 11114 | 961.2 | 1.00 | 1270 |
| single-65536-csa-cp8r0 | cudnn_flashmla@compare | 1249 | 150.9 | 0.47 | 5955 | 339.4 | 0.54 | 1294 |
| single-65536-csa-cp8r0 | cute@compare | 2420 | 123.3 | 0.92 | 10905 | 415.9 | 0.98 | 1270 |
| single-65536-csa-cp8r0 | cute_ws@compare | 1157 | 57.2 | 0.44 | 9654 | 270.5 | 0.87 | 1270 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 1245 | 86.7 | 0.47 | - | - | - | 556 |
| single-65536-csa-cp8r4 | tilelang@main | 2711 | 717.6 | 1.00 | 12259 | 886.8 | 1.00 | 1270 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1324 | 86.3 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r4 | tilelang@compare | 2786 | 616.9 | 1.00 | 12340 | 906.4 | 1.00 | 1270 |
| single-65536-csa-cp8r4 | cudnn_flashmla@compare | 1334 | 154.8 | 0.48 | 6486 | 353.0 | 0.53 | 1294 |
| single-65536-csa-cp8r4 | cute@compare | 2607 | 95.3 | 0.94 | 12159 | 345.6 | 0.99 | 1270 |
| single-65536-csa-cp8r4 | cute_ws@compare | 1230 | 63.2 | 0.44 | 10773 | 276.8 | 0.87 | 1270 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 1324 | 95.9 | 0.48 | - | - | - | 556 |
| single-65536-csa-cp8r7 | tilelang@main | 2709 | 675.0 | 1.00 | 13123 | 872.4 | 1.00 | 1270 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1331 | 78.8 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r7 | tilelang@compare | 2759 | 672.2 | 1.00 | 13224 | 943.2 | 1.00 | 1270 |
| single-65536-csa-cp8r7 | cudnn_flashmla@compare | 1349 | 158.2 | 0.49 | 6824 | 387.3 | 0.52 | 1294 |
| single-65536-csa-cp8r7 | cute@compare | 2588 | 108.2 | 0.94 | 12792 | 610.1 | 0.97 | 1270 |
| single-65536-csa-cp8r7 | cute_ws@compare | 1244 | 61.8 | 0.45 | 11601 | 312.8 | 0.88 | 1270 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 1337 | 94.9 | 0.48 | - | - | - | 556 |
| single-65536-hca-cp1 | tilelang@main | 16321 | 181.6 | 1.00 | 63704 | 622.9 | 1.00 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 7958 | 238.2 | 0.49 | - | - | - | 4448 |
| single-65536-hca-cp1 | tilelang@compare | 16227 | 320.9 | 1.00 | 63723 | 769.9 | 1.00 | 8434 |
| single-65536-hca-cp1 | cudnn_flashmla@compare | 7985 | 311.4 | 0.49 | 37441 | -19.0 | 0.59 | 8626 |
| single-65536-hca-cp1 | cute@compare | 15387 | -82.9 | 0.95 | 63294 | -124.7 | 0.99 | 8434 |
| single-65536-hca-cp1 | cute_ws@compare | 7114 | 537.8 | 0.44 | 54896 | 163.4 | 0.86 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref@compare | 8372 | 56.7 | 0.52 | - | - | - | 4448 |
| single-65536-hca-cp8r0 | tilelang@main | 1365 | 708.7 | 1.00 | 4606 | 828.1 | 1.00 | 1224 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 751.0 | 84.2 | 0.55 | - | - | - | 556 |
| single-65536-hca-cp8r0 | tilelang@compare | 1358 | 710.3 | 1.00 | 4610 | 840.7 | 1.00 | 1224 |
| single-65536-hca-cp8r0 | cudnn_flashmla@compare | 759.8 | 154.8 | 0.56 | 3136 | 296.4 | 0.68 | 1248 |
| single-65536-hca-cp8r0 | cute@compare | 1268 | 109.5 | 0.93 | 4518 | 217.5 | 0.98 | 1224 |
| single-65536-hca-cp8r0 | cute_ws@compare | 620.0 | 54.0 | 0.46 | 3872 | 112.8 | 0.84 | 1224 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 753.9 | 81.7 | 0.56 | - | - | - | 556 |
| single-65536-hca-cp8r4 | tilelang@main | 2123 | 673.9 | 1.00 | 8699 | 883.4 | 1.00 | 1224 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1114 | 70.8 | 0.52 | - | - | - | 556 |
| single-65536-hca-cp8r4 | tilelang@compare | 2133 | 684.7 | 1.00 | 8693 | 944.4 | 1.00 | 1224 |
| single-65536-hca-cp8r4 | cudnn_flashmla@compare | 1123 | 151.0 | 0.53 | 4947 | 330.3 | 0.57 | 1248 |
| single-65536-hca-cp8r4 | cute@compare | 2024 | 103.3 | 0.95 | 8597 | 252.1 | 0.99 | 1224 |
| single-65536-hca-cp8r4 | cute_ws@compare | 1011 | 55.0 | 0.47 | 7591 | 129.5 | 0.87 | 1224 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 1114 | 87.5 | 0.52 | - | - | - | 556 |
| single-65536-hca-cp8r7 | tilelang@main | 2715 | 674.6 | 1.00 | 11760 | 929.3 | 1.00 | 1224 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 1305 | 79.3 | 0.48 | - | - | - | 556 |
| single-65536-hca-cp8r7 | tilelang@compare | 2751 | 660.3 | 1.00 | 11843 | 947.0 | 1.00 | 1224 |
| single-65536-hca-cp8r7 | cudnn_flashmla@compare | 1321 | 149.8 | 0.48 | 6252 | 316.5 | 0.53 | 1248 |
| single-65536-hca-cp8r7 | cute@compare | 2592 | 110.3 | 0.94 | 11711 | 302.0 | 0.99 | 1224 |
| single-65536-hca-cp8r7 | cute_ws@compare | 1208 | 63.0 | 0.44 | 10350 | 192.0 | 0.87 | 1224 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 1305 | 88.2 | 0.47 | - | - | - | 556 |
| single-65536-sliding-cp1 | tilelang@main | 8998 | 652.7 | 1.00 | 29308 | 838.2 | 1.00 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4089 | 32.9 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp1 | tilelang@compare | 9085 | 624.9 | 1.00 | 29328 | 825.5 | 1.00 | 8432 |
| single-65536-sliding-cp1 | cudnn_flashmla@compare | 4147 | 85.5 | 0.46 | 19994 | 294.0 | 0.68 | 8496 |
| single-65536-sliding-cp1 | cute@compare | 8353 | 69.0 | 0.92 | 28654 | 233.2 | 0.98 | 8432 |
| single-65536-sliding-cp1 | cute_ws@compare | 3155 | 67.6 | 0.35 | 23495 | 145.4 | 0.80 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4079 | 159.4 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp8r0 | tilelang@main | 1148 | 723.0 | 1.00 | 3834 | 823.8 | 1.00 | 1222 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 522.4 | 139.0 | 0.45 | - | - | - | 524 |
| single-65536-sliding-cp8r0 | tilelang@compare | 1154 | 716.3 | 1.00 | 3822 | 895.8 | 1.00 | 1222 |
| single-65536-sliding-cp8r0 | cudnn_flashmla@compare | 543.6 | 187.9 | 0.47 | 2627 | 455.0 | 0.69 | 1230 |
| single-65536-sliding-cp8r0 | cute@compare | 1063 | 122.6 | 0.92 | 3750 | 239.5 | 0.98 | 1222 |
| single-65536-sliding-cp8r0 | cute_ws@compare | 409.7 | 52.6 | 0.35 | 3089 | 154.4 | 0.81 | 1222 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 529.0 | 135.3 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r4 | tilelang@main | 1153 | 690.8 | 1.00 | 3867 | 826.9 | 1.00 | 1222 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 533.2 | 127.5 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r4 | tilelang@compare | 1158 | 702.9 | 1.00 | 3864 | 884.4 | 1.00 | 1222 |
| single-65536-sliding-cp8r4 | cudnn_flashmla@compare | 535.9 | 198.8 | 0.46 | 2622 | 445.3 | 0.68 | 1230 |
| single-65536-sliding-cp8r4 | cute@compare | 1070 | 120.9 | 0.92 | 3785 | 242.4 | 0.98 | 1222 |
| single-65536-sliding-cp8r4 | cute_ws@compare | 409.2 | 53.1 | 0.35 | 3135 | 158.5 | 0.81 | 1222 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 525.6 | 134.7 | 0.45 | - | - | - | 524 |
| single-65536-sliding-cp8r7 | tilelang@main | 1154 | 698.5 | 1.00 | 3861 | 855.4 | 1.00 | 1222 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 529.4 | 133.0 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r7 | tilelang@compare | 1155 | 705.7 | 1.00 | 3884 | 859.9 | 1.00 | 1222 |
| single-65536-sliding-cp8r7 | cudnn_flashmla@compare | 541.8 | 190.7 | 0.47 | 2635 | 416.8 | 0.68 | 1230 |
| single-65536-sliding-cp8r7 | cute@compare | 1066 | 125.9 | 0.92 | 3782 | 235.3 | 0.97 | 1222 |
| single-65536-sliding-cp8r7 | cute_ws@compare | 408.4 | 56.2 | 0.35 | 3141 | 133.6 | 0.81 | 1222 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 532.4 | 128.7 | 0.46 | - | - | - | 524 |
| short-65536-csa-cp1 | tilelang@main | 16000 | 242.3 | 1.00 | 62509 | 660.3 | 1.00 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 7830 | 315.8 | 0.49 | - | - | - | 4448 |
| short-65536-csa-cp1 | tilelang@compare | 16330 | 16.1 | 1.00 | 62490 | 725.9 | 1.00 | 8480 |
| short-65536-csa-cp1 | cudnn_flashmla@compare | 7848 | 294.6 | 0.48 | 36543 | 199.1 | 0.58 | 8672 |
| short-65536-csa-cp1 | cute@compare | 14915 | 81.4 | 0.91 | 62096 | -21.6 | 0.99 | 8480 |
| short-65536-csa-cp1 | cute_ws@compare | 7096 | 567.1 | 0.43 | 53902 | 188.2 | 0.86 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref@compare | 8069 | 255.4 | 0.49 | - | - | - | 4448 |
| short-65536-csa-cp8r0 | tilelang@main | 1610 | 676.7 | 1.00 | 5968 | 811.5 | 1.00 | 1270 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 825.1 | 82.7 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r0 | tilelang@compare | 1614 | 683.4 | 1.00 | 5957 | 870.0 | 1.00 | 1270 |
| short-65536-csa-cp8r0 | cudnn_flashmla@compare | 836.5 | 151.6 | 0.52 | 3674 | 296.3 | 0.62 | 1294 |
| short-65536-csa-cp8r0 | cute@compare | 1518 | 104.1 | 0.94 | 5870 | 223.4 | 0.99 | 1270 |
| short-65536-csa-cp8r0 | cute_ws@compare | 717.5 | 54.7 | 0.44 | 5068 | 129.7 | 0.85 | 1270 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 834.8 | 77.1 | 0.52 | - | - | - | 556 |
| short-65536-csa-cp8r4 | tilelang@main | 2195 | 666.7 | 1.00 | 9179 | 896.3 | 1.00 | 1270 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1090 | 79.7 | 0.50 | - | - | - | 556 |
| short-65536-csa-cp8r4 | tilelang@compare | 2210 | 662.4 | 1.00 | 9188 | 983.3 | 1.00 | 1270 |
| short-65536-csa-cp8r4 | cudnn_flashmla@compare | 1098 | 154.7 | 0.50 | 5127 | 302.8 | 0.56 | 1294 |
| short-65536-csa-cp8r4 | cute@compare | 2105 | 100.7 | 0.95 | 9061 | 285.7 | 0.99 | 1270 |
| short-65536-csa-cp8r4 | cute_ws@compare | 993.8 | 57.0 | 0.45 | 7976 | 176.1 | 0.87 | 1270 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 1097 | 83.1 | 0.50 | - | - | - | 556 |
| short-65536-csa-cp8r7 | tilelang@main | 1997 | 667.0 | 1.00 | 8087 | 855.0 | 1.00 | 1270 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1012 | 76.5 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r7 | tilelang@compare | 1995 | 711.1 | 1.00 | 8069 | 888.1 | 1.00 | 1270 |
| short-65536-csa-cp8r7 | cudnn_flashmla@compare | 1017 | 155.1 | 0.51 | 4635 | 287.8 | 0.57 | 1294 |
| short-65536-csa-cp8r7 | cute@compare | 1889 | 118.5 | 0.95 | 7962 | 231.0 | 0.99 | 1270 |
| short-65536-csa-cp8r7 | cute_ws@compare | 893.4 | 68.7 | 0.45 | 6978 | 123.7 | 0.86 | 1270 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 1012 | 86.7 | 0.51 | - | - | - | 556 |
| short-65536-hca-cp1 | tilelang@main | 10309 | 612.7 | 1.00 | 32762 | 770.6 | 1.00 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5376 | 1.8 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp1 | tilelang@compare | 10386 | 569.4 | 1.00 | 32786 | 807.3 | 1.00 | 8482 |
| short-65536-hca-cp1 | cudnn_flashmla@compare | 5446 | 39.4 | 0.52 | 22538 | 143.5 | 0.69 | 8530 |
| short-65536-hca-cp1 | cute@compare | 9689 | -12.0 | 0.93 | 32050 | 211.3 | 0.98 | 8482 |
| short-65536-hca-cp1 | cute_ws@compare | 4719 | 19.9 | 0.45 | 27106 | 187.0 | 0.83 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref@compare | 5377 | 24.7 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp8r0 | tilelang@main | 1296 | 723.7 | 1.00 | 4168 | 858.9 | 1.00 | 1229 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 687.6 | 123.7 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r0 | tilelang@compare | 1292 | 738.4 | 1.00 | 4162 | 888.4 | 1.00 | 1229 |
| short-65536-hca-cp8r0 | cudnn_flashmla@compare | 694.4 | 177.2 | 0.54 | 2876 | 309.2 | 0.69 | 1236 |
| short-65536-hca-cp8r0 | cute@compare | 1207 | 138.3 | 0.93 | 4079 | 266.3 | 0.98 | 1229 |
| short-65536-hca-cp8r0 | cute_ws@compare | 588.9 | 53.4 | 0.46 | 3456 | 119.2 | 0.83 | 1229 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 681.0 | 122.7 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r4 | tilelang@main | 1337 | 717.1 | 1.00 | 4340 | 857.5 | 1.00 | 1229 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 704.0 | 128.4 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r4 | tilelang@compare | 1339 | 741.1 | 1.00 | 4342 | 926.0 | 1.00 | 1229 |
| short-65536-hca-cp8r4 | cudnn_flashmla@compare | 714.9 | 186.5 | 0.53 | 2988 | 334.8 | 0.69 | 1236 |
| short-65536-hca-cp8r4 | cute@compare | 1245 | 139.8 | 0.93 | 4250 | 272.4 | 0.98 | 1229 |
| short-65536-hca-cp8r4 | cute_ws@compare | 606.3 | 59.2 | 0.45 | 3623 | 110.3 | 0.83 | 1229 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 696.6 | 141.7 | 0.52 | - | - | - | 532 |
| short-65536-hca-cp8r7 | tilelang@main | 1320 | 720.0 | 1.00 | 4272 | 894.3 | 1.00 | 1229 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 702.8 | 117.0 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r7 | tilelang@compare | 1316 | 734.6 | 1.00 | 4272 | 895.7 | 1.00 | 1229 |
| short-65536-hca-cp8r7 | cudnn_flashmla@compare | 710.3 | 176.1 | 0.54 | 2949 | 312.3 | 0.69 | 1236 |
| short-65536-hca-cp8r7 | cute@compare | 1231 | 138.7 | 0.94 | 4181 | 270.4 | 0.98 | 1229 |
| short-65536-hca-cp8r7 | cute_ws@compare | 598.9 | 57.3 | 0.45 | 3561 | 117.0 | 0.83 | 1229 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 697.7 | 123.1 | 0.53 | - | - | - | 532 |
| short-65536-sliding-cp1 | tilelang@main | 8974 | 637.9 | 1.00 | 28844 | 797.0 | 1.00 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4081 | 23.9 | 0.45 | - | - | - | 4192 |
| short-65536-sliding-cp1 | tilelang@compare | 9064 | 594.8 | 1.00 | 28831 | 845.0 | 1.00 | 8432 |
| short-65536-sliding-cp1 | cudnn_flashmla@compare | 4154 | 68.5 | 0.46 | 19786 | 220.4 | 0.69 | 8496 |
| short-65536-sliding-cp1 | cute@compare | 8340 | 28.6 | 0.92 | 28172 | 218.5 | 0.98 | 8432 |
| short-65536-sliding-cp1 | cute_ws@compare | 3156 | 74.3 | 0.35 | 23007 | 186.4 | 0.80 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4100 | 22.6 | 0.45 | - | - | - | 4192 |
| short-65536-sliding-cp8r0 | tilelang@main | 1138 | 725.1 | 1.00 | 3714 | 842.5 | 1.00 | 1222 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 524.0 | 141.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r0 | tilelang@compare | 1135 | 717.1 | 1.00 | 3711 | 871.8 | 1.00 | 1222 |
| short-65536-sliding-cp8r0 | cudnn_flashmla@compare | 540.5 | 194.8 | 0.48 | 2570 | 420.0 | 0.69 | 1230 |
| short-65536-sliding-cp8r0 | cute@compare | 1054 | 121.0 | 0.93 | 3640 | 224.5 | 0.98 | 1222 |
| short-65536-sliding-cp8r0 | cute_ws@compare | 411.6 | 54.7 | 0.36 | 2999 | 145.0 | 0.81 | 1222 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 529.0 | 135.8 | 0.47 | - | - | - | 524 |
| short-65536-sliding-cp8r4 | tilelang@main | 1149 | 688.8 | 1.00 | 3813 | 810.9 | 1.00 | 1222 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 526.8 | 133.6 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r4 | tilelang@compare | 1151 | 688.8 | 1.00 | 3803 | 876.6 | 1.00 | 1222 |
| short-65536-sliding-cp8r4 | cudnn_flashmla@compare | 542.9 | 194.7 | 0.47 | 2604 | 427.8 | 0.68 | 1230 |
| short-65536-sliding-cp8r4 | cute@compare | 1061 | 124.6 | 0.92 | 3724 | 218.3 | 0.98 | 1222 |
| short-65536-sliding-cp8r4 | cute_ws@compare | 409.1 | 54.5 | 0.36 | 3076 | 128.1 | 0.81 | 1222 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 522.5 | 143.6 | 0.45 | - | - | - | 524 |
| short-65536-sliding-cp8r7 | tilelang@main | 1153 | 688.0 | 1.00 | 3777 | 855.3 | 1.00 | 1222 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 526.2 | 135.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r7 | tilelang@compare | 1154 | 678.9 | 1.00 | 3779 | 860.0 | 1.00 | 1222 |
| short-65536-sliding-cp8r7 | cudnn_flashmla@compare | 540.3 | 189.9 | 0.47 | 2600 | 403.9 | 0.69 | 1230 |
| short-65536-sliding-cp8r7 | cute@compare | 1065 | 118.0 | 0.92 | 3682 | 232.7 | 0.97 | 1222 |
| short-65536-sliding-cp8r7 | cute_ws@compare | 411.4 | 53.8 | 0.36 | 3037 | 147.2 | 0.80 | 1222 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 531.6 | 129.7 | 0.46 | - | - | - | 524 |
| heavy-65536-csa-cp1 | tilelang@main | 17414 | -41.5 | 1.00 | 69035 | 660.4 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8439 | 381.8 | 0.48 | - | - | - | 4448 |
| heavy-65536-csa-cp1 | tilelang@compare | 17344 | 113.6 | 1.00 | 69031 | 776.6 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | cudnn_flashmla@compare | 8440 | 370.6 | 0.49 | 40237 | -77.8 | 0.58 | 8672 |
| heavy-65536-csa-cp1 | cute@compare | 16198 | -19.7 | 0.93 | 68630 | -177.1 | 0.99 | 8480 |
| heavy-65536-csa-cp1 | cute_ws@compare | 7723 | 651.4 | 0.45 | 59954 | 149.7 | 0.87 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@compare | 8789 | 250.6 | 0.51 | - | - | - | 4448 |
| heavy-65536-csa-cp8r0 | tilelang@main | 1561 | 676.9 | 1.00 | 5628 | 791.6 | 1.00 | 1270 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 818.9 | 82.0 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r0 | tilelang@compare | 1562 | 687.2 | 1.00 | 5624 | 850.8 | 1.00 | 1270 |
| heavy-65536-csa-cp8r0 | cudnn_flashmla@compare | 821.6 | 149.5 | 0.53 | 3524 | 295.9 | 0.63 | 1294 |
| heavy-65536-csa-cp8r0 | cute@compare | 1471 | 108.0 | 0.94 | 5528 | 219.5 | 0.98 | 1270 |
| heavy-65536-csa-cp8r0 | cute_ws@compare | 708.2 | 55.2 | 0.45 | 4774 | 123.2 | 0.85 | 1270 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 814.9 | 87.8 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r4 | tilelang@main | 2719 | 683.2 | 1.00 | 12075 | 898.7 | 1.00 | 1270 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1326 | 75.9 | 0.49 | - | - | - | 556 |
| heavy-65536-csa-cp8r4 | tilelang@compare | 2816 | 579.4 | 1.00 | 12151 | 940.4 | 1.00 | 1270 |
| heavy-65536-csa-cp8r4 | cudnn_flashmla@compare | 1341 | 152.9 | 0.48 | 6419 | 359.7 | 0.53 | 1294 |
| heavy-65536-csa-cp8r4 | cute@compare | 2615 | 156.2 | 0.93 | 11968 | 377.6 | 0.98 | 1270 |
| heavy-65536-csa-cp8r4 | cute_ws@compare | 1233 | 64.5 | 0.44 | 10599 | 278.6 | 0.87 | 1270 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 1328 | 98.2 | 0.47 | - | - | - | 556 |
| heavy-65536-csa-cp8r7 | tilelang@main | 1888 | 661.1 | 1.00 | 7479 | 820.1 | 1.00 | 1270 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 972.7 | 74.8 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r7 | tilelang@compare | 1889 | 705.3 | 1.00 | 7480 | 867.7 | 1.00 | 1270 |
| heavy-65536-csa-cp8r7 | cudnn_flashmla@compare | 978.5 | 154.6 | 0.52 | 4370 | 309.7 | 0.58 | 1294 |
| heavy-65536-csa-cp8r7 | cute@compare | 1787 | 118.6 | 0.95 | 7366 | 239.4 | 0.98 | 1270 |
| heavy-65536-csa-cp8r7 | cute_ws@compare | 857.1 | 60.6 | 0.45 | 6445 | 133.9 | 0.86 | 1270 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 969.4 | 92.6 | 0.51 | - | - | - | 556 |
| heavy-65536-hca-cp1 | tilelang@main | 11197 | 516.8 | 1.00 | 37594 | 678.6 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5693 | -10.7 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp1 | tilelang@compare | 11261 | 509.7 | 1.00 | 37637 | 705.9 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | cudnn_flashmla@compare | 5845 | 15.0 | 0.52 | 24743 | 137.8 | 0.66 | 8594 |
| heavy-65536-hca-cp1 | cute@compare | 10554 | -0.6 | 0.94 | 37134 | 56.3 | 0.99 | 8530 |
| heavy-65536-hca-cp1 | cute_ws@compare | 4897 | 114.8 | 0.43 | 31258 | 194.8 | 0.83 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@compare | 5709 | 88.8 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp8r0 | tilelang@main | 1261 | 730.2 | 1.00 | 3956 | 864.8 | 1.00 | 1235 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 660.8 | 116.2 | 0.52 | - | - | - | 540 |
| heavy-65536-hca-cp8r0 | tilelang@compare | 1257 | 725.2 | 1.00 | 3958 | 897.5 | 1.00 | 1235 |
| heavy-65536-hca-cp8r0 | cudnn_flashmla@compare | 684.6 | 164.1 | 0.54 | 2795 | 306.7 | 0.71 | 1244 |
| heavy-65536-hca-cp8r0 | cute@compare | 1173 | 130.4 | 0.93 | 3864 | 264.1 | 0.98 | 1235 |
| heavy-65536-hca-cp8r0 | cute_ws@compare | 552.5 | 56.3 | 0.44 | 3250 | 115.1 | 0.82 | 1235 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 663.1 | 114.7 | 0.53 | - | - | - | 540 |
| heavy-65536-hca-cp8r4 | tilelang@main | 1655 | 722.9 | 1.00 | 6188 | 861.1 | 1.00 | 1235 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 811.0 | 117.0 | 0.49 | - | - | - | 540 |
| heavy-65536-hca-cp8r4 | tilelang@compare | 1660 | 734.7 | 1.00 | 6196 | 912.3 | 1.00 | 1235 |
| heavy-65536-hca-cp8r4 | cudnn_flashmla@compare | 837.4 | 165.6 | 0.50 | 3768 | 291.6 | 0.61 | 1244 |
| heavy-65536-hca-cp8r4 | cute@compare | 1554 | 136.4 | 0.94 | 6097 | 261.4 | 0.98 | 1235 |
| heavy-65536-hca-cp8r4 | cute_ws@compare | 712.0 | 59.6 | 0.43 | 5244 | 139.1 | 0.85 | 1235 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 817.7 | 117.1 | 0.49 | - | - | - | 540 |
| heavy-65536-hca-cp8r7 | tilelang@main | 1308 | 698.6 | 1.00 | 4132 | 878.0 | 1.00 | 1235 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 693.7 | 105.4 | 0.53 | - | - | - | 540 |
| heavy-65536-hca-cp8r7 | tilelang@compare | 1308 | 731.6 | 1.00 | 4130 | 919.8 | 1.00 | 1235 |
| heavy-65536-hca-cp8r7 | cudnn_flashmla@compare | 716.4 | 159.3 | 0.55 | 2903 | 312.7 | 0.70 | 1244 |
| heavy-65536-hca-cp8r7 | cute@compare | 1218 | 127.7 | 0.93 | 4046 | 251.3 | 0.98 | 1235 |
| heavy-65536-hca-cp8r7 | cute_ws@compare | 589.5 | 51.4 | 0.45 | 3423 | 112.8 | 0.83 | 1235 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 693.7 | 104.2 | 0.53 | - | - | - | 540 |
| heavy-65536-sliding-cp1 | tilelang@main | 8945 | 671.5 | 1.00 | 28442 | 840.0 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4091 | 20.5 | 0.46 | - | - | - | 4192 |
| heavy-65536-sliding-cp1 | tilelang@compare | 8990 | 629.5 | 1.00 | 28476 | 797.9 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | cudnn_flashmla@compare | 4161 | 69.5 | 0.46 | 19578 | 292.1 | 0.69 | 8496 |
| heavy-65536-sliding-cp1 | cute@compare | 8308 | 25.5 | 0.92 | 27794 | 204.5 | 0.98 | 8432 |
| heavy-65536-sliding-cp1 | cute_ws@compare | 3174 | 53.5 | 0.35 | 22636 | 199.5 | 0.79 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4109 | 54.6 | 0.46 | - | - | - | 4192 |
| heavy-65536-sliding-cp8r0 | tilelang@main | 1132 | 721.2 | 1.00 | 3582 | 823.1 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 528.4 | 141.3 | 0.47 | - | - | - | 524 |
| heavy-65536-sliding-cp8r0 | tilelang@compare | 1130 | 708.1 | 1.00 | 3586 | 863.9 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r0 | cudnn_flashmla@compare | 539.0 | 187.3 | 0.48 | 2507 | 425.6 | 0.70 | 1230 |
| heavy-65536-sliding-cp8r0 | cute@compare | 1041 | 121.7 | 0.92 | 3500 | 238.1 | 0.98 | 1222 |
| heavy-65536-sliding-cp8r0 | cute_ws@compare | 417.9 | 50.0 | 0.37 | 2871 | 134.2 | 0.80 | 1222 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 522.6 | 141.2 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r4 | tilelang@main | 1152 | 688.6 | 1.00 | 3866 | 811.3 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 524.6 | 131.3 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r4 | tilelang@compare | 1151 | 706.8 | 1.00 | 3859 | 882.3 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r4 | cudnn_flashmla@compare | 544.4 | 188.5 | 0.47 | 2626 | 424.1 | 0.68 | 1230 |
| heavy-65536-sliding-cp8r4 | cute@compare | 1069 | 124.7 | 0.93 | 3793 | 229.5 | 0.98 | 1222 |
| heavy-65536-sliding-cp8r4 | cute_ws@compare | 411.3 | 51.4 | 0.36 | 3129 | 143.7 | 0.81 | 1222 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 523.7 | 140.9 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r7 | tilelang@main | 1148 | 675.9 | 1.00 | 3690 | 833.4 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 525.3 | 140.4 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r7 | tilelang@compare | 1147 | 711.5 | 1.00 | 3687 | 860.7 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r7 | cudnn_flashmla@compare | 541.8 | 190.7 | 0.47 | 2556 | 427.9 | 0.69 | 1230 |
| heavy-65536-sliding-cp8r7 | cute@compare | 1057 | 125.8 | 0.92 | 3600 | 232.0 | 0.98 | 1222 |
| heavy-65536-sliding-cp8r7 | cute_ws@compare | 408.5 | 57.7 | 0.36 | 2956 | 172.0 | 0.80 | 1222 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 527.3 | 137.2 | 0.46 | - | - | - | 524 |
| tiny-65536-csa-cp1 | tilelang@main | 10414 | 470.2 | 1.00 | 31178 | 622.0 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5690 | -30.5 | 0.55 | - | - | - | 4448 |
| tiny-65536-csa-cp1 | tilelang@compare | 10430 | 489.8 | 1.00 | 31178 | 655.8 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | cudnn_flashmla@compare | 5702 | 31.5 | 0.55 | 21426 | 118.2 | 0.69 | 8671 |
| tiny-65536-csa-cp1 | cute@compare | 9744 | 32.4 | 0.93 | 30515 | 96.6 | 0.98 | 8479 |
| tiny-65536-csa-cp1 | cute_ws@compare | 4877 | 44.5 | 0.47 | 25614 | 181.7 | 0.82 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@compare | 5687 | -19.6 | 0.55 | - | - | - | 4448 |
| tiny-65536-csa-cp8r0 | tilelang@main | 1324 | 680.4 | 1.00 | 4061 | 830.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 727.4 | 81.8 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r0 | tilelang@compare | 1324 | 687.8 | 1.00 | 4062 | 864.9 | 1.00 | 1269 |
| tiny-65536-csa-cp8r0 | cudnn_flashmla@compare | 736.3 | 152.4 | 0.56 | 2800 | 326.3 | 0.69 | 1293 |
| tiny-65536-csa-cp8r0 | cute@compare | 1239 | 109.3 | 0.94 | 3974 | 260.8 | 0.98 | 1269 |
| tiny-65536-csa-cp8r0 | cute_ws@compare | 618.0 | 62.3 | 0.47 | 3357 | 172.9 | 0.83 | 1269 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 736.7 | 78.8 | 0.56 | - | - | - | 556 |
| tiny-65536-csa-cp8r4 | tilelang@main | 1326 | 681.4 | 1.00 | 4066 | 801.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 731.1 | 92.1 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r4 | tilelang@compare | 1327 | 682.0 | 1.00 | 4062 | 846.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r4 | cudnn_flashmla@compare | 737.4 | 152.9 | 0.56 | 2807 | 305.9 | 0.69 | 1293 |
| tiny-65536-csa-cp8r4 | cute@compare | 1242 | 110.0 | 0.94 | 3970 | 229.4 | 0.98 | 1269 |
| tiny-65536-csa-cp8r4 | cute_ws@compare | 627.0 | 57.9 | 0.47 | 3359 | 123.6 | 0.83 | 1269 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 727.4 | 85.6 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r7 | tilelang@main | 1330 | 734.8 | 1.00 | 4068 | 810.1 | 1.00 | 1269 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 728.8 | 102.6 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r7 | tilelang@compare | 1337 | 688.9 | 1.00 | 4063 | 852.1 | 1.00 | 1269 |
| tiny-65536-csa-cp8r7 | cudnn_flashmla@compare | 736.7 | 149.7 | 0.55 | 2806 | 299.0 | 0.69 | 1293 |
| tiny-65536-csa-cp8r7 | cute@compare | 1251 | 103.5 | 0.94 | 3975 | 237.3 | 0.98 | 1269 |
| tiny-65536-csa-cp8r7 | cute_ws@compare | 624.8 | 57.9 | 0.47 | 3353 | 134.5 | 0.83 | 1269 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 729.0 | 90.2 | 0.55 | - | - | - | 556 |
| tiny-65536-hca-cp1 | tilelang@main | 8038 | 651.6 | 1.00 | 21189 | 833.7 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4147 | 25.6 | 0.52 | - | - | - | 4192 |
| tiny-65536-hca-cp1 | tilelang@compare | 8043 | 646.6 | 1.00 | 21204 | 827.8 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | cudnn_flashmla@compare | 4184 | 92.3 | 0.52 | 16558 | 288.7 | 0.78 | 8496 |
| tiny-65536-hca-cp1 | cute@compare | 7361 | 72.0 | 0.92 | 20535 | 209.1 | 0.97 | 8432 |
| tiny-65536-hca-cp1 | cute_ws@compare | 3623 | 30.7 | 0.45 | 16765 | 187.8 | 0.79 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@compare | 4111 | 47.3 | 0.51 | - | - | - | 4192 |
| tiny-65536-hca-cp8r0 | tilelang@main | 1023 | 682.2 | 1.00 | 2791 | 837.3 | 1.00 | 1222 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 527.5 | 130.4 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r0 | tilelang@compare | 1021 | 698.1 | 1.00 | 2788 | 854.6 | 1.00 | 1222 |
| tiny-65536-hca-cp8r0 | cudnn_flashmla@compare | 538.7 | 189.9 | 0.53 | 2185 | 392.9 | 0.78 | 1230 |
| tiny-65536-hca-cp8r0 | cute@compare | 939.5 | 118.9 | 0.92 | 2710 | 216.9 | 0.97 | 1222 |
| tiny-65536-hca-cp8r0 | cute_ws@compare | 448.2 | 69.1 | 0.44 | 2241 | 111.2 | 0.80 | 1222 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 535.7 | 125.1 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r4 | tilelang@main | 1031 | 700.4 | 1.00 | 2795 | 813.5 | 1.00 | 1222 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 528.8 | 133.5 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r4 | tilelang@compare | 1028 | 707.1 | 1.00 | 2792 | 848.5 | 1.00 | 1222 |
| tiny-65536-hca-cp8r4 | cudnn_flashmla@compare | 540.0 | 190.5 | 0.53 | 2189 | 410.1 | 0.78 | 1230 |
| tiny-65536-hca-cp8r4 | cute@compare | 947.2 | 124.1 | 0.92 | 2713 | 221.1 | 0.97 | 1222 |
| tiny-65536-hca-cp8r4 | cute_ws@compare | 451.6 | 60.2 | 0.44 | 2225 | 115.3 | 0.80 | 1222 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 526.9 | 140.6 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r7 | tilelang@main | 1025 | 690.8 | 1.00 | 2797 | 833.6 | 1.00 | 1222 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 530.4 | 135.8 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r7 | tilelang@compare | 1025 | 697.5 | 1.00 | 2797 | 871.7 | 1.00 | 1222 |
| tiny-65536-hca-cp8r7 | cudnn_flashmla@compare | 546.4 | 187.0 | 0.53 | 2186 | 417.8 | 0.78 | 1230 |
| tiny-65536-hca-cp8r7 | cute@compare | 943.0 | 122.7 | 0.92 | 2715 | 227.0 | 0.97 | 1222 |
| tiny-65536-hca-cp8r7 | cute_ws@compare | 464.3 | 58.7 | 0.45 | 2243 | 111.1 | 0.80 | 1222 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 525.9 | 140.5 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp1 | tilelang@main | 8031 | 685.8 | 1.00 | 21186 | 850.2 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4116 | 62.0 | 0.51 | - | - | - | 4192 |
| tiny-65536-sliding-cp1 | tilelang@compare | 8029 | 661.0 | 1.00 | 21199 | 815.7 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | cudnn_flashmla@compare | 4198 | 73.1 | 0.52 | 16584 | 278.4 | 0.78 | 8496 |
| tiny-65536-sliding-cp1 | cute@compare | 7372 | 59.1 | 0.92 | 20528 | 216.9 | 0.97 | 8432 |
| tiny-65536-sliding-cp1 | cute_ws@compare | 3570 | 102.0 | 0.44 | 16749 | 180.5 | 0.79 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@compare | 4131 | 36.0 | 0.51 | - | - | - | 4192 |
| tiny-65536-sliding-cp8r0 | tilelang@main | 1021 | 723.6 | 1.00 | 2791 | 839.4 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 525.1 | 143.2 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r0 | tilelang@compare | 1021 | 720.3 | 1.00 | 2792 | 867.1 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r0 | cudnn_flashmla@compare | 536.8 | 201.8 | 0.53 | 2180 | 416.1 | 0.78 | 1230 |
| tiny-65536-sliding-cp8r0 | cute@compare | 939.8 | 122.1 | 0.92 | 2709 | 221.3 | 0.97 | 1222 |
| tiny-65536-sliding-cp8r0 | cute_ws@compare | 468.7 | 54.0 | 0.46 | 2242 | 108.6 | 0.80 | 1222 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 527.5 | 143.3 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp8r4 | tilelang@main | 1028 | 687.6 | 1.00 | 2791 | 821.8 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 522.8 | 139.4 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r4 | tilelang@compare | 1027 | 698.9 | 1.00 | 2790 | 864.5 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r4 | cudnn_flashmla@compare | 536.6 | 196.8 | 0.52 | 2189 | 412.8 | 0.78 | 1230 |
| tiny-65536-sliding-cp8r4 | cute@compare | 948.3 | 120.0 | 0.92 | 2708 | 227.1 | 0.97 | 1222 |
| tiny-65536-sliding-cp8r4 | cute_ws@compare | 464.0 | 51.0 | 0.45 | 2223 | 120.9 | 0.80 | 1222 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 525.3 | 138.2 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r7 | tilelang@main | 1017 | 687.1 | 1.00 | 2786 | 842.6 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 529.2 | 135.9 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp8r7 | tilelang@compare | 1025 | 692.7 | 1.00 | 2797 | 862.7 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r7 | cudnn_flashmla@compare | 544.8 | 190.7 | 0.53 | 2185 | 400.2 | 0.78 | 1230 |
| tiny-65536-sliding-cp8r7 | cute@compare | 943.6 | 118.7 | 0.92 | 2719 | 217.0 | 0.97 | 1222 |
| tiny-65536-sliding-cp8r7 | cute_ws@compare | 468.4 | 53.3 | 0.46 | 2247 | 109.6 | 0.80 | 1222 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 532.0 | 131.0 | 0.52 | - | - | - | 524 |

Useful FLOPs count valid slots only (fwd 4HD, bwd 10HD per slot); `exec/useful` counts the slots each
backend's tiles touch, or every padded slot for an arm without tile information. TFLOP/s divide
useful FLOPs by op-boundary time (higher is better);
`% peak` is f+b against 989.5 dense BF16 TFLOP/s (https://www.nvidia.com/en-us/data-center/h200/ (H200 SXM BF16 1,979 TFLOPS with sparsity, halved)).

| item | backend | f+b GFLOP | exec/useful fwd | exec/useful bwd | fwd TFLOP/s | f+b TFLOP/s | % peak |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 356.8 | 1.09 | 1.05 | 85.2 | 112.0 | 11.3 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 356.8 | 1.69 | 1.69 | 265.5 | - | - |
| single-2048-csa-cp1 | tilelang@compare | 356.8 | 1.09 | 1.05 | 84.2 | 109.2 | 11.0 |
| single-2048-csa-cp1 | cudnn_flashmla@compare | 356.8 | 1.09 | 1.09 | 233.0 | 183.9 | 18.6 |
| single-2048-csa-cp1 | cute@compare | 356.8 | 1.09 | 1.05 | 167.8 | 135.7 | 13.7 |
| single-2048-csa-cp1 | cute_ws@compare | 356.8 | 1.18 | 1.05 | 335.0 | 143.5 | 14.5 |
| single-2048-csa-cp1 | flashmla_fwd_ref@compare | 356.8 | 1.69 | 1.69 | 267.2 | - | - |
| single-2048-csa-cp8r0 | tilelang@main | 15.0 | 1.49 | 1.36 | 5.9 | 7.5 | 0.8 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 15.0 | 5.00 | 5.00 | 23.5 | - | - |
| single-2048-csa-cp8r0 | tilelang@compare | 15.0 | 1.49 | 1.36 | 5.6 | 7.1 | 0.7 |
| single-2048-csa-cp8r0 | cudnn_flashmla@compare | 15.0 | 1.49 | 1.49 | 13.5 | 10.2 | 1.0 |
| single-2048-csa-cp8r0 | cute@compare | 15.0 | 1.49 | 1.36 | 23.6 | 10.3 | 1.0 |
| single-2048-csa-cp8r0 | cute_ws@compare | 15.0 | 1.99 | 1.36 | 49.9 | 11.5 | 1.2 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 15.0 | 5.00 | 5.00 | 22.0 | - | - |
| single-2048-csa-cp8r4 | tilelang@main | 48.8 | 1.08 | 1.04 | 18.3 | 24.1 | 2.4 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 48.8 | 1.54 | 1.54 | 70.6 | - | - |
| single-2048-csa-cp8r4 | tilelang@compare | 48.8 | 1.08 | 1.04 | 18.1 | 23.0 | 2.3 |
| single-2048-csa-cp8r4 | cudnn_flashmla@compare | 48.8 | 1.08 | 1.08 | 44.9 | 33.1 | 3.3 |
| single-2048-csa-cp8r4 | cute@compare | 48.8 | 1.08 | 1.04 | 69.7 | 33.7 | 3.4 |
| single-2048-csa-cp8r4 | cute_ws@compare | 48.8 | 1.23 | 1.04 | 141.5 | 37.8 | 3.8 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 48.8 | 1.54 | 1.54 | 68.9 | - | - |
| single-2048-csa-cp8r7 | tilelang@main | 71.4 | 1.05 | 1.03 | 26.3 | 35.3 | 3.6 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 71.4 | 1.05 | 1.05 | 100.5 | - | - |
| single-2048-csa-cp8r7 | tilelang@compare | 71.4 | 1.05 | 1.03 | 25.8 | 33.7 | 3.4 |
| single-2048-csa-cp8r7 | cudnn_flashmla@compare | 71.4 | 1.05 | 1.05 | 64.7 | 48.6 | 4.9 |
| single-2048-csa-cp8r7 | cute@compare | 71.4 | 1.05 | 1.03 | 94.3 | 48.9 | 4.9 |
| single-2048-csa-cp8r7 | cute_ws@compare | 71.4 | 1.05 | 1.03 | 194.1 | 55.0 | 5.6 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 71.4 | 1.05 | 1.05 | 97.9 | - | - |
| single-2048-hca-cp1 | tilelang@main | 123.6 | 1.41 | 1.18 | 32.5 | 49.4 | 5.0 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 123.6 | 1.95 | 1.95 | 103.5 | - | - |
| single-2048-hca-cp1 | tilelang@compare | 123.6 | 1.41 | 1.18 | 32.0 | 48.1 | 4.9 |
| single-2048-hca-cp1 | cudnn_flashmla@compare | 123.6 | 1.41 | 1.41 | 87.4 | 78.0 | 7.9 |
| single-2048-hca-cp1 | cute@compare | 123.6 | 1.41 | 1.18 | 73.0 | 64.4 | 6.5 |
| single-2048-hca-cp1 | cute_ws@compare | 123.6 | 1.89 | 1.18 | 158.9 | 71.4 | 7.2 |
| single-2048-hca-cp1 | flashmla_fwd_ref@compare | 123.6 | 1.95 | 1.95 | 103.1 | - | - |
| single-2048-hca-cp8r0 | tilelang@main | 11.4 | 1.49 | 1.24 | 4.3 | 5.3 | 0.5 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 11.4 | 2.65 | 2.65 | 17.2 | - | - |
| single-2048-hca-cp8r0 | tilelang@compare | 11.4 | 1.49 | 1.24 | 4.1 | 5.1 | 0.5 |
| single-2048-hca-cp8r0 | cudnn_flashmla@compare | 11.4 | 1.49 | 1.49 | 10.1 | 7.6 | 0.8 |
| single-2048-hca-cp8r0 | cute@compare | 11.4 | 1.49 | 1.24 | 16.4 | 7.2 | 0.7 |
| single-2048-hca-cp8r0 | cute_ws@compare | 11.4 | 1.99 | 1.24 | 42.4 | 8.4 | 0.8 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 11.4 | 2.65 | 2.65 | 17.1 | - | - |
| single-2048-hca-cp8r4 | tilelang@main | 16.0 | 1.41 | 1.17 | 5.8 | 7.4 | 0.8 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 16.0 | 1.88 | 1.88 | 23.6 | - | - |
| single-2048-hca-cp8r4 | tilelang@compare | 16.0 | 1.41 | 1.17 | 5.7 | 7.2 | 0.7 |
| single-2048-hca-cp8r4 | cudnn_flashmla@compare | 16.0 | 1.41 | 1.41 | 14.5 | 10.8 | 1.1 |
| single-2048-hca-cp8r4 | cute@compare | 16.0 | 1.41 | 1.17 | 22.4 | 10.4 | 1.0 |
| single-2048-hca-cp8r4 | cute_ws@compare | 16.0 | 1.88 | 1.17 | 56.8 | 11.4 | 1.2 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 16.0 | 1.88 | 1.88 | 23.5 | - | - |
| single-2048-hca-cp8r7 | tilelang@main | 16.7 | 1.35 | 1.12 | 5.9 | 7.7 | 0.8 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 16.7 | 1.80 | 1.80 | 24.7 | - | - |
| single-2048-hca-cp8r7 | tilelang@compare | 16.7 | 1.35 | 1.12 | 6.1 | 7.5 | 0.8 |
| single-2048-hca-cp8r7 | cudnn_flashmla@compare | 16.7 | 1.35 | 1.35 | 15.4 | 11.5 | 1.2 |
| single-2048-hca-cp8r7 | cute@compare | 16.7 | 1.35 | 1.12 | 23.9 | 10.7 | 1.1 |
| single-2048-hca-cp8r7 | cute_ws@compare | 16.7 | 1.80 | 1.12 | 59.6 | 12.3 | 1.2 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 16.7 | 1.80 | 1.80 | 25.1 | - | - |
| single-2048-sliding-cp1 | tilelang@main | 116.5 | 1.02 | 1.01 | 32.8 | 50.9 | 5.1 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 111.6 | - | - |
| single-2048-sliding-cp1 | tilelang@compare | 116.5 | 1.02 | 1.01 | 32.4 | 48.8 | 4.9 |
| single-2048-sliding-cp1 | cudnn_flashmla@compare | 116.5 | 1.02 | 1.02 | 92.0 | 76.2 | 7.7 |
| single-2048-sliding-cp1 | cute@compare | 116.5 | 1.02 | 1.01 | 81.1 | 66.4 | 6.7 |
| single-2048-sliding-cp1 | cute_ws@compare | 116.5 | 1.03 | 1.01 | 194.3 | 72.9 | 7.4 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@compare | 116.5 | 1.03 | 1.03 | 113.3 | - | - |
| single-2048-sliding-cp8r0 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.4 | 5.5 | 0.6 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 18.0 | - | - |
| single-2048-sliding-cp8r0 | tilelang@compare | 11.3 | 1.16 | 1.08 | 4.3 | 5.3 | 0.5 |
| single-2048-sliding-cp8r0 | cudnn_flashmla@compare | 11.3 | 1.16 | 1.16 | 10.3 | 7.7 | 0.8 |
| single-2048-sliding-cp8r0 | cute@compare | 11.3 | 1.16 | 1.08 | 18.8 | 7.6 | 0.8 |
| single-2048-sliding-cp8r0 | cute_ws@compare | 11.3 | 1.33 | 1.08 | 44.8 | 8.6 | 0.9 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 11.3 | 1.33 | 1.33 | 18.0 | - | - |
| single-2048-sliding-cp8r4 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.8 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 24.0 | - | - |
| single-2048-sliding-cp8r4 | tilelang@compare | 15.0 | 1.00 | 1.00 | 5.7 | 7.1 | 0.7 |
| single-2048-sliding-cp8r4 | cudnn_flashmla@compare | 15.0 | 1.00 | 1.00 | 13.8 | 10.2 | 1.0 |
| single-2048-sliding-cp8r4 | cute@compare | 15.0 | 1.00 | 1.00 | 25.4 | 10.2 | 1.0 |
| single-2048-sliding-cp8r4 | cute_ws@compare | 15.0 | 1.00 | 1.00 | 57.2 | 11.5 | 1.2 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 15.0 | 1.00 | 1.00 | 23.8 | - | - |
| single-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.7 | - | - |
| single-2048-sliding-cp8r7 | tilelang@compare | 15.0 | 1.00 | 1.00 | 5.7 | 7.1 | 0.7 |
| single-2048-sliding-cp8r7 | cudnn_flashmla@compare | 15.0 | 1.00 | 1.00 | 13.5 | 10.2 | 1.0 |
| single-2048-sliding-cp8r7 | cute@compare | 15.0 | 1.00 | 1.00 | 24.9 | 10.3 | 1.0 |
| single-2048-sliding-cp8r7 | cute_ws@compare | 15.0 | 1.00 | 1.00 | 57.8 | 11.7 | 1.2 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 15.0 | 1.00 | 1.00 | 22.9 | - | - |
| short-2048-csa-cp1 | tilelang@main | 233.1 | 1.16 | 1.10 | 58.8 | 84.6 | 8.5 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 233.1 | 2.58 | 2.58 | 181.0 | - | - |
| short-2048-csa-cp1 | tilelang@compare | 233.1 | 1.16 | 1.10 | 59.0 | 81.1 | 8.2 |
| short-2048-csa-cp1 | cudnn_flashmla@compare | 233.1 | 1.16 | 1.16 | 154.8 | 133.5 | 13.5 |
| short-2048-csa-cp1 | cute@compare | 233.1 | 1.16 | 1.10 | 124.1 | 104.9 | 10.6 |
| short-2048-csa-cp1 | cute_ws@compare | 233.1 | 1.30 | 1.10 | 254.9 | 112.6 | 11.4 |
| short-2048-csa-cp1 | flashmla_fwd_ref@compare | 233.1 | 2.58 | 2.58 | 181.3 | - | - |
| short-2048-csa-cp8r0 | tilelang@main | 15.0 | 1.49 | 1.36 | 5.7 | 7.4 | 0.8 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 15.0 | 5.00 | 5.00 | 22.9 | - | - |
| short-2048-csa-cp8r0 | tilelang@compare | 15.0 | 1.49 | 1.36 | 5.6 | 7.1 | 0.7 |
| short-2048-csa-cp8r0 | cudnn_flashmla@compare | 15.0 | 1.49 | 1.49 | 13.6 | 10.1 | 1.0 |
| short-2048-csa-cp8r0 | cute@compare | 15.0 | 1.49 | 1.36 | 24.0 | 10.3 | 1.0 |
| short-2048-csa-cp8r0 | cute_ws@compare | 15.0 | 1.99 | 1.36 | 48.9 | 11.5 | 1.2 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 15.0 | 5.00 | 5.00 | 22.7 | - | - |
| short-2048-csa-cp8r4 | tilelang@main | 19.6 | 1.43 | 1.30 | 7.6 | 9.7 | 1.0 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 19.6 | 3.83 | 3.83 | 29.9 | - | - |
| short-2048-csa-cp8r4 | tilelang@compare | 19.6 | 1.43 | 1.30 | 7.4 | 9.4 | 0.9 |
| short-2048-csa-cp8r4 | cudnn_flashmla@compare | 19.6 | 1.43 | 1.43 | 18.2 | 13.6 | 1.4 |
| short-2048-csa-cp8r4 | cute@compare | 19.6 | 1.43 | 1.30 | 30.3 | 13.7 | 1.4 |
| short-2048-csa-cp8r4 | cute_ws@compare | 19.6 | 1.81 | 1.30 | 62.8 | 15.4 | 1.6 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 19.6 | 3.83 | 3.83 | 28.4 | - | - |
| short-2048-csa-cp8r7 | tilelang@main | 39.9 | 1.09 | 1.05 | 14.9 | 19.6 | 2.0 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 39.9 | 1.89 | 1.89 | 59.5 | - | - |
| short-2048-csa-cp8r7 | tilelang@compare | 39.9 | 1.09 | 1.05 | 14.9 | 19.1 | 1.9 |
| short-2048-csa-cp8r7 | cudnn_flashmla@compare | 39.9 | 1.09 | 1.09 | 37.0 | 27.3 | 2.8 |
| short-2048-csa-cp8r7 | cute@compare | 39.9 | 1.09 | 1.05 | 57.5 | 28.0 | 2.8 |
| short-2048-csa-cp8r7 | cute_ws@compare | 39.9 | 1.13 | 1.05 | 125.0 | 31.3 | 3.2 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 39.9 | 1.89 | 1.89 | 59.2 | - | - |
| short-2048-hca-cp1 | tilelang@main | 116.1 | 1.46 | 1.21 | 30.7 | 46.7 | 4.7 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 116.1 | 2.07 | 2.07 | 97.5 | - | - |
| short-2048-hca-cp1 | tilelang@compare | 116.1 | 1.46 | 1.21 | 30.9 | 46.2 | 4.7 |
| short-2048-hca-cp1 | cudnn_flashmla@compare | 116.1 | 1.46 | 1.46 | 82.4 | 74.4 | 7.5 |
| short-2048-hca-cp1 | cute@compare | 116.1 | 1.46 | 1.21 | 70.0 | 61.6 | 6.2 |
| short-2048-hca-cp1 | cute_ws@compare | 116.1 | 1.94 | 1.21 | 151.1 | 69.4 | 7.0 |
| short-2048-hca-cp1 | flashmla_fwd_ref@compare | 116.1 | 2.07 | 2.07 | 97.3 | - | - |
| short-2048-hca-cp8r0 | tilelang@main | 11.4 | 1.49 | 1.24 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 11.4 | 2.65 | 2.65 | 17.0 | - | - |
| short-2048-hca-cp8r0 | tilelang@compare | 11.4 | 1.49 | 1.24 | 4.1 | 5.2 | 0.5 |
| short-2048-hca-cp8r0 | cudnn_flashmla@compare | 11.4 | 1.49 | 1.49 | 10.3 | 7.8 | 0.8 |
| short-2048-hca-cp8r0 | cute@compare | 11.4 | 1.49 | 1.24 | 16.3 | 7.3 | 0.7 |
| short-2048-hca-cp8r0 | cute_ws@compare | 11.4 | 1.99 | 1.24 | 41.5 | 8.5 | 0.9 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 11.4 | 2.65 | 2.65 | 16.6 | - | - |
| short-2048-hca-cp8r4 | tilelang@main | 11.5 | 1.47 | 1.22 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 11.5 | 2.61 | 2.61 | 17.0 | - | - |
| short-2048-hca-cp8r4 | tilelang@compare | 11.5 | 1.47 | 1.22 | 4.2 | 5.2 | 0.5 |
| short-2048-hca-cp8r4 | cudnn_flashmla@compare | 11.5 | 1.47 | 1.47 | 10.8 | 7.8 | 0.8 |
| short-2048-hca-cp8r4 | cute@compare | 11.5 | 1.47 | 1.22 | 16.9 | 7.4 | 0.7 |
| short-2048-hca-cp8r4 | cute_ws@compare | 11.5 | 1.96 | 1.22 | 42.0 | 8.4 | 0.9 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 11.5 | 2.61 | 2.61 | 17.3 | - | - |
| short-2048-hca-cp8r7 | tilelang@main | 15.8 | 1.43 | 1.19 | 5.7 | 7.2 | 0.7 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 15.8 | 1.91 | 1.91 | 23.8 | - | - |
| short-2048-hca-cp8r7 | tilelang@compare | 15.8 | 1.43 | 1.19 | 5.7 | 7.1 | 0.7 |
| short-2048-hca-cp8r7 | cudnn_flashmla@compare | 15.8 | 1.43 | 1.43 | 14.5 | 10.7 | 1.1 |
| short-2048-hca-cp8r7 | cute@compare | 15.8 | 1.43 | 1.19 | 22.5 | 10.2 | 1.0 |
| short-2048-hca-cp8r7 | cute_ws@compare | 15.8 | 1.91 | 1.19 | 55.9 | 11.5 | 1.2 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 15.8 | 1.91 | 1.91 | 23.2 | - | - |
| short-2048-sliding-cp1 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.1 | 48.4 | 4.9 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 109.9 | - | - |
| short-2048-sliding-cp1 | tilelang@compare | 112.8 | 1.03 | 1.02 | 32.3 | 47.9 | 4.8 |
| short-2048-sliding-cp1 | cudnn_flashmla@compare | 112.8 | 1.03 | 1.03 | 90.8 | 73.7 | 7.4 |
| short-2048-sliding-cp1 | cute@compare | 112.8 | 1.03 | 1.02 | 79.6 | 65.5 | 6.6 |
| short-2048-sliding-cp1 | cute_ws@compare | 112.8 | 1.07 | 1.02 | 190.2 | 71.4 | 7.2 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@compare | 112.8 | 1.07 | 1.07 | 110.8 | - | - |
| short-2048-sliding-cp8r0 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.3 | 5.5 | 0.6 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 17.8 | - | - |
| short-2048-sliding-cp8r0 | tilelang@compare | 11.3 | 1.16 | 1.08 | 4.3 | 5.3 | 0.5 |
| short-2048-sliding-cp8r0 | cudnn_flashmla@compare | 11.3 | 1.16 | 1.16 | 10.6 | 7.6 | 0.8 |
| short-2048-sliding-cp8r0 | cute@compare | 11.3 | 1.16 | 1.08 | 18.9 | 7.6 | 0.8 |
| short-2048-sliding-cp8r0 | cute_ws@compare | 11.3 | 1.33 | 1.08 | 44.0 | 8.7 | 0.9 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 11.3 | 1.33 | 1.33 | 17.7 | - | - |
| short-2048-sliding-cp8r4 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.3 | 5.6 | 0.6 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 17.0 | - | - |
| short-2048-sliding-cp8r4 | tilelang@compare | 11.3 | 1.16 | 1.08 | 4.4 | 5.3 | 0.5 |
| short-2048-sliding-cp8r4 | cudnn_flashmla@compare | 11.3 | 1.16 | 1.16 | 10.5 | 7.7 | 0.8 |
| short-2048-sliding-cp8r4 | cute@compare | 11.3 | 1.16 | 1.08 | 19.2 | 7.8 | 0.8 |
| short-2048-sliding-cp8r4 | cute_ws@compare | 11.3 | 1.33 | 1.08 | 43.4 | 8.8 | 0.9 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 11.3 | 1.33 | 1.33 | 17.6 | - | - |
| short-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.2 | - | - |
| short-2048-sliding-cp8r7 | tilelang@compare | 15.0 | 1.00 | 1.00 | 5.8 | 7.2 | 0.7 |
| short-2048-sliding-cp8r7 | cudnn_flashmla@compare | 15.0 | 1.00 | 1.00 | 13.6 | 10.2 | 1.0 |
| short-2048-sliding-cp8r7 | cute@compare | 15.0 | 1.00 | 1.00 | 25.1 | 10.3 | 1.0 |
| short-2048-sliding-cp8r7 | cute_ws@compare | 15.0 | 1.00 | 1.00 | 57.7 | 11.7 | 1.2 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 15.0 | 1.00 | 1.00 | 23.9 | - | - |
| heavy-2048-csa-cp1 | tilelang@main | 272.5 | 1.16 | 1.09 | 66.5 | 92.2 | 9.3 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 272.5 | 2.21 | 2.21 | 199.8 | - | - |
| heavy-2048-csa-cp1 | tilelang@compare | 272.5 | 1.16 | 1.09 | 67.5 | 90.2 | 9.1 |
| heavy-2048-csa-cp1 | cudnn_flashmla@compare | 272.5 | 1.16 | 1.16 | 171.1 | 150.5 | 15.2 |
| heavy-2048-csa-cp1 | cute@compare | 272.5 | 1.16 | 1.09 | 136.9 | 115.3 | 11.7 |
| heavy-2048-csa-cp1 | cute_ws@compare | 272.5 | 1.29 | 1.09 | 280.3 | 123.0 | 12.4 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@compare | 272.5 | 2.21 | 2.21 | 199.6 | - | - |
| heavy-2048-csa-cp8r0 | tilelang@main | 10.3 | 2.16 | 1.86 | 3.9 | 5.0 | 0.5 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 10.3 | 7.32 | 7.32 | 15.5 | - | - |
| heavy-2048-csa-cp8r0 | tilelang@compare | 10.3 | 2.16 | 1.86 | 3.8 | 4.8 | 0.5 |
| heavy-2048-csa-cp8r0 | cudnn_flashmla@compare | 10.3 | 2.16 | 2.16 | 9.4 | 6.9 | 0.7 |
| heavy-2048-csa-cp8r0 | cute@compare | 10.3 | 2.16 | 1.86 | 16.4 | 7.0 | 0.7 |
| heavy-2048-csa-cp8r0 | cute_ws@compare | 10.3 | 2.89 | 1.86 | 34.0 | 7.8 | 0.8 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 10.3 | 7.32 | 7.32 | 15.6 | - | - |
| heavy-2048-csa-cp8r4 | tilelang@main | 37.7 | 1.10 | 1.05 | 13.8 | 18.2 | 1.8 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 37.7 | 2.00 | 2.00 | 53.6 | - | - |
| heavy-2048-csa-cp8r4 | tilelang@compare | 37.7 | 1.10 | 1.05 | 14.1 | 17.8 | 1.8 |
| heavy-2048-csa-cp8r4 | cudnn_flashmla@compare | 37.7 | 1.10 | 1.10 | 34.7 | 25.8 | 2.6 |
| heavy-2048-csa-cp8r4 | cute@compare | 37.7 | 1.10 | 1.05 | 56.3 | 25.7 | 2.6 |
| heavy-2048-csa-cp8r4 | cute_ws@compare | 37.7 | 1.20 | 1.05 | 115.1 | 28.8 | 2.9 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 37.7 | 2.00 | 2.00 | 54.6 | - | - |
| heavy-2048-csa-cp8r7 | tilelang@main | 60.2 | 1.06 | 1.03 | 21.7 | 29.6 | 3.0 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 60.2 | 1.25 | 1.25 | 82.1 | - | - |
| heavy-2048-csa-cp8r7 | tilelang@compare | 60.2 | 1.06 | 1.03 | 21.3 | 28.1 | 2.8 |
| heavy-2048-csa-cp8r7 | cudnn_flashmla@compare | 60.2 | 1.06 | 1.06 | 55.5 | 40.5 | 4.1 |
| heavy-2048-csa-cp8r7 | cute@compare | 60.2 | 1.06 | 1.03 | 82.9 | 40.9 | 4.1 |
| heavy-2048-csa-cp8r7 | cute_ws@compare | 60.2 | 1.12 | 1.03 | 166.6 | 46.3 | 4.7 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 60.2 | 1.25 | 1.25 | 83.9 | - | - |
| heavy-2048-hca-cp1 | tilelang@main | 113.7 | 1.44 | 1.20 | 30.3 | 46.2 | 4.7 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 113.7 | 2.11 | 2.11 | 97.0 | - | - |
| heavy-2048-hca-cp1 | tilelang@compare | 113.7 | 1.44 | 1.20 | 30.1 | 45.5 | 4.6 |
| heavy-2048-hca-cp1 | cudnn_flashmla@compare | 113.7 | 1.44 | 1.44 | 81.5 | 73.6 | 7.4 |
| heavy-2048-hca-cp1 | cute@compare | 113.7 | 1.44 | 1.20 | 68.5 | 61.5 | 6.2 |
| heavy-2048-hca-cp1 | cute_ws@compare | 113.7 | 1.92 | 1.20 | 151.7 | 68.0 | 6.9 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@compare | 113.7 | 2.11 | 2.11 | 96.1 | - | - |
| heavy-2048-hca-cp8r0 | tilelang@main | 8.2 | 1.57 | 1.28 | 3.0 | 3.8 | 0.4 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 8.2 | 3.68 | 3.68 | 12.3 | - | - |
| heavy-2048-hca-cp8r0 | tilelang@compare | 8.2 | 1.57 | 1.28 | 3.0 | 3.7 | 0.4 |
| heavy-2048-hca-cp8r0 | cudnn_flashmla@compare | 8.2 | 1.57 | 1.57 | 7.3 | 5.6 | 0.6 |
| heavy-2048-hca-cp8r0 | cute@compare | 8.2 | 1.57 | 1.28 | 12.1 | 5.2 | 0.5 |
| heavy-2048-hca-cp8r0 | cute_ws@compare | 8.2 | 2.21 | 1.28 | 30.3 | 6.0 | 0.6 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 8.2 | 3.68 | 3.68 | 12.4 | - | - |
| heavy-2048-hca-cp8r4 | tilelang@main | 15.7 | 1.44 | 1.20 | 5.7 | 7.4 | 0.7 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 15.7 | 1.92 | 1.92 | 23.6 | - | - |
| heavy-2048-hca-cp8r4 | tilelang@compare | 15.7 | 1.44 | 1.20 | 5.5 | 7.0 | 0.7 |
| heavy-2048-hca-cp8r4 | cudnn_flashmla@compare | 15.7 | 1.44 | 1.44 | 13.9 | 10.6 | 1.1 |
| heavy-2048-hca-cp8r4 | cute@compare | 15.7 | 1.44 | 1.20 | 21.9 | 10.1 | 1.0 |
| heavy-2048-hca-cp8r4 | cute_ws@compare | 15.7 | 1.92 | 1.20 | 54.3 | 11.6 | 1.2 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 15.7 | 1.92 | 1.92 | 23.0 | - | - |
| heavy-2048-hca-cp8r7 | tilelang@main | 16.4 | 1.38 | 1.15 | 6.0 | 7.6 | 0.8 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 16.4 | 1.83 | 1.83 | 24.7 | - | - |
| heavy-2048-hca-cp8r7 | tilelang@compare | 16.4 | 1.38 | 1.15 | 5.9 | 7.3 | 0.7 |
| heavy-2048-hca-cp8r7 | cudnn_flashmla@compare | 16.4 | 1.38 | 1.38 | 14.6 | 11.1 | 1.1 |
| heavy-2048-hca-cp8r7 | cute@compare | 16.4 | 1.38 | 1.15 | 23.0 | 10.5 | 1.1 |
| heavy-2048-hca-cp8r7 | cute_ws@compare | 16.4 | 1.83 | 1.15 | 56.4 | 12.3 | 1.2 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 16.4 | 1.83 | 1.83 | 24.0 | - | - |
| heavy-2048-sliding-cp1 | tilelang@main | 109.1 | 1.05 | 1.03 | 31.2 | 47.5 | 4.8 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 109.1 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-2048-sliding-cp1 | tilelang@compare | 109.1 | 1.05 | 1.03 | 31.0 | 46.5 | 4.7 |
| heavy-2048-sliding-cp1 | cudnn_flashmla@compare | 109.1 | 1.05 | 1.05 | 87.9 | 71.8 | 7.3 |
| heavy-2048-sliding-cp1 | cute@compare | 109.1 | 1.05 | 1.03 | 77.2 | 63.9 | 6.5 |
| heavy-2048-sliding-cp1 | cute_ws@compare | 109.1 | 1.10 | 1.03 | 185.2 | 70.1 | 7.1 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@compare | 109.1 | 1.10 | 1.10 | 107.4 | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@main | 8.1 | 1.39 | 1.19 | 3.2 | 4.0 | 0.4 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 8.1 | 1.85 | 1.85 | 12.9 | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@compare | 8.1 | 1.39 | 1.19 | 3.1 | 3.8 | 0.4 |
| heavy-2048-sliding-cp8r0 | cudnn_flashmla@compare | 8.1 | 1.39 | 1.39 | 7.3 | 5.5 | 0.6 |
| heavy-2048-sliding-cp8r0 | cute@compare | 8.1 | 1.39 | 1.19 | 13.7 | 5.6 | 0.6 |
| heavy-2048-sliding-cp8r0 | cute_ws@compare | 8.1 | 1.85 | 1.19 | 31.0 | 6.3 | 0.6 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 8.1 | 1.85 | 1.85 | 12.6 | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.8 | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@compare | 15.0 | 1.00 | 1.00 | 5.7 | 7.1 | 0.7 |
| heavy-2048-sliding-cp8r4 | cudnn_flashmla@compare | 15.0 | 1.00 | 1.00 | 13.6 | 10.3 | 1.0 |
| heavy-2048-sliding-cp8r4 | cute@compare | 15.0 | 1.00 | 1.00 | 24.5 | 10.3 | 1.0 |
| heavy-2048-sliding-cp8r4 | cute_ws@compare | 15.0 | 1.00 | 1.00 | 56.5 | 11.6 | 1.2 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 15.0 | 1.00 | 1.00 | 22.7 | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.3 | 0.7 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.5 | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@compare | 15.0 | 1.00 | 1.00 | 5.6 | 7.1 | 0.7 |
| heavy-2048-sliding-cp8r7 | cudnn_flashmla@compare | 15.0 | 1.00 | 1.00 | 13.4 | 10.3 | 1.0 |
| heavy-2048-sliding-cp8r7 | cute@compare | 15.0 | 1.00 | 1.00 | 24.7 | 10.2 | 1.0 |
| heavy-2048-sliding-cp8r7 | cute_ws@compare | 15.0 | 1.00 | 1.00 | 56.0 | 11.6 | 1.2 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 15.0 | 1.00 | 1.00 | 22.9 | - | - |
| tiny-2048-csa-cp1 | tilelang@main | 53.6 | 3.28 | 2.72 | 14.7 | 22.7 | 2.3 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 53.6 | 11.22 | 11.22 | 45.9 | - | - |
| tiny-2048-csa-cp1 | tilelang@compare | 53.6 | 3.28 | 2.72 | 14.4 | 22.1 | 2.2 |
| tiny-2048-csa-cp1 | cudnn_flashmla@compare | 53.6 | 3.28 | 3.28 | 38.7 | 35.1 | 3.5 |
| tiny-2048-csa-cp1 | cute@compare | 53.6 | 3.28 | 2.72 | 33.9 | 30.4 | 3.1 |
| tiny-2048-csa-cp1 | cute_ws@compare | 53.6 | 4.40 | 2.72 | 66.9 | 33.6 | 3.4 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@compare | 53.6 | 11.22 | 11.22 | 45.5 | - | - |
| tiny-2048-csa-cp8r0 | tilelang@main | 6.6 | 3.31 | 2.74 | 2.5 | 3.2 | 0.3 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 6.6 | 11.38 | 11.38 | 9.8 | - | - |
| tiny-2048-csa-cp8r0 | tilelang@compare | 6.6 | 3.31 | 2.74 | 2.5 | 3.1 | 0.3 |
| tiny-2048-csa-cp8r0 | cudnn_flashmla@compare | 6.6 | 3.31 | 3.31 | 6.0 | 4.5 | 0.5 |
| tiny-2048-csa-cp8r0 | cute@compare | 6.6 | 3.31 | 2.74 | 10.7 | 4.6 | 0.5 |
| tiny-2048-csa-cp8r0 | cute_ws@compare | 6.6 | 4.44 | 2.74 | 21.4 | 5.1 | 0.5 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@compare | 6.6 | 11.38 | 11.38 | 9.9 | - | - |
| tiny-2048-csa-cp8r4 | tilelang@main | 4.7 | 4.58 | 3.79 | 1.8 | 2.3 | 0.2 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 4.7 | 15.89 | 15.89 | 7.2 | - | - |
| tiny-2048-csa-cp8r4 | tilelang@compare | 4.7 | 4.58 | 3.79 | 1.8 | 2.2 | 0.2 |
| tiny-2048-csa-cp8r4 | cudnn_flashmla@compare | 4.7 | 4.58 | 4.58 | 4.3 | 3.2 | 0.3 |
| tiny-2048-csa-cp8r4 | cute@compare | 4.7 | 4.58 | 3.79 | 7.7 | 3.2 | 0.3 |
| tiny-2048-csa-cp8r4 | cute_ws@compare | 4.7 | 6.17 | 3.79 | 15.7 | 3.7 | 0.4 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@compare | 4.7 | 15.89 | 15.89 | 7.0 | - | - |
| tiny-2048-csa-cp8r7 | tilelang@main | 8.6 | 2.59 | 2.15 | 3.2 | 4.2 | 0.4 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 8.6 | 8.76 | 8.76 | 12.9 | - | - |
| tiny-2048-csa-cp8r7 | tilelang@compare | 8.6 | 2.59 | 2.15 | 3.2 | 4.0 | 0.4 |
| tiny-2048-csa-cp8r7 | cudnn_flashmla@compare | 8.6 | 2.59 | 2.59 | 7.8 | 5.8 | 0.6 |
| tiny-2048-csa-cp8r7 | cute@compare | 8.6 | 2.59 | 2.15 | 13.9 | 5.9 | 0.6 |
| tiny-2048-csa-cp8r7 | cute_ws@compare | 8.6 | 3.46 | 2.15 | 28.4 | 6.6 | 0.7 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@compare | 8.6 | 8.76 | 8.76 | 12.9 | - | - |
| tiny-2048-hca-cp1 | tilelang@main | 43.2 | 1.79 | 1.36 | 12.8 | 20.6 | 2.1 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 43.2 | 2.79 | 2.79 | 42.2 | - | - |
| tiny-2048-hca-cp1 | tilelang@compare | 43.2 | 1.79 | 1.36 | 12.8 | 20.1 | 2.0 |
| tiny-2048-hca-cp1 | cudnn_flashmla@compare | 43.2 | 1.79 | 1.79 | 33.9 | 28.9 | 2.9 |
| tiny-2048-hca-cp1 | cute@compare | 43.2 | 1.79 | 1.36 | 32.9 | 28.5 | 2.9 |
| tiny-2048-hca-cp1 | cute_ws@compare | 43.2 | 2.79 | 1.36 | 73.4 | 31.8 | 3.2 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@compare | 43.2 | 2.79 | 2.79 | 42.3 | - | - |
| tiny-2048-hca-cp8r0 | tilelang@main | 5.3 | 1.76 | 1.37 | 2.1 | 2.6 | 0.3 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 5.3 | 2.83 | 2.83 | 8.3 | - | - |
| tiny-2048-hca-cp8r0 | tilelang@compare | 5.3 | 1.76 | 1.37 | 2.1 | 2.5 | 0.3 |
| tiny-2048-hca-cp8r0 | cudnn_flashmla@compare | 5.3 | 1.76 | 1.76 | 4.8 | 3.6 | 0.4 |
| tiny-2048-hca-cp8r0 | cute@compare | 5.3 | 1.76 | 1.37 | 9.0 | 3.7 | 0.4 |
| tiny-2048-hca-cp8r0 | cute_ws@compare | 5.3 | 2.83 | 1.37 | 20.8 | 4.1 | 0.4 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@compare | 5.3 | 2.83 | 2.83 | 8.4 | - | - |
| tiny-2048-hca-cp8r4 | tilelang@main | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 3.8 | 3.94 | 3.94 | 6.0 | - | - |
| tiny-2048-hca-cp8r4 | tilelang@compare | 3.8 | 2.23 | 1.52 | 1.5 | 1.8 | 0.2 |
| tiny-2048-hca-cp8r4 | cudnn_flashmla@compare | 3.8 | 2.23 | 2.23 | 3.5 | 2.5 | 0.3 |
| tiny-2048-hca-cp8r4 | cute@compare | 3.8 | 2.23 | 1.52 | 6.4 | 2.6 | 0.3 |
| tiny-2048-hca-cp8r4 | cute_ws@compare | 3.8 | 3.94 | 1.52 | 14.5 | 2.9 | 0.3 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@compare | 3.8 | 3.94 | 3.94 | 5.9 | - | - |
| tiny-2048-hca-cp8r7 | tilelang@main | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 6.9 | 2.18 | 2.18 | 10.4 | - | - |
| tiny-2048-hca-cp8r7 | tilelang@compare | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-hca-cp8r7 | cudnn_flashmla@compare | 6.9 | 1.60 | 1.60 | 6.2 | 4.6 | 0.5 |
| tiny-2048-hca-cp8r7 | cute@compare | 6.9 | 1.60 | 1.28 | 11.5 | 4.7 | 0.5 |
| tiny-2048-hca-cp8r7 | cute_ws@compare | 6.9 | 2.18 | 1.28 | 25.7 | 5.3 | 0.5 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@compare | 6.9 | 2.18 | 2.18 | 10.7 | - | - |
| tiny-2048-sliding-cp1 | tilelang@main | 43.2 | 1.79 | 1.36 | 12.9 | 20.5 | 2.1 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 43.2 | 2.79 | 2.79 | 42.4 | - | - |
| tiny-2048-sliding-cp1 | tilelang@compare | 43.2 | 1.79 | 1.36 | 12.8 | 19.5 | 2.0 |
| tiny-2048-sliding-cp1 | cudnn_flashmla@compare | 43.2 | 1.79 | 1.79 | 34.3 | 28.4 | 2.9 |
| tiny-2048-sliding-cp1 | cute@compare | 43.2 | 1.79 | 1.36 | 32.6 | 28.0 | 2.8 |
| tiny-2048-sliding-cp1 | cute_ws@compare | 43.2 | 2.79 | 1.36 | 73.5 | 31.2 | 3.2 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@compare | 43.2 | 2.79 | 2.79 | 41.7 | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@main | 5.3 | 1.76 | 1.37 | 2.1 | 2.5 | 0.3 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 5.3 | 2.83 | 2.83 | 8.5 | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@compare | 5.3 | 1.76 | 1.37 | 2.0 | 2.4 | 0.2 |
| tiny-2048-sliding-cp8r0 | cudnn_flashmla@compare | 5.3 | 1.76 | 1.76 | 4.8 | 3.5 | 0.4 |
| tiny-2048-sliding-cp8r0 | cute@compare | 5.3 | 1.76 | 1.37 | 9.0 | 3.5 | 0.4 |
| tiny-2048-sliding-cp8r0 | cute_ws@compare | 5.3 | 2.83 | 1.37 | 20.5 | 4.0 | 0.4 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@compare | 5.3 | 2.83 | 2.83 | 8.2 | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@main | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 3.8 | 3.94 | 3.94 | 6.1 | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@compare | 3.8 | 2.23 | 1.52 | 1.4 | 1.8 | 0.2 |
| tiny-2048-sliding-cp8r4 | cudnn_flashmla@compare | 3.8 | 2.23 | 2.23 | 3.5 | 2.6 | 0.3 |
| tiny-2048-sliding-cp8r4 | cute@compare | 3.8 | 2.23 | 1.52 | 6.7 | 2.6 | 0.3 |
| tiny-2048-sliding-cp8r4 | cute_ws@compare | 3.8 | 3.94 | 1.52 | 15.2 | 3.0 | 0.3 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@compare | 3.8 | 3.94 | 3.94 | 6.2 | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@main | 6.9 | 1.60 | 1.28 | 2.6 | 3.4 | 0.3 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 6.9 | 2.18 | 2.18 | 10.6 | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@compare | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-sliding-cp8r7 | cudnn_flashmla@compare | 6.9 | 1.60 | 1.60 | 6.3 | 4.7 | 0.5 |
| tiny-2048-sliding-cp8r7 | cute@compare | 6.9 | 1.60 | 1.28 | 11.6 | 4.8 | 0.5 |
| tiny-2048-sliding-cp8r7 | cute_ws@compare | 6.9 | 2.18 | 1.28 | 26.6 | 5.3 | 0.5 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@compare | 6.9 | 2.18 | 2.18 | 10.8 | - | - |
| single-4096-csa-cp1 | tilelang@main | 958.1 | 1.03 | 1.02 | 144.8 | 160.4 | 16.2 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 958.1 | 1.26 | 1.26 | 382.7 | - | - |
| single-4096-csa-cp1 | tilelang@compare | 958.1 | 1.03 | 1.02 | 144.4 | 159.3 | 16.1 |
| single-4096-csa-cp1 | cudnn_flashmla@compare | 958.1 | 1.03 | 1.03 | 350.3 | 305.3 | 30.8 |
| single-4096-csa-cp1 | cute@compare | 958.1 | 1.03 | 1.02 | 215.2 | 179.7 | 18.2 |
| single-4096-csa-cp1 | cute_ws@compare | 958.1 | 1.07 | 1.02 | 452.9 | 202.6 | 20.5 |
| single-4096-csa-cp1 | flashmla_fwd_ref@compare | 958.1 | 1.26 | 1.26 | 383.1 | - | - |
| single-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.8 | 20.3 | 2.0 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 57.2 | - | - |
| single-4096-csa-cp8r0 | tilelang@compare | 41.3 | 1.27 | 1.18 | 14.7 | 19.6 | 2.0 |
| single-4096-csa-cp8r0 | cudnn_flashmla@compare | 41.3 | 1.27 | 1.27 | 37.7 | 27.8 | 2.8 |
| single-4096-csa-cp8r0 | cute@compare | 41.3 | 1.27 | 1.18 | 52.7 | 28.6 | 2.9 |
| single-4096-csa-cp8r0 | cute_ws@compare | 41.3 | 1.45 | 1.18 | 109.9 | 32.2 | 3.2 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 41.3 | 3.64 | 3.64 | 55.9 | - | - |
| single-4096-csa-cp8r4 | tilelang@main | 150.3 | 1.00 | 1.00 | 49.4 | 64.4 | 6.5 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 150.3 | 1.00 | 1.00 | 172.9 | - | - |
| single-4096-csa-cp8r4 | tilelang@compare | 150.3 | 1.00 | 1.00 | 48.2 | 62.7 | 6.3 |
| single-4096-csa-cp8r4 | cudnn_flashmla@compare | 150.3 | 1.00 | 1.00 | 137.2 | 101.9 | 10.3 |
| single-4096-csa-cp8r4 | cute@compare | 150.3 | 1.00 | 1.00 | 141.3 | 85.5 | 8.6 |
| single-4096-csa-cp8r4 | cute_ws@compare | 150.3 | 1.00 | 1.00 | 287.3 | 94.4 | 9.5 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 150.3 | 1.00 | 1.00 | 169.6 | - | - |
| single-4096-csa-cp8r7 | tilelang@main | 150.3 | 1.00 | 1.00 | 48.8 | 64.0 | 6.5 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 150.3 | 1.00 | 1.00 | 170.5 | - | - |
| single-4096-csa-cp8r7 | tilelang@compare | 150.3 | 1.00 | 1.00 | 49.0 | 61.7 | 6.2 |
| single-4096-csa-cp8r7 | cudnn_flashmla@compare | 150.3 | 1.00 | 1.00 | 137.7 | 101.0 | 10.2 |
| single-4096-csa-cp8r7 | cute@compare | 150.3 | 1.00 | 1.00 | 141.8 | 85.1 | 8.6 |
| single-4096-csa-cp8r7 | cute_ws@compare | 150.3 | 1.00 | 1.00 | 292.5 | 93.4 | 9.4 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 150.3 | 1.00 | 1.00 | 170.2 | - | - |
| single-4096-hca-cp1 | tilelang@main | 265.9 | 1.34 | 1.11 | 53.0 | 84.9 | 8.6 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 265.9 | 1.81 | 1.81 | 150.3 | - | - |
| single-4096-hca-cp1 | tilelang@compare | 265.9 | 1.34 | 1.11 | 53.2 | 83.2 | 8.4 |
| single-4096-hca-cp1 | cudnn_flashmla@compare | 265.9 | 1.34 | 1.34 | 133.2 | 125.8 | 12.7 |
| single-4096-hca-cp1 | cute@compare | 265.9 | 1.34 | 1.11 | 96.0 | 104.3 | 10.5 |
| single-4096-hca-cp1 | cute_ws@compare | 265.9 | 1.78 | 1.11 | 204.3 | 112.8 | 11.4 |
| single-4096-hca-cp1 | flashmla_fwd_ref@compare | 265.9 | 1.81 | 1.81 | 150.2 | - | - |
| single-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.2 | 12.6 | 1.3 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.1 | - | - |
| single-4096-hca-cp8r0 | tilelang@compare | 26.7 | 1.48 | 1.23 | 9.0 | 11.9 | 1.2 |
| single-4096-hca-cp8r0 | cudnn_flashmla@compare | 26.7 | 1.48 | 1.48 | 23.6 | 18.0 | 1.8 |
| single-4096-hca-cp8r0 | cute@compare | 26.7 | 1.48 | 1.23 | 31.5 | 16.9 | 1.7 |
| single-4096-hca-cp8r0 | cute_ws@compare | 26.7 | 1.97 | 1.23 | 76.0 | 19.5 | 2.0 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 26.7 | 2.25 | 2.25 | 34.5 | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 34.2 | 1.32 | 1.10 | 11.8 | 16.0 | 1.6 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 34.2 | 1.76 | 1.76 | 45.0 | - | - |
| single-4096-hca-cp8r4 | tilelang@compare | 34.2 | 1.32 | 1.10 | 10.7 | 15.2 | 1.5 |
| single-4096-hca-cp8r4 | cudnn_flashmla@compare | 34.2 | 1.32 | 1.32 | 29.7 | 23.0 | 2.3 |
| single-4096-hca-cp8r4 | cute@compare | 34.2 | 1.32 | 1.10 | 38.6 | 21.7 | 2.2 |
| single-4096-hca-cp8r4 | cute_ws@compare | 34.2 | 1.76 | 1.10 | 92.6 | 25.2 | 2.5 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 34.2 | 1.76 | 1.76 | 43.0 | - | - |
| single-4096-hca-cp8r7 | tilelang@main | 37.0 | 1.22 | 1.02 | 12.8 | 17.1 | 1.7 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 37.0 | 1.63 | 1.63 | 48.7 | - | - |
| single-4096-hca-cp8r7 | tilelang@compare | 37.0 | 1.22 | 1.02 | 12.5 | 16.0 | 1.6 |
| single-4096-hca-cp8r7 | cudnn_flashmla@compare | 37.0 | 1.22 | 1.22 | 33.0 | 24.3 | 2.5 |
| single-4096-hca-cp8r7 | cute@compare | 37.0 | 1.22 | 1.02 | 43.0 | 23.3 | 2.4 |
| single-4096-hca-cp8r7 | cute_ws@compare | 37.0 | 1.63 | 1.02 | 102.1 | 27.4 | 2.8 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 37.0 | 1.63 | 1.63 | 48.2 | - | - |
| single-4096-sliding-cp1 | tilelang@main | 236.8 | 1.01 | 1.00 | 52.2 | 82.8 | 8.4 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 236.8 | 1.02 | 1.02 | 161.0 | - | - |
| single-4096-sliding-cp1 | tilelang@compare | 236.8 | 1.01 | 1.00 | 50.6 | 80.6 | 8.1 |
| single-4096-sliding-cp1 | cudnn_flashmla@compare | 236.8 | 1.01 | 1.01 | 139.6 | 118.8 | 12.0 |
| single-4096-sliding-cp1 | cute@compare | 236.8 | 1.01 | 1.00 | 100.7 | 103.9 | 10.5 |
| single-4096-sliding-cp1 | cute_ws@compare | 236.8 | 1.02 | 1.00 | 250.8 | 111.9 | 11.3 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@compare | 236.8 | 1.02 | 1.02 | 160.3 | - | - |
| single-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 37.8 | - | - |
| single-4096-sliding-cp8r0 | tilelang@compare | 26.3 | 1.07 | 1.03 | 9.5 | 12.3 | 1.2 |
| single-4096-sliding-cp8r0 | cudnn_flashmla@compare | 26.3 | 1.07 | 1.07 | 24.4 | 17.8 | 1.8 |
| single-4096-sliding-cp8r0 | cute@compare | 26.3 | 1.07 | 1.03 | 36.2 | 17.8 | 1.8 |
| single-4096-sliding-cp8r0 | cute_ws@compare | 26.3 | 1.14 | 1.03 | 84.8 | 20.2 | 2.0 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 26.3 | 1.14 | 1.14 | 36.7 | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 30.1 | 1.00 | 1.00 | 42.3 | - | - |
| single-4096-sliding-cp8r4 | tilelang@compare | 30.1 | 1.00 | 1.00 | 10.8 | 14.2 | 1.4 |
| single-4096-sliding-cp8r4 | cudnn_flashmla@compare | 30.1 | 1.00 | 1.00 | 27.1 | 20.6 | 2.1 |
| single-4096-sliding-cp8r4 | cute@compare | 30.1 | 1.00 | 1.00 | 41.2 | 20.6 | 2.1 |
| single-4096-sliding-cp8r4 | cute_ws@compare | 30.1 | 1.00 | 1.00 | 98.4 | 23.5 | 2.4 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 30.1 | 1.00 | 1.00 | 42.9 | - | - |
| single-4096-sliding-cp8r7 | tilelang@main | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 30.1 | 1.00 | 1.00 | 44.3 | - | - |
| single-4096-sliding-cp8r7 | tilelang@compare | 30.1 | 1.00 | 1.00 | 11.0 | 14.2 | 1.4 |
| single-4096-sliding-cp8r7 | cudnn_flashmla@compare | 30.1 | 1.00 | 1.00 | 27.5 | 20.6 | 2.1 |
| single-4096-sliding-cp8r7 | cute@compare | 30.1 | 1.00 | 1.00 | 41.4 | 20.7 | 2.1 |
| single-4096-sliding-cp8r7 | cute_ws@compare | 30.1 | 1.00 | 1.00 | 97.3 | 23.4 | 2.4 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 30.1 | 1.00 | 1.00 | 43.0 | - | - |
| short-4096-csa-cp1 | tilelang@main | 449.8 | 1.18 | 1.11 | 84.4 | 116.1 | 11.7 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 449.8 | 2.67 | 2.67 | 232.4 | - | - |
| short-4096-csa-cp1 | tilelang@compare | 449.8 | 1.18 | 1.11 | 85.1 | 115.3 | 11.7 |
| short-4096-csa-cp1 | cudnn_flashmla@compare | 449.8 | 1.18 | 1.18 | 209.3 | 190.4 | 19.2 |
| short-4096-csa-cp1 | cute@compare | 449.8 | 1.18 | 1.11 | 143.0 | 139.3 | 14.1 |
| short-4096-csa-cp1 | cute_ws@compare | 449.8 | 1.33 | 1.11 | 300.3 | 150.1 | 15.2 |
| short-4096-csa-cp1 | flashmla_fwd_ref@compare | 449.8 | 2.67 | 2.67 | 231.5 | - | - |
| short-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.3 | 20.1 | 2.0 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 53.1 | - | - |
| short-4096-csa-cp8r0 | tilelang@compare | 41.3 | 1.27 | 1.18 | 14.6 | 19.7 | 2.0 |
| short-4096-csa-cp8r0 | cudnn_flashmla@compare | 41.3 | 1.27 | 1.27 | 37.3 | 28.4 | 2.9 |
| short-4096-csa-cp8r0 | cute@compare | 41.3 | 1.27 | 1.18 | 51.7 | 28.7 | 2.9 |
| short-4096-csa-cp8r0 | cute_ws@compare | 41.3 | 1.45 | 1.18 | 108.6 | 32.2 | 3.3 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 41.3 | 3.64 | 3.64 | 55.9 | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 44.6 | 1.21 | 1.13 | 14.5 | 20.8 | 2.1 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 44.6 | 3.37 | 3.37 | 56.1 | - | - |
| short-4096-csa-cp8r4 | tilelang@compare | 44.6 | 1.21 | 1.13 | 15.4 | 21.1 | 2.1 |
| short-4096-csa-cp8r4 | cudnn_flashmla@compare | 44.6 | 1.21 | 1.21 | 41.5 | 30.4 | 3.1 |
| short-4096-csa-cp8r4 | cute@compare | 44.6 | 1.21 | 1.13 | 55.8 | 30.8 | 3.1 |
| short-4096-csa-cp8r4 | cute_ws@compare | 44.6 | 1.38 | 1.13 | 115.0 | 34.6 | 3.5 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 44.6 | 3.37 | 3.37 | 59.2 | - | - |
| short-4096-csa-cp8r7 | tilelang@main | 42.3 | 1.26 | 1.17 | 14.8 | 20.6 | 2.1 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 42.3 | 3.55 | 3.55 | 57.4 | - | - |
| short-4096-csa-cp8r7 | tilelang@compare | 42.3 | 1.26 | 1.17 | 15.2 | 20.2 | 2.0 |
| short-4096-csa-cp8r7 | cudnn_flashmla@compare | 42.3 | 1.26 | 1.26 | 39.6 | 28.8 | 2.9 |
| short-4096-csa-cp8r7 | cute@compare | 42.3 | 1.26 | 1.17 | 53.4 | 29.3 | 3.0 |
| short-4096-csa-cp8r7 | cute_ws@compare | 42.3 | 1.46 | 1.17 | 111.6 | 32.6 | 3.3 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 42.3 | 3.55 | 3.55 | 57.4 | - | - |
| short-4096-hca-cp1 | tilelang@main | 228.1 | 1.46 | 1.22 | 46.2 | 73.8 | 7.5 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 228.1 | 2.11 | 2.11 | 130.3 | - | - |
| short-4096-hca-cp1 | tilelang@compare | 228.1 | 1.46 | 1.22 | 46.7 | 73.9 | 7.5 |
| short-4096-hca-cp1 | cudnn_flashmla@compare | 228.1 | 1.46 | 1.46 | 115.8 | 111.8 | 11.3 |
| short-4096-hca-cp1 | cute@compare | 228.1 | 1.46 | 1.22 | 85.3 | 92.9 | 9.4 |
| short-4096-hca-cp1 | cute_ws@compare | 228.1 | 1.95 | 1.22 | 181.7 | 100.5 | 10.2 |
| short-4096-hca-cp1 | flashmla_fwd_ref@compare | 228.1 | 2.11 | 2.11 | 130.0 | - | - |
| short-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.0 | 12.2 | 1.2 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.3 | - | - |
| short-4096-hca-cp8r0 | tilelang@compare | 26.7 | 1.48 | 1.23 | 9.1 | 12.1 | 1.2 |
| short-4096-hca-cp8r0 | cudnn_flashmla@compare | 26.7 | 1.48 | 1.48 | 24.0 | 18.1 | 1.8 |
| short-4096-hca-cp8r0 | cute@compare | 26.7 | 1.48 | 1.23 | 31.5 | 17.2 | 1.7 |
| short-4096-hca-cp8r0 | cute_ws@compare | 26.7 | 1.97 | 1.23 | 76.5 | 19.9 | 2.0 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 26.7 | 2.25 | 2.25 | 35.1 | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 28.3 | 1.46 | 1.23 | 9.8 | 13.1 | 1.3 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.3 | 2.13 | 2.13 | 38.8 | - | - |
| short-4096-hca-cp8r4 | tilelang@compare | 28.3 | 1.46 | 1.23 | 9.8 | 12.8 | 1.3 |
| short-4096-hca-cp8r4 | cudnn_flashmla@compare | 28.3 | 1.46 | 1.46 | 25.6 | 19.2 | 1.9 |
| short-4096-hca-cp8r4 | cute@compare | 28.3 | 1.46 | 1.23 | 33.7 | 18.3 | 1.8 |
| short-4096-hca-cp8r4 | cute_ws@compare | 28.3 | 1.92 | 1.23 | 81.5 | 21.3 | 2.2 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 28.3 | 2.13 | 2.13 | 37.8 | - | - |
| short-4096-hca-cp8r7 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.2 | 12.2 | 1.2 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.7 | - | - |
| short-4096-hca-cp8r7 | tilelang@compare | 26.7 | 1.48 | 1.23 | 9.2 | 12.2 | 1.2 |
| short-4096-hca-cp8r7 | cudnn_flashmla@compare | 26.7 | 1.48 | 1.48 | 23.9 | 18.4 | 1.9 |
| short-4096-hca-cp8r7 | cute@compare | 26.7 | 1.48 | 1.23 | 31.6 | 17.2 | 1.7 |
| short-4096-hca-cp8r7 | cute_ws@compare | 26.7 | 1.97 | 1.23 | 76.5 | 19.8 | 2.0 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 26.7 | 2.25 | 2.25 | 34.7 | - | - |
| short-4096-sliding-cp1 | tilelang@main | 221.9 | 1.04 | 1.02 | 48.3 | 78.1 | 7.9 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 221.9 | 1.08 | 1.08 | 149.8 | - | - |
| short-4096-sliding-cp1 | tilelang@compare | 221.9 | 1.04 | 1.02 | 49.2 | 77.3 | 7.8 |
| short-4096-sliding-cp1 | cudnn_flashmla@compare | 221.9 | 1.04 | 1.04 | 130.9 | 111.9 | 11.3 |
| short-4096-sliding-cp1 | cute@compare | 221.9 | 1.04 | 1.02 | 95.2 | 99.5 | 10.1 |
| short-4096-sliding-cp1 | cute_ws@compare | 221.9 | 1.08 | 1.02 | 233.1 | 107.0 | 10.8 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@compare | 221.9 | 1.08 | 1.08 | 152.9 | - | - |
| short-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 38.1 | - | - |
| short-4096-sliding-cp8r0 | tilelang@compare | 26.3 | 1.07 | 1.03 | 9.5 | 12.4 | 1.2 |
| short-4096-sliding-cp8r0 | cudnn_flashmla@compare | 26.3 | 1.07 | 1.07 | 24.2 | 17.8 | 1.8 |
| short-4096-sliding-cp8r0 | cute@compare | 26.3 | 1.07 | 1.03 | 36.6 | 18.0 | 1.8 |
| short-4096-sliding-cp8r0 | cute_ws@compare | 26.3 | 1.14 | 1.03 | 84.5 | 20.3 | 2.0 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 26.3 | 1.14 | 1.14 | 37.0 | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 27.9 | 1.04 | 1.02 | 10.0 | 13.7 | 1.4 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 27.9 | 1.08 | 1.08 | 40.3 | - | - |
| short-4096-sliding-cp8r4 | tilelang@compare | 27.9 | 1.04 | 1.02 | 10.0 | 13.1 | 1.3 |
| short-4096-sliding-cp8r4 | cudnn_flashmla@compare | 27.9 | 1.04 | 1.04 | 25.3 | 18.8 | 1.9 |
| short-4096-sliding-cp8r4 | cute@compare | 27.9 | 1.04 | 1.02 | 38.0 | 19.0 | 1.9 |
| short-4096-sliding-cp8r4 | cute_ws@compare | 27.9 | 1.08 | 1.02 | 88.1 | 21.3 | 2.2 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 27.9 | 1.08 | 1.08 | 39.2 | - | - |
| short-4096-sliding-cp8r7 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.7 | 12.6 | 1.3 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 38.4 | - | - |
| short-4096-sliding-cp8r7 | tilelang@compare | 26.3 | 1.07 | 1.03 | 9.4 | 12.3 | 1.2 |
| short-4096-sliding-cp8r7 | cudnn_flashmla@compare | 26.3 | 1.07 | 1.07 | 23.7 | 17.9 | 1.8 |
| short-4096-sliding-cp8r7 | cute@compare | 26.3 | 1.07 | 1.03 | 36.2 | 17.9 | 1.8 |
| short-4096-sliding-cp8r7 | cute_ws@compare | 26.3 | 1.14 | 1.03 | 85.0 | 20.3 | 2.1 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 26.3 | 1.14 | 1.14 | 38.0 | - | - |
| heavy-4096-csa-cp1 | tilelang@main | 356.7 | 1.29 | 1.19 | 69.7 | 100.6 | 10.2 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 356.7 | 3.37 | 3.37 | 194.2 | - | - |
| heavy-4096-csa-cp1 | tilelang@compare | 356.7 | 1.29 | 1.19 | 69.1 | 99.1 | 10.0 |
| heavy-4096-csa-cp1 | cudnn_flashmla@compare | 356.7 | 1.29 | 1.29 | 170.5 | 156.2 | 15.8 |
| heavy-4096-csa-cp1 | cute@compare | 356.7 | 1.29 | 1.19 | 120.8 | 122.9 | 12.4 |
| heavy-4096-csa-cp1 | cute_ws@compare | 356.7 | 1.52 | 1.19 | 248.0 | 131.0 | 13.2 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@compare | 356.7 | 3.37 | 3.37 | 192.1 | - | - |
| heavy-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.9 | 20.3 | 2.1 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 56.7 | - | - |
| heavy-4096-csa-cp8r0 | tilelang@compare | 41.3 | 1.27 | 1.18 | 14.5 | 19.4 | 2.0 |
| heavy-4096-csa-cp8r0 | cudnn_flashmla@compare | 41.3 | 1.27 | 1.27 | 37.2 | 27.8 | 2.8 |
| heavy-4096-csa-cp8r0 | cute@compare | 41.3 | 1.27 | 1.18 | 51.9 | 28.4 | 2.9 |
| heavy-4096-csa-cp8r0 | cute_ws@compare | 41.3 | 1.45 | 1.18 | 109.2 | 31.4 | 3.2 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 41.3 | 3.64 | 3.64 | 55.8 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 45.5 | 1.20 | 1.12 | 15.9 | 22.2 | 2.2 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 45.5 | 3.30 | 3.30 | 59.9 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@compare | 45.5 | 1.20 | 1.12 | 16.0 | 21.5 | 2.2 |
| heavy-4096-csa-cp8r4 | cudnn_flashmla@compare | 45.5 | 1.20 | 1.20 | 41.3 | 30.7 | 3.1 |
| heavy-4096-csa-cp8r4 | cute@compare | 45.5 | 1.20 | 1.12 | 56.9 | 31.2 | 3.2 |
| heavy-4096-csa-cp8r4 | cute_ws@compare | 45.5 | 1.37 | 1.12 | 117.3 | 34.9 | 3.5 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 45.5 | 3.30 | 3.30 | 59.6 | - | - |
| heavy-4096-csa-cp8r7 | tilelang@main | 29.6 | 1.51 | 1.38 | 9.9 | 14.2 | 1.4 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 29.6 | 5.09 | 5.09 | 39.1 | - | - |
| heavy-4096-csa-cp8r7 | tilelang@compare | 29.6 | 1.51 | 1.38 | 10.6 | 13.8 | 1.4 |
| heavy-4096-csa-cp8r7 | cudnn_flashmla@compare | 29.6 | 1.51 | 1.51 | 26.4 | 19.6 | 2.0 |
| heavy-4096-csa-cp8r7 | cute@compare | 29.6 | 1.51 | 1.38 | 38.2 | 20.0 | 2.0 |
| heavy-4096-csa-cp8r7 | cute_ws@compare | 29.6 | 2.02 | 1.38 | 77.7 | 22.6 | 2.3 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 29.6 | 5.09 | 5.09 | 39.9 | - | - |
| heavy-4096-hca-cp1 | tilelang@main | 207.2 | 1.47 | 1.23 | 43.1 | 69.5 | 7.0 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 207.2 | 2.32 | 2.32 | 122.6 | - | - |
| heavy-4096-hca-cp1 | tilelang@compare | 207.2 | 1.47 | 1.23 | 41.8 | 68.5 | 6.9 |
| heavy-4096-hca-cp1 | cudnn_flashmla@compare | 207.2 | 1.47 | 1.47 | 107.1 | 102.5 | 10.4 |
| heavy-4096-hca-cp1 | cute@compare | 207.2 | 1.47 | 1.23 | 78.9 | 86.8 | 8.8 |
| heavy-4096-hca-cp1 | cute_ws@compare | 207.2 | 1.96 | 1.23 | 172.3 | 94.5 | 9.5 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@compare | 207.2 | 2.32 | 2.32 | 122.0 | - | - |
| heavy-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.1 | 12.4 | 1.3 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.5 | - | - |
| heavy-4096-hca-cp8r0 | tilelang@compare | 26.7 | 1.48 | 1.23 | 9.1 | 11.9 | 1.2 |
| heavy-4096-hca-cp8r0 | cudnn_flashmla@compare | 26.7 | 1.48 | 1.48 | 23.5 | 17.8 | 1.8 |
| heavy-4096-hca-cp8r0 | cute@compare | 26.7 | 1.48 | 1.23 | 31.5 | 17.0 | 1.7 |
| heavy-4096-hca-cp8r0 | cute_ws@compare | 26.7 | 1.97 | 1.23 | 77.9 | 19.6 | 2.0 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 26.7 | 2.25 | 2.25 | 35.2 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 28.7 | 1.46 | 1.22 | 9.9 | 13.3 | 1.3 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.7 | 2.10 | 2.10 | 39.5 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@compare | 28.7 | 1.46 | 1.22 | 10.0 | 12.7 | 1.3 |
| heavy-4096-hca-cp8r4 | cudnn_flashmla@compare | 28.7 | 1.46 | 1.46 | 25.5 | 19.1 | 1.9 |
| heavy-4096-hca-cp8r4 | cute@compare | 28.7 | 1.46 | 1.22 | 33.5 | 18.0 | 1.8 |
| heavy-4096-hca-cp8r4 | cute_ws@compare | 28.7 | 1.92 | 1.22 | 82.5 | 21.0 | 2.1 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 28.7 | 2.10 | 2.10 | 38.3 | - | - |
| heavy-4096-hca-cp8r7 | tilelang@main | 22.7 | 1.49 | 1.24 | 7.7 | 10.4 | 1.1 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 22.7 | 2.65 | 2.65 | 30.1 | - | - |
| heavy-4096-hca-cp8r7 | tilelang@compare | 22.7 | 1.49 | 1.24 | 7.9 | 10.3 | 1.0 |
| heavy-4096-hca-cp8r7 | cudnn_flashmla@compare | 22.7 | 1.49 | 1.49 | 20.7 | 15.3 | 1.6 |
| heavy-4096-hca-cp8r7 | cute@compare | 22.7 | 1.49 | 1.24 | 27.5 | 14.5 | 1.5 |
| heavy-4096-hca-cp8r7 | cute_ws@compare | 22.7 | 1.99 | 1.24 | 67.8 | 16.8 | 1.7 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 22.7 | 2.65 | 2.65 | 31.0 | - | - |
| heavy-4096-sliding-cp1 | tilelang@main | 203.2 | 1.09 | 1.04 | 45.3 | 69.8 | 7.1 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 203.2 | 1.18 | 1.18 | 138.7 | - | - |
| heavy-4096-sliding-cp1 | tilelang@compare | 203.2 | 1.09 | 1.04 | 45.5 | 72.4 | 7.3 |
| heavy-4096-sliding-cp1 | cudnn_flashmla@compare | 203.2 | 1.09 | 1.09 | 120.0 | 105.2 | 10.6 |
| heavy-4096-sliding-cp1 | cute@compare | 203.2 | 1.09 | 1.04 | 87.9 | 94.2 | 9.5 |
| heavy-4096-sliding-cp1 | cute_ws@compare | 203.2 | 1.18 | 1.04 | 213.6 | 101.2 | 10.2 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@compare | 203.2 | 1.18 | 1.18 | 139.1 | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.6 | 1.3 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 37.9 | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@compare | 26.3 | 1.07 | 1.03 | 9.7 | 12.4 | 1.3 |
| heavy-4096-sliding-cp8r0 | cudnn_flashmla@compare | 26.3 | 1.07 | 1.07 | 24.3 | 18.0 | 1.8 |
| heavy-4096-sliding-cp8r0 | cute@compare | 26.3 | 1.07 | 1.03 | 36.9 | 18.2 | 1.8 |
| heavy-4096-sliding-cp8r0 | cute_ws@compare | 26.3 | 1.14 | 1.03 | 84.8 | 20.5 | 2.1 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 26.3 | 1.14 | 1.14 | 37.7 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 28.3 | 1.04 | 1.02 | 10.3 | 13.8 | 1.4 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 28.3 | 1.06 | 1.06 | 41.6 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@compare | 28.3 | 1.04 | 1.02 | 10.3 | 13.5 | 1.4 |
| heavy-4096-sliding-cp8r4 | cudnn_flashmla@compare | 28.3 | 1.04 | 1.04 | 26.0 | 19.6 | 2.0 |
| heavy-4096-sliding-cp8r4 | cute@compare | 28.3 | 1.04 | 1.02 | 39.4 | 19.8 | 2.0 |
| heavy-4096-sliding-cp8r4 | cute_ws@compare | 28.3 | 1.06 | 1.02 | 93.2 | 21.8 | 2.2 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 28.3 | 1.06 | 1.06 | 41.2 | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@main | 22.6 | 1.16 | 1.08 | 8.4 | 10.8 | 1.1 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 22.6 | 1.33 | 1.33 | 32.7 | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@compare | 22.6 | 1.16 | 1.08 | 8.2 | 10.7 | 1.1 |
| heavy-4096-sliding-cp8r7 | cudnn_flashmla@compare | 22.6 | 1.16 | 1.16 | 20.5 | 15.5 | 1.6 |
| heavy-4096-sliding-cp8r7 | cute@compare | 22.6 | 1.16 | 1.08 | 31.5 | 15.5 | 1.6 |
| heavy-4096-sliding-cp8r7 | cute_ws@compare | 22.6 | 1.33 | 1.08 | 71.7 | 17.7 | 1.8 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 22.6 | 1.33 | 1.33 | 32.1 | - | - |
| tiny-4096-csa-cp1 | tilelang@main | 104.6 | 3.35 | 2.77 | 21.8 | 35.8 | 3.6 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 104.6 | 11.49 | 11.49 | 59.6 | - | - |
| tiny-4096-csa-cp1 | tilelang@compare | 104.6 | 3.35 | 2.77 | 22.1 | 35.3 | 3.6 |
| tiny-4096-csa-cp1 | cudnn_flashmla@compare | 104.6 | 3.35 | 3.35 | 54.1 | 53.0 | 5.4 |
| tiny-4096-csa-cp1 | cute@compare | 104.6 | 3.35 | 2.77 | 39.9 | 45.5 | 4.6 |
| tiny-4096-csa-cp1 | cute_ws@compare | 104.6 | 4.50 | 2.77 | 78.5 | 48.3 | 4.9 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@compare | 104.6 | 11.49 | 11.49 | 60.9 | - | - |
| tiny-4096-csa-cp8r0 | tilelang@main | 11.9 | 3.67 | 3.04 | 4.3 | 5.9 | 0.6 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 11.9 | 12.64 | 12.64 | 16.1 | - | - |
| tiny-4096-csa-cp8r0 | tilelang@compare | 11.9 | 3.67 | 3.04 | 4.2 | 5.7 | 0.6 |
| tiny-4096-csa-cp8r0 | cudnn_flashmla@compare | 11.9 | 3.67 | 3.67 | 11.0 | 8.1 | 0.8 |
| tiny-4096-csa-cp8r0 | cute@compare | 11.9 | 3.67 | 3.04 | 15.6 | 8.2 | 0.8 |
| tiny-4096-csa-cp8r0 | cute_ws@compare | 11.9 | 4.94 | 3.04 | 31.4 | 9.2 | 0.9 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@compare | 11.9 | 12.64 | 12.64 | 16.2 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 15.1 | 2.92 | 2.42 | 5.3 | 7.4 | 0.7 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 15.1 | 9.98 | 9.98 | 20.5 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@compare | 15.1 | 2.92 | 2.42 | 5.3 | 7.2 | 0.7 |
| tiny-4096-csa-cp8r4 | cudnn_flashmla@compare | 15.1 | 2.92 | 2.92 | 14.0 | 10.3 | 1.0 |
| tiny-4096-csa-cp8r4 | cute@compare | 15.1 | 2.92 | 2.42 | 19.7 | 10.5 | 1.1 |
| tiny-4096-csa-cp8r4 | cute_ws@compare | 15.1 | 3.92 | 2.42 | 40.0 | 11.7 | 1.2 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@compare | 15.1 | 9.98 | 9.98 | 20.4 | - | - |
| tiny-4096-csa-cp8r7 | tilelang@main | 14.5 | 3.04 | 2.52 | 5.3 | 7.1 | 0.7 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 14.5 | 10.36 | 10.36 | 19.7 | - | - |
| tiny-4096-csa-cp8r7 | tilelang@compare | 14.5 | 3.04 | 2.52 | 5.2 | 6.8 | 0.7 |
| tiny-4096-csa-cp8r7 | cudnn_flashmla@compare | 14.5 | 3.04 | 3.04 | 13.4 | 9.9 | 1.0 |
| tiny-4096-csa-cp8r7 | cute@compare | 14.5 | 3.04 | 2.52 | 19.0 | 9.9 | 1.0 |
| tiny-4096-csa-cp8r7 | cute_ws@compare | 14.5 | 4.07 | 2.52 | 38.7 | 11.1 | 1.1 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@compare | 14.5 | 10.36 | 10.36 | 19.9 | - | - |
| tiny-4096-hca-cp1 | tilelang@main | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 84.3 | 2.85 | 2.85 | 58.4 | - | - |
| tiny-4096-hca-cp1 | tilelang@compare | 84.3 | 1.81 | 1.37 | 19.6 | 34.5 | 3.5 |
| tiny-4096-hca-cp1 | cudnn_flashmla@compare | 84.3 | 1.81 | 1.81 | 50.4 | 47.7 | 4.8 |
| tiny-4096-hca-cp1 | cute@compare | 84.3 | 1.81 | 1.37 | 39.5 | 46.5 | 4.7 |
| tiny-4096-hca-cp1 | cute_ws@compare | 84.3 | 2.85 | 1.37 | 85.8 | 51.0 | 5.2 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@compare | 84.3 | 2.85 | 2.85 | 57.3 | - | - |
| tiny-4096-hca-cp8r0 | tilelang@main | 9.6 | 1.91 | 1.41 | 3.6 | 4.7 | 0.5 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 9.6 | 3.14 | 3.14 | 14.1 | - | - |
| tiny-4096-hca-cp8r0 | tilelang@compare | 9.6 | 1.91 | 1.41 | 3.5 | 4.5 | 0.5 |
| tiny-4096-hca-cp8r0 | cudnn_flashmla@compare | 9.6 | 1.91 | 1.91 | 8.8 | 6.6 | 0.7 |
| tiny-4096-hca-cp8r0 | cute@compare | 9.6 | 1.91 | 1.41 | 14.0 | 6.5 | 0.7 |
| tiny-4096-hca-cp8r0 | cute_ws@compare | 9.6 | 3.14 | 1.41 | 31.5 | 7.4 | 0.7 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@compare | 9.6 | 3.14 | 3.14 | 14.0 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.5 | 6.0 | 0.6 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.6 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@compare | 12.1 | 1.66 | 1.32 | 4.4 | 5.7 | 0.6 |
| tiny-4096-hca-cp8r4 | cudnn_flashmla@compare | 12.1 | 1.66 | 1.66 | 11.1 | 8.2 | 0.8 |
| tiny-4096-hca-cp8r4 | cute@compare | 12.1 | 1.66 | 1.32 | 17.4 | 8.3 | 0.8 |
| tiny-4096-hca-cp8r4 | cute_ws@compare | 12.1 | 2.48 | 1.32 | 39.5 | 9.4 | 0.9 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@compare | 12.1 | 2.48 | 2.48 | 17.3 | - | - |
| tiny-4096-hca-cp8r7 | tilelang@main | 11.7 | 1.70 | 1.33 | 4.3 | 5.7 | 0.6 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 11.7 | 2.57 | 2.57 | 16.6 | - | - |
| tiny-4096-hca-cp8r7 | tilelang@compare | 11.7 | 1.70 | 1.33 | 4.3 | 5.6 | 0.6 |
| tiny-4096-hca-cp8r7 | cudnn_flashmla@compare | 11.7 | 1.70 | 1.70 | 10.7 | 8.0 | 0.8 |
| tiny-4096-hca-cp8r7 | cute@compare | 11.7 | 1.70 | 1.33 | 16.5 | 8.0 | 0.8 |
| tiny-4096-hca-cp8r7 | cute_ws@compare | 11.7 | 2.57 | 1.33 | 38.3 | 9.1 | 0.9 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@compare | 11.7 | 2.57 | 2.57 | 16.7 | - | - |
| tiny-4096-sliding-cp1 | tilelang@main | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 84.3 | 2.85 | 2.85 | 58.9 | - | - |
| tiny-4096-sliding-cp1 | tilelang@compare | 84.3 | 1.81 | 1.37 | 19.5 | 34.1 | 3.5 |
| tiny-4096-sliding-cp1 | cudnn_flashmla@compare | 84.3 | 1.81 | 1.81 | 50.1 | 47.3 | 4.8 |
| tiny-4096-sliding-cp1 | cute@compare | 84.3 | 1.81 | 1.37 | 39.3 | 46.5 | 4.7 |
| tiny-4096-sliding-cp1 | cute_ws@compare | 84.3 | 2.85 | 1.37 | 84.1 | 50.3 | 5.1 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@compare | 84.3 | 2.85 | 2.85 | 58.1 | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@main | 9.6 | 1.91 | 1.41 | 3.5 | 4.7 | 0.5 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 9.6 | 3.14 | 3.14 | 13.6 | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@compare | 9.6 | 1.91 | 1.41 | 3.5 | 4.5 | 0.5 |
| tiny-4096-sliding-cp8r0 | cudnn_flashmla@compare | 9.6 | 1.91 | 1.91 | 8.9 | 6.5 | 0.7 |
| tiny-4096-sliding-cp8r0 | cute@compare | 9.6 | 1.91 | 1.41 | 14.0 | 6.5 | 0.7 |
| tiny-4096-sliding-cp8r0 | cute_ws@compare | 9.6 | 3.14 | 1.41 | 31.4 | 7.4 | 0.7 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@compare | 9.6 | 3.14 | 3.14 | 13.8 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.4 | 5.9 | 0.6 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.1 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@compare | 12.1 | 1.66 | 1.32 | 4.4 | 5.7 | 0.6 |
| tiny-4096-sliding-cp8r4 | cudnn_flashmla@compare | 12.1 | 1.66 | 1.66 | 11.2 | 8.3 | 0.8 |
| tiny-4096-sliding-cp8r4 | cute@compare | 12.1 | 1.66 | 1.32 | 17.4 | 8.3 | 0.8 |
| tiny-4096-sliding-cp8r4 | cute_ws@compare | 12.1 | 2.48 | 1.32 | 39.2 | 9.3 | 0.9 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@compare | 12.1 | 2.48 | 2.48 | 17.4 | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@main | 11.7 | 1.70 | 1.33 | 4.2 | 5.6 | 0.6 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 11.7 | 2.57 | 2.57 | 16.5 | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@compare | 11.7 | 1.70 | 1.33 | 4.3 | 5.5 | 0.6 |
| tiny-4096-sliding-cp8r7 | cudnn_flashmla@compare | 11.7 | 1.70 | 1.70 | 10.7 | 7.9 | 0.8 |
| tiny-4096-sliding-cp8r7 | cute@compare | 11.7 | 1.70 | 1.33 | 17.0 | 7.9 | 0.8 |
| tiny-4096-sliding-cp8r7 | cute_ws@compare | 11.7 | 2.57 | 1.33 | 38.3 | 9.0 | 0.9 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@compare | 11.7 | 2.57 | 2.57 | 16.5 | - | - |
| single-16384-csa-cp1 | tilelang@main | 4565.9 | 1.01 | 1.00 | 220.5 | 193.8 | 19.6 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 4565.9 | 1.05 | 1.05 | 507.6 | - | - |
| single-16384-csa-cp1 | tilelang@compare | 4565.9 | 1.01 | 1.00 | 221.2 | 192.5 | 19.5 |
| single-16384-csa-cp1 | cudnn_flashmla@compare | 4565.9 | 1.01 | 1.01 | 495.5 | 359.9 | 36.4 |
| single-16384-csa-cp1 | cute@compare | 4565.9 | 1.01 | 1.00 | 252.6 | 199.8 | 20.2 |
| single-16384-csa-cp1 | cute_ws@compare | 4565.9 | 1.01 | 1.00 | 536.3 | 226.9 | 22.9 |
| single-16384-csa-cp1 | flashmla_fwd_ref@compare | 4565.9 | 1.05 | 1.05 | 472.5 | - | - |
| single-16384-csa-cp8r0 | tilelang@main | 356.8 | 1.09 | 1.05 | 82.7 | 110.0 | 11.1 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 356.8 | 1.69 | 1.69 | 244.1 | - | - |
| single-16384-csa-cp8r0 | tilelang@compare | 356.8 | 1.09 | 1.05 | 82.6 | 108.9 | 11.0 |
| single-16384-csa-cp8r0 | cudnn_flashmla@compare | 356.8 | 1.09 | 1.09 | 212.2 | 183.1 | 18.5 |
| single-16384-csa-cp8r0 | cute@compare | 356.8 | 1.09 | 1.05 | 162.0 | 135.7 | 13.7 |
| single-16384-csa-cp8r0 | cute_ws@compare | 356.8 | 1.18 | 1.05 | 331.5 | 143.4 | 14.5 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 356.8 | 1.69 | 1.69 | 242.8 | - | - |
| single-16384-csa-cp8r4 | tilelang@main | 601.3 | 1.00 | 1.00 | 121.9 | 147.4 | 14.9 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 601.3 | 1.00 | 1.00 | 347.0 | - | - |
| single-16384-csa-cp8r4 | tilelang@compare | 601.3 | 1.00 | 1.00 | 121.0 | 146.1 | 14.8 |
| single-16384-csa-cp8r4 | cudnn_flashmla@compare | 601.3 | 1.00 | 1.00 | 305.2 | 258.6 | 26.1 |
| single-16384-csa-cp8r4 | cute@compare | 601.3 | 1.00 | 1.00 | 215.2 | 173.4 | 17.5 |
| single-16384-csa-cp8r4 | cute_ws@compare | 601.3 | 1.00 | 1.00 | 447.3 | 181.1 | 18.3 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 601.3 | 1.00 | 1.00 | 349.5 | - | - |
| single-16384-csa-cp8r7 | tilelang@main | 601.3 | 1.00 | 1.00 | 122.3 | 146.5 | 14.8 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 601.3 | 1.00 | 1.00 | 345.9 | - | - |
| single-16384-csa-cp8r7 | tilelang@compare | 601.3 | 1.00 | 1.00 | 121.9 | 145.4 | 14.7 |
| single-16384-csa-cp8r7 | cudnn_flashmla@compare | 601.3 | 1.00 | 1.00 | 311.5 | 257.3 | 26.0 |
| single-16384-csa-cp8r7 | cute@compare | 601.3 | 1.00 | 1.00 | 216.0 | 173.0 | 17.5 |
| single-16384-csa-cp8r7 | cute_ws@compare | 601.3 | 1.00 | 1.00 | 448.5 | 181.0 | 18.3 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 601.3 | 1.00 | 1.00 | 347.4 | - | - |
| single-16384-hca-cp1 | tilelang@main | 1435.7 | 1.17 | 1.08 | 114.8 | 131.7 | 13.3 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1435.7 | 1.34 | 1.34 | 275.4 | - | - |
| single-16384-hca-cp1 | tilelang@compare | 1435.7 | 1.17 | 1.08 | 116.0 | 132.0 | 13.3 |
| single-16384-hca-cp1 | cudnn_flashmla@compare | 1435.7 | 1.17 | 1.17 | 263.3 | 219.7 | 22.2 |
| single-16384-hca-cp1 | cute@compare | 1435.7 | 1.17 | 1.08 | 147.9 | 142.9 | 14.4 |
| single-16384-hca-cp1 | cute_ws@compare | 1435.7 | 1.34 | 1.08 | 328.5 | 170.6 | 17.2 |
| single-16384-hca-cp1 | flashmla_fwd_ref@compare | 1435.7 | 1.34 | 1.34 | 276.1 | - | - |
| single-16384-hca-cp8r0 | tilelang@main | 123.6 | 1.41 | 1.18 | 32.7 | 51.0 | 5.2 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 123.6 | 1.95 | 1.95 | 103.7 | - | - |
| single-16384-hca-cp8r0 | tilelang@compare | 123.6 | 1.41 | 1.18 | 33.0 | 50.0 | 5.1 |
| single-16384-hca-cp8r0 | cudnn_flashmla@compare | 123.6 | 1.41 | 1.41 | 88.0 | 77.8 | 7.9 |
| single-16384-hca-cp8r0 | cute@compare | 123.6 | 1.41 | 1.18 | 77.0 | 67.8 | 6.8 |
| single-16384-hca-cp8r0 | cute_ws@compare | 123.6 | 1.89 | 1.18 | 158.4 | 73.9 | 7.5 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 123.6 | 1.95 | 1.95 | 103.1 | - | - |
| single-16384-hca-cp8r4 | tilelang@main | 187.4 | 1.26 | 1.11 | 48.8 | 70.3 | 7.1 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 187.4 | 1.28 | 1.28 | 157.6 | - | - |
| single-16384-hca-cp8r4 | tilelang@compare | 187.4 | 1.26 | 1.11 | 48.6 | 68.6 | 6.9 |
| single-16384-hca-cp8r4 | cudnn_flashmla@compare | 187.4 | 1.26 | 1.26 | 132.5 | 109.0 | 11.0 |
| single-16384-hca-cp8r4 | cute@compare | 187.4 | 1.26 | 1.11 | 104.7 | 91.1 | 9.2 |
| single-16384-hca-cp8r4 | cute_ws@compare | 187.4 | 1.28 | 1.11 | 238.0 | 98.5 | 10.0 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 187.4 | 1.28 | 1.28 | 156.8 | - | - |
| single-16384-hca-cp8r7 | tilelang@main | 232.5 | 1.03 | 1.03 | 60.1 | 82.5 | 8.3 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 232.5 | 1.03 | 1.03 | 193.2 | - | - |
| single-16384-hca-cp8r7 | tilelang@compare | 232.5 | 1.03 | 1.03 | 59.1 | 80.6 | 8.1 |
| single-16384-hca-cp8r7 | cudnn_flashmla@compare | 232.5 | 1.03 | 1.03 | 160.6 | 132.8 | 13.4 |
| single-16384-hca-cp8r7 | cute@compare | 232.5 | 1.03 | 1.03 | 128.3 | 107.1 | 10.8 |
| single-16384-hca-cp8r7 | cute_ws@compare | 232.5 | 1.03 | 1.03 | 294.5 | 114.3 | 11.6 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 232.5 | 1.03 | 1.03 | 193.8 | - | - |
| single-16384-sliding-cp1 | tilelang@main | 958.3 | 1.00 | 1.00 | 92.1 | 115.7 | 11.7 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 958.3 | 1.00 | 1.00 | 235.2 | - | - |
| single-16384-sliding-cp1 | tilelang@compare | 958.3 | 1.00 | 1.00 | 92.2 | 115.3 | 11.6 |
| single-16384-sliding-cp1 | cudnn_flashmla@compare | 958.3 | 1.00 | 1.00 | 220.1 | 179.5 | 18.1 |
| single-16384-sliding-cp1 | cute@compare | 958.3 | 1.00 | 1.00 | 123.7 | 127.8 | 12.9 |
| single-16384-sliding-cp1 | cute_ws@compare | 958.3 | 1.00 | 1.00 | 322.1 | 157.8 | 16.0 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@compare | 958.3 | 1.00 | 1.00 | 236.2 | - | - |
| single-16384-sliding-cp8r0 | tilelang@main | 116.5 | 1.02 | 1.01 | 32.7 | 49.9 | 5.0 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 111.1 | - | - |
| single-16384-sliding-cp8r0 | tilelang@compare | 116.5 | 1.02 | 1.01 | 33.0 | 48.4 | 4.9 |
| single-16384-sliding-cp8r0 | cudnn_flashmla@compare | 116.5 | 1.02 | 1.02 | 92.2 | 74.6 | 7.5 |
| single-16384-sliding-cp8r0 | cute@compare | 116.5 | 1.02 | 1.01 | 80.7 | 67.0 | 6.8 |
| single-16384-sliding-cp8r0 | cute_ws@compare | 116.5 | 1.03 | 1.01 | 194.1 | 73.1 | 7.4 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 116.5 | 1.03 | 1.03 | 113.7 | - | - |
| single-16384-sliding-cp8r4 | tilelang@main | 120.3 | 1.00 | 1.00 | 34.2 | 50.3 | 5.1 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 115.8 | - | - |
| single-16384-sliding-cp8r4 | tilelang@compare | 120.3 | 1.00 | 1.00 | 33.8 | 49.7 | 5.0 |
| single-16384-sliding-cp8r4 | cudnn_flashmla@compare | 120.3 | 1.00 | 1.00 | 95.1 | 77.3 | 7.8 |
| single-16384-sliding-cp8r4 | cute@compare | 120.3 | 1.00 | 1.00 | 81.8 | 67.7 | 6.8 |
| single-16384-sliding-cp8r4 | cute_ws@compare | 120.3 | 1.00 | 1.00 | 200.9 | 73.9 | 7.5 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 120.3 | 1.00 | 1.00 | 114.8 | - | - |
| single-16384-sliding-cp8r7 | tilelang@main | 120.3 | 1.00 | 1.00 | 34.5 | 50.3 | 5.1 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 116.9 | - | - |
| single-16384-sliding-cp8r7 | tilelang@compare | 120.3 | 1.00 | 1.00 | 33.7 | 49.0 | 5.0 |
| single-16384-sliding-cp8r7 | cudnn_flashmla@compare | 120.3 | 1.00 | 1.00 | 95.9 | 77.0 | 7.8 |
| single-16384-sliding-cp8r7 | cute@compare | 120.3 | 1.00 | 1.00 | 82.0 | 67.4 | 6.8 |
| single-16384-sliding-cp8r7 | cute_ws@compare | 120.3 | 1.00 | 1.00 | 201.1 | 73.4 | 7.4 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 120.3 | 1.00 | 1.00 | 116.3 | - | - |
| short-16384-csa-cp1 | tilelang@main | 2610.6 | 1.11 | 1.06 | 165.4 | 163.4 | 16.5 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 2610.6 | 1.84 | 1.84 | 381.2 | - | - |
| short-16384-csa-cp1 | tilelang@compare | 2610.6 | 1.11 | 1.06 | 165.9 | 163.2 | 16.5 |
| short-16384-csa-cp1 | cudnn_flashmla@compare | 2610.6 | 1.11 | 1.11 | 368.6 | 295.3 | 29.8 |
| short-16384-csa-cp1 | cute@compare | 2610.6 | 1.11 | 1.06 | 201.9 | 172.3 | 17.4 |
| short-16384-csa-cp1 | cute_ws@compare | 2610.6 | 1.20 | 1.06 | 416.2 | 198.0 | 20.0 |
| short-16384-csa-cp1 | flashmla_fwd_ref@compare | 2610.6 | 1.84 | 1.84 | 375.7 | - | - |
| short-16384-csa-cp8r0 | tilelang@main | 280.2 | 1.14 | 1.08 | 68.1 | 93.5 | 9.5 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 280.2 | 2.15 | 2.15 | 205.3 | - | - |
| short-16384-csa-cp8r0 | tilelang@compare | 280.2 | 1.14 | 1.08 | 67.3 | 93.0 | 9.4 |
| short-16384-csa-cp8r0 | cudnn_flashmla@compare | 280.2 | 1.14 | 1.14 | 177.3 | 153.3 | 15.5 |
| short-16384-csa-cp8r0 | cute@compare | 280.2 | 1.14 | 1.08 | 140.3 | 118.4 | 12.0 |
| short-16384-csa-cp8r0 | cute_ws@compare | 280.2 | 1.26 | 1.08 | 287.9 | 126.6 | 12.8 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 280.2 | 2.15 | 2.15 | 207.7 | - | - |
| short-16384-csa-cp8r4 | tilelang@main | 410.5 | 1.07 | 1.04 | 93.9 | 121.2 | 12.3 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 410.5 | 1.46 | 1.46 | 271.8 | - | - |
| short-16384-csa-cp8r4 | tilelang@compare | 410.5 | 1.07 | 1.04 | 92.4 | 118.6 | 12.0 |
| short-16384-csa-cp8r4 | cudnn_flashmla@compare | 410.5 | 1.07 | 1.07 | 238.6 | 202.7 | 20.5 |
| short-16384-csa-cp8r4 | cute@compare | 410.5 | 1.07 | 1.04 | 179.8 | 146.1 | 14.8 |
| short-16384-csa-cp8r4 | cute_ws@compare | 410.5 | 1.12 | 1.04 | 368.2 | 154.7 | 15.6 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 410.5 | 1.46 | 1.46 | 271.7 | - | - |
| short-16384-csa-cp8r7 | tilelang@main | 352.9 | 1.10 | 1.06 | 83.3 | 108.9 | 11.0 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 352.9 | 1.70 | 1.70 | 247.1 | - | - |
| short-16384-csa-cp8r7 | tilelang@compare | 352.9 | 1.10 | 1.06 | 83.7 | 106.3 | 10.7 |
| short-16384-csa-cp8r7 | cudnn_flashmla@compare | 352.9 | 1.10 | 1.10 | 212.9 | 179.6 | 18.2 |
| short-16384-csa-cp8r7 | cute@compare | 352.9 | 1.10 | 1.06 | 164.0 | 133.6 | 13.5 |
| short-16384-csa-cp8r7 | cute_ws@compare | 352.9 | 1.20 | 1.06 | 330.5 | 142.0 | 14.3 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 352.9 | 1.70 | 1.70 | 244.5 | - | - |
| short-16384-hca-cp1 | tilelang@main | 963.6 | 1.42 | 1.18 | 82.5 | 104.5 | 10.6 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 963.6 | 2.00 | 2.00 | 188.2 | - | - |
| short-16384-hca-cp1 | tilelang@compare | 963.6 | 1.42 | 1.18 | 82.8 | 105.4 | 10.7 |
| short-16384-hca-cp1 | cudnn_flashmla@compare | 963.6 | 1.42 | 1.42 | 178.0 | 163.8 | 16.6 |
| short-16384-hca-cp1 | cute@compare | 963.6 | 1.42 | 1.18 | 108.2 | 115.9 | 11.7 |
| short-16384-hca-cp1 | cute_ws@compare | 963.6 | 1.90 | 1.18 | 223.7 | 138.8 | 14.0 |
| short-16384-hca-cp1 | flashmla_fwd_ref@compare | 963.6 | 2.00 | 2.00 | 188.8 | - | - |
| short-16384-hca-cp8r0 | tilelang@main | 117.6 | 1.44 | 1.20 | 31.4 | 47.0 | 4.8 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 117.6 | 2.05 | 2.05 | 99.1 | - | - |
| short-16384-hca-cp8r0 | tilelang@compare | 117.6 | 1.44 | 1.20 | 30.7 | 46.1 | 4.7 |
| short-16384-hca-cp8r0 | cudnn_flashmla@compare | 117.6 | 1.44 | 1.44 | 82.3 | 73.7 | 7.4 |
| short-16384-hca-cp8r0 | cute@compare | 117.6 | 1.44 | 1.20 | 71.6 | 61.5 | 6.2 |
| short-16384-hca-cp8r0 | cute_ws@compare | 117.6 | 1.92 | 1.20 | 154.3 | 69.1 | 7.0 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 117.6 | 2.05 | 2.05 | 98.2 | - | - |
| short-16384-hca-cp8r4 | tilelang@main | 125.5 | 1.39 | 1.16 | 33.2 | 49.5 | 5.0 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 125.5 | 1.92 | 1.92 | 106.2 | - | - |
| short-16384-hca-cp8r4 | tilelang@compare | 125.5 | 1.39 | 1.16 | 32.5 | 47.5 | 4.8 |
| short-16384-hca-cp8r4 | cudnn_flashmla@compare | 125.5 | 1.39 | 1.39 | 88.0 | 77.0 | 7.8 |
| short-16384-hca-cp8r4 | cute@compare | 125.5 | 1.39 | 1.16 | 74.1 | 64.0 | 6.5 |
| short-16384-hca-cp8r4 | cute_ws@compare | 125.5 | 1.86 | 1.16 | 162.2 | 71.2 | 7.2 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 125.5 | 1.92 | 1.92 | 104.0 | - | - |
| short-16384-hca-cp8r7 | tilelang@main | 119.9 | 1.41 | 1.18 | 31.7 | 46.9 | 4.7 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 119.9 | 2.01 | 2.01 | 101.1 | - | - |
| short-16384-hca-cp8r7 | tilelang@compare | 119.9 | 1.41 | 1.18 | 31.3 | 46.6 | 4.7 |
| short-16384-hca-cp8r7 | cudnn_flashmla@compare | 119.9 | 1.41 | 1.41 | 82.7 | 74.6 | 7.5 |
| short-16384-hca-cp8r7 | cute@compare | 119.9 | 1.41 | 1.18 | 71.2 | 62.1 | 6.3 |
| short-16384-hca-cp8r7 | cute_ws@compare | 119.9 | 1.88 | 1.18 | 155.6 | 69.6 | 7.0 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 119.9 | 2.01 | 2.01 | 99.8 | - | - |
| short-16384-sliding-cp1 | tilelang@main | 913.6 | 1.03 | 1.01 | 87.4 | 112.0 | 11.3 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 913.6 | 1.05 | 1.05 | 223.3 | - | - |
| short-16384-sliding-cp1 | tilelang@compare | 913.6 | 1.03 | 1.01 | 88.3 | 112.5 | 11.4 |
| short-16384-sliding-cp1 | cudnn_flashmla@compare | 913.6 | 1.03 | 1.03 | 209.3 | 173.8 | 17.6 |
| short-16384-sliding-cp1 | cute@compare | 913.6 | 1.03 | 1.01 | 118.5 | 124.5 | 12.6 |
| short-16384-sliding-cp1 | cute_ws@compare | 913.6 | 1.05 | 1.01 | 306.8 | 153.8 | 15.5 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@compare | 913.6 | 1.05 | 1.05 | 223.9 | - | - |
| short-16384-sliding-cp8r0 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.2 | 48.9 | 4.9 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 108.8 | - | - |
| short-16384-sliding-cp8r0 | tilelang@compare | 112.8 | 1.03 | 1.02 | 32.1 | 46.7 | 4.7 |
| short-16384-sliding-cp8r0 | cudnn_flashmla@compare | 112.8 | 1.03 | 1.03 | 88.5 | 72.3 | 7.3 |
| short-16384-sliding-cp8r0 | cute@compare | 112.8 | 1.03 | 1.02 | 78.7 | 64.7 | 6.5 |
| short-16384-sliding-cp8r0 | cute_ws@compare | 112.8 | 1.07 | 1.02 | 190.3 | 70.9 | 7.2 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 112.8 | 1.07 | 1.07 | 109.1 | - | - |
| short-16384-sliding-cp8r4 | tilelang@main | 116.5 | 1.02 | 1.01 | 33.5 | 49.2 | 5.0 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 113.1 | - | - |
| short-16384-sliding-cp8r4 | tilelang@compare | 116.5 | 1.02 | 1.01 | 32.8 | 48.5 | 4.9 |
| short-16384-sliding-cp8r4 | cudnn_flashmla@compare | 116.5 | 1.02 | 1.02 | 91.3 | 75.4 | 7.6 |
| short-16384-sliding-cp8r4 | cute@compare | 116.5 | 1.02 | 1.01 | 79.2 | 66.3 | 6.7 |
| short-16384-sliding-cp8r4 | cute_ws@compare | 116.5 | 1.03 | 1.01 | 194.9 | 71.9 | 7.3 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 116.5 | 1.03 | 1.03 | 112.0 | - | - |
| short-16384-sliding-cp8r7 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.2 | 47.8 | 4.8 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 109.4 | - | - |
| short-16384-sliding-cp8r7 | tilelang@compare | 112.8 | 1.03 | 1.02 | 30.9 | 47.0 | 4.7 |
| short-16384-sliding-cp8r7 | cudnn_flashmla@compare | 112.8 | 1.03 | 1.03 | 89.1 | 72.4 | 7.3 |
| short-16384-sliding-cp8r7 | cute@compare | 112.8 | 1.03 | 1.02 | 77.6 | 64.0 | 6.5 |
| short-16384-sliding-cp8r7 | cute_ws@compare | 112.8 | 1.07 | 1.02 | 188.9 | 70.1 | 7.1 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 112.8 | 1.07 | 1.07 | 109.5 | - | - |
| heavy-16384-csa-cp1 | tilelang@main | 1795.9 | 1.22 | 1.14 | 129.3 | 138.2 | 14.0 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1795.9 | 2.68 | 2.68 | 297.3 | - | - |
| heavy-16384-csa-cp1 | tilelang@compare | 1795.9 | 1.22 | 1.14 | 129.2 | 139.1 | 14.1 |
| heavy-16384-csa-cp1 | cudnn_flashmla@compare | 1795.9 | 1.22 | 1.22 | 286.7 | 241.0 | 24.4 |
| heavy-16384-csa-cp1 | cute@compare | 1795.9 | 1.22 | 1.14 | 160.8 | 148.3 | 15.0 |
| heavy-16384-csa-cp1 | cute_ws@compare | 1795.9 | 1.40 | 1.14 | 329.5 | 172.2 | 17.4 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@compare | 1795.9 | 2.68 | 2.68 | 296.1 | - | - |
| heavy-16384-csa-cp8r0 | tilelang@main | 154.2 | 1.36 | 1.24 | 40.3 | 59.9 | 6.1 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 154.2 | 3.90 | 3.90 | 127.1 | - | - |
| heavy-16384-csa-cp8r0 | tilelang@compare | 154.2 | 1.36 | 1.24 | 40.5 | 59.1 | 6.0 |
| heavy-16384-csa-cp8r0 | cudnn_flashmla@compare | 154.2 | 1.36 | 1.36 | 106.9 | 93.4 | 9.4 |
| heavy-16384-csa-cp8r0 | cute@compare | 154.2 | 1.36 | 1.24 | 90.1 | 78.0 | 7.9 |
| heavy-16384-csa-cp8r0 | cute_ws@compare | 154.2 | 1.65 | 1.24 | 185.1 | 84.3 | 8.5 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 154.2 | 3.90 | 3.90 | 125.7 | - | - |
| heavy-16384-csa-cp8r4 | tilelang@main | 219.5 | 1.19 | 1.12 | 56.2 | 79.5 | 8.0 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 219.5 | 2.74 | 2.74 | 172.9 | - | - |
| heavy-16384-csa-cp8r4 | tilelang@compare | 219.5 | 1.19 | 1.12 | 55.2 | 76.3 | 7.7 |
| heavy-16384-csa-cp8r4 | cudnn_flashmla@compare | 219.5 | 1.19 | 1.19 | 141.9 | 124.7 | 12.6 |
| heavy-16384-csa-cp8r4 | cute@compare | 219.5 | 1.19 | 1.12 | 119.0 | 99.7 | 10.1 |
| heavy-16384-csa-cp8r4 | cute_ws@compare | 219.5 | 1.35 | 1.12 | 244.0 | 107.5 | 10.9 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 219.5 | 2.74 | 2.74 | 171.1 | - | - |
| heavy-16384-csa-cp8r7 | tilelang@main | 410.5 | 1.06 | 1.03 | 92.8 | 117.9 | 11.9 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 410.5 | 1.46 | 1.46 | 273.7 | - | - |
| heavy-16384-csa-cp8r7 | tilelang@compare | 410.5 | 1.06 | 1.03 | 92.1 | 116.7 | 11.8 |
| heavy-16384-csa-cp8r7 | cudnn_flashmla@compare | 410.5 | 1.06 | 1.06 | 233.3 | 198.3 | 20.0 |
| heavy-16384-csa-cp8r7 | cute@compare | 410.5 | 1.06 | 1.03 | 175.0 | 143.3 | 14.5 |
| heavy-16384-csa-cp8r7 | cute_ws@compare | 410.5 | 1.12 | 1.03 | 359.9 | 151.6 | 15.3 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 410.5 | 1.46 | 1.46 | 266.5 | - | - |
| heavy-16384-hca-cp1 | tilelang@main | 847.5 | 1.45 | 1.21 | 74.5 | 97.4 | 9.8 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 847.5 | 2.27 | 2.27 | 172.5 | - | - |
| heavy-16384-hca-cp1 | tilelang@compare | 847.5 | 1.45 | 1.21 | 75.4 | 98.4 | 9.9 |
| heavy-16384-hca-cp1 | cudnn_flashmla@compare | 847.5 | 1.45 | 1.45 | 164.2 | 150.6 | 15.2 |
| heavy-16384-hca-cp1 | cute@compare | 847.5 | 1.45 | 1.21 | 98.9 | 108.3 | 10.9 |
| heavy-16384-hca-cp1 | cute_ws@compare | 847.5 | 1.94 | 1.21 | 208.2 | 130.8 | 13.2 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@compare | 847.5 | 2.27 | 2.27 | 173.3 | - | - |
| heavy-16384-hca-cp8r0 | tilelang@main | 99.2 | 1.48 | 1.23 | 26.6 | 40.5 | 4.1 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 99.2 | 2.42 | 2.42 | 84.5 | - | - |
| heavy-16384-hca-cp8r0 | tilelang@compare | 99.2 | 1.48 | 1.23 | 26.4 | 39.5 | 4.0 |
| heavy-16384-hca-cp8r0 | cudnn_flashmla@compare | 99.2 | 1.48 | 1.48 | 71.2 | 63.4 | 6.4 |
| heavy-16384-hca-cp8r0 | cute@compare | 99.2 | 1.48 | 1.23 | 61.8 | 53.6 | 5.4 |
| heavy-16384-hca-cp8r0 | cute_ws@compare | 99.2 | 1.97 | 1.23 | 137.0 | 60.1 | 6.1 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 99.2 | 2.42 | 2.42 | 84.9 | - | - |
| heavy-16384-hca-cp8r4 | tilelang@main | 112.5 | 1.46 | 1.22 | 30.1 | 45.2 | 4.6 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 112.5 | 2.14 | 2.14 | 94.1 | - | - |
| heavy-16384-hca-cp8r4 | tilelang@compare | 112.5 | 1.46 | 1.22 | 30.2 | 43.7 | 4.4 |
| heavy-16384-hca-cp8r4 | cudnn_flashmla@compare | 112.5 | 1.46 | 1.46 | 79.5 | 70.1 | 7.1 |
| heavy-16384-hca-cp8r4 | cute@compare | 112.5 | 1.46 | 1.22 | 68.4 | 58.9 | 6.0 |
| heavy-16384-hca-cp8r4 | cute_ws@compare | 112.5 | 1.94 | 1.22 | 151.7 | 65.8 | 6.6 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 112.5 | 2.14 | 2.14 | 94.9 | - | - |
| heavy-16384-hca-cp8r7 | tilelang@main | 129.0 | 1.40 | 1.17 | 33.9 | 49.2 | 5.0 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 129.0 | 1.86 | 1.86 | 106.5 | - | - |
| heavy-16384-hca-cp8r7 | tilelang@compare | 129.0 | 1.40 | 1.17 | 33.3 | 48.7 | 4.9 |
| heavy-16384-hca-cp8r7 | cudnn_flashmla@compare | 129.0 | 1.40 | 1.40 | 90.7 | 78.4 | 7.9 |
| heavy-16384-hca-cp8r7 | cute@compare | 129.0 | 1.40 | 1.17 | 75.7 | 65.2 | 6.6 |
| heavy-16384-hca-cp8r7 | cute_ws@compare | 129.0 | 1.86 | 1.17 | 163.7 | 72.5 | 7.3 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 129.0 | 1.86 | 1.86 | 105.9 | - | - |
| heavy-16384-sliding-cp1 | tilelang@main | 820.4 | 1.09 | 1.04 | 78.7 | 104.0 | 10.5 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 820.4 | 1.17 | 1.17 | 199.8 | - | - |
| heavy-16384-sliding-cp1 | tilelang@compare | 820.4 | 1.09 | 1.04 | 79.2 | 104.3 | 10.5 |
| heavy-16384-sliding-cp1 | cudnn_flashmla@compare | 820.4 | 1.09 | 1.09 | 187.4 | 159.3 | 16.1 |
| heavy-16384-sliding-cp1 | cute@compare | 820.4 | 1.09 | 1.04 | 106.7 | 116.0 | 11.7 |
| heavy-16384-sliding-cp1 | cute_ws@compare | 820.4 | 1.17 | 1.04 | 269.2 | 143.9 | 14.5 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@compare | 820.4 | 1.17 | 1.17 | 199.7 | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@main | 97.9 | 1.11 | 1.06 | 27.6 | 42.7 | 4.3 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 97.9 | 1.23 | 1.23 | 93.9 | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@compare | 97.9 | 1.11 | 1.06 | 27.6 | 41.4 | 4.2 |
| heavy-16384-sliding-cp8r0 | cudnn_flashmla@compare | 97.9 | 1.11 | 1.11 | 77.5 | 63.8 | 6.4 |
| heavy-16384-sliding-cp8r0 | cute@compare | 97.9 | 1.11 | 1.06 | 69.4 | 57.5 | 5.8 |
| heavy-16384-sliding-cp8r0 | cute_ws@compare | 97.9 | 1.23 | 1.06 | 164.3 | 62.9 | 6.4 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 97.9 | 1.23 | 1.23 | 94.0 | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@main | 109.5 | 1.05 | 1.02 | 31.8 | 47.0 | 4.7 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 109.5 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@compare | 109.5 | 1.05 | 1.02 | 31.1 | 46.1 | 4.7 |
| heavy-16384-sliding-cp8r4 | cudnn_flashmla@compare | 109.5 | 1.05 | 1.05 | 86.3 | 70.7 | 7.1 |
| heavy-16384-sliding-cp8r4 | cute@compare | 109.5 | 1.05 | 1.02 | 76.1 | 63.2 | 6.4 |
| heavy-16384-sliding-cp8r4 | cute_ws@compare | 109.5 | 1.10 | 1.02 | 183.5 | 68.9 | 7.0 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 109.5 | 1.10 | 1.10 | 105.8 | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@main | 120.3 | 1.00 | 1.00 | 33.6 | 50.0 | 5.1 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 115.9 | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@compare | 120.3 | 1.00 | 1.00 | 33.4 | 48.6 | 4.9 |
| heavy-16384-sliding-cp8r7 | cudnn_flashmla@compare | 120.3 | 1.00 | 1.00 | 94.7 | 76.9 | 7.8 |
| heavy-16384-sliding-cp8r7 | cute@compare | 120.3 | 1.00 | 1.00 | 80.8 | 67.2 | 6.8 |
| heavy-16384-sliding-cp8r7 | cute_ws@compare | 120.3 | 1.00 | 1.00 | 198.6 | 73.5 | 7.4 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 120.3 | 1.00 | 1.00 | 112.8 | - | - |
| tiny-16384-csa-cp1 | tilelang@main | 391.4 | 3.56 | 2.95 | 34.0 | 44.9 | 4.5 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 391.4 | 12.29 | 12.29 | 75.8 | - | - |
| tiny-16384-csa-cp1 | tilelang@compare | 391.4 | 3.56 | 2.95 | 33.9 | 44.6 | 4.5 |
| tiny-16384-csa-cp1 | cudnn_flashmla@compare | 391.4 | 3.56 | 3.56 | 72.8 | 70.4 | 7.1 |
| tiny-16384-csa-cp1 | cute@compare | 391.4 | 3.56 | 2.95 | 44.0 | 49.5 | 5.0 |
| tiny-16384-csa-cp1 | cute_ws@compare | 391.4 | 4.79 | 2.95 | 86.0 | 59.2 | 6.0 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@compare | 391.4 | 12.29 | 12.29 | 75.7 | - | - |
| tiny-16384-csa-cp8r0 | tilelang@main | 50.6 | 3.45 | 2.86 | 13.9 | 21.3 | 2.2 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 50.6 | 11.89 | 11.89 | 43.2 | - | - |
| tiny-16384-csa-cp8r0 | tilelang@compare | 50.6 | 3.45 | 2.86 | 13.6 | 20.7 | 2.1 |
| tiny-16384-csa-cp8r0 | cudnn_flashmla@compare | 50.6 | 3.45 | 3.45 | 35.5 | 32.6 | 3.3 |
| tiny-16384-csa-cp8r0 | cute@compare | 50.6 | 3.45 | 2.86 | 31.8 | 28.5 | 2.9 |
| tiny-16384-csa-cp8r0 | cute_ws@compare | 50.6 | 4.64 | 2.86 | 63.5 | 31.4 | 3.2 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@compare | 50.6 | 11.89 | 11.89 | 42.4 | - | - |
| tiny-16384-csa-cp8r4 | tilelang@main | 48.5 | 3.60 | 2.98 | 13.2 | 20.6 | 2.1 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 48.5 | 12.39 | 12.39 | 40.5 | - | - |
| tiny-16384-csa-cp8r4 | tilelang@compare | 48.5 | 3.60 | 2.98 | 13.3 | 20.1 | 2.0 |
| tiny-16384-csa-cp8r4 | cudnn_flashmla@compare | 48.5 | 3.60 | 3.60 | 34.6 | 31.4 | 3.2 |
| tiny-16384-csa-cp8r4 | cute@compare | 48.5 | 3.60 | 2.98 | 30.7 | 27.5 | 2.8 |
| tiny-16384-csa-cp8r4 | cute_ws@compare | 48.5 | 4.84 | 2.98 | 60.9 | 30.0 | 3.0 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@compare | 48.5 | 12.39 | 12.39 | 41.4 | - | - |
| tiny-16384-csa-cp8r7 | tilelang@main | 43.9 | 3.96 | 3.27 | 12.1 | 18.5 | 1.9 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 43.9 | 13.71 | 13.71 | 37.5 | - | - |
| tiny-16384-csa-cp8r7 | tilelang@compare | 43.9 | 3.96 | 3.27 | 12.0 | 18.0 | 1.8 |
| tiny-16384-csa-cp8r7 | cudnn_flashmla@compare | 43.9 | 3.96 | 3.96 | 31.8 | 27.9 | 2.8 |
| tiny-16384-csa-cp8r7 | cute@compare | 43.9 | 3.96 | 3.27 | 27.7 | 24.7 | 2.5 |
| tiny-16384-csa-cp8r7 | cute_ws@compare | 43.9 | 5.33 | 3.27 | 54.9 | 27.2 | 2.8 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@compare | 43.9 | 13.71 | 13.71 | 37.1 | - | - |
| tiny-16384-hca-cp1 | tilelang@main | 315.4 | 1.89 | 1.40 | 33.3 | 51.0 | 5.2 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 315.4 | 3.05 | 3.05 | 76.6 | - | - |
| tiny-16384-hca-cp1 | tilelang@compare | 315.4 | 1.89 | 1.40 | 33.2 | 51.0 | 5.2 |
| tiny-16384-hca-cp1 | cudnn_flashmla@compare | 315.4 | 1.89 | 1.89 | 71.7 | 70.5 | 7.1 |
| tiny-16384-hca-cp1 | cute@compare | 315.4 | 1.89 | 1.40 | 45.6 | 58.4 | 5.9 |
| tiny-16384-hca-cp1 | cute_ws@compare | 315.4 | 3.05 | 1.40 | 92.4 | 72.4 | 7.3 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@compare | 315.4 | 3.05 | 3.05 | 76.3 | - | - |
| tiny-16384-hca-cp8r0 | tilelang@main | 40.7 | 1.86 | 1.39 | 11.9 | 19.0 | 1.9 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-hca-cp8r0 | tilelang@compare | 40.7 | 1.86 | 1.39 | 11.7 | 18.7 | 1.9 |
| tiny-16384-hca-cp8r0 | cudnn_flashmla@compare | 40.7 | 1.86 | 1.86 | 32.3 | 26.8 | 2.7 |
| tiny-16384-hca-cp8r0 | cute@compare | 40.7 | 1.86 | 1.39 | 30.4 | 26.7 | 2.7 |
| tiny-16384-hca-cp8r0 | cute_ws@compare | 40.7 | 2.95 | 1.39 | 69.6 | 29.6 | 3.0 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@compare | 40.7 | 2.95 | 2.95 | 38.8 | - | - |
| tiny-16384-hca-cp8r4 | tilelang@main | 39.1 | 1.88 | 1.40 | 11.5 | 18.7 | 1.9 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 39.1 | 3.07 | 3.07 | 38.1 | - | - |
| tiny-16384-hca-cp8r4 | tilelang@compare | 39.1 | 1.88 | 1.40 | 11.4 | 17.9 | 1.8 |
| tiny-16384-hca-cp8r4 | cudnn_flashmla@compare | 39.1 | 1.88 | 1.88 | 31.0 | 26.1 | 2.6 |
| tiny-16384-hca-cp8r4 | cute@compare | 39.1 | 1.88 | 1.40 | 29.1 | 25.7 | 2.6 |
| tiny-16384-hca-cp8r4 | cute_ws@compare | 39.1 | 3.07 | 1.40 | 66.6 | 28.7 | 2.9 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@compare | 39.1 | 3.07 | 3.07 | 37.7 | - | - |
| tiny-16384-hca-cp8r7 | tilelang@main | 35.4 | 2.01 | 1.45 | 10.6 | 16.6 | 1.7 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 35.4 | 3.40 | 3.40 | 34.3 | - | - |
| tiny-16384-hca-cp8r7 | tilelang@compare | 35.4 | 2.01 | 1.45 | 10.4 | 16.5 | 1.7 |
| tiny-16384-hca-cp8r7 | cudnn_flashmla@compare | 35.4 | 2.01 | 2.01 | 27.9 | 23.6 | 2.4 |
| tiny-16384-hca-cp8r7 | cute@compare | 35.4 | 2.01 | 1.45 | 27.2 | 23.7 | 2.4 |
| tiny-16384-hca-cp8r7 | cute_ws@compare | 35.4 | 3.40 | 1.45 | 60.1 | 26.5 | 2.7 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@compare | 35.4 | 3.40 | 3.40 | 34.3 | - | - |
| tiny-16384-sliding-cp1 | tilelang@main | 315.4 | 1.89 | 1.40 | 33.0 | 51.2 | 5.2 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 315.4 | 3.05 | 3.05 | 76.3 | - | - |
| tiny-16384-sliding-cp1 | tilelang@compare | 315.4 | 1.89 | 1.40 | 33.0 | 50.9 | 5.1 |
| tiny-16384-sliding-cp1 | cudnn_flashmla@compare | 315.4 | 1.89 | 1.89 | 71.6 | 70.5 | 7.1 |
| tiny-16384-sliding-cp1 | cute@compare | 315.4 | 1.89 | 1.40 | 45.6 | 58.3 | 5.9 |
| tiny-16384-sliding-cp1 | cute_ws@compare | 315.4 | 3.05 | 1.40 | 91.9 | 72.3 | 7.3 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@compare | 315.4 | 3.05 | 3.05 | 76.4 | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@main | 40.7 | 1.86 | 1.39 | 12.1 | 19.0 | 1.9 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@compare | 40.7 | 1.86 | 1.39 | 12.1 | 18.3 | 1.9 |
| tiny-16384-sliding-cp8r0 | cudnn_flashmla@compare | 40.7 | 1.86 | 1.86 | 32.4 | 26.0 | 2.6 |
| tiny-16384-sliding-cp8r0 | cute@compare | 40.7 | 1.86 | 1.39 | 30.5 | 26.1 | 2.6 |
| tiny-16384-sliding-cp8r0 | cute_ws@compare | 40.7 | 2.95 | 1.39 | 69.7 | 28.9 | 2.9 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@compare | 40.7 | 2.95 | 2.95 | 38.9 | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@main | 39.1 | 1.88 | 1.40 | 11.6 | 18.0 | 1.8 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 39.1 | 3.07 | 3.07 | 37.4 | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@compare | 39.1 | 1.88 | 1.40 | 11.4 | 17.9 | 1.8 |
| tiny-16384-sliding-cp8r4 | cudnn_flashmla@compare | 39.1 | 1.88 | 1.88 | 30.5 | 25.7 | 2.6 |
| tiny-16384-sliding-cp8r4 | cute@compare | 39.1 | 1.88 | 1.40 | 29.1 | 25.7 | 2.6 |
| tiny-16384-sliding-cp8r4 | cute_ws@compare | 39.1 | 3.07 | 1.40 | 65.8 | 28.5 | 2.9 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@compare | 39.1 | 3.07 | 3.07 | 37.6 | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@main | 35.4 | 2.01 | 1.45 | 10.5 | 16.5 | 1.7 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 35.4 | 3.40 | 3.40 | 34.5 | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@compare | 35.4 | 2.01 | 1.45 | 10.4 | 16.2 | 1.6 |
| tiny-16384-sliding-cp8r7 | cudnn_flashmla@compare | 35.4 | 2.01 | 2.01 | 27.9 | 23.3 | 2.4 |
| tiny-16384-sliding-cp8r7 | cute@compare | 35.4 | 2.01 | 1.45 | 26.8 | 23.5 | 2.4 |
| tiny-16384-sliding-cp8r7 | cute_ws@compare | 35.4 | 3.40 | 1.45 | 59.3 | 26.2 | 2.6 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@compare | 35.4 | 3.40 | 3.40 | 33.9 | - | - |
| single-49208-csa-cp1 | tilelang@main | 14203.1 | 1.00 | 1.00 | 247.1 | 200.9 | 20.3 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 14203.1 | 1.02 | 1.02 | 488.4 | - | - |
| single-49208-csa-cp1 | tilelang@compare | 14203.1 | 1.00 | 1.00 | 246.0 | 200.5 | 20.3 |
| single-49208-csa-cp1 | cudnn_flashmla@compare | 14203.1 | 1.00 | 1.00 | 488.9 | 359.7 | 36.3 |
| single-49208-csa-cp1 | cute@compare | 14203.1 | 1.00 | 1.00 | 264.6 | 203.8 | 20.6 |
| single-49208-csa-cp1 | cute_ws@compare | 14203.1 | 1.00 | 1.00 | 514.7 | 230.0 | 23.2 |
| single-49208-csa-cp1 | flashmla_fwd_ref@compare | 14203.1 | 1.02 | 1.02 | 475.6 | - | - |
| single-49208-csa-cp8r0 | tilelang@main | 1561.5 | 1.02 | 1.01 | 174.6 | 175.0 | 17.7 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 1561.5 | 1.16 | 1.16 | 434.8 | - | - |
| single-49208-csa-cp8r0 | tilelang@compare | 1561.5 | 1.02 | 1.01 | 174.0 | 173.9 | 17.6 |
| single-49208-csa-cp8r0 | cudnn_flashmla@compare | 1561.5 | 1.02 | 1.02 | 405.2 | 329.9 | 33.3 |
| single-49208-csa-cp8r0 | cute@compare | 1561.5 | 1.02 | 1.01 | 236.0 | 189.1 | 19.1 |
| single-49208-csa-cp8r0 | cute_ws@compare | 1561.5 | 1.04 | 1.01 | 490.7 | 216.4 | 21.9 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 1561.5 | 1.16 | 1.16 | 435.4 | - | - |
| single-49208-csa-cp8r4 | tilelang@main | 1805.9 | 1.00 | 1.00 | 188.0 | 180.1 | 18.2 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1805.9 | 1.00 | 1.00 | 468.9 | - | - |
| single-49208-csa-cp8r4 | tilelang@compare | 1805.9 | 1.00 | 1.00 | 184.6 | 178.2 | 18.0 |
| single-49208-csa-cp8r4 | cudnn_flashmla@compare | 1805.9 | 1.00 | 1.00 | 431.9 | 345.3 | 34.9 |
| single-49208-csa-cp8r4 | cute@compare | 1805.9 | 1.00 | 1.00 | 247.0 | 192.6 | 19.5 |
| single-49208-csa-cp8r4 | cute_ws@compare | 1805.9 | 1.00 | 1.00 | 519.8 | 220.0 | 22.2 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 1805.9 | 1.00 | 1.00 | 462.0 | - | - |
| single-49208-csa-cp8r7 | tilelang@main | 1805.9 | 1.00 | 1.00 | 188.0 | 174.3 | 17.6 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1805.9 | 1.00 | 1.00 | 467.1 | - | - |
| single-49208-csa-cp8r7 | tilelang@compare | 1805.9 | 1.00 | 1.00 | 187.5 | 172.6 | 17.4 |
| single-49208-csa-cp8r7 | cudnn_flashmla@compare | 1805.9 | 1.00 | 1.00 | 438.1 | 338.7 | 34.2 |
| single-49208-csa-cp8r7 | cute@compare | 1805.9 | 1.00 | 1.00 | 249.3 | 186.2 | 18.8 |
| single-49208-csa-cp8r7 | cute_ws@compare | 1805.9 | 1.00 | 1.00 | 521.8 | 211.8 | 21.4 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 1805.9 | 1.00 | 1.00 | 464.2 | - | - |
| single-49208-hca-cp1 | tilelang@main | 7213.9 | 1.10 | 1.05 | 179.3 | 169.5 | 17.1 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 7213.9 | 1.60 | 1.60 | 381.5 | - | - |
| single-49208-hca-cp1 | tilelang@compare | 7213.9 | 1.10 | 1.05 | 178.7 | 169.2 | 17.1 |
| single-49208-hca-cp1 | cudnn_flashmla@compare | 7213.9 | 1.10 | 1.10 | 378.2 | 288.4 | 29.1 |
| single-49208-hca-cp1 | cute@compare | 7213.9 | 1.10 | 1.05 | 197.5 | 173.7 | 17.6 |
| single-49208-hca-cp1 | cute_ws@compare | 7213.9 | 1.20 | 1.05 | 411.5 | 200.9 | 20.3 |
| single-49208-hca-cp1 | flashmla_fwd_ref@compare | 7213.9 | 1.60 | 1.60 | 365.7 | - | - |
| single-49208-hca-cp8r0 | tilelang@main | 423.9 | 1.26 | 1.12 | 70.8 | 99.8 | 10.1 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 423.9 | 3.41 | 3.41 | 180.6 | - | - |
| single-49208-hca-cp8r0 | tilelang@compare | 423.9 | 1.26 | 1.12 | 69.9 | 98.7 | 10.0 |
| single-49208-hca-cp8r0 | cudnn_flashmla@compare | 423.9 | 1.26 | 1.26 | 161.6 | 154.3 | 15.6 |
| single-49208-hca-cp8r0 | cute@compare | 423.9 | 1.26 | 1.12 | 112.8 | 118.2 | 11.9 |
| single-49208-hca-cp8r0 | cute_ws@compare | 423.9 | 1.69 | 1.12 | 229.5 | 138.4 | 14.0 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 423.9 | 3.41 | 3.41 | 178.5 | - | - |
| single-49208-hca-cp8r4 | tilelang@main | 970.0 | 1.11 | 1.05 | 129.2 | 147.8 | 14.9 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 970.0 | 1.49 | 1.49 | 345.9 | - | - |
| single-49208-hca-cp8r4 | tilelang@compare | 970.0 | 1.11 | 1.05 | 128.6 | 146.7 | 14.8 |
| single-49208-hca-cp8r4 | cudnn_flashmla@compare | 970.0 | 1.11 | 1.11 | 317.5 | 267.9 | 27.1 |
| single-49208-hca-cp8r4 | cute@compare | 970.0 | 1.11 | 1.05 | 185.1 | 164.7 | 16.6 |
| single-49208-hca-cp8r4 | cute_ws@compare | 970.0 | 1.12 | 1.05 | 413.6 | 193.2 | 19.5 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 970.0 | 1.49 | 1.49 | 343.8 | - | - |
| single-49208-hca-cp8r7 | tilelang@main | 1376.8 | 1.05 | 1.03 | 161.9 | 167.0 | 16.9 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 1376.8 | 1.05 | 1.05 | 416.5 | - | - |
| single-49208-hca-cp8r7 | tilelang@compare | 1376.8 | 1.05 | 1.03 | 160.3 | 167.1 | 16.9 |
| single-49208-hca-cp8r7 | cudnn_flashmla@compare | 1376.8 | 1.05 | 1.05 | 385.7 | 315.6 | 31.9 |
| single-49208-hca-cp8r7 | cute@compare | 1376.8 | 1.05 | 1.03 | 220.6 | 183.0 | 18.5 |
| single-49208-hca-cp8r7 | cute_ws@compare | 1376.8 | 1.05 | 1.03 | 479.9 | 211.1 | 21.3 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 1376.8 | 1.05 | 1.05 | 415.6 | - | - |
| single-49208-sliding-cp1 | tilelang@main | 2885.8 | 1.00 | 1.00 | 110.9 | 126.1 | 12.7 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 2885.8 | 1.00 | 1.00 | 263.8 | - | - |
| single-49208-sliding-cp1 | tilelang@compare | 2885.8 | 1.00 | 1.00 | 110.5 | 125.9 | 12.7 |
| single-49208-sliding-cp1 | cudnn_flashmla@compare | 2885.8 | 1.00 | 1.00 | 254.7 | 187.6 | 19.0 |
| single-49208-sliding-cp1 | cute@compare | 2885.8 | 1.00 | 1.00 | 129.6 | 132.3 | 13.4 |
| single-49208-sliding-cp1 | cute_ws@compare | 2885.8 | 1.00 | 1.00 | 339.8 | 161.9 | 16.4 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@compare | 2885.8 | 1.00 | 1.00 | 261.7 | - | - |
| single-49208-sliding-cp8r0 | tilelang@main | 357.5 | 1.01 | 1.00 | 64.9 | 95.5 | 9.6 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 357.5 | 1.01 | 1.01 | 188.1 | - | - |
| single-49208-sliding-cp8r0 | tilelang@compare | 357.5 | 1.01 | 1.00 | 65.0 | 94.8 | 9.6 |
| single-49208-sliding-cp8r0 | cudnn_flashmla@compare | 357.5 | 1.01 | 1.01 | 167.1 | 139.7 | 14.1 |
| single-49208-sliding-cp8r0 | cute@compare | 357.5 | 1.01 | 1.00 | 109.7 | 116.6 | 11.8 |
| single-49208-sliding-cp8r0 | cute_ws@compare | 357.5 | 1.01 | 1.00 | 277.3 | 133.3 | 13.5 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 357.5 | 1.01 | 1.01 | 186.1 | - | - |
| single-49208-sliding-cp8r4 | tilelang@main | 361.2 | 1.00 | 1.00 | 66.3 | 95.5 | 9.6 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.4 | - | - |
| single-49208-sliding-cp8r4 | tilelang@compare | 361.2 | 1.00 | 1.00 | 65.1 | 94.7 | 9.6 |
| single-49208-sliding-cp8r4 | cudnn_flashmla@compare | 361.2 | 1.00 | 1.00 | 168.1 | 141.4 | 14.3 |
| single-49208-sliding-cp8r4 | cute@compare | 361.2 | 1.00 | 1.00 | 109.6 | 116.0 | 11.7 |
| single-49208-sliding-cp8r4 | cute_ws@compare | 361.2 | 1.00 | 1.00 | 279.6 | 132.6 | 13.4 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 361.2 | 1.00 | 1.00 | 192.0 | - | - |
| single-49208-sliding-cp8r7 | tilelang@main | 361.2 | 1.00 | 1.00 | 65.6 | 95.3 | 9.6 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.1 | - | - |
| single-49208-sliding-cp8r7 | tilelang@compare | 361.2 | 1.00 | 1.00 | 65.2 | 94.7 | 9.6 |
| single-49208-sliding-cp8r7 | cudnn_flashmla@compare | 361.2 | 1.00 | 1.00 | 165.1 | 140.4 | 14.2 |
| single-49208-sliding-cp8r7 | cute@compare | 361.2 | 1.00 | 1.00 | 109.9 | 115.8 | 11.7 |
| single-49208-sliding-cp8r7 | cute_ws@compare | 361.2 | 1.00 | 1.00 | 278.5 | 131.5 | 13.3 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 361.2 | 1.00 | 1.00 | 188.5 | - | - |
| short-49208-csa-cp1 | tilelang@main | 9266.5 | 1.07 | 1.04 | 203.8 | 181.4 | 18.3 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 9266.5 | 1.56 | 1.56 | 425.2 | - | - |
| short-49208-csa-cp1 | tilelang@compare | 9266.5 | 1.07 | 1.04 | 201.7 | 181.1 | 18.3 |
| short-49208-csa-cp1 | cudnn_flashmla@compare | 9266.5 | 1.07 | 1.07 | 420.2 | 314.9 | 31.8 |
| short-49208-csa-cp1 | cute@compare | 9266.5 | 1.07 | 1.04 | 221.6 | 184.7 | 18.7 |
| short-49208-csa-cp1 | cute_ws@compare | 9266.5 | 1.13 | 1.04 | 454.9 | 211.8 | 21.4 |
| short-49208-csa-cp1 | flashmla_fwd_ref@compare | 9266.5 | 1.56 | 1.56 | 396.7 | - | - |
| short-49208-csa-cp8r0 | tilelang@main | 832.8 | 1.14 | 1.08 | 116.8 | 138.5 | 14.0 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 832.8 | 2.17 | 2.17 | 300.8 | - | - |
| short-49208-csa-cp8r0 | tilelang@compare | 832.8 | 1.14 | 1.08 | 115.9 | 137.9 | 13.9 |
| short-49208-csa-cp8r0 | cudnn_flashmla@compare | 832.8 | 1.14 | 1.14 | 275.1 | 245.6 | 24.8 |
| short-49208-csa-cp8r0 | cute@compare | 832.8 | 1.14 | 1.08 | 171.9 | 156.2 | 15.8 |
| short-49208-csa-cp8r0 | cute_ws@compare | 832.8 | 1.26 | 1.08 | 360.0 | 182.1 | 18.4 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 832.8 | 2.17 | 2.17 | 299.5 | - | - |
| short-49208-csa-cp8r4 | tilelang@main | 1351.2 | 1.04 | 1.02 | 161.8 | 167.1 | 16.9 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1351.2 | 1.34 | 1.34 | 402.3 | - | - |
| short-49208-csa-cp8r4 | tilelang@compare | 1351.2 | 1.04 | 1.02 | 159.9 | 166.6 | 16.8 |
| short-49208-csa-cp8r4 | cudnn_flashmla@compare | 1351.2 | 1.04 | 1.04 | 373.1 | 311.3 | 31.5 |
| short-49208-csa-cp8r4 | cute@compare | 1351.2 | 1.04 | 1.02 | 220.7 | 182.6 | 18.5 |
| short-49208-csa-cp8r4 | cute_ws@compare | 1351.2 | 1.08 | 1.02 | 454.8 | 209.0 | 21.1 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 1351.2 | 1.34 | 1.34 | 399.3 | - | - |
| short-49208-csa-cp8r7 | tilelang@main | 1017.3 | 1.09 | 1.05 | 134.5 | 149.4 | 15.1 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1017.3 | 1.78 | 1.78 | 338.6 | - | - |
| short-49208-csa-cp8r7 | tilelang@compare | 1017.3 | 1.09 | 1.05 | 134.1 | 149.3 | 15.1 |
| short-49208-csa-cp8r7 | cudnn_flashmla@compare | 1017.3 | 1.09 | 1.09 | 314.1 | 271.3 | 27.4 |
| short-49208-csa-cp8r7 | cute@compare | 1017.3 | 1.09 | 1.05 | 192.1 | 166.7 | 16.9 |
| short-49208-csa-cp8r7 | cute_ws@compare | 1017.3 | 1.18 | 1.05 | 395.6 | 192.8 | 19.5 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 1017.3 | 1.78 | 1.78 | 338.8 | - | - |
| short-49208-hca-cp1 | tilelang@main | 3131.2 | 1.36 | 1.16 | 105.2 | 119.6 | 12.1 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 3131.2 | 1.85 | 1.85 | 218.0 | - | - |
| short-49208-hca-cp1 | tilelang@compare | 3131.2 | 1.36 | 1.16 | 104.9 | 119.5 | 12.1 |
| short-49208-hca-cp1 | cudnn_flashmla@compare | 3131.2 | 1.36 | 1.36 | 213.2 | 179.9 | 18.2 |
| short-49208-hca-cp1 | cute@compare | 3131.2 | 1.36 | 1.16 | 121.3 | 125.0 | 12.6 |
| short-49208-hca-cp1 | cute_ws@compare | 3131.2 | 1.78 | 1.16 | 250.5 | 147.8 | 14.9 |
| short-49208-hca-cp1 | flashmla_fwd_ref@compare | 3131.2 | 1.85 | 1.85 | 217.4 | - | - |
| short-49208-hca-cp8r0 | tilelang@main | 352.9 | 1.44 | 1.20 | 59.3 | 86.7 | 8.8 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 352.9 | 2.05 | 2.05 | 155.9 | - | - |
| short-49208-hca-cp8r0 | tilelang@compare | 352.9 | 1.44 | 1.20 | 57.9 | 86.1 | 8.7 |
| short-49208-hca-cp8r0 | cudnn_flashmla@compare | 352.9 | 1.44 | 1.44 | 138.4 | 133.6 | 13.5 |
| short-49208-hca-cp8r0 | cute@compare | 352.9 | 1.44 | 1.20 | 94.1 | 103.8 | 10.5 |
| short-49208-hca-cp8r0 | cute_ws@compare | 352.9 | 1.92 | 1.20 | 197.2 | 121.1 | 12.2 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 352.9 | 2.05 | 2.05 | 151.4 | - | - |
| short-49208-hca-cp8r4 | tilelang@main | 398.0 | 1.33 | 1.14 | 65.1 | 95.1 | 9.6 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 398.0 | 1.82 | 1.82 | 170.9 | - | - |
| short-49208-hca-cp8r4 | tilelang@compare | 398.0 | 1.33 | 1.14 | 65.2 | 93.6 | 9.5 |
| short-49208-hca-cp8r4 | cudnn_flashmla@compare | 398.0 | 1.33 | 1.33 | 154.5 | 146.2 | 14.8 |
| short-49208-hca-cp8r4 | cute@compare | 398.0 | 1.33 | 1.14 | 104.1 | 112.4 | 11.4 |
| short-49208-hca-cp8r4 | cute_ws@compare | 398.0 | 1.78 | 1.14 | 219.6 | 131.4 | 13.3 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 398.0 | 1.82 | 1.82 | 170.2 | - | - |
| short-49208-hca-cp8r7 | tilelang@main | 369.7 | 1.42 | 1.18 | 61.0 | 88.7 | 9.0 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 369.7 | 1.95 | 1.95 | 159.0 | - | - |
| short-49208-hca-cp8r7 | tilelang@compare | 369.7 | 1.42 | 1.18 | 60.2 | 88.5 | 8.9 |
| short-49208-hca-cp8r7 | cudnn_flashmla@compare | 369.7 | 1.42 | 1.42 | 142.8 | 139.1 | 14.1 |
| short-49208-hca-cp8r7 | cute@compare | 369.7 | 1.42 | 1.18 | 97.3 | 106.4 | 10.8 |
| short-49208-hca-cp8r7 | cute_ws@compare | 369.7 | 1.89 | 1.18 | 203.9 | 125.2 | 12.7 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 369.7 | 1.95 | 1.95 | 157.8 | - | - |
| short-49208-sliding-cp1 | tilelang@main | 2785.1 | 1.02 | 1.01 | 107.2 | 123.1 | 12.4 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 2785.1 | 1.04 | 1.04 | 254.8 | - | - |
| short-49208-sliding-cp1 | tilelang@compare | 2785.1 | 1.02 | 1.01 | 106.2 | 122.9 | 12.4 |
| short-49208-sliding-cp1 | cudnn_flashmla@compare | 2785.1 | 1.02 | 1.02 | 245.0 | 183.0 | 18.5 |
| short-49208-sliding-cp1 | cute@compare | 2785.1 | 1.02 | 1.01 | 125.4 | 129.4 | 13.1 |
| short-49208-sliding-cp1 | cute_ws@compare | 2785.1 | 1.04 | 1.01 | 329.2 | 158.5 | 16.0 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@compare | 2785.1 | 1.04 | 1.04 | 252.4 | - | - |
| short-49208-sliding-cp8r0 | tilelang@main | 338.8 | 1.03 | 1.02 | 61.9 | 92.5 | 9.3 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 338.8 | 1.07 | 1.07 | 178.9 | - | - |
| short-49208-sliding-cp8r0 | tilelang@compare | 338.8 | 1.03 | 1.02 | 61.9 | 91.5 | 9.3 |
| short-49208-sliding-cp8r0 | cudnn_flashmla@compare | 338.8 | 1.03 | 1.03 | 158.4 | 133.8 | 13.5 |
| short-49208-sliding-cp8r0 | cute@compare | 338.8 | 1.03 | 1.02 | 105.6 | 113.0 | 11.4 |
| short-49208-sliding-cp8r0 | cute_ws@compare | 338.8 | 1.07 | 1.02 | 261.9 | 127.9 | 12.9 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 338.8 | 1.07 | 1.07 | 178.1 | - | - |
| short-49208-sliding-cp8r4 | tilelang@main | 353.7 | 1.01 | 1.01 | 64.4 | 94.6 | 9.6 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 353.7 | 1.02 | 1.02 | 185.2 | - | - |
| short-49208-sliding-cp8r4 | tilelang@compare | 353.7 | 1.01 | 1.01 | 64.4 | 94.2 | 9.5 |
| short-49208-sliding-cp8r4 | cudnn_flashmla@compare | 353.7 | 1.01 | 1.01 | 164.3 | 139.9 | 14.1 |
| short-49208-sliding-cp8r4 | cute@compare | 353.7 | 1.01 | 1.01 | 108.3 | 115.5 | 11.7 |
| short-49208-sliding-cp8r4 | cute_ws@compare | 353.7 | 1.02 | 1.01 | 275.0 | 131.8 | 13.3 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 353.7 | 1.02 | 1.02 | 184.5 | - | - |
| short-49208-sliding-cp8r7 | tilelang@main | 350.0 | 1.02 | 1.01 | 64.3 | 93.5 | 9.4 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 350.0 | 1.03 | 1.03 | 186.3 | - | - |
| short-49208-sliding-cp8r7 | tilelang@compare | 350.0 | 1.02 | 1.01 | 63.1 | 92.9 | 9.4 |
| short-49208-sliding-cp8r7 | cudnn_flashmla@compare | 350.0 | 1.02 | 1.02 | 160.7 | 137.4 | 13.9 |
| short-49208-sliding-cp8r7 | cute@compare | 350.0 | 1.02 | 1.01 | 107.0 | 114.8 | 11.6 |
| short-49208-sliding-cp8r7 | cute_ws@compare | 350.0 | 1.03 | 1.01 | 270.6 | 131.6 | 13.3 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 350.0 | 1.03 | 1.03 | 182.1 | - | - |
| heavy-49208-csa-cp1 | tilelang@main | 11959.6 | 1.03 | 1.02 | 229.1 | 193.8 | 19.6 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 11959.6 | 1.21 | 1.21 | 462.2 | - | - |
| heavy-49208-csa-cp1 | tilelang@compare | 11959.6 | 1.03 | 1.02 | 228.7 | 193.7 | 19.6 |
| heavy-49208-csa-cp1 | cudnn_flashmla@compare | 11959.6 | 1.03 | 1.03 | 461.7 | 344.9 | 34.9 |
| heavy-49208-csa-cp1 | cute@compare | 11959.6 | 1.03 | 1.02 | 246.8 | 196.6 | 19.9 |
| heavy-49208-csa-cp1 | cute_ws@compare | 11959.6 | 1.05 | 1.02 | 497.1 | 223.7 | 22.6 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@compare | 11959.6 | 1.21 | 1.21 | 442.3 | - | - |
| heavy-49208-csa-cp8r0 | tilelang@main | 663.7 | 1.22 | 1.14 | 98.5 | 122.4 | 12.4 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 663.7 | 2.72 | 2.72 | 249.6 | - | - |
| heavy-49208-csa-cp8r0 | tilelang@compare | 663.7 | 1.22 | 1.14 | 97.8 | 121.9 | 12.3 |
| heavy-49208-csa-cp8r0 | cudnn_flashmla@compare | 663.7 | 1.22 | 1.22 | 229.0 | 211.3 | 21.4 |
| heavy-49208-csa-cp8r0 | cute@compare | 663.7 | 1.22 | 1.14 | 148.2 | 139.3 | 14.1 |
| heavy-49208-csa-cp8r0 | cute_ws@compare | 663.7 | 1.41 | 1.14 | 305.5 | 163.9 | 16.6 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 663.7 | 2.72 | 2.72 | 251.2 | - | - |
| heavy-49208-csa-cp8r4 | tilelang@main | 1663.9 | 1.01 | 1.01 | 180.8 | 177.6 | 17.9 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1663.9 | 1.09 | 1.09 | 449.3 | - | - |
| heavy-49208-csa-cp8r4 | tilelang@compare | 1663.9 | 1.01 | 1.01 | 179.3 | 176.2 | 17.8 |
| heavy-49208-csa-cp8r4 | cudnn_flashmla@compare | 1663.9 | 1.01 | 1.01 | 415.1 | 337.9 | 34.1 |
| heavy-49208-csa-cp8r4 | cute@compare | 1663.9 | 1.01 | 1.01 | 242.2 | 191.3 | 19.3 |
| heavy-49208-csa-cp8r4 | cute_ws@compare | 1663.9 | 1.03 | 1.01 | 506.8 | 218.6 | 22.1 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 1663.9 | 1.09 | 1.09 | 445.1 | - | - |
| heavy-49208-csa-cp8r7 | tilelang@main | 1454.9 | 1.03 | 1.02 | 167.8 | 169.5 | 17.1 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1454.9 | 1.24 | 1.24 | 419.7 | - | - |
| heavy-49208-csa-cp8r7 | tilelang@compare | 1454.9 | 1.03 | 1.02 | 167.0 | 168.2 | 17.0 |
| heavy-49208-csa-cp8r7 | cudnn_flashmla@compare | 1454.9 | 1.03 | 1.03 | 390.0 | 320.3 | 32.4 |
| heavy-49208-csa-cp8r7 | cute@compare | 1454.9 | 1.03 | 1.02 | 227.1 | 183.3 | 18.5 |
| heavy-49208-csa-cp8r7 | cute_ws@compare | 1454.9 | 1.06 | 1.02 | 474.7 | 210.2 | 21.2 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 1454.9 | 1.24 | 1.24 | 420.0 | - | - |
| heavy-49208-hca-cp1 | tilelang@main | 4077.2 | 1.21 | 1.09 | 128.2 | 137.7 | 13.9 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4077.2 | 2.13 | 2.13 | 272.6 | - | - |
| heavy-49208-hca-cp1 | tilelang@compare | 4077.2 | 1.21 | 1.09 | 127.9 | 137.4 | 13.9 |
| heavy-49208-hca-cp1 | cudnn_flashmla@compare | 4077.2 | 1.21 | 1.21 | 263.6 | 214.5 | 21.7 |
| heavy-49208-hca-cp1 | cute@compare | 4077.2 | 1.21 | 1.09 | 146.0 | 143.1 | 14.5 |
| heavy-49208-hca-cp1 | cute_ws@compare | 4077.2 | 1.43 | 1.09 | 315.3 | 168.4 | 17.0 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@compare | 4077.2 | 2.13 | 2.13 | 266.6 | - | - |
| heavy-49208-hca-cp8r0 | tilelang@main | 318.8 | 1.45 | 1.21 | 53.5 | 81.5 | 8.2 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 318.8 | 3.40 | 3.40 | 141.7 | - | - |
| heavy-49208-hca-cp8r0 | tilelang@compare | 318.8 | 1.45 | 1.21 | 53.6 | 80.5 | 8.1 |
| heavy-49208-hca-cp8r0 | cudnn_flashmla@compare | 318.8 | 1.45 | 1.45 | 127.7 | 123.2 | 12.4 |
| heavy-49208-hca-cp8r0 | cute@compare | 318.8 | 1.45 | 1.21 | 88.0 | 98.4 | 9.9 |
| heavy-49208-hca-cp8r0 | cute_ws@compare | 318.8 | 1.94 | 1.21 | 187.1 | 114.3 | 11.6 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 318.8 | 3.40 | 3.40 | 141.2 | - | - |
| heavy-49208-hca-cp8r4 | tilelang@main | 438.1 | 1.24 | 1.11 | 71.4 | 99.7 | 10.1 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 438.1 | 2.47 | 2.47 | 185.4 | - | - |
| heavy-49208-hca-cp8r4 | tilelang@compare | 438.1 | 1.24 | 1.11 | 70.8 | 98.9 | 10.0 |
| heavy-49208-hca-cp8r4 | cudnn_flashmla@compare | 438.1 | 1.24 | 1.24 | 169.3 | 158.3 | 16.0 |
| heavy-49208-hca-cp8r4 | cute@compare | 438.1 | 1.24 | 1.11 | 113.2 | 117.9 | 11.9 |
| heavy-49208-hca-cp8r4 | cute_ws@compare | 438.1 | 1.65 | 1.11 | 237.1 | 137.7 | 13.9 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 438.1 | 2.47 | 2.47 | 186.8 | - | - |
| heavy-49208-hca-cp8r7 | tilelang@main | 507.2 | 1.25 | 1.08 | 78.6 | 108.5 | 11.0 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 507.2 | 2.14 | 2.14 | 203.5 | - | - |
| heavy-49208-hca-cp8r7 | tilelang@compare | 507.2 | 1.25 | 1.08 | 79.2 | 109.1 | 11.0 |
| heavy-49208-hca-cp8r7 | cudnn_flashmla@compare | 507.2 | 1.25 | 1.25 | 185.8 | 178.2 | 18.0 |
| heavy-49208-hca-cp8r7 | cute@compare | 507.2 | 1.25 | 1.08 | 123.3 | 128.1 | 12.9 |
| heavy-49208-hca-cp8r7 | cute_ws@compare | 507.2 | 1.60 | 1.08 | 254.2 | 151.5 | 15.3 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 507.2 | 2.14 | 2.14 | 204.6 | - | - |
| heavy-49208-sliding-cp1 | tilelang@main | 2781.4 | 1.02 | 1.01 | 106.7 | 123.2 | 12.4 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 2781.4 | 1.04 | 1.04 | 253.5 | - | - |
| heavy-49208-sliding-cp1 | tilelang@compare | 2781.4 | 1.02 | 1.01 | 106.6 | 122.9 | 12.4 |
| heavy-49208-sliding-cp1 | cudnn_flashmla@compare | 2781.4 | 1.02 | 1.02 | 244.2 | 182.6 | 18.5 |
| heavy-49208-sliding-cp1 | cute@compare | 2781.4 | 1.02 | 1.01 | 125.1 | 129.4 | 13.1 |
| heavy-49208-sliding-cp1 | cute_ws@compare | 2781.4 | 1.04 | 1.01 | 326.8 | 158.5 | 16.0 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@compare | 2781.4 | 1.04 | 1.04 | 253.4 | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@main | 309.0 | 1.08 | 1.04 | 56.2 | 86.1 | 8.7 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 309.0 | 1.17 | 1.17 | 162.8 | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@compare | 309.0 | 1.08 | 1.04 | 56.3 | 85.4 | 8.6 |
| heavy-49208-sliding-cp8r0 | cudnn_flashmla@compare | 309.0 | 1.08 | 1.08 | 144.0 | 123.6 | 12.5 |
| heavy-49208-sliding-cp8r0 | cute@compare | 309.0 | 1.08 | 1.04 | 95.9 | 106.1 | 10.7 |
| heavy-49208-sliding-cp8r0 | cute_ws@compare | 309.0 | 1.17 | 1.04 | 233.8 | 120.4 | 12.2 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 309.0 | 1.17 | 1.17 | 160.9 | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@main | 361.2 | 1.00 | 1.00 | 65.9 | 96.0 | 9.7 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.5 | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@compare | 361.2 | 1.00 | 1.00 | 64.4 | 94.9 | 9.6 |
| heavy-49208-sliding-cp8r4 | cudnn_flashmla@compare | 361.2 | 1.00 | 1.00 | 168.2 | 141.5 | 14.3 |
| heavy-49208-sliding-cp8r4 | cute@compare | 361.2 | 1.00 | 1.00 | 110.0 | 116.3 | 11.7 |
| heavy-49208-sliding-cp8r4 | cute_ws@compare | 361.2 | 1.00 | 1.00 | 279.9 | 132.3 | 13.4 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 361.2 | 1.00 | 1.00 | 189.0 | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@main | 353.7 | 1.01 | 1.01 | 64.5 | 93.9 | 9.5 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 353.7 | 1.02 | 1.02 | 186.1 | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@compare | 353.7 | 1.01 | 1.01 | 63.8 | 93.1 | 9.4 |
| heavy-49208-sliding-cp8r7 | cudnn_flashmla@compare | 353.7 | 1.01 | 1.01 | 163.6 | 138.2 | 14.0 |
| heavy-49208-sliding-cp8r7 | cute@compare | 353.7 | 1.01 | 1.01 | 108.3 | 114.7 | 11.6 |
| heavy-49208-sliding-cp8r7 | cute_ws@compare | 353.7 | 1.02 | 1.01 | 273.8 | 130.7 | 13.2 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 353.7 | 1.02 | 1.02 | 183.1 | - | - |
| tiny-49208-csa-cp1 | tilelang@main | 1202.7 | 3.49 | 2.89 | 40.9 | 49.9 | 5.0 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 1202.7 | 12.01 | 12.01 | 80.0 | - | - |
| tiny-49208-csa-cp1 | tilelang@compare | 1202.7 | 3.49 | 2.89 | 41.0 | 49.8 | 5.0 |
| tiny-49208-csa-cp1 | cudnn_flashmla@compare | 1202.7 | 3.49 | 3.49 | 79.2 | 73.9 | 7.5 |
| tiny-49208-csa-cp1 | cute@compare | 1202.7 | 3.49 | 2.89 | 46.7 | 52.2 | 5.3 |
| tiny-49208-csa-cp1 | cute_ws@compare | 1202.7 | 4.69 | 2.89 | 92.5 | 61.9 | 6.3 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@compare | 1202.7 | 12.01 | 12.01 | 80.4 | - | - |
| tiny-49208-csa-cp8r0 | tilelang@main | 150.2 | 3.50 | 2.89 | 25.2 | 38.8 | 3.9 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 150.2 | 12.03 | 12.03 | 65.2 | - | - |
| tiny-49208-csa-cp8r0 | tilelang@compare | 150.2 | 3.50 | 2.89 | 25.3 | 38.5 | 3.9 |
| tiny-49208-csa-cp8r0 | cudnn_flashmla@compare | 150.2 | 3.50 | 3.50 | 58.3 | 60.6 | 6.1 |
| tiny-49208-csa-cp8r0 | cute@compare | 150.2 | 3.50 | 2.89 | 41.1 | 46.7 | 4.7 |
| tiny-49208-csa-cp8r0 | cute_ws@compare | 150.2 | 4.70 | 2.89 | 80.7 | 55.8 | 5.6 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@compare | 150.2 | 12.03 | 12.03 | 65.4 | - | - |
| tiny-49208-csa-cp8r4 | tilelang@main | 156.7 | 3.35 | 2.78 | 26.6 | 40.1 | 4.1 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 156.7 | 11.53 | 11.53 | 68.2 | - | - |
| tiny-49208-csa-cp8r4 | tilelang@compare | 156.7 | 3.35 | 2.78 | 26.5 | 39.9 | 4.0 |
| tiny-49208-csa-cp8r4 | cudnn_flashmla@compare | 156.7 | 3.35 | 3.35 | 61.3 | 62.6 | 6.3 |
| tiny-49208-csa-cp8r4 | cute@compare | 156.7 | 3.35 | 2.78 | 42.8 | 48.2 | 4.9 |
| tiny-49208-csa-cp8r4 | cute_ws@compare | 156.7 | 4.50 | 2.78 | 83.8 | 57.8 | 5.8 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@compare | 156.7 | 11.53 | 11.53 | 67.9 | - | - |
| tiny-49208-csa-cp8r7 | tilelang@main | 144.3 | 3.63 | 3.01 | 24.4 | 36.9 | 3.7 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 144.3 | 12.51 | 12.51 | 62.5 | - | - |
| tiny-49208-csa-cp8r7 | tilelang@compare | 144.3 | 3.63 | 3.01 | 24.4 | 36.8 | 3.7 |
| tiny-49208-csa-cp8r7 | cudnn_flashmla@compare | 144.3 | 3.63 | 3.63 | 56.8 | 57.6 | 5.8 |
| tiny-49208-csa-cp8r7 | cute@compare | 144.3 | 3.63 | 3.01 | 39.4 | 44.8 | 4.5 |
| tiny-49208-csa-cp8r7 | cute_ws@compare | 144.3 | 4.89 | 3.01 | 77.6 | 53.2 | 5.4 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@compare | 144.3 | 12.51 | 12.51 | 62.8 | - | - |
| tiny-49208-hca-cp1 | tilelang@main | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 969.0 | 2.98 | 2.98 | 87.5 | - | - |
| tiny-49208-hca-cp1 | tilelang@compare | 969.0 | 1.86 | 1.39 | 41.3 | 57.8 | 5.8 |
| tiny-49208-hca-cp1 | cudnn_flashmla@compare | 969.0 | 1.86 | 1.86 | 84.3 | 75.7 | 7.6 |
| tiny-49208-hca-cp1 | cute@compare | 969.0 | 1.86 | 1.39 | 49.2 | 61.9 | 6.3 |
| tiny-49208-hca-cp1 | cute_ws@compare | 969.0 | 2.98 | 1.39 | 101.0 | 76.1 | 7.7 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@compare | 969.0 | 2.98 | 2.98 | 87.3 | - | - |
| tiny-49208-hca-cp8r0 | tilelang@main | 121.0 | 1.86 | 1.39 | 23.4 | 41.0 | 4.1 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 121.0 | 2.99 | 2.99 | 63.8 | - | - |
| tiny-49208-hca-cp8r0 | tilelang@compare | 121.0 | 1.86 | 1.39 | 22.3 | 40.8 | 4.1 |
| tiny-49208-hca-cp8r0 | cudnn_flashmla@compare | 121.0 | 1.86 | 1.86 | 54.2 | 54.8 | 5.5 |
| tiny-49208-hca-cp8r0 | cute@compare | 121.0 | 1.86 | 1.39 | 41.0 | 53.0 | 5.4 |
| tiny-49208-hca-cp8r0 | cute_ws@compare | 121.0 | 2.99 | 1.39 | 85.1 | 60.9 | 6.2 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@compare | 121.0 | 2.99 | 2.99 | 61.6 | - | - |
| tiny-49208-hca-cp8r4 | tilelang@main | 126.2 | 1.83 | 1.38 | 24.3 | 42.3 | 4.3 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 126.2 | 2.86 | 2.86 | 66.3 | - | - |
| tiny-49208-hca-cp8r4 | tilelang@compare | 126.2 | 1.83 | 1.38 | 24.1 | 41.9 | 4.2 |
| tiny-49208-hca-cp8r4 | cudnn_flashmla@compare | 126.2 | 1.83 | 1.83 | 58.3 | 57.1 | 5.8 |
| tiny-49208-hca-cp8r4 | cute@compare | 126.2 | 1.83 | 1.38 | 42.8 | 54.7 | 5.5 |
| tiny-49208-hca-cp8r4 | cute_ws@compare | 126.2 | 2.86 | 1.38 | 89.8 | 62.6 | 6.3 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@compare | 126.2 | 2.86 | 2.86 | 66.2 | - | - |
| tiny-49208-hca-cp8r7 | tilelang@main | 116.3 | 1.90 | 1.41 | 22.6 | 39.4 | 4.0 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 116.3 | 3.11 | 3.11 | 61.5 | - | - |
| tiny-49208-hca-cp8r7 | tilelang@compare | 116.3 | 1.90 | 1.41 | 22.7 | 39.3 | 4.0 |
| tiny-49208-hca-cp8r7 | cudnn_flashmla@compare | 116.3 | 1.90 | 1.90 | 54.7 | 53.2 | 5.4 |
| tiny-49208-hca-cp8r7 | cute@compare | 116.3 | 1.90 | 1.41 | 39.8 | 51.6 | 5.2 |
| tiny-49208-hca-cp8r7 | cute_ws@compare | 116.3 | 3.11 | 1.41 | 82.6 | 59.1 | 6.0 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@compare | 116.3 | 3.11 | 3.11 | 61.4 | - | - |
| tiny-49208-sliding-cp1 | tilelang@main | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 969.0 | 2.98 | 2.98 | 87.2 | - | - |
| tiny-49208-sliding-cp1 | tilelang@compare | 969.0 | 1.86 | 1.39 | 41.2 | 57.6 | 5.8 |
| tiny-49208-sliding-cp1 | cudnn_flashmla@compare | 969.0 | 1.86 | 1.86 | 84.3 | 75.8 | 7.7 |
| tiny-49208-sliding-cp1 | cute@compare | 969.0 | 1.86 | 1.39 | 49.2 | 61.8 | 6.3 |
| tiny-49208-sliding-cp1 | cute_ws@compare | 969.0 | 2.98 | 1.39 | 100.9 | 76.1 | 7.7 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@compare | 969.0 | 2.98 | 2.98 | 87.3 | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@main | 121.0 | 1.86 | 1.39 | 23.6 | 41.1 | 4.1 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 121.0 | 2.99 | 2.99 | 64.0 | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@compare | 121.0 | 1.86 | 1.39 | 23.5 | 40.0 | 4.0 |
| tiny-49208-sliding-cp8r0 | cudnn_flashmla@compare | 121.0 | 1.86 | 1.86 | 56.7 | 54.3 | 5.5 |
| tiny-49208-sliding-cp8r0 | cute@compare | 121.0 | 1.86 | 1.39 | 41.4 | 52.9 | 5.3 |
| tiny-49208-sliding-cp8r0 | cute_ws@compare | 121.0 | 2.99 | 1.39 | 85.6 | 59.7 | 6.0 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@compare | 121.0 | 2.99 | 2.99 | 63.4 | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@main | 126.2 | 1.83 | 1.38 | 24.3 | 42.4 | 4.3 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 126.2 | 2.86 | 2.86 | 66.5 | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@compare | 126.2 | 1.83 | 1.38 | 23.6 | 41.9 | 4.2 |
| tiny-49208-sliding-cp8r4 | cudnn_flashmla@compare | 126.2 | 1.83 | 1.83 | 58.4 | 56.6 | 5.7 |
| tiny-49208-sliding-cp8r4 | cute@compare | 126.2 | 1.83 | 1.38 | 42.3 | 54.5 | 5.5 |
| tiny-49208-sliding-cp8r4 | cute_ws@compare | 126.2 | 2.86 | 1.38 | 89.4 | 62.6 | 6.3 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@compare | 126.2 | 2.86 | 2.86 | 66.1 | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@main | 116.3 | 1.90 | 1.41 | 22.7 | 39.4 | 4.0 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 116.3 | 3.11 | 3.11 | 61.3 | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@compare | 116.3 | 1.90 | 1.41 | 22.2 | 39.2 | 4.0 |
| tiny-49208-sliding-cp8r7 | cudnn_flashmla@compare | 116.3 | 1.90 | 1.90 | 54.5 | 53.4 | 5.4 |
| tiny-49208-sliding-cp8r7 | cute@compare | 116.3 | 1.90 | 1.41 | 39.7 | 51.4 | 5.2 |
| tiny-49208-sliding-cp8r7 | cute_ws@compare | 116.3 | 3.11 | 1.41 | 84.2 | 59.0 | 6.0 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@compare | 116.3 | 3.11 | 3.11 | 60.2 | - | - |
| single-65536-csa-cp1 | tilelang@main | 18997.0 | 1.00 | 1.00 | 250.2 | 199.0 | 20.1 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 18997.0 | 1.01 | 1.01 | 476.7 | - | - |
| single-65536-csa-cp1 | tilelang@compare | 18997.0 | 1.00 | 1.00 | 249.9 | 198.9 | 20.1 |
| single-65536-csa-cp1 | cudnn_flashmla@compare | 18997.0 | 1.00 | 1.00 | 473.4 | 356.2 | 36.0 |
| single-65536-csa-cp1 | cute@compare | 18997.0 | 1.00 | 1.00 | 267.8 | 202.1 | 20.4 |
| single-65536-csa-cp1 | cute_ws@compare | 18997.0 | 1.00 | 1.00 | 495.9 | 226.2 | 22.9 |
| single-65536-csa-cp1 | flashmla_fwd_ref@compare | 18997.0 | 1.01 | 1.01 | 473.0 | - | - |
| single-65536-csa-cp8r0 | tilelang@main | 2160.7 | 1.02 | 1.01 | 189.1 | 181.1 | 18.3 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 2160.7 | 1.11 | 1.11 | 465.9 | - | - |
| single-65536-csa-cp8r0 | tilelang@compare | 2160.7 | 1.02 | 1.01 | 190.9 | 178.9 | 18.1 |
| single-65536-csa-cp8r0 | cudnn_flashmla@compare | 2160.7 | 1.02 | 1.02 | 441.0 | 343.3 | 34.7 |
| single-65536-csa-cp8r0 | cute@compare | 2160.7 | 1.02 | 1.01 | 242.8 | 190.9 | 19.3 |
| single-65536-csa-cp8r0 | cute_ws@compare | 2160.7 | 1.03 | 1.01 | 508.6 | 217.7 | 22.0 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 2160.7 | 1.11 | 1.11 | 463.5 | - | - |
| single-65536-csa-cp8r4 | tilelang@main | 2405.2 | 1.00 | 1.00 | 200.4 | 183.0 | 18.5 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 487.2 | - | - |
| single-65536-csa-cp8r4 | tilelang@compare | 2405.2 | 1.00 | 1.00 | 201.9 | 181.6 | 18.4 |
| single-65536-csa-cp8r4 | cudnn_flashmla@compare | 2405.2 | 1.00 | 1.00 | 461.5 | 351.7 | 35.5 |
| single-65536-csa-cp8r4 | cute@compare | 2405.2 | 1.00 | 1.00 | 254.3 | 192.3 | 19.4 |
| single-65536-csa-cp8r4 | cute_ws@compare | 2405.2 | 1.00 | 1.00 | 531.2 | 217.7 | 22.0 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 2405.2 | 1.00 | 1.00 | 484.0 | - | - |
| single-65536-csa-cp8r7 | tilelang@main | 2405.2 | 1.00 | 1.00 | 203.1 | 171.9 | 17.4 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 487.3 | - | - |
| single-65536-csa-cp8r7 | tilelang@compare | 2405.2 | 1.00 | 1.00 | 200.3 | 169.8 | 17.2 |
| single-65536-csa-cp8r7 | cudnn_flashmla@compare | 2405.2 | 1.00 | 1.00 | 455.9 | 333.6 | 33.7 |
| single-65536-csa-cp8r7 | cute@compare | 2405.2 | 1.00 | 1.00 | 254.9 | 179.5 | 18.1 |
| single-65536-csa-cp8r7 | cute_ws@compare | 2405.2 | 1.00 | 1.00 | 526.2 | 201.9 | 20.4 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 2405.2 | 1.00 | 1.00 | 480.0 | - | - |
| single-65536-hca-cp1 | tilelang@main | 11526.3 | 1.08 | 1.04 | 199.6 | 179.2 | 18.1 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 11526.3 | 1.67 | 1.67 | 401.8 | - | - |
| single-65536-hca-cp1 | tilelang@compare | 11526.3 | 1.08 | 1.04 | 199.0 | 178.7 | 18.1 |
| single-65536-hca-cp1 | cudnn_flashmla@compare | 11526.3 | 1.08 | 1.08 | 396.9 | 308.0 | 31.1 |
| single-65536-hca-cp1 | cute@compare | 11526.3 | 1.08 | 1.04 | 215.2 | 182.5 | 18.4 |
| single-65536-hca-cp1 | cute_ws@compare | 11526.3 | 1.17 | 1.04 | 430.4 | 209.3 | 21.2 |
| single-65536-hca-cp1 | flashmla_fwd_ref@compare | 11526.3 | 1.67 | 1.67 | 390.7 | - | - |
| single-65536-hca-cp8r0 | tilelang@main | 595.7 | 1.20 | 1.10 | 82.1 | 109.6 | 11.1 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 595.7 | 4.04 | 4.04 | 203.8 | - | - |
| single-65536-hca-cp8r0 | tilelang@compare | 595.7 | 1.20 | 1.10 | 82.3 | 109.3 | 11.0 |
| single-65536-hca-cp8r0 | cudnn_flashmla@compare | 595.7 | 1.20 | 1.20 | 186.1 | 173.6 | 17.5 |
| single-65536-hca-cp8r0 | cute@compare | 595.7 | 1.20 | 1.10 | 123.6 | 125.8 | 12.7 |
| single-65536-hca-cp8r0 | cute_ws@compare | 595.7 | 1.60 | 1.10 | 252.5 | 149.5 | 15.1 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 595.7 | 4.04 | 4.04 | 203.7 | - | - |
| single-65536-hca-cp8r4 | tilelang@main | 1561.5 | 1.08 | 1.04 | 159.5 | 163.0 | 16.5 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1561.5 | 1.54 | 1.54 | 376.5 | - | - |
| single-65536-hca-cp8r4 | tilelang@compare | 1561.5 | 1.08 | 1.04 | 158.3 | 162.0 | 16.4 |
| single-65536-hca-cp8r4 | cudnn_flashmla@compare | 1561.5 | 1.08 | 1.08 | 350.3 | 295.9 | 29.9 |
| single-65536-hca-cp8r4 | cute@compare | 1561.5 | 1.08 | 1.04 | 209.7 | 176.5 | 17.8 |
| single-65536-hca-cp8r4 | cute_ws@compare | 1561.5 | 1.23 | 1.04 | 418.4 | 202.3 | 20.4 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 1561.5 | 1.54 | 1.54 | 371.2 | - | - |
| single-65536-hca-cp8r7 | tilelang@main | 2283.1 | 1.05 | 1.03 | 192.4 | 179.9 | 18.2 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 2283.1 | 1.05 | 1.05 | 471.2 | - | - |
| single-65536-hca-cp8r7 | tilelang@compare | 2283.1 | 1.05 | 1.03 | 191.2 | 178.5 | 18.0 |
| single-65536-hca-cp8r7 | cudnn_flashmla@compare | 2283.1 | 1.05 | 1.05 | 443.4 | 347.6 | 35.1 |
| single-65536-hca-cp8r7 | cute@compare | 2283.1 | 1.05 | 1.03 | 241.3 | 190.0 | 19.2 |
| single-65536-hca-cp8r7 | cute_ws@compare | 2283.1 | 1.05 | 1.03 | 513.3 | 216.6 | 21.9 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 2283.1 | 1.05 | 1.05 | 468.1 | - | - |
| single-65536-sliding-cp1 | tilelang@main | 3844.6 | 1.00 | 1.00 | 113.8 | 127.5 | 12.9 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 3844.6 | 1.00 | 1.00 | 266.5 | - | - |
| single-65536-sliding-cp1 | tilelang@compare | 3844.6 | 1.00 | 1.00 | 113.1 | 127.5 | 12.9 |
| single-65536-sliding-cp1 | cudnn_flashmla@compare | 3844.6 | 1.00 | 1.00 | 259.5 | 189.5 | 19.2 |
| single-65536-sliding-cp1 | cute@compare | 3844.6 | 1.00 | 1.00 | 130.4 | 133.1 | 13.5 |
| single-65536-sliding-cp1 | cute_ws@compare | 3844.6 | 1.00 | 1.00 | 340.8 | 162.6 | 16.4 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@compare | 3844.6 | 1.00 | 1.00 | 259.2 | - | - |
| single-65536-sliding-cp8r0 | tilelang@main | 477.3 | 1.00 | 1.00 | 72.9 | 102.5 | 10.4 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 477.3 | 1.01 | 1.01 | 206.2 | - | - |
| single-65536-sliding-cp8r0 | tilelang@compare | 477.3 | 1.00 | 1.00 | 72.9 | 101.2 | 10.2 |
| single-65536-sliding-cp8r0 | cudnn_flashmla@compare | 477.3 | 1.00 | 1.00 | 186.4 | 154.9 | 15.7 |
| single-65536-sliding-cp8r0 | cute@compare | 477.3 | 1.00 | 1.00 | 115.0 | 119.6 | 12.1 |
| single-65536-sliding-cp8r0 | cute_ws@compare | 477.3 | 1.01 | 1.00 | 295.0 | 147.2 | 14.9 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 477.3 | 1.01 | 1.01 | 205.3 | - | - |
| single-65536-sliding-cp8r4 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.6 | 102.5 | 10.4 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 208.0 | - | - |
| single-65536-sliding-cp8r4 | tilelang@compare | 481.0 | 1.00 | 1.00 | 73.9 | 101.3 | 10.2 |
| single-65536-sliding-cp8r4 | cudnn_flashmla@compare | 481.0 | 1.00 | 1.00 | 187.1 | 156.8 | 15.8 |
| single-65536-sliding-cp8r4 | cute@compare | 481.0 | 1.00 | 1.00 | 115.4 | 119.4 | 12.1 |
| single-65536-sliding-cp8r4 | cute_ws@compare | 481.0 | 1.00 | 1.00 | 297.3 | 146.0 | 14.8 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 481.0 | 1.00 | 1.00 | 208.2 | - | - |
| single-65536-sliding-cp8r7 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.2 | 102.0 | 10.3 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 207.5 | - | - |
| single-65536-sliding-cp8r7 | tilelang@compare | 481.0 | 1.00 | 1.00 | 73.9 | 101.4 | 10.2 |
| single-65536-sliding-cp8r7 | cudnn_flashmla@compare | 481.0 | 1.00 | 1.00 | 187.6 | 157.6 | 15.9 |
| single-65536-sliding-cp8r7 | cute@compare | 481.0 | 1.00 | 1.00 | 115.3 | 119.7 | 12.1 |
| single-65536-sliding-cp8r7 | cute_ws@compare | 481.0 | 1.00 | 1.00 | 295.8 | 146.9 | 14.8 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 481.0 | 1.00 | 1.00 | 207.9 | - | - |
| short-65536-csa-cp1 | tilelang@main | 11209.9 | 1.08 | 1.05 | 197.2 | 177.5 | 17.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 11209.9 | 1.72 | 1.72 | 393.2 | - | - |
| short-65536-csa-cp1 | tilelang@compare | 11209.9 | 1.08 | 1.05 | 195.9 | 177.3 | 17.9 |
| short-65536-csa-cp1 | cudnn_flashmla@compare | 11209.9 | 1.08 | 1.08 | 393.4 | 305.1 | 30.8 |
| short-65536-csa-cp1 | cute@compare | 11209.9 | 1.08 | 1.05 | 213.6 | 180.6 | 18.3 |
| short-65536-csa-cp1 | cute_ws@compare | 11209.9 | 1.16 | 1.05 | 418.0 | 207.2 | 20.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref@compare | 11209.9 | 1.72 | 1.72 | 384.7 | - | - |
| short-65536-csa-cp8r0 | tilelang@main | 885.0 | 1.18 | 1.11 | 110.6 | 130.5 | 13.2 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 885.0 | 2.72 | 2.72 | 278.5 | - | - |
| short-65536-csa-cp8r0 | tilelang@compare | 885.0 | 1.18 | 1.11 | 110.1 | 129.6 | 13.1 |
| short-65536-csa-cp8r0 | cudnn_flashmla@compare | 885.0 | 1.18 | 1.18 | 255.9 | 222.9 | 22.5 |
| short-65536-csa-cp8r0 | cute@compare | 885.0 | 1.18 | 1.11 | 155.9 | 145.2 | 14.7 |
| short-65536-csa-cp8r0 | cute_ws@compare | 885.0 | 1.34 | 1.11 | 327.4 | 170.3 | 17.2 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 885.0 | 2.72 | 2.72 | 277.3 | - | - |
| short-65536-csa-cp8r4 | tilelang@main | 1689.8 | 1.05 | 1.03 | 168.7 | 167.7 | 16.9 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1689.8 | 1.42 | 1.42 | 412.8 | - | - |
| short-65536-csa-cp8r4 | tilelang@compare | 1689.8 | 1.05 | 1.03 | 168.1 | 166.1 | 16.8 |
| short-65536-csa-cp8r4 | cudnn_flashmla@compare | 1689.8 | 1.05 | 1.05 | 385.4 | 311.2 | 31.5 |
| short-65536-csa-cp8r4 | cute@compare | 1689.8 | 1.05 | 1.03 | 218.9 | 180.8 | 18.3 |
| short-65536-csa-cp8r4 | cute_ws@compare | 1689.8 | 1.10 | 1.03 | 459.5 | 207.3 | 20.9 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 1689.8 | 1.42 | 1.42 | 409.1 | - | - |
| short-65536-csa-cp8r7 | tilelang@main | 1414.7 | 1.08 | 1.05 | 151.7 | 158.2 | 16.0 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1414.7 | 1.70 | 1.70 | 371.4 | - | - |
| short-65536-csa-cp8r7 | tilelang@compare | 1414.7 | 1.08 | 1.05 | 149.4 | 157.9 | 16.0 |
| short-65536-csa-cp8r7 | cudnn_flashmla@compare | 1414.7 | 1.08 | 1.08 | 344.9 | 287.4 | 29.0 |
| short-65536-csa-cp8r7 | cute@compare | 1414.7 | 1.08 | 1.05 | 201.4 | 172.7 | 17.5 |
| short-65536-csa-cp8r7 | cute_ws@compare | 1414.7 | 1.16 | 1.05 | 420.1 | 199.2 | 20.1 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 1414.7 | 1.70 | 1.70 | 367.9 | - | - |
| short-65536-hca-cp1 | tilelang@main | 3930.2 | 1.40 | 1.17 | 102.8 | 117.2 | 11.8 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 3930.2 | 1.96 | 1.96 | 208.8 | - | - |
| short-65536-hca-cp1 | tilelang@compare | 3930.2 | 1.40 | 1.17 | 102.5 | 117.0 | 11.8 |
| short-65536-hca-cp1 | cudnn_flashmla@compare | 3930.2 | 1.40 | 1.40 | 204.7 | 173.3 | 17.5 |
| short-65536-hca-cp1 | cute@compare | 3930.2 | 1.40 | 1.17 | 116.0 | 121.8 | 12.3 |
| short-65536-hca-cp1 | cute_ws@compare | 3930.2 | 1.87 | 1.17 | 237.0 | 144.0 | 14.6 |
| short-65536-hca-cp1 | flashmla_fwd_ref@compare | 3930.2 | 1.96 | 1.96 | 207.9 | - | - |
| short-65536-hca-cp8r0 | tilelang@main | 455.7 | 1.46 | 1.22 | 64.5 | 90.7 | 9.2 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 455.7 | 2.11 | 2.11 | 160.5 | - | - |
| short-65536-hca-cp8r0 | tilelang@compare | 455.7 | 1.46 | 1.22 | 64.1 | 90.2 | 9.1 |
| short-65536-hca-cp8r0 | cudnn_flashmla@compare | 455.7 | 1.46 | 1.46 | 149.4 | 143.1 | 14.5 |
| short-65536-hca-cp8r0 | cute@compare | 455.7 | 1.46 | 1.22 | 96.8 | 104.9 | 10.6 |
| short-65536-hca-cp8r0 | cute_ws@compare | 455.7 | 1.95 | 1.22 | 202.7 | 127.5 | 12.9 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 455.7 | 2.11 | 2.11 | 162.0 | - | - |
| short-65536-hca-cp8r4 | tilelang@main | 514.5 | 1.37 | 1.14 | 71.6 | 99.0 | 10.0 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 514.5 | 1.87 | 1.87 | 176.6 | - | - |
| short-65536-hca-cp8r4 | tilelang@compare | 514.5 | 1.37 | 1.14 | 70.7 | 97.7 | 9.9 |
| short-65536-hca-cp8r4 | cudnn_flashmla@compare | 514.5 | 1.37 | 1.37 | 163.1 | 154.8 | 15.6 |
| short-65536-hca-cp8r4 | cute@compare | 514.5 | 1.37 | 1.14 | 106.2 | 113.8 | 11.5 |
| short-65536-hca-cp8r4 | cute_ws@compare | 514.5 | 1.83 | 1.14 | 220.9 | 137.8 | 13.9 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 514.5 | 1.87 | 1.87 | 175.4 | - | - |
| short-65536-hca-cp8r7 | tilelang@main | 491.0 | 1.40 | 1.17 | 68.7 | 95.0 | 9.6 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 491.0 | 1.96 | 1.96 | 171.1 | - | - |
| short-65536-hca-cp8r7 | tilelang@compare | 491.0 | 1.40 | 1.17 | 68.4 | 95.0 | 9.6 |
| short-65536-hca-cp8r7 | cudnn_flashmla@compare | 491.0 | 1.40 | 1.40 | 158.2 | 150.6 | 15.2 |
| short-65536-hca-cp8r7 | cute@compare | 491.0 | 1.40 | 1.17 | 102.4 | 110.3 | 11.1 |
| short-65536-hca-cp8r7 | cute_ws@compare | 491.0 | 1.87 | 1.17 | 213.8 | 133.5 | 13.5 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 491.0 | 1.96 | 1.96 | 170.9 | - | - |
| short-65536-sliding-cp1 | tilelang@main | 3673.0 | 1.02 | 1.01 | 109.2 | 123.9 | 12.5 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 3673.0 | 1.05 | 1.05 | 255.6 | - | - |
| short-65536-sliding-cp1 | tilelang@compare | 3673.0 | 1.02 | 1.01 | 108.7 | 123.8 | 12.5 |
| short-65536-sliding-cp1 | cudnn_flashmla@compare | 3673.0 | 1.02 | 1.02 | 248.6 | 183.6 | 18.6 |
| short-65536-sliding-cp1 | cute@compare | 3673.0 | 1.02 | 1.01 | 125.4 | 129.4 | 13.1 |
| short-65536-sliding-cp1 | cute_ws@compare | 3673.0 | 1.05 | 1.01 | 324.8 | 158.4 | 16.0 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@compare | 3673.0 | 1.05 | 1.05 | 254.6 | - | - |
| short-65536-sliding-cp8r0 | tilelang@main | 443.7 | 1.04 | 1.02 | 68.1 | 97.4 | 9.8 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 443.7 | 1.08 | 1.08 | 190.4 | - | - |
| short-65536-sliding-cp8r0 | tilelang@compare | 443.7 | 1.04 | 1.02 | 68.5 | 96.8 | 9.8 |
| short-65536-sliding-cp8r0 | cudnn_flashmla@compare | 443.7 | 1.04 | 1.04 | 172.4 | 148.4 | 15.0 |
| short-65536-sliding-cp8r0 | cute@compare | 443.7 | 1.04 | 1.02 | 107.9 | 114.8 | 11.6 |
| short-65536-sliding-cp8r0 | cute_ws@compare | 443.7 | 1.08 | 1.02 | 271.9 | 141.1 | 14.3 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 443.7 | 1.08 | 1.08 | 190.7 | - | - |
| short-65536-sliding-cp8r4 | tilelang@main | 469.9 | 1.01 | 1.01 | 73.0 | 101.6 | 10.3 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 469.9 | 1.02 | 1.02 | 203.3 | - | - |
| short-65536-sliding-cp8r4 | tilelang@compare | 469.9 | 1.01 | 1.01 | 73.0 | 100.4 | 10.1 |
| short-65536-sliding-cp8r4 | cudnn_flashmla@compare | 469.9 | 1.01 | 1.01 | 182.0 | 155.0 | 15.7 |
| short-65536-sliding-cp8r4 | cute@compare | 469.9 | 1.01 | 1.01 | 113.2 | 119.2 | 12.0 |
| short-65536-sliding-cp8r4 | cute_ws@compare | 469.9 | 1.02 | 1.01 | 289.6 | 146.6 | 14.8 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 469.9 | 1.02 | 1.02 | 201.5 | - | - |
| short-65536-sliding-cp8r7 | tilelang@main | 458.7 | 1.02 | 1.01 | 71.2 | 99.0 | 10.0 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 458.7 | 1.05 | 1.05 | 197.9 | - | - |
| short-65536-sliding-cp8r7 | tilelang@compare | 458.7 | 1.02 | 1.01 | 71.5 | 98.9 | 10.0 |
| short-65536-sliding-cp8r7 | cudnn_flashmla@compare | 458.7 | 1.02 | 1.02 | 179.5 | 152.7 | 15.4 |
| short-65536-sliding-cp8r7 | cute@compare | 458.7 | 1.02 | 1.01 | 110.7 | 117.2 | 11.8 |
| short-65536-sliding-cp8r7 | cute_ws@compare | 458.7 | 1.05 | 1.01 | 281.7 | 144.1 | 14.6 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 458.7 | 1.05 | 1.05 | 198.1 | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 12785.2 | 1.07 | 1.04 | 210.3 | 183.4 | 18.5 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 12785.2 | 1.50 | 1.50 | 414.1 | - | - |
| heavy-65536-csa-cp1 | tilelang@compare | 12785.2 | 1.07 | 1.04 | 209.3 | 183.1 | 18.5 |
| heavy-65536-csa-cp1 | cudnn_flashmla@compare | 12785.2 | 1.07 | 1.07 | 414.6 | 318.4 | 32.2 |
| heavy-65536-csa-cp1 | cute@compare | 12785.2 | 1.07 | 1.04 | 225.8 | 186.8 | 18.9 |
| heavy-65536-csa-cp1 | cute_ws@compare | 12785.2 | 1.12 | 1.04 | 436.2 | 212.7 | 21.5 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@compare | 12785.2 | 1.50 | 1.50 | 404.1 | - | - |
| heavy-65536-csa-cp8r0 | tilelang@main | 771.9 | 1.27 | 1.18 | 98.5 | 120.2 | 12.2 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 771.9 | 3.12 | 3.12 | 244.8 | - | - |
| heavy-65536-csa-cp8r0 | tilelang@compare | 771.9 | 1.27 | 1.18 | 98.1 | 119.2 | 12.0 |
| heavy-65536-csa-cp8r0 | cudnn_flashmla@compare | 771.9 | 1.27 | 1.27 | 227.1 | 202.1 | 20.4 |
| heavy-65536-csa-cp8r0 | cute@compare | 771.9 | 1.27 | 1.18 | 139.7 | 134.3 | 13.6 |
| heavy-65536-csa-cp8r0 | cute_ws@compare | 771.9 | 1.49 | 1.18 | 288.9 | 157.6 | 15.9 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 771.9 | 3.12 | 3.12 | 244.3 | - | - |
| heavy-65536-csa-cp8r4 | tilelang@main | 2405.2 | 1.00 | 1.00 | 202.0 | 185.4 | 18.7 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 490.2 | - | - |
| heavy-65536-csa-cp8r4 | tilelang@compare | 2405.2 | 1.00 | 1.00 | 202.4 | 183.7 | 18.6 |
| heavy-65536-csa-cp8r4 | cudnn_flashmla@compare | 2405.2 | 1.00 | 1.00 | 460.1 | 354.8 | 35.9 |
| heavy-65536-csa-cp8r4 | cute@compare | 2405.2 | 1.00 | 1.00 | 247.9 | 194.8 | 19.7 |
| heavy-65536-csa-cp8r4 | cute_ws@compare | 2405.2 | 1.00 | 1.00 | 529.8 | 221.1 | 22.3 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 2405.2 | 1.00 | 1.00 | 481.8 | - | - |
| heavy-65536-csa-cp8r7 | tilelang@main | 1250.3 | 1.12 | 1.08 | 140.1 | 150.7 | 15.2 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1250.3 | 1.92 | 1.92 | 341.0 | - | - |
| heavy-65536-csa-cp8r7 | tilelang@compare | 1250.3 | 1.12 | 1.08 | 137.7 | 149.8 | 15.1 |
| heavy-65536-csa-cp8r7 | cudnn_flashmla@compare | 1250.3 | 1.12 | 1.12 | 315.3 | 267.2 | 27.0 |
| heavy-65536-csa-cp8r7 | cute@compare | 1250.3 | 1.12 | 1.08 | 187.5 | 164.4 | 16.6 |
| heavy-65536-csa-cp8r7 | cute_ws@compare | 1250.3 | 1.23 | 1.08 | 389.3 | 190.1 | 19.2 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 1250.3 | 1.92 | 1.92 | 336.4 | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 5141.1 | 1.25 | 1.11 | 125.4 | 134.3 | 13.6 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5141.1 | 2.25 | 2.25 | 258.5 | - | - |
| heavy-65536-hca-cp1 | tilelang@compare | 5141.1 | 1.25 | 1.11 | 124.8 | 134.1 | 13.6 |
| heavy-65536-hca-cp1 | cudnn_flashmla@compare | 5141.1 | 1.25 | 1.25 | 250.7 | 206.6 | 20.9 |
| heavy-65536-hca-cp1 | cute@compare | 5141.1 | 1.25 | 1.11 | 139.2 | 138.2 | 14.0 |
| heavy-65536-hca-cp1 | cute_ws@compare | 5141.1 | 1.52 | 1.11 | 293.1 | 163.5 | 16.5 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@compare | 5141.1 | 2.25 | 2.25 | 253.4 | - | - |
| heavy-65536-hca-cp8r0 | tilelang@main | 408.9 | 1.46 | 1.22 | 58.7 | 84.8 | 8.6 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 408.9 | 3.53 | 3.53 | 150.4 | - | - |
| heavy-65536-hca-cp8r0 | tilelang@compare | 408.9 | 1.46 | 1.22 | 58.9 | 84.2 | 8.5 |
| heavy-65536-hca-cp8r0 | cudnn_flashmla@compare | 408.9 | 1.46 | 1.46 | 137.7 | 131.9 | 13.3 |
| heavy-65536-hca-cp8r0 | cute@compare | 408.9 | 1.46 | 1.22 | 89.6 | 99.1 | 10.0 |
| heavy-65536-hca-cp8r0 | cute_ws@compare | 408.9 | 1.95 | 1.22 | 191.9 | 121.5 | 12.3 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 408.9 | 3.53 | 3.53 | 150.2 | - | - |
| heavy-65536-hca-cp8r4 | tilelang@main | 965.4 | 1.12 | 1.06 | 116.0 | 136.9 | 13.8 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 965.4 | 1.49 | 1.49 | 297.2 | - | - |
| heavy-65536-hca-cp8r4 | tilelang@compare | 965.4 | 1.12 | 1.06 | 115.2 | 135.8 | 13.7 |
| heavy-65536-hca-cp8r4 | cudnn_flashmla@compare | 965.4 | 1.12 | 1.12 | 275.0 | 237.8 | 24.0 |
| heavy-65536-hca-cp8r4 | cute@compare | 965.4 | 1.12 | 1.06 | 163.2 | 151.8 | 15.3 |
| heavy-65536-hca-cp8r4 | cute_ws@compare | 965.4 | 1.25 | 1.06 | 357.5 | 179.3 | 18.1 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 965.4 | 1.49 | 1.49 | 295.1 | - | - |
| heavy-65536-hca-cp8r7 | tilelang@main | 458.3 | 1.40 | 1.17 | 65.3 | 91.5 | 9.2 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 458.3 | 3.15 | 3.15 | 163.9 | - | - |
| heavy-65536-hca-cp8r7 | tilelang@compare | 458.3 | 1.40 | 1.17 | 64.2 | 90.8 | 9.2 |
| heavy-65536-hca-cp8r7 | cudnn_flashmla@compare | 458.3 | 1.40 | 1.40 | 149.5 | 142.5 | 14.4 |
| heavy-65536-hca-cp8r7 | cute@compare | 458.3 | 1.40 | 1.17 | 97.3 | 106.6 | 10.8 |
| heavy-65536-hca-cp8r7 | cute_ws@compare | 458.3 | 1.87 | 1.17 | 204.3 | 129.6 | 13.1 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 458.3 | 3.15 | 3.15 | 164.1 | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 3542.5 | 1.04 | 1.02 | 105.3 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 3542.5 | 1.09 | 1.09 | 246.2 | - | - |
| heavy-65536-sliding-cp1 | tilelang@compare | 3542.5 | 1.04 | 1.02 | 105.2 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | cudnn_flashmla@compare | 3542.5 | 1.04 | 1.04 | 239.2 | 178.3 | 18.0 |
| heavy-65536-sliding-cp1 | cute@compare | 3542.5 | 1.04 | 1.02 | 121.5 | 126.5 | 12.8 |
| heavy-65536-sliding-cp1 | cute_ws@compare | 3542.5 | 1.09 | 1.02 | 313.6 | 155.1 | 15.7 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@compare | 3542.5 | 1.09 | 1.09 | 243.1 | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@main | 399.0 | 1.10 | 1.05 | 61.5 | 90.6 | 9.2 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 399.0 | 1.21 | 1.21 | 170.2 | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@compare | 399.0 | 1.10 | 1.05 | 62.0 | 89.7 | 9.1 |
| heavy-65536-sliding-cp8r0 | cudnn_flashmla@compare | 399.0 | 1.10 | 1.10 | 156.9 | 136.1 | 13.8 |
| heavy-65536-sliding-cp8r0 | cute@compare | 399.0 | 1.10 | 1.05 | 98.0 | 106.7 | 10.8 |
| heavy-65536-sliding-cp8r0 | cute_ws@compare | 399.0 | 1.21 | 1.05 | 243.6 | 132.8 | 13.4 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 399.0 | 1.21 | 1.21 | 171.7 | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.7 | 102.8 | 10.4 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 209.5 | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@compare | 481.0 | 1.00 | 1.00 | 74.0 | 101.5 | 10.3 |
| heavy-65536-sliding-cp8r4 | cudnn_flashmla@compare | 481.0 | 1.00 | 1.00 | 187.5 | 157.7 | 15.9 |
| heavy-65536-sliding-cp8r4 | cute@compare | 481.0 | 1.00 | 1.00 | 115.1 | 119.6 | 12.1 |
| heavy-65536-sliding-cp8r4 | cute_ws@compare | 481.0 | 1.00 | 1.00 | 297.1 | 147.0 | 14.9 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 481.0 | 1.00 | 1.00 | 206.8 | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@main | 428.8 | 1.06 | 1.03 | 67.2 | 94.8 | 9.6 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 428.8 | 1.12 | 1.12 | 184.1 | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@compare | 428.8 | 1.06 | 1.03 | 65.9 | 94.3 | 9.5 |
| heavy-65536-sliding-cp8r7 | cudnn_flashmla@compare | 428.8 | 1.06 | 1.06 | 167.3 | 143.7 | 14.5 |
| heavy-65536-sliding-cp8r7 | cute@compare | 428.8 | 1.06 | 1.03 | 103.6 | 111.9 | 11.3 |
| heavy-65536-sliding-cp8r7 | cute_ws@compare | 428.8 | 1.12 | 1.03 | 262.8 | 137.1 | 13.9 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 428.8 | 1.12 | 1.12 | 184.4 | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 1621.2 | 3.45 | 2.86 | 42.6 | 51.0 | 5.2 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 1621.2 | 11.87 | 11.87 | 81.8 | - | - |
| tiny-65536-csa-cp1 | tilelang@compare | 1621.2 | 3.45 | 2.86 | 42.4 | 50.9 | 5.1 |
| tiny-65536-csa-cp1 | cudnn_flashmla@compare | 1621.2 | 3.45 | 3.45 | 80.8 | 75.3 | 7.6 |
| tiny-65536-csa-cp1 | cute@compare | 1621.2 | 3.45 | 2.86 | 47.4 | 53.0 | 5.4 |
| tiny-65536-csa-cp1 | cute_ws@compare | 1621.2 | 4.64 | 2.86 | 94.1 | 62.8 | 6.4 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@compare | 1621.2 | 11.87 | 11.87 | 81.7 | - | - |
| tiny-65536-csa-cp8r0 | tilelang@main | 202.8 | 3.45 | 2.86 | 28.9 | 41.5 | 4.2 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 202.8 | 11.86 | 11.86 | 71.6 | - | - |
| tiny-65536-csa-cp8r0 | tilelang@compare | 202.8 | 3.45 | 2.86 | 28.8 | 41.2 | 4.2 |
| tiny-65536-csa-cp8r0 | cudnn_flashmla@compare | 202.8 | 3.45 | 3.45 | 65.2 | 64.9 | 6.6 |
| tiny-65536-csa-cp8r0 | cute@compare | 202.8 | 3.45 | 2.86 | 43.0 | 47.9 | 4.8 |
| tiny-65536-csa-cp8r0 | cute_ws@compare | 202.8 | 4.63 | 2.86 | 85.2 | 57.5 | 5.8 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@compare | 202.8 | 11.86 | 11.86 | 71.1 | - | - |
| tiny-65536-csa-cp8r4 | tilelang@main | 202.7 | 3.45 | 2.86 | 28.8 | 41.6 | 4.2 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 202.7 | 11.87 | 11.87 | 70.3 | - | - |
| tiny-65536-csa-cp8r4 | tilelang@compare | 202.7 | 3.45 | 2.86 | 28.8 | 41.3 | 4.2 |
| tiny-65536-csa-cp8r4 | cudnn_flashmla@compare | 202.7 | 3.45 | 3.45 | 65.0 | 65.1 | 6.6 |
| tiny-65536-csa-cp8r4 | cute@compare | 202.7 | 3.45 | 2.86 | 42.8 | 48.3 | 4.9 |
| tiny-65536-csa-cp8r4 | cute_ws@compare | 202.7 | 4.64 | 2.86 | 84.6 | 58.2 | 5.9 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@compare | 202.7 | 11.87 | 11.87 | 71.2 | - | - |
| tiny-65536-csa-cp8r7 | tilelang@main | 201.4 | 3.47 | 2.88 | 27.9 | 41.3 | 4.2 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 201.4 | 11.94 | 11.94 | 69.2 | - | - |
| tiny-65536-csa-cp8r7 | tilelang@compare | 201.4 | 3.47 | 2.88 | 28.4 | 41.0 | 4.1 |
| tiny-65536-csa-cp8r7 | cudnn_flashmla@compare | 201.4 | 3.47 | 3.47 | 64.9 | 64.9 | 6.6 |
| tiny-65536-csa-cp8r7 | cute@compare | 201.4 | 3.47 | 2.88 | 42.5 | 47.8 | 4.8 |
| tiny-65536-csa-cp8r7 | cute_ws@compare | 201.4 | 4.67 | 2.88 | 84.3 | 57.8 | 5.8 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@compare | 201.4 | 11.94 | 11.94 | 70.2 | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.4 | - | - |
| tiny-65536-hca-cp1 | tilelang@compare | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-hca-cp1 | cudnn_flashmla@compare | 1306.0 | 1.85 | 1.85 | 87.3 | 77.5 | 7.8 |
| tiny-65536-hca-cp1 | cute@compare | 1306.0 | 1.85 | 1.39 | 50.2 | 63.0 | 6.4 |
| tiny-65536-hca-cp1 | cute_ws@compare | 1306.0 | 2.95 | 1.39 | 102.1 | 77.0 | 7.8 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@compare | 1306.0 | 2.95 | 2.95 | 89.7 | - | - |
| tiny-65536-hca-cp8r0 | tilelang@main | 163.4 | 1.85 | 1.39 | 27.4 | 45.0 | 4.6 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 163.4 | 2.94 | 2.94 | 71.0 | - | - |
| tiny-65536-hca-cp8r0 | tilelang@compare | 163.4 | 1.85 | 1.39 | 27.2 | 44.8 | 4.5 |
| tiny-65536-hca-cp8r0 | cudnn_flashmla@compare | 163.4 | 1.85 | 1.85 | 64.1 | 63.4 | 6.4 |
| tiny-65536-hca-cp8r0 | cute@compare | 163.4 | 1.85 | 1.39 | 44.1 | 55.8 | 5.6 |
| tiny-65536-hca-cp8r0 | cute_ws@compare | 163.4 | 2.94 | 1.39 | 90.2 | 69.5 | 7.0 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@compare | 163.4 | 2.94 | 2.94 | 70.6 | - | - |
| tiny-65536-hca-cp8r4 | tilelang@main | 163.3 | 1.85 | 1.39 | 26.9 | 45.2 | 4.6 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-hca-cp8r4 | tilelang@compare | 163.3 | 1.85 | 1.39 | 26.9 | 44.9 | 4.5 |
| tiny-65536-hca-cp8r4 | cudnn_flashmla@compare | 163.3 | 1.85 | 1.85 | 63.9 | 62.8 | 6.3 |
| tiny-65536-hca-cp8r4 | cute@compare | 163.3 | 1.85 | 1.39 | 43.5 | 55.7 | 5.6 |
| tiny-65536-hca-cp8r4 | cute_ws@compare | 163.3 | 2.95 | 1.39 | 91.1 | 69.8 | 7.1 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@compare | 163.3 | 2.95 | 2.95 | 69.9 | - | - |
| tiny-65536-hca-cp8r7 | tilelang@main | 162.3 | 1.86 | 1.39 | 27.0 | 44.7 | 4.5 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 162.3 | 2.96 | 2.96 | 69.6 | - | - |
| tiny-65536-hca-cp8r7 | tilelang@compare | 162.3 | 1.86 | 1.39 | 26.9 | 44.2 | 4.5 |
| tiny-65536-hca-cp8r7 | cudnn_flashmla@compare | 162.3 | 1.86 | 1.86 | 63.2 | 62.3 | 6.3 |
| tiny-65536-hca-cp8r7 | cute@compare | 162.3 | 1.86 | 1.39 | 43.5 | 55.2 | 5.6 |
| tiny-65536-hca-cp8r7 | cute_ws@compare | 162.3 | 2.96 | 1.39 | 88.6 | 68.9 | 7.0 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@compare | 162.3 | 2.96 | 2.96 | 69.6 | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.8 | 59.3 | 6.0 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.3 | - | - |
| tiny-65536-sliding-cp1 | tilelang@compare | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-sliding-cp1 | cudnn_flashmla@compare | 1306.0 | 1.85 | 1.85 | 87.4 | 77.5 | 7.8 |
| tiny-65536-sliding-cp1 | cute@compare | 1306.0 | 1.85 | 1.39 | 50.2 | 63.0 | 6.4 |
| tiny-65536-sliding-cp1 | cute_ws@compare | 1306.0 | 2.95 | 1.39 | 101.6 | 77.1 | 7.8 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@compare | 1306.0 | 2.95 | 2.95 | 89.6 | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@main | 163.4 | 1.85 | 1.39 | 26.7 | 45.0 | 4.5 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 163.4 | 2.94 | 2.94 | 69.8 | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@compare | 163.4 | 1.85 | 1.39 | 26.8 | 44.6 | 4.5 |
| tiny-65536-sliding-cp8r0 | cudnn_flashmla@compare | 163.4 | 1.85 | 1.85 | 63.2 | 62.9 | 6.4 |
| tiny-65536-sliding-cp8r0 | cute@compare | 163.4 | 1.85 | 1.39 | 44.0 | 55.7 | 5.6 |
| tiny-65536-sliding-cp8r0 | cute_ws@compare | 163.4 | 2.94 | 1.39 | 89.3 | 69.5 | 7.0 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@compare | 163.4 | 2.94 | 2.94 | 69.6 | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@main | 163.3 | 1.85 | 1.39 | 27.2 | 45.2 | 4.6 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@compare | 163.3 | 1.85 | 1.39 | 27.0 | 44.7 | 4.5 |
| tiny-65536-sliding-cp8r4 | cudnn_flashmla@compare | 163.3 | 1.85 | 1.85 | 63.6 | 62.8 | 6.3 |
| tiny-65536-sliding-cp8r4 | cute@compare | 163.3 | 1.85 | 1.39 | 43.7 | 55.6 | 5.6 |
| tiny-65536-sliding-cp8r4 | cute_ws@compare | 163.3 | 2.95 | 1.39 | 90.6 | 69.7 | 7.0 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@compare | 163.3 | 2.95 | 2.95 | 70.3 | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@main | 162.3 | 1.86 | 1.39 | 27.2 | 44.7 | 4.5 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 162.3 | 2.96 | 2.96 | 69.7 | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@compare | 162.3 | 1.86 | 1.39 | 27.0 | 44.3 | 4.5 |
| tiny-65536-sliding-cp8r7 | cudnn_flashmla@compare | 162.3 | 1.86 | 1.86 | 63.0 | 62.8 | 6.3 |
| tiny-65536-sliding-cp8r7 | cute@compare | 162.3 | 1.86 | 1.39 | 43.6 | 55.3 | 5.6 |
| tiny-65536-sliding-cp8r7 | cute_ws@compare | 162.3 | 2.96 | 1.39 | 88.9 | 68.9 | 7.0 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@compare | 162.3 | 2.96 | 2.96 | 69.9 | - | - |

Correctness failures (excluded from timing): 0
