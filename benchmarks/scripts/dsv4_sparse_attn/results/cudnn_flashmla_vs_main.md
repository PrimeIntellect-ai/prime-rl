- `main`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git 6d5cf0180
- `cudnn_flashmla`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git e28cc525e
- corpus hash b5b289171983c8c6; synthetic corpus: random-weight CSA picks are near-uniform, while a
  trained indexer favors recent and neighboring entries, so CSA gather locality here is pessimistic.

Op-boundary time per call in µs (lower is better): median over rounds, p20-p80 across rounds.
`/TL` is this time divided by tilelang's in the same run.

| item | backend | fwd µs | fwd p20-p80 | fwd /TL | f+b µs | f+b p20-p80 | f+b /TL |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 1197 | 1180-1209 | 1.00 | 3186 | 3183-3216 | 1.00 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 384.0 | 381.9-387.6 | 0.32 | - | - | - |
| single-2048-csa-cp1 | tilelang@cudnn_flashmla | 1210 | 1194-1230 | 1.00 | 3266 | 3255-3295 | 1.00 |
| single-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 438.1 | 436.4-441.2 | 0.36 | 1925 | 1918-1953 | 0.59 |
| single-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 377.8 | 376.6-380.6 | 0.31 | - | - | - |
| single-2048-csa-cp8r0 | tilelang@main | 733.5 | 729.6-769.4 | 1.00 | 2001 | 1987-2043 | 1.00 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 182.5 | 181.2-196.9 | 0.25 | - | - | - |
| single-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 766.1 | 761.7-775.4 | 1.00 | 2075 | 2061-2101 | 1.00 |
| single-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 305.5 | 298.7-306.3 | 0.40 | 1449 | 1435-1456 | 0.70 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 187.8 | 183.5-191.1 | 0.25 | - | - | - |
| single-2048-csa-cp8r4 | tilelang@main | 763.3 | 755.8-769.4 | 1.00 | 2028 | 2010-2035 | 1.00 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 197.7 | 194.4-198.5 | 0.26 | - | - | - |
| single-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 776.8 | 768.3-793.7 | 1.00 | 2081 | 2070-2096 | 1.00 |
| single-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 305.5 | 303.6-313.7 | 0.39 | 1448 | 1443-1534 | 0.70 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 197.1 | 194.8-199.0 | 0.25 | - | - | - |
| single-2048-csa-cp8r7 | tilelang@main | 774.3 | 771.8-795.5 | 1.00 | 2025 | 2024-2030 | 1.00 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 202.8 | 198.5-208.2 | 0.26 | - | - | - |
| single-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 800.2 | 779.3-809.3 | 1.00 | 2093 | 2082-2128 | 1.00 |
| single-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 306.2 | 299.6-315.3 | 0.38 | 1446 | 1439-1455 | 0.69 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 212.8 | 206.7-213.6 | 0.27 | - | - | - |
| single-2048-hca-cp1 | tilelang@main | 1085 | 1075-1099 | 1.00 | 2504 | 2477-2531 | 1.00 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 341.1 | 339.6-343.2 | 0.31 | - | - | - |
| single-2048-hca-cp1 | tilelang@cudnn_flashmla | 1103 | 1087-1116 | 1.00 | 2531 | 2524-2548 | 1.00 |
| single-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 406.7 | 404.0-418.8 | 0.37 | 1571 | 1563-1578 | 0.62 |
| single-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 346.7 | 342.2-358.4 | 0.31 | - | - | - |
| single-2048-hca-cp8r0 | tilelang@main | 763.9 | 761.9-769.2 | 1.00 | 2152 | 2142-2403 | 1.00 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 188.7 | 185.0-194.9 | 0.25 | - | - | - |
| single-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 823.5 | 788.1-839.0 | 1.00 | 2230 | 2204-2258 | 1.00 |
| single-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 322.1 | 311.3-347.2 | 0.39 | 1476 | 1455-1529 | 0.66 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 191.0 | 189.2-194.4 | 0.23 | - | - | - |
| single-2048-hca-cp8r4 | tilelang@main | 793.0 | 784.8-810.2 | 1.00 | 2158 | 2111-2193 | 1.00 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 193.9 | 191.6-196.0 | 0.24 | - | - | - |
| single-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 784.5 | 782.3-808.5 | 1.00 | 2220 | 2213-2221 | 1.00 |
| single-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 312.6 | 306.9-317.9 | 0.40 | 1466 | 1456-1484 | 0.66 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 192.3 | 186.2-193.8 | 0.25 | - | - | - |
| single-2048-hca-cp8r7 | tilelang@main | 806.9 | 791.4-813.7 | 1.00 | 2178 | 2146-2204 | 1.00 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 193.5 | 189.0-195.7 | 0.24 | - | - | - |
| single-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 778.8 | 776.1-793.4 | 1.00 | 2193 | 2170-2197 | 1.00 |
| single-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 311.6 | 302.1-318.2 | 0.40 | 1467 | 1439-1485 | 0.67 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 191.8 | 188.7-196.2 | 0.25 | - | - | - |
| single-2048-sliding-cp1 | tilelang@main | 1016 | 1002-1021 | 1.00 | 2288 | 2278-2306 | 1.00 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 298.4 | 293.8-303.0 | 0.29 | - | - | - |
| single-2048-sliding-cp1 | tilelang@cudnn_flashmla | 1011 | 998.2-1019 | 1.00 | 2373 | 2367-2399 | 1.00 |
| single-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 358.0 | 355.8-360.8 | 0.35 | 1518 | 1516-1527 | 0.64 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 293.7 | 290.6-297.0 | 0.29 | - | - | - |
| single-2048-sliding-cp8r0 | tilelang@main | 732.5 | 723.8-761.5 | 1.00 | 2047 | 2013-2148 | 1.00 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 179.7 | 174.7-186.8 | 0.25 | - | - | - |
| single-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 748.9 | 736.7-761.9 | 1.00 | 2096 | 2088-2117 | 1.00 |
| single-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 309.0 | 301.6-311.2 | 0.41 | 1464 | 1450-1467 | 0.70 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 180.9 | 175.4-184.7 | 0.24 | - | - | - |
| single-2048-sliding-cp8r4 | tilelang@main | 739.5 | 732.5-778.8 | 1.00 | 2018 | 2006-2039 | 1.00 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 179.0 | 174.0-188.2 | 0.24 | - | - | - |
| single-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 764.2 | 744.5-767.7 | 1.00 | 2096 | 2094-2107 | 1.00 |
| single-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 306.4 | 297.9-308.7 | 0.40 | 1465 | 1458-1476 | 0.70 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 179.3 | 175.8-181.3 | 0.23 | - | - | - |
| single-2048-sliding-cp8r7 | tilelang@main | 739.5 | 735.6-746.3 | 1.00 | 2034 | 2020-2067 | 1.00 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 180.9 | 179.1-190.1 | 0.24 | - | - | - |
| single-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 759.6 | 743.9-769.3 | 1.00 | 2116 | 2115-2132 | 1.00 |
| single-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 304.3 | 300.0-310.8 | 0.40 | 1461 | 1451-1493 | 0.69 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 180.3 | 177.1-183.1 | 0.24 | - | - | - |
| short-2048-csa-cp1 | tilelang@main | 1132 | 1121-1135 | 1.00 | 2756 | 2748-2783 | 1.00 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 368.0 | 365.2-371.5 | 0.33 | - | - | - |
| short-2048-csa-cp1 | tilelang@cudnn_flashmla | 1137 | 1129-1150 | 1.00 | 2845 | 2814-2875 | 1.00 |
| short-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 425.5 | 423.2-430.9 | 0.37 | 1727 | 1716-1734 | 0.61 |
| short-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 367.8 | 366.3-370.3 | 0.32 | - | - | - |
| short-2048-csa-cp8r0 | tilelang@main | 753.1 | 737.6-770.5 | 1.00 | 2023 | 2006-2054 | 1.00 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 187.7 | 179.4-191.9 | 0.25 | - | - | - |
| short-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 742.9 | 733.1-767.8 | 1.00 | 2116 | 2100-2124 | 1.00 |
| short-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 303.1 | 302.7-333.9 | 0.41 | 1465 | 1460-1473 | 0.69 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 185.8 | 181.8-202.2 | 0.25 | - | - | - |
| short-2048-csa-cp8r4 | tilelang@main | 742.5 | 736.4-755.8 | 1.00 | 2033 | 2020-2063 | 1.00 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 187.7 | 183.3-192.1 | 0.25 | - | - | - |
| short-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 764.3 | 755.9-775.9 | 1.00 | 2114 | 2088-2115 | 1.00 |
| short-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 304.7 | 302.9-318.9 | 0.40 | 1455 | 1449-1469 | 0.69 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 194.4 | 190.0-196.9 | 0.25 | - | - | - |
| short-2048-csa-cp8r7 | tilelang@main | 762.2 | 749.4-771.5 | 1.00 | 2037 | 2025-2095 | 1.00 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 191.5 | 186.9-192.3 | 0.25 | - | - | - |
| short-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 791.5 | 772.0-831.7 | 1.00 | 2110 | 2087-2160 | 1.00 |
| short-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 323.0 | 305.3-339.1 | 0.41 | 1477 | 1460-1478 | 0.70 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 198.9 | 194.3-212.8 | 0.25 | - | - | - |
| short-2048-hca-cp1 | tilelang@main | 1079 | 1074-1103 | 1.00 | 2489 | 2485-2573 | 1.00 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 340.4 | 337.8-344.8 | 0.32 | - | - | - |
| short-2048-hca-cp1 | tilelang@cudnn_flashmla | 1088 | 1080-1104 | 1.00 | 2525 | 2516-2542 | 1.00 |
| short-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 397.8 | 394.9-410.5 | 0.37 | 1563 | 1551-1567 | 0.62 |
| short-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 336.6 | 334.4-337.9 | 0.31 | - | - | - |
| short-2048-hca-cp8r0 | tilelang@main | 787.4 | 779.9-790.5 | 1.00 | 2154 | 2142-2191 | 1.00 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 191.5 | 187.6-192.1 | 0.24 | - | - | - |
| short-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 784.2 | 779.8-811.2 | 1.00 | 2162 | 2154-2173 | 1.00 |
| short-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 307.2 | 305.0-316.6 | 0.39 | 1449 | 1445-1456 | 0.67 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 186.2 | 183.5-189.8 | 0.24 | - | - | - |
| short-2048-hca-cp8r4 | tilelang@main | 795.0 | 779.4-801.7 | 1.00 | 2162 | 2154-2181 | 1.00 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 193.3 | 190.6-194.5 | 0.24 | - | - | - |
| short-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 777.1 | 765.8-792.9 | 1.00 | 2204 | 2197-2207 | 1.00 |
| short-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 305.3 | 301.6-308.7 | 0.39 | 1467 | 1451-1472 | 0.67 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 186.7 | 185.0-188.3 | 0.24 | - | - | - |
| short-2048-hca-cp8r7 | tilelang@main | 792.8 | 779.8-798.3 | 1.00 | 2193 | 2161-2199 | 1.00 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 188.9 | 188.3-192.7 | 0.24 | - | - | - |
| short-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 785.9 | 782.9-790.8 | 1.00 | 2154 | 2152-2202 | 1.00 |
| short-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 312.5 | 311.7-312.8 | 0.40 | 1473 | 1456-1500 | 0.68 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 193.6 | 192.9-195.9 | 0.25 | - | - | - |
| short-2048-sliding-cp1 | tilelang@main | 1003 | 994.3-1018 | 1.00 | 2330 | 2322-2398 | 1.00 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 293.3 | 291.6-301.8 | 0.29 | - | - | - |
| short-2048-sliding-cp1 | tilelang@cudnn_flashmla | 995.2 | 982.6-997.2 | 1.00 | 2455 | 2452-2467 | 1.00 |
| short-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 351.4 | 347.1-362.9 | 0.35 | 1631 | 1615-1648 | 0.66 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 289.8 | 287.4-309.2 | 0.29 | - | - | - |
| short-2048-sliding-cp8r0 | tilelang@main | 746.8 | 734.3-752.0 | 1.00 | 2042 | 2039-2056 | 1.00 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 181.9 | 179.7-187.0 | 0.24 | - | - | - |
| short-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 742.7 | 734.5-755.9 | 1.00 | 2081 | 2068-2084 | 1.00 |
| short-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 299.1 | 298.1-301.6 | 0.40 | 1461 | 1456-1466 | 0.70 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 175.5 | 172.9-183.5 | 0.24 | - | - | - |
| short-2048-sliding-cp8r4 | tilelang@main | 756.9 | 751.3-773.4 | 1.00 | 2022 | 2012-2028 | 1.00 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 189.8 | 187.4-194.1 | 0.25 | - | - | - |
| short-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 749.8 | 743.0-751.0 | 1.00 | 2101 | 2078-2228 | 1.00 |
| short-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 303.2 | 299.9-310.5 | 0.40 | 1450 | 1435-1458 | 0.69 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 180.8 | 176.8-182.1 | 0.24 | - | - | - |
| short-2048-sliding-cp8r7 | tilelang@main | 736.4 | 731.7-751.1 | 1.00 | 2044 | 2038-2069 | 1.00 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 185.1 | 179.8-188.5 | 0.25 | - | - | - |
| short-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 758.9 | 744.6-767.1 | 1.00 | 2118 | 2102-2164 | 1.00 |
| short-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 306.0 | 298.4-313.5 | 0.40 | 1463 | 1454-1632 | 0.69 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 179.0 | 177.8-182.2 | 0.24 | - | - | - |
| heavy-2048-csa-cp1 | tilelang@main | 1171 | 1161-1186 | 1.00 | 2955 | 2937-2956 | 1.00 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 389.7 | 383.9-390.4 | 0.33 | - | - | - |
| heavy-2048-csa-cp1 | tilelang@cudnn_flashmla | 1163 | 1156-1181 | 1.00 | 2976 | 2959-2979 | 1.00 |
| heavy-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 440.4 | 438.9-443.9 | 0.38 | 1777 | 1771-1785 | 0.60 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 386.7 | 382.7-387.6 | 0.33 | - | - | - |
| heavy-2048-csa-cp8r0 | tilelang@main | 747.5 | 737.9-759.0 | 1.00 | 2043 | 2029-2055 | 1.00 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 189.1 | 186.1-190.5 | 0.25 | - | - | - |
| heavy-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 745.2 | 733.7-772.0 | 1.00 | 2108 | 2108-2124 | 1.00 |
| heavy-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 307.0 | 297.1-314.1 | 0.41 | 1468 | 1461-1481 | 0.70 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 185.0 | 181.7-192.9 | 0.25 | - | - | - |
| heavy-2048-csa-cp8r4 | tilelang@main | 777.6 | 755.6-789.9 | 1.00 | 2065 | 2036-2070 | 1.00 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 200.7 | 189.6-207.8 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 773.0 | 767.2-778.7 | 1.00 | 2098 | 2086-2114 | 1.00 |
| heavy-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 305.3 | 303.4-321.1 | 0.39 | 1447 | 1434-1454 | 0.69 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 189.3 | 185.9-192.9 | 0.24 | - | - | - |
| heavy-2048-csa-cp8r7 | tilelang@main | 793.2 | 765.5-859.1 | 1.00 | 2034 | 2029-2075 | 1.00 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 209.7 | 204.5-224.9 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 779.8 | 773.6-786.0 | 1.00 | 2108 | 2086-2114 | 1.00 |
| heavy-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 299.1 | 296.8-307.8 | 0.38 | 1467 | 1442-1479 | 0.70 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 196.2 | 194.3-201.4 | 0.25 | - | - | - |
| heavy-2048-hca-cp1 | tilelang@main | 1074 | 1069-1080 | 1.00 | 2464 | 2458-2470 | 1.00 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 335.2 | 329.4-337.8 | 0.31 | - | - | - |
| heavy-2048-hca-cp1 | tilelang@cudnn_flashmla | 1117 | 1090-1227 | 1.00 | 2527 | 2497-2554 | 1.00 |
| heavy-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 396.9 | 394.1-462.9 | 0.36 | 1554 | 1538-1598 | 0.62 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 337.1 | 327.4-370.3 | 0.30 | - | - | - |
| heavy-2048-hca-cp8r0 | tilelang@main | 774.2 | 771.3-778.2 | 1.00 | 2135 | 2131-2143 | 1.00 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 189.3 | 188.5-195.6 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 786.3 | 776.2-800.4 | 1.00 | 2240 | 2206-2308 | 1.00 |
| heavy-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 307.8 | 306.7-315.5 | 0.39 | 1476 | 1454-1485 | 0.66 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 185.1 | 184.2-187.5 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r4 | tilelang@main | 782.0 | 771.9-791.9 | 1.00 | 2122 | 2120-2137 | 1.00 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 189.5 | 183.3-193.8 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 802.5 | 792.0-810.4 | 1.00 | 2281 | 2231-2909 | 1.00 |
| heavy-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 307.5 | 303.8-315.3 | 0.38 | 1483 | 1472-1777 | 0.65 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 188.2 | 186.4-193.4 | 0.23 | - | - | - |
| heavy-2048-hca-cp8r7 | tilelang@main | 786.8 | 778.1-795.3 | 1.00 | 2157 | 2147-2182 | 1.00 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 189.8 | 187.0-192.2 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 799.7 | 789.5-818.0 | 1.00 | 2220 | 2202-2229 | 1.00 |
| heavy-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 312.2 | 309.9-314.5 | 0.39 | 1490 | 1488-1497 | 0.67 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 192.5 | 191.5-195.4 | 0.24 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang@main | 1000 | 986.0-1002 | 1.00 | 2294 | 2292-2303 | 1.00 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 291.8 | 289.1-294.8 | 0.29 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang@cudnn_flashmla | 996.4 | 988.9-1009 | 1.00 | 2366 | 2355-2372 | 1.00 |
| heavy-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 354.0 | 347.6-364.7 | 0.36 | 1515 | 1509-1525 | 0.64 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 291.5 | 290.3-292.4 | 0.29 | - | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@main | 736.0 | 727.2-829.0 | 1.00 | 2039 | 2024-2043 | 1.00 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 180.0 | 176.3-183.7 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 761.8 | 754.7-771.8 | 1.00 | 2127 | 2117-2167 | 1.00 |
| heavy-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 311.9 | 304.8-314.6 | 0.41 | 1482 | 1466-1484 | 0.70 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 188.6 | 182.9-190.7 | 0.25 | - | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@main | 742.3 | 731.2-750.7 | 1.00 | 2036 | 2019-2052 | 1.00 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 180.7 | 175.9-183.0 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 740.7 | 735.8-758.2 | 1.00 | 2101 | 2074-2137 | 1.00 |
| heavy-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 305.5 | 301.9-313.5 | 0.41 | 1461 | 1450-1473 | 0.70 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 182.3 | 180.5-184.5 | 0.25 | - | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@main | 743.6 | 736.9-744.8 | 1.00 | 2071 | 2052-2104 | 1.00 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 182.7 | 175.3-184.9 | 0.25 | - | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 752.5 | 750.8-755.0 | 1.00 | 2141 | 2131-2149 | 1.00 |
| heavy-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 311.0 | 309.8-312.3 | 0.41 | 1482 | 1478-1497 | 0.69 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 182.9 | 182.0-186.5 | 0.24 | - | - | - |
| tiny-2048-csa-cp1 | tilelang@main | 1043 | 1040-1056 | 1.00 | 2359 | 2333-2369 | 1.00 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 333.9 | 329.9-336.8 | 0.32 | - | - | - |
| tiny-2048-csa-cp1 | tilelang@cudnn_flashmla | 1069 | 1054-1070 | 1.00 | 2391 | 2369-2392 | 1.00 |
| tiny-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 397.0 | 394.0-402.1 | 0.37 | 1506 | 1500-1515 | 0.63 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 335.4 | 334.0-337.8 | 0.31 | - | - | - |
| tiny-2048-csa-cp8r0 | tilelang@main | 753.9 | 747.6-789.7 | 1.00 | 2056 | 2032-2068 | 1.00 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 192.5 | 185.3-201.4 | 0.26 | - | - | - |
| tiny-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 752.1 | 744.9-757.6 | 1.00 | 2119 | 2111-2163 | 1.00 |
| tiny-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 306.3 | 301.6-308.2 | 0.41 | 1485 | 1465-1498 | 0.70 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 187.5 | 183.6-189.5 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r4 | tilelang@main | 748.5 | 741.9-755.9 | 1.00 | 2029 | 2015-2055 | 1.00 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 187.9 | 184.4-189.9 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 760.5 | 750.6-763.7 | 1.00 | 2141 | 2104-2147 | 1.00 |
| tiny-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 310.9 | 306.8-315.2 | 0.41 | 1486 | 1476-1496 | 0.69 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 188.7 | 186.9-191.2 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r7 | tilelang@main | 761.0 | 757.8-769.9 | 1.00 | 2040 | 2032-2079 | 1.00 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 189.9 | 186.1-198.6 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 755.3 | 746.1-763.2 | 1.00 | 2101 | 2070-2145 | 1.00 |
| tiny-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 305.6 | 303.6-312.1 | 0.40 | 1477 | 1459-1487 | 0.70 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 184.4 | 180.6-185.7 | 0.24 | - | - | - |
| tiny-2048-hca-cp1 | tilelang@main | 962.7 | 956.2-975.0 | 1.00 | 2093 | 2080-2128 | 1.00 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 292.2 | 290.6-305.2 | 0.30 | - | - | - |
| tiny-2048-hca-cp1 | tilelang@cudnn_flashmla | 973.0 | 966.0-974.6 | 1.00 | 2187 | 2166-2188 | 1.00 |
| tiny-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 355.2 | 349.8-359.8 | 0.37 | 1493 | 1490-1499 | 0.68 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 295.8 | 291.7-298.6 | 0.30 | - | - | - |
| tiny-2048-hca-cp8r0 | tilelang@main | 736.5 | 733.4-737.7 | 1.00 | 2030 | 2013-2052 | 1.00 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 182.6 | 176.8-187.7 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 746.3 | 738.6-749.3 | 1.00 | 2092 | 2087-2102 | 1.00 |
| tiny-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 305.5 | 303.6-317.0 | 0.41 | 1463 | 1445-1470 | 0.70 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 181.7 | 179.1-184.4 | 0.24 | - | - | - |
| tiny-2048-hca-cp8r4 | tilelang@main | 735.0 | 723.8-756.2 | 1.00 | 2049 | 2011-2074 | 1.00 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 180.9 | 179.2-183.4 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 734.4 | 729.5-748.0 | 1.00 | 2130 | 2120-2134 | 1.00 |
| tiny-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 307.0 | 299.3-315.4 | 0.42 | 1474 | 1461-1481 | 0.69 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 183.5 | 180.5-187.3 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r7 | tilelang@main | 751.2 | 738.8-819.8 | 1.00 | 2099 | 2044-2164 | 1.00 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 190.0 | 184.2-195.8 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 752.0 | 743.4-757.4 | 1.00 | 2107 | 2100-2123 | 1.00 |
| tiny-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 308.3 | 300.3-310.7 | 0.41 | 1478 | 1475-1481 | 0.70 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 176.8 | 174.8-181.6 | 0.24 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang@main | 957.8 | 952.0-962.9 | 1.00 | 2108 | 2103-2118 | 1.00 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 290.5 | 284.1-294.3 | 0.30 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang@cudnn_flashmla | 964.2 | 955.1-969.4 | 1.00 | 2176 | 2170-2194 | 1.00 |
| tiny-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 353.3 | 352.6-354.5 | 0.37 | 1498 | 1492-1510 | 0.69 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 292.4 | 289.8-294.8 | 0.30 | - | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@main | 731.7 | 730.4-739.6 | 1.00 | 2136 | 2058-2216 | 1.00 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 179.8 | 173.2-180.5 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 757.6 | 748.0-758.9 | 1.00 | 2122 | 2111-2144 | 1.00 |
| tiny-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 307.2 | 304.5-313.9 | 0.41 | 1475 | 1471-1485 | 0.70 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 184.8 | 182.0-186.2 | 0.24 | - | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@main | 738.5 | 727.1-742.4 | 1.00 | 2024 | 2010-2050 | 1.00 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 178.4 | 171.8-182.8 | 0.24 | - | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 743.4 | 735.6-746.9 | 1.00 | 2115 | 2108-2125 | 1.00 |
| tiny-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 300.9 | 298.2-315.2 | 0.40 | 1464 | 1455-1467 | 0.69 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 182.1 | 174.5-185.9 | 0.24 | - | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@main | 758.1 | 748.7-774.6 | 1.00 | 2049 | 2038-2151 | 1.00 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 186.8 | 182.1-192.1 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 751.8 | 747.8-758.2 | 1.00 | 2120 | 2114-2138 | 1.00 |
| tiny-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 312.9 | 310.4-320.2 | 0.42 | 1469 | 1463-1472 | 0.69 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 185.1 | 182.8-186.4 | 0.25 | - | - | - |
| single-4096-csa-cp1 | tilelang@main | 1890 | 1888-1954 | 1.00 | 5972 | 5965-5976 | 1.00 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 715.4 | 711.7-718.7 | 0.38 | - | - | - |
| single-4096-csa-cp1 | tilelang@cudnn_flashmla | 1908 | 1906-1928 | 1.00 | 6010 | 5986-6016 | 1.00 |
| single-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 786.8 | 779.5-788.3 | 0.41 | 3124 | 3118-3138 | 0.52 |
| single-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 717.7 | 715.9-720.3 | 0.38 | - | - | - |
| single-4096-csa-cp8r0 | tilelang@main | 795.7 | 791.3-798.4 | 1.00 | 2037 | 2027-2047 | 1.00 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 206.3 | 203.2-212.2 | 0.26 | - | - | - |
| single-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 812.0 | 803.9-848.1 | 1.00 | 2121 | 2110-2128 | 1.00 |
| single-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 305.2 | 297.6-318.4 | 0.38 | 1486 | 1471-1498 | 0.70 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 207.8 | 205.6-210.5 | 0.26 | - | - | - |
| single-4096-csa-cp8r4 | tilelang@main | 869.0 | 862.6-883.1 | 1.00 | 2333 | 2321-2341 | 1.00 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 248.4 | 245.0-250.7 | 0.29 | - | - | - |
| single-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 889.3 | 880.5-902.7 | 1.00 | 2421 | 2405-2432 | 1.00 |
| single-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 310.5 | 308.6-321.4 | 0.35 | 1497 | 1481-1501 | 0.62 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 251.3 | 248.2-256.0 | 0.28 | - | - | - |
| single-4096-csa-cp8r7 | tilelang@main | 880.8 | 872.3-893.7 | 1.00 | 2349 | 2340-2358 | 1.00 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 252.0 | 246.9-253.2 | 0.29 | - | - | - |
| single-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 893.5 | 883.4-897.1 | 1.00 | 2385 | 2381-2405 | 1.00 |
| single-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 312.2 | 310.9-317.1 | 0.35 | 1471 | 1461-1476 | 0.62 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 253.5 | 251.3-259.5 | 0.28 | - | - | - |
| single-4096-hca-cp1 | tilelang@main | 1432 | 1411-1475 | 1.00 | 3134 | 3126-3142 | 1.00 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 505.5 | 499.0-532.2 | 0.35 | - | - | - |
| single-4096-hca-cp1 | tilelang@cudnn_flashmla | 1429 | 1423-1433 | 1.00 | 3216 | 3191-3253 | 1.00 |
| single-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 572.9 | 569.1-576.2 | 0.40 | 2124 | 2109-2125 | 0.66 |
| single-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 508.4 | 507.5-514.2 | 0.36 | - | - | - |
| single-4096-hca-cp8r0 | tilelang@main | 828.2 | 816.4-834.6 | 1.00 | 2126 | 2122-2148 | 1.00 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 217.1 | 213.7-217.8 | 0.26 | - | - | - |
| single-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 865.1 | 840.2-869.6 | 1.00 | 2204 | 2191-2219 | 1.00 |
| single-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 313.3 | 310.2-323.6 | 0.36 | 1469 | 1462-1498 | 0.67 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 213.9 | 211.6-217.5 | 0.25 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 824.2 | 816.0-832.5 | 1.00 | 2143 | 2120-2164 | 1.00 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 216.8 | 214.5-219.7 | 0.26 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 836.6 | 830.9-849.6 | 1.00 | 2246 | 2237-2257 | 1.00 |
| single-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 313.3 | 308.8-314.8 | 0.37 | 1499 | 1487-1513 | 0.67 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 216.9 | 215.1-217.8 | 0.26 | - | - | - |
| single-4096-hca-cp8r7 | tilelang@main | 827.1 | 813.1-838.7 | 1.00 | 2159 | 2139-2187 | 1.00 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 217.2 | 212.4-222.6 | 0.26 | - | - | - |
| single-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 844.9 | 838.9-859.6 | 1.00 | 2212 | 2202-2231 | 1.00 |
| single-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 318.1 | 313.6-324.3 | 0.38 | 1476 | 1468-1502 | 0.67 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 223.7 | 218.2-228.4 | 0.26 | - | - | - |
| single-4096-sliding-cp1 | tilelang@main | 1297 | 1285-1352 | 1.00 | 2859 | 2858-2862 | 1.00 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 420.2 | 412.8-423.1 | 0.32 | - | - | - |
| single-4096-sliding-cp1 | tilelang@cudnn_flashmla | 1294 | 1284-1302 | 1.00 | 2920 | 2909-2936 | 1.00 |
| single-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 483.1 | 477.5-488.2 | 0.37 | 2007 | 2000-2011 | 0.69 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 418.6 | 415.3-420.5 | 0.32 | - | - | - |
| single-4096-sliding-cp8r0 | tilelang@main | 790.1 | 775.7-812.3 | 1.00 | 2034 | 2027-2045 | 1.00 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 199.3 | 197.8-208.9 | 0.25 | - | - | - |
| single-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 799.0 | 793.6-803.2 | 1.00 | 2140 | 2137-2154 | 1.00 |
| single-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 312.1 | 311.7-316.4 | 0.39 | 1481 | 1476-1489 | 0.69 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 201.0 | 200.0-203.8 | 0.25 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 791.1 | 779.7-892.0 | 1.00 | 2057 | 2032-2070 | 1.00 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 197.7-217.5 | 0.26 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 802.1 | 796.4-812.1 | 1.00 | 2113 | 2103-2161 | 1.00 |
| single-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 304.2 | 303.5-309.7 | 0.38 | 1470 | 1435-1517 | 0.70 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 200.0 | 198.7-204.3 | 0.25 | - | - | - |
| single-4096-sliding-cp8r7 | tilelang@main | 785.2 | 769.9-787.5 | 1.00 | 2065 | 2049-2079 | 1.00 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 194.0 | 192.7-204.3 | 0.25 | - | - | - |
| single-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 803.4 | 787.9-809.7 | 1.00 | 2133 | 2115-2141 | 1.00 |
| single-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 311.8 | 309.2-318.7 | 0.39 | 1470 | 1462-1492 | 0.69 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 204.4 | 198.2-206.1 | 0.25 | - | - | - |
| short-4096-csa-cp1 | tilelang@main | 1523 | 1513-1536 | 1.00 | 3873 | 3871-3875 | 1.00 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 552.9 | 546.9-555.2 | 0.36 | - | - | - |
| short-4096-csa-cp1 | tilelang@cudnn_flashmla | 1532 | 1521-1540 | 1.00 | 3878 | 3863-3897 | 1.00 |
| short-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 614.2 | 613.0-614.8 | 0.40 | 2370 | 2355-2376 | 0.61 |
| short-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 552.7 | 549.6-554.7 | 0.36 | - | - | - |
| short-4096-csa-cp8r0 | tilelang@main | 827.7 | 802.6-859.4 | 1.00 | 2054 | 2046-2075 | 1.00 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 222.4 | 213.5-232.8 | 0.27 | - | - | - |
| short-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 809.9 | 793.4-820.4 | 1.00 | 2133 | 2127-2136 | 1.00 |
| short-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 302.3 | 301.5-315.7 | 0.37 | 1496 | 1483-1501 | 0.70 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 211.6 | 207.5-213.1 | 0.26 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 876.9 | 821.4-901.4 | 1.00 | 2145 | 2070-2654 | 1.00 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 227.1 | 220.9-229.2 | 0.26 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 811.5 | 809.9-823.0 | 1.00 | 2127 | 2125-2138 | 1.00 |
| short-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 314.3 | 308.6-317.1 | 0.39 | 1483 | 1479-1492 | 0.70 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 218.4 | 214.2-224.6 | 0.27 | - | - | - |
| short-4096-csa-cp8r7 | tilelang@main | 818.1 | 806.9-826.3 | 1.00 | 2060 | 2047-2075 | 1.00 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 210.7 | 208.5-216.6 | 0.26 | - | - | - |
| short-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 803.6 | 800.8-815.5 | 1.00 | 2090 | 2074-2114 | 1.00 |
| short-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 307.7 | 303.8-313.4 | 0.38 | 1480 | 1465-1488 | 0.71 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 212.8 | 209.1-218.8 | 0.26 | - | - | - |
| short-4096-hca-cp1 | tilelang@main | 1411 | 1404-1416 | 1.00 | 3089 | 3082-3102 | 1.00 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 500.1 | 497.7-502.9 | 0.35 | - | - | - |
| short-4096-hca-cp1 | tilelang@cudnn_flashmla | 1405 | 1399-1423 | 1.00 | 3137 | 3130-3173 | 1.00 |
| short-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 568.9 | 565.8-577.0 | 0.40 | 2048 | 2032-2058 | 0.65 |
| short-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 502.3 | 500.1-508.2 | 0.36 | - | - | - |
| short-4096-hca-cp8r0 | tilelang@main | 844.5 | 832.2-863.5 | 1.00 | 2180 | 2165-2244 | 1.00 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 216.0 | 211.8-217.3 | 0.26 | - | - | - |
| short-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 834.4 | 824.0-858.1 | 1.00 | 2225 | 2218-2266 | 1.00 |
| short-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 316.5 | 311.5-321.0 | 0.38 | 1481 | 1470-1525 | 0.67 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 216.8 | 211.5-227.5 | 0.26 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 826.5 | 818.9-845.1 | 1.00 | 2157 | 2154-2171 | 1.00 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 208.2 | 206.1-217.1 | 0.25 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 857.5 | 826.0-906.4 | 1.00 | 2243 | 2241-2256 | 1.00 |
| short-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 326.3 | 322.1-332.6 | 0.38 | 1487 | 1475-1496 | 0.66 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 216.8 | 215.2-220.5 | 0.25 | - | - | - |
| short-4096-hca-cp8r7 | tilelang@main | 831.1 | 816.4-835.0 | 1.00 | 2197 | 2167-2202 | 1.00 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 214.2 | 209.4-216.6 | 0.26 | - | - | - |
| short-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 845.7 | 840.0-862.8 | 1.00 | 2228 | 2224-2237 | 1.00 |
| short-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 319.8 | 313.8-327.0 | 0.38 | 1482 | 1478-1493 | 0.67 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 220.3 | 219.2-223.7 | 0.26 | - | - | - |
| short-4096-sliding-cp1 | tilelang@main | 1312 | 1272-1315 | 1.00 | 2842 | 2832-2846 | 1.00 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 423.2 | 414.6-434.4 | 0.32 | - | - | - |
| short-4096-sliding-cp1 | tilelang@cudnn_flashmla | 1293 | 1280-1299 | 1.00 | 2891 | 2871-2905 | 1.00 |
| short-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 481.8 | 475.7-496.9 | 0.37 | 1972 | 1970-1984 | 0.68 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 420.9 | 411.5-425.0 | 0.33 | - | - | - |
| short-4096-sliding-cp8r0 | tilelang@main | 793.8 | 787.3-805.0 | 1.00 | 2036 | 2030-2048 | 1.00 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 197.4 | 192.0-199.4 | 0.25 | - | - | - |
| short-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 795.2 | 791.7-800.4 | 1.00 | 2132 | 2118-2146 | 1.00 |
| short-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 311.8 | 308.3-312.5 | 0.39 | 1501 | 1476-1532 | 0.70 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 202.3 | 199.1-205.1 | 0.25 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 794.0 | 780.5-815.6 | 1.00 | 2041 | 2033-2066 | 1.00 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 197.7 | 193.1-216.8 | 0.25 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 798.7 | 796.5-801.0 | 1.00 | 2129 | 2090-2207 | 1.00 |
| short-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 309.9 | 305.7-316.2 | 0.39 | 1482 | 1464-1499 | 0.70 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 198.4 | 197.6-206.5 | 0.25 | - | - | - |
| short-4096-sliding-cp8r7 | tilelang@main | 774.8 | 766.8-785.9 | 1.00 | 2086 | 2079-2120 | 1.00 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 195.7 | 192.6-198.1 | 0.25 | - | - | - |
| short-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 793.7 | 788.9-798.8 | 1.00 | 2116 | 2102-2121 | 1.00 |
| short-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 316.1 | 310.3-319.6 | 0.40 | 1470 | 1452-1471 | 0.69 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 203.8 | 202.9-208.1 | 0.26 | - | - | - |
| heavy-4096-csa-cp1 | tilelang@main | 1462 | 1446-1472 | 1.00 | 3545 | 3535-3555 | 1.00 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 524.9 | 521.9-530.8 | 0.36 | - | - | - |
| heavy-4096-csa-cp1 | tilelang@cudnn_flashmla | 1464 | 1463-1482 | 1.00 | 3571 | 3561-3589 | 1.00 |
| heavy-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 596.8 | 591.5-600.6 | 0.41 | 2233 | 2228-2260 | 0.63 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 531.0 | 530.2-538.4 | 0.36 | - | - | - |
| heavy-4096-csa-cp8r0 | tilelang@main | 794.4 | 789.7-815.2 | 1.00 | 2033 | 2023-2050 | 1.00 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 208.3 | 203.7-215.7 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 832.6 | 827.2-845.4 | 1.00 | 2127 | 2118-2154 | 1.00 |
| heavy-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 314.1 | 308.7-317.1 | 0.38 | 1486 | 1475-1496 | 0.70 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 214.3 | 213.4-215.8 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 817.9 | 805.9-831.3 | 1.00 | 2049 | 2038-2194 | 1.00 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 217.3 | 211.7-218.4 | 0.27 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 819.2 | 816.5-825.1 | 1.00 | 2125 | 2118-2135 | 1.00 |
| heavy-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 304.4 | 302.4-310.7 | 0.37 | 1479 | 1467-1483 | 0.70 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 213.4 | 210.3-213.6 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r7 | tilelang@main | 849.9 | 811.7-856.0 | 1.00 | 2089 | 2057-2341 | 1.00 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 215.8 | 214.5-218.0 | 0.25 | - | - | - |
| heavy-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 801.8 | 792.4-810.3 | 1.00 | 2100 | 2072-2131 | 1.00 |
| heavy-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 306.3 | 303.6-309.1 | 0.38 | 1458 | 1442-1475 | 0.69 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 211.2 | 209.9-212.2 | 0.26 | - | - | - |
| heavy-4096-hca-cp1 | tilelang@main | 1375 | 1372-1386 | 1.00 | 2982 | 2967-2988 | 1.00 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 482.7 | 480.6-489.0 | 0.35 | - | - | - |
| heavy-4096-hca-cp1 | tilelang@cudnn_flashmla | 1394 | 1383-1404 | 1.00 | 3032 | 3019-3105 | 1.00 |
| heavy-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 558.5 | 546.3-561.3 | 0.40 | 2027 | 2017-2028 | 0.67 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 491.4 | 487.4-500.0 | 0.35 | - | - | - |
| heavy-4096-hca-cp8r0 | tilelang@main | 835.5 | 828.0-844.7 | 1.00 | 2153 | 2137-2162 | 1.00 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 214.5 | 211.6-218.6 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 864.4 | 832.6-897.1 | 1.00 | 2193 | 2190-2261 | 1.00 |
| heavy-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 315.1 | 310.4-322.0 | 0.36 | 1481 | 1466-1516 | 0.68 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 212.3 | 208.3-219.7 | 0.25 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 827.9 | 820.9-840.8 | 1.00 | 2160 | 2157-2177 | 1.00 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 207.4 | 205.8-212.8 | 0.25 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 827.1 | 817.6-851.4 | 1.00 | 2240 | 2227-2256 | 1.00 |
| heavy-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 313.9 | 311.0-317.6 | 0.38 | 1489 | 1477-1533 | 0.66 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 212.2 | 211.5-215.7 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r7 | tilelang@main | 838.2 | 821.0-847.4 | 1.00 | 2180 | 2177-2181 | 1.00 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 215.7 | 212.8-219.9 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 854.3 | 825.1-855.7 | 1.00 | 2220 | 2213-2229 | 1.00 |
| heavy-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 308.7 | 302.6-316.3 | 0.36 | 1483 | 1466-1503 | 0.67 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 206.5 | 203.3-208.2 | 0.24 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang@main | 1283 | 1276-1286 | 1.00 | 2911 | 2843-2955 | 1.00 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 418.8 | 418.5-423.0 | 0.33 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang@cudnn_flashmla | 1286 | 1282-1291 | 1.00 | 2852 | 2842-2857 | 1.00 |
| heavy-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 479.2 | 474.8-482.2 | 0.37 | 1962 | 1946-1965 | 0.69 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 417.4 | 414.7-418.9 | 0.32 | - | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@main | 793.9 | 787.1-802.0 | 1.00 | 2083 | 2065-2088 | 1.00 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 198.7 | 195.6-205.1 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 810.5 | 796.3-815.8 | 1.00 | 2143 | 2126-2158 | 1.00 |
| heavy-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 313.2 | 308.1-323.1 | 0.39 | 1475 | 1467-1484 | 0.69 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 203.2 | 197.4-210.0 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 780.6 | 767.1-784.0 | 1.00 | 2050 | 2037-2053 | 1.00 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 194.3 | 190.0-200.3 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 794.3 | 787.7-822.9 | 1.00 | 2108 | 2086-2138 | 1.00 |
| heavy-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 307.9 | 303.1-336.2 | 0.39 | 1460 | 1450-1472 | 0.69 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 199.6 | 192.0-212.2 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@main | 772.0 | 762.3-785.7 | 1.00 | 2099 | 2076-2132 | 1.00 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 197.5 | 194.8-199.0 | 0.26 | - | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 788.5 | 785.0-804.0 | 1.00 | 2100 | 2096-2112 | 1.00 |
| heavy-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 315.6 | 302.8-318.5 | 0.40 | 1473 | 1467-1487 | 0.70 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 201.9 | 196.0-204.2 | 0.26 | - | - | - |
| tiny-4096-csa-cp1 | tilelang@main | 1374 | 1368-1389 | 1.00 | 2919 | 2904-2930 | 1.00 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 501.5 | 498.9-506.4 | 0.37 | - | - | - |
| tiny-4096-csa-cp1 | tilelang@cudnn_flashmla | 1391 | 1373-1400 | 1.00 | 2922 | 2906-2948 | 1.00 |
| tiny-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 555.1 | 555.0-568.0 | 0.40 | 1951 | 1947-1970 | 0.67 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 498.4 | 493.4-501.6 | 0.36 | - | - | - |
| tiny-4096-csa-cp8r0 | tilelang@main | 796.5 | 784.8-820.9 | 1.00 | 2020 | 2015-2135 | 1.00 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 210.9 | 210.1-221.9 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 792.1 | 787.5-794.5 | 1.00 | 2122 | 2110-2134 | 1.00 |
| tiny-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 306.5 | 305.8-309.5 | 0.39 | 1501 | 1499-1548 | 0.71 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 209.1 | 207.3-213.3 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 809.3 | 793.1-830.6 | 1.00 | 2034 | 2014-2048 | 1.00 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 209.6 | 207.2-216.1 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 811.7 | 796.3-821.6 | 1.00 | 2123 | 2096-2158 | 1.00 |
| tiny-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 309.0 | 302.6-320.7 | 0.38 | 1495 | 1479-1501 | 0.70 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 207.8 | 205.8-215.1 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r7 | tilelang@main | 786.0 | 777.7-790.5 | 1.00 | 2043 | 2030-2080 | 1.00 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 210.6 | 206.3-211.8 | 0.27 | - | - | - |
| tiny-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 789.1 | 785.5-799.4 | 1.00 | 2103 | 2096-2109 | 1.00 |
| tiny-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 299.5 | 296.3-307.4 | 0.38 | 1476 | 1467-1485 | 0.70 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 208.6 | 208.4-209.2 | 0.26 | - | - | - |
| tiny-4096-hca-cp1 | tilelang@main | 1227 | 1218-1246 | 1.00 | 2409 | 2400-2413 | 1.00 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 412.3 | 408.0-430.9 | 0.34 | - | - | - |
| tiny-4096-hca-cp1 | tilelang@cudnn_flashmla | 1224 | 1214-1259 | 1.00 | 2462 | 2447-2467 | 1.00 |
| tiny-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 480.8 | 477.9-487.3 | 0.39 | 1778 | 1770-1779 | 0.72 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 417.2 | 414.4-420.8 | 0.34 | - | - | - |
| tiny-4096-hca-cp8r0 | tilelang@main | 770.4 | 766.3-773.0 | 1.00 | 2033 | 2020-2052 | 1.00 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 194.8 | 192.6-198.4 | 0.25 | - | - | - |
| tiny-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 791.6 | 784.3-795.8 | 1.00 | 2123 | 2101-2135 | 1.00 |
| tiny-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 313.7 | 308.7-317.4 | 0.40 | 1465 | 1454-1466 | 0.69 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 201.4 | 199.4-204.5 | 0.25 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 767.4 | 763.6-775.5 | 1.00 | 2023 | 2018-2038 | 1.00 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 196.6 | 192.9-202.8 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 794.3 | 774.3-796.1 | 1.00 | 2154 | 2137-2172 | 1.00 |
| tiny-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 308.7 | 306.5-313.3 | 0.39 | 1492 | 1479-1495 | 0.69 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 199.7 | 195.8-202.7 | 0.25 | - | - | - |
| tiny-4096-hca-cp8r7 | tilelang@main | 784.1 | 768.8-793.5 | 1.00 | 2053 | 2045-2065 | 1.00 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 201.5 | 196.4-202.1 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 791.8 | 773.3-808.9 | 1.00 | 2122 | 2108-2144 | 1.00 |
| tiny-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 310.7 | 307.9-313.0 | 0.39 | 1472 | 1455-1477 | 0.69 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 202.2 | 196.5-203.7 | 0.26 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang@main | 1226 | 1215-1234 | 1.00 | 2407 | 2386-2415 | 1.00 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 408.8 | 408.3-411.3 | 0.33 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang@cudnn_flashmla | 1228 | 1223-1245 | 1.00 | 2485 | 2464-2495 | 1.00 |
| tiny-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 478.9 | 476.1-486.1 | 0.39 | 1789 | 1786-1790 | 0.72 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 416.8 | 415.8-421.4 | 0.34 | - | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@main | 775.3 | 771.9-783.8 | 1.00 | 2030 | 2005-2076 | 1.00 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 200.7 | 197.0-203.9 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 777.3 | 771.2-806.9 | 1.00 | 2119 | 2111-2123 | 1.00 |
| tiny-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 304.0 | 302.1-307.8 | 0.39 | 1475 | 1451-1480 | 0.70 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 195.1 | 192.5-198.8 | 0.25 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 781.6 | 770.1-787.6 | 1.00 | 2040 | 2028-2056 | 1.00 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 201.3-214.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 806.3 | 796.4-819.0 | 1.00 | 2114 | 2100-2141 | 1.00 |
| tiny-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 304.5 | 300.6-306.7 | 0.38 | 1542 | 1480-1581 | 0.73 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 194.8 | 193.1-198.7 | 0.24 | - | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@main | 786.4 | 772.9-790.8 | 1.00 | 2081 | 2076-2129 | 1.00 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 202.2 | 194.2-206.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 805.0 | 795.1-864.4 | 1.00 | 2150 | 2138-2166 | 1.00 |
| tiny-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 314.9 | 314.5-329.8 | 0.39 | 1515 | 1485-1522 | 0.70 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 203.7 | 200.6-205.8 | 0.25 | - | - | - |
| single-16384-csa-cp1 | tilelang@main | 5918 | 5886-6013 | 1.00 | 23560 | 23556-23572 | 1.00 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 2570 | 2560-2654 | 0.43 | - | - | - |
| single-16384-csa-cp1 | tilelang@cudnn_flashmla | 6020 | 5961-6091 | 1.00 | 23876 | 23651-23919 | 1.00 |
| single-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 2637 | 2628-2845 | 0.44 | 12754 | 12664-12773 | 0.53 |
| single-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 2677 | 2660-2753 | 0.44 | - | - | - |
| single-16384-csa-cp8r0 | tilelang@main | 1232 | 1216-1235 | 1.00 | 3243 | 3230-3250 | 1.00 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 417.6 | 414.9-421.8 | 0.34 | - | - | - |
| single-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 1241 | 1227-1256 | 1.00 | 3299 | 3276-3361 | 1.00 |
| single-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 482.1 | 481.7-488.4 | 0.39 | 1966 | 1965-1975 | 0.60 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 417.2 | 414.2-419.0 | 0.34 | - | - | - |
| single-16384-csa-cp8r4 | tilelang@main | 1409 | 1392-1436 | 1.00 | 4079 | 4072-4087 | 1.00 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 495.2 | 492.1-501.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 1423 | 1409-1434 | 1.00 | 4123 | 4119-4149 | 1.00 |
| single-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 549.0 | 548.0-559.2 | 0.39 | 2331 | 2317-2339 | 0.57 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 488.5 | 485.8-494.5 | 0.34 | - | - | - |
| single-16384-csa-cp8r7 | tilelang@main | 1405 | 1396-1424 | 1.00 | 4104 | 4096-4125 | 1.00 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 496.7 | 493.8-498.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 1426 | 1422-1468 | 1.00 | 4126 | 4099-4144 | 1.00 |
| single-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 556.8 | 555.8-569.9 | 0.39 | 2321 | 2317-2328 | 0.56 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 499.0 | 498.1-505.3 | 0.35 | - | - | - |
| single-16384-hca-cp1 | tilelang@main | 3573 | 3561-3617 | 1.00 | 10905 | 10899-10911 | 1.00 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1490 | 1486-1513 | 0.42 | - | - | - |
| single-16384-hca-cp1 | tilelang@cudnn_flashmla | 3582 | 3561-3590 | 1.00 | 10869 | 10861-10875 | 1.00 |
| single-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1570 | 1566-1575 | 0.44 | 6532 | 6524-6536 | 0.60 |
| single-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1488 | 1488-1493 | 0.42 | - | - | - |
| single-16384-hca-cp8r0 | tilelang@main | 1081 | 1067-1109 | 1.00 | 2422 | 2395-2446 | 1.00 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 340.5 | 338.5-348.6 | 0.32 | - | - | - |
| single-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 1056 | 1052-1065 | 1.00 | 2470 | 2448-2556 | 1.00 |
| single-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 399.6 | 396.5-402.9 | 0.38 | 1596 | 1593-1604 | 0.65 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 337.8 | 333.8-342.9 | 0.32 | - | - | - |
| single-16384-hca-cp8r4 | tilelang@main | 1097 | 1094-1101 | 1.00 | 2665 | 2641-2689 | 1.00 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 339.9 | 338.0-344.9 | 0.31 | - | - | - |
| single-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 1112 | 1107-1124 | 1.00 | 2776 | 2765-2784 | 1.00 |
| single-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 401.8 | 400.3-407.6 | 0.36 | 1719 | 1702-1729 | 0.62 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 341.7 | 338.7-344.3 | 0.31 | - | - | - |
| single-16384-hca-cp8r7 | tilelang@main | 1106 | 1097-1108 | 1.00 | 2819 | 2816-2866 | 1.00 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 343.8 | 338.7-344.0 | 0.31 | - | - | - |
| single-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 1111 | 1100-1114 | 1.00 | 2827 | 2805-2841 | 1.00 |
| single-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 403.2 | 398.5-409.0 | 0.36 | 1727 | 1722-1732 | 0.61 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 342.2 | 338.2-345.1 | 0.31 | - | - | - |
| single-16384-sliding-cp1 | tilelang@main | 2973 | 2970-2979 | 1.00 | 8286 | 8275-8288 | 1.00 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 1164 | 1159-1174 | 0.39 | - | - | - |
| single-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2988 | 2969-3007 | 1.00 | 8300 | 8275-8313 | 1.00 |
| single-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1242 | 1238-1244 | 0.42 | 5335 | 5320-5338 | 0.64 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1167 | 1158-1169 | 0.39 | - | - | - |
| single-16384-sliding-cp8r0 | tilelang@main | 1019 | 1002-1035 | 1.00 | 2333 | 2317-2348 | 1.00 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 299.8 | 294.8-317.2 | 0.29 | - | - | - |
| single-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 1018 | 1007-1025 | 1.00 | 2406 | 2396-2424 | 1.00 |
| single-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 354.5 | 353.9-356.9 | 0.35 | 1546 | 1534-1576 | 0.64 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 293.4 | 292.5-298.3 | 0.29 | - | - | - |
| single-16384-sliding-cp8r4 | tilelang@main | 1004 | 986.3-1009 | 1.00 | 2390 | 2382-2476 | 1.00 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 296.7 | 289.8-313.0 | 0.30 | - | - | - |
| single-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 1004 | 996.7-1010 | 1.00 | 2397 | 2391-2415 | 1.00 |
| single-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 350.6 | 347.8-355.7 | 0.35 | 1552 | 1534-1558 | 0.65 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 289.6 | 288.3-295.2 | 0.29 | - | - | - |
| single-16384-sliding-cp8r7 | tilelang@main | 996.2 | 987.6-1004 | 1.00 | 2389 | 2385-2438 | 1.00 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 294.0 | 292.2-302.3 | 0.30 | - | - | - |
| single-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 1012 | 1010-1030 | 1.00 | 2414 | 2411-2439 | 1.00 |
| single-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 353.7 | 353.3-355.5 | 0.35 | 1546 | 1539-1570 | 0.64 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 295.5 | 291.3-296.4 | 0.29 | - | - | - |
| short-16384-csa-cp1 | tilelang@main | 4509 | 4494-4544 | 1.00 | 15976 | 15975-16005 | 1.00 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 1957 | 1952-1959 | 0.43 | - | - | - |
| short-16384-csa-cp1 | tilelang@cudnn_flashmla | 4520 | 4486-4605 | 1.00 | 16071 | 16048-16078 | 1.00 |
| short-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 2013 | 2010-2049 | 0.45 | 8883 | 8820-8910 | 0.55 |
| short-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1953 | 1947-1981 | 0.43 | - | - | - |
| short-16384-csa-cp8r0 | tilelang@main | 1175 | 1156-1200 | 1.00 | 2997 | 2984-3003 | 1.00 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 390.1 | 384.5-394.8 | 0.33 | - | - | - |
| short-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 1174 | 1172-1179 | 1.00 | 3044 | 3038-3062 | 1.00 |
| short-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 451.8 | 449.7-453.1 | 0.38 | 1852 | 1839-1865 | 0.61 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 386.2 | 385.7-393.9 | 0.33 | - | - | - |
| short-16384-csa-cp8r4 | tilelang@main | 1249 | 1236-1261 | 1.00 | 3387 | 3384-3406 | 1.00 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 431.6 | 429.3-432.5 | 0.35 | - | - | - |
| short-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 1249 | 1247-1258 | 1.00 | 3450 | 3441-3453 | 1.00 |
| short-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 490.0 | 487.0-491.2 | 0.39 | 2029 | 2024-2037 | 0.59 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 431.8 | 426.4-434.1 | 0.35 | - | - | - |
| short-16384-csa-cp8r7 | tilelang@main | 1211 | 1195-1219 | 1.00 | 3241 | 3236-3255 | 1.00 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 408.0 | 406.4-415.0 | 0.34 | - | - | - |
| short-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 1225 | 1218-1239 | 1.00 | 3233 | 3227-3261 | 1.00 |
| short-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 473.0 | 470.3-473.8 | 0.39 | 1947 | 1934-1958 | 0.60 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 413.7 | 411.7-416.3 | 0.34 | - | - | - |
| short-16384-hca-cp1 | tilelang@main | 3337 | 3333-3349 | 1.00 | 9225 | 9192-9255 | 1.00 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 1463 | 1461-1465 | 0.44 | - | - | - |
| short-16384-hca-cp1 | tilelang@cudnn_flashmla | 3326 | 3321-3335 | 1.00 | 9144 | 9125-9164 | 1.00 |
| short-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1543 | 1538-1549 | 0.46 | 5882 | 5877-5903 | 0.64 |
| short-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1462 | 1459-1466 | 0.44 | - | - | - |
| short-16384-hca-cp8r0 | tilelang@main | 1070 | 1063-1086 | 1.00 | 2499 | 2480-2505 | 1.00 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 338.9 | 332.9-342.3 | 0.32 | - | - | - |
| short-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 1083 | 1077-1108 | 1.00 | 2519 | 2510-2551 | 1.00 |
| short-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 402.1 | 395.8-406.1 | 0.37 | 1575 | 1573-1591 | 0.63 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 337.6 | 335.1-343.1 | 0.31 | - | - | - |
| short-16384-hca-cp8r4 | tilelang@main | 1080 | 1076-1091 | 1.00 | 2532 | 2525-2555 | 1.00 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 337.4 | 334.8-343.4 | 0.31 | - | - | - |
| short-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 1088 | 1081-1093 | 1.00 | 2606 | 2589-2615 | 1.00 |
| short-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 402.2 | 397.2-409.2 | 0.37 | 1609 | 1603-1615 | 0.62 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 341.3 | 340.7-344.0 | 0.31 | - | - | - |
| short-16384-hca-cp8r7 | tilelang@main | 1081 | 1069-1086 | 1.00 | 2558 | 2548-2561 | 1.00 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 338.8 | 338.8-342.2 | 0.31 | - | - | - |
| short-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 1082 | 1077-1092 | 1.00 | 2558 | 2547-2567 | 1.00 |
| short-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 405.3 | 403.8-408.2 | 0.37 | 1595 | 1591-1603 | 0.62 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 340.8 | 338.7-346.1 | 0.32 | - | - | - |
| short-16384-sliding-cp1 | tilelang@main | 2985 | 2970-3000 | 1.00 | 8155 | 8151-8192 | 1.00 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 1169 | 1162-1175 | 0.39 | - | - | - |
| short-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2957 | 2946-2980 | 1.00 | 8143 | 8134-8156 | 1.00 |
| short-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1237 | 1235-1241 | 0.42 | 5273 | 5260-5280 | 0.65 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1163 | 1158-1165 | 0.39 | - | - | - |
| short-16384-sliding-cp8r0 | tilelang@main | 1001 | 987.1-1006 | 1.00 | 2305 | 2292-2331 | 1.00 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 296.3 | 293.4-300.4 | 0.30 | - | - | - |
| short-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 1002 | 998.4-1028 | 1.00 | 2404 | 2399-2407 | 1.00 |
| short-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 359.4 | 356.7-363.8 | 0.36 | 1557 | 1541-1565 | 0.65 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 293.6 | 293.1-296.8 | 0.29 | - | - | - |
| short-16384-sliding-cp8r4 | tilelang@main | 993.9 | 985.0-1070 | 1.00 | 2370 | 2342-2396 | 1.00 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 294.4 | 287.7-309.9 | 0.30 | - | - | - |
| short-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 1012 | 999.6-1018 | 1.00 | 2384 | 2376-2397 | 1.00 |
| short-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 353.8 | 350.3-358.2 | 0.35 | 1533 | 1528-1551 | 0.64 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 290.0 | 288.9-293.9 | 0.29 | - | - | - |
| short-16384-sliding-cp8r7 | tilelang@main | 1001 | 987.3-1004 | 1.00 | 2361 | 2358-2371 | 1.00 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 294.5 | 290.2-294.9 | 0.29 | - | - | - |
| short-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 1016 | 1010-1028 | 1.00 | 2409 | 2394-2427 | 1.00 |
| short-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 362.5 | 360.3-364.2 | 0.36 | 1546 | 1544-1560 | 0.64 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 297.6 | 296.5-303.3 | 0.29 | - | - | - |
| heavy-16384-csa-cp1 | tilelang@main | 3969 | 3954-4002 | 1.00 | 12997 | 12956-13003 | 1.00 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1726 | 1725-1731 | 0.43 | - | - | - |
| heavy-16384-csa-cp1 | tilelang@cudnn_flashmla | 3950 | 3935-4000 | 1.00 | 12893 | 12891-12904 | 1.00 |
| heavy-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1789 | 1787-1790 | 0.45 | 7443 | 7432-7472 | 0.58 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1722 | 1717-1728 | 0.44 | - | - | - |
| heavy-16384-csa-cp8r0 | tilelang@main | 1092 | 1069-1121 | 1.00 | 2575 | 2548-2608 | 1.00 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 346.5 | 345.0-355.2 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 1094 | 1091-1119 | 1.00 | 2661 | 2637-2668 | 1.00 |
| heavy-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 413.5 | 411.6-421.0 | 0.38 | 1666 | 1652-1673 | 0.63 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 353.0 | 351.7-355.6 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r4 | tilelang@main | 1117 | 1112-1126 | 1.00 | 2762 | 2761-2778 | 1.00 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 362.7 | 357.6-370.0 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 1128 | 1120-1130 | 1.00 | 2847 | 2826-2856 | 1.00 |
| heavy-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 434.2 | 430.0-437.4 | 0.39 | 1745 | 1744-1752 | 0.61 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 370.6 | 366.1-373.2 | 0.33 | - | - | - |
| heavy-16384-csa-cp8r7 | tilelang@main | 1264 | 1261-1266 | 1.00 | 3482 | 3467-3491 | 1.00 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 428.5 | 427.4-434.8 | 0.34 | - | - | - |
| heavy-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 1281 | 1269-1283 | 1.00 | 3480 | 3460-3569 | 1.00 |
| heavy-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 498.9 | 498.3-500.7 | 0.39 | 2041 | 2022-2049 | 0.59 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 438.1 | 433.5-441.1 | 0.34 | - | - | - |
| heavy-16384-hca-cp1 | tilelang@main | 3252 | 3222-3290 | 1.00 | 8705 | 8691-8740 | 1.00 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 1404 | 1396-1410 | 0.43 | - | - | - |
| heavy-16384-hca-cp1 | tilelang@cudnn_flashmla | 3226 | 3218-3229 | 1.00 | 8678 | 8631-8694 | 1.00 |
| heavy-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1479 | 1467-1495 | 0.46 | 5630 | 5625-5645 | 0.65 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1402 | 1401-1403 | 0.43 | - | - | - |
| heavy-16384-hca-cp8r0 | tilelang@main | 1068 | 1060-1096 | 1.00 | 2448 | 2445-2477 | 1.00 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 335.5 | 331.0-346.8 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 1055 | 1050-1071 | 1.00 | 2460 | 2452-2499 | 1.00 |
| heavy-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 386.4 | 385.0-393.5 | 0.37 | 1545 | 1535-1563 | 0.63 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 325.7 | 324.7-330.1 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r4 | tilelang@main | 1070 | 1066-1081 | 1.00 | 2488 | 2479-2500 | 1.00 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 341.7 | 338.1-344.5 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 1071 | 1061-1077 | 1.00 | 2550 | 2541-2574 | 1.00 |
| heavy-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 396.4 | 395.0-401.2 | 0.37 | 1598 | 1592-1608 | 0.63 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 336.2 | 333.7-337.1 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r7 | tilelang@main | 1088 | 1082-1106 | 1.00 | 2621 | 2616-2649 | 1.00 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 346.1 | 343.6-346.5 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 1099 | 1094-1117 | 1.00 | 2639 | 2621-2668 | 1.00 |
| heavy-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 411.9 | 406.2-413.8 | 0.37 | 1636 | 1605-1827 | 0.62 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 349.5 | 348.9-352.3 | 0.32 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang@main | 2979 | 2958-3008 | 1.00 | 7887 | 7879-7894 | 1.00 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 1173 | 1173-1183 | 0.39 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2947 | 2943-2973 | 1.00 | 7861 | 7861-7888 | 1.00 |
| heavy-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1242 | 1239-1246 | 0.42 | 5154 | 5149-5171 | 0.66 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1167 | 1163-1177 | 0.40 | - | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@main | 1013 | 1002-1040 | 1.00 | 2292 | 2285-2305 | 1.00 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 297.7 | 295.7-301.9 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 996.5 | 992.5-1002 | 1.00 | 2350 | 2330-2371 | 1.00 |
| heavy-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 357.4 | 353.7-366.6 | 0.36 | 1526 | 1521-1542 | 0.65 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 295.3 | 293.3-300.1 | 0.30 | - | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@main | 982.7 | 980.0-1006 | 1.00 | 2331 | 2316-2337 | 1.00 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 293.1 | 290.1-298.7 | 0.30 | - | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 1014 | 1001-1068 | 1.00 | 2355 | 2348-2360 | 1.00 |
| heavy-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 356.3 | 355.5-358.2 | 0.35 | 1523 | 1518-1523 | 0.65 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 294.6 | 293.4-296.2 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@main | 1022 | 994.6-1025 | 1.00 | 2405 | 2394-2439 | 1.00 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 296.4 | 293.9-310.3 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 1016 | 1014-1036 | 1.00 | 2491 | 2474-2521 | 1.00 |
| heavy-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 360.7 | 353.5-366.5 | 0.35 | 1572 | 1566-1580 | 0.63 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 300.6 | 293.6-301.5 | 0.30 | - | - | - |
| tiny-16384-csa-cp1 | tilelang@main | 3287 | 3284-3298 | 1.00 | 8726 | 8702-8759 | 1.00 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 1476 | 1471-1478 | 0.45 | - | - | - |
| tiny-16384-csa-cp1 | tilelang@cudnn_flashmla | 3311 | 3294-3346 | 1.00 | 8697 | 8696-8713 | 1.00 |
| tiny-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1537 | 1532-1538 | 0.46 | 5550 | 5543-5557 | 0.64 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1477 | 1474-1478 | 0.45 | - | - | - |
| tiny-16384-csa-cp8r0 | tilelang@main | 1043 | 1041-1055 | 1.00 | 2376 | 2356-2382 | 1.00 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 334.5 | 334.0-340.0 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 1061 | 1052-1117 | 1.00 | 2418 | 2409-2441 | 1.00 |
| tiny-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 399.7 | 395.9-407.7 | 0.38 | 1545 | 1539-1550 | 0.64 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 337.1 | 331.5-343.7 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r4 | tilelang@main | 1048 | 1040-1052 | 1.00 | 2358 | 2354-2374 | 1.00 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 342.2 | 335.7-346.6 | 0.33 | - | - | - |
| tiny-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 1046 | 1040-1069 | 1.00 | 2397 | 2381-2410 | 1.00 |
| tiny-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 397.7 | 395.7-398.6 | 0.38 | 1529 | 1521-1544 | 0.64 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 343.2 | 339.4-344.8 | 0.33 | - | - | - |
| tiny-16384-csa-cp8r7 | tilelang@main | 1032 | 1022-1076 | 1.00 | 2369 | 2363-2372 | 1.00 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 334.5 | 327.7-343.5 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 1056 | 1049-1064 | 1.00 | 2373 | 2362-2380 | 1.00 |
| tiny-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 393.8 | 391.1-398.7 | 0.37 | 1511 | 1505-1519 | 0.64 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 335.9 | 333.6-337.2 | 0.32 | - | - | - |
| tiny-16384-hca-cp1 | tilelang@main | 2707 | 2700-2714 | 1.00 | 6184 | 6177-6195 | 1.00 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 1176 | 1174-1178 | 0.43 | - | - | - |
| tiny-16384-hca-cp1 | tilelang@cudnn_flashmla | 2741 | 2731-2744 | 1.00 | 6200 | 6186-6205 | 1.00 |
| tiny-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1262 | 1260-1277 | 0.46 | 4481 | 4478-4490 | 0.72 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1179 | 1176-1184 | 0.43 | - | - | - |
| tiny-16384-hca-cp8r0 | tilelang@main | 981.1 | 976.9-1003 | 1.00 | 2144 | 2130-2255 | 1.00 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 296.6 | 289.3-309.1 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 972.0 | 971.4-984.0 | 1.00 | 2174 | 2167-2186 | 1.00 |
| tiny-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 359.4 | 354.1-360.6 | 0.37 | 1494 | 1490-1499 | 0.69 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 293.6 | 292.4-295.2 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r4 | tilelang@main | 968.2 | 959.1-972.4 | 1.00 | 2095 | 2090-2123 | 1.00 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 293.1 | 289.9-295.4 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 987.1 | 978.3-1039 | 1.00 | 2174 | 2156-2184 | 1.00 |
| tiny-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 358.9 | 356.2-361.0 | 0.36 | 1501 | 1491-1510 | 0.69 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 295.3 | 295.0-297.4 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r7 | tilelang@main | 955.4 | 942.5-968.8 | 1.00 | 2130 | 2120-2172 | 1.00 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 294.3 | 290.2-298.9 | 0.31 | - | - | - |
| tiny-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 974.4 | 968.9-983.5 | 1.00 | 2147 | 2120-2153 | 1.00 |
| tiny-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 359.3 | 357.5-363.0 | 0.37 | 1485 | 1476-1498 | 0.69 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 299.1 | 298.1-301.5 | 0.31 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang@main | 2728 | 2711-2754 | 1.00 | 6163 | 6161-6164 | 1.00 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 1181 | 1174-1182 | 0.43 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2731 | 2721-2744 | 1.00 | 6199 | 6194-6204 | 1.00 |
| tiny-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1255 | 1252-1261 | 0.46 | 4474 | 4460-4477 | 0.72 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1180 | 1177-1182 | 0.43 | - | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@main | 962.6 | 956.2-977.8 | 1.00 | 2142 | 2126-2157 | 1.00 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 297.0 | 295.1-301.5 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 990.1 | 976.8-998.1 | 1.00 | 2187 | 2184-2215 | 1.00 |
| tiny-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 360.3 | 359.7-369.2 | 0.36 | 1516 | 1510-1536 | 0.69 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 299.2 | 298.5-302.6 | 0.30 | - | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@main | 962.2 | 954.6-977.3 | 1.00 | 2174 | 2113-2294 | 1.00 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 299.2 | 290.4-299.6 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 974.6 | 961.8-980.6 | 1.00 | 2140 | 2131-2149 | 1.00 |
| tiny-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 355.4 | 347.5-364.5 | 0.36 | 1483 | 1462-1495 | 0.69 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 292.3 | 290.8-297.5 | 0.30 | - | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@main | 961.0 | 949.7-972.0 | 1.00 | 2140 | 2129-2174 | 1.00 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 293.2 | 292.1-301.2 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 960.9 | 954.1-989.5 | 1.00 | 2149 | 2139-2205 | 1.00 |
| tiny-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 357.2 | 354.7-360.6 | 0.37 | 1503 | 1495-1551 | 0.70 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 293.5 | 291.9-296.2 | 0.31 | - | - | - |
| single-49208-csa-cp1 | tilelang@main | 16424 | 16369-16661 | 1.00 | 70708 | 70691-70733 | 1.00 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 8309 | 8259-8522 | 0.51 | - | - | - |
| single-49208-csa-cp1 | tilelang@cudnn_flashmla | 16483 | 16431-16617 | 1.00 | 71011 | 70739-71067 | 1.00 |
| single-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 8387 | 8364-8585 | 0.51 | 39439 | 39343-39467 | 0.56 |
| single-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 8542 | 8533-8592 | 0.52 | - | - | - |
| single-49208-csa-cp8r0 | tilelang@main | 2556 | 2547-2584 | 1.00 | 8923 | 8921-8925 | 1.00 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 1026 | 1020-1028 | 0.40 | - | - | - |
| single-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 2584 | 2566-2602 | 1.00 | 9060 | 9032-9081 | 1.00 |
| single-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 1101 | 1094-1106 | 0.43 | 4770 | 4766-4779 | 0.53 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 1022 | 1019-1024 | 0.40 | - | - | - |
| single-49208-csa-cp8r4 | tilelang@main | 2744 | 2725-2754 | 1.00 | 10025 | 10005-10044 | 1.00 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1100 | 1100-1108 | 0.40 | - | - | - |
| single-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 2762 | 2737-2813 | 1.00 | 10126 | 10107-10141 | 1.00 |
| single-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1177 | 1173-1192 | 0.43 | 5236 | 5230-5249 | 0.52 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1106 | 1099-1107 | 0.40 | - | - | - |
| single-49208-csa-cp8r7 | tilelang@main | 2745 | 2736-2763 | 1.00 | 10361 | 10302-10387 | 1.00 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1105 | 1101-1108 | 0.40 | - | - | - |
| single-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 2779 | 2765-2794 | 1.00 | 10456 | 10414-10494 | 1.00 |
| single-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1174 | 1173-1181 | 0.42 | 5311 | 5300-5348 | 0.51 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1105 | 1102-1113 | 0.40 | - | - | - |
| single-49208-hca-cp1 | tilelang@main | 11496 | 11468-11639 | 1.00 | 42559 | 42547-42587 | 1.00 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 5402 | 5373-5651 | 0.47 | - | - | - |
| single-49208-hca-cp1 | tilelang@cudnn_flashmla | 11494 | 11470-11586 | 1.00 | 42813 | 42576-42848 | 1.00 |
| single-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 5524 | 5454-5657 | 0.48 | 25025 | 25011-25056 | 0.58 |
| single-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5616 | 5592-5650 | 0.49 | - | - | - |
| single-49208-hca-cp8r0 | tilelang@main | 1710 | 1698-1758 | 1.00 | 4247 | 4246-4326 | 1.00 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 670.5 | 667.3-673.7 | 0.39 | - | - | - |
| single-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 1731 | 1715-1760 | 1.00 | 4283 | 4276-4288 | 1.00 |
| single-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 740.0 | 737.7-759.8 | 0.43 | 2727 | 2719-2735 | 0.64 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 671.5 | 668.8-677.2 | 0.39 | - | - | - |
| single-49208-hca-cp8r4 | tilelang@main | 2145 | 2139-2186 | 1.00 | 6565 | 6561-6572 | 1.00 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 801.2 | 799.3-802.6 | 0.37 | - | - | - |
| single-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 2163 | 2149-2169 | 1.00 | 6591 | 6581-6597 | 1.00 |
| single-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 866.1 | 860.3-880.1 | 0.40 | 3623 | 3620-3628 | 0.55 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 800.6 | 797.4-805.2 | 0.37 | - | - | - |
| single-49208-hca-cp8r7 | tilelang@main | 2430 | 2427-2432 | 1.00 | 8244 | 8226-8262 | 1.00 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 944.5 | 939.2-955.3 | 0.39 | - | - | - |
| single-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 2446 | 2439-2452 | 1.00 | 8266 | 8238-8272 | 1.00 |
| single-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1012 | 1008-1016 | 0.41 | 4364 | 4352-4369 | 0.53 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 943.6 | 934.6-945.7 | 0.39 | - | - | - |
| single-49208-sliding-cp1 | tilelang@main | 7433 | 7429-7462 | 1.00 | 22888 | 22878-22898 | 1.00 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 3125 | 3112-3128 | 0.42 | - | - | - |
| single-49208-sliding-cp1 | tilelang@cudnn_flashmla | 7425 | 7416-7439 | 1.00 | 22910 | 22906-22927 | 1.00 |
| single-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3223 | 3221-3224 | 0.43 | 15385 | 15380-15404 | 0.67 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3115 | 3107-3122 | 0.42 | - | - | - |
| single-49208-sliding-cp8r0 | tilelang@main | 1574 | 1566-1595 | 1.00 | 3745 | 3734-3748 | 1.00 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 543.0 | 540.3-547.6 | 0.35 | - | - | - |
| single-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 1596 | 1594-1602 | 1.00 | 3769 | 3757-3781 | 1.00 |
| single-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 612.7 | 604.8-618.0 | 0.38 | 2539 | 2531-2560 | 0.67 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 545.0 | 542.5-546.6 | 0.34 | - | - | - |
| single-49208-sliding-cp8r4 | tilelang@main | 1557 | 1554-1558 | 1.00 | 3783 | 3768-3790 | 1.00 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 539.1 | 537.2-545.8 | 0.35 | - | - | - |
| single-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 1581 | 1578-1589 | 1.00 | 3782 | 3776-3788 | 1.00 |
| single-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 605.5 | 602.3-608.8 | 0.38 | 2533 | 2526-2541 | 0.67 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 540.1 | 538.4-541.0 | 0.34 | - | - | - |
| single-49208-sliding-cp8r7 | tilelang@main | 1572 | 1565-1598 | 1.00 | 3792 | 3777-3798 | 1.00 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 540.1 | 536.8-541.1 | 0.34 | - | - | - |
| single-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 1603 | 1586-1615 | 1.00 | 3845 | 3830-3892 | 1.00 |
| single-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 606.4 | 602.8-610.2 | 0.38 | 2611 | 2552-2650 | 0.68 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 541.2 | 539.5-543.1 | 0.34 | - | - | - |
| short-49208-csa-cp1 | tilelang@main | 12988 | 12944-13072 | 1.00 | 51070 | 51061-51101 | 1.00 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 6227 | 6191-6569 | 0.48 | - | - | - |
| short-49208-csa-cp1 | tilelang@cudnn_flashmla | 13047 | 12989-13146 | 1.00 | 51348 | 51179-51419 | 1.00 |
| short-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 6398 | 6349-6635 | 0.49 | 29401 | 29344-29525 | 0.57 |
| short-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 6617 | 6606-6658 | 0.51 | - | - | - |
| short-49208-csa-cp8r0 | tilelang@main | 2037 | 2013-2042 | 1.00 | 6012 | 6007-6017 | 1.00 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 791.1 | 785.7-795.3 | 0.39 | - | - | - |
| short-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 2041 | 2028-2046 | 1.00 | 6006 | 6000-6017 | 1.00 |
| short-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 860.1 | 855.0-864.1 | 0.42 | 3388 | 3383-3394 | 0.56 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 790.4 | 788.0-795.1 | 0.39 | - | - | - |
| short-49208-csa-cp8r4 | tilelang@main | 2386 | 2373-2409 | 1.00 | 8086 | 8074-8104 | 1.00 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 959.5 | 958.1-965.9 | 0.40 | - | - | - |
| short-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 2397 | 2396-2415 | 1.00 | 8147 | 8138-8149 | 1.00 |
| short-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1024 | 1020-1031 | 0.43 | 4336 | 4331-4362 | 0.53 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 954.7 | 952.2-956.3 | 0.40 | - | - | - |
| short-49208-csa-cp8r7 | tilelang@main | 2161 | 2150-2181 | 1.00 | 6809 | 6800-6810 | 1.00 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 858.3 | 856.4-860.1 | 0.40 | - | - | - |
| short-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 2180 | 2164-2195 | 1.00 | 6786 | 6779-6791 | 1.00 |
| short-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 919.0 | 917.7-922.9 | 0.42 | 3739 | 3737-3743 | 0.55 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 856.4 | 853.7-871.8 | 0.39 | - | - | - |
| short-49208-hca-cp1 | tilelang@main | 8505 | 8498-8522 | 1.00 | 26175 | 26167-26201 | 1.00 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 4103 | 4093-4107 | 0.48 | - | - | - |
| short-49208-hca-cp1 | tilelang@cudnn_flashmla | 8486 | 8462-8489 | 1.00 | 26185 | 26180-26197 | 1.00 |
| short-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 4176 | 4172-4192 | 0.49 | 17410 | 17389-17417 | 0.66 |
| short-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4096 | 4088-4098 | 0.48 | - | - | - |
| short-49208-hca-cp8r0 | tilelang@main | 1701 | 1696-1716 | 1.00 | 4068 | 4063-4077 | 1.00 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 646.6 | 644.3-648.9 | 0.38 | - | - | - |
| short-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 1755 | 1731-1767 | 1.00 | 4089 | 4082-4092 | 1.00 |
| short-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 721.7 | 720.6-731.6 | 0.41 | 2611 | 2600-2618 | 0.64 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 653.5 | 651.7-664.0 | 0.37 | - | - | - |
| short-49208-hca-cp8r4 | tilelang@main | 1747 | 1727-1765 | 1.00 | 4187 | 4177-4188 | 1.00 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 665.3 | 661.2-667.1 | 0.38 | - | - | - |
| short-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 1751 | 1735-1756 | 1.00 | 4226 | 4225-4228 | 1.00 |
| short-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 735.3 | 729.0-735.8 | 0.42 | 2692 | 2684-2716 | 0.64 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 666.0 | 664.6-669.0 | 0.38 | - | - | - |
| short-49208-hca-cp8r7 | tilelang@main | 1733 | 1727-1755 | 1.00 | 4169 | 4154-4176 | 1.00 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 664.6 | 659.7-670.8 | 0.38 | - | - | - |
| short-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 1750 | 1738-1756 | 1.00 | 4148 | 4146-4160 | 1.00 |
| short-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 736.7 | 734.9-741.7 | 0.42 | 2630 | 2616-2644 | 0.63 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 669.0 | 665.1-672.2 | 0.38 | - | - | - |
| short-49208-sliding-cp1 | tilelang@main | 7423 | 7418-7435 | 1.00 | 22632 | 22622-22641 | 1.00 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 3123 | 3117-3127 | 0.42 | - | - | - |
| short-49208-sliding-cp1 | tilelang@cudnn_flashmla | 7422 | 7405-7436 | 1.00 | 22617 | 22608-22626 | 1.00 |
| short-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3224 | 3222-3230 | 0.43 | 15217 | 15212-15223 | 0.67 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3122 | 3114-3123 | 0.42 | - | - | - |
| short-49208-sliding-cp8r0 | tilelang@main | 1565 | 1552-1569 | 1.00 | 3663 | 3651-3680 | 1.00 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 541.0 | 537.3-548.1 | 0.35 | - | - | - |
| short-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 1574 | 1566-1587 | 1.00 | 3688 | 3682-3694 | 1.00 |
| short-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 610.6 | 608.4-615.5 | 0.39 | 2504 | 2490-2514 | 0.68 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 546.9 | 545.9-552.0 | 0.35 | - | - | - |
| short-49208-sliding-cp8r4 | tilelang@main | 1569 | 1551-1574 | 1.00 | 3739 | 3720-3763 | 1.00 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 545.8 | 542.3-547.2 | 0.35 | - | - | - |
| short-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 1566 | 1558-1567 | 1.00 | 3735 | 3725-3745 | 1.00 |
| short-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 611.4 | 609.5-616.4 | 0.39 | 2519 | 2502-2541 | 0.67 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 545.3 | 541.6-546.2 | 0.35 | - | - | - |
| short-49208-sliding-cp8r7 | tilelang@main | 1556 | 1551-1565 | 1.00 | 3743 | 3735-3746 | 1.00 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 536.9 | 535.8-539.6 | 0.34 | - | - | - |
| short-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 1587 | 1574-1605 | 1.00 | 3734 | 3731-3746 | 1.00 |
| short-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 608.2 | 604.0-609.4 | 0.38 | 2530 | 2524-2531 | 0.68 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 544.9 | 542.2-545.8 | 0.34 | - | - | - |
| heavy-49208-csa-cp1 | tilelang@main | 14914 | 14896-14994 | 1.00 | 61708 | 61688-61710 | 1.00 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 7394 | 7345-7676 | 0.50 | - | - | - |
| heavy-49208-csa-cp1 | tilelang@cudnn_flashmla | 14921 | 14860-14969 | 1.00 | 62042 | 61696-62107 | 1.00 |
| heavy-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 7497 | 7226-7678 | 0.50 | 34637 | 34530-34728 | 0.56 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 7661 | 7618-7673 | 0.51 | - | - | - |
| heavy-49208-csa-cp8r0 | tilelang@main | 1925 | 1921-1931 | 1.00 | 5420 | 5416-5427 | 1.00 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 759.8 | 758.7-765.0 | 0.39 | - | - | - |
| heavy-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 1936 | 1926-1961 | 1.00 | 5414 | 5410-5419 | 1.00 |
| heavy-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 818.7 | 814.0-826.9 | 0.42 | 3131 | 3129-3136 | 0.58 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 751.9 | 747.9-755.5 | 0.39 | - | - | - |
| heavy-49208-csa-cp8r4 | tilelang@main | 2629 | 2613-2647 | 1.00 | 9369 | 9360-9377 | 1.00 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1058 | 1054-1062 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 2635 | 2624-2654 | 1.00 | 9482 | 9473-9498 | 1.00 |
| heavy-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1129 | 1125-1132 | 0.43 | 4945 | 4939-4986 | 0.52 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1059 | 1054-1062 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r7 | tilelang@main | 2477 | 2464-2492 | 1.00 | 8582 | 8578-8592 | 1.00 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 990.5 | 988.4-996.9 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 2482 | 2481-2499 | 1.00 | 8649 | 8613-8679 | 1.00 |
| heavy-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1058 | 1050-1059 | 0.43 | 4547 | 4541-4549 | 0.53 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 985.7 | 984.1-989.8 | 0.40 | - | - | - |
| heavy-49208-hca-cp1 | tilelang@main | 9089 | 9066-9107 | 1.00 | 29605 | 29574-29619 | 1.00 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4274 | 4269-4284 | 0.47 | - | - | - |
| heavy-49208-hca-cp1 | tilelang@cudnn_flashmla | 9072 | 9068-9084 | 1.00 | 29675 | 29667-29698 | 1.00 |
| heavy-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 4416 | 4407-4435 | 0.49 | 19005 | 19002-19008 | 0.64 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4268 | 4261-4271 | 0.47 | - | - | - |
| heavy-49208-hca-cp8r0 | tilelang@main | 1702 | 1695-1730 | 1.00 | 3912 | 3899-3948 | 1.00 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 642.8 | 634.5-652.1 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 1698 | 1682-1721 | 1.00 | 3897 | 3892-3902 | 1.00 |
| heavy-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 701.8 | 695.3-710.5 | 0.41 | 2535 | 2529-2540 | 0.65 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 635.8 | 629.0-639.1 | 0.37 | - | - | - |
| heavy-49208-hca-cp8r4 | tilelang@main | 1753 | 1742-1768 | 1.00 | 4393 | 4384-4416 | 1.00 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 675.0 | 669.0-676.8 | 0.39 | - | - | - |
| heavy-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 1770 | 1759-1780 | 1.00 | 4418 | 4412-4472 | 1.00 |
| heavy-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 744.0 | 731.9-749.5 | 0.42 | 2762 | 2756-2813 | 0.63 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 669.8 | 668.0-673.6 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r7 | tilelang@main | 1845 | 1815-1849 | 1.00 | 4673 | 4665-4690 | 1.00 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 712.1 | 708.3-714.2 | 0.39 | - | - | - |
| heavy-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 1843 | 1830-1846 | 1.00 | 4670 | 4650-4682 | 1.00 |
| heavy-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 778.2 | 771.8-784.1 | 0.42 | 2924 | 2918-2932 | 0.63 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 712.0 | 706.7-714.4 | 0.39 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang@main | 7448 | 7405-7474 | 1.00 | 22583 | 22581-22614 | 1.00 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 3135 | 3122-3149 | 0.42 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang@cudnn_flashmla | 7413 | 7401-7433 | 1.00 | 22588 | 22583-22592 | 1.00 |
| heavy-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3225 | 3216-3230 | 0.44 | 15215 | 15211-15252 | 0.67 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3115 | 3111-3120 | 0.42 | - | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@main | 1572 | 1556-1586 | 1.00 | 3588 | 3579-3605 | 1.00 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 542.2 | 536.7-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 1589 | 1564-1602 | 1.00 | 3610 | 3597-3617 | 1.00 |
| heavy-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 613.0 | 605.6-622.3 | 0.39 | 2465 | 2462-2481 | 0.68 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 541.8 | 538.8-544.9 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@main | 1565 | 1560-1583 | 1.00 | 3761 | 3760-3770 | 1.00 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 538.9 | 537.2-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 1565 | 1553-1569 | 1.00 | 3784 | 3783-3789 | 1.00 |
| heavy-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 601.2 | 597.5-608.0 | 0.38 | 2519 | 2516-2537 | 0.67 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 540.4 | 535.7-546.8 | 0.35 | - | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@main | 1567 | 1558-1572 | 1.00 | 3766 | 3757-3772 | 1.00 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 543.2 | 540.1-547.6 | 0.35 | - | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 1571 | 1560-1577 | 1.00 | 3770 | 3761-3784 | 1.00 |
| heavy-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 601.1 | 598.6-603.5 | 0.38 | 2535 | 2532-2563 | 0.67 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 534.8 | 533.4-541.0 | 0.34 | - | - | - |
| tiny-49208-csa-cp1 | tilelang@main | 8393 | 8360-8427 | 1.00 | 24104 | 24092-24115 | 1.00 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 4295 | 4284-4297 | 0.51 | - | - | - |
| tiny-49208-csa-cp1 | tilelang@cudnn_flashmla | 8376 | 8360-8390 | 1.00 | 24149 | 24135-24152 | 1.00 |
| tiny-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 4340 | 4332-4351 | 0.52 | 16276 | 16270-16281 | 0.67 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4275 | 4274-4279 | 0.51 | - | - | - |
| tiny-49208-csa-cp8r0 | tilelang@main | 1704 | 1673-1720 | 1.00 | 3875 | 3869-3881 | 1.00 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 658.2 | 656.7-659.3 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 1687 | 1679-1716 | 1.00 | 3874 | 3868-3884 | 1.00 |
| tiny-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 721.6 | 716.5-729.4 | 0.43 | 2475 | 2458-2481 | 0.64 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 654.4 | 652.9-656.9 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r4 | tilelang@main | 1680 | 1674-1688 | 1.00 | 3906 | 3895-3913 | 1.00 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 656.4 | 651.7-664.8 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 1694 | 1690-1708 | 1.00 | 3907 | 3903-3925 | 1.00 |
| tiny-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 721.5 | 717.6-723.9 | 0.43 | 2502 | 2490-2502 | 0.64 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 656.3 | 653.7-657.9 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r7 | tilelang@main | 1689 | 1670-1693 | 1.00 | 3912 | 3898-3930 | 1.00 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 659.4 | 655.1-660.4 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 1708 | 1691-1712 | 1.00 | 3890 | 3878-3919 | 1.00 |
| tiny-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 730.5 | 723.3-731.7 | 0.43 | 2465 | 2461-2484 | 0.63 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 660.0 | 656.5-660.8 | 0.39 | - | - | - |
| tiny-49208-hca-cp1 | tilelang@main | 6744 | 6728-6770 | 1.00 | 16786 | 16781-16790 | 1.00 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 3164 | 3162-3181 | 0.47 | - | - | - |
| tiny-49208-hca-cp1 | tilelang@cudnn_flashmla | 6720 | 6715-6769 | 1.00 | 16785 | 16772-16793 | 1.00 |
| tiny-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 3270 | 3264-3279 | 0.49 | 12797 | 12793-12806 | 0.76 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3162 | 3158-3165 | 0.47 | - | - | - |
| tiny-49208-hca-cp8r0 | tilelang@main | 1479 | 1468-1496 | 1.00 | 2952 | 2948-2958 | 1.00 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 541.4 | 536.4-544.8 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 1514 | 1476-1532 | 1.00 | 2973 | 2960-2981 | 1.00 |
| tiny-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 616.3 | 610.1-617.1 | 0.41 | 2187 | 2183-2210 | 0.74 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 548.9 | 545.1-549.6 | 0.36 | - | - | - |
| tiny-49208-hca-cp8r4 | tilelang@main | 1485 | 1472-1546 | 1.00 | 2981 | 2974-2984 | 1.00 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 544.0 | 539.4-559.9 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 1485 | 1482-1490 | 1.00 | 3016 | 3005-3040 | 1.00 |
| tiny-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 602.1 | 598.9-612.6 | 0.41 | 2226 | 2219-2244 | 0.74 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 537.8 | 536.2-540.7 | 0.36 | - | - | - |
| tiny-49208-hca-cp8r7 | tilelang@main | 1474 | 1461-1497 | 1.00 | 2953 | 2945-2961 | 1.00 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 540.6 | 538.1-542.2 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 1493 | 1480-1503 | 1.00 | 2953 | 2946-2977 | 1.00 |
| tiny-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 607.7 | 602.5-607.9 | 0.41 | 2186 | 2174-2210 | 0.74 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 541.8 | 539.1-544.3 | 0.36 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang@main | 6733 | 6721-6755 | 1.00 | 16801 | 16796-16808 | 1.00 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 3173 | 3173-3185 | 0.47 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang@cudnn_flashmla | 6722 | 6707-6767 | 1.00 | 16756 | 16751-16769 | 1.00 |
| tiny-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3280 | 3274-3288 | 0.49 | 12799 | 12782-12815 | 0.76 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3168 | 3163-3174 | 0.47 | - | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@main | 1463 | 1458-1465 | 1.00 | 2947 | 2942-2964 | 1.00 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 540.3 | 536.1-543.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 1476 | 1472-1486 | 1.00 | 2964 | 2961-2975 | 1.00 |
| tiny-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 614.6 | 609.3-615.1 | 0.42 | 2211 | 2207-2214 | 0.75 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 547.5 | 546.1-551.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@main | 1485 | 1470-1510 | 1.00 | 2976 | 2975-2983 | 1.00 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 542.5 | 538.3-548.4 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 1494 | 1490-1500 | 1.00 | 3000 | 2995-3009 | 1.00 |
| tiny-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 612.3 | 603.5-616.4 | 0.41 | 2211 | 2194-2215 | 0.74 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 546.4 | 543.3-559.0 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@main | 1465 | 1463-1477 | 1.00 | 2951 | 2947-2968 | 1.00 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 542.3 | 536.3-545.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 1484 | 1479-1493 | 1.00 | 2976 | 2966-2978 | 1.00 |
| tiny-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 611.2 | 606.4-615.3 | 0.41 | 2207 | 2203-2210 | 0.74 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 545.6 | 544.2-550.2 | 0.37 | - | - | - |
| single-65536-csa-cp1 | tilelang@main | 21694 | 21617-21724 | 1.00 | 95444 | 95324-95476 | 1.00 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 11385 | 11377-11470 | 0.52 | - | - | - |
| single-65536-csa-cp1 | tilelang@cudnn_flashmla | 21717 | 21638-21763 | 1.00 | 95533 | 95401-95583 | 1.00 |
| single-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 11442 | 11363-11467 | 0.53 | 53422 | 53114-55473 | 0.56 |
| single-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 11470 | 11438-11482 | 0.53 | - | - | - |
| single-65536-csa-cp8r0 | tilelang@main | 3265 | 3257-3300 | 1.00 | 11931 | 11922-11967 | 1.00 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 1325 | 1322-1327 | 0.41 | - | - | - |
| single-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 3293 | 3262-3327 | 1.00 | 12055 | 12033-12124 | 1.00 |
| single-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 1402 | 1393-1421 | 0.43 | 6307 | 6287-6344 | 0.52 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 1329 | 1326-1341 | 0.40 | - | - | - |
| single-65536-csa-cp8r4 | tilelang@main | 3429 | 3394-3504 | 1.00 | 13145 | 13143-13204 | 1.00 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1410 | 1405-1413 | 0.41 | - | - | - |
| single-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 3442 | 3422-3469 | 1.00 | 13295 | 13259-13339 | 1.00 |
| single-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1480 | 1472-1495 | 0.43 | 6886 | 6843-6947 | 0.52 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1407 | 1404-1422 | 0.41 | - | - | - |
| single-65536-csa-cp8r7 | tilelang@main | 3384 | 3367-3434 | 1.00 | 13996 | 13970-14040 | 1.00 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1410 | 1408-1414 | 0.42 | - | - | - |
| single-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 3466 | 3422-3549 | 1.00 | 14127 | 14018-14158 | 1.00 |
| single-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1514 | 1489-1522 | 0.44 | 7215 | 7187-7369 | 0.51 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1418 | 1413-1421 | 0.41 | - | - | - |
| single-65536-hca-cp1 | tilelang@main | 16503 | 16484-16545 | 1.00 | 64327 | 64312-64341 | 1.00 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 8196 | 7971-8458 | 0.50 | - | - | - |
| single-65536-hca-cp1 | tilelang@cudnn_flashmla | 16565 | 16522-16611 | 1.00 | 64570 | 64309-64745 | 1.00 |
| single-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 8349 | 8301-8479 | 0.50 | 37436 | 37349-37445 | 0.58 |
| single-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 8424 | 8382-8435 | 0.51 | - | - | - |
| single-65536-hca-cp8r0 | tilelang@main | 2074 | 2053-2077 | 1.00 | 5434 | 5425-5441 | 1.00 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 835.2 | 832.6-836.7 | 0.40 | - | - | - |
| single-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 2045 | 2033-2069 | 1.00 | 5419 | 5410-5420 | 1.00 |
| single-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 901.5 | 890.8-913.3 | 0.44 | 3411 | 3408-3424 | 0.63 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 841.9 | 834.7-846.2 | 0.41 | - | - | - |
| single-65536-hca-cp8r4 | tilelang@main | 2797 | 2781-2799 | 1.00 | 9582 | 9559-9586 | 1.00 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1185 | 1183-1190 | 0.42 | - | - | - |
| single-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 2796 | 2790-2805 | 1.00 | 9666 | 9627-9670 | 1.00 |
| single-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1259 | 1254-1264 | 0.45 | 5277 | 5269-5293 | 0.55 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1189 | 1185-1191 | 0.43 | - | - | - |
| single-65536-hca-cp8r7 | tilelang@main | 3390 | 3380-3403 | 1.00 | 12689 | 12685-12694 | 1.00 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 1384 | 1378-1390 | 0.41 | - | - | - |
| single-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 3418 | 3409-3495 | 1.00 | 12780 | 12775-12800 | 1.00 |
| single-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1457 | 1450-1463 | 0.43 | 6609 | 6545-6687 | 0.52 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1381 | 1380-1384 | 0.40 | - | - | - |
| single-65536-sliding-cp1 | tilelang@main | 9650 | 9637-9666 | 1.00 | 30146 | 30143-30155 | 1.00 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4122 | 4112-4130 | 0.43 | - | - | - |
| single-65536-sliding-cp1 | tilelang@cudnn_flashmla | 9660 | 9639-9675 | 1.00 | 30125 | 30109-30150 | 1.00 |
| single-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4210 | 4209-4226 | 0.44 | 20280 | 20268-20302 | 0.67 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4097 | 4093-4105 | 0.42 | - | - | - |
| single-65536-sliding-cp8r0 | tilelang@main | 1871 | 1858-1899 | 1.00 | 4658 | 4655-4673 | 1.00 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 661.3 | 655.0-663.8 | 0.35 | - | - | - |
| single-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1857 | 1856-1878 | 1.00 | 4690 | 4683-4699 | 1.00 |
| single-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 737.2 | 728.5-739.3 | 0.40 | 3047 | 3034-3054 | 0.65 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 665.0 | 659.2-669.0 | 0.36 | - | - | - |
| single-65536-sliding-cp8r4 | tilelang@main | 1843 | 1830-1855 | 1.00 | 4694 | 4692-4707 | 1.00 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 660.7 | 659.4-664.0 | 0.36 | - | - | - |
| single-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1865 | 1855-1877 | 1.00 | 4716 | 4712-4723 | 1.00 |
| single-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 730.1 | 726.4-747.0 | 0.39 | 3028 | 3020-3033 | 0.64 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 660.0 | 658.4-663.4 | 0.35 | - | - | - |
| single-65536-sliding-cp8r7 | tilelang@main | 1852 | 1844-1863 | 1.00 | 4716 | 4715-4727 | 1.00 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 662.4 | 658.8-665.8 | 0.36 | - | - | - |
| single-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1878 | 1872-1888 | 1.00 | 4743 | 4741-4755 | 1.00 |
| single-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 732.3 | 726.4-735.2 | 0.39 | 3054 | 3049-3081 | 0.64 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 660.9 | 659.7-663.5 | 0.35 | - | - | - |
| short-65536-csa-cp1 | tilelang@main | 16242 | 16220-16289 | 1.00 | 63170 | 63160-63183 | 1.00 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 8146 | 8080-8320 | 0.50 | - | - | - |
| short-65536-csa-cp1 | tilelang@cudnn_flashmla | 16255 | 16210-16379 | 1.00 | 63353 | 63142-63539 | 1.00 |
| short-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 8226 | 8135-8341 | 0.51 | 36815 | 36797-36884 | 0.58 |
| short-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 8272 | 8264-8302 | 0.51 | - | - | - |
| short-65536-csa-cp8r0 | tilelang@main | 2287 | 2281-2291 | 1.00 | 6780 | 6771-6782 | 1.00 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 907.8 | 902.6-914.0 | 0.40 | - | - | - |
| short-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 2302 | 2298-2356 | 1.00 | 6791 | 6778-6795 | 1.00 |
| short-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 985.4 | 978.3-991.4 | 0.43 | 3960 | 3959-3965 | 0.58 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 918.0 | 905.4-923.2 | 0.40 | - | - | - |
| short-65536-csa-cp8r4 | tilelang@main | 2862 | 2860-2870 | 1.00 | 10076 | 10047-10086 | 1.00 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1170 | 1169-1173 | 0.41 | - | - | - |
| short-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 2892 | 2885-2902 | 1.00 | 10176 | 10163-10188 | 1.00 |
| short-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1247 | 1241-1259 | 0.43 | 5437 | 5427-5446 | 0.53 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1176 | 1173-1177 | 0.41 | - | - | - |
| short-65536-csa-cp8r7 | tilelang@main | 2664 | 2649-2673 | 1.00 | 8942 | 8934-8970 | 1.00 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1088 | 1087-1093 | 0.41 | - | - | - |
| short-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 2674 | 2672-2688 | 1.00 | 8941 | 8935-8949 | 1.00 |
| short-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1154 | 1153-1174 | 0.43 | 4918 | 4908-4928 | 0.55 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1090 | 1084-1101 | 0.41 | - | - | - |
| short-65536-hca-cp1 | tilelang@main | 10922 | 10906-10936 | 1.00 | 33532 | 33528-33548 | 1.00 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5378 | 5372-5387 | 0.49 | - | - | - |
| short-65536-hca-cp1 | tilelang@cudnn_flashmla | 10951 | 10907-10965 | 1.00 | 33579 | 33558-33601 | 1.00 |
| short-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 5472 | 5466-5477 | 0.50 | 22694 | 22685-22701 | 0.68 |
| short-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5373 | 5370-5376 | 0.49 | - | - | - |
| short-65536-hca-cp8r0 | tilelang@main | 2020 | 2012-2043 | 1.00 | 5027 | 5026-5032 | 1.00 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 811.2 | 808.9-814.9 | 0.40 | - | - | - |
| short-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 2041 | 2025-2046 | 1.00 | 5069 | 5062-5072 | 1.00 |
| short-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 872.6 | 871.6-893.4 | 0.43 | 3196 | 3192-3200 | 0.63 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 809.3 | 808.4-811.2 | 0.40 | - | - | - |
| short-65536-hca-cp8r4 | tilelang@main | 2054 | 2049-2060 | 1.00 | 5198 | 5193-5210 | 1.00 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 832.4 | 828.5-836.2 | 0.41 | - | - | - |
| short-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 2072 | 2060-2082 | 1.00 | 5245 | 5232-5257 | 1.00 |
| short-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 896.6 | 892.8-901.7 | 0.43 | 3320 | 3317-3324 | 0.63 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 828.6 | 825.2-833.1 | 0.40 | - | - | - |
| short-65536-hca-cp8r7 | tilelang@main | 2040 | 2035-2045 | 1.00 | 5167 | 5154-5172 | 1.00 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 819.8 | 815.1-823.4 | 0.40 | - | - | - |
| short-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 2070 | 2065-2073 | 1.00 | 5167 | 5152-5251 | 1.00 |
| short-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 894.5 | 886.4-898.9 | 0.43 | 3264 | 3259-3337 | 0.63 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 831.4 | 828.7-832.5 | 0.40 | - | - | - |
| short-65536-sliding-cp1 | tilelang@main | 9612 | 9605-9628 | 1.00 | 29641 | 29634-29651 | 1.00 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4105 | 4097-4107 | 0.43 | - | - | - |
| short-65536-sliding-cp1 | tilelang@cudnn_flashmla | 9656 | 9634-9699 | 1.00 | 29695 | 29684-29707 | 1.00 |
| short-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4228 | 4221-4231 | 0.44 | 20040 | 20018-20053 | 0.67 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4106 | 4104-4111 | 0.43 | - | - | - |
| short-65536-sliding-cp8r0 | tilelang@main | 1863 | 1854-1872 | 1.00 | 4557 | 4556-4558 | 1.00 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 665.9 | 661.3-668.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1850 | 1841-1875 | 1.00 | 4594 | 4589-4597 | 1.00 |
| short-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 730.0 | 723.8-739.1 | 0.39 | 2999 | 2985-3004 | 0.65 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 668.0 | 667.0-672.8 | 0.36 | - | - | - |
| short-65536-sliding-cp8r4 | tilelang@main | 1838 | 1829-1848 | 1.00 | 4624 | 4622-4646 | 1.00 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 660.4 | 656.3-663.9 | 0.36 | - | - | - |
| short-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1851 | 1846-1856 | 1.00 | 4643 | 4639-4682 | 1.00 |
| short-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 730.1 | 725.6-733.2 | 0.39 | 3001 | 2972-3020 | 0.65 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 659.8 | 656.2-661.1 | 0.36 | - | - | - |
| short-65536-sliding-cp8r7 | tilelang@main | 1841 | 1834-1848 | 1.00 | 4633 | 4620-4639 | 1.00 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 662.2 | 661.1-666.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1844 | 1839-1857 | 1.00 | 4647 | 4641-4668 | 1.00 |
| short-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 731.5 | 721.9-733.0 | 0.40 | 3014 | 3011-3028 | 0.65 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 658.2 | 656.9-661.2 | 0.36 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 17373 | 17342-17419 | 1.00 | 69696 | 69684-69701 | 1.00 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8820 | 8665-9037 | 0.51 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@cudnn_flashmla | 17465 | 17399-17477 | 1.00 | 69951 | 69730-69992 | 1.00 |
| heavy-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 8946 | 8913-9072 | 0.51 | 39981 | 39858-40148 | 0.57 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 9041 | 8975-9076 | 0.52 | - | - | - |
| heavy-65536-csa-cp8r0 | tilelang@main | 2238 | 2232-2242 | 1.00 | 6420 | 6417-6435 | 1.00 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 900.9 | 888.9-902.5 | 0.40 | - | - | - |
| heavy-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 2244 | 2237-2248 | 1.00 | 6429 | 6428-6438 | 1.00 |
| heavy-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 969.8 | 960.2-979.3 | 0.43 | 3805 | 3798-3812 | 0.59 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 901.3 | 897.9-904.3 | 0.40 | - | - | - |
| heavy-65536-csa-cp8r4 | tilelang@main | 3403 | 3379-3442 | 1.00 | 12973 | 12936-12999 | 1.00 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1402 | 1397-1405 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 3464 | 3442-3536 | 1.00 | 13154 | 13111-13169 | 1.00 |
| heavy-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1515 | 1480-1523 | 0.44 | 6836 | 6804-6933 | 0.52 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1406 | 1404-1408 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r7 | tilelang@main | 2550 | 2543-2560 | 1.00 | 8299 | 8297-8303 | 1.00 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1048 | 1046-1052 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 2587 | 2574-2607 | 1.00 | 8302 | 8290-8330 | 1.00 |
| heavy-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1128 | 1116-1130 | 0.44 | 4662 | 4660-4667 | 0.56 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1046 | 1043-1059 | 0.40 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 11714 | 11677-11736 | 1.00 | 38272 | 38241-38292 | 1.00 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5682 | 5674-5693 | 0.49 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@cudnn_flashmla | 11752 | 11745-11772 | 1.00 | 38336 | 38309-38504 | 1.00 |
| heavy-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 5868 | 5850-5886 | 0.50 | 24921 | 24869-24934 | 0.65 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5700 | 5680-5716 | 0.49 | - | - | - |
| heavy-65536-hca-cp8r0 | tilelang@main | 1991 | 1967-1996 | 1.00 | 4821 | 4813-4828 | 1.00 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 777.0 | 775.5-781.9 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 1985 | 1972-1995 | 1.00 | 4836 | 4833-4847 | 1.00 |
| heavy-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 845.9 | 844.6-853.4 | 0.43 | 3102 | 3096-3105 | 0.64 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 779.0 | 775.1-781.9 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r4 | tilelang@main | 2378 | 2363-2384 | 1.00 | 7050 | 7045-7053 | 1.00 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 928.0 | 924.2-932.5 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 2383 | 2376-2411 | 1.00 | 7083 | 7079-7090 | 1.00 |
| heavy-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 988.6 | 987.3-999.1 | 0.41 | 4072 | 4063-4079 | 0.57 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 927.9 | 924.6-928.5 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r7 | tilelang@main | 2006 | 2002-2013 | 1.00 | 5010 | 5008-5012 | 1.00 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 799.1 | 797.0-808.2 | 0.40 | - | - | - |
| heavy-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 2021 | 2011-2039 | 1.00 | 5003 | 4995-5015 | 1.00 |
| heavy-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 865.2 | 862.5-871.5 | 0.43 | 3215 | 3203-3224 | 0.64 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 804.3 | 803.4-804.9 | 0.40 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 9616 | 9585-9660 | 1.00 | 29282 | 29271-29295 | 1.00 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4111 | 4108-4115 | 0.43 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@cudnn_flashmla | 9598 | 9583-9632 | 1.00 | 29266 | 29262-29300 | 1.00 |
| heavy-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4220 | 4219-4228 | 0.44 | 19854 | 19851-19874 | 0.68 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4107 | 4105-4112 | 0.43 | - | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@main | 1853 | 1835-1865 | 1.00 | 4405 | 4399-4411 | 1.00 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 669.7 | 664.8-674.5 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1837 | 1831-1841 | 1.00 | 4442 | 4434-4450 | 1.00 |
| heavy-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 731.8 | 728.9-740.0 | 0.40 | 2915 | 2911-2924 | 0.66 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 664.0 | 658.6-671.0 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@main | 1840 | 1825-1842 | 1.00 | 4678 | 4674-4682 | 1.00 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 655.9 | 655.0-657.4 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1851 | 1850-1859 | 1.00 | 4711 | 4705-4723 | 1.00 |
| heavy-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 730.3 | 722.6-735.2 | 0.39 | 3024 | 3009-3036 | 0.64 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 659.4 | 658.3-660.5 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@main | 1824 | 1817-1827 | 1.00 | 4524 | 4521-4533 | 1.00 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 665.7 | 659.8-668.6 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1852 | 1851-1861 | 1.00 | 4553 | 4548-4561 | 1.00 |
| heavy-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 725.3 | 722.2-728.0 | 0.39 | 2960 | 2952-2972 | 0.65 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 659.8 | 656.8-661.0 | 0.36 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 10884 | 10872-10885 | 1.00 | 31800 | 31799-31808 | 1.00 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5659 | 5656-5664 | 0.52 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@cudnn_flashmla | 10898 | 10880-10906 | 1.00 | 31816 | 31803-31834 | 1.00 |
| tiny-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 5737 | 5731-5739 | 0.53 | 21544 | 21532-21547 | 0.68 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5658 | 5656-5663 | 0.52 | - | - | - |
| tiny-65536-csa-cp8r0 | tilelang@main | 2004 | 1988-2021 | 1.00 | 4891 | 4874-4905 | 1.00 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 809.2 | 808.8-818.0 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 2004 | 1996-2014 | 1.00 | 4864 | 4858-4876 | 1.00 |
| tiny-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 874.9 | 871.4-879.6 | 0.44 | 3084 | 3079-3086 | 0.63 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 813.5 | 806.2-814.9 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r4 | tilelang@main | 2008 | 1980-2018 | 1.00 | 4867 | 4862-4872 | 1.00 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 823.2 | 820.0-828.9 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 2005 | 2001-2020 | 1.00 | 4891 | 4889-4895 | 1.00 |
| tiny-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 887.4 | 879.6-890.9 | 0.44 | 3100 | 3095-3104 | 0.63 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 817.8 | 812.9-818.6 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r7 | tilelang@main | 2064 | 2023-2100 | 1.00 | 4878 | 4870-4892 | 1.00 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 831.3 | 809.5-891.6 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 1998 | 1994-2007 | 1.00 | 4902 | 4891-4908 | 1.00 |
| tiny-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 874.0 | 870.4-880.3 | 0.44 | 3099 | 3094-3124 | 0.63 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 808.0 | 804.0-810.0 | 0.40 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 8689 | 8676-8718 | 1.00 | 22022 | 21996-22029 | 1.00 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4173 | 4168-4184 | 0.48 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@cudnn_flashmla | 8662 | 8658-8687 | 1.00 | 22046 | 22037-22048 | 1.00 |
| tiny-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 4262 | 4258-4273 | 0.49 | 16878 | 16875-16883 | 0.77 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4151 | 4150-4158 | 0.48 | - | - | - |
| tiny-65536-hca-cp8r0 | tilelang@main | 1706 | 1702-1728 | 1.00 | 3628 | 3625-3630 | 1.00 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 657.9 | 656.0-673.7 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 1725 | 1719-1743 | 1.00 | 3643 | 3642-3645 | 1.00 |
| tiny-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 727.0 | 724.5-733.0 | 0.42 | 2584 | 2579-2597 | 0.71 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 662.2 | 657.4-666.0 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r4 | tilelang@main | 1732 | 1714-1738 | 1.00 | 3608 | 3597-3618 | 1.00 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 662.3 | 656.5-663.6 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 1731 | 1731-1737 | 1.00 | 3658 | 3658-3670 | 1.00 |
| tiny-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 726.6 | 724.2-735.3 | 0.42 | 2619 | 2613-2621 | 0.72 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 659.9 | 656.9-662.0 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r7 | tilelang@main | 1716 | 1711-1718 | 1.00 | 3631 | 3624-3641 | 1.00 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 666.2 | 661.7-670.2 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 1761 | 1756-1781 | 1.00 | 3647 | 3637-3650 | 1.00 |
| tiny-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 735.5 | 727.6-739.9 | 0.42 | 2586 | 2582-2595 | 0.71 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 658.9 | 657.6-663.5 | 0.37 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 8717 | 8700-8727 | 1.00 | 22036 | 22028-22065 | 1.00 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4178 | 4164-4186 | 0.48 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@cudnn_flashmla | 8674 | 8660-8714 | 1.00 | 21991 | 21986-21995 | 1.00 |
| tiny-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4267 | 4256-4268 | 0.49 | 16857 | 16845-16860 | 0.77 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4156 | 4153-4161 | 0.48 | - | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@main | 1745 | 1740-1764 | 1.00 | 3631 | 3625-3633 | 1.00 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 668.3 | 661.3-672.0 | 0.38 | - | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1749 | 1737-1752 | 1.00 | 3643 | 3638-3646 | 1.00 |
| tiny-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 742.3 | 735.7-749.0 | 0.42 | 2585 | 2575-2600 | 0.71 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 671.1 | 669.7-677.6 | 0.38 | - | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@main | 1716 | 1698-1722 | 1.00 | 3613 | 3604-3624 | 1.00 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 662.2 | 654.3-664.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1741 | 1723-1743 | 1.00 | 3641 | 3630-3642 | 1.00 |
| tiny-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 731.7 | 723.1-734.3 | 0.42 | 2563 | 2559-2586 | 0.70 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 661.8 | 660.4-662.6 | 0.38 | - | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@main | 1705 | 1695-1721 | 1.00 | 3629 | 3621-3632 | 1.00 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 665.0 | 659.9-666.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1713 | 1701-1726 | 1.00 | 3644 | 3636-3645 | 1.00 |
| tiny-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 725.7 | 720.5-727.8 | 0.42 | 2579 | 2568-2592 | 0.71 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 656.0 | 653.8-659.3 | 0.38 | - | - | - |

GPU busy time per call in µs from profiler traces (lower is better); `host` is op-boundary minus GPU
busy time (launch overhead and gaps); `/TL` divides GPU busy time by tilelang's; `peak MiB` is the
allocation above the inputs during one call, forward+backward where the backend has it, else forward.

| item | backend | fwd gpu µs | fwd host | fwd /TL | f+b gpu µs | f+b host | f+b /TL | peak MiB |
|---|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 539.5 | 657.7 | 1.00 | 2161 | 1025 | 1.00 | 265 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 284.0 | 100.0 | 0.53 | - | - | - | 139 |
| single-2048-csa-cp1 | tilelang@cudnn_flashmla | 538.5 | 671.1 | 1.00 | 2155 | 1111 | 1.00 | 265 |
| single-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 290.7 | 147.4 | 0.54 | 1256 | 668.3 | 0.58 | 271 |
| single-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 282.6 | 95.2 | 0.52 | - | - | - | 139 |
| single-2048-csa-cp8r0 | tilelang@main | 60.1 | 673.4 | 1.00 | 221.7 | 1780 | 1.00 | 40 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.0 | 138.4 | 0.73 | - | - | - | 17 |
| single-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 60.1 | 706.0 | 1.00 | 221.5 | 1854 | 1.00 | 40 |
| single-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 52.1 | 253.4 | 0.87 | 184.7 | 1264 | 0.83 | 40 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 44.0 | 143.7 | 0.73 | - | - | - | 17 |
| single-2048-csa-cp8r4 | tilelang@main | 84.4 | 679.0 | 1.00 | 356.6 | 1672 | 1.00 | 40 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 57.8 | 139.9 | 0.68 | - | - | - | 17 |
| single-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 83.5 | 693.3 | 1.00 | 354.9 | 1726 | 1.00 | 40 |
| single-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 64.9 | 240.6 | 0.78 | 249.4 | 1199 | 0.70 | 40 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 57.1 | 140.0 | 0.68 | - | - | - | 17 |
| single-2048-csa-cp8r7 | tilelang@main | 101.7 | 672.6 | 1.00 | 455.4 | 1569 | 1.00 | 40 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 64.1 | 138.7 | 0.63 | - | - | - | 17 |
| single-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 100.9 | 699.3 | 1.00 | 452.8 | 1640 | 1.00 | 40 |
| single-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 71.6 | 234.6 | 0.71 | 291.9 | 1154 | 0.64 | 40 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 63.8 | 148.9 | 0.63 | - | - | - | 17 |
| single-2048-hca-cp1 | tilelang@main | 362.9 | 722.3 | 1.00 | 1166 | 1338 | 1.00 | 265 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 199.3 | 141.7 | 0.55 | - | - | - | 133 |
| single-2048-hca-cp1 | tilelang@cudnn_flashmla | 360.3 | 742.7 | 1.00 | 1164 | 1367 | 1.00 | 265 |
| single-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 206.6 | 200.1 | 0.57 | 809.4 | 762.0 | 0.70 | 267 |
| single-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 196.4 | 150.3 | 0.55 | - | - | - | 133 |
| single-2048-hca-cp8r0 | tilelang@main | 57.6 | 706.3 | 1.00 | 206.3 | 1945 | 1.00 | 38 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 41.6 | 147.1 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 57.0 | 766.6 | 1.00 | 205.9 | 2024 | 1.00 | 38 |
| single-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 49.3 | 272.8 | 0.87 | 170.7 | 1305 | 0.83 | 39 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 42.0 | 149.1 | 0.74 | - | - | - | 17 |
| single-2048-hca-cp8r4 | tilelang@main | 61.4 | 731.7 | 1.00 | 219.3 | 1938 | 1.00 | 38 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 44.9 | 149.0 | 0.73 | - | - | - | 17 |
| single-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 61.9 | 722.6 | 1.00 | 219.1 | 2000 | 1.00 | 38 |
| single-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 52.7 | 259.9 | 0.85 | 182.8 | 1283 | 0.83 | 39 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 44.6 | 147.7 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r7 | tilelang@main | 62.1 | 744.8 | 1.00 | 219.6 | 1958 | 1.00 | 38 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.2 | 148.3 | 0.73 | - | - | - | 17 |
| single-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 62.0 | 716.9 | 1.00 | 219.6 | 1974 | 1.00 | 38 |
| single-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 52.4 | 259.2 | 0.85 | 183.6 | 1284 | 0.84 | 39 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 44.9 | 146.9 | 0.73 | - | - | - | 17 |
| single-2048-sliding-cp1 | tilelang@main | 312.4 | 703.5 | 1.00 | 1033 | 1255 | 1.00 | 264 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 150.8 | 147.6 | 0.48 | - | - | - | 131 |
| single-2048-sliding-cp1 | tilelang@cudnn_flashmla | 312.4 | 698.1 | 1.00 | 1030 | 1342 | 1.00 | 264 |
| single-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 157.4 | 200.5 | 0.50 | 711.8 | 806.5 | 0.69 | 266 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 148.7 | 145.1 | 0.48 | - | - | - | 131 |
| single-2048-sliding-cp8r0 | tilelang@main | 51.4 | 681.1 | 1.00 | 190.3 | 1857 | 1.00 | 38 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.6 | 144.1 | 0.69 | - | - | - | 16 |
| single-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 51.6 | 697.3 | 1.00 | 190.8 | 1905 | 1.00 | 38 |
| single-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 43.6 | 265.4 | 0.85 | 160.9 | 1303 | 0.84 | 38 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 36.0 | 144.8 | 0.70 | - | - | - | 16 |
| single-2048-sliding-cp8r4 | tilelang@main | 53.4 | 686.1 | 1.00 | 195.9 | 1822 | 1.00 | 38 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.5 | 143.5 | 0.66 | - | - | - | 16 |
| single-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 53.4 | 710.8 | 1.00 | 196.0 | 1900 | 1.00 | 38 |
| single-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 43.7 | 262.7 | 0.82 | 163.1 | 1302 | 0.83 | 38 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 36.4 | 143.0 | 0.68 | - | - | - | 16 |
| single-2048-sliding-cp8r7 | tilelang@main | 53.2 | 686.3 | 1.00 | 195.3 | 1838 | 1.00 | 38 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 36.2 | 144.7 | 0.68 | - | - | - | 16 |
| single-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 53.1 | 706.6 | 1.00 | 195.2 | 1920 | 1.00 | 38 |
| single-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 43.8 | 260.5 | 0.83 | 164.4 | 1296 | 0.84 | 38 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 36.3 | 144.1 | 0.68 | - | - | - | 16 |
| short-2048-csa-cp1 | tilelang@main | 442.8 | 688.9 | 1.00 | 1638 | 1118 | 1.00 | 265 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 234.8 | 133.2 | 0.53 | - | - | - | 139 |
| short-2048-csa-cp1 | tilelang@cudnn_flashmla | 440.2 | 696.4 | 1.00 | 1632 | 1213 | 1.00 | 265 |
| short-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 239.5 | 186.1 | 0.54 | 1010 | 716.6 | 0.62 | 271 |
| short-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 232.9 | 134.9 | 0.53 | - | - | - | 139 |
| short-2048-csa-cp8r0 | tilelang@main | 60.1 | 693.0 | 1.00 | 221.8 | 1801 | 1.00 | 40 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.4 | 143.4 | 0.74 | - | - | - | 17 |
| short-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 60.6 | 682.3 | 1.00 | 221.8 | 1894 | 1.00 | 40 |
| short-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 52.3 | 250.8 | 0.86 | 184.0 | 1281 | 0.83 | 40 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 44.2 | 141.6 | 0.73 | - | - | - | 17 |
| short-2048-csa-cp8r4 | tilelang@main | 67.8 | 674.7 | 1.00 | 262.9 | 1770 | 1.00 | 40 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 48.9 | 138.8 | 0.72 | - | - | - | 17 |
| short-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 68.0 | 696.3 | 1.00 | 261.3 | 1852 | 1.00 | 40 |
| short-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 56.3 | 248.4 | 0.83 | 201.7 | 1254 | 0.77 | 40 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 49.4 | 144.9 | 0.73 | - | - | - | 17 |
| short-2048-csa-cp8r7 | tilelang@main | 76.4 | 685.8 | 1.00 | 319.2 | 1718 | 1.00 | 40 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 50.9 | 140.6 | 0.67 | - | - | - | 17 |
| short-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 76.1 | 715.4 | 1.00 | 319.8 | 1790 | 1.00 | 40 |
| short-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 58.6 | 264.4 | 0.77 | 226.6 | 1250 | 0.71 | 40 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 51.2 | 147.7 | 0.67 | - | - | - | 17 |
| short-2048-hca-cp1 | tilelang@main | 354.6 | 724.6 | 1.00 | 1143 | 1346 | 1.00 | 265 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 196.7 | 143.8 | 0.55 | - | - | - | 133 |
| short-2048-hca-cp1 | tilelang@cudnn_flashmla | 353.3 | 734.3 | 1.00 | 1137 | 1388 | 1.00 | 265 |
| short-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 203.7 | 194.1 | 0.58 | 795.5 | 767.6 | 0.70 | 267 |
| short-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 194.4 | 142.2 | 0.55 | - | - | - | 133 |
| short-2048-hca-cp8r0 | tilelang@main | 57.2 | 730.2 | 1.00 | 205.4 | 1949 | 1.00 | 38 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 42.0 | 149.5 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 57.7 | 726.5 | 1.00 | 206.3 | 1956 | 1.00 | 38 |
| short-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 49.3 | 257.9 | 0.86 | 172.4 | 1277 | 0.84 | 39 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 41.3 | 144.9 | 0.72 | - | - | - | 17 |
| short-2048-hca-cp8r4 | tilelang@main | 58.6 | 736.4 | 1.00 | 211.8 | 1950 | 1.00 | 38 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 43.2 | 150.1 | 0.74 | - | - | - | 17 |
| short-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 58.4 | 718.7 | 1.00 | 212.1 | 1992 | 1.00 | 38 |
| short-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 50.9 | 254.4 | 0.87 | 176.0 | 1291 | 0.83 | 39 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 43.0 | 143.7 | 0.74 | - | - | - | 17 |
| short-2048-hca-cp8r7 | tilelang@main | 62.0 | 730.7 | 1.00 | 219.8 | 1973 | 1.00 | 38 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.4 | 143.5 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 61.7 | 724.2 | 1.00 | 219.7 | 1934 | 1.00 | 38 |
| short-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 52.3 | 260.3 | 0.85 | 182.0 | 1291 | 0.83 | 39 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 44.9 | 148.7 | 0.73 | - | - | - | 17 |
| short-2048-sliding-cp1 | tilelang@main | 307.0 | 695.7 | 1.00 | 1011 | 1319 | 1.00 | 264 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 147.7 | 145.6 | 0.48 | - | - | - | 131 |
| short-2048-sliding-cp1 | tilelang@cudnn_flashmla | 306.0 | 689.2 | 1.00 | 1010 | 1445 | 1.00 | 264 |
| short-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 158.1 | 193.3 | 0.52 | 703.9 | 927.3 | 0.70 | 266 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 151.7 | 138.1 | 0.50 | - | - | - | 131 |
| short-2048-sliding-cp8r0 | tilelang@main | 51.5 | 695.3 | 1.00 | 190.0 | 1852 | 1.00 | 38 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 36.3 | 145.5 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 51.2 | 691.4 | 1.00 | 190.6 | 1891 | 1.00 | 38 |
| short-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 43.8 | 255.3 | 0.86 | 161.4 | 1300 | 0.85 | 38 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 35.9 | 139.6 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r4 | tilelang@main | 51.5 | 705.5 | 1.00 | 191.7 | 1831 | 1.00 | 38 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 36.2 | 153.6 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 51.2 | 698.6 | 1.00 | 191.4 | 1910 | 1.00 | 38 |
| short-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 44.1 | 259.2 | 0.86 | 160.5 | 1289 | 0.84 | 38 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 36.4 | 144.5 | 0.71 | - | - | - | 16 |
| short-2048-sliding-cp8r7 | tilelang@main | 53.4 | 683.0 | 1.00 | 196.2 | 1848 | 1.00 | 38 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.6 | 149.5 | 0.67 | - | - | - | 16 |
| short-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 53.4 | 705.5 | 1.00 | 196.5 | 1921 | 1.00 | 38 |
| short-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 44.3 | 261.7 | 0.83 | 163.7 | 1300 | 0.83 | 38 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 36.5 | 142.4 | 0.68 | - | - | - | 16 |
| heavy-2048-csa-cp1 | tilelang@main | 476.0 | 694.7 | 1.00 | 1821 | 1134 | 1.00 | 265 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 254.4 | 135.3 | 0.53 | - | - | - | 139 |
| heavy-2048-csa-cp1 | tilelang@cudnn_flashmla | 473.5 | 689.7 | 1.00 | 1817 | 1159 | 1.00 | 265 |
| heavy-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 260.4 | 180.0 | 0.55 | 1092 | 685.5 | 0.60 | 271 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 253.1 | 133.5 | 0.53 | - | - | - | 139 |
| heavy-2048-csa-cp8r0 | tilelang@main | 60.1 | 687.4 | 1.00 | 214.2 | 1829 | 1.00 | 40 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.1 | 145.0 | 0.73 | - | - | - | 17 |
| heavy-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 60.3 | 684.9 | 1.00 | 214.5 | 1894 | 1.00 | 40 |
| heavy-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 51.9 | 255.1 | 0.86 | 177.3 | 1291 | 0.83 | 40 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 44.4 | 140.7 | 0.74 | - | - | - | 17 |
| heavy-2048-csa-cp8r4 | tilelang@main | 75.4 | 702.2 | 1.00 | 312.4 | 1752 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 51.7 | 149.0 | 0.69 | - | - | - | 17 |
| heavy-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 75.1 | 697.9 | 1.00 | 310.7 | 1787 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 59.0 | 246.2 | 0.79 | 224.9 | 1222 | 0.72 | 40 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 51.4 | 137.9 | 0.68 | - | - | - | 17 |
| heavy-2048-csa-cp8r7 | tilelang@main | 93.5 | 699.7 | 1.00 | 410.4 | 1624 | 1.00 | 40 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 60.9 | 148.8 | 0.65 | - | - | - | 17 |
| heavy-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 92.8 | 686.9 | 1.00 | 406.8 | 1701 | 1.00 | 40 |
| heavy-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 68.0 | 231.1 | 0.73 | 269.8 | 1197 | 0.66 | 40 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 61.1 | 135.1 | 0.66 | - | - | - | 17 |
| heavy-2048-hca-cp1 | tilelang@main | 344.4 | 729.5 | 1.00 | 1109 | 1355 | 1.00 | 265 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 188.5 | 146.7 | 0.55 | - | - | - | 133 |
| heavy-2048-hca-cp1 | tilelang@cudnn_flashmla | 343.5 | 773.7 | 1.00 | 1106 | 1421 | 1.00 | 265 |
| heavy-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 197.1 | 199.8 | 0.57 | 771.7 | 782.4 | 0.70 | 267 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 188.3 | 148.8 | 0.55 | - | - | - | 133 |
| heavy-2048-hca-cp8r0 | tilelang@main | 54.0 | 720.3 | 1.00 | 187.5 | 1948 | 1.00 | 38 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 41.0 | 148.3 | 0.76 | - | - | - | 17 |
| heavy-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 53.9 | 732.4 | 1.00 | 188.1 | 2052 | 1.00 | 38 |
| heavy-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 48.4 | 259.4 | 0.90 | 164.8 | 1311 | 0.88 | 39 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 41.2 | 143.9 | 0.76 | - | - | - | 17 |
| heavy-2048-hca-cp8r4 | tilelang@main | 61.8 | 720.2 | 1.00 | 219.6 | 1903 | 1.00 | 38 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 45.0 | 144.6 | 0.73 | - | - | - | 17 |
| heavy-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 62.0 | 740.5 | 1.00 | 218.6 | 2062 | 1.00 | 38 |
| heavy-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 52.7 | 254.8 | 0.85 | 182.0 | 1301 | 0.83 | 39 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 44.9 | 143.4 | 0.72 | - | - | - | 17 |
| heavy-2048-hca-cp8r7 | tilelang@main | 61.9 | 725.0 | 1.00 | 219.4 | 1938 | 1.00 | 38 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 45.1 | 144.7 | 0.73 | - | - | - | 17 |
| heavy-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 62.0 | 737.7 | 1.00 | 219.5 | 2001 | 1.00 | 38 |
| heavy-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 52.5 | 259.7 | 0.85 | 182.3 | 1307 | 0.83 | 39 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 45.5 | 147.0 | 0.73 | - | - | - | 17 |
| heavy-2048-sliding-cp1 | tilelang@main | 305.0 | 695.3 | 1.00 | 1000 | 1294 | 1.00 | 264 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 150.5 | 141.3 | 0.49 | - | - | - | 131 |
| heavy-2048-sliding-cp1 | tilelang@cudnn_flashmla | 303.1 | 693.4 | 1.00 | 997.1 | 1369 | 1.00 | 264 |
| heavy-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 158.5 | 195.5 | 0.52 | 693.9 | 821.1 | 0.70 | 266 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 146.7 | 144.8 | 0.48 | - | - | - | 131 |
| heavy-2048-sliding-cp8r0 | tilelang@main | 49.9 | 686.1 | 1.00 | 180.0 | 1859 | 1.00 | 38 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.3 | 144.8 | 0.71 | - | - | - | 16 |
| heavy-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 49.8 | 712.0 | 1.00 | 180.4 | 1946 | 1.00 | 38 |
| heavy-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 43.8 | 268.1 | 0.88 | 157.0 | 1325 | 0.87 | 38 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 36.4 | 152.2 | 0.73 | - | - | - | 16 |
| heavy-2048-sliding-cp8r4 | tilelang@main | 53.5 | 688.8 | 1.00 | 195.8 | 1841 | 1.00 | 38 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.8 | 144.9 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 53.3 | 687.4 | 1.00 | 195.7 | 1905 | 1.00 | 38 |
| heavy-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 43.8 | 261.7 | 0.82 | 163.3 | 1298 | 0.83 | 38 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 36.1 | 146.2 | 0.68 | - | - | - | 16 |
| heavy-2048-sliding-cp8r7 | tilelang@main | 53.2 | 690.4 | 1.00 | 195.4 | 1876 | 1.00 | 38 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.7 | 147.0 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 53.2 | 699.4 | 1.00 | 196.4 | 1945 | 1.00 | 38 |
| heavy-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 43.5 | 267.5 | 0.82 | 163.2 | 1319 | 0.83 | 38 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 36.1 | 146.8 | 0.68 | - | - | - | 16 |
| tiny-2048-csa-cp1 | tilelang@main | 358.8 | 683.7 | 1.00 | 1100 | 1259 | 1.00 | 265 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 202.3 | 131.6 | 0.56 | - | - | - | 139 |
| tiny-2048-csa-cp1 | tilelang@cudnn_flashmla | 357.4 | 711.4 | 1.00 | 1095 | 1296 | 1.00 | 265 |
| tiny-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 210.1 | 186.9 | 0.59 | 756.2 | 749.9 | 0.69 | 271 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 200.5 | 134.9 | 0.56 | - | - | - | 139 |
| tiny-2048-csa-cp8r0 | tilelang@main | 59.5 | 694.4 | 1.00 | 208.2 | 1848 | 1.00 | 40 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 44.4 | 148.0 | 0.75 | - | - | - | 17 |
| tiny-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 59.5 | 692.6 | 1.00 | 207.5 | 1912 | 1.00 | 40 |
| tiny-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 51.4 | 255.0 | 0.86 | 175.2 | 1310 | 0.84 | 40 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 43.6 | 143.9 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r4 | tilelang@main | 59.6 | 688.9 | 1.00 | 206.9 | 1822 | 1.00 | 40 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 43.5 | 144.4 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 59.4 | 701.1 | 1.00 | 206.5 | 1934 | 1.00 | 40 |
| tiny-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 51.5 | 259.4 | 0.87 | 172.3 | 1314 | 0.83 | 40 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 43.7 | 145.0 | 0.74 | - | - | - | 17 |
| tiny-2048-csa-cp8r7 | tilelang@main | 60.4 | 700.6 | 1.00 | 209.9 | 1830 | 1.00 | 40 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 44.1 | 145.9 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 59.8 | 695.5 | 1.00 | 210.5 | 1890 | 1.00 | 40 |
| tiny-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 51.5 | 254.1 | 0.86 | 176.4 | 1301 | 0.84 | 40 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 43.9 | 140.5 | 0.73 | - | - | - | 17 |
| tiny-2048-hca-cp1 | tilelang@main | 270.8 | 691.8 | 1.00 | 772.0 | 1321 | 1.00 | 264 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 148.6 | 143.6 | 0.55 | - | - | - | 131 |
| tiny-2048-hca-cp1 | tilelang@cudnn_flashmla | 270.7 | 702.3 | 1.00 | 769.4 | 1418 | 1.00 | 264 |
| tiny-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 157.4 | 197.8 | 0.58 | 600.4 | 892.6 | 0.78 | 266 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 147.6 | 148.2 | 0.55 | - | - | - | 131 |
| tiny-2048-hca-cp8r0 | tilelang@main | 48.4 | 688.1 | 1.00 | 166.6 | 1863 | 1.00 | 38 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 35.5 | 147.1 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 48.1 | 698.3 | 1.00 | 165.4 | 1926 | 1.00 | 38 |
| tiny-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 43.5 | 262.0 | 0.90 | 150.2 | 1313 | 0.91 | 38 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 36.1 | 145.6 | 0.75 | - | - | - | 16 |
| tiny-2048-hca-cp8r4 | tilelang@main | 48.4 | 686.5 | 1.00 | 164.1 | 1885 | 1.00 | 38 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 35.5 | 145.4 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 48.4 | 686.0 | 1.00 | 164.2 | 1966 | 1.00 | 38 |
| tiny-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 43.4 | 263.6 | 0.89 | 149.0 | 1325 | 0.91 | 38 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 36.1 | 147.4 | 0.75 | - | - | - | 16 |
| tiny-2048-hca-cp8r7 | tilelang@main | 50.0 | 701.3 | 1.00 | 175.8 | 1923 | 1.00 | 38 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 35.6 | 154.3 | 0.71 | - | - | - | 16 |
| tiny-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 49.9 | 702.1 | 1.00 | 176.6 | 1930 | 1.00 | 38 |
| tiny-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 43.5 | 264.9 | 0.87 | 155.1 | 1323 | 0.88 | 38 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 36.1 | 140.7 | 0.72 | - | - | - | 16 |
| tiny-2048-sliding-cp1 | tilelang@main | 271.0 | 686.8 | 1.00 | 772.5 | 1336 | 1.00 | 264 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 149.0 | 141.5 | 0.55 | - | - | - | 131 |
| tiny-2048-sliding-cp1 | tilelang@cudnn_flashmla | 271.0 | 693.2 | 1.00 | 769.7 | 1406 | 1.00 | 264 |
| tiny-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 158.5 | 194.9 | 0.58 | 598.3 | 899.8 | 0.78 | 266 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 147.8 | 144.6 | 0.55 | - | - | - | 131 |
| tiny-2048-sliding-cp8r0 | tilelang@main | 48.3 | 683.4 | 1.00 | 165.9 | 1970 | 1.00 | 38 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 35.2 | 144.5 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 48.3 | 709.3 | 1.00 | 165.8 | 1956 | 1.00 | 38 |
| tiny-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 43.5 | 263.7 | 0.90 | 149.0 | 1326 | 0.90 | 38 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 36.1 | 148.7 | 0.75 | - | - | - | 16 |
| tiny-2048-sliding-cp8r4 | tilelang@main | 48.3 | 690.2 | 1.00 | 164.1 | 1860 | 1.00 | 38 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 35.3 | 143.1 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 48.7 | 694.7 | 1.00 | 165.0 | 1950 | 1.00 | 38 |
| tiny-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 43.7 | 257.1 | 0.90 | 149.4 | 1314 | 0.91 | 38 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 36.1 | 146.0 | 0.74 | - | - | - | 16 |
| tiny-2048-sliding-cp8r7 | tilelang@main | 50.1 | 708.0 | 1.00 | 175.6 | 1874 | 1.00 | 38 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 35.6 | 151.2 | 0.71 | - | - | - | 16 |
| tiny-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 50.3 | 701.4 | 1.00 | 176.7 | 1943 | 1.00 | 38 |
| tiny-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 43.6 | 269.3 | 0.87 | 153.7 | 1316 | 0.87 | 38 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 35.9 | 149.2 | 0.71 | - | - | - | 16 |
| single-4096-csa-cp1 | tilelang@main | 1208 | 682.8 | 1.00 | 5145 | 827.0 | 1.00 | 530 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 593.7 | 121.7 | 0.49 | - | - | - | 278 |
| single-4096-csa-cp1 | tilelang@cudnn_flashmla | 1220 | 688.3 | 1.00 | 5144 | 865.6 | 1.00 | 530 |
| single-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 608.0 | 178.8 | 0.50 | 2780 | 344.3 | 0.54 | 542 |
| single-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 597.5 | 120.3 | 0.49 | - | - | - | 278 |
| single-4096-csa-cp8r0 | tilelang@main | 112.0 | 683.6 | 1.00 | 410.0 | 1627 | 1.00 | 79 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.0 | 138.3 | 0.61 | - | - | - | 35 |
| single-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 112.6 | 699.3 | 1.00 | 410.3 | 1710 | 1.00 | 79 |
| single-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 77.2 | 228.0 | 0.69 | 305.1 | 1181 | 0.74 | 81 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 68.5 | 139.3 | 0.61 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang@main | 191.2 | 677.8 | 1.00 | 856.0 | 1477 | 1.00 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 109.5 | 138.9 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 191.1 | 698.1 | 1.00 | 858.1 | 1563 | 1.00 | 79 |
| single-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 117.7 | 192.8 | 0.62 | 511.8 | 985.6 | 0.60 | 81 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 108.9 | 142.4 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r7 | tilelang@main | 191.7 | 689.2 | 1.00 | 859.6 | 1490 | 1.00 | 79 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 109.4 | 142.6 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 190.3 | 703.1 | 1.00 | 860.3 | 1525 | 1.00 | 79 |
| single-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 117.2 | 195.0 | 0.62 | 512.9 | 957.8 | 0.60 | 81 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 109.1 | 144.4 | 0.57 | - | - | - | 35 |
| single-4096-hca-cp1 | tilelang@main | 691.9 | 740.4 | 1.00 | 2233 | 900.3 | 1.00 | 530 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 366.3 | 139.2 | 0.53 | - | - | - | 266 |
| single-4096-hca-cp1 | tilelang@cudnn_flashmla | 689.8 | 739.1 | 1.00 | 2234 | 981.5 | 1.00 | 530 |
| single-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 372.8 | 200.1 | 0.54 | 1508 | 616.0 | 0.67 | 533 |
| single-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 365.2 | 143.2 | 0.53 | - | - | - | 266 |
| single-4096-hca-cp8r0 | tilelang@main | 102.0 | 726.2 | 1.00 | 349.1 | 1777 | 1.00 | 77 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 66.4 | 150.7 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 101.5 | 763.6 | 1.00 | 351.2 | 1853 | 1.00 | 77 |
| single-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 73.7 | 239.6 | 0.73 | 275.7 | 1194 | 0.79 | 77 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 64.0 | 149.9 | 0.63 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang@main | 107.4 | 716.8 | 1.00 | 370.8 | 1772 | 1.00 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 70.2 | 146.6 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 106.8 | 729.8 | 1.00 | 369.2 | 1877 | 1.00 | 77 |
| single-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 77.5 | 235.8 | 0.73 | 290.8 | 1208 | 0.79 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 68.6 | 148.4 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r7 | tilelang@main | 107.1 | 720.0 | 1.00 | 372.8 | 1786 | 1.00 | 77 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 68.8 | 148.4 | 0.64 | - | - | - | 33 |
| single-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 107.2 | 737.7 | 1.00 | 371.5 | 1841 | 1.00 | 77 |
| single-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 77.4 | 240.7 | 0.72 | 292.1 | 1184 | 0.79 | 77 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 68.8 | 154.9 | 0.64 | - | - | - | 33 |
| single-4096-sliding-cp1 | tilelang@main | 594.3 | 703.0 | 1.00 | 1945 | 914.2 | 1.00 | 527 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 273.9 | 146.3 | 0.46 | - | - | - | 262 |
| single-4096-sliding-cp1 | tilelang@cudnn_flashmla | 593.8 | 700.0 | 1.00 | 1944 | 975.7 | 1.00 | 527 |
| single-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 288.4 | 194.7 | 0.49 | 1307 | 700.3 | 0.67 | 531 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 272.2 | 146.4 | 0.46 | - | - | - | 262 |
| single-4096-sliding-cp8r0 | tilelang@main | 90.6 | 699.5 | 1.00 | 319.4 | 1715 | 1.00 | 76 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 53.1 | 146.2 | 0.59 | - | - | - | 33 |
| single-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 89.2 | 709.8 | 1.00 | 319.9 | 1820 | 1.00 | 76 |
| single-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 61.0 | 251.1 | 0.68 | 248.9 | 1232 | 0.78 | 77 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 53.4 | 147.6 | 0.60 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@main | 92.4 | 698.6 | 1.00 | 327.4 | 1730 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.7 | 149.4 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 92.1 | 710.0 | 1.00 | 326.9 | 1786 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 61.8 | 242.4 | 0.67 | 254.2 | 1216 | 0.78 | 77 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 53.7 | 146.3 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r7 | tilelang@main | 92.2 | 693.0 | 1.00 | 327.1 | 1738 | 1.00 | 76 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.9 | 140.2 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 92.3 | 711.1 | 1.00 | 326.9 | 1806 | 1.00 | 76 |
| single-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 62.4 | 249.5 | 0.68 | 253.5 | 1217 | 0.78 | 77 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 53.2 | 151.2 | 0.58 | - | - | - | 33 |
| short-4096-csa-cp1 | tilelang@main | 829.2 | 694.2 | 1.00 | 3054 | 819.3 | 1.00 | 530 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 428.1 | 124.9 | 0.52 | - | - | - | 278 |
| short-4096-csa-cp1 | tilelang@cudnn_flashmla | 829.9 | 702.6 | 1.00 | 3051 | 827.4 | 1.00 | 530 |
| short-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 434.3 | 179.9 | 0.52 | 1848 | 521.7 | 0.61 | 542 |
| short-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 426.0 | 126.6 | 0.51 | - | - | - | 278 |
| short-4096-csa-cp8r0 | tilelang@main | 112.0 | 715.7 | 1.00 | 410.6 | 1643 | 1.00 | 79 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.3 | 154.1 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 111.7 | 698.2 | 1.00 | 408.9 | 1724 | 1.00 | 79 |
| short-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 76.4 | 226.0 | 0.68 | 303.9 | 1192 | 0.74 | 81 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 68.1 | 143.6 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang@main | 114.7 | 762.2 | 1.00 | 432.7 | 1712 | 1.00 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.3 | 154.8 | 0.63 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 114.2 | 697.3 | 1.00 | 431.7 | 1695 | 1.00 | 79 |
| short-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 80.5 | 233.8 | 0.70 | 317.9 | 1165 | 0.74 | 81 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 73.1 | 145.3 | 0.64 | - | - | - | 35 |
| short-4096-csa-cp8r7 | tilelang@main | 112.8 | 705.3 | 1.00 | 430.6 | 1629 | 1.00 | 79 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 68.9 | 141.8 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 112.4 | 691.2 | 1.00 | 431.4 | 1659 | 1.00 | 79 |
| short-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 77.2 | 230.5 | 0.69 | 309.0 | 1171 | 0.72 | 81 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 69.1 | 143.6 | 0.61 | - | - | - | 35 |
| short-4096-hca-cp1 | tilelang@main | 666.3 | 744.5 | 1.00 | 2131 | 958.7 | 1.00 | 530 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 359.6 | 140.5 | 0.54 | - | - | - | 266 |
| short-4096-hca-cp1 | tilelang@cudnn_flashmla | 666.3 | 738.5 | 1.00 | 2130 | 1007 | 1.00 | 530 |
| short-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 369.7 | 199.2 | 0.55 | 1450 | 597.8 | 0.68 | 533 |
| short-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 361.4 | 140.9 | 0.54 | - | - | - | 266 |
| short-4096-hca-cp8r0 | tilelang@main | 100.2 | 744.3 | 1.00 | 347.4 | 1833 | 1.00 | 77 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 64.7 | 151.3 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 100.6 | 733.8 | 1.00 | 348.4 | 1876 | 1.00 | 77 |
| short-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 73.7 | 242.8 | 0.73 | 272.9 | 1208 | 0.78 | 77 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 64.9 | 151.9 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang@main | 101.5 | 725.0 | 1.00 | 354.4 | 1803 | 1.00 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 65.3 | 142.9 | 0.64 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 101.0 | 756.5 | 1.00 | 353.9 | 1889 | 1.00 | 77 |
| short-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 73.0 | 253.3 | 0.72 | 273.8 | 1213 | 0.77 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 64.8 | 152.0 | 0.64 | - | - | - | 33 |
| short-4096-hca-cp8r7 | tilelang@main | 102.6 | 728.6 | 1.00 | 359.5 | 1837 | 1.00 | 77 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 67.0 | 147.2 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 102.7 | 743.0 | 1.00 | 359.0 | 1869 | 1.00 | 77 |
| short-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 75.3 | 244.4 | 0.73 | 278.7 | 1203 | 0.78 | 77 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 67.0 | 153.3 | 0.65 | - | - | - | 33 |
| short-4096-sliding-cp1 | tilelang@main | 584.0 | 727.8 | 1.00 | 1896 | 946.0 | 1.00 | 527 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 271.4 | 151.8 | 0.46 | - | - | - | 262 |
| short-4096-sliding-cp1 | tilelang@cudnn_flashmla | 585.8 | 707.5 | 1.00 | 1898 | 992.7 | 1.00 | 527 |
| short-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 278.8 | 203.1 | 0.48 | 1283 | 688.7 | 0.68 | 531 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 274.2 | 146.6 | 0.47 | - | - | - | 262 |
| short-4096-sliding-cp8r0 | tilelang@main | 89.3 | 704.5 | 1.00 | 317.4 | 1719 | 1.00 | 76 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 53.0 | 144.4 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 89.9 | 705.3 | 1.00 | 316.5 | 1816 | 1.00 | 76 |
| short-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 62.3 | 249.5 | 0.69 | 251.8 | 1249 | 0.80 | 77 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 54.3 | 148.1 | 0.60 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@main | 89.9 | 704.2 | 1.00 | 319.7 | 1722 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.1 | 144.6 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 90.3 | 708.4 | 1.00 | 319.8 | 1809 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 61.3 | 248.6 | 0.68 | 251.6 | 1230 | 0.79 | 77 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 54.0 | 144.5 | 0.60 | - | - | - | 33 |
| short-4096-sliding-cp8r7 | tilelang@main | 90.8 | 684.0 | 1.00 | 323.9 | 1762 | 1.00 | 76 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 54.0 | 141.7 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 90.5 | 703.2 | 1.00 | 322.9 | 1793 | 1.00 | 76 |
| short-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 62.4 | 253.6 | 0.69 | 250.4 | 1219 | 0.78 | 77 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 53.6 | 150.2 | 0.59 | - | - | - | 33 |
| heavy-4096-csa-cp1 | tilelang@main | 770.8 | 691.6 | 1.00 | 2718 | 827.3 | 1.00 | 530 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 404.9 | 120.0 | 0.53 | - | - | - | 278 |
| heavy-4096-csa-cp1 | tilelang@cudnn_flashmla | 771.9 | 692.3 | 1.00 | 2722 | 848.4 | 1.00 | 530 |
| heavy-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 414.3 | 182.5 | 0.54 | 1698 | 534.5 | 0.62 | 542 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 406.3 | 124.8 | 0.53 | - | - | - | 278 |
| heavy-4096-csa-cp8r0 | tilelang@main | 111.8 | 682.7 | 1.00 | 410.6 | 1622 | 1.00 | 79 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 68.1 | 140.3 | 0.61 | - | - | - | 35 |
| heavy-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 112.0 | 720.6 | 1.00 | 409.8 | 1718 | 1.00 | 79 |
| heavy-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 76.0 | 238.1 | 0.68 | 305.6 | 1181 | 0.75 | 81 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 67.9 | 146.4 | 0.61 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang@main | 114.6 | 703.3 | 1.00 | 434.1 | 1615 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.1 | 145.2 | 0.63 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 114.1 | 705.1 | 1.00 | 434.5 | 1691 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 80.9 | 223.6 | 0.71 | 318.1 | 1161 | 0.73 | 81 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 72.6 | 140.8 | 0.64 | - | - | - | 35 |
| heavy-4096-csa-cp8r7 | tilelang@main | 105.7 | 744.1 | 1.00 | 378.3 | 1711 | 1.00 | 79 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 68.6 | 147.1 | 0.65 | - | - | - | 35 |
| heavy-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 105.9 | 695.9 | 1.00 | 378.1 | 1722 | 1.00 | 79 |
| heavy-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 76.6 | 229.7 | 0.72 | 286.3 | 1172 | 0.76 | 81 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 68.3 | 142.9 | 0.64 | - | - | - | 35 |
| heavy-4096-hca-cp1 | tilelang@main | 647.0 | 727.5 | 1.00 | 2030 | 952.6 | 1.00 | 530 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 343.9 | 138.8 | 0.53 | - | - | - | 266 |
| heavy-4096-hca-cp1 | tilelang@cudnn_flashmla | 646.9 | 747.1 | 1.00 | 2023 | 1009 | 1.00 | 530 |
| heavy-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 350.6 | 207.9 | 0.54 | 1394 | 633.6 | 0.69 | 533 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 339.8 | 151.6 | 0.53 | - | - | - | 266 |
| heavy-4096-hca-cp8r0 | tilelang@main | 100.5 | 735.0 | 1.00 | 348.5 | 1804 | 1.00 | 77 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 65.0 | 149.5 | 0.65 | - | - | - | 33 |
| heavy-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 101.2 | 763.2 | 1.00 | 349.8 | 1843 | 1.00 | 77 |
| heavy-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 72.8 | 242.3 | 0.72 | 273.3 | 1207 | 0.78 | 77 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 64.3 | 148.0 | 0.64 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang@main | 102.2 | 725.7 | 1.00 | 357.1 | 1803 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 64.6 | 142.8 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 102.3 | 724.8 | 1.00 | 356.9 | 1883 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 73.0 | 240.9 | 0.71 | 274.5 | 1214 | 0.77 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 64.6 | 147.6 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r7 | tilelang@main | 97.6 | 740.6 | 1.00 | 338.4 | 1841 | 1.00 | 77 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 61.8 | 154.0 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 97.3 | 757.0 | 1.00 | 336.4 | 1883 | 1.00 | 77 |
| heavy-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 69.5 | 239.3 | 0.71 | 261.6 | 1221 | 0.78 | 77 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 61.2 | 145.3 | 0.63 | - | - | - | 33 |
| heavy-4096-sliding-cp1 | tilelang@main | 582.4 | 700.3 | 1.00 | 1840 | 1071 | 1.00 | 527 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 269.1 | 149.7 | 0.46 | - | - | - | 262 |
| heavy-4096-sliding-cp1 | tilelang@cudnn_flashmla | 581.8 | 704.3 | 1.00 | 1839 | 1013 | 1.00 | 527 |
| heavy-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 283.8 | 195.3 | 0.49 | 1259 | 703.3 | 0.68 | 531 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 271.9 | 145.5 | 0.47 | - | - | - | 262 |
| heavy-4096-sliding-cp8r0 | tilelang@main | 88.9 | 704.9 | 1.00 | 318.0 | 1765 | 1.00 | 76 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 52.5 | 146.2 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 89.9 | 720.6 | 1.00 | 317.2 | 1826 | 1.00 | 76 |
| heavy-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 62.2 | 251.0 | 0.69 | 250.7 | 1224 | 0.79 | 77 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 53.7 | 149.5 | 0.60 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@main | 90.6 | 690.0 | 1.00 | 321.4 | 1729 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.9 | 140.4 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 90.7 | 703.6 | 1.00 | 321.4 | 1786 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 61.3 | 246.6 | 0.68 | 254.6 | 1206 | 0.79 | 77 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 54.2 | 145.4 | 0.60 | - | - | - | 33 |
| heavy-4096-sliding-cp8r7 | tilelang@main | 87.9 | 684.1 | 1.00 | 309.0 | 1790 | 1.00 | 76 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.5 | 144.1 | 0.61 | - | - | - | 33 |
| heavy-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 87.7 | 700.8 | 1.00 | 309.1 | 1791 | 1.00 | 76 |
| heavy-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 62.0 | 253.6 | 0.71 | 244.1 | 1229 | 0.79 | 77 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 53.7 | 148.2 | 0.61 | - | - | - | 33 |
| tiny-4096-csa-cp1 | tilelang@main | 680.3 | 693.4 | 1.00 | 2071 | 848.0 | 1.00 | 530 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 372.6 | 128.9 | 0.55 | - | - | - | 278 |
| tiny-4096-csa-cp1 | tilelang@cudnn_flashmla | 679.8 | 710.8 | 1.00 | 2073 | 849.7 | 1.00 | 530 |
| tiny-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 380.5 | 174.5 | 0.56 | 1392 | 559.1 | 0.67 | 542 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 371.0 | 127.4 | 0.55 | - | - | - | 278 |
| tiny-4096-csa-cp8r0 | tilelang@main | 103.6 | 692.9 | 1.00 | 340.7 | 1680 | 1.00 | 79 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 66.6 | 144.3 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 103.2 | 688.9 | 1.00 | 340.8 | 1782 | 1.00 | 79 |
| tiny-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 74.7 | 231.8 | 0.72 | 267.0 | 1234 | 0.78 | 81 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 67.0 | 142.1 | 0.65 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang@main | 104.5 | 704.8 | 1.00 | 345.8 | 1688 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 67.1 | 142.5 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 104.2 | 707.5 | 1.00 | 345.7 | 1777 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 75.5 | 233.5 | 0.72 | 272.9 | 1222 | 0.79 | 81 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 67.2 | 140.6 | 0.65 | - | - | - | 35 |
| tiny-4096-csa-cp8r7 | tilelang@main | 104.5 | 681.6 | 1.00 | 346.0 | 1697 | 1.00 | 79 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 67.8 | 142.8 | 0.65 | - | - | - | 35 |
| tiny-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 105.3 | 683.8 | 1.00 | 345.3 | 1757 | 1.00 | 79 |
| tiny-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 75.7 | 223.8 | 0.72 | 274.2 | 1202 | 0.79 | 81 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 67.9 | 140.8 | 0.64 | - | - | - | 35 |
| tiny-4096-hca-cp1 | tilelang@main | 532.2 | 695.1 | 1.00 | 1436 | 973.1 | 1.00 | 527 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 273.7 | 138.6 | 0.51 | - | - | - | 262 |
| tiny-4096-hca-cp1 | tilelang@cudnn_flashmla | 529.7 | 694.2 | 1.00 | 1435 | 1027 | 1.00 | 527 |
| tiny-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 283.1 | 197.7 | 0.53 | 1087 | 690.9 | 0.76 | 531 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 273.9 | 143.3 | 0.52 | - | - | - | 262 |
| tiny-4096-hca-cp8r0 | tilelang@main | 81.8 | 688.6 | 1.00 | 254.2 | 1778 | 1.00 | 76 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 52.5 | 142.3 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 82.4 | 709.2 | 1.00 | 254.2 | 1869 | 1.00 | 76 |
| tiny-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 61.5 | 252.2 | 0.75 | 225.0 | 1240 | 0.89 | 77 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 53.9 | 147.5 | 0.65 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang@main | 84.2 | 683.2 | 1.00 | 267.7 | 1755 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 53.3 | 143.4 | 0.63 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 84.8 | 709.5 | 1.00 | 267.1 | 1887 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 61.6 | 247.1 | 0.73 | 227.2 | 1265 | 0.85 | 77 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 53.5 | 146.2 | 0.63 | - | - | - | 33 |
| tiny-4096-hca-cp8r7 | tilelang@main | 82.2 | 701.9 | 1.00 | 260.4 | 1793 | 1.00 | 76 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 52.8 | 148.7 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 82.5 | 709.3 | 1.00 | 260.3 | 1861 | 1.00 | 76 |
| tiny-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 61.1 | 249.6 | 0.74 | 225.2 | 1247 | 0.86 | 77 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 53.7 | 148.5 | 0.65 | - | - | - | 33 |
| tiny-4096-sliding-cp1 | tilelang@main | 534.3 | 692.2 | 1.00 | 1439 | 968.5 | 1.00 | 527 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 269.6 | 139.2 | 0.50 | - | - | - | 262 |
| tiny-4096-sliding-cp1 | tilelang@cudnn_flashmla | 530.3 | 697.2 | 1.00 | 1437 | 1048 | 1.00 | 527 |
| tiny-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 283.9 | 195.0 | 0.54 | 1087 | 701.4 | 0.76 | 531 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 272.0 | 144.9 | 0.51 | - | - | - | 262 |
| tiny-4096-sliding-cp8r0 | tilelang@main | 81.9 | 693.4 | 1.00 | 254.8 | 1776 | 1.00 | 76 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 52.1 | 148.6 | 0.64 | - | - | - | 33 |
| tiny-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 82.6 | 694.7 | 1.00 | 253.9 | 1865 | 1.00 | 76 |
| tiny-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 60.5 | 243.4 | 0.73 | 225.4 | 1249 | 0.89 | 77 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 53.5 | 141.6 | 0.65 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@main | 84.0 | 697.6 | 1.00 | 267.6 | 1772 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 52.9 | 150.1 | 0.63 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 84.4 | 721.9 | 1.00 | 266.8 | 1847 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 62.1 | 242.4 | 0.74 | 227.3 | 1315 | 0.85 | 77 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 53.9 | 141.0 | 0.64 | - | - | - | 33 |
| tiny-4096-sliding-cp8r7 | tilelang@main | 82.0 | 704.4 | 1.00 | 259.8 | 1821 | 1.00 | 76 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 53.1 | 149.0 | 0.65 | - | - | - | 33 |
| tiny-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 82.3 | 722.7 | 1.00 | 261.0 | 1889 | 1.00 | 76 |
| tiny-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 61.6 | 253.3 | 0.75 | 224.6 | 1290 | 0.86 | 77 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 53.1 | 150.6 | 0.64 | - | - | - | 33 |
| single-16384-csa-cp1 | tilelang@main | 5358 | 559.6 | 1.00 | 22675 | 885.3 | 1.00 | 2120 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 2542 | 28.1 | 0.47 | - | - | - | 1112 |
| single-16384-csa-cp1 | tilelang@cudnn_flashmla | 5439 | 580.6 | 1.00 | 23256 | 620.3 | 1.00 | 2120 |
| single-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 2551 | 86.2 | 0.47 | 12139 | 614.8 | 0.52 | 2168 |
| single-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 2535 | 142.9 | 0.47 | - | - | - | 1112 |
| single-16384-csa-cp8r0 | tilelang@main | 539.8 | 692.6 | 1.00 | 2182 | 1062 | 1.00 | 318 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 283.6 | 134.0 | 0.53 | - | - | - | 139 |
| single-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 542.3 | 698.3 | 1.00 | 2179 | 1121 | 1.00 | 318 |
| single-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 292.5 | 189.6 | 0.54 | 1289 | 677.2 | 0.59 | 324 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 286.1 | 131.1 | 0.53 | - | - | - | 139 |
| single-16384-csa-cp8r4 | tilelang@main | 710.9 | 698.0 | 1.00 | 3186 | 893.3 | 1.00 | 318 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 359.6 | 135.6 | 0.51 | - | - | - | 139 |
| single-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 712.1 | 711.1 | 1.00 | 3176 | 947.3 | 1.00 | 318 |
| single-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 367.7 | 181.3 | 0.52 | 1733 | 597.8 | 0.55 | 324 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 357.4 | 131.1 | 0.50 | - | - | - | 139 |
| single-16384-csa-cp8r7 | tilelang@main | 711.0 | 694.1 | 1.00 | 3201 | 903.7 | 1.00 | 318 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 358.9 | 137.7 | 0.50 | - | - | - | 139 |
| single-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 712.3 | 714.1 | 1.00 | 3172 | 953.6 | 1.00 | 318 |
| single-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 365.2 | 191.6 | 0.51 | 1736 | 585.3 | 0.55 | 324 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 358.9 | 140.1 | 0.50 | - | - | - | 139 |
| single-16384-hca-cp1 | tilelang@main | 2873 | 700.6 | 1.00 | 10008 | 896.7 | 1.00 | 2108 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1399 | 90.2 | 0.49 | - | - | - | 1064 |
| single-16384-hca-cp1 | tilelang@cudnn_flashmla | 2852 | 729.7 | 1.00 | 9947 | 922.3 | 1.00 | 2108 |
| single-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1407 | 163.0 | 0.49 | 6237 | 294.7 | 0.63 | 2132 |
| single-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1386 | 102.2 | 0.49 | - | - | - | 1064 |
| single-16384-hca-cp8r0 | tilelang@main | 360.1 | 720.7 | 1.00 | 1178 | 1245 | 1.00 | 306 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 196.9 | 143.6 | 0.55 | - | - | - | 133 |
| single-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 358.6 | 697.4 | 1.00 | 1177 | 1293 | 1.00 | 306 |
| single-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 206.5 | 193.1 | 0.58 | 826.5 | 770.0 | 0.70 | 309 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 197.3 | 140.5 | 0.55 | - | - | - | 133 |
| single-16384-hca-cp8r4 | tilelang@main | 411.5 | 685.4 | 1.00 | 1468 | 1197 | 1.00 | 306 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 200.6 | 139.3 | 0.49 | - | - | - | 133 |
| single-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 410.8 | 701.3 | 1.00 | 1466 | 1310 | 1.00 | 306 |
| single-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 209.3 | 192.5 | 0.51 | 939.2 | 780.0 | 0.64 | 309 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 199.4 | 142.4 | 0.49 | - | - | - | 133 |
| single-16384-hca-cp8r7 | tilelang@main | 416.3 | 690.1 | 1.00 | 1589 | 1230 | 1.00 | 306 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 200.7 | 143.1 | 0.48 | - | - | - | 133 |
| single-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 414.5 | 696.9 | 1.00 | 1589 | 1238 | 1.00 | 306 |
| single-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 209.1 | 194.1 | 0.50 | 974.2 | 752.4 | 0.61 | 309 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 200.8 | 141.3 | 0.48 | - | - | - | 133 |
| single-16384-sliding-cp1 | tilelang@main | 2290 | 682.5 | 1.00 | 7470 | 816.3 | 1.00 | 2108 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 1049 | 115.4 | 0.46 | - | - | - | 1048 |
| single-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2291 | 696.6 | 1.00 | 7433 | 867.0 | 1.00 | 2108 |
| single-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1058 | 183.9 | 0.46 | 5003 | 331.6 | 0.67 | 2124 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1046 | 121.7 | 0.46 | - | - | - | 1048 |
| single-16384-sliding-cp8r0 | tilelang@main | 310.8 | 707.7 | 1.00 | 1046 | 1288 | 1.00 | 306 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 150.5 | 149.3 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 312.9 | 704.9 | 1.00 | 1046 | 1361 | 1.00 | 306 |
| single-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 158.4 | 196.1 | 0.51 | 733.8 | 811.8 | 0.70 | 308 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 147.9 | 145.4 | 0.47 | - | - | - | 131 |
| single-16384-sliding-cp8r4 | tilelang@main | 315.2 | 689.1 | 1.00 | 1080 | 1310 | 1.00 | 306 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 150.5 | 146.2 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 313.8 | 690.4 | 1.00 | 1086 | 1311 | 1.00 | 306 |
| single-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 159.0 | 191.5 | 0.51 | 744.4 | 808.1 | 0.69 | 308 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 148.6 | 141.0 | 0.47 | - | - | - | 131 |
| single-16384-sliding-cp8r7 | tilelang@main | 314.2 | 682.0 | 1.00 | 1084 | 1305 | 1.00 | 306 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 150.6 | 143.4 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 316.0 | 696.4 | 1.00 | 1083 | 1332 | 1.00 | 306 |
| single-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 160.9 | 192.8 | 0.51 | 748.4 | 797.8 | 0.69 | 308 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 151.1 | 144.4 | 0.48 | - | - | - | 131 |
| short-16384-csa-cp1 | tilelang@main | 3853 | 656.0 | 1.00 | 15075 | 901.5 | 1.00 | 2120 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 1929 | 28.0 | 0.50 | - | - | - | 1112 |
| short-16384-csa-cp1 | tilelang@cudnn_flashmla | 3822 | 698.0 | 1.00 | 15223 | 848.1 | 1.00 | 2120 |
| short-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1933 | 80.3 | 0.51 | 8658 | 225.0 | 0.57 | 2168 |
| short-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1936 | 17.3 | 0.51 | - | - | - | 1112 |
| short-16384-csa-cp8r0 | tilelang@main | 478.1 | 697.2 | 1.00 | 1852 | 1145 | 1.00 | 317 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 254.2 | 135.9 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 481.8 | 692.2 | 1.00 | 1859 | 1186 | 1.00 | 317 |
| short-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 263.9 | 187.9 | 0.55 | 1132 | 720.1 | 0.61 | 324 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 253.6 | 132.6 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r4 | tilelang@main | 566.4 | 683.1 | 1.00 | 2358 | 1029 | 1.00 | 317 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 295.5 | 136.1 | 0.52 | - | - | - | 139 |
| short-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 566.3 | 683.1 | 1.00 | 2359 | 1091 | 1.00 | 317 |
| short-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 304.8 | 185.2 | 0.54 | 1363 | 665.7 | 0.58 | 324 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 295.5 | 136.3 | 0.52 | - | - | - | 139 |
| short-16384-csa-cp8r7 | tilelang@main | 524.8 | 685.9 | 1.00 | 2140 | 1101 | 1.00 | 317 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 276.9 | 131.2 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 524.2 | 700.6 | 1.00 | 2138 | 1095 | 1.00 | 317 |
| short-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 285.0 | 188.1 | 0.54 | 1259 | 688.3 | 0.59 | 324 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 276.1 | 137.5 | 0.53 | - | - | - | 139 |
| short-16384-hca-cp1 | tilelang@main | 2599 | 738.1 | 1.00 | 8299 | 926.1 | 1.00 | 2120 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 1361 | 102.2 | 0.52 | - | - | - | 1064 |
| short-16384-hca-cp1 | tilelang@cudnn_flashmla | 2612 | 713.7 | 1.00 | 8241 | 903.7 | 1.00 | 2120 |
| short-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1390 | 153.6 | 0.53 | 5590 | 292.2 | 0.68 | 2132 |
| short-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1377 | 84.9 | 0.53 | - | - | - | 1064 |
| short-16384-hca-cp8r0 | tilelang@main | 349.8 | 720.5 | 1.00 | 1148 | 1351 | 1.00 | 307 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 196.9 | 142.0 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 348.0 | 734.7 | 1.00 | 1149 | 1370 | 1.00 | 307 |
| short-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 204.9 | 197.2 | 0.59 | 817.8 | 757.1 | 0.71 | 309 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 194.2 | 143.5 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r4 | tilelang@main | 362.1 | 717.9 | 1.00 | 1211 | 1322 | 1.00 | 307 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 200.7 | 136.7 | 0.55 | - | - | - | 133 |
| short-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 361.2 | 726.4 | 1.00 | 1213 | 1393 | 1.00 | 307 |
| short-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 208.1 | 194.1 | 0.58 | 843.5 | 765.6 | 0.70 | 309 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 199.8 | 141.5 | 0.55 | - | - | - | 133 |
| short-16384-hca-cp8r7 | tilelang@main | 355.1 | 726.0 | 1.00 | 1179 | 1380 | 1.00 | 307 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 198.1 | 140.8 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 354.2 | 727.5 | 1.00 | 1177 | 1382 | 1.00 | 307 |
| short-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 206.2 | 199.1 | 0.58 | 826.7 | 768.2 | 0.70 | 309 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 195.3 | 145.5 | 0.55 | - | - | - | 133 |
| short-16384-sliding-cp1 | tilelang@main | 2264 | 720.8 | 1.00 | 7323 | 832.1 | 1.00 | 2108 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 1041 | 128.7 | 0.46 | - | - | - | 1048 |
| short-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2268 | 689.7 | 1.00 | 7274 | 868.4 | 1.00 | 2108 |
| short-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1062 | 175.8 | 0.47 | 4935 | 338.0 | 0.68 | 2124 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1033 | 129.7 | 0.46 | - | - | - | 1048 |
| short-16384-sliding-cp8r0 | tilelang@main | 303.6 | 697.8 | 1.00 | 1025 | 1281 | 1.00 | 306 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 149.7 | 146.6 | 0.49 | - | - | - | 131 |
| short-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 304.4 | 697.4 | 1.00 | 1023 | 1380 | 1.00 | 306 |
| short-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 159.2 | 200.2 | 0.52 | 726.6 | 830.3 | 0.71 | 308 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 148.2 | 145.4 | 0.49 | - | - | - | 131 |
| short-16384-sliding-cp8r4 | tilelang@main | 314.2 | 679.7 | 1.00 | 1072 | 1298 | 1.00 | 306 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 147.1 | 147.3 | 0.47 | - | - | - | 131 |
| short-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 313.9 | 697.7 | 1.00 | 1069 | 1315 | 1.00 | 306 |
| short-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 159.2 | 194.6 | 0.51 | 735.4 | 797.9 | 0.69 | 308 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 150.2 | 139.8 | 0.48 | - | - | - | 131 |
| short-16384-sliding-cp8r7 | tilelang@main | 312.3 | 689.1 | 1.00 | 1051 | 1310 | 1.00 | 306 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 150.9 | 143.6 | 0.48 | - | - | - | 131 |
| short-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 311.7 | 704.3 | 1.00 | 1053 | 1356 | 1.00 | 306 |
| short-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 158.7 | 203.8 | 0.51 | 734.6 | 811.4 | 0.70 | 308 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 150.6 | 147.0 | 0.48 | - | - | - | 131 |
| heavy-16384-csa-cp1 | tilelang@main | 3305 | 664.5 | 1.00 | 12082 | 914.7 | 1.00 | 2120 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1691 | 34.9 | 0.51 | - | - | - | 1112 |
| heavy-16384-csa-cp1 | tilelang@cudnn_flashmla | 3291 | 658.4 | 1.00 | 12028 | 865.2 | 1.00 | 2120 |
| heavy-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1698 | 90.7 | 0.52 | 7229 | 214.1 | 0.60 | 2168 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1694 | 28.1 | 0.51 | - | - | - | 1112 |
| heavy-16384-csa-cp8r0 | tilelang@main | 395.5 | 697.0 | 1.00 | 1364 | 1211 | 1.00 | 317 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 213.5 | 133.0 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 393.5 | 700.9 | 1.00 | 1364 | 1297 | 1.00 | 317 |
| heavy-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 223.0 | 190.5 | 0.57 | 903.5 | 762.6 | 0.66 | 323 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 214.9 | 138.2 | 0.55 | - | - | - | 139 |
| heavy-16384-csa-cp8r4 | tilelang@main | 433.6 | 683.1 | 1.00 | 1614 | 1148 | 1.00 | 317 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 232.8 | 129.9 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 431.4 | 696.1 | 1.00 | 1609 | 1238 | 1.00 | 317 |
| heavy-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 241.0 | 193.2 | 0.56 | 1018 | 727.4 | 0.63 | 323 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 234.7 | 135.9 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r7 | tilelang@main | 579.5 | 684.9 | 1.00 | 2413 | 1069 | 1.00 | 317 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 301.6 | 126.9 | 0.52 | - | - | - | 139 |
| heavy-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 576.9 | 703.6 | 1.00 | 2407 | 1073 | 1.00 | 317 |
| heavy-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 308.8 | 190.1 | 0.54 | 1386 | 655.1 | 0.58 | 323 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 301.4 | 136.6 | 0.52 | - | - | - | 139 |
| heavy-16384-hca-cp1 | tilelang@main | 2507 | 744.8 | 1.00 | 7784 | 921.8 | 1.00 | 2120 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 1309 | 94.9 | 0.52 | - | - | - | 1064 |
| heavy-16384-hca-cp1 | tilelang@cudnn_flashmla | 2502 | 724.1 | 1.00 | 7751 | 926.8 | 1.00 | 2120 |
| heavy-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1314 | 165.4 | 0.53 | 5320 | 310.0 | 0.69 | 2132 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1300 | 101.9 | 0.52 | - | - | - | 1064 |
| heavy-16384-hca-cp8r0 | tilelang@main | 333.5 | 734.1 | 1.00 | 1069 | 1379 | 1.00 | 308 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 184.9 | 150.6 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 332.3 | 723.2 | 1.00 | 1071 | 1389 | 1.00 | 308 |
| heavy-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 193.1 | 193.3 | 0.58 | 768.9 | 776.0 | 0.72 | 309 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 183.8 | 141.9 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r4 | tilelang@main | 343.8 | 725.9 | 1.00 | 1127 | 1361 | 1.00 | 308 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 190.5 | 151.2 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 345.1 | 726.3 | 1.00 | 1129 | 1421 | 1.00 | 308 |
| heavy-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 201.6 | 194.8 | 0.58 | 800.3 | 797.5 | 0.71 | 309 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 190.6 | 145.6 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r7 | tilelang@main | 366.8 | 721.2 | 1.00 | 1231 | 1390 | 1.00 | 308 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 203.8 | 142.3 | 0.56 | - | - | - | 133 |
| heavy-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 366.2 | 732.3 | 1.00 | 1234 | 1405 | 1.00 | 308 |
| heavy-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 210.1 | 201.8 | 0.57 | 852.3 | 783.4 | 0.69 | 309 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 200.7 | 148.8 | 0.55 | - | - | - | 133 |
| heavy-16384-sliding-cp1 | tilelang@main | 2247 | 731.8 | 1.00 | 7046 | 841.1 | 1.00 | 2108 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 1042 | 130.6 | 0.46 | - | - | - | 1048 |
| heavy-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2254 | 692.9 | 1.00 | 7020 | 840.6 | 1.00 | 2108 |
| heavy-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1072 | 169.7 | 0.48 | 4822 | 332.3 | 0.69 | 2124 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1046 | 120.3 | 0.46 | - | - | - | 1048 |
| heavy-16384-sliding-cp8r0 | tilelang@main | 300.8 | 712.1 | 1.00 | 974.6 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 148.3 | 149.4 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 301.0 | 695.6 | 1.00 | 973.9 | 1376 | 1.00 | 306 |
| heavy-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 158.4 | 199.0 | 0.53 | 702.1 | 823.7 | 0.72 | 308 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 151.2 | 144.1 | 0.50 | - | - | - | 131 |
| heavy-16384-sliding-cp8r4 | tilelang@main | 303.7 | 679.0 | 1.00 | 1014 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 149.5 | 143.6 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 304.8 | 709.7 | 1.00 | 1016 | 1339 | 1.00 | 306 |
| heavy-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 159.0 | 197.3 | 0.52 | 718.5 | 804.4 | 0.71 | 308 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 148.0 | 146.5 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r7 | tilelang@main | 314.8 | 707.2 | 1.00 | 1082 | 1323 | 1.00 | 306 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 147.4 | 149.0 | 0.47 | - | - | - | 131 |
| heavy-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 315.0 | 701.3 | 1.00 | 1083 | 1408 | 1.00 | 306 |
| heavy-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 158.6 | 202.1 | 0.50 | 748.9 | 823.0 | 0.69 | 308 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 149.4 | 151.1 | 0.47 | - | - | - | 131 |
| tiny-16384-csa-cp1 | tilelang@main | 2636 | 650.2 | 1.00 | 7892 | 833.7 | 1.00 | 2120 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 1437 | 39.5 | 0.54 | - | - | - | 1112 |
| tiny-16384-csa-cp1 | tilelang@cudnn_flashmla | 2638 | 673.7 | 1.00 | 7894 | 803.2 | 1.00 | 2120 |
| tiny-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1441 | 96.6 | 0.55 | 5339 | 210.6 | 0.68 | 2168 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1446 | 30.9 | 0.55 | - | - | - | 1112 |
| tiny-16384-csa-cp8r0 | tilelang@main | 354.2 | 688.8 | 1.00 | 1109 | 1267 | 1.00 | 317 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 200.1 | 134.4 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 352.6 | 708.9 | 1.00 | 1108 | 1310 | 1.00 | 317 |
| tiny-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 208.2 | 191.6 | 0.59 | 779.2 | 765.4 | 0.70 | 323 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 199.6 | 137.5 | 0.57 | - | - | - | 139 |
| tiny-16384-csa-cp8r4 | tilelang@main | 356.2 | 692.0 | 1.00 | 1108 | 1250 | 1.00 | 317 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 200.5 | 141.7 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 356.2 | 690.2 | 1.00 | 1107 | 1290 | 1.00 | 317 |
| tiny-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 208.1 | 189.5 | 0.58 | 778.9 | 750.2 | 0.70 | 323 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 200.7 | 142.5 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r7 | tilelang@main | 356.2 | 675.8 | 1.00 | 1099 | 1270 | 1.00 | 317 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 200.3 | 134.2 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 355.1 | 700.7 | 1.00 | 1097 | 1277 | 1.00 | 317 |
| tiny-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 210.6 | 183.2 | 0.59 | 771.6 | 739.7 | 0.70 | 323 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 200.9 | 135.0 | 0.57 | - | - | - | 139 |
| tiny-16384-hca-cp1 | tilelang@main | 2024 | 683.5 | 1.00 | 5355 | 828.5 | 1.00 | 2108 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 1052 | 124.0 | 0.52 | - | - | - | 1048 |
| tiny-16384-hca-cp1 | tilelang@cudnn_flashmla | 2021 | 720.5 | 1.00 | 5351 | 849.0 | 1.00 | 2108 |
| tiny-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1068 | 193.5 | 0.53 | 4154 | 327.1 | 0.78 | 2124 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1052 | 127.2 | 0.52 | - | - | - | 1048 |
| tiny-16384-hca-cp8r0 | tilelang@main | 278.4 | 702.7 | 1.00 | 784.0 | 1360 | 1.00 | 306 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 149.1 | 147.5 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 278.8 | 693.2 | 1.00 | 784.8 | 1390 | 1.00 | 306 |
| tiny-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 157.7 | 201.7 | 0.57 | 620.9 | 872.9 | 0.79 | 308 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 147.7 | 145.9 | 0.53 | - | - | - | 131 |
| tiny-16384-hca-cp8r4 | tilelang@main | 276.8 | 691.4 | 1.00 | 771.3 | 1324 | 1.00 | 306 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 148.8 | 144.3 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 276.1 | 711.0 | 1.00 | 772.5 | 1402 | 1.00 | 306 |
| tiny-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 157.6 | 201.3 | 0.57 | 616.9 | 884.4 | 0.80 | 308 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 148.1 | 147.2 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r7 | tilelang@main | 268.8 | 686.6 | 1.00 | 754.2 | 1376 | 1.00 | 306 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 149.0 | 145.3 | 0.55 | - | - | - | 131 |
| tiny-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 267.9 | 706.5 | 1.00 | 752.4 | 1394 | 1.00 | 306 |
| tiny-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 157.8 | 201.5 | 0.59 | 607.4 | 877.6 | 0.81 | 308 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 146.5 | 152.7 | 0.55 | - | - | - | 131 |
| tiny-16384-sliding-cp1 | tilelang@main | 2025 | 703.4 | 1.00 | 5346 | 817.0 | 1.00 | 2108 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 1050 | 130.1 | 0.52 | - | - | - | 1048 |
| tiny-16384-sliding-cp1 | tilelang@cudnn_flashmla | 2022 | 709.5 | 1.00 | 5347 | 852.4 | 1.00 | 2108 |
| tiny-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1067 | 188.2 | 0.53 | 4157 | 316.5 | 0.78 | 2124 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1055 | 125.2 | 0.52 | - | - | - | 1048 |
| tiny-16384-sliding-cp8r0 | tilelang@main | 278.4 | 684.2 | 1.00 | 784.9 | 1357 | 1.00 | 306 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 147.7 | 149.3 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 278.9 | 711.2 | 1.00 | 783.4 | 1404 | 1.00 | 306 |
| tiny-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 158.0 | 202.3 | 0.57 | 618.6 | 897.8 | 0.79 | 308 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 148.9 | 150.3 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r4 | tilelang@main | 276.5 | 685.8 | 1.00 | 772.0 | 1402 | 1.00 | 306 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 147.7 | 151.4 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 277.2 | 697.4 | 1.00 | 773.7 | 1366 | 1.00 | 306 |
| tiny-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 159.9 | 195.5 | 0.58 | 616.8 | 866.0 | 0.80 | 308 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 148.5 | 143.8 | 0.54 | - | - | - | 131 |
| tiny-16384-sliding-cp8r7 | tilelang@main | 267.7 | 693.4 | 1.00 | 754.5 | 1385 | 1.00 | 306 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 148.1 | 145.0 | 0.55 | - | - | - | 131 |
| tiny-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 267.5 | 693.4 | 1.00 | 753.1 | 1396 | 1.00 | 306 |
| tiny-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 157.3 | 199.9 | 0.59 | 612.6 | 890.8 | 0.81 | 308 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 148.5 | 145.0 | 0.56 | - | - | - | 131 |
| single-49208-csa-cp1 | tilelang@main | 16237 | 187.0 | 1.00 | 69993 | 715.1 | 1.00 | 6368 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 7737 | 571.7 | 0.48 | - | - | - | 3340 |
| single-49208-csa-cp1 | tilelang@cudnn_flashmla | 16575 | -92.6 | 1.00 | 70558 | 452.3 | 1.00 | 6368 |
| single-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 7756 | 630.9 | 0.47 | 39257 | 182.3 | 0.56 | 6513 |
| single-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 7875 | 667.2 | 0.48 | - | - | - | 3340 |
| single-49208-csa-cp8r0 | tilelang@main | 1866 | 689.5 | 1.00 | 8084 | 839.4 | 1.00 | 954 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 922.1 | 104.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 1874 | 710.0 | 1.00 | 8209 | 850.5 | 1.00 | 954 |
| single-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 928.9 | 171.8 | 0.50 | 4418 | 351.6 | 0.54 | 972 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 923.7 | 98.4 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r4 | tilelang@main | 2035 | 709.5 | 1.00 | 9123 | 902.5 | 1.00 | 954 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 999.9 | 100.6 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 2052 | 709.5 | 1.00 | 9297 | 829.3 | 1.00 | 954 |
| single-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1009 | 167.6 | 0.49 | 4901 | 335.3 | 0.53 | 973 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1002 | 103.8 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r7 | tilelang@main | 2039 | 705.7 | 1.00 | 9444 | 917.3 | 1.00 | 954 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1003 | 101.6 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 2051 | 728.4 | 1.00 | 9671 | 785.6 | 1.00 | 954 |
| single-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1012 | 162.8 | 0.49 | 4985 | 326.8 | 0.52 | 973 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1005 | 100.3 | 0.49 | - | - | - | 418 |
| single-49208-hca-cp1 | tilelang@main | 11188 | 307.9 | 1.00 | 41810 | 748.7 | 1.00 | 6334 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 5350 | 51.8 | 0.48 | - | - | - | 3292 |
| single-49208-hca-cp1 | tilelang@cudnn_flashmla | 11133 | 360.4 | 1.00 | 42108 | 704.7 | 1.00 | 6334 |
| single-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 5374 | 150.1 | 0.48 | 24444 | 580.9 | 0.58 | 6454 |
| single-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5356 | 260.1 | 0.48 | - | - | - | 3292 |
| single-49208-hca-cp8r0 | tilelang@main | 1029 | 680.6 | 1.00 | 3422 | 825.2 | 1.00 | 919 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 555.3 | 115.3 | 0.54 | - | - | - | 412 |
| single-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 1030 | 701.5 | 1.00 | 3427 | 856.2 | 1.00 | 919 |
| single-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 559.1 | 180.9 | 0.54 | 2335 | 392.8 | 0.68 | 934 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 553.2 | 118.2 | 0.54 | - | - | - | 412 |
| single-49208-hca-cp8r4 | tilelang@main | 1459 | 686.0 | 1.00 | 5739 | 825.7 | 1.00 | 919 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 691.9 | 109.3 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 1458 | 704.5 | 1.00 | 5742 | 849.6 | 1.00 | 919 |
| single-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 702.6 | 163.5 | 0.48 | 3291 | 331.7 | 0.57 | 934 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 689.2 | 111.5 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r7 | tilelang@main | 1762 | 668.0 | 1.00 | 7389 | 854.6 | 1.00 | 919 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 831.5 | 113.1 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 1747 | 699.0 | 1.00 | 7375 | 891.1 | 1.00 | 919 |
| single-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 837.3 | 174.3 | 0.48 | 4030 | 333.9 | 0.55 | 934 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 825.2 | 118.4 | 0.47 | - | - | - | 412 |
| single-49208-sliding-cp1 | tilelang@main | 6768 | 664.9 | 1.00 | 22045 | 843.6 | 1.00 | 6332 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 3083 | 41.8 | 0.46 | - | - | - | 3148 |
| single-49208-sliding-cp1 | tilelang@cudnn_flashmla | 6776 | 648.9 | 1.00 | 22051 | 859.0 | 1.00 | 6332 |
| single-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3115 | 108.1 | 0.46 | 15055 | 330.5 | 0.68 | 6380 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3078 | 37.2 | 0.45 | - | - | - | 3148 |
| single-49208-sliding-cp8r0 | tilelang@main | 873.5 | 700.3 | 1.00 | 2906 | 838.8 | 1.00 | 918 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 402.7 | 140.4 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 874.3 | 721.5 | 1.00 | 2912 | 857.1 | 1.00 | 918 |
| single-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 417.4 | 195.3 | 0.48 | 1991 | 548.0 | 0.68 | 925 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 398.5 | 146.5 | 0.46 | - | - | - | 393 |
| single-49208-sliding-cp8r4 | tilelang@main | 878.6 | 678.7 | 1.00 | 2949 | 833.9 | 1.00 | 918 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 404.1 | 135.1 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 877.5 | 703.9 | 1.00 | 2950 | 832.2 | 1.00 | 918 |
| single-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 413.8 | 191.6 | 0.47 | 2008 | 525.6 | 0.68 | 925 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 401.8 | 138.3 | 0.46 | - | - | - | 393 |
| single-49208-sliding-cp8r7 | tilelang@main | 880.4 | 691.7 | 1.00 | 2941 | 850.3 | 1.00 | 918 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 401.2 | 139.0 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 883.9 | 719.1 | 1.00 | 2960 | 885.3 | 1.00 | 918 |
| single-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 415.1 | 191.3 | 0.47 | 2013 | 598.1 | 0.68 | 925 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 400.8 | 140.4 | 0.45 | - | - | - | 393 |
| short-49208-csa-cp1 | tilelang@main | 12688 | 299.8 | 1.00 | 50333 | 736.6 | 1.00 | 6368 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 6166 | 60.7 | 0.49 | - | - | - | 3340 |
| short-49208-csa-cp1 | tilelang@cudnn_flashmla | 12806 | 240.6 | 1.00 | 50697 | 651.1 | 1.00 | 6368 |
| short-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 6195 | 203.2 | 0.48 | 29165 | 235.3 | 0.58 | 6513 |
| short-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 6205 | 412.1 | 0.48 | - | - | - | 3340 |
| short-49208-csa-cp8r0 | tilelang@main | 1344 | 692.7 | 1.00 | 5188 | 824.3 | 1.00 | 954 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 686.4 | 104.7 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 1344 | 697.3 | 1.00 | 5195 | 811.6 | 1.00 | 954 |
| short-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 694.1 | 166.0 | 0.52 | 3090 | 297.9 | 0.59 | 972 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 685.6 | 104.8 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r4 | tilelang@main | 1711 | 674.8 | 1.00 | 7251 | 834.8 | 1.00 | 954 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 852.5 | 107.1 | 0.50 | - | - | - | 418 |
| short-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 1717 | 680.1 | 1.00 | 7294 | 853.5 | 1.00 | 954 |
| short-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 863.6 | 160.8 | 0.50 | 4002 | 333.7 | 0.55 | 972 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 854.1 | 100.6 | 0.50 | - | - | - | 418 |
| short-49208-csa-cp8r7 | tilelang@main | 1484 | 676.8 | 1.00 | 5954 | 855.8 | 1.00 | 954 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 756.8 | 101.5 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 1479 | 700.9 | 1.00 | 5950 | 835.3 | 1.00 | 954 |
| short-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 765.0 | 154.0 | 0.52 | 3440 | 299.4 | 0.58 | 972 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 755.9 | 100.5 | 0.51 | - | - | - | 418 |
| short-49208-hca-cp1 | tilelang@main | 7876 | 628.9 | 1.00 | 25343 | 831.9 | 1.00 | 6382 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 4089 | 14.0 | 0.52 | - | - | - | 3197 |
| short-49208-hca-cp1 | tilelang@cudnn_flashmla | 7871 | 614.3 | 1.00 | 25362 | 822.4 | 1.00 | 6382 |
| short-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 4126 | 50.1 | 0.52 | 17239 | 170.8 | 0.68 | 6406 |
| short-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4103 | -7.3 | 0.52 | - | - | - | 3197 |
| short-49208-hca-cp8r0 | tilelang@main | 989.7 | 710.9 | 1.00 | 3199 | 869.2 | 1.00 | 925 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 521.4 | 125.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 993.0 | 762.1 | 1.00 | 3193 | 896.2 | 1.00 | 925 |
| short-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 530.6 | 191.1 | 0.53 | 2196 | 414.1 | 0.69 | 928 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 520.7 | 132.8 | 0.52 | - | - | - | 400 |
| short-49208-hca-cp8r4 | tilelang@main | 1004 | 742.6 | 1.00 | 3328 | 858.1 | 1.00 | 925 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 529.1 | 136.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 1012 | 738.9 | 1.00 | 3334 | 892.0 | 1.00 | 925 |
| short-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 538.1 | 197.3 | 0.53 | 2279 | 412.5 | 0.68 | 928 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 525.0 | 141.0 | 0.52 | - | - | - | 400 |
| short-49208-hca-cp8r7 | tilelang@main | 1012 | 720.8 | 1.00 | 3277 | 892.2 | 1.00 | 925 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 532.2 | 132.3 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 1015 | 734.8 | 1.00 | 3274 | 873.5 | 1.00 | 925 |
| short-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 547.5 | 189.2 | 0.54 | 2240 | 390.2 | 0.68 | 928 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 536.2 | 132.8 | 0.53 | - | - | - | 400 |
| short-49208-sliding-cp1 | tilelang@main | 6741 | 681.3 | 1.00 | 21763 | 869.5 | 1.00 | 6332 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 3084 | 39.0 | 0.46 | - | - | - | 3148 |
| short-49208-sliding-cp1 | tilelang@cudnn_flashmla | 6741 | 681.2 | 1.00 | 21806 | 811.3 | 1.00 | 6332 |
| short-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3120 | 103.6 | 0.46 | 14869 | 348.4 | 0.68 | 6380 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3081 | 41.6 | 0.46 | - | - | - | 3148 |
| short-49208-sliding-cp8r0 | tilelang@main | 864.5 | 700.0 | 1.00 | 2838 | 825.1 | 1.00 | 918 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 397.8 | 143.2 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 866.3 | 707.5 | 1.00 | 2836 | 852.5 | 1.00 | 918 |
| short-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 420.5 | 190.1 | 0.49 | 1965 | 538.6 | 0.69 | 924 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 407.9 | 139.0 | 0.47 | - | - | - | 394 |
| short-49208-sliding-cp8r4 | tilelang@main | 869.7 | 699.7 | 1.00 | 2893 | 846.4 | 1.00 | 918 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 402.9 | 142.9 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 869.5 | 696.1 | 1.00 | 2894 | 841.1 | 1.00 | 918 |
| short-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 415.8 | 195.6 | 0.48 | 1970 | 549.0 | 0.68 | 924 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 402.3 | 143.0 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r7 | tilelang@main | 872.3 | 683.9 | 1.00 | 2886 | 857.6 | 1.00 | 918 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 405.0 | 131.9 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 875.3 | 711.3 | 1.00 | 2883 | 850.1 | 1.00 | 918 |
| short-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 416.7 | 191.5 | 0.48 | 1976 | 554.3 | 0.69 | 924 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 405.0 | 139.9 | 0.46 | - | - | - | 394 |
| heavy-49208-csa-cp1 | tilelang@main | 14881 | 32.9 | 1.00 | 60982 | 725.7 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 7056 | 338.1 | 0.47 | - | - | - | 3340 |
| heavy-49208-csa-cp1 | tilelang@cudnn_flashmla | 14803 | 118.0 | 1.00 | 61453 | 588.4 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 7061 | 436.8 | 0.48 | 34164 | 473.1 | 0.56 | 6513 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 7141 | 520.5 | 0.48 | - | - | - | 3340 |
| heavy-49208-csa-cp8r0 | tilelang@main | 1239 | 685.7 | 1.00 | 4603 | 816.8 | 1.00 | 954 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 650.2 | 109.6 | 0.52 | - | - | - | 418 |
| heavy-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 1240 | 696.1 | 1.00 | 4595 | 819.1 | 1.00 | 954 |
| heavy-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 652.2 | 166.5 | 0.53 | 2837 | 294.0 | 0.62 | 972 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 649.9 | 102.1 | 0.52 | - | - | - | 418 |
| heavy-49208-csa-cp8r4 | tilelang@main | 1940 | 688.9 | 1.00 | 8493 | 875.9 | 1.00 | 954 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 953.3 | 104.7 | 0.49 | - | - | - | 418 |
| heavy-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 1949 | 686.7 | 1.00 | 8635 | 847.0 | 1.00 | 954 |
| heavy-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 964.0 | 165.2 | 0.49 | 4608 | 337.2 | 0.53 | 972 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 955.4 | 104.0 | 0.49 | - | - | - | 418 |
| heavy-49208-csa-cp8r7 | tilelang@main | 1792 | 685.2 | 1.00 | 7721 | 860.7 | 1.00 | 954 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 889.3 | 101.1 | 0.50 | - | - | - | 418 |
| heavy-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 1801 | 681.2 | 1.00 | 7777 | 871.5 | 1.00 | 954 |
| heavy-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 900.9 | 157.1 | 0.50 | 4215 | 331.7 | 0.54 | 972 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 893.5 | 92.2 | 0.50 | - | - | - | 418 |
| heavy-49208-hca-cp1 | tilelang@main | 8488 | 600.9 | 1.00 | 28842 | 762.6 | 1.00 | 6394 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4251 | 22.7 | 0.50 | - | - | - | 3244 |
| heavy-49208-hca-cp1 | tilelang@cudnn_flashmla | 8435 | 637.6 | 1.00 | 28865 | 810.1 | 1.00 | 6394 |
| heavy-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 4364 | 52.6 | 0.52 | 18802 | 203.3 | 0.65 | 6454 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4246 | 21.7 | 0.50 | - | - | - | 3244 |
| heavy-49208-hca-cp8r0 | tilelang@main | 956.8 | 745.4 | 1.00 | 3055 | 857.0 | 1.00 | 926 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 510.4 | 132.4 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 958.9 | 739.5 | 1.00 | 3044 | 853.5 | 1.00 | 926 |
| heavy-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 527.5 | 174.3 | 0.55 | 2157 | 378.0 | 0.71 | 934 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 508.9 | 126.9 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r4 | tilelang@main | 1044 | 709.2 | 1.00 | 3546 | 847.7 | 1.00 | 926 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 550.7 | 124.3 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 1038 | 731.4 | 1.00 | 3511 | 906.7 | 1.00 | 926 |
| heavy-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 571.0 | 173.0 | 0.55 | 2365 | 397.4 | 0.67 | 934 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 552.7 | 117.1 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r7 | tilelang@main | 1105 | 739.6 | 1.00 | 3792 | 880.6 | 1.00 | 926 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 591.4 | 120.7 | 0.54 | - | - | - | 406 |
| heavy-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 1098 | 745.0 | 1.00 | 3772 | 897.8 | 1.00 | 926 |
| heavy-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 605.7 | 172.6 | 0.55 | 2518 | 406.1 | 0.67 | 934 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 584.0 | 128.0 | 0.53 | - | - | - | 406 |
| heavy-49208-sliding-cp1 | tilelang@main | 6750 | 697.8 | 1.00 | 21733 | 849.9 | 1.00 | 6332 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 3069 | 65.7 | 0.45 | - | - | - | 3148 |
| heavy-49208-sliding-cp1 | tilelang@cudnn_flashmla | 6732 | 680.7 | 1.00 | 21780 | 807.2 | 1.00 | 6332 |
| heavy-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3117 | 108.0 | 0.46 | 14909 | 306.1 | 0.68 | 6380 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3074 | 40.7 | 0.46 | - | - | - | 3148 |
| heavy-49208-sliding-cp8r0 | tilelang@main | 861.4 | 710.5 | 1.00 | 2755 | 833.2 | 1.00 | 918 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 402.5 | 139.7 | 0.47 | - | - | - | 394 |
| heavy-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 860.6 | 728.6 | 1.00 | 2743 | 867.4 | 1.00 | 918 |
| heavy-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 413.7 | 199.3 | 0.48 | 1921 | 544.1 | 0.70 | 925 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 397.8 | 144.0 | 0.46 | - | - | - | 393 |
| heavy-49208-sliding-cp8r4 | tilelang@main | 877.6 | 687.8 | 1.00 | 2942 | 819.4 | 1.00 | 918 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 403.9 | 135.0 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 880.4 | 684.5 | 1.00 | 2944 | 839.9 | 1.00 | 918 |
| heavy-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 418.6 | 182.7 | 0.48 | 2004 | 515.4 | 0.68 | 925 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 401.7 | 138.6 | 0.46 | - | - | - | 393 |
| heavy-49208-sliding-cp8r7 | tilelang@main | 882.7 | 684.6 | 1.00 | 2924 | 841.8 | 1.00 | 918 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 402.0 | 141.2 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 879.0 | 691.8 | 1.00 | 2918 | 851.9 | 1.00 | 918 |
| heavy-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 415.3 | 185.8 | 0.47 | 1983 | 552.2 | 0.68 | 925 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 398.2 | 136.5 | 0.45 | - | - | - | 393 |
| tiny-49208-csa-cp1 | tilelang@main | 7840 | 553.1 | 1.00 | 23410 | 693.2 | 1.00 | 6368 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 4272 | 23.0 | 0.54 | - | - | - | 3340 |
| tiny-49208-csa-cp1 | tilelang@cudnn_flashmla | 7839 | 536.7 | 1.00 | 23449 | 699.5 | 1.00 | 6368 |
| tiny-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 4286 | 54.2 | 0.55 | 16105 | 170.6 | 0.69 | 6512 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4280 | -4.8 | 0.55 | - | - | - | 3340 |
| tiny-49208-csa-cp8r0 | tilelang@main | 998.7 | 705.4 | 1.00 | 3072 | 802.9 | 1.00 | 953 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 556.1 | 102.0 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 1000 | 686.1 | 1.00 | 3066 | 807.7 | 1.00 | 953 |
| tiny-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 561.3 | 160.4 | 0.56 | 2128 | 347.1 | 0.69 | 971 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 551.4 | 103.0 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r4 | tilelang@main | 1012 | 668.1 | 1.00 | 3097 | 809.2 | 1.00 | 953 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 554.1 | 102.4 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 1003 | 690.9 | 1.00 | 3081 | 826.1 | 1.00 | 953 |
| tiny-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 557.5 | 164.0 | 0.56 | 2128 | 373.7 | 0.69 | 971 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 549.3 | 107.0 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r7 | tilelang@main | 1007 | 682.4 | 1.00 | 3080 | 832.5 | 1.00 | 953 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 560.0 | 99.4 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 1005 | 703.1 | 1.00 | 3082 | 807.8 | 1.00 | 953 |
| tiny-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 563.4 | 167.1 | 0.56 | 2128 | 336.8 | 0.69 | 971 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 553.2 | 106.8 | 0.55 | - | - | - | 418 |
| tiny-49208-hca-cp1 | tilelang@main | 6058 | 685.3 | 1.00 | 15953 | 833.4 | 1.00 | 6332 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 3126 | 38.3 | 0.52 | - | - | - | 3148 |
| tiny-49208-hca-cp1 | tilelang@cudnn_flashmla | 6031 | 689.4 | 1.00 | 15893 | 891.8 | 1.00 | 6332 |
| tiny-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 3144 | 126.1 | 0.52 | 12490 | 306.8 | 0.79 | 6380 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3112 | 50.0 | 0.52 | - | - | - | 3148 |
| tiny-49208-hca-cp8r0 | tilelang@main | 774.9 | 704.2 | 1.00 | 2125 | 827.8 | 1.00 | 918 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 401.6 | 139.8 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 769.0 | 744.6 | 1.00 | 2119 | 854.7 | 1.00 | 918 |
| tiny-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 412.7 | 203.6 | 0.54 | 1667 | 520.7 | 0.79 | 925 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 399.1 | 149.8 | 0.52 | - | - | - | 393 |
| tiny-49208-hca-cp8r4 | tilelang@main | 783.0 | 701.8 | 1.00 | 2153 | 827.3 | 1.00 | 918 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 404.4 | 139.6 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 783.5 | 701.4 | 1.00 | 2154 | 862.1 | 1.00 | 918 |
| tiny-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 413.2 | 188.9 | 0.53 | 1672 | 554.5 | 0.78 | 925 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 403.1 | 134.7 | 0.51 | - | - | - | 393 |
| tiny-49208-hca-cp8r7 | tilelang@main | 780.8 | 692.8 | 1.00 | 2111 | 841.6 | 1.00 | 918 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 401.6 | 139.0 | 0.51 | - | - | - | 394 |
| tiny-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 782.9 | 709.8 | 1.00 | 2105 | 847.9 | 1.00 | 918 |
| tiny-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 416.2 | 191.5 | 0.53 | 1655 | 531.0 | 0.79 | 925 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 401.5 | 140.4 | 0.51 | - | - | - | 393 |
| tiny-49208-sliding-cp1 | tilelang@main | 6059 | 674.1 | 1.00 | 15952 | 849.2 | 1.00 | 6332 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 3124 | 48.6 | 0.52 | - | - | - | 3148 |
| tiny-49208-sliding-cp1 | tilelang@cudnn_flashmla | 6063 | 658.9 | 1.00 | 15916 | 839.7 | 1.00 | 6332 |
| tiny-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3142 | 138.3 | 0.52 | 12445 | 353.8 | 0.78 | 6380 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3110 | 58.4 | 0.51 | - | - | - | 3148 |
| tiny-49208-sliding-cp8r0 | tilelang@main | 773.6 | 689.3 | 1.00 | 2123 | 823.5 | 1.00 | 918 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 403.5 | 136.8 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 771.1 | 705.4 | 1.00 | 2114 | 849.8 | 1.00 | 918 |
| tiny-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 413.9 | 200.7 | 0.54 | 1668 | 543.2 | 0.79 | 925 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 399.6 | 148.0 | 0.52 | - | - | - | 393 |
| tiny-49208-sliding-cp8r4 | tilelang@main | 785.5 | 699.1 | 1.00 | 2156 | 819.1 | 1.00 | 918 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 402.9 | 139.6 | 0.51 | - | - | - | 394 |
| tiny-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 785.1 | 709.1 | 1.00 | 2154 | 845.9 | 1.00 | 918 |
| tiny-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 415.2 | 197.2 | 0.53 | 1676 | 534.4 | 0.78 | 925 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 405.0 | 141.4 | 0.52 | - | - | - | 393 |
| tiny-49208-sliding-cp8r7 | tilelang@main | 779.6 | 685.1 | 1.00 | 2107 | 843.6 | 1.00 | 918 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 406.0 | 136.3 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 779.4 | 704.2 | 1.00 | 2108 | 868.3 | 1.00 | 918 |
| tiny-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 413.9 | 197.3 | 0.53 | 1653 | 553.4 | 0.78 | 925 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 403.4 | 142.2 | 0.52 | - | - | - | 393 |
| single-65536-csa-cp1 | tilelang@main | 21804 | -109.6 | 1.00 | 94799 | 644.7 | 1.00 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 10352 | 1033 | 0.47 | - | - | - | 4448 |
| single-65536-csa-cp1 | tilelang@cudnn_flashmla | 21623 | 94.5 | 1.00 | 95299 | 234.2 | 1.00 | 8480 |
| single-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 10373 | 1069 | 0.48 | 52887 | 535.2 | 0.55 | 8672 |
| single-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 10756 | 713.8 | 0.50 | - | - | - | 4448 |
| single-65536-csa-cp8r0 | tilelang@main | 2543 | 722.8 | 1.00 | 11048 | 882.7 | 1.00 | 1270 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 1248 | 76.7 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 2616 | 677.3 | 1.00 | 11162 | 892.5 | 1.00 | 1270 |
| single-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 1256 | 146.3 | 0.48 | 5992 | 315.1 | 0.54 | 1294 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 1243 | 85.7 | 0.48 | - | - | - | 556 |
| single-65536-csa-cp8r4 | tilelang@main | 2711 | 717.6 | 1.00 | 12259 | 886.8 | 1.00 | 1270 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1324 | 86.3 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 2719 | 722.6 | 1.00 | 12582 | 713.0 | 1.00 | 1270 |
| single-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1337 | 143.3 | 0.49 | 6501 | 385.0 | 0.52 | 1294 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1326 | 81.6 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r7 | tilelang@main | 2709 | 675.0 | 1.00 | 13123 | 872.4 | 1.00 | 1270 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1331 | 78.8 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 2817 | 649.2 | 1.00 | 13637 | 490.2 | 1.00 | 1270 |
| single-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1347 | 166.9 | 0.48 | 6833 | 382.5 | 0.50 | 1294 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1336 | 82.4 | 0.47 | - | - | - | 556 |
| single-65536-hca-cp1 | tilelang@main | 16321 | 181.6 | 1.00 | 63704 | 622.9 | 1.00 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 7958 | 238.2 | 0.49 | - | - | - | 4448 |
| single-65536-hca-cp1 | tilelang@cudnn_flashmla | 16274 | 291.2 | 1.00 | 64118 | 451.8 | 1.00 | 8434 |
| single-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 8002 | 347.5 | 0.49 | 37306 | 129.8 | 0.58 | 8626 |
| single-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 8078 | 346.3 | 0.50 | - | - | - | 4448 |
| single-65536-hca-cp8r0 | tilelang@main | 1365 | 708.7 | 1.00 | 4606 | 828.1 | 1.00 | 1224 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 751.0 | 84.2 | 0.55 | - | - | - | 556 |
| single-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 1363 | 681.8 | 1.00 | 4622 | 797.1 | 1.00 | 1224 |
| single-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 755.7 | 145.8 | 0.55 | 3145 | 266.3 | 0.68 | 1248 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 756.0 | 85.8 | 0.55 | - | - | - | 556 |
| single-65536-hca-cp8r4 | tilelang@main | 2123 | 673.9 | 1.00 | 8699 | 883.4 | 1.00 | 1224 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1114 | 70.8 | 0.52 | - | - | - | 556 |
| single-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 2122 | 674.3 | 1.00 | 8754 | 911.5 | 1.00 | 1224 |
| single-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1121 | 137.9 | 0.53 | 4936 | 340.3 | 0.56 | 1248 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1106 | 82.5 | 0.52 | - | - | - | 556 |
| single-65536-hca-cp8r7 | tilelang@main | 2715 | 674.6 | 1.00 | 11760 | 929.3 | 1.00 | 1224 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 1305 | 79.3 | 0.48 | - | - | - | 556 |
| single-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 2757 | 660.4 | 1.00 | 11925 | 854.1 | 1.00 | 1224 |
| single-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1323 | 133.7 | 0.48 | 6247 | 362.2 | 0.52 | 1248 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1312 | 69.2 | 0.48 | - | - | - | 556 |
| single-65536-sliding-cp1 | tilelang@main | 8998 | 652.7 | 1.00 | 29308 | 838.2 | 1.00 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4089 | 32.9 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp1 | tilelang@cudnn_flashmla | 9005 | 654.7 | 1.00 | 29322 | 803.5 | 1.00 | 8432 |
| single-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4133 | 77.2 | 0.46 | 20017 | 263.3 | 0.68 | 8496 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4068 | 29.0 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp8r0 | tilelang@main | 1148 | 723.0 | 1.00 | 3834 | 823.8 | 1.00 | 1222 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 522.4 | 139.0 | 0.45 | - | - | - | 524 |
| single-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1155 | 702.2 | 1.00 | 3830 | 860.2 | 1.00 | 1222 |
| single-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 536.9 | 200.3 | 0.46 | 2628 | 418.9 | 0.69 | 1230 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 521.5 | 143.4 | 0.45 | - | - | - | 524 |
| single-65536-sliding-cp8r4 | tilelang@main | 1153 | 690.8 | 1.00 | 3867 | 826.9 | 1.00 | 1222 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 533.2 | 127.5 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1151 | 713.7 | 1.00 | 3879 | 836.8 | 1.00 | 1222 |
| single-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 541.0 | 189.1 | 0.47 | 2628 | 400.0 | 0.68 | 1230 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 521.4 | 138.6 | 0.45 | - | - | - | 524 |
| single-65536-sliding-cp8r7 | tilelang@main | 1154 | 698.5 | 1.00 | 3861 | 855.4 | 1.00 | 1222 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 529.4 | 133.0 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1153 | 724.6 | 1.00 | 3860 | 883.8 | 1.00 | 1222 |
| single-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 539.8 | 192.5 | 0.47 | 2632 | 422.3 | 0.68 | 1230 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 524.8 | 136.1 | 0.46 | - | - | - | 524 |
| short-65536-csa-cp1 | tilelang@main | 16000 | 242.3 | 1.00 | 62509 | 660.3 | 1.00 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 7830 | 315.8 | 0.49 | - | - | - | 4448 |
| short-65536-csa-cp1 | tilelang@cudnn_flashmla | 15959 | 296.7 | 1.00 | 62988 | 364.8 | 1.00 | 8480 |
| short-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 7826 | 399.9 | 0.49 | 36267 | 547.4 | 0.58 | 8672 |
| short-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 8046 | 226.5 | 0.50 | - | - | - | 4448 |
| short-65536-csa-cp8r0 | tilelang@main | 1610 | 676.7 | 1.00 | 5968 | 811.5 | 1.00 | 1270 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 825.1 | 82.7 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 1612 | 689.6 | 1.00 | 5956 | 834.4 | 1.00 | 1270 |
| short-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 832.8 | 152.6 | 0.52 | 3678 | 282.0 | 0.62 | 1294 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 833.0 | 85.1 | 0.52 | - | - | - | 556 |
| short-65536-csa-cp8r4 | tilelang@main | 2195 | 666.7 | 1.00 | 9179 | 896.3 | 1.00 | 1270 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1090 | 79.7 | 0.50 | - | - | - | 556 |
| short-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 2209 | 683.2 | 1.00 | 9338 | 838.0 | 1.00 | 1270 |
| short-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1101 | 146.0 | 0.50 | 5140 | 296.8 | 0.55 | 1294 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1096 | 79.8 | 0.50 | - | - | - | 556 |
| short-65536-csa-cp8r7 | tilelang@main | 1997 | 667.0 | 1.00 | 8087 | 855.0 | 1.00 | 1270 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1012 | 76.5 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 1993 | 680.9 | 1.00 | 8072 | 868.8 | 1.00 | 1270 |
| short-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1021 | 133.0 | 0.51 | 4630 | 288.2 | 0.57 | 1294 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1007 | 82.8 | 0.51 | - | - | - | 556 |
| short-65536-hca-cp1 | tilelang@main | 10309 | 612.7 | 1.00 | 32762 | 770.6 | 1.00 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5376 | 1.8 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp1 | tilelang@cudnn_flashmla | 10327 | 623.4 | 1.00 | 32754 | 825.7 | 1.00 | 8482 |
| short-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 5435 | 36.7 | 0.53 | 22490 | 203.4 | 0.69 | 8530 |
| short-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5392 | -19.7 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp8r0 | tilelang@main | 1296 | 723.7 | 1.00 | 4168 | 858.9 | 1.00 | 1229 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 687.6 | 123.7 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 1290 | 751.8 | 1.00 | 4169 | 899.9 | 1.00 | 1229 |
| short-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 691.1 | 181.4 | 0.54 | 2866 | 330.4 | 0.69 | 1236 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 678.3 | 131.0 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r4 | tilelang@main | 1337 | 717.1 | 1.00 | 4340 | 857.5 | 1.00 | 1229 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 704.0 | 128.4 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 1333 | 738.7 | 1.00 | 4345 | 900.5 | 1.00 | 1229 |
| short-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 713.2 | 183.5 | 0.53 | 2986 | 333.4 | 0.69 | 1236 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 701.1 | 127.5 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r7 | tilelang@main | 1320 | 720.0 | 1.00 | 4272 | 894.3 | 1.00 | 1229 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 702.8 | 117.0 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 1319 | 750.9 | 1.00 | 4275 | 892.8 | 1.00 | 1229 |
| short-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 706.5 | 188.0 | 0.54 | 2950 | 314.8 | 0.69 | 1236 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 694.3 | 137.1 | 0.53 | - | - | - | 532 |
| short-65536-sliding-cp1 | tilelang@main | 8974 | 637.9 | 1.00 | 28844 | 797.0 | 1.00 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4081 | 23.9 | 0.45 | - | - | - | 4192 |
| short-65536-sliding-cp1 | tilelang@cudnn_flashmla | 8985 | 671.4 | 1.00 | 28861 | 833.9 | 1.00 | 8432 |
| short-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4150 | 78.5 | 0.46 | 19750 | 290.2 | 0.68 | 8496 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4093 | 12.6 | 0.46 | - | - | - | 4192 |
| short-65536-sliding-cp8r0 | tilelang@main | 1138 | 725.1 | 1.00 | 3714 | 842.5 | 1.00 | 1222 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 524.0 | 141.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1138 | 711.8 | 1.00 | 3719 | 875.1 | 1.00 | 1222 |
| short-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 543.7 | 186.3 | 0.48 | 2580 | 418.3 | 0.69 | 1230 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 529.3 | 138.7 | 0.47 | - | - | - | 524 |
| short-65536-sliding-cp8r4 | tilelang@main | 1149 | 688.8 | 1.00 | 3813 | 810.9 | 1.00 | 1222 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 526.8 | 133.6 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1145 | 706.5 | 1.00 | 3808 | 834.7 | 1.00 | 1222 |
| short-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 542.1 | 188.0 | 0.47 | 2613 | 388.3 | 0.69 | 1230 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 523.1 | 136.7 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r7 | tilelang@main | 1153 | 688.0 | 1.00 | 3777 | 855.3 | 1.00 | 1222 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 526.2 | 135.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1151 | 693.4 | 1.00 | 3777 | 870.0 | 1.00 | 1222 |
| short-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 542.6 | 188.8 | 0.47 | 2589 | 425.4 | 0.69 | 1230 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 521.0 | 137.1 | 0.45 | - | - | - | 524 |
| heavy-65536-csa-cp1 | tilelang@main | 17414 | -41.5 | 1.00 | 69035 | 660.4 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8439 | 381.8 | 0.48 | - | - | - | 4448 |
| heavy-65536-csa-cp1 | tilelang@cudnn_flashmla | 17414 | 51.1 | 1.00 | 69494 | 457.0 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 8446 | 500.0 | 0.49 | 39712 | 268.1 | 0.57 | 8672 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 8584 | 456.6 | 0.49 | - | - | - | 4448 |
| heavy-65536-csa-cp8r0 | tilelang@main | 1561 | 676.9 | 1.00 | 5628 | 791.6 | 1.00 | 1270 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 818.9 | 82.0 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 1562 | 682.3 | 1.00 | 5617 | 812.2 | 1.00 | 1270 |
| heavy-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 818.5 | 151.3 | 0.52 | 3524 | 281.2 | 0.63 | 1294 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 814.6 | 86.7 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r4 | tilelang@main | 2719 | 683.2 | 1.00 | 12075 | 898.7 | 1.00 | 1270 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1326 | 75.9 | 0.49 | - | - | - | 556 |
| heavy-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 2713 | 751.4 | 1.00 | 12320 | 833.5 | 1.00 | 1270 |
| heavy-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1335 | 179.6 | 0.49 | 6444 | 391.9 | 0.52 | 1294 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1327 | 78.9 | 0.49 | - | - | - | 556 |
| heavy-65536-csa-cp8r7 | tilelang@main | 1888 | 661.1 | 1.00 | 7479 | 820.1 | 1.00 | 1270 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 972.7 | 74.8 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 1889 | 697.5 | 1.00 | 7469 | 832.5 | 1.00 | 1270 |
| heavy-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 979.8 | 148.6 | 0.52 | 4376 | 286.0 | 0.59 | 1294 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 969.2 | 77.1 | 0.51 | - | - | - | 556 |
| heavy-65536-hca-cp1 | tilelang@main | 11197 | 516.8 | 1.00 | 37594 | 678.6 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5693 | -10.7 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp1 | tilelang@cudnn_flashmla | 11198 | 554.1 | 1.00 | 37870 | 465.5 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 5824 | 43.8 | 0.52 | 24761 | 160.0 | 0.65 | 8594 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5683 | 17.7 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp8r0 | tilelang@main | 1261 | 730.2 | 1.00 | 3956 | 864.8 | 1.00 | 1235 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 660.8 | 116.2 | 0.52 | - | - | - | 540 |
| heavy-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 1260 | 725.2 | 1.00 | 3952 | 883.9 | 1.00 | 1235 |
| heavy-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 684.7 | 161.2 | 0.54 | 2798 | 303.7 | 0.71 | 1244 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 660.0 | 119.0 | 0.52 | - | - | - | 540 |
| heavy-65536-hca-cp8r4 | tilelang@main | 1655 | 722.9 | 1.00 | 6188 | 861.1 | 1.00 | 1235 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 811.0 | 117.0 | 0.49 | - | - | - | 540 |
| heavy-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 1652 | 731.2 | 1.00 | 6200 | 883.8 | 1.00 | 1235 |
| heavy-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 837.1 | 151.5 | 0.51 | 3752 | 319.7 | 0.61 | 1244 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 813.8 | 114.1 | 0.49 | - | - | - | 540 |
| heavy-65536-hca-cp8r7 | tilelang@main | 1308 | 698.6 | 1.00 | 4132 | 878.0 | 1.00 | 1235 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 693.7 | 105.4 | 0.53 | - | - | - | 540 |
| heavy-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 1302 | 718.7 | 1.00 | 4126 | 877.1 | 1.00 | 1235 |
| heavy-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 711.1 | 154.0 | 0.55 | 2909 | 305.9 | 0.71 | 1244 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 689.1 | 115.3 | 0.53 | - | - | - | 540 |
| heavy-65536-sliding-cp1 | tilelang@main | 8945 | 671.5 | 1.00 | 28442 | 840.0 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4091 | 20.5 | 0.46 | - | - | - | 4192 |
| heavy-65536-sliding-cp1 | tilelang@cudnn_flashmla | 8941 | 657.6 | 1.00 | 28459 | 807.2 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4145 | 75.0 | 0.46 | 19605 | 249.4 | 0.69 | 8496 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4098 | 8.8 | 0.46 | - | - | - | 4192 |
| heavy-65536-sliding-cp8r0 | tilelang@main | 1132 | 721.2 | 1.00 | 3582 | 823.1 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 528.4 | 141.3 | 0.47 | - | - | - | 524 |
| heavy-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1131 | 706.0 | 1.00 | 3589 | 852.3 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 544.4 | 187.5 | 0.48 | 2519 | 395.9 | 0.70 | 1230 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 525.4 | 138.5 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r4 | tilelang@main | 1152 | 688.6 | 1.00 | 3866 | 811.3 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 524.6 | 131.3 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1155 | 695.5 | 1.00 | 3871 | 839.9 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 535.2 | 195.1 | 0.46 | 2636 | 387.7 | 0.68 | 1230 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 521.2 | 138.2 | 0.45 | - | - | - | 524 |
| heavy-65536-sliding-cp8r7 | tilelang@main | 1148 | 675.9 | 1.00 | 3690 | 833.4 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 525.3 | 140.4 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1146 | 705.9 | 1.00 | 3685 | 868.1 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 542.4 | 182.9 | 0.47 | 2547 | 413.5 | 0.69 | 1230 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 521.6 | 138.2 | 0.46 | - | - | - | 524 |
| tiny-65536-csa-cp1 | tilelang@main | 10414 | 470.2 | 1.00 | 31178 | 622.0 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5690 | -30.5 | 0.55 | - | - | - | 4448 |
| tiny-65536-csa-cp1 | tilelang@cudnn_flashmla | 10432 | 466.0 | 1.00 | 31181 | 635.2 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 5690 | 47.2 | 0.55 | 21411 | 132.8 | 0.69 | 8671 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5660 | -2.0 | 0.54 | - | - | - | 4448 |
| tiny-65536-csa-cp8r0 | tilelang@main | 1324 | 680.4 | 1.00 | 4061 | 830.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 727.4 | 81.8 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 1325 | 678.7 | 1.00 | 4058 | 805.4 | 1.00 | 1269 |
| tiny-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 732.4 | 142.6 | 0.55 | 2805 | 278.0 | 0.69 | 1293 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 727.0 | 86.5 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r4 | tilelang@main | 1326 | 681.4 | 1.00 | 4066 | 801.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 731.1 | 92.1 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 1326 | 678.3 | 1.00 | 4057 | 833.5 | 1.00 | 1269 |
| tiny-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 738.0 | 149.5 | 0.56 | 2810 | 289.3 | 0.69 | 1293 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 725.3 | 92.5 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r7 | tilelang@main | 1330 | 734.8 | 1.00 | 4068 | 810.1 | 1.00 | 1269 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 728.8 | 102.6 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 1331 | 667.6 | 1.00 | 4064 | 838.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 737.9 | 136.1 | 0.55 | 2807 | 291.4 | 0.69 | 1293 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 727.3 | 80.7 | 0.55 | - | - | - | 556 |
| tiny-65536-hca-cp1 | tilelang@main | 8038 | 651.6 | 1.00 | 21189 | 833.7 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4147 | 25.6 | 0.52 | - | - | - | 4192 |
| tiny-65536-hca-cp1 | tilelang@cudnn_flashmla | 8028 | 633.5 | 1.00 | 21206 | 840.6 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 4188 | 74.8 | 0.52 | 16563 | 315.5 | 0.78 | 8496 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4140 | 11.2 | 0.52 | - | - | - | 4192 |
| tiny-65536-hca-cp8r0 | tilelang@main | 1023 | 682.2 | 1.00 | 2791 | 837.3 | 1.00 | 1222 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 527.5 | 130.4 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 1023 | 702.3 | 1.00 | 2790 | 853.1 | 1.00 | 1222 |
| tiny-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 546.2 | 180.7 | 0.53 | 2180 | 404.1 | 0.78 | 1230 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 523.6 | 138.7 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r4 | tilelang@main | 1031 | 700.4 | 1.00 | 2795 | 813.5 | 1.00 | 1222 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 528.8 | 133.5 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 1030 | 701.3 | 1.00 | 2794 | 864.5 | 1.00 | 1222 |
| tiny-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 542.1 | 184.5 | 0.53 | 2179 | 440.8 | 0.78 | 1230 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 523.3 | 136.6 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r7 | tilelang@main | 1025 | 690.8 | 1.00 | 2797 | 833.6 | 1.00 | 1222 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 530.4 | 135.8 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 1016 | 745.5 | 1.00 | 2791 | 856.1 | 1.00 | 1222 |
| tiny-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 538.3 | 197.1 | 0.53 | 2183 | 402.9 | 0.78 | 1230 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 526.1 | 132.8 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp1 | tilelang@main | 8031 | 685.8 | 1.00 | 21186 | 850.2 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4116 | 62.0 | 0.51 | - | - | - | 4192 |
| tiny-65536-sliding-cp1 | tilelang@cudnn_flashmla | 8037 | 637.4 | 1.00 | 21201 | 789.7 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 4192 | 75.1 | 0.52 | 16580 | 276.3 | 0.78 | 8496 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4132 | 24.4 | 0.51 | - | - | - | 4192 |
| tiny-65536-sliding-cp8r0 | tilelang@main | 1021 | 723.6 | 1.00 | 2791 | 839.4 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 525.1 | 143.2 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 1023 | 726.1 | 1.00 | 2790 | 852.7 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 538.3 | 204.0 | 0.53 | 2184 | 401.4 | 0.78 | 1230 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 519.5 | 151.6 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r4 | tilelang@main | 1028 | 687.6 | 1.00 | 2791 | 821.8 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 522.8 | 139.4 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 1030 | 711.3 | 1.00 | 2793 | 847.9 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 543.1 | 188.6 | 0.53 | 2187 | 376.1 | 0.78 | 1230 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 523.6 | 138.1 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r7 | tilelang@main | 1017 | 687.1 | 1.00 | 2786 | 842.6 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 529.2 | 135.9 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 1018 | 694.8 | 1.00 | 2790 | 854.7 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 539.5 | 186.2 | 0.53 | 2185 | 394.0 | 0.78 | 1230 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 517.8 | 138.3 | 0.51 | - | - | - | 524 |

Useful FLOPs count valid slots only (fwd 4HD, bwd 10HD per slot); `exec/useful` counts the slots each
backend's tiles touch, or every padded slot for an arm without tile information. TFLOP/s divide
useful FLOPs by op-boundary time (higher is better);
`% peak` is f+b against 989.5 dense BF16 TFLOP/s (https://www.nvidia.com/en-us/data-center/h200/ (H200 SXM BF16 1,979 TFLOPS with sparsity, halved)).

| item | backend | f+b GFLOP | exec/useful fwd | exec/useful bwd | fwd TFLOP/s | f+b TFLOP/s | % peak |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang@main | 356.8 | 1.09 | 1.05 | 85.2 | 112.0 | 11.3 |
| single-2048-csa-cp1 | flashmla_fwd_ref@main | 356.8 | 1.69 | 1.69 | 265.5 | - | - |
| single-2048-csa-cp1 | tilelang@cudnn_flashmla | 356.8 | 1.09 | 1.05 | 84.3 | 109.2 | 11.0 |
| single-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 356.8 | 1.09 | 1.09 | 232.7 | 185.4 | 18.7 |
| single-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 356.8 | 1.69 | 1.69 | 269.8 | - | - |
| single-2048-csa-cp8r0 | tilelang@main | 15.0 | 1.49 | 1.36 | 5.9 | 7.5 | 0.8 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@main | 15.0 | 5.00 | 5.00 | 23.5 | - | - |
| single-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 15.0 | 1.49 | 1.36 | 5.6 | 7.2 | 0.7 |
| single-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 15.0 | 1.49 | 1.49 | 14.1 | 10.4 | 1.0 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 15.0 | 5.00 | 5.00 | 22.9 | - | - |
| single-2048-csa-cp8r4 | tilelang@main | 48.8 | 1.08 | 1.04 | 18.3 | 24.1 | 2.4 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@main | 48.8 | 1.54 | 1.54 | 70.6 | - | - |
| single-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 48.8 | 1.08 | 1.04 | 18.0 | 23.5 | 2.4 |
| single-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 48.8 | 1.08 | 1.08 | 45.7 | 33.7 | 3.4 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 48.8 | 1.54 | 1.54 | 70.8 | - | - |
| single-2048-csa-cp8r7 | tilelang@main | 71.4 | 1.05 | 1.03 | 26.3 | 35.3 | 3.6 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@main | 71.4 | 1.05 | 1.05 | 100.5 | - | - |
| single-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 71.4 | 1.05 | 1.03 | 25.5 | 34.1 | 3.4 |
| single-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 71.4 | 1.05 | 1.05 | 66.6 | 49.4 | 5.0 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 71.4 | 1.05 | 1.05 | 95.9 | - | - |
| single-2048-hca-cp1 | tilelang@main | 123.6 | 1.41 | 1.18 | 32.5 | 49.4 | 5.0 |
| single-2048-hca-cp1 | flashmla_fwd_ref@main | 123.6 | 1.95 | 1.95 | 103.5 | - | - |
| single-2048-hca-cp1 | tilelang@cudnn_flashmla | 123.6 | 1.41 | 1.18 | 32.0 | 48.8 | 4.9 |
| single-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 123.6 | 1.41 | 1.41 | 86.8 | 78.6 | 7.9 |
| single-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 123.6 | 1.95 | 1.95 | 101.8 | - | - |
| single-2048-hca-cp8r0 | tilelang@main | 11.4 | 1.49 | 1.24 | 4.3 | 5.3 | 0.5 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@main | 11.4 | 2.65 | 2.65 | 17.2 | - | - |
| single-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 11.4 | 1.49 | 1.24 | 3.9 | 5.1 | 0.5 |
| single-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 11.4 | 1.49 | 1.49 | 10.1 | 7.7 | 0.8 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 11.4 | 2.65 | 2.65 | 17.0 | - | - |
| single-2048-hca-cp8r4 | tilelang@main | 16.0 | 1.41 | 1.17 | 5.8 | 7.4 | 0.8 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@main | 16.0 | 1.88 | 1.88 | 23.6 | - | - |
| single-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 16.0 | 1.41 | 1.17 | 5.8 | 7.2 | 0.7 |
| single-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 16.0 | 1.41 | 1.41 | 14.7 | 10.9 | 1.1 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 16.0 | 1.88 | 1.88 | 23.8 | - | - |
| single-2048-hca-cp8r7 | tilelang@main | 16.7 | 1.35 | 1.12 | 5.9 | 7.7 | 0.8 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@main | 16.7 | 1.80 | 1.80 | 24.7 | - | - |
| single-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 16.7 | 1.35 | 1.12 | 6.1 | 7.6 | 0.8 |
| single-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 16.7 | 1.35 | 1.35 | 15.3 | 11.4 | 1.2 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 16.7 | 1.80 | 1.80 | 24.9 | - | - |
| single-2048-sliding-cp1 | tilelang@main | 116.5 | 1.02 | 1.01 | 32.8 | 50.9 | 5.1 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 111.6 | - | - |
| single-2048-sliding-cp1 | tilelang@cudnn_flashmla | 116.5 | 1.02 | 1.01 | 32.9 | 49.1 | 5.0 |
| single-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 116.5 | 1.02 | 1.02 | 93.0 | 76.7 | 7.8 |
| single-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 116.5 | 1.03 | 1.03 | 113.4 | - | - |
| single-2048-sliding-cp8r0 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.4 | 5.5 | 0.6 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 18.0 | - | - |
| single-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 11.3 | 1.16 | 1.08 | 4.3 | 5.4 | 0.5 |
| single-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 11.3 | 1.16 | 1.16 | 10.5 | 7.7 | 0.8 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 11.3 | 1.33 | 1.33 | 17.9 | - | - |
| single-2048-sliding-cp8r4 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.8 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 24.0 | - | - |
| single-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 5.6 | 7.2 | 0.7 |
| single-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 14.0 | 10.3 | 1.0 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 23.9 | - | - |
| single-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.7 | - | - |
| single-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 5.7 | 7.1 | 0.7 |
| single-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 14.1 | 10.3 | 1.0 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 23.8 | - | - |
| short-2048-csa-cp1 | tilelang@main | 233.1 | 1.16 | 1.10 | 58.8 | 84.6 | 8.5 |
| short-2048-csa-cp1 | flashmla_fwd_ref@main | 233.1 | 2.58 | 2.58 | 181.0 | - | - |
| short-2048-csa-cp1 | tilelang@cudnn_flashmla | 233.1 | 1.16 | 1.10 | 58.6 | 81.9 | 8.3 |
| short-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 233.1 | 1.16 | 1.16 | 156.5 | 135.0 | 13.6 |
| short-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 233.1 | 2.58 | 2.58 | 181.1 | - | - |
| short-2048-csa-cp8r0 | tilelang@main | 15.0 | 1.49 | 1.36 | 5.7 | 7.4 | 0.8 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@main | 15.0 | 5.00 | 5.00 | 22.9 | - | - |
| short-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 15.0 | 1.49 | 1.36 | 5.8 | 7.1 | 0.7 |
| short-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 15.0 | 1.49 | 1.49 | 14.2 | 10.3 | 1.0 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 15.0 | 5.00 | 5.00 | 23.1 | - | - |
| short-2048-csa-cp8r4 | tilelang@main | 19.6 | 1.43 | 1.30 | 7.6 | 9.7 | 1.0 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@main | 19.6 | 3.83 | 3.83 | 29.9 | - | - |
| short-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 19.6 | 1.43 | 1.30 | 7.3 | 9.3 | 0.9 |
| short-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 19.6 | 1.43 | 1.43 | 18.4 | 13.5 | 1.4 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 19.6 | 3.83 | 3.83 | 28.8 | - | - |
| short-2048-csa-cp8r7 | tilelang@main | 39.9 | 1.09 | 1.05 | 14.9 | 19.6 | 2.0 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@main | 39.9 | 1.89 | 1.89 | 59.5 | - | - |
| short-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 39.9 | 1.09 | 1.05 | 14.4 | 18.9 | 1.9 |
| short-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 39.9 | 1.09 | 1.09 | 35.3 | 27.0 | 2.7 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 39.9 | 1.89 | 1.89 | 57.3 | - | - |
| short-2048-hca-cp1 | tilelang@main | 116.1 | 1.46 | 1.21 | 30.7 | 46.7 | 4.7 |
| short-2048-hca-cp1 | flashmla_fwd_ref@main | 116.1 | 2.07 | 2.07 | 97.5 | - | - |
| short-2048-hca-cp1 | tilelang@cudnn_flashmla | 116.1 | 1.46 | 1.21 | 30.5 | 46.0 | 4.6 |
| short-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 116.1 | 1.46 | 1.46 | 83.4 | 74.3 | 7.5 |
| short-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 116.1 | 2.07 | 2.07 | 98.6 | - | - |
| short-2048-hca-cp8r0 | tilelang@main | 11.4 | 1.49 | 1.24 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@main | 11.4 | 2.65 | 2.65 | 17.0 | - | - |
| short-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 11.4 | 1.49 | 1.24 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 11.4 | 1.49 | 1.49 | 10.6 | 7.8 | 0.8 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 11.4 | 2.65 | 2.65 | 17.4 | - | - |
| short-2048-hca-cp8r4 | tilelang@main | 11.5 | 1.47 | 1.22 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@main | 11.5 | 2.61 | 2.61 | 17.0 | - | - |
| short-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 11.5 | 1.47 | 1.22 | 4.2 | 5.2 | 0.5 |
| short-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 11.5 | 1.47 | 1.47 | 10.8 | 7.9 | 0.8 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 11.5 | 2.61 | 2.61 | 17.6 | - | - |
| short-2048-hca-cp8r7 | tilelang@main | 15.8 | 1.43 | 1.19 | 5.7 | 7.2 | 0.7 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@main | 15.8 | 1.91 | 1.91 | 23.8 | - | - |
| short-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 15.8 | 1.43 | 1.19 | 5.7 | 7.3 | 0.7 |
| short-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 15.8 | 1.43 | 1.43 | 14.4 | 10.7 | 1.1 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 15.8 | 1.91 | 1.91 | 23.2 | - | - |
| short-2048-sliding-cp1 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.1 | 48.4 | 4.9 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 109.9 | - | - |
| short-2048-sliding-cp1 | tilelang@cudnn_flashmla | 112.8 | 1.03 | 1.02 | 32.4 | 46.0 | 4.6 |
| short-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 112.8 | 1.03 | 1.03 | 91.7 | 69.2 | 7.0 |
| short-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 112.8 | 1.07 | 1.07 | 111.2 | - | - |
| short-2048-sliding-cp8r0 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.3 | 5.5 | 0.6 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 17.8 | - | - |
| short-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 11.3 | 1.16 | 1.08 | 4.3 | 5.4 | 0.5 |
| short-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 11.3 | 1.16 | 1.16 | 10.8 | 7.7 | 0.8 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 11.3 | 1.33 | 1.33 | 18.4 | - | - |
| short-2048-sliding-cp8r4 | tilelang@main | 11.3 | 1.16 | 1.08 | 4.3 | 5.6 | 0.6 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 11.3 | 1.33 | 1.33 | 17.0 | - | - |
| short-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 11.3 | 1.16 | 1.08 | 4.3 | 5.4 | 0.5 |
| short-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 11.3 | 1.16 | 1.16 | 10.7 | 7.8 | 0.8 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 11.3 | 1.33 | 1.33 | 17.9 | - | - |
| short-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.2 | - | - |
| short-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 5.7 | 7.1 | 0.7 |
| short-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 14.0 | 10.3 | 1.0 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 24.0 | - | - |
| heavy-2048-csa-cp1 | tilelang@main | 272.5 | 1.16 | 1.09 | 66.5 | 92.2 | 9.3 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@main | 272.5 | 2.21 | 2.21 | 199.8 | - | - |
| heavy-2048-csa-cp1 | tilelang@cudnn_flashmla | 272.5 | 1.16 | 1.09 | 66.9 | 91.6 | 9.3 |
| heavy-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 272.5 | 1.16 | 1.16 | 176.8 | 153.3 | 15.5 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 272.5 | 2.21 | 2.21 | 201.4 | - | - |
| heavy-2048-csa-cp8r0 | tilelang@main | 10.3 | 2.16 | 1.86 | 3.9 | 5.0 | 0.5 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@main | 10.3 | 7.32 | 7.32 | 15.5 | - | - |
| heavy-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 10.3 | 2.16 | 1.86 | 3.9 | 4.9 | 0.5 |
| heavy-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 10.3 | 2.16 | 2.16 | 9.6 | 7.0 | 0.7 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 10.3 | 7.32 | 7.32 | 15.9 | - | - |
| heavy-2048-csa-cp8r4 | tilelang@main | 37.7 | 1.10 | 1.05 | 13.8 | 18.2 | 1.8 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@main | 37.7 | 2.00 | 2.00 | 53.6 | - | - |
| heavy-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 37.7 | 1.10 | 1.05 | 13.9 | 18.0 | 1.8 |
| heavy-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 37.7 | 1.10 | 1.10 | 35.3 | 26.0 | 2.6 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 37.7 | 2.00 | 2.00 | 56.9 | - | - |
| heavy-2048-csa-cp8r7 | tilelang@main | 60.2 | 1.06 | 1.03 | 21.7 | 29.6 | 3.0 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@main | 60.2 | 1.25 | 1.25 | 82.1 | - | - |
| heavy-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 60.2 | 1.06 | 1.03 | 22.1 | 28.6 | 2.9 |
| heavy-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 60.2 | 1.06 | 1.06 | 57.5 | 41.0 | 4.1 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 60.2 | 1.25 | 1.25 | 87.7 | - | - |
| heavy-2048-hca-cp1 | tilelang@main | 113.7 | 1.44 | 1.20 | 30.3 | 46.2 | 4.7 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@main | 113.7 | 2.11 | 2.11 | 97.0 | - | - |
| heavy-2048-hca-cp1 | tilelang@cudnn_flashmla | 113.7 | 1.44 | 1.20 | 29.1 | 45.0 | 4.5 |
| heavy-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 113.7 | 1.44 | 1.44 | 81.9 | 73.2 | 7.4 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 113.7 | 2.11 | 2.11 | 96.4 | - | - |
| heavy-2048-hca-cp8r0 | tilelang@main | 8.2 | 1.57 | 1.28 | 3.0 | 3.8 | 0.4 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@main | 8.2 | 3.68 | 3.68 | 12.3 | - | - |
| heavy-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 8.2 | 1.57 | 1.28 | 3.0 | 3.6 | 0.4 |
| heavy-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 8.2 | 1.57 | 1.57 | 7.6 | 5.5 | 0.6 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 8.2 | 3.68 | 3.68 | 12.6 | - | - |
| heavy-2048-hca-cp8r4 | tilelang@main | 15.7 | 1.44 | 1.20 | 5.7 | 7.4 | 0.7 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@main | 15.7 | 1.92 | 1.92 | 23.6 | - | - |
| heavy-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 15.7 | 1.44 | 1.20 | 5.6 | 6.9 | 0.7 |
| heavy-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 15.7 | 1.44 | 1.44 | 14.6 | 10.6 | 1.1 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 15.7 | 1.92 | 1.92 | 23.8 | - | - |
| heavy-2048-hca-cp8r7 | tilelang@main | 16.4 | 1.38 | 1.15 | 6.0 | 7.6 | 0.8 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@main | 16.4 | 1.83 | 1.83 | 24.7 | - | - |
| heavy-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 16.4 | 1.38 | 1.15 | 5.9 | 7.4 | 0.7 |
| heavy-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 16.4 | 1.38 | 1.38 | 15.0 | 11.0 | 1.1 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 16.4 | 1.83 | 1.83 | 24.3 | - | - |
| heavy-2048-sliding-cp1 | tilelang@main | 109.1 | 1.05 | 1.03 | 31.2 | 47.5 | 4.8 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@main | 109.1 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-2048-sliding-cp1 | tilelang@cudnn_flashmla | 109.1 | 1.05 | 1.03 | 31.3 | 46.1 | 4.7 |
| heavy-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 109.1 | 1.05 | 1.05 | 88.0 | 72.0 | 7.3 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 109.1 | 1.10 | 1.10 | 106.9 | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@main | 8.1 | 1.39 | 1.19 | 3.2 | 4.0 | 0.4 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 8.1 | 1.85 | 1.85 | 12.9 | - | - |
| heavy-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 8.1 | 1.39 | 1.19 | 3.1 | 3.8 | 0.4 |
| heavy-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 8.1 | 1.39 | 1.39 | 7.5 | 5.5 | 0.6 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 8.1 | 1.85 | 1.85 | 12.3 | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.8 | - | - |
| heavy-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 5.8 | 7.2 | 0.7 |
| heavy-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 14.1 | 10.3 | 1.0 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 23.6 | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@main | 15.0 | 1.00 | 1.00 | 5.8 | 7.3 | 0.7 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 15.0 | 1.00 | 1.00 | 23.5 | - | - |
| heavy-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 5.7 | 7.0 | 0.7 |
| heavy-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 13.8 | 10.1 | 1.0 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 15.0 | 1.00 | 1.00 | 23.5 | - | - |
| tiny-2048-csa-cp1 | tilelang@main | 53.6 | 3.28 | 2.72 | 14.7 | 22.7 | 2.3 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@main | 53.6 | 11.22 | 11.22 | 45.9 | - | - |
| tiny-2048-csa-cp1 | tilelang@cudnn_flashmla | 53.6 | 3.28 | 2.72 | 14.3 | 22.4 | 2.3 |
| tiny-2048-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 53.6 | 3.28 | 3.28 | 38.6 | 35.6 | 3.6 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 53.6 | 11.22 | 11.22 | 45.6 | - | - |
| tiny-2048-csa-cp8r0 | tilelang@main | 6.6 | 3.31 | 2.74 | 2.5 | 3.2 | 0.3 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@main | 6.6 | 11.38 | 11.38 | 9.8 | - | - |
| tiny-2048-csa-cp8r0 | tilelang@cudnn_flashmla | 6.6 | 3.31 | 2.74 | 2.5 | 3.1 | 0.3 |
| tiny-2048-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 6.6 | 3.31 | 3.31 | 6.2 | 4.4 | 0.4 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 6.6 | 11.38 | 11.38 | 10.1 | - | - |
| tiny-2048-csa-cp8r4 | tilelang@main | 4.7 | 4.58 | 3.79 | 1.8 | 2.3 | 0.2 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@main | 4.7 | 15.89 | 15.89 | 7.2 | - | - |
| tiny-2048-csa-cp8r4 | tilelang@cudnn_flashmla | 4.7 | 4.58 | 3.79 | 1.8 | 2.2 | 0.2 |
| tiny-2048-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 4.7 | 4.58 | 4.58 | 4.3 | 3.2 | 0.3 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 4.7 | 15.89 | 15.89 | 7.2 | - | - |
| tiny-2048-csa-cp8r7 | tilelang@main | 8.6 | 2.59 | 2.15 | 3.2 | 4.2 | 0.4 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@main | 8.6 | 8.76 | 8.76 | 12.9 | - | - |
| tiny-2048-csa-cp8r7 | tilelang@cudnn_flashmla | 8.6 | 2.59 | 2.15 | 3.2 | 4.1 | 0.4 |
| tiny-2048-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 8.6 | 2.59 | 2.59 | 8.0 | 5.8 | 0.6 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 8.6 | 8.76 | 8.76 | 13.3 | - | - |
| tiny-2048-hca-cp1 | tilelang@main | 43.2 | 1.79 | 1.36 | 12.8 | 20.6 | 2.1 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@main | 43.2 | 2.79 | 2.79 | 42.2 | - | - |
| tiny-2048-hca-cp1 | tilelang@cudnn_flashmla | 43.2 | 1.79 | 1.36 | 12.7 | 19.7 | 2.0 |
| tiny-2048-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 43.2 | 1.79 | 1.79 | 34.7 | 28.9 | 2.9 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 43.2 | 2.79 | 2.79 | 41.7 | - | - |
| tiny-2048-hca-cp8r0 | tilelang@main | 5.3 | 1.76 | 1.37 | 2.1 | 2.6 | 0.3 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@main | 5.3 | 2.83 | 2.83 | 8.3 | - | - |
| tiny-2048-hca-cp8r0 | tilelang@cudnn_flashmla | 5.3 | 1.76 | 1.37 | 2.0 | 2.5 | 0.3 |
| tiny-2048-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 5.3 | 1.76 | 1.76 | 5.0 | 3.6 | 0.4 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 5.3 | 2.83 | 2.83 | 8.4 | - | - |
| tiny-2048-hca-cp8r4 | tilelang@main | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@main | 3.8 | 3.94 | 3.94 | 6.0 | - | - |
| tiny-2048-hca-cp8r4 | tilelang@cudnn_flashmla | 3.8 | 2.23 | 1.52 | 1.5 | 1.8 | 0.2 |
| tiny-2048-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 3.8 | 2.23 | 2.23 | 3.6 | 2.6 | 0.3 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 3.8 | 3.94 | 3.94 | 5.9 | - | - |
| tiny-2048-hca-cp8r7 | tilelang@main | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@main | 6.9 | 2.18 | 2.18 | 10.4 | - | - |
| tiny-2048-hca-cp8r7 | tilelang@cudnn_flashmla | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 6.9 | 1.60 | 1.60 | 6.4 | 4.7 | 0.5 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 6.9 | 2.18 | 2.18 | 11.1 | - | - |
| tiny-2048-sliding-cp1 | tilelang@main | 43.2 | 1.79 | 1.36 | 12.9 | 20.5 | 2.1 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@main | 43.2 | 2.79 | 2.79 | 42.4 | - | - |
| tiny-2048-sliding-cp1 | tilelang@cudnn_flashmla | 43.2 | 1.79 | 1.36 | 12.8 | 19.8 | 2.0 |
| tiny-2048-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 43.2 | 1.79 | 1.79 | 34.9 | 28.8 | 2.9 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 43.2 | 2.79 | 2.79 | 42.2 | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@main | 5.3 | 1.76 | 1.37 | 2.1 | 2.5 | 0.3 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@main | 5.3 | 2.83 | 2.83 | 8.5 | - | - |
| tiny-2048-sliding-cp8r0 | tilelang@cudnn_flashmla | 5.3 | 1.76 | 1.37 | 2.0 | 2.5 | 0.3 |
| tiny-2048-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 5.3 | 1.76 | 1.76 | 4.9 | 3.6 | 0.4 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 5.3 | 2.83 | 2.83 | 8.2 | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@main | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@main | 3.8 | 3.94 | 3.94 | 6.1 | - | - |
| tiny-2048-sliding-cp8r4 | tilelang@cudnn_flashmla | 3.8 | 2.23 | 1.52 | 1.5 | 1.8 | 0.2 |
| tiny-2048-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 3.8 | 2.23 | 2.23 | 3.6 | 2.6 | 0.3 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 3.8 | 3.94 | 3.94 | 6.0 | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@main | 6.9 | 1.60 | 1.28 | 2.6 | 3.4 | 0.3 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@main | 6.9 | 2.18 | 2.18 | 10.6 | - | - |
| tiny-2048-sliding-cp8r7 | tilelang@cudnn_flashmla | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 6.9 | 1.60 | 1.60 | 6.3 | 4.7 | 0.5 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 6.9 | 2.18 | 2.18 | 10.7 | - | - |
| single-4096-csa-cp1 | tilelang@main | 958.1 | 1.03 | 1.02 | 144.8 | 160.4 | 16.2 |
| single-4096-csa-cp1 | flashmla_fwd_ref@main | 958.1 | 1.26 | 1.26 | 382.7 | - | - |
| single-4096-csa-cp1 | tilelang@cudnn_flashmla | 958.1 | 1.03 | 1.02 | 143.5 | 159.4 | 16.1 |
| single-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 958.1 | 1.03 | 1.03 | 347.9 | 306.7 | 31.0 |
| single-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 958.1 | 1.26 | 1.26 | 381.4 | - | - |
| single-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.8 | 20.3 | 2.0 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 57.2 | - | - |
| single-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 41.3 | 1.27 | 1.18 | 14.5 | 19.5 | 2.0 |
| single-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 41.3 | 1.27 | 1.27 | 38.7 | 27.8 | 2.8 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 41.3 | 3.64 | 3.64 | 56.8 | - | - |
| single-4096-csa-cp8r4 | tilelang@main | 150.3 | 1.00 | 1.00 | 49.4 | 64.4 | 6.5 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 150.3 | 1.00 | 1.00 | 172.9 | - | - |
| single-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 150.3 | 1.00 | 1.00 | 48.3 | 62.1 | 6.3 |
| single-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 150.3 | 1.00 | 1.00 | 138.3 | 100.4 | 10.1 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 150.3 | 1.00 | 1.00 | 170.9 | - | - |
| single-4096-csa-cp8r7 | tilelang@main | 150.3 | 1.00 | 1.00 | 48.8 | 64.0 | 6.5 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@main | 150.3 | 1.00 | 1.00 | 170.5 | - | - |
| single-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 150.3 | 1.00 | 1.00 | 48.1 | 63.0 | 6.4 |
| single-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 150.3 | 1.00 | 1.00 | 137.6 | 102.2 | 10.3 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 150.3 | 1.00 | 1.00 | 169.4 | - | - |
| single-4096-hca-cp1 | tilelang@main | 265.9 | 1.34 | 1.11 | 53.0 | 84.9 | 8.6 |
| single-4096-hca-cp1 | flashmla_fwd_ref@main | 265.9 | 1.81 | 1.81 | 150.3 | - | - |
| single-4096-hca-cp1 | tilelang@cudnn_flashmla | 265.9 | 1.34 | 1.11 | 53.2 | 82.7 | 8.4 |
| single-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 265.9 | 1.34 | 1.34 | 132.6 | 125.2 | 12.7 |
| single-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 265.9 | 1.81 | 1.81 | 149.4 | - | - |
| single-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.2 | 12.6 | 1.3 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.1 | - | - |
| single-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 26.7 | 1.48 | 1.23 | 8.8 | 12.1 | 1.2 |
| single-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 26.7 | 1.48 | 1.48 | 24.3 | 18.2 | 1.8 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 26.7 | 2.25 | 2.25 | 35.6 | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 34.2 | 1.32 | 1.10 | 11.8 | 16.0 | 1.6 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 34.2 | 1.76 | 1.76 | 45.0 | - | - |
| single-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 34.2 | 1.32 | 1.10 | 11.7 | 15.2 | 1.5 |
| single-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 34.2 | 1.32 | 1.32 | 31.2 | 22.8 | 2.3 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 34.2 | 1.76 | 1.76 | 45.0 | - | - |
| single-4096-hca-cp8r7 | tilelang@main | 37.0 | 1.22 | 1.02 | 12.8 | 17.1 | 1.7 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@main | 37.0 | 1.63 | 1.63 | 48.7 | - | - |
| single-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 37.0 | 1.22 | 1.02 | 12.5 | 16.7 | 1.7 |
| single-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 37.0 | 1.22 | 1.22 | 33.2 | 25.1 | 2.5 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 37.0 | 1.63 | 1.63 | 47.3 | - | - |
| single-4096-sliding-cp1 | tilelang@main | 236.8 | 1.01 | 1.00 | 52.2 | 82.8 | 8.4 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@main | 236.8 | 1.02 | 1.02 | 161.0 | - | - |
| single-4096-sliding-cp1 | tilelang@cudnn_flashmla | 236.8 | 1.01 | 1.00 | 52.3 | 81.1 | 8.2 |
| single-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 236.8 | 1.01 | 1.01 | 140.0 | 118.0 | 11.9 |
| single-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 236.8 | 1.02 | 1.02 | 161.6 | - | - |
| single-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 37.8 | - | - |
| single-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 26.3 | 1.07 | 1.03 | 9.4 | 12.3 | 1.2 |
| single-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 26.3 | 1.07 | 1.07 | 24.1 | 17.8 | 1.8 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 26.3 | 1.14 | 1.14 | 37.4 | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 30.1 | 1.00 | 1.00 | 42.3 | - | - |
| single-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 30.1 | 1.00 | 1.00 | 10.7 | 14.2 | 1.4 |
| single-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 30.1 | 1.00 | 1.00 | 28.2 | 20.5 | 2.1 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 30.1 | 1.00 | 1.00 | 42.9 | - | - |
| single-4096-sliding-cp8r7 | tilelang@main | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 30.1 | 1.00 | 1.00 | 44.3 | - | - |
| single-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 30.1 | 1.00 | 1.00 | 10.7 | 14.1 | 1.4 |
| single-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 30.1 | 1.00 | 1.00 | 27.5 | 20.4 | 2.1 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 30.1 | 1.00 | 1.00 | 42.0 | - | - |
| short-4096-csa-cp1 | tilelang@main | 449.8 | 1.18 | 1.11 | 84.4 | 116.1 | 11.7 |
| short-4096-csa-cp1 | flashmla_fwd_ref@main | 449.8 | 2.67 | 2.67 | 232.4 | - | - |
| short-4096-csa-cp1 | tilelang@cudnn_flashmla | 449.8 | 1.18 | 1.11 | 83.9 | 116.0 | 11.7 |
| short-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 449.8 | 1.18 | 1.18 | 209.3 | 189.8 | 19.2 |
| short-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 449.8 | 2.67 | 2.67 | 232.6 | - | - |
| short-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.3 | 20.1 | 2.0 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 53.1 | - | - |
| short-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 41.3 | 1.27 | 1.18 | 14.6 | 19.4 | 2.0 |
| short-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 41.3 | 1.27 | 1.27 | 39.0 | 27.6 | 2.8 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 41.3 | 3.64 | 3.64 | 55.8 | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 44.6 | 1.21 | 1.13 | 14.5 | 20.8 | 2.1 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 44.6 | 3.37 | 3.37 | 56.1 | - | - |
| short-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 44.6 | 1.21 | 1.13 | 15.7 | 21.0 | 2.1 |
| short-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 44.6 | 1.21 | 1.21 | 40.6 | 30.1 | 3.0 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 44.6 | 3.37 | 3.37 | 58.4 | - | - |
| short-4096-csa-cp8r7 | tilelang@main | 42.3 | 1.26 | 1.17 | 14.8 | 20.6 | 2.1 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@main | 42.3 | 3.55 | 3.55 | 57.4 | - | - |
| short-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 42.3 | 1.26 | 1.17 | 15.1 | 20.3 | 2.0 |
| short-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 42.3 | 1.26 | 1.26 | 39.3 | 28.6 | 2.9 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 42.3 | 3.55 | 3.55 | 56.9 | - | - |
| short-4096-hca-cp1 | tilelang@main | 228.1 | 1.46 | 1.22 | 46.2 | 73.8 | 7.5 |
| short-4096-hca-cp1 | flashmla_fwd_ref@main | 228.1 | 2.11 | 2.11 | 130.3 | - | - |
| short-4096-hca-cp1 | tilelang@cudnn_flashmla | 228.1 | 1.46 | 1.22 | 46.4 | 72.7 | 7.3 |
| short-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 228.1 | 1.46 | 1.46 | 114.6 | 111.4 | 11.3 |
| short-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 228.1 | 2.11 | 2.11 | 129.8 | - | - |
| short-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.0 | 12.2 | 1.2 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.3 | - | - |
| short-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 26.7 | 1.48 | 1.23 | 9.1 | 12.0 | 1.2 |
| short-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 26.7 | 1.48 | 1.48 | 24.1 | 18.0 | 1.8 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 26.7 | 2.25 | 2.25 | 35.2 | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 28.3 | 1.46 | 1.23 | 9.8 | 13.1 | 1.3 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.3 | 2.13 | 2.13 | 38.8 | - | - |
| short-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 28.3 | 1.46 | 1.23 | 9.4 | 12.6 | 1.3 |
| short-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 28.3 | 1.46 | 1.46 | 24.8 | 19.0 | 1.9 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 28.3 | 2.13 | 2.13 | 37.3 | - | - |
| short-4096-hca-cp8r7 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.2 | 12.2 | 1.2 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.7 | - | - |
| short-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 26.7 | 1.48 | 1.23 | 9.0 | 12.0 | 1.2 |
| short-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 26.7 | 1.48 | 1.48 | 23.9 | 18.0 | 1.8 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 26.7 | 2.25 | 2.25 | 34.7 | - | - |
| short-4096-sliding-cp1 | tilelang@main | 221.9 | 1.04 | 1.02 | 48.3 | 78.1 | 7.9 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@main | 221.9 | 1.08 | 1.08 | 149.8 | - | - |
| short-4096-sliding-cp1 | tilelang@cudnn_flashmla | 221.9 | 1.04 | 1.02 | 49.0 | 76.8 | 7.8 |
| short-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 221.9 | 1.04 | 1.04 | 131.6 | 112.5 | 11.4 |
| short-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 221.9 | 1.08 | 1.08 | 150.6 | - | - |
| short-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 38.1 | - | - |
| short-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 26.3 | 1.07 | 1.03 | 9.5 | 12.4 | 1.2 |
| short-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 26.3 | 1.07 | 1.07 | 24.1 | 17.5 | 1.8 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 26.3 | 1.14 | 1.14 | 37.2 | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 27.9 | 1.04 | 1.02 | 10.0 | 13.7 | 1.4 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 27.9 | 1.08 | 1.08 | 40.3 | - | - |
| short-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 27.9 | 1.04 | 1.02 | 10.0 | 13.1 | 1.3 |
| short-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 27.9 | 1.04 | 1.04 | 25.7 | 18.8 | 1.9 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 27.9 | 1.08 | 1.08 | 40.1 | - | - |
| short-4096-sliding-cp8r7 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.7 | 12.6 | 1.3 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 38.4 | - | - |
| short-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 26.3 | 1.07 | 1.03 | 9.5 | 12.4 | 1.3 |
| short-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 26.3 | 1.07 | 1.07 | 23.8 | 17.9 | 1.8 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 26.3 | 1.14 | 1.14 | 36.9 | - | - |
| heavy-4096-csa-cp1 | tilelang@main | 356.7 | 1.29 | 1.19 | 69.7 | 100.6 | 10.2 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@main | 356.7 | 3.37 | 3.37 | 194.2 | - | - |
| heavy-4096-csa-cp1 | tilelang@cudnn_flashmla | 356.7 | 1.29 | 1.19 | 69.6 | 99.9 | 10.1 |
| heavy-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 356.7 | 1.29 | 1.29 | 170.8 | 159.8 | 16.1 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 356.7 | 3.37 | 3.37 | 191.9 | - | - |
| heavy-4096-csa-cp8r0 | tilelang@main | 41.3 | 1.27 | 1.18 | 14.9 | 20.3 | 2.1 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@main | 41.3 | 3.64 | 3.64 | 56.7 | - | - |
| heavy-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 41.3 | 1.27 | 1.18 | 14.2 | 19.4 | 2.0 |
| heavy-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 41.3 | 1.27 | 1.27 | 37.6 | 27.8 | 2.8 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 41.3 | 3.64 | 3.64 | 55.1 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 45.5 | 1.20 | 1.12 | 15.9 | 22.2 | 2.2 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 45.5 | 3.30 | 3.30 | 59.9 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 45.5 | 1.20 | 1.12 | 15.9 | 21.4 | 2.2 |
| heavy-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 45.5 | 1.20 | 1.20 | 42.7 | 30.8 | 3.1 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 45.5 | 3.30 | 3.30 | 61.0 | - | - |
| heavy-4096-csa-cp8r7 | tilelang@main | 29.6 | 1.51 | 1.38 | 9.9 | 14.2 | 1.4 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@main | 29.6 | 5.09 | 5.09 | 39.1 | - | - |
| heavy-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 29.6 | 1.51 | 1.38 | 10.5 | 14.1 | 1.4 |
| heavy-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 29.6 | 1.51 | 1.51 | 27.6 | 20.3 | 2.0 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 29.6 | 5.09 | 5.09 | 40.0 | - | - |
| heavy-4096-hca-cp1 | tilelang@main | 207.2 | 1.47 | 1.23 | 43.1 | 69.5 | 7.0 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@main | 207.2 | 2.32 | 2.32 | 122.6 | - | - |
| heavy-4096-hca-cp1 | tilelang@cudnn_flashmla | 207.2 | 1.47 | 1.23 | 42.5 | 68.3 | 6.9 |
| heavy-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 207.2 | 1.47 | 1.47 | 106.0 | 102.2 | 10.3 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 207.2 | 2.32 | 2.32 | 120.4 | - | - |
| heavy-4096-hca-cp8r0 | tilelang@main | 26.7 | 1.48 | 1.23 | 9.1 | 12.4 | 1.3 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@main | 26.7 | 2.25 | 2.25 | 35.5 | - | - |
| heavy-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 26.7 | 1.48 | 1.23 | 8.8 | 12.2 | 1.2 |
| heavy-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 26.7 | 1.48 | 1.48 | 24.2 | 18.0 | 1.8 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 26.7 | 2.25 | 2.25 | 35.9 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 28.7 | 1.46 | 1.22 | 9.9 | 13.3 | 1.3 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.7 | 2.10 | 2.10 | 39.5 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 28.7 | 1.46 | 1.22 | 9.9 | 12.8 | 1.3 |
| heavy-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 28.7 | 1.46 | 1.46 | 26.1 | 19.3 | 1.9 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 28.7 | 2.10 | 2.10 | 38.6 | - | - |
| heavy-4096-hca-cp8r7 | tilelang@main | 22.7 | 1.49 | 1.24 | 7.7 | 10.4 | 1.1 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@main | 22.7 | 2.65 | 2.65 | 30.1 | - | - |
| heavy-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 22.7 | 1.49 | 1.24 | 7.6 | 10.2 | 1.0 |
| heavy-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 22.7 | 1.49 | 1.49 | 21.0 | 15.3 | 1.5 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 22.7 | 2.65 | 2.65 | 31.4 | - | - |
| heavy-4096-sliding-cp1 | tilelang@main | 203.2 | 1.09 | 1.04 | 45.3 | 69.8 | 7.1 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@main | 203.2 | 1.18 | 1.18 | 138.7 | - | - |
| heavy-4096-sliding-cp1 | tilelang@cudnn_flashmla | 203.2 | 1.09 | 1.04 | 45.1 | 71.2 | 7.2 |
| heavy-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 203.2 | 1.09 | 1.09 | 121.2 | 103.6 | 10.5 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 203.2 | 1.18 | 1.18 | 139.1 | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@main | 26.3 | 1.07 | 1.03 | 9.5 | 12.6 | 1.3 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 26.3 | 1.14 | 1.14 | 37.9 | - | - |
| heavy-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 26.3 | 1.07 | 1.03 | 9.3 | 12.3 | 1.2 |
| heavy-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 26.3 | 1.07 | 1.07 | 24.0 | 17.9 | 1.8 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 26.3 | 1.14 | 1.14 | 37.0 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 28.3 | 1.04 | 1.02 | 10.3 | 13.8 | 1.4 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 28.3 | 1.06 | 1.06 | 41.6 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 28.3 | 1.04 | 1.02 | 10.2 | 13.4 | 1.4 |
| heavy-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 28.3 | 1.04 | 1.04 | 26.2 | 19.4 | 2.0 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 28.3 | 1.06 | 1.06 | 40.5 | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@main | 22.6 | 1.16 | 1.08 | 8.4 | 10.8 | 1.1 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 22.6 | 1.33 | 1.33 | 32.7 | - | - |
| heavy-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 22.6 | 1.16 | 1.08 | 8.2 | 10.8 | 1.1 |
| heavy-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 22.6 | 1.16 | 1.16 | 20.5 | 15.3 | 1.6 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 22.6 | 1.33 | 1.33 | 32.0 | - | - |
| tiny-4096-csa-cp1 | tilelang@main | 104.6 | 3.35 | 2.77 | 21.8 | 35.8 | 3.6 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@main | 104.6 | 11.49 | 11.49 | 59.6 | - | - |
| tiny-4096-csa-cp1 | tilelang@cudnn_flashmla | 104.6 | 3.35 | 2.77 | 21.5 | 35.8 | 3.6 |
| tiny-4096-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 104.6 | 3.35 | 3.35 | 53.9 | 53.6 | 5.4 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 104.6 | 11.49 | 11.49 | 60.0 | - | - |
| tiny-4096-csa-cp8r0 | tilelang@main | 11.9 | 3.67 | 3.04 | 4.3 | 5.9 | 0.6 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@main | 11.9 | 12.64 | 12.64 | 16.1 | - | - |
| tiny-4096-csa-cp8r0 | tilelang@cudnn_flashmla | 11.9 | 3.67 | 3.04 | 4.3 | 5.6 | 0.6 |
| tiny-4096-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 11.9 | 3.67 | 3.67 | 11.1 | 7.9 | 0.8 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 11.9 | 12.64 | 12.64 | 16.2 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 15.1 | 2.92 | 2.42 | 5.3 | 7.4 | 0.7 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 15.1 | 9.98 | 9.98 | 20.5 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@cudnn_flashmla | 15.1 | 2.92 | 2.42 | 5.3 | 7.1 | 0.7 |
| tiny-4096-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 15.1 | 2.92 | 2.92 | 13.9 | 10.1 | 1.0 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 15.1 | 9.98 | 9.98 | 20.7 | - | - |
| tiny-4096-csa-cp8r7 | tilelang@main | 14.5 | 3.04 | 2.52 | 5.3 | 7.1 | 0.7 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@main | 14.5 | 10.36 | 10.36 | 19.7 | - | - |
| tiny-4096-csa-cp8r7 | tilelang@cudnn_flashmla | 14.5 | 3.04 | 2.52 | 5.3 | 6.9 | 0.7 |
| tiny-4096-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 14.5 | 3.04 | 3.04 | 13.8 | 9.8 | 1.0 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 14.5 | 10.36 | 10.36 | 19.9 | - | - |
| tiny-4096-hca-cp1 | tilelang@main | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@main | 84.3 | 2.85 | 2.85 | 58.4 | - | - |
| tiny-4096-hca-cp1 | tilelang@cudnn_flashmla | 84.3 | 1.81 | 1.37 | 19.7 | 34.2 | 3.5 |
| tiny-4096-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 84.3 | 1.81 | 1.81 | 50.1 | 47.4 | 4.8 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 84.3 | 2.85 | 2.85 | 57.7 | - | - |
| tiny-4096-hca-cp8r0 | tilelang@main | 9.6 | 1.91 | 1.41 | 3.6 | 4.7 | 0.5 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@main | 9.6 | 3.14 | 3.14 | 14.1 | - | - |
| tiny-4096-hca-cp8r0 | tilelang@cudnn_flashmla | 9.6 | 1.91 | 1.41 | 3.5 | 4.5 | 0.5 |
| tiny-4096-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 9.6 | 1.91 | 1.91 | 8.7 | 6.5 | 0.7 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 9.6 | 3.14 | 3.14 | 13.6 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.5 | 6.0 | 0.6 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.6 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@cudnn_flashmla | 12.1 | 1.66 | 1.32 | 4.4 | 5.6 | 0.6 |
| tiny-4096-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 12.1 | 1.66 | 1.66 | 11.2 | 8.1 | 0.8 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 12.1 | 2.48 | 2.48 | 17.3 | - | - |
| tiny-4096-hca-cp8r7 | tilelang@main | 11.7 | 1.70 | 1.33 | 4.3 | 5.7 | 0.6 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@main | 11.7 | 2.57 | 2.57 | 16.6 | - | - |
| tiny-4096-hca-cp8r7 | tilelang@cudnn_flashmla | 11.7 | 1.70 | 1.33 | 4.2 | 5.5 | 0.6 |
| tiny-4096-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 11.7 | 1.70 | 1.70 | 10.7 | 7.9 | 0.8 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 11.7 | 2.57 | 2.57 | 16.5 | - | - |
| tiny-4096-sliding-cp1 | tilelang@main | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@main | 84.3 | 2.85 | 2.85 | 58.9 | - | - |
| tiny-4096-sliding-cp1 | tilelang@cudnn_flashmla | 84.3 | 1.81 | 1.37 | 19.6 | 33.9 | 3.4 |
| tiny-4096-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 84.3 | 1.81 | 1.81 | 50.3 | 47.1 | 4.8 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 84.3 | 2.85 | 2.85 | 57.8 | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@main | 9.6 | 1.91 | 1.41 | 3.5 | 4.7 | 0.5 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@main | 9.6 | 3.14 | 3.14 | 13.6 | - | - |
| tiny-4096-sliding-cp8r0 | tilelang@cudnn_flashmla | 9.6 | 1.91 | 1.41 | 3.5 | 4.5 | 0.5 |
| tiny-4096-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 9.6 | 1.91 | 1.91 | 9.0 | 6.5 | 0.7 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 9.6 | 3.14 | 3.14 | 14.0 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.4 | 5.9 | 0.6 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.1 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@cudnn_flashmla | 12.1 | 1.66 | 1.32 | 4.3 | 5.7 | 0.6 |
| tiny-4096-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 12.1 | 1.66 | 1.66 | 11.4 | 7.9 | 0.8 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 12.1 | 2.48 | 2.48 | 17.8 | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@main | 11.7 | 1.70 | 1.33 | 4.2 | 5.6 | 0.6 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@main | 11.7 | 2.57 | 2.57 | 16.5 | - | - |
| tiny-4096-sliding-cp8r7 | tilelang@cudnn_flashmla | 11.7 | 1.70 | 1.33 | 4.1 | 5.4 | 0.5 |
| tiny-4096-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 11.7 | 1.70 | 1.70 | 10.6 | 7.7 | 0.8 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 11.7 | 2.57 | 2.57 | 16.4 | - | - |
| single-16384-csa-cp1 | tilelang@main | 4565.9 | 1.01 | 1.00 | 220.5 | 193.8 | 19.6 |
| single-16384-csa-cp1 | flashmla_fwd_ref@main | 4565.9 | 1.05 | 1.05 | 507.6 | - | - |
| single-16384-csa-cp1 | tilelang@cudnn_flashmla | 4565.9 | 1.01 | 1.00 | 216.7 | 191.2 | 19.3 |
| single-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 4565.9 | 1.01 | 1.01 | 494.7 | 358.0 | 36.2 |
| single-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4565.9 | 1.05 | 1.05 | 487.2 | - | - |
| single-16384-csa-cp8r0 | tilelang@main | 356.8 | 1.09 | 1.05 | 82.7 | 110.0 | 11.1 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@main | 356.8 | 1.69 | 1.69 | 244.1 | - | - |
| single-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 356.8 | 1.09 | 1.05 | 82.2 | 108.2 | 10.9 |
| single-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 356.8 | 1.09 | 1.09 | 211.5 | 181.5 | 18.3 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 356.8 | 1.69 | 1.69 | 244.4 | - | - |
| single-16384-csa-cp8r4 | tilelang@main | 601.3 | 1.00 | 1.00 | 121.9 | 147.4 | 14.9 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@main | 601.3 | 1.00 | 1.00 | 347.0 | - | - |
| single-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 601.3 | 1.00 | 1.00 | 120.7 | 145.8 | 14.7 |
| single-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 601.3 | 1.00 | 1.00 | 313.0 | 257.9 | 26.1 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 601.3 | 1.00 | 1.00 | 351.7 | - | - |
| single-16384-csa-cp8r7 | tilelang@main | 601.3 | 1.00 | 1.00 | 122.3 | 146.5 | 14.8 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@main | 601.3 | 1.00 | 1.00 | 345.9 | - | - |
| single-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 601.3 | 1.00 | 1.00 | 120.4 | 145.7 | 14.7 |
| single-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 601.3 | 1.00 | 1.00 | 308.5 | 259.1 | 26.2 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 601.3 | 1.00 | 1.00 | 344.3 | - | - |
| single-16384-hca-cp1 | tilelang@main | 1435.7 | 1.17 | 1.08 | 114.8 | 131.7 | 13.3 |
| single-16384-hca-cp1 | flashmla_fwd_ref@main | 1435.7 | 1.34 | 1.34 | 275.4 | - | - |
| single-16384-hca-cp1 | tilelang@cudnn_flashmla | 1435.7 | 1.17 | 1.08 | 114.5 | 132.1 | 13.3 |
| single-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1435.7 | 1.17 | 1.17 | 261.3 | 219.8 | 22.2 |
| single-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1435.7 | 1.34 | 1.34 | 275.6 | - | - |
| single-16384-hca-cp8r0 | tilelang@main | 123.6 | 1.41 | 1.18 | 32.7 | 51.0 | 5.2 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@main | 123.6 | 1.95 | 1.95 | 103.7 | - | - |
| single-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 123.6 | 1.41 | 1.18 | 33.4 | 50.0 | 5.1 |
| single-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 123.6 | 1.41 | 1.41 | 88.4 | 77.4 | 7.8 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 123.6 | 1.95 | 1.95 | 104.5 | - | - |
| single-16384-hca-cp8r4 | tilelang@main | 187.4 | 1.26 | 1.11 | 48.8 | 70.3 | 7.1 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@main | 187.4 | 1.28 | 1.28 | 157.6 | - | - |
| single-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 187.4 | 1.26 | 1.11 | 48.2 | 67.5 | 6.8 |
| single-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 187.4 | 1.26 | 1.26 | 133.3 | 109.0 | 11.0 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 187.4 | 1.28 | 1.28 | 156.7 | - | - |
| single-16384-hca-cp8r7 | tilelang@main | 232.5 | 1.03 | 1.03 | 60.1 | 82.5 | 8.3 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@main | 232.5 | 1.03 | 1.03 | 193.2 | - | - |
| single-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 232.5 | 1.03 | 1.03 | 59.8 | 82.3 | 8.3 |
| single-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 232.5 | 1.03 | 1.03 | 164.8 | 134.7 | 13.6 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 232.5 | 1.03 | 1.03 | 194.2 | - | - |
| single-16384-sliding-cp1 | tilelang@main | 958.3 | 1.00 | 1.00 | 92.1 | 115.7 | 11.7 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@main | 958.3 | 1.00 | 1.00 | 235.2 | - | - |
| single-16384-sliding-cp1 | tilelang@cudnn_flashmla | 958.3 | 1.00 | 1.00 | 91.6 | 115.5 | 11.7 |
| single-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 958.3 | 1.00 | 1.00 | 220.5 | 179.6 | 18.2 |
| single-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 958.3 | 1.00 | 1.00 | 234.6 | - | - |
| single-16384-sliding-cp8r0 | tilelang@main | 116.5 | 1.02 | 1.01 | 32.7 | 49.9 | 5.0 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 111.1 | - | - |
| single-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 116.5 | 1.02 | 1.01 | 32.7 | 48.4 | 4.9 |
| single-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 116.5 | 1.02 | 1.02 | 93.9 | 75.4 | 7.6 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 116.5 | 1.03 | 1.03 | 113.5 | - | - |
| single-16384-sliding-cp8r4 | tilelang@main | 120.3 | 1.00 | 1.00 | 34.2 | 50.3 | 5.1 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 115.8 | - | - |
| single-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 34.2 | 50.2 | 5.1 |
| single-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 98.0 | 77.5 | 7.8 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 118.6 | - | - |
| single-16384-sliding-cp8r7 | tilelang@main | 120.3 | 1.00 | 1.00 | 34.5 | 50.3 | 5.1 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 116.9 | - | - |
| single-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 33.9 | 49.8 | 5.0 |
| single-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 97.2 | 77.8 | 7.9 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 116.3 | - | - |
| short-16384-csa-cp1 | tilelang@main | 2610.6 | 1.11 | 1.06 | 165.4 | 163.4 | 16.5 |
| short-16384-csa-cp1 | flashmla_fwd_ref@main | 2610.6 | 1.84 | 1.84 | 381.2 | - | - |
| short-16384-csa-cp1 | tilelang@cudnn_flashmla | 2610.6 | 1.11 | 1.06 | 165.0 | 162.4 | 16.4 |
| short-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 2610.6 | 1.11 | 1.11 | 370.5 | 293.9 | 29.7 |
| short-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 2610.6 | 1.84 | 1.84 | 381.8 | - | - |
| short-16384-csa-cp8r0 | tilelang@main | 280.2 | 1.14 | 1.08 | 68.1 | 93.5 | 9.5 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@main | 280.2 | 2.15 | 2.15 | 205.3 | - | - |
| short-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 280.2 | 1.14 | 1.08 | 68.2 | 92.1 | 9.3 |
| short-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 280.2 | 1.14 | 1.14 | 177.2 | 151.3 | 15.3 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 280.2 | 2.15 | 2.15 | 207.3 | - | - |
| short-16384-csa-cp8r4 | tilelang@main | 410.5 | 1.07 | 1.04 | 93.9 | 121.2 | 12.3 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@main | 410.5 | 1.46 | 1.46 | 271.8 | - | - |
| short-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 410.5 | 1.07 | 1.04 | 93.9 | 119.0 | 12.0 |
| short-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 410.5 | 1.07 | 1.07 | 239.4 | 202.3 | 20.4 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 410.5 | 1.46 | 1.46 | 271.7 | - | - |
| short-16384-csa-cp8r7 | tilelang@main | 352.9 | 1.10 | 1.06 | 83.3 | 108.9 | 11.0 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@main | 352.9 | 1.70 | 1.70 | 247.1 | - | - |
| short-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 352.9 | 1.10 | 1.06 | 82.3 | 109.2 | 11.0 |
| short-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 352.9 | 1.10 | 1.10 | 213.1 | 181.3 | 18.3 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 352.9 | 1.70 | 1.70 | 243.7 | - | - |
| short-16384-hca-cp1 | tilelang@main | 963.6 | 1.42 | 1.18 | 82.5 | 104.5 | 10.6 |
| short-16384-hca-cp1 | flashmla_fwd_ref@main | 963.6 | 2.00 | 2.00 | 188.2 | - | - |
| short-16384-hca-cp1 | tilelang@cudnn_flashmla | 963.6 | 1.42 | 1.18 | 82.8 | 105.4 | 10.6 |
| short-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 963.6 | 1.42 | 1.42 | 178.4 | 163.8 | 16.6 |
| short-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 963.6 | 2.00 | 2.00 | 188.3 | - | - |
| short-16384-hca-cp8r0 | tilelang@main | 117.6 | 1.44 | 1.20 | 31.4 | 47.0 | 4.8 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@main | 117.6 | 2.05 | 2.05 | 99.1 | - | - |
| short-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 117.6 | 1.44 | 1.20 | 31.0 | 46.7 | 4.7 |
| short-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 117.6 | 1.44 | 1.44 | 83.5 | 74.7 | 7.5 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 117.6 | 2.05 | 2.05 | 99.5 | - | - |
| short-16384-hca-cp8r4 | tilelang@main | 125.5 | 1.39 | 1.16 | 33.2 | 49.5 | 5.0 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@main | 125.5 | 1.92 | 1.92 | 106.2 | - | - |
| short-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 125.5 | 1.39 | 1.16 | 33.0 | 48.1 | 4.9 |
| short-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 125.5 | 1.39 | 1.39 | 89.1 | 78.0 | 7.9 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 125.5 | 1.92 | 1.92 | 105.0 | - | - |
| short-16384-hca-cp8r7 | tilelang@main | 119.9 | 1.41 | 1.18 | 31.7 | 46.9 | 4.7 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@main | 119.9 | 2.01 | 2.01 | 101.1 | - | - |
| short-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 119.9 | 1.41 | 1.18 | 31.7 | 46.9 | 4.7 |
| short-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 119.9 | 1.41 | 1.41 | 84.5 | 75.2 | 7.6 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 119.9 | 2.01 | 2.01 | 100.5 | - | - |
| short-16384-sliding-cp1 | tilelang@main | 913.6 | 1.03 | 1.01 | 87.4 | 112.0 | 11.3 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@main | 913.6 | 1.05 | 1.05 | 223.3 | - | - |
| short-16384-sliding-cp1 | tilelang@cudnn_flashmla | 913.6 | 1.03 | 1.01 | 88.3 | 112.2 | 11.3 |
| short-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 913.6 | 1.03 | 1.03 | 210.9 | 173.2 | 17.5 |
| short-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 913.6 | 1.05 | 1.05 | 224.5 | - | - |
| short-16384-sliding-cp8r0 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.2 | 48.9 | 4.9 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 108.8 | - | - |
| short-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 112.8 | 1.03 | 1.02 | 32.2 | 46.9 | 4.7 |
| short-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 112.8 | 1.03 | 1.03 | 89.7 | 72.5 | 7.3 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 112.8 | 1.07 | 1.07 | 109.8 | - | - |
| short-16384-sliding-cp8r4 | tilelang@main | 116.5 | 1.02 | 1.01 | 33.5 | 49.2 | 5.0 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 116.5 | 1.03 | 1.03 | 113.1 | - | - |
| short-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 116.5 | 1.02 | 1.01 | 32.9 | 48.9 | 4.9 |
| short-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 116.5 | 1.02 | 1.02 | 94.1 | 76.0 | 7.7 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 116.5 | 1.03 | 1.03 | 114.8 | - | - |
| short-16384-sliding-cp8r7 | tilelang@main | 112.8 | 1.03 | 1.02 | 32.2 | 47.8 | 4.8 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 112.8 | 1.07 | 1.07 | 109.4 | - | - |
| short-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 112.8 | 1.03 | 1.02 | 31.7 | 46.8 | 4.7 |
| short-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 112.8 | 1.03 | 1.03 | 88.9 | 73.0 | 7.4 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 112.8 | 1.07 | 1.07 | 108.3 | - | - |
| heavy-16384-csa-cp1 | tilelang@main | 1795.9 | 1.22 | 1.14 | 129.3 | 138.2 | 14.0 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@main | 1795.9 | 2.68 | 2.68 | 297.3 | - | - |
| heavy-16384-csa-cp1 | tilelang@cudnn_flashmla | 1795.9 | 1.22 | 1.14 | 129.9 | 139.3 | 14.1 |
| heavy-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1795.9 | 1.22 | 1.22 | 286.8 | 241.3 | 24.4 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1795.9 | 2.68 | 2.68 | 298.0 | - | - |
| heavy-16384-csa-cp8r0 | tilelang@main | 154.2 | 1.36 | 1.24 | 40.3 | 59.9 | 6.1 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@main | 154.2 | 3.90 | 3.90 | 127.1 | - | - |
| heavy-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 154.2 | 1.36 | 1.24 | 40.3 | 57.9 | 5.9 |
| heavy-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 154.2 | 1.36 | 1.36 | 106.5 | 92.5 | 9.4 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 154.2 | 3.90 | 3.90 | 124.8 | - | - |
| heavy-16384-csa-cp8r4 | tilelang@main | 219.5 | 1.19 | 1.12 | 56.2 | 79.5 | 8.0 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@main | 219.5 | 2.74 | 2.74 | 172.9 | - | - |
| heavy-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 219.5 | 1.19 | 1.12 | 55.6 | 77.1 | 7.8 |
| heavy-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 219.5 | 1.19 | 1.19 | 144.5 | 125.8 | 12.7 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 219.5 | 2.74 | 2.74 | 169.2 | - | - |
| heavy-16384-csa-cp8r7 | tilelang@main | 410.5 | 1.06 | 1.03 | 92.8 | 117.9 | 11.9 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@main | 410.5 | 1.46 | 1.46 | 273.7 | - | - |
| heavy-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 410.5 | 1.06 | 1.03 | 91.6 | 118.0 | 11.9 |
| heavy-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 410.5 | 1.06 | 1.06 | 235.1 | 201.1 | 20.3 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 410.5 | 1.46 | 1.46 | 267.7 | - | - |
| heavy-16384-hca-cp1 | tilelang@main | 847.5 | 1.45 | 1.21 | 74.5 | 97.4 | 9.8 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@main | 847.5 | 2.27 | 2.27 | 172.5 | - | - |
| heavy-16384-hca-cp1 | tilelang@cudnn_flashmla | 847.5 | 1.45 | 1.21 | 75.1 | 97.7 | 9.9 |
| heavy-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 847.5 | 1.45 | 1.45 | 163.7 | 150.5 | 15.2 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 847.5 | 2.27 | 2.27 | 172.7 | - | - |
| heavy-16384-hca-cp8r0 | tilelang@main | 99.2 | 1.48 | 1.23 | 26.6 | 40.5 | 4.1 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@main | 99.2 | 2.42 | 2.42 | 84.5 | - | - |
| heavy-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 99.2 | 1.48 | 1.23 | 26.9 | 40.3 | 4.1 |
| heavy-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 99.2 | 1.48 | 1.48 | 73.4 | 64.2 | 6.5 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 99.2 | 2.42 | 2.42 | 87.0 | - | - |
| heavy-16384-hca-cp8r4 | tilelang@main | 112.5 | 1.46 | 1.22 | 30.1 | 45.2 | 4.6 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@main | 112.5 | 2.14 | 2.14 | 94.1 | - | - |
| heavy-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 112.5 | 1.46 | 1.22 | 30.0 | 44.1 | 4.5 |
| heavy-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 112.5 | 1.46 | 1.46 | 81.1 | 70.4 | 7.1 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 112.5 | 2.14 | 2.14 | 95.6 | - | - |
| heavy-16384-hca-cp8r7 | tilelang@main | 129.0 | 1.40 | 1.17 | 33.9 | 49.2 | 5.0 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@main | 129.0 | 1.86 | 1.86 | 106.5 | - | - |
| heavy-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 129.0 | 1.40 | 1.17 | 33.5 | 48.9 | 4.9 |
| heavy-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 129.0 | 1.40 | 1.40 | 89.5 | 78.8 | 8.0 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 129.0 | 1.86 | 1.86 | 105.4 | - | - |
| heavy-16384-sliding-cp1 | tilelang@main | 820.4 | 1.09 | 1.04 | 78.7 | 104.0 | 10.5 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@main | 820.4 | 1.17 | 1.17 | 199.8 | - | - |
| heavy-16384-sliding-cp1 | tilelang@cudnn_flashmla | 820.4 | 1.09 | 1.04 | 79.5 | 104.4 | 10.5 |
| heavy-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 820.4 | 1.09 | 1.09 | 188.7 | 159.2 | 16.1 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 820.4 | 1.17 | 1.17 | 200.9 | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@main | 97.9 | 1.11 | 1.06 | 27.6 | 42.7 | 4.3 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 97.9 | 1.23 | 1.23 | 93.9 | - | - |
| heavy-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 97.9 | 1.11 | 1.06 | 28.1 | 41.7 | 4.2 |
| heavy-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 97.9 | 1.11 | 1.11 | 78.2 | 64.2 | 6.5 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 97.9 | 1.23 | 1.23 | 94.7 | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@main | 109.5 | 1.05 | 1.02 | 31.8 | 47.0 | 4.7 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 109.5 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 109.5 | 1.05 | 1.02 | 30.8 | 46.5 | 4.7 |
| heavy-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 109.5 | 1.05 | 1.05 | 87.8 | 71.9 | 7.3 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 109.5 | 1.10 | 1.10 | 106.2 | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@main | 120.3 | 1.00 | 1.00 | 33.6 | 50.0 | 5.1 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 120.3 | 1.00 | 1.00 | 115.9 | - | - |
| heavy-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 33.8 | 48.3 | 4.9 |
| heavy-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 95.3 | 76.5 | 7.7 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 120.3 | 1.00 | 1.00 | 114.3 | - | - |
| tiny-16384-csa-cp1 | tilelang@main | 391.4 | 3.56 | 2.95 | 34.0 | 44.9 | 4.5 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@main | 391.4 | 12.29 | 12.29 | 75.8 | - | - |
| tiny-16384-csa-cp1 | tilelang@cudnn_flashmla | 391.4 | 3.56 | 2.95 | 33.8 | 45.0 | 4.5 |
| tiny-16384-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 391.4 | 3.56 | 3.56 | 72.7 | 70.5 | 7.1 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 391.4 | 12.29 | 12.29 | 75.7 | - | - |
| tiny-16384-csa-cp8r0 | tilelang@main | 50.6 | 3.45 | 2.86 | 13.9 | 21.3 | 2.2 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@main | 50.6 | 11.89 | 11.89 | 43.2 | - | - |
| tiny-16384-csa-cp8r0 | tilelang@cudnn_flashmla | 50.6 | 3.45 | 2.86 | 13.6 | 20.9 | 2.1 |
| tiny-16384-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 50.6 | 3.45 | 3.45 | 36.2 | 32.7 | 3.3 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 50.6 | 11.89 | 11.89 | 42.9 | - | - |
| tiny-16384-csa-cp8r4 | tilelang@main | 48.5 | 3.60 | 2.98 | 13.2 | 20.6 | 2.1 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@main | 48.5 | 12.39 | 12.39 | 40.5 | - | - |
| tiny-16384-csa-cp8r4 | tilelang@cudnn_flashmla | 48.5 | 3.60 | 2.98 | 13.3 | 20.3 | 2.0 |
| tiny-16384-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 48.5 | 3.60 | 3.60 | 34.9 | 31.7 | 3.2 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 48.5 | 12.39 | 12.39 | 40.4 | - | - |
| tiny-16384-csa-cp8r7 | tilelang@main | 43.9 | 3.96 | 3.27 | 12.1 | 18.5 | 1.9 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@main | 43.9 | 13.71 | 13.71 | 37.5 | - | - |
| tiny-16384-csa-cp8r7 | tilelang@cudnn_flashmla | 43.9 | 3.96 | 3.27 | 11.9 | 18.5 | 1.9 |
| tiny-16384-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 43.9 | 3.96 | 3.96 | 31.8 | 29.0 | 2.9 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 43.9 | 13.71 | 13.71 | 37.3 | - | - |
| tiny-16384-hca-cp1 | tilelang@main | 315.4 | 1.89 | 1.40 | 33.3 | 51.0 | 5.2 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@main | 315.4 | 3.05 | 3.05 | 76.6 | - | - |
| tiny-16384-hca-cp1 | tilelang@cudnn_flashmla | 315.4 | 1.89 | 1.40 | 32.9 | 50.9 | 5.1 |
| tiny-16384-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 315.4 | 1.89 | 1.89 | 71.4 | 70.4 | 7.1 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 315.4 | 3.05 | 3.05 | 76.4 | - | - |
| tiny-16384-hca-cp8r0 | tilelang@main | 40.7 | 1.86 | 1.39 | 11.9 | 19.0 | 1.9 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@main | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-hca-cp8r0 | tilelang@cudnn_flashmla | 40.7 | 1.86 | 1.39 | 12.0 | 18.7 | 1.9 |
| tiny-16384-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 40.7 | 1.86 | 1.86 | 32.4 | 27.3 | 2.8 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 40.7 | 2.95 | 2.95 | 39.7 | - | - |
| tiny-16384-hca-cp8r4 | tilelang@main | 39.1 | 1.88 | 1.40 | 11.5 | 18.7 | 1.9 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@main | 39.1 | 3.07 | 3.07 | 38.1 | - | - |
| tiny-16384-hca-cp8r4 | tilelang@cudnn_flashmla | 39.1 | 1.88 | 1.40 | 11.3 | 18.0 | 1.8 |
| tiny-16384-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 39.1 | 1.88 | 1.88 | 31.1 | 26.1 | 2.6 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 39.1 | 3.07 | 3.07 | 37.8 | - | - |
| tiny-16384-hca-cp8r7 | tilelang@main | 35.4 | 2.01 | 1.45 | 10.6 | 16.6 | 1.7 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@main | 35.4 | 3.40 | 3.40 | 34.3 | - | - |
| tiny-16384-hca-cp8r7 | tilelang@cudnn_flashmla | 35.4 | 2.01 | 1.45 | 10.4 | 16.5 | 1.7 |
| tiny-16384-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 35.4 | 2.01 | 2.01 | 28.1 | 23.8 | 2.4 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 35.4 | 3.40 | 3.40 | 33.8 | - | - |
| tiny-16384-sliding-cp1 | tilelang@main | 315.4 | 1.89 | 1.40 | 33.0 | 51.2 | 5.2 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@main | 315.4 | 3.05 | 3.05 | 76.3 | - | - |
| tiny-16384-sliding-cp1 | tilelang@cudnn_flashmla | 315.4 | 1.89 | 1.40 | 33.0 | 50.9 | 5.1 |
| tiny-16384-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 315.4 | 1.89 | 1.89 | 71.8 | 70.5 | 7.1 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 315.4 | 3.05 | 3.05 | 76.4 | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@main | 40.7 | 1.86 | 1.39 | 12.1 | 19.0 | 1.9 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@main | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-sliding-cp8r0 | tilelang@cudnn_flashmla | 40.7 | 1.86 | 1.39 | 11.8 | 18.6 | 1.9 |
| tiny-16384-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 40.7 | 1.86 | 1.86 | 32.3 | 26.9 | 2.7 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 40.7 | 2.95 | 2.95 | 38.9 | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@main | 39.1 | 1.88 | 1.40 | 11.6 | 18.0 | 1.8 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@main | 39.1 | 3.07 | 3.07 | 37.4 | - | - |
| tiny-16384-sliding-cp8r4 | tilelang@cudnn_flashmla | 39.1 | 1.88 | 1.40 | 11.5 | 18.3 | 1.8 |
| tiny-16384-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 39.1 | 1.88 | 1.88 | 31.4 | 26.4 | 2.7 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 39.1 | 3.07 | 3.07 | 38.2 | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@main | 35.4 | 2.01 | 1.45 | 10.5 | 16.5 | 1.7 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@main | 35.4 | 3.40 | 3.40 | 34.5 | - | - |
| tiny-16384-sliding-cp8r7 | tilelang@cudnn_flashmla | 35.4 | 2.01 | 1.45 | 10.5 | 16.5 | 1.7 |
| tiny-16384-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 35.4 | 2.01 | 2.01 | 28.3 | 23.5 | 2.4 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 35.4 | 3.40 | 3.40 | 34.4 | - | - |
| single-49208-csa-cp1 | tilelang@main | 14203.1 | 1.00 | 1.00 | 247.1 | 200.9 | 20.3 |
| single-49208-csa-cp1 | flashmla_fwd_ref@main | 14203.1 | 1.02 | 1.02 | 488.4 | - | - |
| single-49208-csa-cp1 | tilelang@cudnn_flashmla | 14203.1 | 1.00 | 1.00 | 246.2 | 200.0 | 20.2 |
| single-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 14203.1 | 1.00 | 1.00 | 483.9 | 360.1 | 36.4 |
| single-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 14203.1 | 1.02 | 1.02 | 475.1 | - | - |
| single-49208-csa-cp8r0 | tilelang@main | 1561.5 | 1.02 | 1.01 | 174.6 | 175.0 | 17.7 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@main | 1561.5 | 1.16 | 1.16 | 434.8 | - | - |
| single-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 1561.5 | 1.02 | 1.01 | 172.6 | 172.3 | 17.4 |
| single-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 1561.5 | 1.02 | 1.02 | 405.3 | 327.4 | 33.1 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 1561.5 | 1.16 | 1.16 | 436.5 | - | - |
| single-49208-csa-cp8r4 | tilelang@main | 1805.9 | 1.00 | 1.00 | 188.0 | 180.1 | 18.2 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1805.9 | 1.00 | 1.00 | 468.9 | - | - |
| single-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 1805.9 | 1.00 | 1.00 | 186.8 | 178.3 | 18.0 |
| single-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1805.9 | 1.00 | 1.00 | 438.4 | 344.9 | 34.9 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1805.9 | 1.00 | 1.00 | 466.6 | - | - |
| single-49208-csa-cp8r7 | tilelang@main | 1805.9 | 1.00 | 1.00 | 188.0 | 174.3 | 17.6 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1805.9 | 1.00 | 1.00 | 467.1 | - | - |
| single-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 1805.9 | 1.00 | 1.00 | 185.7 | 172.7 | 17.5 |
| single-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1805.9 | 1.00 | 1.00 | 439.3 | 340.0 | 34.4 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1805.9 | 1.00 | 1.00 | 467.0 | - | - |
| single-49208-hca-cp1 | tilelang@main | 7213.9 | 1.10 | 1.05 | 179.3 | 169.5 | 17.1 |
| single-49208-hca-cp1 | flashmla_fwd_ref@main | 7213.9 | 1.60 | 1.60 | 381.5 | - | - |
| single-49208-hca-cp1 | tilelang@cudnn_flashmla | 7213.9 | 1.10 | 1.05 | 179.3 | 168.5 | 17.0 |
| single-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 7213.9 | 1.10 | 1.10 | 373.1 | 288.3 | 29.1 |
| single-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 7213.9 | 1.60 | 1.60 | 367.0 | - | - |
| single-49208-hca-cp8r0 | tilelang@main | 423.9 | 1.26 | 1.12 | 70.8 | 99.8 | 10.1 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@main | 423.9 | 3.41 | 3.41 | 180.6 | - | - |
| single-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 423.9 | 1.26 | 1.12 | 70.0 | 99.0 | 10.0 |
| single-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 423.9 | 1.26 | 1.26 | 163.7 | 155.4 | 15.7 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 423.9 | 3.41 | 3.41 | 180.4 | - | - |
| single-49208-hca-cp8r4 | tilelang@main | 970.0 | 1.11 | 1.05 | 129.2 | 147.8 | 14.9 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@main | 970.0 | 1.49 | 1.49 | 345.9 | - | - |
| single-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 970.0 | 1.11 | 1.05 | 128.1 | 147.2 | 14.9 |
| single-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 970.0 | 1.11 | 1.11 | 320.0 | 267.7 | 27.1 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 970.0 | 1.49 | 1.49 | 346.2 | - | - |
| single-49208-hca-cp8r7 | tilelang@main | 1376.8 | 1.05 | 1.03 | 161.9 | 167.0 | 16.9 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@main | 1376.8 | 1.05 | 1.05 | 416.5 | - | - |
| single-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 1376.8 | 1.05 | 1.03 | 160.8 | 166.6 | 16.8 |
| single-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1376.8 | 1.05 | 1.05 | 388.9 | 315.5 | 31.9 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1376.8 | 1.05 | 1.05 | 416.9 | - | - |
| single-49208-sliding-cp1 | tilelang@main | 2885.8 | 1.00 | 1.00 | 110.9 | 126.1 | 12.7 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@main | 2885.8 | 1.00 | 1.00 | 263.8 | - | - |
| single-49208-sliding-cp1 | tilelang@cudnn_flashmla | 2885.8 | 1.00 | 1.00 | 111.0 | 126.0 | 12.7 |
| single-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 2885.8 | 1.00 | 1.00 | 255.8 | 187.6 | 19.0 |
| single-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 2885.8 | 1.00 | 1.00 | 264.7 | - | - |
| single-49208-sliding-cp8r0 | tilelang@main | 357.5 | 1.01 | 1.00 | 64.9 | 95.5 | 9.6 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 357.5 | 1.01 | 1.01 | 188.1 | - | - |
| single-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 357.5 | 1.01 | 1.00 | 64.0 | 94.8 | 9.6 |
| single-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 357.5 | 1.01 | 1.01 | 166.7 | 140.8 | 14.2 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 357.5 | 1.01 | 1.01 | 187.4 | - | - |
| single-49208-sliding-cp8r4 | tilelang@main | 361.2 | 1.00 | 1.00 | 66.3 | 95.5 | 9.6 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.4 | - | - |
| single-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 65.3 | 95.5 | 9.7 |
| single-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 170.4 | 142.6 | 14.4 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 191.1 | - | - |
| single-49208-sliding-cp8r7 | tilelang@main | 361.2 | 1.00 | 1.00 | 65.6 | 95.3 | 9.6 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.1 | - | - |
| single-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 64.4 | 93.9 | 9.5 |
| single-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 170.2 | 138.4 | 14.0 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 190.7 | - | - |
| short-49208-csa-cp1 | tilelang@main | 9266.5 | 1.07 | 1.04 | 203.8 | 181.4 | 18.3 |
| short-49208-csa-cp1 | flashmla_fwd_ref@main | 9266.5 | 1.56 | 1.56 | 425.2 | - | - |
| short-49208-csa-cp1 | tilelang@cudnn_flashmla | 9266.5 | 1.07 | 1.04 | 202.9 | 180.5 | 18.2 |
| short-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 9266.5 | 1.07 | 1.07 | 413.8 | 315.2 | 31.9 |
| short-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 9266.5 | 1.56 | 1.56 | 400.1 | - | - |
| short-49208-csa-cp8r0 | tilelang@main | 832.8 | 1.14 | 1.08 | 116.8 | 138.5 | 14.0 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@main | 832.8 | 2.17 | 2.17 | 300.8 | - | - |
| short-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 832.8 | 1.14 | 1.08 | 116.6 | 138.7 | 14.0 |
| short-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 832.8 | 1.14 | 1.14 | 276.7 | 245.8 | 24.8 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 832.8 | 2.17 | 2.17 | 301.1 | - | - |
| short-49208-csa-cp8r4 | tilelang@main | 1351.2 | 1.04 | 1.02 | 161.8 | 167.1 | 16.9 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1351.2 | 1.34 | 1.34 | 402.3 | - | - |
| short-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 1351.2 | 1.04 | 1.02 | 161.1 | 165.8 | 16.8 |
| short-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1351.2 | 1.04 | 1.04 | 376.9 | 311.7 | 31.5 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1351.2 | 1.34 | 1.34 | 404.4 | - | - |
| short-49208-csa-cp8r7 | tilelang@main | 1017.3 | 1.09 | 1.05 | 134.5 | 149.4 | 15.1 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1017.3 | 1.78 | 1.78 | 338.6 | - | - |
| short-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 1017.3 | 1.09 | 1.05 | 133.3 | 149.9 | 15.2 |
| short-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1017.3 | 1.09 | 1.09 | 316.3 | 272.0 | 27.5 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1017.3 | 1.78 | 1.78 | 339.4 | - | - |
| short-49208-hca-cp1 | tilelang@main | 3131.2 | 1.36 | 1.16 | 105.2 | 119.6 | 12.1 |
| short-49208-hca-cp1 | flashmla_fwd_ref@main | 3131.2 | 1.85 | 1.85 | 218.0 | - | - |
| short-49208-hca-cp1 | tilelang@cudnn_flashmla | 3131.2 | 1.36 | 1.16 | 105.4 | 119.6 | 12.1 |
| short-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 3131.2 | 1.36 | 1.36 | 214.2 | 179.9 | 18.2 |
| short-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3131.2 | 1.85 | 1.85 | 218.4 | - | - |
| short-49208-hca-cp8r0 | tilelang@main | 352.9 | 1.44 | 1.20 | 59.3 | 86.7 | 8.8 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@main | 352.9 | 2.05 | 2.05 | 155.9 | - | - |
| short-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 352.9 | 1.44 | 1.20 | 57.5 | 86.3 | 8.7 |
| short-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 352.9 | 1.44 | 1.44 | 139.7 | 135.2 | 13.7 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 352.9 | 2.05 | 2.05 | 154.3 | - | - |
| short-49208-hca-cp8r4 | tilelang@main | 398.0 | 1.33 | 1.14 | 65.1 | 95.1 | 9.6 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@main | 398.0 | 1.82 | 1.82 | 170.9 | - | - |
| short-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 398.0 | 1.33 | 1.14 | 64.9 | 94.2 | 9.5 |
| short-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 398.0 | 1.33 | 1.33 | 154.6 | 147.9 | 14.9 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 398.0 | 1.82 | 1.82 | 170.7 | - | - |
| short-49208-hca-cp8r7 | tilelang@main | 369.7 | 1.42 | 1.18 | 61.0 | 88.7 | 9.0 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@main | 369.7 | 1.95 | 1.95 | 159.0 | - | - |
| short-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 369.7 | 1.42 | 1.18 | 60.4 | 89.1 | 9.0 |
| short-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 369.7 | 1.42 | 1.42 | 143.4 | 140.6 | 14.2 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 369.7 | 1.95 | 1.95 | 157.9 | - | - |
| short-49208-sliding-cp1 | tilelang@main | 2785.1 | 1.02 | 1.01 | 107.2 | 123.1 | 12.4 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@main | 2785.1 | 1.04 | 1.04 | 254.8 | - | - |
| short-49208-sliding-cp1 | tilelang@cudnn_flashmla | 2785.1 | 1.02 | 1.01 | 107.2 | 123.1 | 12.4 |
| short-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 2785.1 | 1.02 | 1.02 | 246.9 | 183.0 | 18.5 |
| short-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 2785.1 | 1.04 | 1.04 | 254.9 | - | - |
| short-49208-sliding-cp8r0 | tilelang@main | 338.8 | 1.03 | 1.02 | 61.9 | 92.5 | 9.3 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 338.8 | 1.07 | 1.07 | 178.9 | - | - |
| short-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 338.8 | 1.03 | 1.02 | 61.5 | 91.9 | 9.3 |
| short-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 338.8 | 1.03 | 1.03 | 158.5 | 135.3 | 13.7 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 338.8 | 1.07 | 1.07 | 177.0 | - | - |
| short-49208-sliding-cp8r4 | tilelang@main | 353.7 | 1.01 | 1.01 | 64.4 | 94.6 | 9.6 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 353.7 | 1.02 | 1.02 | 185.2 | - | - |
| short-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 353.7 | 1.01 | 1.01 | 64.6 | 94.7 | 9.6 |
| short-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 353.7 | 1.01 | 1.01 | 165.3 | 140.4 | 14.2 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 353.7 | 1.02 | 1.02 | 185.3 | - | - |
| short-49208-sliding-cp8r7 | tilelang@main | 350.0 | 1.02 | 1.01 | 64.3 | 93.5 | 9.4 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 350.0 | 1.03 | 1.03 | 186.3 | - | - |
| short-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 350.0 | 1.02 | 1.01 | 63.0 | 93.7 | 9.5 |
| short-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 350.0 | 1.02 | 1.02 | 164.4 | 138.3 | 14.0 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 350.0 | 1.03 | 1.03 | 183.5 | - | - |
| heavy-49208-csa-cp1 | tilelang@main | 11959.6 | 1.03 | 1.02 | 229.1 | 193.8 | 19.6 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@main | 11959.6 | 1.21 | 1.21 | 462.2 | - | - |
| heavy-49208-csa-cp1 | tilelang@cudnn_flashmla | 11959.6 | 1.03 | 1.02 | 229.0 | 192.8 | 19.5 |
| heavy-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 11959.6 | 1.03 | 1.03 | 455.8 | 345.3 | 34.9 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 11959.6 | 1.21 | 1.21 | 446.0 | - | - |
| heavy-49208-csa-cp8r0 | tilelang@main | 663.7 | 1.22 | 1.14 | 98.5 | 122.4 | 12.4 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@main | 663.7 | 2.72 | 2.72 | 249.6 | - | - |
| heavy-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 663.7 | 1.22 | 1.14 | 97.9 | 122.6 | 12.4 |
| heavy-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 663.7 | 1.22 | 1.22 | 231.6 | 212.0 | 21.4 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 663.7 | 2.72 | 2.72 | 252.2 | - | - |
| heavy-49208-csa-cp8r4 | tilelang@main | 1663.9 | 1.01 | 1.01 | 180.8 | 177.6 | 17.9 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@main | 1663.9 | 1.09 | 1.09 | 449.3 | - | - |
| heavy-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 1663.9 | 1.01 | 1.01 | 180.4 | 175.5 | 17.7 |
| heavy-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1663.9 | 1.01 | 1.01 | 421.0 | 336.5 | 34.0 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1663.9 | 1.09 | 1.09 | 448.7 | - | - |
| heavy-49208-csa-cp8r7 | tilelang@main | 1454.9 | 1.03 | 1.02 | 167.8 | 169.5 | 17.1 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@main | 1454.9 | 1.24 | 1.24 | 419.7 | - | - |
| heavy-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 1454.9 | 1.03 | 1.02 | 167.5 | 168.2 | 17.0 |
| heavy-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1454.9 | 1.03 | 1.03 | 392.9 | 320.0 | 32.3 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1454.9 | 1.24 | 1.24 | 421.7 | - | - |
| heavy-49208-hca-cp1 | tilelang@main | 4077.2 | 1.21 | 1.09 | 128.2 | 137.7 | 13.9 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@main | 4077.2 | 2.13 | 2.13 | 272.6 | - | - |
| heavy-49208-hca-cp1 | tilelang@cudnn_flashmla | 4077.2 | 1.21 | 1.09 | 128.4 | 137.4 | 13.9 |
| heavy-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 4077.2 | 1.21 | 1.21 | 263.8 | 214.5 | 21.7 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 4077.2 | 2.13 | 2.13 | 273.0 | - | - |
| heavy-49208-hca-cp8r0 | tilelang@main | 318.8 | 1.45 | 1.21 | 53.5 | 81.5 | 8.2 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@main | 318.8 | 3.40 | 3.40 | 141.7 | - | - |
| heavy-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 318.8 | 1.45 | 1.21 | 53.6 | 81.8 | 8.3 |
| heavy-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 318.8 | 1.45 | 1.45 | 129.8 | 125.7 | 12.7 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 318.8 | 3.40 | 3.40 | 143.3 | - | - |
| heavy-49208-hca-cp8r4 | tilelang@main | 438.1 | 1.24 | 1.11 | 71.4 | 99.7 | 10.1 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@main | 438.1 | 2.47 | 2.47 | 185.4 | - | - |
| heavy-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 438.1 | 1.24 | 1.11 | 70.7 | 99.2 | 10.0 |
| heavy-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 438.1 | 1.24 | 1.24 | 168.2 | 158.6 | 16.0 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 438.1 | 2.47 | 2.47 | 186.9 | - | - |
| heavy-49208-hca-cp8r7 | tilelang@main | 507.2 | 1.25 | 1.08 | 78.6 | 108.5 | 11.0 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@main | 507.2 | 2.14 | 2.14 | 203.5 | - | - |
| heavy-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 507.2 | 1.25 | 1.08 | 78.6 | 108.6 | 11.0 |
| heavy-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 507.2 | 1.25 | 1.25 | 186.2 | 173.5 | 17.5 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 507.2 | 2.14 | 2.14 | 203.5 | - | - |
| heavy-49208-sliding-cp1 | tilelang@main | 2781.4 | 1.02 | 1.01 | 106.7 | 123.2 | 12.4 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@main | 2781.4 | 1.04 | 1.04 | 253.5 | - | - |
| heavy-49208-sliding-cp1 | tilelang@cudnn_flashmla | 2781.4 | 1.02 | 1.01 | 107.2 | 123.1 | 12.4 |
| heavy-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 2781.4 | 1.02 | 1.02 | 246.4 | 182.8 | 18.5 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 2781.4 | 1.04 | 1.04 | 255.1 | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@main | 309.0 | 1.08 | 1.04 | 56.2 | 86.1 | 8.7 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 309.0 | 1.17 | 1.17 | 162.8 | - | - |
| heavy-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 309.0 | 1.08 | 1.04 | 55.6 | 85.6 | 8.6 |
| heavy-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 309.0 | 1.08 | 1.08 | 144.0 | 125.4 | 12.7 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 309.0 | 1.17 | 1.17 | 162.9 | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@main | 361.2 | 1.00 | 1.00 | 65.9 | 96.0 | 9.7 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 361.2 | 1.00 | 1.00 | 191.5 | - | - |
| heavy-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 65.9 | 95.5 | 9.6 |
| heavy-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 171.6 | 143.4 | 14.5 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 361.2 | 1.00 | 1.00 | 191.0 | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@main | 353.7 | 1.01 | 1.01 | 64.5 | 93.9 | 9.5 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 353.7 | 1.02 | 1.02 | 186.1 | - | - |
| heavy-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 353.7 | 1.01 | 1.01 | 64.3 | 93.8 | 9.5 |
| heavy-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 353.7 | 1.01 | 1.01 | 168.1 | 139.5 | 14.1 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 353.7 | 1.02 | 1.02 | 189.0 | - | - |
| tiny-49208-csa-cp1 | tilelang@main | 1202.7 | 3.49 | 2.89 | 40.9 | 49.9 | 5.0 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@main | 1202.7 | 12.01 | 12.01 | 80.0 | - | - |
| tiny-49208-csa-cp1 | tilelang@cudnn_flashmla | 1202.7 | 3.49 | 2.89 | 41.0 | 49.8 | 5.0 |
| tiny-49208-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1202.7 | 3.49 | 3.49 | 79.2 | 73.9 | 7.5 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1202.7 | 12.01 | 12.01 | 80.4 | - | - |
| tiny-49208-csa-cp8r0 | tilelang@main | 150.2 | 3.50 | 2.89 | 25.2 | 38.8 | 3.9 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@main | 150.2 | 12.03 | 12.03 | 65.2 | - | - |
| tiny-49208-csa-cp8r0 | tilelang@cudnn_flashmla | 150.2 | 3.50 | 2.89 | 25.4 | 38.8 | 3.9 |
| tiny-49208-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 150.2 | 3.50 | 3.50 | 59.5 | 60.7 | 6.1 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 150.2 | 12.03 | 12.03 | 65.6 | - | - |
| tiny-49208-csa-cp8r4 | tilelang@main | 156.7 | 3.35 | 2.78 | 26.6 | 40.1 | 4.1 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@main | 156.7 | 11.53 | 11.53 | 68.2 | - | - |
| tiny-49208-csa-cp8r4 | tilelang@cudnn_flashmla | 156.7 | 3.35 | 2.78 | 26.4 | 40.1 | 4.1 |
| tiny-49208-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 156.7 | 3.35 | 3.35 | 62.0 | 62.6 | 6.3 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 156.7 | 11.53 | 11.53 | 68.2 | - | - |
| tiny-49208-csa-cp8r7 | tilelang@main | 144.3 | 3.63 | 3.01 | 24.4 | 36.9 | 3.7 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@main | 144.3 | 12.51 | 12.51 | 62.5 | - | - |
| tiny-49208-csa-cp8r7 | tilelang@cudnn_flashmla | 144.3 | 3.63 | 3.01 | 24.1 | 37.1 | 3.7 |
| tiny-49208-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 144.3 | 3.63 | 3.63 | 56.4 | 58.6 | 5.9 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 144.3 | 12.51 | 12.51 | 62.5 | - | - |
| tiny-49208-hca-cp1 | tilelang@main | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@main | 969.0 | 2.98 | 2.98 | 87.5 | - | - |
| tiny-49208-hca-cp1 | tilelang@cudnn_flashmla | 969.0 | 1.86 | 1.39 | 41.2 | 57.7 | 5.8 |
| tiny-49208-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 969.0 | 1.86 | 1.86 | 84.7 | 75.7 | 7.7 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 969.0 | 2.98 | 2.98 | 87.5 | - | - |
| tiny-49208-hca-cp8r0 | tilelang@main | 121.0 | 1.86 | 1.39 | 23.4 | 41.0 | 4.1 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@main | 121.0 | 2.99 | 2.99 | 63.8 | - | - |
| tiny-49208-hca-cp8r0 | tilelang@cudnn_flashmla | 121.0 | 1.86 | 1.39 | 22.8 | 40.7 | 4.1 |
| tiny-49208-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 121.0 | 1.86 | 1.86 | 56.1 | 55.3 | 5.6 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 121.0 | 2.99 | 2.99 | 63.0 | - | - |
| tiny-49208-hca-cp8r4 | tilelang@main | 126.2 | 1.83 | 1.38 | 24.3 | 42.3 | 4.3 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@main | 126.2 | 2.86 | 2.86 | 66.3 | - | - |
| tiny-49208-hca-cp8r4 | tilelang@cudnn_flashmla | 126.2 | 1.83 | 1.38 | 24.3 | 41.8 | 4.2 |
| tiny-49208-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 126.2 | 1.83 | 1.83 | 59.9 | 56.7 | 5.7 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 126.2 | 2.86 | 2.86 | 67.0 | - | - |
| tiny-49208-hca-cp8r7 | tilelang@main | 116.3 | 1.90 | 1.41 | 22.6 | 39.4 | 4.0 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@main | 116.3 | 3.11 | 3.11 | 61.5 | - | - |
| tiny-49208-hca-cp8r7 | tilelang@cudnn_flashmla | 116.3 | 1.90 | 1.41 | 22.3 | 39.4 | 4.0 |
| tiny-49208-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 116.3 | 1.90 | 1.90 | 54.7 | 53.2 | 5.4 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 116.3 | 3.11 | 3.11 | 61.3 | - | - |
| tiny-49208-sliding-cp1 | tilelang@main | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@main | 969.0 | 2.98 | 2.98 | 87.2 | - | - |
| tiny-49208-sliding-cp1 | tilelang@cudnn_flashmla | 969.0 | 1.86 | 1.39 | 41.2 | 57.8 | 5.8 |
| tiny-49208-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 969.0 | 1.86 | 1.86 | 84.4 | 75.7 | 7.7 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 969.0 | 2.98 | 2.98 | 87.4 | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@main | 121.0 | 1.86 | 1.39 | 23.6 | 41.1 | 4.1 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@main | 121.0 | 2.99 | 2.99 | 64.0 | - | - |
| tiny-49208-sliding-cp8r0 | tilelang@cudnn_flashmla | 121.0 | 1.86 | 1.39 | 23.4 | 40.8 | 4.1 |
| tiny-49208-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 121.0 | 1.86 | 1.86 | 56.2 | 54.7 | 5.5 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 121.0 | 2.99 | 2.99 | 63.1 | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@main | 126.2 | 1.83 | 1.38 | 24.3 | 42.4 | 4.3 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@main | 126.2 | 2.86 | 2.86 | 66.5 | - | - |
| tiny-49208-sliding-cp8r4 | tilelang@cudnn_flashmla | 126.2 | 1.83 | 1.38 | 24.1 | 42.1 | 4.3 |
| tiny-49208-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 126.2 | 1.83 | 1.83 | 58.9 | 57.1 | 5.8 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 126.2 | 2.86 | 2.86 | 66.0 | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@main | 116.3 | 1.90 | 1.41 | 22.7 | 39.4 | 4.0 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@main | 116.3 | 3.11 | 3.11 | 61.3 | - | - |
| tiny-49208-sliding-cp8r7 | tilelang@cudnn_flashmla | 116.3 | 1.90 | 1.41 | 22.4 | 39.1 | 3.9 |
| tiny-49208-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 116.3 | 1.90 | 1.90 | 54.4 | 52.7 | 5.3 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 116.3 | 3.11 | 3.11 | 60.9 | - | - |
| single-65536-csa-cp1 | tilelang@main | 18997.0 | 1.00 | 1.00 | 250.2 | 199.0 | 20.1 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 18997.0 | 1.01 | 1.01 | 476.7 | - | - |
| single-65536-csa-cp1 | tilelang@cudnn_flashmla | 18997.0 | 1.00 | 1.00 | 249.9 | 198.9 | 20.1 |
| single-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 18997.0 | 1.00 | 1.00 | 474.4 | 355.6 | 35.9 |
| single-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 18997.0 | 1.01 | 1.01 | 473.2 | - | - |
| single-65536-csa-cp8r0 | tilelang@main | 2160.7 | 1.02 | 1.01 | 189.1 | 181.1 | 18.3 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@main | 2160.7 | 1.11 | 1.11 | 465.9 | - | - |
| single-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 2160.7 | 1.02 | 1.01 | 187.5 | 179.2 | 18.1 |
| single-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 2160.7 | 1.02 | 1.02 | 440.2 | 342.6 | 34.6 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 2160.7 | 1.11 | 1.11 | 464.6 | - | - |
| single-65536-csa-cp8r4 | tilelang@main | 2405.2 | 1.00 | 1.00 | 200.4 | 183.0 | 18.5 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 487.2 | - | - |
| single-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 199.7 | 180.9 | 18.3 |
| single-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 464.3 | 349.3 | 35.3 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 488.3 | - | - |
| single-65536-csa-cp8r7 | tilelang@main | 2405.2 | 1.00 | 1.00 | 203.1 | 171.9 | 17.4 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 487.3 | - | - |
| single-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 198.3 | 170.2 | 17.2 |
| single-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 453.8 | 333.4 | 33.7 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 484.6 | - | - |
| single-65536-hca-cp1 | tilelang@main | 11526.3 | 1.08 | 1.04 | 199.6 | 179.2 | 18.1 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 11526.3 | 1.67 | 1.67 | 401.8 | - | - |
| single-65536-hca-cp1 | tilelang@cudnn_flashmla | 11526.3 | 1.08 | 1.04 | 198.8 | 178.5 | 18.0 |
| single-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 11526.3 | 1.08 | 1.08 | 394.4 | 307.9 | 31.1 |
| single-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 11526.3 | 1.67 | 1.67 | 390.9 | - | - |
| single-65536-hca-cp8r0 | tilelang@main | 595.7 | 1.20 | 1.10 | 82.1 | 109.6 | 11.1 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@main | 595.7 | 4.04 | 4.04 | 203.8 | - | - |
| single-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 595.7 | 1.20 | 1.10 | 83.2 | 109.9 | 11.1 |
| single-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 595.7 | 1.20 | 1.20 | 188.8 | 174.7 | 17.7 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 595.7 | 4.04 | 4.04 | 202.2 | - | - |
| single-65536-hca-cp8r4 | tilelang@main | 1561.5 | 1.08 | 1.04 | 159.5 | 163.0 | 16.5 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@main | 1561.5 | 1.54 | 1.54 | 376.5 | - | - |
| single-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 1561.5 | 1.08 | 1.04 | 159.5 | 161.6 | 16.3 |
| single-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1561.5 | 1.08 | 1.08 | 354.5 | 295.9 | 29.9 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1561.5 | 1.54 | 1.54 | 375.2 | - | - |
| single-65536-hca-cp8r7 | tilelang@main | 2283.1 | 1.05 | 1.03 | 192.4 | 179.9 | 18.2 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@main | 2283.1 | 1.05 | 1.05 | 471.2 | - | - |
| single-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 2283.1 | 1.05 | 1.03 | 190.9 | 178.7 | 18.1 |
| single-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 2283.1 | 1.05 | 1.05 | 447.8 | 345.4 | 34.9 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 2283.1 | 1.05 | 1.05 | 472.4 | - | - |
| single-65536-sliding-cp1 | tilelang@main | 3844.6 | 1.00 | 1.00 | 113.8 | 127.5 | 12.9 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 3844.6 | 1.00 | 1.00 | 266.5 | - | - |
| single-65536-sliding-cp1 | tilelang@cudnn_flashmla | 3844.6 | 1.00 | 1.00 | 113.7 | 127.6 | 12.9 |
| single-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3844.6 | 1.00 | 1.00 | 260.9 | 189.6 | 19.2 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3844.6 | 1.00 | 1.00 | 268.1 | - | - |
| single-65536-sliding-cp8r0 | tilelang@main | 477.3 | 1.00 | 1.00 | 72.9 | 102.5 | 10.4 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 477.3 | 1.01 | 1.01 | 206.2 | - | - |
| single-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 477.3 | 1.00 | 1.00 | 73.4 | 101.8 | 10.3 |
| single-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 477.3 | 1.00 | 1.00 | 185.0 | 156.7 | 15.8 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 477.3 | 1.01 | 1.01 | 205.1 | - | - |
| single-65536-sliding-cp8r4 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.6 | 102.5 | 10.4 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 208.0 | - | - |
| single-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 73.7 | 102.0 | 10.3 |
| single-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 188.2 | 158.9 | 16.1 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 208.2 | - | - |
| single-65536-sliding-cp8r7 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.2 | 102.0 | 10.3 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 207.5 | - | - |
| single-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 73.2 | 101.4 | 10.2 |
| single-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 187.7 | 157.5 | 15.9 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 208.0 | - | - |
| short-65536-csa-cp1 | tilelang@main | 11209.9 | 1.08 | 1.05 | 197.2 | 177.5 | 17.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 11209.9 | 1.72 | 1.72 | 393.2 | - | - |
| short-65536-csa-cp1 | tilelang@cudnn_flashmla | 11209.9 | 1.08 | 1.05 | 197.0 | 176.9 | 17.9 |
| short-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 11209.9 | 1.08 | 1.08 | 389.4 | 304.5 | 30.8 |
| short-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 11209.9 | 1.72 | 1.72 | 387.2 | - | - |
| short-65536-csa-cp8r0 | tilelang@main | 885.0 | 1.18 | 1.11 | 110.6 | 130.5 | 13.2 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@main | 885.0 | 2.72 | 2.72 | 278.5 | - | - |
| short-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 885.0 | 1.18 | 1.11 | 109.9 | 130.3 | 13.2 |
| short-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 885.0 | 1.18 | 1.18 | 256.6 | 223.5 | 22.6 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 885.0 | 2.72 | 2.72 | 275.4 | - | - |
| short-65536-csa-cp8r4 | tilelang@main | 1689.8 | 1.05 | 1.03 | 168.7 | 167.7 | 16.9 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@main | 1689.8 | 1.42 | 1.42 | 412.8 | - | - |
| short-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 1689.8 | 1.05 | 1.03 | 166.9 | 166.1 | 16.8 |
| short-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 1689.8 | 1.05 | 1.05 | 387.3 | 310.8 | 31.4 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 1689.8 | 1.42 | 1.42 | 410.6 | - | - |
| short-65536-csa-cp8r7 | tilelang@main | 1414.7 | 1.08 | 1.05 | 151.7 | 158.2 | 16.0 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1414.7 | 1.70 | 1.70 | 371.4 | - | - |
| short-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 1414.7 | 1.08 | 1.05 | 151.1 | 158.2 | 16.0 |
| short-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1414.7 | 1.08 | 1.08 | 350.4 | 287.6 | 29.1 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1414.7 | 1.70 | 1.70 | 370.9 | - | - |
| short-65536-hca-cp1 | tilelang@main | 3930.2 | 1.40 | 1.17 | 102.8 | 117.2 | 11.8 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 3930.2 | 1.96 | 1.96 | 208.8 | - | - |
| short-65536-hca-cp1 | tilelang@cudnn_flashmla | 3930.2 | 1.40 | 1.17 | 102.5 | 117.0 | 11.8 |
| short-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 3930.2 | 1.40 | 1.40 | 205.2 | 173.2 | 17.5 |
| short-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3930.2 | 1.96 | 1.96 | 209.0 | - | - |
| short-65536-hca-cp8r0 | tilelang@main | 455.7 | 1.46 | 1.22 | 64.5 | 90.7 | 9.2 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@main | 455.7 | 2.11 | 2.11 | 160.5 | - | - |
| short-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 455.7 | 1.46 | 1.22 | 63.8 | 89.9 | 9.1 |
| short-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 455.7 | 1.46 | 1.46 | 149.2 | 142.6 | 14.4 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 455.7 | 2.11 | 2.11 | 160.9 | - | - |
| short-65536-hca-cp8r4 | tilelang@main | 514.5 | 1.37 | 1.14 | 71.6 | 99.0 | 10.0 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@main | 514.5 | 1.87 | 1.87 | 176.6 | - | - |
| short-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 514.5 | 1.37 | 1.14 | 70.9 | 98.1 | 9.9 |
| short-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 514.5 | 1.37 | 1.37 | 164.0 | 155.0 | 15.7 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 514.5 | 1.87 | 1.87 | 177.4 | - | - |
| short-65536-hca-cp8r7 | tilelang@main | 491.0 | 1.40 | 1.17 | 68.7 | 95.0 | 9.6 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@main | 491.0 | 1.96 | 1.96 | 171.1 | - | - |
| short-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 491.0 | 1.40 | 1.17 | 67.8 | 95.0 | 9.6 |
| short-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 491.0 | 1.40 | 1.40 | 156.8 | 150.4 | 15.2 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 491.0 | 1.96 | 1.96 | 168.7 | - | - |
| short-65536-sliding-cp1 | tilelang@main | 3673.0 | 1.02 | 1.01 | 109.2 | 123.9 | 12.5 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 3673.0 | 1.05 | 1.05 | 255.6 | - | - |
| short-65536-sliding-cp1 | tilelang@cudnn_flashmla | 3673.0 | 1.02 | 1.01 | 108.7 | 123.7 | 12.5 |
| short-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3673.0 | 1.02 | 1.02 | 248.2 | 183.3 | 18.5 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3673.0 | 1.05 | 1.05 | 255.6 | - | - |
| short-65536-sliding-cp8r0 | tilelang@main | 443.7 | 1.04 | 1.02 | 68.1 | 97.4 | 9.8 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 443.7 | 1.08 | 1.08 | 190.4 | - | - |
| short-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 443.7 | 1.04 | 1.02 | 68.5 | 96.6 | 9.8 |
| short-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 443.7 | 1.04 | 1.04 | 173.7 | 148.0 | 15.0 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 443.7 | 1.08 | 1.08 | 189.8 | - | - |
| short-65536-sliding-cp8r4 | tilelang@main | 469.9 | 1.01 | 1.01 | 73.0 | 101.6 | 10.3 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 469.9 | 1.02 | 1.02 | 203.3 | - | - |
| short-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 469.9 | 1.01 | 1.01 | 72.5 | 101.2 | 10.2 |
| short-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 469.9 | 1.01 | 1.01 | 183.9 | 156.5 | 15.8 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 469.9 | 1.02 | 1.02 | 203.4 | - | - |
| short-65536-sliding-cp8r7 | tilelang@main | 458.7 | 1.02 | 1.01 | 71.2 | 99.0 | 10.0 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 458.7 | 1.05 | 1.05 | 197.9 | - | - |
| short-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 458.7 | 1.02 | 1.01 | 71.1 | 98.7 | 10.0 |
| short-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 458.7 | 1.02 | 1.02 | 179.2 | 152.2 | 15.4 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 458.7 | 1.05 | 1.05 | 199.1 | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 12785.2 | 1.07 | 1.04 | 210.3 | 183.4 | 18.5 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 12785.2 | 1.50 | 1.50 | 414.1 | - | - |
| heavy-65536-csa-cp1 | tilelang@cudnn_flashmla | 12785.2 | 1.07 | 1.04 | 209.2 | 182.8 | 18.5 |
| heavy-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 12785.2 | 1.07 | 1.07 | 408.3 | 319.8 | 32.3 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 12785.2 | 1.50 | 1.50 | 404.0 | - | - |
| heavy-65536-csa-cp8r0 | tilelang@main | 771.9 | 1.27 | 1.18 | 98.5 | 120.2 | 12.2 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@main | 771.9 | 3.12 | 3.12 | 244.8 | - | - |
| heavy-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 771.9 | 1.27 | 1.18 | 98.3 | 120.1 | 12.1 |
| heavy-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 771.9 | 1.27 | 1.27 | 227.4 | 202.8 | 20.5 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 771.9 | 3.12 | 3.12 | 244.7 | - | - |
| heavy-65536-csa-cp8r4 | tilelang@main | 2405.2 | 1.00 | 1.00 | 202.0 | 185.4 | 18.7 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@main | 2405.2 | 1.00 | 1.00 | 490.2 | - | - |
| heavy-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 198.4 | 182.9 | 18.5 |
| heavy-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 453.7 | 351.8 | 35.6 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 2405.2 | 1.00 | 1.00 | 488.6 | - | - |
| heavy-65536-csa-cp8r7 | tilelang@main | 1250.3 | 1.12 | 1.08 | 140.1 | 150.7 | 15.2 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@main | 1250.3 | 1.92 | 1.92 | 341.0 | - | - |
| heavy-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 1250.3 | 1.12 | 1.08 | 138.1 | 150.6 | 15.2 |
| heavy-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 1250.3 | 1.12 | 1.12 | 316.6 | 268.2 | 27.1 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 1250.3 | 1.92 | 1.92 | 341.4 | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 5141.1 | 1.25 | 1.11 | 125.4 | 134.3 | 13.6 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5141.1 | 2.25 | 2.25 | 258.5 | - | - |
| heavy-65536-hca-cp1 | tilelang@cudnn_flashmla | 5141.1 | 1.25 | 1.11 | 125.0 | 134.1 | 13.6 |
| heavy-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 5141.1 | 1.25 | 1.25 | 250.3 | 206.3 | 20.8 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 5141.1 | 2.25 | 2.25 | 257.7 | - | - |
| heavy-65536-hca-cp8r0 | tilelang@main | 408.9 | 1.46 | 1.22 | 58.7 | 84.8 | 8.6 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@main | 408.9 | 3.53 | 3.53 | 150.4 | - | - |
| heavy-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 408.9 | 1.46 | 1.22 | 58.9 | 84.6 | 8.5 |
| heavy-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 408.9 | 1.46 | 1.46 | 138.1 | 131.8 | 13.3 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 408.9 | 3.53 | 3.53 | 150.0 | - | - |
| heavy-65536-hca-cp8r4 | tilelang@main | 965.4 | 1.12 | 1.06 | 116.0 | 136.9 | 13.8 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@main | 965.4 | 1.49 | 1.49 | 297.2 | - | - |
| heavy-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 965.4 | 1.12 | 1.06 | 115.7 | 136.3 | 13.8 |
| heavy-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 965.4 | 1.12 | 1.12 | 279.0 | 237.1 | 24.0 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 965.4 | 1.49 | 1.49 | 297.3 | - | - |
| heavy-65536-hca-cp8r7 | tilelang@main | 458.3 | 1.40 | 1.17 | 65.3 | 91.5 | 9.2 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@main | 458.3 | 3.15 | 3.15 | 163.9 | - | - |
| heavy-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 458.3 | 1.40 | 1.17 | 64.8 | 91.6 | 9.3 |
| heavy-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 458.3 | 1.40 | 1.40 | 151.3 | 142.5 | 14.4 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 458.3 | 3.15 | 3.15 | 162.8 | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 3542.5 | 1.04 | 1.02 | 105.3 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 3542.5 | 1.09 | 1.09 | 246.2 | - | - |
| heavy-65536-sliding-cp1 | tilelang@cudnn_flashmla | 3542.5 | 1.04 | 1.02 | 105.5 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 3542.5 | 1.04 | 1.04 | 239.9 | 178.4 | 18.0 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 3542.5 | 1.09 | 1.09 | 246.5 | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@main | 399.0 | 1.10 | 1.05 | 61.5 | 90.6 | 9.2 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 399.0 | 1.21 | 1.21 | 170.2 | - | - |
| heavy-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 399.0 | 1.10 | 1.05 | 62.1 | 89.8 | 9.1 |
| heavy-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 399.0 | 1.10 | 1.10 | 155.8 | 136.9 | 13.8 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 399.0 | 1.21 | 1.21 | 171.7 | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@main | 481.0 | 1.00 | 1.00 | 74.7 | 102.8 | 10.4 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 481.0 | 1.00 | 1.00 | 209.5 | - | - |
| heavy-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 74.3 | 102.1 | 10.3 |
| heavy-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 188.2 | 159.1 | 16.1 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 481.0 | 1.00 | 1.00 | 208.4 | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@main | 428.8 | 1.06 | 1.03 | 67.2 | 94.8 | 9.6 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 428.8 | 1.12 | 1.12 | 184.1 | - | - |
| heavy-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 428.8 | 1.06 | 1.03 | 66.2 | 94.2 | 9.5 |
| heavy-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 428.8 | 1.06 | 1.06 | 168.9 | 144.9 | 14.6 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 428.8 | 1.12 | 1.12 | 185.7 | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 1621.2 | 3.45 | 2.86 | 42.6 | 51.0 | 5.2 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 1621.2 | 11.87 | 11.87 | 81.8 | - | - |
| tiny-65536-csa-cp1 | tilelang@cudnn_flashmla | 1621.2 | 3.45 | 2.86 | 42.5 | 51.0 | 5.1 |
| tiny-65536-csa-cp1 | cudnn_flashmla@cudnn_flashmla | 1621.2 | 3.45 | 3.45 | 80.7 | 75.3 | 7.6 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1621.2 | 11.87 | 11.87 | 81.9 | - | - |
| tiny-65536-csa-cp8r0 | tilelang@main | 202.8 | 3.45 | 2.86 | 28.9 | 41.5 | 4.2 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@main | 202.8 | 11.86 | 11.86 | 71.6 | - | - |
| tiny-65536-csa-cp8r0 | tilelang@cudnn_flashmla | 202.8 | 3.45 | 2.86 | 28.9 | 41.7 | 4.2 |
| tiny-65536-csa-cp8r0 | cudnn_flashmla@cudnn_flashmla | 202.8 | 3.45 | 3.45 | 66.2 | 65.8 | 6.6 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 202.8 | 11.86 | 11.86 | 71.2 | - | - |
| tiny-65536-csa-cp8r4 | tilelang@main | 202.7 | 3.45 | 2.86 | 28.8 | 41.6 | 4.2 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@main | 202.7 | 11.87 | 11.87 | 70.3 | - | - |
| tiny-65536-csa-cp8r4 | tilelang@cudnn_flashmla | 202.7 | 3.45 | 2.86 | 28.9 | 41.4 | 4.2 |
| tiny-65536-csa-cp8r4 | cudnn_flashmla@cudnn_flashmla | 202.7 | 3.45 | 3.45 | 65.3 | 65.4 | 6.6 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 202.7 | 11.87 | 11.87 | 70.8 | - | - |
| tiny-65536-csa-cp8r7 | tilelang@main | 201.4 | 3.47 | 2.88 | 27.9 | 41.3 | 4.2 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@main | 201.4 | 11.94 | 11.94 | 69.2 | - | - |
| tiny-65536-csa-cp8r7 | tilelang@cudnn_flashmla | 201.4 | 3.47 | 2.88 | 28.8 | 41.1 | 4.2 |
| tiny-65536-csa-cp8r7 | cudnn_flashmla@cudnn_flashmla | 201.4 | 3.47 | 3.47 | 65.8 | 65.0 | 6.6 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 201.4 | 11.94 | 11.94 | 71.2 | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.4 | - | - |
| tiny-65536-hca-cp1 | tilelang@cudnn_flashmla | 1306.0 | 1.85 | 1.39 | 43.1 | 59.2 | 6.0 |
| tiny-65536-hca-cp1 | cudnn_flashmla@cudnn_flashmla | 1306.0 | 1.85 | 1.85 | 87.5 | 77.4 | 7.8 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1306.0 | 2.95 | 2.95 | 89.9 | - | - |
| tiny-65536-hca-cp8r0 | tilelang@main | 163.4 | 1.85 | 1.39 | 27.4 | 45.0 | 4.6 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@main | 163.4 | 2.94 | 2.94 | 71.0 | - | - |
| tiny-65536-hca-cp8r0 | tilelang@cudnn_flashmla | 163.4 | 1.85 | 1.39 | 27.1 | 44.8 | 4.5 |
| tiny-65536-hca-cp8r0 | cudnn_flashmla@cudnn_flashmla | 163.4 | 1.85 | 1.85 | 64.2 | 63.2 | 6.4 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 163.4 | 2.94 | 2.94 | 70.5 | - | - |
| tiny-65536-hca-cp8r4 | tilelang@main | 163.3 | 1.85 | 1.39 | 26.9 | 45.2 | 4.6 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@main | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-hca-cp8r4 | tilelang@cudnn_flashmla | 163.3 | 1.85 | 1.39 | 26.9 | 44.6 | 4.5 |
| tiny-65536-hca-cp8r4 | cudnn_flashmla@cudnn_flashmla | 163.3 | 1.85 | 1.85 | 64.2 | 62.3 | 6.3 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 163.3 | 2.95 | 2.95 | 70.7 | - | - |
| tiny-65536-hca-cp8r7 | tilelang@main | 162.3 | 1.86 | 1.39 | 27.0 | 44.7 | 4.5 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@main | 162.3 | 2.96 | 2.96 | 69.6 | - | - |
| tiny-65536-hca-cp8r7 | tilelang@cudnn_flashmla | 162.3 | 1.86 | 1.39 | 26.3 | 44.5 | 4.5 |
| tiny-65536-hca-cp8r7 | cudnn_flashmla@cudnn_flashmla | 162.3 | 1.86 | 1.86 | 63.0 | 62.8 | 6.3 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 162.3 | 2.96 | 2.96 | 70.4 | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.8 | 59.3 | 6.0 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.3 | - | - |
| tiny-65536-sliding-cp1 | tilelang@cudnn_flashmla | 1306.0 | 1.85 | 1.39 | 43.0 | 59.4 | 6.0 |
| tiny-65536-sliding-cp1 | cudnn_flashmla@cudnn_flashmla | 1306.0 | 1.85 | 1.85 | 87.4 | 77.5 | 7.8 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@cudnn_flashmla | 1306.0 | 2.95 | 2.95 | 89.8 | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@main | 163.4 | 1.85 | 1.39 | 26.7 | 45.0 | 4.5 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@main | 163.4 | 2.94 | 2.94 | 69.8 | - | - |
| tiny-65536-sliding-cp8r0 | tilelang@cudnn_flashmla | 163.4 | 1.85 | 1.39 | 26.7 | 44.8 | 4.5 |
| tiny-65536-sliding-cp8r0 | cudnn_flashmla@cudnn_flashmla | 163.4 | 1.85 | 1.85 | 62.9 | 63.2 | 6.4 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref@cudnn_flashmla | 163.4 | 2.94 | 2.94 | 69.6 | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@main | 163.3 | 1.85 | 1.39 | 27.2 | 45.2 | 4.6 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@main | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-sliding-cp8r4 | tilelang@cudnn_flashmla | 163.3 | 1.85 | 1.39 | 26.8 | 44.8 | 4.5 |
| tiny-65536-sliding-cp8r4 | cudnn_flashmla@cudnn_flashmla | 163.3 | 1.85 | 1.85 | 63.8 | 63.7 | 6.4 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref@cudnn_flashmla | 163.3 | 2.95 | 2.95 | 70.5 | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@main | 162.3 | 1.86 | 1.39 | 27.2 | 44.7 | 4.5 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@main | 162.3 | 2.96 | 2.96 | 69.7 | - | - |
| tiny-65536-sliding-cp8r7 | tilelang@cudnn_flashmla | 162.3 | 1.86 | 1.39 | 27.1 | 44.5 | 4.5 |
| tiny-65536-sliding-cp8r7 | cudnn_flashmla@cudnn_flashmla | 162.3 | 1.86 | 1.86 | 63.9 | 62.9 | 6.4 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref@cudnn_flashmla | 162.3 | 2.96 | 2.96 | 70.7 | - | - |

Correctness failures (excluded from timing): 0
