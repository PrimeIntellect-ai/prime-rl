- `main`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git 6d5cf0180
- corpus hash b5b289171983c8c6; synthetic corpus: random-weight CSA picks are near-uniform, while a
  trained indexer favors recent and neighboring entries, so CSA gather locality here is pessimistic.

Op-boundary time per call in µs (lower is better): median over rounds, p20-p80 across rounds.
`/TL` is this time divided by tilelang's in the same run.

| item | backend | fwd µs | fwd p20-p80 | fwd /TL | f+b µs | f+b p20-p80 | f+b /TL |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang | 1197 | 1180-1209 | 1.00 | 3186 | 3183-3216 | 1.00 |
| single-2048-csa-cp1 | flashmla_fwd_ref | 384.0 | 381.9-387.6 | 0.32 | - | - | - |
| single-2048-csa-cp8r0 | tilelang | 733.5 | 729.6-769.4 | 1.00 | 2001 | 1987-2043 | 1.00 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref | 182.5 | 181.2-196.9 | 0.25 | - | - | - |
| single-2048-csa-cp8r4 | tilelang | 763.3 | 755.8-769.4 | 1.00 | 2028 | 2010-2035 | 1.00 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref | 197.7 | 194.4-198.5 | 0.26 | - | - | - |
| single-2048-csa-cp8r7 | tilelang | 774.3 | 771.8-795.5 | 1.00 | 2025 | 2024-2030 | 1.00 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref | 202.8 | 198.5-208.2 | 0.26 | - | - | - |
| single-2048-hca-cp1 | tilelang | 1085 | 1075-1099 | 1.00 | 2504 | 2477-2531 | 1.00 |
| single-2048-hca-cp1 | flashmla_fwd_ref | 341.1 | 339.6-343.2 | 0.31 | - | - | - |
| single-2048-hca-cp8r0 | tilelang | 763.9 | 761.9-769.2 | 1.00 | 2152 | 2142-2403 | 1.00 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref | 188.7 | 185.0-194.9 | 0.25 | - | - | - |
| single-2048-hca-cp8r4 | tilelang | 793.0 | 784.8-810.2 | 1.00 | 2158 | 2111-2193 | 1.00 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref | 193.9 | 191.6-196.0 | 0.24 | - | - | - |
| single-2048-hca-cp8r7 | tilelang | 806.9 | 791.4-813.7 | 1.00 | 2178 | 2146-2204 | 1.00 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref | 193.5 | 189.0-195.7 | 0.24 | - | - | - |
| single-2048-sliding-cp1 | tilelang | 1016 | 1002-1021 | 1.00 | 2288 | 2278-2306 | 1.00 |
| single-2048-sliding-cp1 | flashmla_fwd_ref | 298.4 | 293.8-303.0 | 0.29 | - | - | - |
| single-2048-sliding-cp8r0 | tilelang | 732.5 | 723.8-761.5 | 1.00 | 2047 | 2013-2148 | 1.00 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref | 179.7 | 174.7-186.8 | 0.25 | - | - | - |
| single-2048-sliding-cp8r4 | tilelang | 739.5 | 732.5-778.8 | 1.00 | 2018 | 2006-2039 | 1.00 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref | 179.0 | 174.0-188.2 | 0.24 | - | - | - |
| single-2048-sliding-cp8r7 | tilelang | 739.5 | 735.6-746.3 | 1.00 | 2034 | 2020-2067 | 1.00 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref | 180.9 | 179.1-190.1 | 0.24 | - | - | - |
| short-2048-csa-cp1 | tilelang | 1132 | 1121-1135 | 1.00 | 2756 | 2748-2783 | 1.00 |
| short-2048-csa-cp1 | flashmla_fwd_ref | 368.0 | 365.2-371.5 | 0.33 | - | - | - |
| short-2048-csa-cp8r0 | tilelang | 753.1 | 737.6-770.5 | 1.00 | 2023 | 2006-2054 | 1.00 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref | 187.7 | 179.4-191.9 | 0.25 | - | - | - |
| short-2048-csa-cp8r4 | tilelang | 742.5 | 736.4-755.8 | 1.00 | 2033 | 2020-2063 | 1.00 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref | 187.7 | 183.3-192.1 | 0.25 | - | - | - |
| short-2048-csa-cp8r7 | tilelang | 762.2 | 749.4-771.5 | 1.00 | 2037 | 2025-2095 | 1.00 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref | 191.5 | 186.9-192.3 | 0.25 | - | - | - |
| short-2048-hca-cp1 | tilelang | 1079 | 1074-1103 | 1.00 | 2489 | 2485-2573 | 1.00 |
| short-2048-hca-cp1 | flashmla_fwd_ref | 340.4 | 337.8-344.8 | 0.32 | - | - | - |
| short-2048-hca-cp8r0 | tilelang | 787.4 | 779.9-790.5 | 1.00 | 2154 | 2142-2191 | 1.00 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref | 191.5 | 187.6-192.1 | 0.24 | - | - | - |
| short-2048-hca-cp8r4 | tilelang | 795.0 | 779.4-801.7 | 1.00 | 2162 | 2154-2181 | 1.00 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref | 193.3 | 190.6-194.5 | 0.24 | - | - | - |
| short-2048-hca-cp8r7 | tilelang | 792.8 | 779.8-798.3 | 1.00 | 2193 | 2161-2199 | 1.00 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref | 188.9 | 188.3-192.7 | 0.24 | - | - | - |
| short-2048-sliding-cp1 | tilelang | 1003 | 994.3-1018 | 1.00 | 2330 | 2322-2398 | 1.00 |
| short-2048-sliding-cp1 | flashmla_fwd_ref | 293.3 | 291.6-301.8 | 0.29 | - | - | - |
| short-2048-sliding-cp8r0 | tilelang | 746.8 | 734.3-752.0 | 1.00 | 2042 | 2039-2056 | 1.00 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref | 181.9 | 179.7-187.0 | 0.24 | - | - | - |
| short-2048-sliding-cp8r4 | tilelang | 756.9 | 751.3-773.4 | 1.00 | 2022 | 2012-2028 | 1.00 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref | 189.8 | 187.4-194.1 | 0.25 | - | - | - |
| short-2048-sliding-cp8r7 | tilelang | 736.4 | 731.7-751.1 | 1.00 | 2044 | 2038-2069 | 1.00 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref | 185.1 | 179.8-188.5 | 0.25 | - | - | - |
| heavy-2048-csa-cp1 | tilelang | 1171 | 1161-1186 | 1.00 | 2955 | 2937-2956 | 1.00 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref | 389.7 | 383.9-390.4 | 0.33 | - | - | - |
| heavy-2048-csa-cp8r0 | tilelang | 747.5 | 737.9-759.0 | 1.00 | 2043 | 2029-2055 | 1.00 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref | 189.1 | 186.1-190.5 | 0.25 | - | - | - |
| heavy-2048-csa-cp8r4 | tilelang | 777.6 | 755.6-789.9 | 1.00 | 2065 | 2036-2070 | 1.00 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref | 200.7 | 189.6-207.8 | 0.26 | - | - | - |
| heavy-2048-csa-cp8r7 | tilelang | 793.2 | 765.5-859.1 | 1.00 | 2034 | 2029-2075 | 1.00 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref | 209.7 | 204.5-224.9 | 0.26 | - | - | - |
| heavy-2048-hca-cp1 | tilelang | 1074 | 1069-1080 | 1.00 | 2464 | 2458-2470 | 1.00 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref | 335.2 | 329.4-337.8 | 0.31 | - | - | - |
| heavy-2048-hca-cp8r0 | tilelang | 774.2 | 771.3-778.2 | 1.00 | 2135 | 2131-2143 | 1.00 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref | 189.3 | 188.5-195.6 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r4 | tilelang | 782.0 | 771.9-791.9 | 1.00 | 2122 | 2120-2137 | 1.00 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref | 189.5 | 183.3-193.8 | 0.24 | - | - | - |
| heavy-2048-hca-cp8r7 | tilelang | 786.8 | 778.1-795.3 | 1.00 | 2157 | 2147-2182 | 1.00 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref | 189.8 | 187.0-192.2 | 0.24 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang | 1000 | 986.0-1002 | 1.00 | 2294 | 2292-2303 | 1.00 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref | 291.8 | 289.1-294.8 | 0.29 | - | - | - |
| heavy-2048-sliding-cp8r0 | tilelang | 736.0 | 727.2-829.0 | 1.00 | 2039 | 2024-2043 | 1.00 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref | 180.0 | 176.3-183.7 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r4 | tilelang | 742.3 | 731.2-750.7 | 1.00 | 2036 | 2019-2052 | 1.00 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref | 180.7 | 175.9-183.0 | 0.24 | - | - | - |
| heavy-2048-sliding-cp8r7 | tilelang | 743.6 | 736.9-744.8 | 1.00 | 2071 | 2052-2104 | 1.00 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref | 182.7 | 175.3-184.9 | 0.25 | - | - | - |
| tiny-2048-csa-cp1 | tilelang | 1043 | 1040-1056 | 1.00 | 2359 | 2333-2369 | 1.00 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref | 333.9 | 329.9-336.8 | 0.32 | - | - | - |
| tiny-2048-csa-cp8r0 | tilelang | 753.9 | 747.6-789.7 | 1.00 | 2056 | 2032-2068 | 1.00 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref | 192.5 | 185.3-201.4 | 0.26 | - | - | - |
| tiny-2048-csa-cp8r4 | tilelang | 748.5 | 741.9-755.9 | 1.00 | 2029 | 2015-2055 | 1.00 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref | 187.9 | 184.4-189.9 | 0.25 | - | - | - |
| tiny-2048-csa-cp8r7 | tilelang | 761.0 | 757.8-769.9 | 1.00 | 2040 | 2032-2079 | 1.00 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref | 189.9 | 186.1-198.6 | 0.25 | - | - | - |
| tiny-2048-hca-cp1 | tilelang | 962.7 | 956.2-975.0 | 1.00 | 2093 | 2080-2128 | 1.00 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref | 292.2 | 290.6-305.2 | 0.30 | - | - | - |
| tiny-2048-hca-cp8r0 | tilelang | 736.5 | 733.4-737.7 | 1.00 | 2030 | 2013-2052 | 1.00 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref | 182.6 | 176.8-187.7 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r4 | tilelang | 735.0 | 723.8-756.2 | 1.00 | 2049 | 2011-2074 | 1.00 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref | 180.9 | 179.2-183.4 | 0.25 | - | - | - |
| tiny-2048-hca-cp8r7 | tilelang | 751.2 | 738.8-819.8 | 1.00 | 2099 | 2044-2164 | 1.00 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref | 190.0 | 184.2-195.8 | 0.25 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang | 957.8 | 952.0-962.9 | 1.00 | 2108 | 2103-2118 | 1.00 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref | 290.5 | 284.1-294.3 | 0.30 | - | - | - |
| tiny-2048-sliding-cp8r0 | tilelang | 731.7 | 730.4-739.6 | 1.00 | 2136 | 2058-2216 | 1.00 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref | 179.8 | 173.2-180.5 | 0.25 | - | - | - |
| tiny-2048-sliding-cp8r4 | tilelang | 738.5 | 727.1-742.4 | 1.00 | 2024 | 2010-2050 | 1.00 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref | 178.4 | 171.8-182.8 | 0.24 | - | - | - |
| tiny-2048-sliding-cp8r7 | tilelang | 758.1 | 748.7-774.6 | 1.00 | 2049 | 2038-2151 | 1.00 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref | 186.8 | 182.1-192.1 | 0.25 | - | - | - |
| single-4096-csa-cp1 | tilelang | 1890 | 1888-1954 | 1.00 | 5972 | 5965-5976 | 1.00 |
| single-4096-csa-cp1 | flashmla_fwd_ref | 715.4 | 711.7-718.7 | 0.38 | - | - | - |
| single-4096-csa-cp8r0 | tilelang | 795.7 | 791.3-798.4 | 1.00 | 2037 | 2027-2047 | 1.00 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref | 206.3 | 203.2-212.2 | 0.26 | - | - | - |
| single-4096-csa-cp8r4 | tilelang | 869.0 | 862.6-883.1 | 1.00 | 2333 | 2321-2341 | 1.00 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref | 248.4 | 245.0-250.7 | 0.29 | - | - | - |
| single-4096-csa-cp8r7 | tilelang | 880.8 | 872.3-893.7 | 1.00 | 2349 | 2340-2358 | 1.00 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref | 252.0 | 246.9-253.2 | 0.29 | - | - | - |
| single-4096-hca-cp1 | tilelang | 1432 | 1411-1475 | 1.00 | 3134 | 3126-3142 | 1.00 |
| single-4096-hca-cp1 | flashmla_fwd_ref | 505.5 | 499.0-532.2 | 0.35 | - | - | - |
| single-4096-hca-cp8r0 | tilelang | 828.2 | 816.4-834.6 | 1.00 | 2126 | 2122-2148 | 1.00 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref | 217.1 | 213.7-217.8 | 0.26 | - | - | - |
| single-4096-hca-cp8r4 | tilelang | 824.2 | 816.0-832.5 | 1.00 | 2143 | 2120-2164 | 1.00 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref | 216.8 | 214.5-219.7 | 0.26 | - | - | - |
| single-4096-hca-cp8r7 | tilelang | 827.1 | 813.1-838.7 | 1.00 | 2159 | 2139-2187 | 1.00 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref | 217.2 | 212.4-222.6 | 0.26 | - | - | - |
| single-4096-sliding-cp1 | tilelang | 1297 | 1285-1352 | 1.00 | 2859 | 2858-2862 | 1.00 |
| single-4096-sliding-cp1 | flashmla_fwd_ref | 420.2 | 412.8-423.1 | 0.32 | - | - | - |
| single-4096-sliding-cp8r0 | tilelang | 790.1 | 775.7-812.3 | 1.00 | 2034 | 2027-2045 | 1.00 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref | 199.3 | 197.8-208.9 | 0.25 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang | 791.1 | 779.7-892.0 | 1.00 | 2057 | 2032-2070 | 1.00 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref | 203.1 | 197.7-217.5 | 0.26 | - | - | - |
| single-4096-sliding-cp8r7 | tilelang | 785.2 | 769.9-787.5 | 1.00 | 2065 | 2049-2079 | 1.00 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref | 194.0 | 192.7-204.3 | 0.25 | - | - | - |
| short-4096-csa-cp1 | tilelang | 1523 | 1513-1536 | 1.00 | 3873 | 3871-3875 | 1.00 |
| short-4096-csa-cp1 | flashmla_fwd_ref | 552.9 | 546.9-555.2 | 0.36 | - | - | - |
| short-4096-csa-cp8r0 | tilelang | 827.7 | 802.6-859.4 | 1.00 | 2054 | 2046-2075 | 1.00 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref | 222.4 | 213.5-232.8 | 0.27 | - | - | - |
| short-4096-csa-cp8r4 | tilelang | 876.9 | 821.4-901.4 | 1.00 | 2145 | 2070-2654 | 1.00 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref | 227.1 | 220.9-229.2 | 0.26 | - | - | - |
| short-4096-csa-cp8r7 | tilelang | 818.1 | 806.9-826.3 | 1.00 | 2060 | 2047-2075 | 1.00 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref | 210.7 | 208.5-216.6 | 0.26 | - | - | - |
| short-4096-hca-cp1 | tilelang | 1411 | 1404-1416 | 1.00 | 3089 | 3082-3102 | 1.00 |
| short-4096-hca-cp1 | flashmla_fwd_ref | 500.1 | 497.7-502.9 | 0.35 | - | - | - |
| short-4096-hca-cp8r0 | tilelang | 844.5 | 832.2-863.5 | 1.00 | 2180 | 2165-2244 | 1.00 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref | 216.0 | 211.8-217.3 | 0.26 | - | - | - |
| short-4096-hca-cp8r4 | tilelang | 826.5 | 818.9-845.1 | 1.00 | 2157 | 2154-2171 | 1.00 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref | 208.2 | 206.1-217.1 | 0.25 | - | - | - |
| short-4096-hca-cp8r7 | tilelang | 831.1 | 816.4-835.0 | 1.00 | 2197 | 2167-2202 | 1.00 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref | 214.2 | 209.4-216.6 | 0.26 | - | - | - |
| short-4096-sliding-cp1 | tilelang | 1312 | 1272-1315 | 1.00 | 2842 | 2832-2846 | 1.00 |
| short-4096-sliding-cp1 | flashmla_fwd_ref | 423.2 | 414.6-434.4 | 0.32 | - | - | - |
| short-4096-sliding-cp8r0 | tilelang | 793.8 | 787.3-805.0 | 1.00 | 2036 | 2030-2048 | 1.00 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref | 197.4 | 192.0-199.4 | 0.25 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang | 794.0 | 780.5-815.6 | 1.00 | 2041 | 2033-2066 | 1.00 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref | 197.7 | 193.1-216.8 | 0.25 | - | - | - |
| short-4096-sliding-cp8r7 | tilelang | 774.8 | 766.8-785.9 | 1.00 | 2086 | 2079-2120 | 1.00 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref | 195.7 | 192.6-198.1 | 0.25 | - | - | - |
| heavy-4096-csa-cp1 | tilelang | 1462 | 1446-1472 | 1.00 | 3545 | 3535-3555 | 1.00 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref | 524.9 | 521.9-530.8 | 0.36 | - | - | - |
| heavy-4096-csa-cp8r0 | tilelang | 794.4 | 789.7-815.2 | 1.00 | 2033 | 2023-2050 | 1.00 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref | 208.3 | 203.7-215.7 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang | 817.9 | 805.9-831.3 | 1.00 | 2049 | 2038-2194 | 1.00 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref | 217.3 | 211.7-218.4 | 0.27 | - | - | - |
| heavy-4096-csa-cp8r7 | tilelang | 849.9 | 811.7-856.0 | 1.00 | 2089 | 2057-2341 | 1.00 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref | 215.8 | 214.5-218.0 | 0.25 | - | - | - |
| heavy-4096-hca-cp1 | tilelang | 1375 | 1372-1386 | 1.00 | 2982 | 2967-2988 | 1.00 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref | 482.7 | 480.6-489.0 | 0.35 | - | - | - |
| heavy-4096-hca-cp8r0 | tilelang | 835.5 | 828.0-844.7 | 1.00 | 2153 | 2137-2162 | 1.00 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref | 214.5 | 211.6-218.6 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang | 827.9 | 820.9-840.8 | 1.00 | 2160 | 2157-2177 | 1.00 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref | 207.4 | 205.8-212.8 | 0.25 | - | - | - |
| heavy-4096-hca-cp8r7 | tilelang | 838.2 | 821.0-847.4 | 1.00 | 2180 | 2177-2181 | 1.00 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref | 215.7 | 212.8-219.9 | 0.26 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang | 1283 | 1276-1286 | 1.00 | 2911 | 2843-2955 | 1.00 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref | 418.8 | 418.5-423.0 | 0.33 | - | - | - |
| heavy-4096-sliding-cp8r0 | tilelang | 793.9 | 787.1-802.0 | 1.00 | 2083 | 2065-2088 | 1.00 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref | 198.7 | 195.6-205.1 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang | 780.6 | 767.1-784.0 | 1.00 | 2050 | 2037-2053 | 1.00 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref | 194.3 | 190.0-200.3 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r7 | tilelang | 772.0 | 762.3-785.7 | 1.00 | 2099 | 2076-2132 | 1.00 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref | 197.5 | 194.8-199.0 | 0.26 | - | - | - |
| tiny-4096-csa-cp1 | tilelang | 1374 | 1368-1389 | 1.00 | 2919 | 2904-2930 | 1.00 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref | 501.5 | 498.9-506.4 | 0.37 | - | - | - |
| tiny-4096-csa-cp8r0 | tilelang | 796.5 | 784.8-820.9 | 1.00 | 2020 | 2015-2135 | 1.00 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref | 210.9 | 210.1-221.9 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang | 809.3 | 793.1-830.6 | 1.00 | 2034 | 2014-2048 | 1.00 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref | 209.6 | 207.2-216.1 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r7 | tilelang | 786.0 | 777.7-790.5 | 1.00 | 2043 | 2030-2080 | 1.00 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref | 210.6 | 206.3-211.8 | 0.27 | - | - | - |
| tiny-4096-hca-cp1 | tilelang | 1227 | 1218-1246 | 1.00 | 2409 | 2400-2413 | 1.00 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref | 412.3 | 408.0-430.9 | 0.34 | - | - | - |
| tiny-4096-hca-cp8r0 | tilelang | 770.4 | 766.3-773.0 | 1.00 | 2033 | 2020-2052 | 1.00 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref | 194.8 | 192.6-198.4 | 0.25 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang | 767.4 | 763.6-775.5 | 1.00 | 2023 | 2018-2038 | 1.00 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref | 196.6 | 192.9-202.8 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r7 | tilelang | 784.1 | 768.8-793.5 | 1.00 | 2053 | 2045-2065 | 1.00 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref | 201.5 | 196.4-202.1 | 0.26 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang | 1226 | 1215-1234 | 1.00 | 2407 | 2386-2415 | 1.00 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref | 408.8 | 408.3-411.3 | 0.33 | - | - | - |
| tiny-4096-sliding-cp8r0 | tilelang | 775.3 | 771.9-783.8 | 1.00 | 2030 | 2005-2076 | 1.00 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref | 200.7 | 197.0-203.9 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang | 781.6 | 770.1-787.6 | 1.00 | 2040 | 2028-2056 | 1.00 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref | 203.1 | 201.3-214.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r7 | tilelang | 786.4 | 772.9-790.8 | 1.00 | 2081 | 2076-2129 | 1.00 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref | 202.2 | 194.2-206.2 | 0.26 | - | - | - |
| single-16384-csa-cp1 | tilelang | 5918 | 5886-6013 | 1.00 | 23560 | 23556-23572 | 1.00 |
| single-16384-csa-cp1 | flashmla_fwd_ref | 2570 | 2560-2654 | 0.43 | - | - | - |
| single-16384-csa-cp8r0 | tilelang | 1232 | 1216-1235 | 1.00 | 3243 | 3230-3250 | 1.00 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref | 417.6 | 414.9-421.8 | 0.34 | - | - | - |
| single-16384-csa-cp8r4 | tilelang | 1409 | 1392-1436 | 1.00 | 4079 | 4072-4087 | 1.00 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref | 495.2 | 492.1-501.8 | 0.35 | - | - | - |
| single-16384-csa-cp8r7 | tilelang | 1405 | 1396-1424 | 1.00 | 4104 | 4096-4125 | 1.00 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref | 496.7 | 493.8-498.8 | 0.35 | - | - | - |
| single-16384-hca-cp1 | tilelang | 3573 | 3561-3617 | 1.00 | 10905 | 10899-10911 | 1.00 |
| single-16384-hca-cp1 | flashmla_fwd_ref | 1490 | 1486-1513 | 0.42 | - | - | - |
| single-16384-hca-cp8r0 | tilelang | 1081 | 1067-1109 | 1.00 | 2422 | 2395-2446 | 1.00 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref | 340.5 | 338.5-348.6 | 0.32 | - | - | - |
| single-16384-hca-cp8r4 | tilelang | 1097 | 1094-1101 | 1.00 | 2665 | 2641-2689 | 1.00 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref | 339.9 | 338.0-344.9 | 0.31 | - | - | - |
| single-16384-hca-cp8r7 | tilelang | 1106 | 1097-1108 | 1.00 | 2819 | 2816-2866 | 1.00 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref | 343.8 | 338.7-344.0 | 0.31 | - | - | - |
| single-16384-sliding-cp1 | tilelang | 2973 | 2970-2979 | 1.00 | 8286 | 8275-8288 | 1.00 |
| single-16384-sliding-cp1 | flashmla_fwd_ref | 1164 | 1159-1174 | 0.39 | - | - | - |
| single-16384-sliding-cp8r0 | tilelang | 1019 | 1002-1035 | 1.00 | 2333 | 2317-2348 | 1.00 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref | 299.8 | 294.8-317.2 | 0.29 | - | - | - |
| single-16384-sliding-cp8r4 | tilelang | 1004 | 986.3-1009 | 1.00 | 2390 | 2382-2476 | 1.00 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref | 296.7 | 289.8-313.0 | 0.30 | - | - | - |
| single-16384-sliding-cp8r7 | tilelang | 996.2 | 987.6-1004 | 1.00 | 2389 | 2385-2438 | 1.00 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref | 294.0 | 292.2-302.3 | 0.30 | - | - | - |
| short-16384-csa-cp1 | tilelang | 4509 | 4494-4544 | 1.00 | 15976 | 15975-16005 | 1.00 |
| short-16384-csa-cp1 | flashmla_fwd_ref | 1957 | 1952-1959 | 0.43 | - | - | - |
| short-16384-csa-cp8r0 | tilelang | 1175 | 1156-1200 | 1.00 | 2997 | 2984-3003 | 1.00 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref | 390.1 | 384.5-394.8 | 0.33 | - | - | - |
| short-16384-csa-cp8r4 | tilelang | 1249 | 1236-1261 | 1.00 | 3387 | 3384-3406 | 1.00 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref | 431.6 | 429.3-432.5 | 0.35 | - | - | - |
| short-16384-csa-cp8r7 | tilelang | 1211 | 1195-1219 | 1.00 | 3241 | 3236-3255 | 1.00 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref | 408.0 | 406.4-415.0 | 0.34 | - | - | - |
| short-16384-hca-cp1 | tilelang | 3337 | 3333-3349 | 1.00 | 9225 | 9192-9255 | 1.00 |
| short-16384-hca-cp1 | flashmla_fwd_ref | 1463 | 1461-1465 | 0.44 | - | - | - |
| short-16384-hca-cp8r0 | tilelang | 1070 | 1063-1086 | 1.00 | 2499 | 2480-2505 | 1.00 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref | 338.9 | 332.9-342.3 | 0.32 | - | - | - |
| short-16384-hca-cp8r4 | tilelang | 1080 | 1076-1091 | 1.00 | 2532 | 2525-2555 | 1.00 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref | 337.4 | 334.8-343.4 | 0.31 | - | - | - |
| short-16384-hca-cp8r7 | tilelang | 1081 | 1069-1086 | 1.00 | 2558 | 2548-2561 | 1.00 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref | 338.8 | 338.8-342.2 | 0.31 | - | - | - |
| short-16384-sliding-cp1 | tilelang | 2985 | 2970-3000 | 1.00 | 8155 | 8151-8192 | 1.00 |
| short-16384-sliding-cp1 | flashmla_fwd_ref | 1169 | 1162-1175 | 0.39 | - | - | - |
| short-16384-sliding-cp8r0 | tilelang | 1001 | 987.1-1006 | 1.00 | 2305 | 2292-2331 | 1.00 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref | 296.3 | 293.4-300.4 | 0.30 | - | - | - |
| short-16384-sliding-cp8r4 | tilelang | 993.9 | 985.0-1070 | 1.00 | 2370 | 2342-2396 | 1.00 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref | 294.4 | 287.7-309.9 | 0.30 | - | - | - |
| short-16384-sliding-cp8r7 | tilelang | 1001 | 987.3-1004 | 1.00 | 2361 | 2358-2371 | 1.00 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref | 294.5 | 290.2-294.9 | 0.29 | - | - | - |
| heavy-16384-csa-cp1 | tilelang | 3969 | 3954-4002 | 1.00 | 12997 | 12956-13003 | 1.00 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref | 1726 | 1725-1731 | 0.43 | - | - | - |
| heavy-16384-csa-cp8r0 | tilelang | 1092 | 1069-1121 | 1.00 | 2575 | 2548-2608 | 1.00 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref | 346.5 | 345.0-355.2 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r4 | tilelang | 1117 | 1112-1126 | 1.00 | 2762 | 2761-2778 | 1.00 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref | 362.7 | 357.6-370.0 | 0.32 | - | - | - |
| heavy-16384-csa-cp8r7 | tilelang | 1264 | 1261-1266 | 1.00 | 3482 | 3467-3491 | 1.00 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref | 428.5 | 427.4-434.8 | 0.34 | - | - | - |
| heavy-16384-hca-cp1 | tilelang | 3252 | 3222-3290 | 1.00 | 8705 | 8691-8740 | 1.00 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref | 1404 | 1396-1410 | 0.43 | - | - | - |
| heavy-16384-hca-cp8r0 | tilelang | 1068 | 1060-1096 | 1.00 | 2448 | 2445-2477 | 1.00 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref | 335.5 | 331.0-346.8 | 0.31 | - | - | - |
| heavy-16384-hca-cp8r4 | tilelang | 1070 | 1066-1081 | 1.00 | 2488 | 2479-2500 | 1.00 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref | 341.7 | 338.1-344.5 | 0.32 | - | - | - |
| heavy-16384-hca-cp8r7 | tilelang | 1088 | 1082-1106 | 1.00 | 2621 | 2616-2649 | 1.00 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref | 346.1 | 343.6-346.5 | 0.32 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang | 2979 | 2958-3008 | 1.00 | 7887 | 7879-7894 | 1.00 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref | 1173 | 1173-1183 | 0.39 | - | - | - |
| heavy-16384-sliding-cp8r0 | tilelang | 1013 | 1002-1040 | 1.00 | 2292 | 2285-2305 | 1.00 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref | 297.7 | 295.7-301.9 | 0.29 | - | - | - |
| heavy-16384-sliding-cp8r4 | tilelang | 982.7 | 980.0-1006 | 1.00 | 2331 | 2316-2337 | 1.00 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref | 293.1 | 290.1-298.7 | 0.30 | - | - | - |
| heavy-16384-sliding-cp8r7 | tilelang | 1022 | 994.6-1025 | 1.00 | 2405 | 2394-2439 | 1.00 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref | 296.4 | 293.9-310.3 | 0.29 | - | - | - |
| tiny-16384-csa-cp1 | tilelang | 3287 | 3284-3298 | 1.00 | 8726 | 8702-8759 | 1.00 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref | 1476 | 1471-1478 | 0.45 | - | - | - |
| tiny-16384-csa-cp8r0 | tilelang | 1043 | 1041-1055 | 1.00 | 2376 | 2356-2382 | 1.00 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref | 334.5 | 334.0-340.0 | 0.32 | - | - | - |
| tiny-16384-csa-cp8r4 | tilelang | 1048 | 1040-1052 | 1.00 | 2358 | 2354-2374 | 1.00 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref | 342.2 | 335.7-346.6 | 0.33 | - | - | - |
| tiny-16384-csa-cp8r7 | tilelang | 1032 | 1022-1076 | 1.00 | 2369 | 2363-2372 | 1.00 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref | 334.5 | 327.7-343.5 | 0.32 | - | - | - |
| tiny-16384-hca-cp1 | tilelang | 2707 | 2700-2714 | 1.00 | 6184 | 6177-6195 | 1.00 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref | 1176 | 1174-1178 | 0.43 | - | - | - |
| tiny-16384-hca-cp8r0 | tilelang | 981.1 | 976.9-1003 | 1.00 | 2144 | 2130-2255 | 1.00 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref | 296.6 | 289.3-309.1 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r4 | tilelang | 968.2 | 959.1-972.4 | 1.00 | 2095 | 2090-2123 | 1.00 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref | 293.1 | 289.9-295.4 | 0.30 | - | - | - |
| tiny-16384-hca-cp8r7 | tilelang | 955.4 | 942.5-968.8 | 1.00 | 2130 | 2120-2172 | 1.00 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref | 294.3 | 290.2-298.9 | 0.31 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang | 2728 | 2711-2754 | 1.00 | 6163 | 6161-6164 | 1.00 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref | 1181 | 1174-1182 | 0.43 | - | - | - |
| tiny-16384-sliding-cp8r0 | tilelang | 962.6 | 956.2-977.8 | 1.00 | 2142 | 2126-2157 | 1.00 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref | 297.0 | 295.1-301.5 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r4 | tilelang | 962.2 | 954.6-977.3 | 1.00 | 2174 | 2113-2294 | 1.00 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref | 299.2 | 290.4-299.6 | 0.31 | - | - | - |
| tiny-16384-sliding-cp8r7 | tilelang | 961.0 | 949.7-972.0 | 1.00 | 2140 | 2129-2174 | 1.00 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref | 293.2 | 292.1-301.2 | 0.31 | - | - | - |
| single-49208-csa-cp1 | tilelang | 16424 | 16369-16661 | 1.00 | 70708 | 70691-70733 | 1.00 |
| single-49208-csa-cp1 | flashmla_fwd_ref | 8309 | 8259-8522 | 0.51 | - | - | - |
| single-49208-csa-cp8r0 | tilelang | 2556 | 2547-2584 | 1.00 | 8923 | 8921-8925 | 1.00 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref | 1026 | 1020-1028 | 0.40 | - | - | - |
| single-49208-csa-cp8r4 | tilelang | 2744 | 2725-2754 | 1.00 | 10025 | 10005-10044 | 1.00 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref | 1100 | 1100-1108 | 0.40 | - | - | - |
| single-49208-csa-cp8r7 | tilelang | 2745 | 2736-2763 | 1.00 | 10361 | 10302-10387 | 1.00 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref | 1105 | 1101-1108 | 0.40 | - | - | - |
| single-49208-hca-cp1 | tilelang | 11496 | 11468-11639 | 1.00 | 42559 | 42547-42587 | 1.00 |
| single-49208-hca-cp1 | flashmla_fwd_ref | 5402 | 5373-5651 | 0.47 | - | - | - |
| single-49208-hca-cp8r0 | tilelang | 1710 | 1698-1758 | 1.00 | 4247 | 4246-4326 | 1.00 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref | 670.5 | 667.3-673.7 | 0.39 | - | - | - |
| single-49208-hca-cp8r4 | tilelang | 2145 | 2139-2186 | 1.00 | 6565 | 6561-6572 | 1.00 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref | 801.2 | 799.3-802.6 | 0.37 | - | - | - |
| single-49208-hca-cp8r7 | tilelang | 2430 | 2427-2432 | 1.00 | 8244 | 8226-8262 | 1.00 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref | 944.5 | 939.2-955.3 | 0.39 | - | - | - |
| single-49208-sliding-cp1 | tilelang | 7433 | 7429-7462 | 1.00 | 22888 | 22878-22898 | 1.00 |
| single-49208-sliding-cp1 | flashmla_fwd_ref | 3125 | 3112-3128 | 0.42 | - | - | - |
| single-49208-sliding-cp8r0 | tilelang | 1574 | 1566-1595 | 1.00 | 3745 | 3734-3748 | 1.00 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref | 543.0 | 540.3-547.6 | 0.35 | - | - | - |
| single-49208-sliding-cp8r4 | tilelang | 1557 | 1554-1558 | 1.00 | 3783 | 3768-3790 | 1.00 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref | 539.1 | 537.2-545.8 | 0.35 | - | - | - |
| single-49208-sliding-cp8r7 | tilelang | 1572 | 1565-1598 | 1.00 | 3792 | 3777-3798 | 1.00 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref | 540.1 | 536.8-541.1 | 0.34 | - | - | - |
| short-49208-csa-cp1 | tilelang | 12988 | 12944-13072 | 1.00 | 51070 | 51061-51101 | 1.00 |
| short-49208-csa-cp1 | flashmla_fwd_ref | 6227 | 6191-6569 | 0.48 | - | - | - |
| short-49208-csa-cp8r0 | tilelang | 2037 | 2013-2042 | 1.00 | 6012 | 6007-6017 | 1.00 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref | 791.1 | 785.7-795.3 | 0.39 | - | - | - |
| short-49208-csa-cp8r4 | tilelang | 2386 | 2373-2409 | 1.00 | 8086 | 8074-8104 | 1.00 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref | 959.5 | 958.1-965.9 | 0.40 | - | - | - |
| short-49208-csa-cp8r7 | tilelang | 2161 | 2150-2181 | 1.00 | 6809 | 6800-6810 | 1.00 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref | 858.3 | 856.4-860.1 | 0.40 | - | - | - |
| short-49208-hca-cp1 | tilelang | 8505 | 8498-8522 | 1.00 | 26175 | 26167-26201 | 1.00 |
| short-49208-hca-cp1 | flashmla_fwd_ref | 4103 | 4093-4107 | 0.48 | - | - | - |
| short-49208-hca-cp8r0 | tilelang | 1701 | 1696-1716 | 1.00 | 4068 | 4063-4077 | 1.00 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref | 646.6 | 644.3-648.9 | 0.38 | - | - | - |
| short-49208-hca-cp8r4 | tilelang | 1747 | 1727-1765 | 1.00 | 4187 | 4177-4188 | 1.00 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref | 665.3 | 661.2-667.1 | 0.38 | - | - | - |
| short-49208-hca-cp8r7 | tilelang | 1733 | 1727-1755 | 1.00 | 4169 | 4154-4176 | 1.00 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref | 664.6 | 659.7-670.8 | 0.38 | - | - | - |
| short-49208-sliding-cp1 | tilelang | 7423 | 7418-7435 | 1.00 | 22632 | 22622-22641 | 1.00 |
| short-49208-sliding-cp1 | flashmla_fwd_ref | 3123 | 3117-3127 | 0.42 | - | - | - |
| short-49208-sliding-cp8r0 | tilelang | 1565 | 1552-1569 | 1.00 | 3663 | 3651-3680 | 1.00 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref | 541.0 | 537.3-548.1 | 0.35 | - | - | - |
| short-49208-sliding-cp8r4 | tilelang | 1569 | 1551-1574 | 1.00 | 3739 | 3720-3763 | 1.00 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref | 545.8 | 542.3-547.2 | 0.35 | - | - | - |
| short-49208-sliding-cp8r7 | tilelang | 1556 | 1551-1565 | 1.00 | 3743 | 3735-3746 | 1.00 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref | 536.9 | 535.8-539.6 | 0.34 | - | - | - |
| heavy-49208-csa-cp1 | tilelang | 14914 | 14896-14994 | 1.00 | 61708 | 61688-61710 | 1.00 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref | 7394 | 7345-7676 | 0.50 | - | - | - |
| heavy-49208-csa-cp8r0 | tilelang | 1925 | 1921-1931 | 1.00 | 5420 | 5416-5427 | 1.00 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref | 759.8 | 758.7-765.0 | 0.39 | - | - | - |
| heavy-49208-csa-cp8r4 | tilelang | 2629 | 2613-2647 | 1.00 | 9369 | 9360-9377 | 1.00 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref | 1058 | 1054-1062 | 0.40 | - | - | - |
| heavy-49208-csa-cp8r7 | tilelang | 2477 | 2464-2492 | 1.00 | 8582 | 8578-8592 | 1.00 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref | 990.5 | 988.4-996.9 | 0.40 | - | - | - |
| heavy-49208-hca-cp1 | tilelang | 9089 | 9066-9107 | 1.00 | 29605 | 29574-29619 | 1.00 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref | 4274 | 4269-4284 | 0.47 | - | - | - |
| heavy-49208-hca-cp8r0 | tilelang | 1702 | 1695-1730 | 1.00 | 3912 | 3899-3948 | 1.00 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref | 642.8 | 634.5-652.1 | 0.38 | - | - | - |
| heavy-49208-hca-cp8r4 | tilelang | 1753 | 1742-1768 | 1.00 | 4393 | 4384-4416 | 1.00 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref | 675.0 | 669.0-676.8 | 0.39 | - | - | - |
| heavy-49208-hca-cp8r7 | tilelang | 1845 | 1815-1849 | 1.00 | 4673 | 4665-4690 | 1.00 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref | 712.1 | 708.3-714.2 | 0.39 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang | 7448 | 7405-7474 | 1.00 | 22583 | 22581-22614 | 1.00 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref | 3135 | 3122-3149 | 0.42 | - | - | - |
| heavy-49208-sliding-cp8r0 | tilelang | 1572 | 1556-1586 | 1.00 | 3588 | 3579-3605 | 1.00 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref | 542.2 | 536.7-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r4 | tilelang | 1565 | 1560-1583 | 1.00 | 3761 | 3760-3770 | 1.00 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref | 538.9 | 537.2-543.6 | 0.34 | - | - | - |
| heavy-49208-sliding-cp8r7 | tilelang | 1567 | 1558-1572 | 1.00 | 3766 | 3757-3772 | 1.00 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref | 543.2 | 540.1-547.6 | 0.35 | - | - | - |
| tiny-49208-csa-cp1 | tilelang | 8393 | 8360-8427 | 1.00 | 24104 | 24092-24115 | 1.00 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref | 4295 | 4284-4297 | 0.51 | - | - | - |
| tiny-49208-csa-cp8r0 | tilelang | 1704 | 1673-1720 | 1.00 | 3875 | 3869-3881 | 1.00 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref | 658.2 | 656.7-659.3 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r4 | tilelang | 1680 | 1674-1688 | 1.00 | 3906 | 3895-3913 | 1.00 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref | 656.4 | 651.7-664.8 | 0.39 | - | - | - |
| tiny-49208-csa-cp8r7 | tilelang | 1689 | 1670-1693 | 1.00 | 3912 | 3898-3930 | 1.00 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref | 659.4 | 655.1-660.4 | 0.39 | - | - | - |
| tiny-49208-hca-cp1 | tilelang | 6744 | 6728-6770 | 1.00 | 16786 | 16781-16790 | 1.00 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref | 3164 | 3162-3181 | 0.47 | - | - | - |
| tiny-49208-hca-cp8r0 | tilelang | 1479 | 1468-1496 | 1.00 | 2952 | 2948-2958 | 1.00 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref | 541.4 | 536.4-544.8 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r4 | tilelang | 1485 | 1472-1546 | 1.00 | 2981 | 2974-2984 | 1.00 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref | 544.0 | 539.4-559.9 | 0.37 | - | - | - |
| tiny-49208-hca-cp8r7 | tilelang | 1474 | 1461-1497 | 1.00 | 2953 | 2945-2961 | 1.00 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref | 540.6 | 538.1-542.2 | 0.37 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang | 6733 | 6721-6755 | 1.00 | 16801 | 16796-16808 | 1.00 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref | 3173 | 3173-3185 | 0.47 | - | - | - |
| tiny-49208-sliding-cp8r0 | tilelang | 1463 | 1458-1465 | 1.00 | 2947 | 2942-2964 | 1.00 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref | 540.3 | 536.1-543.5 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r4 | tilelang | 1485 | 1470-1510 | 1.00 | 2976 | 2975-2983 | 1.00 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref | 542.5 | 538.3-548.4 | 0.37 | - | - | - |
| tiny-49208-sliding-cp8r7 | tilelang | 1465 | 1463-1477 | 1.00 | 2951 | 2947-2968 | 1.00 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref | 542.3 | 536.3-545.5 | 0.37 | - | - | - |
| single-65536-csa-cp1 | tilelang | 21694 | 21617-21724 | 1.00 | 95444 | 95324-95476 | 1.00 |
| single-65536-csa-cp1 | flashmla_fwd_ref | 11385 | 11377-11470 | 0.52 | - | - | - |
| single-65536-csa-cp8r0 | tilelang | 3265 | 3257-3300 | 1.00 | 11931 | 11922-11967 | 1.00 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref | 1325 | 1322-1327 | 0.41 | - | - | - |
| single-65536-csa-cp8r4 | tilelang | 3429 | 3394-3504 | 1.00 | 13145 | 13143-13204 | 1.00 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref | 1410 | 1405-1413 | 0.41 | - | - | - |
| single-65536-csa-cp8r7 | tilelang | 3384 | 3367-3434 | 1.00 | 13996 | 13970-14040 | 1.00 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref | 1410 | 1408-1414 | 0.42 | - | - | - |
| single-65536-hca-cp1 | tilelang | 16503 | 16484-16545 | 1.00 | 64327 | 64312-64341 | 1.00 |
| single-65536-hca-cp1 | flashmla_fwd_ref | 8196 | 7971-8458 | 0.50 | - | - | - |
| single-65536-hca-cp8r0 | tilelang | 2074 | 2053-2077 | 1.00 | 5434 | 5425-5441 | 1.00 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref | 835.2 | 832.6-836.7 | 0.40 | - | - | - |
| single-65536-hca-cp8r4 | tilelang | 2797 | 2781-2799 | 1.00 | 9582 | 9559-9586 | 1.00 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref | 1185 | 1183-1190 | 0.42 | - | - | - |
| single-65536-hca-cp8r7 | tilelang | 3390 | 3380-3403 | 1.00 | 12689 | 12685-12694 | 1.00 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref | 1384 | 1378-1390 | 0.41 | - | - | - |
| single-65536-sliding-cp1 | tilelang | 9650 | 9637-9666 | 1.00 | 30146 | 30143-30155 | 1.00 |
| single-65536-sliding-cp1 | flashmla_fwd_ref | 4122 | 4112-4130 | 0.43 | - | - | - |
| single-65536-sliding-cp8r0 | tilelang | 1871 | 1858-1899 | 1.00 | 4658 | 4655-4673 | 1.00 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref | 661.3 | 655.0-663.8 | 0.35 | - | - | - |
| single-65536-sliding-cp8r4 | tilelang | 1843 | 1830-1855 | 1.00 | 4694 | 4692-4707 | 1.00 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref | 660.7 | 659.4-664.0 | 0.36 | - | - | - |
| single-65536-sliding-cp8r7 | tilelang | 1852 | 1844-1863 | 1.00 | 4716 | 4715-4727 | 1.00 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref | 662.4 | 658.8-665.8 | 0.36 | - | - | - |
| short-65536-csa-cp1 | tilelang | 16242 | 16220-16289 | 1.00 | 63170 | 63160-63183 | 1.00 |
| short-65536-csa-cp1 | flashmla_fwd_ref | 8146 | 8080-8320 | 0.50 | - | - | - |
| short-65536-csa-cp8r0 | tilelang | 2287 | 2281-2291 | 1.00 | 6780 | 6771-6782 | 1.00 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref | 907.8 | 902.6-914.0 | 0.40 | - | - | - |
| short-65536-csa-cp8r4 | tilelang | 2862 | 2860-2870 | 1.00 | 10076 | 10047-10086 | 1.00 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref | 1170 | 1169-1173 | 0.41 | - | - | - |
| short-65536-csa-cp8r7 | tilelang | 2664 | 2649-2673 | 1.00 | 8942 | 8934-8970 | 1.00 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref | 1088 | 1087-1093 | 0.41 | - | - | - |
| short-65536-hca-cp1 | tilelang | 10922 | 10906-10936 | 1.00 | 33532 | 33528-33548 | 1.00 |
| short-65536-hca-cp1 | flashmla_fwd_ref | 5378 | 5372-5387 | 0.49 | - | - | - |
| short-65536-hca-cp8r0 | tilelang | 2020 | 2012-2043 | 1.00 | 5027 | 5026-5032 | 1.00 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref | 811.2 | 808.9-814.9 | 0.40 | - | - | - |
| short-65536-hca-cp8r4 | tilelang | 2054 | 2049-2060 | 1.00 | 5198 | 5193-5210 | 1.00 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref | 832.4 | 828.5-836.2 | 0.41 | - | - | - |
| short-65536-hca-cp8r7 | tilelang | 2040 | 2035-2045 | 1.00 | 5167 | 5154-5172 | 1.00 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref | 819.8 | 815.1-823.4 | 0.40 | - | - | - |
| short-65536-sliding-cp1 | tilelang | 9612 | 9605-9628 | 1.00 | 29641 | 29634-29651 | 1.00 |
| short-65536-sliding-cp1 | flashmla_fwd_ref | 4105 | 4097-4107 | 0.43 | - | - | - |
| short-65536-sliding-cp8r0 | tilelang | 1863 | 1854-1872 | 1.00 | 4557 | 4556-4558 | 1.00 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref | 665.9 | 661.3-668.2 | 0.36 | - | - | - |
| short-65536-sliding-cp8r4 | tilelang | 1838 | 1829-1848 | 1.00 | 4624 | 4622-4646 | 1.00 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref | 660.4 | 656.3-663.9 | 0.36 | - | - | - |
| short-65536-sliding-cp8r7 | tilelang | 1841 | 1834-1848 | 1.00 | 4633 | 4620-4639 | 1.00 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref | 662.2 | 661.1-666.2 | 0.36 | - | - | - |
| heavy-65536-csa-cp1 | tilelang | 17373 | 17342-17419 | 1.00 | 69696 | 69684-69701 | 1.00 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref | 8820 | 8665-9037 | 0.51 | - | - | - |
| heavy-65536-csa-cp8r0 | tilelang | 2238 | 2232-2242 | 1.00 | 6420 | 6417-6435 | 1.00 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref | 900.9 | 888.9-902.5 | 0.40 | - | - | - |
| heavy-65536-csa-cp8r4 | tilelang | 3403 | 3379-3442 | 1.00 | 12973 | 12936-12999 | 1.00 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref | 1402 | 1397-1405 | 0.41 | - | - | - |
| heavy-65536-csa-cp8r7 | tilelang | 2550 | 2543-2560 | 1.00 | 8299 | 8297-8303 | 1.00 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref | 1048 | 1046-1052 | 0.41 | - | - | - |
| heavy-65536-hca-cp1 | tilelang | 11714 | 11677-11736 | 1.00 | 38272 | 38241-38292 | 1.00 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref | 5682 | 5674-5693 | 0.49 | - | - | - |
| heavy-65536-hca-cp8r0 | tilelang | 1991 | 1967-1996 | 1.00 | 4821 | 4813-4828 | 1.00 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref | 777.0 | 775.5-781.9 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r4 | tilelang | 2378 | 2363-2384 | 1.00 | 7050 | 7045-7053 | 1.00 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref | 928.0 | 924.2-932.5 | 0.39 | - | - | - |
| heavy-65536-hca-cp8r7 | tilelang | 2006 | 2002-2013 | 1.00 | 5010 | 5008-5012 | 1.00 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref | 799.1 | 797.0-808.2 | 0.40 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang | 9616 | 9585-9660 | 1.00 | 29282 | 29271-29295 | 1.00 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref | 4111 | 4108-4115 | 0.43 | - | - | - |
| heavy-65536-sliding-cp8r0 | tilelang | 1853 | 1835-1865 | 1.00 | 4405 | 4399-4411 | 1.00 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref | 669.7 | 664.8-674.5 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r4 | tilelang | 1840 | 1825-1842 | 1.00 | 4678 | 4674-4682 | 1.00 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref | 655.9 | 655.0-657.4 | 0.36 | - | - | - |
| heavy-65536-sliding-cp8r7 | tilelang | 1824 | 1817-1827 | 1.00 | 4524 | 4521-4533 | 1.00 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref | 665.7 | 659.8-668.6 | 0.36 | - | - | - |
| tiny-65536-csa-cp1 | tilelang | 10884 | 10872-10885 | 1.00 | 31800 | 31799-31808 | 1.00 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref | 5659 | 5656-5664 | 0.52 | - | - | - |
| tiny-65536-csa-cp8r0 | tilelang | 2004 | 1988-2021 | 1.00 | 4891 | 4874-4905 | 1.00 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref | 809.2 | 808.8-818.0 | 0.40 | - | - | - |
| tiny-65536-csa-cp8r4 | tilelang | 2008 | 1980-2018 | 1.00 | 4867 | 4862-4872 | 1.00 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref | 823.2 | 820.0-828.9 | 0.41 | - | - | - |
| tiny-65536-csa-cp8r7 | tilelang | 2064 | 2023-2100 | 1.00 | 4878 | 4870-4892 | 1.00 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref | 831.3 | 809.5-891.6 | 0.40 | - | - | - |
| tiny-65536-hca-cp1 | tilelang | 8689 | 8676-8718 | 1.00 | 22022 | 21996-22029 | 1.00 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref | 4173 | 4168-4184 | 0.48 | - | - | - |
| tiny-65536-hca-cp8r0 | tilelang | 1706 | 1702-1728 | 1.00 | 3628 | 3625-3630 | 1.00 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref | 657.9 | 656.0-673.7 | 0.39 | - | - | - |
| tiny-65536-hca-cp8r4 | tilelang | 1732 | 1714-1738 | 1.00 | 3608 | 3597-3618 | 1.00 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref | 662.3 | 656.5-663.6 | 0.38 | - | - | - |
| tiny-65536-hca-cp8r7 | tilelang | 1716 | 1711-1718 | 1.00 | 3631 | 3624-3641 | 1.00 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref | 666.2 | 661.7-670.2 | 0.39 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang | 8717 | 8700-8727 | 1.00 | 22036 | 22028-22065 | 1.00 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref | 4178 | 4164-4186 | 0.48 | - | - | - |
| tiny-65536-sliding-cp8r0 | tilelang | 1745 | 1740-1764 | 1.00 | 3631 | 3625-3633 | 1.00 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref | 668.3 | 661.3-672.0 | 0.38 | - | - | - |
| tiny-65536-sliding-cp8r4 | tilelang | 1716 | 1698-1722 | 1.00 | 3613 | 3604-3624 | 1.00 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref | 662.2 | 654.3-664.8 | 0.39 | - | - | - |
| tiny-65536-sliding-cp8r7 | tilelang | 1705 | 1695-1721 | 1.00 | 3629 | 3621-3632 | 1.00 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref | 665.0 | 659.9-666.8 | 0.39 | - | - | - |

GPU busy time per call in µs from profiler traces (lower is better); `host` is op-boundary minus GPU
busy time (launch overhead and gaps); `/TL` divides GPU busy time by tilelang's; `peak MiB` is the
allocation above the inputs during one call, forward+backward where the backend has it, else forward.

| item | backend | fwd gpu µs | fwd host | fwd /TL | f+b gpu µs | f+b host | f+b /TL | peak MiB |
|---|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang | 539.5 | 657.7 | 1.00 | 2161 | 1025 | 1.00 | 265 |
| single-2048-csa-cp1 | flashmla_fwd_ref | 284.0 | 100.0 | 0.53 | - | - | - | 139 |
| single-2048-csa-cp8r0 | tilelang | 60.1 | 673.4 | 1.00 | 221.7 | 1780 | 1.00 | 40 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref | 44.0 | 138.4 | 0.73 | - | - | - | 17 |
| single-2048-csa-cp8r4 | tilelang | 84.4 | 679.0 | 1.00 | 356.6 | 1672 | 1.00 | 40 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref | 57.8 | 139.9 | 0.68 | - | - | - | 17 |
| single-2048-csa-cp8r7 | tilelang | 101.7 | 672.6 | 1.00 | 455.4 | 1569 | 1.00 | 40 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref | 64.1 | 138.7 | 0.63 | - | - | - | 17 |
| single-2048-hca-cp1 | tilelang | 362.9 | 722.3 | 1.00 | 1166 | 1338 | 1.00 | 265 |
| single-2048-hca-cp1 | flashmla_fwd_ref | 199.3 | 141.7 | 0.55 | - | - | - | 133 |
| single-2048-hca-cp8r0 | tilelang | 57.6 | 706.3 | 1.00 | 206.3 | 1945 | 1.00 | 38 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref | 41.6 | 147.1 | 0.72 | - | - | - | 17 |
| single-2048-hca-cp8r4 | tilelang | 61.4 | 731.7 | 1.00 | 219.3 | 1938 | 1.00 | 38 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref | 44.9 | 149.0 | 0.73 | - | - | - | 17 |
| single-2048-hca-cp8r7 | tilelang | 62.1 | 744.8 | 1.00 | 219.6 | 1958 | 1.00 | 38 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref | 45.2 | 148.3 | 0.73 | - | - | - | 17 |
| single-2048-sliding-cp1 | tilelang | 312.4 | 703.5 | 1.00 | 1033 | 1255 | 1.00 | 264 |
| single-2048-sliding-cp1 | flashmla_fwd_ref | 150.8 | 147.6 | 0.48 | - | - | - | 131 |
| single-2048-sliding-cp8r0 | tilelang | 51.4 | 681.1 | 1.00 | 190.3 | 1857 | 1.00 | 38 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref | 35.6 | 144.1 | 0.69 | - | - | - | 16 |
| single-2048-sliding-cp8r4 | tilelang | 53.4 | 686.1 | 1.00 | 195.9 | 1822 | 1.00 | 38 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref | 35.5 | 143.5 | 0.66 | - | - | - | 16 |
| single-2048-sliding-cp8r7 | tilelang | 53.2 | 686.3 | 1.00 | 195.3 | 1838 | 1.00 | 38 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref | 36.2 | 144.7 | 0.68 | - | - | - | 16 |
| short-2048-csa-cp1 | tilelang | 442.8 | 688.9 | 1.00 | 1638 | 1118 | 1.00 | 265 |
| short-2048-csa-cp1 | flashmla_fwd_ref | 234.8 | 133.2 | 0.53 | - | - | - | 139 |
| short-2048-csa-cp8r0 | tilelang | 60.1 | 693.0 | 1.00 | 221.8 | 1801 | 1.00 | 40 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref | 44.4 | 143.4 | 0.74 | - | - | - | 17 |
| short-2048-csa-cp8r4 | tilelang | 67.8 | 674.7 | 1.00 | 262.9 | 1770 | 1.00 | 40 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref | 48.9 | 138.8 | 0.72 | - | - | - | 17 |
| short-2048-csa-cp8r7 | tilelang | 76.4 | 685.8 | 1.00 | 319.2 | 1718 | 1.00 | 40 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref | 50.9 | 140.6 | 0.67 | - | - | - | 17 |
| short-2048-hca-cp1 | tilelang | 354.6 | 724.6 | 1.00 | 1143 | 1346 | 1.00 | 265 |
| short-2048-hca-cp1 | flashmla_fwd_ref | 196.7 | 143.8 | 0.55 | - | - | - | 133 |
| short-2048-hca-cp8r0 | tilelang | 57.2 | 730.2 | 1.00 | 205.4 | 1949 | 1.00 | 38 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref | 42.0 | 149.5 | 0.73 | - | - | - | 17 |
| short-2048-hca-cp8r4 | tilelang | 58.6 | 736.4 | 1.00 | 211.8 | 1950 | 1.00 | 38 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref | 43.2 | 150.1 | 0.74 | - | - | - | 17 |
| short-2048-hca-cp8r7 | tilelang | 62.0 | 730.7 | 1.00 | 219.8 | 1973 | 1.00 | 38 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref | 45.4 | 143.5 | 0.73 | - | - | - | 17 |
| short-2048-sliding-cp1 | tilelang | 307.0 | 695.7 | 1.00 | 1011 | 1319 | 1.00 | 264 |
| short-2048-sliding-cp1 | flashmla_fwd_ref | 147.7 | 145.6 | 0.48 | - | - | - | 131 |
| short-2048-sliding-cp8r0 | tilelang | 51.5 | 695.3 | 1.00 | 190.0 | 1852 | 1.00 | 38 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref | 36.3 | 145.5 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r4 | tilelang | 51.5 | 705.5 | 1.00 | 191.7 | 1831 | 1.00 | 38 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref | 36.2 | 153.6 | 0.70 | - | - | - | 16 |
| short-2048-sliding-cp8r7 | tilelang | 53.4 | 683.0 | 1.00 | 196.2 | 1848 | 1.00 | 38 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref | 35.6 | 149.5 | 0.67 | - | - | - | 16 |
| heavy-2048-csa-cp1 | tilelang | 476.0 | 694.7 | 1.00 | 1821 | 1134 | 1.00 | 265 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref | 254.4 | 135.3 | 0.53 | - | - | - | 139 |
| heavy-2048-csa-cp8r0 | tilelang | 60.1 | 687.4 | 1.00 | 214.2 | 1829 | 1.00 | 40 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref | 44.1 | 145.0 | 0.73 | - | - | - | 17 |
| heavy-2048-csa-cp8r4 | tilelang | 75.4 | 702.2 | 1.00 | 312.4 | 1752 | 1.00 | 40 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref | 51.7 | 149.0 | 0.69 | - | - | - | 17 |
| heavy-2048-csa-cp8r7 | tilelang | 93.5 | 699.7 | 1.00 | 410.4 | 1624 | 1.00 | 40 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref | 60.9 | 148.8 | 0.65 | - | - | - | 17 |
| heavy-2048-hca-cp1 | tilelang | 344.4 | 729.5 | 1.00 | 1109 | 1355 | 1.00 | 265 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref | 188.5 | 146.7 | 0.55 | - | - | - | 133 |
| heavy-2048-hca-cp8r0 | tilelang | 54.0 | 720.3 | 1.00 | 187.5 | 1948 | 1.00 | 38 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref | 41.0 | 148.3 | 0.76 | - | - | - | 17 |
| heavy-2048-hca-cp8r4 | tilelang | 61.8 | 720.2 | 1.00 | 219.6 | 1903 | 1.00 | 38 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref | 45.0 | 144.6 | 0.73 | - | - | - | 17 |
| heavy-2048-hca-cp8r7 | tilelang | 61.9 | 725.0 | 1.00 | 219.4 | 1938 | 1.00 | 38 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref | 45.1 | 144.7 | 0.73 | - | - | - | 17 |
| heavy-2048-sliding-cp1 | tilelang | 305.0 | 695.3 | 1.00 | 1000 | 1294 | 1.00 | 264 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref | 150.5 | 141.3 | 0.49 | - | - | - | 131 |
| heavy-2048-sliding-cp8r0 | tilelang | 49.9 | 686.1 | 1.00 | 180.0 | 1859 | 1.00 | 38 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref | 35.3 | 144.8 | 0.71 | - | - | - | 16 |
| heavy-2048-sliding-cp8r4 | tilelang | 53.5 | 688.8 | 1.00 | 195.8 | 1841 | 1.00 | 38 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref | 35.8 | 144.9 | 0.67 | - | - | - | 16 |
| heavy-2048-sliding-cp8r7 | tilelang | 53.2 | 690.4 | 1.00 | 195.4 | 1876 | 1.00 | 38 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref | 35.7 | 147.0 | 0.67 | - | - | - | 16 |
| tiny-2048-csa-cp1 | tilelang | 358.8 | 683.7 | 1.00 | 1100 | 1259 | 1.00 | 265 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref | 202.3 | 131.6 | 0.56 | - | - | - | 139 |
| tiny-2048-csa-cp8r0 | tilelang | 59.5 | 694.4 | 1.00 | 208.2 | 1848 | 1.00 | 40 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref | 44.4 | 148.0 | 0.75 | - | - | - | 17 |
| tiny-2048-csa-cp8r4 | tilelang | 59.6 | 688.9 | 1.00 | 206.9 | 1822 | 1.00 | 40 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref | 43.5 | 144.4 | 0.73 | - | - | - | 17 |
| tiny-2048-csa-cp8r7 | tilelang | 60.4 | 700.6 | 1.00 | 209.9 | 1830 | 1.00 | 40 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref | 44.1 | 145.9 | 0.73 | - | - | - | 17 |
| tiny-2048-hca-cp1 | tilelang | 270.8 | 691.8 | 1.00 | 772.0 | 1321 | 1.00 | 264 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref | 148.6 | 143.6 | 0.55 | - | - | - | 131 |
| tiny-2048-hca-cp8r0 | tilelang | 48.4 | 688.1 | 1.00 | 166.6 | 1863 | 1.00 | 38 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref | 35.5 | 147.1 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r4 | tilelang | 48.4 | 686.5 | 1.00 | 164.1 | 1885 | 1.00 | 38 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref | 35.5 | 145.4 | 0.73 | - | - | - | 16 |
| tiny-2048-hca-cp8r7 | tilelang | 50.0 | 701.3 | 1.00 | 175.8 | 1923 | 1.00 | 38 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref | 35.6 | 154.3 | 0.71 | - | - | - | 16 |
| tiny-2048-sliding-cp1 | tilelang | 271.0 | 686.8 | 1.00 | 772.5 | 1336 | 1.00 | 264 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref | 149.0 | 141.5 | 0.55 | - | - | - | 131 |
| tiny-2048-sliding-cp8r0 | tilelang | 48.3 | 683.4 | 1.00 | 165.9 | 1970 | 1.00 | 38 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref | 35.2 | 144.5 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r4 | tilelang | 48.3 | 690.2 | 1.00 | 164.1 | 1860 | 1.00 | 38 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref | 35.3 | 143.1 | 0.73 | - | - | - | 16 |
| tiny-2048-sliding-cp8r7 | tilelang | 50.1 | 708.0 | 1.00 | 175.6 | 1874 | 1.00 | 38 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref | 35.6 | 151.2 | 0.71 | - | - | - | 16 |
| single-4096-csa-cp1 | tilelang | 1208 | 682.8 | 1.00 | 5145 | 827.0 | 1.00 | 530 |
| single-4096-csa-cp1 | flashmla_fwd_ref | 593.7 | 121.7 | 0.49 | - | - | - | 278 |
| single-4096-csa-cp8r0 | tilelang | 112.0 | 683.6 | 1.00 | 410.0 | 1627 | 1.00 | 79 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref | 68.0 | 138.3 | 0.61 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang | 191.2 | 677.8 | 1.00 | 856.0 | 1477 | 1.00 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref | 109.5 | 138.9 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r7 | tilelang | 191.7 | 689.2 | 1.00 | 859.6 | 1490 | 1.00 | 79 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref | 109.4 | 142.6 | 0.57 | - | - | - | 35 |
| single-4096-hca-cp1 | tilelang | 691.9 | 740.4 | 1.00 | 2233 | 900.3 | 1.00 | 530 |
| single-4096-hca-cp1 | flashmla_fwd_ref | 366.3 | 139.2 | 0.53 | - | - | - | 266 |
| single-4096-hca-cp8r0 | tilelang | 102.0 | 726.2 | 1.00 | 349.1 | 1777 | 1.00 | 77 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref | 66.4 | 150.7 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang | 107.4 | 716.8 | 1.00 | 370.8 | 1772 | 1.00 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref | 70.2 | 146.6 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r7 | tilelang | 107.1 | 720.0 | 1.00 | 372.8 | 1786 | 1.00 | 77 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref | 68.8 | 148.4 | 0.64 | - | - | - | 33 |
| single-4096-sliding-cp1 | tilelang | 594.3 | 703.0 | 1.00 | 1945 | 914.2 | 1.00 | 527 |
| single-4096-sliding-cp1 | flashmla_fwd_ref | 273.9 | 146.3 | 0.46 | - | - | - | 262 |
| single-4096-sliding-cp8r0 | tilelang | 90.6 | 699.5 | 1.00 | 319.4 | 1715 | 1.00 | 76 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref | 53.1 | 146.2 | 0.59 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang | 92.4 | 698.6 | 1.00 | 327.4 | 1730 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref | 53.7 | 149.4 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r7 | tilelang | 92.2 | 693.0 | 1.00 | 327.1 | 1738 | 1.00 | 76 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref | 53.9 | 140.2 | 0.58 | - | - | - | 33 |
| short-4096-csa-cp1 | tilelang | 829.2 | 694.2 | 1.00 | 3054 | 819.3 | 1.00 | 530 |
| short-4096-csa-cp1 | flashmla_fwd_ref | 428.1 | 124.9 | 0.52 | - | - | - | 278 |
| short-4096-csa-cp8r0 | tilelang | 112.0 | 715.7 | 1.00 | 410.6 | 1643 | 1.00 | 79 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref | 68.3 | 154.1 | 0.61 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang | 114.7 | 762.2 | 1.00 | 432.7 | 1712 | 1.00 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref | 72.3 | 154.8 | 0.63 | - | - | - | 35 |
| short-4096-csa-cp8r7 | tilelang | 112.8 | 705.3 | 1.00 | 430.6 | 1629 | 1.00 | 79 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref | 68.9 | 141.8 | 0.61 | - | - | - | 35 |
| short-4096-hca-cp1 | tilelang | 666.3 | 744.5 | 1.00 | 2131 | 958.7 | 1.00 | 530 |
| short-4096-hca-cp1 | flashmla_fwd_ref | 359.6 | 140.5 | 0.54 | - | - | - | 266 |
| short-4096-hca-cp8r0 | tilelang | 100.2 | 744.3 | 1.00 | 347.4 | 1833 | 1.00 | 77 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref | 64.7 | 151.3 | 0.65 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang | 101.5 | 725.0 | 1.00 | 354.4 | 1803 | 1.00 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref | 65.3 | 142.9 | 0.64 | - | - | - | 33 |
| short-4096-hca-cp8r7 | tilelang | 102.6 | 728.6 | 1.00 | 359.5 | 1837 | 1.00 | 77 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref | 67.0 | 147.2 | 0.65 | - | - | - | 33 |
| short-4096-sliding-cp1 | tilelang | 584.0 | 727.8 | 1.00 | 1896 | 946.0 | 1.00 | 527 |
| short-4096-sliding-cp1 | flashmla_fwd_ref | 271.4 | 151.8 | 0.46 | - | - | - | 262 |
| short-4096-sliding-cp8r0 | tilelang | 89.3 | 704.5 | 1.00 | 317.4 | 1719 | 1.00 | 76 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref | 53.0 | 144.4 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang | 89.9 | 704.2 | 1.00 | 319.7 | 1722 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref | 53.1 | 144.6 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r7 | tilelang | 90.8 | 684.0 | 1.00 | 323.9 | 1762 | 1.00 | 76 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref | 54.0 | 141.7 | 0.59 | - | - | - | 33 |
| heavy-4096-csa-cp1 | tilelang | 770.8 | 691.6 | 1.00 | 2718 | 827.3 | 1.00 | 530 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref | 404.9 | 120.0 | 0.53 | - | - | - | 278 |
| heavy-4096-csa-cp8r0 | tilelang | 111.8 | 682.7 | 1.00 | 410.6 | 1622 | 1.00 | 79 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref | 68.1 | 140.3 | 0.61 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang | 114.6 | 703.3 | 1.00 | 434.1 | 1615 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref | 72.1 | 145.2 | 0.63 | - | - | - | 35 |
| heavy-4096-csa-cp8r7 | tilelang | 105.7 | 744.1 | 1.00 | 378.3 | 1711 | 1.00 | 79 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref | 68.6 | 147.1 | 0.65 | - | - | - | 35 |
| heavy-4096-hca-cp1 | tilelang | 647.0 | 727.5 | 1.00 | 2030 | 952.6 | 1.00 | 530 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref | 343.9 | 138.8 | 0.53 | - | - | - | 266 |
| heavy-4096-hca-cp8r0 | tilelang | 100.5 | 735.0 | 1.00 | 348.5 | 1804 | 1.00 | 77 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref | 65.0 | 149.5 | 0.65 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang | 102.2 | 725.7 | 1.00 | 357.1 | 1803 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref | 64.6 | 142.8 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r7 | tilelang | 97.6 | 740.6 | 1.00 | 338.4 | 1841 | 1.00 | 77 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref | 61.8 | 154.0 | 0.63 | - | - | - | 33 |
| heavy-4096-sliding-cp1 | tilelang | 582.4 | 700.3 | 1.00 | 1840 | 1071 | 1.00 | 527 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref | 269.1 | 149.7 | 0.46 | - | - | - | 262 |
| heavy-4096-sliding-cp8r0 | tilelang | 88.9 | 704.9 | 1.00 | 318.0 | 1765 | 1.00 | 76 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref | 52.5 | 146.2 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang | 90.6 | 690.0 | 1.00 | 321.4 | 1729 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref | 53.9 | 140.4 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r7 | tilelang | 87.9 | 684.1 | 1.00 | 309.0 | 1790 | 1.00 | 76 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref | 53.5 | 144.1 | 0.61 | - | - | - | 33 |
| tiny-4096-csa-cp1 | tilelang | 680.3 | 693.4 | 1.00 | 2071 | 848.0 | 1.00 | 530 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref | 372.6 | 128.9 | 0.55 | - | - | - | 278 |
| tiny-4096-csa-cp8r0 | tilelang | 103.6 | 692.9 | 1.00 | 340.7 | 1680 | 1.00 | 79 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref | 66.6 | 144.3 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang | 104.5 | 704.8 | 1.00 | 345.8 | 1688 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref | 67.1 | 142.5 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r7 | tilelang | 104.5 | 681.6 | 1.00 | 346.0 | 1697 | 1.00 | 79 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref | 67.8 | 142.8 | 0.65 | - | - | - | 35 |
| tiny-4096-hca-cp1 | tilelang | 532.2 | 695.1 | 1.00 | 1436 | 973.1 | 1.00 | 527 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref | 273.7 | 138.6 | 0.51 | - | - | - | 262 |
| tiny-4096-hca-cp8r0 | tilelang | 81.8 | 688.6 | 1.00 | 254.2 | 1778 | 1.00 | 76 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref | 52.5 | 142.3 | 0.64 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang | 84.2 | 683.2 | 1.00 | 267.7 | 1755 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref | 53.3 | 143.4 | 0.63 | - | - | - | 33 |
| tiny-4096-hca-cp8r7 | tilelang | 82.2 | 701.9 | 1.00 | 260.4 | 1793 | 1.00 | 76 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref | 52.8 | 148.7 | 0.64 | - | - | - | 33 |
| tiny-4096-sliding-cp1 | tilelang | 534.3 | 692.2 | 1.00 | 1439 | 968.5 | 1.00 | 527 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref | 269.6 | 139.2 | 0.50 | - | - | - | 262 |
| tiny-4096-sliding-cp8r0 | tilelang | 81.9 | 693.4 | 1.00 | 254.8 | 1776 | 1.00 | 76 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref | 52.1 | 148.6 | 0.64 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang | 84.0 | 697.6 | 1.00 | 267.6 | 1772 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref | 52.9 | 150.1 | 0.63 | - | - | - | 33 |
| tiny-4096-sliding-cp8r7 | tilelang | 82.0 | 704.4 | 1.00 | 259.8 | 1821 | 1.00 | 76 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref | 53.1 | 149.0 | 0.65 | - | - | - | 33 |
| single-16384-csa-cp1 | tilelang | 5358 | 559.6 | 1.00 | 22675 | 885.3 | 1.00 | 2120 |
| single-16384-csa-cp1 | flashmla_fwd_ref | 2542 | 28.1 | 0.47 | - | - | - | 1112 |
| single-16384-csa-cp8r0 | tilelang | 539.8 | 692.6 | 1.00 | 2182 | 1062 | 1.00 | 318 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref | 283.6 | 134.0 | 0.53 | - | - | - | 139 |
| single-16384-csa-cp8r4 | tilelang | 710.9 | 698.0 | 1.00 | 3186 | 893.3 | 1.00 | 318 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref | 359.6 | 135.6 | 0.51 | - | - | - | 139 |
| single-16384-csa-cp8r7 | tilelang | 711.0 | 694.1 | 1.00 | 3201 | 903.7 | 1.00 | 318 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref | 358.9 | 137.7 | 0.50 | - | - | - | 139 |
| single-16384-hca-cp1 | tilelang | 2873 | 700.6 | 1.00 | 10008 | 896.7 | 1.00 | 2108 |
| single-16384-hca-cp1 | flashmla_fwd_ref | 1399 | 90.2 | 0.49 | - | - | - | 1064 |
| single-16384-hca-cp8r0 | tilelang | 360.1 | 720.7 | 1.00 | 1178 | 1245 | 1.00 | 306 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref | 196.9 | 143.6 | 0.55 | - | - | - | 133 |
| single-16384-hca-cp8r4 | tilelang | 411.5 | 685.4 | 1.00 | 1468 | 1197 | 1.00 | 306 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref | 200.6 | 139.3 | 0.49 | - | - | - | 133 |
| single-16384-hca-cp8r7 | tilelang | 416.3 | 690.1 | 1.00 | 1589 | 1230 | 1.00 | 306 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref | 200.7 | 143.1 | 0.48 | - | - | - | 133 |
| single-16384-sliding-cp1 | tilelang | 2290 | 682.5 | 1.00 | 7470 | 816.3 | 1.00 | 2108 |
| single-16384-sliding-cp1 | flashmla_fwd_ref | 1049 | 115.4 | 0.46 | - | - | - | 1048 |
| single-16384-sliding-cp8r0 | tilelang | 310.8 | 707.7 | 1.00 | 1046 | 1288 | 1.00 | 306 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref | 150.5 | 149.3 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r4 | tilelang | 315.2 | 689.1 | 1.00 | 1080 | 1310 | 1.00 | 306 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref | 150.5 | 146.2 | 0.48 | - | - | - | 131 |
| single-16384-sliding-cp8r7 | tilelang | 314.2 | 682.0 | 1.00 | 1084 | 1305 | 1.00 | 306 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref | 150.6 | 143.4 | 0.48 | - | - | - | 131 |
| short-16384-csa-cp1 | tilelang | 3853 | 656.0 | 1.00 | 15075 | 901.5 | 1.00 | 2120 |
| short-16384-csa-cp1 | flashmla_fwd_ref | 1929 | 28.0 | 0.50 | - | - | - | 1112 |
| short-16384-csa-cp8r0 | tilelang | 478.1 | 697.2 | 1.00 | 1852 | 1145 | 1.00 | 317 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref | 254.2 | 135.9 | 0.53 | - | - | - | 139 |
| short-16384-csa-cp8r4 | tilelang | 566.4 | 683.1 | 1.00 | 2358 | 1029 | 1.00 | 317 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref | 295.5 | 136.1 | 0.52 | - | - | - | 139 |
| short-16384-csa-cp8r7 | tilelang | 524.8 | 685.9 | 1.00 | 2140 | 1101 | 1.00 | 317 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref | 276.9 | 131.2 | 0.53 | - | - | - | 139 |
| short-16384-hca-cp1 | tilelang | 2599 | 738.1 | 1.00 | 8299 | 926.1 | 1.00 | 2120 |
| short-16384-hca-cp1 | flashmla_fwd_ref | 1361 | 102.2 | 0.52 | - | - | - | 1064 |
| short-16384-hca-cp8r0 | tilelang | 349.8 | 720.5 | 1.00 | 1148 | 1351 | 1.00 | 307 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref | 196.9 | 142.0 | 0.56 | - | - | - | 133 |
| short-16384-hca-cp8r4 | tilelang | 362.1 | 717.9 | 1.00 | 1211 | 1322 | 1.00 | 307 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref | 200.7 | 136.7 | 0.55 | - | - | - | 133 |
| short-16384-hca-cp8r7 | tilelang | 355.1 | 726.0 | 1.00 | 1179 | 1380 | 1.00 | 307 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref | 198.1 | 140.8 | 0.56 | - | - | - | 133 |
| short-16384-sliding-cp1 | tilelang | 2264 | 720.8 | 1.00 | 7323 | 832.1 | 1.00 | 2108 |
| short-16384-sliding-cp1 | flashmla_fwd_ref | 1041 | 128.7 | 0.46 | - | - | - | 1048 |
| short-16384-sliding-cp8r0 | tilelang | 303.6 | 697.8 | 1.00 | 1025 | 1281 | 1.00 | 306 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref | 149.7 | 146.6 | 0.49 | - | - | - | 131 |
| short-16384-sliding-cp8r4 | tilelang | 314.2 | 679.7 | 1.00 | 1072 | 1298 | 1.00 | 306 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref | 147.1 | 147.3 | 0.47 | - | - | - | 131 |
| short-16384-sliding-cp8r7 | tilelang | 312.3 | 689.1 | 1.00 | 1051 | 1310 | 1.00 | 306 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref | 150.9 | 143.6 | 0.48 | - | - | - | 131 |
| heavy-16384-csa-cp1 | tilelang | 3305 | 664.5 | 1.00 | 12082 | 914.7 | 1.00 | 2120 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref | 1691 | 34.9 | 0.51 | - | - | - | 1112 |
| heavy-16384-csa-cp8r0 | tilelang | 395.5 | 697.0 | 1.00 | 1364 | 1211 | 1.00 | 317 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref | 213.5 | 133.0 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r4 | tilelang | 433.6 | 683.1 | 1.00 | 1614 | 1148 | 1.00 | 317 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref | 232.8 | 129.9 | 0.54 | - | - | - | 139 |
| heavy-16384-csa-cp8r7 | tilelang | 579.5 | 684.9 | 1.00 | 2413 | 1069 | 1.00 | 317 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref | 301.6 | 126.9 | 0.52 | - | - | - | 139 |
| heavy-16384-hca-cp1 | tilelang | 2507 | 744.8 | 1.00 | 7784 | 921.8 | 1.00 | 2120 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref | 1309 | 94.9 | 0.52 | - | - | - | 1064 |
| heavy-16384-hca-cp8r0 | tilelang | 333.5 | 734.1 | 1.00 | 1069 | 1379 | 1.00 | 308 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref | 184.9 | 150.6 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r4 | tilelang | 343.8 | 725.9 | 1.00 | 1127 | 1361 | 1.00 | 308 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref | 190.5 | 151.2 | 0.55 | - | - | - | 133 |
| heavy-16384-hca-cp8r7 | tilelang | 366.8 | 721.2 | 1.00 | 1231 | 1390 | 1.00 | 308 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref | 203.8 | 142.3 | 0.56 | - | - | - | 133 |
| heavy-16384-sliding-cp1 | tilelang | 2247 | 731.8 | 1.00 | 7046 | 841.1 | 1.00 | 2108 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref | 1042 | 130.6 | 0.46 | - | - | - | 1048 |
| heavy-16384-sliding-cp8r0 | tilelang | 300.8 | 712.1 | 1.00 | 974.6 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref | 148.3 | 149.4 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r4 | tilelang | 303.7 | 679.0 | 1.00 | 1014 | 1317 | 1.00 | 306 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref | 149.5 | 143.6 | 0.49 | - | - | - | 131 |
| heavy-16384-sliding-cp8r7 | tilelang | 314.8 | 707.2 | 1.00 | 1082 | 1323 | 1.00 | 306 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref | 147.4 | 149.0 | 0.47 | - | - | - | 131 |
| tiny-16384-csa-cp1 | tilelang | 2636 | 650.2 | 1.00 | 7892 | 833.7 | 1.00 | 2120 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref | 1437 | 39.5 | 0.54 | - | - | - | 1112 |
| tiny-16384-csa-cp8r0 | tilelang | 354.2 | 688.8 | 1.00 | 1109 | 1267 | 1.00 | 317 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref | 200.1 | 134.4 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r4 | tilelang | 356.2 | 692.0 | 1.00 | 1108 | 1250 | 1.00 | 317 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref | 200.5 | 141.7 | 0.56 | - | - | - | 139 |
| tiny-16384-csa-cp8r7 | tilelang | 356.2 | 675.8 | 1.00 | 1099 | 1270 | 1.00 | 317 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref | 200.3 | 134.2 | 0.56 | - | - | - | 139 |
| tiny-16384-hca-cp1 | tilelang | 2024 | 683.5 | 1.00 | 5355 | 828.5 | 1.00 | 2108 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref | 1052 | 124.0 | 0.52 | - | - | - | 1048 |
| tiny-16384-hca-cp8r0 | tilelang | 278.4 | 702.7 | 1.00 | 784.0 | 1360 | 1.00 | 306 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref | 149.1 | 147.5 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r4 | tilelang | 276.8 | 691.4 | 1.00 | 771.3 | 1324 | 1.00 | 306 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref | 148.8 | 144.3 | 0.54 | - | - | - | 131 |
| tiny-16384-hca-cp8r7 | tilelang | 268.8 | 686.6 | 1.00 | 754.2 | 1376 | 1.00 | 306 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref | 149.0 | 145.3 | 0.55 | - | - | - | 131 |
| tiny-16384-sliding-cp1 | tilelang | 2025 | 703.4 | 1.00 | 5346 | 817.0 | 1.00 | 2108 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref | 1050 | 130.1 | 0.52 | - | - | - | 1048 |
| tiny-16384-sliding-cp8r0 | tilelang | 278.4 | 684.2 | 1.00 | 784.9 | 1357 | 1.00 | 306 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref | 147.7 | 149.3 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r4 | tilelang | 276.5 | 685.8 | 1.00 | 772.0 | 1402 | 1.00 | 306 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref | 147.7 | 151.4 | 0.53 | - | - | - | 131 |
| tiny-16384-sliding-cp8r7 | tilelang | 267.7 | 693.4 | 1.00 | 754.5 | 1385 | 1.00 | 306 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref | 148.1 | 145.0 | 0.55 | - | - | - | 131 |
| single-49208-csa-cp1 | tilelang | 16237 | 187.0 | 1.00 | 69993 | 715.1 | 1.00 | 6368 |
| single-49208-csa-cp1 | flashmla_fwd_ref | 7737 | 571.7 | 0.48 | - | - | - | 3340 |
| single-49208-csa-cp8r0 | tilelang | 1866 | 689.5 | 1.00 | 8084 | 839.4 | 1.00 | 954 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref | 922.1 | 104.1 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r4 | tilelang | 2035 | 709.5 | 1.00 | 9123 | 902.5 | 1.00 | 954 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref | 999.9 | 100.6 | 0.49 | - | - | - | 418 |
| single-49208-csa-cp8r7 | tilelang | 2039 | 705.7 | 1.00 | 9444 | 917.3 | 1.00 | 954 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref | 1003 | 101.6 | 0.49 | - | - | - | 418 |
| single-49208-hca-cp1 | tilelang | 11188 | 307.9 | 1.00 | 41810 | 748.7 | 1.00 | 6334 |
| single-49208-hca-cp1 | flashmla_fwd_ref | 5350 | 51.8 | 0.48 | - | - | - | 3292 |
| single-49208-hca-cp8r0 | tilelang | 1029 | 680.6 | 1.00 | 3422 | 825.2 | 1.00 | 919 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref | 555.3 | 115.3 | 0.54 | - | - | - | 412 |
| single-49208-hca-cp8r4 | tilelang | 1459 | 686.0 | 1.00 | 5739 | 825.7 | 1.00 | 919 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref | 691.9 | 109.3 | 0.47 | - | - | - | 412 |
| single-49208-hca-cp8r7 | tilelang | 1762 | 668.0 | 1.00 | 7389 | 854.6 | 1.00 | 919 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref | 831.5 | 113.1 | 0.47 | - | - | - | 412 |
| single-49208-sliding-cp1 | tilelang | 6768 | 664.9 | 1.00 | 22045 | 843.6 | 1.00 | 6332 |
| single-49208-sliding-cp1 | flashmla_fwd_ref | 3083 | 41.8 | 0.46 | - | - | - | 3148 |
| single-49208-sliding-cp8r0 | tilelang | 873.5 | 700.3 | 1.00 | 2906 | 838.8 | 1.00 | 918 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref | 402.7 | 140.4 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r4 | tilelang | 878.6 | 678.7 | 1.00 | 2949 | 833.9 | 1.00 | 918 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref | 404.1 | 135.1 | 0.46 | - | - | - | 394 |
| single-49208-sliding-cp8r7 | tilelang | 880.4 | 691.7 | 1.00 | 2941 | 850.3 | 1.00 | 918 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref | 401.2 | 139.0 | 0.46 | - | - | - | 394 |
| short-49208-csa-cp1 | tilelang | 12688 | 299.8 | 1.00 | 50333 | 736.6 | 1.00 | 6368 |
| short-49208-csa-cp1 | flashmla_fwd_ref | 6166 | 60.7 | 0.49 | - | - | - | 3340 |
| short-49208-csa-cp8r0 | tilelang | 1344 | 692.7 | 1.00 | 5188 | 824.3 | 1.00 | 954 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref | 686.4 | 104.7 | 0.51 | - | - | - | 418 |
| short-49208-csa-cp8r4 | tilelang | 1711 | 674.8 | 1.00 | 7251 | 834.8 | 1.00 | 954 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref | 852.5 | 107.1 | 0.50 | - | - | - | 418 |
| short-49208-csa-cp8r7 | tilelang | 1484 | 676.8 | 1.00 | 5954 | 855.8 | 1.00 | 954 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref | 756.8 | 101.5 | 0.51 | - | - | - | 418 |
| short-49208-hca-cp1 | tilelang | 7876 | 628.9 | 1.00 | 25343 | 831.9 | 1.00 | 6382 |
| short-49208-hca-cp1 | flashmla_fwd_ref | 4089 | 14.0 | 0.52 | - | - | - | 3197 |
| short-49208-hca-cp8r0 | tilelang | 989.7 | 710.9 | 1.00 | 3199 | 869.2 | 1.00 | 925 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref | 521.4 | 125.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r4 | tilelang | 1004 | 742.6 | 1.00 | 3328 | 858.1 | 1.00 | 925 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref | 529.1 | 136.2 | 0.53 | - | - | - | 400 |
| short-49208-hca-cp8r7 | tilelang | 1012 | 720.8 | 1.00 | 3277 | 892.2 | 1.00 | 925 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref | 532.2 | 132.3 | 0.53 | - | - | - | 400 |
| short-49208-sliding-cp1 | tilelang | 6741 | 681.3 | 1.00 | 21763 | 869.5 | 1.00 | 6332 |
| short-49208-sliding-cp1 | flashmla_fwd_ref | 3084 | 39.0 | 0.46 | - | - | - | 3148 |
| short-49208-sliding-cp8r0 | tilelang | 864.5 | 700.0 | 1.00 | 2838 | 825.1 | 1.00 | 918 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref | 397.8 | 143.2 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r4 | tilelang | 869.7 | 699.7 | 1.00 | 2893 | 846.4 | 1.00 | 918 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref | 402.9 | 142.9 | 0.46 | - | - | - | 394 |
| short-49208-sliding-cp8r7 | tilelang | 872.3 | 683.9 | 1.00 | 2886 | 857.6 | 1.00 | 918 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref | 405.0 | 131.9 | 0.46 | - | - | - | 394 |
| heavy-49208-csa-cp1 | tilelang | 14881 | 32.9 | 1.00 | 60982 | 725.7 | 1.00 | 6368 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref | 7056 | 338.1 | 0.47 | - | - | - | 3340 |
| heavy-49208-csa-cp8r0 | tilelang | 1239 | 685.7 | 1.00 | 4603 | 816.8 | 1.00 | 954 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref | 650.2 | 109.6 | 0.52 | - | - | - | 418 |
| heavy-49208-csa-cp8r4 | tilelang | 1940 | 688.9 | 1.00 | 8493 | 875.9 | 1.00 | 954 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref | 953.3 | 104.7 | 0.49 | - | - | - | 418 |
| heavy-49208-csa-cp8r7 | tilelang | 1792 | 685.2 | 1.00 | 7721 | 860.7 | 1.00 | 954 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref | 889.3 | 101.1 | 0.50 | - | - | - | 418 |
| heavy-49208-hca-cp1 | tilelang | 8488 | 600.9 | 1.00 | 28842 | 762.6 | 1.00 | 6394 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref | 4251 | 22.7 | 0.50 | - | - | - | 3244 |
| heavy-49208-hca-cp8r0 | tilelang | 956.8 | 745.4 | 1.00 | 3055 | 857.0 | 1.00 | 926 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref | 510.4 | 132.4 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r4 | tilelang | 1044 | 709.2 | 1.00 | 3546 | 847.7 | 1.00 | 926 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref | 550.7 | 124.3 | 0.53 | - | - | - | 406 |
| heavy-49208-hca-cp8r7 | tilelang | 1105 | 739.6 | 1.00 | 3792 | 880.6 | 1.00 | 926 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref | 591.4 | 120.7 | 0.54 | - | - | - | 406 |
| heavy-49208-sliding-cp1 | tilelang | 6750 | 697.8 | 1.00 | 21733 | 849.9 | 1.00 | 6332 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref | 3069 | 65.7 | 0.45 | - | - | - | 3148 |
| heavy-49208-sliding-cp8r0 | tilelang | 861.4 | 710.5 | 1.00 | 2755 | 833.2 | 1.00 | 918 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref | 402.5 | 139.7 | 0.47 | - | - | - | 394 |
| heavy-49208-sliding-cp8r4 | tilelang | 877.6 | 687.8 | 1.00 | 2942 | 819.4 | 1.00 | 918 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref | 403.9 | 135.0 | 0.46 | - | - | - | 394 |
| heavy-49208-sliding-cp8r7 | tilelang | 882.7 | 684.6 | 1.00 | 2924 | 841.8 | 1.00 | 918 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref | 402.0 | 141.2 | 0.46 | - | - | - | 394 |
| tiny-49208-csa-cp1 | tilelang | 7840 | 553.1 | 1.00 | 23410 | 693.2 | 1.00 | 6368 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref | 4272 | 23.0 | 0.54 | - | - | - | 3340 |
| tiny-49208-csa-cp8r0 | tilelang | 998.7 | 705.4 | 1.00 | 3072 | 802.9 | 1.00 | 953 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref | 556.1 | 102.0 | 0.56 | - | - | - | 418 |
| tiny-49208-csa-cp8r4 | tilelang | 1012 | 668.1 | 1.00 | 3097 | 809.2 | 1.00 | 953 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref | 554.1 | 102.4 | 0.55 | - | - | - | 418 |
| tiny-49208-csa-cp8r7 | tilelang | 1007 | 682.4 | 1.00 | 3080 | 832.5 | 1.00 | 953 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref | 560.0 | 99.4 | 0.56 | - | - | - | 418 |
| tiny-49208-hca-cp1 | tilelang | 6058 | 685.3 | 1.00 | 15953 | 833.4 | 1.00 | 6332 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref | 3126 | 38.3 | 0.52 | - | - | - | 3148 |
| tiny-49208-hca-cp8r0 | tilelang | 774.9 | 704.2 | 1.00 | 2125 | 827.8 | 1.00 | 918 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref | 401.6 | 139.8 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r4 | tilelang | 783.0 | 701.8 | 1.00 | 2153 | 827.3 | 1.00 | 918 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref | 404.4 | 139.6 | 0.52 | - | - | - | 394 |
| tiny-49208-hca-cp8r7 | tilelang | 780.8 | 692.8 | 1.00 | 2111 | 841.6 | 1.00 | 918 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref | 401.6 | 139.0 | 0.51 | - | - | - | 394 |
| tiny-49208-sliding-cp1 | tilelang | 6059 | 674.1 | 1.00 | 15952 | 849.2 | 1.00 | 6332 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref | 3124 | 48.6 | 0.52 | - | - | - | 3148 |
| tiny-49208-sliding-cp8r0 | tilelang | 773.6 | 689.3 | 1.00 | 2123 | 823.5 | 1.00 | 918 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref | 403.5 | 136.8 | 0.52 | - | - | - | 394 |
| tiny-49208-sliding-cp8r4 | tilelang | 785.5 | 699.1 | 1.00 | 2156 | 819.1 | 1.00 | 918 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref | 402.9 | 139.6 | 0.51 | - | - | - | 394 |
| tiny-49208-sliding-cp8r7 | tilelang | 779.6 | 685.1 | 1.00 | 2107 | 843.6 | 1.00 | 918 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref | 406.0 | 136.3 | 0.52 | - | - | - | 394 |
| single-65536-csa-cp1 | tilelang | 21804 | -109.6 | 1.00 | 94799 | 644.7 | 1.00 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref | 10352 | 1033 | 0.47 | - | - | - | 4448 |
| single-65536-csa-cp8r0 | tilelang | 2543 | 722.8 | 1.00 | 11048 | 882.7 | 1.00 | 1270 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref | 1248 | 76.7 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r4 | tilelang | 2711 | 717.6 | 1.00 | 12259 | 886.8 | 1.00 | 1270 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref | 1324 | 86.3 | 0.49 | - | - | - | 556 |
| single-65536-csa-cp8r7 | tilelang | 2709 | 675.0 | 1.00 | 13123 | 872.4 | 1.00 | 1270 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref | 1331 | 78.8 | 0.49 | - | - | - | 556 |
| single-65536-hca-cp1 | tilelang | 16321 | 181.6 | 1.00 | 63704 | 622.9 | 1.00 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref | 7958 | 238.2 | 0.49 | - | - | - | 4448 |
| single-65536-hca-cp8r0 | tilelang | 1365 | 708.7 | 1.00 | 4606 | 828.1 | 1.00 | 1224 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref | 751.0 | 84.2 | 0.55 | - | - | - | 556 |
| single-65536-hca-cp8r4 | tilelang | 2123 | 673.9 | 1.00 | 8699 | 883.4 | 1.00 | 1224 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref | 1114 | 70.8 | 0.52 | - | - | - | 556 |
| single-65536-hca-cp8r7 | tilelang | 2715 | 674.6 | 1.00 | 11760 | 929.3 | 1.00 | 1224 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref | 1305 | 79.3 | 0.48 | - | - | - | 556 |
| single-65536-sliding-cp1 | tilelang | 8998 | 652.7 | 1.00 | 29308 | 838.2 | 1.00 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref | 4089 | 32.9 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp8r0 | tilelang | 1148 | 723.0 | 1.00 | 3834 | 823.8 | 1.00 | 1222 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref | 522.4 | 139.0 | 0.45 | - | - | - | 524 |
| single-65536-sliding-cp8r4 | tilelang | 1153 | 690.8 | 1.00 | 3867 | 826.9 | 1.00 | 1222 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref | 533.2 | 127.5 | 0.46 | - | - | - | 524 |
| single-65536-sliding-cp8r7 | tilelang | 1154 | 698.5 | 1.00 | 3861 | 855.4 | 1.00 | 1222 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref | 529.4 | 133.0 | 0.46 | - | - | - | 524 |
| short-65536-csa-cp1 | tilelang | 16000 | 242.3 | 1.00 | 62509 | 660.3 | 1.00 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref | 7830 | 315.8 | 0.49 | - | - | - | 4448 |
| short-65536-csa-cp8r0 | tilelang | 1610 | 676.7 | 1.00 | 5968 | 811.5 | 1.00 | 1270 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref | 825.1 | 82.7 | 0.51 | - | - | - | 556 |
| short-65536-csa-cp8r4 | tilelang | 2195 | 666.7 | 1.00 | 9179 | 896.3 | 1.00 | 1270 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref | 1090 | 79.7 | 0.50 | - | - | - | 556 |
| short-65536-csa-cp8r7 | tilelang | 1997 | 667.0 | 1.00 | 8087 | 855.0 | 1.00 | 1270 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref | 1012 | 76.5 | 0.51 | - | - | - | 556 |
| short-65536-hca-cp1 | tilelang | 10309 | 612.7 | 1.00 | 32762 | 770.6 | 1.00 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref | 5376 | 1.8 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp8r0 | tilelang | 1296 | 723.7 | 1.00 | 4168 | 858.9 | 1.00 | 1229 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref | 687.6 | 123.7 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r4 | tilelang | 1337 | 717.1 | 1.00 | 4340 | 857.5 | 1.00 | 1229 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref | 704.0 | 128.4 | 0.53 | - | - | - | 532 |
| short-65536-hca-cp8r7 | tilelang | 1320 | 720.0 | 1.00 | 4272 | 894.3 | 1.00 | 1229 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref | 702.8 | 117.0 | 0.53 | - | - | - | 532 |
| short-65536-sliding-cp1 | tilelang | 8974 | 637.9 | 1.00 | 28844 | 797.0 | 1.00 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref | 4081 | 23.9 | 0.45 | - | - | - | 4192 |
| short-65536-sliding-cp8r0 | tilelang | 1138 | 725.1 | 1.00 | 3714 | 842.5 | 1.00 | 1222 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref | 524.0 | 141.9 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r4 | tilelang | 1149 | 688.8 | 1.00 | 3813 | 810.9 | 1.00 | 1222 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref | 526.8 | 133.6 | 0.46 | - | - | - | 524 |
| short-65536-sliding-cp8r7 | tilelang | 1153 | 688.0 | 1.00 | 3777 | 855.3 | 1.00 | 1222 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref | 526.2 | 135.9 | 0.46 | - | - | - | 524 |
| heavy-65536-csa-cp1 | tilelang | 17414 | -41.5 | 1.00 | 69035 | 660.4 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref | 8439 | 381.8 | 0.48 | - | - | - | 4448 |
| heavy-65536-csa-cp8r0 | tilelang | 1561 | 676.9 | 1.00 | 5628 | 791.6 | 1.00 | 1270 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref | 818.9 | 82.0 | 0.52 | - | - | - | 556 |
| heavy-65536-csa-cp8r4 | tilelang | 2719 | 683.2 | 1.00 | 12075 | 898.7 | 1.00 | 1270 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref | 1326 | 75.9 | 0.49 | - | - | - | 556 |
| heavy-65536-csa-cp8r7 | tilelang | 1888 | 661.1 | 1.00 | 7479 | 820.1 | 1.00 | 1270 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref | 972.7 | 74.8 | 0.52 | - | - | - | 556 |
| heavy-65536-hca-cp1 | tilelang | 11197 | 516.8 | 1.00 | 37594 | 678.6 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref | 5693 | -10.7 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp8r0 | tilelang | 1261 | 730.2 | 1.00 | 3956 | 864.8 | 1.00 | 1235 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref | 660.8 | 116.2 | 0.52 | - | - | - | 540 |
| heavy-65536-hca-cp8r4 | tilelang | 1655 | 722.9 | 1.00 | 6188 | 861.1 | 1.00 | 1235 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref | 811.0 | 117.0 | 0.49 | - | - | - | 540 |
| heavy-65536-hca-cp8r7 | tilelang | 1308 | 698.6 | 1.00 | 4132 | 878.0 | 1.00 | 1235 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref | 693.7 | 105.4 | 0.53 | - | - | - | 540 |
| heavy-65536-sliding-cp1 | tilelang | 8945 | 671.5 | 1.00 | 28442 | 840.0 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref | 4091 | 20.5 | 0.46 | - | - | - | 4192 |
| heavy-65536-sliding-cp8r0 | tilelang | 1132 | 721.2 | 1.00 | 3582 | 823.1 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref | 528.4 | 141.3 | 0.47 | - | - | - | 524 |
| heavy-65536-sliding-cp8r4 | tilelang | 1152 | 688.6 | 1.00 | 3866 | 811.3 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref | 524.6 | 131.3 | 0.46 | - | - | - | 524 |
| heavy-65536-sliding-cp8r7 | tilelang | 1148 | 675.9 | 1.00 | 3690 | 833.4 | 1.00 | 1222 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref | 525.3 | 140.4 | 0.46 | - | - | - | 524 |
| tiny-65536-csa-cp1 | tilelang | 10414 | 470.2 | 1.00 | 31178 | 622.0 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref | 5690 | -30.5 | 0.55 | - | - | - | 4448 |
| tiny-65536-csa-cp8r0 | tilelang | 1324 | 680.4 | 1.00 | 4061 | 830.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref | 727.4 | 81.8 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r4 | tilelang | 1326 | 681.4 | 1.00 | 4066 | 801.2 | 1.00 | 1269 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref | 731.1 | 92.1 | 0.55 | - | - | - | 556 |
| tiny-65536-csa-cp8r7 | tilelang | 1330 | 734.8 | 1.00 | 4068 | 810.1 | 1.00 | 1269 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref | 728.8 | 102.6 | 0.55 | - | - | - | 556 |
| tiny-65536-hca-cp1 | tilelang | 8038 | 651.6 | 1.00 | 21189 | 833.7 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref | 4147 | 25.6 | 0.52 | - | - | - | 4192 |
| tiny-65536-hca-cp8r0 | tilelang | 1023 | 682.2 | 1.00 | 2791 | 837.3 | 1.00 | 1222 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref | 527.5 | 130.4 | 0.52 | - | - | - | 524 |
| tiny-65536-hca-cp8r4 | tilelang | 1031 | 700.4 | 1.00 | 2795 | 813.5 | 1.00 | 1222 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref | 528.8 | 133.5 | 0.51 | - | - | - | 524 |
| tiny-65536-hca-cp8r7 | tilelang | 1025 | 690.8 | 1.00 | 2797 | 833.6 | 1.00 | 1222 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref | 530.4 | 135.8 | 0.52 | - | - | - | 524 |
| tiny-65536-sliding-cp1 | tilelang | 8031 | 685.8 | 1.00 | 21186 | 850.2 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref | 4116 | 62.0 | 0.51 | - | - | - | 4192 |
| tiny-65536-sliding-cp8r0 | tilelang | 1021 | 723.6 | 1.00 | 2791 | 839.4 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref | 525.1 | 143.2 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r4 | tilelang | 1028 | 687.6 | 1.00 | 2791 | 821.8 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref | 522.8 | 139.4 | 0.51 | - | - | - | 524 |
| tiny-65536-sliding-cp8r7 | tilelang | 1017 | 687.1 | 1.00 | 2786 | 842.6 | 1.00 | 1222 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref | 529.2 | 135.9 | 0.52 | - | - | - | 524 |

Useful FLOPs count valid slots only (fwd 4HD, bwd 10HD per slot); `exec/useful` counts the slots each
backend's tiles touch, or every padded slot for an arm without tile information. TFLOP/s divide
useful FLOPs by op-boundary time (higher is better);
`% peak` is f+b against 989.5 dense BF16 TFLOP/s (https://www.nvidia.com/en-us/data-center/h200/ (H200 SXM BF16 1,979 TFLOPS with sparsity, halved)).

| item | backend | f+b GFLOP | exec/useful fwd | exec/useful bwd | fwd TFLOP/s | f+b TFLOP/s | % peak |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang | 356.8 | 1.09 | 1.05 | 85.2 | 112.0 | 11.3 |
| single-2048-csa-cp1 | flashmla_fwd_ref | 356.8 | 1.69 | 1.69 | 265.5 | - | - |
| single-2048-csa-cp8r0 | tilelang | 15.0 | 1.49 | 1.36 | 5.9 | 7.5 | 0.8 |
| single-2048-csa-cp8r0 | flashmla_fwd_ref | 15.0 | 5.00 | 5.00 | 23.5 | - | - |
| single-2048-csa-cp8r4 | tilelang | 48.8 | 1.08 | 1.04 | 18.3 | 24.1 | 2.4 |
| single-2048-csa-cp8r4 | flashmla_fwd_ref | 48.8 | 1.54 | 1.54 | 70.6 | - | - |
| single-2048-csa-cp8r7 | tilelang | 71.4 | 1.05 | 1.03 | 26.3 | 35.3 | 3.6 |
| single-2048-csa-cp8r7 | flashmla_fwd_ref | 71.4 | 1.05 | 1.05 | 100.5 | - | - |
| single-2048-hca-cp1 | tilelang | 123.6 | 1.41 | 1.18 | 32.5 | 49.4 | 5.0 |
| single-2048-hca-cp1 | flashmla_fwd_ref | 123.6 | 1.95 | 1.95 | 103.5 | - | - |
| single-2048-hca-cp8r0 | tilelang | 11.4 | 1.49 | 1.24 | 4.3 | 5.3 | 0.5 |
| single-2048-hca-cp8r0 | flashmla_fwd_ref | 11.4 | 2.65 | 2.65 | 17.2 | - | - |
| single-2048-hca-cp8r4 | tilelang | 16.0 | 1.41 | 1.17 | 5.8 | 7.4 | 0.8 |
| single-2048-hca-cp8r4 | flashmla_fwd_ref | 16.0 | 1.88 | 1.88 | 23.6 | - | - |
| single-2048-hca-cp8r7 | tilelang | 16.7 | 1.35 | 1.12 | 5.9 | 7.7 | 0.8 |
| single-2048-hca-cp8r7 | flashmla_fwd_ref | 16.7 | 1.80 | 1.80 | 24.7 | - | - |
| single-2048-sliding-cp1 | tilelang | 116.5 | 1.02 | 1.01 | 32.8 | 50.9 | 5.1 |
| single-2048-sliding-cp1 | flashmla_fwd_ref | 116.5 | 1.03 | 1.03 | 111.6 | - | - |
| single-2048-sliding-cp8r0 | tilelang | 11.3 | 1.16 | 1.08 | 4.4 | 5.5 | 0.6 |
| single-2048-sliding-cp8r0 | flashmla_fwd_ref | 11.3 | 1.33 | 1.33 | 18.0 | - | - |
| single-2048-sliding-cp8r4 | tilelang | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.8 |
| single-2048-sliding-cp8r4 | flashmla_fwd_ref | 15.0 | 1.00 | 1.00 | 24.0 | - | - |
| single-2048-sliding-cp8r7 | tilelang | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| single-2048-sliding-cp8r7 | flashmla_fwd_ref | 15.0 | 1.00 | 1.00 | 23.7 | - | - |
| short-2048-csa-cp1 | tilelang | 233.1 | 1.16 | 1.10 | 58.8 | 84.6 | 8.5 |
| short-2048-csa-cp1 | flashmla_fwd_ref | 233.1 | 2.58 | 2.58 | 181.0 | - | - |
| short-2048-csa-cp8r0 | tilelang | 15.0 | 1.49 | 1.36 | 5.7 | 7.4 | 0.8 |
| short-2048-csa-cp8r0 | flashmla_fwd_ref | 15.0 | 5.00 | 5.00 | 22.9 | - | - |
| short-2048-csa-cp8r4 | tilelang | 19.6 | 1.43 | 1.30 | 7.6 | 9.7 | 1.0 |
| short-2048-csa-cp8r4 | flashmla_fwd_ref | 19.6 | 3.83 | 3.83 | 29.9 | - | - |
| short-2048-csa-cp8r7 | tilelang | 39.9 | 1.09 | 1.05 | 14.9 | 19.6 | 2.0 |
| short-2048-csa-cp8r7 | flashmla_fwd_ref | 39.9 | 1.89 | 1.89 | 59.5 | - | - |
| short-2048-hca-cp1 | tilelang | 116.1 | 1.46 | 1.21 | 30.7 | 46.7 | 4.7 |
| short-2048-hca-cp1 | flashmla_fwd_ref | 116.1 | 2.07 | 2.07 | 97.5 | - | - |
| short-2048-hca-cp8r0 | tilelang | 11.4 | 1.49 | 1.24 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r0 | flashmla_fwd_ref | 11.4 | 2.65 | 2.65 | 17.0 | - | - |
| short-2048-hca-cp8r4 | tilelang | 11.5 | 1.47 | 1.22 | 4.1 | 5.3 | 0.5 |
| short-2048-hca-cp8r4 | flashmla_fwd_ref | 11.5 | 2.61 | 2.61 | 17.0 | - | - |
| short-2048-hca-cp8r7 | tilelang | 15.8 | 1.43 | 1.19 | 5.7 | 7.2 | 0.7 |
| short-2048-hca-cp8r7 | flashmla_fwd_ref | 15.8 | 1.91 | 1.91 | 23.8 | - | - |
| short-2048-sliding-cp1 | tilelang | 112.8 | 1.03 | 1.02 | 32.1 | 48.4 | 4.9 |
| short-2048-sliding-cp1 | flashmla_fwd_ref | 112.8 | 1.07 | 1.07 | 109.9 | - | - |
| short-2048-sliding-cp8r0 | tilelang | 11.3 | 1.16 | 1.08 | 4.3 | 5.5 | 0.6 |
| short-2048-sliding-cp8r0 | flashmla_fwd_ref | 11.3 | 1.33 | 1.33 | 17.8 | - | - |
| short-2048-sliding-cp8r4 | tilelang | 11.3 | 1.16 | 1.08 | 4.3 | 5.6 | 0.6 |
| short-2048-sliding-cp8r4 | flashmla_fwd_ref | 11.3 | 1.33 | 1.33 | 17.0 | - | - |
| short-2048-sliding-cp8r7 | tilelang | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| short-2048-sliding-cp8r7 | flashmla_fwd_ref | 15.0 | 1.00 | 1.00 | 23.2 | - | - |
| heavy-2048-csa-cp1 | tilelang | 272.5 | 1.16 | 1.09 | 66.5 | 92.2 | 9.3 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref | 272.5 | 2.21 | 2.21 | 199.8 | - | - |
| heavy-2048-csa-cp8r0 | tilelang | 10.3 | 2.16 | 1.86 | 3.9 | 5.0 | 0.5 |
| heavy-2048-csa-cp8r0 | flashmla_fwd_ref | 10.3 | 7.32 | 7.32 | 15.5 | - | - |
| heavy-2048-csa-cp8r4 | tilelang | 37.7 | 1.10 | 1.05 | 13.8 | 18.2 | 1.8 |
| heavy-2048-csa-cp8r4 | flashmla_fwd_ref | 37.7 | 2.00 | 2.00 | 53.6 | - | - |
| heavy-2048-csa-cp8r7 | tilelang | 60.2 | 1.06 | 1.03 | 21.7 | 29.6 | 3.0 |
| heavy-2048-csa-cp8r7 | flashmla_fwd_ref | 60.2 | 1.25 | 1.25 | 82.1 | - | - |
| heavy-2048-hca-cp1 | tilelang | 113.7 | 1.44 | 1.20 | 30.3 | 46.2 | 4.7 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref | 113.7 | 2.11 | 2.11 | 97.0 | - | - |
| heavy-2048-hca-cp8r0 | tilelang | 8.2 | 1.57 | 1.28 | 3.0 | 3.8 | 0.4 |
| heavy-2048-hca-cp8r0 | flashmla_fwd_ref | 8.2 | 3.68 | 3.68 | 12.3 | - | - |
| heavy-2048-hca-cp8r4 | tilelang | 15.7 | 1.44 | 1.20 | 5.7 | 7.4 | 0.7 |
| heavy-2048-hca-cp8r4 | flashmla_fwd_ref | 15.7 | 1.92 | 1.92 | 23.6 | - | - |
| heavy-2048-hca-cp8r7 | tilelang | 16.4 | 1.38 | 1.15 | 6.0 | 7.6 | 0.8 |
| heavy-2048-hca-cp8r7 | flashmla_fwd_ref | 16.4 | 1.83 | 1.83 | 24.7 | - | - |
| heavy-2048-sliding-cp1 | tilelang | 109.1 | 1.05 | 1.03 | 31.2 | 47.5 | 4.8 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref | 109.1 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-2048-sliding-cp8r0 | tilelang | 8.1 | 1.39 | 1.19 | 3.2 | 4.0 | 0.4 |
| heavy-2048-sliding-cp8r0 | flashmla_fwd_ref | 8.1 | 1.85 | 1.85 | 12.9 | - | - |
| heavy-2048-sliding-cp8r4 | tilelang | 15.0 | 1.00 | 1.00 | 5.8 | 7.4 | 0.7 |
| heavy-2048-sliding-cp8r4 | flashmla_fwd_ref | 15.0 | 1.00 | 1.00 | 23.8 | - | - |
| heavy-2048-sliding-cp8r7 | tilelang | 15.0 | 1.00 | 1.00 | 5.8 | 7.3 | 0.7 |
| heavy-2048-sliding-cp8r7 | flashmla_fwd_ref | 15.0 | 1.00 | 1.00 | 23.5 | - | - |
| tiny-2048-csa-cp1 | tilelang | 53.6 | 3.28 | 2.72 | 14.7 | 22.7 | 2.3 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref | 53.6 | 11.22 | 11.22 | 45.9 | - | - |
| tiny-2048-csa-cp8r0 | tilelang | 6.6 | 3.31 | 2.74 | 2.5 | 3.2 | 0.3 |
| tiny-2048-csa-cp8r0 | flashmla_fwd_ref | 6.6 | 11.38 | 11.38 | 9.8 | - | - |
| tiny-2048-csa-cp8r4 | tilelang | 4.7 | 4.58 | 3.79 | 1.8 | 2.3 | 0.2 |
| tiny-2048-csa-cp8r4 | flashmla_fwd_ref | 4.7 | 15.89 | 15.89 | 7.2 | - | - |
| tiny-2048-csa-cp8r7 | tilelang | 8.6 | 2.59 | 2.15 | 3.2 | 4.2 | 0.4 |
| tiny-2048-csa-cp8r7 | flashmla_fwd_ref | 8.6 | 8.76 | 8.76 | 12.9 | - | - |
| tiny-2048-hca-cp1 | tilelang | 43.2 | 1.79 | 1.36 | 12.8 | 20.6 | 2.1 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref | 43.2 | 2.79 | 2.79 | 42.2 | - | - |
| tiny-2048-hca-cp8r0 | tilelang | 5.3 | 1.76 | 1.37 | 2.1 | 2.6 | 0.3 |
| tiny-2048-hca-cp8r0 | flashmla_fwd_ref | 5.3 | 2.83 | 2.83 | 8.3 | - | - |
| tiny-2048-hca-cp8r4 | tilelang | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-hca-cp8r4 | flashmla_fwd_ref | 3.8 | 3.94 | 3.94 | 6.0 | - | - |
| tiny-2048-hca-cp8r7 | tilelang | 6.9 | 1.60 | 1.28 | 2.6 | 3.3 | 0.3 |
| tiny-2048-hca-cp8r7 | flashmla_fwd_ref | 6.9 | 2.18 | 2.18 | 10.4 | - | - |
| tiny-2048-sliding-cp1 | tilelang | 43.2 | 1.79 | 1.36 | 12.9 | 20.5 | 2.1 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref | 43.2 | 2.79 | 2.79 | 42.4 | - | - |
| tiny-2048-sliding-cp8r0 | tilelang | 5.3 | 1.76 | 1.37 | 2.1 | 2.5 | 0.3 |
| tiny-2048-sliding-cp8r0 | flashmla_fwd_ref | 5.3 | 2.83 | 2.83 | 8.5 | - | - |
| tiny-2048-sliding-cp8r4 | tilelang | 3.8 | 2.23 | 1.52 | 1.5 | 1.9 | 0.2 |
| tiny-2048-sliding-cp8r4 | flashmla_fwd_ref | 3.8 | 3.94 | 3.94 | 6.1 | - | - |
| tiny-2048-sliding-cp8r7 | tilelang | 6.9 | 1.60 | 1.28 | 2.6 | 3.4 | 0.3 |
| tiny-2048-sliding-cp8r7 | flashmla_fwd_ref | 6.9 | 2.18 | 2.18 | 10.6 | - | - |
| single-4096-csa-cp1 | tilelang | 958.1 | 1.03 | 1.02 | 144.8 | 160.4 | 16.2 |
| single-4096-csa-cp1 | flashmla_fwd_ref | 958.1 | 1.26 | 1.26 | 382.7 | - | - |
| single-4096-csa-cp8r0 | tilelang | 41.3 | 1.27 | 1.18 | 14.8 | 20.3 | 2.0 |
| single-4096-csa-cp8r0 | flashmla_fwd_ref | 41.3 | 3.64 | 3.64 | 57.2 | - | - |
| single-4096-csa-cp8r4 | tilelang | 150.3 | 1.00 | 1.00 | 49.4 | 64.4 | 6.5 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref | 150.3 | 1.00 | 1.00 | 172.9 | - | - |
| single-4096-csa-cp8r7 | tilelang | 150.3 | 1.00 | 1.00 | 48.8 | 64.0 | 6.5 |
| single-4096-csa-cp8r7 | flashmla_fwd_ref | 150.3 | 1.00 | 1.00 | 170.5 | - | - |
| single-4096-hca-cp1 | tilelang | 265.9 | 1.34 | 1.11 | 53.0 | 84.9 | 8.6 |
| single-4096-hca-cp1 | flashmla_fwd_ref | 265.9 | 1.81 | 1.81 | 150.3 | - | - |
| single-4096-hca-cp8r0 | tilelang | 26.7 | 1.48 | 1.23 | 9.2 | 12.6 | 1.3 |
| single-4096-hca-cp8r0 | flashmla_fwd_ref | 26.7 | 2.25 | 2.25 | 35.1 | - | - |
| single-4096-hca-cp8r4 | tilelang | 34.2 | 1.32 | 1.10 | 11.8 | 16.0 | 1.6 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref | 34.2 | 1.76 | 1.76 | 45.0 | - | - |
| single-4096-hca-cp8r7 | tilelang | 37.0 | 1.22 | 1.02 | 12.8 | 17.1 | 1.7 |
| single-4096-hca-cp8r7 | flashmla_fwd_ref | 37.0 | 1.63 | 1.63 | 48.7 | - | - |
| single-4096-sliding-cp1 | tilelang | 236.8 | 1.01 | 1.00 | 52.2 | 82.8 | 8.4 |
| single-4096-sliding-cp1 | flashmla_fwd_ref | 236.8 | 1.02 | 1.02 | 161.0 | - | - |
| single-4096-sliding-cp8r0 | tilelang | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| single-4096-sliding-cp8r0 | flashmla_fwd_ref | 26.3 | 1.14 | 1.14 | 37.8 | - | - |
| single-4096-sliding-cp8r4 | tilelang | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref | 30.1 | 1.00 | 1.00 | 42.3 | - | - |
| single-4096-sliding-cp8r7 | tilelang | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r7 | flashmla_fwd_ref | 30.1 | 1.00 | 1.00 | 44.3 | - | - |
| short-4096-csa-cp1 | tilelang | 449.8 | 1.18 | 1.11 | 84.4 | 116.1 | 11.7 |
| short-4096-csa-cp1 | flashmla_fwd_ref | 449.8 | 2.67 | 2.67 | 232.4 | - | - |
| short-4096-csa-cp8r0 | tilelang | 41.3 | 1.27 | 1.18 | 14.3 | 20.1 | 2.0 |
| short-4096-csa-cp8r0 | flashmla_fwd_ref | 41.3 | 3.64 | 3.64 | 53.1 | - | - |
| short-4096-csa-cp8r4 | tilelang | 44.6 | 1.21 | 1.13 | 14.5 | 20.8 | 2.1 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref | 44.6 | 3.37 | 3.37 | 56.1 | - | - |
| short-4096-csa-cp8r7 | tilelang | 42.3 | 1.26 | 1.17 | 14.8 | 20.6 | 2.1 |
| short-4096-csa-cp8r7 | flashmla_fwd_ref | 42.3 | 3.55 | 3.55 | 57.4 | - | - |
| short-4096-hca-cp1 | tilelang | 228.1 | 1.46 | 1.22 | 46.2 | 73.8 | 7.5 |
| short-4096-hca-cp1 | flashmla_fwd_ref | 228.1 | 2.11 | 2.11 | 130.3 | - | - |
| short-4096-hca-cp8r0 | tilelang | 26.7 | 1.48 | 1.23 | 9.0 | 12.2 | 1.2 |
| short-4096-hca-cp8r0 | flashmla_fwd_ref | 26.7 | 2.25 | 2.25 | 35.3 | - | - |
| short-4096-hca-cp8r4 | tilelang | 28.3 | 1.46 | 1.23 | 9.8 | 13.1 | 1.3 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref | 28.3 | 2.13 | 2.13 | 38.8 | - | - |
| short-4096-hca-cp8r7 | tilelang | 26.7 | 1.48 | 1.23 | 9.2 | 12.2 | 1.2 |
| short-4096-hca-cp8r7 | flashmla_fwd_ref | 26.7 | 2.25 | 2.25 | 35.7 | - | - |
| short-4096-sliding-cp1 | tilelang | 221.9 | 1.04 | 1.02 | 48.3 | 78.1 | 7.9 |
| short-4096-sliding-cp1 | flashmla_fwd_ref | 221.9 | 1.08 | 1.08 | 149.8 | - | - |
| short-4096-sliding-cp8r0 | tilelang | 26.3 | 1.07 | 1.03 | 9.5 | 12.9 | 1.3 |
| short-4096-sliding-cp8r0 | flashmla_fwd_ref | 26.3 | 1.14 | 1.14 | 38.1 | - | - |
| short-4096-sliding-cp8r4 | tilelang | 27.9 | 1.04 | 1.02 | 10.0 | 13.7 | 1.4 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref | 27.9 | 1.08 | 1.08 | 40.3 | - | - |
| short-4096-sliding-cp8r7 | tilelang | 26.3 | 1.07 | 1.03 | 9.7 | 12.6 | 1.3 |
| short-4096-sliding-cp8r7 | flashmla_fwd_ref | 26.3 | 1.14 | 1.14 | 38.4 | - | - |
| heavy-4096-csa-cp1 | tilelang | 356.7 | 1.29 | 1.19 | 69.7 | 100.6 | 10.2 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref | 356.7 | 3.37 | 3.37 | 194.2 | - | - |
| heavy-4096-csa-cp8r0 | tilelang | 41.3 | 1.27 | 1.18 | 14.9 | 20.3 | 2.1 |
| heavy-4096-csa-cp8r0 | flashmla_fwd_ref | 41.3 | 3.64 | 3.64 | 56.7 | - | - |
| heavy-4096-csa-cp8r4 | tilelang | 45.5 | 1.20 | 1.12 | 15.9 | 22.2 | 2.2 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref | 45.5 | 3.30 | 3.30 | 59.9 | - | - |
| heavy-4096-csa-cp8r7 | tilelang | 29.6 | 1.51 | 1.38 | 9.9 | 14.2 | 1.4 |
| heavy-4096-csa-cp8r7 | flashmla_fwd_ref | 29.6 | 5.09 | 5.09 | 39.1 | - | - |
| heavy-4096-hca-cp1 | tilelang | 207.2 | 1.47 | 1.23 | 43.1 | 69.5 | 7.0 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref | 207.2 | 2.32 | 2.32 | 122.6 | - | - |
| heavy-4096-hca-cp8r0 | tilelang | 26.7 | 1.48 | 1.23 | 9.1 | 12.4 | 1.3 |
| heavy-4096-hca-cp8r0 | flashmla_fwd_ref | 26.7 | 2.25 | 2.25 | 35.5 | - | - |
| heavy-4096-hca-cp8r4 | tilelang | 28.7 | 1.46 | 1.22 | 9.9 | 13.3 | 1.3 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref | 28.7 | 2.10 | 2.10 | 39.5 | - | - |
| heavy-4096-hca-cp8r7 | tilelang | 22.7 | 1.49 | 1.24 | 7.7 | 10.4 | 1.1 |
| heavy-4096-hca-cp8r7 | flashmla_fwd_ref | 22.7 | 2.65 | 2.65 | 30.1 | - | - |
| heavy-4096-sliding-cp1 | tilelang | 203.2 | 1.09 | 1.04 | 45.3 | 69.8 | 7.1 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref | 203.2 | 1.18 | 1.18 | 138.7 | - | - |
| heavy-4096-sliding-cp8r0 | tilelang | 26.3 | 1.07 | 1.03 | 9.5 | 12.6 | 1.3 |
| heavy-4096-sliding-cp8r0 | flashmla_fwd_ref | 26.3 | 1.14 | 1.14 | 37.9 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang | 28.3 | 1.04 | 1.02 | 10.3 | 13.8 | 1.4 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref | 28.3 | 1.06 | 1.06 | 41.6 | - | - |
| heavy-4096-sliding-cp8r7 | tilelang | 22.6 | 1.16 | 1.08 | 8.4 | 10.8 | 1.1 |
| heavy-4096-sliding-cp8r7 | flashmla_fwd_ref | 22.6 | 1.33 | 1.33 | 32.7 | - | - |
| tiny-4096-csa-cp1 | tilelang | 104.6 | 3.35 | 2.77 | 21.8 | 35.8 | 3.6 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref | 104.6 | 11.49 | 11.49 | 59.6 | - | - |
| tiny-4096-csa-cp8r0 | tilelang | 11.9 | 3.67 | 3.04 | 4.3 | 5.9 | 0.6 |
| tiny-4096-csa-cp8r0 | flashmla_fwd_ref | 11.9 | 12.64 | 12.64 | 16.1 | - | - |
| tiny-4096-csa-cp8r4 | tilelang | 15.1 | 2.92 | 2.42 | 5.3 | 7.4 | 0.7 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref | 15.1 | 9.98 | 9.98 | 20.5 | - | - |
| tiny-4096-csa-cp8r7 | tilelang | 14.5 | 3.04 | 2.52 | 5.3 | 7.1 | 0.7 |
| tiny-4096-csa-cp8r7 | flashmla_fwd_ref | 14.5 | 10.36 | 10.36 | 19.7 | - | - |
| tiny-4096-hca-cp1 | tilelang | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref | 84.3 | 2.85 | 2.85 | 58.4 | - | - |
| tiny-4096-hca-cp8r0 | tilelang | 9.6 | 1.91 | 1.41 | 3.6 | 4.7 | 0.5 |
| tiny-4096-hca-cp8r0 | flashmla_fwd_ref | 9.6 | 3.14 | 3.14 | 14.1 | - | - |
| tiny-4096-hca-cp8r4 | tilelang | 12.1 | 1.66 | 1.32 | 4.5 | 6.0 | 0.6 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref | 12.1 | 2.48 | 2.48 | 17.6 | - | - |
| tiny-4096-hca-cp8r7 | tilelang | 11.7 | 1.70 | 1.33 | 4.3 | 5.7 | 0.6 |
| tiny-4096-hca-cp8r7 | flashmla_fwd_ref | 11.7 | 2.57 | 2.57 | 16.6 | - | - |
| tiny-4096-sliding-cp1 | tilelang | 84.3 | 1.81 | 1.37 | 19.6 | 35.0 | 3.5 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref | 84.3 | 2.85 | 2.85 | 58.9 | - | - |
| tiny-4096-sliding-cp8r0 | tilelang | 9.6 | 1.91 | 1.41 | 3.5 | 4.7 | 0.5 |
| tiny-4096-sliding-cp8r0 | flashmla_fwd_ref | 9.6 | 3.14 | 3.14 | 13.6 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang | 12.1 | 1.66 | 1.32 | 4.4 | 5.9 | 0.6 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref | 12.1 | 2.48 | 2.48 | 17.1 | - | - |
| tiny-4096-sliding-cp8r7 | tilelang | 11.7 | 1.70 | 1.33 | 4.2 | 5.6 | 0.6 |
| tiny-4096-sliding-cp8r7 | flashmla_fwd_ref | 11.7 | 2.57 | 2.57 | 16.5 | - | - |
| single-16384-csa-cp1 | tilelang | 4565.9 | 1.01 | 1.00 | 220.5 | 193.8 | 19.6 |
| single-16384-csa-cp1 | flashmla_fwd_ref | 4565.9 | 1.05 | 1.05 | 507.6 | - | - |
| single-16384-csa-cp8r0 | tilelang | 356.8 | 1.09 | 1.05 | 82.7 | 110.0 | 11.1 |
| single-16384-csa-cp8r0 | flashmla_fwd_ref | 356.8 | 1.69 | 1.69 | 244.1 | - | - |
| single-16384-csa-cp8r4 | tilelang | 601.3 | 1.00 | 1.00 | 121.9 | 147.4 | 14.9 |
| single-16384-csa-cp8r4 | flashmla_fwd_ref | 601.3 | 1.00 | 1.00 | 347.0 | - | - |
| single-16384-csa-cp8r7 | tilelang | 601.3 | 1.00 | 1.00 | 122.3 | 146.5 | 14.8 |
| single-16384-csa-cp8r7 | flashmla_fwd_ref | 601.3 | 1.00 | 1.00 | 345.9 | - | - |
| single-16384-hca-cp1 | tilelang | 1435.7 | 1.17 | 1.08 | 114.8 | 131.7 | 13.3 |
| single-16384-hca-cp1 | flashmla_fwd_ref | 1435.7 | 1.34 | 1.34 | 275.4 | - | - |
| single-16384-hca-cp8r0 | tilelang | 123.6 | 1.41 | 1.18 | 32.7 | 51.0 | 5.2 |
| single-16384-hca-cp8r0 | flashmla_fwd_ref | 123.6 | 1.95 | 1.95 | 103.7 | - | - |
| single-16384-hca-cp8r4 | tilelang | 187.4 | 1.26 | 1.11 | 48.8 | 70.3 | 7.1 |
| single-16384-hca-cp8r4 | flashmla_fwd_ref | 187.4 | 1.28 | 1.28 | 157.6 | - | - |
| single-16384-hca-cp8r7 | tilelang | 232.5 | 1.03 | 1.03 | 60.1 | 82.5 | 8.3 |
| single-16384-hca-cp8r7 | flashmla_fwd_ref | 232.5 | 1.03 | 1.03 | 193.2 | - | - |
| single-16384-sliding-cp1 | tilelang | 958.3 | 1.00 | 1.00 | 92.1 | 115.7 | 11.7 |
| single-16384-sliding-cp1 | flashmla_fwd_ref | 958.3 | 1.00 | 1.00 | 235.2 | - | - |
| single-16384-sliding-cp8r0 | tilelang | 116.5 | 1.02 | 1.01 | 32.7 | 49.9 | 5.0 |
| single-16384-sliding-cp8r0 | flashmla_fwd_ref | 116.5 | 1.03 | 1.03 | 111.1 | - | - |
| single-16384-sliding-cp8r4 | tilelang | 120.3 | 1.00 | 1.00 | 34.2 | 50.3 | 5.1 |
| single-16384-sliding-cp8r4 | flashmla_fwd_ref | 120.3 | 1.00 | 1.00 | 115.8 | - | - |
| single-16384-sliding-cp8r7 | tilelang | 120.3 | 1.00 | 1.00 | 34.5 | 50.3 | 5.1 |
| single-16384-sliding-cp8r7 | flashmla_fwd_ref | 120.3 | 1.00 | 1.00 | 116.9 | - | - |
| short-16384-csa-cp1 | tilelang | 2610.6 | 1.11 | 1.06 | 165.4 | 163.4 | 16.5 |
| short-16384-csa-cp1 | flashmla_fwd_ref | 2610.6 | 1.84 | 1.84 | 381.2 | - | - |
| short-16384-csa-cp8r0 | tilelang | 280.2 | 1.14 | 1.08 | 68.1 | 93.5 | 9.5 |
| short-16384-csa-cp8r0 | flashmla_fwd_ref | 280.2 | 2.15 | 2.15 | 205.3 | - | - |
| short-16384-csa-cp8r4 | tilelang | 410.5 | 1.07 | 1.04 | 93.9 | 121.2 | 12.3 |
| short-16384-csa-cp8r4 | flashmla_fwd_ref | 410.5 | 1.46 | 1.46 | 271.8 | - | - |
| short-16384-csa-cp8r7 | tilelang | 352.9 | 1.10 | 1.06 | 83.3 | 108.9 | 11.0 |
| short-16384-csa-cp8r7 | flashmla_fwd_ref | 352.9 | 1.70 | 1.70 | 247.1 | - | - |
| short-16384-hca-cp1 | tilelang | 963.6 | 1.42 | 1.18 | 82.5 | 104.5 | 10.6 |
| short-16384-hca-cp1 | flashmla_fwd_ref | 963.6 | 2.00 | 2.00 | 188.2 | - | - |
| short-16384-hca-cp8r0 | tilelang | 117.6 | 1.44 | 1.20 | 31.4 | 47.0 | 4.8 |
| short-16384-hca-cp8r0 | flashmla_fwd_ref | 117.6 | 2.05 | 2.05 | 99.1 | - | - |
| short-16384-hca-cp8r4 | tilelang | 125.5 | 1.39 | 1.16 | 33.2 | 49.5 | 5.0 |
| short-16384-hca-cp8r4 | flashmla_fwd_ref | 125.5 | 1.92 | 1.92 | 106.2 | - | - |
| short-16384-hca-cp8r7 | tilelang | 119.9 | 1.41 | 1.18 | 31.7 | 46.9 | 4.7 |
| short-16384-hca-cp8r7 | flashmla_fwd_ref | 119.9 | 2.01 | 2.01 | 101.1 | - | - |
| short-16384-sliding-cp1 | tilelang | 913.6 | 1.03 | 1.01 | 87.4 | 112.0 | 11.3 |
| short-16384-sliding-cp1 | flashmla_fwd_ref | 913.6 | 1.05 | 1.05 | 223.3 | - | - |
| short-16384-sliding-cp8r0 | tilelang | 112.8 | 1.03 | 1.02 | 32.2 | 48.9 | 4.9 |
| short-16384-sliding-cp8r0 | flashmla_fwd_ref | 112.8 | 1.07 | 1.07 | 108.8 | - | - |
| short-16384-sliding-cp8r4 | tilelang | 116.5 | 1.02 | 1.01 | 33.5 | 49.2 | 5.0 |
| short-16384-sliding-cp8r4 | flashmla_fwd_ref | 116.5 | 1.03 | 1.03 | 113.1 | - | - |
| short-16384-sliding-cp8r7 | tilelang | 112.8 | 1.03 | 1.02 | 32.2 | 47.8 | 4.8 |
| short-16384-sliding-cp8r7 | flashmla_fwd_ref | 112.8 | 1.07 | 1.07 | 109.4 | - | - |
| heavy-16384-csa-cp1 | tilelang | 1795.9 | 1.22 | 1.14 | 129.3 | 138.2 | 14.0 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref | 1795.9 | 2.68 | 2.68 | 297.3 | - | - |
| heavy-16384-csa-cp8r0 | tilelang | 154.2 | 1.36 | 1.24 | 40.3 | 59.9 | 6.1 |
| heavy-16384-csa-cp8r0 | flashmla_fwd_ref | 154.2 | 3.90 | 3.90 | 127.1 | - | - |
| heavy-16384-csa-cp8r4 | tilelang | 219.5 | 1.19 | 1.12 | 56.2 | 79.5 | 8.0 |
| heavy-16384-csa-cp8r4 | flashmla_fwd_ref | 219.5 | 2.74 | 2.74 | 172.9 | - | - |
| heavy-16384-csa-cp8r7 | tilelang | 410.5 | 1.06 | 1.03 | 92.8 | 117.9 | 11.9 |
| heavy-16384-csa-cp8r7 | flashmla_fwd_ref | 410.5 | 1.46 | 1.46 | 273.7 | - | - |
| heavy-16384-hca-cp1 | tilelang | 847.5 | 1.45 | 1.21 | 74.5 | 97.4 | 9.8 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref | 847.5 | 2.27 | 2.27 | 172.5 | - | - |
| heavy-16384-hca-cp8r0 | tilelang | 99.2 | 1.48 | 1.23 | 26.6 | 40.5 | 4.1 |
| heavy-16384-hca-cp8r0 | flashmla_fwd_ref | 99.2 | 2.42 | 2.42 | 84.5 | - | - |
| heavy-16384-hca-cp8r4 | tilelang | 112.5 | 1.46 | 1.22 | 30.1 | 45.2 | 4.6 |
| heavy-16384-hca-cp8r4 | flashmla_fwd_ref | 112.5 | 2.14 | 2.14 | 94.1 | - | - |
| heavy-16384-hca-cp8r7 | tilelang | 129.0 | 1.40 | 1.17 | 33.9 | 49.2 | 5.0 |
| heavy-16384-hca-cp8r7 | flashmla_fwd_ref | 129.0 | 1.86 | 1.86 | 106.5 | - | - |
| heavy-16384-sliding-cp1 | tilelang | 820.4 | 1.09 | 1.04 | 78.7 | 104.0 | 10.5 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref | 820.4 | 1.17 | 1.17 | 199.8 | - | - |
| heavy-16384-sliding-cp8r0 | tilelang | 97.9 | 1.11 | 1.06 | 27.6 | 42.7 | 4.3 |
| heavy-16384-sliding-cp8r0 | flashmla_fwd_ref | 97.9 | 1.23 | 1.23 | 93.9 | - | - |
| heavy-16384-sliding-cp8r4 | tilelang | 109.5 | 1.05 | 1.02 | 31.8 | 47.0 | 4.7 |
| heavy-16384-sliding-cp8r4 | flashmla_fwd_ref | 109.5 | 1.10 | 1.10 | 106.8 | - | - |
| heavy-16384-sliding-cp8r7 | tilelang | 120.3 | 1.00 | 1.00 | 33.6 | 50.0 | 5.1 |
| heavy-16384-sliding-cp8r7 | flashmla_fwd_ref | 120.3 | 1.00 | 1.00 | 115.9 | - | - |
| tiny-16384-csa-cp1 | tilelang | 391.4 | 3.56 | 2.95 | 34.0 | 44.9 | 4.5 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref | 391.4 | 12.29 | 12.29 | 75.8 | - | - |
| tiny-16384-csa-cp8r0 | tilelang | 50.6 | 3.45 | 2.86 | 13.9 | 21.3 | 2.2 |
| tiny-16384-csa-cp8r0 | flashmla_fwd_ref | 50.6 | 11.89 | 11.89 | 43.2 | - | - |
| tiny-16384-csa-cp8r4 | tilelang | 48.5 | 3.60 | 2.98 | 13.2 | 20.6 | 2.1 |
| tiny-16384-csa-cp8r4 | flashmla_fwd_ref | 48.5 | 12.39 | 12.39 | 40.5 | - | - |
| tiny-16384-csa-cp8r7 | tilelang | 43.9 | 3.96 | 3.27 | 12.1 | 18.5 | 1.9 |
| tiny-16384-csa-cp8r7 | flashmla_fwd_ref | 43.9 | 13.71 | 13.71 | 37.5 | - | - |
| tiny-16384-hca-cp1 | tilelang | 315.4 | 1.89 | 1.40 | 33.3 | 51.0 | 5.2 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref | 315.4 | 3.05 | 3.05 | 76.6 | - | - |
| tiny-16384-hca-cp8r0 | tilelang | 40.7 | 1.86 | 1.39 | 11.9 | 19.0 | 1.9 |
| tiny-16384-hca-cp8r0 | flashmla_fwd_ref | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-hca-cp8r4 | tilelang | 39.1 | 1.88 | 1.40 | 11.5 | 18.7 | 1.9 |
| tiny-16384-hca-cp8r4 | flashmla_fwd_ref | 39.1 | 3.07 | 3.07 | 38.1 | - | - |
| tiny-16384-hca-cp8r7 | tilelang | 35.4 | 2.01 | 1.45 | 10.6 | 16.6 | 1.7 |
| tiny-16384-hca-cp8r7 | flashmla_fwd_ref | 35.4 | 3.40 | 3.40 | 34.3 | - | - |
| tiny-16384-sliding-cp1 | tilelang | 315.4 | 1.89 | 1.40 | 33.0 | 51.2 | 5.2 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref | 315.4 | 3.05 | 3.05 | 76.3 | - | - |
| tiny-16384-sliding-cp8r0 | tilelang | 40.7 | 1.86 | 1.39 | 12.1 | 19.0 | 1.9 |
| tiny-16384-sliding-cp8r0 | flashmla_fwd_ref | 40.7 | 2.95 | 2.95 | 39.2 | - | - |
| tiny-16384-sliding-cp8r4 | tilelang | 39.1 | 1.88 | 1.40 | 11.6 | 18.0 | 1.8 |
| tiny-16384-sliding-cp8r4 | flashmla_fwd_ref | 39.1 | 3.07 | 3.07 | 37.4 | - | - |
| tiny-16384-sliding-cp8r7 | tilelang | 35.4 | 2.01 | 1.45 | 10.5 | 16.5 | 1.7 |
| tiny-16384-sliding-cp8r7 | flashmla_fwd_ref | 35.4 | 3.40 | 3.40 | 34.5 | - | - |
| single-49208-csa-cp1 | tilelang | 14203.1 | 1.00 | 1.00 | 247.1 | 200.9 | 20.3 |
| single-49208-csa-cp1 | flashmla_fwd_ref | 14203.1 | 1.02 | 1.02 | 488.4 | - | - |
| single-49208-csa-cp8r0 | tilelang | 1561.5 | 1.02 | 1.01 | 174.6 | 175.0 | 17.7 |
| single-49208-csa-cp8r0 | flashmla_fwd_ref | 1561.5 | 1.16 | 1.16 | 434.8 | - | - |
| single-49208-csa-cp8r4 | tilelang | 1805.9 | 1.00 | 1.00 | 188.0 | 180.1 | 18.2 |
| single-49208-csa-cp8r4 | flashmla_fwd_ref | 1805.9 | 1.00 | 1.00 | 468.9 | - | - |
| single-49208-csa-cp8r7 | tilelang | 1805.9 | 1.00 | 1.00 | 188.0 | 174.3 | 17.6 |
| single-49208-csa-cp8r7 | flashmla_fwd_ref | 1805.9 | 1.00 | 1.00 | 467.1 | - | - |
| single-49208-hca-cp1 | tilelang | 7213.9 | 1.10 | 1.05 | 179.3 | 169.5 | 17.1 |
| single-49208-hca-cp1 | flashmla_fwd_ref | 7213.9 | 1.60 | 1.60 | 381.5 | - | - |
| single-49208-hca-cp8r0 | tilelang | 423.9 | 1.26 | 1.12 | 70.8 | 99.8 | 10.1 |
| single-49208-hca-cp8r0 | flashmla_fwd_ref | 423.9 | 3.41 | 3.41 | 180.6 | - | - |
| single-49208-hca-cp8r4 | tilelang | 970.0 | 1.11 | 1.05 | 129.2 | 147.8 | 14.9 |
| single-49208-hca-cp8r4 | flashmla_fwd_ref | 970.0 | 1.49 | 1.49 | 345.9 | - | - |
| single-49208-hca-cp8r7 | tilelang | 1376.8 | 1.05 | 1.03 | 161.9 | 167.0 | 16.9 |
| single-49208-hca-cp8r7 | flashmla_fwd_ref | 1376.8 | 1.05 | 1.05 | 416.5 | - | - |
| single-49208-sliding-cp1 | tilelang | 2885.8 | 1.00 | 1.00 | 110.9 | 126.1 | 12.7 |
| single-49208-sliding-cp1 | flashmla_fwd_ref | 2885.8 | 1.00 | 1.00 | 263.8 | - | - |
| single-49208-sliding-cp8r0 | tilelang | 357.5 | 1.01 | 1.00 | 64.9 | 95.5 | 9.6 |
| single-49208-sliding-cp8r0 | flashmla_fwd_ref | 357.5 | 1.01 | 1.01 | 188.1 | - | - |
| single-49208-sliding-cp8r4 | tilelang | 361.2 | 1.00 | 1.00 | 66.3 | 95.5 | 9.6 |
| single-49208-sliding-cp8r4 | flashmla_fwd_ref | 361.2 | 1.00 | 1.00 | 191.4 | - | - |
| single-49208-sliding-cp8r7 | tilelang | 361.2 | 1.00 | 1.00 | 65.6 | 95.3 | 9.6 |
| single-49208-sliding-cp8r7 | flashmla_fwd_ref | 361.2 | 1.00 | 1.00 | 191.1 | - | - |
| short-49208-csa-cp1 | tilelang | 9266.5 | 1.07 | 1.04 | 203.8 | 181.4 | 18.3 |
| short-49208-csa-cp1 | flashmla_fwd_ref | 9266.5 | 1.56 | 1.56 | 425.2 | - | - |
| short-49208-csa-cp8r0 | tilelang | 832.8 | 1.14 | 1.08 | 116.8 | 138.5 | 14.0 |
| short-49208-csa-cp8r0 | flashmla_fwd_ref | 832.8 | 2.17 | 2.17 | 300.8 | - | - |
| short-49208-csa-cp8r4 | tilelang | 1351.2 | 1.04 | 1.02 | 161.8 | 167.1 | 16.9 |
| short-49208-csa-cp8r4 | flashmla_fwd_ref | 1351.2 | 1.34 | 1.34 | 402.3 | - | - |
| short-49208-csa-cp8r7 | tilelang | 1017.3 | 1.09 | 1.05 | 134.5 | 149.4 | 15.1 |
| short-49208-csa-cp8r7 | flashmla_fwd_ref | 1017.3 | 1.78 | 1.78 | 338.6 | - | - |
| short-49208-hca-cp1 | tilelang | 3131.2 | 1.36 | 1.16 | 105.2 | 119.6 | 12.1 |
| short-49208-hca-cp1 | flashmla_fwd_ref | 3131.2 | 1.85 | 1.85 | 218.0 | - | - |
| short-49208-hca-cp8r0 | tilelang | 352.9 | 1.44 | 1.20 | 59.3 | 86.7 | 8.8 |
| short-49208-hca-cp8r0 | flashmla_fwd_ref | 352.9 | 2.05 | 2.05 | 155.9 | - | - |
| short-49208-hca-cp8r4 | tilelang | 398.0 | 1.33 | 1.14 | 65.1 | 95.1 | 9.6 |
| short-49208-hca-cp8r4 | flashmla_fwd_ref | 398.0 | 1.82 | 1.82 | 170.9 | - | - |
| short-49208-hca-cp8r7 | tilelang | 369.7 | 1.42 | 1.18 | 61.0 | 88.7 | 9.0 |
| short-49208-hca-cp8r7 | flashmla_fwd_ref | 369.7 | 1.95 | 1.95 | 159.0 | - | - |
| short-49208-sliding-cp1 | tilelang | 2785.1 | 1.02 | 1.01 | 107.2 | 123.1 | 12.4 |
| short-49208-sliding-cp1 | flashmla_fwd_ref | 2785.1 | 1.04 | 1.04 | 254.8 | - | - |
| short-49208-sliding-cp8r0 | tilelang | 338.8 | 1.03 | 1.02 | 61.9 | 92.5 | 9.3 |
| short-49208-sliding-cp8r0 | flashmla_fwd_ref | 338.8 | 1.07 | 1.07 | 178.9 | - | - |
| short-49208-sliding-cp8r4 | tilelang | 353.7 | 1.01 | 1.01 | 64.4 | 94.6 | 9.6 |
| short-49208-sliding-cp8r4 | flashmla_fwd_ref | 353.7 | 1.02 | 1.02 | 185.2 | - | - |
| short-49208-sliding-cp8r7 | tilelang | 350.0 | 1.02 | 1.01 | 64.3 | 93.5 | 9.4 |
| short-49208-sliding-cp8r7 | flashmla_fwd_ref | 350.0 | 1.03 | 1.03 | 186.3 | - | - |
| heavy-49208-csa-cp1 | tilelang | 11959.6 | 1.03 | 1.02 | 229.1 | 193.8 | 19.6 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref | 11959.6 | 1.21 | 1.21 | 462.2 | - | - |
| heavy-49208-csa-cp8r0 | tilelang | 663.7 | 1.22 | 1.14 | 98.5 | 122.4 | 12.4 |
| heavy-49208-csa-cp8r0 | flashmla_fwd_ref | 663.7 | 2.72 | 2.72 | 249.6 | - | - |
| heavy-49208-csa-cp8r4 | tilelang | 1663.9 | 1.01 | 1.01 | 180.8 | 177.6 | 17.9 |
| heavy-49208-csa-cp8r4 | flashmla_fwd_ref | 1663.9 | 1.09 | 1.09 | 449.3 | - | - |
| heavy-49208-csa-cp8r7 | tilelang | 1454.9 | 1.03 | 1.02 | 167.8 | 169.5 | 17.1 |
| heavy-49208-csa-cp8r7 | flashmla_fwd_ref | 1454.9 | 1.24 | 1.24 | 419.7 | - | - |
| heavy-49208-hca-cp1 | tilelang | 4077.2 | 1.21 | 1.09 | 128.2 | 137.7 | 13.9 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref | 4077.2 | 2.13 | 2.13 | 272.6 | - | - |
| heavy-49208-hca-cp8r0 | tilelang | 318.8 | 1.45 | 1.21 | 53.5 | 81.5 | 8.2 |
| heavy-49208-hca-cp8r0 | flashmla_fwd_ref | 318.8 | 3.40 | 3.40 | 141.7 | - | - |
| heavy-49208-hca-cp8r4 | tilelang | 438.1 | 1.24 | 1.11 | 71.4 | 99.7 | 10.1 |
| heavy-49208-hca-cp8r4 | flashmla_fwd_ref | 438.1 | 2.47 | 2.47 | 185.4 | - | - |
| heavy-49208-hca-cp8r7 | tilelang | 507.2 | 1.25 | 1.08 | 78.6 | 108.5 | 11.0 |
| heavy-49208-hca-cp8r7 | flashmla_fwd_ref | 507.2 | 2.14 | 2.14 | 203.5 | - | - |
| heavy-49208-sliding-cp1 | tilelang | 2781.4 | 1.02 | 1.01 | 106.7 | 123.2 | 12.4 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref | 2781.4 | 1.04 | 1.04 | 253.5 | - | - |
| heavy-49208-sliding-cp8r0 | tilelang | 309.0 | 1.08 | 1.04 | 56.2 | 86.1 | 8.7 |
| heavy-49208-sliding-cp8r0 | flashmla_fwd_ref | 309.0 | 1.17 | 1.17 | 162.8 | - | - |
| heavy-49208-sliding-cp8r4 | tilelang | 361.2 | 1.00 | 1.00 | 65.9 | 96.0 | 9.7 |
| heavy-49208-sliding-cp8r4 | flashmla_fwd_ref | 361.2 | 1.00 | 1.00 | 191.5 | - | - |
| heavy-49208-sliding-cp8r7 | tilelang | 353.7 | 1.01 | 1.01 | 64.5 | 93.9 | 9.5 |
| heavy-49208-sliding-cp8r7 | flashmla_fwd_ref | 353.7 | 1.02 | 1.02 | 186.1 | - | - |
| tiny-49208-csa-cp1 | tilelang | 1202.7 | 3.49 | 2.89 | 40.9 | 49.9 | 5.0 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref | 1202.7 | 12.01 | 12.01 | 80.0 | - | - |
| tiny-49208-csa-cp8r0 | tilelang | 150.2 | 3.50 | 2.89 | 25.2 | 38.8 | 3.9 |
| tiny-49208-csa-cp8r0 | flashmla_fwd_ref | 150.2 | 12.03 | 12.03 | 65.2 | - | - |
| tiny-49208-csa-cp8r4 | tilelang | 156.7 | 3.35 | 2.78 | 26.6 | 40.1 | 4.1 |
| tiny-49208-csa-cp8r4 | flashmla_fwd_ref | 156.7 | 11.53 | 11.53 | 68.2 | - | - |
| tiny-49208-csa-cp8r7 | tilelang | 144.3 | 3.63 | 3.01 | 24.4 | 36.9 | 3.7 |
| tiny-49208-csa-cp8r7 | flashmla_fwd_ref | 144.3 | 12.51 | 12.51 | 62.5 | - | - |
| tiny-49208-hca-cp1 | tilelang | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref | 969.0 | 2.98 | 2.98 | 87.5 | - | - |
| tiny-49208-hca-cp8r0 | tilelang | 121.0 | 1.86 | 1.39 | 23.4 | 41.0 | 4.1 |
| tiny-49208-hca-cp8r0 | flashmla_fwd_ref | 121.0 | 2.99 | 2.99 | 63.8 | - | - |
| tiny-49208-hca-cp8r4 | tilelang | 126.2 | 1.83 | 1.38 | 24.3 | 42.3 | 4.3 |
| tiny-49208-hca-cp8r4 | flashmla_fwd_ref | 126.2 | 2.86 | 2.86 | 66.3 | - | - |
| tiny-49208-hca-cp8r7 | tilelang | 116.3 | 1.90 | 1.41 | 22.6 | 39.4 | 4.0 |
| tiny-49208-hca-cp8r7 | flashmla_fwd_ref | 116.3 | 3.11 | 3.11 | 61.5 | - | - |
| tiny-49208-sliding-cp1 | tilelang | 969.0 | 1.86 | 1.39 | 41.1 | 57.7 | 5.8 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref | 969.0 | 2.98 | 2.98 | 87.2 | - | - |
| tiny-49208-sliding-cp8r0 | tilelang | 121.0 | 1.86 | 1.39 | 23.6 | 41.1 | 4.1 |
| tiny-49208-sliding-cp8r0 | flashmla_fwd_ref | 121.0 | 2.99 | 2.99 | 64.0 | - | - |
| tiny-49208-sliding-cp8r4 | tilelang | 126.2 | 1.83 | 1.38 | 24.3 | 42.4 | 4.3 |
| tiny-49208-sliding-cp8r4 | flashmla_fwd_ref | 126.2 | 2.86 | 2.86 | 66.5 | - | - |
| tiny-49208-sliding-cp8r7 | tilelang | 116.3 | 1.90 | 1.41 | 22.7 | 39.4 | 4.0 |
| tiny-49208-sliding-cp8r7 | flashmla_fwd_ref | 116.3 | 3.11 | 3.11 | 61.3 | - | - |
| single-65536-csa-cp1 | tilelang | 18997.0 | 1.00 | 1.00 | 250.2 | 199.0 | 20.1 |
| single-65536-csa-cp1 | flashmla_fwd_ref | 18997.0 | 1.01 | 1.01 | 476.7 | - | - |
| single-65536-csa-cp8r0 | tilelang | 2160.7 | 1.02 | 1.01 | 189.1 | 181.1 | 18.3 |
| single-65536-csa-cp8r0 | flashmla_fwd_ref | 2160.7 | 1.11 | 1.11 | 465.9 | - | - |
| single-65536-csa-cp8r4 | tilelang | 2405.2 | 1.00 | 1.00 | 200.4 | 183.0 | 18.5 |
| single-65536-csa-cp8r4 | flashmla_fwd_ref | 2405.2 | 1.00 | 1.00 | 487.2 | - | - |
| single-65536-csa-cp8r7 | tilelang | 2405.2 | 1.00 | 1.00 | 203.1 | 171.9 | 17.4 |
| single-65536-csa-cp8r7 | flashmla_fwd_ref | 2405.2 | 1.00 | 1.00 | 487.3 | - | - |
| single-65536-hca-cp1 | tilelang | 11526.3 | 1.08 | 1.04 | 199.6 | 179.2 | 18.1 |
| single-65536-hca-cp1 | flashmla_fwd_ref | 11526.3 | 1.67 | 1.67 | 401.8 | - | - |
| single-65536-hca-cp8r0 | tilelang | 595.7 | 1.20 | 1.10 | 82.1 | 109.6 | 11.1 |
| single-65536-hca-cp8r0 | flashmla_fwd_ref | 595.7 | 4.04 | 4.04 | 203.8 | - | - |
| single-65536-hca-cp8r4 | tilelang | 1561.5 | 1.08 | 1.04 | 159.5 | 163.0 | 16.5 |
| single-65536-hca-cp8r4 | flashmla_fwd_ref | 1561.5 | 1.54 | 1.54 | 376.5 | - | - |
| single-65536-hca-cp8r7 | tilelang | 2283.1 | 1.05 | 1.03 | 192.4 | 179.9 | 18.2 |
| single-65536-hca-cp8r7 | flashmla_fwd_ref | 2283.1 | 1.05 | 1.05 | 471.2 | - | - |
| single-65536-sliding-cp1 | tilelang | 3844.6 | 1.00 | 1.00 | 113.8 | 127.5 | 12.9 |
| single-65536-sliding-cp1 | flashmla_fwd_ref | 3844.6 | 1.00 | 1.00 | 266.5 | - | - |
| single-65536-sliding-cp8r0 | tilelang | 477.3 | 1.00 | 1.00 | 72.9 | 102.5 | 10.4 |
| single-65536-sliding-cp8r0 | flashmla_fwd_ref | 477.3 | 1.01 | 1.01 | 206.2 | - | - |
| single-65536-sliding-cp8r4 | tilelang | 481.0 | 1.00 | 1.00 | 74.6 | 102.5 | 10.4 |
| single-65536-sliding-cp8r4 | flashmla_fwd_ref | 481.0 | 1.00 | 1.00 | 208.0 | - | - |
| single-65536-sliding-cp8r7 | tilelang | 481.0 | 1.00 | 1.00 | 74.2 | 102.0 | 10.3 |
| single-65536-sliding-cp8r7 | flashmla_fwd_ref | 481.0 | 1.00 | 1.00 | 207.5 | - | - |
| short-65536-csa-cp1 | tilelang | 11209.9 | 1.08 | 1.05 | 197.2 | 177.5 | 17.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref | 11209.9 | 1.72 | 1.72 | 393.2 | - | - |
| short-65536-csa-cp8r0 | tilelang | 885.0 | 1.18 | 1.11 | 110.6 | 130.5 | 13.2 |
| short-65536-csa-cp8r0 | flashmla_fwd_ref | 885.0 | 2.72 | 2.72 | 278.5 | - | - |
| short-65536-csa-cp8r4 | tilelang | 1689.8 | 1.05 | 1.03 | 168.7 | 167.7 | 16.9 |
| short-65536-csa-cp8r4 | flashmla_fwd_ref | 1689.8 | 1.42 | 1.42 | 412.8 | - | - |
| short-65536-csa-cp8r7 | tilelang | 1414.7 | 1.08 | 1.05 | 151.7 | 158.2 | 16.0 |
| short-65536-csa-cp8r7 | flashmla_fwd_ref | 1414.7 | 1.70 | 1.70 | 371.4 | - | - |
| short-65536-hca-cp1 | tilelang | 3930.2 | 1.40 | 1.17 | 102.8 | 117.2 | 11.8 |
| short-65536-hca-cp1 | flashmla_fwd_ref | 3930.2 | 1.96 | 1.96 | 208.8 | - | - |
| short-65536-hca-cp8r0 | tilelang | 455.7 | 1.46 | 1.22 | 64.5 | 90.7 | 9.2 |
| short-65536-hca-cp8r0 | flashmla_fwd_ref | 455.7 | 2.11 | 2.11 | 160.5 | - | - |
| short-65536-hca-cp8r4 | tilelang | 514.5 | 1.37 | 1.14 | 71.6 | 99.0 | 10.0 |
| short-65536-hca-cp8r4 | flashmla_fwd_ref | 514.5 | 1.87 | 1.87 | 176.6 | - | - |
| short-65536-hca-cp8r7 | tilelang | 491.0 | 1.40 | 1.17 | 68.7 | 95.0 | 9.6 |
| short-65536-hca-cp8r7 | flashmla_fwd_ref | 491.0 | 1.96 | 1.96 | 171.1 | - | - |
| short-65536-sliding-cp1 | tilelang | 3673.0 | 1.02 | 1.01 | 109.2 | 123.9 | 12.5 |
| short-65536-sliding-cp1 | flashmla_fwd_ref | 3673.0 | 1.05 | 1.05 | 255.6 | - | - |
| short-65536-sliding-cp8r0 | tilelang | 443.7 | 1.04 | 1.02 | 68.1 | 97.4 | 9.8 |
| short-65536-sliding-cp8r0 | flashmla_fwd_ref | 443.7 | 1.08 | 1.08 | 190.4 | - | - |
| short-65536-sliding-cp8r4 | tilelang | 469.9 | 1.01 | 1.01 | 73.0 | 101.6 | 10.3 |
| short-65536-sliding-cp8r4 | flashmla_fwd_ref | 469.9 | 1.02 | 1.02 | 203.3 | - | - |
| short-65536-sliding-cp8r7 | tilelang | 458.7 | 1.02 | 1.01 | 71.2 | 99.0 | 10.0 |
| short-65536-sliding-cp8r7 | flashmla_fwd_ref | 458.7 | 1.05 | 1.05 | 197.9 | - | - |
| heavy-65536-csa-cp1 | tilelang | 12785.2 | 1.07 | 1.04 | 210.3 | 183.4 | 18.5 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref | 12785.2 | 1.50 | 1.50 | 414.1 | - | - |
| heavy-65536-csa-cp8r0 | tilelang | 771.9 | 1.27 | 1.18 | 98.5 | 120.2 | 12.2 |
| heavy-65536-csa-cp8r0 | flashmla_fwd_ref | 771.9 | 3.12 | 3.12 | 244.8 | - | - |
| heavy-65536-csa-cp8r4 | tilelang | 2405.2 | 1.00 | 1.00 | 202.0 | 185.4 | 18.7 |
| heavy-65536-csa-cp8r4 | flashmla_fwd_ref | 2405.2 | 1.00 | 1.00 | 490.2 | - | - |
| heavy-65536-csa-cp8r7 | tilelang | 1250.3 | 1.12 | 1.08 | 140.1 | 150.7 | 15.2 |
| heavy-65536-csa-cp8r7 | flashmla_fwd_ref | 1250.3 | 1.92 | 1.92 | 341.0 | - | - |
| heavy-65536-hca-cp1 | tilelang | 5141.1 | 1.25 | 1.11 | 125.4 | 134.3 | 13.6 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref | 5141.1 | 2.25 | 2.25 | 258.5 | - | - |
| heavy-65536-hca-cp8r0 | tilelang | 408.9 | 1.46 | 1.22 | 58.7 | 84.8 | 8.6 |
| heavy-65536-hca-cp8r0 | flashmla_fwd_ref | 408.9 | 3.53 | 3.53 | 150.4 | - | - |
| heavy-65536-hca-cp8r4 | tilelang | 965.4 | 1.12 | 1.06 | 116.0 | 136.9 | 13.8 |
| heavy-65536-hca-cp8r4 | flashmla_fwd_ref | 965.4 | 1.49 | 1.49 | 297.2 | - | - |
| heavy-65536-hca-cp8r7 | tilelang | 458.3 | 1.40 | 1.17 | 65.3 | 91.5 | 9.2 |
| heavy-65536-hca-cp8r7 | flashmla_fwd_ref | 458.3 | 3.15 | 3.15 | 163.9 | - | - |
| heavy-65536-sliding-cp1 | tilelang | 3542.5 | 1.04 | 1.02 | 105.3 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref | 3542.5 | 1.09 | 1.09 | 246.2 | - | - |
| heavy-65536-sliding-cp8r0 | tilelang | 399.0 | 1.10 | 1.05 | 61.5 | 90.6 | 9.2 |
| heavy-65536-sliding-cp8r0 | flashmla_fwd_ref | 399.0 | 1.21 | 1.21 | 170.2 | - | - |
| heavy-65536-sliding-cp8r4 | tilelang | 481.0 | 1.00 | 1.00 | 74.7 | 102.8 | 10.4 |
| heavy-65536-sliding-cp8r4 | flashmla_fwd_ref | 481.0 | 1.00 | 1.00 | 209.5 | - | - |
| heavy-65536-sliding-cp8r7 | tilelang | 428.8 | 1.06 | 1.03 | 67.2 | 94.8 | 9.6 |
| heavy-65536-sliding-cp8r7 | flashmla_fwd_ref | 428.8 | 1.12 | 1.12 | 184.1 | - | - |
| tiny-65536-csa-cp1 | tilelang | 1621.2 | 3.45 | 2.86 | 42.6 | 51.0 | 5.2 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref | 1621.2 | 11.87 | 11.87 | 81.8 | - | - |
| tiny-65536-csa-cp8r0 | tilelang | 202.8 | 3.45 | 2.86 | 28.9 | 41.5 | 4.2 |
| tiny-65536-csa-cp8r0 | flashmla_fwd_ref | 202.8 | 11.86 | 11.86 | 71.6 | - | - |
| tiny-65536-csa-cp8r4 | tilelang | 202.7 | 3.45 | 2.86 | 28.8 | 41.6 | 4.2 |
| tiny-65536-csa-cp8r4 | flashmla_fwd_ref | 202.7 | 11.87 | 11.87 | 70.3 | - | - |
| tiny-65536-csa-cp8r7 | tilelang | 201.4 | 3.47 | 2.88 | 27.9 | 41.3 | 4.2 |
| tiny-65536-csa-cp8r7 | flashmla_fwd_ref | 201.4 | 11.94 | 11.94 | 69.2 | - | - |
| tiny-65536-hca-cp1 | tilelang | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref | 1306.0 | 2.95 | 2.95 | 89.4 | - | - |
| tiny-65536-hca-cp8r0 | tilelang | 163.4 | 1.85 | 1.39 | 27.4 | 45.0 | 4.6 |
| tiny-65536-hca-cp8r0 | flashmla_fwd_ref | 163.4 | 2.94 | 2.94 | 71.0 | - | - |
| tiny-65536-hca-cp8r4 | tilelang | 163.3 | 1.85 | 1.39 | 26.9 | 45.2 | 4.6 |
| tiny-65536-hca-cp8r4 | flashmla_fwd_ref | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-hca-cp8r7 | tilelang | 162.3 | 1.86 | 1.39 | 27.0 | 44.7 | 4.5 |
| tiny-65536-hca-cp8r7 | flashmla_fwd_ref | 162.3 | 2.96 | 2.96 | 69.6 | - | - |
| tiny-65536-sliding-cp1 | tilelang | 1306.0 | 1.85 | 1.39 | 42.8 | 59.3 | 6.0 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref | 1306.0 | 2.95 | 2.95 | 89.3 | - | - |
| tiny-65536-sliding-cp8r0 | tilelang | 163.4 | 1.85 | 1.39 | 26.7 | 45.0 | 4.5 |
| tiny-65536-sliding-cp8r0 | flashmla_fwd_ref | 163.4 | 2.94 | 2.94 | 69.8 | - | - |
| tiny-65536-sliding-cp8r4 | tilelang | 163.3 | 1.85 | 1.39 | 27.2 | 45.2 | 4.6 |
| tiny-65536-sliding-cp8r4 | flashmla_fwd_ref | 163.3 | 2.95 | 2.95 | 70.4 | - | - |
| tiny-65536-sliding-cp8r7 | tilelang | 162.3 | 1.86 | 1.39 | 27.2 | 44.7 | 4.5 |
| tiny-65536-sliding-cp8r7 | flashmla_fwd_ref | 162.3 | 2.96 | 2.96 | 69.7 | - | - |

Correctness failures (excluded from timing): 0
