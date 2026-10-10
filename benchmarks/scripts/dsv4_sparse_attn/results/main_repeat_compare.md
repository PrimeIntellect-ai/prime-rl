- `main`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git 6d5cf0180
- `main-repeat`: NVIDIA H200, driver 580.173.02, SM clock 345 MHz (max 1980 MHz), power limit 700.00 W, host prime-nebius-puku-h200-gpu-059, git 6d5cf0180
- corpus hash b5b289171983c8c6; synthetic corpus: random-weight CSA picks are near-uniform, while a
  trained indexer favors recent and neighboring entries, so CSA gather locality here is pessimistic.

Op-boundary time per call in µs (lower is better): median over rounds, p20-p80 across rounds.
`/TL` is this time divided by tilelang's in the same run.

| item | backend | fwd µs | fwd p20-p80 | fwd /TL | f+b µs | f+b p20-p80 | f+b /TL |
|---|---|---|---|---|---|---|---|
| single-4096-csa-cp8r4 | tilelang@main | 869.0 | 862.6-883.1 | 1.00 | 2333 | 2321-2341 | 1.00 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 248.4 | 245.0-250.7 | 0.29 | - | - | - |
| single-4096-csa-cp8r4 | tilelang@main-repeat | 841.0 | 829.1-864.3 | 1.00 | 2283 | 2280-2309 | 1.00 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 210.9 | 209.4-215.9 | 0.25 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 824.2 | 816.0-832.5 | 1.00 | 2143 | 2120-2164 | 1.00 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 216.8 | 214.5-219.7 | 0.26 | - | - | - |
| single-4096-hca-cp8r4 | tilelang@main-repeat | 814.1 | 802.0-819.2 | 1.00 | 2105 | 2092-2117 | 1.00 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 217.9 | 205.9-235.9 | 0.27 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 791.1 | 779.7-892.0 | 1.00 | 2057 | 2032-2070 | 1.00 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 197.7-217.5 | 0.26 | - | - | - |
| single-4096-sliding-cp8r4 | tilelang@main-repeat | 775.8 | 766.7-810.9 | 1.00 | 2013 | 2004-2017 | 1.00 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 195.0 | 191.8-197.9 | 0.25 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 876.9 | 821.4-901.4 | 1.00 | 2145 | 2070-2654 | 1.00 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 227.1 | 220.9-229.2 | 0.26 | - | - | - |
| short-4096-csa-cp8r4 | tilelang@main-repeat | 804.1 | 803.4-812.9 | 1.00 | 2020 | 1990-2049 | 1.00 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 206.8 | 205.1-208.2 | 0.26 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 826.5 | 818.9-845.1 | 1.00 | 2157 | 2154-2171 | 1.00 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 208.2 | 206.1-217.1 | 0.25 | - | - | - |
| short-4096-hca-cp8r4 | tilelang@main-repeat | 819.8 | 811.4-830.2 | 1.00 | 2113 | 2101-2116 | 1.00 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 204.1 | 200.3-205.0 | 0.25 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 794.0 | 780.5-815.6 | 1.00 | 2041 | 2033-2066 | 1.00 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 197.7 | 193.1-216.8 | 0.25 | - | - | - |
| short-4096-sliding-cp8r4 | tilelang@main-repeat | 766.1 | 755.2-770.5 | 1.00 | 2011 | 1990-2030 | 1.00 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 195.7 | 190.5-197.9 | 0.26 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 817.9 | 805.9-831.3 | 1.00 | 2049 | 2038-2194 | 1.00 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 217.3 | 211.7-218.4 | 0.27 | - | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main-repeat | 802.1 | 788.6-811.8 | 1.00 | 2028 | 2013-2066 | 1.00 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 206.0 | 200.5-207.8 | 0.26 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 827.9 | 820.9-840.8 | 1.00 | 2160 | 2157-2177 | 1.00 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 207.4 | 205.8-212.8 | 0.25 | - | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main-repeat | 822.9 | 813.9-826.0 | 1.00 | 2098 | 2091-2115 | 1.00 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 205.2 | 204.7-209.0 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 780.6 | 767.1-784.0 | 1.00 | 2050 | 2037-2053 | 1.00 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 194.3 | 190.0-200.3 | 0.25 | - | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main-repeat | 784.0 | 773.5-789.4 | 1.00 | 2059 | 2034-2082 | 1.00 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 197.0 | 193.9-199.4 | 0.25 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 809.3 | 793.1-830.6 | 1.00 | 2034 | 2014-2048 | 1.00 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 209.6 | 207.2-216.1 | 0.26 | - | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main-repeat | 785.2 | 782.3-795.9 | 1.00 | 2019 | 2001-2030 | 1.00 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 207.8 | 205.3-211.3 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 767.4 | 763.6-775.5 | 1.00 | 2023 | 2018-2038 | 1.00 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 196.6 | 192.9-202.8 | 0.26 | - | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main-repeat | 771.1 | 760.9-785.4 | 1.00 | 2017 | 2006-2037 | 1.00 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 191.4 | 189.2-196.6 | 0.25 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 781.6 | 770.1-787.6 | 1.00 | 2040 | 2028-2056 | 1.00 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 203.1 | 201.3-214.2 | 0.26 | - | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main-repeat | 772.3 | 759.9-852.9 | 1.00 | 2014 | 2007-2036 | 1.00 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 194.8 | 188.0-196.2 | 0.25 | - | - | - |
| single-65536-csa-cp1 | tilelang@main | 21694 | 21617-21724 | 1.00 | 95444 | 95324-95476 | 1.00 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 11385 | 11377-11470 | 0.52 | - | - | - |
| single-65536-csa-cp1 | tilelang@main-repeat | 21778 | 21732-21878 | 1.00 | 95347 | 95252-95414 | 1.00 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 11282 | 11138-11354 | 0.52 | - | - | - |
| single-65536-hca-cp1 | tilelang@main | 16503 | 16484-16545 | 1.00 | 64327 | 64312-64341 | 1.00 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 8196 | 7971-8458 | 0.50 | - | - | - |
| single-65536-hca-cp1 | tilelang@main-repeat | 16555 | 16482-16579 | 1.00 | 64341 | 64325-64369 | 1.00 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 8263 | 8146-8381 | 0.50 | - | - | - |
| single-65536-sliding-cp1 | tilelang@main | 9650 | 9637-9666 | 1.00 | 30146 | 30143-30155 | 1.00 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4122 | 4112-4130 | 0.43 | - | - | - |
| single-65536-sliding-cp1 | tilelang@main-repeat | 9614 | 9610-9647 | 1.00 | 30096 | 30085-30109 | 1.00 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4105 | 4096-4110 | 0.43 | - | - | - |
| short-65536-csa-cp1 | tilelang@main | 16242 | 16220-16289 | 1.00 | 63170 | 63160-63183 | 1.00 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 8146 | 8080-8320 | 0.50 | - | - | - |
| short-65536-csa-cp1 | tilelang@main-repeat | 16221 | 16188-16327 | 1.00 | 63124 | 63120-63142 | 1.00 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 8125 | 8022-8325 | 0.50 | - | - | - |
| short-65536-hca-cp1 | tilelang@main | 10922 | 10906-10936 | 1.00 | 33532 | 33528-33548 | 1.00 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5378 | 5372-5387 | 0.49 | - | - | - |
| short-65536-hca-cp1 | tilelang@main-repeat | 10927 | 10897-10954 | 1.00 | 33517 | 33501-33530 | 1.00 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 5383 | 5367-5388 | 0.49 | - | - | - |
| short-65536-sliding-cp1 | tilelang@main | 9612 | 9605-9628 | 1.00 | 29641 | 29634-29651 | 1.00 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4105 | 4097-4107 | 0.43 | - | - | - |
| short-65536-sliding-cp1 | tilelang@main-repeat | 9584 | 9577-9603 | 1.00 | 29621 | 29609-29626 | 1.00 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4107 | 4098-4112 | 0.43 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 17373 | 17342-17419 | 1.00 | 69696 | 69684-69701 | 1.00 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8820 | 8665-9037 | 0.51 | - | - | - |
| heavy-65536-csa-cp1 | tilelang@main-repeat | 17375 | 17343-17450 | 1.00 | 69637 | 69619-69654 | 1.00 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 8829 | 8732-8939 | 0.51 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 11714 | 11677-11736 | 1.00 | 38272 | 38241-38292 | 1.00 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5682 | 5674-5693 | 0.49 | - | - | - |
| heavy-65536-hca-cp1 | tilelang@main-repeat | 11756 | 11678-11767 | 1.00 | 38237 | 38224-38245 | 1.00 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 5681 | 5667-5697 | 0.48 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 9616 | 9585-9660 | 1.00 | 29282 | 29271-29295 | 1.00 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4111 | 4108-4115 | 0.43 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang@main-repeat | 9574 | 9563-9585 | 1.00 | 29226 | 29225-29228 | 1.00 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4111 | 4107-4116 | 0.43 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 10884 | 10872-10885 | 1.00 | 31800 | 31799-31808 | 1.00 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5659 | 5656-5664 | 0.52 | - | - | - |
| tiny-65536-csa-cp1 | tilelang@main-repeat | 10882 | 10865-10914 | 1.00 | 31768 | 31758-31769 | 1.00 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 5673 | 5663-5683 | 0.52 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 8689 | 8676-8718 | 1.00 | 22022 | 21996-22029 | 1.00 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4173 | 4168-4184 | 0.48 | - | - | - |
| tiny-65536-hca-cp1 | tilelang@main-repeat | 8646 | 8632-8668 | 1.00 | 21958 | 21945-21970 | 1.00 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 4159 | 4157-4175 | 0.48 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 8717 | 8700-8727 | 1.00 | 22036 | 22028-22065 | 1.00 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4178 | 4164-4186 | 0.48 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang@main-repeat | 8707 | 8641-8747 | 1.00 | 21983 | 21952-22068 | 1.00 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4165 | 4161-4169 | 0.48 | - | - | - |

GPU busy time per call in µs from profiler traces (lower is better); `host` is op-boundary minus GPU
busy time (launch overhead and gaps); `/TL` divides GPU busy time by tilelang's; `peak MiB` is the
allocation above the inputs during one call, forward+backward where the backend has it, else forward.

| item | backend | fwd gpu µs | fwd host | fwd /TL | f+b gpu µs | f+b host | f+b /TL | peak MiB |
|---|---|---|---|---|---|---|---|---|
| single-4096-csa-cp8r4 | tilelang@main | 191.2 | 677.8 | 1.00 | 856.0 | 1477 | 1.00 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 109.5 | 138.9 | 0.57 | - | - | - | 35 |
| single-4096-csa-cp8r4 | tilelang@main-repeat | 192.6 | 648.4 | 1.00 | 863.8 | 1419 | 1.00 | 79 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 110.2 | 100.7 | 0.57 | - | - | - | 35 |
| single-4096-hca-cp8r4 | tilelang@main | 107.4 | 716.8 | 1.00 | 370.8 | 1772 | 1.00 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 70.2 | 146.6 | 0.65 | - | - | - | 33 |
| single-4096-hca-cp8r4 | tilelang@main-repeat | 107.1 | 707.0 | 1.00 | 370.4 | 1735 | 1.00 | 77 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 69.7 | 148.2 | 0.65 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@main | 92.4 | 698.6 | 1.00 | 327.4 | 1730 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.7 | 149.4 | 0.58 | - | - | - | 33 |
| single-4096-sliding-cp8r4 | tilelang@main-repeat | 92.5 | 683.3 | 1.00 | 328.2 | 1684 | 1.00 | 76 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 54.5 | 140.4 | 0.59 | - | - | - | 33 |
| short-4096-csa-cp8r4 | tilelang@main | 114.7 | 762.2 | 1.00 | 432.7 | 1712 | 1.00 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.3 | 154.8 | 0.63 | - | - | - | 35 |
| short-4096-csa-cp8r4 | tilelang@main-repeat | 115.2 | 688.9 | 1.00 | 434.9 | 1585 | 1.00 | 79 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 73.5 | 133.2 | 0.64 | - | - | - | 35 |
| short-4096-hca-cp8r4 | tilelang@main | 101.5 | 725.0 | 1.00 | 354.4 | 1803 | 1.00 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 65.3 | 142.9 | 0.64 | - | - | - | 33 |
| short-4096-hca-cp8r4 | tilelang@main-repeat | 101.7 | 718.1 | 1.00 | 355.1 | 1758 | 1.00 | 77 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 64.8 | 139.2 | 0.64 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@main | 89.9 | 704.2 | 1.00 | 319.7 | 1722 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.1 | 144.6 | 0.59 | - | - | - | 33 |
| short-4096-sliding-cp8r4 | tilelang@main-repeat | 90.8 | 675.3 | 1.00 | 321.9 | 1690 | 1.00 | 76 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 54.1 | 141.6 | 0.60 | - | - | - | 33 |
| heavy-4096-csa-cp8r4 | tilelang@main | 114.6 | 703.3 | 1.00 | 434.1 | 1615 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 72.1 | 145.2 | 0.63 | - | - | - | 35 |
| heavy-4096-csa-cp8r4 | tilelang@main-repeat | 115.1 | 686.9 | 1.00 | 436.9 | 1591 | 1.00 | 79 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 73.2 | 132.8 | 0.64 | - | - | - | 35 |
| heavy-4096-hca-cp8r4 | tilelang@main | 102.2 | 725.7 | 1.00 | 357.1 | 1803 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 64.6 | 142.8 | 0.63 | - | - | - | 33 |
| heavy-4096-hca-cp8r4 | tilelang@main-repeat | 102.7 | 720.2 | 1.00 | 358.1 | 1740 | 1.00 | 77 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 65.6 | 139.6 | 0.64 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@main | 90.6 | 690.0 | 1.00 | 321.4 | 1729 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 53.9 | 140.4 | 0.59 | - | - | - | 33 |
| heavy-4096-sliding-cp8r4 | tilelang@main-repeat | 90.9 | 693.2 | 1.00 | 322.9 | 1736 | 1.00 | 76 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 53.9 | 143.1 | 0.59 | - | - | - | 33 |
| tiny-4096-csa-cp8r4 | tilelang@main | 104.5 | 704.8 | 1.00 | 345.8 | 1688 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 67.1 | 142.5 | 0.64 | - | - | - | 35 |
| tiny-4096-csa-cp8r4 | tilelang@main-repeat | 104.8 | 680.4 | 1.00 | 346.5 | 1673 | 1.00 | 79 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 67.6 | 140.1 | 0.65 | - | - | - | 35 |
| tiny-4096-hca-cp8r4 | tilelang@main | 84.2 | 683.2 | 1.00 | 267.7 | 1755 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 53.3 | 143.4 | 0.63 | - | - | - | 33 |
| tiny-4096-hca-cp8r4 | tilelang@main-repeat | 84.4 | 686.7 | 1.00 | 267.7 | 1749 | 1.00 | 76 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 53.3 | 138.1 | 0.63 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@main | 84.0 | 697.6 | 1.00 | 267.6 | 1772 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 52.9 | 150.1 | 0.63 | - | - | - | 33 |
| tiny-4096-sliding-cp8r4 | tilelang@main-repeat | 84.4 | 687.9 | 1.00 | 267.4 | 1747 | 1.00 | 76 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 53.8 | 141.0 | 0.64 | - | - | - | 33 |
| single-65536-csa-cp1 | tilelang@main | 21804 | -109.6 | 1.00 | 94799 | 644.7 | 1.00 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 10352 | 1033 | 0.47 | - | - | - | 4448 |
| single-65536-csa-cp1 | tilelang@main-repeat | 21591 | 186.9 | 1.00 | 94646 | 700.7 | 1.00 | 8480 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 10700 | 582.3 | 0.50 | - | - | - | 4448 |
| single-65536-hca-cp1 | tilelang@main | 16321 | 181.6 | 1.00 | 63704 | 622.9 | 1.00 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 7958 | 238.2 | 0.49 | - | - | - | 4448 |
| single-65536-hca-cp1 | tilelang@main-repeat | 16362 | 192.8 | 1.00 | 63692 | 649.1 | 1.00 | 8434 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 7963 | 300.5 | 0.49 | - | - | - | 4448 |
| single-65536-sliding-cp1 | tilelang@main | 8998 | 652.7 | 1.00 | 29308 | 838.2 | 1.00 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 4089 | 32.9 | 0.45 | - | - | - | 4192 |
| single-65536-sliding-cp1 | tilelang@main-repeat | 9001 | 612.9 | 1.00 | 29298 | 798.6 | 1.00 | 8432 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4078 | 27.1 | 0.45 | - | - | - | 4192 |
| short-65536-csa-cp1 | tilelang@main | 16000 | 242.3 | 1.00 | 62509 | 660.3 | 1.00 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 7830 | 315.8 | 0.49 | - | - | - | 4448 |
| short-65536-csa-cp1 | tilelang@main-repeat | 16321 | -99.6 | 1.00 | 62532 | 592.4 | 1.00 | 8480 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 7804 | 321.2 | 0.48 | - | - | - | 4448 |
| short-65536-hca-cp1 | tilelang@main | 10309 | 612.7 | 1.00 | 32762 | 770.6 | 1.00 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 5376 | 1.8 | 0.52 | - | - | - | 4256 |
| short-65536-hca-cp1 | tilelang@main-repeat | 10302 | 624.6 | 1.00 | 32733 | 784.0 | 1.00 | 8482 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 5386 | -2.8 | 0.52 | - | - | - | 4256 |
| short-65536-sliding-cp1 | tilelang@main | 8974 | 637.9 | 1.00 | 28844 | 797.0 | 1.00 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 4081 | 23.9 | 0.45 | - | - | - | 4192 |
| short-65536-sliding-cp1 | tilelang@main-repeat | 8961 | 623.0 | 1.00 | 28820 | 801.3 | 1.00 | 8432 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4092 | 15.2 | 0.46 | - | - | - | 4192 |
| heavy-65536-csa-cp1 | tilelang@main | 17414 | -41.5 | 1.00 | 69035 | 660.4 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 8439 | 381.8 | 0.48 | - | - | - | 4448 |
| heavy-65536-csa-cp1 | tilelang@main-repeat | 17487 | -111.4 | 1.00 | 69014 | 622.7 | 1.00 | 8480 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 8430 | 399.0 | 0.48 | - | - | - | 4448 |
| heavy-65536-hca-cp1 | tilelang@main | 11197 | 516.8 | 1.00 | 37594 | 678.6 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5693 | -10.7 | 0.51 | - | - | - | 4320 |
| heavy-65536-hca-cp1 | tilelang@main-repeat | 11200 | 555.6 | 1.00 | 37609 | 627.5 | 1.00 | 8530 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 5709 | -27.2 | 0.51 | - | - | - | 4320 |
| heavy-65536-sliding-cp1 | tilelang@main | 8945 | 671.5 | 1.00 | 28442 | 840.0 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 4091 | 20.5 | 0.46 | - | - | - | 4192 |
| heavy-65536-sliding-cp1 | tilelang@main-repeat | 8943 | 631.3 | 1.00 | 28423 | 803.4 | 1.00 | 8432 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4100 | 10.8 | 0.46 | - | - | - | 4192 |
| tiny-65536-csa-cp1 | tilelang@main | 10414 | 470.2 | 1.00 | 31178 | 622.0 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 5690 | -30.5 | 0.55 | - | - | - | 4448 |
| tiny-65536-csa-cp1 | tilelang@main-repeat | 10413 | 469.0 | 1.00 | 31164 | 603.9 | 1.00 | 8479 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 5686 | -13.6 | 0.55 | - | - | - | 4448 |
| tiny-65536-hca-cp1 | tilelang@main | 8038 | 651.6 | 1.00 | 21189 | 833.7 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 4147 | 25.6 | 0.52 | - | - | - | 4192 |
| tiny-65536-hca-cp1 | tilelang@main-repeat | 8029 | 616.3 | 1.00 | 21153 | 805.6 | 1.00 | 8432 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 4138 | 20.9 | 0.52 | - | - | - | 4192 |
| tiny-65536-sliding-cp1 | tilelang@main | 8031 | 685.8 | 1.00 | 21186 | 850.2 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 4116 | 62.0 | 0.51 | - | - | - | 4192 |
| tiny-65536-sliding-cp1 | tilelang@main-repeat | 8034 | 673.1 | 1.00 | 21177 | 805.9 | 1.00 | 8432 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 4123 | 42.0 | 0.51 | - | - | - | 4192 |

Useful FLOPs count valid slots only (fwd 4HD, bwd 10HD per slot); `exec/useful` counts the slots each
backend's tiles touch, or every padded slot for an arm without tile information. TFLOP/s divide
useful FLOPs by op-boundary time (higher is better);
`% peak` is f+b against 989.5 dense BF16 TFLOP/s (https://www.nvidia.com/en-us/data-center/h200/ (H200 SXM BF16 1,979 TFLOPS with sparsity, halved)).

| item | backend | f+b GFLOP | exec/useful fwd | exec/useful bwd | fwd TFLOP/s | f+b TFLOP/s | % peak |
|---|---|---|---|---|---|---|---|
| single-4096-csa-cp8r4 | tilelang@main | 150.3 | 1.00 | 1.00 | 49.4 | 64.4 | 6.5 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main | 150.3 | 1.00 | 1.00 | 172.9 | - | - |
| single-4096-csa-cp8r4 | tilelang@main-repeat | 150.3 | 1.00 | 1.00 | 51.1 | 65.9 | 6.7 |
| single-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 150.3 | 1.00 | 1.00 | 203.7 | - | - |
| single-4096-hca-cp8r4 | tilelang@main | 34.2 | 1.32 | 1.10 | 11.8 | 16.0 | 1.6 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main | 34.2 | 1.76 | 1.76 | 45.0 | - | - |
| single-4096-hca-cp8r4 | tilelang@main-repeat | 34.2 | 1.32 | 1.10 | 12.0 | 16.2 | 1.6 |
| single-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 34.2 | 1.76 | 1.76 | 44.8 | - | - |
| single-4096-sliding-cp8r4 | tilelang@main | 30.1 | 1.00 | 1.00 | 10.9 | 14.6 | 1.5 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 30.1 | 1.00 | 1.00 | 42.3 | - | - |
| single-4096-sliding-cp8r4 | tilelang@main-repeat | 30.1 | 1.00 | 1.00 | 11.1 | 14.9 | 1.5 |
| single-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 30.1 | 1.00 | 1.00 | 44.1 | - | - |
| short-4096-csa-cp8r4 | tilelang@main | 44.6 | 1.21 | 1.13 | 14.5 | 20.8 | 2.1 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main | 44.6 | 3.37 | 3.37 | 56.1 | - | - |
| short-4096-csa-cp8r4 | tilelang@main-repeat | 44.6 | 1.21 | 1.13 | 15.9 | 22.1 | 2.2 |
| short-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 44.6 | 3.37 | 3.37 | 61.7 | - | - |
| short-4096-hca-cp8r4 | tilelang@main | 28.3 | 1.46 | 1.23 | 9.8 | 13.1 | 1.3 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.3 | 2.13 | 2.13 | 38.8 | - | - |
| short-4096-hca-cp8r4 | tilelang@main-repeat | 28.3 | 1.46 | 1.23 | 9.9 | 13.4 | 1.4 |
| short-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 28.3 | 2.13 | 2.13 | 39.6 | - | - |
| short-4096-sliding-cp8r4 | tilelang@main | 27.9 | 1.04 | 1.02 | 10.0 | 13.7 | 1.4 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 27.9 | 1.08 | 1.08 | 40.3 | - | - |
| short-4096-sliding-cp8r4 | tilelang@main-repeat | 27.9 | 1.04 | 1.02 | 10.4 | 13.9 | 1.4 |
| short-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 27.9 | 1.08 | 1.08 | 40.7 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main | 45.5 | 1.20 | 1.12 | 15.9 | 22.2 | 2.2 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main | 45.5 | 3.30 | 3.30 | 59.9 | - | - |
| heavy-4096-csa-cp8r4 | tilelang@main-repeat | 45.5 | 1.20 | 1.12 | 16.2 | 22.4 | 2.3 |
| heavy-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 45.5 | 3.30 | 3.30 | 63.1 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main | 28.7 | 1.46 | 1.22 | 9.9 | 13.3 | 1.3 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main | 28.7 | 2.10 | 2.10 | 39.5 | - | - |
| heavy-4096-hca-cp8r4 | tilelang@main-repeat | 28.7 | 1.46 | 1.22 | 10.0 | 13.7 | 1.4 |
| heavy-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 28.7 | 2.10 | 2.10 | 39.9 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main | 28.3 | 1.04 | 1.02 | 10.3 | 13.8 | 1.4 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 28.3 | 1.06 | 1.06 | 41.6 | - | - |
| heavy-4096-sliding-cp8r4 | tilelang@main-repeat | 28.3 | 1.04 | 1.02 | 10.3 | 13.7 | 1.4 |
| heavy-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 28.3 | 1.06 | 1.06 | 41.0 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main | 15.1 | 2.92 | 2.42 | 5.3 | 7.4 | 0.7 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main | 15.1 | 9.98 | 9.98 | 20.5 | - | - |
| tiny-4096-csa-cp8r4 | tilelang@main-repeat | 15.1 | 2.92 | 2.42 | 5.5 | 7.5 | 0.8 |
| tiny-4096-csa-cp8r4 | flashmla_fwd_ref@main-repeat | 15.1 | 9.98 | 9.98 | 20.7 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.5 | 6.0 | 0.6 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.6 | - | - |
| tiny-4096-hca-cp8r4 | tilelang@main-repeat | 12.1 | 1.66 | 1.32 | 4.5 | 6.0 | 0.6 |
| tiny-4096-hca-cp8r4 | flashmla_fwd_ref@main-repeat | 12.1 | 2.48 | 2.48 | 18.1 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main | 12.1 | 1.66 | 1.32 | 4.4 | 5.9 | 0.6 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main | 12.1 | 2.48 | 2.48 | 17.1 | - | - |
| tiny-4096-sliding-cp8r4 | tilelang@main-repeat | 12.1 | 1.66 | 1.32 | 4.5 | 6.0 | 0.6 |
| tiny-4096-sliding-cp8r4 | flashmla_fwd_ref@main-repeat | 12.1 | 2.48 | 2.48 | 17.8 | - | - |
| single-65536-csa-cp1 | tilelang@main | 18997.0 | 1.00 | 1.00 | 250.2 | 199.0 | 20.1 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main | 18997.0 | 1.01 | 1.01 | 476.7 | - | - |
| single-65536-csa-cp1 | tilelang@main-repeat | 18997.0 | 1.00 | 1.00 | 249.2 | 199.2 | 20.1 |
| single-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 18997.0 | 1.01 | 1.01 | 481.1 | - | - |
| single-65536-hca-cp1 | tilelang@main | 11526.3 | 1.08 | 1.04 | 199.6 | 179.2 | 18.1 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main | 11526.3 | 1.67 | 1.67 | 401.8 | - | - |
| single-65536-hca-cp1 | tilelang@main-repeat | 11526.3 | 1.08 | 1.04 | 198.9 | 179.1 | 18.1 |
| single-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 11526.3 | 1.67 | 1.67 | 398.5 | - | - |
| single-65536-sliding-cp1 | tilelang@main | 3844.6 | 1.00 | 1.00 | 113.8 | 127.5 | 12.9 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main | 3844.6 | 1.00 | 1.00 | 266.5 | - | - |
| single-65536-sliding-cp1 | tilelang@main-repeat | 3844.6 | 1.00 | 1.00 | 114.3 | 127.7 | 12.9 |
| single-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 3844.6 | 1.00 | 1.00 | 267.6 | - | - |
| short-65536-csa-cp1 | tilelang@main | 11209.9 | 1.08 | 1.05 | 197.2 | 177.5 | 17.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main | 11209.9 | 1.72 | 1.72 | 393.2 | - | - |
| short-65536-csa-cp1 | tilelang@main-repeat | 11209.9 | 1.08 | 1.05 | 197.4 | 177.6 | 17.9 |
| short-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 11209.9 | 1.72 | 1.72 | 394.2 | - | - |
| short-65536-hca-cp1 | tilelang@main | 3930.2 | 1.40 | 1.17 | 102.8 | 117.2 | 11.8 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main | 3930.2 | 1.96 | 1.96 | 208.8 | - | - |
| short-65536-hca-cp1 | tilelang@main-repeat | 3930.2 | 1.40 | 1.17 | 102.8 | 117.3 | 11.9 |
| short-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 3930.2 | 1.96 | 1.96 | 208.6 | - | - |
| short-65536-sliding-cp1 | tilelang@main | 3673.0 | 1.02 | 1.01 | 109.2 | 123.9 | 12.5 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main | 3673.0 | 1.05 | 1.05 | 255.6 | - | - |
| short-65536-sliding-cp1 | tilelang@main-repeat | 3673.0 | 1.02 | 1.01 | 109.5 | 124.0 | 12.5 |
| short-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 3673.0 | 1.05 | 1.05 | 255.5 | - | - |
| heavy-65536-csa-cp1 | tilelang@main | 12785.2 | 1.07 | 1.04 | 210.3 | 183.4 | 18.5 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main | 12785.2 | 1.50 | 1.50 | 414.1 | - | - |
| heavy-65536-csa-cp1 | tilelang@main-repeat | 12785.2 | 1.07 | 1.04 | 210.2 | 183.6 | 18.6 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 12785.2 | 1.50 | 1.50 | 413.7 | - | - |
| heavy-65536-hca-cp1 | tilelang@main | 5141.1 | 1.25 | 1.11 | 125.4 | 134.3 | 13.6 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main | 5141.1 | 2.25 | 2.25 | 258.5 | - | - |
| heavy-65536-hca-cp1 | tilelang@main-repeat | 5141.1 | 1.25 | 1.11 | 125.0 | 134.5 | 13.6 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 5141.1 | 2.25 | 2.25 | 258.5 | - | - |
| heavy-65536-sliding-cp1 | tilelang@main | 3542.5 | 1.04 | 1.02 | 105.3 | 121.0 | 12.2 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main | 3542.5 | 1.09 | 1.09 | 246.2 | - | - |
| heavy-65536-sliding-cp1 | tilelang@main-repeat | 3542.5 | 1.04 | 1.02 | 105.7 | 121.2 | 12.2 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 3542.5 | 1.09 | 1.09 | 246.2 | - | - |
| tiny-65536-csa-cp1 | tilelang@main | 1621.2 | 3.45 | 2.86 | 42.6 | 51.0 | 5.2 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main | 1621.2 | 11.87 | 11.87 | 81.8 | - | - |
| tiny-65536-csa-cp1 | tilelang@main-repeat | 1621.2 | 3.45 | 2.86 | 42.6 | 51.0 | 5.2 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref@main-repeat | 1621.2 | 11.87 | 11.87 | 81.7 | - | - |
| tiny-65536-hca-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.9 | 59.3 | 6.0 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.4 | - | - |
| tiny-65536-hca-cp1 | tilelang@main-repeat | 1306.0 | 1.85 | 1.39 | 43.2 | 59.5 | 6.0 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref@main-repeat | 1306.0 | 2.95 | 2.95 | 89.7 | - | - |
| tiny-65536-sliding-cp1 | tilelang@main | 1306.0 | 1.85 | 1.39 | 42.8 | 59.3 | 6.0 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main | 1306.0 | 2.95 | 2.95 | 89.3 | - | - |
| tiny-65536-sliding-cp1 | tilelang@main-repeat | 1306.0 | 1.85 | 1.39 | 42.9 | 59.4 | 6.0 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref@main-repeat | 1306.0 | 2.95 | 2.95 | 89.6 | - | - |

Correctness failures (excluded from timing): 0
