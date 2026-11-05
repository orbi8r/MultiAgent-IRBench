| tag | method | dev_online | dev_library | dev_literature | dev_mean | test_on_call | test_due_diligence | test_executive_assistants | test_help_desk | test_textbook_team | test_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f2_z2_ftpub | public trace ftpub over z2 K200, list builder | 0.8702 | 0.8085 | 0.8433 | 0.8407 | 0.8030 | 0.8315 | 0.8256 | 0.7189 | 0.7348 | 0.7827 |
| f2_ftpub_s1 | ftpub recipe retrained with seed 1 | 0.8628 | 0.7839 | 0.8433 | 0.8300 | 0.7939 | 0.8285 | 0.8215 | 0.7169 | 0.7496 | 0.7821 |
| f2_ftpub_s2 | ftpub recipe retrained with seed 2 | 0.8577 | 0.7879 | 0.8333 | 0.8263 | 0.7961 | 0.8219 | 0.8070 | 0.7118 | 0.7489 | 0.7771 |
| f2_ftpub_s3 | ftpub recipe retrained with seed 3 | 0.8643 | 0.7942 | 0.8656 | 0.8413 | 0.7930 | 0.8115 | 0.8126 | 0.7050 | 0.7219 | 0.7688 |
| f2_ftpub4b2x_exp_fuse0_k200_m25 | 4B ftpub4b2x (two batches of public rows) over fuse0 top 200 plus the top 25 of each expansion list | 0.8840 | 0.8430 | 0.9056 | 0.8775 | 0.8222 | 0.8548 | 0.8359 | 0.7660 | 0.7863 | 0.8131 |
| f2_ftpub4b2x_fuse0_k200 | 4B ftpub4b2x (two batches of public rows) over fuse0 top 200 | 0.8874 | 0.8217 | 0.9056 | 0.8716 | 0.8230 | 0.8363 | 0.8470 | 0.7405 | 0.7344 | 0.7962 |
| f2_ftpub4b_exp_fuse0_k200_m25 | 4B ftpub4b over fuse0 top 200 plus the top 25 of each expansion list | 0.8791 | 0.8131 | 0.8844 | 0.8589 | 0.8117 | 0.8315 | 0.8367 | 0.7686 | 0.7602 | 0.8017 |
| f2_ftpub4b_fuse0_k200 | 4B ftpub4b over fuse0 top 200 | 0.8768 | 0.8131 | 0.8889 | 0.8596 | 0.8091 | 0.8374 | 0.8322 | 0.7409 | 0.7283 | 0.7896 |
| f3_probe_P1 | ftpub (trained flat) over z2 K200 with the P1 structure prefix on the candidate text, list builder | 0.8806 | 0.7993 | 0.8444 | 0.8414 |  |  |  |  |  |  |
| f3_probe_P2 | ftpub (trained flat) over z2 K200 with the P2 structure prefix on the candidate text, list builder | 0.8704 | 0.7853 | 0.8433 | 0.8330 |  |  |  |  |  |  |
| f3_probe_P3 | ftpub (trained flat) over z2 K200 with the P3 structure prefix on the candidate text, list builder | 0.8772 | 0.7955 | 0.8433 | 0.8387 |  |  |  |  |  |  |
| f3_sp06_P1_s1 | ftpub recipe trained on P1 prefixed public rows over z2 K200, list builder, seed 1 | 0.8779 | 0.7973 | 0.8544 | 0.8432 |  |  |  |  |  |  |
| f3_sp06_P2_s1 | ftpub recipe trained on P2 prefixed public rows over z2 K200, list builder, seed 1 | 0.8749 | 0.8026 | 0.8633 | 0.8469 | 0.7830 | 0.8007 | 0.7870 | 0.7012 | 0.7178 | 0.7579 |
| f3_sp06_P2_s2 | ftpub recipe trained on P2 prefixed public rows over z2 K200, list builder, seed 2 | 0.8772 | 0.7914 | 0.8367 | 0.8351 | 0.7848 | 0.8041 | 0.7974 | 0.7144 | 0.7541 | 0.7710 |
| f3_sp06_P2_s3 | ftpub recipe trained on P2 prefixed public rows over z2 K200, list builder, seed 3 | 0.8553 | 0.7957 | 0.8433 | 0.8314 | 0.7983 | 0.8311 | 0.8111 | 0.6926 | 0.7041 | 0.7674 |
| f3_sp4b_P2 | Qwen/Qwen3-Reranker-4B trained on P2 prefixed public rows, fuse0 top 200; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0 | 0.8844 | 0.8121 | 0.9000 | 0.8655 | 0.8156 | 0.8170 | 0.8411 | 0.7290 | 0.7215 | 0.7848 |
| f3_sp4b_P2_exp | Qwen/Qwen3-Reranker-4B trained on P2 prefixed public rows, fuse0 top 200 plus the top 25 of each expansion list; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0 | 0.8787 | 0.8168 | 0.9000 | 0.8652 | 0.8152 | 0.8359 | 0.8456 | 0.7579 | 0.7737 | 0.8057 |
| t3_ftpub4b2x | Qwen/Qwen3-Reranker-4B adapter ftpub4b2x, fuse0 top 200 plus the top 25 of each expansion list, scoring only |  |  |  |  |  |  |  |  |  |  |
| f3_sect06_c_as | sectional rerank method C, node type as only, config chosen leave one run out | 0.8658 | 0.8061 | 0.8478 | 0.8399 |  |  |  |  |  |  |
| f3_sect06_c_best | sectional rerank method C, config ('as', 'soft', 0.5), frozen from the dev runs | 0.8658 | 0.8061 | 0.8478 | 0.8399 | 0.7985 | 0.8341 | 0.8233 | 0.7278 | 0.7381 | 0.7844 |
| f3_sect06_c_event | sectional rerank method C, node type event only, config chosen leave one run out | 0.8702 | 0.7981 | 0.8433 | 0.8372 |  |  |  |  |  |  |
| f3_sect06_c_file | sectional rerank method C, node type file only, config chosen leave one run out | 0.8702 | 0.8038 | 0.8433 | 0.8391 |  |  |  |  |  |  |
| f3_sect06_c_sec | sectional rerank method C, node type sec only, config chosen leave one run out | 0.8702 | 0.8085 | 0.8433 | 0.8407 |  |  |  |  |  |  |
| f3_sect06_d | sectional rerank method D, lightgbm {'n_estimators': 100, 'num_leaves': 7}, frozen from the dev runs | 0.8394 | 0.7664 | 0.8211 | 0.8089 | 0.8143 | 0.8019 | 0.8015 | 0.7104 | 0.7239 | 0.7704 |
| f3_sect4b_c_best | sectional rerank method C, config ('flat', '-', 0.0), frozen from the dev runs | 0.8889 | 0.8147 | 0.8811 | 0.8616 | 0.8207 | 0.8289 | 0.8348 | 0.7349 | 0.7344 | 0.7908 |
| f3_sect4b_c_best_exp | sectional rerank method C, config ('as', 'hard', 2.0), frozen from the dev runs | 0.8855 | 0.8513 | 0.8878 | 0.8749 | 0.8156 | 0.8585 | 0.8159 | 0.7549 | 0.7907 | 0.8071 |
| f3_sect4b_d | sectional rerank method D, lightgbm {'n_estimators': 100, 'num_leaves': 7}, frozen from the dev runs | 0.8689 | 0.8127 | 0.8783 | 0.8533 | 0.8170 | 0.8200 | 0.8237 | 0.7342 | 0.7293 | 0.7848 |
| f3_sect4b_d_exp | sectional rerank method D, lightgbm {'n_estimators': 100, 'num_leaves': 7}, frozen from the dev runs | 0.8794 | 0.8143 | 0.8811 | 0.8583 | 0.8463 | 0.8511 | 0.8326 | 0.7390 | 0.7830 | 0.8104 |
