| tag | method | dev_online | dev_library | dev_literature | dev_mean | test_on_call | test_due_diligence | test_executive_assistants | test_help_desk | test_textbook_team | test_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25_ctx_alone | contextual BM25 alone, no list builder | 0.5880 | 0.6310 | 0.5120 | 0.5770 | 0.4707 | 0.5419 | 0.5874 | 0.4553 | 0.4622 | 0.5035 |
| example_bm25 | repo baseline example_bm25.py | 0.5010 | 0.4880 | 0.5090 | 0.4990 | 0.3606 | 0.4359 | 0.4181 | 0.4287 | 0.4074 | 0.4101 |
| bgem3_nolb | bge reranker v2 m3 zero shot, no list builder | 0.6070 | 0.7200 |  |  | 0.4719 | 0.6152 | 0.6244 | 0.5763 | 0.5409 | 0.5657 |
| c4b | Qwen3 Reranker 4B on the 0.6B top 50 (cascade) |  |  |  | 0.8060 | 0.7754 | 0.8074 | 0.7422 | 0.7046 | 0.7309 | 0.7521 |
| ens_zs06_bgem3_w025 | z sum of Qwen 0.6B and bge v2 m3, weight 0.25 |  |  |  |  | 0.7617 | 0.8056 | 0.8033 | 0.6564 | 0.6683 | 0.7391 |
| ens_zs06_bgem3_w05 | z sum of Qwen 0.6B and bge v2 m3, weight 0.5 |  |  |  |  | 0.7543 | 0.8022 | 0.7922 | 0.6507 | 0.6617 | 0.7322 |
| ens_zs06_bgem3_w1 | z sum of Qwen 0.6B and bge v2 m3, weight 1 |  |  |  |  | 0.7457 | 0.8052 | 0.7978 | 0.6527 | 0.6554 | 0.7313 |
| lw4 | listwise Qwen3 4B Instruct 2507 over zero shot 0.6B top 10 | 0.8640 | 0.7980 | 0.7960 | 0.8190 |  |  |  |  |  |  |
| nr_c8b | Qwen3 Reranker 8B on the 0.6B top 50 | 0.8680 | 0.7960 | 0.8270 | 0.8250 |  |  |  |  |  |  |
| nr_lw8 | listwise Qwen3 8B over zero shot 0.6B top 10 | 0.8510 | 0.7940 | 0.8200 | 0.8220 |  |  |  |  |  |  |
| nr_lw8pub | listwise Qwen3 8B over the public 0.6B |  |  |  |  |  |  |  |  |  |  |
| nr_t2 | timed t2: lexical K 200, 2 GPUs | 0.8390 | 0.7430 | 0.7860 | 0.7900 |  |  |  |  |  |  |
| t1 | timed t1: lexical K 100, public reader, 1 GPU (run_method.py) | 0.8410 | 0.7350 | 0.7820 | 0.7820 | 0.7398 | 0.7441 | 0.7511 | 0.6533 | 0.6952 | 0.7167 |
| z2_first | z2 first stage list as is (no reader) |  |  |  |  | 0.4256 | 0.5013 | 0.5870 | 0.5076 | 0.4665 | 0.4976 |
| z3 | per qtype instructions (QINSTR=1) | 0.8230 | 0.7710 | 0.7740 | 0.7890 | 0.7722 | 0.7456 | 0.7422 | 0.6800 | 0.6707 | 0.7221 |
| zs06 | Qwen3 Reranker 0.6B zero shot | 0.8410 | 0.7720 | 0.7960 | 0.8030 | 0.7389 | 0.7522 | 0.7489 | 0.6889 | 0.6919 | 0.7241 |
| zs06_nolb | Qwen3 Reranker 0.6B zero shot, reader order as is (no list builder) | 0.6330 | 0.7490 |  |  | 0.4689 | 0.5496 | 0.6222 | 0.6156 | 0.5156 | 0.5544 |
| zs4b | Qwen3 Reranker 4B zero shot over the full z2 top 200 |  |  |  |  | 0.7724 | 0.8085 | 0.7448 | 0.7021 | 0.7491 | 0.7554 |
| ens_ftpub4b2x_ftpub2x_w05 | ensemble z(ftpub4b2x) + 0.5 z(ftpub2x) | 0.8760 | 0.8180 | 0.8770 | 0.8570 | 0.8096 | 0.8333 | 0.8315 | 0.7244 | 0.7622 | 0.7922 |
| ens_ftpub4b2x_ftpub_w05 | ensemble z(ftpub4b2x) + 0.5 z(ftpub) | 0.8850 | 0.8320 | 0.8920 | 0.8700 | 0.7954 | 0.8326 | 0.8148 | 0.7370 | 0.7556 | 0.7871 |
| ens_ftpub4b_ftpub_w025 | ensemble z(ftpub4b) + w z(ftpub), w=0.25 | 0.8810 | 0.8100 | 0.8970 | 0.8630 | 0.7759 | 0.8307 | 0.8278 | 0.7496 | 0.7587 | 0.7886 |
| ens_ftpub4b_ftpub_w05 | ensemble z(ftpub4b) + w z(ftpub), w=0.5 | 0.8860 | 0.8070 | 0.8940 | 0.8620 | 0.7663 | 0.8319 | 0.8300 | 0.7419 | 0.7617 | 0.7863 |
| ens_ftpub4b_ftpub_w1 | ensemble z(ftpub4b) + w z(ftpub), w=1 | 0.8820 | 0.8040 | 0.8880 | 0.8580 | 0.7646 | 0.8281 | 0.8300 | 0.7352 | 0.7565 | 0.7829 |
| ens_zs4b_zs06_w05 | ensemble z(zero shot 4B) + 0.5 z(zero shot 0.6B) |  |  |  |  | 0.7878 | 0.8152 | 0.7556 | 0.7118 | 0.7619 | 0.7664 |
| ftpub | ftpub: 0.6B LoRA on public SWE agent traces (batch 1) | 0.8700 | 0.8090 | 0.8430 | 0.8410 | 0.8030 | 0.8315 | 0.8256 | 0.7189 | 0.7348 | 0.7827 |
| ftpub2x | ftpub2x: 0.6B on both public batches (10199 rows) | 0.8840 | 0.7910 | 0.8230 | 0.8330 | 0.8378 | 0.8333 | 0.8337 | 0.7066 | 0.7648 | 0.7953 |
| ftpub4b | ftpub4b: 4B LoRA on the same public rows | 0.8610 | 0.8130 | 0.8870 | 0.8540 | 0.7693 | 0.8293 | 0.8033 | 0.7520 | 0.7531 | 0.7814 |
| ftpub4b2x | ftpub4b2x: 4B on both public batches | 0.8890 | 0.8260 | 0.9060 | 0.8740 | 0.7893 | 0.8367 | 0.8159 | 0.7394 | 0.7481 | 0.7859 |
| ft4b_lib | ft4b fold adapter trained on the two dev runs other than library |  | 0.7910 |  | 0.7910 | 0.8078 | 0.8385 | 0.8456 | 0.7385 | 0.7526 | 0.7966 |
| ft4b_lit | ft4b fold adapter trained on the two dev runs other than literature |  |  | 0.9110 | 0.9110 | 0.7531 | 0.8315 | 0.8122 | 0.7197 | 0.7585 | 0.7750 |
| ft4b_on | ft4b fold adapter trained on the two dev runs other than online | 0.8510 |  |  | 0.8510 | 0.7811 | 0.8363 | 0.8226 | 0.7305 | 0.7537 | 0.7848 |
| ftreal_lib | ftreal_lib: 0.6B on 2 dev runs' answer keys, 136 rows |  | 0.6890 |  | 0.6890 | 0.6491 | 0.7407 | 0.6778 | 0.5996 | 0.6600 | 0.6654 |
| ftsyn_lib | ftsyn fold adapter trained on the two dev runs other than library |  | 0.7780 |  | 0.7780 | 0.7996 | 0.8085 | 0.7693 | 0.7009 | 0.7393 | 0.7635 |
| ftsyn_lit | ftsyn fold adapter trained on the two dev runs other than literature |  |  | 0.8390 | 0.8390 | 0.7733 | 0.8141 | 0.8067 | 0.6934 | 0.7244 | 0.7624 |
| ftsyn_on | ftsyn fold adapter trained on the two dev runs other than online | 0.8450 |  |  | 0.8450 | 0.7728 | 0.7852 | 0.8211 | 0.7217 | 0.7289 | 0.7659 |
| fttrans | fttrans: 0.6B on synthetic rows of all 3 dev runs | 0.7970 | 0.7860 | 0.8520 | 0.8120 | 0.8020 | 0.8256 | 0.7722 | 0.6770 | 0.7254 | 0.7604 |
| ftwho_lib | ftwho_lib: 0.6B on other runs' who questions |  |  |  |  | 0.7193 | 0.7593 | 0.8059 | 0.6435 | 0.7115 | 0.7279 |
| nr_ltr | feat.py + ltr.py LightGBM lambdarank, PROBE |  |  |  | 0.7320 |  |  |  |  |  |  |
| ce_bgeb | student bge reranker base hard 256 as reader over the z2 list |  |  |  | 0.7430 | 0.7444 | 0.7067 | 0.7807 | 0.5899 | 0.6661 | 0.6976 |
| ce_bgeb512 | student bge reranker base hard 512 as reader over the z2 list |  |  |  | 0.7630 | 0.7737 | 0.7315 | 0.8026 | 0.6366 | 0.6696 | 0.7228 |
| ce_l12 | student MiniLM L12 hard 256 as reader over the z2 list |  |  |  | 0.7350 | 0.7067 | 0.7193 | 0.7974 | 0.5983 | 0.6524 | 0.6948 |
| ce_l12_512 | student MiniLM L12 hard 512 as reader over the z2 list |  |  |  | 0.7490 | 0.7696 | 0.7289 | 0.7981 | 0.6422 | 0.6526 | 0.7183 |
| ce_l6 | student MiniLM L6 hard 256 as reader over the z2 list |  |  |  | 0.7240 | 0.6907 | 0.6779 | 0.7785 | 0.5524 | 0.6235 | 0.6646 |
| ds_bgeb | student ds_bgeb: bge base, KL to the 4B + 0.5 0.6B ensemble, 512, 2 epochs as reader over the z2 list |  |  |  | 0.7720 | 0.7750 | 0.7459 | 0.8037 | 0.6516 | 0.7093 | 0.7371 |
| nr_ds_gtemb | ds_gtemb: gte reranker modernbert base |  |  |  |  |  |  |  |  |  |  |
| nr_ds_m3 | ds_m3: bge reranker v2 m3 student |  |  |  |  |  |  |  |  |  |  |
| cas_rstul12_k100_n10 | MiniLM L12 student on fuse0 top 100, ftpub reader on the student's top 10 |  |  |  | 0.8030 | 0.8022 | 0.7304 | 0.8293 | 0.6083 | 0.6828 | 0.7306 |
| cas_rstul12_k100_n20 | MiniLM L12 student on fuse0 top 100, ftpub reader on the student's top 20 |  |  |  | 0.8170 | 0.8185 | 0.7659 | 0.8148 | 0.6480 | 0.6867 | 0.7468 |
| cas_rstul12_k100_n30 | MiniLM L12 student on fuse0 top 100, ftpub reader on the student's top 30 |  |  |  | 0.8200 | 0.8193 | 0.7652 | 0.8270 | 0.6524 | 0.6915 | 0.7511 |
| cas_rstul12_k100_n5 | MiniLM L12 student on fuse0 top 100, ftpub reader on the student's top 5 |  |  |  | 0.8010 | 0.7889 | 0.7006 | 0.8237 | 0.5846 | 0.6694 | 0.7134 |
| cas_rstul6_k50_n20 | MiniLM L6 student on fuse0 top 50, ftpub reader on the student's top 20 |  |  |  | 0.7820 | 0.8059 | 0.7715 | 0.7937 | 0.6067 | 0.6398 | 0.7235 |
| cas_rstul6_k50_n30 | MiniLM L6 student on fuse0 top 50, ftpub reader on the student's top 30 |  |  |  | 0.8140 | 0.8074 | 0.7681 | 0.7948 | 0.6044 | 0.6665 | 0.7283 |
| rstul12_fuse0_k100 | MiniLM L12 student alone over fuse0 top 100 |  |  |  | 0.7760 | 0.7739 | 0.6902 | 0.8159 | 0.5746 | 0.6341 | 0.6977 |
| rstul12_fuse0_k50 | MiniLM L12 student alone over fuse0 top 50 |  |  |  | 0.7610 | 0.7600 | 0.6699 | 0.7926 | 0.5348 | 0.6311 | 0.6777 |
| rstul6_fuse0_k100 | MiniLM L6 student alone over fuse0 top 100 |  |  |  | 0.7320 | 0.7613 | 0.6807 | 0.7637 | 0.5885 | 0.6263 | 0.6841 |
| rstul6_fuse0_k50 | MiniLM L6 student alone over fuse0 top 50 |  |  |  | 0.7260 | 0.7606 | 0.6737 | 0.7707 | 0.5419 | 0.6191 | 0.6732 |
| fs_base_any10 | BM25 ctx + plain RRF any@10, non location atoms |  |  |  | 0.6260 |  |  |  |  |  | 0.5990 |
| fs_base_any100 | BM25 ctx + plain RRF any@100, non location atoms |  |  |  | 0.9360 |  |  |  |  |  | 0.8660 |
| fs_base_any200 | BM25 ctx + plain RRF any@200, non location atoms |  |  |  | 0.9590 |  |  |  |  |  | 0.9170 |
| fs_base_any25 | BM25 ctx + plain RRF any@25, non location atoms |  |  |  | 0.8480 |  |  |  |  |  | 0.7170 |
| fs_base_any50 | BM25 ctx + plain RRF any@50, non location atoms |  |  |  | 0.9240 |  |  |  |  |  | 0.8150 |
| fs_nov_any10 | novelty pruned BM25 ctx + plain RRF any@10, non location atoms |  |  |  | 0.7890 |  |  |  |  |  | 0.7070 |
| fs_nov_any100 | novelty pruned BM25 ctx + plain RRF any@100, non location atoms |  |  |  | 0.9590 |  |  |  |  |  | 0.9460 |
| fs_nov_any200 | novelty pruned BM25 ctx + plain RRF any@200, non location atoms |  |  |  | 0.9590 |  |  |  |  |  | 0.9620 |
| fs_nov_any25 | novelty pruned BM25 ctx + plain RRF any@25, non location atoms |  |  |  | 0.8830 |  |  |  |  |  | 0.8030 |
| fs_nov_any50 | novelty pruned BM25 ctx + plain RRF any@50, non location atoms |  |  |  | 0.9530 |  |  |  |  |  | 0.8790 |
| rec200_arch2_rrf | arch2_rrf recall@200 (key free, lexical seeds) | 0.9280 | 0.9120 | 0.8810 | 0.9070 | 0.9090 | 0.8940 | 0.9290 | 0.9110 | 0.8120 | 0.8900 |
| rec200_asker | asker recall@200 (key free, lexical seeds) | 0.7970 | 0.6620 | 0.7460 | 0.7310 | 0.8310 | 0.8710 | 0.8290 | 0.8230 | 0.7870 | 0.8290 |
| rec200_authshift | authshift recall@200 (key free, lexical seeds) | 0.8550 | 0.7870 | 0.8210 | 0.8190 | 0.6620 | 0.8940 | 0.8000 | 0.8100 | 0.8250 | 0.8010 |
| rec200_base | base recall@200 (key free, lexical seeds) | 0.8700 | 0.9250 | 0.8210 | 0.8750 | 0.7790 | 0.8710 | 0.8570 | 0.8860 | 0.8000 | 0.8390 |
| rec200_bm_ctx | bm_ctx recall@200 (key free, lexical seeds) | 0.8700 | 0.9000 | 0.8210 | 0.8660 | 0.8440 | 0.9060 | 0.8860 | 0.8730 | 0.8000 | 0.8620 |
| rec200_bm_nov | bm_nov recall@200 (key free, lexical seeds) | 0.8840 | 0.9250 | 0.8960 | 0.9030 | 0.8440 | 0.9290 | 0.8860 | 0.9110 | 0.8750 | 0.8900 |
| rec200_bm_param | bm_param recall@200 (key free, lexical seeds) | 0.8550 | 0.9250 | 0.8360 | 0.8750 | 0.8440 | 0.8940 | 0.8430 | 0.8860 | 0.8000 | 0.8540 |
| rec200_bm_plain | bm_plain recall@200 (key free, lexical seeds) | 0.8120 | 0.8870 | 0.8360 | 0.8470 | 0.7010 | 0.8120 | 0.7570 | 0.8480 | 0.8000 | 0.7850 |
| rec200_bm_raw | bm_raw recall@200 (key free, lexical seeds) | 0.7390 | 0.7500 | 0.8060 | 0.7640 | 0.6750 | 0.7410 | 0.7290 | 0.7470 | 0.7000 | 0.7190 |
| rec200_filegraph | filegraph recall@200 (key free, lexical seeds) | 0.5800 | 0.3620 | 0.4180 | 0.4490 | 0.5320 | 0.4820 | 0.4860 | 0.5320 | 0.5500 | 0.5170 |
| rec200_firstwrite | firstwrite recall@200 (key free, lexical seeds) | 0.6090 | 0.5120 | 0.6420 | 0.5830 | 0.4550 | 0.6350 | 0.5570 | 0.5190 | 0.4500 | 0.5240 |
| rec200_fuse0 | fuse0 recall@200 (key free, lexical seeds) | 0.8700 | 0.9250 | 0.8360 | 0.8800 | 0.8570 | 0.9180 | 0.8860 | 0.9110 | 0.8120 | 0.8770 |
| rec200_idlink | idlink recall@200 (key free, lexical seeds) | 0.5650 | 0.6000 | 0.5820 | 0.5600 | 0.3900 | 0.6000 | 0.3860 | 0.5950 | 0.3250 | 0.4630 |
| rec200_nbr | nbr recall@200 (key free, lexical seeds) | 0.7830 | 0.7500 | 0.7760 | 0.7690 | 0.7400 | 0.8470 | 0.8430 | 0.7340 | 0.8380 | 0.8010 |
| rec200_qdecomp | qdecomp recall@200 (key free, lexical seeds) | 0.8700 | 0.9500 | 0.8210 | 0.8840 | 0.8050 | 0.8710 | 0.8570 | 0.8990 | 0.8380 | 0.8540 |
| rec200_qtype | qtype recall@200 (key free, lexical seeds) | 0.8990 | 0.9250 | 0.8510 | 0.8940 | 0.8310 | 0.8940 | 0.8710 | 0.8860 | 0.7870 | 0.8540 |
| rec200_rm3 | rm3 recall@200 (key free, lexical seeds) | 0.8700 | 0.9120 | 0.8510 | 0.8800 | 0.8180 | 0.8710 | 0.8860 | 0.8990 | 0.7750 | 0.8490 |
| fam_arch2_rrf_ftpub | ftpub over the top 200 of family arch2_rrf |  |  |  |  | 0.8219 | 0.8252 | 0.8478 | 0.7156 | 0.7222 | 0.7865 |
| fam_asker_ftpub | ftpub over the top 200 of family asker |  |  |  |  | 0.7689 | 0.7111 | 0.7215 | 0.6574 | 0.6893 | 0.7096 |
| fam_authshift_ftpub | ftpub over the top 200 of family authshift |  |  |  |  | 0.6578 | 0.7965 | 0.6770 | 0.6422 | 0.6863 | 0.6920 |
| fam_base_ftpub | ftpub over the top 200 of family base |  |  |  |  | 0.8096 | 0.8133 | 0.8322 | 0.6989 | 0.7415 | 0.7791 |
| fam_bm_ctx_ftpub | ftpub over the top 200 of family bm_ctx |  |  |  |  | 0.8207 | 0.8289 | 0.8322 | 0.6911 | 0.7315 | 0.7809 |
| fam_bm_nov_ftpub | ftpub over the top 200 of family bm_nov |  |  |  |  | 0.7552 | 0.8059 | 0.8470 | 0.7056 | 0.7196 | 0.7667 |
| fam_bm_param_ftpub | ftpub over the top 200 of family bm_param |  |  |  |  | 0.8241 | 0.8230 | 0.8344 | 0.6989 | 0.7237 | 0.7808 |
| fam_bm_plain_ftpub | ftpub over the top 200 of family bm_plain |  |  |  |  | 0.7374 | 0.7519 | 0.7422 | 0.6822 | 0.7396 | 0.7307 |
| fam_bm_raw_ftpub | ftpub over the top 200 of family bm_raw |  |  |  |  | 0.7130 | 0.7604 | 0.7389 | 0.6256 | 0.7067 | 0.7089 |
| fam_filegraph_ftpub | ftpub over the top 200 of family filegraph |  |  |  |  | 0.5422 | 0.5402 | 0.4859 | 0.3593 | 0.5361 | 0.4928 |
| fam_firstwrite_ftpub | ftpub over the top 200 of family firstwrite |  |  |  |  | 0.6017 | 0.6033 | 0.5878 | 0.4704 | 0.5352 | 0.5597 |
| fam_idlink_ftpub | ftpub over the top 200 of family idlink |  |  |  |  | 0.4717 | 0.4719 | 0.4133 | 0.4881 | 0.3748 | 0.4440 |
| fam_nbr_ftpub | ftpub over the top 200 of family nbr |  |  |  |  | 0.7011 | 0.7348 | 0.7507 | 0.5919 | 0.7443 | 0.7046 |
| fam_pass1_rrf_ftpub | ftpub over the top 200 of family pass1_rrf |  |  |  |  | 0.8322 | 0.8289 | 0.8567 | 0.7156 | 0.7370 | 0.7941 |
| fam_qdecomp_ftpub | ftpub over the top 200 of family qdecomp |  |  |  |  | 0.8089 | 0.8111 | 0.8344 | 0.6944 | 0.7470 | 0.7792 |
| fam_qtype_ftpub | ftpub over the top 200 of family qtype |  |  |  |  | 0.8141 | 0.8267 | 0.8500 | 0.6922 | 0.7359 | 0.7838 |
| fam_rm3_ftpub | ftpub over the top 200 of family rm3 |  |  |  |  | 0.8163 | 0.8156 | 0.8522 | 0.7111 | 0.7156 | 0.7821 |
| time_d7_cpu_total | CPU first stage per run (tokenise, novelty, indexes, pass 1 lists, two expansion rounds, fusion) |  |  |  |  | 13.9000 | 12.4000 | 11.3000 | 14.7000 | 15.1000 | 13.5000 |
| time_fuse0 | fuse0 first stage only (index plus queries, pure Python) |  |  |  |  | 10.9000 | 10.0000 | 9.1000 | 11.4000 | 11.8000 | 10.6000 |
| ftpub_base_k20 | ftpub scores the top 20 of base, no expansion |  |  |  | 0.7160 | 0.6683 | 0.6663 | 0.6607 | 0.4756 | 0.6156 | 0.6173 |
| ftpub_exit_tau_m0p01 | ftpub early exit: top 20 of fuse0, ranks 20 to 100 too if best log p(yes) is below -0.01 |  |  |  |  | 0.7852 | 0.7993 | 0.7689 | 0.6122 | 0.7007 | 0.7333 |
| ftpub_exit_tau_m0p05 | ftpub early exit: top 20 of fuse0, ranks 20 to 100 too if best log p(yes) is below -0.05 |  |  |  | 0.8400 | 0.7785 | 0.7807 | 0.7689 | 0.5856 | 0.6985 | 0.7224 |
| ftpub_exit_tau_m0p2 | ftpub early exit: top 20 of fuse0, ranks 20 to 100 too if best log p(yes) is below -0.2 |  |  |  | 0.8330 | 0.7652 | 0.7807 | 0.7704 | 0.5744 | 0.6985 | 0.7179 |
| ftpub_exit_tau_m0p5 | ftpub early exit: top 20 of fuse0, ranks 20 to 100 too if best log p(yes) is below -0.5 |  |  |  | 0.8250 | 0.7674 | 0.7807 | 0.7604 | 0.5767 | 0.6585 | 0.7087 |
| ftpub_exit_tau_m1p0 | ftpub early exit: top 20 of fuse0, ranks 20 to 100 too if best log p(yes) is below -1.0 |  |  |  | 0.8180 | 0.7519 | 0.7752 | 0.7493 | 0.5767 | 0.6585 | 0.7023 |
| ftpub_exit_tau_m2p0 | ftpub early exit: top 20 of fuse0, ranks 20 to 100 too if best log p(yes) is below -2.0 |  |  |  | 0.8000 | 0.7407 | 0.7752 | 0.7493 | 0.5700 | 0.6585 | 0.6987 |
| ftpub_exp_base_k20_m10 | ftpub over base top 20 plus the top 10 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.7478 | 0.7374 | 0.7341 | 0.5789 | 0.6963 | 0.6989 |
| ftpub_exp_base_k20_m25 | ftpub over base top 20 plus the top 25 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.7926 | 0.7619 | 0.7978 | 0.6067 | 0.7289 | 0.7376 |
| ftpub_exp_base_k20_m5 | ftpub over base top 20 plus the top 5 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.7311 | 0.7130 | 0.6841 | 0.5733 | 0.6585 | 0.6720 |
| ftpub_exp_fuse0_k100_m10 | ftpub over fuse0 top 100 plus the top 10 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8237 | 0.8037 | 0.8456 | 0.7011 | 0.7237 | 0.7796 |
| ftpub_exp_fuse0_k100_m25 | ftpub over fuse0 top 100 plus the top 25 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8356 | 0.8111 | 0.8344 | 0.7178 | 0.7504 | 0.7899 |
| ftpub_exp_fuse0_k100_m5 | ftpub over fuse0 top 100 plus the top 5 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8230 | 0.8037 | 0.8322 | 0.6878 | 0.7126 | 0.7719 |
| ftpub_exp_fuse0_k200_m10 | ftpub over fuse0 top 200 plus the top 10 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8352 | 0.8285 | 0.8411 | 0.7144 | 0.7504 | 0.7939 |
| ftpub_exp_fuse0_k200_m25 | ftpub over fuse0 top 200 plus the top 25 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8389 | 0.8322 | 0.8478 | 0.7122 | 0.7756 | 0.8013 |
| ftpub_exp_fuse0_k200_m5 | ftpub over fuse0 top 200 plus the top 5 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8330 | 0.8285 | 0.8567 | 0.7144 | 0.7370 | 0.7939 |
| ftpub_exp_fuse0_k20_m10 | ftpub over fuse0 top 20 plus the top 10 of each expansion list seeded by the reader's top 5 |  |  |  | 0.7860 | 0.7567 | 0.7685 | 0.7593 | 0.6267 | 0.7015 | 0.7225 |
| ftpub_exp_fuse0_k20_m25 | ftpub over fuse0 top 20 plus the top 25 of each expansion list seeded by the reader's top 5 |  |  |  | 0.8140 | 0.8111 | 0.7770 | 0.7870 | 0.6644 | 0.7406 | 0.7560 |
| ftpub_exp_fuse0_k20_m5 | ftpub over fuse0 top 20 plus the top 5 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.7211 | 0.7019 | 0.7370 | 0.6289 | 0.6681 | 0.6914 |
| ftpub_exp_fuse0_k30_m10 | ftpub over fuse0 top 30 plus the top 10 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.7856 | 0.7641 | 0.7963 | 0.6533 | 0.6904 | 0.7379 |
| ftpub_exp_fuse0_k30_m25 | ftpub over fuse0 top 30 plus the top 25 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8081 | 0.7741 | 0.8189 | 0.6744 | 0.7185 | 0.7588 |
| ftpub_exp_fuse0_k30_m5 | ftpub over fuse0 top 30 plus the top 5 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.7833 | 0.7396 | 0.7807 | 0.6589 | 0.6748 | 0.7275 |
| ftpub_exp_fuse0_k50_m10 | ftpub over fuse0 top 50 plus the top 10 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8074 | 0.7841 | 0.8200 | 0.6556 | 0.7087 | 0.7551 |
| ftpub_exp_fuse0_k50_m25 | ftpub over fuse0 top 50 plus the top 25 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8193 | 0.7985 | 0.8522 | 0.6756 | 0.7400 | 0.7771 |
| ftpub_exp_fuse0_k50_m5 | ftpub over fuse0 top 50 plus the top 5 of each expansion list seeded by the reader's top 5 |  |  |  |  | 0.8096 | 0.7730 | 0.8200 | 0.6356 | 0.7087 | 0.7494 |
| ftpub_fuse0_k100 | ftpub scores the top 100 of fuse0, no expansion |  |  |  | 0.8460 | 0.8141 | 0.8015 | 0.8300 | 0.6611 | 0.7093 | 0.7632 |
| ftpub_fuse0_k20 | ftpub scores the top 20 of fuse0, no expansion |  |  |  | 0.7490 | 0.7022 | 0.6796 | 0.7226 | 0.5411 | 0.6407 | 0.6573 |
| ftpub_fuse0_k200 | ftpub scores the top 200 of fuse0, no expansion |  |  |  | 0.8510 | 0.8322 | 0.8285 | 0.8567 | 0.7156 | 0.7370 | 0.7940 |
| ftpub_fuse0_k30 | ftpub scores the top 30 of fuse0, no expansion |  |  |  | 0.7920 | 0.7433 | 0.7419 | 0.7741 | 0.6044 | 0.6615 | 0.7050 |
| ftpub_fuse0_k50 | ftpub scores the top 50 of fuse0, no expansion |  |  |  | 0.8210 | 0.8126 | 0.7752 | 0.8178 | 0.6033 | 0.6998 | 0.7417 |
| d2q_none | BM25 ctx reader with list builder, no expansion | 0.7620 | 0.6750 | 0.7510 | 0.7300 | 0.7502 | 0.7181 | 0.7052 | 0.5524 | 0.6026 | 0.6657 |
| d2q_qwen_concat | BM25 ctx reader with list builder, doc2query qwen expansion, concat | 0.7710 | 0.6970 | 0.7490 | 0.7390 | 0.7448 | 0.7319 | 0.7207 | 0.5493 | 0.6067 | 0.6707 |
| d2q_qwen_field0p5 | BM25 ctx reader with list builder, doc2query qwen expansion, field0p5 |  |  |  | 0.7230 | 0.7176 | 0.7345 | 0.7048 | 0.5580 | 0.6259 | 0.6682 |
| d2q_qwen_field1p0 | BM25 ctx reader with list builder, doc2query qwen expansion, field1p0 |  |  |  | 0.6960 | 0.7093 | 0.7227 | 0.6581 | 0.5556 | 0.5970 | 0.6485 |
| d2q_t5_concat | BM25 ctx reader with list builder, doc2query t5 expansion, concat | 0.7940 | 0.6780 | 0.7610 | 0.7450 | 0.7391 | 0.7507 | 0.7474 | 0.5467 | 0.6652 | 0.6898 |
| d2q_t5_field0p5 | BM25 ctx reader with list builder, doc2query t5 expansion, field0p5 |  |  |  | 0.7140 | 0.7531 | 0.7185 | 0.7781 | 0.5559 | 0.6415 | 0.6894 |
| d2q_t5_field1p0 | BM25 ctx reader with list builder, doc2query t5 expansion, field1p0 |  |  |  | 0.6890 | 0.7094 | 0.6919 | 0.7496 | 0.5513 | 0.6133 | 0.6631 |
| d2q_t5plusqwen_concat | BM25 ctx reader with list builder, doc2query t5plusqwen expansion, concat |  |  |  |  | 0.7504 | 0.7511 | 0.7452 | 0.5463 | 0.6604 | 0.6907 |
| d2q_t5plusqwen_field0p5 | BM25 ctx reader with list builder, doc2query t5plusqwen expansion, field0p5 |  |  |  |  | 0.7452 | 0.7434 | 0.7541 | 0.5420 | 0.6422 | 0.6854 |
| d2q_t5plusqwen_field1p0 | BM25 ctx reader with list builder, doc2query t5plusqwen expansion, field1p0 |  |  |  |  | 0.6811 | 0.7323 | 0.7378 | 0.5439 | 0.6215 | 0.6633 |
| algread_bm25 | algread BM25 ctx reader + list builder | 0.7620 | 0.6750 | 0.7510 | 0.7290 | 0.7502 | 0.7181 | 0.7052 | 0.5524 | 0.6026 | 0.6657 |
| algread_hand | algread BM25 + hand features (proximity, answer span, brevity, outcome kind) | 0.7680 | 0.6890 | 0.7410 | 0.7330 | 0.7519 | 0.7133 | 0.7000 | 0.5614 | 0.5963 | 0.6646 |
| algread_p0_a0_b0_k0 | algread grid WP=0 WA=0 WB=0 WK=0 | 0.7620 | 0.6750 | 0.7510 | 0.7290 | 0.7502 | 0.7181 | 0.7052 | 0.5524 | 0.6026 | 0.6657 |
| algread_p0_a0_b0_k0d5 | algread grid WP=0 WA=0 WB=0 WK=0.5 | 0.7640 | 0.6770 | 0.7480 | 0.7300 | 0.7524 | 0.7226 | 0.7059 | 0.5546 | 0.6048 | 0.6681 |
| algread_p0_a0_b0d2_k0 | algread grid WP=0 WA=0 WB=0.2 WK=0 | 0.7670 | 0.6670 | 0.7410 | 0.7250 | 0.7513 | 0.7244 | 0.7004 | 0.5401 | 0.6137 | 0.6660 |
| algread_p0_a0_b0d2_k0d5 | algread grid WP=0 WA=0 WB=0.2 WK=0.5 | 0.7630 | 0.6730 | 0.7360 | 0.7240 | 0.7513 | 0.7296 | 0.6978 | 0.5398 | 0.6137 | 0.6664 |
| algread_p0_a0_b0d5_k0 | algread grid WP=0 WA=0 WB=0.5 WK=0 | 0.7400 | 0.6660 | 0.7210 | 0.7090 | 0.7231 | 0.7244 | 0.7181 | 0.5116 | 0.6081 | 0.6571 |
| algread_p0_a0_b0d5_k0d5 | algread grid WP=0 WA=0 WB=0.5 WK=0.5 | 0.7520 | 0.6730 | 0.7200 | 0.7150 | 0.7254 | 0.7478 | 0.7244 | 0.5194 | 0.6044 | 0.6643 |
| algread_p0_a0d5_b0_k0 | algread grid WP=0 WA=0.5 WB=0 WK=0 | 0.7640 | 0.6760 | 0.7480 | 0.7290 | 0.7457 | 0.7215 | 0.7207 | 0.5576 | 0.6081 | 0.6707 |
| algread_p0_a0d5_b0_k0d5 | algread grid WP=0 WA=0.5 WB=0 WK=0.5 | 0.7680 | 0.6780 | 0.7460 | 0.7310 | 0.7502 | 0.7270 | 0.7126 | 0.5576 | 0.6070 | 0.6709 |
| algread_p0_a0d5_b0d2_k0 | algread grid WP=0 WA=0.5 WB=0.2 WK=0 | 0.7690 | 0.6650 | 0.7390 | 0.7240 | 0.7491 | 0.7422 | 0.7248 | 0.5394 | 0.6059 | 0.6723 |
| algread_p0_a0d5_b0d2_k0d5 | algread grid WP=0 WA=0.5 WB=0.2 WK=0.5 | 0.7700 | 0.6760 | 0.7310 | 0.7260 | 0.7469 | 0.7496 | 0.7311 | 0.5390 | 0.6048 | 0.6743 |
| algread_p0_a0d5_b0d5_k0 | algread grid WP=0 WA=0.5 WB=0.5 WK=0 | 0.7460 | 0.6680 | 0.7340 | 0.7160 | 0.7409 | 0.7256 | 0.7315 | 0.5116 | 0.6037 | 0.6627 |
| algread_p0_a0d5_b0d5_k0d5 | algread grid WP=0 WA=0.5 WB=0.5 WK=0.5 | 0.7560 | 0.6780 | 0.7290 | 0.7210 | 0.7409 | 0.7456 | 0.7378 | 0.5194 | 0.5956 | 0.6678 |
| algread_p0_a1_b0_k0 | algread grid WP=0 WA=1 WB=0 WK=0 | 0.7640 | 0.6670 | 0.7420 | 0.7240 | 0.7324 | 0.7070 | 0.7252 | 0.5553 | 0.6104 | 0.6661 |
| algread_p0_a1_b0_k0d5 | algread grid WP=0 WA=1 WB=0 WK=0.5 | 0.7700 | 0.6780 | 0.7410 | 0.7300 | 0.7369 | 0.7259 | 0.7281 | 0.5605 | 0.6104 | 0.6724 |
| algread_p0_a1_b0d2_k0 | algread grid WP=0 WA=1 WB=0.2 WK=0 | 0.7710 | 0.6720 | 0.7370 | 0.7270 | 0.7335 | 0.7244 | 0.7270 | 0.5416 | 0.6059 | 0.6665 |
| algread_p0_a1_b0d2_k0d5 | algread grid WP=0 WA=1 WB=0.2 WK=0.5 | 0.7750 | 0.6810 | 0.7310 | 0.7290 | 0.7335 | 0.7307 | 0.7333 | 0.5413 | 0.6059 | 0.6690 |
| algread_p0_a1_b0d5_k0 | algread grid WP=0 WA=1 WB=0.5 WK=0 | 0.7480 | 0.6750 | 0.7290 | 0.7180 | 0.7409 | 0.7122 | 0.7515 | 0.5109 | 0.5959 | 0.6623 |
| algread_p0_a1_b0d5_k0d5 | algread grid WP=0 WA=1 WB=0.5 WK=0.5 | 0.7540 | 0.6780 | 0.7260 | 0.7190 | 0.7431 | 0.7478 | 0.7578 | 0.5187 | 0.5889 | 0.6713 |
| algread_p1_a0_b0_k0 | algread grid WP=1 WA=0 WB=0 WK=0 | 0.7610 | 0.6740 | 0.7420 | 0.7250 | 0.7563 | 0.7100 | 0.7056 | 0.5525 | 0.5985 | 0.6646 |
| algread_p1_a0_b0_k0d5 | algread grid WP=1 WA=0 WB=0 WK=0.5 | 0.7680 | 0.6890 | 0.7410 | 0.7330 | 0.7519 | 0.7133 | 0.7000 | 0.5614 | 0.5963 | 0.6646 |
| algread_p1_a0_b0d2_k0 | algread grid WP=1 WA=0 WB=0.2 WK=0 | 0.7660 | 0.6720 | 0.7340 | 0.7240 | 0.7511 | 0.7178 | 0.7126 | 0.5416 | 0.6141 | 0.6674 |
| algread_p1_a0_b0d2_k0d5 | algread grid WP=1 WA=0 WB=0.2 WK=0.5 | 0.7680 | 0.6780 | 0.7320 | 0.7260 | 0.7530 | 0.7230 | 0.7126 | 0.5372 | 0.6081 | 0.6668 |
| algread_p1_a0_b0d5_k0 | algread grid WP=1 WA=0 WB=0.5 WK=0 | 0.7460 | 0.6530 | 0.7290 | 0.7100 | 0.7380 | 0.7252 | 0.7159 | 0.5140 | 0.6059 | 0.6598 |
| algread_p1_a0_b0d5_k0d5 | algread grid WP=1 WA=0 WB=0.5 WK=0.5 | 0.7490 | 0.6670 | 0.7270 | 0.7140 | 0.7391 | 0.7441 | 0.7193 | 0.5207 | 0.6037 | 0.6654 |
| algread_p1_a0d5_b0_k0 | algread grid WP=1 WA=0.5 WB=0 WK=0 | 0.7590 | 0.6800 | 0.7370 | 0.7250 | 0.7341 | 0.7178 | 0.7300 | 0.5540 | 0.6007 | 0.6673 |
| algread_p1_a0d5_b0_k0d5 | algread grid WP=1 WA=0.5 WB=0 WK=0.5 | 0.7680 | 0.6900 | 0.7370 | 0.7320 | 0.7341 | 0.7344 | 0.7289 | 0.5629 | 0.5985 | 0.6718 |
| algread_p1_a0d5_b0d2_k0 | algread grid WP=1 WA=0.5 WB=0.2 WK=0 | 0.7680 | 0.6730 | 0.7300 | 0.7240 | 0.7444 | 0.7211 | 0.7304 | 0.5453 | 0.6141 | 0.6711 |
| algread_p1_a0d5_b0d2_k0d5 | algread grid WP=1 WA=0.5 WB=0.2 WK=0.5 | 0.7700 | 0.6830 | 0.7260 | 0.7260 | 0.7485 | 0.7252 | 0.7304 | 0.5409 | 0.6104 | 0.6711 |
| algread_p1_a0d5_b0d5_k0 | algread grid WP=1 WA=0.5 WB=0.5 WK=0 | 0.7480 | 0.6570 | 0.7340 | 0.7130 | 0.7380 | 0.7174 | 0.7293 | 0.5184 | 0.6059 | 0.6618 |
| algread_p1_a0d5_b0d5_k0d5 | algread grid WP=1 WA=0.5 WB=0.5 WK=0.5 | 0.7520 | 0.6680 | 0.7310 | 0.7170 | 0.7413 | 0.7496 | 0.7326 | 0.5251 | 0.6037 | 0.6705 |
| algread_p1_a1_b0_k0 | algread grid WP=1 WA=1 WB=0 WK=0 | 0.7610 | 0.6770 | 0.7390 | 0.7260 | 0.7363 | 0.6989 | 0.7322 | 0.5510 | 0.6019 | 0.6641 |
| algread_p1_a1_b0_k0d5 | algread grid WP=1 WA=1 WB=0 WK=0.5 | 0.7700 | 0.6860 | 0.7370 | 0.7310 | 0.7363 | 0.7156 | 0.7333 | 0.5621 | 0.6030 | 0.6701 |
| algread_p1_a1_b0d2_k0 | algread grid WP=1 WA=1 WB=0.2 WK=0 | 0.7750 | 0.6800 | 0.7340 | 0.7300 | 0.7311 | 0.7089 | 0.7348 | 0.5453 | 0.6019 | 0.6644 |
| algread_p1_a1_b0d2_k0d5 | algread grid WP=1 WA=1 WB=0.2 WK=0.5 | 0.7750 | 0.6880 | 0.7280 | 0.7300 | 0.7352 | 0.7263 | 0.7348 | 0.5431 | 0.5970 | 0.6673 |
| algread_p1_a1_b0d5_k0 | algread grid WP=1 WA=1 WB=0.5 WK=0 | 0.7520 | 0.6650 | 0.7290 | 0.7160 | 0.7335 | 0.7207 | 0.7359 | 0.5199 | 0.5948 | 0.6610 |
| algread_p1_a1_b0d5_k0d5 | algread grid WP=1 WA=1 WB=0.5 WK=0.5 | 0.7550 | 0.6730 | 0.7260 | 0.7180 | 0.7369 | 0.7519 | 0.7393 | 0.5244 | 0.5904 | 0.6685 |
| algread_p2_a0_b0_k0 | algread grid WP=2 WA=0 WB=0 WK=0 | 0.7430 | 0.6640 | 0.7300 | 0.7120 | 0.7474 | 0.6941 | 0.7137 | 0.5467 | 0.6104 | 0.6625 |
| algread_p2_a0_b0_k0d5 | algread grid WP=2 WA=0 WB=0 WK=0.5 | 0.7430 | 0.6740 | 0.7280 | 0.7150 | 0.7459 | 0.7141 | 0.7159 | 0.5545 | 0.6126 | 0.6686 |
| algread_p2_a0_b0d2_k0 | algread grid WP=2 WA=0 WB=0.2 WK=0 | 0.7400 | 0.6680 | 0.7270 | 0.7110 | 0.7448 | 0.7041 | 0.7111 | 0.5281 | 0.6093 | 0.6595 |
| algread_p2_a0_b0d2_k0d5 | algread grid WP=2 WA=0 WB=0.2 WK=0.5 | 0.7430 | 0.6750 | 0.7240 | 0.7140 | 0.7437 | 0.7181 | 0.7067 | 0.5292 | 0.6056 | 0.6607 |
| algread_p2_a0_b0d5_k0 | algread grid WP=2 WA=0 WB=0.5 WK=0 | 0.7250 | 0.6590 | 0.7140 | 0.7000 | 0.7083 | 0.6996 | 0.7207 | 0.5118 | 0.5881 | 0.6457 |
| algread_p2_a0_b0d5_k0d5 | algread grid WP=2 WA=0 WB=0.5 WK=0.5 | 0.7300 | 0.6710 | 0.7140 | 0.7050 | 0.7309 | 0.7296 | 0.7204 | 0.5177 | 0.5889 | 0.6575 |
| algread_p2_a0d5_b0_k0 | algread grid WP=2 WA=0.5 WB=0 WK=0 | 0.7620 | 0.6700 | 0.7320 | 0.7210 | 0.7252 | 0.6974 | 0.7181 | 0.5460 | 0.5993 | 0.6572 |
| algread_p2_a0d5_b0_k0d5 | algread grid WP=2 WA=0.5 WB=0 WK=0.5 | 0.7610 | 0.6730 | 0.7300 | 0.7210 | 0.7281 | 0.7152 | 0.7181 | 0.5560 | 0.6037 | 0.6642 |
| algread_p2_a0d5_b0d2_k0 | algread grid WP=2 WA=0.5 WB=0.2 WK=0 | 0.7410 | 0.6530 | 0.7240 | 0.7060 | 0.7426 | 0.7041 | 0.7289 | 0.5303 | 0.6093 | 0.6630 |
| algread_p2_a0d5_b0d2_k0d5 | algread grid WP=2 WA=0.5 WB=0.2 WK=0.5 | 0.7430 | 0.6730 | 0.7200 | 0.7120 | 0.7437 | 0.7204 | 0.7267 | 0.5314 | 0.6100 | 0.6664 |
| algread_p2_a0d5_b0d5_k0 | algread grid WP=2 WA=0.5 WB=0.5 WK=0 | 0.7280 | 0.6600 | 0.7190 | 0.7030 | 0.7039 | 0.7007 | 0.7074 | 0.5140 | 0.5881 | 0.6428 |
| algread_p2_a0d5_b0d5_k0d5 | algread grid WP=2 WA=0.5 WB=0.5 WK=0.5 | 0.7320 | 0.6730 | 0.7190 | 0.7080 | 0.7265 | 0.7363 | 0.7070 | 0.5199 | 0.5933 | 0.6566 |
| algread_p2_a1_b0_k0 | algread grid WP=2 WA=1 WB=0 WK=0 | 0.7620 | 0.6630 | 0.7370 | 0.7210 | 0.7296 | 0.6819 | 0.7337 | 0.5460 | 0.6026 | 0.6588 |
| algread_p2_a1_b0_k0d5 | algread grid WP=2 WA=1 WB=0 WK=0.5 | 0.7610 | 0.6660 | 0.7370 | 0.7210 | 0.7304 | 0.7007 | 0.7359 | 0.5560 | 0.6059 | 0.6658 |
| algread_p2_a1_b0d2_k0 | algread grid WP=2 WA=1 WB=0.2 WK=0 | 0.7630 | 0.6640 | 0.7400 | 0.7220 | 0.7315 | 0.7063 | 0.7156 | 0.5296 | 0.5959 | 0.6558 |
| algread_p2_a1_b0d2_k0d5 | algread grid WP=2 WA=1 WB=0.2 WK=0.5 | 0.7660 | 0.6700 | 0.7330 | 0.7230 | 0.7304 | 0.7215 | 0.7111 | 0.5351 | 0.5967 | 0.6589 |
| algread_p2_a1_b0d5_k0 | algread grid WP=2 WA=1 WB=0.5 WK=0 | 0.7280 | 0.6680 | 0.7230 | 0.7060 | 0.7039 | 0.7019 | 0.7430 | 0.5140 | 0.5904 | 0.6506 |
| algread_p2_a1_b0d5_k0d5 | algread grid WP=2 WA=1 WB=0.5 WK=0.5 | 0.7320 | 0.6770 | 0.7170 | 0.7090 | 0.7243 | 0.7352 | 0.7404 | 0.5199 | 0.5911 | 0.6622 |
| convprf_b0_w0 | conversation pseudo relevance B=0 W=0 | 0.7620 | 0.6750 | 0.7510 | 0.7290 | 0.7502 | 0.7181 | 0.7052 | 0.5524 | 0.6026 | 0.6657 |
| convprf_b0_w2 | conversation pseudo relevance B=0 W=2 | 0.7620 | 0.6750 | 0.7510 | 0.7290 | 0.7502 | 0.7181 | 0.7052 | 0.5524 | 0.6026 | 0.6657 |
| convprf_b0_w5 | conversation pseudo relevance B=0 W=5 | 0.7620 | 0.6750 | 0.7510 | 0.7290 | 0.7502 | 0.7181 | 0.7052 | 0.5524 | 0.6026 | 0.6657 |
| convprf_b0d2_w0 | conversation pseudo relevance B=0.2 W=0 | 0.7490 | 0.6700 | 0.7430 | 0.7210 | 0.7496 | 0.7130 | 0.7193 | 0.5378 | 0.6263 | 0.6692 |
| convprf_b0d2_w2 | conversation pseudo relevance B=0.2 W=2 | 0.7590 | 0.6770 | 0.7440 | 0.7270 | 0.7519 | 0.7019 | 0.7344 | 0.5489 | 0.6204 | 0.6715 |
| convprf_b0d2_w5 | conversation pseudo relevance B=0.2 W=5 | 0.7570 | 0.6730 | 0.7440 | 0.7250 | 0.7474 | 0.6774 | 0.7311 | 0.5533 | 0.6226 | 0.6664 |
| convprf_b0d4_w0 | conversation pseudo relevance B=0.4 W=0 | 0.7490 | 0.6590 | 0.7380 | 0.7150 | 0.7463 | 0.7074 | 0.7181 | 0.5144 | 0.6293 | 0.6631 |
| convprf_b0d4_w2 | conversation pseudo relevance B=0.4 W=2 | 0.7630 | 0.6650 | 0.7390 | 0.7220 | 0.7507 | 0.7052 | 0.7389 | 0.5389 | 0.6226 | 0.6713 |
| convprf_b0d4_w5 | conversation pseudo relevance B=0.4 W=5 | 0.7630 | 0.6670 | 0.7370 | 0.7230 | 0.7485 | 0.6996 | 0.7311 | 0.5444 | 0.6248 | 0.6697 |
| convprf_b0d8_w0 | conversation pseudo relevance B=0.8 W=0 | 0.7450 | 0.6420 | 0.7260 | 0.7040 | 0.7244 | 0.7163 | 0.7185 | 0.5113 | 0.6226 | 0.6586 |
| convprf_b0d8_w2 | conversation pseudo relevance B=0.8 W=2 | 0.7600 | 0.6770 | 0.7400 | 0.7260 | 0.7450 | 0.7067 | 0.7452 | 0.5356 | 0.6217 | 0.6708 |
| convprf_b0d8_w5 | conversation pseudo relevance B=0.8 W=5 | 0.7570 | 0.6680 | 0.7420 | 0.7220 | 0.7461 | 0.7178 | 0.7419 | 0.5400 | 0.6278 | 0.6747 |
| filelevel_bm25_files_all_fused | file level retrieval, BM25 reader, bm25_files_all_fused | 0.6140 | 0.5830 | 0.6500 | 0.6250 | 0.7076 | 0.6790 | 0.5983 | 0.5972 | 0.5029 | 0.6205 |
| filelevel_bm25_files_all_only | file level retrieval, BM25 reader, bm25_files_all_only | 0.6340 | 0.5690 | 0.7060 | 0.6540 | 0.7135 | 0.6914 | 0.5726 | 0.5278 | 0.5146 | 0.6162 |
| filelevel_bm25_files_wr_fused | file level retrieval, BM25 reader, bm25_files_wr_fused | 0.6410 | 0.5140 | 0.5390 | 0.5730 | 0.7018 | 0.6296 | 0.5641 | 0.5972 | 0.5380 | 0.6104 |
| filelevel_bm25_files_wr_only | file level retrieval, BM25 reader, bm25_files_wr_only | 0.5690 | 0.4720 | 0.5670 | 0.5510 | 0.7427 | 0.6049 | 0.7350 | 0.5278 | 0.5205 | 0.6320 |
| filelevel_bm25_solver | file level retrieval, BM25 reader, bm25_solver | 0.7060 | 0.6110 | 0.7560 | 0.7110 | 0.8538 | 0.7284 | 0.6410 | 0.5972 | 0.4971 | 0.6739 |
| fast_S100_30 | whole fast method S100_30: fuse0 top 100, ds_bgeb student, ftpub reader on its top 30 | 0.8840 | 0.7600 | 0.8230 | 0.8230 | 0.8248 | 0.7833 | 0.8381 | 0.6800 | 0.7200 | 0.7693 |
| fast_X100 | whole fast method X100: early exit, ftpub reader top 20, ranks to 100 if best log p(yes) < -0.05 | 0.8710 | 0.7570 | 0.8320 | 0.8210 | 0.7807 | 0.7837 | 0.7667 | 0.5856 | 0.7030 | 0.7239 |
| nr_fast_multi_gpu | 2 GPU and 4 GPU columns of S100_30 and X100 |  |  |  |  |  |  |  |  |  |  |
| time_fast_S100_30_1gpu | seconds per run, S100_30, 1 GPU, model load excluded |  |  |  |  | 50.4000 | 44.4000 | 40.3000 | 42.2000 | 43.3000 | 44.1000 |
| time_fast_X100_1gpu | seconds per run, X100, 1 GPU, model load excluded |  |  |  |  | 66.3000 | 54.1000 | 46.2000 | 52.9000 | 51.8000 | 54.3000 |
