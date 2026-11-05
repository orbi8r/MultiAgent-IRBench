| tag | method | dev_online | dev_library | dev_literature | dev_mean | test_on_call | test_due_diligence | test_executive_assistants | test_help_desk | test_textbook_team | test_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f2_z2_first | z2 first stage alone, top 200 in fused order | 0.5820 | 0.6286 | 0.5289 | 0.5798 | 0.4256 | 0.5013 | 0.5870 | 0.5076 | 0.4665 | 0.4976 |
| f2_z2_ftpub_raw | public trace ftpub over z2 K200, reader order without the list builder | 0.6848 | 0.7837 | 0.5956 | 0.6880 | 0.5348 | 0.6504 | 0.7159 | 0.6559 | 0.5915 | 0.6297 |
| f2_z2_zs06 | zero shot Qwen3 Reranker 0.6B over z2 K200, list builder | 0.8413 | 0.7719 | 0.7961 | 0.8031 | 0.7389 | 0.7522 | 0.7489 | 0.6889 | 0.6919 | 0.7241 |
| f2_z2_zs06_raw | zero shot Qwen3 Reranker 0.6B over z2 K200, reader order without the list builder | 0.6104 | 0.7238 | 0.5672 | 0.6338 | 0.4689 | 0.5496 | 0.6222 | 0.6156 | 0.5156 | 0.5544 |
| f2_pub25 | ftpub recipe on every 4th public row (1241 of 4964) | 0.8764 | 0.7890 | 0.8433 | 0.8363 | 0.7459 | 0.8144 | 0.7815 | 0.6873 | 0.7115 | 0.7481 |
| f2_pub50 | ftpub recipe on every 2nd public row (2482 of 4964) | 0.8730 | 0.7808 | 0.8422 | 0.8320 | 0.7835 | 0.8174 | 0.8004 | 0.6878 | 0.7526 | 0.7683 |
| f2_c4b | cascade: 4B rereads the zs06 top 50 of z2 | 0.8403 | 0.7695 | 0.8289 | 0.8129 | 0.7754 | 0.8074 | 0.7422 | 0.7046 | 0.7309 | 0.7521 |
| f2_c8b | cascade: 8B rereads the zs06 top 50 of z2 | 0.8692 | 0.7991 | 0.8206 | 0.8296 | 0.7917 | 0.8041 | 0.7796 | 0.6951 | 0.7207 | 0.7582 |
| f2_ds_bgeb | student ds_bgeb over z2 K200 | 0.7763 | 0.7268 | 0.8122 | 0.7718 | 0.7750 | 0.7459 | 0.8037 | 0.6516 | 0.7093 | 0.7371 |
| f2_z8b | zero shot Qwen3 Reranker 8B over z2 K200 | 0.8715 | 0.7980 | 0.8083 | 0.8259 | 0.7909 | 0.8096 | 0.7863 | 0.6929 | 0.7104 | 0.7580 |
| f2_lw4 | listwise Qwen3 4B Instruct reads the zs06 top 10 of z2 | 0.8643 | 0.7979 | 0.7961 | 0.8194 | 0.7693 | 0.7600 | 0.7759 | 0.7063 | 0.7144 | 0.7452 |
| f2_lw4pub | listwise Qwen3 4B Instruct reads the ftpub top 10 of z2 | 0.8791 | 0.8046 | 0.8783 | 0.8540 | 0.8074 | 0.8219 | 0.8322 | 0.7433 | 0.7504 | 0.7910 |
| f2_lw8 | listwise Qwen3 8B (nf4) reads the zs06 top 10 of z2 | 0.8564 | 0.7908 | 0.8256 | 0.8242 | 0.7619 | 0.7756 | 0.7659 | 0.7033 | 0.6989 | 0.7411 |
| f2_lw8pub | listwise Qwen3 8B (nf4) reads the ftpub top 10 of z2 | 0.8802 | 0.7951 | 0.8644 | 0.8466 | 0.8080 | 0.8407 | 0.8300 | 0.7293 | 0.7443 | 0.7904 |
| f2_fast_K50 | fast pipeline config K50: ftpub on fuse0 top 50 | 0.8738 | 0.7597 | 0.8167 | 0.8167 | 0.8193 | 0.7737 | 0.7985 | 0.6044 | 0.6865 | 0.7365 |
| f2_fast_S200_30 | fast pipeline config S200_30: ds_bgeb student on fuse0 top 200, ftpub reads the student top 30 | 0.8787 | 0.7664 | 0.8344 | 0.8265 | 0.8311 | 0.8189 | 0.8519 | 0.6978 | 0.7370 | 0.7873 |
| f2_fast_S50_20 | fast pipeline config S50_20: ds_bgeb student on fuse0 top 50, ftpub reads the student top 20 | 0.8949 | 0.7411 | 0.8056 | 0.8139 | 0.8081 | 0.7759 | 0.8156 | 0.6122 | 0.6865 | 0.7397 |
| f2_fast_X200 | fast pipeline config X200: early exit, ftpub on top 20 and up to 200 when unsure | 0.8666 | 0.7778 | 0.8411 | 0.8285 | 0.7930 | 0.7904 | 0.7733 | 0.6089 | 0.7252 | 0.7381 |
| f2_fast_X50 | fast pipeline config X50: early exit, ftpub on top 20 and up to 50 when unsure | 0.8624 | 0.7313 | 0.8167 | 0.8035 | 0.7837 | 0.7559 | 0.7696 | 0.5533 | 0.6807 | 0.7087 |
| t2_c4b | lexical + dense + zs06 over K200 + 4B over the top 50, scoring only |  |  |  |  |  |  |  |  |  |  |
| t2_ds_bgeb | lexical + dense + ds_bgeb over K200, scoring only |  |  |  |  |  |  |  |  |  |  |
| t2_ftpub_k200 | ftpub on fuse0 top 200, timed_run K200 total without model load |  |  |  |  |  |  |  |  |  |  |
| f2_maxb320 | ftpub over z2 K200 with the body capped at 320 tokens | 0.8741 | 0.7757 | 0.8372 | 0.8290 | 0.7389 | 0.7919 | 0.8048 | 0.6887 | 0.7226 | 0.7494 |
| f2_maxb640 | ftpub over z2 K200 with the body capped at 640 tokens | 0.8696 | 0.7991 | 0.8456 | 0.8381 | 0.8352 | 0.8107 | 0.8256 | 0.7189 | 0.7326 | 0.7846 |
| f2_exp_fuse0_k200_m100 | ftpub on fuse0 top 200 plus the top 100 of each expansion list | 0.9027 | 0.8121 | 0.8389 | 0.8512 | 0.8300 | 0.8589 | 0.8456 | 0.7167 | 0.7778 | 0.8058 |
| f2_exp_fuse0_k200_m50 | ftpub on fuse0 top 200 plus the top 50 of each expansion list | 0.8947 | 0.8215 | 0.8389 | 0.8517 | 0.8267 | 0.8433 | 0.8411 | 0.7222 | 0.7756 | 0.8018 |
| f2_fl_first1_k100 | first stage fusion variant first1, top 100 read by ftpub | 0.8762 | 0.7801 | 0.8211 | 0.8258 | 0.8204 | 0.8159 | 0.8200 | 0.6378 | 0.7133 | 0.7615 |
| f2_fl_first1_k100_m25 | first stage fusion variant first1, top 100 read by ftpub plus the top 25 of each expansion list | 0.8743 | 0.7920 | 0.8211 | 0.8291 | 0.8141 | 0.8193 | 0.8411 | 0.6378 | 0.7356 | 0.7696 |
| f2_fl_first3_k100 | first stage fusion variant first3, top 100 read by ftpub | 0.8581 | 0.7679 | 0.8233 | 0.8164 | 0.8107 | 0.7952 | 0.8067 | 0.6422 | 0.6981 | 0.7506 |
| f2_fl_first3_k100_m25 | first stage fusion variant first3, top 100 read by ftpub plus the top 25 of each expansion list | 0.8596 | 0.7991 | 0.8189 | 0.8258 | 0.8652 | 0.8070 | 0.8122 | 0.7211 | 0.7489 | 0.7909 |
| f2_fl_first5_k100 | first stage fusion variant first5, top 100 read by ftpub | 0.8989 | 0.7963 | 0.8456 | 0.8469 | 0.8163 | 0.8015 | 0.8167 | 0.6578 | 0.7304 | 0.7645 |
| f2_fl_first5_k100_m25 | first stage fusion variant first5, top 100 read by ftpub plus the top 25 of each expansion list | 0.8981 | 0.8132 | 0.8411 | 0.8508 | 0.8374 | 0.8111 | 0.8344 | 0.6989 | 0.7600 | 0.7884 |
| f2_fl_no_bm_ctx_k100 | first stage fusion variant no_bm_ctx, top 100 read by ftpub | 0.8989 | 0.7967 | 0.8478 | 0.8478 | 0.8141 | 0.8015 | 0.8189 | 0.6467 | 0.7104 | 0.7583 |
| f2_fl_no_bm_ctx_k100_m25 | first stage fusion variant no_bm_ctx, top 100 read by ftpub plus the top 25 of each expansion list | 0.8981 | 0.8132 | 0.8611 | 0.8575 | 0.8356 | 0.8111 | 0.8344 | 0.7144 | 0.7504 | 0.7892 |
| f2_fl_no_bm_nov_k100 | first stage fusion variant no_bm_nov, top 100 read by ftpub | 0.8966 | 0.7943 | 0.8367 | 0.8425 | 0.8163 | 0.8041 | 0.8289 | 0.6578 | 0.7089 | 0.7632 |
| f2_fl_no_bm_nov_k100_m25 | first stage fusion variant no_bm_nov, top 100 read by ftpub plus the top 25 of each expansion list | 0.8959 | 0.8132 | 0.8344 | 0.8479 | 0.8319 | 0.8226 | 0.8500 | 0.7156 | 0.7407 | 0.7921 |
| f2_fl_no_bm_plain_k100 | first stage fusion variant no_bm_plain, top 100 read by ftpub | 0.9012 | 0.7920 | 0.8367 | 0.8433 | 0.8174 | 0.8037 | 0.8456 | 0.6689 | 0.7370 | 0.7745 |
| f2_fl_no_bm_plain_k100_m25 | first stage fusion variant no_bm_plain, top 100 read by ftpub plus the top 25 of each expansion list | 0.8981 | 0.7920 | 0.8322 | 0.8408 | 0.8363 | 0.8111 | 0.8478 | 0.7022 | 0.7693 | 0.7933 |
| f2_fl_no_bm_raw_k100 | first stage fusion variant no_bm_raw, top 100 read by ftpub | 0.8955 | 0.7967 | 0.8478 | 0.8467 | 0.8152 | 0.8037 | 0.8589 | 0.6622 | 0.7211 | 0.7722 |
| f2_fl_no_bm_raw_k100_m25 | first stage fusion variant no_bm_raw, top 100 read by ftpub plus the top 25 of each expansion list | 0.8925 | 0.7920 | 0.8433 | 0.8426 | 0.8319 | 0.8133 | 0.8500 | 0.7022 | 0.7548 | 0.7904 |
| f2_fl_no_qdecomp_k100 | first stage fusion variant no_qdecomp, top 100 read by ftpub | 0.8989 | 0.7920 | 0.8478 | 0.8462 | 0.8185 | 0.8015 | 0.8300 | 0.6611 | 0.7126 | 0.7647 |
| f2_fl_no_qdecomp_k100_m25 | first stage fusion variant no_qdecomp, top 100 read by ftpub plus the top 25 of each expansion list | 0.8981 | 0.7920 | 0.8411 | 0.8437 | 0.8419 | 0.8111 | 0.8344 | 0.7022 | 0.7504 | 0.7880 |
| f2_fl_no_qtype_k100 | first stage fusion variant no_qtype, top 100 read by ftpub | 0.8989 | 0.7931 | 0.8456 | 0.8459 | 0.8141 | 0.8015 | 0.8189 | 0.6467 | 0.7259 | 0.7614 |
| f2_fl_no_qtype_k100_m25 | first stage fusion variant no_qtype, top 100 read by ftpub plus the top 25 of each expansion list | 0.8981 | 0.8132 | 0.8611 | 0.8575 | 0.8356 | 0.8111 | 0.8344 | 0.7144 | 0.7659 | 0.7923 |
| f2_fl_no_rm3_k100 | first stage fusion variant no_rm3, top 100 read by ftpub | 0.8989 | 0.7943 | 0.8456 | 0.8463 | 0.8141 | 0.8015 | 0.8433 | 0.6633 | 0.7093 | 0.7663 |
| f2_fl_no_rm3_k100_m25 | first stage fusion variant no_rm3, top 100 read by ftpub plus the top 25 of each expansion list | 0.8981 | 0.7920 | 0.8433 | 0.8445 | 0.8333 | 0.8178 | 0.8344 | 0.7178 | 0.7504 | 0.7907 |
| pl_dp_exact | P2 RL style Model B, 12 feature linear policy, coordinate ascent on the BAR metric, dedup exact | 0.4739 | 0.5485 | 0.4359 | 0.4861 | 0.3683 | 0.4687 | 0.5159 | 0.3997 | 0.4906 | 0.4486 |
| pl_dp_mh85 | P2 RL style Model B, 12 feature linear policy, coordinate ascent on the BAR metric, dedup mh85 | 0.4856 | 0.5674 | 0.4411 | 0.4980 | 0.3689 | 0.4650 | 0.4874 | 0.3919 | 0.4791 | 0.4385 |
| pl_modelA_exact | P4 RL style Model A (neural RL reranker, Plackett Luce REINFORCE), dedup exact | 0.4471 | 0.3712 | 0.3137 | 0.3773 | 0.3085 | 0.3787 | 0.4363 | 0.3556 | 0.4248 | 0.3808 |
| pl_modelA_mh85 | P4 RL style Model A (neural RL reranker, Plackett Luce REINFORCE), dedup mh85 | 0.4392 | 0.3783 | 0.3259 | 0.3811 | 0.3093 | 0.3830 | 0.4174 | 0.3572 | 0.4257 | 0.3785 |
| pl_h1_dppool | the original 12 feature Model B refitted on our pool, coordinate ascent on the BAR metric, raw order | 0.6319 | 0.5686 | 0.5548 | 0.5851 | 0.3978 | 0.5496 | 0.6000 | 0.4813 | 0.4463 | 0.4950 |
| pl_h1_dppool_lb | the original 12 feature Model B refitted on our pool, coordinate ascent on the BAR metric, through our list builder | 0.7118 | 0.6087 | 0.6581 | 0.6596 | 0.5689 | 0.6219 | 0.6893 | 0.5509 | 0.5524 | 0.5967 |
| pl_h2_dpft | the original Model B with the ftpub z score as a 13th feature, raw order | 0.7326 | 0.7569 | 0.6393 | 0.7096 | 0.6235 | 0.7485 | 0.7430 | 0.6804 | 0.6559 | 0.6903 |
| pl_h2_dpft_lb | the original Model B with the ftpub z score as a 13th feature, through our list builder | 0.8313 | 0.7687 | 0.8233 | 0.8078 | 0.8028 | 0.8363 | 0.8004 | 0.7111 | 0.6867 | 0.7674 |
| pl_h3_blend | fusion z(the original Model B score) + w z(ftpub), w chosen on dev, raw order | 0.7320 | 0.7577 | 0.6385 | 0.7094 | 0.5985 | 0.7074 | 0.7274 | 0.6967 | 0.6474 | 0.6755 |
| pl_h3_blend_lb | fusion z(the original Model B score) + w z(ftpub), w chosen on dev, through our list builder | 0.8562 | 0.7707 | 0.8463 | 0.8244 | 0.7802 | 0.8233 | 0.8011 | 0.7252 | 0.7359 | 0.7731 |
| pl_h4_seeds | our pipeline (ftpub order, list builder) with the original Model B choosing the location provenance seeds | 0.7780 | 0.8002 | 0.7189 | 0.7657 | 0.7337 | 0.7733 | 0.8144 | 0.7100 | 0.7044 | 0.7472 |
