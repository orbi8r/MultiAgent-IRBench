| tag | method | dev_online | dev_library | dev_literature | dev_mean | test_on_call | test_due_diligence | test_executive_assistants | test_help_desk | test_textbook_team | test_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b3_random_s0 | uniform random permutation of all pieces per question, seed 0, no list builder | 0.0824 | 0.0378 | 0.0656 | 0.0619 | 0.1204 | 0.0778 | 0.1230 | 0.0706 | 0.0867 | 0.0957 |
| b3_random_s1 | uniform random permutation of all pieces per question, seed 1, no list builder | 0.0824 | 0.0225 | 0.0400 | 0.0483 | 0.0761 | 0.0880 | 0.1481 | 0.0793 | 0.1081 | 0.0999 |
| b3_random_s2 | uniform random permutation of all pieces per question, seed 2, no list builder | 0.0754 | 0.0390 | 0.0433 | 0.0526 | 0.0556 | 0.0681 | 0.1137 | 0.0496 | 0.1044 | 0.0783 |
| b3_recent | newest piece first, same list for every question, no list builder | 0.0571 | 0.0154 | 0.0322 | 0.0349 | 0.0974 | 0.0444 | 0.0204 | 0.0207 | 0.0522 | 0.0470 |
| b3_oldest | chronological, oldest piece first, same list for every question, no list builder | 0.0159 | 0.0154 | 0.0000 | 0.0104 | 0.0267 | 0.0341 | 0.0037 | 0.0111 | 0.0119 | 0.0175 |
| b3_long | longest piece first (tokens), same list for every question, no list builder | 0.0045 | 0.0189 | 0.0267 | 0.0167 | 0.0067 | 0.0300 | 0.0156 | 0.0344 | 0.0096 | 0.0193 |
| b3_short | shortest piece first (tokens), same list for every question, no list builder | 0.0068 | 0.0000 | 0.0089 | 0.0052 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b3_bm25_ws | BM25 over plain piece text, lowercase whitespace split, no list builder | 0.4233 | 0.3950 | 0.3933 | 0.4039 | 0.2459 | 0.3641 | 0.3574 | 0.3637 | 0.3307 | 0.3324 |
| b3_bm25_word | BM25 over plain piece text, lowercase \w+ tokens, no list builder | 0.5011 | 0.4903 | 0.5037 | 0.4984 | 0.3606 | 0.4359 | 0.4170 | 0.4265 | 0.4063 | 0.4093 |
| b3_bm25_stem | BM25 over plain piece text, \w+ tokens, English stop words removed, Porter stem, no list builder | 0.4940 | 0.5317 | 0.4867 | 0.5041 | 0.3724 | 0.4441 | 0.4704 | 0.4280 | 0.3670 | 0.4164 |
| b3_bm25_ident | BM25 over plain piece text, identifier split (camelCase, snake_case, paths on / . _ -), lowercase, no list builder | 0.4955 | 0.4947 | 0.4881 | 0.4928 | 0.3872 | 0.4652 | 0.4048 | 0.4376 | 0.4307 | 0.4251 |
| b3_dense_minilm | off the shelf dense, all-MiniLM-L6-v2 cosine over the ctx text, no list builder | 0.3704 | 0.4704 | 0.4072 | 0.4160 | 0.2783 | 0.3504 | 0.4781 | 0.3592 | 0.3580 | 0.3648 |
| b3_dense_bgeb | off the shelf dense, bge-base-en-v1.5 cosine over the ctx text, query instruction, no list builder | 0.4490 | 0.5140 | 0.4722 | 0.4784 | 0.3476 | 0.4341 | 0.4996 | 0.3687 | 0.3770 | 0.4054 |
| b3_dense_e5b | off the shelf dense, e5-base-v2 cosine over the ctx text, query and passage prefixes, no list builder | 0.4516 | 0.4863 | 0.4650 | 0.4676 | 0.3794 | 0.4422 | 0.5448 | 0.4063 | 0.4072 | 0.4360 |
| b3_dense_qwen3e | off the shelf dense, Qwen3-Embedding-0.6B cosine over the ctx text, its query instruction, no list builder | 0.4573 | 0.5798 | 0.4656 | 0.5009 | 0.3167 | 0.4042 | 0.4837 | 0.3980 | 0.3922 | 0.3990 |
| b3_rrf_stem_bgeb | reciprocal rank fusion of b3_bm25_stem and b3_dense_bgeb, no list builder | 0.5136 | 0.5943 | 0.5344 | 0.5475 | 0.4261 | 0.5159 | 0.5359 | 0.4870 | 0.4185 | 0.4767 |
| b3_recent_lb | newest piece first through the list builder | 0.0559 | 0.0165 | 0.0278 | 0.0334 | 0.0985 | 0.0526 | 0.0133 | 0.0174 | 0.0456 | 0.0455 |
| b3_bm25_word_lb | BM25 \w+ through the list builder | 0.6761 | 0.5513 | 0.6887 | 0.6387 | 0.6169 | 0.6022 | 0.5493 | 0.5024 | 0.5291 | 0.5600 |
| b3_bm25_stem_lb | BM25 stop words and Porter stem through the list builder | 0.6878 | 0.5990 | 0.7006 | 0.6625 | 0.6680 | 0.6315 | 0.5970 | 0.5232 | 0.5378 | 0.5915 |
| b3_dense_bgeb_lb | bge-base-en-v1.5 dense through the list builder | 0.5873 | 0.5932 | 0.6522 | 0.6109 | 0.5676 | 0.6537 | 0.6381 | 0.4120 | 0.5241 | 0.5591 |
| b3_dense_qwen3e_lb | Qwen3-Embedding-0.6B dense through the list builder | 0.6315 | 0.6054 | 0.6144 | 0.6171 | 0.5809 | 0.6035 | 0.6548 | 0.4756 | 0.5543 | 0.5738 |
| b3_rrf_stem_bgeb_lb | RRF of BM25 stem and bge-base dense through the list builder | 0.7154 | 0.6617 | 0.7139 | 0.6970 | 0.6898 | 0.7030 | 0.7319 | 0.5564 | 0.5707 | 0.6504 |
