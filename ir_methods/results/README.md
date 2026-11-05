# results

The result tables behind the document, as they were at the time of writing. Open them with any csv reader.

| File | Rows | Content |
|---|---|---|
| `results_test5.csv` | 220 | the frozen test study: zero shot readers, ftpub readers and variants, students, families, expansion, fast pipeline timing |
| `results_full2.csv` | 57 | the main study rerun with leave one run out dev scores and frozen test scores, with bootstrap intervals and timing |
| `results_full3.csv` | 28 | the structure aware variants with paired gains over the flat controls |
| `results_full3b.csv` | 22 | the lexical and dense baselines on all eight public runs |
| `tag_index.csv` | 330 | one row per result tag of the document |
| `doc_script_names.csv` | 77 | script and recipe names as the document spells them, with the file that has them now (the files were renamed after the document was written) |

`tag_index.csv` columns: `tag`, `results_file` (which csv holds its row), `section` (doc section), `method`, `entry_point` (the script that produced it, relative to `ir_methods/`), `recipe` (the recipe that runs it) and `frozen_setting`. A tag in several csv files is kept from the last of test5, full2, full3, full3b, which is the rule of the document build.

Columns of the result csv files: `tag`, `section`, `method`, `doc_kind`, `test_clean` (clean or flagged), `frozen_setting`, the three dev run scores and `dev_mean`, the five test run scores and `test_mean`, `test_minus_dev`, `dev_time`, `test_time`, `completeness`, `clean_reason` and notes. The later study files add bootstrap intervals and paired gain columns. Scores are the `score.py` v5_abs value from 0 to 1.

Rows that the document leaves out (tags it drops, and rows fitted on an older answer key) are not here. To rebuild a table from scratch run the `99_aggregate.sh` recipe of its study.

Each csv has a markdown twin (`results_full2.md` and so on) with the tag, method and per run scores only, for reading on GitHub. Rebuild one with `python3 aggregate/csv_to_md.py results/results_full2.csv > results/results_full2.md`.
