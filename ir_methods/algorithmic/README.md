# algorithmic

Readers with no neural model and nothing fitted on the runs. They rerank the lexical top 200 and go through the same list builder as every other reader. Both scripts score against the answer key as a diagnostic, so they set `IRBENCH_KEYS=1` themselves. Run them from `$IRBENCH_WORK` after `first_stage/lexical_candidates.py` and `first_stage/cpu_families.py`.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `algorithmic_reader.py` | score = z(BM25 context) + weights on answer span proximity, answer type, brevity and outcome kind. Writes every cell of the weight grid as a submission. `INTERNAL_SCORE=1` also prints the in sample and leave one run out tables | `data/<run>/fam_cpu.json` | `$SUBDIR/algread_<cell>/<run>.json`, tags `algread_*` |
| `file_level_reader.py` | for location questions, retrieves files instead of pieces (BM25 over everything that touched a path), then runs the provenance solver on the top 5 files | `data/<run>/fam_cpu.json`, `lex.json` | `$SUBDIR/filelevel_<variant>/<run>.json`, tags `filelevel_*` |

Recipe: `recipes/test5/30_cpu_families.sh`.
