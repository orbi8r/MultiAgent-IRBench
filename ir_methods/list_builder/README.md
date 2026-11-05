# list_builder

The last stage: from scored candidates to the ordered list of piece ids that `score.py` reads.

| Script | What it does |
|---|---|
| `build_list.py` | the list builder: reranked candidates with repeats dropped and the earliest copy kept, with the provenance solver in front for location questions. `python build_list.py TAG MODEL [--write OUTDIR]` scores each run with the public scorer |
| `file_provenance.py` | file provenance for location questions: which file a piece belongs to, where the file went later (mv, cp, git mv), which pieces state the final path. Keys no run, agent or folder name |
| `write_submissions.py MODE ROW ...` | writes frozen submissions to `$IRBENCH_OUT/subs/<ROW>/<run>.json` from saved reader scores. Modes `gpu`, `raw`, `first`, `ens` (z scored blend), `lw` (listwise picks first), `casc` |
| `write_pool_submissions.py pub\|stu\|fam` | submissions for the pool based rows: early exit grids and expansion (`pub`), student and cascade rows (`stu`), family sweeps read by an adapter (`fam`) |

The frozen settings of the document are `EARLIEST=1`, `LOC_SCAN=30`, `UTIL_A=0`. `write_submissions.py` and `write_pool_submissions.py` set them in code.
