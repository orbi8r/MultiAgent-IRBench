# training_data

Training rows for the readers and students. None of it comes from the benchmark: it starts from public SWE agent trajectories (the Hugging Face dataset nebius/SWE-agent-trajectories) and local models.

Order of use:

1. `convert_public_traces.py OUTDIR NRUNS T` turns trajectories into pseudo team records in the benchmark format. Each pseudo run bundles T trajectories, one per member. `OFFSET` skips trajectories already used, `K0` is the number of the first pseudo run, so a second disjoint batch is made with `OFFSET=<NRUNS*T> K0=<NRUNS>`.
2. `prepare_public_runs.py pieces RUNDIR DATADIR` writes the contextual pieces of a pseudo run.
3. `generate_questions.py RUN N MODEL` has a local instruct model write N questions a teammate might ask, each answered by one piece. Output `data/<run>/synth.jsonl`.
4. `prepare_public_runs.py rows DATADIR` builds `train_synth.jsonl`, rows of `{query, pos, negs}`.

`make_train_rows.py synth RUN` builds the same rows for a benchmark run from its own synthetic questions. `make_train_rows.py real RUN` builds rows from that run's answer key, and exists only for the probe adapters of section 5.4 that were trained on dev records on purpose.

Recipe: `recipes/test5/10_train_public.sh`.
