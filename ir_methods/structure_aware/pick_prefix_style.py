"""Picks the prefix style with the higher dev mean, writes the gate, freezes the winner and scores its test subs only when the gate is go.
    python pick_prefix_style.py STYLES SEED          e.g. "P1 P2" 1  ->  $IRBENCH_OUT/gate.json, frozen file of the winner, test subs when go
Dev tags f3_sp06_<style>_s<seed> come from the dev runs of the earlier sp06 steps (out/*/subs/dev). Test is touched once, after the freeze, for the winner only."""
import json, os, subprocess, sys
import study_helpers, dev_gate

styles, seed = sys.argv[1].split(), sys.argv[2]
suf = os.environ.get("TAGSUF", "")  # smokes only
ctl = f"f2_ftpub_s{seed}"
tags = {s: f"f3_sp06_{s}_s{seed}{suf}" for s in styles}
means = {t: dev_gate.pooled(dev_gate.dev_scores(t)) for t in tags.values()}
win = max(styles, key=lambda s: (means[tags[s]], -styles.index(s)))  # a tie keeps the first style
g = dev_gate.gate(tags[win], ctl)
g.update(style=win, dev_means=means, rule=f"dev mean beats {ctl} by {dev_gate.GAIN} and is not worse on 2 of 3 dev runs")
study_helpers.atomic(study_helpers.OUT / "gate.json", g, indent=1)
print(json.dumps(g, indent=1), flush=True)
model = f"Qwen/Qwen3-Reranker-0.6B+lora={study_helpers.BASE}/adapters3/{tags[win]}"
cmd = [sys.executable, study_helpers.script("score_prefix_reader.py"), "--tag", tags[win], "--model", model, "--style", win, "--freeze", "--norow", "--splits", "test" if g["go"] else "none",
       "--train", f"prefixed rows {win} of pub_00 to pub_14, 4964 rows, 1 epoch, seed {seed}, train_prefix_reader.py (train_rr recipe)",
       "--section", "9.2 full3 trained prefix 0.6B", "--method", f"ftpub recipe trained on {win} prefixed public rows over z2 K200, list builder",
       "--note", f"dev decided the style: {json.dumps(means)}; gate {g['why']}",
       "--evidence", json.dumps(dict(dev_means=means, gate=g["why"], go=g["go"]))]
subprocess.run(cmd, check=True)
if not g["go"]:
    (study_helpers.OUT / "SKIPPED_TEST.txt").write_text("gate no go, no test sub written: " + g["why"] + "\n")
