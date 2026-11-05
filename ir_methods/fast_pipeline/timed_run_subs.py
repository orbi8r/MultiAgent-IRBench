"""Turn the lists that timed_run.py wrote into frozen submissions and a timing file.
Inputs: $IRBENCH_OUT/fast/<tag>_1g_<run>.json for the five test runs (timed_run.py output, the lists and the times).
Outputs: $IRBENCH_OUT/subs/fast_<config>/<run>.json for every reader config, and $IRBENCH_OUT/fast/times_<tag>.json.
    python timed_run_subs.py [tag]     (default tag: fastd)"""
import json, os, sys
OUT = os.environ["IRBENCH_OUT"]
TAG = sys.argv[1] if len(sys.argv) > 1 else "fastd"
RUNS = "on_call due_diligence executive_assistants help_desk textbook_team".split()
times = {}
for r in RUNS:
    p = f"{OUT}/fast/{TAG}_1g_{r}.json"
    if not os.path.exists(p):
        continue
    J = json.load(open(p)); times[r] = J["times"]
    for cfg, L in J["lists"].items():
        d = f"{OUT}/subs/fast_{cfg}"; os.makedirs(d, exist_ok=True)
        json.dump(L, open(f"{d}/{r}.json", "w"))
json.dump(times, open(f"{OUT}/fast/times_{TAG}.json", "w"))
print("timed_run_subs", TAG, sorted(times))
