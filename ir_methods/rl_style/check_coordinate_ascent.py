"""Check: hybrid_lib.fit_cd on the original 12 features gives the same weights as the original SubmodularDirectPolicy.fit on the same items (dev run, exact dedup).
    python check_coordinate_ascent.py [n_iter]"""
import pickle, sys
import numpy as np
from rl_common import DEV, FEATS, load_key
import score as S
from method.reranker import SubmodularDirectPolicy
from hybrid_lib import her12, prep, fit_cd

n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1
r = DEV[0]
items = pickle.load(open(FEATS / f"items_{r}_exact.pkl", "rb"))
run = S.load_run(f"runs/{r}")
key = load_key(r)
data = [dict(it, tokens_map=run["tokens"], answer_entry=key[q], credits=S.credits(key[q], run), run_text=run["text"]) for q, it in items.items() if q in key]
init = [1.0] + [0.0] * 11
p = SubmodularDirectPolicy(weights=np.array(init, dtype=np.float32))
sc = p.fit(data, n_iter=n_iter, verbose=False)
fast = [prep(d, her12(d["block_ids"], d["retrieval_feats"], d["struct_feats"], run["tokens"])) for d in data]
w, sc2 = fit_cd(fast, init, n_iter)
print("the original fit", round(float(sc), 5), "ours", round(sc2, 5), "max |dw|", float(np.abs(np.array(p.weights) - w).max()))
assert abs(sc - sc2) < 1e-4 and np.allclose(p.weights, w, atol=1e-5), "fit_cd differs from the original fit"
print("SELFTEST OK")
