"""The RL trainer of the repo's method/ package, run under our scorer. This file holds no copy of the trainer.
It reads method/rl_trainer.py, makes the one edit below on the text and runs the result as the module patched_rl_trainer,
so `from patched_rl_trainer import RLTrainer` gives the original class with one changed line.

The edit: in train_epoch the reward call `score_query(answer_entry, ranked_ids, tokens_map)` becomes
`score_query(answer_entry, ranked_ids, item["run"], credit=item["credits"])`. The repo score.py takes the loaded run and the
credit table of the question, and fit_model_a.py puts both into every item.
Left as it is: evaluate() keeps the old call and reads above_floor and full_support, which the repo score.py does not return.
fit_model_a.py never calls it (train is called with eval_data=None and no imitation epochs), so the unchanged lines never run.
Run from the repo root so that `import method` finds the package."""
import sys
import types
from pathlib import Path

import method

SOURCE = Path(method.__file__).parent / "rl_trainer.py"
OLD = "\n                result = score_query(answer_entry, ranked_ids, tokens_map)"
NEW = '\n                result = score_query(answer_entry, ranked_ids, item["run"], credit=item["credits"])'

text = SOURCE.read_text(encoding="utf-8")
assert text.count(OLD) == 1, "method/rl_trainer.py is not the version this patch was written for"
module = types.ModuleType(__name__)
module.__file__ = str(SOURCE)
sys.modules[__name__] = module
exec(compile(text.replace(OLD, NEW), str(SOURCE), "exec"), module.__dict__)
