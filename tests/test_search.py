"""Checks the evasion search itself, not just its constraints.

Covers three properties the constraint checks do not:
  * the accept-only-improvements loop can never raise the model's attack score;
  * a fixed seed makes the whole search repeatable;
  * summarise() computes the reported mean/std over seeds correctly.

Run: python tests/test_search.py   (downloads the data on first run)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nids_robustness as n

Xtr, ytr, Xte, yte = n.load()
lo, hi = n.bounds_from(Xtr)
model = n.make_model("rf", 0).fit(Xtr.iloc[:20000], ytr[:20000])
att = Xte[yte == 1].iloc[:500].reset_index(drop=True)

# 1. the search never raises the attack score, on any record
adv = n.evade(model, att, lo, hi, np.random.default_rng(0), iters=10)
before = model.predict_proba(att)[:, 1]
after = model.predict_proba(adv)[:, 1]
assert (after <= before + 1e-9).all(), "search raised a record's attack score"
assert (after < before).any(), "search changed no score at all - is it doing anything?"

# 2. fixed seed => identical result; iters=0 leaves every record untouched
again = n.evade(model, att, lo, hi, np.random.default_rng(0), iters=10)
assert adv.equals(again), "same seed gave a different result"
zero = n.evade(model, att, lo, hi, np.random.default_rng(0), iters=0)
assert zero.reset_index(drop=True).equals(att), "iters=0 altered a record"

# 3. summarise() reports the true mean/std (ddof=0) over the per-seed rows
def row(seed, acc, detect):
    m = {"acc": acc, "detect": detect, "fpr": 0.01,
         "detect_under_attack": 0.5, "detect_under_strong_attack": 0.25}
    return {"seed": seed, "model": "rf", "val_clean_acc": acc,
            "holdout_baseline": dict(m), "holdout_hardened": dict(m),
            "kddtest_exploratory_baseline": dict(m), "kddtest_exploratory_hardened": dict(m)}

rows = [row(0, 0.90, 0.80), row(1, 0.94, 0.90), row(2, 0.92, 0.85)]
s = n.summarise(rows)
assert len(s) == 4, f"expected 4 model/split/version rows, got {len(s)}"
r = next(x for x in s if x["split"] == "holdout" and x["version"] == "baseline")
assert r["n_seeds"] == 3
assert r["acc_mean"] == round(float(np.mean([0.90, 0.94, 0.92])), 4)
assert r["acc_std"] == round(float(np.std([0.90, 0.94, 0.92], ddof=0)), 4)
assert r["detect_mean"] == round(float(np.mean([0.80, 0.90, 0.85])), 4)
assert r["detect_std"] == round(float(np.std([0.80, 0.90, 0.85], ddof=0)), 4)

print("OK: scores never rise, fixed-seed runs repeat, iters=0 is a no-op, summary maths checked")
