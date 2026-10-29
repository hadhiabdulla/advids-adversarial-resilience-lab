"""Checks that the stress test only makes allowed changes. Run: python tests/test_constraints.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nids_robustness as n

Xtr, ytr, Xte, yte = n.load()
lo, hi = n.bounds_from(Xtr)
model = n.make_model("rf", 0).fit(Xtr.iloc[:20000], ytr[:20000])
att = Xte[yte == 1].iloc[:500]
adv = n.evade(model, att, lo, hi, np.random.default_rng(0), iters=10)
orig = att.reset_index(drop=True)

frozen = [c for c in orig.columns if c not in n.SHAPEABLE]
assert (orig[frozen].values == adv[frozen].values).all(), "a frozen feature changed"
for f in n.SHAPEABLE:
    o = orig[f].values
    assert (adv[f].values >= np.minimum(lo[f], o) - 1e-9).all() and (adv[f].values <= np.maximum(hi[f], o) + 1e-9).all(), f"{f} out of bounds"
for f in n.INT_FEATS:
    changed = adv[f].values != orig[f].values
    assert (adv[f][changed] == np.rint(adv[f][changed])).all(), f"{f} not integer"
for f in n.RATE_FEATS:
    assert adv[f].between(0, 1).all(), f"{f} outside [0,1]"
ok = (orig.same_srv_rate + orig.diff_srv_rate) <= 1 + 1e-9
assert ((adv.same_srv_rate + adv.diff_srv_rate) <= 1 + 1e-9)[ok].all(), "rates sum above 1"
ok = orig["srv_count"] <= orig["count"]
assert (adv["srv_count"] <= adv["count"])[ok].all(), "srv_count above count"
assert len(adv) == len(orig)
# the do-nothing candidate must leave records unchanged
same = n.project(orig, orig, lo, hi)
assert (same.values == orig.values).all(), "projection altered an unperturbed record"
print("OK: frozen features untouched, bounds, integer and rate constraints hold on", len(adv), "records")
