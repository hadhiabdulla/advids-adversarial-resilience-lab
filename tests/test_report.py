"""Checks the report table and the robustness figure, without the dataset.

Uses a small synthetic summary, so it runs anywhere in seconds - no download,
no model training. Run: python tests/test_report.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nids_robustness as n


def fake_summary():
    def row(model, split, version, detect, under):
        return {"model": model, "split": split, "version": version, "n_seeds": 2,
                "acc_mean": 0.9, "acc_std": 0.01, "detect_mean": detect, "detect_std": 0.01,
                "fpr_mean": 0.02, "fpr_std": 0.005,
                "detect_under_attack_mean": under, "detect_under_attack_std": 0.03,
                "detect_under_strong_attack_mean": under / 2, "detect_under_strong_attack_std": 0.02}
    return [row(m, s, v, d, u)
            for m in ("rf", "mlp") for s in ("holdout", "kddtest_exploratory")
            for v, d, u in (("baseline", 0.95, 0.40), ("hardened", 0.96, 0.90))]


S = fake_summary()
tmp = tempfile.mkdtemp()

# 1. the markdown report covers both splits, every model/version row, and the post-hoc label
md = os.path.join(tmp, "results.md")
n.write_report(S, md)
text = open(md).read()
assert "Holdout (20% of KDDTrain+" in text, "holdout section missing"
assert "KDDTest+ (exploratory benchmark, see README)" in text, "exploratory section missing"
assert text.count("post-hoc") >= 2, "stronger-attack column must stay labelled post-hoc"
for line in text.splitlines():
    if line.startswith("| rf | baseline |"):
        cells = [c.strip() for c in line.split("|")]
        assert cells[5] == "40.0 +/- 3.0%", f"under-attack cell wrong: {cells[5]}"
        assert cells[6] == "20.0 +/- 2.0%", f"stronger-attack cell wrong: {cells[6]}"
        break
else:
    raise AssertionError("rf/baseline row missing")
assert sum(1 for l in text.splitlines() if l.startswith("| ")) == 2 * (1 + 4), \
    "each split should have a header row and 4 model rows"

# 2. the figure: one PNG with both splits, from summary data alone
png = os.path.join(tmp, "robustness.png")
n.plot(S, png)
data = open(png, "rb").read()
assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000, "plot did not produce a real PNG"
# every summary row must belong to a split the plot knows about
known = {"holdout", "kddtest_exploratory"}
assert all(r["split"] in known for r in S)

print("OK: report tables and figure checked (no dataset needed)")
