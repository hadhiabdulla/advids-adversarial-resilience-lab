"""
AdvIDS - An Adversarial Resilience Lab
(robustness stress-testing of ML network intrusion detection on NSL-KDD).

What this does
  1. Trains two detectors (Random Forest, MLP) on NSL-KDD connection records.
  2. Stress-tests them with a black-box, feature-space evasion search that may
     only change a small set of "shapeable" features, within fixed bounds.
  3. Hardens each model with adversarial training and re-tests it, attacking
     the hardened model too (the attacker adapts).

IMPORTANT - what this is NOT
  This is a FEATURE-SPACE stress test. It changes numbers in the NSL-KDD
  records. It does not generate packets, and a changed record is not proven
  to be a real, working attack on a real network. The list of shapeable
  features is an ASSUMPTION (see SHAPEABLE below).

Frozen protocol (written before the final run)
  * Data: official KDDTrain+ / KDDTest+.
  * KDDTrain+ is split per seed into 60% train / 20% validation / 20% holdout
    (stratified). Models and the defence are fitted on train only. The
    validation split is used only for monitoring. The holdout is evaluated once
    per seed, after everything is fixed. No hyper-parameter was tuned on it.
  * KDDTest+ is reported as a SECONDARY, EXPLORATORY benchmark: a quick
    feasibility run looked at KDDTest+ results before this protocol was frozen,
    so it is not an untouched test set.
  * Defence settings (fixed in advance, not tuned): 2 rounds of adversarial
    training, adversarial copies of train-split attack records labelled attack.
  * Seeds 0-4. We report mean and standard deviation. Whatever the defence
    does (helps, partly helps, costs clean accuracy) is reported as is.
"""
import argparse, json, os, sys, urllib.request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]
COLS = ("duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,"
        "num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,"
        "num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,"
        "is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,"
        "same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,"
        "dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,"
        "dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,"
        "dst_host_rerror_rate,dst_host_srv_rerror_rate,label,difficulty").split(",")
CAT = ["protocol_type", "service", "flag"]

# ASSUMPTION: features an attacker could plausibly shape by pacing, padding or
# spreading connections. Count-style features are integers, rate features are
# in [0, 1]. Everything else (protocol, service, flag, content features such as
# num_failed_logins, root_shell ...) is frozen because it is tied to what the
# attack actually does.
SHAPEABLE = ["duration", "src_bytes", "count", "srv_count",
             "same_srv_rate", "diff_srv_rate", "dst_host_count"]
INT_FEATS = {"duration", "src_bytes", "count", "srv_count", "dst_host_count"}
RATE_FEATS = {"same_srv_rate", "diff_srv_rate"}


def fetch_data(d="data"):
    os.makedirs(d, exist_ok=True)
    for f in FILES:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print("downloading", f)
            urllib.request.urlretrieve(BASE + f.replace("+", "%2B"), p)
    return [os.path.join(d, f) for f in FILES]


def load(d="data"):
    ptr, pte = fetch_data(d)
    tr = pd.read_csv(ptr, names=COLS)
    te = pd.read_csv(pte, names=COLS)
    for df in (tr, te):
        df["y"] = (df.label != "normal").astype(int)
    allv = pd.concat([tr, te])
    enc = {c: {v: i for i, v in enumerate(sorted(allv[c].unique()))} for c in CAT}

    def feats(df):
        x = df.drop(columns=["label", "difficulty", "y"]).copy()
        for c in CAT:
            x[c] = x[c].map(enc[c])
        return x.astype(float)
    return feats(tr), tr.y.values, feats(te), te.y.values


# ---------------------------------------------------------------- attack
def bounds_from(Xtrain):
    lo = Xtrain[SHAPEABLE].quantile(0.01)
    hi = Xtrain[SHAPEABLE].quantile(0.99)
    for f in INT_FEATS:               # integer features get integer bounds
        lo[f], hi[f] = np.ceil(lo[f]), np.floor(hi[f])
    return lo, hi


def project(orig, cand, lo, hi):
    """Force a candidate to obey the constraints relative to the original.
    Each feature may move inside [min(lo, original), max(hi, original)], so the
    unmodified record is always allowed and unperturbed rows are never altered."""
    out = orig.copy()
    for f in SHAPEABLE:
        o = orig[f].values
        v = np.clip(cand[f].values, np.minimum(lo[f], o), np.maximum(hi[f], o))
        if f in INT_FEATS:
            v = np.rint(v)
        if f in RATE_FEATS:
            v = np.clip(v, 0.0, 1.0)
        out[f] = v
    # same_srv_rate + diff_srv_rate cannot exceed 1 (only enforced when the original obeys it)
    ok = (orig["same_srv_rate"].values + orig["diff_srv_rate"].values) <= 1.0 + 1e-9
    over = ((out["same_srv_rate"].values + out["diff_srv_rate"].values) > 1.0 + 1e-9) & ok
    out.loc[over, "diff_srv_rate"] = 1.0 - out.loc[over, "same_srv_rate"]
    # srv_count cannot exceed count (only enforced when the original obeys it)
    bad = (out["srv_count"] > out["count"]) & (orig["srv_count"] <= orig["count"])
    out.loc[bad, "srv_count"] = out.loc[bad, "count"]
    return out


def evade(model, X_att, lo, hi, rng, iters=30, sigma=0.3):
    """Black-box random search: keep a perturbation if the attack score drops."""
    orig = X_att.reset_index(drop=True)
    adv = project(orig, orig, lo, hi)
    best = model.predict_proba(adv)[:, 1]
    for _ in range(iters):
        cand = orig.copy()
        for f in SHAPEABLE:
            cand[f] = orig[f].values + rng.normal(0, sigma, len(orig)) * (hi[f] - lo[f])
        cand = project(orig, cand, lo, hi)
        p = model.predict_proba(cand)[:, 1]
        m = p < best
        adv.loc[m] = cand.loc[m]
        best = np.where(m, p, best)
    return adv


# ---------------------------------------------------------------- models
def make_model(kind, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    return make_pipeline(StandardScaler(),
                         MLPClassifier((64, 32), max_iter=60, random_state=seed))


def metrics(model, X, y):
    p = model.predict(X)
    return {"acc": float((p == y).mean()),
            "detect": float(p[y == 1].mean()),
            "fpr": float(p[y == 0].mean())}


def run_seed(kind, seed, Xtr_full, ytr_full, Xte, yte, iters, rounds=2):
    rng = np.random.default_rng(seed)
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        Xtr_full, ytr_full, test_size=0.4, random_state=seed, stratify=ytr_full)
    X_val, X_ho, y_val, y_ho = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
    lo, hi = bounds_from(X_tr)

    base = make_model(kind, seed).fit(X_tr, y_tr)
    res = {"seed": seed, "model": kind, "val_clean_acc": float(base.score(X_val, y_val))}

    # defence: adversarial training, built from TRAIN split only
    Xd, yd = X_tr, y_tr
    cur = base
    for _ in range(rounds):
        att = X_tr[y_tr == 1]
        adv = evade(cur, att, lo, hi, rng, iters)
        Xd = pd.concat([Xd.reset_index(drop=True), adv], ignore_index=True)
        yd = np.concatenate([np.asarray(yd), np.ones(len(adv), dtype=int)])
        cur = make_model(kind, seed).fit(Xd, yd)
    hard = cur

    for name, (Xe, ye) in {"holdout": (X_ho, y_ho), "kddtest_exploratory": (Xte, yte)}.items():
        att_rows = Xe[ye == 1]
        for tag, mdl in (("baseline", base), ("hardened", hard)):
            m = metrics(mdl, Xe, ye)
            adv = evade(mdl, att_rows, lo, hi, np.random.default_rng(seed + 1000), iters)
            m["detect_under_attack"] = float(mdl.predict(adv).mean())
            # POST-HOC addition (decided after seeing the first results): a stronger
            # attacker with larger steps and more search iterations, same constraints.
            adv2 = evade(mdl, att_rows, lo, hi, np.random.default_rng(seed + 2000), iters * 4, sigma=0.6)
            m["detect_under_strong_attack"] = float(mdl.predict(adv2).mean())
            res[f"{name}_{tag}"] = m
    return res


def summarise(rows):
    df = pd.json_normalize(rows)
    out = []
    for kind in df.model.unique():
        d = df[df.model == kind]
        for split in ("holdout", "kddtest_exploratory"):
            for tag in ("baseline", "hardened"):
                r = {"model": kind, "split": split, "version": tag, "n_seeds": len(d)}
                for k in ("acc", "detect", "fpr", "detect_under_attack", "detect_under_strong_attack"):
                    c = d[f"{split}_{tag}.{k}"]
                    r[k + "_mean"] = round(float(c.mean()), 4)
                    r[k + "_std"] = round(float(c.std(ddof=0)), 4)
                out.append(r)
    return out


def write_report(summary, path):
    L = ["# Results (generated by nids_robustness.py)", "",
         "Values are mean +/- std over seeds. `detect` = share of attack records flagged,",
         "`detect_under_attack` = share still flagged after the feature-space evasion search",
         "(the attacker re-targets each model, including the hardened one), `fpr` = normal records flagged.", ""]
    for split, title in (("holdout", "Holdout (20% of KDDTrain+, evaluated once per seed)"),
                         ("kddtest_exploratory", "KDDTest+ (exploratory benchmark, see README)")):
        L += [f"## {title}", "",
              "| Model | Version | Accuracy | Detect | Detect under attack | Detect under stronger attack (post-hoc) | False-positive rate |",
              "|---|---|---|---|---|---|---|"]
        for r in summary:
            if r["split"] != split:
                continue
            f = lambda k: f"{100*r[k+'_mean']:.1f} +/- {100*r[k+'_std']:.1f}%"
            L.append(f"| {r['model']} | {r['version']} | {f('acc')} | {f('detect')} | {f('detect_under_attack')} | {f('detect_under_strong_attack')} | {f('fpr')} |")
        L.append("")
    open(path, "w").write("\n".join(L))


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary if r["split"] == "holdout"]
    labels = [f"{r['model'].upper()}\n{r['version']}" for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, [100*r["detect_mean"] for r in rows], w, label="Clean", color="#3b82f6")
    ax.bar(x + w/2, [100*r["detect_under_attack_mean"] for r in rows], w,
           yerr=[100*r["detect_under_attack_std"] for r in rows], label="Under feature-space attack", color="#ef4444")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Attack detection rate (%)")
    ax.set_title("Holdout: attack detection, clean vs under stress test"); ax.legend(); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30, help="evasion search iterations")
    ap.add_argument("--models", default="rf,mlp")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Xtr, ytr, Xte, yte = load()
    rows = []
    for kind in a.models.split(","):
        for s in range(a.seeds):
            print(f"model={kind} seed={s}", flush=True)
            rows.append(run_seed(kind, s, Xtr, ytr, Xte, yte, a.iters))
            json.dump(rows, open(os.path.join(a.out, "raw_results.json"), "w"), indent=1)
    summ = summarise(rows)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    write_report(summ, os.path.join(a.out, "results.md"))
    plot(summ, os.path.join(a.out, "robustness_holdout.png"))
    print(open(os.path.join(a.out, "results.md")).read())
