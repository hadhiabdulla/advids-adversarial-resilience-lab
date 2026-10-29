# AdvIDS - An Adversarial Resilience Lab

Stress-tests machine-learning intrusion detectors and measures whether adversarial training helps. Two detectors (Random Forest and a small neural network) are trained on the NSL-KDD connection records, attacked with a constrained black-box evasion search, hardened with adversarial training, and attacked again. Every number in this repo comes from running the code.

## What this is, and is not

* It is a **feature-space stress test**. It edits numbers in NSL-KDD records. It does **not** craft packets, and an edited record is not shown to be a working attack on a real network.
* The attacker may change only 7 features: `duration`, `src_bytes`, `count`, `srv_count`, `same_srv_rate`, `diff_srv_rate`, `dst_host_count`. Choosing them as "shapeable" (by pacing, padding or spreading connections) is an **assumption**, not a measured fact. Everything else stays fixed. Changes stay inside the 1st-99th percentile range of the training data (or the record's own value if outside it), integer features stay integers, rates stay in [0, 1], `same_srv_rate + diff_srv_rate <= 1`, `srv_count <= count`. `tests/test_constraints.py` checks all of this.
* The attack is a simple random search against a model's output score (30 iterations). It is one attack family, not a worst-case bound.

## Run it

```bash
git clone https://github.com/hadhiabdulla/advids-adversarial-resilience-lab.git
cd advids-adversarial-resilience-lab
pip install -r requirements.txt
python tests/test_constraints.py          # constraint checks (downloads the data on first run)
python nids_robustness.py                 # full experiment: 5 seeds x 2 models
```

CPU only, no GPU. Measured on a small shared Linux sandbox: about 1 minute per Random Forest seed and 1.3 minutes per neural-network seed, roughly 8 minutes for the full run. Needs numpy, pandas, scikit-learn and matplotlib (versions used: scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, Python 3.10). Fewer seeds: `python nids_robustness.py --seeds 1`.

## Protocol (fixed before the final run)

* KDDTrain+ is split per seed into 60% train, 20% validation and 20% holdout (stratified). Models and the defence are fitted on **train only**; nothing was tuned on the holdout.
* Defence settings were fixed in advance, not tuned: 2 rounds of adversarial training. Each round attacks the current model's training-set attack records and adds the perturbed copies, labelled as attacks, to the training data.
* Seeds 0-4; tables show mean +/- standard deviation.
* **Exploratory benchmark:** before the protocol was frozen, a quick feasibility run looked at Random Forest results on the official KDDTest+ file. KDDTest+ is therefore reported as an exploratory benchmark, not an untouched test set.
* **Post-hoc addition:** the "stronger attacker" column (larger steps, 4x the search iterations) was added after seeing the first results. It is labelled post-hoc and is not part of the original protocol.
* The holdout comes from the same distribution as the training data, so near-perfect clean accuracy there is expected. KDDTest+ has a known distribution shift from training data, which is why clean accuracy there is much lower.

## Results

20% holdout of KDDTrain+ (5 seeds). "Detect" is the share of attack records flagged. The attacker re-targets each model, including the hardened one.

| Model | Version | Accuracy | Detect | Under attack | Under stronger attack (post-hoc) | False positives |
|---|---|---|---|---|---|---|
| RF | baseline | 99.9 +/- 0.0% | 99.8 +/- 0.0% | 36.8 +/- 10.1% | 15.4 +/- 6.0% | 0.1 +/- 0.0% |
| RF | hardened | 99.9 +/- 0.0% | 99.8 +/- 0.0% | 99.8 +/- 0.0% | 99.8 +/- 0.0% | 0.1 +/- 0.0% |
| MLP | baseline | 99.6 +/- 0.1% | 99.6 +/- 0.2% | 87.2 +/- 5.3% | 60.4 +/- 24.8% | 0.4 +/- 0.1% |
| MLP | hardened | 99.5 +/- 0.1% | 99.7 +/- 0.1% | 99.5 +/- 0.1% | 98.3 +/- 0.3% | 0.6 +/- 0.2% |

KDDTest+ (exploratory, 5 seeds):

| Model | Version | Accuracy | Detect | Under attack | Under stronger attack (post-hoc) | False positives |
|---|---|---|---|---|---|---|
| RF | baseline | 76.9 +/- 0.7% | 61.5 +/- 1.3% | 29.5 +/- 6.4% | 14.8 +/- 6.6% | 2.8 +/- 0.0% |
| RF | hardened | 80.0 +/- 0.8% | 67.2 +/- 1.3% | 64.0 +/- 1.9% | 60.1 +/- 1.2% | 3.1 +/- 0.0% |
| MLP | baseline | 78.5 +/- 1.3% | 64.2 +/- 2.4% | 48.5 +/- 5.0% | 30.8 +/- 13.6% | 2.7 +/- 0.1% |
| MLP | hardened | 81.5 +/- 1.6% | 70.6 +/- 2.3% | 66.4 +/- 3.4% | 59.5 +/- 3.1% | 4.0 +/- 1.8% |

What to take from it:

* The baseline detectors lose a large share of their detections under the constrained attack, the Random Forest much more than the neural network.
* Adversarial training recovers almost all of it on the holdout. That is the best case: the defence is trained against the same attack family used to test it, on data from the same distribution. The harder KDDTest+ shows a partial recovery only (about 60% detection under the stronger attack, against 15-31% before), with a small rise in false positives for the hardened neural network (2.7% to 4.0%).
* Nothing here shows robustness against other attack types, a smarter adversary, or real traffic.

Raw per-seed numbers: `results/raw_results.json`, summary: `results/summary.json`, tables: `results/results.md`.

## Data and provenance

NSL-KDD (Tavallaee, Bagheri, Lu and Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set", 2009), an improved version of KDD Cup 99. The original University of New Brunswick page (https://www.unb.ca/cic/datasets/nsl.html) now says the dataset is no longer available, and the licence terms could not be confirmed from it. For that reason **the data is not included in this repo**. `nids_robustness.py` downloads the two files on first run from a public GitHub mirror (github.com/defcom17/NSL_KDD) into `data/`. Please cite the paper if you use the dataset, and check its terms yourself.

## Files

`nids_robustness.py` data loading, attack, defence, evaluation, report and plot. `tests/test_constraints.py` checks the attack only makes allowed changes. `results/` outputs of the full run and its log.
