"""Score the COCO full runs from the logged samples, with and without the label audit.

Usage (inside WSL, conda env cmpe297, from the project folder):
    python src/score_coco.py

COCO-only version of score.py: same metrics, bootstrap and audit views, applied to the COCO
set alone, with tables written to results/tables/coco/. The numbers are identical to the
COCO rows of score.py (same bootstrap stream: COCO is stream 0 there too).

Predictions are re-parsed from the raw responses (results_io), joined on question_id to the
frozen set, which must match its MANIFEST.json entry. Audit outcomes are joined at scoring
time; the frozen file is never edited. Three label views (PROJECT_PLAN.md 6.4):

- (a) full_original: all 9,000 questions with the original labels. Primary view.
- (b) audit_corrected: only the questions whose (image, object) is in the Stage A sample, with
  the reviewer's verdict as the label (present -> yes, absent -> no). Uncertain items are
  excluded and counted in n_uncertain_excluded. audit_original scores the same subset with
  the original labels, so the effect of the corrections is separated from the subset itself.
- (c) label quality: Stage A verdicts against the original labels, and the share of reviewed
  Stage B ordinary-prompt false positives (and false negatives) tagged "label error".

Split "all" pools the splits. In view (a) it is all 9,000 rows; in view (b) it is the distinct
audited questions (a question occurs in several splits with the same response, checked here).

Uncertainty (PROJECT_PLAN.md 7.1): a bootstrap over images (--bootstrap resamples, --seed),
since the questions about one image are not independent. One set of resamples is shared by
every prompt and split. It gives percentile 95% CIs on the headline metrics (accuracy, F1,
hallucination rate, false-negative rate) in view (a), and CIs on the hallucination-rate
differences random vs popular and random vs adversarial under the ordinary prompt, each
resample computing both splits on the same images. A difference whose interval excludes 0 is
called significant. Prompts are compared descriptively, not tested.

Outputs in results/tables/coco/: metrics_long.csv (one row per prompt x split x label view;
CI columns filled for view (a)), metrics_full_original.md, metrics_headline_ci.md,
split_comparison.csv/.md, metrics_audited.md, label_quality.md.
"""
import argparse
import csv
import sys
from collections import Counter

import numpy as np

from freeze_eval_set_coco import verify
from pope_sampling import SPLITS
from results_io import PROJECT, PROMPT_KINDS, load_predictions, load_questions

DS = "coco"
AUDIT_DIR = PROJECT / "data/audit"
TABLES = PROJECT / "results/tables/coco"
VERDICT_LABEL = {"present": "yes", "absent": "no"}
LABEL_ERROR_SHARE_NOTE = 0.05  # above this share of audited negatives judged present, the report says so
COUNT_KEYS = ["n", "n_pos", "n_neg", "tp", "fp", "tn", "fn", "abstain_pos", "abstain_neg", "invalid"]
RATE_KEYS = ["accuracy", "precision", "recall", "f1", "specificity", "hallucination_rate", "fn_rate",
             "yes_ratio", "abstention_rate", "abstention_rate_pos", "abstention_rate_neg", "class_balance"]
HEADLINE = ["accuracy", "f1", "hallucination_rate", "fn_rate"]
CI_KEYS = [f"{m}_ci_{end}" for m in HEADLINE for end in ("lo", "hi")]
COLUMNS = (["dataset", "prompt", "split", "label_view"] + COUNT_KEYS + ["n_uncertain_excluded"] + RATE_KEYS
           + CI_KEYS)
SPLIT_COMPARISONS = [("random", "popular"), ("random", "adversarial")]
COMPARISON_PROMPT = "ordinary"
BOOTSTRAP_STREAM = 0  # the COCO stream index score.py uses, so both scripts draw the same resamples


def ratio(a, b):
    return a / b if b else float("nan")


def metrics(pairs) -> dict:
    """pairs: (label, prediction) with label in {yes, no}, prediction in {yes, no, abstain, invalid}.

    Abstentions and invalid parses are never correct; both count in the abstention rates,
    and invalid is also reported on its own.
    """
    c = Counter()
    for label, pred in pairs:
        c[label, "abstain" if pred == "invalid" else pred] += 1
        c["invalid"] += pred == "invalid"
    tp, fn, ab_pos = c["yes", "yes"], c["yes", "no"], c["yes", "abstain"]
    fp, tn, ab_neg = c["no", "yes"], c["no", "no"], c["no", "abstain"]
    n_pos, n_neg = tp + fn + ab_pos, fp + tn + ab_neg
    n = n_pos + n_neg
    return {
        "n": n, "n_pos": n_pos, "n_neg": n_neg, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "abstain_pos": ab_pos, "abstain_neg": ab_neg, "invalid": c["invalid"],
        "accuracy": ratio(tp + tn, n),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, n_pos),
        "f1": ratio(2 * tp, tp + fp + n_pos),  # harmonic mean of precision and recall
        "specificity": ratio(tn, n_neg),
        "hallucination_rate": ratio(fp, n_neg),
        "fn_rate": ratio(fn, n_pos),
        "yes_ratio": ratio(tp + fp, n),
        "abstention_rate": ratio(ab_pos + ab_neg, n),
        "abstention_rate_pos": ratio(ab_pos, n_pos),
        "abstention_rate_neg": ratio(ab_neg, n_neg),
        "class_balance": ratio(n_pos, n),
    }


# ---- audit inputs ---------------------------------------------------------------------------
def read_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_verdicts(list_csv, labels_glob: str) -> list[dict]:
    """Rows of an audit list, each with the `verdict` and `tag` from the reviewer label files.

    Reviewers split the work by list, so the labels for one list may sit in several files
    (one per reviewer); every item must have exactly one verdict.
    """
    items = read_csv(list_csv)
    labels = {}
    for path in sorted(AUDIT_DIR.glob(labels_glob)):
        for r in read_csv(path):
            if r["audit_id"] in labels:
                sys.exit(f"{r['audit_id']} has more than one verdict ({path.name})")
            labels[r["audit_id"]] = r
    missing = [it["audit_id"] for it in items if it["audit_id"] not in labels]
    if missing:
        sys.exit(f"{list_csv.name}: {len(missing)} item(s) not labelled yet, e.g. {missing[:5]}")
    for it in items:
        lab = labels[it["audit_id"]]
        if (lab["image"], lab["object"]) != (it["image"], it["object"]):
            sys.exit(f"{it['audit_id']}: labels file and {list_csv.name} disagree on image/object")
        it["verdict"], it["tag"] = lab["verdict"], lab["tag"]
    return items


def stage_a() -> list[dict]:
    return load_verdicts(AUDIT_DIR / f"audit_sample_{DS}.csv", f"labels_{DS}_*.csv")


LABEL_ERROR_VERDICT = {"fp": "present", "fn": "absent"}  # the verdict that means the dataset label is wrong


def stage_b(kind: str) -> list[dict]:
    """Stage B items; exits if a "label error" tag and the verdict disagree."""
    items = load_verdicts(AUDIT_DIR / f"audit_{kind}_{DS}.csv", f"labels_{kind}_{DS}_*.csv")
    bad = [it["audit_id"] for it in items
           if (it["tag"] == "label error") != (it["verdict"] == LABEL_ERROR_VERDICT[kind])]
    if bad:
        sys.exit(f"Stage B {kind}: tag and verdict disagree on label error for {bad[:5]}; re-review them")
    return items


# ---- bootstrap over images ------------------------------------------------------------------
def resample_weights(n_images: int, n_boot: int, seed: int, stream: int = BOOTSTRAP_STREAM) -> np.ndarray:
    """(n_boot, n_images): how many times each image is drawn in each resample."""
    rng = np.random.default_rng([seed, stream])
    draws = rng.integers(0, n_images, size=(n_boot, n_images))
    return np.stack([np.bincount(d, minlength=n_images) for d in draws])


def image_counts(qs, preds, images: list[str]) -> np.ndarray:
    """(n_images, 6) per-image counts: tp, fp, tn, fn, n_pos, n_neg."""
    index = {im: i for i, im in enumerate(images)}
    c = np.zeros((len(images), 6))
    for q in qs:
        i, pos, pred = index[q["image"]], q["answer"] == "yes", preds[q["question_id"]]["prediction"]
        c[i, 4 if pos else 5] += 1
        if pred == "yes":
            c[i, 0 if pos else 1] += 1
        elif pred == "no":
            c[i, 3 if pos else 2] += 1
    return c


def headline(totals: np.ndarray) -> dict:
    """Headline metrics from summed counts (last axis as in image_counts); same formulas as metrics()."""
    tp, fp, tn, fn, n_pos, n_neg = np.moveaxis(totals, -1, 0)
    return {"accuracy": (tp + tn) / (n_pos + n_neg), "f1": 2 * tp / (tp + fp + n_pos),
            "hallucination_rate": fp / n_neg, "fn_rate": fn / n_pos}


def ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def split_comparisons(questions, preds, images, weights) -> list[dict]:
    """Hallucination-rate difference (second split minus first) with its bootstrap CI."""
    rate = {}
    for split in SPLITS:
        c = image_counts([q for q in questions if q["category"] == split], preds, images)
        rate[split] = (headline(c.sum(0))["hallucination_rate"], headline(weights @ c)["hallucination_rate"])
    out = []
    for a, b in SPLIT_COMPARISONS:
        lo, hi = ci(rate[b][1] - rate[a][1])
        out.append({"dataset": DS, "prompt": COMPARISON_PROMPT, "metric": "hallucination_rate",
                    "split_a": a, "split_b": b, "rate_a": rate[a][0], "rate_b": rate[b][0],
                    "difference": rate[b][0] - rate[a][0], "ci_lo": lo, "ci_hi": hi,
                    "significant": int(lo > 0 or hi < 0)})
    return out


# ---- views ------------------------------------------------------------------------------------
def score_rows(questions, preds, prompt: str, audit, images, weights) -> list[dict]:
    """metrics_long rows for one prompt: view (a), and both label sets of view (b)."""
    out = []

    def add(split, view, pairs, n_uncertain=""):
        out.append({"dataset": DS, "prompt": prompt, "split": split, "label_view": view,
                    **metrics(pairs), "n_uncertain_excluded": n_uncertain})

    for split in SPLITS + ("all",):
        qs = [q for q in questions if split in ("all", q["category"])]
        add(split, "full_original", [(q["answer"], preds[q["question_id"]]["prediction"]) for q in qs])
        boot = headline(weights @ image_counts(qs, preds, images))
        for m in HEADLINE:
            out[-1][f"{m}_ci_lo"], out[-1][f"{m}_ci_hi"] = ci(boot[m])

    verdict = {(it["image"], it["object"]): it["verdict"] for it in audit}
    for split in SPLITS + ("all",):
        qs = [q for q in questions if (q["image"], q["object"]) in verdict and split in ("all", q["category"])]
        if split == "all":  # one item per audited question: its copies across splits share a response
            first = {}
            for q in qs:
                key = q["image"], q["object"]
                if key in first and preds[q["question_id"]]["response"] != preds[first[key]["question_id"]]["response"]:
                    sys.exit(f"{prompt} {key}: copies across splits got different responses")
                first.setdefault(key, q)
            qs = list(first.values())
        kept = [q for q in qs if verdict[q["image"], q["object"]] in VERDICT_LABEL]
        n_unc = len(qs) - len(kept)
        add(split, "audit_original", [(q["answer"], preds[q["question_id"]]["prediction"]) for q in kept], n_unc)
        add(split, "audit_corrected", [(VERDICT_LABEL[verdict[q["image"], q["object"]]],
                                        preds[q["question_id"]]["prediction"]) for q in kept], n_unc)
    return out


def label_quality(audit: list[dict]) -> dict:
    """View (c)."""
    neg = Counter(it["verdict"] for it in audit if it["answer"] == "no")
    pos = Counter(it["verdict"] for it in audit if it["answer"] == "yes")
    q = {"neg": neg, "n_neg": sum(neg.values()), "pos": pos, "n_pos": sum(pos.values())}
    for kind in ("fp", "fn"):
        items = stage_b(kind)
        q[kind] = Counter(it["tag"] for it in items)
        q[f"n_{kind}"] = len(items)
    return q


# ---- output -------------------------------------------------------------------------------------
def pct(x) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


METRIC_HEAD = ["Acc", "Prec", "Rec", "F1", "Spec", "Halluc (FPR)", "FNR", "Yes ratio", "Abstain",
               "Abst. pos", "Abst. neg"]


def metric_cells(r: dict) -> list[str]:
    return [pct(r[k]) for k in ("accuracy", "precision", "recall", "f1", "specificity", "hallucination_rate",
                                "fn_rate", "yes_ratio", "abstention_rate", "abstention_rate_pos",
                                "abstention_rate_neg")]


def write_full_md(rows: list[dict]) -> str:
    body = [[r["prompt"], r["split"], *metric_cells(r)] for r in rows if r["label_view"] == "full_original"]
    return ("# View (a): full set, original labels\n"
            "All metrics in %. N = 3,000 per split (1,500 yes / 1,500 no, class balance 50%); "
            "\"all\" pools the three splits (N = 9,000). Abstain counts abstentions plus invalid parses "
            "(invalid parses are listed in metrics_long.csv).\n\n"
            + md_table(["Prompt", "Split", *METRIC_HEAD], body))


def with_ci(r: dict, m: str) -> str:
    return f"{pct(r[m])} [{pct(r[f'{m}_ci_lo'])}, {pct(r[f'{m}_ci_hi'])}]"


def write_ci_md(rows: list[dict], n_boot: int, seed: int) -> str:
    body = [[r["prompt"], r["split"], *(with_ci(r, m) for m in HEADLINE)] for r in rows
            if r["label_view"] == "full_original"]
    return ("# Headline metrics with 95% CIs (view a)\n"
            f"Full set, original labels. Point estimate [95% CI] in %, percentile bootstrap over images "
            f"({n_boot:,} resamples, seed {seed}).\n\n"
            + md_table(["Prompt", "Split", "Accuracy", "F1", "Halluc (FPR)", "FNR"], body))


def write_comparison_md(comparisons: list[dict], n_boot: int, seed: int) -> str:
    body = [[f"{c['split_a']} vs {c['split_b']}", pct(c["rate_a"]), pct(c["rate_b"]),
             f"{100 * c['difference']:+.1f}", f"[{100 * c['ci_lo']:+.1f}, {100 * c['ci_hi']:+.1f}]",
             "yes" if c["significant"] else "no"] for c in comparisons]
    return ("# Split comparisons: hallucination rate, ordinary prompt\n"
            f"Difference = second split minus first, in percentage points. 95% CI from a paired bootstrap over "
            f"images ({n_boot:,} resamples, seed {seed}): each resample scores both splits on the same images. "
            "Significant = the interval excludes 0.\n\n"
            + md_table(["Comparison", "FPR first (%)", "FPR second (%)", "Difference (pp)", "95% CI (pp)",
                        "Significant"], body))


def write_audited_md(rows: list[dict]) -> str:
    body = [[r["prompt"], r["label_view"].removeprefix("audit_"), r["n"], r["n_pos"], r["n_neg"],
             r["n_uncertain_excluded"], *metric_cells(r)] for r in rows
            if r["split"] == "all" and r["label_view"].startswith("audit_")]
    return ("# View (b): Stage A audited subset\n"
            "The 300 audited questions (150 drawn as positives, 150 as negatives), each counted once. "
            "*original* scores them with the dataset labels, *corrected* with the reviewer's verdicts; items "
            "judged uncertain are excluded from both and counted in \"Uncertain\". All metrics in %. Per-split "
            "rows are in metrics_long.csv.\n\n"
            + md_table(["Prompt", "Labels", "N", "N pos", "N neg", "Uncertain", *METRIC_HEAD], body))


def write_quality_md(q: dict) -> str:
    def share(c, k, n):
        return f"{c[k]} ({pct(ratio(c[k], n))}%)"

    out = ["# View (c): label-quality summary\n",
           "Stage A: blind single-reviewer verdicts on the predeclared audit sample. Stage B: "
           "reviewed ordinary-prompt errors (not blind: reviewers know each item is a model error).\n\n"]
    out.append(md_table(["Items", "Judged present", "Judged absent", "Judged uncertain"], [
        [f"Stage A negatives (dataset says *no*), n={q['n_neg']}", share(q["neg"], "present", q["n_neg"]),
         share(q["neg"], "absent", q["n_neg"]), share(q["neg"], "uncertain", q["n_neg"])],
        [f"Stage A positives (dataset says *yes*), n={q['n_pos']}", share(q["pos"], "present", q["n_pos"]),
         share(q["pos"], "absent", q["n_pos"]), share(q["pos"], "uncertain", q["n_pos"])],
    ]))
    out.append("\n" + md_table(["Stage B list", "Reviewed", "Tagged *label error*"], [
        ["Ordinary-prompt false positives", q["n_fp"], share(q["fp"], "label error", q["n_fp"])],
        ["Ordinary-prompt false negatives", q["n_fn"], share(q["fn"], "label error", q["n_fn"])],
    ]))
    out.append("\n" + md_table(["Stage B false-positive tag", "Count"], [[t, n] for t, n in q["fp"].most_common()]))
    out.append("\n" + md_table(["Stage B false-negative tag", "Count"], [[t, n] for t, n in q["fn"].most_common()]))
    frac = ratio(q["neg"]["present"], q["n_neg"])
    if frac > LABEL_ERROR_SHARE_NOTE:
        out.append(f"\n**Note:** {pct(frac)}% of audited negatives were judged present (above "
                   f"{pct(LABEL_ERROR_SHARE_NOTE)}%), so part of the measured hallucination rate is label error; "
                   "see view (b).\n")
    return "".join(out)


def load_checked(prompt, questions) -> dict:
    """Predictions for one task; exits unless there is exactly one response per question."""
    preds = load_predictions(DS, prompt)
    missing = [q["question_id"] for q in questions if q["question_id"] not in preds]
    extra = set(preds) - {q["question_id"] for q in questions}
    if missing or extra:
        sys.exit(f"{prompt}: {len(missing)} question(s) without a response, {len(extra)} unknown id(s)")
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples over images")
    ap.add_argument("--seed", type=int, default=297)
    args = ap.parse_args()

    problems = verify()
    if problems:
        sys.exit("frozen set does not match MANIFEST.json:\n  " + "\n  ".join(problems))
    questions = load_questions(DS)
    images = sorted({q["image"] for q in questions})
    weights = resample_weights(len(images), args.bootstrap, args.seed)
    audit = stage_a()
    rows, comparisons = [], []
    for prompt in PROMPT_KINDS:
        preds = load_checked(prompt, questions)
        rows += score_rows(questions, preds, prompt, audit, images, weights)
        if prompt == COMPARISON_PROMPT:
            comparisons += split_comparisons(questions, preds, images, weights)
    quality = label_quality(audit)

    def fmt(r):
        return {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}

    TABLES.mkdir(parents=True, exist_ok=True)
    for name, cols, data in (("metrics_long.csv", COLUMNS, rows),
                             ("split_comparison.csv", list(comparisons[0]), comparisons)):
        with open(TABLES / name, "w", newline="") as f:
            writer = csv.DictWriter(f, cols, restval="", lineterminator="\n")
            writer.writeheader()
            writer.writerows(fmt(r) for r in data)
    markdown = {"metrics_full_original.md": write_full_md(rows),
                "metrics_headline_ci.md": write_ci_md(rows, args.bootstrap, args.seed),
                "split_comparison.md": write_comparison_md(comparisons, args.bootstrap, args.seed),
                "metrics_audited.md": write_audited_md(rows),
                "label_quality.md": write_quality_md(quality)}
    for name, text in markdown.items():
        (TABLES / name).write_text(text, newline="\n", encoding="utf-8")
    print(f"wrote results/tables/coco/: metrics_long.csv ({len(rows)} rows), split_comparison.csv, {', '.join(markdown)}")


if __name__ == "__main__":
    main()
