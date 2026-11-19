"""Effect studies and figures.

Usage (inside WSL, conda env cmpe297, from the project folder; run src/score.py first, whose
metrics_long.csv supplies the bootstrap CIs used here):
    python src/analyze.py

Unit: distinct (image, object) questions. A positive occurs in all three splits and some
negatives in two (popular + adversarial), always with the same response (checked here), so
"pooled" counts each question once. Per-split breakdowns use the questions in that split.
Full set, original labels (view a) throughout.

Tables (results/tables/, a .csv and a .md each unless noted):
- effect_size:      recall, false-negative and abstention rate of positives by size bin.
- effect_cooc:      hallucination rate of negatives by quartile of cooc_score (quartile edges
                    from the dataset's pooled negatives) and by freq_rank band, pooled and per split.
- prompt_flips:     questions whose answer changes between two prompts, with the full
                    answer-transition counts; the .md adds where ordinary's answers go under
                    the evidence prompt (does permission to abstain remove hallucinations or
                    correct answers?).
- top_objects:      top 10 most hallucinated and most missed objects (ordinary prompt).
- dataset_contrast.md: COCO vs GQA per prompt and split, with the CIs from score.py.

Figures (results/figures/): fig_fpr_by_split, fig_recall_by_size, fig_fpr_by_cooc,
fig_prompt_comparison, fig_examples_fp (genuine hallucinations from Stage B) and
fig_examples_label_error (Stage B false positives the reviewer found present). The example
ids are listed in EXAMPLES_FP / EXAMPLES_LABEL_ERROR.
"""
import argparse
import csv
import os
import sys
import textwrap
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from freeze_eval_set import verify  # noqa: E402
from pope_sampling import SPLITS, article, image_path  # noqa: E402
from results_io import DATASETS, PROJECT, PROMPT_KINDS, load_predictions, load_questions  # noqa: E402
from score import TABLES, md_table, metrics, pct, read_csv, stage_b  # noqa: E402

FIGURES = PROJECT / "results/figures"
DATA_ROOT = os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data"))
SIZE_BINS = ["small", "medium", "large"]
FREQ_BANDS = [(1, 5), (6, 20), (21, 50), (51, None)]
FLIP_PAIRS = [("ordinary", "neutral"), ("ordinary", "evidence"), ("neutral", "evidence")]
ANSWERS = ["yes", "no", "abstain"]
TOP_K = 10
TOP_PROMPT = "ordinary"
# Example figures: Stage B false-positive ids, hand-picked on 2026-11-18. They started as a
# seeded draw (one per tag first); borderline images were swapped for clear ones after review.
EXAMPLES_FP = ["coco_fp029", "coco_fp022", "coco_fp006", "coco_fp097",    # verdict absent:
               "gqa_fp099", "gqa_fp040", "gqa_fp097", "gqa_fp013"]       # genuine hallucinations
EXAMPLES_LABEL_ERROR = ["coco_fp028", "gqa_fp031", "gqa_fp071"]          # verdict present
TILE = (640, 480)  # example images are fitted into this canvas

# Chart tokens: the dataviz reference palette, light mode. Prompts take categorical slots 1-3
# (the slots validated all-pairs); abstention is neutral gray, not a series hue.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PROMPT_COLOR = {"ordinary": "#2a78d6", "neutral": "#eb6834", "evidence": "#1baf7a"}
OUTCOME_COLOR = {"correct": "#2a78d6", "wrong": "#eb6834", "abstain": AXIS}


# ---- data ------------------------------------------------------------------------------------
def distinct_questions(questions, preds) -> list[dict]:
    """One item per (image, object) question: the frozen row plus `splits`, and per prompt the
    raw `response` and parsed `pred` (invalid parses count as abstentions, as in score.py)."""
    items = {}
    for q in questions:
        key = q["image"], q["object"]
        resp = {p: preds[p][q["question_id"]]["response"] for p in PROMPT_KINDS}
        if key not in items:
            items[key] = {**q, "splits": [], "response": resp,
                          "pred": {p: preds[p][q["question_id"]]["prediction"].replace("invalid", "abstain")
                                   for p in PROMPT_KINDS}}
        elif items[key]["response"] != resp:
            sys.exit(f"{q['dataset']} {key}: copies across splits got different responses")
        items[key]["splits"].append(q["category"])
    return list(items.values())


def in_split(it, split) -> bool:
    return split == "pooled" or split in it["splits"]


# ---- effect studies ----------------------------------------------------------------------------
def size_effect(ds, items) -> list[dict]:
    rows = []
    for prompt in PROMPT_KINDS:
        for size in SIZE_BINS:
            m = metrics([("yes", it["pred"][prompt]) for it in items if it["answer"] == "yes" and it["size_bin"] == size])
            rows.append({"dataset": ds, "prompt": prompt, "size_bin": size, "n": m["n"], "recall": m["recall"],
                         "fn_rate": m["fn_rate"], "abstention_rate": m["abstention_rate_pos"]})
    return rows


def cooc_groups(items):
    """(group index of a negative, group labels) for cooc quartiles and for freq_rank bands."""
    scores = [it["cooc_score"] for it in items if it["answer"] == "no"]
    edges = np.quantile(scores, [0.25, 0.5, 0.75])
    bounds = [min(scores), *edges, max(scores)]
    quart_labels = [f"Q{i + 1} ({bounds[i]:.2f}-{bounds[i + 1]:.2f})" for i in range(4)]
    band_labels = [f"{lo}-{hi}" if hi else f"{lo}+" for lo, hi in FREQ_BANDS]

    def band(it):
        rank = it["freq_rank"]
        return next(i for i, (lo, hi) in enumerate(FREQ_BANDS) if rank >= lo and (hi is None or rank <= hi))

    return {"cooc_quartile": (lambda it: int(np.searchsorted(edges, it["cooc_score"], side="right")), quart_labels),
            "freq_rank_band": (band, band_labels)}


def cooc_effect(ds, items) -> list[dict]:
    rows = []
    for grouping, (group_of, labels) in cooc_groups(items).items():
        for split in ("pooled",) + SPLITS:
            neg = [it for it in items if it["answer"] == "no" and in_split(it, split)]
            for prompt in PROMPT_KINDS:
                for g, label in enumerate(labels):
                    m = metrics([("no", it["pred"][prompt]) for it in neg if group_of(it) == g])
                    rows.append({"dataset": ds, "grouping": grouping, "split": split, "prompt": prompt, "group": label,
                                 "n": m["n"], "hallucination_rate": m["hallucination_rate"],
                                 "abstention_rate": m["abstention_rate_neg"]})
    return rows


def prompt_flips(ds, items) -> list[dict]:
    rows = []
    for a, b in FLIP_PAIRS:
        for label in ("yes", "no"):
            sub = [it for it in items if it["answer"] == label]
            c = Counter((it["pred"][a], it["pred"][b]) for it in sub)
            rows.append({"dataset": ds, "prompt_a": a, "prompt_b": b, "label": label, "n": len(sub),
                         "flipped": sum(v for (x, y), v in c.items() if x != y),
                         **{f"{x}->{y}": c[x, y] for x in ANSWERS for y in ANSWERS}})
    return rows


def top_objects(ds, items) -> list[dict]:
    rows = []
    for kind, label, wrong in (("hallucinated", "no", "yes"), ("missed", "yes", "no")):
        n = Counter(it["object"] for it in items if it["answer"] == label)
        err = Counter(it["object"] for it in items if it["answer"] == label and it["pred"][TOP_PROMPT] == wrong)
        ranked = sorted(err, key=lambda o: (-err[o], -err[o] / n[o], o))[:TOP_K]
        rows += [{"dataset": ds, "kind": kind, "prompt": TOP_PROMPT, "rank": i + 1, "object": o, "errors": err[o],
                  "n": n[o], "rate": err[o] / n[o]} for i, o in enumerate(ranked)]
    return rows


# ---- markdown ----------------------------------------------------------------------------------
def size_md(rows) -> str:
    out = ["# Object size: positives by size bin\n",
           "Distinct positive questions; size bin of the largest instance (COCO thresholds: small < 32², "
           "medium < 96² pixels). Recall + FNR + abstention = 100%. Rates in %.\n"]
    for ds in DATASETS:
        body = [[r["prompt"], r["size_bin"], r["n"], pct(r["recall"]), pct(r["fn_rate"]), pct(r["abstention_rate"])]
                for r in rows if r["dataset"] == ds]
        out += [f"\n## {ds.upper()}\n\n", md_table(["Prompt", "Size", "N", "Recall", "FNR", "Abstain"], body)]
    return "".join(out)


def cooc_md(rows) -> str:
    out = ["# Co-occurrence and object frequency: negatives\n",
           "Hallucination rate (%) of negatives. *cooc_score* = mean over the image's annotated objects of "
           "P(probe object | that object); quartile edges from each dataset's pooled negatives. *freq_rank* = "
           "the probe object's frequency rank in the selected images (1 = most frequent). Pooled = each "
           "distinct question once. N per cell in effect_cooc.csv.\n"]
    for ds in DATASETS:
        out.append(f"\n## {ds.upper()}\n")
        for grouping, title in (("cooc_quartile", "By co-occurrence quartile"), ("freq_rank_band", "By frequency rank")):
            sel = [r for r in rows if r["dataset"] == ds and r["grouping"] == grouping]
            groups = list(dict.fromkeys(r["group"] for r in sel))
            cell = {(r["split"], r["prompt"], r["group"]): r for r in sel}
            body = [[split, prompt, *(f"{pct(cell[split, prompt, g]['hallucination_rate'])} (n={cell[split, prompt, g]['n']})"
                                      for g in groups)]
                    for split in ("pooled",) + SPLITS for prompt in PROMPT_KINDS]
            out += [f"\n### {title}\n\n", md_table(["Split", "Prompt", *groups], body)]
    return "".join(out)


def flips_md(rows) -> str:
    out = ["# Question wording: answer flips between prompts\n",
           "Distinct questions whose parsed answer differs between two prompts (invalid parses count as "
           "abstentions). Full transition counts in prompt_flips.csv.\n\n"]
    body = [[r["dataset"].upper(), f"{r['prompt_a']} vs {r['prompt_b']}", "positives" if r["label"] == "yes" else "negatives",
             r["n"], f"{r['flipped']} ({pct(r['flipped'] / r['n'])}%)"] for r in rows]
    out.append(md_table(["Dataset", "Prompts", "Questions", "N", "Answer changed"], body))
    out.append("\n## Where the ordinary prompt's answers go under the evidence prompt\n\n"
               "Does permission to abstain remove hallucinations, or correct answers? Counts of distinct questions.\n")
    for ds in DATASETS:
        sel = {r["label"]: r for r in rows if r["dataset"] == ds and (r["prompt_a"], r["prompt_b"]) == ("ordinary", "evidence")}
        body = []
        for label, name in (("no", "negatives"), ("yes", "positives")):
            r = sel[label]
            for x in ("yes", "no"):
                total = sum(r[f"{x}->{y}"] for y in ANSWERS)
                if not total:
                    continue
                what = {("no", "yes"): "hallucination", ("no", "no"): "correct", ("yes", "yes"): "correct",
                        ("yes", "no"): "miss"}[label, x]
                body.append([f"{name}: ordinary *{x}* ({what})", total,
                             *(f"{r[f'{x}->{y}']} ({pct(r[f'{x}->{y}'] / total)}%)" for y in ANSWERS)])
        out += [f"\n### {ds.upper()}\n\n", md_table(["Ordinary answer", "N", "Evidence: yes", "Evidence: no",
                                                    "Evidence: not sure"], body)]
    return "".join(out)


def top_md(rows) -> str:
    out = [f"# Category-specific errors ({TOP_PROMPT} prompt)\n",
           "Top objects by number of errors over distinct questions; rate = errors / questions about that "
           "object with that label.\n"]
    for ds in DATASETS:
        for kind, head in (("hallucinated", "Most hallucinated (answered yes, labelled no)"),
                           ("missed", "Most missed (answered no, labelled yes)")):
            body = [[r["rank"], r["object"], r["errors"], r["n"], pct(r["rate"])] for r in rows
                    if r["dataset"] == ds and r["kind"] == kind]
            out += [f"\n## {ds.upper()}: {head}\n\n", md_table(["#", "Object", "Errors", "N", "Rate (%)"], body)]
    return "".join(out)


def contrast_md(long_rows) -> str:
    cell = {(r["dataset"], r["prompt"], r["split"]): r for r in long_rows if r["label_view"] == "full_original"}
    metric_names = [("accuracy", "Accuracy"), ("f1", "F1"), ("hallucination_rate", "Halluc (FPR)"), ("fn_rate", "FNR")]

    def fmt(r, m):
        return f"{pct(float(r[m]))} [{pct(float(r[m + '_ci_lo']))}, {pct(float(r[m + '_ci_hi']))}]"

    out = ["# Dataset contrast: COCO vs GQA\n",
           "Same prompt and split, full set with original labels. Point estimate [95% bootstrap CI over images] "
           "in %, from score.py.\n"]
    for m, name in metric_names:
        body = [[prompt, split, fmt(cell["coco", prompt, split], m), fmt(cell["gqa", prompt, split], m),
                 f"{100 * (float(cell['gqa', prompt, split][m]) - float(cell['coco', prompt, split][m])):+.1f}"]
                for prompt in PROMPT_KINDS for split in SPLITS + ("all",)]
        out += [f"\n## {name}\n\n", md_table(["Prompt", "Split", "COCO", "GQA", "GQA - COCO (pp)"], body)]
    return "".join(out)


# ---- figures -----------------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": MUTED, "ytick.left": False,
        "legend.frameon": False, "legend.fontsize": 9, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def prompt_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=PROMPT_COLOR[p]) for p in PROMPT_KINDS]
    fig.legend(handles, [f"{p} prompt" for p in PROMPT_KINDS], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), handlelength=1.2)


def grouped_bars(ax, categories, values, errors=None):
    """values[prompt] -> list of % per category; errors[prompt] -> (lower, upper) offsets."""
    width = 0.26
    x = np.arange(len(categories))
    for i, prompt in enumerate(PROMPT_KINDS):
        ax.bar(x + (i - 1) * width, values[prompt], width, color=PROMPT_COLOR[prompt], edgecolor=SURFACE,
               linewidth=1.5, yerr=None if errors is None else errors[prompt],
               error_kw={"ecolor": INK2, "elinewidth": 0.9, "capsize": 0})
    ax.set_xticks(x, categories)
    ax.tick_params(axis="x", length=0)


def fig_fpr_by_split(long_rows, path):
    cell = {(r["dataset"], r["prompt"], r["split"]): r for r in long_rows if r["label_view"] == "full_original"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        vals = {p: [100 * float(cell[ds, p, s]["hallucination_rate"]) for s in SPLITS] for p in PROMPT_KINDS}
        errs = {p: np.array([[vals[p][j] - 100 * float(cell[ds, p, s]["hallucination_rate_ci_lo"]),
                              100 * float(cell[ds, p, s]["hallucination_rate_ci_hi"]) - vals[p][j]]
                             for j, s in enumerate(SPLITS)]).T for p in PROMPT_KINDS}
        grouped_bars(ax, list(SPLITS), vals, errs)
        ax.set_title(ds.upper(), loc="left")
    axes[0].set_ylabel("Hallucination rate (%)\n'yes' on absent objects")
    prompt_legend(fig)
    fig.text(0.5, -0.04, "Negative sampling split. Error bars: 95% bootstrap CI over images.", ha="center",
             color=MUTED, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_recall_by_size(size_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        cell = {(r["prompt"], r["size_bin"]): r for r in size_rows if r["dataset"] == ds}
        grouped_bars(ax, [f"{s}\n(n={cell['ordinary', s]['n']})" for s in SIZE_BINS],
                     {p: [100 * cell[p, s]["recall"] for s in SIZE_BINS] for p in PROMPT_KINDS})
        ax.set_title(ds.upper(), loc="left")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Recall (%)\n'yes' on present objects")
    prompt_legend(fig)
    fig.text(0.5, -0.06, "Size of the object's largest instance (COCO area thresholds).", ha="center",
             color=MUTED, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_fpr_by_cooc(cooc_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sel = [r for r in cooc_rows if r["dataset"] == ds and r["grouping"] == "cooc_quartile" and r["split"] == "pooled"]
        groups = list(dict.fromkeys(r["group"] for r in sel))
        x = np.arange(len(groups))
        for prompt in PROMPT_KINDS:
            y = [100 * next(r for r in sel if r["prompt"] == prompt and r["group"] == g)["hallucination_rate"] for g in groups]
            ax.plot(x, y, color=PROMPT_COLOR[prompt], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(x, [g.replace(" (", "\n(") for g in groups])
        ax.tick_params(axis="x", length=0)
        ax.set_title(ds.upper(), loc="left")
    top = max(r["hallucination_rate"] for r in cooc_rows if r["grouping"] == "cooc_quartile" and r["split"] == "pooled")
    axes[0].set_ylim(0, 100 * top * 1.08)
    axes[0].set_ylabel("Hallucination rate (%)")
    prompt_legend(fig)
    fig.text(0.5, -0.08, "Quartile of the probe object's co-occurrence with the image's annotated objects "
             "(pooled over splits).", ha="center", color=MUTED, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_prompt_comparison(items_by_ds, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.2), sharex=True)
    for row, ds in enumerate(DATASETS):
        for col, (label, name) in enumerate((("no", "negatives (object absent)"), ("yes", "positives (object present)"))):
            ax = axes[row, col]
            sub = [it for it in items_by_ds[ds] if it["answer"] == label]
            for y, prompt in enumerate(PROMPT_KINDS):
                c = Counter(it["pred"][prompt] for it in sub)
                shares = {"correct": c[label], "wrong": c["yes" if label == "no" else "no"], "abstain": c["abstain"]}
                left = 0.0
                for outcome, count in shares.items():
                    w = 100 * count / len(sub)
                    ax.barh(y, w, left=left, height=0.62, color=OUTCOME_COLOR[outcome], edgecolor=SURFACE, linewidth=1.5)
                    if w >= 7:
                        ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8, color=INK)
                    left += w
            ax.set_yticks(range(len(PROMPT_KINDS)), PROMPT_KINDS)
            ax.invert_yaxis()
            ax.tick_params(axis="y", length=0, labelcolor=INK2)
            ax.grid(False)
            ax.set_xlim(0, 100)
            ax.set_title(f"{ds.upper()}: {name}", loc="left", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Share of distinct questions (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLOR[o]) for o in OUTCOME_COLOR]
    fig.legend(handles, ["correct", "wrong (hallucination on negatives, miss on positives)", "abstain (not sure)"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def pick_examples():
    """The listed Stage B false positives; stops if one no longer has the verdict its figure needs."""
    items = {it["audit_id"]: it for ds in DATASETS for it in stage_b("fp", ds)}
    picked = []
    for ids, verdict in ((EXAMPLES_FP, "absent"), (EXAMPLES_LABEL_ERROR, "present")):
        wrong = [a for a in ids if items[a]["verdict"] != verdict]
        if wrong:
            sys.exit(f"examples {wrong} no longer have verdict {verdict!r}; update the lists in analyze.py")
        picked.append([items[a] for a in ids])
    return picked


def fig_examples(examples, evidence_resp, path, ncols, caption):
    nrows = -(-len(examples) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, it in zip(axes.flat, examples):
        img = Image.open(image_path(DATA_ROOT, it["dataset"], it["image"])).convert("RGB")
        img.thumbnail(TILE)
        tile = Image.new("RGB", TILE, SURFACE)
        tile.paste(img, ((TILE[0] - img.width) // 2, (TILE[1] - img.height) // 2))
        ax.imshow(tile)
        ax.set_title(textwrap.fill(f"{it['dataset'].upper()}: is there {article(it['object'])} {it['object']}?", 34), fontsize=9, loc="left")
        ev = evidence_resp[it["dataset"], it["image"], it["object"]]
        ax.text(0, -0.04, textwrap.fill(f"ordinary: {it['response']}  ·  evidence: {ev}", 40) + "\n"
                + textwrap.fill(caption(it), 40), transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path)
    plt.close(fig)


# ---- main --------------------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    problems = verify()
    if problems:
        sys.exit("frozen sets do not match MANIFEST.json:\n  " + "\n  ".join(problems))
    long_csv = TABLES / "metrics_long.csv"
    if not long_csv.exists():
        sys.exit("results/tables/metrics_long.csv is missing; run src/score.py first")
    long_rows = read_csv(long_csv)

    items_by_ds, size_rows, cooc_rows, flip_rows, top_rows = {}, [], [], [], []
    for ds in DATASETS:
        questions = load_questions(ds)
        preds = {p: load_predictions(ds, p) for p in PROMPT_KINDS}
        items = distinct_questions(questions, preds)
        items_by_ds[ds] = items
        size_rows += size_effect(ds, items)
        cooc_rows += cooc_effect(ds, items)
        flip_rows += prompt_flips(ds, items)
        top_rows += top_objects(ds, items)

    TABLES.mkdir(parents=True, exist_ok=True)
    for name, rows, text in (("effect_size", size_rows, size_md(size_rows)),
                             ("effect_cooc", cooc_rows, cooc_md(cooc_rows)),
                             ("prompt_flips", flip_rows, flips_md(flip_rows)),
                             ("top_objects", top_rows, top_md(top_rows))):
        write_csv(TABLES / f"{name}.csv", rows)
        (TABLES / f"{name}.md").write_text(text, newline="\n")
    (TABLES / "dataset_contrast.md").write_text(contrast_md(long_rows), newline="\n")

    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    fig_fpr_by_split(long_rows, FIGURES / "fig_fpr_by_split.png")
    fig_recall_by_size(size_rows, FIGURES / "fig_recall_by_size.png")
    fig_fpr_by_cooc(cooc_rows, FIGURES / "fig_fpr_by_cooc.png")
    fig_prompt_comparison(items_by_ds, FIGURES / "fig_prompt_comparison.png")
    evidence_resp = {(it["dataset"], it["image"], it["object"]): it["response"]["evidence"].strip()
                     for items in items_by_ds.values() for it in items}
    real, label_errors = pick_examples()
    fig_examples(real, evidence_resp, FIGURES / "fig_examples_fp.png", 4,
                 lambda it: f"reviewer: absent · tag: {it['tag']}")
    fig_examples(label_errors, evidence_resp, FIGURES / "fig_examples_label_error.png", 3,
                 lambda it: "dataset label: no · reviewer: present (label error)")
    print("wrote results/tables/: effect_size, effect_cooc, prompt_flips, top_objects (.csv + .md), dataset_contrast.md")
    print("wrote results/figures/: " + ", ".join(sorted(p.name for p in FIGURES.glob("fig_*.png"))))


if __name__ == "__main__":
    main()
