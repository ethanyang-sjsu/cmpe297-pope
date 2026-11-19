"""Unit tests for the fixed response parser, the prompt templates, and the metric arithmetic.

Run from the project folder:  pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parser import parse  # noqa: E402
from prompts import MAX_NEW_TOKENS, PROMPTS, build_prompt  # noqa: E402


@pytest.mark.parametrize("response, expected", [
    # plain answers, any case, with punctuation or markdown
    ("Yes", "yes"), ("yes.", "yes"), ("YES!", "yes"), ("**Yes**", "yes"), ('"yes"', "yes"),
    ("No", "no"), ("no.", "no"), (" No ", "no"),
    # answers followed by an explanation
    ("Yes, there is a dog.", "yes"), ("Yes. There is a dog on the couch.", "yes"),
    ("no, there isn't", "no"), ("No, there is no giraffe in the image.", "no"),
    # the evidence prompt's "not visible" option means no
    ("Not visible", "no"), ("not visible.", "no"), ("It is not visible.", "no"),
    ("It’s not visible", "no"),
    # abstentions, including hedged commitments
    ("Not sure", "abstain"), ("not sure.", "abstain"), ("Unsure", "abstain"),
    ("I cannot tell.", "abstain"), ("I can't tell from the image", "abstain"),
    ("Yes, but it's unclear.", "abstain"), ("No, I'm not sure.", "abstain"),
    ("I don't know", "abstain"),
    # anything else is invalid, never guessed
    ("", "invalid"), (None, "invalid"), ("   ", "invalid"), ("...", "invalid"),
    ("There is a dog in the image.", "invalid"), ("The image shows a kitchen.", "invalid"),
    ("yeah", "invalid"), ("nope", "invalid"),
    # whole-word matching: these words only start with yes/no/abstention text
    ("Yesterday's photo", "invalid"), ("Nothing like that", "invalid"),
    ("Note: the image is dark", "invalid"), ("Unsurpassed view", "invalid"),
])
def test_parse(response, expected):
    assert parse(response) == expected


def test_prompts_use_object_and_article():
    for kind in PROMPTS:
        assert "an elephant" in build_prompt(kind, "elephant")
        assert "a dining table" in build_prompt(kind, "dining table")
        assert "{" not in build_prompt(kind, "car")


def test_only_evidence_prompt_permits_abstention():
    assert "not sure" in build_prompt("evidence", "car")
    for kind in ("ordinary", "neutral"):
        assert parse(build_prompt(kind, "car")) != "abstain"  # no abstention wording leaks in
        assert "not sure" not in build_prompt(kind, "car")


def test_matched_token_budget():
    assert MAX_NEW_TOKENS == 16


# ---- metric arithmetic (src/score.py) ----------------------------------------------------------
from score import headline, image_counts, metrics, resample_weights  # noqa: E402

MIXED = ([("yes", "yes")] * 7 + [("yes", "no")] * 2 + [("yes", "abstain")] + [("yes", "invalid")]
         + [("no", "no")] * 5 + [("no", "yes")] * 3 + [("no", "abstain")] * 2 + [("no", "invalid")])


def test_rates_on_negatives_sum_to_one():
    m = metrics(MIXED)
    assert m["specificity"] + m["hallucination_rate"] + m["abstention_rate_neg"] == pytest.approx(1)
    assert m["recall"] + m["fn_rate"] + m["abstention_rate_pos"] == pytest.approx(1)


def test_metric_definitions():
    m = metrics(MIXED)
    assert (m["n"], m["n_pos"], m["n_neg"], m["invalid"]) == (22, 11, 11, 2)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (7, 2, 3, 5)
    assert m["accuracy"] == pytest.approx(12 / 22)            # abstentions are never correct
    assert m["precision"] == pytest.approx(7 / 10)
    assert m["recall"] == pytest.approx(7 / 11)               # over all positives, abstentions included
    assert m["f1"] == pytest.approx(2 * (7 / 10) * (7 / 11) / (7 / 10 + 7 / 11))
    assert m["hallucination_rate"] == pytest.approx(3 / 11)
    assert m["abstention_rate"] == pytest.approx(5 / 22)      # abstain + invalid
    assert m["yes_ratio"] == pytest.approx(10 / 22)
    assert m["class_balance"] == pytest.approx(0.5)


def test_undefined_precision_is_nan():
    m = metrics([("yes", "no"), ("no", "no")])
    assert m["precision"] != m["precision"] and m["f1"] == 0


def test_bootstrap_formulas_match_metrics():
    preds = {f"q{i}": {"prediction": p} for i, (_, p) in enumerate(MIXED)}
    qs = [{"question_id": f"q{i}", "answer": a, "image": f"im{i % 4}"} for i, (a, _) in enumerate(MIXED)]
    images = [f"im{i}" for i in range(4)]
    point = headline(image_counts(qs, preds, images).sum(0))
    m = metrics(MIXED)
    for k, v in point.items():
        assert v == pytest.approx(m[k])


def test_resample_weights_deterministic():
    w = resample_weights(500, 20, 297, "coco")
    assert (w == resample_weights(500, 20, 297, "coco")).all()
    assert (w.sum(1) == 500).all() and not (w == resample_weights(500, 20, 297, "gqa")).all()
