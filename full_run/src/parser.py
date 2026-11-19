"""Fixed response parser: model output -> "yes" | "no" | "abstain" | "invalid".

Rules, applied in order to the lower-cased response with markdown, quotes and surrounding
punctuation removed:
  1. any abstention phrase anywhere (whole words, e.g. "not sure", "unsure",
     "cannot tell")                                        -> "abstain"
  2. first word "yes"                                       -> "yes"
  3. first word "no", or the response starts with "not visible" or
     "it is not visible"                                    -> "no"
  4. anything else, including empty output                  -> "invalid"

Rule 1 comes first so hedged answers ("Yes, but it's unclear") count as abstentions, not
as commitments. The parser is deliberately strict and is never tuned on results: anything
it cannot classify is reported as "invalid" (counted with abstentions in the abstention
rate and also reported separately), so parser failures stay visible instead of being
guessed at.
"""
import re

ABSTAIN_PHRASES = [
    "not sure", "unsure", "uncertain", "not certain", "unclear",
    "cannot tell", "can't tell", "can not tell", "cannot determine", "can't determine",
    "cannot be determined", "hard to tell", "difficult to tell", "impossible to tell",
    "i don't know", "i do not know",
]
_ABSTAIN_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in ABSTAIN_PHRASES) + r")\b")
_NO_PREFIXES = ("not visible", "it is not visible", "it's not visible")
_STRIP = " \t\r\n\"'`*_.,!?:;()[]"


def normalize(response: str) -> str:
    text = response.lower().replace("’", "'").strip().strip(_STRIP)
    return re.sub(r"\s+", " ", text)


def parse(response: str | None) -> str:
    text = normalize(response or "")
    if _ABSTAIN_RE.search(text):
        return "abstain"
    words = re.findall(r"[a-z']+", text)
    if not words:
        return "invalid"
    if words[0] == "yes":
        return "yes"
    if words[0] == "no" or text.startswith(_NO_PREFIXES):
        return "no"
    return "invalid"
