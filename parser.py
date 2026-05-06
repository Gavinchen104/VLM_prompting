"""
parser.py — Map free-text model outputs to one of 7 HAM10000 class codes.

Returns one of: mel, nv, bcc, akiec, bkl, df, vasc, or PARSE_FAIL.

Design notes:
- Strict format first ("Final answer: <code>"), then graceful fallbacks.
- PARSE_FAIL is a *real* outcome — never silently dropped, never counted as wrong.
  The parse-failure rate per condition is itself a measurement of prompt quality.
- Synonyms are accepted. Many models say "melanoma" instead of "mel".
"""

import re

CLASSES = {"mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"}

# Lowercased synonyms → canonical class code
SYNONYMS = {
    # melanoma
    "melanoma": "mel",
    "malignant melanoma": "mel",
    # melanocytic nevus
    "melanocytic nevus": "nv",
    "nevus": "nv",
    "nevi": "nv",
    "mole": "nv",
    "benign nevus": "nv",
    # basal cell carcinoma
    "basal cell carcinoma": "bcc",
    "bcc": "bcc",
    # actinic keratosis / intraepithelial carcinoma
    "actinic keratosis": "akiec",
    "intraepithelial carcinoma": "akiec",
    "bowen": "akiec",
    "bowen's disease": "akiec",
    # benign keratosis
    "benign keratosis": "bkl",
    "seborrheic keratosis": "bkl",
    "solar lentigo": "bkl",
    "lichen planus-like keratosis": "bkl",
    # dermatofibroma
    "dermatofibroma": "df",
    # vascular lesion
    "vascular lesion": "vasc",
    "vascular": "vasc",
    "hemangioma": "vasc",
    "angioma": "vasc",
    "pyogenic granuloma": "vasc",
}


def _normalize(s: str) -> str:
    """lowercase + collapse whitespace + strip punctuation around words."""
    s = s.lower()
    s = re.sub(r"[*_`#]", "", s)        # strip markdown
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse(output: str) -> tuple[str, str]:
    """
    Returns (parsed_label, parse_method).
    parse_method is one of: 'strict', 'synonym', 'fallback', 'fail'.
    """
    if not output:
        return "PARSE_FAIL", "fail"

    text = _normalize(output)

    # ----- 1. Strict: "Final answer: <code>" -----
    m = re.search(r"final answer\s*[:\-]\s*([a-z][a-z\- ]*)", text)
    if m:
        cand = m.group(1).strip().split()[0]   # first word after the colon
        cand = cand.rstrip(".,;:")
        if cand in CLASSES:
            return cand, "strict"
        # Try the multi-word phrase (up to 4 words after the colon)
        phrase = " ".join(m.group(1).strip().split()[:4]).rstrip(".,;:")
        for syn, code in SYNONYMS.items():
            if phrase.startswith(syn):
                return code, "synonym"

    # ----- 2. Synonym anywhere in the last ~200 chars (the answer is usually at the end) -----
    tail = text[-300:]
    for syn, code in sorted(SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        # word-boundary match to avoid 'mole' inside 'molecule'
        if re.search(rf"\b{re.escape(syn)}\b", tail):
            return code, "synonym"

    # ----- 3. Fallback: any class code as a standalone token, anywhere -----
    for cls in CLASSES:
        if re.search(rf"\b{cls}\b", text):
            return cls, "fallback"

    return "PARSE_FAIL", "fail"


# Quick self-test
if __name__ == "__main__":
    cases = [
        ("Final answer: mel",                                          "mel",       "strict"),
        ("...some reasoning...\nFinal answer: nv",                     "nv",        "strict"),
        ("Final Answer: melanoma",                                     "mel",       "synonym"),
        ("I think this is a basal cell carcinoma.",                    "bcc",       "synonym"),
        ("This appears to be a benign nevus or mole.",                 "nv",        "synonym"),
        ("Final answer: glioma",                                       "PARSE_FAIL", "fail"),
        ("",                                                           "PARSE_FAIL", "fail"),
        ("The lesion shows BKL features. Final answer: bkl",           "bkl",       "strict"),
        ("Looks like a hemangioma to me.",                             "vasc",      "synonym"),
    ]
    for inp, want_lbl, want_method in cases:
        got_lbl, got_method = parse(inp)
        ok = (got_lbl == want_lbl and got_method == want_method)
        print(f"{'PASS' if ok else 'FAIL':4} | {got_lbl:11} {got_method:8} | {inp[:60]!r}")
