"""Text normalisation, tokenisation and lightweight entity extraction."""
import re
import string

STOP = set(
    "a an the of in on at to for and or is was were are be been by with as from that this which who whom "
    "what when where why how did does do has have had it its his her their he she they than then into over "
    "under between after before during about not no also but if so such can could would should may might".split()
)

_ART = re.compile(r"\b(a|an|the)\b")
_CAP = re.compile(r"[A-Z0-9][\w'’\-\.]*")
_JOIN = {"of", "the", "de", "von", "van", "du", "la", "le"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def tokens(s: str):
    return re.findall(r"\w+", s.lower())


def content_tokens(s: str):
    return [t for t in tokens(s) if t not in STOP]


def _clean_entity(e: str) -> str:
    e = e.strip(" .,'’-")
    e = re.sub(r"^(The|A|An)\s+", "", e)
    return e.strip()


def extract_entities_regex(text: str):
    """Capitalised spans (allowing joiners such as 'of', 'the') plus 4-digit years.

    Deterministic, dependency free. Sentence-initial stop words are discarded.
    """
    words = text.split()
    ents, cur, pending = [], [], []

    def flush():
        nonlocal cur, pending
        if cur:
            ents.append(" ".join(cur))
        cur, pending = [], []

    for i, w in enumerate(words):
        bare = w.strip(".,;:!?()[]\"'\u201c\u201d")
        if bare.isdigit() and len(bare) == 4:  # years are standalone entities
            flush()
            ents.append(bare)
        elif _CAP.fullmatch(bare) and bare[:1].isupper() and not (
            bare.lower() in STOP and (i == 0 or words[i - 1].endswith((".", "?", "!")))
        ):
            cur += pending + [bare]
            pending = []
            if w[-1] in ".,;:!?)" and not bare.endswith("."):
                flush()
        elif bare.lower() in _JOIN and cur:
            pending.append(bare)
        else:
            flush()
    flush()
    out = []
    for e in ents:
        e = _clean_entity(e)
        if len(e) > 1 and e.lower() not in STOP:
            out.append(e)
    return out


_SPACY = None


_SPACY_MEMO = {}


def extract_entities_spacy(text: str):
    global _SPACY
    if text in _SPACY_MEMO:
        return _SPACY_MEMO[text]
    if _SPACY is None:
        import spacy  # optional dependency: pip install spacy && python -m spacy download en_core_web_sm

        _SPACY = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    out = [_clean_entity(e.text) for e in _SPACY(text).ents if len(e.text) > 1]
    _SPACY_MEMO[text] = out
    return out


def extract_entities(text: str, mode: str = "regex"):
    if mode == "spacy":
        return extract_entities_spacy(text)
    if mode == "both":  # union of rule-based spans and spaCy NER
        return extract_entities_regex(text) + extract_entities_spacy(text)
    return extract_entities_regex(text)


def ekey(e: str) -> str:
    """Canonical entity key."""
    return " ".join(tokens(e))
