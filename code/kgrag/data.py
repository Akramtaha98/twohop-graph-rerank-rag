"""Dataset loaders. Every loader returns (examples, passages).

example: dict(qid, question, answer, sup_pids: set[int], cand_pids: list[int])
passages: list of dict(title, text); the corpus is the pooled, de-duplicated set of all
paragraphs of the sampled questions (open-domain-style retrieval over the pool).
"""
import json
import random
from pathlib import Path


class Corpus:
    def __init__(self):
        self.passages, self._ix = [], {}

    def add(self, title, text):
        key = (title, text)
        if key not in self._ix:
            self._ix[key] = len(self.passages)
            self.passages.append({"title": title, "text": text})
        return self._ix[key]


def _hotpot_like(records, n, seed):
    """HotpotQA / 2WikiMultiHopQA official JSON format (context + supporting_facts)."""
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    corp, exs = Corpus(), []
    for r in records[:n]:
        ctx = r["context"]
        if isinstance(ctx, dict):  # HF hotpot_qa format
            ctx = list(zip(ctx["title"], ctx["sentences"]))
            sf = set(r["supporting_facts"]["title"])
        else:
            sf = {t for t, _ in r["supporting_facts"]}
        cand, sup = [], set()
        for title, sents in ctx:
            pid = corp.add(title, " ".join(s.strip() for s in sents))
            cand.append(pid)
            if title in sf:
                sup.add(pid)
        exs.append(dict(qid=str(r.get("id", r.get("_id"))), question=r["question"], answer=r["answer"], sup_pids=sup, cand_pids=cand, qtype=r.get("type", "")))
    return exs, corp.passages


def load_hotpot(path=None, n=1000, seed=0, split="validation"):
    if path:
        recs = json.loads(Path(path).read_text())
    else:
        from datasets import load_dataset  # pip install datasets

        recs = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    return _hotpot_like(recs, n, seed)


def load_2wiki(path=None, n=1000, seed=0):
    """2WikiMultiHopQA: local official json, or the HF parquet copy (validation split) when path is None."""
    if path:
        return _hotpot_like(json.loads(Path(path).read_text()), n, seed)
    from datasets import load_dataset

    return _hotpot_like(load_dataset("framolfese/2WikiMultihopQA", split="validation"), n, seed)


def load_musique(path, n=1000, seed=0):
    """Official MuSiQue jsonl (musique_ans_v1.0_dev.jsonl)."""
    rng = random.Random(seed)
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl", repo_type="dataset")
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("answerable", True)]
    rng.shuffle(recs)
    corp, exs = Corpus(), []
    for r in recs[:n]:
        cand, sup = [], set()
        for p in r["paragraphs"]:
            pid = corp.add(p["title"], p["paragraph_text"])
            cand.append(pid)
            if p.get("is_supporting"):
                sup.add(pid)
        exs.append(dict(qid=r["id"], question=r["question"], answer=r["answer"], sup_pids=sup, cand_pids=cand, qtype=r["id"].split("__")[0]))
    return exs, corp.passages


_SYL = ["ka", "ro", "vel", "mi", "tor", "an", "sul", "bre", "dun", "pa", "lix", "ora", "zen", "cal", "mar", "wen", "thi", "gor"]


def load_toy(n=200, seed=0):
    """Synthetic 2-hop questions (film -> director -> birthplace) for offline pipeline tests only."""
    rng = random.Random(seed)
    used = set()

    def name(k=3):
        while True:
            s = "".join(rng.choice(_SYL) for _ in range(k)).capitalize()
            if s not in used:
                used.add(s)
                return s

    corp, gold, rows = Corpus(), [], []
    for _ in range(n):
        film, d1, d2, city = name(4), name(3), name(3), name(4)
        director = f"{d1} {d2}"
        year = str(rng.randint(1950, 2020))
        a = corp.add(film, f"{film} is a film released in {year} and directed by {director}.")
        b = corp.add(director, f"{director} is a film director who was born in {city}.")
        gold.append((a, b))
        rows.append((film, city))
    exs = []
    allp = list(range(len(corp.passages)))
    for i, ((a, b), (film, city)) in enumerate(zip(gold, rows)):
        distract = rng.sample([p for p in allp if p not in (a, b)], 4)
        cand = [a, b] + distract
        rng.shuffle(cand)
        exs.append(dict(qid=f"toy{i}", question=f"Where was the director of {film} born?", answer=city, sup_pids={a, b}, cand_pids=cand))
    return exs, corp.passages


def load(name, path=None, n=1000, seed=0):
    if name == "toy":
        return load_toy(n, seed)
    if name == "hotpot":
        return load_hotpot(path, n, seed)
    if name == "2wiki":
        return load_2wiki(path, n, seed)
    if name == "musique":
        return load_musique(path, n, seed)
    raise ValueError(name)
