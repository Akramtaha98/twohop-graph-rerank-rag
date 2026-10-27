"""MultiHop-RAG loader (Tang & Yang 2024): 609 news articles, 2,556 queries (null queries are dropped).

Articles are split into word windows; each window is one passage whose title is the article headline. Supporting passages are the
windows that contain the evidence sentence ("fact") of each evidence item (best token overlap if no exact match).
Each passage also carries "doc" (article index); each example carries "sup_docs" for a document-level metric.

Data: HF `yixuantt/MultiHopRAG` (configs "corpus" and "MultiHopRAG"), or local JSON files via --path "corpus.json,MultiHopRAG.json",
or the GitHub raw files as a last resort. CHECK the field names printed on the first run (body/text, evidence_list/fact/title).
"""
import json
import random
import re
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/{}"


def _norm(s):
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def _raw(path):
    if path:
        c, q = [x.strip() for x in path.split(",")]
        return json.loads(Path(c).read_text()), json.loads(Path(q).read_text())
    try:
        from datasets import load_dataset

        return list(load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")), list(load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train"))
    except Exception as e:  # noqa: BLE001
        print("HF load failed (%s); trying GitHub raw files" % e)
        get = lambda f: json.loads(urllib.request.urlopen(RAW.format(f), timeout=60).read())  # noqa: E731
        return get("corpus.json"), get("MultiHopRAG.json")


def load_multihoprag(path=None, n=100000, seed=0, chunk_words=200):
    corpus, qas = _raw(path)
    print("corpus fields:", sorted(corpus[0].keys()), "| query fields:", sorted(qas[0].keys()))
    passages, by_title = [], {}
    for di, a in enumerate(corpus):
        title, body = a.get("title", ""), a.get("body", a.get("text", ""))
        words = body.split()
        wins = [words[i : i + chunk_words] for i in range(0, max(1, len(words)), chunk_words)]
        if len(wins) > 1 and len(wins[-1]) < 30:  # merge a tiny tail
            wins[-2] += wins.pop()
        ids = []
        for w in wins:
            ids.append(len(passages))
            passages.append({"title": title, "text": " ".join(w), "doc": di})
        by_title.setdefault(_norm(title), []).extend(ids)
    ptoks = [set(_norm(p["text"]).split()) for p in passages]
    pnorm = [_norm(p["text"]) for p in passages]
    rows = [q for q in qas if q.get("evidence_list") and q.get("question_type") != "null_query"]
    random.Random(seed).shuffle(rows)
    exs, miss = [], 0
    for qi, q in enumerate(rows[:n]):
        sup, docs = set(), set()
        for ev in q["evidence_list"]:
            ids = by_title.get(_norm(ev.get("title")))
            if not ids:
                miss += 1
                continue
            fact = _norm(ev.get("fact"))
            hit = [i for i in ids if fact and fact in pnorm[i]]
            if not hit:
                ft = set(fact.split())
                hit = [max(ids, key=lambda i: len(ft & ptoks[i]))]
            sup.update(hit)
            docs.add(passages[ids[0]]["doc"])
        if not sup:
            continue
        exs.append(dict(qid=f"mhr{qi}", question=q["query"], answer=q["answer"], sup_pids=sup, cand_pids=[], qtype=q["question_type"], sup_docs=docs))
    print(f"multihoprag: {len(exs)} questions, {len(passages)} passages, {miss} evidence items without a matching article title")
    return exs, passages
