"""Optional: LoRA supervised fine-tuning of the generator on retrieved evidence (answer-only loss).

    python -m kgrag.train_lora --n_train 2000 --out adapters/qwen_hotpot
Requires: torch, transformers, peft (and a GPU for reasonable speed). Not run in the authoring sandbox.
"""
import argparse
import random

from . import data as D
from .generate import SYSTEM, build_context, build_prompt
from .retrievers import Dense, Hybrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense", default="tfidf", choices=["tfidf", "bge"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    exs, passages = D.load_hotpot(None, a.n_train, a.seed, split="train")  # train split: never used for evaluation
    dense = Dense(a.dense).fit(passages)
    ret = Hybrid(dense).fit(passages)

    tok = AutoTokenizer.from_pretrained(a.model)
    import transformers as _tf

    dkey = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(a.model, **{dkey: torch.float16 if torch.cuda.is_available() else torch.float32})
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    model.train()
    for ep in range(a.epochs):
        random.shuffle(exs)
        for i, ex in enumerate(exs):
            pids, extra = ret.search(ex["question"], a.k)
            ctx = build_context(passages, pids, extra.get("bridges"), chain=True)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(ex["question"], ctx)}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0][-1536:]
            a_ids = tok(ex["answer"] + tok.eos_token, return_tensors="pt", add_special_tokens=False).input_ids[0]
            ids = torch.cat([p_ids, a_ids]).unsqueeze(0).to(dev)
            labels = ids.clone()
            labels[0, : len(p_ids)] = -100  # loss on the answer only
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            if (i + 1) % 8 == 0:  # gradient accumulation over 8 examples
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if i % 100 == 0:
                print(f"epoch {ep} step {i} loss {loss.item():.3f}", flush=True)
    model.save_pretrained(a.out)
    print("saved adapter to", a.out)


if __name__ == "__main__":
    main()
