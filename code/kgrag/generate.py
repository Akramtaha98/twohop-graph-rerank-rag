"""Answer generators: a Hugging Face causal LM (optionally LoRA-adapted) and an offline extractive stub.

The extractive stub exists only so the whole pipeline can be smoke-tested without GPUs or model
downloads. Its numbers must never be reported as results.
"""
import re

from .text import extract_entities, tokens, ekey

SYSTEM = (
    "You answer multi-hop questions using only the evidence provided. "
    "Reply with the shortest possible answer (an entity, date, number or yes/no) and nothing else."
)


def build_context(passages, pids, bridges=None, chain=True):
    parts = []
    for i, pid in enumerate(pids, 1):
        p = passages[pid]
        line = f"[{i}] {p['title']}: {p['text']}"
        if chain and bridges and bridges.get(pid):
            line += f" (linked via: {', '.join(bridges[pid])})"
        parts.append(line)
    return "\n".join(parts)


def build_prompt(question, context):
    return f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer:"


class Extractive:
    name = "extractive"

    def __call__(self, question, context):
        qt = set(tokens(question))
        best, best_s = None, -1
        for sent in re.split(r"(?<=[.!?])\s+|\n", context):
            s = len(qt & set(tokens(sent)))
            if s > best_s:
                best, best_s = sent, s
        if not best:
            return ""
        cands = [e for e in extract_entities(best) if ekey(e) not in qt and not set(tokens(e)) <= qt]
        cands = [c for c in cands if not re.fullmatch(r"\[?\d+\]?", c)]
        return cands[-1] if cands else ""


class HFGenerator:
    name = "hf"

    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", adapter=None, max_new_tokens=32, load_4bit=False, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        cuda = torch.cuda.is_available()
        mps = (not cuda) and torch.backends.mps.is_available()  # Apple silicon (M1-M4)
        import transformers as _tf

        dkey = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"  # name changed in transformers 5
        kw = {dkey: torch.float16 if (cuda or mps) else torch.float32}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto" if cuda else None, **kw)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        if mps:
            self.model.to("mps")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, question, context):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(question, context)}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)  # works on old and new transformers
        n_in = enc["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        ans = self.tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
        return ans.splitlines()[0].strip().rstrip(".") if ans else ""
