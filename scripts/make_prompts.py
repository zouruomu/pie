# Writes N unique prompts whose chat-rendered length is exactly TARGET tokens
# (system prompt + user turn + generation prompt), matching the site's Raw
# workload shape (512 in, 256 out).
import json
import sys

from transformers import AutoTokenizer

model, target, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tok = AutoTokenizer.from_pretrained(model)
system = "You are a helpful benchmarking assistant."
words = ("the quick brown fox jumps over the lazy dog and then walks back home "
         "again to rest before the next long day begins").split()


def rendered_len(user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False).input_ids)


prompts = []
for i in range(n):
    head = f"Request #{i}. Write a short story about a robot. "
    body = lambda k: head + " ".join(words[j % len(words)] for j in range(k))
    lo, hi = 0, 2 * target
    while lo < hi:  # largest k whose prompt still fits in target
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if rendered_len(body(mid)) <= target else (lo, mid - 1)
    p = body(lo)
    assert rendered_len(p) == target, (i, rendered_len(p))
    prompts.append(p)
json.dump(prompts, open(out, "w"))
print(f"{n} prompts, {target} tokens each -> {out}")
