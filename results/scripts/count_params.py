# Logical parameter count of a safetensors snapshot: NVFP4 planes (U8, two e2m1 a byte) count two a byte;
# block scales, global scales and the vision tower are reported apart.
import json, struct, sys
from pathlib import Path
snap = Path(sys.argv[1])
tot = {}
for f in sorted(snap.glob("*.safetensors")):
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        hdr = json.loads(h.read(n))
    for name, t in hdr.items():
        if name == "__metadata__":
            continue
        e = 1
        for d in t["shape"]:
            e *= d
        if "scale" in name:
            k = "scales"
        elif not name.startswith("language_model."):
            k = "vision"
        elif "embed_tokens" in name or "lm_head" in name:
            k = "embedding"
        elif ".experts." in name:
            k = "routed"
            if t["dtype"] == "U8":
                e *= 2
        elif "shared_expert" in name:
            k = "shared"
        else:
            k = "other"
        tot[k] = tot.get(k, 0) + e
for k, v in sorted(tot.items()):
    print(f"{k:10s} {v/1e9:9.3f} B")
lm = sum(v for k, v in tot.items() if k not in ("scales", "vision"))
print(f"{'text':10s} {lm/1e9:9.3f} B (weights, no scales)")
cfg = json.load(open(snap / "config.json"))["text_config"]
kept, k = cfg["num_local_experts"], cfg["num_experts_per_tok"]
print(f"experts kept {kept}, top-{k}; active (top-{k} + shared + rest) {(lm - tot['routed'] * (1 - k / kept))/1e9:.3f} B")
