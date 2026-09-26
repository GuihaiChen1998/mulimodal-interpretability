"""Extract a random subset of LLaVA-Pretrain (558K) images + captions from images.zip.

Usage: python prep_llava_subset.py SRC_DIR OUT_DIR N SEED
Writes OUT_DIR/images/<relpath> and OUT_DIR/data.json [{id, image, instruction, caption}].
"""

import json
import os
import random
import sys
import zipfile

src, out, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
data = json.load(open(os.path.join(src, "blip_laion_cc_sbu_558k.json")))
random.Random(seed).shuffle(data)
sub = data[:n]
os.makedirs(os.path.join(out, "images"), exist_ok=True)
with zipfile.ZipFile(os.path.join(src, "images.zip")) as z:
    names = set(z.namelist())
    prefix = "" if sub[0]["image"] in names else "images/"
    rows = []
    for d in sub:
        member = prefix + d["image"]
        dst = os.path.join(out, "images", d["image"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(z.read(member))
        human = d["conversations"][0]["value"].replace("<image>", "").strip()
        rows.append({"id": d["id"], "image": dst, "instruction": human, "caption": d["conversations"][1]["value"]})
json.dump(rows, open(os.path.join(out, "data.json"), "w"), indent=1)
print(f"wrote {len(rows)} samples to {out}")
