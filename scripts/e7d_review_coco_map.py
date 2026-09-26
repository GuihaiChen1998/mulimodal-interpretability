"""E7d: apply Claude's review (labels + overrides) to coco2concept_v1.json -> coco2concept_v1_reviewed.json + a table.

Labels: exact = object-level concept; related = superordinate / scene / use-context concept;
none = no acceptable concept found. Overrides pick another candidate from the same pool.
"""
import json, sys

src, dst, md = sys.argv[1:4]
m = json.load(open(src))
OVERRIDE = {  # category: (concept_id, reason)
    "person": (31442, "top spec; object-level 'humans' beats 'Specific Individuals'"),
    "airplane": (10605, "higher spec and clip; 'Planes' also covers geometry"),
    "zebra": (26481, "higher spec; same concept as giraffe"),
    "skateboard": (29412, "current pick is 'Bicycles and Motorcycles'; highest clip in pool"),
    "cup": (17856, "current pick is the measuring-cup sense"),
    "broccoli": (17182, "vegetable-specific; consistent with carrot"),
    "cake": (17504, "clip 0.82 vs 0.73; 'Holiday Feast Foods' is too broad"),
    "refrigerator": (33499, "current pick is 'Cool' slang; tie on score"),
    "teddy bear": (6832, "toy concept instead of 'Cute Affectionate Child Scenes'"),
    "kite": (21555, "exact 'Kite Flying' (clip 0.91) though inactive in masked read-out; current pick is 'Drones'"),
}
LABEL = {
 "exact": ["bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","bird","cat","dog",
           "horse","sheep","cow","elephant","sports ball","kite","tennis racket","knife","apple","sandwich","pizza",
           "chair","bed","dining table","tv","laptop","mouse","keyboard","cell phone","microwave","book","clock",
           "toothbrush","person","bowl"],
 "related": ["fire hydrant","stop sign","parking meter","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
             "suitcase","frisbee","skis","snowboard","baseball bat","baseball glove","skateboard","surfboard","bottle",
             "wine glass","cup","fork","spoon","banana","orange","broccoli","carrot","hot dog","donut","cake","couch",
             "potted plant","toilet","remote","oven","toaster","sink","refrigerator","vase","scissors","teddy bear",
             "hair drier"],
 "none": ["bench"],
}
lab = {c: k for k, cs in LABEL.items() for c in cs}
assert set(lab) == set(m), set(m) ^ set(lab)
rows = []
for c, v in m.items():
    pool = {d["id"]: d for d in v["candidates"]}
    auto = v["primary"]
    if c in OVERRIDE:
        cid, why = OVERRIDE[c]
        if cid not in pool:  # candidates were truncated to 6 in the file; keep name from the reason
            pool[cid] = {"id": cid, "name": None}
        v["primary_auto"], v["primary"], v["override_reason"] = auto, cid, why
        v["primary_name"] = pool[cid]["name"] or v["primary_name"]
    v["review_label"] = lab[c]
    rows.append((c, v))
json.dump(m, open(dst, "w"), indent=1)
cnt = {k: sum(1 for _, v in rows if v["review_label"] == k) for k in LABEL}
with open(md, "w") as f:
    f.write(f"exact {cnt['exact']} / related {cnt['related']} / none {cnt['none']}; overrides {len(OVERRIDE)}\n\n")
    f.write("| COCO | 主概念 (id) | 判定 | 自动置信度 | 描述数 | 修改 |\n|---|---|---|---|---|---|\n")
    zh = {"exact": "✅ 精确", "related": "🟡 相关/上位", "none": "❌ 无合适概念"}
    for c, v in rows:
        ch = f"原为 {v['primary_auto']}：{v['override_reason']}" if "primary_auto" in v else ""
        f.write(f"| {c} | {v['primary_name']} ({v['primary']}) | {zh[v['review_label']]} | {v['confidence']} | {v['n_captions']} | {ch} |\n")
print(cnt)
