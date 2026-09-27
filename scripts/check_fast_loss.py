"""Check mdm_loss_fast == mdm_loss (full interpretable forward): loss, connector gradients, and speed."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.mdm import build_mdm_batch, mdm_loss, mdm_loss_fast  # noqa: E402
from smm.steerling_io import answer_ids, chat_prompt_ids, load_steerling  # noqa: E402
from smm.vlm import build_connector  # noqa: E402

lm, tok = load_steerling()
mask_id = tok.convert_tokens_to_ids("<|mask|>")
res = {}
for name in ("mlp", "resampler"):
    torch.manual_seed(0)
    proj = build_connector(name, 1024).cuda()
    feats = torch.randn(8, 576, 1024, device="cuda", dtype=torch.bfloat16)
    P = [chat_prompt_ids(tok, "Describe the image.")] * 8
    A = [answer_ids(tok, f"a photo of item number {i} on a wooden table next to a window") for i in range(8)]
    txt, lab, w = build_mdm_batch(P, A, n_prefix=proj.n_tokens, mask_id=mask_id, t_fixed=0.5,
                                  gen=torch.Generator().manual_seed(0))
    lab, w = lab.cuda(), w.cuda()

    def run(fast):
        proj.zero_grad()
        x = torch.cat([proj(feats.float()).to(torch.bfloat16), lm.transformer.tok_emb(txt.cuda())], 1)
        loss = mdm_loss_fast(lm, x, lab, w) if fast else mdm_loss(lm(None, input_embeds=x, minimal_output=True)[0], lab, w)
        loss.backward()
        return loss.item(), torch.cat([p.grad.flatten().float() for p in proj.parameters()])

    (lf, gf), (ls, gs) = run(True), run(False)
    t = {}
    for fast in (True, False):
        for _ in range(2):
            run(fast)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5):
            run(fast)
        torch.cuda.synchronize(); t[fast] = (time.time() - t0) / 5
    res[name] = {"loss_fast": lf, "loss_full": ls, "grad_rel_diff": ((gf - gs).norm() / gs.norm()).item(),
                 "s_per_step_fast": round(t[True], 3), "s_per_step_full": round(t[False], 3),
                 "speedup": round(t[False] / t[True], 2), "seq_len": lab.shape[1]}
    print(name, res[name], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=2)
