"""M0 diagnostics: how much of a prediction goes through Steerling's named concepts.

Both measures are taken at MASKED target positions (the only positions an MDM is trained to predict).

1. Logit contribution (exact): logit_y = W_y.known_features + W_y.unk_hat + W_y.epsilon
   (composed == hidden thanks to the epsilon correction, so the three terms sum to the logit).
2. SIM symmetry-II violation: with h = final hidden state at the target position, x = input embeddings
   of the conditioning context (image tokens, or the reference-caption tokens),
       g_f = d(W_y . h)/dx,     g_k = d(w_k . h)/dx   for the top-K known concepts at that position
   (w_k = known-head predictor row; concept logit c_k = w_k . h). violation(K) = ||g_f - P_K g_f||^2 / ||g_f||^2,
   P_K = orthogonal projection onto span{g_k}. Low violation = the prediction's dependence on the context
   is expressible through the named concepts. Random-K concepts give the chance level at every K; the
   headline metric is explained-beyond-chance(K) = 1 - violation_top(K) / violation_random(K).
   Raw violations are confounded by context size (576 image tokens vs ~12 caption tokens), hence the
   size-matched control (see sim_violation's matched_m).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def logit_contributions(model, x: torch.Tensor, positions: list[int], targets: list[int]) -> dict:
    """x: [1, T, D] input embeddings. Returns per-target contributions and log-probs."""
    logits, o = model(None, input_embeds=x, minimal_output=False)
    pos = torch.tensor(positions, device=x.device)
    y = torch.tensor(targets, device=x.device)
    W = model.transformer.lm_head.weight[y].float()
    kf, uh, eps = (t[0, pos].float() for t in (o.known_features, o.unk_hat, o.epsilon))
    kc, dc, ec = (W * kf).sum(-1), (W * uh).sum(-1), (W * eps).sum(-1)
    lp = torch.log_softmax(logits[0, pos].float(), -1).gather(-1, y[:, None])[:, 0]
    return {
        "known_c": kc.tolist(), "disc_c": dc.tolist(), "eps_c": ec.tolist(),
        "logit": logits[0, pos].float().gather(-1, y[:, None])[:, 0].tolist(),
        "logp": lp.tolist(),
        "known_topk_ids": o.known_topk_indices[0, pos].tolist(),
        "known_topk_w": torch.sigmoid(o.known_topk_logits[0, pos].float()).tolist(),
    }


def sim_violation(model, x_parts: list[torch.Tensor], ctx_idx: int, position: int, target: int,
                  ks: tuple[int, ...] = (8, 32, 128, 256), chunk: int = 16, seed: int = 0,
                  matched_m: int | None = None) -> dict:
    """x_parts: embedding pieces [1, t_i, D] concatenated along T; x_parts[ctx_idx] is the context.

    Returns violation(K) for the top-K active concepts and for K random concepts (nested sets, so one
    QR per set gives every K). If matched_m is given, the same quantities are also computed after
    restricting the context to its matched_m tokens with the largest ||g_f|| (size-matched control:
    e.g. m = number of reference-caption tokens in the text condition)."""
    ctx = x_parts[ctx_idx].detach().clone().requires_grad_(True)
    parts = [p.detach() if i != ctx_idx else ctx for i, p in enumerate(x_parts)]
    h = model.transformer(None, input_embeds=torch.cat(parts, 1), return_hidden=True)[0, position]
    Wp = model.known_head.concept_predictor.weight[: model.known_head.n_concepts]
    kmax = max(ks)
    with torch.no_grad():
        top = (Wp.float() @ h.float()).topk(kmax).indices
        g = torch.Generator(device="cpu").manual_seed(seed)
        rnd = torch.randperm(Wp.shape[0], generator=g)[:kmax].to(h.device)

    def vjps(U):
        outs = []
        for i in range(0, U.shape[0], chunk):
            gi, = torch.autograd.grad(h, ctx, grad_outputs=U[i: i + chunk].to(h.dtype),
                                      is_grads_batched=True, retain_graph=True)
            outs.append(gi.reshape(gi.shape[0], -1).float())
        return torch.cat(outs)

    gf = vjps(model.transformer.lm_head.weight[target][None])[0]
    G_top, G_rnd = vjps(Wp[top]), vjps(Wp[rnd])
    del h

    def violation(G, v):
        Q, _ = torch.linalg.qr(G.T)  # orthonormal columns; the first k span the first k rows of G
        out = {}
        for k in ks:
            proj = Q[:, :k] @ (Q[:, :k].T @ v)
            out[k] = ((v - proj).norm() ** 2 / v.norm() ** 2).item()
        return out

    T_ctx, D = ctx.shape[1], ctx.shape[2]
    res = {"gf_norm": gf.norm().item(), "ctx_tokens": T_ctx,
           "top": violation(G_top, gf), "random": violation(G_rnd, gf), "top_ids": top[:32].tolist()}
    if matched_m is not None and matched_m < T_ctx:
        tok_norm = gf.view(T_ctx, D).norm(dim=-1)
        keep = tok_norm.topk(matched_m).indices.sort().values
        cols = (keep[:, None] * D + torch.arange(D, device=keep.device)).reshape(-1)
        gm = gf[cols]
        res["matched"] = {"m": matched_m, "gf_share": (gm.norm() ** 2 / gf.norm() ** 2).item(),
                          "top": violation(G_top[:, cols], gm), "random": violation(G_rnd[:, cols], gm)}
    return res
