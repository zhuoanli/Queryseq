"""QuerySeq and every baseline it is compared against, as one architecture.

The whole comparison rests on isolating *where the gate comes from*.  So the
encoder, the fusion, the prediction head, the optimiser and the training
schedule are identical across methods, and the only thing that changes is what
the routing gate is allowed to depend on:

    mode            gate g_isc depends on          answers
    ------------    ---------------------------    ------------------------
    uniform         nothing (all available)        mean fusion reference
    fixed           a preset subset                fixed-subset policies
    random          a random subset per study      does the budget alone explain it
    patient         patient evidence v_is          is the query needed at all
    finding         query identity only            is it just a lookup table
    label_id        v_is and a learned per-finding is text semantics needed
                    embedding (no text)
    queryseq        v_is and the query text        the method

Giving the baselines a weaker gate but the same capacity elsewhere is the point:
if QuerySeq wins, it wins because of the conditioning, not because it had a
bigger network or a better learning rate.

Gates are sigmoids, not a softmax over inputs.  A softmax sums to one no matter
how many inputs it reads, which makes "read fewer inputs" unexpressible and a
cost penalty meaningless; independent sigmoid gates let the model genuinely
abstain from an input and make sum_s g_isc a real budget.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ["uniform", "fixed", "random", "patient", "finding", "label_id",
         "queryseq"]


def mlp(i, h, o, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(),
                         nn.Dropout(p), nn.Linear(h, o))


class QuerySeq(nn.Module):
    def __init__(self, dim_img=1408, dim_txt=768, d=256, n_seq=4, n_find=37,
                 mode="queryseq", tau=1.0, p=0.1, query_head=True):
        super().__init__()
        assert mode in MODES, mode
        self.mode, self.n_seq, self.d, self.tau = mode, n_seq, d, tau
        # When the head is query-conditioned it can already specialise its
        # readout per finding, which may make the gate redundant.  Setting
        # query_head=False replaces it with an independent per-finding readout,
        # so the *only* path by which the query can influence the prediction is
        # the router.  That is the clean test of whether routing carries
        # finding-specific information.
        self.query_head = query_head
        self.out_w = nn.Embedding(n_find, d)
        self.out_b = nn.Embedding(n_find, 1)
        self.enc = mlp(dim_img, 2 * d, d, p)
        self.qenc = mlp(dim_txt, 2 * d, d, p)
        self.seq_bias = nn.Parameter(torch.zeros(n_seq))
        # per-finding learned embedding: the ablation that asks whether the
        # text carries anything a free per-finding vector could not learn
        self.id_emb = nn.Embedding(n_find, d)
        # patient-only gate: a score per input from the image alone
        self.pgate = nn.Linear(d, 1)
        # finding-only gate: an unconstrained lookup table, by construction
        self.fgate = nn.Parameter(torch.zeros(n_find, n_seq))
        self.head = mlp(3 * d, 2 * d, 1, p)

    def gates(self, V, Q, avail, fixed_mask=None, gen=None):
        """g of shape (B, n_seq, F), already masked to available inputs."""
        B, S, _ = V.shape
        Fn = Q.shape[0]
        H = self.enc(V)                                     # (B, S, d)
        if self.mode == "uniform":
            e = torch.zeros(B, S, Fn, device=V.device)
        elif self.mode == "fixed":
            m = fixed_mask.to(V.device).view(1, S, 1).float()
            e = (m - 0.5) * 20.0                            # hard on/off
            e = e.expand(B, S, Fn)
        elif self.mode == "random":
            m = fixed_mask.to(V.device).float()             # (B, S) per study
            e = ((m - 0.5) * 20.0).unsqueeze(-1).expand(B, S, Fn)
        elif self.mode == "patient":
            e = self.pgate(H) + self.seq_bias.view(1, S, 1)  # (B, S, 1)
            e = e.expand(B, S, Fn)
        elif self.mode == "finding":
            e = self.fgate.t().view(1, S, Fn).expand(B, S, Fn)
        else:
            K = self.id_emb.weight if self.mode == "label_id" else self.qenc(Q)
            e = torch.einsum("bsd,fd->bsf", H, F.normalize(K, dim=-1))
            e = e / self.tau + self.seq_bias.view(1, S, 1)
        g = torch.sigmoid(e) * avail.unsqueeze(-1)
        return g, H

    def fuse_predict(self, g, H, Q):
        # NOTE: the normalisation below makes predictions invariant to a
        # uniform rescaling of all gates, so (a) the training cost penalty on
        # raw g induces shrinkage, not sparsity, and (b) any thresholded
        # count of g is not a measure of inputs used.  Hard top-k masking is
        # the only mechanism that actually removes inputs.  A reviewer's
        # audit traced a published "collapse to chance" number to exactly
        # this confusion.
        z = torch.einsum("bsf,bsd->bfd", g, H)
        z = z / (g.sum(1).unsqueeze(-1) + 1e-6)             # (B, F, d)
        if not self.query_head:
            w = self.out_w.weight.unsqueeze(0)               # (1, F, d)
            return (z * w).sum(-1) + self.out_b.weight.t()
        q = self.qenc(Q).unsqueeze(0).expand(z.shape[0], -1, -1)
        return self.head(torch.cat([z, q, z * q], -1)).squeeze(-1)

    def forward(self, V, Q, avail, fixed_mask=None, topk=None):
        g, H = self.gates(V, Q, avail, fixed_mask)
        if topk is not None:
            g = hard_topk(g, avail, topk)
        return self.fuse_predict(g, H, Q), g


def hard_topk(g, avail, k):
    """Keep the k highest-gated available inputs, straight-through.

    Evaluation at an exact budget has to be a real selection, not a soft
    reweighting that still peeks at every input.  The forward pass uses the hard
    mask; the backward pass sees the soft gate so the router still trains.
    """
    B, S, Fn = g.shape
    gm = g.masked_fill(avail.unsqueeze(-1) == 0, -1e9)
    kk = int(min(k, S))
    idx = gm.topk(kk, dim=1).indices
    hard = torch.zeros_like(g).scatter_(1, idx, 1.0) * avail.unsqueeze(-1)
    return hard + (g - g.detach()) * hard


def budget_of(g, thresh=0.5):
    """Average number of inputs a gate actually reads."""
    return (g > thresh).float().sum(1).mean().item()


def loss_fn(logit, y, teacher, g, valid, alpha, beta, pos_weight):
    """Classification + full-to-subset consistency + input cost.

    The consistency term is what lets one model serve every availability
    pattern: a masked view is asked to reproduce what the same model concluded
    with all inputs present, so subsets inherit the full-input decision boundary
    instead of each learning their own.
    """
    bce = F.binary_cross_entropy_with_logits(
        logit, y, pos_weight=pos_weight, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    out = {"bce": bce}
    total = bce
    if alpha > 0 and teacher is not None:
        p_t = torch.sigmoid(teacher.detach())
        cons = F.binary_cross_entropy_with_logits(
            logit, p_t, reduction="none")
        cons = (cons * valid).sum() / valid.sum().clamp(min=1)
        out["cons"] = cons
        total = total + alpha * cons
    if beta > 0:
        cost = g.sum(1).mean()
        out["cost"] = cost
        total = total + beta * cost
    out["total"] = total
    return total, out


def sample_avail(B, n_seq, rng, device, min_k=1):
    """Random availability pattern per study, drawn over all 2^n - 1 subsets.

    Training against every pattern rather than a fixed dropout rate is what
    makes the any-subset evaluation legitimate -- no subset is out of
    distribution at test time.
    """
    while True:
        m = torch.from_numpy(
            (rng.random((B, n_seq)) < 0.5).astype(np.float32)).to(device)
        bad = m.sum(1) < min_k
        if not bad.any():
            return m
        m[bad] = 1.0
        return m
