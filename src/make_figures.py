"""Every figure in the paper, each from a CSV in results/.

Colour decisions follow the project's dataviz rules and were computed, not
eyeballed (src/validate_palette.py):

  categorical  the five method series use slots checked on the adjacent
               pairlist -- worst CVD dE 17.7, worst normal-vision dE 33.6,
               both well clear of the 8 / 15 floors.  Every series also carries
               a distinct marker and dash pattern, so identity survives
               greyscale printing and is never colour-alone.
  diverging    signed quantities (utility relative to reading every input,
               and synergy) use two hues through a neutral grey midpoint.  A
               sequential ramp here would hide the sign, which is the whole
               point of those panels.
  sequential   routing weights are a magnitude, so one hue, light to dark.

Each figure degrades gracefully when its CSV is absent, so a partial results
directory still builds the paper.
"""
import glob
import json
import os
import time
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import SEQUENCES, robust_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"

METHOD_STYLE = {
    "uniform":  ("#008300", "o", (0, (1, 0)),    "Mean fusion"),
    "finding":  ("#e87ba4", "s", (0, (4, 2)),    "Finding-only gate"),
    "patient":  ("#4a3aa7", "^", (0, (1, 1.6)),  "Patient-only router"),
    "label_id": ("#eb6834", "D", (0, (5, 1, 1, 1)), "Learned finding-ID"),
    "queryseq": ("#2a78d6", "o", (0, (1, 0)),    "QuerySeq"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
DIVERGING = LinearSegmentedColormap.from_list(
    "qs_div", ["#2a78d6", "#8fb9e8", "#e8e8e6", "#f4a883", "#eb6834"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "qs_seq", ["#f4f4f2", "#8fb9e8", "#2a78d6", "#123a68"])

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.axisbelow": True, "legend.frameon": False,
})


def have(name):
    """Delegate to the shared resolver so figures, tables and macros
    cannot disagree about which input set they are drawing."""
    from common import result_path
    return result_path(name)


def _tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_atlas():
    """Figure 1: the utility atlas -- where input value actually lives."""
    f = have("finding_sequence_matrix.csv")
    if not f:
        return
    d = robust_csv(f)
    piv = d.pivot_table(index="finding", columns="subset",
                        values="auroc_mean")
    # The all-inputs column of whatever set is active.  This was pinned to the
    # four-sequence subset, so under k5 the lookup missed, the function
    # returned silently, and the paper kept shipping the *previous* input
    # set's atlas figure while every number around it had moved on.
    full = max(piv.columns, key=lambda c: len(c.split("+")))
    if len(full.split("+")) != len(SEQUENCES):
        raise RuntimeError(
            f"fig_atlas: widest subset in {os.path.basename(f)} is '{full}' "
            f"({len(full.split('+'))} inputs) but the active set has "
            f"{len(SEQUENCES)}. Refusing to draw an atlas for a different "
            f"input set.")
    order = sorted(piv.columns, key=lambda c: (len(c.split("+")), c))
    piv = piv[order]
    npos = d.groupby("finding").n_pos.first()
    # most-affected findings first; a 33-row heatmap is unreadable at column width
    spread = (piv.max(1) - piv.min(1)).sort_values(ascending=False)
    sel = spread.head(18).index
    rel = piv.loc[sel].sub(piv.loc[sel, full], axis=0)

    fig = plt.figure(figsize=(6.6, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.05, 1.05], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    v = float(np.nanmax(np.abs(rel.to_numpy())))
    im = ax.imshow(rel.to_numpy(), cmap=DIVERGING, aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{s[:26]} ({npos[s]})" for s in sel])
    ax.set_title(f"A  Utility relative to reading all {len(SEQUENCES)} inputs",
                 loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    cb.set_label(f"$\\Delta$AUROC vs all {len(SEQUENCES)}", size=6.5)
    cb.ax.tick_params(labelsize=6)
    # mark each finding's best subset
    for r, s in enumerate(sel):
        ax.add_patch(plt.Rectangle(
            (list(order).index(piv.loc[s].idxmax()) - .5, r - .5), 1, 1,
            fill=False, ec=INK, lw=0.9))

    sf = have("shapley.csv")
    if sf:
        sh = robust_csv(sf)
        sp = sh.pivot_table(index="finding", columns="sequence",
                            values="shapley").loc[sel]
        ax2 = fig.add_subplot(gs[1])
        w = float(np.nanmax(np.abs(sp.to_numpy())))
        im2 = ax2.imshow(sp.to_numpy(), cmap=DIVERGING, aspect="auto",
                         norm=TwoSlopeNorm(vcenter=0, vmin=-w, vmax=w))
        ax2.set_xticks(range(sp.shape[1]))
        ax2.set_xticklabels(sp.columns, rotation=90)
        ax2.set_yticks([])
        ax2.set_title("B  Shapley value", loc="left")
        ax2.grid(False)
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.05, pad=0.02)
        cb2.ax.tick_params(labelsize=6)

    yf = have("synergy.csv")
    if yf:
        sy = robust_csv(yf)
        # The empty context is degenerate (|psi| forced toward 0.44 by
        # U(empty)=0.5) and the text promises non-empty contexts only; an
        # earlier version plotted all rows and its left tail was entirely
        # this artefact.
        if "context" in sy.columns:
            sy = sy[sy.context != "none"]
        ax3 = fig.add_subplot(gs[2])
        ax3.axvline(0, color=MUTED, lw=0.7)
        ax3.hist(sy.synergy, bins=40, color="#8fb9e8", edgecolor=INK,
                 linewidth=0.3)
        ax3.set_xlabel("pairwise synergy")
        ax3.set_ylabel("count")
        ax3.set_title("C  Pairwise interactions", loc="left")
        _tidy(ax3)
        frac = float((sy.synergy.abs() > 0.005).mean())
        ax3.annotate(f"{100 * frac:.0f}% of pairs\n$|\\psi|>0.005$",
                     xy=(0.97, 0.93), xycoords="axes fraction",
                     ha="right", va="top", fontsize=6, color=MUTED)
    fig.savefig(f"{FIG}/fig1_atlas.pdf")
    plt.close(fig)


def fig_frontier():
    """Figure 3: accuracy against how many inputs the model actually reads.

    Reads the nqh (independent per-finding readout) run so the frontier
    matches Table 1 and the section-4.4 macros; the shared-head run is a
    different configuration and must not be mixed in.
    """
    f = have("method_macro_nqh.csv")
    if not f:
        return
    d = robust_csv(f)
    d = d[d.axis == "budget"]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.1))
    for ax, metric, lab in zip(
            axes, ["macro_auroc", "worst_finding"],
            ["macro AUROC", "worst-finding AUROC"]):
        for m, (c, mk, ls, lbl) in METHOD_STYLE.items():
            s = d[d.method == m]
            if s.empty:
                continue
            g = s.groupby("budget")[metric]
            mu, sd = g.mean(), g.std()
            ax.plot(mu.index, mu.values, color=c, marker=mk, linestyle=ls,
                    lw=1.4, ms=4, label=lbl, zorder=3)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.12,
                            lw=0)
        # every budget the data holds, explicitly.  This was a hardcoded
        # [1, 2, 3, 4]: the five-input model's k=5 endpoint was plotted but
        # untitled, so the figure read as a four-input relic.
        ax.set_xticks(sorted(d.budget.dropna().unique()))
        ax.set_xlabel("inputs read (budget $k$)")
        ax.set_ylabel(lab)
        _tidy(ax)
    axes[0].legend(loc="lower right", ncol=1)
    fig.savefig(f"{FIG}/fig3_frontier.pdf")
    plt.close(fig)


def fig_routing():
    """Figure 4: routing is finding-specific, and follows meaning not form."""
    from common import SEQSET, SEQ_SHORT, SEQUENCES, Cohort, finding_filter
    # Tagged path, never the bare k4-era file: the untagged
    # gates_queryseq_s0.npy is a four-sequence, 1,477-study matrix, and an
    # earlier version of this function silently drew it under five k5 labels.
    gp = f"{ROOT}/data_local/gates_queryseq_s0_{SEQSET}.npy"
    if not os.path.exists(gp):
        return
    G = np.load(gp).astype(np.float32)               # (n, seq, finding)
    if G.shape[1] != len(SEQUENCES):
        raise RuntimeError(
            f"fig_routing: gates file has {G.shape[1]} sequences but the "
            f"active set has {len(SEQUENCES)} -- refusing to mislabel")
    # A clean supplement extract has the gate array but not the feature
    # caches Cohort() needs, so the finding names / eligibility indices are
    # persisted beside the gates and used as a fallback (a reviewer's
    # clean-extract rebuild failed exactly here).
    sup = f"{ROOT}/data_local/fig_support_{SEQSET}.json"
    try:
        te = Cohort("test", complete_case=True)
        keep = list(finding_filter(te.Y, te.findings, min_pos=20))
        names_all, n_te = list(te.findings), len(te)
        with open(sup, "w") as fh:
            json.dump({"n_test_cc": n_te, "findings": names_all,
                       "keep_minpos20": [int(j) for j in keep]}, fh)
    except Exception:
        with open(sup) as fh:
            meta = json.load(fh)
        keep, names_all, n_te = (meta["keep_minpos20"], meta["findings"],
                                 meta["n_test_cc"])
    if G.shape[0] != n_te:
        raise RuntimeError(
            f"fig_routing: gates file has {G.shape[0]} studies, cohort has "
            f"{n_te}")
    R = G.mean(0).T[keep]                            # (F, seq)
    names = [names_all[j][:26] for j in keep]
    order = np.argsort(-R.std(1))[:18]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8),
                             gridspec_kw={"width_ratios": [1, 1.25],
                                          "wspace": 0.55})
    ax = axes[0]
    im = ax.imshow(R[order], cmap=SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(len(SEQUENCES)))
    ax.set_xticklabels([SEQ_SHORT[s] for s in SEQUENCES], rotation=90)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_title("A  Mean routing weight", loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cb.ax.tick_params(labelsize=5, pad=1)

    ax2 = axes[1]
    q = have("query_diagnostics.csv")
    if q:
        d = robust_csv(q)
        agg = d.groupby("variant").macro_auroc.agg(["mean", "std"])
        para = [i for i in agg.index if i.startswith("para:")]
        ctrl = [i for i in agg.index if i in ("shuffled", "counterfactual")]
        xs = para + ctrl
        cols = ["#2a78d6"] * len(para) + ["#eb6834"] * len(ctrl)
        ax2.bar(range(len(xs)), agg.loc[xs, "mean"],
                yerr=agg.loc[xs, "std"].fillna(0), color=cols,
                edgecolor=INK, linewidth=0.4, capsize=2)
        ax2.set_xticks(range(len(xs)))
        ax2.set_xticklabels([x.replace("para:", "") for x in xs], rotation=45,
                            ha="right")
        ax2.set_ylabel("macro AUROC")
        ax2.set_ylim(0.5, None)
        ax2.set_title("B  Query paraphrase (blue) vs. broken query (orange)",
                      loc="left")
        _tidy(ax2)
    fig.savefig(f"{FIG}/fig4_routing.pdf")
    plt.close(fig)


def fig_external():
    """Figure 5: held-out scanner vendor and field strength."""
    f = have("external_macro.csv")
    if not f:
        return
    d = robust_csv(f)
    da = d[d.axis == "availability"]
    if da.empty:
        return
    shifts = sorted(da["shift"].unique())
    fig, axes = plt.subplots(1, len(shifts), figsize=(7.1, 2.3), squeeze=False)
    for ax, sh in zip(axes[0], shifts):
        s = da[da["shift"] == sh]
        ms = [m for m in METHOD_STYLE if m in set(s.method)]
        x = np.arange(3)
        w = 0.8 / max(len(ms), 1)
        for i, m in enumerate(ms):
            c, _, _, lbl = METHOD_STYLE[m]
            v = s[s.method == m][["macro_full", "macro_avg_subset",
                                  "macro_worst_subset"]]
            ax.bar(x + i * w - 0.4 + w / 2, v.mean(), w * 0.9,
                   yerr=v.std().fillna(0), color=c, label=lbl,
                   edgecolor=INK, linewidth=0.3, capsize=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"all {len(SEQUENCES)}", "avg subset",
                            "worst subset"])
        ax.set_ylabel("macro AUROC")
        ax.set_ylim(0.5, None)
        held = s.held_out.iloc[0]
        ax.set_title(f"held-out {sh}: {held}", loc="left")
        _tidy(ax)
    # a legend inside the axes sat on top of the bars; one shared legend
    # below the panels keeps every bar readable
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.savefig(f"{FIG}/fig5_external.pdf")
    plt.close(fig)


def fig_confound():
    """Availability leaks the label: why the lattice is complete-case."""
    f = have("availability_confound.csv")
    if not f:
        return
    d = robust_csv(f).sort_values("auroc", ascending=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.bar(range(len(d)), d.auroc, color="#eb6834", edgecolor=INK,
           linewidth=0.3)
    ax.set_xticks([])
    ax.set_xlabel(f"findings ({len(d)}), sorted")
    ax.set_ylabel("AUROC from availability alone")
    ax.set_ylim(0.4, None)
    ax.annotate(f"macro {d.auroc.mean():.3f}", xy=(0.97, 0.93),
                xycoords="axes fraction", ha="right", fontsize=6.5,
                color=MUTED)
    _tidy(ax)
    fig.savefig(f"{FIG}/fig2_confound.pdf")
    plt.close(fig)


def fig_optimism():
    """Observed optimism against its permutation distribution, k5 and k7.

    Rebuilt to draw ONLY from the nested null_optimism outputs.  The previous
    version read selection_optimism.csv, a four-sequence-era file, so its
    "real lattice" series predated both the k5 cohort and the strictly nested
    estimator -- a stale-source error of exactly the class this project keeps
    a gate for.
    """
    f = have("null_optimism.csv")
    if not f:
        return
    d = robust_csv(f).sort_values("n_candidates")

    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.fill_between(d.n_candidates, d.null_p05, d.null_p95,
                    color="#eb6834", alpha=0.15, lw=0,
                    label="permutation reference (5--95th pct.)")
    ax.plot(d.n_candidates, d.null_median, color="#eb6834", marker="s",
            linestyle=(0, (4, 2)), lw=1.4, ms=4, label="null median")
    ax.plot(d.n_candidates, d.observed, color="#2a78d6", marker="o",
            linestyle=(0, (1, 0)), lw=1.4, ms=4,
            label="observed (nested)", zorder=4)

    deep = f"{RES}/null_optimism_k7.csv"
    if os.path.exists(deep):
        from common.data import RESULTS_USED
        RESULTS_USED.add("null_optimism_k7.csv")
        k7 = robust_csv(deep).sort_values("n_candidates")
        ax.plot(k7.n_candidates, k7.null_median, color="#4a3aa7", marker="^",
                linestyle=(0, (1, 1.6)), lw=1.4, ms=4,
                label="null median, deep lattice\n($K{=}127$, smaller cohort)")
        ax.fill_between(k7.n_candidates, k7.null_p05, k7.null_p95,
                        color="#4a3aa7", alpha=0.10, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("candidate subsets searched ($K$)")
    ax.set_ylabel("optimism (AUROC)")
    ax.legend(loc="upper left", fontsize=6)
    _tidy(ax)
    fig.savefig(f"{FIG}/fig6_optimism.pdf")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    # "ok" used to mean "the call returned", which a function that returned
    # early on a missing column also satisfies -- so a figure could go stale
    # for weeks while the driver reported success every time.  It now means
    # "this call wrote a file", verified by mtime.
    def snapshot():
        # (mtime, size) rather than mtime alone: Lustre truncates mtime to
        # whole seconds, so a file written moments ago can look older than the
        # wall clock reading taken just before the call.
        return {p: (os.path.getmtime(p), os.path.getsize(p))
                for p in glob.glob(f"{FIG}/*.pdf")}

    before = snapshot()
    failed = []
    for fn in (fig_atlas, fig_confound, fig_frontier, fig_routing,
               fig_optimism, fig_external):
        try:
            fn()
            after = snapshot()
            wrote = sorted(os.path.basename(p) for p in after
                           if before.get(p) != after[p])
            if not wrote:
                raise RuntimeError(
                    f"{fn.__name__} returned without writing a figure -- it "
                    f"almost certainly hit an early return on data from a "
                    f"different input set")
            print(f"  {fn.__name__} wrote {' '.join(wrote)}", flush=True)
            before = after
        except Exception as e:                                # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  {fn.__name__} FAILED: {type(e).__name__}: {e}",
                  flush=True)
    print("figures in", FIG)
    if failed:
        # a reviewer's clean-extract rebuild printed FAILED and still
        # exited 0, so the submission gate treated a broken figure path as
        # success -- any expected-figure failure is now a hard error
        print(f"FAILED figures: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
