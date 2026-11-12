#%%
"""
The measured YRI1/YRI2 (sister) eigenvalue against theory (supplementary figure).

a  against the exact formula nu_3 = eps_0 + N phi_tilde (tau/tau_bar) Fst
b  against the approximation nu_3 = N phi_tilde Fst (assumes eps_0 = 0, tau = tau_bar)

One point per design cell (median over replicates), y = x dotted. Each sweep family is one
series, coded by both colour and marker shape. The two phi_tilde families differ in design:
one holds n1 + n2 = 500 with unequal sisters (N = 1000), the other keeps the sisters equal and
grows their share (N varies with phi_tilde).

Reads results/analysis/SisterEigenvalue/summary.csv
Writes results/figures/supplement/EigenvalueAfricanSplit.pdf
"""
import os

import pandas as pd

import genetics_style as gs

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterEigenvalue")

W, H = 130, 72
AX, LEFT, GAP, BOTTOM = 47, 13, 19, 21
NAMES = {"Fst": r"F_{ST}", "N": "N", "H": r"\tilde\phi"}

# (row selector, varied parameter(s), fixed columns, marker, extra note)
SERIES = [
    (lambda s: s.family == "Fst", "Fst", ["N", "H"], "o", None),
    (lambda s: s.family == "N", "N", ["Fst", "H"], "s", None),
    (lambda s: s.design.str.startswith("H_A"), "H", ["Fst", "N"], "^",
     r"$\phi_1 \neq \phi_2$"),
    (lambda s: s.design.str.startswith("H_B"), ("H", "N"), ["Fst"], "v",
     r"$\phi_1 = \phi_2$"),
    (lambda s: s.family == "scatter", None, [], "D", None),
]


def _fmt(v):
    v = float(f"{float(v):.3g}")
    return f"{v:.0f}" if v == int(v) else f"{v:.3g}"


def _series_label(d, varied, fixed_cols, note=None):
    """'varying Fst (N = 1000, phi_tilde = 0.25)'."""
    if varied is None:
        return r"$N$, $\tilde\phi$, $F_{ST}$ all varied"
    varied_cols = [varied] if isinstance(varied, str) else list(varied)
    head = "varying " + " & ".join(rf"${NAMES[c]}$" for c in varied_cols)
    shown = lambda c: gs.individuals(d[c].iloc[0]) if c == "N" else d[c].iloc[0]
    fixed = [rf"${NAMES[c]} = {_fmt(shown(c))}$" for c in ["Fst", "N", "H"] if c in fixed_cols]
    if note:
        fixed.append(note)
    return f"{head} ({', '.join(fixed)})" if fixed else head


def panel(ax, s, x_col, x_label, lim):
    ax.plot(lim, lim, ":", color="0.35", linewidth=0.8, zorder=1, label=r"$y = x$")
    for (select, varied, fixed_cols, marker, note), c in zip(SERIES, gs.CATEGORICAL):
        d = s[select(s)]
        if d.empty:
            continue
        ax.scatter(d[x_col], d.nu_meas, s=14, marker=marker, facecolors="none",
                   edgecolors=c, linewidths=0.7, zorder=3,
                   label=_series_label(d, varied, fixed_cols, note))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(x_label)


def make_figure(analysis_dir=ANALYSIS_DIR):
    gs.setup()
    s = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))
    lo = min(s.nu_analytical.min(), s.nu_pred.min(), s.nu_meas.min()) * 0.8
    hi = max(s.nu_analytical.max(), s.nu_pred.max(), s.nu_meas.max()) * 1.2

    fig = gs.blank_figure(W, H)
    ax_a = gs.axes_mm(fig, LEFT, BOTTOM, AX, AX)
    ax_b = gs.axes_mm(fig, LEFT + AX + GAP, BOTTOM, AX, AX)
    panel(ax_a, s, "nu_analytical",
          r"exact $\nu_3 = \varepsilon_0 + 2N\tilde\phi\,(\tau/\bar\tau)\,F_{ST}$", [lo, hi])
    panel(ax_b, s, "nu_pred", r"approximate $\nu_3 = 2N\tilde\phi F_{ST}$", [lo, hi])
    for ax in (ax_a, ax_b):
        ax.set_ylabel(r"measured $\nu_3$ (median)")

    # y = x keyed inside each panel's empty corner; series in two blocks centred
    # under the panels, on one shared row grid
    handles, _ = ax_a.get_legend_handles_labels()
    for ax in (ax_a, ax_b):
        ax.legend(handles=handles[:1], loc="upper left", handlelength=1.6)
    series = handles[1:]
    left_block = [series[i] for i in (0, 1, 4) if i < len(series)]
    right_block = [series[i] for i in (2, 3) if i < len(series)]
    centers = [LEFT + AX / 2, LEFT + AX + GAP + AX / 2]
    gs.grid_key(fig, [left_block, right_block], centers, BOTTOM - 9.5, pitch_mm=3.2,
                handlelength=1.2, handletextpad=0.3)
    gs.label_panel_mm(fig, "a", 1, H - 1)
    gs.label_panel_mm(fig, "b", LEFT + AX + GAP - 12, H - 1)
    out = gs.save(fig, "EigenvalueAfricanSplit", "supplement")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
