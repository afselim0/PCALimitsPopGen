"""GENETICS figure style: import and call setup() before plotting.

Every figure is built at its final printed size (single column 87.9 mm, double column /
supplement 184.3 mm max) and saved WITHOUT bbox_inches="tight", so the PDF width is the width
on the page and LaTeX does no scaling. Sizes: 7 pt labels, 6 pt ticks and legends, 9 pt bold
lowercase panel letters, Arial throughout.
"""
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.transforms import ScaledTranslation

MM = 1 / 25.4
WIDTH = {"single": 87.9 * MM, "double": 184.3 * MM, "supp": 184.3 * MM}
MAX_HEIGHT = 200 * MM  # leaves room for the caption on a 241 mm text block

LABEL_PT, TICK_PT, LEGEND_PT, PANEL_PT = 7, 6, 6, 9
PLOIDY = 2            # tables count haplotypes; every figure labels diploid individuals


# Okabe-Ito: distinguishable under the common colour-vision deficiencies
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
# Order to use when a figure needs fewer colours (most distinct first)
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#000000", "#F0E442"]
# Two highlighted populations against a grey background (YRI1/YRI2, A/B)
POP1, POP2, BACKGROUND = "#D55E00", "#0072B2", "#C8C8C8"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURE_DIR = os.path.join(_ROOT, "results", "figures")
PNG_DIR = os.path.join(_ROOT, "results", "png_figures")


_SYSTEM_FONTS = ["/usr/share/fonts/msttcorefonts", "/usr/share/fonts/liberation-sans",
                 "/usr/share/fonts/truetype/msttcorefonts",
                 "/usr/share/fonts/truetype/liberation"]


def _register_fonts():
    # matplotlib's font cache does not scan these directories on the cluster
    from matplotlib import font_manager
    for d in _SYSTEM_FONTS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().startswith(("arial", "liberationsans")) and f.lower().endswith(".ttf") \
                    and "black" not in f.lower() and "ariblk" not in f.lower():
                try:
                    font_manager.fontManager.addfont(os.path.join(d, f))
                except Exception:
                    pass


def setup():
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # Arial's old timestamps
    _register_fonts()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        # STIX sans for maths: Arial's italic nu is indistinguishable from v
        "mathtext.fontset": "stixsans",
        "font.size": LABEL_PT, "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT, "legend.title_fontsize": LEGEND_PT,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2, "axes.titlepad": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "axes.unicode_minus": True,
        "image.cmap": "viridis",
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3, "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
        "legend.columnspacing": 1.0,
        "errorbar.capsize": 1.5,
        "patch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, keep text editable
        "savefig.dpi": 600, "savefig.bbox": None, "savefig.pad_inches": 0,
        "figure.constrained_layout.use": False,
    })


def figure(width="single", height_mm=None, **kw):
    w = WIDTH[width] if isinstance(width, str) else width * MM
    h = (height_mm * MM) if height_mm else w * 0.75
    if h > MAX_HEIGHT + 1e-9:
        raise ValueError(f"height {h/MM:.0f} mm exceeds {MAX_HEIGHT/MM:.0f} mm")
    return plt.subplots(figsize=(w, h), **kw)


def blank_figure(width_mm, height_mm):
    """An empty figure of exact size; place axes with axes_mm."""
    if height_mm * MM > MAX_HEIGHT + 1e-9:
        raise ValueError(f"height {height_mm:.0f} mm exceeds {MAX_HEIGHT/MM:.0f} mm")
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def axes_mm(fig, left, bottom, width, height, **kw):
    """Add axes at an absolute position in mm from the figure's lower-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def label_panels(axes, labels="abcdefghijklmnop", x=-0.18, y=1.04):
    """Bold lowercase panel labels at the upper left of each axes."""
    for ax, lab in zip(list(axes), labels):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=PANEL_PT,
                fontweight="bold", va="bottom", ha="left")


def label_panel_mm(fig, letter, x_mm, y_mm):
    """A panel label with its top-left corner at an absolute figure position (mm)."""
    W, H = fig.get_size_inches() / MM
    fig.text(x_mm / W, y_mm / H, letter, fontsize=PANEL_PT, fontweight="bold",
             ha="left", va="top")


def seq_colors(n, cmap=cm.viridis, lo=0.0, hi=0.85):
    """n colours along a sequential map, stopping short of viridis' pale yellow end."""
    if n == 1:
        return [cmap((lo + hi) / 2)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def chain_color(deme, K, cmap=cm.viridis):
    """Colour of deme `deme` on a 1D chain of K demes."""
    return cmap(deme / max(K - 1, 1))


# Bivariate colours for a K x K grid: bilinear blend of four Okabe-Ito corners, which (unlike
# the previous blue/red/green/yellow set) stay apart under deuteranomaly. RGB blending keeps
# colour distance monotone in grid distance along both axes, with no hue wraparound.
_CORNER_00 = np.array(colors.to_rgb("#0072B2"))   # blue
_CORNER_0K = np.array(colors.to_rgb("#D55E00"))   # vermillion
_CORNER_K0 = np.array(colors.to_rgb("#009E73"))   # bluish green
_CORNER_KK = np.array(colors.to_rgb("#F0E442"))   # yellow
GRID_CORNERS = ["#0072B2", "#D55E00", "#009E73", "#F0E442"]


def grid_color(row, col, K):
    """Bilinear-interpolated colour for grid position (row, col) in a K x K grid."""
    u = col / max(K - 1, 1)
    v = row / max(K - 1, 1)
    top = _CORNER_00 * (1 - u) + _CORNER_0K * u
    bottom = _CORNER_K0 * (1 - u) + _CORNER_KK * u
    return tuple(top * (1 - v) + bottom * v)


def save(fig, name, section):
    """Write results/figures/<section>/<name>.pdf and a PNG preview, at exact size.

    No bbox_inches="tight": it changes the width. section is "main" or "supplement".
    """
    pdf_dir = os.path.join(FIGURE_DIR, section)
    png_dir = os.path.join(PNG_DIR, section)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    pdf = os.path.join(pdf_dir, f"{name}.pdf")
    png = os.path.join(png_dir, f"{name}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    return pdf


TITLE_H = 2.8   # mm reserved above a row key for its title


def _legend_width_mm(fig, leg):
    fig.canvas.draw()
    return leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / MM


KEY_STAGES = ((1.0, 1.2, 0.4), (0.6, 0.9, 0.35), (0.4, 0.7, 0.3), (0.3, 0.55, 0.25))


def row_key(ax, handles, labels, below_mm, title=None, max_width_mm=None, stage=None, **kw):
    """
    One-row legend centred under `ax` (title centred above the row), `below_mm` below the
    axes bottom. Unless `stage` is given, tightens spacing through KEY_STAGES until the row
    fits `max_width_mm` (default: axes width + 2 mm); it never wraps. Returns (legend, stage)
    so sibling panels can be redrawn at one common spacing.
    """
    fig = ax.figure
    H = fig.get_size_inches()[1] / MM
    ax_h = ax.get_position().height * H
    limit = max_width_mm or ax.get_position().width * fig.get_size_inches()[0] / MM + 2.0
    stages = range(len(KEY_STAGES)) if stage is None else [stage]
    for k in stages:
        spacing, hl, pad = KEY_STAGES[k]
        # anchored at the row's centre line, so rows with tall labels still line up
        row_mid = below_mm + (TITLE_H if title else 0) + 1.4
        opts = dict(loc="center", bbox_to_anchor=(0.5, -row_mid / ax_h),
                    borderaxespad=0, borderpad=0, ncol=len(handles), columnspacing=spacing,
                    handlelength=hl, handletextpad=pad)
        opts.update(kw)
        leg = ax.legend(handles, labels, **opts)
        if stage is not None or _legend_width_mm(fig, leg) <= limit:
            break
    if title:
        # fixed baseline, so titles with and without descenders sit on one line
        ax.text(0.5, -(below_mm + TITLE_H - 0.8) / ax_h, title, transform=ax.transAxes,
                ha="center", va="baseline", fontsize=LEGEND_PT)
    return leg, k


def matched_row_keys(axes, entries, below_mm, titles):
    """row_key on several panels at the tightest spacing any one of them needs."""
    k = max(row_key(ax, h, l, below_mm, title="x")[1] for ax, (h, l) in zip(axes, entries))
    for ax in axes:
        for t in list(ax.texts):
            if t.get_text() == "x":
                t.remove()
    return [row_key(ax, h, l, below_mm, title=t, stage=k)[0]
            for ax, (h, l), t in zip(axes, entries, titles)]


def column_key(ax, handles, below_mm, **kw):
    """A single-column legend centred under `ax` (added as an extra artist)."""
    fig = ax.figure
    ax_h = ax.get_position().height * fig.get_size_inches()[1] / MM
    opts = dict(loc="upper center", bbox_to_anchor=(0.5, -below_mm / ax_h),
                borderaxespad=0, handlelength=2.0)
    opts.update(kw)
    from matplotlib.legend import Legend
    leg = Legend(ax, handles, [h.get_label() for h in handles], **opts)
    ax.add_artist(leg)
    return leg


def grid_key(fig, columns, centers_mm, top_mm, pitch_mm=3.0, **kw):
    """
    Legend blocks laid on one shared row grid: column j is centred on centers_mm[j] (mm from
    the left), its entries left-aligned within it, row i at top_mm - i * pitch_mm. Separate
    one-entry legends, so rows line up across columns even when label heights differ.
    """
    from matplotlib.legend import Legend
    W, H = fig.get_size_inches() / MM
    opts = dict(loc="center left", borderaxespad=0, borderpad=0, handlelength=2.0,
                frameon=False)
    opts.update(kw)
    for handles, cx in zip(columns, centers_mm):
        probe = Legend(fig, handles, [h.get_label() for h in handles], **opts)
        fig.legends.append(probe)
        width = _legend_width_mm(fig, probe)
        fig.legends.remove(probe)
        x0 = cx - width / 2
        for i, h in enumerate(handles):
            leg = Legend(fig, [h], [h.get_label()],
                         bbox_to_anchor=(x0 / W, (top_mm - (i + 0.5) * pitch_mm) / H),
                         **opts)
            fig.legends.append(leg)


def individuals(n):
    """A stored sample count (haplotypes) as the number of individuals it is drawn as."""
    return n / PLOIDY
