#%%
import os

from matplotlib import font_manager, rcParams
from cycler import cycler
okabe_ito = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#E69F00',  # orange
    '#CC79A7',  # pink
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#000000',  # black
]

# ---------------------------------------------------------------------------
# Canonical type sizes (points). ONE table for every figure in the manuscript.
#
# These are absolute sizes for the final 6.5 in page and are never rescaled by
# figure size. Every figure uses the same number for the same ROLE, so titles
# match titles and legends match legends across the whole set.
#
# Sized for the page, not for the screen: at 6.5 in a two-panel row gives each
# panel ~2.6 in of width, and a side legend at 11 pt is physically wider than
# half of that -- it cannot be placed without either crushing the axes into a
# rectangle or landing on top of them. 8 pt is the largest that leaves a square
# panel with its legend beside it.
# ---------------------------------------------------------------------------
PANEL_LETTER_PT = 12   # the bold A, B, C
TITLE_PT        = 10   # panel titles
LABEL_PT        = 8.5  # x and y axis labels
TICK_PT         = 7
LEGEND_PT       = 10   # every legend: side keys, in-axes keys, the shared mark key
ANNOT_PT        = 9    # in-axes annotation (tick-row labels, schematic captions)


_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _register_bundled_fonts(font_dir=_FONT_DIR):
    """
    Add the bundled STIX Two faces to matplotlib's font list.

    Bundled in-repo rather than assumed installed: STIX Two ships with no Linux distro and
    is not on this cluster, so a figure built elsewhere would silently fall back to a
    different face and change every metric on the page. Missing files are not an error --
    font.serif falls through to STIXGeneral, then DejaVu Serif.
    """
    if not os.path.isdir(font_dir):
        return
    for name in sorted(os.listdir(font_dir)):
        if name.lower().endswith((".ttf", ".otf")):
            try:
                font_manager.fontManager.addfont(os.path.join(font_dir, name))
            except Exception:
                pass


def setup_plotting_style():
    # -- Output / vector settings --
    rcParams['savefig.format'] = 'pdf'
    rcParams['pdf.fonttype'] = 42       # embed fonts as actual text, not paths (editable/searchable in PDF)
    rcParams['ps.fonttype'] = 42
    rcParams['svg.fonttype'] = 'none'   # keep text as text if exporting SVG
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.02
    rcParams['savefig.transparent'] = False
    rcParams['figure.dpi'] = 500        # for on-screen preview only; irrelevant for vector export
    rcParams['savefig.dpi'] = 'figure'  # dpi setting only matters if you rasterize elements

    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    rcParams['axes.linewidth'] = 0.6    # 2pt reads as heavy/cartoonish at print size; 1–1.3 is crisper
    rcParams['axes.labelpad'] = 4
    rcParams['axes.titlepad'] = 6
    rcParams['axes.unicode_minus'] = False  # avoids a missing-glyph minus sign with STIX in some cases

    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 0.6
    rcParams['ytick.major.width'] = 0.6
    rcParams['xtick.major.size'] = 2.5
    rcParams['ytick.major.size'] = 2.5
    rcParams['xtick.minor.visible'] = False
    rcParams['ytick.minor.visible'] = False
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.labelsize'] = TICK_PT
    rcParams['ytick.labelsize'] = TICK_PT

    # -- Fonts (match the LaTeX document) --
    # STIX Two Text is the modern STIX cut and the closest free match to a 12 pt Computer
    # Modern / Times-family body. text.usetex is NOT used: no LaTeX binary is installed
    # here (no latex/dvipng on PATH), and mathtext with the stix fontset renders the same
    # glyph shapes without the toolchain dependency. See _register_bundled_fonts.
    _register_bundled_fonts()
    rcParams['font.size'] = LABEL_PT
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif']
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.rm'] = 'STIX Two Text'
    rcParams['mathtext.it'] = 'STIX Two Text:italic'
    rcParams['mathtext.bf'] = 'STIX Two Text:bold'
    # mathtext 'custom' consults every generic family, so set the ones matplotlib would
    # otherwise warn about falling back on.
    rcParams['mathtext.sf'] = 'STIX Two Text'
    rcParams['mathtext.tt'] = 'STIX Two Text'
    rcParams['mathtext.cal'] = 'STIX Two Text:italic'
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    rcParams['axes.titlesize'] = TITLE_PT
    rcParams['axes.labelsize'] = LABEL_PT

    # -- Lines and markers --
    rcParams['lines.linewidth'] = 0.9
    rcParams['lines.markersize'] = 3.5
    rcParams['lines.markeredgewidth'] = 0.8
    rcParams['lines.solid_capstyle'] = 'round'

    # -- Bars --
    rcParams['patch.linewidth'] = 0.8
    rcParams['patch.edgecolor'] = 'black'
    rcParams['patch.force_edgecolor'] = True   # gives bars a defined edge instead of looking flat/plastic

    # -- Legend --
    rcParams['legend.frameon'] = False
    rcParams['legend.fontsize'] = LEGEND_PT
    rcParams['legend.title_fontsize'] = LEGEND_PT
    rcParams['legend.handlelength'] = 1.2
    rcParams['legend.labelspacing'] = 0.25
    rcParams['legend.borderaxespad'] = 0.4
    rcParams['legend.handletextpad'] = 0.5
    rcParams['legend.borderpad'] = 0.3
    rcParams['legend.columnspacing'] = 1.0

    # -- Color cycle (colorblind-safe, works for lines/bars/scatter) --
    # Based on Okabe-Ito palette — distinguishable in grayscale and for common CVD types
    rcParams['axes.prop_cycle'] = cycler(color=okabe_ito)

    # -- Figure size --
    rcParams['figure.figsize'] = (6, 4)  # 6x6 is unusually square for most single-panel figures; 3:2 reads better
    rcParams['figure.constrained_layout.use'] = False  # Turn auto-spacing off for more control

    # -- Errorbars (common in scientific figures) --
    rcParams['errorbar.capsize'] = 3

setup_plotting_style()
# %%