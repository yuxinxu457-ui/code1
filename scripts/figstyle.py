"""Shared figure style for the paper's data figures.

Figures are drawn at their printed size (IEEE double column: 7.16 in full width, 3.5 in single column),
so the 8 pt text below is the size the reader sees. Palette (validated for colour-vision deficiency and
contrast): blue = two-layer system / deployed memory, orange = window alone / one-cell code, purple and teal
for further alternatives, greys for ground truth and context.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FULL_W = 7.16      # in, IEEE two-column figure width
COL_W = 3.5        # in, IEEE single-column figure width

BLUE, ORANGE, PURPLE, TEAL = '#2a78d6', '#eb6834', '#7b5ea7', '#2a9d8f'
INK, MUTED, FAINT, GRID = '#1a1a1a', '#5b5a56', '#b9b8b3', '#e8e7e3'
TRUTH = '#9d9c97'          # ground-truth path
SHADE = '#f1f0ec'          # background bands (e.g. laps of loop B)
LIGHT_BLUE = '#cfe0f5'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'axes.labelcolor': INK, 'axes.titlepad': 5, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.8, 'ytick.major.size': 2.8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'lines.linewidth': 1.4, 'lines.solid_capstyle': 'round',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Helvetica Neue', 'mathtext.it': 'Helvetica Neue:italic',
    'mathtext.bf': 'Helvetica Neue:bold', 'mathtext.sf': 'Helvetica Neue',
})


def panel_title(ax, letter, text=''):
    """Bold panel letter followed by a plain descriptive title, left-aligned just above the axes."""
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, 6), textcoords='offset points',
                fontsize=9, fontweight='bold', ha='left', va='bottom', annotation_clip=False)
    if text:
        ax.annotate(text, (0, 1), xycoords='axes fraction', xytext=(11, 6), textcoords='offset points',
                    fontsize=8.5, ha='left', va='bottom', annotation_clip=False)


def save(fig, path_pdf):
    """Save a vector PDF and a 300-dpi PNG next to it."""
    fig.savefig(path_pdf, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(path_pdf.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
