import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter

# Marker styles for different codes
marker_styles = {
    'gross': 'o',
    'bacon': 's',
    'hh': '^',
    'surface': 'D',
    'steane': 'X',
    'color': '*',
    'other': ''
}

# Code display names
code_rename_map = {
    'bacon': 'Bacon-Shor',
    'hh': 'Heavy-hex',
    'gross': 'Gross'
}

# Error type mapping
error_type_map = {
    'Constant': 'constant',
    'SI1000': 'modsi1000'
}

decoder_map = {
    'mwpm': 'MWPM',
    'bposd_batch': 'BP-OSD Batch',
    'bposd_chk': 'BP-OSD PCM',
    'bposd': 'BP-OSD',
    'chromobius': 'Chromobius'
}

# Backend display names
backend_rename_map = {
    "real_willow": "Willow",
    "real_willow_ns": "Willow",
    "real_infleqtion": "Infleqtion (w/ s.)",
    "real_nsinfleqtion": "Infleqtion",
    "real_nsapollo": "Apollo (w/o s.)",
    "real_apollo": "Apollo",
    "real_flamingo": "DQC",
    "real_flamingo_1_qpu": "1 QPU",
    "real_nighthawk": "DQC",
    "real_nighthawk_1_qpu": "1 QPU"
}

# Color palette and hatches for bars
code_palette = sns.color_palette("pastel", n_colors=6)
code_hatches = ["/", "\\", "//", "++", "xx", "**"]

# Figure sizes and font
WIDE_FIGSIZE = 6
HEIGHT_FIGSIZE = 2.2
FONTSIZE = 12
BAR_WIDTH = 0.2  # constant bar width for consistency with size plot
group_spacing = 0.4

# Highlight backend sizes for specific codes and error types
HIGHLIGHT = {
    ('surface', 'Constant'): [400, 500],
    ('hh', 'Constant'): [450],
    ('color', 'Constant'): [400, 500],
    ('bacon', 'Constant'): [400, 450],
    ('surface', 'SI1000'): [400, 500],
    ('hh', 'SI1000'): [450],
    ('color', 'SI1000'): [400, 500],
    ('bacon', 'SI1000'): [400, 450],
}


# Matplotlib rcParams settings
tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Font sizes
    "axes.labelsize": FONTSIZE,
    "font.size": FONTSIZE,
    "legend.fontsize": FONTSIZE - 2,
    "xtick.labelsize": FONTSIZE - 2,
    "ytick.labelsize": FONTSIZE - 2,
    "axes.titlesize": 10,
    # Line and marker styles
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": "black",
    # Error bar cap size
    "errorbar.capsize": 3,
}

plt.rcParams.update(tex_fonts)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42