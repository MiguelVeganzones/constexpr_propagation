import matplotlib.pyplot as plt
import matplotlib as mpl

root_style = {
    "figure.figsize": (7, 5),

    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 16,

    # Axes
    "axes.linewidth": 1.5,
    "axes.labelsize": 18,
    "axes.titlesize": 18,

    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,

    # Lines
    "lines.linewidth": 2,

    # Legend
    "legend.frameon": False,
    "legend.fontsize": 14,

    # No grid
    "axes.grid": False,

    # Save quality
    "savefig.dpi": 300,
}

mpl.rcParams.update(root_style)
