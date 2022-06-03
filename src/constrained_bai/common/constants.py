import os
from collections import defaultdict
from matplotlib.markers import MarkerStyle

PLOTS_FOLDER = "plots"

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
PLOTS_PATH = os.path.join(BASE_PATH, PLOTS_FOLDER)


# source: https://gist.github.com/thriveth/8560036
color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#e41a1c",
    "#dede00",
    "#999999",
    "#f781bf",
    "#a65628",
    "#984ea3",
]

hatch_cycle = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

marker_cycle = ["*", "^", "o", "x", "d", "+", "v", "."]


PLOT_COLORS = {
    "uniform": "#4daf4a",
    "adaptive_uniform_all": "#44aa99",
    "adaptive_uniform_all_tuned": "#ff7f00",
    "g-allocation": "#984ea3",
    "adaptive_maxvar_all": "#332288",
    "adaptive_maxvar_all_tuned": "#f781bf",
    "adaptive_static": "#e41a1c",
    "adaptive": "#377eb8",
    "adaptive_tuned": "#a65628",
    "adaptive_uniform": "#999933",
    "adaptive_uniform_tuned": "#ddcc77",
    "adaptive_greedy_reward_feasible": "#555555",
    "adaptive_greedy_reward_feasible_tuned": "#555555",
    "adaptive_greedy_reward_uncertain": "#dede00",
    "adaptive_greedy_reward_uncertain_tuned": "#555555",
    "oracle": "#888888",
}
PLOT_COLORS = defaultdict(lambda: "blue", PLOT_COLORS)


PLOT_MARKERS = {
    "uniform": MarkerStyle(marker="s", fillstyle="none"),
    "adaptive_uniform_all": MarkerStyle(marker="s", fillstyle="none"),
    "adaptive_uniform_all_tuned": MarkerStyle(marker="s", fillstyle="none"),
    "g-allocation": MarkerStyle(marker="^", fillstyle="none"),
    "adaptive_maxvar_all": MarkerStyle(marker="^", fillstyle="none"),
    "adaptive_maxvar_all_tuned": MarkerStyle(marker="^", fillstyle="none"),
    "adaptive_static": MarkerStyle(marker="o", fillstyle="none"),
    "adaptive": MarkerStyle(marker="o", fillstyle="none"),
    "adaptive_tuned": MarkerStyle(marker="o", fillstyle="none"),
    "adaptive_uniform": "x",
    "adaptive_uniform_tuned": "x",
    "adaptive_greedy_reward_uncertain": MarkerStyle(marker="v", fillstyle="none"),
    "adaptive_greedy_reward_uncertain_tuned": MarkerStyle(marker="v", fillstyle="none"),
    "oracle": MarkerStyle(marker="d", fillstyle="none"),
}
PLOT_MARKERS = defaultdict(lambda: ".", PLOT_MARKERS)

PLOT_LINESTYLE = {
    "uniform": "solid",
    "adaptive_uniform_all": "dotted",
    "adaptive_uniform_all_tuned": "dashed",
    "g-allocation": "solid",
    "adaptive_maxvar_all": "dotted",
    "adaptive_maxvar_all_tuned": "dashed",
    "adaptive_static": "solid",
    "adaptive": "dotted",
    "adaptive_tuned": "dashed",
    "adaptive_uniform": "dotted",
    "adaptive_uniform_tuned": "dashed",
    "oracle": "solid",
}
PLOT_LINESTYLE = defaultdict(lambda: "-", PLOT_LINESTYLE)

PLOT_ALPHA = {
    "adaptive_static": 1.0,
    # "adaptive_tuned": 1.0,
    # "adaptive": 1.0,
}
PLOT_ALPHA = defaultdict(lambda: 0.9, PLOT_ALPHA)

PLOT_ZORDER = {
    "adaptive_static": 2,
    # "adaptive_tuned": 2,
    # "adaptive": 2,
}
PLOT_ZORDER = defaultdict(lambda: 1, PLOT_ZORDER)
