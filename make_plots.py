"""
This is a helper script to create plots from sacred results.
Parameter values are hardcoded in the script for now.
"""

import argparse
import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict
from functools import partial
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
from frozendict import frozendict

from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from constrained_bai.common.constants import (
    PLOT_ALPHA,
    PLOT_COLORS,
    PLOT_MARKERS,
    PLOT_LINESTYLE,
    PLOT_ZORDER,
    PLOTS_PATH,
    color_cycle,
    hatch_cycle,
    marker_cycle,
)
from constrained_bai.util.plotting import (
    plot_result_percentiles,
    set_plot_style,
)
from constrained_bai.util.results import FileExperimentResults


def load(results_folder, experiment_label, config_query=None):
    if experiment_label is not None:
        result = subprocess.check_output(
            f"grep '{experiment_label}' -r {results_folder} "
            "| grep config | cut -f 1 -d : | rev | cut -d / -f 2- | rev",
            shell=True,
        ).decode()
        subdirs = result.split("\n")
    else:
        subdirs = [x[0] for x in os.walk(results_folder)]
    experiments = []
    for i, subdir in enumerate(subdirs):
        print(i, subdir)
        try:
            experiment = FileExperimentResults(subdir)
        except Exception as e:
            # print(e)
            continue
        valid = True
        if config_query is not None:
            for key, value in config_query.items():
                if experiment.config[key] != value:
                    # print(f"{key}, {experiment.config[key]}, {value}")
                    valid = False
        if valid:
            experiments.append(experiment)
    return experiments


def extract_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def get_method_label(method):
    return method


def get_dict_mult_key(dictionary, mult_key):
    res = dictionary
    for key in mult_key.split("."):
        res = res[key]
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default=None)
    parser.add_argument("--remote_results_folder", type=str, default=None)
    parser.add_argument("--tmp_folder", type=str, default="/tmp")
    parser.add_argument("--experiment_label", type=str, default=None)
    parser.add_argument("--paper_style", action="store_true")
    parser.add_argument("--forbid_running", action="store_true")
    parser.add_argument("--allow_failed", action="store_true")
    parser.add_argument("--allow_interrupted", action="store_true")
    parser.add_argument("--no_title", action="store_true")
    parser.add_argument("--output_format", type=str, default="pdf")
    parser.add_argument("--xmin", type=int, default=None)
    parser.add_argument("--xmax", type=int, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--vline", type=float, default=None)
    parser.add_argument("--hline", type=float, default=None)
    parser.add_argument("--hlines", type=str, default=None)
    parser.add_argument("--no_markers", action="store_true")
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--stddev", action="store_true")
    parser.add_argument("--stderr", action="store_true")
    parser.add_argument("--no_error", action="store_true")
    parser.add_argument("--xlabel", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--aspect", type=float, default=1.15)
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--extract_legend", action="store_true")
    parser.add_argument("--markers_every", type=int, default=1)
    parser.add_argument("--n_xticks", type=int, default=5)
    parser.add_argument("--use_colors_markers", action="store_true")
    parser.add_argument("--no_yaxis", action="store_true")
    parser.add_argument("--fix_axis_size", action="store_true")
    parser.add_argument("--legend_pos", type=str, default="upper_left")
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--independent_variable", type=str, default=None)
    parser.add_argument("--int_xaxis", action="store_true")
    parser.add_argument("--anytime", action="store_true")
    parser.add_argument("--barchart", action="store_true")
    parser.add_argument("--method_key", type=str, default="method")
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--no_replace", action="store_true")
    parser.add_argument("--xtick_rotation", type=int, default=-90)
    return parser.parse_args()


def get_next_style(color, hatch, marker):
    color = (color + 1) % len(color_cycle)
    hatch = (hatch + 1) % len(hatch_cycle)
    marker = (marker + 1) % len(marker_cycle)
    return color, hatch, marker


def main():
    args = parse_args()

    if args.results_folder is None and args.remote_results_folder is None:
        raise ValueError("results_folder or remote_results_folder has to be given")

    if args.remote_results_folder is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tmp_result_folder = os.path.join(args.tmp_folder, timestamp)
        subprocess.run(
            [
                "rsync",
                "-av",
                "-e ssh",
                "--exclude",
                "*.txt",
                "--exclude",
                "*.pkl",
                args.remote_results_folder,
                tmp_result_folder,
            ]
        )
        args.results_folder = tmp_result_folder

    if args.paper_style:
        set_plot_style()

    print(f"Loading '{args.experiment_label}' from '{args.results_folder}'.")

    # list exps
    all_experiments = load(
        args.results_folder, experiment_label=args.experiment_label, config_query=None
    )
    print("Found {} experiments.".format(len(all_experiments)))

    experiments_by_exp_method = dict()

    if args.metrics is not None:
        metrics = args.metrics.split(",")
    elif args.anytime:
        metrics = ["cos_similarity", "regret", "constraint"]
    else:
        metrics = ["iterations", "correct", "constraint_violation", "runtime"]

    methods = set()
    results_dict = dict()

    for experiment in all_experiments:
        if (
            args.forbid_running
            and experiment.status == "RUNNING"
            or ((not args.allow_failed) and experiment.status == "FAILED")
            or ((not args.allow_interrupted) and experiment.status == "INTERRUPTED")
        ):
            print("Ignoring experiment with status:", experiment.status)
        elif experiment.status != "COMPLETED":
            # hacky modification because current experiment implementation needs things to finish
            print("Warning: Excluding experiment with status", experiment.status)
        else:
            if "experiment" in experiment.config:
                exp = experiment.config["experiment"]
            else:
                exp = None
            if exp is None or exp["instance"] == args.instance:
                method = get_dict_mult_key(experiment.config, args.method_key)
                method = str(method)
                methods.add(method)

                for metric in metrics:
                    if args.anytime:
                        if metric == "regret":
                            steps, values = experiment.get_metric("reward")
                            values = experiment.info["best_reward"] - values
                        else:
                            steps, values = experiment.get_metric(metric)

                        if metric not in results_dict.keys():
                            results_dict[metric] = dict()
                        if method not in results_dict[metric]:
                            results_dict[metric][method] = dict()
                        min_xval = np.min(steps)
                        max_xval = np.max(steps)

                        res = dict()
                        for xval, yval in zip(steps, values):
                            res[xval] = yval

                        for xval in range(50):
                            yval = None
                            if xval in res:
                                yval = res[xval]
                            elif xval > max_xval:
                                if metric == "regret" or metric == "constraint":
                                    # TODO: take real regret
                                    yval = 0
                            if yval is not None:
                                if xval not in results_dict[metric][method]:
                                    results_dict[metric][method][xval] = []
                                results_dict[metric][method][xval].append(yval)
                    else:
                        if args.independent_variable:
                            xval = get_dict_mult_key(
                                experiment.config, args.independent_variable
                            )
                        else:
                            xval = 0

                        if metric == "runtime":
                            yval = experiment.get_runtime()
                        elif metric == "regret":
                            yval = experiment.result["reward"]
                            yval = experiment.info["best_reward"] - yval
                        else:
                            if metric not in experiment.result:
                                continue
                            yval = experiment.result[metric]
                            yval = float(yval)

                        if metric not in results_dict.keys():
                            results_dict[metric] = dict()
                        if method not in results_dict[metric]:
                            results_dict[metric][method] = dict()
                        if xval not in results_dict[metric][method]:
                            results_dict[metric][method][xval] = []
                        results_dict[metric][method][xval].append(yval)

    methods_to_plot = methods
    barchart_names = dict()
    color_overwrite = dict()

    # methods_to_plot = [
    #     "uniform",
    #     # "adaptive_uniform_all",
    #     "adaptive_uniform_all_tuned",
    #     "g-allocation",
    #     # "adaptive_maxvar_all",
    #     "adaptive_maxvar_all_tuned",
    #     "adaptive_static",
    #     "adaptive",
    #     "adaptive_tuned",
    #     # "adaptive_uniform",
    #     # "adaptive_uniform_tuned",
    #     "adaptive_greedy_reward_uncertain",
    #     "adaptive_greedy_reward_uncertain_tuned",
    #     "oracle",
    # ]

    # For driver experiments -- Barchart
    # methods_to_plot = [
    #     "uniform",
    #     "adaptive_uniform_all",
    #     "g-allocation",
    #     "adaptive_maxvar_all",
    #     "adaptive_static",
    #     "adaptive",
    #     "adaptive_greedy_reward_uncertain",
    #     "adaptive_uniform",
    #     # "oracle",
    # ]
    # barchart_names = {
    #     "uniform": "Uniform",
    #     "adaptive_uniform_all": "Adaptive Uniform",
    #     "g-allocation": "G-Allocation",
    #     "adaptive_maxvar_all": "Greedy MaxVar",
    #     "adaptive_static": "ACOL",
    #     "adaptive": "G-ACOL",
    #     "adaptive_greedy_reward_uncertain": "MaxRew-$\mathcal{U}$",
    #     "adaptive_uniform": "G-ACOL Uniform",
    #     "oracle": "Oracle",
    # }
    # for consistent colors in driver barchart vs beta chart
    # color_overwrite = {
    #     "adaptive": "#a65628",
    #     "adaptive_maxvar_all": "#f781bf",
    #     "adaptive_uniform_all": "#ff7f00",
    #     "adaptive_greedy_reward_uncertain": "#555555",
    # }
    # for presentation
    # color_overwrite = {
    #     "uniform": "#444444",
    #     "g-allocation": "#999999",
    #     "adaptive_static": "#e41a1c",
    # }

    # For driver experiments -- beta charts
    methods_to_plot = [
        "adaptive_tuned",  # G-ACOL
        "adaptive_uniform_all_tuned",  # Adaptive Uniform
        "adaptive_maxvar_all_tuned",  # Greedy MaxVar
        "adaptive_greedy_reward_uncertain_tuned",  # MaxRew-U
        "adaptive_uniform_tuned",  # G-ACOL Uniform
    ]

    # For regret comparison experiment -- Barchart
    # methods_to_plot = [
    #     "adaptive_greedy_reward_feasible",
    #     "adaptive_greedy_reward_uncertain",
    #     "adaptive"
    # ]
    # barchart_names = {
    #     "adaptive_greedy_reward_feasible": "MaxRew-$\mathcal{F}$",
    #     "adaptive_greedy_reward_uncertain": "MaxRew-$\mathcal{U}$",
    #     "adaptive": "G-ACOL",
    # }

    for metric in results_dict.keys():
        print(f"Creating plots of {metric}")
        fig_height, fig_width = plt.figaspect(args.aspect)
        fig = plt.figure(figsize=(fig_width, fig_height))
        color, hatch, marker = 0, 0, 0
        legend_handles, legend_labels = [], []

        if args.barchart:
            xvalues = []
            yvalues = []
            yerr = []
            labels = []
            raw_labels = []
            i = 1

            for met in methods_to_plot:
                values = np.array(
                    sum([v for k, v in results_dict[metric][met].items()], [])
                )
                mean = np.mean(values)
                stderr = np.std(values) / np.sqrt(len(values))
                xvalues.append(i)
                yvalues.append(mean)
                yerr.append(stderr)

                label = met
                raw_labels.append(label)
                if label in barchart_names:
                    label = barchart_names[label]
                if args.paper_style:
                    label = label.replace("_", "\_")
                labels.append(label)
                print(label, len(values), mean, stderr)

                i += 1

            idxsort = np.argsort(yvalues)
            yvalues = np.array(yvalues)[idxsort]
            yerr = np.array(yerr)[idxsort]
            labels = np.array(labels)[idxsort]
            raw_labels = np.array(raw_labels)[idxsort]

            if args.no_error:
                yerr = None

            barlist = plt.bar(xvalues, yvalues, tick_label=labels, yerr=yerr)

            if args.use_colors_markers:
                for label, bar in zip(raw_labels, barlist):
                    if label in PLOT_COLORS:
                        bar_color = PLOT_COLORS[label]
                    if label in color_overwrite:
                        bar_color = color_overwrite[label]
                    bar.set_color(bar_color)

        else:
            for met in methods_to_plot:
                if met not in results_dict[metric]:
                    continue

                print(f"met: {met}", end=" ")

                xvalues = []
                yvalues = []

                print("yvalue len:", end=" ")
                for xval in results_dict[metric][met].keys():
                    if (not args.anytime) or xval < 50:
                        yvals = results_dict[metric][met][xval]
                        xvalues.append(xval)
                        yvalues.append(yvals)
                        print(len(yvals), end="  ")
                print()

                if args.use_colors_markers:
                    if met in color_overwrite:
                        plot_color = color_overwrite[met]
                    else:
                        plot_color = PLOT_COLORS[met]
                    plot_marker = PLOT_MARKERS[met]
                    plot_linestyle = PLOT_LINESTYLE[met]
                    plot_alpha = PLOT_ALPHA[met]
                    plot_zorder = PLOT_ZORDER[met]
                else:
                    plot_color = color_cycle[color]
                    plot_marker = marker_cycle[marker]
                    plot_linestyle = "-"
                    plot_alpha = 1
                    plot_zorder = 1

                if args.no_markers:
                    plot_marker = None

                if args.paper_style and not args.no_replace:
                    met = met.replace("_", "\_")

                legend_handles, legend_labels = plot_result_percentiles(
                    xvalues,
                    yvalues,
                    met,
                    plot_color,
                    hatch,
                    legend_handles,
                    legend_labels,
                    fix_negative=(
                        metric == "regret" or metric == "selected_policy_regret"
                    ),
                    markers_every=args.markers_every,
                    plot_mean=args.mean,
                    plot_stddev=args.stddev,
                    plot_stderr=args.stderr,
                    dont_plot_percentiles=args.no_error,
                    marker=plot_marker,
                    alpha=plot_alpha,
                    zorder=plot_zorder,
                    linestyle=plot_linestyle,
                )
                color, hatch, marker = get_next_style(color, hatch, marker)

        if not args.no_legend and not args.barchart:
            # sort both labels and handles by labels
            legend_labels, legend_handles = zip(
                *sorted(zip(legend_labels, legend_handles), key=lambda t: t[0])
            )

            if not args.paper_style:
                if args.legend_pos == "upper_right":
                    # uppper right
                    legend_kwargs = {
                        "loc": "center left",
                        "bbox_to_anchor": (0.5, 0.8),
                    }
                elif args.legend_pos == "outside_right":
                    legend_kwargs = {
                        "loc": "center left",
                        "bbox_to_anchor": (1, 0.5),
                    }
                else:
                    # lower right
                    legend_kwargs = {
                        "loc": "center left",
                        "bbox_to_anchor": (0.5, 0.2),
                    }
            else:
                legend_kwargs = dict()

            leg = plt.legend(
                legend_handles,
                legend_labels,
                fontsize=5,
                handlelength=2,
                prop={"size": 20},
                **legend_kwargs,
            )

        if not args.no_title:
            title = args.instance
            if args.paper_style and not args.no_replace:
                title = title.replace("_", "\_")
            plt.title(title)

        if args.xmin is not None and args.xmax is not None:
            plt.xlim(args.xmin, args.xmax)

        if args.ymin is not None and args.ymax is not None:
            plt.ylim(args.ymin, args.ymax)

        if args.hlines is not None:
            hline_values = [float(x) for x in args.hlines.split(",")]
            for hline_i, hline_val in enumerate(hline_values):
                color = hline_i % len(color_cycle)
                plt.axhline(
                    hline_val, linestyle="--", color=color_cycle[color], zorder=0
                )

        if args.hline is not None:
            plt.axhline(args.hline, linestyle="--", color="black", zorder=0)

        if args.vline is not None:
            plt.axvline(args.vline, linestyle="--", color="black", zorder=0)

        if args.int_xaxis:
            # for only integer x values
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # force sceintific notation for y axis
        plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        if args.barchart:
            plt.xticks(rotation=args.xtick_rotation)

        if args.paper_style:
            if args.xmax is None:
                xmax = plt.gca().get_xlim()[1]
            else:
                xmax = args.xmax

            if not args.barchart:
                plt.locator_params(nbins=args.n_xticks - 1, axis="x")

            plt.gca().tick_params(
                axis="both",
                direction="out",
                bottom=True,
                left=True,
                top=False,
                right=False,
            )

            if not args.barchart:
                # remove trailing zeros from x axis tick labels
                plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%g"))

            plt.xticks()
            plt.yticks()

        if args.xlabel is None:
            xlabel = args.independent_variable
        else:
            xlabel = args.xlabel

        if args.ylabel is None:
            ylabel = metric
        else:
            ylabel = args.ylabel

        if args.paper_style and not args.no_replace:
            xlabel = xlabel.replace("_", "\_")
            ylabel = ylabel.replace("_", "\_")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if args.logy:
            plt.yscale("log")

        if args.no_yaxis:
            plt.gca().axes.get_yaxis().set_visible(False)

        if args.fix_axis_size:
            ax = plt.gca()
            ax.set_box_aspect(1 / args.aspect)
            # fig_size = fig.get_size_inches()
            # ax_height, ax_width = fig.gca().get_position().size * fig_size
            # ax_target_width = ax_height*args.aspect
            # fig_size[1] *= ax_target_width / ax_width
            # fig.set_size_inches(fig_size)

        plt.tight_layout()

        filename = f"{args.instance}_{args.independent_variable}"
        filename += "." + args.output_format

        filename = "{}_{}".format(metric, filename)
        if args.experiment_label is not None:
            folder_path = os.path.join(PLOTS_PATH, args.experiment_label)
        else:
            folder_path = PLOTS_PATH
        os.makedirs(folder_path, exist_ok=True)

        if args.output_format == "svg":
            plt.rcParams["svg.fonttype"] = "none"

        if args.extract_legend:
            extract_legend(leg, filename=os.path.join(folder_path, "legend.pdf"))

        full_path = os.path.join(folder_path, filename)
        print("Writing to", full_path)
        plt.savefig(os.path.join(folder_path, filename), format=args.output_format)

        del fig


if __name__ == "__main__":
    main()
