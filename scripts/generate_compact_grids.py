#!/usr/bin/env python3
"""Generate compact, paper-style task grids.

Edit ``PLOT_GROUPS`` to control which runs become separate lines.  Groups are
evaluated in order and use pandas query syntax.  Edit ``FIGURES`` to choose the
tasks, their order, grid width, and bin filtering.  The script always writes
exactly one SVG per figure specification.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from generate_plot_df import NoDataForTask, _prepare_task_plot
from plot_utils import BinFilter, load_summary_dataframe


@dataclass(frozen=True)
class PlotGroup:
    label: str
    query: str
    color: str
    marker: str


@dataclass(frozen=True)
class FigureSpec:
    output_name: str
    title: str
    tasks: tuple[str, ...]
    ncols: int
    bin_filter: BinFilter


# Each entry is one line in every subplot. Queries are intentionally explicit,
# so adding, removing, or regrouping models only requires changing this tuple.
PLOT_GROUPS = (
    PlotGroup(
        "Hybrid (NoPE)",
        "(arch == 'olmohyb') or ((arch == 'hyb') and (pe == 'False'))",
        "#0072B2",
        "o",
    ),
    PlotGroup(
        "SSM",
        "arch in ['ssm', 'olmossm']",
        "#009E73",
        "s",
    ),
    PlotGroup(
        "Transformer (NoPE)",
        "(arch == 'olmolm') or ((arch == 'lm') and (pe != 'True'))",
        "#D55E00",
        "^",
    ),
    PlotGroup(
        "Hybrid (RoPE)",
        "(arch == 'hyb') and (pe == 'True')",
        "#56B4E9",
        "D",
    ),
    PlotGroup(
        "Transformer (APE)",
        "(arch == 'lm') and (pe == 'True')",
        "#CC79A7",
        "v",
    ),
)


# Task order here is subplot order. A task without matching data still gets an
# axis, making missing results visible instead of silently changing the layout.
FIGURES = (
    FigureSpec(
        output_name="algorithmic_tasks.svg",
        title="Length generalization on algorithmic tasks",
        tasks=(
            "addition",
            "multiplication",
            "bin_majority_interleave",
            "bin_majority",
            "majority",
            "mkar",
            "mqar",
            "parity",
            "repeat_copy",
            "selective_copy",
            "sort",
            "unique_copy",
            "dyck_2",
            "parity_majority",
            "selective_state_tracking",
        ),
        ncols=5,
        bin_filter=BinFilter(bin_counts=frozenset({3})),
    ),
    FigureSpec(
        output_name="formal_languages.svg",
        title="Length generalization on formal languages",
        tasks=(
            "012_star_0_2_star",
            "aa_star",
            "aa_star_bb_star",
            "abab_star",
            "ab_star_d_bc_star",
            "d_2",
            "d_3",
            "d_4",
            "d_12",
            "tomita_1",
            "tomita_2",
            "tomita_3",
            "tomita_4",
            "tomita_5",
            "tomita_6",
            "tomita_7",
        ),
        ncols=4,
        bin_filter=BinFilter(first_bins=3),
    ),
)


def _assign_plot_groups(df):
    """Keep configured runs and attach their configured line label."""
    import pandas as pd

    frames = []
    assigned = pd.Series(False, index=df.index)
    for group in PLOT_GROUPS:
        try:
            matches = df.eval(group.query).fillna(False).astype(bool)
        except Exception as exc:
            raise SystemExit(
                f"Invalid query for plot group {group.label!r}: {group.query!r}: {exc}"
            ) from exc
        overlap = matches & assigned
        if overlap.any():
            raise SystemExit(
                f"Plot group {group.label!r} overlaps an earlier group "
                f"for {int(overlap.sum())} rows."
            )
        selected = df.loc[matches].copy()
        selected["plot_group"] = group.label
        frames.append(selected)
        assigned |= matches
    if not frames:
        return df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def _display_title(task: str) -> str:
    special = {
        "mkar": "MKAR",
        "mqar": "MQAR",
        "dyck_2": "Dyck-2",
    }
    return special.get(task, task.replace("_", " ").title())


def _group_label(series_key: tuple[str, frozenset[str]]) -> str:
    prefix = "grp:plot_group="
    sid = series_key[0]
    return sid[len(prefix) :] if sid.startswith(prefix) else sid


def _prepare(df, task: str):
    try:
        return _prepare_task_plot(
            df,
            task=task,
            group_by=["plot_group"],
            group_label_mode="group",
            group_custom_labels=[],
            max_aggregation="max",
            max_bin_weight=1.1,
            max_acc_threshold=0.98,
            x_ticks_mode="bins",
            x_tick_step=1,
            x_axis_break=None,
            num_bins=None,
            merge_bins=True,
        )
    except NoDataForTask:
        return None


def _draw_axis(ax, prepared, task: str, styles: dict[str, PlotGroup]) -> None:
    ax.set_title(_display_title(task), fontsize=8, fontweight="semibold", pad=2)
    ax.set_ylim(-4, 104)
    ax.set_yticks((0, 50, 100))
    ax.tick_params(axis="both", labelsize=6, length=2, width=0.6, pad=1)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)

    if prepared is None:
        ax.set_xticks((0, 1, 2), ("1", "2", "3"))
        ax.set_xlim(-0.35, 2.35)
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888888",
            fontsize=7,
        )
        return

    max_x = 0
    for key in prepared["sub_keys"]:
        if key not in prepared["max_series"]:
            continue
        _old_label, xs, means, _stds = prepared["max_series"][key]
        label = _group_label(key)
        style = styles[label]
        max_x = max(max_x, *(int(round(x)) for x in xs))
        ax.plot(
            xs,
            means,
            color=style.color,
            marker=style.marker,
            label=label,
            linewidth=1.15,
            markersize=2.8,
            markeredgewidth=0.45,
        )
    ticks = list(range(max_x + 1))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks])
    ax.set_xlim(-0.3, max(max_x + 0.3, 0.3))


def _draw_figure(df, spec: FigureSpec, output_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise SystemExit("Install pandas and matplotlib to generate compact grids.") from exc

    filtered = spec.bin_filter.apply(df)
    nrows = math.ceil(len(spec.tasks) / spec.ncols)
    fig, axes = plt.subplots(
        nrows,
        spec.ncols,
        figsize=(spec.ncols * 1.55, nrows * 1.25 + 0.55),
        squeeze=False,
        sharey=True,
    )
    styles = {group.label: group for group in PLOT_GROUPS}
    for ax, task in zip(axes.flat, spec.tasks, strict=False):
        _draw_axis(ax, _prepare(filtered, task), task, styles)
    for ax in axes.flat[len(spec.tasks) :]:
        ax.set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=group.color,
            marker=group.marker,
            linewidth=1.15,
            markersize=3.2,
            label=group.label,
        )
        for group in PLOT_GROUPS
    ]
    fig.suptitle(spec.title, fontsize=11, fontweight="bold", y=0.995)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(handles),
        frameon=False,
        fontsize=6.5,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.supxlabel("Validation bin", fontsize=7, y=0.01)
    fig.supylabel("Accuracy (%)", fontsize=7, x=0.008)
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.075,
        top=0.86,
        wspace=0.16,
        hspace=0.38,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / spec.output_name
    # Suppress Matplotlib's generation timestamp so unchanged data produces an
    # unchanged tracked SVG and the workflow does not create empty-noise commits.
    fig.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Date": None},
    )
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=repo_root / "summary.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "workflow-plots",
        help="Output directory for the two SVG files.",
    )
    args = parser.parse_args()

    df = _assign_plot_groups(load_summary_dataframe(args.csv))
    if df.empty:
        raise SystemExit("No rows matched PLOT_GROUPS.")
    for spec in FIGURES:
        _draw_figure(df, spec, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
