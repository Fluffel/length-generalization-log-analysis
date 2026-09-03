#!/usr/bin/env python3
"""Regenerate plots and summarise them in HTML contact sheets.

Runs the ``run_*_plot.sh`` scripts for ``--arch`` (each writes one SVG per task),
then writes one HTML page per group that lays those SVGs out in a grid of
``--columns`` columns.

Each panel lists the models that appear in that graph's legend — the same
selection ``generate_plot_df.py`` used to draw the SVG — with that run's
accuracy in every plotted bin.  A panel gets a green border when at least one
of those plotted series stays at or above ``--threshold`` in every bin.  Keep
filters, bin trimming and grouping are read from the shell scripts, so the
HTML always describes exactly what was plotted.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dataframe_query_utils import apply_keep_remove_filters
from generate_plot_df import NoDataForTask, _prepare_task_plot
from plot_utils import BinFilter, load_summary_dataframe

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]

print(f"script dirctory: {REPO_ROOT}")

_TASKS_ARRAY_RE = re.compile(r"tasks=\(\s*(.*?)\)", re.DOTALL)
_TOKEN_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'|(\S+)")
_TASK_PLACEHOLDERS = ("${TASK}", "$TASK")


@dataclass(frozen=True)
class PlotScript:
    """A ``run_*_plot.sh`` script and everything it tells us about its plots."""

    name: str          # short id used in the HTML filename
    title: str         # heading shown on the page
    path: Path
    tasks: list[str]
    plot_template: str  # path with ${TASK} placeholder, relative to the repo root
    csv_path: Path
    keep: list[str]
    group_by: list[str]
    max_aggregation: str
    max_acc_threshold: float
    max_bin_weight: float
    num_bins: int | None
    first_bins: int | None
    merge_bins: bool

    def plot_path(self, task: str) -> Path:
        rel = self.plot_template
        for placeholder in _TASK_PLACEHOLDERS:
            rel = rel.replace(placeholder, task)
        return REPO_ROOT / rel


@dataclass(frozen=True)
class PlottedSeries:
    """One legend entry: the run(s) actually drawn, with per-bin accuracy."""

    label: str
    bin_labels: tuple[str, ...]
    accuracies: tuple[float, ...]  # fractions in [0, 1], one per plotted bin

    def passes(self, threshold: float) -> bool:
        return bool(self.accuracies) and all(acc >= threshold for acc in self.accuracies)


@dataclass(frozen=True)
class Panel:
    """One task's plot plus the series shown in its legend."""

    task: str
    plot_path: Path
    series: list[PlottedSeries]
    threshold: float

    @property
    def highlighted(self) -> bool:
        return any(s.passes(self.threshold) for s in self.series)


_FLAG_VALUE_RE = r"[=\s]+(?:\"([^\"]*)\"|'([^']*)'|(\S+))"


def _flag_value(text: str, flag: str) -> str | None:
    """Value of ``--flag value`` / ``--flag=value`` in a shell script."""
    m = re.search(rf"{re.escape(flag)}{_FLAG_VALUE_RE}", text)
    if not m:
        return None
    return next(g for g in m.groups() if g is not None)


def _flag_values(text: str, flag: str) -> list[str]:
    """Every ``--flag value`` occurrence, in order."""
    values: list[str] = []
    for groups in re.findall(rf"{re.escape(flag)}{_FLAG_VALUE_RE}", text):
        values.append(next(g for g in groups if g))
    return values


def _has_flag(text: str, flag: str) -> bool:
    return re.search(rf"(?:^|\s){re.escape(flag)}(?:\s|$)", text) is not None


def _optional_int_flag(text: str, flag: str) -> int | None:
    raw = _flag_value(text, flag)
    if raw is None or not raw.isdigit():
        return None
    return int(raw)


def _optional_float_flag(text: str, flag: str, default: float) -> float:
    raw = _flag_value(text, flag)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_task_array(text: str, script: Path) -> list[str]:
    m = _TASKS_ARRAY_RE.search(text)
    if not m:
        raise SystemExit(f"{script}: could not find a tasks=( ... ) array.")
    tasks: list[str] = []
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0]
        for quoted_d, quoted_s, bare in _TOKEN_RE.findall(line):
            token = quoted_d or quoted_s or bare
            if token:
                tasks.append(token)
    if not tasks:
        raise SystemExit(f"{script}: tasks=( ... ) array is empty.")
    return tasks


def parse_plot_script(path: Path, *, name: str, title: str) -> PlotScript:
    if not path.exists():
        raise SystemExit(f"Plot script not found: {path}")
    text = path.read_text()

    template = _flag_value(text, "--output")
    if not template:
        raise SystemExit(f"{path}: could not find an --output path.")
    if not any(p in template for p in _TASK_PLACEHOLDERS):
        raise SystemExit(
            f"{path}: --output {template!r} has no ${{TASK}} placeholder, so the "
            "script does not write one plot per task."
        )

    csv_value = _flag_value(text, "--csv") or "summary.csv"
    keep = _flag_values(text, "--keep")
    group_by = _flag_values(text, "--group-by")
    max_aggregation = _flag_value(text, "--max-aggregation") or "max"
    if max_aggregation == "mean":
        max_aggregation = "pareto_mean"

    return PlotScript(
        name=name,
        title=title,
        path=path,
        tasks=_parse_task_array(text, path),
        plot_template=template,
        csv_path=(REPO_ROOT / csv_value).resolve(),
        keep=keep,
        group_by=group_by,
        max_aggregation=max_aggregation,
        max_acc_threshold=_optional_float_flag(text, "--max-acc-threshold", 0.98),
        max_bin_weight=_optional_float_flag(text, "--max-bin-weight", 1.1),
        num_bins=_optional_int_flag(text, "--num-bins"),
        first_bins=_optional_int_flag(text, "--first-bins"),
        merge_bins=_has_flag(text, "--merge-bins"),
    )


def run_plot_script(script: PlotScript) -> int:
    """Run the shell script from the repo root with our own interpreter first on PATH."""
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    print(f"Running {script.path.relative_to(REPO_ROOT)} ...", flush=True)
    proc = subprocess.run(["bash", str(script.path)], cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        print(f"  {script.path.name} exited with code {proc.returncode}.", flush=True)
    return proc.returncode


def stale_plots(script: PlotScript, *, newer_than: float) -> list[str]:
    """Tasks whose SVG is left over from a previous run rather than rewritten.

    A plot script can fail for one task and leave the old SVG in place, which
    would then be published as if it were current.  A task with no SVG at all
    is fine: it simply has no runs yet, and the page says so.
    """
    return [
        task
        for task in script.tasks
        if (path := script.plot_path(task)).exists()
        and path.stat().st_mtime < newer_than
    ]


def script_bin_filter(script: PlotScript, *, num_bins: int | None) -> BinFilter:
    """The same run/bin selection the plot script applies before drawing."""
    counts: frozenset[int] = frozenset()
    n = num_bins if num_bins is not None else script.num_bins
    if n is not None:
        counts = frozenset({n})
    return BinFilter(bin_counts=counts, first_bins=script.first_bins)


def filter_summary_for_script(df, script: PlotScript, *, num_bins: int | None):
    """Apply the plot script's ``--keep`` and bin filters to a long-form summary."""
    df = apply_keep_remove_filters(df, script.keep, [])
    return script_bin_filter(script, num_bins=num_bins).apply(df)


def _bin_labels_for_series(prepared: dict, xs: list[float]) -> tuple[str, ...]:
    if prepared.get("use_bins"):
        ticks = prepared.get("ordinal_tick_labels") or []
        labels: list[str] = []
        for x in xs:
            idx = int(round(x))
            if 0 <= idx < len(ticks):
                labels.append(str(ticks[idx]))
            else:
                labels.append(f"bin{idx + 1}")
        return tuple(labels)
    return tuple(str(int(x)) if float(x).is_integer() else f"{x:g}" for x in xs)


def plotted_series_for_task(df, script: PlotScript, task: str) -> list[PlottedSeries]:
    """Legend entries for *task*, using the same selection as the SVG."""
    sub = df[df["task"].astype(str) == task] if "task" in df.columns else df
    if sub.empty:
        return []
    try:
        prepared = _prepare_task_plot(
            sub,
            task=task,
            group_by=script.group_by,
            group_label_mode="model",
            group_custom_labels=[],
            max_aggregation=script.max_aggregation,
            max_bin_weight=script.max_bin_weight,
            max_acc_threshold=script.max_acc_threshold,
            x_ticks_mode="bins",
            x_tick_step=10,
            x_axis_break=None,
            num_bins=None,  # already applied via BinFilter
            merge_bins=script.merge_bins,
        )
    except NoDataForTask:
        return []

    series: list[PlottedSeries] = []
    max_series = prepared["max_series"]
    for sk in prepared["sub_keys"]:
        if sk not in max_series:
            continue
        label, xs, means, _stds = max_series[sk]
        series.append(
            PlottedSeries(
                label=label,
                bin_labels=_bin_labels_for_series(prepared, xs),
                accuracies=tuple(m / 100.0 for m in means),
            )
        )
    return series


def build_panels(
    df,
    script: PlotScript,
    *,
    threshold: float,
    num_bins: int | None,
) -> list[Panel]:
    filtered = filter_summary_for_script(df, script, num_bins=num_bins)
    return [
        Panel(
            task=task,
            plot_path=script.plot_path(task),
            series=plotted_series_for_task(filtered, script, task),
            threshold=threshold,
        )
        for task in script.tasks
    ]


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; line-height: 1.6; }}
  .legend-dot {{ display: inline-block; width: 0.8rem; height: 0.8rem;
                 border: 3px solid #2e7d32; border-radius: 3px;
                 vertical-align: middle; }}
  .grid {{ display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr));
           gap: 1.25rem; }}
  .panel {{ border: 3px solid #d8d8d8; border-radius: 8px; padding: 0.75rem;
            background: #fff; }}
  .panel.pass {{ border-color: #2e7d32; background: #f3fbf4; }}
  .panel h2 {{ font-size: 1rem; margin: 0 0 0.5rem; }}
  .panel img {{ width: 100%; height: auto; display: block; }}
  .models {{ font-size: 0.8rem; color: #444; margin: 0.5rem 0 0;
             padding-left: 1.15rem; word-break: break-all; line-height: 1.5; }}
  .models li {{ margin: 0.15rem 0; }}
  .panel.pass .models {{ color: #2e7d32; }}
  .missing {{ font-size: 0.9rem; color: #b71c1c; padding: 2rem 0; text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  {meta}
</div>
<div class="grid">
{panels}
</div>
</body>
</html>
"""


def _format_series_caption(series: PlottedSeries) -> str:
    bins = ", ".join(
        f"{html.escape(label)} {acc * 100:.1f}%"
        for label, acc in zip(series.bin_labels, series.accuracies, strict=False)
    )
    if not bins:
        return html.escape(series.label)
    return f"{html.escape(series.label)} ({bins})"


def _render_panel(panel: Panel, html_dir: Path) -> str:
    task = html.escape(panel.task)
    parts = [f"<h2>{task}</h2>"]
    if panel.plot_path.exists():
        # The SVG filenames are stable, so a browser that has already seen a
        # page would keep serving the cached plot next to freshly generated
        # HTML.  The plot's mtime as a query string forces a refetch whenever
        # the plot actually changed.
        src = html.escape(os.path.relpath(panel.plot_path, html_dir))
        version = int(panel.plot_path.stat().st_mtime)
        parts.append(f'<img src="{src}?v={version}" alt="{task}">')
    else:
        parts.append('<div class="missing">no plot generated</div>')
    if panel.series:
        items = "".join(
            f"<li>{_format_series_caption(s)}</li>" for s in panel.series
        )
        parts.append(f'<ul class="models">{items}</ul>')
    classes = "panel pass" if panel.highlighted else "panel"
    body = "\n    ".join(parts)
    return f'  <div class="{classes}">\n    {body}\n  </div>'


def _bins_note(script: PlotScript, num_bins: int | None) -> str:
    n = num_bins if num_bins is not None else script.num_bins
    parts: list[str] = []
    if n is not None:
        parts.append(f"{n} bins")
    if script.first_bins is not None:
        parts.append(f"first {script.first_bins} bins")
    return ", ".join(parts) if parts else "all bin counts"


def render_page(
    script: PlotScript,
    panels: list[Panel],
    *,
    arch: str,
    threshold: float,
    columns: int,
    num_bins: int | None,
    html_dir: Path,
) -> str:
    passed = sum(1 for p in panels if p.highlighted)
    missing = sum(1 for p in panels if not p.plot_path.exists())
    meta_lines = [
        f"<span class='legend-dot'></span> green border: a plotted "
        f"<strong>{html.escape(arch)}</strong> series is at or above "
        f"{threshold * 100:.1f}% in every shown bin "
        f"({passed} of {len(panels)} tasks).",
        "Captions list the models in each graph's legend, with accuracy per bin.",
        f"Source: {html.escape(str(script.csv_path.relative_to(REPO_ROOT)))}, "
        f"{_bins_note(script, num_bins)}, plots from {html.escape(script.path.name)}.",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.",
    ]
    if missing:
        meta_lines.append(f"{missing} task(s) produced no plot.")
    return _PAGE_TEMPLATE.format(
        title=html.escape(script.title),
        columns=columns,
        meta="<br>\n  ".join(meta_lines),
        panels="\n".join(_render_panel(p, html_dir) for p in panels),
    )

SCRIPT_FILES = {"ssm": {"formal": "run_formal_lang_ssm_plot.sh",
                        "tasks": "run_tasks_ssm_plot.sh"},
                "hyb": {"formal": "run_formal_lang_hyb_plot.sh",
                        "tasks": "run_tasks_hyb_plot.sh"}}

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the formal-language and task plots, then write one "
            "HTML overview per group with the plots side by side. Captions list "
            "the models in each graph's legend; panels where a plotted series "
            "stays at or above the accuracy threshold in every bin get a green "
            "border."
        )
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Which plot scripts to run (ssm or hyb). Architecture filters "
        "come from each script's --keep flag, not this name alone.",
    )
    parser.add_argument(
        "--columns",
        "--ncols",
        dest="columns",
        type=int,
        default=3,
        help="Number of plots per row in the HTML pages (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "site",
        help="Directory for the HTML pages.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Minimum accuracy required in every bin, as a fraction or a "
        "percentage (default: 0.98).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Summary CSV used for the highlight check (default: the --csv value "
        "of each shell script).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Bin count a run must have to be plotted (default: the --num-bins "
        "value of each shell script).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only rebuild the HTML pages from the plots that are already on disk.",
    )
    args = parser.parse_args()

    if args.columns < 1:
        raise SystemExit("--columns must be >= 1.")
    threshold = args.threshold / 100.0 if args.threshold > 1 else args.threshold
    if not 0 <= threshold < 1:
        raise SystemExit("--threshold must be a fraction in [0, 1) or a percentage.")

    script_files = SCRIPT_FILES[args.arch]
    scripts = [
        parse_plot_script(
            SCRIPT_DIR / script_files["formal"],
            name="formal",
            title=f"Formal languages — {args.arch} plots",
        ),
        parse_plot_script(
            SCRIPT_DIR / script_files["tasks"],
            name="tasks",
            title=f"Algorithmic tasks — {args.arch} plots",
        ),
    ]

    # Anything older than this was left over from the checkout rather than
    # written by this run.  One second of slack absorbs filesystem timestamp
    # granularity.
    run_started = time.time() - 1
    failures: list[str] = []

    if not args.skip_plots:
        for script in scripts:
            if run_plot_script(script) != 0:
                failures.append(f"{script.path.name} exited non-zero")
        for script in scripts:
            if stale := stale_plots(script, newer_than=run_started):
                failures.append(
                    f"{script.path.name} did not regenerate: {', '.join(stale)}"
                )

    html_dir = args.output
    html_dir.mkdir(parents=True, exist_ok=True)

    for script in scripts:
        csv_path = args.csv or script.csv_path
        num_bins = args.num_bins if args.num_bins is not None else None
        df = load_summary_dataframe(csv_path)
        filtered = filter_summary_for_script(df, script, num_bins=num_bins)
        if filtered.empty:
            keep = ", ".join(script.keep) if script.keep else "no --keep"
            available = ", ".join(sorted(set(df["arch"].astype(str))))
            print(
                f"Warning: no rows left for {script.path.name} after {keep}; "
                f"available arch values: {available}"
            )

        panels = build_panels(
            df, script, threshold=threshold, num_bins=num_bins
        )
        page = render_page(
            script,
            panels,
            arch=args.arch,
            threshold=threshold,
            columns=args.columns,
            num_bins=num_bins,
            html_dir=html_dir,
        )
        out_path = html_dir / f"{script.name}_{args.arch}_plots.html"
        out_path.write_text(page)

        passed = [p.task for p in panels if p.highlighted]
        print(
            f"Wrote {out_path} ({len(panels)} plots, "
            f"{len(passed)} above {threshold * 100:.1f}% for {args.arch})"
        )
        if passed:
            print(f"  highlighted: {', '.join(passed)}")

    if failures:
        # Reported only after the HTML is written, so the pages are still there
        # to inspect, but with a non-zero status so CI never publishes a site
        # whose plots are older than its data.
        print("\nPlot generation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
