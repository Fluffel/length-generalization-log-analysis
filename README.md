# Intro
This repository contains the pipeline to quickly generate result plots and an overview of the length-generalization experiments for this repository https://github.com/Fluffel/length_generalization. This repository is only for convenience and experimental.

Results can be found here: https://fluffel.github.io/length-generalization-log-analysis/

## Compact comparison grids

`scripts/generate_compact_grids.py` creates two tight, paper-style SVG figures:
one for algorithmic tasks and one for formal languages. The workflow commits
updated files to `workflow-plots/` on `main`, making them directly visible in
the GitHub repository. They are not included in the GitHub Pages site.

The configuration is kept at the top of the script:

- `PLOT_GROUPS` defines the model lines with pandas query expressions, labels,
  colors, and markers.
- `FIGURES` defines each figure's task order, column count, and bin filter.

Run it locally after generating `summary.csv`:

```bash
python scripts/generate_compact_grids.py
```

The two files are written to `workflow-plots/` by default.