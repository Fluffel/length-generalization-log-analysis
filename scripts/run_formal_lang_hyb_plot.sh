#!/usr/bin/env bash
# Plot every language that can be plotted.  Missing or unplottable languages
# are skipped (and any leftover SVG is removed) so the rest of the suite and
# the HTML overview still get written.  Exit code 3 from generate_plot_df.py
# means "no runs for this language yet".
set -uo pipefail

tasks=(
"012_star_0_2_star"
"aa_star"
"aa_star_bb_star"
"abab_star"
"ab_star_d_bc_star"
"d_2"
"d_3"
"d_4"
"d_12"
"tomita_1"
"tomita_2"
"tomita_3"
"tomita_4"
"tomita_5"
"tomita_6"
"tomita_7"
)

failed=()
skipped=()
for TASK in "${tasks[@]}"; do
    python scripts/generate_plot_df.py \
        --task "$TASK" \
        --keep arch=hyb,olmohyb \
        --group-by arch \
        --output "site/plots/hyb/formal/${TASK}.svg" \
        --csv summary.csv \
        --legend-loc "lower left" \
        --max-aggregation max \
        --first-bins 3 \
        --merge-bins
    status=$?
    if [[ $status -eq 3 ]]; then
        skipped+=("$TASK")
        rm -f "site/plots/hyb/formal/${TASK}.svg"
    elif [[ $status -ne 0 ]]; then
        failed+=("$TASK")
        rm -f "site/plots/hyb/formal/${TASK}.svg"
    fi
done

[[ ${#skipped[@]} -gt 0 ]] && echo "No runs yet, skipped: ${skipped[*]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed to plot: ${failed[*]}" >&2
fi