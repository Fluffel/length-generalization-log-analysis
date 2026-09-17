#!/usr/bin/env bash
# Plot every task that can be plotted.  Missing or unplottable tasks are
# skipped (and any leftover SVG is removed) so the rest of the suite and the
# HTML overview still get written.  Exit code 3 from generate_plot_df.py means
# "no runs for this task yet".
set -uo pipefail

tasks=(
    "addition"
    "bin_majority_interleave"
    "bin_majority"
    "majority"
    "mkar"
    "mqar"
    "parity"
    "repeat_copy"
    "selective_copy"
    "sort"
    "unique_copy"
    "dyck_2"
    "parity_majority"
    "selective_state_tracking"
)

failed=()
skipped=()
for TASK in "${tasks[@]}"; do
    python scripts/generate_plot_df.py \
        --task "$TASK" \
        --keep arch=ssm,olmossm \
        --group-by arch \
        --output "site/plots/ssm/tasks/${TASK}.svg" \
        --csv summary.csv \
        --legend-loc "lower left" \
        --max-aggregation max \
        --num-bins 3 \
        --merge-bins
    status=$?
    if [[ $status -eq 3 ]]; then
        skipped+=("$TASK")
        rm -f "site/plots/ssm/tasks/${TASK}.svg"
    elif [[ $status -ne 0 ]]; then
        failed+=("$TASK")
        rm -f "site/plots/ssm/tasks/${TASK}.svg"
    fi
done

[[ ${#skipped[@]} -gt 0 ]] && echo "No runs yet, skipped: ${skipped[*]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed to plot: ${failed[*]}" >&2
fi