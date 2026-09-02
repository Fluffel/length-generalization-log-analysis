#!/usr/bin/env bash
# Fail the whole script when any task fails to plot.  Without this the loop
# keeps going and exits with the status of the last task only, so a broken
# plot would be published as whatever SVG happened to be on disk already.
# Exit code 3 means "no runs for this task yet", which is not a failure.
set -uo pipefail

tasks=(
    "addition"
    "bin_majority_interleave"
    "bin_majority"
    "flipflop"
    "majority"
    "mkar"
    "mqar"
    "parity"
    "repeat_copy"
    "selective_copy"
    "sort"
    "unique_copy"
)

failed=()
skipped=()
for TASK in "${tasks[@]}"; do
    python scripts/generate_plot_df.py \
        --task "$TASK" \
        --keep arch=hyb,olmohyb \
        --group-by arch \
        --output "site/plots/hyb/tasks/${TASK}.svg" \
        --csv summary.csv \
        --legend-loc "lower left" \
        --num-bins 3 \
        --merge-bins
    status=$?
    if [[ $status -eq 3 ]]; then
        skipped+=("$TASK")
    elif [[ $status -ne 0 ]]; then
        failed+=("$TASK")
    fi
done

[[ ${#skipped[@]} -gt 0 ]] && echo "No runs yet, skipped: ${skipped[*]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed to plot: ${failed[*]}" >&2
    exit 1
fi