# Robustness experiment summary

- Source CSV: `outputs\robustness\ml1m_hstu_robustness.csv`
- Rows: 15000
- Users: 500
- Top-k: 10

## Main observations

- Lowest mean top-k overlap: `replace random` at `last` = 0.2493.
- Highest mean Jensen-Shannon divergence: `replace random` at `last` = 0.001667.

## Generated files

- `outputs\robustness\plots\robustness_summary_by_position.csv`
- `outputs\robustness\plots\robustness_summary_by_history_length.csv`
- `outputs\robustness\plots\mean_topk_overlap_by_position.svg`
- `outputs\robustness\plots\mean_js_divergence_by_position.svg`
- `outputs\robustness\plots\topk_overlap_boxplot_by_position.svg`
- `outputs\robustness\plots\topk_overlap_distribution_by_position.svg`
- `outputs\robustness\plots\mean_topk_overlap_by_history_length.svg`

The history-length plot uses equal-count bins over users, then averages row-level robustness metrics inside each bin.
