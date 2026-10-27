# Archived SGLang native result reanalysis

Native output_lens can come from stream usage or requested-length fallback; raw SSE usage, terminal framing, and exact 1024 generated tokens are not independently certified by this JSONL. Upstream combines reasoning_content/content.

Only request_rate Infinity is normalized. Original producer statuses and every native file SHA remain in analysis.json.

| Load | Producer status | Exit/process | Reanalysis | Completed | Output token/s | Wall seconds |
|---|---|---|---|---:|---:|---:|
| round1_4096_1024_c1 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 8 | 14.874251041324312 | 594.186152 |
| round2_4096_1024_c1 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 8 | 14.969114857514182 | 590.638239 |
| round1_4096_1024_c8 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 16 | 55.500322638562125 | 347.858115 |
| round2_4096_1024_c8 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 16 | 56.19793641620571 | 338.989474 |
| round1_4096_1024_c64 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 128 | 54.981844020132336 | 2541.494164 |
| round2_4096_1024_c64 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 128 | 54.94470222148663 | 2539.035043 |
| round1_16384_1024_c1 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 8 | 12.93264644599626 | 689.080753 |
| round2_16384_1024_c1 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 8 | 13.008174177528527 | 683.859863 |
| round1_16384_1024_c8 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 16 | 22.635446708815785 | 788.727658 |
| round2_16384_1024_c8 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 16 | 22.78194711454868 | 784.919829 |
| round1_16384_1024_c64 | interrupted | 130/interrupted | not_eligible |  |  | 3485.93223 |
| round2_16384_1024_c64 | not_attempted_interrupted | None/None | not_eligible |  |  | None |

Eligible native client reports: 10/12. Official acceptance: false. Exact output token proof: false.
