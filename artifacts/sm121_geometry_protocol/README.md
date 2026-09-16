# SM121 per-warp eligibility vs backend-saturation protocol

The OMMA control stall is a per-warp eligibility field. It is **not** a global tensor-pipe issue-period knob.
This protocol therefore starts with one 32-thread block per SM, where there is only one requested warp per SM and warp rotation cannot hide that warp's eligibility delay. It then increases the requested warp population and repeats equivalent populations with 64- and 128-thread blocks.

The saturated 128-thread / 8-blocks-per-SM silicon run already showed only a ~0.89% median gain from stall8/y0 (943.501 TFLOP/s) to the best stall7/y1 variant (951.895 TFLOP/s), far below 2 PF. The geometry sweep is therefore diagnostic, not a new stall-based 2PF prediction: it tests whether low-population warps are stall-limited while the high-population result is backend-limited.

Interpretation:

- If `spacing_stall7` / `spacing_stall6` beat the 16-OMMA base at one warp/SM, the compiler's conservative per-warp schedule has measurable slack.
- If that advantage shrinks as requested warps/SM increase, warp rotation is hiding per-warp latency and a shared backend limit is taking over.
- If the 31-OMMA fill variants raise **normalized measured OMMA/s** at high warp population while preserving the arithmetic witness, the backend can accept a higher aggregate OMMA rate.
- If normalized OMMA/s plateaus near the existing ~1-PF result regardless of stall/fill changes, the warp-OMMA backend is the likely limit and the stall-only lane should be closed.

Every candidate runs the per-accumulator arithmetic witness before timing. The timing program reports CUDA occupancy limits (`max_active_blocks_per_sm`, `max_resident_warps_per_sm`) alongside requested grid warps/SM. No PFLOP/s value is inferred from a stall field.
