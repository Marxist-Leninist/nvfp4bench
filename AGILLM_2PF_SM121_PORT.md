# AGILLM 2 PFLOP SM121 research fork

Base: secYOUre/nvfp4bench commit df8e8f469b0df929d7c5ba58ff26c5dfdd4e920c
Target: GB10 / SM121 at fixed ~2.5 GHz. Keep the existing dense-equivalent sparse NVFP4 FLOP accounting unchanged.

## Current measured baseline

- Verified open-source register-resident packed+sparse NVFP4 path: ~992 TFLOP/s best in our tuned fork, ~984 TFLOP/s sustained-best median family.
- SASS hot loop is `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X` with independent accumulators and zero spills.
- 2 PFLOP at 48 SM * 2.5 GHz requires ~16,667 dense-equivalent FLOP/cycle/SM, almost exactly 2x the present ~8,267 FLOP/cycle/SM.

## What can actually be ported from datacenter Blackwell

CUTLASS config proves SM121 already receives some SM100-derived primitives:

- LDSM/STSM are enabled for SM121.
- SM120-family TMA is enabled for SM121.
- `CUTE_ARCH_TCGEN05_TMEM_ENABLED` is deliberately *not* enabled for SM120/SM121; it is enabled for SM100/101/103 families.

Therefore the useful port is the *pipeline architecture*, not blindly enabling tcgen05:

1. port SM100-style staged/TMA operand movement to the SM121 warp-MMA backend;
2. use LDSM/STSM-native fragment layouts instead of manual gathers;
3. software-pipeline independent OMMA streams aggressively across warp schedulers;
4. investigate CTA-cluster/software pairing only as a scheduling strategy, never count it as 2-SM MMA unless hardware evidence proves it;
5. inspect generated SASS and issue counters for a genuine second OMMA issue domain.

## Hard gate for a real 2 PF result

A candidate only counts if all are true:

- fixed FLOP formula, identical to the existing verified benchmark;
- measured wall-clock kernel time;
- arithmetic witness remains valid;
- SASS shows real additional MMA issue, not duplicated accounting;
- no clock/power increase beyond the fixed platform setting;
- repeatable across multiple runs;
- production RPV16/DBlock trainer is not paused, killed, replaced or used as the benchmark victim.

## Three hypotheses

### H1: hidden tcgen05/TMEM compatibility
Low prior. Test compiler/ISA and cubin/SASS evidence before any device execution. If SM121 does not decode the instruction or lacks TMEM state, abandon this lane.

### H2: dual OMMA issue on SM121
Highest-value path. Existing ~1 PF loop is one OMMA stream. Search for scheduler/partition combinations that permit two independent sparse packed OMMA streams to retire concurrently. Evidence must be instruction/cycle counters plus unchanged FLOP accounting.

### H3: SM100 pipeline semantics over warp OMMA
Most practical for useful GEMM/training. Port TMA + LDSM/STSM + warp specialization concepts onto `mma.sync`/`mma.sp`. This may not yield 2 PF micropeak, but is the strongest route to turning ~1 PF micropeak into substantially higher end-to-end GEMM/training efficiency.


## Static legality result: wider packed-NVFP4 matrix shapes are closed

On CUDA 13 / `ptxas -arch=sm_121a`, a systematic static sweep covered every matrix-shape token present in the installed ptxas binary plus nearby plausible variants (26 candidates total) against both packed NVFP4 dense and packed+sparse PTX families.

Accepted forms were exactly:

- packed dense: `m16n8k64`
- packed + structured sparse: `m16n8k128`

All other candidates were rejected as **illegal matrix shapes**, including `m16n8k256`; there were no operand-mismatch-only candidates suggesting a larger legal shape with merely different register-vector widths. This closes the hidden-k256/wider-shape hypothesis for the programmable `sm_121a` warp-MMA interface.

Artifacts: `src/legal_shape_sweep.py` and `/workspace/gb10_2pflop_openfork_20260916/static_recon/legal_shape_sweep/results.json`.

The remaining honest 2-PF hypotheses are therefore issue-rate/scheduler-domain improvements using the legal `m16n8k128` sparse OMMA, or a different legal instruction encoding with the same logical shape. FLOP accounting stays fixed.
