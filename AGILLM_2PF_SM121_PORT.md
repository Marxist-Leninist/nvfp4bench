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

## 2026-09-16 static gates closed and active issue-rate experiment

### Closed lanes

- **tcgen05/TMEM on SM121:** closed. CUDA 13 `ptxas` rejects `tcgen05.mma`, `tcgen05.relinquish_alloc_permit`, `.cta_group::{1,2}`, and TMEM-style features for `sm_121a`; the same toolchain accepts them for datacenter Blackwell targets.
- **Hidden wider packed-NVFP4 warp shape:** closed by `src/legal_shape_sweep.py`. Of 26 assembler-known/plausible shapes, packed dense accepts only `m16n8k64`; packed+sparse accepts only `m16n8k128`. `m16n8k256` is explicitly an illegal sparse-MMA shape.
- **Alternate packed scale encodings:** exhaustive static cross-product accepts only `2X+UE8M0` and `4X+UE4M3`; both remain `m16n8k128` sparse and therefore carry the same fixed logical 32,768 FLOPs/warp instruction.
- **FP16 accumulator shortcut:** rejected by `ptxas` for packed sparse NVFP4; the exposed form is F32 accumulate.

### Compiler scheduling finding

Controlled source probes compare an 8-OMMA true accumulator-dependency chain with 8 OMMAs using independent accumulator groups. `ptxas` emits the same steady schedule for both:

- one 16-byte `NOP` between adjacent sparse OMMAs;
- steady sparse-OMMA upper control word `0x000fde00...`;
- SM120 control decoder: `stall=(ctrl>>41)&0xf`, `yield=(ctrl>>45)&1`;
- therefore steady sparse OMMA carries **stall=15, yield=0**;
- sequence-opening/terminal form `0x000fe200...` decodes **stall=1, yield=1**;
- NOP form `0x000fc200...` decodes **stall=1, yield=0**.

The stall-field interpretation is independently calibrated by ordinary compiler-generated FFMA control words (`0x...fc800` -> stall 8, `0x...fca00` -> stall 10). The key inference is narrow: **ptxas does not relax sparse-OMMA scheduling merely because accumulators are independent.** It does *not* prove hardware accepts a shorter sparse-OMMA issue interval.

### Active hardware question

Existing verified register-resident packed+sparse result is about 0.252 OMMA/SM/cycle (~0.99 PFLOP/s dense-equivalent at the observed platform point). At 48 SM and a fixed 2.5 GHz accounting clock, 2 PFLOP/s requires:

```
2e15 / (48 * 2.5e9 * 32768) = 0.508626 OMMA / SM / cycle
```

The active question is therefore no longer “find a bigger FLOP count”. It is:

> Can independent `OMMA.SF.SP.168128` instructions execute correctly with a materially shorter control stall / issue cadence than ptxas's steady stall=15 schedule?

### Staged experiments (not yet run on production GPU)

1. `src/peak_2pf_probe.cu`: 2X/4X, one-bank and dual-bank issue-rate / launch-topology sweep, fixed FLOP accounting.
2. `src/peak_multibank_static.cu`: 1x16, 2x8, 4x4, 8x2, 16x1 operand-bank sweep. 4x4 is a useful ILP point; launch-bound pressure can reach 62 regs/thread but adds substantial MOV/R2UR traffic.
3. `src/peak_2pf_cluster_probe.cu`: matched ordinary-vs-2CTA-cluster test with identical total CTAs, OMMA count and FLOP formula. This tests scheduling only; it is not called hardware 2-SM MMA.
4. `src/peak_sass_patch_target.cu` + `src/patch_sm121_omma_nops.py`: countable NOP-fill series. Static cubins contain 16/20/23/26/30 sparse OMMAs per warp-iteration for 0/25/50/75/100% fill. NVIDIA disassemblers accept all variants.
5. `src/verify_sass_patch_manifest.py`: byte proof that patched variants change only aligned 16-byte NOP slots inside the target `.text`; all nonpatched bytes, control flow, section layout and companion metadata remain byte-identical.
6. `src/peak_sass_patch_target_v2.cu` + `src/run_sass_patch_correctness_v2.cpp`: bank-separated arithmetic witness. Inserted donor OMMAs remain inside their operand bank and timing is rejected unless each bank's accumulated output scales by the exact predicted `(8+added)/8` ratio.
7. `src/run_2pf_guarded.sh`: fail-closed GPU guard. It never stops/signals a trainer and requires an explicit SG benchmark fence plus an empty CUDA/process census before launch.

### Promotion rule

No patched result counts as a 2PF improvement unless, in order:

1. base cubin loads/runs correctly;
2. patched cubin loads/runs correctly;
3. bank-separated arithmetic witness passes;
4. wall-clock timing uses the fixed 32,768 FLOPs per sparse OMMA actually present in the binary;
5. repeated trials show the gain and the measured OMMA/SM/cycle moves toward 0.508626;
6. production RPV16/DBlock remains outside the benchmark window and is never stopped by this lane.

