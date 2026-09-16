# SM100 tcgen05 -> SM121 reconstruction notes

Static compiler/disassembler work only. None of these probes execute a CUDA kernel.

## Verified facts on CUDA 13.0.88

- `tcgen05` compiles for the **architecture-specific** target `sm_100a`.
- The identical PTX is rejected by `ptxas` for `sm_121a`; both `tcgen05.mma` and `cta_group::{1,2}` are rejected.
- SM121 packed sparse NVFP4 compiles through warp-level `mma.sp...kind::mxf4nvf4`, emitted as
  `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X`.
- SM100 `tcgen05.mma...mxf4nvf4` decompiles to `UTCOMMA.4X`; `cta_group::2` decompiles to
  `UTCOMMA.2CTA.4X`.
- In the observed UTCOMMA encoding, 1CTA vs 2CTA differs by one bit in the upper 64-bit word:
  `...080000..` -> `...082000..` (bit 21 of the upper word / overall instruction bit 85).
- Flipping the corresponding overall bit 85 in the SM121 OMMA encoding does **not** expose a 2CTA mnemonic.
- Reflagging an SM100 cubin as SM121 causes `nvdisasm` to reject UTCOMMA as an illegal/unrecognized uC operation.
  This rules out the simplest "ptxas gate only" theory.
- Exhaustively enumerating the local SM121 OMMA modifier field at instruction bits 80..86 yields only:
  dense/sparse x E8/UE4M3 x 4X/non-4X. No `8X`, K=256, or `2CTA` form appears in that field.

## Baseline encodings

SM121 sparse packed NVFP4 OMMA observed in the probe:

```
OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X
low64  = 0x703002040404747f
high64 = 0x000fe20000053eff
```

SM100 tcgen05 sparse packed NVFP4:

```
UTCOMMA.4X      ... high64 contains ...080000...
UTCOMMA.2CTA.4X ... high64 contains ...082000...
```

The two families have different opcode classes (`...75ea` UTCOMMA vs `...747f` OMMA), so literal opcode copying is not a viable port.

## Reconstruction direction

The useful SM100 design ideas can still be ported without pretending UTCOMMA exists on SM121:

1. Keep SM121 TMA plus LDSM/STSM staging.
2. Replace TMEM accumulation with register-blocked OMMA accumulator banks.
3. Reconstruct CTA-group cooperation in software: paired CTAs own disjoint output fragments and share only staging/scheduling metadata.
4. Search for a genuinely independent second OMMA issue path/opclass. Do not relabel two ordinary CTAs as 2x throughput.
5. Treat >1 PFLOP as valid only when measured issued work/time increases under the existing dense-equivalent sparse-NVFP4 accounting.

The next static lane is opcode-class mapping and a dual-bank OMMA reconstruction. Runtime testing remains deferred while the production GB10 trainer owns the GPU.

## Additional decoder and dual-stream results

- Rewriting only the ELF architecture flags of a valid SM100A UTCOMMA cubin from `sm_100a` to `sm_121a`
  makes `nvdisasm` reject the UTCOMMA instruction as an **unrecognized uC operation / illegal instruction**.
  This is additional evidence that the UTCOMMA execution class is not merely hidden behind a PTX frontend gate on SM121.
- Exhaustive static decode of the obvious SM121 OMMA modifier bits 80..86 produced exactly eight mnemonic families:
  dense/sparse x E8/UE4M3 x 4X/non-4X. It exposed no `2CTA`, `8X`, or K=256 form.
- Exhaustively varying instruction bytes 13 and 15 did not reveal another OMMA mnemonic. Most changes are either
  accepted scheduling/control encodings or rejected decoder states.

A separate static dual-bank reconstruction (`src/peak_dual_omma_static.cu`) compiles for SM121A without executing:

| Static form | Registers | Spill stores/loads | Static OMMA count in cubin |
|---|---:|---:|---:|
| one bank, 16 accumulators | 74 | 0 / 0 | 464 |
| two banks, 8+8 accumulators | 81 | 0 / 0 | 464 |
| two banks, 16+16 accumulators | 138 | 0 / 0 | 928 |

This proves the compiler can preserve two independent OMMA operand/accumulator banks cheaply in the 8+8 form,
but it does **not** prove dual issue. Runtime throughput measurement is intentionally deferred while production owns GB10.

## SM121 2-CTA software reconstruction is compiler-legal

A static `__cluster_dims__(2,1,1)` probe compiles successfully for `sm_121a`. SASS contains explicit cluster metadata plus `CGAERRBAR`, `UCGABAR_ARV`, and `UCGABAR_WAIT`. A cluster-2 OMMA reconstruction therefore exists as a legal software building block even though hardware UTCOMMA/TCGEN05 is absent.

`src/sm121_cluster2_omma_static.cu` combines an explicit two-CTA cluster with the legal sparse packed OMMA path and compiles with 78 registers, zero spills. This is not counted as hardware 2CTA MMA.

The six legal issue-rate probe variants also compile statically with zero spills. One-bank16 uses 74 registers, two-bank8 uses 81, and two-bank16 uses 138 for both 2X/E8 and 4X/UE4M3 forms. The hot SASS stream remains one OMMA every 0x20 bytes in the fully unrolled core; the two-bank8 form occasionally has a 0x30-byte gap. Static scheduling therefore provides no evidence yet of a hidden second tensor issue opcode/domain.
