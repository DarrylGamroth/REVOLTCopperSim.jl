# Copper atmospheric and closed-loop convergence

CUDA, RTX 3080, 2026-10-21. All production defaults remain unchanged.

Three paired seeds at r0 = 0.10, 0.15, 0.20 m; 300 causal feedback updates per run, 50-frame burn-in, 31 open-loop rate/slope samples. Reference: 1440² / 256 points. Each primary case uses its own complete 277-column calibration with fixed 5% SVD cutoff and gain 0.15.

| Pupil / points | Open rate error range % | Open slope error range % | Worst same-command slope error % | Max absolute mean residual RMS difference % | Median plant ms |
|---|---:|---:|---:|---:|---:|
| 480 / 32 | 1.284–1.856 | 2.044–3.009 | 4.554 | 0.644 | 2.564 |
| 180 / 64 | 1.266–1.612 | 1.628–1.883 | 2.196 | 1.123 | 0.736 |
| 240 / 96 | 0.599–0.626 | 0.596–0.753 | 0.952 | 0.100 | 1.851 |
| 240 / 128 | 0.591–0.620 | 0.638–0.688 | 0.921 | 0.181 | 2.455 |

## Finer-reference checks

Seed 1, r0=0.15 m, with the fixed 1440²/256 controller:

| Check | Photon-rate image error % | Open slope error % | Mean residual RMS change (nm) |
|---|---:|---:|---:|
| 1920 / 256 | 0.0390 | 0.0285 | -0.00926 |
| 1440 / 512 | 0.0247 | 0.0890 | +0.02416 |

These are observed sensitivities for one seed/strength, not continuum error bounds.


The maximum paired total-photon-rate discrepancy is 0.00669%; the table reports photon-rate image errors.

Error ranges span the nine paired seed/strength run aggregates. Residual differences compare the means of frames 51–300 with the corresponding fine-reference loop. Lower residual RMS is better control performance, but a difference from the reference is not greater numerical fidelity. Temporal samples and different r0 scalings of one seed are correlated.

These tests alone do not qualify an unconditional default replacement. They use noiseless diagnostic slopes, not the operational pyRTC mask/noisy detector controller. Earlier [spatial-frequency tests](../2026-10-21-copper-convergence/README.md) also exposed sampling errors outside the original calibration/probe band. No independent physical oracle is claimed.

The atmosphere is the native finite-screen model: 960-pixel master pupil and 2880² periodic screens over 3.66 m. Missing low-frequency power is not restored with subharmonics; nominal r0 does not establish full atmospheric statistics. All grids sample the same continuous bilinear screens. See [protocol](records/protocol.md) and [complete numerical reductions](DETAILS.md).

Additional 1920²/256 and 1440²/512 runs at seed 1, r0=0.15 m test plant sensitivity with the same 1440²/256 controller. They are not independently recalibrated anchors. Per-frame data, science-grid sensitivity and same-command comparisons are retained in [metrics.json](metrics.json).

Plant times include captured completion but exclude atmosphere sampling, transfers, host control and science diagnostics. They are sequential service times, not complete RTC loop latency or fixed-rate deadline evidence. All recorded plant allocation observations were zero host bytes.

## Reproduction

Use the archived isolated benchmark environment and recorded local source revisions; registered application environments were not modified. Restore source paths for the local checkout before loading the archived manifest. The exact commands are in records/.

```sh
python3 benchmarks/run_atmospheric.py OUTPUT --stage tests
python3 benchmarks/run_atmospheric.py OUTPUT --stage main
python3 benchmarks/run_atmospheric.py OUTPUT --stage anchors
python3 benchmarks/persist_atmospheric.py OUTPUT ARCHIVE
```

The first attempt exhausted preparation memory after all seed-1 records completed; its log is preserved. Remaining seeds and individual anchors use isolated Julia processes with the same numerical harness. Source, record and compressed-array hashes are retained in provenance.json.
