"""Verify, reduce and archive a completed paired atmospheric CUDA campaign."""
import argparse
import gzip
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from analyze_atmospheric import analyze, REQUIRED, ANCHORS

ROOT = Path(__file__).resolve().parents[1]

def persist(base, output):
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for stage in ('main', 'anchors'):
        for source in (base / stage).iterdir():
            if source.is_file():
                target = raw / source.name
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError(f'conflicting artifact: {source.name}')
                shutil.copy2(source, target)
    metrics = analyze(raw, output)
    assert len(metrics['runs']) == 47 and metrics['pair_count'] == 9
    assert {(p['seed'], p['r0_m']) for p in metrics['coverage']} == {
        (seed, r0) for seed in (1, 2, 3) for r0 in (.10, .15, .20)}
    for pair in metrics['coverage']:
        expected = REQUIRED | ANCHORS if (pair['seed'], pair['r0_m']) == (1, .15) else REQUIRED
        assert {tuple(c) for c in pair['cases']} == expected
    harness = hashlib.sha256((ROOT/'benchmarks/copper_atmospheric.jl').read_bytes()).hexdigest()
    assert all(v['metadata']['harness_sha256'] == harness for v in metrics['runs'].values())
    shutil.move(output/'README.md', output/'DETAILS.md')
    records = output/'records'
    records.mkdir(exist_ok=True)
    for source in base.iterdir():
        if source.is_file() and source.suffix in ('.log', '.json', '.txt', '.md', '.py'):
            if source.suffix == '.log':
                (records/(source.name+'.gz')).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
            else:
                shutil.copy2(source, records/source.name)
    for name in ('Project.toml', 'Manifest.toml'):
        (records/(name+'.gz')).write_bytes(gzip.compress((ROOT/'benchmarks'/name).read_bytes(), mtime=0))
    sources = [ROOT/'benchmarks'/name for name in ('copper_atmospheric.jl', 'run_atmospheric.py',
               'analyze_atmospheric.py', 'persist_atmospheric.py')]
    provenance = {'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources},
                  'raw_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()},
                  'record_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in records.iterdir()}}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    lines = ['# Copper atmospheric and closed-loop convergence', '',
        'CUDA, RTX 3080, 2026-10-21. All production defaults remain unchanged.', '',
        'Three paired seeds at r0 = 0.10, 0.15, 0.20 m; 300 causal feedback updates per run, '
        '50-frame burn-in, 31 open-loop rate/slope samples. Reference: 1440² / 256 points. '
        'Each primary case uses its own complete 277-column calibration with fixed 5% SVD cutoff and gain 0.15.', '',
        '| Pupil / points | Open rate error range % | Open slope error range % | Worst same-command slope error % | Max absolute mean residual RMS difference % | Median plant ms |',
        '|---|---:|---:|---:|---:|---:|']
    for n, m in ((480, 32), (180, 64), (240, 96), (240, 128)):
        runs = [v for v in metrics['runs'].values() if
                (v['metadata']['resolution'], v['metadata']['modulation_points']) == (n, m)]
        assert len(runs) == 9
        rate = [100*v['open_rate']['frobenius_relative_l2'] for v in runs]
        slope = [100*v['open_slope_response']['frobenius_relative_l2'] for v in runs]
        same = max(100*v['same_command']['summary']['slope_response_relative_l2']['worst'] for v in runs)
        rms = max(100*abs(v['steady']['residual_rms_mean_ratio']-1) for v in runs)
        latency = statistics.median(v['plant_p50_us']/1000 for v in runs)
        lines.append(f'| {n} / {m} | {min(rate):.3f}–{max(rate):.3f} | {min(slope):.3f}–{max(slope):.3f} | {same:.3f} | {rms:.3f} | {latency:.3f} |')
    lines += ['', '## Finer-reference checks', '',
        'Seed 1, r0=0.15 m, with the fixed 1440²/256 controller:', '',
        '| Check | Photon-rate image error % | Open slope error % | Mean residual RMS change (nm) |',
        '|---|---:|---:|---:|']
    for key, label in (('n1920-m256-r15-s1', '1920 / 256'), ('n1440-m512-r15-s1', '1440 / 512')):
        value = metrics['runs'][key]
        lines.append(f'| {label} | {100*value["open_rate"]["frobenius_relative_l2"]:.4f} | '
            f'{100*value["open_slope_response"]["frobenius_relative_l2"]:.4f} | '
            f'{value["steady"]["residual_rms_mean_difference_nm"]:+.5f} |')
    lines += ['', 'These are observed sensitivities for one seed/strength, not continuum error bounds.', '']
    throughput = max(v['photon_rate_total']['errors']['relative_l2_by_frame']['worst'] for v in metrics['runs'].values())
    lines += ['', f'The maximum paired total-photon-rate discrepancy is {100*throughput:.5f}%; the table reports photon-rate image errors.', '', 'Error ranges span the nine paired seed/strength run aggregates. Residual differences '
        'compare the means of frames 51–300 with the corresponding fine-reference loop. Lower residual '
        'RMS is better control performance, but a difference from the reference is not greater numerical fidelity. '
        'Temporal samples and different r0 scalings of one seed are correlated.', '',
        'These tests alone do not qualify an unconditional default replacement. They use noiseless '
        'diagnostic slopes, not the operational pyRTC mask/noisy detector controller. Earlier '
        '[spatial-frequency tests](../2026-10-21-copper-convergence/README.md) also exposed sampling errors '
        'outside the original calibration/probe band. No independent physical oracle is claimed.', '',
        'The atmosphere is the native finite-screen model: 960-pixel master pupil and 2880² periodic '
        'screens over 3.66 m. Missing low-frequency power is not restored with subharmonics; nominal r0 '
        'does not establish full atmospheric statistics. All grids sample the same continuous bilinear '
        'screens. See [protocol](records/protocol.md) and [complete numerical reductions](DETAILS.md).', '',
        'Additional 1920²/256 and 1440²/512 runs at seed 1, r0=0.15 m test plant sensitivity with the '
        'same 1440²/256 controller. They are not independently recalibrated anchors. Per-frame data, '
        'science-grid sensitivity and same-command comparisons are retained in [metrics.json](metrics.json).', '',
        'Plant times include captured completion but exclude atmosphere sampling, transfers, host control '
        'and science diagnostics. They are sequential service times, not complete RTC loop latency or '
        'fixed-rate deadline evidence. All recorded plant allocation observations were zero host bytes.', '',
        '## Reproduction', '', 'Use the archived isolated benchmark environment and recorded local source '
        'revisions; registered application environments were not modified. Restore source paths for the '
        'local checkout before loading the archived manifest. The exact commands are in records/.', '',
        '```sh', 'python3 benchmarks/run_atmospheric.py OUTPUT --stage tests',
        'python3 benchmarks/run_atmospheric.py OUTPUT --stage main',
        'python3 benchmarks/run_atmospheric.py OUTPUT --stage anchors',
        'python3 benchmarks/persist_atmospheric.py OUTPUT ARCHIVE', '```', '',
        'The first attempt exhausted preparation memory after all seed-1 records completed; its log is '
        'preserved. Remaining seeds and individual anchors use isolated Julia processes with the same '
        'numerical harness. Source, record and compressed-array hashes are retained in provenance.json.', '']
    (output/'README.md').write_text('\n'.join(lines))
    print(f'Archived {len(metrics["runs"])} runs and {len(provenance["raw_sha256"])} raw artifacts')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    persist(args.base.resolve(), args.output.resolve())
