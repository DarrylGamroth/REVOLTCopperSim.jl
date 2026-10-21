"""Validate and archive the completed local Copper convergence campaign."""
import argparse,gzip,hashlib,json,platform,shutil,statistics,subprocess,tomllib
from pathlib import Path
import numpy as np
from analyze_convergence import load_record,relative,slopes
ROOT=Path(__file__).resolve().parents[1]
CASES=('n480-m32','n180-m64','n120-m96','n240-m96','n240-m128','n480-m128')
ENDPOINTS=('calibration_graph','calibration_hil','atmosphere_graph','atmosphere_hil')

def persist(base,output):
    metrics=json.loads((base/'analysis/metrics.json').read_text())['cases']
    assert len(metrics)==48
    reference_record,reference=load_record(base/'full/n1440-m256.toml')
    benchmark={}
    for case in CASES:
        rows=[]
        for r in (1,2,3):
            d=tomllib.loads((base/f'benchmark-{r}/{case}.toml').read_text())
            assert d['harness_sha256']==reference_record['harness_sha256']
            assert d['graph_sha256']==reference_record['graph_sha256']
            for endpoint in ENDPOINTS:
                x=d[endpoint]
                assert len(x['samples_ns'])==128 and x['host_bytes']==[0]*16
                assert statistics.median(x['samples_ns'])==x['p50_ns']
                assert x['gc_collections']==0
            rows.append(d)
        benchmark[case]={e:dict(p50_us=statistics.median(x[e]['p50_ns']/1000 for x in rows),
            run_min_us=min(x[e]['p50_ns']/1000 for x in rows),run_max_us=max(x[e]['p50_ns']/1000 for x in rows)) for e in ENDPOINTS}
    spatial=json.loads((base/'analysis/spatial-frequency.json').read_text())
    assert len(spatial)==9
    # Anchors measured independently with 11 columns, versus their same-run 1440/256.
    anchor_dir=base/'reference-check'
    ad,ar=load_record(anchor_dir/'n1440-m256.toml')
    anchors=[]
    for case in ('n1920-m256','n1440-m512'):
        d,a=load_record(anchor_dir/(case+'.toml'))
        anchors.append(dict(case=case,
            rate_max=max(relative(a['rates'][:,:,k],ar['rates'][:,:,k]) for k in range(13)),
            calibration=relative(a['interaction'],ar['interaction']),
            slope_response_max=max(relative(slopes(a['rates'][:,:,k],ad['divisor'])-slopes(a['rates'][:,:,0],ad['divisor']),
                slopes(ar['rates'][:,:,k],ad['divisor'])-slopes(ar['rates'][:,:,0],ad['divisor'])) for k in range(2,13))))
    output.mkdir(parents=True,exist_ok=True)
    raw=output/'raw';raw.mkdir(exist_ok=True)
    for directory in ('full','reference-check','pilot','spatial-frequency','benchmark-1','benchmark-2','benchmark-3'):
        if (base/directory).exists():shutil.copytree(base/directory,raw/directory,dirs_exist_ok=True)
    for p in base.glob('*.log'):
        (raw/(p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    for p in base.glob('commands-*.json'):shutil.copy2(p,raw/p.name)
    for name in ('pilot-harness.jl','reference-check-harness.jl','provenance.md','independent-review.md'):
        if (base/name).exists():shutil.copy2(base/name,raw/name)
    for name in ('Manifest.toml','Project.toml'):
        p=ROOT/'benchmarks'/name
        (raw/(name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    for name in ('convergence.svg','accuracy_latency.svg','summary.csv','metrics.json','spatial-frequency.json','plot-version.txt'):
        source=base/'analysis'/name
        if source.suffix in ('.svg','.csv'):
            (output/name).write_text('\n'.join(line.rstrip() for line in source.read_text().splitlines())+'\n')
        else:
            shutil.copy2(source,output/name)
    (output/'benchmark-summary.json').write_text(json.dumps(benchmark,indent=2)+'\n')
    smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True,capture_output=True)
    module_version=Path('/proc/driver/nvidia/version')
    provenance=dict(julia='1.13.2',python=platform.python_version(),numpy=np.__version__,
        source_revision=reference_record['copper_revision'],aos_revision=reference_record['aos_revision'],
        harness_sha256=reference_record['harness_sha256'],graph_sha256=reference_record['graph_sha256'],
        gpu_during_run=reference_record['device'],nvml_post_run_returncode=smi.returncode,
        nvml_post_run_output=smi.stdout+smi.stderr,
        driver_module_post_run=module_version.read_text() if module_version.exists() else None,
        system=subprocess.check_output(['uname','-a'],text=True).strip(),cpu=subprocess.check_output(['lscpu'],text=True),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'benchmarks').glob('*.py')})
    assert hashlib.sha256((ROOT/'benchmarks/copper_convergence.jl').read_bytes()).hexdigest()==provenance['harness_sha256']
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    checksums={str(p.relative_to(raw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file()}
    (output/'raw-sha256.json').write_text(json.dumps(checksums,indent=2,sort_keys=True)+'\n')
    lines=['# Copper pupil and modulation convergence','',
'2026-10-21. CUDA investigation on RTX 3080. Production defaults remain 480²/32 points. This study changes numerical discretization only in private benchmark construction; physical parameters and public constructors are unchanged.','',
'## Findings','',
'1. Copper 480 is inherited configuration, not a demonstrated minimum. Copper samples 30-pixel pupil images at 16 grid samples/pixel; GPI 240 samples 60-pixel pupil images at 4. Their padded FFT grids are 1024² and 960², respectively. The [provenance record](raw/provenance.md) traces the source comments and git history and identifies source contracts that were unavailable for direct inspection.',
'2. The current 32-point quadrature is not converged for the tested small-tilt slopes: 480/32 differs from 1440/256 by 23.10% in worst flat-subtracted probe response, despite a 2.52% aggregate interaction-matrix difference. The full 277 columns are necessary: the 11-column pilot underestimated some matrix errors.',
'3. 180/64 improves all four aggregate screening metrics for the original 13 probes and smooth Gaussian-DM responses while costing roughly one-third of the current noisy HIL cycle. It is a candidate for those tested inputs, not a universally accuracy-preserving default.',
'4. Additional spatial-frequency checks reject that universal claim. A 192/191 cycles-per-aperture probe aliases on 180 to 12/11 cycles; even 32- and 64-cycle probes, below its Nyquist limit, have large response errors. No production sampling default was changed.',
'',
'## Full-size timings and convergence','',
'Each error uses the same Float32 CUDA physical model, fixed detector geometry, common 716-pixel slope support and finest-flat incidence divisor. Slopes are FGA I4Q X/Y after transposing AOS(x,y) to image(row=y,column=x). Relative slope errors use each case\'s own flat-subtracted response. Flat/piston slope errors are absolute only. Calibration is the complete 1432×277 finite central-difference response to ±20 nm actuator surface-OPD commands. All commands retain canonical HSDM277 order.','',
'| Pupil / points | Max rate error % | Max slope-response error % | Matrix error % | Worst column % | Noisy captured HIL ms | Run-median range ms |',
'|---|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        m=metrics[case];t=benchmark[case]['atmosphere_hil']
        lines.append(f'| {m["resolution"]} / {m["modulation_points"]} | {100*m["rate_max_relative_l2"]:.3f} | {100*m["slope_response_max_relative_l2"]:.3f} | {100*m["calibration_relative_l2"]:.3f} | {100*m["calibration_max_column_relative_l2"]:.3f} | {t["p50_us"]/1000:.3f} | {t["run_min_us"]/1000:.3f}–{t["run_max_us"]/1000:.3f} |')
    speedup=benchmark['n480-m32']['atmosphere_hil']['p50_us']/benchmark['n180-m64']['atmosphere_hil']['p50_us']
    lines+=['',f'180/64 measured {speedup:.2f}× faster than 480/32 in this completed noisy HIL endpoint. The 120/96 case is faster still but worsens the worst calibration-column metric. The 240/96 and 240/128 cases offer lower reference errors at intermediate cost. These are descriptive results for this finite workload matrix, not a global optimum or a physical accuracy acceptance threshold.','',
'![Convergence errors](convergence.svg)','',
'![Accuracy versus latency](accuracy_latency.svg)','',
'The [complete 48-case table](summary.csv), [per-probe/column/spectrum metrics](metrics.json), [repeat timing summary](benchmark-summary.json) and raw arrays are retained. Fixed-phase trapezoidal angular quadrature converges non-monotonically; increasing point count does not guarantee every metric decreases at every step. Relative singular-value threshold counts at 5% are diagnostics, not mathematical rank or operational reconstructor qualification.','',
'## Finer-reference checks','',
'The primary reference is 1440²/256. Full 277-column spatial and angular anchors are 960²/256 and 1440²/128; see the complete table. Separate 1920²/256 and 1440²/512 checks use all 13 probes and 11 selected actuator columns. Their differences against the same-run 1440²/256 are:','',
'| Finer check | Max photon-rate change % | Max slope-response change % | Sampled matrix change % |','|---|---:|---:|---:|']
    for a in anchors:lines.append(f'| {a["case"]} | {100*a["rate_max"]:.4f} | {100*a["slope_response_max"]:.4f} | {100*a["calibration"]:.4f} |')
    lines+=['',
'These are observed reference sensitivities, not rigorous continuum error bounds. They apply to these probes and sampled columns. In particular, the finer spatial-frequency probes have their own sensitivity checks below; the small sensitivities above do not qualify all spatial frequencies.','',
'## Additional spatial-frequency checks','',
'These checks use 32/31, 64/63, 96/95 and 192/191 cycle 2D sinusoidal OPD patterns with 20 nm coefficients, plus a deterministic multiscale field with a 100 nm coefficient and f⁻⁵⁄⁶ amplitude envelope. The latter is not 100 nm RMS and is not a statistically validated atmospheric ensemble.','',
'| Case | 192-cycle rate error % | Flat-subtracted response error RMS | Reference response RMS | Relative response error % | Multiscale slope error % |','|---|---:|---:|---:|---:|---:|']
    for case in ('n480-m32','n180-m64','n240-m96','n480-m128','n1920-m256','n1440-m512'):
        rows=spatial[case];p=rows[4]
        lines.append(f'| {case} | {100*p["rate_relative_l2"]:.3f} | {p["slope_response_error_rms"]:.6g} | {p["reference_slope_response_rms"]:.6g} | {100*p["slope_response_relative_l2"]:.2f} | {100*rows[-1]["slope_response_relative_l2"]:.3f} |')
    lines+=['',
'The full spatial-frequency results include raw and flat-subtracted errors. A small reference-response norm can produce a large relative error, so absolute response errors are shown. The high-frequency reference is itself sensitive: 1920/256 changes the 192/191 response by 14.07% (absolute RMS 7.074×10⁻⁵) and the 96/95 response by 10.97%; doubling angular samples to 512 changes the 192/191 response by 0.811%. These sensitivities preclude treating this high-frequency reference as converged, but they are far smaller than the 180-grid alias error. Even the 480 preset is not well converged for every high-frequency probe. Nyquist representability is necessary but insufficient for accurate propagation. Passing the original low-frequency probes cannot establish fidelity for all inputs below Nyquist.','',
'## Method, validation and limits','',
'- 48 full cases: 120,180,240,360,480,720 crossed with 8,16,24,32,48,64 points; plus 96,128,256 at 120,240,480; plus the three primary/reference-anchor cases. Each has 13 OPD probes and 554 signed actuator exposures. Inputs are analytic functions on the exact centred pupil grid, without interpolation between resolutions.',
'- Fixed 1.22 m aperture, 30 pixel pupils, 64×64 detector, 5 λ/D modulation radius, 550 nm source, source flux, exposure, Gaussian DM model and 277-command mapping. Reference comparison uses noiseless photon-rate outputs before detector noise/quantization; benchmark endpoints include separate noiseless calibration and noisy five-layer-atmosphere graphs.',
'- The padded side is 64 Npupil/30. Thus focal sample spacing stays 0.46875 λ/D across every pupil/reference grid. The experiment does not refine focal spacing or test the shifted-mask approximation against pupil-tilt propagation. It is same-model discretization convergence, not an independent SPECULA, bench, turbulence-statistics or closed-loop oracle.',
'- The primary sweep has 32 warmups, 16 allocation samples and 128 timing samples per completed endpoint. Selected cases repeat in three fresh Julia processes, with case order reversed in round 2 and rotated in round 3. Reported times are medians of process medians. Endpoints include completion; HIL additionally stages the 64×64 host frame and adopts the matching command. There is no transport, external controller or queueing.',
'- Every selected endpoint measured 0 host bytes in all 48 allocation samples; all 45 timed sweep cases also passed zero-allocation assertions. Repeated timed intervals had no GC. This does not claim arbitrary transport, device-memory-pool or fixed-rate deadline behavior.',
'- Julia 1.13.2, CUDA 6.4.2, KA 0.9.45, RTX 3080, CPU 2 affinity, 1 compute thread, no interactive pool, 1 GC thread and BLAS/FFT 1. GPU clocks were uncontrolled on a shared host. Numerical reductions use Float64; graphs and archived arrays use Float32. Arrays are little-endian, column-major `.f32.gz`, with shape/type/checksum in companion TOML.',
'- CPU preparation regression 10/10, measurement regression 19/19, Python reductions 3/3, CUDA oracle/capture 5/5 passed. CPU/CUDA maximum photon-rate error 2.876×10⁻⁶; selected calibration error 9.879×10⁻⁶. The cross-check is not an independent scientific model.',
'- All CUDA runs completed successfully. A later archival NVML query reported a driver/library mismatch. The post-run error and loaded kernel-module version are recorded in provenance; no driver reset or service change was performed.',
'- Independent Astra review checked coordinates, normalization, command clearing, reference checks, FGA axis conversion, source hashes and the 192→12-cycle alias identity. Source and data findings are retained in the review record.','',
'## Reproduction','',
'Use an isolated validation environment with the archived project/manifest. This run used local AOS 0.2.2 source; registered application environments were not modified. The benchmark project requires those capture/HIL fixes; its ordinary registered setup becomes usable when that AOS release is available. Do not add development overrides to maintained application environments.','',
'```sh',
'OPENBLAS_NUM_THREADS=1 python3 benchmarks/run_convergence.py /tmp/copper-convergence --stage all',
'OPENBLAS_NUM_THREADS=1 python3 benchmarks/analyze_convergence.py /tmp/copper-convergence/full /tmp/copper-convergence/analysis --extended --plot --spatial /tmp/copper-convergence/spatial-frequency',
'```','',
'Analysis requires NumPy; figures additionally require Matplotlib. The run records its versions. `run_convergence.py` writes exact commands and logs and serializes GPU jobs. It does not install dependencies or connect to RTC services. Existing case metadata check the harness SHA, `graphs.jl` SHA, AOS git HEAD and pilot/full/benchmark flags; use a new output directory after changing the environment. Pilot/reference-check snapshots are archived because source/provenance hardening occurred before freezing the full sweep at commit `'+reference_record['copper_revision']+'`.','',
'Production defaults remain unchanged. A default change needs a defined spatial-frequency/phase-amplitude envelope and quantitative error budgets for the intended workload, followed by statistical-atmosphere and closed-loop validation. The measured frontier supplies candidates for that decision.','']
    text='\n'.join(lines)
    (output/'README.md').write_text(text)
    print('Persisted',len(metrics),'full cases and',len(checksums),'raw artifacts')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();persist(a.base,a.output)
