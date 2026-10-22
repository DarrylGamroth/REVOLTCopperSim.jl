"""Offline verification of the final 47-record atmospheric archive and anchors."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import gzip,hashlib,json,tomllib
from pathlib import Path
import numpy as np
repo=Path('/home/dgamroth/workspaces/codex/REVOLTCopperSim-atmospheric')
archive=repo/'benchmarks/results/2026-10-21-copper-atmospheric'
evidence=Path('/home/dgamroth/workspaces/codex/copper-atmospheric-evidence')
raw=archive/'raw'; metrics=json.loads((archive/'metrics.json').read_text())
provenance=json.loads((archive/'provenance.json').read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def arr(record,key,folder=raw):
    d=record[key]; p=folder/d['file']; assert sha(p)==d['sha256']
    a=np.frombuffer(gzip.decompress(p.read_bytes()),dtype='<f4').reshape(d['shape'],order='F').astype(float)
    assert np.isfinite(a).all();return a

def slopes(a,div):
    x=[];y=[]
    for j in range(30):
      for i in range(30):
        if (i-14.5)**2+(j-14.5)**2<=225:
          q=[a[c+j,r+i] for r,c in ((1,1),(33,1),(33,33),(1,33))]
          x.append((q[0]-q[1]+q[3]-q[2])/div);y.append((q[0]-q[3]+q[1]-q[2])/div)
    return np.array(x+y)
def relative(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def close(a,b):assert np.isclose(a,b,rtol=1e-11,atol=1e-13),(a,b)

assert len(provenance['raw_sha256'])==833
for group,folder in (('raw_sha256',raw),('record_sha256',archive/'records')):
    assert set(provenance[group])=={p.name for p in folder.iterdir() if p.is_file()}
    for name,h in provenance[group].items():assert sha(folder/name)==h,(group,name)
for name,h in provenance['source_hashes'].items():assert sha(repo/name)==h,name
assert sha(repo/'benchmarks/analyze_atmospheric.py')==metrics['analyzer_sha256']
metadata={p.stem:tomllib.loads(p.read_text()) for p in raw.glob('*.toml')}
expected={(n,m,r0,s) for n,m in ((1440,256),(480,32),(180,64),(240,96),(240,128)) for r0 in (.10,.15,.20) for s in (1,2,3)}|{(1920,256,.15,1),(1440,512,.15,1)}
assert {(d['resolution'],d['modulation_points'],d['r0_m'],d['seed']) for d in metadata.values()}==expected
assert len(metadata)==47 and set(metadata)==set(metrics['runs']) and metrics['pair_count']==9
harness=sha(repo/'benchmarks/copper_atmospheric.jl'); assert harness=='f0c9f2e384cc6eee5e7e8c487e137eb441e17bfa29b19bacc6999d76c1215669'
reference_totals={};max_throughput=0.
main_before={f"n{x['n']}-m{x['m']}-r{round(100*x['r0'])}-s{x['seed']}":x for x in json.loads((evidence/'independent-review-main.json').read_text())}
for name,d in metadata.items():
    v=metrics['runs'][name]
    assert v['record_sha256']==sha(raw/(name+'.toml'))
    assert d['harness_sha256']==harness
    assert d['frames']==300 and d['burn_in']==50 and d['open_frame_ids']==[1,*range(10,301,10)]
    rate=arr(d,'rates');flat=arr(d,'flat');assert np.all(rate>=0) and np.all(flat>=0)
    rn=f"n1440-m256-r{round(100*d['r0_m'])}-s{d['seed']}"
    if rn not in reference_totals:reference_totals[rn]=arr(metadata[rn],'rates').sum(axis=(0,1))
    total=rate.sum(axis=(0,1));expected_total=reference_totals[rn]
    discrepancy=np.abs(total-expected_total)/expected_total
    max_throughput=max(max_throughput,float(discrepancy.max()))
    tp=v['photon_rate_total'];close(tp['own_mean_photons_s'],float(total.mean()));close(tp['reference_mean_photons_s'],float(expected_total.mean()))
    close(tp['errors']['frobenius_relative_l2'],relative(total,expected_total));close(tp['errors']['relative_l2_by_frame']['worst'],float(discrepancy.max()))
    if name in main_before:
        old=main_before[name]
        close(v['open_rate']['frobenius_relative_l2'],old['rate_error']);close(v['open_slope_response']['frobenius_relative_l2'],old['slope_error'])
        close(v['steady']['residual_rms_mean_ratio'],old['residual_ratio']);close(v['steady']['strehl_mean'],old['strehl'])
        assert v['case_role']==('numerical_reference' if (d['resolution'],d['modulation_points'])==(1440,256) else 'own_calibrated_candidate')

rn='n1440-m256-r15-s1';rd=metadata[rn]
ra={k:arr(rd,k) for k in ('rates','flat','open_slopes','closed_slopes','commands','diagnostics')}
caldir=repo/'benchmarks/results/2026-10-21-copper-convergence/raw/full'
cd=tomllib.loads((caldir/'n1440-m256.toml').read_text());a=arr(cd,'interaction',caldir)
u,s,vh=np.linalg.svd(a,full_matrices=False);keep=s>.05*s[0];inverse=(vh[keep].T/s[keep])@u[:,keep].T
anchors=[]
for name in ('n1920-m256-r15-s1','n1440-m512-r15-s1'):
    d=metadata[name];v=metrics['runs'][name];a={k:arr(d,k) for k in ra}
    assert (d['calibration_resolution'],d['calibration_points'])==(1440,256)
    for k in ('screen_hashes','seed','r0_m','dt_s','gain','rcond','divisor','master_pupil_resolution','science_resolution','science_wavelength_m','harness_sha256','base_harness_sha256','graph_sha256','aos_revision','copper_revision'):
        assert d[k]==rd[k],(name,k)
    assert d['retained_rank']==int(keep.sum())==231
    assert v['case_role']=='reference_calibration_plant_sensitivity'
    assert not v['same_command']['available'] and 'optional anchor' in v['same_command']['unavailable_reason']
    assert all('own_rate' not in c and 'fine_rate' not in c for c in d['checks'])
    assert np.array_equal(a['diagnostics'][2:4],ra['diagnostics'][2:4])
    assert np.all(a['commands'][:,0]==0)
    signal=np.stack([slopes(a['rates'][:,:,k],d['divisor'])-slopes(a['flat'],d['divisor']) for k in range(31)],axis=1)
    assert np.allclose(signal,a['open_slopes'],rtol=2e-6,atol=2e-9)
    causal=float(np.max(abs(a['commands'][:,1:]-a['commands'][:,:-1]+d['gain']*inverse@a['closed_slopes'][:,:-1])));assert causal<1e-11
    post=np.array(d['open_frame_ids'])>50
    result={'case':name,'rank':int(keep.sum()),'causal_error_m':causal,
      'open_rate_relative_l2':relative(a['rates'],ra['rates']),
      'open_slope_relative_l2':relative(a['open_slopes'],ra['open_slopes']),
      'steady_open_rate_relative_l2':relative(a['rates'][:,:,post],ra['rates'][:,:,post]),
      'steady_open_slope_relative_l2':relative(a['open_slopes'][:,post],ra['open_slopes'][:,post]),
      'closed_trajectory_slope_relative_l2':relative(a['closed_slopes'],ra['closed_slopes']),
      'command_trajectory_relative_l2':relative(a['commands'],ra['commands']),
      'residual_rms_mean_difference_nm':float(np.mean(a['diagnostics'][0,50:]-ra['diagnostics'][0,50:])*1e9),
      'residual_rms_mean_ratio':float(np.mean(a['diagnostics'][0,50:])/np.mean(ra['diagnostics'][0,50:])),
      'strehl_mean_difference':float(np.mean(a['diagnostics'][1,50:]-ra['diagnostics'][1,50:]))}
    for key,category in (('open_rate_relative_l2','open_rate'),('open_slope_relative_l2','open_slope_response'),('steady_open_rate_relative_l2','steady_open_rate'),('steady_open_slope_relative_l2','steady_open_slope_response'),('closed_trajectory_slope_relative_l2','closed_trajectory_slope_response'),('command_trajectory_relative_l2','command_trajectory')):close(result[key],v[category]['frobenius_relative_l2'])
    for key in ('residual_rms_mean_difference_nm','residual_rms_mean_ratio','strehl_mean_difference'):close(result[key],v['steady'][key])
    result['max_science_480_vs_1440_rms_nm']=max(abs(a['diagnostics'][0,c['frame']-1]-c['science_1440_rms_m'])*1e9 for c in d['checks'])
    result['max_science_paired_effect_rms_nm']=max(abs((a['diagnostics'][0,c['frame']-1]-ra['diagnostics'][0,c['frame']-1])-(c['science_1440_rms_m']-rc['science_1440_rms_m']))*1e9 for c,rc in zip(d['checks'],rd['checks']))
    anchors.append(result)
readme=(archive/'README.md').read_text()
assert f'{100*max_throughput:.5f}%' in readme
assert 'They are not independently recalibrated anchors' in readme
assert 'Missing low-frequency power' in readme and 'not greater numerical fidelity' in readme
result={'raw_files_verified':len(provenance['raw_sha256']),'supporting_records_verified':len(provenance['record_sha256']),
    'run_count':47,'pair_count':9,'source_hashes_verified':provenance['source_hashes'],
    'maximum_total_photon_rate_fractional_discrepancy':max_throughput,'anchors':anchors}
(evidence/'independent-review-final.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
