import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,gzip,tomllib,hashlib,json
from pathlib import Path
import numpy as np
repo=Path('/home/dgamroth/workspaces/codex/REVOLTCopperSim-atmospheric')
raw=Path('/home/dgamroth/workspaces/codex/copper-atmospheric-evidence/main')
cal=repo/'benchmarks/results/2026-10-21-copper-convergence/raw/full'
sys.path.insert(0,str(repo/'benchmarks'))
import analyze_atmospheric as reducer

def readarr(folder,desc):
    b=(folder/desc['file']).read_bytes()
    assert hashlib.sha256(b).hexdigest()==desc['sha256']
    a=np.frombuffer(gzip.decompress(b),dtype='<f4').reshape(desc['shape'],order='F').astype(float)
    assert np.isfinite(a).all()
    return a

def slope(a,divisor):
    x=[]; y=[]
    for j in range(30):
      for i in range(30):
        if (i-14.5)**2+(j-14.5)**2<=225:
          q=[a[c+j,r+i] for r,c in ((1,1),(33,1),(33,33),(1,33))]
          x.append((q[0]-q[1]+q[3]-q[2])/divisor)
          y.append((q[0]-q[3]+q[1]-q[2])/divisor)
    return np.array(x+y)

def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
records={}; inverses={}
for path in sorted(raw.glob('*-s1.toml')):
    d=tomllib.loads(path.read_text()); assert d['seed']==1
    a={k:readarr(raw,d[k]) for k in ('rates','flat','open_slopes','closed_slopes','commands','diagnostics')}
    records[(d['resolution'],d['modulation_points'],d['r0_m'])]=(path,d,a)
assert len(records)==15
rows=[]
for (n,m,r0),(path,d,a) in records.items():
    _,rd,ra=records[(1440,256,r0)]
    assert d['screen_hashes']==rd['screen_hashes'] and d['open_frame_ids']==rd['open_frame_ids']
    post=np.array(d['open_frame_ids'])>50
    assert sum(post)==25 and a['diagnostics'].shape[1]==300
    assert np.array_equal(a['diagnostics'][2:4],ra['diagnostics'][2:4])
    ss=np.stack([slope(a['rates'][:,:,k],d['divisor'])-slope(a['flat'],d['divisor']) for k in range(31)],axis=1)
    slope_abs=float(np.max(np.abs(ss-a['open_slopes'])))
    assert np.allclose(ss,a['open_slopes'],rtol=2e-6,atol=2e-9)
    assert np.array_equal(a['commands'][:,0],np.zeros(277))
    assert a['diagnostics'][0,0]==a['diagnostics'][2,0] and a['diagnostics'][1,0]==a['diagnostics'][3,0]
    if (n,m) not in inverses:
        cd=tomllib.loads((cal/f'n{n}-m{m}.toml').read_text())
        assert cd['divisor']==d['divisor'] and cd['calibration_columns']==list(range(1,278))
        aa=readarr(cal,cd['interaction']); u,s,vh=np.linalg.svd(aa,full_matrices=False)
        keep=s>0.05*s[0]; inverses[n,m]=((vh[keep].T/s[keep])@u[:,keep].T,int(keep.sum()))
    inv,rank=inverses[n,m]; assert rank==d['retained_rank']
    causal=np.max(np.abs(a['commands'][:,1:]-a['commands'][:,:-1]+d['gain']*inv@a['closed_slopes'][:,:-1]))
    assert causal<1e-11
    sci=[]; same=[]
    for c in d['checks']:
        k=c['frame']-1
        sci.append([abs(a['diagnostics'][0,k]-c['science_1440_rms_m'])*1e9,
                    abs(a['diagnostics'][1,k]-c['science_1440_strehl']),
                    abs(c['science_960_rms_m']-c['science_1440_rms_m'])*1e9,
                    abs(c['science_960_strehl']-c['science_1440_strehl'])])
        if 'own_rate' in c:
            own=readarr(raw,c['own_rate']); fine=readarr(raw,c['fine_rate'])
            os=slope(own,d['divisor'])-slope(a['flat'],d['divisor'])
            fs=slope(fine,d['divisor'])-slope(ra['flat'],d['divisor'])
            rr=rel(own,fine); sr=rel(os,fs); se=float(np.sqrt(np.mean((os-fs)**2)))
            assert np.isclose(rr,c['same_command_rate_relative_l2'],rtol=5e-5,atol=1e-10)
            assert np.isclose(sr,c['same_command_slope_relative_l2'],rtol=5e-5,atol=1e-10)
            assert np.isclose(se,c['same_command_slope_error_rms'],rtol=5e-5,atol=1e-10)
            same.append([c['frame'],rr,sr,se])
    result={'n':n,'m':m,'r0':r0,'rank':rank,'rate_error':rel(a['rates'],ra['rates']),
        'slope_error':rel(a['open_slopes'],ra['open_slopes']),
        'post_rate_error':rel(a['rates'][:,:,post],ra['rates'][:,:,post]),
        'post_slope_error':rel(a['open_slopes'][:,post],ra['open_slopes'][:,post]),
        'residual_nm':float(a['diagnostics'][0,50:].mean()*1e9),
        'residual_ratio':float(a['diagnostics'][0,50:].mean()/ra['diagnostics'][0,50:].mean()),
        'strehl':float(a['diagnostics'][1,50:].mean()),
        'open_nm':float(a['diagnostics'][2,50:].mean()*1e9),
        'command_norm_ratio':float(np.linalg.norm(a['commands'][:,50:])/np.linalg.norm(ra['commands'][:,50:])),
        'causal_error_m':float(causal),'open_slope_archive_roundoff':slope_abs,
        'science_max_errors':np.max(sci,axis=0).tolist(),'same_command':same}
    # Independently recomputed quantities compared against current reducer output.
    dd,ar=reducer.load_record(path); rr,rar=reducer.load_record(records[(1440,256,r0)][0])
    reduced=reducer.compare(dd,ar,rr,rar,50)
    assert np.isclose(reduced['open_rate']['frobenius_relative_l2'],result['rate_error'],rtol=1e-12,atol=1e-14)
    assert np.isclose(reduced['open_slope_response']['frobenius_relative_l2'],result['slope_error'],rtol=1e-12,atol=1e-14)
    assert np.isclose(reduced['steady']['residual_rms_mean_ratio'],result['residual_ratio'],rtol=1e-12)
    assert np.isclose(reduced['steady_open_rate']['frobenius_relative_l2'],result['post_rate_error'],rtol=1e-12,atol=1e-14)
    assert np.isclose(reduced['steady_open_slope_response']['frobenius_relative_l2'],result['post_slope_error'],rtol=1e-12,atol=1e-14)
    assert np.isclose(reduced['steady']['strehl_mean'],result['strehl'],rtol=1e-12)
    rows.append(result)
print(json.dumps(rows,indent=2))
Path('/tmp/atmospheric-independent-seed1.json').write_text(json.dumps(rows,indent=2)+'\n')
