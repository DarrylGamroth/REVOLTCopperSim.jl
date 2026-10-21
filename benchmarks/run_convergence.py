"""Serial CUDA campaign driver. Activate an isolated Julia benchmark environment first."""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXTENDED='''include(ARGS[2]); using .CopperConvergence
const C=CopperConvergence
C.CUDA.allowscalar(false); C.BLAS.set_num_threads(1); C.AOS.Backends.set_fft_provider_threads!(1)
dir=ARGS[1]; ref=C.TOML.parsefile(joinpath(dir,"n1440-m256.toml"))
for n in (120,240,480), m in (96,128,256)
    C.run_case(dir,n,m;reference=ref)
end
'''
REFERENCE='''include(ARGS[3]); using .CopperConvergence
const C=CopperConvergence
C.CUDA.allowscalar(false); C.BLAS.set_num_threads(1); C.AOS.Backends.set_fft_provider_threads!(1)
ref=C.TOML.parsefile(joinpath(ARGS[2],"n1440-m256.toml"))
for (n,m) in ((1440,256),(1920,256),(1440,512))
    C.run_case(ARGS[1],n,m;reference=ref,full_calibration=false,benchmark=false)
end
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--stage',choices=('all','sweep','extended','reference','benchmark','spatial','oracle'),default='all')
    p.add_argument('--project',type=Path,default=ROOT/'benchmarks')
    p.add_argument('--julia',default='julia');p.add_argument('--cpu',default='2')
    p.add_argument('--cases',default='480:32,180:64,120:96,240:96,240:128,480:128')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    prefix=['taskset','-c',a.cpu,a.julia,'--startup-file=no','--threads=1,0','--gcthreads=1',f'--project={a.project.resolve()}']
    env=dict(os.environ,JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    commands=[]
    def run(label,args):
        cmd=prefix+list(map(str,args));commands.append(dict(stage=label,argv=cmd))
        record=a.output/f'commands-{a.stage}.json'
        record.write_text(json.dumps(commands,indent=2)+'\n')
        print(label,flush=True)
        with (a.output/f'{label}.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    stages=('sweep','extended','reference','spatial','oracle','benchmark') if a.stage=='all' else (a.stage,)
    script=ROOT/'benchmarks/copper_convergence.jl'
    for stage in stages:
        if stage=='sweep':run('full',[script,a.output/'full','sweep'])
        elif stage=='extended':run('extended',['-e',EXTENDED,a.output/'full',script])
        elif stage=='reference':run('reference-check',['-e',REFERENCE,a.output/'reference-check',a.output/'full',script])
        elif stage=='spatial':run('spatial-frequency',[ROOT/'benchmarks/copper_spatial_frequency.jl',a.output/'spatial-frequency','1440:256,1920:256,1440:512,'+a.cases])
        elif stage=='oracle':run('cuda-oracle',[ROOT/'test/benchmarks/cuda_convergence.jl'])
        else:
            pairs=a.cases.split(',')
            for round in range(1,4):
                order=pairs if round==1 else list(reversed(pairs)) if round==2 else pairs[2:]+pairs[:2]
                run(f'benchmark-{round}',[script,a.output/f'benchmark-{round}','benchmark',','.join(order)])
if __name__=='__main__':main()
