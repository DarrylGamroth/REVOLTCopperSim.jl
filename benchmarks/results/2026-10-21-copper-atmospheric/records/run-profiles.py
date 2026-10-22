"""Serial unprofiled repetitions followed by a separate CUPTI diagnostic."""
from pathlib import Path
import json, os, subprocess
base=Path(__file__).resolve().parent
out=base/'aos-profiles';out.mkdir(exist_ok=True)
aos=Path('/home/dgamroth/workspaces/codex/AdaptiveOpticsSim-copper-review')
copper=Path('/home/dgamroth/workspaces/codex/REVOLTCopperSim-atmospheric')
for label,samples,profile in [('run1',128,'0'),('run2',128,'0'),('run3',128,'0'),('diagnostic',16,'1')]:
    command=['taskset','-c','2','julia',f'--project={base / "profile-env"}',
             '--threads=1,0','--gcthreads=1','--startup-file=no',
             str(aos/'benchmarks/profile_copper_cuda.jl'),str(copper),str(out/(label+'.toml')),str(samples)]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',AOS_COPPER_PROFILE=profile)
    (out/(label+'-command.json')).write_text(json.dumps({'argv':command,'cwd':str(aos),
        'environment':{'OPENBLAS_NUM_THREADS':'1','AOS_COPPER_PROFILE':profile}},indent=2)+'\n')
    with (out/(label+'.log')).open('w') as log:
        subprocess.run(command,cwd=aos,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
