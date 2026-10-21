"""Run the local paired CUDA study serially in the benchmark environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--stage', choices=('tests', 'main', 'anchors'), required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = ['taskset', '-c', '2', 'julia', '--project=benchmarks', '--threads=1,0',
              '--gcthreads=1', '--startup-file=no']
    calibration = 'benchmarks/results/2026-10-21-copper-convergence/raw/full'
    if args.stage == 'tests':
        command = prefix + ['test/benchmarks/atmospheric_parity.jl']
    else:
        command = prefix + ['benchmarks/copper_atmospheric.jl', str(args.output.resolve()/args.stage), calibration,
                            '300', '1,2,3' if args.stage == 'main' else '1',
                            '.10,.15,.20' if args.stage == 'main' else '.15']
        if args.stage == 'anchors':
            command += ['1920:256,1440:512']
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', COPPER_TEST_CUDA='1')
    (args.output/f'command-{args.stage}.json').write_text(json.dumps(
        {'cwd': str(ROOT), 'argv': command, 'environment':
         {k: env[k] for k in ('OPENBLAS_NUM_THREADS', 'COPPER_TEST_CUDA')}}, indent=2)+'\n')
    with (args.output/f'{args.stage}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

if __name__ == '__main__':
    main()
