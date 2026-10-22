"""Run paired CUDA validation serially, isolating each seed's FFT owners."""
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
    p.add_argument('--seeds', default='1,2,3')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = ['taskset', '-c', '2', 'julia', '--project=benchmarks', '--threads=1,0',
              '--gcthreads=1', '--startup-file=no']
    calibration = 'benchmarks/results/2026-10-21-copper-convergence/raw/full'
    if args.stage == 'tests':
        jobs = [('tests', prefix + ['test/benchmarks/atmospheric_parity.jl'])]
    else:
        base = prefix + ['benchmarks/copper_atmospheric.jl', str(args.output.resolve()/args.stage), calibration, '300']
        if args.stage == 'main':
            jobs = [(f'main-seed{seed}', base + [str(seed), '.10,.15,.20'])
                    for seed in map(int, args.seeds.split(','))]
        else:
            jobs = [(f'anchor-{pair.replace(":", "-")}', base + ['1', '.15', pair])
                    for pair in ('1920:256', '1440:512')]
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', COPPER_TEST_CUDA='1')
    for label, command in jobs:
        (args.output/f'command-{label}.json').write_text(json.dumps(
            {'cwd': str(ROOT), 'argv': command, 'environment':
             {k: env[k] for k in ('OPENBLAS_NUM_THREADS', 'COPPER_TEST_CUDA')}}, indent=2)+'\n')
        with (args.output/f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

if __name__ == '__main__':
    main()
