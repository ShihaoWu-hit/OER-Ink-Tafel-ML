"""Compute the manuscript revision analyses using the released repository inputs."""
from pathlib import Path
import argparse, os, subprocess, sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['variance','baseline','posterior','four','boundaries','probability','high-focus','ale','all'])
    parser.add_argument('--repo', type=Path, help='existing repository root; defaults to the location of this code')
    parser.add_argument('--output', type=Path, required=True, help='new output directory; existing results are not overwritten')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--resume', action='store_true', help='reuse matching local fold caches and regenerate summary tables')
    args = parser.parse_args()
    if args.jobs < 1: parser.error('--jobs must be positive')
    if args.repo: os.environ['TAFEL_REPO_ROOT'] = str(args.repo.resolve())
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('output is not empty; use a new directory or explicitly pass --resume')
    from repository import ROOT
    protected = ROOT / 'Data'
    if out == protected or protected in out.parents:
        parser.error('output must not be inside the released Data directory')
    out.mkdir(parents=True, exist_ok=True)
    stages = ['variance','baseline','posterior','four','boundaries','probability','high-focus','ale'] if args.stage == 'all' else [args.stage]
    here = Path(__file__).resolve().parent
    for stage in stages:
        print('=== '+stage+' ===', flush=True)
        if stage == 'variance':
            subprocess.run([sys.executable, str(here/'variance_decomposition.py'), '--output-dir', str(out/'variance')], check=True)
        elif stage == 'baseline':
            subprocess.run([sys.executable, str(here/'baseline_comparison.py'), '--output', str(out/'baseline')], check=True)
            subprocess.run([sys.executable, str(here/'baseline_contrasts.py'), '--results', str(out/'baseline')], check=True)
        else:
            from importlib import import_module
            modules = {'posterior':('posterior_resolution',False), 'four':('four_regions',True),
                       'boundaries':('boundary_sensitivity',True), 'probability':('probability_stability',True),
                       'high-focus':('high_focus_stability',False), 'ale':('effect_size',True)}
            name, parallel = modules[stage]
            import_module('tafel_si_s7.'+name).run(out/name, **({'jobs':args.jobs} if parallel else {}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
