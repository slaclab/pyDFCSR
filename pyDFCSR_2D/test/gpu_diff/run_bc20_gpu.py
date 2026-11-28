"""
Indranil's BC20 run (test/bc20_indranil/run_indranil.py, default b176_v2: the L1-17.6284 beam, 1e6
particles, the recommended recipe) on the GPU: backend torch, CUDA, torch_compile, torch_bands.
Same run/record machinery (run_dfcsr_bc20.main); output in benchmark_results/bc20_indranil/runs_gpu.

    CC=$CONDA_PREFIX/bin/gcc python run_bc20_gpu.py [run name]
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'bc20_indranil'))
import run_indranil as RI  # noqa: E402

GPU = dict(backend='torch', device='cuda', torch_compile=True, torch_bands=True)
_config = RI.config


def config(name):
    f = _config(name)
    cfg = yaml.safe_load(open(f))
    cfg['CSR_computation'].update(GPU)
    yaml.safe_dump(cfg, open(f, 'w'), sort_keys=False)
    return f


if __name__ == '__main__':
    RI.OUT = os.path.join(RI.ROOT, 'test', 'benchmark_results', 'bc20_indranil', 'runs_gpu')
    os.makedirs(RI.OUT, exist_ok=True)
    RI.config = config
    RI.main(sys.argv[1] if len(sys.argv) > 1 else 'b176_v2')
