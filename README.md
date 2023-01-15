# Fast work distribution for composable task scheduling engines

## Install dependencies
Using conda:
```bash
conda env create -f environment.yml
conda activate benchmarks
```

## Build & Run
```bash
make clean
make release
make bench # runs benchmarks and saves results to ./bench_results
```

## Plot results
```bash
conda env create -f benchplot.yml
conda activate benchplot
python3 benchplot.py # plots benchmark results and saves images to ./bench_results/images
```
