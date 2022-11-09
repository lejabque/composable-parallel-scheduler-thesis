.PHONY: all
all: release test bench_noop

release:
	cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -C cmake-build-release -j$(shell nproc)

debug:
	cmake -B cmake-build-debug -S . -DCMAKE_BUILD_TYPE=Debug && make -C cmake-build-debug -j$(shell nproc)

clean:
	rm -rf cmake-build-*

clean_bench:
	rm -f bench_results/*

bench_dir:
	mkdir -p bench_results

# todo: separate balanced/unbalanced
bench_spmv:
	@echo ----------------------------------------------------------------------------
	@echo -e "Bench\tRows\tColumns            \tIters\tTime\tCPU\tUnit"
	@echo ----------------------------------------------------------------------------
	@for x in $(shell ls -1 cmake-build-release/benchmarks/bench_spmv_* | xargs -n 1 basename | sort ) ; do numactl -N 0 cmake-build-release/benchmarks/$$x --benchmark_out_format=json --benchmark_out=bench_results/$$x.json; done


bench_reduce:
	@echo ----------------------------------------------------------------------------
	@echo -e "Bench\tRows\tColumns            \tIters\tTime\tCPU\tUnit"
	@echo ----------------------------------------------------------------------------
	@for x in $(shell ls -1 cmake-build-release/benchmarks/bench_reduce_* | xargs -n 1 basename | sort ) ; do numactl -N 0 cmake-build-release/benchmarks/$$x --benchmark_out_format=json --benchmark_out=bench_results/$$x.json; done


bench_scan:
	@echo ----------------------------------------------------------------------------
	@echo -e "Bench\tRows\tColumns            \tIters\tTime\tCPU\tUnit"
	@echo ----------------------------------------------------------------------------
	@for x in $(shell ls -1 cmake-build-release/benchmarks/bench_scan_* | xargs -n 1 basename | sort ) ; do numactl -N 0 cmake-build-release/benchmarks/$$x --benchmark_out_format=json --benchmark_out=bench_results/$$x.json; done


bench: clean_bench bench_dir release bench_spmv bench_reduce bench_scan

