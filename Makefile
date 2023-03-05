UNAME_S := $(shell uname -s)

NUMACTL_BIND :=
ifeq ($(UNAME_S),Linux)
	NUMACTL_BIND = numactl -N 0
endif

OMP_FLAGS := OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite KMP_AFFINITY="granularity=core,compact" LIBOMP_NUM_HIDDEN_HELPER_THREADS=0

release:
	cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -C cmake-build-release -j$(shell nproc)

debug:
	cmake -B cmake-build-debug -S . -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON && make -C cmake-build-debug -j$(shell nproc)

clean:
	rm -rf cmake-build-* benchmarks/cmake-build-* scheduling_dist/cmake-build-*

clean_bench:
	rm -rf raw_results/*

bench_dir:
	mkdir -p raw_results

bench_spmv:
	@mkdir -p raw_results/spmv
	@for x in $(shell ls -1 cmake-build-release/benchmarks/bench_spmv* | xargs -n 1 basename | sort ) ; do $(OMP_FLAGS) cmake-build-release/benchmarks/$$x --benchmark_out_format=json --benchmark_out=raw_results/spmv/$$x.json; done

bench_spin:
	@mkdir -p raw_results/spin
	@for x in $(shell ls -1 cmake-build-release/benchmarks/bench_spin* | xargs -n 1 basename | sort ) ; do $(OMP_FLAGS) cmake-build-release/benchmarks/$$x --benchmark_out_format=json --benchmark_out=raw_results/spin/$$x.json; done

bench_reduce:
	@mkdir -p raw_results/reduce
	@for x in $(shell ls -1 cmake-build-release/benchmarks/bench_reduce_* | xargs -n 1 basename | sort ) ; do $(OMP_FLAGS) cmake-build-release/benchmarks/$$x --benchmark_out_format=json --benchmark_out=raw_results/reduce/$$x.json; done

bench_scan:
	@mkdir -p raw_results/scan
	@for x in $(shell ls -1 cmake-build-release/benchmarks/bench_scan_* | xargs -n 1 basename | sort ) ; do $(OMP_FLAGS) cmake-build-release/benchmarks/$$x --benchmark_out_format=json --benchmark_out=raw_results/scan/$$x.json; done

run_scheduling_dist:
	@mkdir -p raw_results/scheduling_dist
	@for x in $(shell ls -1 cmake-build-release/scheduling_dist/scheduling_dist_* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; $(OMP_FLAGS) cmake-build-release/scheduling_dist/$$x > raw_results/scheduling_dist/$$x.json; done

bench: clean_bench bench_dir clean release bench_spmv bench_spin bench_reduce bench_scan run_scheduling_dist

bench_tests:
	@for x in $(shell ls -1 cmake-build-debug/benchmarks/tests/*tests* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; $(OMP_FLAGS) cmake-build-debug/benchmarks/tests/$$x; done

tests: debug bench_tests
