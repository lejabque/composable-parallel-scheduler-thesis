.PHONY: all
all: release test bench_noop

release_benchmarks:
	cd benchmarks && cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -C cmake-build-release -j$(shell nproc)

release_scheduling:
	cd scheduling_time && cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -C cmake-build-release -j$(shell nproc)

release_scheduling_dist:
	cd scheduling_dist && cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -C cmake-build-release -j$(shell nproc)

debug_benchmarks:
	cd benchmarks && cmake -B cmake-build-debug -S . -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug && make -C cmake-build-debug -j$(shell nproc)

clean:
	rm -rf cmake-build-* benchmarks/cmake-build-* scheduling_time/cmake-build-*

clean_bench:
	rm -rf bench_results/*

bench_dir:
	mkdir -p bench_results

# todo: separate balanced/unbalanced
bench_spmv:
	@echo ----------------------------------------------------------------------------
	@echo "Bench\tRows\tColumns            \tIters\tTime\tCPU\tUnit"
	@echo ----------------------------------------------------------------------------
	@for x in $(shell ls -1 benchmarks/cmake-build-release/bench_spmv_* | xargs -n 1 basename | sort ) ; do numactl -N 0 benchmarks/cmake-build-release/$$x --benchmark_out_format=json --benchmark_out=bench_results/$$x.json; done

bench_reduce:
	@echo ----------------------------------------------------------------------------
	@echo "Bench\tRows\tColumns            \tIters\tTime\tCPU\tUnit"
	@echo ----------------------------------------------------------------------------
	@for x in $(shell ls -1 benchmarks/cmake-build-release/bench_reduce_* | xargs -n 1 basename | sort ) ; do numactl -N 0 benchmarks/cmake-build-release/$$x --benchmark_out_format=json --benchmark_out=bench_results/$$x.json; done

bench_scan:
	@echo ----------------------------------------------------------------------------
	@echo "Bench\tRows\tColumns            \tIters\tTime\tCPU\tUnit"
	@echo ----------------------------------------------------------------------------
	@for x in $(shell ls -1 benchmarks/cmake-build-release/bench_scan_* | xargs -n 1 basename | sort ) ; do numactl -N 0 benchmarks/cmake-build-release/$$x --benchmark_out_format=json --benchmark_out=bench_results/$$x.json; done

bench_scheduling:
	@for x in $(shell ls -1 scheduling_time/cmake-build-release/bench_scheduling_* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; scheduling_time/cmake-build-release/$$x > bench_results/$$x.json; done

run_scheduling_dist:
	@for x in $(shell ls -1 scheduling_dist/cmake-build-release/scheduling_dist_* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; scheduling_dist/cmake-build-release/$$x > bench_results/$$x.json; done

bench: clean_bench bench_dir release_benchmarks release_scheduling bench_spmv bench_reduce bench_scan bench_scheduling

bench_tests:
	@for x in $(shell ls -1 benchmarks/cmake-build-debug/tests/*tests* | xargs -n 1 basename | sort ) ; do numactl -N 0 benchmarks/cmake-build-release/tests/$$x; done

tests: debug_benchmarks bench_tests
