UNAME_S := $(shell uname -s)

NUMACTL_BIND :=
ifeq ($(UNAME_S),Linux)
	NUMACTL_BIND = numactl -N 0
endif

OMP_FLAGS := OMP_MAX_ACTIVE_LEVELS=8 OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite KMP_AFFINITY="granularity=core,compact" LIBOMP_NUM_HIDDEN_HELPER_THREADS=0

release:
	USE_LB4OMP=$(LB4OMP) cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -C cmake-build-release -j$(shell nproc)

debug:
	cmake -B cmake-build-debug -S . -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON && make -C cmake-build-debug -j$(shell nproc)

clean:
	rm -rf cmake-build-* benchmarks/cmake-build-* scheduling_dist/cmake-build-*

clean_bench:
	rm -rf raw_results/*

bench_dir:
	mkdir -p raw_results

bench_spmv:
	./run_bench.sh spmv

bench_spin:
	./run_bench.sh spin

bench_reduce:
	./run_bench.sh reduce

bench_scan:
	./run_bench.sh scan

bench_mmul:
	./run_bench.sh mmul

bench_mtranspose:
	./run_bench.sh mtranspose

run_scheduling_dist:
	@mkdir -p raw_results/scheduling_dist
	@for x in $(shell ls -1 cmake-build-release/scheduling_dist/scheduling_dist_* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; $(OMP_FLAGS) cmake-build-release/scheduling_dist/$$x > raw_results/scheduling_dist/$$x.json; done

run_trace_spin:
	@mkdir -p raw_results/trace_spin
	@for x in $(shell ls -1 cmake-build-release/trace_spin/trace_spin_* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; $(OMP_FLAGS) cmake-build-release/trace_spin/$$x > raw_results/trace_spin/$$x.json; done

run_timespan_tuner:
	@for x in $(shell ls -1 cmake-build-release/timespan_tuner/timespan_tuner_* | xargs -n 1 basename | sort ) ; do $(OMP_FLAGS) cmake-build-release/timespan_tuner/$$x; done

bench_tests:
	@set -e; for x in $(shell ls -1 cmake-build-debug/benchmarks/tests/*tests* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; $(OMP_FLAGS) cmake-build-debug/benchmarks/tests/$$x; done

lib_tests:
	@set -e; for x in $(shell ls -1 cmake-build-debug/include/tests/*tests* | xargs -n 1 basename | sort ) ; do echo "Running $$x"; $(OMP_FLAGS) cmake-build-debug/include/tests/$$x; done

tests: debug bench_tests lib_tests

install_lb4omp:
	@cp -n ~/miniconda3/envs/benchmarks/lib/libomp.so ~/miniconda3/envs/benchmarks/lib/libomp-backup.so
	@cd ~/diploma/LB4OMP && rm -rf build && mkdir build && cd build \
		&& cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLIBOMP_HAVE___RDTSC=ON -DLIBOMP_HAVE_X86INTRIN_H=ON .. \
		&& make && cp runtime/src/libomp.so ~/miniconda3/envs/benchmarks/lib/libomp.so

remove_lb4omp:
	@cp ~/miniconda3/envs/benchmarks/lib/libomp-backup.so ~/miniconda3/envs/benchmarks/lib/libomp.so

run_benchmarks: clean_bench bench_dir bench_spmv bench_spin bench_reduce bench_scan bench_mmul bench_mtranspose

bench: LB4OMP=0
bench: clean release run_benchmarks

bench_lb4omp: LB4OMP=1
bench_lb4omp: clean install_lb4omp release run_benchmarks remove_lb4omp

