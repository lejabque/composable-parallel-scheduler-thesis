 make -C cmake-build-release -j$(shell nproc);

rm -rf results;
mkdir results;
for spin_count in 100 1000 10000 100000 1000000 10000000 100000000; do
    ./cmake-build-release/tbb_overhead_task_size $spin_count > results/tbb_overhead_task_size_${spin_count}.json;
done
