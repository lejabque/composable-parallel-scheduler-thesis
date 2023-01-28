# read all files tbb_overhead_task_size_*.json

import json
import matplotlib.pyplot as plt
import os

bench_by_spin_count = {} # spin_count -> iter -> thread_id -> time

for file in os.listdir("results"):
    if file.startswith("tbb_overhead_task_size_") and file.endswith(".json"):
        with open(os.path.join("results", file)) as f:
            bench = json.load(f)
            spin_count = int(file.split(".")[0].split("_")[4])
            bench_by_spin_count[spin_count] = bench


def calc_averages(main_thread):
    averages = {}
    print("Average times:")
    for spin_count, bench in sorted(bench_by_spin_count.items()):
        avg = 0
        for iter in bench.values():
            for thread_id, time in iter.items():
                if main_thread and thread_id == "0":
                    avg += time
                elif not main_thread and thread_id != "0":
                    avg += time
        avg /= len(bench)
        averages[spin_count] = avg
        print(avg)
    return averages


plt.xscale("log")
plt.plot(*zip(*sorted(calc_averages(main_thread=True).items())), label="Main thread")
plt.plot(*zip(*sorted(calc_averages(main_thread=False).items())), label="Other threads")
plt.legend()
plt.xlabel("Spin count")
plt.ylabel("Average clock ticks")

plt.savefig("tbb_overhead_task_size.png")

