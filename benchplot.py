import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import os


def split_bench_name(s):
    s = s.split("_")
    prefix = []
    for part in s:
        if part.islower():
            prefix.append(part)
        else:
            break
    return "_".join(prefix), "_".join(s[len(prefix):])


# return new plot
def plot_benchmark(benchmarks, title):
    benchmarks = sorted(benchmarks.items(), key=lambda x: x[0])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    min_time = sorted(benchmarks, key=lambda x: x[1])[0][1]
    ax1.set_title("Normalized perfomance of " + title)
    print([(name, min_time / time) for name, time in benchmarks])
    ax1 = ax1.barh(*zip(*[(name, min_time / time) for name, time in benchmarks]))

    ax2.set_title("Time of " + title + ", us")
    ax2 = ax2.barh(*zip(*benchmarks))
    return fig


def parse_benchmarks(folder_name):
    benchmarks = {}
    scheduling_bench = {}
    for bench_file in os.listdir(folder_name):
        if bench_file.endswith('.json'):
            with open(os.path.join(folder_name, bench_file)) as f:
                bench = json.load(f)
                name = bench_file.split(".")[0]
                # TODO: take not only last bench
                if "benchmarks" in bench:
                    res = bench["benchmarks"][-1]
                    if "real_time" in res:
                        benchmarks[name] = res["real_time"]
                    elif "manual_time" in res:
                        benchmarks[name] = res["manual_time"]
                    elif "cpu_time" in res:
                        benchmarks[name] = res["cpu_time"]
                elif name.startswith("bench_scheduling"):
                    scheduling_bench[name] = bench
    benchmarks_by_type = {}

    for name, bench in benchmarks.items():
        bench_type, bench_mode = split_bench_name(name)
        if bench_type not in benchmarks_by_type:
            benchmarks_by_type[bench_type] = {}
        benchmarks_by_type[bench_type][bench_mode] = bench
    return benchmarks_by_type, scheduling_bench


def plot_scheduling_benchmarks(scheduling_times):
    # x for thread_idx, y for time
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for bench_type, times in sorted(scheduling_times.items(), key=lambda x: max(x[1])):
        print(bench_type, times)
        ax.plot(range(len(times)), times, label=bench_type)
    ax.legend()
    return fig


if __name__ == "__main__":
    # fetch folder from args or use current folder
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "bench_results"
    benchmarks_by_type, scheduling_bench = parse_benchmarks(folder_name)
    res_path = os.path.join(folder_name, "images")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # plot main benchmarks
    for bench_type, bench in benchmarks_by_type.items():
        fig = plot_benchmark(bench, bench_type)
        fig.savefig(os.path.join(res_path, bench_type + '.png'))

    # plot scheduling benchmarks
    scheduling_times = {}
    for bench_type, res in scheduling_bench.items():
        # print thread num, used threads
        print(bench_type, res["thread_num"], res["used_threads"])
        if res["thread_num"] != res["used_threads"]:
            print(f"Maybe something went wrong with {bench_type}: not all threads were used")
        scheduling_times[bench_type] = res["start_times"]

    fig = plot_scheduling_benchmarks(scheduling_times)
    fig.savefig(os.path.join(res_path, 'scheduling_time.png'))

    # plot average scheduling time for all benchs
    avg_times = {}
    for bench_type, times in scheduling_times.items():
        avg_times[bench_type.split("_", 2)[-1]] = sum(times) / len(times)
    fig = plot_benchmark(avg_times, "average scheduling time")
    fig.savefig(os.path.join(res_path, 'avg_scheduling_time.png'))
