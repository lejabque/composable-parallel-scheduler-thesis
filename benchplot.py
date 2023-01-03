import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec


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
    ax1.set_xlabel("Normalized perfomance of " + title)
    print([(name, min_time / time) for name, time in benchmarks])
    ax1 = ax1.barh(*zip(*[(name, min_time / time) for name, time in benchmarks]))

    ax2.set_xlabel("Time of " + title + ", us")
    ax2 = ax2.barh(*zip(*benchmarks))
    return fig


def parse_benchmarks(folder_name):
    benchmarks_by_type = {}
    for bench_file in os.listdir(folder_name):
        if not bench_file.endswith('.json'):
            continue
        with open(os.path.join(folder_name, bench_file)) as f:
            print(bench_file)
            bench = json.load(f)
            name = bench_file.split(".")[0]
            bench_type, bench_mode = split_bench_name(name)
            if bench_type not in benchmarks_by_type:
                benchmarks_by_type[bench_type] = {}
            if "benchmarks" in bench:
                # TODO: take not only last bench
                res = bench["benchmarks"][-1]
                if "real_time" in res:
                    report = res["real_time"]
                elif "manual_time" in res:
                    report = res["manual_time"]
                elif "cpu_time" in res:
                    report = res["cpu_time"]
                benchmarks_by_type[bench_type][bench_mode] = report
            else:
                benchmarks_by_type[bench_type][bench_mode] = bench
    return benchmarks_by_type


def plot_scheduling_benchmarks(scheduling_times):
    # x for thread_idx, y for time
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for bench_type, times in reversed(sorted(scheduling_times.items(),
                                 key=lambda x: np.max(np.mean(np.asarray(x[1]), axis=0)))):
        # mean time per thread for each idx in range of thread count
        times = np.asarray(times)
        means = np.mean(times, axis=0)
        stds = np.std(times, axis=0)
        print(bench_type, means, stds)
        # TODO: plot stds
        ax.plot(range(len(means)), means, label=bench_type)
    ax.set_title("Scheduling time")
    ax.set_xlabel("Index of thread")
    ax.set_ylabel("Clock ticks")
    ax.legend()
    return fig


def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


def plot_scheduling_dist(scheduling_dist):
    # plot heatmap for map of thread_idx -> tasks list
    row_count = len(scheduling_dist)
    fig = plt.figure(figsize=(10, 15))
    fig.tight_layout()
    gs = GridSpec(row_count, 1, figure=fig)
    # gs.update(wspace=1.5, hspace=1)

    format_axes(fig)
    # # axs.set_title("Scheduling distribution")
    # # axs.set_ylabel("Index of thread")
    # # axs.set_xlabel("Index of task")
    for i in range(row_count):
        ax = fig.add_subplot(gs[i, 0])
        thread_count = len(scheduling_dist[i])
        task_count = max(max(tasks) for tasks in scheduling_dist[i].values()) + 1
        data = np.ones((thread_count, task_count))
        total = 0
        for _, tasks in scheduling_dist[i].items():
            for task_idx in tasks:
                data[total, task_idx] = 0
            total += 1
        ax.imshow(data, cmap='gray')
    return fig


if __name__ == "__main__":
    # fetch folder from args or use current folder
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "bench_results"
    benchmarks = parse_benchmarks(folder_name)
    res_path = os.path.join(folder_name, "images")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # plot main benchmarks
    for bench_type, bench in benchmarks.items():
        if not bench_type.startswith("bench") or bench_type.startswith("bench_scheduling"):
            continue
        fig = plot_benchmark(bench, bench_type)
        fig.savefig(os.path.join(res_path, bench_type + '.png'), bbox_inches='tight')

    # plot with and without barrier
    # group scheduling_times by suffix
    scheduling_times_by_suffix = {}
    for bench_mode, res in benchmarks["bench_scheduling"].items():
        bench_mode, measure_mode = bench_mode.rsplit("_", 1)
        if measure_mode not in scheduling_times_by_suffix:
            scheduling_times_by_suffix[measure_mode] = {}
        scheduling_times_by_suffix[measure_mode][bench_mode] = res["results"]

    for measure_mode, times in scheduling_times_by_suffix.items():
        fig = plot_scheduling_benchmarks(times)
        fig.savefig(os.path.join(res_path, "scheduling_time_" + measure_mode + '.png'), bbox_inches='tight')

    # plot average scheduling time for all benchs
    # avg_times = {}
    # for bench_type, times in scheduling_times.items():
    #     avg_times[bench_type.split("_", 2)[-1]] = sum(times) / len(times)
    # fig = plot_benchmark(avg_times, "average scheduling time")
    # fig.savefig(os.path.join(res_path, 'avg_scheduling_time.png'))

    # plot scheduling dist
    for bench_mode, res in benchmarks["scheduling_dist"].items():
        fig = plot_scheduling_dist(res["results"])
        fig.savefig(os.path.join(res_path, "scheduling_dist_" + bench_mode + '.png'), bbox_inches='tight')

