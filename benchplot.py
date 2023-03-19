import json
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.pyplot import cm
from functools import partial
import itertools
import time
import math
import numpy as np
import re
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


def save_figure(path, fig, name):
    fig.savefig(os.path.join(path, name + ".jpeg"), bbox_inches='tight', format='jpeg')
    fig.savefig(os.path.join(path, name + ".svg"), bbox_inches='tight', format='svg', dpi=1200)


# returns new plot
def plot_benchmark(benchmarks, title, verbose):
    params_count = len(benchmarks)
    fig, axis = plt.subplots(params_count, 2 if verbose else 1, figsize=(36 if verbose else 16, params_count * 6), squeeze=False)
    iter = 0
    for params, bench_results in benchmarks.items():
        bench_results = sorted(bench_results.items(), key=lambda x: x[1])
        min_time = sorted(bench_results, key=lambda x: x[1])[0][1]
        axis[iter][0].barh(*zip(*[(name, min_time / time) for name, time in bench_results]))
        if verbose:
            params_str = ""
            if params != "":
                params_str = " with params " + params
            axis[iter][1].barh(*zip(*bench_results))
            axis[iter][0].set_xlabel(title + params_str + ", normalized (higher is better)", fontsize=14)
            axis[iter][1].set_xlabel(title + params_str + ", absolute time (lower is better), us", fontsize=14)
        iter += 1

    for axs in axis:
        for ax in axs:
            if verbose:
                ax.tick_params(axis='both', which='major', labelsize=14)
                fig.tight_layout()
            else:
                ax.tick_params(axis='both', which='major', labelsize=20)
    return fig


def parse_benchmarks(folder_name):
    benchmarks_by_type = {}
    for bench_file in os.listdir(folder_name):
        if not bench_file.endswith('.json'):
            continue
        with open(os.path.join(folder_name, bench_file)) as f:
            try:
                bench = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print("Error while parsing", bench_file, e, ", skipping")
                continue
            name = bench_file.split(".")[0]
            bench_type, bench_mode = split_bench_name(name)
            if "benchmarks" in bench:
                # TODO: take not only last bench
                for res in bench["benchmarks"]:
                    params = ";".join(x.group()[1:] for x in re.finditer(r'\/\w+:\d+', res["name"]))
                    benchmarks_by_type.setdefault(bench_type, {}).setdefault(params, {})[bench_mode] = res["real_time"]
            else:
                benchmarks_by_type.setdefault(bench_type, {}).setdefault("", {})[bench_mode] = bench
    return benchmarks_by_type


# TODO: refactor results and replace dicts with classes
def plot_scheduling_benchmarks(scheduling_times, verbose):
    # x for thread_idx, y for time
    min_times = [(bench_type, [[min([t["trace"]["execution_start"] for t in tasks]) for tasks in iter["tasks"].values()] for iter in results]) for bench_type, results in scheduling_times.items()]
    runtimes = list(set(name.split("_")[0] for (name, _) in min_times))
    runtimes.append("all")
    plots = {runtime: plt.subplots(figsize=(18, 6)) for runtime in runtimes}
    styles = itertools.cycle(["-", "--", "-.", ":"])
    colors = itertools.cycle('bgrcmk')
    style = next(styles)
    color = next(colors)
    max_y = []
    for bench_type, times in reversed(sorted(min_times,
                                             key=lambda x: np.max(np.mean(np.asarray(x[1]), axis=0)))):
        # todo: better way to visualize?
        # min time per thread for each idx in range of thread count
        times = np.asarray(times)
        # plot distribution of all times for iterations as scatter around time
        means = np.min(times, axis=0)
        means = np.sort(means, axis=0)
        if color == "k":
            style = next(styles)
        color = next(colors)
        _, ax = plots[bench_type.split("_")[0]]
        ax.plot(range(len(means)), means, label=bench_type, linestyle=style, color=color)
        plots["all"][1].plot(range(len(means)), means, label=bench_type, linestyle=style, color=color)
        max_y.append(np.max(means))

    # ignore most slower method
    if len(max_y) > 1:
        top_limit = sorted(max_y)[-2]
        plots["all"][1].set_ylim(top=top_limit, bottom=0)

    for runtime in runtimes:
        _, ax = plots[runtime]
        ax.ticklabel_format(style='plain')
        if verbose:
            ax.set_title("Scheduling time, " + runtime)
        # use yticks bigger and xticks fontsize
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel("Index of thread (sorted by time of first task)", fontsize=20)
        ax.set_ylabel("Cycles", fontsize=20)
        ax.legend(fontsize=14)
        ylimit = ax.get_ylim()
        # make ~10 ticks on y axis but on round numbers (e.x. 500, 1000) only:
        step = (ylimit[1] - ylimit[0]) / 10
        # round step to round number (e.g. if it's 700, round up to 1000)
        step = 10 ** (len(str(int(step))) - 1) * (int(step) // 10 ** (len(str(int(step))) - 1) + 1)
        # round by step to smallest:
        bottom = math.floor(ylimit[0] / step) * step
        bottom = max(bottom, 0)
        ax.set_yticks(np.arange(bottom, ylimit[1], step))
    return plots


def plot_scheduling_dist(scheduling_dist, verbose):
    # plot heatmap for map of thread_idx -> tasks list
    row_count = len(scheduling_dist)
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle("Scheduling distribution, results for each iteration")
    fig.tight_layout()
    gs = GridSpec(row_count, 3, figure=fig, width_ratios=[1, 1, 4])

    for iter in range(row_count):
        ax = fig.add_subplot(gs[iter, 0])
        if iter == 0:
            ax.set_title("Distribution of tasks to threads")
        # ax.get_figure().tight_layout()

        thread_count = len(scheduling_dist[iter]["tasks"])
        task_count = max(max(t["index"] for t in tasks) for tasks in scheduling_dist[iter]["tasks"].values()) + 1
        task_height = int(task_count / thread_count)
        data = np.ones((thread_count * task_height, task_count))
        for thread_id, tasks in sorted(scheduling_dist[iter]["tasks"].items(), key=lambda x: x[0]):
            for t in tasks:
                idx = thread_count - 1 if thread_id == "-1" else int(thread_id)
                # data[idx, t["index"]] = 0
                # fill rectangle by zeros
                data[idx * task_height: (idx + 1) * task_height, t["index"]] = 0
        ax.set_ylabel("Thread")
        ax.set_xlabel("Task")
        ax.xaxis.set_label_position('top')
        ax.imshow(data, cmap='gray', origin='lower')

        def format_task(x, _):
            return str(int(x / task_height))

        ax.yaxis.set_major_formatter(format_task)

    # plot heatmap thread id, cpu id
    for iter in range(row_count):
        ax = fig.add_subplot(gs[iter, 1])
        if iter == 0:
            ax.set_title("Distribution of threads to cpus")
        ax.set_ylabel("Thread")
        ax.set_xlabel("Cpu")
        ax.xaxis.set_label_position('top')
        # ax.get_figure().tight_layout()

        thread_count = len(scheduling_dist[iter]["tasks"].keys())
        cpu_count = len(set(t["cpu"] for tasks in scheduling_dist[iter]["tasks"].values() for t in tasks))
        cpus = {}  # mapping cpu_id -> idx
        threads = {}  # mapping thread_id -> idx
        data = np.ones((thread_count, cpu_count))
        for thread_id, tasks in scheduling_dist[iter]["tasks"].items():
            for t in tasks:
                if thread_id not in threads:
                    threads[thread_id] = len(threads)
                cpu = t["cpu"]
                if cpu not in cpus:
                    cpus[cpu] = len(cpus)
                data[threads[thread_id], cpus[cpu]] = 0
        ax.imshow(data, cmap='gray', origin='lower')

    for iter in range(row_count):
        ax = fig.add_subplot(gs[iter, 2])
        ax.set_ylabel("Clock ticks")
        ax.set_xlabel("Thread")
        # ax.get_figure().tight_layout()
        if iter == 0:
            ax.set_title("Time since start of first executed task for each thread (sorted by time)")

        thread_count = len(scheduling_dist[iter]["tasks"])
        times = [min(t["trace"]["execution_start"] for t in tasks) for tasks in scheduling_dist[iter]["tasks"].values()]
        times = np.sort(np.asarray(times))
        ax.plot(range(len(times)), times, label="Start time")
        times_end = [min(scheduling_dist[iter]["end"] - t["trace"]["execution_end"] for t in tasks) for tasks in scheduling_dist[iter]["tasks"].values()]
        time_end = min(times_end)
        # times_end = np.sort(np.asarray(times_end))
        ax.plot(range(len(times)), [time_end] * len(times_end), label="End time")
        ax.legend()

    return fig


def plot_scheduling_dist_item(item, res_path, verbose):
    bench_mode, res = item
    time_before = time.time()
    fig = plot_scheduling_dist(res["results"], verbose)
    save_figure(res_path, fig, "scheduling_dist_" + bench_mode)
    plt.close()
    print(f"Plotting {bench_mode} took {time.time() - time_before}s")
    for iter in res["results"]:
        for thread_id, tasks in iter["tasks"].items():
            unique_cpus = set(t["cpu"] for t in tasks)
            if len(unique_cpus) > 1:
                print(f"{bench_mode}: thread {thread_id} has tasks executed on differenet cpus: {unique_cpus}")


if __name__ == "__main__":
    # fetch folder from args or use current folder
    folder_name = "raw_results"
    res_path = "bench_results"
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    verbose = not (len(sys.argv) > 1 and sys.argv[1] == "compact")

    subdirs = [d for d in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, d))]
    # plot main benchmarks
    for subdir in subdirs:
        benchmarks = parse_benchmarks(os.path.join(folder_name, subdir))
        if subdir == "scheduling_dist":
            scheduling_times_by_suffix = {}
            for bench_mode, res in benchmarks["scheduling_dist"][""].items():
                bench_mode, measure_mode = bench_mode.rsplit("_", 1)
                if measure_mode not in scheduling_times_by_suffix:
                    scheduling_times_by_suffix[measure_mode] = {}
                scheduling_times_by_suffix[measure_mode][bench_mode] = res["results"]

            for measure_mode, times in scheduling_times_by_suffix.items():
                print("Processing scheduling time", measure_mode)
                plots = plot_scheduling_benchmarks(times, verbose)
                for bench_type, plot in plots.items():
                    fig, ax = plot
                    current_res_path = os.path.join(res_path, "scheduling_time")
                    if not os.path.exists(current_res_path):
                        os.makedirs(current_res_path)
                    save_figure(current_res_path, fig, "scheduling_time_" + bench_type + "_" + measure_mode)
                    plt.close()

            # plot scheduling dist

            current_res_path = os.path.join(res_path, "scheduling_dist")
            if not os.path.exists(current_res_path):
                os.makedirs(current_res_path)
            with Pool() as pool:
                # call the function for each item in parallel
                pool.map(partial(plot_scheduling_dist_item, res_path=current_res_path, verbose=verbose), benchmarks["scheduling_dist"][""].items())

        else:
            current_res_path = os.path.join(res_path, subdir)
            if not os.path.exists(current_res_path):
                os.makedirs(current_res_path)
            for bench_type, bench in benchmarks.items():
                print("Processing", bench_type)
                fig = plot_benchmark(bench, bench_type, verbose)
                save_figure(current_res_path, fig, bench_type)
                plt.close()
