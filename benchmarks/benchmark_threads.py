"""Benchmark multithreaded runNESTvec_parallel against single-threaded runNESTvec.

Example:
    python benchmarks/benchmark_threads.py --events 200000 --threads 1 2 4 8 --repeats 5

Configurations are interleaved (each repeat runs every configuration once) so
that CPU thermal throttling affects them all equally, and the best time of
each is reported.
"""

import argparse
import os
import platform
import statistics
import time
from importlib.metadata import version

# Stop numpy's OpenBLAS threads spinning and competing with NEST's threads.
# Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import nestpy  # noqa: E402


def parse_args():
    n_cpu = os.cpu_count() or 1
    default_threads = sorted({1, 2, 4, n_cpu // 2, n_cpu} - {0})

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, default=200_000, help="events per run")
    parser.add_argument("--threads", type=int, nargs="+", default=default_threads,
                        help="thread counts to test (0 means all cores)")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per configuration")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--detector", default="LZ_WS2024",
                        help="name of a class in nestpy.detectors")
    parser.add_argument("--interaction", default="NR",
                        help="name of a member of nestpy.interactions")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def make_inputs(detector, n_events, seed):
    """Uniform energies (1-50 keV) at uniform positions inside the detector."""
    rng = np.random.default_rng(seed)
    r = detector.get_radius() * np.sqrt(rng.uniform(0, 1, n_events))
    theta = rng.uniform(0, 2 * np.pi, n_events)
    z = rng.uniform(1, detector.get_TopDrift() - 1, n_events)
    positions = np.column_stack((r * np.cos(theta), r * np.sin(theta), z))
    energies = rng.uniform(1, 50, n_events)
    return energies.tolist(), positions.tolist()


def main():
    args = parse_args()

    detector = getattr(nestpy.detectors, args.detector)()
    interaction = getattr(nestpy.interactions, args.interaction)
    energies, positions = make_inputs(detector, args.events, args.seed)
    common = dict(detector=detector, interaction_type=interaction,
                  energies=energies, positions=positions, seed=args.seed)

    configs = {"serial": lambda: nestpy.array.runNESTvec(**common)}
    for n in args.threads:
        configs[f"{n} threads" if n > 0 else "all threads"] = lambda n=n: nestpy.array.runNESTvec_parallel(
            **common, n_threads=n, chunk_size=args.chunk_size)

    print(f"nestpy {version('nestpy')}, NEST {nestpy.__nest_version__}, "
          f"Python {platform.python_version()}, {os.cpu_count()} logical CPUs")
    print(f"{args.events} {args.interaction} events in {args.detector}, "
          f"chunk_size={args.chunk_size}, best of {args.repeats}\n")

    # Warm up (first-call costs, one-off warnings) on a small slice
    warm = dict(common, energies=energies[:1000], positions=positions[:1000])
    nestpy.array.runNESTvec(**warm)
    nestpy.array.runNESTvec_parallel(**warm)

    times = {name: [] for name in configs}
    for _ in range(args.repeats):
        for name, run in configs.items():
            start = time.perf_counter()
            run()
            times[name].append(time.perf_counter() - start)

    serial = min(times["serial"])
    print(f"{'config':>12} {'best (s)':>9} {'median (s)':>11} {'events/s':>10} "
          f"{'speedup':>8} {'efficiency':>11}")
    for name, ts in times.items():
        best = min(ts)
        speedup = serial / best
        n = {"serial": 1, "all threads": os.cpu_count()}.get(name) or int(name.split()[0])
        print(f"{name:>12} {best:9.3f} {statistics.median(ts):11.3f} "
              f"{args.events / best:10.0f} {speedup:7.2f}x {speedup / n:10.0%}")


if __name__ == "__main__":
    main()
