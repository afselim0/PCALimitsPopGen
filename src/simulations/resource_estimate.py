"""
Estimate SLURM time/memory for a config by actually running it, tiny and unparallelized.

Replaces hand-maintained cluster.mem/cluster.time in a YAML: those are easy to get wrong
and easy to leave stale after a config changes. Instead, probe_cost() simulates a handful
of real rows (the cheapest and most expensive by the swept parameter, per outdir_suffix
group if the kind has one) at a tiny L and n_workers=1, timing each and measuring this
process's peak RSS. That gives:

    seconds/SNP, worst case across the probed rows -> scaled to the config's real L for
    a single-threaded time estimate, then divided by the planned n_workers (with margin)
  memory: the probe's own peak RSS (fixed per-process overhead, e.g. msprime/tree
    machinery) plus the analytically-known size of the real N x L boolean array
    (simulate_haplotypes' peak transient is a small multiple of N*L bytes, dominated by
    the final concatenated array across workers -- see utils.simulate_haplotypes), with
    headroom on top.

A tiny-L probe cannot see the N*L array's real memory footprint (L is deliberately small),
which is why that term is added analytically rather than measured -- the probe only
measures the *other* memory a real run would use (baseline interpreter/msprime/demography
overhead), which a probe at any L is a reasonable proxy for.
"""
import os
import resource
import sys
import time
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/simulations/
from demographies import get_kind

PROBE_L = 100
MEM_CEILING_GB = 16.0
MEM_HEADROOM = 2.0          # multiply the estimated peak by this before applying the ceiling
TIME_HEADROOM = 2.0          # multiply the estimated worst-case time by this
DEFAULT_CPUS = 8


def _bytes_to_gb(b):
    return b / 1e9


def _probe_rows(table):
    """The cheapest and most expensive row to probe: first and last per outdir_suffix
    group (stepping-stone kinds), or just first/last of the whole table otherwise. Rows
    are already in swept-parameter order (ascending), since every kind builds its table
    via axis_values/sweep_plan in that order."""
    if "outdir_suffix" in table.columns:
        rows = []
        for _, g in table.groupby("outdir_suffix", sort=False):
            rows += [g.iloc[0], g.iloc[-1]]
        return rows
    return [table.iloc[0], table.iloc[-1]]


def _run_probe(kind_module, row, cfg, probe_dir, queue):
    """Runs in a subprocess: time + peak RSS of one simulate_one call at PROBE_L,
    n_workers=1. Peak RSS is read from getrusage after the call, in this child process
    only, so it reflects just this simulation's cost, not the caller's.

    Every kind's simulate_one reads L from row.L except african_split, which reads
    cfg["fixed"]["L"] directly -- both are overridden so the probe is tiny regardless of
    which one a given kind actually consults.
    """
    probe_row = row.copy()
    probe_row["L"] = PROBE_L
    probe_cfg = {**cfg, "fixed": {**cfg.get("fixed", {}), "L": PROBE_L}}

    # simulate_one doesn't create its own output directory -- normally sweep.run_table /
    # run_one_row do that before calling it, but the probe calls it directly.
    if "outdir_suffix" in row.index:
        os.makedirs(os.path.join(probe_dir, row.outdir_suffix), exist_ok=True)

    t0 = time.time()
    kind_module.simulate_one(probe_row, probe_cfg, probe_dir, n_workers=1)
    elapsed = time.time() - t0

    # ru_maxrss is KB on Linux (where SLURM jobs actually run), bytes on macOS.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = raw if sys.platform == "darwin" else raw * 1024
    queue.put((elapsed, _bytes_to_gb(peak_rss_bytes)))


def probe_cost(cfg, probe_dir, verbose=True):
    """
    Probe cfg's cheapest/most-expensive rows at PROBE_L, n_workers=1, in subprocesses (so
    each probe's peak RSS is isolated and the parent's own memory doesn't pollute it).

    Returns (seconds_per_snp_worst, peak_rss_gb_worst, N_max): N_max is the largest total
    sample size across all probed rows, for the analytic N*L memory term.
    """
    kind_module = get_kind(cfg["kind"])
    table = kind_module.build_parameter_table(cfg)
    rows = _probe_rows(table)
    os.makedirs(probe_dir, exist_ok=True)

    worst_rate, worst_rss, N_max = 0.0, 0.0, 0
    for row in rows:
        q = Queue()
        p = Process(target=_run_probe, args=(kind_module, row, cfg, probe_dir, q))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"probe for simID={row.simID} failed (exit {p.exitcode}); "
                               f"run it directly to see the traceback")
        elapsed, peak_rss_gb = q.get()
        rate = elapsed / PROBE_L
        worst_rate = max(worst_rate, rate)
        worst_rss = max(worst_rss, peak_rss_gb)
        N_max = max(N_max, int(row.N))
        if verbose:
            print(f"  probe simID={row.simID}: {elapsed:.2f}s for {PROBE_L} SNPs "
                  f"({rate*1000:.2f} ms/SNP), peak RSS {peak_rss_gb:.2f} GB, N={int(row.N)}")

    return worst_rate, worst_rss, N_max


def estimate(cfg, probe_dir, n_workers=DEFAULT_CPUS, verbose=True):
    """
    Time/memory estimate for cfg's real L, from a cheap tiny-L probe.

    Returns dict(time_seconds, mem_gb, n_workers) -- n_workers may be reduced from the
    requested value if the memory estimate at that many workers would exceed the ceiling
    (more workers means more concurrent N*chunk_L worker arrays before the final concat).
    """
    # sweep.L_values is what simulate_one actually uses (via row.L) for every kind except
    # african_split, which reads fixed.L directly -- prefer L_values when present so this
    # can't silently disagree with what a real run would simulate.
    L_values = cfg.get("sweep", {}).get("L_values")
    if L_values:
        if len(L_values) != 1:
            raise ValueError("resource estimation needs a single L; got L_values="
                             f"{L_values} -- estimate per L value separately")
        L = L_values[0]
    else:
        L = cfg["fixed"]["L"]

    rate, probe_rss_gb, N_max = probe_cost(cfg, probe_dir, verbose=verbose)

    # analytic N*L bool array term: the final concatenated array (utils.simulate_haplotypes)
    # plus each worker's own chunk held concurrently -- roughly (1 + 1/n_workers) * N*L bytes
    array_gb = _bytes_to_gb(N_max * L)

    def total_mem_gb(workers):
        return (probe_rss_gb + array_gb * (1.0 + 1.0 / workers)) * MEM_HEADROOM

    workers = n_workers
    while workers > 1 and total_mem_gb(workers) > MEM_CEILING_GB:
        workers -= 1
    mem_gb = total_mem_gb(workers)
    if mem_gb > MEM_CEILING_GB and verbose:
        print(f"  warning: even at n_workers=1, estimated memory {mem_gb:.1f} GB exceeds "
              f"the {MEM_CEILING_GB:.0f} GB ceiling -- this config may not fit a single job")

    single_threaded_seconds = rate * L * TIME_HEADROOM
    time_seconds = single_threaded_seconds / workers * 1.3  # parallel speedup rarely linear

    if verbose:
        print(f"  -> estimate: n_workers={workers}, mem={mem_gb:.1f} GB, "
              f"time={time_seconds/60:.0f} min (array term {array_gb:.2f} GB, "
              f"probe overhead {probe_rss_gb:.2f} GB)")

    return dict(time_seconds=time_seconds, mem_gb=mem_gb, n_workers=workers)


def format_mem(gb):
    return f"{max(1, int(round(gb)))}G"


def format_time(seconds):
    seconds = max(60, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
