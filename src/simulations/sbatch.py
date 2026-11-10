"""
Render (and optionally submit) a SLURM array job from a simulation YAML config.

The --array bound is sized automatically from build_parameter_table's row count (cheap --
no simulation cost). mem/time/cpus_per_task are sized automatically too, by actually
running a tiny probe of the config (see resource_estimate.py) rather than relying on
hand-maintained cluster: fields that are easy to leave stale after a config changes. A
config's cluster: block can still set job_name, and can override mem/time/cpus_per_task
explicitly to skip probing (e.g. reusing numbers from a previous probe).
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/simulations/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from slurm_config import slurm_settings
from demographies import get_kind
from resource_estimate import DEFAULT_CPUS, estimate, format_mem, format_time

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(_HERE, "sbatch.template")


def _default_job_name(cfg):
    name = cfg.get("name")
    return f"{cfg['kind']}_{name}" if name and name != "default" else cfg["kind"]


def render_sbatch(cfg, config_path, outdir, sbatch_outdir=None, verbose=True):
    """Render a .sbatch file for cfg, sized from its own parameter table and a resource
    probe. Returns the path written, under <repo_root>/logs/ by default."""
    kind_module = get_kind(cfg["kind"])
    n_rows = len(kind_module.build_parameter_table(cfg))

    cluster = cfg.get("cluster", {})
    job_name = cluster.get("job_name", _default_job_name(cfg))

    if "mem" in cluster and "time" in cluster and "cpus_per_task" in cluster:
        mem, time_str = cluster["mem"], cluster["time"]
        cpus = cluster["cpus_per_task"]
        if verbose:
            print(f"  using cluster: block as-is (mem={mem}, time={time_str}, "
                  f"cpus_per_task={cpus}) -- skipping the resource probe")
    else:
        if verbose:
            print(f"  probing resource needs for kind={cfg['kind']} "
                  f"(no complete cluster: block given)...")
        probe_dir = tempfile.mkdtemp(prefix="pca_rmt_probe_")
        try:
            n_workers = cluster.get("cpus_per_task", DEFAULT_CPUS)
            est = estimate(cfg, probe_dir, n_workers=n_workers, verbose=verbose)
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
        mem = cluster.get("mem", format_mem(est["mem_gb"]))
        time_str = cluster.get("time", format_time(est["time_seconds"]))
        cpus = cluster.get("cpus_per_task", est["n_workers"])

    with open(TEMPLATE_PATH) as f:
        template = f.read()
    rendered = template.format(
        job_name=job_name, n_rows_minus_1=n_rows - 1,
        cpus_per_task=cpus, mem=mem, time=time_str,
        config_path=os.path.abspath(config_path), outdir=outdir, **slurm_settings())

    sbatch_outdir = sbatch_outdir or os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "logs")
    os.makedirs(sbatch_outdir, exist_ok=True)
    path = os.path.join(sbatch_outdir, f"{job_name}.sbatch")
    with open(path, "w") as f:
        f.write(rendered)
    return path


def submit(sbatch_path):
    """sbatch <path>, run from the repo root so relative paths in the template resolve."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run(["sbatch", sbatch_path], cwd=repo_root, check=True)
