#%%
"""
Simulate binary genotype matrices for two populations separated by a target Fst,
under the Balding-Nichols model.

Every column of the output is polymorphic in the sample: candidate loci are
over-drawn and filtered, so L is the number of *segregating* sites, not the number
of draws attempted (see simulate_genotypes).

For each of L independent loci:
    1. an ancestral allele frequency p is drawn from a density proportional to
       1/x on [p_min, p_max], matching the shape of the neutral SFS
    2. the two populations get independent frequencies
           p'_k ~ Beta(p (1-Fst)/Fst, (1-p)(1-Fst)/Fst),   k = 1, 2
       which has mean p and variance Fst * p (1-p): Fst is exactly the expected
       fraction of variance in allele frequency that lies between populations
    3. each population's n samples are independent Bernoulli(p'_k) draws

Unlike a coalescent simulation this is a pure allele-frequency model -- there is
no tree, no linkage and no shared drift beyond the common ancestral frequency --
so it produces exactly the exchangeable, independent-locus structure the RMT
calculation assumes, at a cost of ~L * N random draws instead of L trees.

The 1/x ancestral density needs a lower cutoff: it is not normalizable at 0, and
1/x itself is the density of the *population* frequency spectrum, which has
infinite mass at arbitrarily rare variants. p_min sets that cutoff and is
inverse-CDF sampled in closed form (p = p_min (p_max/p_min)^U).

Rows are ordered population 1 first, then population 2, so the matrix is

    X[:n]   population 1
    X[n:]   population 2

Example:
    python simulate_generic_Fst.py                       # the 100-value default sweep
    python simulate_generic_Fst.py --num-Fst 20 --n 50    # a quick smaller sweep

Outputs (results/simulations/K2_simulations by default):
    simulation_parameters.csv   one row per simulation
    Fst<value>.npy              the 2n x L genotype matrix, saved as np.bool_,
                                with the Fst in scientific notation to 4 decimal
                                places (e.g. Fst1.0000e-04.npy)
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "simulations", "K2_simulations",
)

DEFAULT_N = 500
DEFAULT_L = int(2e5)
DEFAULT_NUM_FST = 100
DEFAULT_MIN_FST = 1e-4
DEFAULT_MAX_FST = 1e-2

# Cutoffs on the ancestral frequency. The lower one is required (1/x is not
# normalizable at 0); the upper one keeps p strictly inside (0, 1) so the Beta
# shape parameters stay positive.
DEFAULT_P_MIN = 1e-4
DEFAULT_P_MAX = 1.0 - 1e-4

DEFAULT_RANDOM_SEED = 42


def format_Fst(Fst):
    """
    The filename stem for a simulation: the Fst in scientific notation to 4
    decimal places, e.g. 1e-4 -> "Fst1.0000e-04".

    Fixed-width and monotone in Fst over the default sweep, so the files sort in
    Fst order, and 4 decimal places resolve the 100 log-spaced values in
    [1e-4, 1e-2] uniquely (their closest neighbours differ in the 2nd digit).
    """
    return f"Fst{Fst:.4e}"


def sample_ancestral_frequencies(L, p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX,
                                 rng=None):
    """
    Draw L ancestral allele frequencies from a density proportional to 1/p on
    [p_min, p_max].

    Inverse-CDF in closed form: with f(p) ∝ 1/p the CDF is
    log(p/p_min)/log(p_max/p_min), so p = p_min * (p_max/p_min)^U for U uniform
    on [0, 1) -- i.e. the frequencies are log-uniform. Vectorized over all L
    loci at once.
    """
    if not (0.0 < p_min < p_max < 1.0):
        raise ValueError(
            f"need 0 < p_min < p_max < 1, got p_min={p_min:g}, p_max={p_max:g}"
        )
    rng = np.random.default_rng() if rng is None else rng
    return p_min * (p_max / p_min) ** rng.random(L)


def draw_loci(n, L, Fst, p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX, rng=None):
    """
    Draw L candidate loci: the ancestral frequencies, the two per-population
    frequencies, and the 2n x L genotypes. No polymorphism filtering.

    Every step is vectorized across the L loci: the ancestral frequencies are one
    length-L draw, the two Beta draws are length-L each, and the Bernoulli draws
    are a single (2n, L) comparison of uniforms against the per-column frequency
    broadcast down the rows. Nothing loops over loci.
    """
    p_anc = sample_ancestral_frequencies(L, p_min=p_min, p_max=p_max, rng=rng)

    # Balding-Nichols: Beta(p*theta, (1-p)*theta) with theta = (1-Fst)/Fst has
    # mean p and variance Fst*p*(1-p). Small Fst means large theta, i.e. a Beta
    # tightly concentrated on p -- little differentiation.
    theta = (1.0 - Fst) / Fst
    p_pop = rng.beta(p_anc * theta, (1.0 - p_anc) * theta, size=(2, L))

    # One uniform per (sample, locus), compared against that locus's population
    # frequency. rng.random((2n, L)) < freq is equivalent to 2n Bernoulli draws
    # per locus but avoids a per-locus call into the generator. The comparison is
    # already np.bool_, which is the stored dtype -- no cast needed.
    freq = np.repeat(p_pop, n, axis=0)          # (2n, L): rows 0..n-1 are pop 1
    X = rng.random((2 * n, L)) < freq
    return X, p_anc, p_pop


def simulate_genotypes(Fst, n=DEFAULT_N, L=DEFAULT_L, p_min=DEFAULT_P_MIN,
                       p_max=DEFAULT_P_MAX, rng=None, batch_factor=1.6,
                       max_batches=100):
    """
    Simulate a 2n x L binary genotype matrix for two populations at the given Fst,
    with n haploid samples per population, in which *every one of the L columns is
    polymorphic in the sample*.

    A raw Balding-Nichols draw wastes columns: with the ancestral frequency drawn
    from a 1/p density most loci are rare, and at n = 500, p_min = 1e-4 only about
    70-80% of columns segregate in the sample -- the rest are all-ancestral (or,
    rarely, all-derived) and carry no information while still counting toward L.

    So candidates are over-drawn and filtered. The retention rate is not known in
    advance (it falls with Fst and rises with n), so rather than scaling L by a
    fixed guess this draws in batches until L polymorphic columns have accumulated,
    then keeps exactly the first L of them. The first batch is sized by
    batch_factor and subsequent batches from the retention rate actually observed,
    so the common case is a single draw with a small surplus.

    Note this conditions the SFS on ascertainment: the retained loci are a
    frequency-biased subsample of the 1/p prior, depleted of the rarest variants
    (at n = 500 the mean ancestral frequency rises from ~0.109 to ~0.139), which is
    the same ascertainment a real polymorphism panel has. Fst itself is left
    essentially unchanged -- a monomorphic column contributes zero to both the
    Hudson numerator and its denominator, so dropping such columns is close to
    neutral for the ratio -- but the frequency spectrum, and hence the shape of the
    eigenvalue bulk, is not the unconditioned one.

    Returns:
        (X, p_anc, p_pop) -- the boolean 2n x L genotype matrix, the length-L
        ancestral frequencies, and the (2, L) per-population frequencies, all three
        restricted to (and aligned on) the retained polymorphic loci.

    X is np.bool_ (1 byte per entry, same footprint as uint8, half of which would
    be wasted on a 0/1 uint8). Downstream arithmetic must cast it: bool arrays
    sum to integers correctly but negate and subtract under boolean logic, so
    `X.astype(np.float64)` before any centering, as compute_sample_covariance
    already does.
    """
    if not (0.0 < Fst < 1.0):
        raise ValueError(f"Fst must lie in (0, 1), got {Fst:g}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if L < 1:
        raise ValueError(f"L must be at least 1, got {L}")
    if batch_factor < 1.0:
        raise ValueError(f"batch_factor must be at least 1, got {batch_factor:g}")

    rng = np.random.default_rng() if rng is None else rng

    X_parts, p_anc_parts, p_pop_parts = [], [], []
    found = 0
    n_drawn = 0
    # First batch: L * batch_factor candidates. Later batches are sized from the
    # observed retention rate, with a 1.3x safety margin and a floor, so a bad
    # first estimate cannot lead to many tiny follow-up draws.
    batch = int(np.ceil(L * batch_factor))

    for _ in range(max_batches):
        X_b, p_anc_b, p_pop_b = draw_loci(
            n, batch, Fst, p_min=p_min, p_max=p_max, rng=rng)

        counts = X_b.sum(axis=0)
        keep = (counts > 0) & (counts < 2 * n)

        X_parts.append(X_b[:, keep])
        p_anc_parts.append(p_anc_b[keep])
        p_pop_parts.append(p_pop_b[:, keep])
        found += int(np.count_nonzero(keep))
        n_drawn += batch

        if found >= L:
            break

        rate = max(found / n_drawn, 1e-3)  # guard a pathological all-monomorphic run
        batch = int(max(np.ceil(1.3 * (L - found) / rate), 1024))
    else:
        raise RuntimeError(
            f"only found {found} polymorphic loci of the {L} requested after "
            f"{max_batches} batches ({n_drawn} candidates) at Fst={Fst:g}, n={n}, "
            f"p_min={p_min:g}; the retention rate is implausibly low."
        )

    # Keep exactly the first L retained columns. Loci are i.i.d., so the leading L
    # are as good as a random choice among those found and avoid a second shuffle.
    X = np.concatenate(X_parts, axis=1)[:, :L]
    p_anc = np.concatenate(p_anc_parts)[:L]
    p_pop = np.concatenate(p_pop_parts, axis=1)[:, :L]

    assert X.shape == (2 * n, L), f"expected {(2 * n, L)}, got {X.shape}"
    return X, p_anc, p_pop


def observed_Fst(X, n):
    """
    Hudson's Fst estimator on the simulated matrix, as a check that the realized
    differentiation matches the target.

    Uses the ratio-of-averages form -- sum of numerators over sum of
    denominators across loci, not the average of per-locus ratios -- which is the
    consistent estimator; the per-locus ratio is badly biased at low frequency.

        numerator_l   = (p1 - p2)^2 - p1(1-p1)/(n-1) - p2(1-p2)/(n-1)
        denominator_l = p1(1-p2) + p2(1-p1)

    The two subtracted terms remove the sampling variance in each population's
    sample frequency, which would otherwise inflate Fst by ~1/n.
    """
    if n < 2:
        raise ValueError(f"the Fst estimator needs n >= 2, got {n}")

    p1 = X[:n].mean(axis=0)
    p2 = X[n:].mean(axis=0)

    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n - 1) - p2 * (1 - p2) / (n - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)

    den_sum = den.sum()
    if den_sum <= 0:
        return np.nan  # every locus monomorphic in the same allele: Fst undefined
    return float(num.sum() / den_sum)


def build_parameter_table(num_Fst=DEFAULT_NUM_FST, min_Fst=DEFAULT_MIN_FST,
                          max_Fst=DEFAULT_MAX_FST, n=DEFAULT_N, L=DEFAULT_L,
                          p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX,
                          random_seed=DEFAULT_RANDOM_SEED):
    """
    One row per simulation over num_Fst log-spaced Fst values.

    Log spacing because the range spans orders of magnitude and the detectability
    transition sits at the low end, where linear spacing would place almost no
    points. Seeds vary per row so the simulations are independent.
    """
    if num_Fst < 1:
        raise ValueError(f"num_Fst must be at least 1, got {num_Fst}")
    if not (0.0 < min_Fst <= max_Fst < 1.0):
        raise ValueError(
            f"need 0 < min_Fst <= max_Fst < 1, got min_Fst={min_Fst:g}, "
            f"max_Fst={max_Fst:g}"
        )

    Fst_values = (np.geomspace(min_Fst, max_Fst, num_Fst) if num_Fst > 1
                  else np.array([min_Fst]))

    params = pd.DataFrame({
        "simID": [format_Fst(f) for f in Fst_values],
        "Fst": Fst_values,
        "n": n,
        "N": 2 * n,
        "L": L,
        "p_min": p_min,
        "p_max": p_max,
        "random_seed": random_seed + np.arange(num_Fst),
    })

    duplicates = params.simID.duplicated()
    if duplicates.any():
        raise ValueError(
            f"{int(duplicates.sum())} Fst values collide after rounding to 4 "
            f"decimal places (e.g. {params.simID[duplicates].iloc[0]}); use "
            f"fewer values or a wider range."
        )
    return params


def simulate_one(simID, Fst, n, L, p_min, p_max, outdir, random_seed):
    """
    Simulate one Fst and save its genotype matrix as <simID>.npy.

    Returns a dict of realized quantities used as diagnostics: a run whose
    observed Fst drifts from the target is not measuring what it claims to.

    Fst_drawn is read off the true population frequencies rather than the
    genotypes: Balding-Nichols gives Var(p') = Fst * p(1-p) per locus, so
    averaging (p'_1 - p'_2)^2 / 2 against p(1-p) recovers Fst without any
    sampling noise from the Bernoulli step. Fst_observed is the genotype-based
    Hudson estimate, which is what a real analysis would see. Both are computed on
    the retained polymorphic loci, i.e. on the matrix actually saved, so they
    reflect the ascertainment described in simulate_genotypes.
    """
    X, p_anc, p_pop = simulate_genotypes(
        Fst, n=n, L=L, p_min=p_min, p_max=p_max,
        rng=np.random.default_rng(random_seed))

    # Every column is polymorphic by construction; verify rather than count, so a
    # regression in the filtering surfaces here instead of silently downstream.
    derived_counts = X.sum(axis=0)
    assert np.all((derived_counts > 0) & (derived_counts < 2 * n)), (
        f"{simID}: {int(np.count_nonzero((derived_counts == 0) | (derived_counts == 2 * n)))} "
        f"monomorphic columns survived filtering"
    )

    np.save(os.path.join(outdir, f"{simID}.npy"), X)

    heterozygosity = (p_anc * (1.0 - p_anc)).sum()
    return {
        "simID": simID,
        "Fst_drawn": float(
            0.5 * ((p_pop[0] - p_pop[1]) ** 2).sum() / heterozygosity),
        "Fst_observed": observed_Fst(X, n),
        "p_anc_mean": float(p_anc.mean()),
        "L_polymorphic": int(X.shape[1]),
    }


def run(num_Fst=DEFAULT_NUM_FST, min_Fst=DEFAULT_MIN_FST,
        max_Fst=DEFAULT_MAX_FST, n=DEFAULT_N, L=DEFAULT_L,
        p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX, outdir=DEFAULT_OUTDIR,
        random_seed=DEFAULT_RANDOM_SEED, dry_run=False, verbose=True):
    """
    Build the parameter table, run every simulation, and write the table with the
    realized diagnostics merged in.

    The table is written after the sweep rather than before it so it carries the
    observed Fst alongside the target.
    """
    params = build_parameter_table(
        num_Fst=num_Fst, min_Fst=min_Fst, max_Fst=max_Fst, n=n, L=L,
        p_min=p_min, p_max=p_max, random_seed=random_seed)

    if verbose:
        print(f"{len(params)} simulations, Fst in "
              f"[{params.Fst.min():.4e}, {params.Fst.max():.4e}] "
              f"(log-spaced), each {2 * n} x {L}")
        gb = len(params) * 2 * n * L / 1e9
        print(f"output -> {outdir}  (~{gb:.2f} GB as bool, 1 byte/entry)")

    if dry_run:
        return params

    os.makedirs(outdir, exist_ok=True)

    records = []
    bar = tqdm(total=len(params), desc="simulations", unit="sim",
               disable=not verbose)
    with bar:
        for r in params.itertuples():
            records.append(simulate_one(
                r.simID, r.Fst, int(r.n), int(r.L), r.p_min, r.p_max,
                outdir, int(r.random_seed)))
            bar.set_postfix_str(r.simID, refresh=False)
            bar.update(1)

    params = params.merge(pd.DataFrame(records), on="simID")

    params_path = os.path.join(outdir, "BN_simulation_parameters.csv")
    params.to_csv(params_path, index=False)
    if verbose:
        rel = (params.Fst_observed - params.Fst).abs() / params.Fst
        print(f"\nwrote {len(params)} simulations to {outdir}")
        print(f"parameters -> {params_path}")
        print(f"observed vs target Fst, relative: median "
              f"{rel.median():.2e}, max {rel.max():.2e}")
    return params


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-Fst", type=int, default=DEFAULT_NUM_FST, help=f"number of log-spaced Fst values (default: {DEFAULT_NUM_FST})")
    p.add_argument("--min-Fst", type=float, default=DEFAULT_MIN_FST, help=f"lower end of the Fst sweep (default: {DEFAULT_MIN_FST:g})")
    p.add_argument("--max-Fst", type=float, default=DEFAULT_MAX_FST, help=f"upper end of the Fst sweep (default: {DEFAULT_MAX_FST:g})")
    p.add_argument("--n", type=int, default=DEFAULT_N, help=f"haploid samples per population (default: {DEFAULT_N}, so N = {2 * DEFAULT_N})")
    p.add_argument("--L", type=int, default=DEFAULT_L, help=f"number of independent polymorphic SNPs (default: {DEFAULT_L:g}); every output column segregates")
    p.add_argument("--p-min", type=float, default=DEFAULT_P_MIN, help=f"lower cutoff on the ancestral frequency (default: {DEFAULT_P_MIN:g})")
    p.add_argument("--p-max", type=float, default=DEFAULT_P_MAX, help=f"upper cutoff on the ancestral frequency (default: {DEFAULT_P_MAX:g})")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--dry-run", action="store_true", help="report the plan and exit without simulating")
    args = p.parse_args()

    run(num_Fst=args.num_Fst, min_Fst=args.min_Fst, max_Fst=args.max_Fst,
        n=args.n, L=args.L, p_min=args.p_min, p_max=args.p_max,
        outdir=args.outdir, random_seed=args.random_seed,
        dry_run=args.dry_run)


if __name__ == "__main__":
    main()
