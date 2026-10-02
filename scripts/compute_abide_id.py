#!/usr/bin/env python3
"""
ABIDE intrinsic-dimension estimator using DADApy.

Implements the ABIDE iteration used in:
Fede-stack/Adaptive-nonparametric-dimensionality-reduction
(src/AdaptiveKLLE.py), and reports the approximate normal 95% CI
described by Di Noia et al.

Input:
    .npy file containing an array X of shape (n_samples, n_features)

Example:
    python abide_id.py data.npy
    python abide_id.py data.npy --n-iter 5 --dthr 6.67 --maxk 200

Dependencies:
    pip install numpy scipy dadapy
"""

import argparse
import numpy as np
from scipy.stats import norm
from dadapy import Data
from src.datautils import  load_attributes, CLIM_ATTRS, TOPO_ATTRS, GEOL_ATTRS, SOIL_ATTRS, VEGE_ATTRS
from src.utils import get_basin_list
C_STAR = 0.2032


def estimate_abide(
    X,
    n_iter=5,
    dthr=6.67,
    maxk=None,
    initial_id=None,
    confidence=0.95,
    verbose=True,
):
    """
    Estimate intrinsic dimension with ABIDE.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input coordinates.
    n_iter : int
        Number of ABIDE iterations. The reference implementation uses
        a fixed number of iterations; 5 is the value suggested in the paper.
    dthr : float
        Likelihood-ratio threshold passed to DADApy's compute_kstar().
        The repository's return_ids_kstar_binomial() default is 6.67.
    maxk : int or None
        Maximum number of nearest neighbours stored by DADApy.
        If None, DADApy's default is used.
    initial_id : float or None
        Initial ID. If None, initialize with the 2NN estimator.
    confidence : float
        Confidence level for the asymptotic normal interval.
    verbose : bool
        Print iteration diagnostics.

    Returns
    -------
    result : dict
        Contains final ID, standard error, CI, final k*, and iteration history.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")
    if X.shape[0] < 3:
        raise ValueError("At least 3 samples are required.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between 0 and 1.")

    data = Data(coordinates=X)

    # Match the reference ABIDE initialization.
    if initial_id is None:
        if maxk is not None:
            data.compute_distances(maxk=maxk)
            data.compute_id_2NN(algorithm="base")
        else:
            data.compute_id_2NN(algorithm="base")
    else:
        if maxk is None:
            data.compute_distances()
        else:
            data.compute_distances(maxk=maxk)
        data.set_id(float(initial_id))

    history = []

    for iteration in range(n_iter):
        d_current = float(data.intrinsic_dim)

        # Adaptive locally homogeneous neighbourhoods.
        data.compute_kstar(Dthr=dthr)
        kstar = np.asarray(data.kstar, dtype=int)

        # Optimal ABIDE shell ratio.
        tau = min(0.95, C_STAR ** (1.0 / d_current))

        # Radius of each point's k_i^*-th neighbour.
        r_outer = np.array(
            [distances[kstar[i]] for i, distances in enumerate(data.distances)],
            dtype=float,
        )
        r_inner = tau * r_outer

        # DADApy distances include the point itself at distance 0.
        # Hence n_inner - 1 is k_A,i and kstar - 1 is k_B,i.
        n_inner = np.sum(data.distances < r_inner[:, None], axis=1)
        kA = n_inner - 1
        kB = kstar - 1

        sum_kA = float(np.sum(kA))
        sum_kB = float(np.sum(kB))

        if sum_kA <= 0 or sum_kB <= 0:
            raise RuntimeError(
                "Degenerate neighbourhood counts encountered. "
                "Try increasing maxk or inspect duplicate/degenerate points."
            )

        # ABIDE / BIDE maximum-likelihood update.
        d_new = np.log(sum_kA / sum_kB) / np.log(tau)

        # Observed Fisher information per observation, Eq. (5):
        #
        # I(d*) = (log tau)^2 * tau^d * mean(k_B) / (1 - tau^d)
        #
        # Total information = n * I(d*), so
        # SE = 1 / sqrt(n I(d*)).
        n_samples = X.shape[0]
        tau_d = tau ** d_new
        mean_kB = np.mean(kB)

        fisher_per_obs = (
            (np.log(tau) ** 2)
            * tau_d
            * mean_kB
            / (1.0 - tau_d)
        )
        fisher_total = n_samples * fisher_per_obs

        if not np.isfinite(fisher_total) or fisher_total <= 0:
            raise RuntimeError("Invalid Fisher information; cannot construct CI.")

        se = 1.0 / np.sqrt(fisher_total)
        z = norm.ppf(0.5 + confidence / 2.0)
        ci_low = d_new - z * se
        ci_high = d_new + z * se

        history.append(
            {
                "iteration": iteration + 1,
                "id": d_new,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "tau": tau,
                "mean_kstar": float(np.mean(kstar)),
                "median_kstar": float(np.median(kstar)),
                "min_kstar": int(np.min(kstar)),
                "max_kstar": int(np.max(kstar)),
            }
        )

        if verbose:
            print(
                f"iter {iteration + 1:2d}: "
                f"ID={d_new:.6f}, "
                f"SE={se:.6f}, "
                f"{100*confidence:.1f}% CI=({ci_low:.6f}, {ci_high:.6f}), "
                f"tau={tau:.6f}, "
                f"mean(k*)={np.mean(kstar):.2f}"
            )

        data.set_id(d_new)

    final = history[-1]

    return {
        "id": final["id"],
        "se": final["se"],
        "ci": (final["ci_low"], final["ci_high"]),
        "confidence": confidence,
        "kstar": np.asarray(data.kstar, dtype=int).copy(),
        "history": history,
    }


def load_attrs():
    basins = get_basin_list()
    
    keep_attrs = (
        SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS +
        TOPO_ATTRS + GEOL_ATTRS )

    df = load_attributes(
        "data/attributes.db",
        basins,
        keep_attributes=keep_attrs
    )

   
    # z-score standardization
    df = (df - df.mean()) / df.std()
    df = df.dropna(axis=1)  # removes zero-variance attributes

    X = df.to_numpy(dtype=float)

    return X

def main():
    parser = argparse.ArgumentParser(
        description="Estimate intrinsic dimension with ABIDE using DADApy."
    )
  
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5,
        help="Number of ABIDE iterations (default: 5).",
    )
    parser.add_argument(
        "--dthr",
        type=float,
        default=6.67,
        help="Threshold for DADApy compute_kstar (default: 6.67).",
    )
    parser.add_argument(
        "--maxk",
        type=int,
        default=None,
        help="Maximum stored nearest-neighbour rank (default: DADApy default).",
    )
    parser.add_argument(
        "--initial-id",
        type=float,
        default=None,
        help="Optional initial intrinsic dimension; default is 2NN.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for normal CI (default: 0.95).",
    )
    args = parser.parse_args()

    X = load_attrs()
 
    result = estimate_abide(
        X,
        n_iter=args.n_iter,
        dthr=args.dthr,
        maxk=args.maxk,
        initial_id=args.initial_id,
        confidence=args.confidence,
        verbose=True,
    )

    lo, hi = result["ci"]
    print("\nFinal ABIDE estimate")
    print("--------------------")
    print(f"ID       : {result['id']:.6f}")
    print(f"SE       : {result['se']:.6f}")
    print(
        f"{100*result['confidence']:.1f}% CI : "
        f"[{lo:.6f}, {hi:.6f}]"
    )


if __name__ == "__main__":
    main()
