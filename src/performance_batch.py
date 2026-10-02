import numpy as np
import pandas as pd
from numba import njit


METRIC_NAMES = (
    "nse", "bias", "stdev", "obs5", "sim5", "obs95", "sim95", "obs0", "sim0",
    "obsL", "simL", "obsH", "simH", "obsFDC", "simFDC", "obsBF", "simBF",
    "abs_nse", "sqrt_nse", "FHV", "FLV", "kge", "FMM", "bias_RR", "log_stdev",
    "r_coeff", "bias_std", "skew_rat", "kurt_rat",
)


@njit(cache=True, error_model="numpy")
def _baseflow_columns(flows):
    """Original single-pass c=0.925 filter, compiled for several columns."""
    result = np.empty(flows.shape[1])
    c = 0.925
    for column in range(flows.shape[1]):
        previous = flows[0, column]
        numerator = 0.0
        denominator = 0.0
        for t in range(1, len(flows)):
            previous = min(flows[t, column], c * previous +
                           (1.0 - c) / 2.0 * (flows[t, column] + flows[t - 1, column]))
            numerator += previous
            denominator += flows[t, column]
        result[column] = numerator / denominator
    return result


def _shared_mask_metrics(obs, sims, record_length):
    """All columns here are finite, nonnegative, and use the same dates."""
    flows = np.column_stack((obs, sims))
    means = flows.mean(axis=0)
    stds = flows.std(axis=0, ddof=1)
    centered = flows - means
    quantiles = np.quantile(flows, [0.05, 0.95, 0.33, 0.66, 0.5], axis=0)
    medians = quantiles[4]
    errors = sims - obs[:, None]
    squared_error = (errors ** 2).mean(axis=0)
    obs_variance = np.mean(centered[:, 0] ** 2)
    root_obs = np.sqrt(obs)
    root_error = np.sqrt(sims) - root_obs[:, None]
    correlation = (centered[:, :1] * centered[:, 1:]).sum(axis=0) / np.sqrt(
        (centered[:, 0] ** 2).sum() * (centered[:, 1:] ** 2).sum(axis=0)
    )
    ratio_mean = means[1:] / means[0]
    ratio_std = stds[1:] / stds[0]
    fdc = (np.log(quantiles[2]) - np.log(quantiles[3])) / (0.66 - 0.33)
    baseflow = _baseflow_columns(flows)
    log_std = np.log(flows + 1e-8).std(axis=0, ddof=1)
    # Pandas preserves the scalar functions' corrected sample moments and their
    # constant-series behavior, without repeatedly indexing by date labels.
    moments = pd.DataFrame(flows)
    skew = moments.skew().to_numpy()
    kurt = moments.kurt().to_numpy()
    zeros = (flows == 0).sum(axis=0)
    high = (flows >= 9 * medians).sum(axis=0)
    low = (flows <= 0.2 * medians).sum(axis=0)

    fhv = np.empty(sims.shape[1])
    flv = np.empty(sims.shape[1])
    for column in range(sims.shape[1]):
        # Preserve the original secondary sort by simulated flow for tied obs.
        # Both metrics use the original (pre-mask) record length for cutoffs.
        order = np.lexsort((-sims[:, column], -obs))
        high_idx = order[:int(record_length * 0.02)]
        low_idx = order[int(record_length * 0.9):]
        fhv[column] = errors[high_idx, column].sum() / obs[high_idx].sum()
        flv[column] = errors[low_idx, column].sum() / obs[low_idx].sum()

    return {
        "nse": 1 - squared_error / obs_variance,
        "bias": (means[1:] - means[0]) / means[0],
        "stdev": ratio_std,
        "obs5": quantiles[0, 0], "sim5": quantiles[0, 1:],
        "obs95": quantiles[1, 0], "sim95": quantiles[1, 1:],
        "obs0": zeros[0], "sim0": zeros[1:],
        "obsL": low[0], "simL": low[1:],
        "obsH": high[0], "simH": high[1:],
        "obsFDC": fdc[0], "simFDC": fdc[1:],
        "obsBF": baseflow[0], "simBF": baseflow[1:],
        "abs_nse": 1 - np.abs(errors).mean(axis=0) / np.abs(centered[:, 0]).mean(),
        "sqrt_nse": 1 - (root_error ** 2).mean(axis=0) / np.var(root_obs),
        "FHV": fhv, "FLV": flv,
        "kge": 1 - np.sqrt((correlation - 1) ** 2 + (ratio_mean - 1) ** 2 + (ratio_std - 1) ** 2),
        "FMM": (np.log(means[1:]) - np.log(means[0])) / np.log(means[0]),
        "bias_RR": errors.sum(axis=0) / obs.sum(),
        "log_stdev": log_std[1:] / log_std[0],
        "r_coeff": correlation,
        "bias_std": (means[1:] - means[0]) / stds[0],
        "skew_rat": skew[1:] / skew[0], "kurt_rat": kurt[1:] / kurt[0],
    }


def calculate_metrics(obs, predictions):
    """Return one row per prediction, using the scalar functions' metric names."""
    obs = np.asarray(obs, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if obs.ndim != 1 or predictions.ndim != 2 or len(obs) != len(predictions):
        raise ValueError("Expected observations (time,) and predictions (time, members).")
    if np.isinf(obs).any() or np.isinf(predictions).any():
        raise ValueError("Infinite streamflow values are not supported.")
    groups = {}
    for column in range(predictions.shape[1]):
        valid = (obs >= 0) & (predictions[:, column] >= 0)
        groups.setdefault(valid.tobytes(), (valid, []))[1].append(column)

    result = pd.DataFrame(np.nan, index=range(predictions.shape[1]), columns=METRIC_NAMES)
    with np.errstate(divide="ignore", invalid="ignore"):
        for valid, columns in groups.values():
            if not valid.any():
                continue
            metrics = _shared_mask_metrics(obs[valid], predictions[valid][:, columns], len(obs))
            for name, values in metrics.items():
                result.loc[columns, name] = values
    return result
