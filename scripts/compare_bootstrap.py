"""Compare models using aligned temporal-bootstrap NSE replicates per catchment.

Inference uses paired percentile intervals and their approximate two-sided test
inversion, with a finite-replicate safeguard. Replicates are not treated as new
independent observations in a t-test. The default BY adjustment covers selected
pairs and common catchments together, allowing dependence between comparisons.
By default, compare successive feature counts and each selected encoded model
with its attributes model, within PUB and GLOBAL. Export a LaTeX table and CSVs.
"""

import argparse
import hashlib
import itertools
import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PAIRING_FIELDS = ("samples_sha256", "method", "seed", "n_bootstrap", "start_date", "end_date")
CORRECTIONS = ("fdr_by", "fdr_bh", "holm", "none")
ROOT = Path(__file__).resolve().parents[1]  # repository root


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="*", help="Optional explicit NSE files; otherwise discover models from src.plot_utils.")
    parser.add_argument("--bootstrap-dir", type=Path, default=ROOT / "analysis/bootstrap")
    parser.add_argument("--eval-period", choices=("val", "test"), default="test")
    parser.add_argument("--block-method", choices=("hydrological_year", "sliding"), default="sliding")
    parser.add_argument("--extra-global-features", type=int, nargs="+", default=[],
                        help="Also discover these GLOBAL feature counts, e.g. 26 for the significance maps.")
    parser.add_argument("--pairs", choices=("paper", "all", "adjacent"),
                        help="paper: sequential and versus-A map pairs (default for discovery); all: every pair "
                        "(default for explicit files); adjacent: neighbouring available models.")
    parser.add_argument("--labels", nargs="+", help="One model label per input file, in the same order.")
    parser.add_argument("--reference", help="Compare each other model against this label; default: all pairs.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("--correction", choices=CORRECTIONS, default="fdr_by")
    parser.add_argument("--min-valid-replicates", type=int, default=1000)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/bootstrap/comparisons")
    parser.add_argument("--plot-maps", action="store_true",
                        help="Also save the four CONUS maps in OUTPUT_DIR/maps; includes GLOBAL-26 in discovery.")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Report model filenames and metadata availability without loading bootstrap arrays or writing results.")
    args = parser.parse_args()
    if args.pairs is None:
        args.pairs = "all" if args.files else "paper"
    if args.files and len(args.files) < 2:
        parser.error("Provide at least two model files.")
    if len(set(path.resolve() for path in args.files)) != len(args.files):
        parser.error("Each model file must be supplied only once.")
    if not args.files and (args.labels is not None or args.reference is not None):
        parser.error("--labels and --reference require explicit model files.")
    if args.files and args.labels is None:
        args.labels = [path.stem.removeprefix("nse_") for path in args.files]
    if args.files and (len(args.labels) != len(args.files) or len(set(args.labels)) != len(args.labels)):
        parser.error("Provide one unique label per model file.")
    if args.reference is not None and args.reference not in args.labels:
        parser.error("--reference must match one of the model labels.")
    if args.reference is not None and args.pairs != "all":
        parser.error("--reference cannot be combined with --pairs paper or adjacent.")
    if args.plot_maps and (args.reference is not None or args.pairs == "adjacent"):
        parser.error("--plot-maps needs paper or all pairs; omit --reference and --pairs adjacent.")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1.")
    if args.min_valid_replicates < 2 or not 0 < args.min_valid_fraction <= 1:
        parser.error("Require at least two valid replicates and a valid fraction in (0, 1].")
    if any(features < 0 for features in args.extra_global_features):
        parser.error("--extra-global-features must be nonnegative.")
    if (args.plot_maps or args.pairs == "paper") and 26 not in args.extra_global_features:
        args.extra_global_features.append(26)
    return args


def select_paper_pairs(labels):
    """Use the exact map pairs, keeping the smaller feature count first in tables."""
    from src.plot_utils import get_bootstrap_comparison_pairs

    selected, skipped = [], []
    for family in ("pub", "global"):
        for kind in ("previous", "attr"):
            for higher, lower in get_bootstrap_comparison_pairs(family, kind):
                missing = [label for label in (lower, higher) if label not in labels]
                if missing:
                    skipped.append({"model_a": lower, "model_b": higher, "missing_models": missing})
                else:
                    selected.append((lower, higher))
    return selected, skipped


def discover_models(directory, eval_period="test", block_method="sliding", specs=None, extra_global_features=(),
                    check_inputs=False):
    """Find configured models with metadata, accepting pub/global and *_ae names."""
    if specs is None:
        from src.plot_utils import get_bootstrap_plot_models

        specs = get_bootstrap_plot_models()
    specs = list(specs)
    for features in extra_global_features:
        if not any(spec["family"] == "global" and spec["encoded_features"] == features for spec in specs):
            specs.append({"family": "global", "file_experiment": "global", "encoded_features": features,
                          "random_features": False, "label": f"GLOBAL-{features}"})
    suffix = "_sliding365" if block_method == "sliding" else ""
    discovered, skipped = {}, []
    for family in ("pub", "global"):
        ordered = sorted(
            (spec for spec in specs if spec["family"] == family and not spec["random_features"]),
            key=lambda spec: float("inf") if spec["encoded_features"] is None else spec["encoded_features"],
        )
        for spec in ordered:
            experiment = spec["file_experiment"]
            aliases = [experiment]
            if experiment in ("pub", "global"):
                aliases.append(f"{experiment}_ae")
            candidates = [
                Path(directory) / f"nse_{alias}_{spec['encoded_features']}_{eval_period}{suffix}.pkl"
                for alias in aliases
            ]
            related = sorted({path for alias in aliases
                              for path in Path(directory).glob(f"*nse_{alias}_{spec['encoded_features']}_*.pkl")
                              if path not in candidates})
            if check_inputs:
                print(f"{spec['label']} ({eval_period}, {block_method}):")
                for path in candidates:
                    print(f"  {path.resolve()}: pickle {'OK' if path.is_file() else 'MISSING'}, "
                          f"JSON {'OK' if path.with_suffix('.json').is_file() else 'MISSING'}")
                for path in related:
                    print(f"  Other filename (not selected): {path.resolve()}")
            ready = [path for path in candidates if path.is_file() and path.with_suffix(".json").is_file()]
            if len(ready) > 1:
                raise ValueError(f"Ambiguous files for {spec['label']}: {ready}. Use a directory with one file per model.")
            if not ready:
                existing = [path for path in candidates if path.is_file()]
                reason = (
                    "missing JSON metadata for " + ", ".join(path.name for path in existing)
                    if existing else f"no NSE pickle matching period={eval_period}, block_method={block_method}"
                )
                skipped.append({
                    "label": spec["label"],
                    "reason": reason,
                    "expected_files": [str(path) for path in candidates],
                    "related_files": [str(path) for path in related],
                })
                continue
            discovered[spec["label"]] = {**spec, "path": ready[0]}
    # An isolated model cannot contribute to a within-family comparison.
    for label, spec in list(discovered.items()):
        if sum(other["family"] == spec["family"] for other in discovered.values()) < 2:
            skipped.append({"label": label, "reason": "no other available model in this family"})
            del discovered[label]
    if not discovered:
        raise ValueError(
            f"No within-family pair found in {directory} for {eval_period}, {block_method}. "
            "At least two PUB or two GLOBAL models from src.plot_utils need both NSE .pkl and .json files."
        )
    return discovered, skipped


def load_model(path):
    """Require provenance before allowing elementwise comparisons of two files."""
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(
            f"Missing metadata: {metadata_path}. Rerun this model with scripts.compute_block_bootstrap "
            "using the shared sampling plan; pairing of legacy or restart outputs cannot be verified."
        )
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or any(key not in metadata for key in PAIRING_FIELDS):
        raise ValueError(f"Incomplete bootstrap metadata: {metadata_path}.")
    if metadata["method"] not in ("october_september_years", "sliding_365_days"):
        raise ValueError(f"Unsupported bootstrap method in {metadata_path}.")
    if not isinstance(metadata["samples_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata["samples_sha256"]):
        raise ValueError(f"Invalid sampling-plan hash in {metadata_path}.")
    n_bootstrap = metadata["n_bootstrap"]
    if type(n_bootstrap) is not int or n_bootstrap < 2:
        raise ValueError(f"Invalid replicate count in {metadata_path}.")
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Expected a nonempty catchment-to-NSE dictionary in {path}.")
    scores = {}
    for basin, values in raw.items():
        basin = str(basin)
        if basin in scores:
            raise ValueError(f"Duplicate catchment ID after string conversion in {path}: {basin}.")
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) != n_bootstrap:
            raise ValueError(f"Basin {basin} in {path}: expected {n_bootstrap} aligned NSE replicates.")
        scores[basin] = values
    return {"path": path, "metadata": metadata, "scores": scores}


def validate_pairing(models):
    reference = next(iter(models.values()))["metadata"]
    for label, model in models.items():
        for key in PAIRING_FIELDS:
            if model["metadata"][key] != reference[key]:
                raise ValueError(
                    f"Model {label}: {key} differs. Compare outputs produced with the same "
                    "sampling plan, block method, evaluation dates, seed and replicate count."
                )
    common_basins = set.intersection(*(set(model["scores"]) for model in models.values()))
    if not common_basins:
        raise ValueError("The model files contain no common catchments.")
    excluded = {
        label: sorted(set(model["scores"]) - common_basins)
        for label, model in models.items()
    }
    if any(excluded.values()):
        warnings.warn(
            f"Using {len(common_basins)} catchments shared by every selected model; "
            "excluded catchment IDs will be listed in comparison_config.json.",
            stacklevel=2,
        )
    return sorted(common_basins), excluded


def paired_comparison(a, b, alpha=0.05, min_valid_replicates=1000, min_valid_fraction=0.95):
    """Invert the paired percentile interval; retain only jointly finite indices."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not len(a):
        raise ValueError("Paired arrays must be nonempty, one-dimensional and equally long.")
    valid = np.isfinite(a) & np.isfinite(b)
    delta = a[valid] - b[valid]
    n_valid = len(delta)
    row = {
        "n_bootstrap": len(a),
        "n_valid": n_valid,
        "valid_fraction": n_valid / len(a),
        "delta_nse_median": float(np.median(delta)) if n_valid else np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "bootstrap_win_fraction_a": float(np.mean(delta > 0)) if n_valid else np.nan,
        "p_value": np.nan,
        "p_resolution": min(1.0, 2 / (n_valid + 1)),
        "status": "insufficient_replicates",
    }
    if n_valid < min_valid_replicates or row["valid_fraction"] < min_valid_fraction:
        return row
    row["ci_lower"], row["ci_upper"] = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
    # Percentile-CI inversion: twice the smaller inclusive tail at H0: delta = 0.
    # Adding one prevents claiming p=0 from a finite number of replicates.
    tail_count = min(np.count_nonzero(delta <= 0), np.count_nonzero(delta >= 0))
    row["p_value"] = min(1.0, 2 * (tail_count + 1) / (n_valid + 1))
    row["status"] = "tested"
    return row


def adjust_pvalues(p_values, method="fdr_by"):
    """Adjust one complete family; untestable entries count as non-rejections."""
    if method not in CORRECTIONS:
        raise ValueError(f"Unknown multiple-comparison correction: {method}.")
    p_values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p_values)
    if p_values.ndim != 1 or np.any((p_values[finite] < 0) | (p_values[finite] > 1)):
        raise ValueError("Expected a one-dimensional array of p-values in [0, 1].")
    if method == "none" or not len(p_values):
        return p_values.copy()
    work = np.where(finite, p_values, 1.0)
    order = np.argsort(work)
    n_tests = len(work)
    if method == "holm":
        sorted_adjusted = np.maximum.accumulate(work[order] * np.arange(n_tests, 0, -1))
    else:
        ranks = np.arange(1, n_tests + 1)
        factor = np.sum(1 / ranks) if method == "fdr_by" else 1.0
        scaled = work[order] * n_tests * factor / ranks
        sorted_adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n_tests)
    adjusted[order] = np.minimum(sorted_adjusted, 1.0)
    adjusted[~finite] = np.nan
    return adjusted


def compare_models(models, reference=None, alpha=0.05, correction="fdr_by",
                   min_valid_replicates=1000, min_valid_fraction=0.95,
                   model_groups=None, adjacent_only=False, selected_pairs=None):
    if len(models) < 2:
        raise ValueError("At least two models are required.")
    if reference is not None and reference not in models:
        raise ValueError(f"Unknown reference model: {reference}.")
    if reference is not None and adjacent_only:
        raise ValueError("Reference comparisons cannot also be adjacent comparisons.")
    if model_groups is not None and set(model_groups) != set(models):
        raise ValueError("Provide one family for every model.")
    groups = model_groups or dict.fromkeys(models, "custom")
    if selected_pairs is not None:
        if reference is not None or adjacent_only:
            raise ValueError("Selected pairs cannot be combined with reference or adjacent comparisons.")
        seen = set()
        for a, b in selected_pairs:
            if a not in models or b not in models or a == b or groups[a] != groups[b]:
                raise ValueError(f"Invalid selected model pair: {a} / {b}.")
            key = frozenset((a, b))
            if key in seen:
                raise ValueError(f"Duplicate selected model pair: {a} / {b}.")
            seen.add(key)
    pairs, excluded = [], {}
    for family in dict.fromkeys(groups.values()):
        members = {label: model for label, model in models.items() if groups[label] == family}
        if len(members) < 2 or (reference is not None and reference not in members):
            continue
        common_basins, group_excluded = validate_pairing(members)
        excluded.update(group_excluded)
        labels = list(members)
        if selected_pairs is not None:
            group_pairs = [(a, b) for a, b in selected_pairs if a in members]
        elif reference is not None:
            group_pairs = [(label, reference) for label in labels if label != reference]
        elif adjacent_only:
            group_pairs = zip(labels[:-1], labels[1:])
        else:
            group_pairs = itertools.combinations(labels, 2)
        pairs.extend((family, a, b, common_basins) for a, b in group_pairs)
    if not pairs:
        raise ValueError("No model pairs are available within the selected families.")
    rows = []
    for family, model_a, model_b, common_basins in pairs:
        for basin in common_basins:
            row = paired_comparison(
                models[model_a]["scores"][basin], models[model_b]["scores"][basin],
                alpha, min_valid_replicates, min_valid_fraction,
            )
            rows.append({"family": family, "model_a": model_a, "model_b": model_b, "basin": basin, **row})
    comparisons = pd.DataFrame(rows)
    comparisons["p_adjusted"] = adjust_pvalues(comparisons["p_value"], correction)
    comparisons["significant_raw"] = comparisons["p_value"] <= alpha
    comparisons["significant_adjusted"] = comparisons["p_adjusted"] <= alpha
    comparisons["result"] = "not_significant"
    comparisons.loc[comparisons["status"] != "tested", "result"] = "insufficient_replicates"
    significant = comparisons["significant_adjusted"]
    comparisons.loc[significant & (comparisons["delta_nse_median"] > 0), "result"] = "a_better"
    comparisons.loc[significant & (comparisons["delta_nse_median"] < 0), "result"] = "b_better"
    summaries = []
    for (model_a, model_b), group in comparisons.groupby(["model_a", "model_b"], sort=False):
        tested = group["status"] == "tested"
        summaries.append({
            "family": group["family"].iloc[0],
            "model_a": model_a,
            "model_b": model_b,
            "n_catchments": len(group),
            "n_tested": int(tested.sum()),
            "a_better": int((group["result"] == "a_better").sum()),
            "b_better": int((group["result"] == "b_better").sum()),
            "not_significant": int((group["result"] == "not_significant").sum()),
            "insufficient_replicates": int((~tested).sum()),
            "median_catchment_delta_nse": group.loc[tested, "delta_nse_median"].median(),
        })
    return comparisons, pd.DataFrame(summaries), excluded


def latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def write_latex_table(summary, output_dir, alpha=0.05, correction="fdr_by"):
    """Export a booktabs tabular fragment for direct inclusion with LaTeX input."""
    lines = [
        r"% Requires \usepackage{booktabs}. Insert with \input{comparison_table.tex}.",
        f"% Paired bootstrap: alpha={alpha:g}, correction={correction}.",
        "% Correction covers all selected pairs and catchments in this run.",
        "% Percentages use all catchments in each comparison.",
        "% Unresolved = no significant winner + insufficient bootstrap replicates.",
        "% Delta NSE = median catchment difference (first model - second model), over tested catchments.",
        "% No detected difference does not establish equivalence.",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Comparison & First wins & Second wins & \shortstack{Unresolved\\catchments} & $\Delta$NSE \\",
        r"\midrule",
    ]
    for family, group in summary.groupby("family", sort=False):
        title = "Selected models" if family == "custom" else family.upper()
        lines.append(r"\multicolumn{5}{l}{" + latex_escape(title) + r"} \\")
        for row in group.itertuples(index=False):
            def count(value):
                return f"{value} ({100 * value / row.n_catchments:.1f}" + r"\%)" if row.n_catchments else str(value)

            a, b = latex_escape(row.model_a), latex_escape(row.model_b)
            wins_a, wins_b = count(row.a_better), count(row.b_better)
            effect = f"{row.median_catchment_delta_nse:+.4f}" if np.isfinite(row.median_catchment_delta_nse) else "--"
            unresolved = row.not_significant + row.insufficient_replicates
            cells = [f"{a} vs {b}", wins_a, wins_b, count(unresolved), effect]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path = Path(output_dir) / "comparison_table.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = get_args()
    try:
        specs, skipped, model_groups = {}, [], None
        if args.files:
            paths = dict(zip(args.labels, args.files))
        else:
            print(f"Reading bootstrap inputs from {args.bootstrap_dir.resolve()}")
            specs, skipped = discover_models(args.bootstrap_dir, args.eval_period, args.block_method,
                                             extra_global_features=args.extra_global_features,
                                             check_inputs=args.check_inputs)
            paths = {label: spec["path"] for label, spec in specs.items()}
            model_groups = {label: spec["family"] for label, spec in specs.items()}
            print("Discovered models: " + ", ".join(paths))
            for model in skipped:
                print(f"Skipping {model['label']}: {model['reason']}.")
                if model.get("related_files"):
                    print("  Other filenames found: " + ", ".join(model["related_files"]))
        if args.check_inputs:
            if args.files:
                for label, path in paths.items():
                    print(f"{label}: {path.resolve()}; pickle {'OK' if path.is_file() else 'MISSING'}, "
                          f"JSON {'OK' if path.with_suffix('.json').is_file() else 'MISSING'}")
            print("Input check finished; no bootstrap arrays loaded or outputs written.")
            return
        selected_pairs, skipped_pairs = None, []
        if args.pairs == "paper":
            selected_pairs, skipped_pairs = select_paper_pairs(paths)
            if not selected_pairs:
                raise ValueError("No sequential or versus-A pairs available. Check model files and PUB/GLOBAL labels, "
                                 "or use --pairs all for custom comparisons.")
            for pair in skipped_pairs:
                print(f"Skipping pair {pair['model_a']} / {pair['model_b']}: missing "
                      + ", ".join(pair["missing_models"]))
            used = {label for pair in selected_pairs for label in pair}
            paths = {label: path for label, path in paths.items() if label in used}
            if model_groups is not None:
                model_groups = {label: model_groups[label] for label in paths}
        models = {label: load_model(path) for label, path in paths.items()}
        comparisons, summary, excluded = compare_models(
            models, args.reference, args.alpha, args.correction,
            args.min_valid_replicates, args.min_valid_fraction,
            model_groups=model_groups, adjacent_only=args.pairs == "adjacent",
            selected_pairs=selected_pairs,
        )
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(args.output_dir / "catchment_comparisons.csv", index=False)
    summary.to_csv(args.output_dir / "pair_summary.csv", index=False)
    config = {
        "models": {
            label: {
                "file": str(model["path"].resolve()),
                "file_sha256": file_sha256(model["path"]),
                "bootstrap_metadata": model["metadata"],
                **({"family": specs[label]["family"], "encoded_features": specs[label]["encoded_features"]}
                   if label in specs else {}),
            }
            for label, model in models.items()
        },
        "reference": args.reference,
        "eval_period": args.eval_period,
        "selection": "explicit files" if args.files else "src.plot_utils.get_bootstrap_plot_models",
        "pairs": args.pairs,
        "selected_pairs": summary[["model_a", "model_b"]].values.tolist(),
        "skipped_pairs": skipped_pairs,
        "skipped_models": skipped,
        "alpha": args.alpha,
        "correction": args.correction,
        "correction_family": "all selected pairs across families; common catchments within each family",
        "n_comparisons": len(comparisons),
        "n_tested": int((comparisons["status"] == "tested").sum()),
        "min_valid_replicates": args.min_valid_replicates,
        "min_valid_fraction": args.min_valid_fraction,
        "excluded_catchments": excluded,
        "delta_definition": "NSE(model_a) - NSE(model_b), replicate by replicate",
        "confidence_interval": "unadjusted equal-tailed percentile interval",
        "test": "approximate percentile-CI inversion with inclusive tails and +1 safeguard",
        "p_value_formula": "min(1, 2 * (min(count(delta <= 0), count(delta >= 0)) + 1) / (n_valid + 1))",
        "interpretation": "not_significant does not establish equivalence; bootstrap win fractions are not posterior probabilities",
    }
    (args.output_dir / "comparison_config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_latex_table(summary, args.output_dir, args.alpha, args.correction)
    print("Paired delta NSE = model_a - model_b; positive values favour model_a.")
    print(f"Correction: {args.correction}, alpha={args.alpha:g}, family size={len(comparisons)}.")
    print(summary.to_string(index=False))
    print(f"Saved comparison_table.tex, CSV results and settings to {args.output_dir.resolve()}")
    if args.plot_maps:
        from scripts.plot_bootstrap_significance import generate_maps

        try:
            generate_maps(args.output_dir, args.output_dir / "maps", args.eval_period)
        except (ValueError, OSError) as error:
            raise SystemExit(f"Comparison results were saved, but CONUS maps could not be generated: {error}") from error
    else:
        import shlex

        print("To generate the CONUS evidence and win/loss maps from these results, run:")
        print(shlex.join(["python", "-m", "scripts.plot_bootstrap_significance",
                          "--comparison-dir", str(args.output_dir.resolve()),
                          "--eval_period", args.eval_period,
                          "--output-dir", str(args.output_dir.resolve() / "maps")]))


if __name__ == "__main__":
    main()
