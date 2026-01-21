#!/usr/bin/env python3
"""Lead/lag logistic regression using weekly ROC strand features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.research.roc_utils import (
    iter_gate_files,
    load_gate_records,
    parse_week_label,
    roc_value,
    summary_lookup,
)


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return fmean(clean) if clean else None


def _share(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(1.0 for v in values if v) / len(values)


def _semantic_share(records: Sequence[Mapping[str, object]], tag: str) -> float | None:
    if not records:
        return None
    tag_lower = tag.lower()
    count = sum(1 for rec in records if tag_lower in [str(t).lower() for t in rec.get("semantic_tags", [])])
    return count / len(records)


def _mr_d5_records(records: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    out = []
    for rec in records:
        if rec.get("regime") != "mean_revert":
            continue
        if rec.get("hazard_decile") != 5:
            continue
        out.append(rec)
    return out


def _compute_week_features(
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object] | None,
    *,
    horizons: Sequence[int],
) -> Dict[str, float | int | None]:
    features: Dict[str, float | int | None] = {}
    mr_records = _mr_d5_records(records)

    def _avg_roc(mr: Sequence[Mapping[str, object]], horizon: int) -> float | None:
        values = [roc_value(rec, horizon) for rec in mr]
        clean = [v for v in values if v is not None]
        return float(fmean(clean)) if clean else None

    for horizon in horizons:
        features[f"mr_d5_avg_roc_{horizon}m"] = _avg_roc(mr_records, horizon)

    pos_flags = [roc_value(rec, 60) is not None and roc_value(rec, 60) > 0 for rec in mr_records]
    features["mr_d5_positive_share_60m"] = _share(pos_flags)
    features["mr_d5_sample_count"] = len(mr_records)
    features["mr_d5_sem_highly_stable_pct"] = _semantic_share(mr_records, "highly_stable")
    features["mr_d5_sem_strengthening_pct"] = _semantic_share(mr_records, "strengthening_structure")
    features["mr_d5_sem_low_hazard_pct"] = _semantic_share(mr_records, "low_hazard_environment")
    features["mr_d5_sem_high_rupture_pct"] = _semantic_share(mr_records, "high_rupture_event")

    coh_values = []
    dom_values = []
    entropy_values = []
    for rec in mr_records:
        structure = rec.get("structure") or {}
        coh_values.append(structure.get("coherence_tau_slope"))
        dom_values.append(structure.get("domain_wall_slope"))
        entropy_values.append(structure.get("entropy"))
    features["mr_d5_avg_coh_tau_slope"] = _mean([v for v in coh_values if isinstance(v, (int, float))])
    features["mr_d5_avg_domain_wall_slope"] = _mean([v for v in dom_values if isinstance(v, (int, float))])
    features["mr_d5_avg_entropy"] = _mean([v for v in entropy_values if isinstance(v, (int, float))])

    if summary and isinstance(summary, Mapping):
        regimes = summary.get("regimes") or {}
        neutral = regimes.get("neutral") if isinstance(regimes, Mapping) else {}
        for horizon in horizons:
            payload = neutral.get(str(horizon)) if isinstance(neutral, Mapping) else None
            if isinstance(payload, Mapping):
                features[f"neutral_avg_roc_{horizon}m"] = payload.get("avg_roc_pct")
                features[f"neutral_positive_{horizon}m"] = payload.get("positive_pct")
    return features


def _build_dataset(
    gates_dir: Path,
    *,
    horizons: Sequence[int],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    summary_map = summary_lookup(gates_dir)
    rows: List[Dict[str, object]] = []
    for gate_file in iter_gate_files(gates_dir):
        window = parse_week_label(gate_file)
        summary = summary_map.get(window.label)
        records = load_gate_records(gate_file)
        features = _compute_week_features(records, summary, horizons=horizons)
        row: Dict[str, object] = {
            "week_label": window.label,
            "week_start": window.start,
            **features,
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)
    return df, summary_map


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_neutral_avg_roc_60m"] = df["neutral_avg_roc_60m"].shift(-1)
    df["target_positive"] = (df["next_neutral_avg_roc_60m"] > 0).astype(float)
    return df.iloc[:-1]


def _filter_dataset(
    df: pd.DataFrame,
    *,
    start_week: str | None,
    end_week: str | None,
    window_weeks: int | None,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered["week_start"] = pd.to_datetime(filtered["week_start"])
    filtered = filtered.sort_values("week_start").reset_index(drop=True)
    if start_week:
        start_ts = pd.to_datetime(start_week)
        filtered = filtered[filtered["week_start"] >= start_ts]
    if end_week:
        end_ts = pd.to_datetime(end_week)
        filtered = filtered[filtered["week_start"] <= end_ts]
    if window_weeks:
        filtered = filtered.tail(window_weeks)
    return filtered.reset_index(drop=True)


def _fit_logit(df: pd.DataFrame, features: List[str]) -> Tuple[sm.Logit, sm.Logit.fit]:
    X = df[features]
    X = sm.add_constant(X, has_constant="add")
    y = df["target_positive"]
    model = sm.Logit(y, X)
    result = model.fit(disp=False, maxiter=200)
    return model, result


def _rolling_accuracy(
    df: pd.DataFrame,
    features: List[str],
    fit_fn,
    *,
    min_train: int = 8,
) -> float | None:
    if len(df) <= min_train:
        return None
    correct = 0
    total = 0
    for idx in range(min_train, len(df)):
        train = df.iloc[:idx]
        test = df.iloc[idx : idx + 1]
        try:
            params = fit_fn(train)
        except Exception:
            continue
        X_test = sm.add_constant(test[features], has_constant="add")
        linpred = float(np.dot(X_test.to_numpy()[0], params))
        prob = 1.0 / (1.0 + np.exp(-linpred))
        pred = 1.0 if prob >= 0.5 else 0.0
        if pred == float(test["target_positive"].iloc[0]):
            correct += 1
        total += 1
    return correct / total if total else None


def _save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_results(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _evaluate_accuracy(model_df: pd.DataFrame, features: List[str], params: np.ndarray) -> float:
    X = sm.add_constant(model_df[features], has_constant="add")
    logits = X.to_numpy() @ params
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(float)
    return float((preds == model_df["target_positive"]).mean())


def _fit_logit_model(model_df: pd.DataFrame, features: List[str]) -> Dict[str, object]:
    X = sm.add_constant(model_df[features], has_constant="add")
    y = model_df["target_positive"]
    model = sm.Logit(y, X)
    result = model.fit(disp=False, maxiter=500)
    accuracy = _evaluate_accuracy(model_df, features, result.params.to_numpy())
    rolling_accuracy = _rolling_accuracy(
        model_df,
        features,
        lambda df: sm.Logit(df["target_positive"], sm.add_constant(df[features], has_constant="add")).fit(
            disp=False, maxiter=200
        ).params.to_numpy(),
    )
    coeffs = result.params.to_dict()
    p_values = result.pvalues.to_dict()
    odds = {key: float(np.exp(val)) for key, val in coeffs.items()}
    return {
        "coefficients": coeffs,
        "p_values": p_values,
        "odds_ratios": odds,
        "accuracy_in_sample": accuracy,
        "rolling_accuracy": rolling_accuracy,
        "llf": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "summary": str(result.summary()),
        "params_vector": result.params.to_numpy().tolist(),
    }


def _fit_ridge_model(
    model_df: pd.DataFrame,
    features: List[str],
    *,
    alpha: float,
) -> Dict[str, object]:
    X = sm.add_constant(model_df[features], has_constant="add")
    y = model_df["target_positive"]
    model = sm.Logit(y, X)
    result = model.fit_regularized(alpha=alpha, L1_wt=0.0, disp=False, maxiter=500)
    params = result.params.to_numpy()
    accuracy = _evaluate_accuracy(model_df, features, params)

    def _ridge_fit_fn(df: pd.DataFrame) -> np.ndarray:
        subX = sm.add_constant(df[features], has_constant="add")
        suby = df["target_positive"]
        submodel = sm.Logit(suby, subX)
        sub_res = submodel.fit_regularized(alpha=alpha, L1_wt=0.0, disp=False, maxiter=500)
        return sub_res.params.to_numpy()

    rolling_accuracy = _rolling_accuracy(model_df, features, _ridge_fit_fn)
    coeffs = result.params.to_dict()
    return {
        "alpha": alpha,
        "coefficients": coeffs,
        "accuracy_in_sample": accuracy,
        "rolling_accuracy": rolling_accuracy,
        "params_vector": params.tolist(),
    }


def _fit_bayesian_model(
    model_df: pd.DataFrame,
    features: List[str],
    *,
    alpha: float,
    draws: int,
    seed: int | None,
) -> Dict[str, object]:
    X = sm.add_constant(model_df[features], has_constant="add")
    y = model_df["target_positive"]
    model = sm.Logit(y, X)
    map_result = model.fit_regularized(alpha=alpha, L1_wt=0.0, disp=False, maxiter=500)
    map_params = map_result.params.to_numpy()
    hessian = model.hessian(map_result.params)
    penalty_hessian = 2.0 * alpha * np.eye(len(map_params))
    precision = -(hessian) + penalty_hessian
    # Stabilise precision matrix if needed.
    jitter = 1e-6
    attempts = 0
    while attempts < 5:
        try:
            cov = np.linalg.inv(precision + jitter * np.eye(len(map_params)))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
            attempts += 1
    else:
        raise RuntimeError("Failed to invert posterior precision matrix for Bayesian approximation")
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=map_params, cov=cov, size=draws)
    accuracy = _evaluate_accuracy(model_df, features, map_params)
    coeff_names = list(map_result.params.index)
    stats = {}
    for idx, name in enumerate(coeff_names):
        coeff_samples = samples[:, idx]
        stats[name] = {
            "mean": float(np.mean(coeff_samples)),
            "std": float(np.std(coeff_samples, ddof=1)),
            "hpd_2.5": float(np.percentile(coeff_samples, 2.5)),
            "hpd_97.5": float(np.percentile(coeff_samples, 97.5)),
            "map": float(map_params[idx]),
        }
    return {
        "alpha": alpha,
        "draws": draws,
        "coefficients": stats,
        "accuracy_in_sample": accuracy,
        "params_vector": map_params.tolist(),
    }


def run(args: argparse.Namespace) -> int:
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    dataset, _ = _build_dataset(Path(args.roc_dir), horizons=horizons)
    dataset = _add_targets(dataset)
    dataset = _filter_dataset(
        dataset,
        start_week=args.start_week,
        end_week=args.end_week,
        window_weeks=args.window_weeks,
    )
    base_features = [
        "mr_d5_avg_roc_60m",
        "mr_d5_avg_roc_90m",
        "mr_d5_positive_share_60m",
        "mr_d5_sem_highly_stable_pct",
        "mr_d5_sem_strengthening_pct",
        "mr_d5_sem_low_hazard_pct",
        "mr_d5_sem_high_rupture_pct",
        "mr_d5_avg_coh_tau_slope",
        "mr_d5_avg_domain_wall_slope",
    ]
    missing = [col for col in base_features if col not in dataset.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")
    model_df = dataset.dropna(subset=["target_positive"])
    variable_columns = [col for col in base_features if model_df[col].nunique(dropna=True) > 1]
    model_df = model_df.dropna(subset=variable_columns)
    if not variable_columns:
        raise SystemExit("No feature columns available after filtering constant/NaN values")
    features = variable_columns

    outputs: Dict[str, object] = {
        "observations": int(len(model_df)),
        "features": features,
    }
    summaries: List[str] = []
    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]

    if "logit" in methods:
        logit_payload = _fit_logit_model(model_df, features)
        summaries.append("=== Logit ===\n" + logit_payload.pop("summary"))
        outputs["logit"] = logit_payload

    if "ridge" in methods:
        ridge_payload = _fit_ridge_model(model_df, features, alpha=args.ridge_alpha)
        outputs["ridge"] = ridge_payload

    if "bayes" in methods:
        bayes_payload = _fit_bayesian_model(
            model_df,
            features,
            alpha=args.bayes_alpha,
            draws=args.bayes_draws,
            seed=args.seed,
        )
        outputs["bayes_laplace"] = bayes_payload

    if args.output_json:
        _save_results(outputs, Path(args.output_json))
    if args.dataset_csv:
        _save_dataset(model_df, Path(args.dataset_csv))
    if args.summary_txt and summaries:
        Path(args.summary_txt).write_text("\n\n".join(summaries), encoding="utf-8")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lead/lag logistic regression for neutral ROC sign prediction")
    parser.add_argument("--roc-dir", default="docs/evidence/roc_history", help="Directory containing ROC exports")
    parser.add_argument("--horizons", default="60,90", help="Comma separated list of horizons used for features")
    parser.add_argument("--methods", default="logit,ridge,bayes", help="Comma separated list: logit,ridge,bayes")
    parser.add_argument("--ridge-alpha", type=float, default=0.5, help="L2 penalty strength for ridge variant")
    parser.add_argument("--bayes-alpha", type=float, default=0.2, help="Gaussian prior precision (alpha) for Laplace approx")
    parser.add_argument("--bayes-draws", type=int, default=4000, help="Monte Carlo draws for Bayesian Laplace approximation")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for Monte Carlo draws")
    parser.add_argument("--start-week", help="Inclusive YYYY-MM-DD filter for week_start")
    parser.add_argument("--end-week", help="Inclusive YYYY-MM-DD filter for week_start")
    parser.add_argument("--window-weeks", type=int, help="Keep only the most recent N weeks after applying start/end filters")
    parser.add_argument("--dataset-csv", default="docs/evidence/lead_lag_features.csv")
    parser.add_argument("--output-json", default="docs/evidence/lead_lag_model.json")
    parser.add_argument("--summary-txt", default="docs/evidence/lead_lag_model.txt")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
