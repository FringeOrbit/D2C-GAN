"""Cross-basin evaluation on the public Espirito Santo Basin well-log dataset.

The source contains GR, DT, RHOB and NPHI together with well coordinates and a
location flag.  DT is mapped to the project's DTC channel and NPHI is converted
from percent to fraction.  The model and training scaler remain frozen.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OptimizedConfig
from models.generator import AdvancedSeqGenerator
from checkpoint_io import load_generator_state


FEATURES = ["GR", "RHOB", "NPHI", "DTC"]
MISSING_RATE_LIST = [0.2, 0.4, 0.6, 0.8]
MISSING_SENTINELS = {-999.25, -999.0, -9999.0}


def load_espirito_dataset(path):
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    required = {"WELL", "DEPT", "GR", "DT", "RHOB", "NPHI", "LOCATION"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Espirito Santo file is missing columns: {sorted(missing)}")

    location = pd.to_numeric(frame["LOCATION"], errors="coerce")
    location_label = location.map({0.0: "onshore", 1.0: "offshore"}).fillna("UNKNOWN")

    out = pd.DataFrame({
        "WELL": frame["WELL"].astype(str).str.strip(),
        "DEPTH_MD": pd.to_numeric(frame["DEPT"], errors="coerce"),
        "GR": pd.to_numeric(frame["GR"], errors="coerce"),
        "RHOB": pd.to_numeric(frame["RHOB"], errors="coerce"),
        "NPHI": pd.to_numeric(frame["NPHI"], errors="coerce") / 100.0,
        "DTC": pd.to_numeric(frame["DT"], errors="coerce"),
        "LITH": location_label,
    })
    out = out.replace(list(MISSING_SENTINELS), np.nan)
    out.loc[out["LITH"].isin(["", "nan", "None"]), "LITH"] = "UNKNOWN"
    out = out.dropna(subset=["WELL", "DEPTH_MD"]).copy()

    # A common ascending order makes the sequence direction consistent.
    out = out.sort_values(["WELL", "DEPTH_MD"], kind="stable").reset_index(drop=True)
    return out


def standardize(frame, scaler):
    raw = frame[FEATURES].to_numpy(dtype=np.float32)
    natural_missing = ~np.isfinite(raw)
    scaled = (raw - scaler.mean_) / scaler.scale_
    scaled[natural_missing] = 0.0
    return scaled.astype(np.float32), natural_missing


def metric_row(well, rate, feature, true_values, pred_values, level="overall", lithology="ALL"):
    errors = pred_values - true_values
    row = {
        "well": well,
        "missing_rate": rate,
        "feature": feature,
        "level": level,
        "lithology": lithology,
        "n": int(len(true_values)),
        "sse": float(np.sum(errors ** 2)) if len(errors) else 0.0,
        "sae": float(np.sum(np.abs(errors))) if len(errors) else 0.0,
        "sum_y": float(np.sum(true_values)) if len(true_values) else 0.0,
        "sum_y2": float(np.sum(true_values ** 2)) if len(true_values) else 0.0,
    }
    if len(true_values) == 0:
        row.update({"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pcc": np.nan})
    else:
        if len(true_values) > 1 and np.std(true_values) > 0 and np.std(pred_values) > 0:
            pcc = float(np.corrcoef(true_values, pred_values)[0, 1])
            pcc = pcc if np.isfinite(pcc) else np.nan
        else:
            pcc = np.nan
        row.update({
            "rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
            "mae": float(mean_absolute_error(true_values, pred_values)),
            "r2": float(r2_score(true_values, pred_values)) if len(true_values) > 1 else np.nan,
            "pcc": pcc,
        })
    return row


def stable_seed(well, rate, seed):
    well_seed = int(hashlib.md5(str(well).encode("utf-8")).hexdigest()[:8], 16)
    return seed + int(rate * 1000) + well_seed


def evaluate_well(model, scaler, config, frame, seed):
    scaled, natural_missing = standardize(frame, scaler)
    original_n = len(frame)
    pad_len = (config.seq_len - (original_n % config.seq_len)) % config.seq_len
    if pad_len:
        scaled = np.pad(scaled, ((0, pad_len), (0, 0)), mode="edge")
        natural_missing = np.pad(natural_missing, ((0, pad_len), (0, 0)), mode="edge")

    model_input = torch.from_numpy(
        scaled.reshape(-1, config.seq_len, len(FEATURES)).transpose(0, 2, 1)
    ).float().to(config.device)
    natural_tensor = torch.from_numpy(
        natural_missing.reshape(-1, config.seq_len, len(FEATURES)).transpose(0, 2, 1)
    ).to(config.device)
    raw_true = frame[FEATURES].to_numpy(dtype=np.float32)
    lith = frame["LITH"].to_numpy()
    well = frame["WELL"].iloc[0]
    rows, lithology_rows, physics_rows, prediction_chunks = [], [], [], []

    for rate in MISSING_RATE_LIST:
        generator = torch.Generator()
        generator.manual_seed(stable_seed(well, rate, seed))
        mask = torch.rand(model_input.shape, generator=generator) > rate
        mask = mask.to(config.device) & ~natural_tensor
        incomplete = model_input.clone()
        incomplete[~mask] = 0.0
        with torch.no_grad():
            prediction = model(incomplete, mask=mask.float())

        scaled_pred = prediction.detach().cpu().numpy().transpose(0, 2, 1).reshape(-1, 4)[:original_n]
        raw_pred = scaler.inverse_transform(scaled_pred)
        flat_mask = mask.detach().cpu().numpy().transpose(0, 2, 1).reshape(-1, 4)[:original_n]

        for feature_idx, feature in enumerate(FEATURES):
            valid = (~flat_mask[:, feature_idx]) & (~natural_missing[:original_n, feature_idx]) & np.isfinite(raw_true[:, feature_idx])
            true_values = raw_true[valid, feature_idx]
            pred_values = raw_pred[valid, feature_idx]
            rows.append(metric_row(well, rate, feature, true_values, pred_values))
            prediction_chunks.append({
                "missing_rate": rate,
                "feature": feature,
                "well": np.asarray([well] * int(valid.sum()), dtype=str),
                "depth": frame["DEPTH_MD"].to_numpy(dtype=np.float32)[valid],
                "lithology": lith[valid].astype(str),
                "y_true": true_values.astype(np.float32),
                "y_pred": pred_values.astype(np.float32),
            })
            for lithology in np.unique(lith[valid]):
                lith_valid = valid & (lith == lithology)
                lithology_rows.append(metric_row(
                    well, rate, feature, raw_true[lith_valid, feature_idx], raw_pred[lith_valid, feature_idx],
                    level="lithology", lithology=str(lithology),
                ))

        # The Wyllie residual is reported as a diagnostic, not as an external
        # This is a diagnostic, not a claim that this dataset calibrates Wyllie.
        nphi_idx, dtc_idx = FEATURES.index("NPHI"), FEATURES.index("DTC")
        valid_pair = (
            (~flat_mask[:, nphi_idx]) & (~flat_mask[:, dtc_idx])
            & (~natural_missing[:original_n, nphi_idx])
            & (~natural_missing[:original_n, dtc_idx])
        )
        dtc_wyllie = raw_pred[:, nphi_idx] * getattr(config, 'wyllie_dt_fluid', 189.0) + (1.0 - raw_pred[:, nphi_idx]) * getattr(config, 'wyllie_dt_matrix', 55.5)
        residual = raw_pred[valid_pair, dtc_idx] - dtc_wyllie[valid_pair]
        true_dtc_wyllie = raw_true[:, nphi_idx] * getattr(config, 'wyllie_dt_fluid', 189.0) + (1.0 - raw_true[:, nphi_idx]) * getattr(config, 'wyllie_dt_matrix', 55.5)
        true_residual = raw_true[valid_pair, dtc_idx] - true_dtc_wyllie[valid_pair]
        physics_rows.append({
            "well": well,
            "missing_rate": rate,
            "n": int(len(residual)),
            "pred_wyllie_rmse": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else np.nan,
            "pred_wyllie_mae": float(np.mean(np.abs(residual))) if len(residual) else np.nan,
            "true_wyllie_rmse": float(np.sqrt(np.mean(true_residual ** 2))) if len(true_residual) else np.nan,
            "true_wyllie_mae": float(np.mean(np.abs(true_residual))) if len(true_residual) else np.nan,
        })
    return rows, lithology_rows, physics_rows, prediction_chunks


def pooled_summary(frame, group_columns):
    grouped = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = int(group["n"].sum())
        sse = float(group["sse"].sum())
        sae = float(group["sae"].sum())
        sum_y = float(group["sum_y"].sum())
        sum_y2 = float(group["sum_y2"].sum())
        total_ss = sum_y2 - (sum_y ** 2 / n) if n else np.nan
        row = dict(zip(group_columns, keys))
        row.update({
            "wells": int(group["well"].nunique()),
            "n": n,
            "rmse": float(np.sqrt(sse / n)) if n else np.nan,
            "mae": float(sae / n) if n else np.nan,
            "r2": float(1.0 - sse / total_ss) if total_ss and total_ss > 0 else np.nan,
        })
        grouped.append(row)
    return pd.DataFrame(grouped)


def attach_pooled_pcc(summary, prediction_chunks, group_columns):
    """Attach pooled Pearson correlation using the saved target/prediction arrays."""
    grouped = {}
    for chunk in prediction_chunks:
        rate = chunk["missing_rate"]
        feature = chunk["feature"]
        if "lithology" in group_columns:
            for lithology in np.unique(chunk["lithology"]):
                mask = chunk["lithology"] == lithology
                key = (rate, feature, str(lithology))
                grouped.setdefault(key, [[], []])
                grouped[key][0].append(chunk["y_true"][mask])
                grouped[key][1].append(chunk["y_pred"][mask])
        else:
            key = (rate, feature)
            grouped.setdefault(key, [[], []])
            grouped[key][0].append(chunk["y_true"])
            grouped[key][1].append(chunk["y_pred"])

    pcc_values = []
    for _, row in summary.iterrows():
        key = tuple(row[column] for column in group_columns)
        true_values = np.concatenate(grouped[key][0]) if key in grouped else np.array([])
        pred_values = np.concatenate(grouped[key][1]) if key in grouped else np.array([])
        if len(true_values) > 1 and np.std(true_values) > 0 and np.std(pred_values) > 0:
            value = float(np.corrcoef(true_values, pred_values)[0, 1])
            pcc_values.append(value if np.isfinite(value) else np.nan)
        else:
            pcc_values.append(np.nan)
    summary = summary.copy()
    summary["pcc"] = pcc_values
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--model-variant", default="unspecified")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    config = OptimizedConfig()
    config.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(args.scaler)
    model = AdvancedSeqGenerator(config).to(config.device)
    model.load_state_dict(load_generator_state(args.checkpoint, config.device), strict=True)
    model.eval()

    frame = load_espirito_dataset(args.data_file)
    if args.audit_only:
        print({"rows": len(frame), "wells": int(frame.WELL.nunique()), "locations": frame.LITH.value_counts(dropna=False).to_dict()})
        print(frame[FEATURES].describe(percentiles=[0.01, 0.5, 0.99]).to_string())
        print("missing", frame[FEATURES].isna().sum().to_dict())
        print("well_sizes", frame.groupby("WELL").size().describe().to_dict())
        return
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_lithology_rows, all_physics, all_prediction_chunks = [], [], [], []
    wells_metadata = []
    for well, well_frame in frame.groupby("WELL", sort=True):
        if len(well_frame) < config.seq_len:
            continue
        rows, lithology_rows, physics_rows, prediction_chunks = evaluate_well(
            model, scaler, config, well_frame.reset_index(drop=True), args.seed
        )
        all_rows.extend(rows)
        all_lithology_rows.extend(lithology_rows)
        all_physics.extend(physics_rows)
        all_prediction_chunks.extend(prediction_chunks)
        wells_metadata.append({
            "well": well,
            "rows": int(len(well_frame)),
            "depth_min": float(well_frame.DEPTH_MD.min()),
            "depth_max": float(well_frame.DEPTH_MD.max()),
            "valid_gr": int(well_frame.GR.notna().sum()),
            "valid_rhob": int(well_frame.RHOB.notna().sum()),
            "valid_nphi": int(well_frame.NPHI.notna().sum()),
            "valid_dtc": int(well_frame.DTC.notna().sum()),
        })

    metrics = pd.DataFrame(all_rows)
    lithology_metrics = pd.DataFrame(all_lithology_rows)
    physics = pd.DataFrame(all_physics)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for rate in MISSING_RATE_LIST:
        for feature in FEATURES:
            chunks = [
                chunk for chunk in all_prediction_chunks
                if chunk["missing_rate"] == rate and chunk["feature"] == feature
            ]
            if not chunks:
                continue
            np.savez_compressed(
                prediction_dir / f"external_es_predictions_{int(rate * 100):02d}_{feature}.npz",
                well=np.concatenate([chunk["well"] for chunk in chunks]),
                depth=np.concatenate([chunk["depth"] for chunk in chunks]),
                lithology=np.concatenate([chunk["lithology"] for chunk in chunks]),
                y_true=np.concatenate([chunk["y_true"] for chunk in chunks]),
                y_pred=np.concatenate([chunk["y_pred"] for chunk in chunks]),
            )
    metrics.to_csv(output_dir / "external_es_metrics_by_well.csv", index=False)
    lithology_metrics.to_csv(output_dir / "external_es_metrics_by_location.csv", index=False)
    physics.to_csv(output_dir / "external_es_wyllie_by_well.csv", index=False)
    summary = pooled_summary(metrics, ["missing_rate", "feature"])
    summary = attach_pooled_pcc(summary, all_prediction_chunks, ["missing_rate", "feature"])
    summary.to_csv(output_dir / "external_es_summary.csv", index=False)
    lithology_summary = pooled_summary(lithology_metrics, ["missing_rate", "feature", "lithology"])
    lithology_summary = attach_pooled_pcc(
        lithology_summary, all_prediction_chunks, ["missing_rate", "feature", "lithology"]
    )
    lithology_summary.to_csv(output_dir / "external_es_location_summary.csv", index=False)
    physics_summary = physics.groupby("missing_rate", as_index=False).agg(
        wells=("well", "nunique"), n=("n", "sum"),
        pred_wyllie_rmse=("pred_wyllie_rmse", "mean"), pred_wyllie_mae=("pred_wyllie_mae", "mean"),
        true_wyllie_rmse=("true_wyllie_rmse", "mean"), true_wyllie_mae=("true_wyllie_mae", "mean"),
    )
    physics_summary.to_csv(output_dir / "external_es_wyllie_summary.csv", index=False)
    metadata = {
        "source": "Comprehensive well-log and geochemical data set of the Espirito Santo Basin, SE Brazil",
        "data_file": str(args.data_file),
        "checkpoint": str(args.checkpoint),
        "scaler": str(args.scaler),
        "model_variant": args.model_variant,
        "curve_mapping": {"GR": "GR", "DT": "DTC", "RHOB": "RHOB", "NPHI_percent": "NPHI_fraction", "Location": "onshore/offshore stratification"},
        "source_rows": int(len(frame)),
        "source_wells": int(frame.WELL.nunique()),
        "sort_order": "ascending depth within each well",
        "missing_rates": MISSING_RATE_LIST,
        "seed": args.seed,
        "window_protocol": "non-overlapping 80-point windows within each well; final padding excluded",
        "mask_protocol": "pointwise random, stable per-well/rate seed; NOT continuous blocks",
        "checkpoint_loading": "restricted deserialization and strict generator state matching",
        "wells": wells_metadata,
        "note": "The model and training scaler are frozen. Metrics are computed only on artificially masked, originally valid targets.",
    }
    (output_dir / "external_es_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(lithology_summary.to_string(index=False))
    print(physics_summary.to_string(index=False))


if __name__ == "__main__":
    main()
