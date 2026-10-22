"""Offline paired atmospheric reduction using only Python and NumPy.

Usage: python3 benchmarks/analyze_atmospheric.py RAW_DIR OUTPUT_DIR
Use --allow-partial --burn-in 4 for a short smoke run. Default production
coverage is five resolutions for every recorded (seed, r0) pair.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_convergence import slopes

REFERENCE = (1440, 256)
REQUIRED = {REFERENCE, (480, 32), (180, 64), (240, 96), (240, 128)}
ANCHORS = {(1920, 256), (1440, 512)}
ROWS = ["residual_opd_rms_m", "on_axis_strehl", "open_opd_rms_m",
        "open_on_axis_strehl", "command_rms_m"]
PAIRED_KEYS = ("frames", "open_frame_ids", "dt_s", "gain", "rcond", "divisor", "screen_hashes",
               "master_pupil_resolution", "science_resolution", "science_wavelength_m",
               "harness_sha256", "base_harness_sha256", "graph_sha256", "aos_revision",
               "copper_revision", "burn_in")
QUALIFICATIONS = (
    "Finite 960-pixel master pupil, native 3n periodic atmosphere screens; all grids "
    "sample the same screens. The 3.66 m canvas omits frequencies below 0.273 m⁻¹, "
    "above 1/L0 = 0.033 m⁻¹; no subharmonics restore this low-frequency power. "
    "Winds travel up to 6 m in 0.6 s, reusing periodic structure. "
    "The 1440 / 256 comparison is a numerical reference, "
    "not an independent physical oracle. No detector noise or pixel quantization "
    "is applied to the host controller. Primary cases use their own calibration "
    "with a 5% singular-value cutoff and fixed gain 0.15. Optional anchors may "
    "reuse the 1440 / 256 calibration, as recorded. Science diagnostics use a "
    "common 480 grid with sparse 960 and 1440 checks. This exercises a causal "
    "host controller and plant, with no actual pyRTC integration. Timing covers "
    "only the warmed static plant endpoint; controller, atmosphere generation, "
    "transfers and science diagnostics are excluded."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_array(directory, record):
    """Verify compressed bytes before decoding the declared column-major array."""
    path = directory / record["file"]
    require(path.resolve().is_relative_to(directory.resolve()), f"array escapes input directory: {path}")
    require(record["type"] == "little-endian Float32 column-major", f"unexpected array encoding: {path}")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == record["sha256"], f"checksum mismatch: {path}")
    raw = gzip.decompress(content)
    shape = tuple(record["shape"])
    require(all(isinstance(n, int) and n > 0 for n in shape), f"invalid shape: {path}")
    require(len(raw) == 4 * int(np.prod(shape)), f"byte count disagrees with shape: {path}")
    values = np.frombuffer(raw, dtype="<f4").reshape(shape, order="F").astype(np.float64)
    require(np.isfinite(values).all(), f"nonfinite array: {path}")
    return values


def load_record(path, record=None):
    record = tomllib.loads(path.read_text()) if record is None else record
    frames = record["frames"]
    open_ids = record["open_frame_ids"]
    require(open_ids == sorted({1, frames, *range(10, frames + 1, 10)}),
            f"{path.name}: unexpected open sample frame identity")
    shapes = {"rates": (64, 64, len(open_ids)), "open_slopes": (1432, len(open_ids)),
              "closed_slopes": (1432, frames), "commands": (277, frames),
              "diagnostics": (5, frames), "flat": (64, 64)}
    arrays = {}
    for key, shape in shapes.items():
        arrays[key] = load_array(path.parent, record[key])
        require(arrays[key].shape == shape, f"{path.name}: invalid {key} shape")
    require(np.all(arrays["rates"] >= 0) and np.all(arrays["flat"] >= 0), f"{path.name}: negative photon rate")
    require(record["diagnostic_rows"] == ROWS, f"{path.name}: diagnostic row ordering differs")
    require(record["gain"] == 0.15 and record["rcond"] == 0.05,
            f"{path.name}: expected fixed gain 0.15 and rcond 0.05")
    require(record["divisor"] > 0 and record["dt_s"] > 0, f"{path.name}: invalid normalization or cadence")
    require(record["master_pupil_resolution"] == 960 and record["science_resolution"] == 480,
            f"{path.name}: unsupported master/science grid")
    case = (record["resolution"], record["modulation_points"])
    calibration = (record["calibration_resolution"], record["calibration_points"])
    require(calibration == (REFERENCE if case in ANCHORS else case),
            f"{path.name}: unexpected calibration provenance")
    require(len(record["screen_hashes"]) == 5, f"{path.name}: expected five atmospheric layers")
    check_frames = [c["frame"] for c in record["checks"]]
    expected = sorted({1, frames, *range(50, frames + 1, 50)})
    require(check_frames == expected, f"{path.name}: science check frame identity differs")
    derived = np.column_stack([slopes(arrays["rates"][:, :, k], record["divisor"])
                               for k in range(len(open_ids))])
    derived -= slopes(arrays["flat"], record["divisor"])[:, None]
    require(np.allclose(derived, arrays["open_slopes"], rtol=2e-6, atol=2e-9),
            f"{path.name}: archived open slopes disagree with rates/flat/order")
    require(np.all(arrays["commands"][:, 0] == 0), f"{path.name}: first command is not zero")
    require(np.allclose(arrays["open_slopes"][:, 0], arrays["closed_slopes"][:, 0],
                        rtol=2e-6, atol=2e-9), f"{path.name}: first closed frame differs from open frame")
    rms = np.sqrt(np.mean(arrays["commands"] ** 2, axis=0))
    require(np.allclose(rms, arrays["diagnostics"][4], rtol=2e-6, atol=1e-13),
            f"{path.name}: diagnostic command RMS differs from applied commands")
    for key in ("open_opd_rms_m", "residual_opd_rms_m"):
        require(np.all(arrays["diagnostics"][ROWS.index(key)] >= 0), f"{path.name}: negative RMS")
    require(np.all((arrays["diagnostics"][[1, 3]] >= 0) &
                   (arrays["diagnostics"][[1, 3]] <= 1 + 1e-6)), f"{path.name}: invalid Strehl")
    for check in record["checks"]:
        for key in ("fine_rate", "own_rate"):
            if key in check:
                values = load_array(path.parent, check[key])
                require(values.shape == (64, 64), f"{path.name}: invalid {key} check shape")
                arrays.setdefault("checks", {}).setdefault(check["frame"], {})[key] = values
    return record, arrays


def distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"median": float(np.median(finite)) if finite.size else None,
            "p95": float(np.quantile(finite, .95)) if finite.size else None,
            "worst": float(np.max(finite)) if finite.size else None,
            "undefined_count": int(values.size - finite.size)}


def error_metrics(actual, expected, frame_ids=None):
    """Frames are the final axis; relative errors with zero signal are explicit."""
    actual = np.asarray(actual, dtype=float).reshape((-1, actual.shape[-1]))
    expected = np.asarray(expected, dtype=float).reshape(actual.shape)
    delta = actual - expected
    denominator = np.linalg.norm(expected, axis=0)
    numerator = np.linalg.norm(delta, axis=0)
    per_frame = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)
    per_frame[(denominator == 0) & (numerator == 0)] = 0
    reference_norm = float(np.linalg.norm(expected))
    total_error = float(np.linalg.norm(delta))
    relative = total_error / reference_norm if reference_norm else (0.0 if total_error == 0 else None)
    rms = np.sqrt(np.mean(delta ** 2, axis=0))
    relative_stats = distribution(per_frame)
    if np.isfinite(per_frame).any():
        index = int(np.nanargmax(per_frame))
        relative_stats["worst_frame"] = frame_ids[index] if frame_ids is not None else index + 1
    else:
        relative_stats["worst_frame"] = None
    return {"frobenius_relative_l2": relative, "absolute_rms": float(np.sqrt(np.mean(delta ** 2))),
            "reference_rms": float(np.sqrt(np.mean(expected ** 2))),
            "relative_l2_by_frame": relative_stats,
            "absolute_rms_by_frame": dict(distribution(rms), worst_frame=(frame_ids[int(np.argmax(rms))]
                if frame_ids is not None else int(np.argmax(rms)) + 1))}


def science_checks(record, arrays):
    rows = []
    for check in record["checks"]:
        k = check["frame"] - 1
        row = {"frame": k + 1, "science_480_rms_m": float(arrays["diagnostics"][0, k]),
               "science_480_strehl": float(arrays["diagnostics"][1, k])}
        for dense in (960, 1440):
            rms = check[f"science_{dense}_rms_m"]
            strehl = check[f"science_{dense}_strehl"]
            require(np.isfinite(rms) and rms > 0 and np.isfinite(strehl) and 0 <= strehl <= 1,
                    f"invalid science diagnostic at frame {k + 1}")
            row[f"science_{dense}_rms_m"] = rms
            row[f"science_{dense}_strehl"] = strehl
            row[f"rms_480_vs_{dense}_relative_error"] = (row["science_480_rms_m"] - rms) / rms
            row[f"rms_480_vs_{dense}_error_nm"] = 1e9 * (row["science_480_rms_m"] - rms)
            row[f"strehl_480_vs_{dense}_error"] = row["science_480_strehl"] - strehl
        row["rms_960_vs_1440_relative_error"] = row["science_960_rms_m"] / row["science_1440_rms_m"] - 1
        row["strehl_960_vs_1440_error"] = row["science_960_strehl"] - row["science_1440_strehl"]
        rows.append(row)
    summary = {}
    for key in rows[0]:
        if "error" in key:
            summary[key + "_absolute"] = distribution([abs(r[key]) for r in rows])
    return {"checks": rows, "summary": summary}


def same_command_checks(record, arrays, reference):
    rows, own_rates, fine_rates, own_signals, fine_signals = [], [], [], [], []
    own_flat = slopes(arrays["flat"], record["divisor"])
    fine_flat = slopes(reference["flat"], record["divisor"])
    for check in record["checks"]:
        if "own_rate" not in check:
            continue
        frame = check["frame"]
        rates = arrays["checks"][frame]
        own = slopes(rates["own_rate"], record["divisor"]) - own_flat
        fine = slopes(rates["fine_rate"], record["divisor"]) - fine_flat
        require(np.allclose(own, arrays["closed_slopes"][:, frame - 1], rtol=2e-6, atol=2e-9),
                f"frame {frame}: same-command own rate differs from archived closed slopes")
        rate_metrics = error_metrics(rates["own_rate"][..., None], rates["fine_rate"][..., None])
        slope_metrics = error_metrics(own[:, None], fine[:, None])
        for key, value in (("same_command_rate_relative_l2", rate_metrics["frobenius_relative_l2"]),
                           ("same_command_slope_relative_l2", slope_metrics["frobenius_relative_l2"]),
                           ("same_command_slope_error_rms", slope_metrics["absolute_rms"])):
            require(value is not None and np.isclose(check[key], value, rtol=5e-5, atol=1e-10),
                    f"frame {frame}: recomputed {key} differs from harness")
        own_rates.append(rates["own_rate"])
        fine_rates.append(rates["fine_rate"])
        own_signals.append(own)
        fine_signals.append(fine)
        rows.append({"frame": frame, "rate_relative_l2": rate_metrics["frobenius_relative_l2"],
                     "slope_response_relative_l2": slope_metrics["frobenius_relative_l2"],
                     "slope_response_error_rms": slope_metrics["absolute_rms"]})
    case = (record["resolution"], record["modulation_points"])
    require(len(rows) == len(record["checks"]) or (not rows and case in ANCHORS | {REFERENCE}),
            "missing same-command checks")
    return {"available": bool(rows),
            "unavailable_reason": None if rows else ("reference trajectory" if case == REFERENCE else
                "optional anchor: concurrent fine-system exposures were not acquired"),
            "checks": rows, "summary": {key: distribution([r[key] for r in rows])
                                        for key in ("rate_relative_l2", "slope_response_relative_l2",
                                                    "slope_response_error_rms")} if rows else {},
            "rate": error_metrics(np.stack(own_rates, axis=-1), np.stack(fine_rates, axis=-1),
                                  [r["frame"] for r in rows]) if rows else None,
            "slope_response": error_metrics(np.stack(own_signals, axis=-1), np.stack(fine_signals, axis=-1),
                                            [r["frame"] for r in rows]) if rows else None}


def compare(record, arrays, reference_record, reference, burn_in):
    for key in PAIRED_KEYS:
        require(record[key] == reference_record[key], f"unpaired metadata: {key}")
    for key in ("seed", "r0_m"):
        require(record[key] == reference_record[key], f"unpaired atmosphere: {key}")
    require(np.allclose(arrays["diagnostics"][[2, 3]], reference["diagnostics"][[2, 3]],
                        rtol=2e-6, atol=1e-12), "common open science diagnostics differ")
    require(0 <= burn_in < record["frames"], "burn-in must leave at least one measured frame")
    own, fine = arrays["diagnostics"][:, burn_in:], reference["diagnostics"][:, burn_in:]
    steady = {"frames": record["frames"] - burn_in,
              "residual_rms_mean_nm": float(1e9 * own[0].mean()),
              "reference_residual_rms_mean_nm": float(1e9 * fine[0].mean()),
              "residual_rms_mean_ratio": float(own[0].mean() / fine[0].mean()),
              "residual_rms_mean_difference_nm": float(1e9 * (own[0] - fine[0]).mean()),
              "strehl_mean": float(own[1].mean()), "reference_strehl_mean": float(fine[1].mean()),
              "strehl_mean_difference": float((own[1] - fine[1]).mean()),
              "open_rms_mean_nm": float(1e9 * own[2].mean()),
              "open_strehl_mean": float(own[3].mean()),
              "command_rms_mean_nm": float(1e9 * own[4].mean()),
              "residual_rms_temporal_nm": distribution(1e9 * own[0]),
              "strehl_temporal": distribution(own[1])}
    steady_open = np.asarray(record["open_frame_ids"]) > burn_in
    total = arrays["rates"].sum(axis=(0, 1))
    fine_total = reference["rates"].sum(axis=(0, 1))
    result = {"photon_rate_total": dict(
                  own_min_photons_s=float(total.min()), own_max_photons_s=float(total.max()),
                  own_mean_photons_s=float(total.mean()), reference_mean_photons_s=float(fine_total.mean()),
                  errors=error_metrics(total[None, :], fine_total[None, :], record["open_frame_ids"])),
              "open_rate": error_metrics(arrays["rates"], reference["rates"], record["open_frame_ids"]),
              "open_slope_response": error_metrics(arrays["open_slopes"], reference["open_slopes"], record["open_frame_ids"]),
              "closed_trajectory_slope_response": error_metrics(arrays["closed_slopes"], reference["closed_slopes"]),
              "command_trajectory": error_metrics(arrays["commands"], reference["commands"]),
              "steady": steady, "steady_open_frame_ids": np.asarray(record["open_frame_ids"])[steady_open].tolist(),
              "steady_open_rate": error_metrics(arrays["rates"][:, :, steady_open], reference["rates"][:, :, steady_open],
                                                np.asarray(record["open_frame_ids"])[steady_open].tolist()) if steady_open.any() else None,
              "steady_open_slope_response": error_metrics(arrays["open_slopes"][:, steady_open],
                    reference["open_slopes"][:, steady_open], np.asarray(record["open_frame_ids"])[steady_open].tolist())
                    if steady_open.any() else None,
              "science_grid": science_checks(record, arrays),
              "same_command": same_command_checks(record, arrays, reference)}
    stats = record.get("plant_timing")
    if stats:
        require(np.all(np.asarray(stats["host_bytes"]) == 0), "plant timing has host allocations")
        require(np.median(stats["samples_ns"]) == stats["p50_ns"], "plant median timing differs")
        result["plant_p50_us"] = stats["p50_ns"] / 1000
    return result


def summary_row(record, result):
    row = {key: record[key] for key in ("resolution", "modulation_points", "seed", "r0_m", "frames",
                                       "calibration_resolution", "calibration_points", "retained_rank")}
    for label in ("open_rate", "open_slope_response"):
        values = result[label]
        row[label + "_relative_l2"] = values["frobenius_relative_l2"]
        row[label + "_absolute_rms"] = values["absolute_rms"]
        for quantile in ("median", "p95", "worst"):
            row[label + "_frame_" + quantile] = values["relative_l2_by_frame"][quantile]
    row["calibration_mode"] = "shared_reference" if (record["resolution"], record["modulation_points"]) in ANCHORS else "own_case"
    row["total_photon_rate_mean_photons_s"] = result["photon_rate_total"]["own_mean_photons_s"]
    row["total_photon_rate_worst_relative_error"] = result["photon_rate_total"]["errors"]["relative_l2_by_frame"]["worst"]
    row["open_samples"] = len(record["open_frame_ids"])
    row["steady_open_samples"] = len(result["steady_open_frame_ids"])
    for label in ("steady_open_rate", "steady_open_slope_response"):
        row[label + "_relative_l2"] = result[label]["frobenius_relative_l2"] if result[label] is not None else None
    row.update({("steady_frames" if key == "frames" else key): value
                for key, value in result["steady"].items() if isinstance(value, (float, int))})
    for label, stats in result["same_command"]["summary"].items():
        row["same_command_" + label + "_worst"] = stats["worst"]
    for label, stats in result["science_grid"]["summary"].items():
        row[label + "_worst"] = stats["worst"]
    row["plant_p50_us"] = result.get("plant_p50_us")
    return row


def across_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["resolution"], row["modulation_points"], row["r0_m"])].append(row)
    summaries = []
    keys = ("open_rate_relative_l2", "open_slope_response_relative_l2",
            "steady_open_rate_relative_l2", "steady_open_slope_response_relative_l2", "residual_rms_mean_nm",
            "residual_rms_mean_ratio", "residual_rms_mean_difference_nm", "strehl_mean", "strehl_mean_difference")
    for (n, m, r0), group in sorted(groups.items()):
        summary = {"resolution": n, "modulation_points": m, "r0_m": r0,
                   "seeds": sorted(r["seed"] for r in group), "independent_seed_count": len(group)}
        for key in keys:
            values = np.array([r[key] for r in group if r[key] is not None])
            summary[key] = {"mean": float(values.mean()) if values.size else None,
                            "std_across_seeds": float(values.std(ddof=1)) if values.size > 1 else None,
                            "min": float(values.min()) if values.size else None,
                            "max": float(values.max()) if values.size else None}
        summaries.append(summary)
    return summaries


def write_readme(output, metrics):
    lines = ["# Paired atmospheric discretization results", "", QUALIFICATIONS, "",
             f"Reference: 1440 / 256. Burn-in: {metrics['burn_in']} frames; frame k uses the command "
             "applied before the next controller update. Sampling is paired by seed, r0, screen hashes "
             "and source/controller metadata. Open slopes have each case's flat response removed.", "",
             "Open exposures are sampled at frame 1, every tenth frame and the last frame (31 samples for a 300-frame run); "
             "closed trajectories and science diagnostics retain every update. Open error statistics "
             "describe only those sampled frames. Post-burn open metrics select actual frame IDs greater "
             "than the burn-in (25 samples at burn 50 for 300 updates). Optional anchor runs are plant "
             "sensitivity checks with the reference calibration, without concurrent fine-system same-command "
             "exposures. Relative errors are aggregate Frobenius norms and temporal median, p95 and worst frame "
             "L2 errors. Absolute slope RMS is dimensionless; rates retain the archived photon-rate "
             "units. Total photon rates and their paired throughput errors are also reported separately from image-shape errors. Undefined relative errors (zero reference signal with nonzero error) are null "
             "and counted. Closed trajectory differences include the accumulated controller response; "
             "sparse same-command checks isolate the plant response to identical applied commands.", "",
             "Each seed is one replicate within each r0. Temporal summaries are descriptive; frames "
             "are correlated and are not independent replicates. Reusing a seed across r0 also does "
             "not create independent atmosphere realizations. Seed dispersion uses sample standard "
             "deviation; a single seed has no dispersion estimate.", "",
             f"Coverage: {len(metrics['runs'])} runs, {metrics['pair_count']} seed/r0 pairs. "
             f"Partial coverage allowed: {metrics['allow_partial']}. See metrics.json for exact coverage "
             "and calibration provenance.", "",
             "| Grid / points | r0 (m) | Seeds | Open rate error | Open slope error | "
             "Residual RMS (nm), seed mean ± SD | Mean Strehl | Residual RMS / reference |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def number(value, percent=False):
        return "undefined" if value is None else (f"{100 * value:.3f}%" if percent else f"{value:.5g}")
    for group in metrics["across_seeds"]:
        rms = group["residual_rms_mean_nm"]
        sd = rms["std_across_seeds"]
        rms_text = number(rms["mean"]) + (" ± " + number(sd) if sd is not None else " (one seed)")
        lines.append(f"| {group['resolution']} / {group['modulation_points']} | {group['r0_m']:.2f} | "
                     f"{group['independent_seed_count']} | {number(group['open_rate_relative_l2']['mean'], True)} | "
                     f"{number(group['open_slope_response_relative_l2']['mean'], True)} | {rms_text} | "
                     f"{number(group['strehl_mean']['mean'])} | {number(group['residual_rms_mean_ratio']['mean'])} |")
    lines += ["", "Science grid errors hold each run's command fixed. Their sparse samples at frame 1, "
              "every 50 frames and the last frame do not bound unsampled frames. metrics.json reports "
              "480 versus 960/1440 RMS errors and Strehl differences, plus 960 versus 1440 differences. "
              "summary.csv contains one row per seed/r0/case. No automatic accuracy threshold or "
              "production preset recommendation is inferred from this finite ensemble.", ""]
    (output / "README.md").write_text("\n".join(lines))


def analyze(directory, output, burn_in=50, allow_partial=False):
    directory, output = Path(directory), Path(output)
    groups = defaultdict(dict)
    for path in sorted(directory.glob("n*-m*-r*-s*.toml")):
        record = tomllib.loads(path.read_text())
        case = (record["resolution"], record["modulation_points"])
        require(case in REQUIRED | ANCHORS, f"unsupported case: {case}")
        pair = (record["seed"], record["r0_m"])
        require(case not in groups[pair], f"duplicate case for seed/r0: {path.name}")
        groups[pair][case] = (path, record)
    require(groups, "no atmospheric run records found")
    results, rows, coverage = {}, [], []
    for (seed, r0), group in sorted(groups.items()):
        require(REFERENCE in group, f"missing reference for seed={seed}, r0={r0}")
        require(allow_partial or REQUIRED <= group.keys(), f"missing primary cases for seed={seed}, r0={r0}")
        reference_record, reference = load_record(*group[REFERENCE])
        coverage.append({"seed": seed, "r0_m": r0, "cases": [list(c) for c in sorted(group)]})
        for case, (path, metadata) in sorted(group.items()):
            record, arrays = (reference_record, reference) if case == REFERENCE else load_record(path, metadata)
            result = compare(record, arrays, reference_record, reference, burn_in)
            result["metadata"] = {key: record[key] for key in (*PAIRED_KEYS, "seed", "r0_m", "resolution",
                                    "modulation_points", "calibration_resolution", "calibration_points", "retained_rank")}
            result["case_role"] = ("reference_calibration_plant_sensitivity" if case in ANCHORS else
                                   "numerical_reference" if case == REFERENCE else "own_calibrated_candidate")
            result["record_file"] = path.name
            result["record_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            results[path.stem] = result
            rows.append(summary_row(record, result))
    metrics = {"reference": "n1440-m256", "burn_in": burn_in, "allow_partial": allow_partial,
               "pair_count": len(groups), "coverage": coverage, "qualifications": QUALIFICATIONS,
               "numpy_version": np.__version__, "python_version": platform.python_version(),
               "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "runs": results, "across_seeds": across_seeds(rows)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_readme(output, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    metrics = analyze(args.directory, args.output, args.burn_in, args.allow_partial)
    print(f"Reduced {len(metrics['runs'])} runs in {metrics['pair_count']} paired seed/r0 groups to {args.output}")
