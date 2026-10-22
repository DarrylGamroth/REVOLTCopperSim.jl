"""Synthetic offline checks; no Julia, GPU or additional Python packages."""
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks"))
import analyze_atmospheric as m


def write_array(directory, name, value):
    value = np.asarray(value, dtype="<f4")
    content = gzip.compress(value.tobytes(order="F"), mtime=0)
    filename = name + ".f32.gz"
    (directory / filename).write_bytes(content)
    return {"file": filename, "shape": list(value.shape),
            "sha256": hashlib.sha256(content).hexdigest(),
            "type": "little-endian Float32 column-major"}


def dump_toml(record):
    """The fixture needs only scalar values, tables and arrays of tables."""
    lines = []
    def table(data, prefix="", header=None):
        if header:
            lines.append(header)
        for key, value in data.items():
            if not isinstance(value, dict) and not (isinstance(value, list) and value and isinstance(value[0], dict)):
                lines.append(f"{key} = {json.dumps(value)}")
        for key, value in data.items():
            path = prefix + key
            if isinstance(value, dict):
                table(value, path + ".", f"[{path}]")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    table(item, path + ".", f"[[{path}]]")
    table(record)
    return "\n".join(lines) + "\n"


def fixture(directory, case, frames=52):
    n, points = case
    name = f"n{n}-m{points}-r15-s1"
    factor = 1 if case == m.REFERENCE else 1.01
    open_ids = sorted({1, frames, *range(10, frames + 1, 10)})
    flat = np.full((64, 64), 100, dtype="<f4")
    rates = np.repeat(flat[:, :, None], frames, axis=2)
    # A positive signal in one quadrant exercises transpose and flat subtraction.
    rates[3:28, 3:28, :] += np.arange(1, frames + 1) / 10
    fine_rates = rates.copy()
    rates = (rates * factor).astype("<f4").astype(float)
    own_flat = (flat * factor).astype("<f4").astype(float)
    flat_signal = m.slopes(own_flat, 400)
    signals = np.column_stack([m.slopes(rates[:, :, k], 400) - flat_signal for k in range(frames)])
    command = np.zeros((277, frames))
    diagnostic = np.array([[2e-7 * factor], [.5 / factor], [3e-7], [.2], [0.]]) * np.ones((1, frames))
    record = {"resolution": n, "modulation_points": points, "seed": 1, "r0_m": .15,
              "frames": frames, "open_frame_ids": open_ids, "burn_in": min(50, frames // 3), "dt_s": .002,
              "gain": .15, "rcond": .05, "divisor": 400, "screen_hashes": [str(k) * 64 for k in range(5)],
              "master_pupil_resolution": 960, "science_resolution": 480, "science_wavelength_m": 550e-9,
              "calibration_resolution": n, "calibration_points": points, "retained_rank": 40,
              "diagnostic_rows": m.ROWS, "checks": []}
    record.update({key: "synthetic-source" for key in ("harness_sha256", "base_harness_sha256", "graph_sha256",
                                                       "aos_revision", "copper_revision")})
    arrays = {"rates": rates[:, :, np.array(open_ids) - 1], "open_slopes": signals[:, np.array(open_ids) - 1],
              "closed_slopes": signals, "commands": command, "diagnostics": diagnostic, "flat": own_flat}
    for key, value in arrays.items():
        record[key] = write_array(directory, name + "-" + key, value)
    for frame in sorted({1, frames, *range(50, frames + 1, 50)}):
        check = {"frame": frame, "science_960_rms_m": 2e-7 * factor * .99,
                 "science_1440_rms_m": 2e-7 * factor * .98,
                 "science_960_strehl": .5 / factor + .001,
                 "science_1440_strehl": .5 / factor + .002}
        if case not in m.ANCHORS | {m.REFERENCE}:
            own_rate, fine_rate = rates[:, :, frame - 1], fine_rates[:, :, frame - 1]
            fine_signal = m.slopes(fine_rate, 400) - m.slopes(flat, 400)
            signal = signals[:, frame - 1]
            check.update({"fine_rate": write_array(directory, name + f"-fine-{frame}", fine_rate),
                          "own_rate": write_array(directory, name + f"-own-{frame}", own_rate),
                          "same_command_rate_relative_l2": float(np.linalg.norm(own_rate - fine_rate) / np.linalg.norm(fine_rate)),
                          "same_command_slope_relative_l2": float(np.linalg.norm(signal - fine_signal) / np.linalg.norm(fine_signal)),
                          "same_command_slope_error_rms": float(np.sqrt(np.mean((signal - fine_signal) ** 2)))})
        record["checks"].append(check)
    path = directory / (name + ".toml")
    path.write_text(dump_toml(record))
    return path, record


class AtmosphericMetrics(unittest.TestCase):
    def test_error_metrics_use_frobenius_and_actual_frame_ids(self):
        reference = np.array([[1., 2.], [0., 0.]])
        actual = np.array([[2., 2.], [0., 0.]])
        result = m.error_metrics(actual, reference, [1, 10])
        self.assertAlmostEqual(result["frobenius_relative_l2"], 1 / np.sqrt(5))
        self.assertEqual(result["relative_l2_by_frame"]["worst_frame"], 1)
        self.assertEqual(result["absolute_rms"], .5)
        self.assertEqual(result["relative_l2_by_frame"]["median"], .5)

    def test_zero_reference_is_explicit(self):
        result = m.error_metrics(np.ones((2, 1)), np.zeros((2, 1)))
        self.assertIsNone(result["frobenius_relative_l2"])
        self.assertEqual(result["relative_l2_by_frame"]["undefined_count"], 1)
        json.dumps(result, allow_nan=False)
        self.assertEqual(m.error_metrics(np.zeros((2, 1)), np.zeros((2, 1)))["frobenius_relative_l2"], 0)

    def test_array_order_checksum_and_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = np.arange(24).reshape((2, 3, 4))
            record = write_array(directory, "test", value)
            np.testing.assert_equal(m.load_array(directory, record), value)
            record["shape"] = [2, 3]
            with self.assertRaisesRegex(ValueError, "byte count"):
                m.load_array(directory, record)
            record["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "checksum"):
                m.load_array(directory, record)

    def test_full_synthetic_reduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for case in m.REQUIRED:
                fixture(directory, case)
            output = directory / "reduced"
            result = m.analyze(directory, output)
            self.assertEqual(len(result["runs"]), 5)
            candidate = result["runs"]["n480-m32-r15-s1"]
            self.assertAlmostEqual(candidate["open_rate"]["frobenius_relative_l2"], .01, places=6)
            self.assertAlmostEqual(candidate["steady"]["residual_rms_mean_ratio"], 1.01, places=6)
            self.assertEqual(candidate["steady"]["frames"], 2)
            self.assertEqual(candidate["steady_open_frame_ids"], [52])
            self.assertIsNotNone(candidate["steady_open_rate"])
            self.assertEqual(len(candidate["same_command"]["checks"]), 3)
            self.assertAlmostEqual(candidate["science_grid"]["checks"][0]["rms_480_vs_1440_relative_error"], 1 / .98 - 1, places=6)
            self.assertIsNone(result["across_seeds"][0]["strehl_mean"]["std_across_seeds"])
            self.assertEqual(json.loads((output / "metrics.json").read_text())["pair_count"], 1)
            self.assertIn("31 samples", (output / "README.md").read_text())
            import csv
            with (output / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["frames"], "52")
            self.assertEqual(row["steady_frames"], "2")
            self.assertEqual(row["open_samples"], "7")

    def test_missing_coverage_and_unpaired_screen_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture(directory, m.REFERENCE)
            path, metadata = fixture(directory, (480, 32))
            with self.assertRaisesRegex(ValueError, "missing primary"):
                m.analyze(directory, directory / "out")
            m.analyze(directory, directory / "out", allow_partial=True)
            metadata["screen_hashes"][0] = "different"
            path.write_text(dump_toml(metadata))
            with self.assertRaisesRegex(ValueError, "screen_hashes"):
                m.analyze(directory, directory / "out", allow_partial=True)

    def test_post_burn_open_uses_frame_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, metadata = fixture(directory, m.REFERENCE, frames=62)
            record, arrays = m.load_record(path, metadata)
            result = m.compare(record, arrays, record, arrays, 50)
            self.assertEqual(result["steady_open_frame_ids"], [60, 62])
            self.assertEqual(result["steady_open_rate"]["frobenius_relative_l2"], 0)

    def test_sparse_frame_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, metadata = fixture(directory, m.REFERENCE)
            metadata["open_frame_ids"][-1] = 49
            with self.assertRaisesRegex(ValueError, "frame identity"):
                m.load_record(path, metadata)

    def test_optional_anchor_uses_reference_calibration_without_same_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture(directory, m.REFERENCE)
            path, metadata = fixture(directory, (1920, 256))
            with self.assertRaisesRegex(ValueError, "calibration provenance"):
                m.load_record(path, metadata)
            metadata["calibration_resolution"], metadata["calibration_points"] = m.REFERENCE
            path.write_text(dump_toml(metadata))
            metrics = m.analyze(directory, directory / "out", allow_partial=True)
            result = metrics["runs"][path.stem]
            self.assertEqual(result["case_role"], "reference_calibration_plant_sensitivity")
            self.assertFalse(result["same_command"]["available"])
            self.assertIsNone(result["same_command"]["rate"])

    def test_seed_dispersion_uses_seed_means(self):
        row = {"resolution": 480, "modulation_points": 32, "r0_m": .15, "seed": 1}
        for key in ("open_rate_relative_l2", "open_slope_response_relative_l2", "residual_rms_mean_nm",
                    "residual_rms_mean_ratio", "residual_rms_mean_difference_nm", "strehl_mean", "strehl_mean_difference",
                    "steady_open_rate_relative_l2", "steady_open_slope_response_relative_l2"):
            row[key] = 1.
        other = dict(row, seed=2, residual_rms_mean_nm=3.)
        result = m.across_seeds([row, other])[0]
        self.assertEqual(result["independent_seed_count"], 2)
        self.assertEqual(result["residual_rms_mean_nm"]["mean"], 2)
        self.assertAlmostEqual(result["residual_rms_mean_nm"]["std_across_seeds"], np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
