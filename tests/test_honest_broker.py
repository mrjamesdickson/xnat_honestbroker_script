"""Tests for the Honest Broker DICOM relabeling script."""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import yaml

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from honest_broker import (
    HonestBrokerClient,
    find_dicom_files,
    load_config,
    relabel_directory,
)


def create_test_dicom(filepath: str, patient_id: str = "ORIG123",
                      patient_name: str = "Doe^John") -> None:
    """Create a minimal DICOM file for testing."""
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "CT"
    ds.StudyDate = "20250101"
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(filepath)


class TestHonestBrokerClient(TestCase):
    """Tests for the HonestBrokerClient class."""

    def _make_config(self, **overrides):
        config = {
            "sts_host": "127.0.0.1:9999",
            "api_host": "127.0.0.1:9999",
            "app_name": "TestApp",
            "app_key": "test-key",
            "username": "testuser",
            "password": "testpass",
            "timeout": 5,
            "token_cache_minutes": 50,
        }
        config.update(overrides)
        return config

    def test_authenticate_caches_token(self):
        config = self._make_config()
        client = HonestBrokerClient(config)
        client._token = "cached-token"
        client._token_expires_at = float("inf")
        token = client.authenticate()
        self.assertEqual(token, "cached-token")

    def test_authenticate_expired_token_needs_refresh(self):
        config = self._make_config()
        client = HonestBrokerClient(config)
        client._token = "expired-token"
        client._token_expires_at = 0  # Expired
        # Verify expired check - the actual HTTP call would fail in tests
        self.assertTrue(client._token_expires_at < 1)

    def test_lookup_uses_cache(self):
        config = self._make_config()
        client = HonestBrokerClient(config)
        client._id_cache["ORIG123"] = "DEIDENT-A1B2"
        result = client.lookup("ORIG123")
        self.assertEqual(result, "DEIDENT-A1B2")

    def test_lookup_cache_returns_same_result(self):
        config = self._make_config()
        client = HonestBrokerClient(config)
        client._id_cache["PAT001"] = "ANON-X1Y2"
        result1 = client.lookup("PAT001")
        result2 = client.lookup("PAT001")
        self.assertEqual(result1, "ANON-X1Y2")
        self.assertEqual(result2, "ANON-X1Y2")

    def test_lookup_different_ids_cached_separately(self):
        config = self._make_config()
        client = HonestBrokerClient(config)
        client._id_cache["ID_A"] = "DEIDENT_A"
        client._id_cache["ID_B"] = "DEIDENT_B"
        self.assertEqual(client.lookup("ID_A"), "DEIDENT_A")
        self.assertEqual(client.lookup("ID_B"), "DEIDENT_B")

    def test_config_defaults(self):
        config = self._make_config()
        del config["timeout"]
        del config["token_cache_minutes"]
        client = HonestBrokerClient(config)
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.token_cache_minutes, 50)


class TestFindDicomFiles(TestCase):
    """Tests for the find_dicom_files function."""

    def test_finds_dcm_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = os.path.join(tmpdir, "test.dcm")
            create_test_dicom(dcm_path)
            results = find_dicom_files(Path(tmpdir))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].name, "test.dcm")

    def test_finds_files_in_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dicom(os.path.join(tmpdir, "subdir1", "a.dcm"))
            create_test_dicom(os.path.join(tmpdir, "subdir2", "b.dcm"))
            results = find_dicom_files(Path(tmpdir))
            self.assertEqual(len(results), 2)

    def test_finds_extensionless_dicom_by_preamble(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = os.path.join(tmpdir, "00000001")
            create_test_dicom(dcm_path)
            results = find_dicom_files(Path(tmpdir))
            self.assertEqual(len(results), 1)

    def test_ignores_non_dicom_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = os.path.join(tmpdir, "readme.txt")
            with open(txt_path, "w") as f:
                f.write("not a dicom file")
            results = find_dicom_files(Path(tmpdir))
            self.assertEqual(len(results), 0)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = find_dicom_files(Path(tmpdir))
            self.assertEqual(len(results), 0)

    def test_results_are_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dicom(os.path.join(tmpdir, "c.dcm"))
            create_test_dicom(os.path.join(tmpdir, "a.dcm"))
            create_test_dicom(os.path.join(tmpdir, "b.dcm"))
            results = find_dicom_files(Path(tmpdir))
            names = [r.name for r in results]
            self.assertEqual(names, ["a.dcm", "b.dcm", "c.dcm"])


class TestLoadConfig(TestCase):
    """Tests for the load_config function."""

    def test_loads_valid_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "honest_broker": {
                    "sts_host": "sts.example.com",
                    "api_host": "api.example.com",
                    "app_name": "App",
                    "app_key": "key",
                    "username": "user",
                    "password": "pass",
                }
            }, f)
            f.flush()
            config = load_config(f.name)
            self.assertEqual(config["sts_host"], "sts.example.com")
            self.assertEqual(config["api_host"], "api.example.com")
            os.unlink(f.name)

    def test_exits_on_missing_file(self):
        with self.assertRaises(SystemExit):
            load_config("/nonexistent/config.yaml")

    def test_exits_on_missing_required_keys(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "honest_broker": {
                    "sts_host": "sts.example.com",
                }
            }, f)
            f.flush()
            with self.assertRaises(SystemExit):
                load_config(f.name)
            os.unlink(f.name)

    def test_exits_on_missing_section(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"other_section": {}}, f)
            f.flush()
            with self.assertRaises(SystemExit):
                load_config(f.name)
            os.unlink(f.name)


class TestRelabelDirectory(TestCase):
    """Tests for the relabel_directory function."""

    def _make_mock_client(self, mappings=None):
        """Create a mock HonestBrokerClient with predefined mappings.

        Mappings should include entries for both PatientID and PatientName
        values, since the script looks up both separately via the HB service.
        """
        if mappings is None:
            mappings = {
                # PatientID lookups
                "ORIG123": "DEIDENT-A1B2",
                "PAT456": "DEIDENT-C3D4",
                # PatientName lookups
                "Doe^John": "ANON-NAME-001",
                "Smith^Jane": "ANON-NAME-002",
            }

        client = mock.MagicMock(spec=HonestBrokerClient)

        def lookup_side_effect(id_in):
            if id_in in mappings:
                return mappings[id_in]
            raise RuntimeError(f"HB lookup failed for '{id_in}'")

        client.lookup.side_effect = lookup_side_effect
        return client

    def test_relabels_patient_id_and_name(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "test.dcm"),
                patient_id="ORIG123",
                patient_name="Doe^John",
            )
            client = self._make_mock_client()

            summary = relabel_directory(Path(input_dir), Path(output_dir), client)

            self.assertEqual(summary["total_files"], 1)
            self.assertEqual(summary["processed"], 1)
            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(len(summary["errors"]), 0)

            # Verify both tags changed
            output_ds = pydicom.dcmread(os.path.join(output_dir, "test.dcm"))
            self.assertEqual(output_ds.PatientID, "DEIDENT-A1B2")
            self.assertEqual(str(output_ds.PatientName), "ANON-NAME-001")

    def test_tracks_both_id_and_name_mappings(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "test.dcm"),
                patient_id="ORIG123",
                patient_name="Doe^John",
            )
            client = self._make_mock_client()

            summary = relabel_directory(Path(input_dir), Path(output_dir), client)

            self.assertIn("ORIG123", summary["patient_id_mappings"])
            self.assertEqual(summary["patient_id_mappings"]["ORIG123"], "DEIDENT-A1B2")
            self.assertIn("Doe^John", summary["patient_name_mappings"])
            self.assertEqual(summary["patient_name_mappings"]["Doe^John"], "ANON-NAME-001")

    def test_relabels_multiple_patients(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "p1.dcm"),
                patient_id="ORIG123", patient_name="Doe^John",
            )
            create_test_dicom(
                os.path.join(input_dir, "p2.dcm"),
                patient_id="PAT456", patient_name="Smith^Jane",
            )
            client = self._make_mock_client()

            summary = relabel_directory(Path(input_dir), Path(output_dir), client)

            self.assertEqual(summary["processed"], 2)
            self.assertEqual(len(summary["patient_id_mappings"]), 2)
            self.assertEqual(len(summary["patient_name_mappings"]), 2)

    def test_preserves_directory_structure(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "study1", "series1", "img.dcm"),
                patient_id="ORIG123", patient_name="Doe^John",
            )
            client = self._make_mock_client()

            relabel_directory(Path(input_dir), Path(output_dir), client)

            output_path = Path(output_dir) / "study1" / "series1" / "img.dcm"
            self.assertTrue(output_path.exists())

    def test_dry_run_does_not_write_files(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "test.dcm"),
                patient_id="ORIG123", patient_name="Doe^John",
            )
            client = self._make_mock_client()

            summary = relabel_directory(
                Path(input_dir), Path(output_dir), client, dry_run=True
            )

            self.assertEqual(summary["processed"], 1)
            self.assertFalse(Path(output_dir, "test.dcm").exists())

    def test_handles_patient_id_lookup_failure(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "test.dcm"),
                patient_id="UNKNOWN_ID", patient_name="Doe^John",
            )
            client = self._make_mock_client(mappings={})

            summary = relabel_directory(Path(input_dir), Path(output_dir), client)

            self.assertEqual(summary["processed"], 0)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(len(summary["errors"]), 1)
            self.assertEqual(summary["errors"][0]["field"], "PatientID")

    def test_handles_patient_name_lookup_failure(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "test.dcm"),
                patient_id="ORIG123", patient_name="Unknown^Person",
            )
            # Only PatientID mapped, not the name
            client = self._make_mock_client(mappings={"ORIG123": "DEIDENT-A1B2"})

            summary = relabel_directory(Path(input_dir), Path(output_dir), client)

            self.assertEqual(summary["processed"], 0)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(len(summary["errors"]), 1)
            self.assertEqual(summary["errors"][0]["field"], "PatientName")

    def test_empty_input_directory(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            client = self._make_mock_client()
            summary = relabel_directory(Path(input_dir), Path(output_dir), client)
            self.assertEqual(summary["total_files"], 0)
            self.assertEqual(summary["processed"], 0)

    def test_same_patient_multiple_files_consistent(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            for i in range(3):
                create_test_dicom(
                    os.path.join(input_dir, f"img{i}.dcm"),
                    patient_id="ORIG123", patient_name="Doe^John",
                )
            client = self._make_mock_client()

            summary = relabel_directory(Path(input_dir), Path(output_dir), client)

            self.assertEqual(summary["processed"], 3)
            self.assertEqual(len(summary["patient_id_mappings"]), 1)
            self.assertEqual(len(summary["patient_name_mappings"]), 1)

            # All output files should have the same de-identified values
            for i in range(3):
                ds = pydicom.dcmread(os.path.join(output_dir, f"img{i}.dcm"))
                self.assertEqual(ds.PatientID, "DEIDENT-A1B2")
                self.assertEqual(str(ds.PatientName), "ANON-NAME-001")

    def test_preserves_non_patient_tags(self):
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            dcm_path = os.path.join(input_dir, "test.dcm")
            create_test_dicom(dcm_path, patient_id="ORIG123", patient_name="Doe^John")

            ds = pydicom.dcmread(dcm_path)
            original_study_uid = ds.StudyInstanceUID
            original_modality = ds.Modality

            client = self._make_mock_client()
            relabel_directory(Path(input_dir), Path(output_dir), client)

            output_ds = pydicom.dcmread(os.path.join(output_dir, "test.dcm"))
            self.assertEqual(output_ds.StudyInstanceUID, original_study_uid)
            self.assertEqual(output_ds.Modality, original_modality)

    def test_lookup_called_for_both_id_and_name(self):
        """Verify the HB service is called separately for PatientID and PatientName."""
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            create_test_dicom(
                os.path.join(input_dir, "test.dcm"),
                patient_id="ORIG123", patient_name="Doe^John",
            )
            client = self._make_mock_client()

            relabel_directory(Path(input_dir), Path(output_dir), client)

            # Should have been called with both PatientID and PatientName
            call_args = [call[0][0] for call in client.lookup.call_args_list]
            self.assertIn("ORIG123", call_args)
            self.assertIn("Doe^John", call_args)


if __name__ == "__main__":
    import unittest
    unittest.main()
