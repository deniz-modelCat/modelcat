import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from modelcat.connector.fetch import (
    clean_supercategories,
    fetch_cli,
    format_for_modelcat,
    generate_dataset_infos,
    parse_roboflow_url,
)
from tests.fixtures.roboflow_export import build_minimal_roboflow_export


class TestParseRoboflowUrl(unittest.TestCase):
    def test_parses_url_with_version(self):
        url = "https://universe.roboflow.com/my-workspace/my-project/dataset/3"
        workspace, project, version = parse_roboflow_url(url)
        self.assertEqual(workspace, "my-workspace")
        self.assertEqual(project, "my-project")
        self.assertEqual(version, 3)

    def test_parses_url_without_version(self):
        url = "https://universe.roboflow.com/my-workspace/my-project"
        workspace, project, version = parse_roboflow_url(url)
        self.assertEqual(workspace, "my-workspace")
        self.assertEqual(project, "my-project")
        self.assertIsNone(version)

    def test_parses_model_path_variant(self):
        url = "https://universe.roboflow.com/ws/proj/model/2"
        workspace, project, version = parse_roboflow_url(url)
        self.assertEqual((workspace, project, version), ("ws", "proj", 2))

    def test_rejects_invalid_url(self):
        with self.assertRaises(ValueError):
            parse_roboflow_url("https://example.com/not-roboflow")


class TestCleanSupercategories(unittest.TestCase):
    def test_removes_empty_categories_and_remaps_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = os.path.join(tmp, "train")
            os.makedirs(split_dir)
            coco = {
                "categories": [
                    {"id": 0, "name": "car", "supercategory": "vehicle"},
                    {"id": 1, "name": "empty", "supercategory": "none"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 0},
                ],
            }
            ann_path = os.path.join(split_dir, "_annotations.coco.json")
            with open(ann_path, "w") as f:
                json.dump(coco, f)

            clean_supercategories(tmp)

            with open(ann_path) as f:
                result = json.load(f)

            self.assertEqual(len(result["categories"]), 1)
            self.assertEqual(result["categories"][0]["name"], "car")
            self.assertEqual(result["categories"][0]["id"], 0)
            self.assertEqual(result["categories"][0]["supercategory"], "none")
            self.assertEqual(result["annotations"][0]["category_id"], 0)


class TestFormatForModelcat(unittest.TestCase):
    def test_converts_roboflow_layout_to_modelcat_layout(self):
        with tempfile.TemporaryDirectory() as rf_dir, tempfile.TemporaryDirectory() as dest:
            build_minimal_roboflow_export(rf_dir, splits=["train", "valid", "test"])

            license_info = format_for_modelcat(rf_dir, dest, "my-project")

            self.assertIsNotNone(license_info)
            self.assertTrue(os.path.isdir(os.path.join(dest, "annotations")))
            self.assertTrue(os.path.isdir(os.path.join(dest, "images", "train")))
            self.assertTrue(os.path.isdir(os.path.join(dest, "images", "validation")))
            self.assertTrue(os.path.isdir(os.path.join(dest, "images", "test")))

            with open(os.path.join(dest, "annotations", "coco_train.json")) as f:
                train_coco = json.load(f)
            self.assertTrue(train_coco["images"][0]["file_name"].startswith("train/"))
            self.assertEqual(train_coco["categories"][0]["id"], 1)


class TestGenerateDatasetInfos(unittest.TestCase):
    def test_writes_dataset_infos_with_personalized_description(self):
        with tempfile.TemporaryDirectory() as rf_dir, tempfile.TemporaryDirectory() as dest:
            build_minimal_roboflow_export(rf_dir)
            rf_license = format_for_modelcat(rf_dir, dest, "forklift-bounding-box")
            generate_dataset_infos(dest, "forklift-bounding-box", rf_license, "detection")

            infos_path = os.path.join(dest, "dataset_infos.json")
            self.assertTrue(os.path.exists(infos_path))

            with open(infos_path) as f:
                infos = json.load(f)

            dataset = infos["forklift-bounding-box"]
            self.assertIn("Forklift Bounding Box", dataset["description"])
            self.assertIn("detection", dataset["description"])
            self.assertEqual(dataset["task_templates"][0]["task"], "detection")
            self.assertEqual(dataset["task_templates"][0]["labels"], ["car"])
            self.assertEqual(dataset["dataset_size"], 3)
            for split in ["train", "validation", "test"]:
                self.assertIn(split, dataset["splits"])

    def test_warns_on_restrictive_license(self):
        with tempfile.TemporaryDirectory() as rf_dir, tempfile.TemporaryDirectory() as dest:
            build_minimal_roboflow_export(
                rf_dir, license_name="CC BY-NC 4.0"
            )
            rf_license = format_for_modelcat(rf_dir, dest, "my-project")

            with self.assertLogs("root", level="WARNING") as logs:
                generate_dataset_infos(dest, "my-project", rf_license, "detection")

            self.assertTrue(
                any("restrictive license" in msg for msg in logs.output)
            )


class TestFetchPipelineIntegration(unittest.TestCase):
    """Fetch formatting output should pass modelcat_validate."""

    def test_formatted_export_passes_validation(self):
        with tempfile.TemporaryDirectory() as rf_dir, tempfile.TemporaryDirectory() as dest:
            build_minimal_roboflow_export(rf_dir)
            clean_supercategories(rf_dir)
            rf_license = format_for_modelcat(rf_dir, dest, "fetch-test-project")
            generate_dataset_infos(dest, "fetch-test-project", rf_license, "detection")

            proc = subprocess.run(
                [sys.executable, "-m", "modelcat.connector.validate", "-d", dest],
                capture_output=True,
                text=True,
                timeout=120,
            )

            self.assertEqual(
                proc.returncode,
                0,
                msg=f"Validation failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
            self.assertIn("Validation passed and signed", proc.stdout)


class TestFetchCli(unittest.TestCase):
    def _patch_roboflow(self, mock_rf):
        mock_module = MagicMock()
        mock_module.Roboflow.return_value = mock_rf
        return patch.dict(sys.modules, {"roboflow": mock_module})

    def test_rejects_invalid_url(self):
        with patch.object(sys, "argv", ["modelcat_fetch", "/tmp/out", "--url", "bad-url"]):
            with patch("modelcat.connector.fetch.ensure_dependencies"):
                with self.assertRaises(SystemExit) as ctx:
                    fetch_cli()
                self.assertEqual(ctx.exception.code, 1)

    def test_rejects_non_empty_save_path(self):
        with tempfile.TemporaryDirectory() as dest:
            with open(os.path.join(dest, "existing.txt"), "w") as f:
                f.write("data")
            url = "https://universe.roboflow.com/ws/proj/1"
            with patch.object(
                sys, "argv", ["modelcat_fetch", dest, "--url", url]
            ):
                with patch("modelcat.connector.fetch.ensure_dependencies"):
                    with patch("modelcat.connector.fetch.get_api_key", return_value="key"):
                        with self.assertRaises(SystemExit) as ctx:
                            fetch_cli()
                        self.assertEqual(ctx.exception.code, 1)

    def test_rejects_unsupported_task_type(self):
        url = "https://universe.roboflow.com/ws/proj/1"
        mock_project = MagicMock()
        mock_project.type = "instance-segmentation"

        mock_workspace = MagicMock()
        mock_workspace.project.return_value = mock_project

        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_workspace

        with tempfile.TemporaryDirectory() as dest:
            with patch.object(sys, "argv", ["modelcat_fetch", dest, "--url", url]):
                with patch("modelcat.connector.fetch.ensure_dependencies"):
                    with patch("modelcat.connector.fetch.get_api_key", return_value="key"):
                        with self._patch_roboflow(mock_rf):
                            with self.assertRaises(SystemExit) as ctx:
                                fetch_cli()
                            self.assertEqual(ctx.exception.code, 1)

    def test_successful_fetch_formats_dataset(self):
        url = "https://universe.roboflow.com/ws/fetch-test/1"

        def fake_download(fmt, location):
            build_minimal_roboflow_export(location)

        mock_version = MagicMock()
        mock_version.download.side_effect = fake_download

        mock_project = MagicMock()
        mock_project.type = "object-detection"
        mock_project.version.return_value = mock_version

        mock_workspace = MagicMock()
        mock_workspace.project.return_value = mock_project

        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_workspace

        with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as dest:
            old_cwd = os.getcwd()
            try:
                os.chdir(cwd)
                with patch.object(
                    sys, "argv", ["modelcat_fetch", dest, "--url", url]
                ):
                    with patch("modelcat.connector.fetch.ensure_dependencies"):
                        with patch(
                            "modelcat.connector.fetch.get_api_key",
                            return_value="fake-key",
                        ):
                            with self._patch_roboflow(mock_rf):
                                stdout = StringIO()
                                with patch("sys.stdout", stdout):
                                    fetch_cli()

                self.assertTrue(os.path.exists(os.path.join(dest, "dataset_infos.json")))
                self.assertTrue(
                    os.path.exists(os.path.join(dest, "annotations", "coco_train.json"))
                )
                self.assertFalse(os.path.exists(os.path.join(cwd, ".rf_temp_download")))
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
