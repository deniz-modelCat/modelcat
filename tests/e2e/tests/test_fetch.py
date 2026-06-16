"""
E2E tests for modelcat_fetch and fetch-to-validate workflow.

Uses real CLI subprocess invocations. Roboflow download is not exercised here;
formatted fetch output is built offline and validated end-to-end.
"""
import os

import pytest

ROBOFLOW_URL = (
    "https://universe.roboflow.com/test-workspace/test-project/dataset/1"
)


@pytest.mark.e2e
class TestFetchCLI:
    def test_missing_url_argument(self, cli):
        """Fetch without --url should exit with argparse error."""
        result = cli.run_fetch_raw(["/tmp/some-path"])
        assert result.exit_code == 2
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_invalid_roboflow_url(self, cli):
        """Invalid Roboflow URL should exit with code 1 and a clear error."""
        result = cli.fetch("/tmp/unused-path", url="https://example.com/not-roboflow")
        assert result.exit_code == 1
        assert result.output_contains("Invalid Roboflow Universe URL")

    def test_non_empty_save_path_rejected(self, cli, tmp_path):
        """Fetch should refuse to write into a non-empty directory."""
        dest = tmp_path / "existing"
        dest.mkdir()
        (dest / "already_here.txt").write_text("data")

        env = os.environ.copy()
        env["ROBOFLOW_API_KEY"] = "fake-key-for-e2e"

        result = cli.fetch(
            str(dest),
            url=ROBOFLOW_URL,
            env=env,
        )
        assert result.exit_code == 1
        assert result.output_contains("not empty")


@pytest.mark.e2e
class TestFetchValidationFlow:
    def test_fetched_dataset_passes_validation(self, cli, fetched_detection_ds):
        """A fetch-formatted dataset should pass modelcat_validate."""
        result = cli.validate(fetched_detection_ds.path)
        assert result.exit_code == 0
        assert result.passed
        assert result.signature is not None
        assert result.error_count == 0

    def test_fetch_output_has_modelcat_structure(self, fetched_detection_ds):
        """Fetch-formatted dataset should have the expected directory layout."""
        root = fetched_detection_ds.path
        assert os.path.isdir(os.path.join(root, "annotations"))
        assert os.path.isdir(os.path.join(root, "images", "train"))
        assert os.path.isdir(os.path.join(root, "images", "validation"))
        assert os.path.isdir(os.path.join(root, "images", "test"))
        assert os.path.isfile(os.path.join(root, "dataset_infos.json"))
        assert os.path.isfile(os.path.join(root, "annotations", "coco_train.json"))

    def test_validate_shows_fetch_next_step_hint_after_format(self, cli, fetched_detection_ds):
        """Validation of fetch output should produce a signature (ready for upload)."""
        result = cli.validate(fetched_detection_ds.path)
        assert result.stdout_contains("Validation passed and signed")
