import os
import os.path as osp
import shutil
import tempfile

from modelcat.connector.fetch import (
    clean_supercategories,
    format_for_modelcat,
    generate_dataset_infos,
)
from tests.fixtures.roboflow_export import build_minimal_roboflow_export


class FetchDatasetBuilder:
    """
    Builds a ModelCat-formatted dataset using the same pipeline as modelcat_fetch,
    without calling the Roboflow API.
    """

    def __init__(self, tmp_dir: str, project_name: str = "fetch-e2e-project"):
        self._path = osp.join(tmp_dir, project_name)
        self._project_name = project_name
        self._task_type = "detection"

    @property
    def path(self) -> str:
        return self._path

    @property
    def project_name(self) -> str:
        return self._project_name

    def build_detection_dataset(self, license_name: str = "MIT") -> "FetchDatasetBuilder":
        rf_dir = tempfile.mkdtemp(dir=osp.dirname(self._path))
        try:
            build_minimal_roboflow_export(
                rf_dir, license_name=license_name
            )
            clean_supercategories(rf_dir)
            rf_license = format_for_modelcat(rf_dir, self._path, self._project_name)
            generate_dataset_infos(
                self._path, self._project_name, rf_license, self._task_type
            )
        finally:
            if osp.exists(rf_dir):
                shutil.rmtree(rf_dir)
        return self
