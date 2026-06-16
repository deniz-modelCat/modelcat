"""Helpers to build minimal Roboflow COCO export fixtures for offline tests."""
import json
import os


def write_minimal_jpeg(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)


def write_roboflow_split(
    base_dir: str,
    split: str,
    image_name: str = "img.jpg",
    category_name: str = "car",
    license_name: str = "MIT",
) -> None:
    """Create one Roboflow split folder with a COCO annotation file and image."""
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    write_minimal_jpeg(os.path.join(split_dir, image_name))

    coco = {
        "licenses": [{"id": 1, "name": license_name, "url": ""}],
        "categories": [
            {"id": 0, "name": category_name, "supercategory": "vehicle"},
        ],
        "images": [
            {
                "id": 1,
                "file_name": image_name,
                "width": 100,
                "height": 100,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }
    with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)


def build_minimal_roboflow_export(
    base_dir: str,
    splits=None,
    category_name: str = "car",
    license_name: str = "MIT",
) -> None:
    """Build a minimal Roboflow COCO export with train/valid/test splits."""
    if splits is None:
        splits = ["train", "valid", "test"]
    for split in splits:
        write_roboflow_split(
            base_dir,
            split,
            image_name=f"{split}_img.jpg",
            category_name=category_name,
            license_name=license_name,
        )
