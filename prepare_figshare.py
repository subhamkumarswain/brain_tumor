import argparse
import hashlib
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import h5py
from PIL import Image


ARCHIVES = (
    (
        "brainTumorDataPublic_1-766.zip",
        "https://ndownloader.figshare.com/files/3381290",
        "74b949ad33f042e6e103523091cd1428",
    ),
    (
        "brainTumorDataPublic_1533-2298.zip",
        "https://ndownloader.figshare.com/files/3381293",
        "8227bf6080cb71f15a88be8d25c79ae7",
    ),
    (
        "brainTumorDataPublic_767-1532.zip",
        "https://ndownloader.figshare.com/files/3381296",
        "7e8a875500d2c8a346f270538e29890e",
    ),
    (
        "brainTumorDataPublic_2299-3064.zip",
        "https://ndownloader.figshare.com/files/3381302",
        "b378a80d6174e5317d59eb28430c6652",
    ),
)
CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(name, url, expected_md5, downloads_dir):
    archive_path = downloads_dir / name
    if archive_path.exists() and md5_file(archive_path) == expected_md5:
        print(f"Verified existing archive: {name}")
        return archive_path

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    print(f"Downloading {name}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial_path.open("wb") as target:
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    print(f"  downloaded {downloaded / (1024 * 1024):.0f} MB")
        actual_md5 = md5_file(partial_path)
        if actual_md5 != expected_md5:
            raise ValueError(f"Checksum mismatch for {name}: {actual_md5}")
        partial_path.replace(archive_path)
    finally:
        if partial_path.exists():
            partial_path.unlink()
    return archive_path


def read_mat_metadata(path):
    with h5py.File(path, "r") as source:
        data = source["cjdata"]
        label = int(np.asarray(data["label"][()]).item())
        patient_values = np.asarray(data["PID"][()]).reshape(-1)
    if patient_values.dtype.kind in "ui":
        patient_id = "".join(chr(int(value)) for value in patient_values).rstrip("\x00")
    else:
        patient_id = "".join(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in patient_values
        ).rstrip("\x00")
    if label not in CLASS_NAMES:
        raise ValueError(f"Unknown class label {label} in {path.name}.")
    return label, patient_id


def assign_patient_splits(records, seed):
    patients_by_class = {label: set() for label in CLASS_NAMES}
    patient_labels = {}
    for record in records:
        previous_label = patient_labels.setdefault(record["patient_id"], record["label"])
        if previous_label != record["label"]:
            raise ValueError(f"Patient {record['patient_id']} has inconsistent class labels.")
        patients_by_class[record["label"]].add(record["patient_id"])

    randomizer = random.Random(seed)
    patient_splits = {}
    for label, patient_ids in patients_by_class.items():
        ordered = sorted(patient_ids)
        randomizer.shuffle(ordered)
        if len(ordered) < 3:
            raise ValueError(f"Not enough patients to split class {CLASS_NAMES[label]}.")
        test_count = max(1, round(len(ordered) * 0.15))
        val_count = max(1, round(len(ordered) * 0.15))
        if test_count + val_count >= len(ordered):
            raise ValueError(f"Not enough patients to split class {CLASS_NAMES[label]}.")
        for patient_id in ordered[:test_count]:
            patient_splits[patient_id] = "test"
        for patient_id in ordered[test_count : test_count + val_count]:
            patient_splits[patient_id] = "val"
        for patient_id in ordered[test_count + val_count :]:
            patient_splits[patient_id] = "train"
    return patient_splits


def save_mri_image(mat_path, destination):
    with h5py.File(mat_path, "r") as source:
        image = np.asarray(source["cjdata"]["image"][()], dtype=np.float32)
    low = float(image.min())
    high = float(image.max())
    if high > low:
        image = (image - low) * (255.0 / (high - low))
    else:
        image = np.zeros_like(image)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(destination)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the public Figshare brain tumor MRI dataset."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    downloads_dir = args.output_dir / "downloads"
    raw_dir = args.output_dir / "raw_mat"
    splits_dir = args.output_dir / f"splits_seed{args.seed}"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, url, expected_md5 in ARCHIVES:
        archive_path = download_archive(name, url, expected_md5, downloads_dir)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if Path(member.filename).name != member.filename or not member.filename.lower().endswith(".mat"):
                    raise ValueError(f"Unexpected archive member: {member.filename}")
                destination = raw_dir / member.filename
                if destination.exists():
                    continue
                partial_path = destination.with_suffix(destination.suffix + ".part")
                with archive.open(member) as source, partial_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                partial_path.replace(destination)

    mat_files = sorted(raw_dir.glob("*.mat"))
    if len(mat_files) != 3064:
        raise ValueError(f"Expected 3064 MRI files, found {len(mat_files)} in {raw_dir}.")

    records = []
    for mat_path in mat_files:
        label, patient_id = read_mat_metadata(mat_path)
        records.append({"path": mat_path, "label": label, "patient_id": patient_id})
    patient_splits = assign_patient_splits(records, args.seed)

    split_patients = {name: set() for name in ("train", "val", "test")}
    split_counts = {name: {class_name: 0 for class_name in CLASS_NAMES.values()} for name in split_patients}
    for record in records:
        split_name = patient_splits[record["patient_id"]]
        class_name = CLASS_NAMES[record["label"]]
        destination_dir = splits_dir / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        save_mri_image(record["path"], destination_dir / f"{record['path'].stem}.png")
        split_patients[split_name].add(record["patient_id"])
        split_counts[split_name][class_name] += 1

    if (split_patients["train"] & split_patients["val"]
            or split_patients["train"] & split_patients["test"]
            or split_patients["val"] & split_patients["test"]):
        raise RuntimeError("Patient overlap detected between dataset splits.")

    print("Prepared image counts by split and class:")
    for split_name, class_counts in split_counts.items():
        print(f"{split_name}: {class_counts}; patients={len(split_patients[split_name])}")
    print(f"Training data directory: {splits_dir}")


if __name__ == "__main__":
    main()