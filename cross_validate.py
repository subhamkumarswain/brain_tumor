import argparse
import csv
import json
import random
import time
from pathlib import Path
from statistics import mean, stdev

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


MODEL_NAMES = ("resnet18", "efficientnet_b0", "densenet121")


class MriDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare frozen CNN feature extractors with patient-level cross-validation."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs_cv"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_patient_id(mat_path):
    with h5py.File(mat_path, "r") as source:
        values = np.asarray(source["cjdata"]["PID"][()]).reshape(-1)
    if values.dtype.kind in "ui":
        return "".join(chr(int(value)) for value in values).rstrip("\x00")
    return "".join(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ).rstrip("\x00")


def gather_samples(data_dir, raw_dir):
    class_names = sorted(path.name for path in (data_dir / "train").iterdir() if path.is_dir())
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    samples = []
    labels = []
    groups = []
    patient_labels = {}
    missing_metadata = []

    for split in ("train", "val"):
        for class_name, target in class_to_index.items():
            class_dir = data_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                mat_path = raw_dir / f"{image_path.stem}.mat"
                if not mat_path.is_file():
                    missing_metadata.append(str(mat_path))
                    continue
                patient_id = read_patient_id(mat_path)
                previous_label = patient_labels.setdefault(patient_id, target)
                if previous_label != target:
                    raise ValueError(f"Patient {patient_id} appears in multiple classes.")
                samples.append((image_path, target))
                labels.append(target)
                groups.append(patient_id)

    if missing_metadata:
        raise FileNotFoundError(f"Missing patient metadata for {len(missing_metadata)} images.")
    if not samples:
        raise ValueError("No training or validation images found.")
    return samples, class_names, np.asarray(labels), np.asarray(groups)


def make_feature_model(name):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        feature_count = 512
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()
        feature_count = 1280
    else:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Identity()
        feature_count = 1024
    return model, feature_count


def extract_features(name, samples, transform, args, device):
    model, feature_count = make_feature_model(name)
    model = model.to(device).eval()
    loader = DataLoader(
        MriDataset(samples, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    all_features = []
    all_labels = []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            features = model(images.to(device, non_blocking=True))
            all_features.append(features.reshape(features.size(0), -1).cpu())
            all_labels.append(labels)
    elapsed = time.perf_counter() - started
    features = torch.cat(all_features)
    labels = torch.cat(all_labels).long()
    if features.shape[1] != feature_count:
        raise RuntimeError(f"Unexpected feature size for {name}: {features.shape[1]}")
    print(f"{name} | extracted {len(labels)} feature vectors in {elapsed:.1f}s")
    return features, labels


def evaluate_folds(features, labels, groups, args, device):
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_results = []
    indices = np.arange(len(labels))

    for fold_number, (train_indices, val_indices) in enumerate(
        splitter.split(indices, labels.numpy(), groups), start=1
    ):
        train_groups = set(groups[train_indices])
        val_groups = set(groups[val_indices])
        if train_groups & val_groups:
            raise RuntimeError(f"Patient leakage detected in fold {fold_number}.")

        train_index_tensor = torch.as_tensor(train_indices, dtype=torch.long)
        val_index_tensor = torch.as_tensor(val_indices, dtype=torch.long)
        train_features = features.index_select(0, train_index_tensor).to(device)
        train_labels = labels.index_select(0, train_index_tensor).to(device)
        val_features = features.index_select(0, val_index_tensor).to(device)
        val_labels = labels.index_select(0, val_index_tensor).to(device)

        torch.manual_seed(args.seed + fold_number)
        classifier = nn.Linear(features.shape[1], int(labels.max().item()) + 1).to(device)
        class_counts = torch.bincount(train_labels, minlength=classifier.out_features).float()
        class_weights = class_counts.sum() / (
            len(class_counts) * class_counts.clamp_min(1)
        )
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )

        for _ in range(args.head_epochs):
            classifier.train()
            permutation = torch.randperm(len(train_labels), device=device)
            for start in range(0, len(permutation), args.batch_size):
                batch_indices = permutation[start : start + args.batch_size]
                logits = classifier(train_features.index_select(0, batch_indices))
                loss = loss_function(logits, train_labels.index_select(0, batch_indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        classifier.eval()
        with torch.inference_mode():
            predictions = classifier(val_features).argmax(dim=1)
        y_true = val_labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        result = {
            "fold": fold_number,
            "validation_patients": len(val_groups),
            "validation_images": len(val_indices),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        fold_results.append(result)
        print(
            f"  fold {fold_number}/{args.folds} | patients {len(val_groups)} | "
            f"accuracy {result['accuracy']:.4f} | macro-F1 {result['macro_f1']:.4f}"
        )

    f1_scores = [result["macro_f1"] for result in fold_results]
    accuracies = [result["accuracy"] for result in fold_results]
    return {
        "folds": fold_results,
        "mean_macro_f1": mean(f1_scores),
        "std_macro_f1": stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
        "mean_accuracy": mean(accuracies),
        "std_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0.0,
    }


def main():
    args = parse_args()
    if args.folds < 2 or args.head_epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("folds must be >= 2; epochs, batch-size, and cpu-threads must be positive.")
    if not args.data_dir.is_dir() or not args.raw_dir.is_dir():
        raise FileNotFoundError("Both --data-dir and --raw-dir must point to existing directories.")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, class_names, labels, groups = gather_samples(args.data_dir, args.raw_dir)
    patient_counts = {
        class_name: len({groups[index] for index, label in enumerate(labels) if label == class_index})
        for class_index, class_name in enumerate(class_names)
    }
    if min(patient_counts.values()) < args.folds:
        raise ValueError(f"Each class needs at least {args.folds} patients; found {patient_counts}.")

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    print(
        f"Device: {device}; images: {len(samples)}; patients: {len(set(groups))}; "
        f"class patients: {patient_counts}; folds: {args.folds}"
    )
    print("Using only train/validation patients; test patients are excluded.")

    results = {
        "protocol": "StratifiedGroupKFold; frozen ImageNet CNN features; train/validation patients only",
        "class_names": class_names,
        "patient_counts_by_class": patient_counts,
        "fold_count": args.folds,
        "models": {},
    }
    for name in args.models:
        features, feature_labels = extract_features(name, samples, transform, args, device)
        if not torch.equal(feature_labels, torch.as_tensor(labels)):
            raise RuntimeError(f"ImageFolder label order changed during {name} feature extraction.")
        results["models"][name] = evaluate_folds(
            features, feature_labels, groups, args, device
        )

    summary = sorted(
        args.models,
        key=lambda name: results["models"][name]["mean_macro_f1"],
        reverse=True,
    )
    results["ranked_by_mean_macro_f1"] = summary
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=("model", "mean_macro_f1", "std_macro_f1", "mean_accuracy", "std_accuracy"),
        )
        writer.writeheader()
        for name in summary:
            model_result = results["models"][name]
            writer.writerow(
                {
                    "model": name,
                    "mean_macro_f1": model_result["mean_macro_f1"],
                    "std_macro_f1": model_result["std_macro_f1"],
                    "mean_accuracy": model_result["mean_accuracy"],
                    "std_accuracy": model_result["std_accuracy"],
                }
            )

    print("\nFive-fold ranking by mean macro-F1:")
    for rank, name in enumerate(summary, start=1):
        model_result = results["models"][name]
        print(
            f"{rank}. {name}: macro-F1={model_result['mean_macro_f1']:.4f} "
            f"+/- {model_result['std_macro_f1']:.4f}; "
            f"accuracy={model_result['mean_accuracy']:.4f} "
            f"+/- {model_result['std_accuracy']:.4f}"
        )
    print(f"Detailed results: {results_path}")


if __name__ == "__main__":
    main()