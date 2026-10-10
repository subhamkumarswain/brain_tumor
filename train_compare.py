import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MODEL_NAMES = ("resnet18", "efficientnet_b0", "densenet121")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare pretrained CNNs for brain MRI tumor classification."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--initial-best-epoch", type=int)
    return parser.parse_args()


def make_model(name, class_count):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, class_count)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    else:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, class_count)
    return model


def make_loaders(data_dir, image_size, batch_size, num_workers, device):
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalization]
    )

    datasets_by_split = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_dir / "val", transform=evaluation_transform),
        "test": datasets.ImageFolder(data_dir / "test", transform=evaluation_transform),
    }
    classes = datasets_by_split["train"].classes
    for split_name, image_dataset in datasets_by_split.items():
        if not image_dataset.samples:
            raise ValueError(f"No images found in {data_dir / split_name}.")
        if image_dataset.class_to_idx != datasets_by_split["train"].class_to_idx:
            raise ValueError(f"Class folders in {split_name} do not match the train split.")

    loaders = {
        split_name: DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        for split_name, image_dataset in datasets_by_split.items()
    }
    train_counts = torch.bincount(
        torch.tensor(datasets_by_split["train"].targets), minlength=len(classes)
    ).float()
    class_weights = train_counts.sum() / (len(classes) * train_counts.clamp_min(1))
    return loaders, classes, class_weights.to(device)


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all = []
    predictions_all = []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_function(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets_all.extend(targets.detach().cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets_all, predictions_all),
        "macro_f1": f1_score(targets_all, predictions_all, average="macro"),
        "targets": targets_all,
        "predictions": predictions_all,
    }


def train_model(name, loaders, classes, class_weights, args, device):
    model = make_model(name, len(classes)).to(device)
    starting_epoch = args.initial_best_epoch or 0
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
        if checkpoint["model_name"] != name or checkpoint["classes"] != classes:
            raise ValueError("Initial checkpoint model or class order does not match this run.")
        model.load_state_dict(checkpoint["state_dict"])
        if args.initial_best_epoch is not None:
            starting_epoch = args.initial_best_epoch
        else:
            starting_epoch = checkpoint.get("best_epoch", 0)
    starting_epoch = args.initial_best_epoch or 0
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_score = -1.0
    best_epoch = starting_epoch
    best_state = None
    if args.initial_checkpoint is not None:
        initial_metrics = run_epoch(model, loaders["val"], loss_function, device)
        best_score = initial_metrics["macro_f1"]
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        print(f"{name} | checkpoint validation macro-F1 {best_score:.4f}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, loaders["train"], loss_function, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], loss_function, device)
        elapsed = time.perf_counter() - started
        print(
            f"{name} | epoch {epoch}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val accuracy {val_metrics['accuracy']:.4f} | "
            f"val macro-F1 {val_metrics['macro_f1']:.4f} | {elapsed:.1f}s"
        )
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = starting_epoch + epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_name": name,
                    "classes": classes,
                    "best_epoch": starting_epoch + epoch,
                    "state_dict": best_state,
                },
                args.output_dir / f"{name}_best.pt",
            )

    model.load_state_dict(best_state)
    checkpoint_path = args.output_dir / f"{name}_best.pt"
    torch.save(
        {
            "model_name": name,
            "classes": classes,
            "best_epoch": best_epoch,
            "state_dict": best_state,
        },
        checkpoint_path,
    )
    return model, best_epoch, best_score, checkpoint_path


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1 or args.cpu_threads < 1:
        raise ValueError("epochs, batch-size, image-size, and cpu-threads must be positive.")
    if args.initial_checkpoint is not None and len(args.models) != 1:
        raise ValueError("--initial-checkpoint requires exactly one model in --models.")
    if args.initial_checkpoint is not None and not args.initial_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.initial_checkpoint}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders, classes, class_weights = make_loaders(
        args.data_dir, args.image_size, args.batch_size, args.num_workers, device
    )
    print(f"Device: {device}; classes: {', '.join(classes)}")
    print("Models: " + ", ".join(args.models))
    print("Test evaluation is disabled." if args.skip_test else "Test evaluation is enabled.")

    results = {}
    trained_models = {}
    for name in args.models:
        model, best_epoch, best_val_f1, checkpoint_path = train_model(
            name, loaders, classes, class_weights, args, device
        )
        trained_models[name] = model
        results[name] = {
            "best_epoch": best_epoch,
            "validation_macro_f1": best_val_f1,
            "checkpoint": str(checkpoint_path),
        }

    selected_name = max(
        args.models, key=lambda model_name: results[model_name]["validation_macro_f1"]
    )
    if not args.skip_test:
        for name, model in trained_models.items():
            loss_function = nn.CrossEntropyLoss(weight=class_weights)
            test_metrics = run_epoch(model, loaders["test"], loss_function, device)
            results[name]["test"] = {
                "loss": test_metrics["loss"],
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "classification_report": classification_report(
                    test_metrics["targets"],
                    test_metrics["predictions"],
                    labels=list(range(len(classes))),
                    target_names=classes,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    test_metrics["targets"],
                    test_metrics["predictions"],
                    labels=list(range(len(classes))),
                ).tolist(),
            }

    results["selected_model"] = selected_name
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    comparison_path = args.output_dir / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=("model", "best_epoch", "validation_macro_f1", "test_accuracy", "test_macro_f1"),
        )
        writer.writeheader()
        for name in args.models:
            writer.writerow(
                {
                    "model": name,
                    "best_epoch": results[name]["best_epoch"],
                    "validation_macro_f1": results[name]["validation_macro_f1"],
                    "test_accuracy": results[name].get("test", {}).get("accuracy", ""),
                    "test_macro_f1": results[name].get("test", {}).get("macro_f1", ""),
                }
            )

    if not args.skip_test:
        print("\nTest-set metrics:")
        for name in args.models:
            test_metrics = results[name]["test"]
            print(
                f"{name}: accuracy={test_metrics['accuracy']:.4f}, "
                f"macro-F1={test_metrics['macro_f1']:.4f}"
            )
    print(f"Best validation macro-F1: {selected_name} ({results[selected_name]['validation_macro_f1']:.4f})")
    print(f"Detailed results: {results_path}")


if __name__ == "__main__":
    main()