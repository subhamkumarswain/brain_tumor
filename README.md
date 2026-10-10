# Brain Tumor MRI Classification

This project compares three ImageNet-pretrained CNNs using the same data splits:

- ResNet-18
- EfficientNet-B0
- DenseNet-121

The model selected for follow-up is the one with the highest validation macro-F1. Each model is then evaluated on the test split; the test set is not used to select the model.

## Dataset

The preparation script uses the public [Cheng brain tumor dataset on Figshare](https://doi.org/10.6084/m9.figshare.1512427.v8), licensed CC BY 4.0. It contains 3,064 contrast-enhanced MRI slices from 233 patients across three classes: meningioma, glioma, and pituitary tumor. The download is about 880 MB. The script verifies the published archive checksums and creates patient-level, class-stratified splits so slices from one patient cannot cross split boundaries.

Prepare the data once:

```powershell
python -m pip install -r requirements.txt
python prepare_figshare.py
```

This creates `data/splits_seed42/train`, `data/splits_seed42/val`, and `data/splits_seed42/test`. The downloaded archives and extracted MATLAB files remain under `data/downloads` and `data/raw_mat`.

## Dataset layout

The trainer expects class-named folders under each split. The class folder names must match across splits. `prepare_figshare.py` creates the following layout:

```text
data/splits_seed42/
  train/
    glioma/
    meningioma/
    pituitary/
  val/
    glioma/
    meningioma/
    pituitary/
  test/
    glioma/
    meningioma/
    pituitary/
```

For a different dataset, arrange image files in the same layout and ensure that patient IDs do not occur in multiple splits.

## Setup and run

Create and activate a virtual environment, then install PyTorch and torchvision for your machine. For CUDA, use the installation command recommended by the official PyTorch selector; otherwise install the CPU builds. Then install the project requirements:

```powershell
python -m pip install -r requirements.txt
```

Run from this project directory after preparing the data:

```powershell
python train_compare.py --data-dir .\data\splits_seed42 --epochs 10 --batch-size 32
```

The preparation step downloads about 880 MB and requires an internet connection. The first training run also downloads pretrained weights. Training runs on CUDA automatically when available; use `--device cpu` to force CPU. For a small dataset or limited memory, lower `--batch-size`.

## Outputs

The `runs/` directory contains one best-validation checkpoint per model, `comparison.csv` with the headline comparison, and `results.json` with per-class metrics and confusion matrices. The selected model is chosen by validation macro-F1, which is more informative than accuracy when class sizes differ.

To continue fine-tuning one model from a checkpoint using validation only, pass `--models`, `--initial-checkpoint`, and `--skip-test`. For example:

```powershell
python train_compare.py --data-dir .\data\splits_seed42 --output-dir .\runs_densenet_tuned --models densenet121 --initial-checkpoint .\runs_initial\densenet121_best.pt --initial-best-epoch 3 --epochs 1 --batch-size 16 --image-size 128 --learning-rate 0.00002 --skip-test
```

## Initial results

The screening run used 3 epochs, 128-pixel images, and a patient-level split. Scores are macro-F1.

| Model | Validation | Test |
|---|---:|---:|
| ResNet-18 | 0.8791 | 0.8987 |
| EfficientNet-B0 | 0.8868 | 0.8830 |
| DenseNet-121 | 0.9149 | 0.8634 |

DenseNet-121 was selected using validation macro-F1. One additional validation-only epoch raised its validation macro-F1 to 0.9341; the next epoch scored 0.9328, so the earlier checkpoint was retained. The initial test scores are exploratory because all three models were evaluated on that test split. Do not use them as an unbiased final performance estimate; final reporting needs a fresh, untouched test set or external dataset.

## Patient-level cross-validation

The follow-up comparison uses five-fold stratified group cross-validation on the 199 train/validation patients. The 34 test patients are excluded. To keep CPU runtime practical, this experiment freezes each ImageNet-pretrained CNN and trains only a linear classification head:

```powershell
python cross_validate.py --data-dir .\data\splits_seed42 --raw-dir .\data\raw_mat --output-dir .\runs_cv --folds 5 --head-epochs 20 --batch-size 32 --image-size 128
```

| Frozen feature extractor | Mean macro-F1 | Fold standard deviation | Mean accuracy |
|---|---:|---:|---:|
| EfficientNet-B0 | **0.8605** | 0.0621 | 0.8737 |
| ResNet-18 | 0.8231 | 0.0623 | 0.8361 |
| DenseNet-121 | 0.8208 | 0.0569 | 0.8330 |

EfficientNet-B0 is the current candidate based on this CV protocol. A separate three-epoch full fine-tuning run selected its epoch-2 checkpoint at validation macro-F1 0.8549. This single validation score is not directly comparable to the cross-validation mean. The candidate checkpoint is `runs_efficientnet_candidate/efficientnet_b0_best.pt`; fold-level results are in `runs_cv/results.json`.

Dataset citation: Cheng, Jun (2017). *brain tumor dataset*. Figshare. https://doi.org/10.6084/m9.figshare.1512427.v8. License: CC BY 4.0.

The local PyTorch build used for these runs is CPU-only; the full initial comparison took about 20 minutes. Use a CUDA-enabled PyTorch build or reduce input size and batch size for faster experiments.

This is an experimental classification baseline, not a clinically validated diagnostic system. Performance depends on dataset quality, split design, scanner/site variation, and independent external validation.