# ASD Face + EEG Fusion

This repository contains scripts for auditing the datasets, extracting EEG features,
training face and EEG models, and evaluating a class-matched face/EEG fusion demo.
The face and EEG datasets contain different people, so fusion pairs examples by
class; it does **not** represent measurements from the same person and is not a
clinical diagnostic tool.

## Project layout

```text
scripts/       Python source code
data/raw/      downloaded archives and extracted datasets (not committed)
data/reference/ auxiliary metadata from exploratory work (not used by pipeline)
results/       generated manifests, features, models, reports, and predictions
notebooks/     Google Colab workflow
docs/          project notes
requirements.txt
```

## 1. Install

Use Python 3.12 and create a virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The face CNN downloads pretrained ResNet-18 weights on its first run. A GPU is
recommended for CNN training; CPU execution is supported but slower.

## 2. Download and place the datasets

Download the **Autistic Children Facial Image Dataset** from
[Kaggle's dataset search](https://www.kaggle.com/datasets?search=Autistic%20Children%20Facial%20Image)
and the
Mendeley **A Multi-Modal Dataset of EEG Signals and M-CHAT Assessments for ASD
Detection** ([DOI 10.17632/4zdf2h8rzw.1](https://doi.org/10.17632/4zdf2h8rzw.1)).
The dataset archives and extracted copies are intentionally excluded from Git.
Place and extract them so the paths look like:

```text
data/raw/
  faces/
    Autistic_Children_Facial_Image.zip
    Autistic_Children_Facial_Image/
      train/  valid/  test/
  eeg/
    ASD-EEG-MCHAT-Dataset.zip
    ASD-EEG-MCHAT-Dataset/
      ASD-EEG-MCHAT-Dataset/
        Raw ASD Data/
        Raw Control Data/
        Extracted Features/
```

For example, after downloading the archives, extract each into its corresponding
`data/raw/faces/` or `data/raw/eeg/` directory. The EEG archive has a nested
`ASD-EEG-MCHAT-Dataset` folder; keep that structure.

## 3. Run the workflow (Windows PowerShell)

Run commands from the `results` directory so generated files are kept together.
Paths to scripts and raw data below are relative to that directory:

```powershell
Set-Location .\results

# Optional dataset audits and face de-duplication manifest
python ..\scripts\inspect_data.py faces --root ..\data\raw\faces\Autistic_Children_Facial_Image
python ..\scripts\eeg_audit.py --root ..\data\raw\eeg\ASD-EEG-MCHAT-Dataset

# Prepare EEG features. v1 is the quick baseline; v2 adds wavelet/connectivity features.
python ..\scripts\prepare_eeg.py --root ..\data\raw\eeg\ASD-EEG-MCHAT-Dataset
python ..\scripts\eeg_baseline.py --data eeg_windows.npz
python ..\scripts\prepare_eeg_v2.py --root ..\data\raw\eeg\ASD-EEG-MCHAT-Dataset\ASD-EEG-MCHAT-Dataset --out eeg_windows_v2.npz
python ..\scripts\eeg_baseline_v2.py --data eeg_windows_v2.npz --n_perm 100

# Face shortcut audit and CNNs (manifest is created by inspect_data.py above)
python ..\scripts\face_shortcut_test.py --root ..\data\raw\faces\Autistic_Children_Facial_Image --manifest faces_clean_manifest.csv
python ..\scripts\face_cnn.py --root ..\data\raw\faces\Autistic_Children_Facial_Image --manifest faces_clean_manifest.csv --variant rgb
python ..\scripts\face_cnn.py --root ..\data\raw\faces\Autistic_Children_Facial_Image --manifest faces_clean_manifest.csv --variant grayeq
python ..\scripts\gradcam.py --root ..\data\raw\faces\Autistic_Children_Facial_Image --manifest faces_clean_manifest.csv --variant rgb

# Fusion and ablation evaluation
python ..\scripts\fusion_eval.py --eeg eeg_windows_v2.npz --variant A --face_out face_out_rgb.npz --face_emb face_emb.npz --reps 20
```

`inspect_data.py faces` must be run before the face-model commands because it
creates `faces_clean_manifest.csv`. For an RGB fusion run, `face_cnn.py --variant
rgb` creates `face_emb.npz`; run both EEG preparation and face training before
fusion. Add `--tag A_grayeq` and `--face_out face_out_grayeq.npz` to the fusion
command for the grayscale/equalized robustness result.

The two EEG preparation scripts intentionally use different root levels:
`prepare_eeg.py` searches recursively from the outer extracted folder, while
`prepare_eeg_v2.py` expects the folder containing `Raw ASD Data/` and
`Raw Control Data/`.

## Google Colab

The `notebooks/asd_fusion_colab.ipynb` notebook contains the face-training and
fusion steps for Colab. Upload the repository's folder structure to Google
Drive, set the notebook's `PROJECT` variable to that folder, and run the cells
from top to bottom. EEG feature files (`eeg_windows_v2.npz`) should be generated
locally or uploaded to `results/` before the fusion step.

## Reproducibility notes

- The requirements pin the package versions for the intended environment. Install them with
  Python 3.12 for the intended environment.
- Pretrained ImageNet weights are fetched by `torchvision` the first time the
  face model runs, so that step requires internet access.
- The EEG sampling rate is not specified in the source dataset. The default EEG
  features therefore omit frequency-band features.
- See [docs/ASD_Face_EEG_Fusion_Project_Plan.md](docs/ASD_Face_EEG_Fusion_Project_Plan.md)
  for dataset audit findings, methodological decisions, and limitations.
