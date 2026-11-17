# Autism Detection: Facial Image + EEG Fusion
### Project context, work done so far, and full implementation plan
*Status as of 17 Nov 2026. Semester major project (college), no paper planned, goal is a working, honest, defensible demo before the evaluation.*

> Historical project notes: the repository has since been reorganized. Use the root
> [README.md](../README.md) for the current folder layout, installation, and run commands.

---

## 0. TL;DR (paste this into a new chat to resume)

- **Goal:** binary autism (ASD vs. typically developing) detection by fusing **facial images** and **EEG**.
- **Datasets (locked):** Kaggle *Autistic Children Facial Image Dataset* (2,926 images) + Mendeley *A Multi-Modal Dataset of EEG Signals and M-CHAT Assessments for ASD Detection* (28 children, 8-channel EEG).
- **Key constraint:** no public dataset has faces *and* EEG from the **same** children, so fusion is **simulated by class-matched pairing** of the two datasets (the reference paper, Vidivelli et al. 2025, does the same and states the assumption "face and EEG belong to the same person"). This must be stated as a limitation.
- **Done:** data audits of both datasets; face de-duplication manifest; EEG window-feature pipeline (`prepare_eeg.py`, run on real data → 26 usable children); EEG leave-one-child-out baseline script written and tested on synthetic data (`eeg_baseline.py`, **not yet run on real data**).
- **Not done yet:** run EEG baseline on real data; face preprocessing + shortcut check; face CNN; fusion + ablation table; report.
- **Environment problem:** Windows "Application Control" blocks `numpy.random`, which breaks pandas and scikit-learn locally. Fix = move to `C:\eeg_asd` with older pinned packages, otherwise use Google Colab.

---

## 1. Project background

| Item | Detail |
|---|---|
| Project | Autism detection from facial image + EEG, fusion model |
| Constraints | Tight schedule; wants a simple implementation with a good, *honest* accuracy; not aiming to publish |
| Reference paper (from the teacher) | Vidivelli, Padmakumari, Shanthi (2025), *Multimodal autism detection: Deep hybrid model with improved feature level fusion* (MADDHM), Computer Methods and Programs in Biomedicine 260, 108492 |
| Earlier reference | Djemal et al. DWT + Shannon entropy + ANN paper for EEG (PMC5412163) |
| Datasets considered earlier | HBN-EEG R1 (136 subjects; ASD labels unclear), Q1K, Zenodo sets, Mendeley ASD EEG + M-CHAT |
| Other face/EEG/multimodal sets reviewed and rejected | MMASD (skeleton/optical flow, no faces/EEG), Move4AS (EEG + motion capture, no faces), autism taste/smell facial video (no EEG), Sun et al. 2023 (EEG is neurotypical only) |

### 1.1 What the reference paper does (and what we take from it)

- **Data:** same Kaggle face dataset + BCIAUT-P300 EEG (15 ASD participants, 105 sessions). Paper states: *"Assume that the Face and EEG belong to the same person."*
- **Pipeline:** Wiener filter (EEG) / Gabor filter (face) → hand-crafted features (EEG: improved singular spectrum entropy, correlation dimension, CSP; face: improved AAM, GLCM, shape-LBP) → feature-level fusion (trimmed-mean weighting) → CNN + Bi-GRU, outputs averaged.
- **Reported:** ≈91% EEG alone, ≈92% face alone, ≈94–97% fused.
- **Weaknesses we should not copy:**
  - Its EEG dataset is described as 15 *ASD-only* participants doing a P300 task; it never explains how ASD-vs-control classification works from it. Our Mendeley set has both groups.
  - Internal inconsistencies: abstract and conclusion give different baseline accuracies; discussion says 89.26% overall while Table 7 says 0.9286; the 80% block of Table 6 has identical FPR and FNR rows for every model; the conclusion mentions eye tracking, which is not used.
  - Friedman/Wilcoxon tests are described incorrectly; no mention of subject-wise splitting.
  - → Treat its numbers as a rough ceiling, **not a target to match**.
- **What we adopt:** same overall structure (preprocess → features → fuse → classify), the same-person assumption (stated openly), and an ablation table (EEG-only, face-only, fused). **What we simplify:** skip the "improved" AAM/SLBT variants and the CNN + Bi-GRU hybrid (see §9 for why).

---

## 2. Datasets

### 2.1 Faces: Kaggle "Autistic Children Facial Image Dataset"

- 2,926 `.jpg` files, 122.52 MB. Folders `train/ valid/ test/`, each with `autistic/` and `non_autistic/`.
- Counts: train 1263 / 1263, valid 100 / 100, test 100 / 100. (The reference paper describes a slightly different version: 1327/1327 train, 140/140 test.)
- Labels are **not clinically verified** (as far as I know the images were collected from the web, so check the Kaggle page). Describe the project as a proof of concept, not a diagnostic tool.

**Audit results (`inspect_data.py faces`)**

| Check | Result |
|---|---|
| Corrupt/unreadable files | 0 |
| Byte-identical duplicates | 0 |
| Near-duplicate pairs (dHash distance ≤ 4) | 35 total: 25 within a split, **10 across splits** (test↔train, valid↔train). Distances: 1 (×1), 2 (×3), 3 (×11), 4 (×20). Pairs at distance 4 may be look-alikes; dropping them costs ~1% of data, so we drop all |
| Clean manifest (`faces_clean_manifest.csv`) | 2,893 kept (33 dropped): train 1258 / 1244, valid 99 / 98, **test 98 / 96** |
| **Shortcut check** | Median size 279×324 (autistic) vs 256×285 (non-autistic); brightness 134.5 vs 143.5; file size 26.0 vs 22.6 KB. The commonest non-autistic sizes all have **height exactly 224** (18–20 images each); no autistic size repeats more than twice → the classes look processed differently, so a CNN could learn photo style instead of faces |

### 2.2 EEG: Mendeley "Multi-Modal Dataset of EEG Signals and M-CHAT Assessments for ASD Detection"

- DOI 10.17632/4zdf2h8rzw.1, CC BY 4.0. **28 children: 15 ASD, 13 typically developing.** NeuroCONCISE FlexEEG, **8 channels**, 10–20 placement. Sampling rate: *not found on the dataset page*.
- Layout:
  - `Extracted Features/`: 4 CSVs (Coefficient Of Variation, Entropy, Kurtosis, Skewness), 28 rows × 8 channels + `Class` (1 = ASD, 0 = control).
  - `Raw ASD Data/EEG Data/`: 15 recordings `ASD_Subject_NN.csv` + `ASD_EEG_File_Info.csv`; `Behavioral Data/` has `ASD-M-CHAT.csv` (15 × 23 binary items) and `Mchat Questions.pdf`.
  - `Raw Control Data/EEG Data/`: 13 recordings `CTRL_Subject_NN.csv` + info file; `Behavioral Data/CTRL-M-CHAT.csv`.
  - Subject names match across EEG and M-CHAT files (`EEG_ASD_01`, …).

**Audit results (`eeg_audit.py`, `prepare_eeg.py`)**

| Finding | Detail / consequence |
|---|---|
| `Entropy.csv` = `Kurtosis.csv` | 100% identical cells. Values look like kurtosis (-3 on a flat channel). The provided features contain **no real entropy**; we compute our own |
| Coefficient of Variation file | 67 cells > 1000, max 1,149,396 → unusable, dropped |
| Dead channels in raw data | Value −8,388,608 is the floor of a 24-bit ADC (unconnected/saturated electrode). ASD_01 ch 4; ASD_03 ch 4 & 5; CTRL_08 ch 8. Handled per window |
| Unusable recordings | **ASD_04 (76 rows), CTRL_08 (97 rows)** → excluded (their rows in the provided feature files were computed from this tiny data, so they are junk too) |
| Scale | Typical amplitude (MAD) ranges from ~27 to ~1.7 million between recordings, in both classes → only scale-independent features are used |
| Recording length | After exclusions, controls average ≈25.0k rows vs ASD ≈18.7k → length could leak the label → every child contributes at most 60 windows |
| M-CHAT | Item sums do **not** separate the groups (ASD median 17, range 4–18; control median 18, range 13–20), probably an item-direction coding issue. Kept **out** of the model (screening tool for the same condition; scoring unverified) |
| 15th ASD recording | The first audit found only 14 `*_Subject_*.csv`; `prepare_eeg.py` found all 15 (ASD_15 has 16,902 rows) |

**Usable after `prepare_eeg.py`:** **26 children (14 ASD, 12 control)**, ≈**1,453 windows** (733 ASD / 720 control, by my count). Thin recordings: ASD_06 (9 windows), ASD_05 (43), ASD_13 (44).

---

## 3. Environment

- Machine: Windows, VS Code, venv. Current folder: `C:\Users\Acer\OneDrive\Desktop\Major_Project-Autism_EEG` (contains the unzipped datasets and the scripts).
- **Problem:** `ImportError: DLL load failed while importing _sfc64: An Application Control policy has blocked this file` (from `numpy.random`, triggered by `import pandas`). pip had just installed very new numpy 2.5.5. Plain numpy and Pillow work, so `inspect_data.py` was rewritten to avoid pandas. scikit-learn will probably hit the same block (unconfirmed).
- **Fix to try (2 minutes):** copy the project outside OneDrive *and* pin older, widely used versions (my guess is that the new numpy release is what Windows distrusts; moving out of OneDrive alone may not help):

```powershell
robocopy . C:\eeg_asd /E /XD .venv
cd C:\eeg_asd
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.5.2 pillow openpyxl
python -c "import numpy.random, pandas, sklearn; print('ok')"
```
  Then open `C:\eeg_asd` in VS Code, select the new `.venv`, delete the old Desktop copy.
- **Fallback / GPU:** Google Colab (free GPU, nothing blocked). Plan on Colab for the face CNN regardless. Do **not** turn off Smart App Control (Windows cannot re-enable it without a reinstall).

---

## 4. Scripts built so far

| Script | Purpose | Status |
|---|---|---|
| `inspect_data.py faces` / `eeg` | Face counts, corrupt files, exact + near-duplicate detection (md5 + dHash), shortcut stats; EEG folder inventory. Writes `faces_report.txt`, `face_duplicates.csv`, `faces_clean_manifest.csv`, `eeg_inventory.txt`. *(Your local copy was edited to use csv/openpyxl instead of pandas.)* | **Run on real data** |
| `eeg_audit.py` | Duplicate feature files, outliers, raw-signal audit (rows, stuck channels, scale), M-CHAT item sums. Needs only numpy. Writes `eeg_audit.txt` | **Run on real data** |
| `prepare_eeg.py` | Raw EEG → window features. Excludes short recordings, drops dead channels per window, cross-checks the dataset's file list. Writes `eeg_windows.npz`, `eeg_subjects.csv` | **Run on real data** (output: 26 children) |
| `eeg_baseline.py` | Leave-one-child-out EEG baseline (LogReg + RandomForest), window- and child-level metrics, permutation test, saves out-of-fold probabilities (`eeg_oof.npz`) and `eeg_loso_children.csv` | Written, tested on synthetic data only; **needs a real run** |

**Feature design in `prepare_eeg.py`** (window 512 samples, step 256, ≤60 evenly spaced windows per child, ≥4 good channels per window): per channel after mean + linear-trend removal → skewness, excess kurtosis, Shannon entropy (32-bin histogram of the standardised window), Hjorth mobility, Hjorth complexity, log relative variance (window variance ÷ the recording's median window variance). Each is averaged and std-ed across good channels → **12 features per window**. All are scale-/offset-invariant (verified: a rescaled and shifted copy gives identical features to ~1e-14). No frequency-band features because the sampling rate is unknown.

---

## 5. Design decisions (locked) and corrections made along the way

**Decisions**
1. Datasets: Kaggle faces + Mendeley EEG (M-CHAT excluded from the model).
2. Fusion is **simulated** by class-matched pairing across the two datasets; stated as an assumption and a limitation (same as the reference paper).
3. EEG features are **recomputed from raw** windows, not taken from the 28-row feature files.
4. EEG validation is **subject-wise** (leave-one-child-out), never window-wise; **child-level** accuracy is the headline number.
5. EEG models are **fixed in advance** (no tuning): LogReg is the pre-declared primary, RandomForest secondary, to avoid selection bias with only 26 children.
6. Faces: use `faces_clean_manifest.csv`, identical preprocessing for both classes, and a metadata-only shortcut test.
7. Fusion compares **late fusion** (probability averaging) and **feature-level fusion** (paper-style), plus both single-modality models, on identical pairs.
8. Modelling runs on Colab if the local numpy/scikit-learn block cannot be fixed.

**Corrections / things learned**
- I first said M-CHAT would trivially leak the label. The audit shows the raw item sum does not separate the groups. It stays out for scoring-uncertainty and conceptual reasons, not because of a demonstrated leak.
- I suspected controls had longer recordings; the audit supports it (≈25.0k vs ≈18.7k rows), hence the per-child window cap.
- "Extracted Features" looked usable at first; it is not (duplicate file, junk CV, junk rows for two children).

---

## 6. Implementation plan: baseline → single-modality models → fusion

Rough effort estimates (hours) assume no more environment surprises; adjust to your deadline.

| Phase | What | Est. |
|---|---|---|
| 0 | Fix environment / set up Colab | 0.5–1 |
| 1 | EEG baseline on real data | 0.5 |
| 2 | Face preprocessing + shortcut test + baselines | 2–3 |
| 3 | Face CNN | 1–2 |
| 4 | EEG improvements (optional, tightly limited) | 1 |
| 5 | Fusion + ablation | 2–3 |
| 6 | Report, figures, viva prep | 4–6 |

### Phase 0: Environment
Do §3. Folder layout (local or Drive):

```
C:\eeg_asd\
  Autistic_Children_Facial_Image\{train,valid,test}\{autistic,non_autistic}\
  ASD-EEG-MCHAT-Dataset\{Extracted Features, Raw ASD Data, Raw Control Data}\
  inspect_data.py  eeg_audit.py  prepare_eeg.py  eeg_baseline.py   (+ scripts still to be built)
  outputs: faces_clean_manifest.csv  eeg_windows.npz  eeg_oof.npz  eeg_loso_children.csv  *_report.txt
```
**Done when:** `python -c "import numpy.random, pandas, sklearn"` prints no error (or Colab is ready with the files uploaded).

### Phase 1: EEG baseline (script already written)
```powershell
python eeg_baseline.py --data eeg_windows.npz        # or on Colab: !python eeg_baseline.py
```
Outputs: window-level and child-level accuracy, balanced accuracy, AUC, confusion matrix and sensitivity/specificity for LogReg and RandomForest; a **permutation test** (200 shuffles of which children are ASD) giving the chance threshold and p-value; per-child table; `eeg_oof.npz` (out-of-fold P(ASD) for every window, needed for fusion).

**How to read it:** chance for child-level accuracy is 14/26 = 53.8% (always guess ASD). On my synthetic no-signal data the shuffled-label accuracy reached ≈73% at the 95th percentile with 26 children, and a single honest run landed at 38%. Your own threshold is printed by the script; expect that child-level accuracy in the 60s to low 70s may be **indistinguishable from luck**. Report it as it is. Window-level accuracy is inflated (neighbouring windows correlate) and must not be the headline.

### Phase 2: Face preprocessing and shortcut test (to build)
1. Load images listed with `keep=True` in `faces_clean_manifest.csv`; keep the provided splits (train 1258/1244, valid 99/98, test 98/96).
2. **Metadata-only baseline (shortcut test):** train a tiny model (logistic regression) on *only* `width, height, aspect ratio, mean brightness, file size, mean R/G/B` of train images; evaluate on test. With 194 test images the 95% binomial interval around chance is ±7%, so **accuracy above ≈57% means the photo properties alone leak the label.** Run this before and after preprocessing.
3. **Standard preprocessing for both classes:** RGB, resize to 224×224, ImageNet normalisation; training augmentation: horizontal flip, rotation ±10°, brightness/contrast jitter 0.3.
4. **Robustness check:** repeat the CNN (Phase 3) on a grayscale + histogram-equalised variant. A big accuracy drop would mean colour/brightness cues were helping.
5. Optional face baseline in the paper's spirit: Gabor filter → GLCM + LBP histogram features → SVM / RandomForest (fast, interpretable, no GPU). Also usable as the face feature vector for feature-level fusion.

### Phase 3: Face CNN (Colab, to build)
- Backbone: ImageNet-pretrained ResNet18 (or MobileNetV2).
- Stage 1: train the new classification head only (≈5 epochs, Adam lr 1e-3). Stage 2: unfreeze the last block (`layer4`), ≈10 epochs, lr 1e-4. Batch 32, fixed seed, early stopping on `valid` loss.
- Evaluate once on `test`: accuracy, balanced accuracy, AUC, confusion matrix, sensitivity/specificity; 95% binomial CI (n = 194).
- Sanity check: Grad-CAM on a few test images. If it highlights backgrounds, borders or hair instead of the face, the model is using shortcuts.
- **Save:** `face_probs_test.npy` (P(ASD) per test image, with file names), `face_probs_valid.npy`, and **frozen ImageNet** 512-d embeddings for train/valid/test (used for feature-level fusion, see Phase 5).
- Do **not** train anything on the fine-tuned CNN's predictions or embeddings for *training* images: they are overfit and would make a fusion classifier over-trust the face branch.

### Phase 4: EEG model improvements (optional, tightly limited)
With 26 children, every extra tried variant inflates the result. Allow **at most 2–3 variants declared up front**, report all of them, and keep LogReg-on-the-default-features as the primary result:
- Wiener filtering of each window before feature extraction (`scipy.signal.wiener`, as in the paper).
- Window length 256 or 1024 samples.
- An SVM in place of / alongside LogReg.

Do **not** do Bi-GRU/CNN on 1,453 correlated windows from 26 children: it will memorise children.

### Phase 5: Fusion and ablation (to build: `fusion_eval.py`)

**Why simulated pairing, and how to keep it honest:** the two datasets contain different people, so fusion uses pairs of *(face image, EEG window)* of the **same class**, formed only inside the same evaluation fold (no image or window on both the train and test side).

**Protocol (leave-one-child-out for EEG, fixed face test split):**
```text
for each held-out child c  (class k = ASD or control):          # 26 folds
    EEG train windows  = windows of all other children           # standardise on these only
    face train images  = clean train split                       # CNN already trained; PCA fit on these only
    TEST pairs   = every clean TEST face of class k  x  a random window of child c
    TRAIN pairs  = every train face  x  a random same-class window from the EEG train children
    repeat R = 20 times with different random pairings

predictors evaluated on the SAME test pairs:
    EEG-only        : p_eeg  = out-of-fold LOSO window probability (from eeg_oof.npz)
    Face-only       : p_face = fine-tuned CNN test probability
    Late fusion     : (p_face + p_eeg) / 2 > 0.5                 # no training needed
    Feature fusion  : LogReg on [PCA-32 of frozen face embedding | 12 EEG features], trained on TRAIN pairs
report pooled over folds, mean ± std over the R repetitions
```
- **Primary fusion = late fusion** (simple, no extra training, no leakage risk). **Feature-level fusion** is the paper-style comparison.
- Optional sanity control: **mismatched pairs** (face of one class + EEG of the other). It shows how strongly each modality pulls the fused decision.

**Final results table** (the one to put in the report):

| Model | Accuracy | Balanced acc. | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| EEG-only (window), and EEG-only (child-level) | | | | | |
| Face-only (CNN) | | | | | |
| Late fusion (average) | | | | | |
| Feature-level fusion | | | | | |
| *(optional)* Face handcrafted (Gabor + GLCM + LBP) | | | | | |
| *(reference)* Metadata-only shortcut model | | | | | |

**Uncertainty:** face-only → binomial CI over 194 images; EEG child-level → 26 children (a result can only move in ≈4% steps); fusion → std over repetitions. State that real uncertainty is dominated by the small number of children and faces.

### Phase 6: Report and viva
1. Problem, motivation, dataset description (include the audit as a strength: duplicates, shortcuts, dead channels).
2. Methods per branch + fusion protocol (include the pairing diagram).
3. Results table + EEG permutation test + Grad-CAM figures.
4. **Limitations (must be explicit):**
   - Face and EEG come from different people; pairing is class-matched, so fused accuracy partly beats each modality *by construction* (two independent votes for the same class). Real same-person data would show correlated errors.
   - Only 26 usable children for EEG; sampling rate unknown; recordings of mixed quality; per-child results are coarse.
   - Face labels are not clinically verified and the images may differ by photo style between classes.
   - Proof of concept only, **not a diagnostic tool**; the data involve children, so no claims about clinical use.
5. Future work: collect paired face + EEG from the same children; larger EEG cohorts; attention-based fusion; clinical validation.

**Likely viva questions to prepare:** Why two datasets? (no paired public data; same assumption as the reference paper) · Why leave-one-child-out? (windows from a child are correlated) · How do you know the CNN is not using background/photo style? (metadata test, preprocessing, Grad-CAM) · Why not use the provided feature files? (duplicate/junk) · Why is EEG accuracy modest? (26 children, 8 channels, heterogeneous recordings; permutation test) · Why M-CHAT excluded?

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Local numpy/scikit-learn stays blocked | Pinned older versions in `C:\eeg_asd`; else Colab |
| EEG accuracy ≈ chance | Report honestly with the permutation test; the project's contribution is the careful pipeline and fusion protocol; the face branch carries the performance |
| Face CNN exploits photo style | Metadata baseline, uniform preprocessing, grayscale check, Grad-CAM |
| Fusion looks "too good" | Explain pairing-by-construction in the limitations; add mismatched-pair control |
| Over-tuning on 26 children | Pre-declare models; ≤2–3 variants; report all |
| Time runs out | Priority order below |

**Priority order if time is short:** (1) EEG baseline real run → (2) face preprocessing + CNN → (3) late-fusion pairing + ablation → (4) feature-level fusion → (5) shortcut robustness/Grad-CAM → (6) optional EEG variants and handcrafted face baseline.

---

## 8. Open items

- **Days left until the evaluation?** (not yet given; the plan can be trimmed once known)
- Did the `C:\eeg_asd` + pinned-versions fix work, or are we on Colab?
- Output of `eeg_baseline.py` on the real `eeg_windows.npz` (send `eeg_baseline_report.txt`).
- Next scripts to be written: face preprocessing + metadata-only test, Colab CNN notebook, `fusion_eval.py`.

---

## 9. Why not copy the paper's full pipeline

- The "improved" AAM/SLBT/singular-spectrum features and the CNN + Bi-GRU on fused vectors add complexity but, with 26 EEG children and ~2.9k face images, they mainly add overfitting risk and implementation time.
- A pretrained ResNet on faces plus simple, scale-free EEG features with subject-wise validation is faster to build, easier to defend, and still follows the paper's structure (preprocess → features → fuse → classify) and its ablation table.

## 10. Quick command reference

```powershell
# audits (already run)
python inspect_data.py faces --root .\Autistic_Children_Facial_Image
python inspect_data.py eeg   --root .\ASD-EEG-MCHAT-Dataset
python eeg_audit.py          --root .\ASD-EEG-MCHAT-Dataset

# EEG features and baseline
python prepare_eeg.py  --root .\ASD-EEG-MCHAT-Dataset
python eeg_baseline.py --data eeg_windows.npz
```
