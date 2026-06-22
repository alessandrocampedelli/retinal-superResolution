# Computational Imaging — Final Project (Group G)

> **Super-Resolution on Medical Retinal Images**  
> Three approaches compared: Variational (FISTA), Deep Learning (U-Net), Generative (DiffPIR)

---

## ⚠️ Important: Google Colab Only

This project is designed to run **exclusively on [Google Colab](https://colab.research.google.com/)** with a GPU runtime enabled.  
Local execution is not officially supported. All notebooks mount Google Drive for persistent storage of datasets, weights, and outputs.

---

## Project Overview

The goal of this project is to solve a **Super-Resolution (SR)** inverse problem on the *Egyptian Medical Retinal Images* dataset. Images are resized to 256×256 pixels, and the task is to recover high-resolution images from degraded low-resolution observations.

**Forward degradation model:**
- Spatial downscaling via average pooling with factors **s ∈ {2, 4}**
- Additive Gaussian noise with levels **σ ∈ {0.005, 0.01}**

This yields 4 degradation scenarios, from easy (s=2, σ=0.005) to hard (s=4, σ=0.01).

**Methods evaluated:**
| # | Method | Type |
|---|--------|------|
| 1 | FISTA + Wavelet L₁ regularization | Variational |
| 2 | Residual U-Net & Hybrid Attention U-Net | End-to-End Deep Learning |
| 3 | DiffPIR (Diffusion Plug-and-Play) | Generative Prior |

---

## Repository Structure

```
CI_Project/                         ← this GitHub repository
│
├── notebooks/
│   ├── 01_Dataset_and_Forward_Model.ipynb   # Dataset download, preprocessing, forward model
│   ├── 02_Variational_FISTA_Wavelets.ipynb  # FISTA solver with Wavelet L₁ regularization
│   ├── 03_DeepLearning_UNet.ipynb           # U-Net training and evaluation (8 models)
│   ├── 04_Generative_DiffPIR.ipynb          # DDPM training + DiffPIR inference
│   └── 05_Results_and_Comparison.ipynb      # Metrics aggregation and final comparison
│
├── outputs/                        # Lightweight figures committed to GitHub
│   ├── comparison/                 # Final comparison plots (NB05)
│   ├── diffpir/                    # DiffPIR-specific plots
│   ├── fista/                      # FISTA-specific plots
│   └── unet/                       # U-Net training curves and qualitative results
│
└── README.md
```

> **Note:** The dataset, model weights, and intermediate tensor files are **not tracked in this repository**. They are stored on Google Drive (see the section below).

---

## Google Drive Folder Structure

Before running any notebook, your Google Drive must contain a folder with the following structure.  
The notebooks will create most subfolders automatically, but the **root folder name must match** the `BASE_DRIVE_FOLDER` variable (default: `CI_Project`).

```
MyDrive/
└── CI_Project/                     ← BASE_DRIVE_FOLDER (configurable)
    │
    ├── dataset/                    ← created automatically by NB01
    │   ├── train/                  # ~3329 preprocessed .pt tensors
    │   ├── val/                    # ~416 preprocessed .pt tensors
    │   └── test/                   # ~417 preprocessed .pt tensors
    │
    ├── weights/                    ← created automatically, populated by NB03 and NB04
    │   ├── resunet_s2_sig0005.pth
    │   ├── resunet_s2_sig001.pth
    │   ├── resunet_s4_sig0005.pth
    │   ├── resunet_s4_sig001.pth
    │   ├── attnunet_s2_sig0005.pth
    │   ├── attnunet_s2_sig001.pth
    │   ├── attnunet_s4_sig0005.pth
    │   ├── attnunet_s4_sig001.pth
    │   └── DDPMDenoiser_retinal.pth
    │
    └── outputs/                    ← created automatically by each notebook
        ├── fista/                  # FISTA results (CSV + plots)
        ├── unet/                   # U-Net results (CSV + training curves)
        ├── diffpir/                # DiffPIR results (CSV + training loss)
        └── comparison/             # Final comparison figures (NB05)
```

> **Tip:** You only need to create the root folder `CI_Project` manually on Google Drive. All subfolders are created automatically when you run the notebooks.

---

## Step-by-Step Execution Guide

### Prerequisites

1. A Google account with Google Drive.
2. Access to [Google Colab](https://colab.research.google.com/) with a **GPU runtime** (`Runtime → Change runtime type → T4 GPU`).
3. A Kaggle account (required only for NB01 to download the dataset).

---

### Step 1 — Clone this repository on Colab

At the start of any session, clone the repository to access the notebooks:

```python
!git clone https://github.com/alessandrocampedelli/CI_Project.git
%cd CI_Project
```

Alternatively, open the notebooks directly from GitHub via the Colab interface (`File → Open notebook → GitHub`).

---

### Step 2 — Run the notebooks in order

> **Important:** Notebooks must be run **sequentially** (01 → 02 → 03 → 04 → 05).  
> Notebooks 02, 03, and 04 are independent of each other once NB01 is complete, and can be run in any order among themselves.  
> Notebook 05 must always be the last to run.

---

#### 📓 Notebook 01 — Dataset and Forward Model
**Purpose:** Downloads the dataset from Kaggle, applies preprocessing, and saves tensors to Google Drive.

**Before running:**
- Configure your Kaggle API credentials in the notebook (the cell will guide you).
- Set `BASE_DRIVE_FOLDER = 'CI_Project'` (or your custom folder name — must match in all other notebooks).

**What it does:**
- Downloads *Egyptian Medical Retinal Images* from Kaggle via `kagglehub`.
- Converts images to grayscale, resizes to 256×256.
- Splits into train/val/test (80/10/10, seed=42).
- Saves preprocessed tensors as `.pt` files to `MyDrive/CI_Project/dataset/`.

**Expected output:** ~4162 `.pt` files saved on Google Drive (≈ 873 MB).  
**⏱ Estimated time:** 10–20 minutes.

---

#### 📓 Notebook 02 — FISTA + Wavelet L₁
**Purpose:** Solves the SR inverse problem using the variational FISTA algorithm with Wavelet L₁ regularization.

**Key configurable parameters:**
| Variable | Default | Description |
|----------|---------|-------------|
| `LAMBDA` | `0.001` | Wavelet L₁ regularization strength |
| `MAX_ITER` | `200` | Maximum FISTA iterations |
| `NUM_TEST_IMGS` | `10` | Number of test images to evaluate |

**What it does:**
- Dynamically clones the `IPPy` library at runtime.
- Runs FISTA on all 4 degradation scenarios.
- Saves per-image PSNR/SSIM results to `outputs/fista/fista_metrics.csv`.
- Saves qualitative figures and convergence plots.

**No training required** — FISTA is a purely mathematical method.  

---

#### 📓 Notebook 03 — Deep Learning U-Net
**Purpose:** Trains and evaluates two U-Net architectures (Residual U-Net and Hybrid Attention U-Net) across all 4 degradation scenarios — 8 models in total.

**Key configurable parameters:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MIDDLE_CH` | `[32, 64, 128, 256]` | Channel progression per encoder level |
| `N_LAYERS` | `2` | Conv layers per block |
| Epochs (in training loop) | `20` | Max training epochs per model |
| Learning rate | `5e-4` | Adam optimizer LR |

**What it does:**
- Trains all 8 models with early stopping, cosine annealing, and gradient clipping.
- Saves model weights to `MyDrive/CI_Project/weights/` (one `.pth` file per model).
- If a weight file already exists on Drive, training for that model is automatically skipped (resume-safe).
- Saves per-image PSNR/SSIM results to `outputs/unet/unet_metrics.csv`.

---

#### 📓 Notebook 04 — Generative DiffPIR
**Purpose:** Trains a DDPM denoiser on clean HR retinal images, then uses it as a Plug-and-Play prior within the DiffPIR inference algorithm (HQS + DDIM).

**Key configurable parameters:**
| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_CH` | `32` | Base channels of the diffusion U-Net |
| `T_DIFFUSION` | `1000` | Total diffusion timesteps |
| `N_TRAIN_EPOCHS` | `100` | Training epochs for the DDPM |
| `BATCH_SIZE` | `16` | Training batch size |
| `LR_DIFFUSION` | `2e-4` | AdamW learning rate |
| `EMA_DECAY` | `0.9999` | EMA weight decay |
| `CKPT_EVERY` | `5` | Save checkpoint every N epochs |
| `DDIM_STEPS` | `100` | Number of DDIM inference steps |
| `RHO_SCALE` | `1.0` | HQS data fidelity strength scaling |
| `NUM_TEST_IMAGES` | `20` | Test images per scenario |

**What it does:**
- Trains a `DiffusionUNet` (~6M parameters) on clean HR images in [-1, 1] space.
- Training is automatically **resumed** from the last checkpoint if `DDPMDenoiser_retinal.pth` exists on Drive and the saved epoch is less than `N_TRAIN_EPOCHS`. Training is skipped entirely if already complete.
- Runs DiffPIR inference (HQS + DDIM) on all 4 degradation scenarios.
- Saves results to `outputs/diffpir/diffpir_metrics.csv`.


> **Note on DiffPIR results:** 100 training epochs on ~3300 images represent a minimal proof-of-concept. The DiffPIR framework is theoretically correct, but the generative prior requires significantly more training (as in the original paper) to achieve competitive PSNR/SSIM.

---

#### 📓 Notebook 05 — Results and Comparison
**Purpose:** Aggregates the CSV results from NB02, NB03, and NB04, and produces all final comparison figures.

**What it does:**
- Loads metrics from all three methods.
- Generates: bar charts, degradation line plots, PSNR boxplots, PSNR vs. SSIM scatter plots, qualitative comparison grids (4×5), and zoom crops around the optic disc.
- Saves all figures to `outputs/comparison/` on Drive.

**Must be run after NB02, NB03, and NB04.**  

---

## Notes on the IPPy Library

All algorithmic logic (operators, proximal solvers, neural network blocks, metrics) is implemented in the custom **`IPPy`** library, maintained in a separate repository:

👉 [github.com/alessandrocampedelli/IPPy](https://github.com/alessandrocampedelli/IPPy)

Every notebook dynamically clones `IPPy` at runtime with:
```python
!git clone https://github.com/alessandrocampedelli/IPPy.git
```

No manual installation is required.

---

## Results Summary

| Method | Best PSNR | Best SSIM | Notes |
|--------|-----------|-----------|-------|
| FISTA + Wavelet L₁ | 37.46 dB | 0.9707 | No training, convergence guaranteed |
| Hybrid Attention U-Net | **42.83 dB** | **0.9774** | Best quantitative performance |
| DiffPIR (100 epochs) | 15.46 dB | 0.2804 | Proof-of-concept; prior under-trained |

---

## Authors

- **Matteo Boscherini**
- **Alessandro Campedelli**

Computational Imaging — Group G — A.Y. 2025/2026
