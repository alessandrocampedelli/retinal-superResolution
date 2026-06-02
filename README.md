# Computational Imaging - Final Project (Group G)

This repository contains the final project for the **Computational Imaging** course. 

## Project Overview

The objective of this project is to solve a **Super-Resolution (SR)** inverse problem on a dataset of high-resolution medical retinal images (*Egyptian Medical Retinal Images*). The images are resized to $256 \times 256$, and the task is to recover the high-quality images from degraded observations.

The forward degradation model includes:
- Spatial downscaling (factors $s=2$ and $s=4$) using average pooling.
- Additive Gaussian noise ($\sigma=0.005$ and $\sigma=0.01$).

### Evaluated Methods
We compare three distinct families of methods to solve the inverse problem:
1. **Variational Method:** Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) with L1 Wavelet regularization.
2. **End-to-End Deep Learning:** U-Net architecture.
3. **Generative Prior:** Diffusion-based Plug-and-Play Image Restoration (DiffPIR).

---

## Repository Structure

To ensure a clean, reproducible, and cloud-friendly workspace, the repository is structured as follows:

```
CI_Project/
│
├── notebooks/                              # Jupyter Notebooks for execution
│   ├── 01_Dataset_and_Forward_Model.ipynb  # Dataset preprocessing and pipeline setup
│   ├── 02_Variational_FISTA_Wavelets.ipynb # FISTA solver implementation & sweep
│   ├── 03_EndToEnd_UNet.ipynb              # U-Net training and evaluation
│   ├── 04_Generative_DiffPIR.ipynb         # DiffPIR execution
│   └── 05_Results_and_Comparison.ipynb     # Metric evaluation and visual comparisons
│
├── outputs/                                # Lightweight output folder
│   ├── comparison/                         # Comparison plots
│   ├── diffpir/                            # DiffPIR specific plots
│   ├── fista/                              # FISTA specific plots
│   └── unet/                               # U-Net specific plots
│
└── README.md                               # Project documentation
```

---

## How to Use This Project

This project is explicitly designed to be run in a cloud environment (**Google Colab** or **Kaggle**) with GPU acceleration.

### 1. Library Dependencies (IPPy)
The underlying algorithmic logic (operators, neural network blocks, metrics) relies on the `IPPy` library. 
To keep this repository clean, **`IPPy` is not included here**. Instead, every notebook dynamically clones the custom `IPPy` fork at runtime:
```bash
!git clone https://github.com/alessandrocampedelli/IPPy.git
```

### 2. Dataset and Weights (Google Drive)
The retinal dataset (~4000 images) and the trained model weights are **not** tracked in this GitHub repository to avoid bloating. 
Instead, they are stored permanently on **Google Drive**.

**Workflow:**
1. Open `01_Dataset_and_Forward_Model.ipynb` in Google Colab.
2. Execute the cells to mount your Google Drive. 
3. The notebook will automatically download the dataset, preprocess it (resize, tensor conversion, train/val/test splits), and save the tensors directly to your Drive.
4. For all subsequent notebooks (`02` to `05`), simply mount your Google Drive. The tensors will be loaded instantaneously without the need for redundant preprocessing.

### 3. Execution Order
To reproduce the experiments, execute the notebooks sequentially from `01` to `05`. 
- **Notebooks 02, 03, and 04** are independent of each other (once the dataset is processed in Notebook 01) and can be executed in any order.
- **Notebook 05** must be executed last, as it gathers the reconstruction results saved by the previous notebooks to compute PSNR and SSIM metrics.