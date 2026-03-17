# PI-INR: Physics-Geometry Fusion for Uncertainty-Aware Early Warning in Adaptive Radiotherapy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-MedIA%20Under%20Review-blue.svg)]()

Official PyTorch implementation of the framework proposed in the paper:  
**"Physics-Geometry Fusion with Uncertainty-Aware Early Warning for Dosimetric Decay in Adaptive Radiotherapy via Implicit Neural Representations"**  
*(Submitted to Medical Image Analysis, 2026)*

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Overview

We propose a novel, zero-dose-computation framework (PI-INR) for real-time dosimetric decay warning in Adaptive Radiotherapy (ART). By bridging **Riemannian geometry**, **Lie derivatives**, and **Evidential Deep Learning (EDL)**, this framework provides a highly interpretable Red-Yellow-Green clinical triage system in under 8 seconds per patient, avoiding the heavy computational burden of traditional Deformable Dose Accumulation (DDA).

### Key Features:
- **Topology-preserving deformable registration** with analytical Jacobian constraint
- **Riemannian dose manifold** for physics-informed regularization
- **Lie derivative entropy (ATI)** for zero-shot dosimetric risk assessment
- **Evidential Deep Learning** for calibrated uncertainty estimation
- **Traffic-light clinical decision system** (Red-Yellow-Green)

---

## 📊 Dataset

This code is designed to work with the **Pancreatic-CT-CBCT-SEG** dataset from The Cancer Imaging Archive (TCIA). The dataset contains 40 patients with pancreatic cancer who underwent longitudinal radiotherapy, including planning CT, daily CBCT, RTDOSE, and RTSTRUCT files.

### Dataset Information

| Item | Details |
|------|---------|
| **Dataset Name** | Pancreatic-CT-CBCT-SEG |
| **Institution** | The Cancer Imaging Archive (TCIA) |
| **DOI** | [10.7937/TCIA.ESHQ-4D90](https://doi.org/10.7937/TCIA.ESHQ-4D90) |
| **Patients** | 40 |
| **Size** | 13.3 GB (Version 2) |
| **Modalities** | CT, RTDOSE, RTSTRUCT |
| **Updated** | 2022-08-23 |
| **License** | CC BY 4.0 |

### Data Download Instructions

The dataset can be directly accessed and downloaded from its official TCIA collection page:

👉 **[Pancreatic-CT-CBCT-SEG Collection on TCIA](https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/)**

**Recommended Download Method:**

1. Visit the official collection page at the link above.
2. Navigate to the **"Data Access"** section.
3. Click the **"Download"** button to obtain the NBIA Data Retriever manifest file (`.tcia` file).
4. Use the **NBIA Data Retriever** (download from [here](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)) to open the manifest file and download the complete dataset (Version 2, approximately 13.3 GB).
5. After downloading, place the data in the following directory structure:

```
PI-INR-MedIA/
└── data/
    └── Pancreatic-CT-CBCT-SEG/
        ├── Pancreas-CT-CB_001/
        ├── Pancreas-CT-CB_002/
        ├── ...
        └── Pancreas-CT-CB_040/
```

> **Important Notes:**
> - The dataset is **NOT included** in this repository due to size constraints. You must download it separately from TCIA.
> - Ensure you download **Version 2** (updated 2022-08-23) as it contains the RTDOSE files required for the full pipeline.
> - The complete dataset is 13.3 GB. Please ensure at least 15 GB of free disk space before downloading.

### Dataset Citation

If you use this dataset, please cite the following:

**Data Citation:**
```bibtex
@article{hong2021breath,
  title={Breath-hold CT and cone-beam CT images with expert manual organ-at-risk segmentations from radiation treatments of locally advanced pancreatic cancer},
  author={Hong, J and Reyngold, M and Crane, C and Cuaron, J and Hajj, C and Mann, J and Zinovoy, M and Yorke, E and LoCastro, E and Apte, AP and Mageras, G},
  journal={The Cancer Imaging Archive},
  year={2021},
  doi={10.7937/TCIA.ESHQ-4D90}
}

**Related Publication:**
```bibtex
@article{han2021deep,
  title={Deep‐learning‐based image registration and automatic segmentation of organs‐at‐risk in cone‐beam CT scans from high‐dose radiation treatment of pancreatic cancer},
  author={Han, X and Hong, J and Reyngold, M and Crane, C and Cuaron, J and Hajj, C and Mann, J and Zinovoy, M and Greer, H and Yorke, E and Mageras, G and Niethammer, M},
  journal={Medical Physics},
  volume={48},
  number={6},
  pages={3084--3095},
  year={2021},
  doi={10.1002/mp.14906}
}
```

⚙️ Installation
Option 1: Using pip (recommended for beginners)
bash
# Clone the repository
git clone https://github.com/YourUsername/PI-INR-MedIA.git
cd PI-INR-MedIA

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Option 2: Using conda
bash
# Clone the repository
git clone https://github.com/YourUsername/PI-INR-MedIA.git
cd PI-INR-MedIA

# Create conda environment
conda env create -f environment.yml
conda activate pi-inr
💻 Usage
Quick Start (Demo)
Run a quick test on a single patient (with reduced epochs for speed):

bash
python demo.py --patient Pancreas-CT-CB_001 --quick
This will:

Process one patient with 100 epochs (instead of 1200)

Generate basic results and statistics

Complete in 2-5 minutes on CPU

Full Pipeline
Process all patients sequentially (40 patients, ~3-5 minutes each on CPU):

bash
# Step 1: Run the full production pipeline
python run_pipeline.py

# Step 2: Generate statistical analysis and Excel tables
python run_statistics.py

# Step 3: Generate publication-ready figures
python run_visualization.py
Advanced Options
bash
# Process only the first 10 patients
python run_pipeline.py --max-patients 10

# Force re-run even if results already exist
python run_pipeline.py --force-rerun
Expected Output Structure

results/
└── MedIA_Ultimate_Run/
    ├── MedIA_Quantitative_Results.csv      # Main results table
    ├── patient_status.csv                   # Data integrity check
    ├── batch_log.txt                         # Processing log
    ├── Pancreas-CT-CB_001/                   # Individual patient results
    │   ├── Warped.nii.gz                      # Registered image
    │   ├── ATI_Map.nii.gz                     # Risk map
    │   ├── Uncertainty.nii.gz                 # Uncertainty map
    │   ├── Disp_X.nii.gz                       # Deformation components
    │   ├── Disp_Y.nii.gz
    │   ├── Disp_Z.nii.gz
    │   ├── Paper_Figure.png                    # 4-panel visualization
    │   └── Clinical_Report.html                # HTML clinical report
    ├── ... (similar for other patients)
    └── MedIA_Final_Excel/                      # Paper-ready outputs
        ├── excel_tables/                        # Excel spreadsheets
        ├── figures/                              # Standard figures
        ├── figures_large_font/                   # Large font figures
        └── text/                                  # Paper paragraphs
📁 Repository Structure

PI-INR-MedIA/
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT license
├── README.md                 # This file
├── requirements.txt          # Python dependencies (pip)
├── environment.yml           # Conda environment file
├── config.yaml               # Configuration file
├── demo.py                   # Quick start demo
├── run_pipeline.py           # Main training + inference pipeline
├── run_statistics.py         # Statistical analysis + Excel generation
├── run_visualization.py      # Paper figure generation
├── data/                     # (Empty) Place TCIA dataset here
└── results/                  # (Empty) Results will be saved here
File Descriptions
File	Description
run_pipeline.py	Main pipeline: data preprocessing, training, inference
run_statistics.py	Statistical analysis, Excel tables, summary reports
run_visualization.py	Generates all paper figures (Figure 5, S1-S6)
demo.py	Quick start demo for testing on a single patient
config.yaml	Configuration file with all hyperparameters
requirements.txt	Python package dependencies (pip)
environment.yml	Conda environment specification
📝 Citation
If you find this code useful in your research, please cite our paper:

bibtex
@article{pi_inr_2026,
  title={Physics-Geometry Fusion with Uncertainty-Aware Early Warning for Dosimetric Decay in Adaptive Radiotherapy via Implicit Neural Representations},
  author={Anonymous Authors},
  journal={Medical Image Analysis (Under Review)},
  year={2026}
}
Also, please cite the dataset if you use it:

bibtex
@article{hong2021breath,
  title={Breath-hold CT and cone-beam CT images with expert manual organ-at-risk segmentations from radiation treatments of locally advanced pancreatic cancer},
  author={Hong, J and Reyngold, M and Crane, C and Cuaron, J and Hajj, C and Mann, J and Zinovoy, M and Yorke, E and LoCastro, E and Apte, AP and Mageras, G},
  journal={The Cancer Imaging Archive},
  year={2021},
  doi={10.7937/TCIA.ESHQ-4D90}
}
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

The Pancreatic-CT-CBCT-SEG dataset is licensed under the CC BY 4.0 license - see the TCIA website for details.

🙏 Acknowledgments
The authors thank the contributors of the Pancreatic-CT-CBCT-SEG dataset for making their data publicly available via TCIA.

We thank the National Cancer Institute for funding the TCIA project.

This work was supported by [your funding sources if any].

⚠️ Important Notes
All code was tested with Python 3.9 and PyTorch 2.0

For GPU acceleration, ensure CUDA is properly installed

Random seed is fixed (42) for full reproducibility

Processing time: ~3-5 minutes per patient on CPU, ~30-60 seconds on GPU (NVIDIA A100)

The dataset is NOT included in this repository - must be downloaded separately from TCIA

If you encounter any issues, please check the Issues page or open a new issue.

🔗 Useful Links
Resource	Link
TCIA Dataset Homepage	https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/
NBIA Retriever Download	https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
Dataset DOI	https://doi.org/10.7937/TCIA.ESHQ-4D90
TCIA Help	help@cancerimagingarchive.net
Happy Researching! 🎉