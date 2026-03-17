PI-INR: Physics-Geometry Fusion for Adaptive Radiotherapy
=========================================================

🚀 OVERVIEW
-----------
Official PyTorch implementation of the paper:
"Physics-Geometry Fusion with Uncertainty-Aware Early Warning for 
Dosimetric Decay in Adaptive Radiotherapy via Implicit Neural Representations"
(Submitted to Medical Image Analysis, 2026)

Key features:
• Topology-preserving deformable registration
• Riemannian dose manifold regularization  
• Lie derivative entropy (ATI) for zero-shot dosimetric risk
• Evidential Deep Learning (EDL) uncertainty estimation
• Red-Yellow-Green clinical decision system


📊 DATASET
-----------
This code uses the Pancreatic-CT-CBCT-SEG dataset from TCIA.

Download:
https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/

Place data in:
./data/Pancreatic-CT-CBCT-SEG/
    ├── Pancreas-CT-CB_001/
    ├── Pancreas-CT-CB_002/
    └── ...


⚙️ INSTALLATION
---------------
Option 1: pip
  pip install -r requirements.txt

Option 2: conda
  conda env create -f environment.yml
  conda activate pi-inr


💻 USAGE
--------
Quick demo (single patient):
  python demo.py --patient Pancreas-CT-CB_001 --quick

Full pipeline:
  python run_pipeline.py
  python run_statistics.py
  python run_visualization.py


📁 OUTPUT STRUCTURE
-------------------
results/MedIA_Ultimate_Run/
├── MedIA_Quantitative_Results.csv        # Main results
├── Pancreas-CT-CB_001/                   # Patient results
│   ├── Warped.nii.gz
│   ├── ATI_Map.nii.gz
│   ├── Uncertainty.nii.gz
│   ├── Disp_X/Y/Z.nii.gz
│   ├── Paper_Figure.png
│   └── Clinical_Report.html
└── MedIA_Final_Excel/                     # Paper-ready outputs
    ├── excel_tables/
    ├── figures/
    └── text/


📝 CITATION
-----------
@article{pi_inr_2026,
  title={Physics-Geometry Fusion with Uncertainty-Aware Early Warning 
         for Dosimetric Decay in Adaptive Radiotherapy via Implicit 
         Neural Representations},
  author={Anonymous Authors},
  journal={Medical Image Analysis (Under Review)},
  year={2026}
}


📜 LICENSE
----------
MIT License - see LICENSE file for details.


⚠️ NOTES
--------
• Python 3.9+, PyTorch 2.0+
• Random seed fixed (42) for reproducibility
• ~3-5 min/patient on CPU, ~30-60 sec on GPU
• Dataset NOT included - download separately from TCIA