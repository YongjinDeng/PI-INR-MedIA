PI-INR: Physics-Geometry Fusion for Adaptive Radiotherapy
Version: 1.0.0
============================================================

OVERVIEW
--------
Official PyTorch implementation of PI-INR for real-time dosimetric
decay warning in Adaptive Radiotherapy (ART).

Key features:
- Topology-preserving deformable registration
- Riemannian dose manifold regularization
- Lie derivative entropy (ATI) for zero-shot risk assessment
- Evidential Deep Learning uncertainty estimation
- Red-Yellow-Green clinical decision system

DATASET
-------
Download Pancreatic-CT-CBCT-SEG from TCIA:
https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/

Place data in: ./data/Pancreatic-CT-CBCT-SEG/

INSTALLATION
------------
pip install -r requirements.txt

or

conda env create -f environment.yml
conda activate pi-inr

USAGE
-----
Quick demo:
  python demo.py --patient Pancreas-CT-CB_001 --quick

Full pipeline:
  python run_pipeline.py
  python run_statistics.py
  python run_visualization.py

OUTPUT
------
Results saved to: ./results/MedIA_Ultimate_Run_Final/
Figures saved to: ./results/MedIA_Ultimate_Run_Final/MedIA_Paper_Figures/

LICENSE
-------
MIT License
