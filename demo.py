"""
PI-INR Demo Script
Quick start guide for running the pipeline on a single patient
Version: 1.0.0
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='PI-INR Demo - Quick Start')
    parser.add_argument('--patient', type=str, default='Pancreas-CT-CB_001',
                        help='Patient ID to process (default: Pancreas-CT-CB_001)')
    parser.add_argument('--quick', action='store_true',
                        help='Run with reduced epochs for quick test (100 epochs)')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip statistics generation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PI-INR Demo - Quick Start v1.0.0")
    print("="*60)
    
    # Check if data exists
    data_path = f"./data/Pancreatic-CT-CBCT-SEG/{args.patient}"
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        print("\nPlease download the Pancreatic-CT-CBCT-SEG dataset from TCIA:")
        print("  https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/")
        print("\nAnd place it in: ./data/Pancreatic-CT-CBCT-SEG/")
        return
    
    print(f"Data found: {data_path}")
    
    # Modify config for quick run
    if args.quick:
        print("Quick mode: reducing epochs to 100")
        os.environ['PI_INR_EPOCHS'] = '100'
    
    print(f"\nProcessing patient: {args.patient}")
    print("   This may take 2-5 minutes on CPU...")
    
    # Run pipeline
    result = subprocess.run([
        sys.executable, 'run_pipeline.py',
        '--max-patients', '1',
        '--force-rerun'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Pipeline failed:")
        print(result.stderr)
        return
    
    print("Pipeline completed successfully!")
    
    # Check results
    csv_path = "./results/MedIA_Ultimate_Run_Final/MedIA_Quantitative_Results.csv"
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        patient_data = df[df['Patient_ID'] == args.patient]
        if len(patient_data) > 0:
            row = patient_data.iloc[0]
            print("\nResults:")
            print(f"   SSIM: {row['SSIM_Before']:.4f} -> {row['SSIM_After']:.4f}")
            print(f"   ATI Risk: {row['High_Risk_Ratio']:.5f}%")
            print(f"   Uncertainty: {row['Mean_Uncert_Risk']:.5f}")
            print(f"   Decision: {row['Decision']}")
    
    # Generate statistics (if not skipped)
    if not args.no_stats:
        print("\nGenerating summary statistics...")
        subprocess.run([sys.executable, 'run_statistics.py'])
        
        print("\nGenerating figures...")
        subprocess.run([sys.executable, 'run_visualization.py'])
    
    print("\n" + "="*60)
    print(f"Demo completed! Results saved in: ./results/MedIA_Ultimate_Run_Final/")
    print("="*60)

if __name__ == "__main__":
    main()
