#!/usr/bin/env python3
"""
run_all.py - Execute complete ML pipeline in one command

This runs all 4 steps in sequence:
1. Download data
2. Extract features
3. Train model
4. Generate visualizations
"""

import subprocess
import sys
import time

def run_script(script_name, description):
    """
    Run a Python script and report status
    """
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Running: {script_name}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed!")
        print(f"Error: {e}")
        return False

def main():
    print("=" * 70)
    print("SUPERNOVA PRECURSOR DETECTION - COMPLETE PIPELINE")
    print("=" * 70)
    print("\nThis will run the full ML pipeline:")
    print("  1. Download supernova data from OSC")
    print("  2. Extract light curve features")
    print("  3. Train anomaly detection model")
    print("  4. Generate visualizations")
    print("\nEstimated time: 3-5 minutes")
    print("=" * 70)
    
    input("\nPress ENTER to start...")
    
    pipeline = [
        ("get_data.py", "Download Data"),
        ("build_features.py", "Extract Features"),
        ("train_model.py", "Train Model"),
        ("visualize.py", "Generate Visualizations")
    ]
    
    total_start = time.time()
    
    for script, description in pipeline:
        success = run_script(script, description)
        if not success:
            print("\n" + "=" * 70)
            print("PIPELINE FAILED")
            print("=" * 70)
            print(f"Error occurred during: {description}")
            print(f"You can run '{script}' manually to debug")
            sys.exit(1)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE! 🎉")
    print("=" * 70)
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print("\n📊 Check your results:")
    print("  - Data files: ./data/")
    print("  - Features: supernova_features.csv")
    print("  - Predictions: anomaly_scores.csv")
    print("  - Visualizations: ./plots/")
    print("\n🔬 Your ML model has:")
    print("  ✓ Downloaded real supernova data")
    print("  ✓ Extracted 16 time-series features")
    print("  ✓ Trained anomaly detector")
    print("  ✓ Identified unusual behavior")
    print("  ✓ Generated publication-quality plots")
    print("\n💡 Next: Open plots/ folder to see visualizations!")
    print("=" * 70)

if __name__ == "__main__":
    main()
