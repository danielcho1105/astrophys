#!/usr/bin/env python3
"""
score_new_supernovae.py - Score new supernovae with your trained model

This script takes NEW supernova names, downloads their data,
extracts features, and scores them with your EXISTING trained model
(without retraining).

Usage:
    python3 score_new_supernovae.py
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import pickle
from scipy import stats
import time

# New supernovae to score
NEW_SUPERNOVAE = [
    "SN2023ixf",   # Recent Type II in M101 (very well observed!)
    "SN2022jli",   # Type Ic
    "SN2021yja",   # Type Ia
    "SN2020tlf",   # Type II with possible precursor
    # Add more names here!
]

DATA_DIR = "data"
MODEL_FILE = "anomaly_detector.pkl"
OUTPUT_FILE = "new_supernova_scores.csv"

def download_supernova(sn_name):
    """Download a supernova JSON file from OSC"""
    year_ranges = [
        "sne-2020-2024",
        "sne-2015-2019", 
        "sne-2010-2014",
        "sne-2005-2009",
    ]
    
    for year_range in year_ranges:
        url = f"https://raw.githubusercontent.com/astrocatalogs/{year_range}/master/{sn_name}.json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                filepath = os.path.join(DATA_DIR, f"{sn_name}.json")
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                return True
        except:
            continue
    
    return False

def parse_photometry(sn_data, sn_name):
    """Extract photometry from JSON"""
    if sn_name not in sn_data:
        return None
    
    event = sn_data[sn_name]
    
    if 'photometry' not in event:
        return None
    
    times, mags, bands = [], [], []
    
    for obs in event['photometry']:
        if 'time' not in obs or 'magnitude' not in obs:
            continue
        
        try:
            times.append(float(obs['time']))
            mags.append(float(obs['magnitude']))
            bands.append(obs.get('band', 'unknown'))
        except (ValueError, TypeError):
            continue
    
    if len(times) < 5:
        return None
    
    return {
        'times': np.array(times),
        'magnitudes': np.array(mags),
        'bands': np.array(bands)
    }

def compute_features(times, mags):
    """Compute the same 16 features used in training"""
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    mags = mags[sort_idx]
    
    features = {}
    
    # Basic statistics
    features['mean_mag'] = np.mean(mags)
    features['median_mag'] = np.median(mags)
    features['std_mag'] = np.std(mags)
    features['mag_range'] = np.max(mags) - np.min(mags)
    features['amplitude'] = np.percentile(mags, 95) - np.percentile(mags, 5)
    features['skewness'] = stats.skew(mags)
    features['kurtosis'] = stats.kurtosis(mags)
    
    # Time-based
    features['n_observations'] = len(mags)
    features['time_span'] = np.max(times) - np.min(times)
    features['median_cadence'] = np.median(np.diff(times)) if len(times) > 1 else 0
    
    # Linear trend
    if len(times) > 2:
        slope, intercept, r_value, _, _ = stats.linregress(times, mags)
        features['linear_trend'] = slope
        features['trend_strength'] = r_value ** 2
    else:
        features['linear_trend'] = 0
        features['trend_strength'] = 0
    
    # Variability
    if len(mags) > 1:
        mag_diffs = np.abs(np.diff(mags))
        features['mean_abs_diff'] = np.mean(mag_diffs)
        features['max_abs_diff'] = np.max(mag_diffs)
        features['std_abs_diff'] = np.std(mag_diffs)
    else:
        features['mean_abs_diff'] = 0
        features['max_abs_diff'] = 0
        features['std_abs_diff'] = 0
    
    features['beyond_1st_magnitude'] = np.sum(mags > 1)
    
    # Autocorrelation
    if len(mags) > 10:
        autocorr = np.corrcoef(mags[:-1], mags[1:])[0, 1]
        features['autocorr'] = autocorr if not np.isnan(autocorr) else 0
    else:
        features['autocorr'] = 0
    
    return features

def get_sn_type(sn_data, sn_name):
    """Extract supernova type"""
    if sn_name not in sn_data:
        return "Unknown"
    
    event = sn_data[sn_name]
    
    if 'claimedtype' in event:
        claimed = event['claimedtype']
        if isinstance(claimed, list) and len(claimed) > 0:
            return claimed[0].get('value', 'Unknown')
    
    return "Unknown"

def main():
    print("=" * 60)
    print("SCORING NEW SUPERNOVAE WITH TRAINED MODEL")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        print(f"\n✗ Model not found: {MODEL_FILE}")
        print("  Run: python3 train_model.py first to train a model")
        return
    
    # Load trained model
    print("\nLoading trained model...")
    with open(MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
    
    print(f"✓ Model loaded ({len(feature_cols)} features)")
    
    # Download new supernovae
    print(f"\nDownloading {len(NEW_SUPERNOVAE)} new supernovae...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    downloaded = []
    for sn_name in NEW_SUPERNOVAE:
        print(f"  Downloading {sn_name}...", end=" ")
        if download_supernova(sn_name):
            print("✓")
            downloaded.append(sn_name)
        else:
            print("✗ Not found")
        time.sleep(0.5)
    
    if not downloaded:
        print("\n✗ No supernovae downloaded successfully")
        return
    
    # Extract features
    print(f"\nExtracting features from {len(downloaded)} supernovae...")
    new_features = []
    
    for sn_name in downloaded:
        filepath = os.path.join(DATA_DIR, f"{sn_name}.json")
        
        try:
            with open(filepath, 'r') as f:
                sn_data = json.load(f)
            
            phot = parse_photometry(sn_data, sn_name)
            if phot is None:
                print(f"  ✗ {sn_name}: No valid photometry")
                continue
            
            features = compute_features(phot['times'], phot['magnitudes'])
            features['name'] = sn_name
            features['sn_type'] = get_sn_type(sn_data, sn_name)
            
            new_features.append(features)
            print(f"  ✓ {sn_name}: {len(phot['times'])} observations")
            
        except Exception as e:
            print(f"  ✗ {sn_name}: {e}")
    
    if not new_features:
        print("\n✗ No valid features extracted")
        return
    
    # Create DataFrame
    df_new = pd.DataFrame(new_features)
    
    # Make sure columns match training data
    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0
    
    # Get feature matrix in correct order
    X_new = df_new[feature_cols].values
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Score with trained model
    print("\nScoring with trained model...")
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    anomaly_scores = -model.score_samples(X_new_scaled)
    
    # Create results
    results = df_new[['name', 'sn_type']].copy()
    results['anomaly_score'] = anomaly_scores
    results['is_anomaly'] = (predictions == -1)
    results['rank'] = results['anomaly_score'].rank(ascending=False, method='min')
    
    # Sort by score
    results = results.sort_values('anomaly_score', ascending=False)
    
    # Save
    results.to_csv(OUTPUT_FILE, index=False)
    
    # Display results
    print("\n" + "=" * 60)
    print("NEW SUPERNOVA RANKINGS")
    print("=" * 60)
    print(results.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("COMPARISON TO EXISTING DATA")
    print("=" * 60)
    
    # Load original scores for comparison
    if os.path.exists("anomaly_scores.csv"):
        df_original = pd.read_csv("anomaly_scores.csv")
        
        print("\nHow new supernovae compare to your training set:")
        print(f"  Highest original score: {df_original['anomaly_score'].max():.3f}")
        print(f"  Lowest original score:  {df_original['anomaly_score'].min():.3f}")
        print(f"  Original threshold:     ~{df_original['anomaly_score'].quantile(0.80):.3f} (top 20%)")
        
        print(f"\nNew supernovae:")
        for _, row in results.iterrows():
            percentile = (df_original['anomaly_score'] < row['anomaly_score']).sum() / len(df_original) * 100
            print(f"  {row['name']:15s} score={row['anomaly_score']:.3f}  " + 
                  f"(would rank in top {100-percentile:.0f}% of original data)")
    
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    print("\nInterpretation:")
    print("  • High anomaly score = Unusual behavior (potential precursor)")
    print("  • is_anomaly=TRUE = Flagged by model")
    print("  • Compare scores to your original data to see relative ranking")

if __name__ == "__main__":
    main()
