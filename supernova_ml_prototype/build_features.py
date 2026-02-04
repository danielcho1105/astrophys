#!/usr/bin/env python3
"""
build_features.py - Extract light curves and compute ML features

This script parses OSC JSON files, extracts photometry (light curves),
and computes statistical features for anomaly detection.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
OUTPUT_FILE = "supernova_features.csv"

def parse_photometry(sn_data, sn_name):
    """
    Extract photometry (time-series brightness measurements) from JSON
    """
    if sn_name not in sn_data:
        return None
    
    event = sn_data[sn_name]
    
    # Look for photometry data
    if 'photometry' not in event:
        return None
    
    photometry = event['photometry']
    
    # Parse into structured format
    times = []
    magnitudes = []
    bands = []
    
    for obs in photometry:
        # Must have time and magnitude
        if 'time' not in obs or 'magnitude' not in obs:
            continue
        
        try:
            time = float(obs['time'])
            mag = float(obs['magnitude'])
            band = obs.get('band', 'unknown')
            
            times.append(time)
            magnitudes.append(mag)
            bands.append(band)
            
        except (ValueError, TypeError):
            continue
    
    if len(times) < 5:  # Need at least 5 points
        return None
    
    return {
        'times': np.array(times),
        'magnitudes': np.array(magnitudes),
        'bands': np.array(bands)
    }

def compute_variability_features(times, mags):
    """
    Compute time-series variability features
    Based on LAISS paper feature extraction
    """
    features = {}
    
    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    mags = mags[sort_idx]
    
    # Basic statistics
    features['mean_mag'] = np.mean(mags)
    features['median_mag'] = np.median(mags)
    features['std_mag'] = np.std(mags)
    features['mag_range'] = np.max(mags) - np.min(mags)
    
    # Variability amplitude
    features['amplitude'] = np.percentile(mags, 95) - np.percentile(mags, 5)
    
    # Skewness and kurtosis (shape of distribution)
    features['skewness'] = stats.skew(mags)
    features['kurtosis'] = stats.kurtosis(mags)
    
    # Time-based features
    features['n_observations'] = len(mags)
    features['time_span'] = np.max(times) - np.min(times)
    features['median_cadence'] = np.median(np.diff(times)) if len(times) > 1 else 0
    
    # Linear trend (is it getting brighter or dimmer?)
    if len(times) > 2:
        slope, intercept, r_value, _, _ = stats.linregress(times, mags)
        features['linear_trend'] = slope
        features['trend_strength'] = r_value ** 2
    else:
        features['linear_trend'] = 0
        features['trend_strength'] = 0
    
    # Magnitude differences (how erratic is the light curve?)
    if len(mags) > 1:
        mag_diffs = np.abs(np.diff(mags))
        features['mean_abs_diff'] = np.mean(mag_diffs)
        features['max_abs_diff'] = np.max(mag_diffs)
        features['std_abs_diff'] = np.std(mag_diffs)
    else:
        features['mean_abs_diff'] = 0
        features['max_abs_diff'] = 0
        features['std_abs_diff'] = 0
    
    # Beyond 1st magnitude (faint objects)
    features['beyond_1st_magnitude'] = np.sum(mags > 1)
    
    # Autocorrelation (periodic behavior)
    if len(mags) > 10:
        # Simple lag-1 autocorrelation
        autocorr = np.corrcoef(mags[:-1], mags[1:])[0, 1]
        features['autocorr'] = autocorr if not np.isnan(autocorr) else 0
    else:
        features['autocorr'] = 0
    
    return features

def get_sn_type(sn_data, sn_name):
    """
    Extract supernova classification from JSON
    """
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
    print("EXTRACTING FEATURES FROM SUPERNOVA LIGHT CURVES")
    print("=" * 60)
    
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    
    if not json_files:
        print(f"\n✗ No JSON files found in {DATA_DIR}/")
        print("  Run: python get_data.py first")
        return
    
    print(f"\nFound {len(json_files)} supernova files")
    print("Extracting photometry and computing features...\n")
    
    all_features = []
    
    for json_file in tqdm(json_files, desc="Processing"):
        filepath = os.path.join(DATA_DIR, json_file)
        sn_name = json_file.replace('.json', '')
        
        try:
            with open(filepath, 'r') as f:
                sn_data = json.load(f)
            
            # Extract light curve
            phot = parse_photometry(sn_data, sn_name)
            
            if phot is None:
                continue
            
            # Compute features
            features = compute_variability_features(phot['times'], phot['magnitudes'])
            features['name'] = sn_name
            features['sn_type'] = get_sn_type(sn_data, sn_name)
            
            all_features.append(features)
            
        except Exception as e:
            print(f"Error processing {sn_name}: {e}")
            continue
    
    if not all_features:
        print("\n✗ No valid light curves extracted")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns
    cols = ['name', 'sn_type'] + [c for c in df.columns if c not in ['name', 'sn_type']]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'=' * 60}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed: {len(all_features)} supernovae")
    print(f"Features per object: {len(df.columns) - 2}")
    print(f"Output saved to: {OUTPUT_FILE}")
    
    # Show summary statistics
    print(f"\n{'=' * 60}")
    print("SUPERNOVA TYPE DISTRIBUTION")
    print(f"{'=' * 60}")
    print(df['sn_type'].value_counts().to_string())
    
    print(f"\n✓ Ready to run: python train_model.py")

if __name__ == "__main__":
    main()
