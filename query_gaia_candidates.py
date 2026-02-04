#!/usr/bin/env python3
"""
query_gaia_candidates.py - Find and score Red Supergiant candidates from Gaia

This script queries the Gaia DR3 database for Red Supergiant stars,
downloads their light curves, extracts features, and scores them with
your trained model to identify potential pre-supernova candidates.

Requirements:
    pip install astroquery

Usage:
    python3 query_gaia_candidates.py
"""

import numpy as np
import pandas as pd
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from astroquery.gaia import Gaia
    GAIA_AVAILABLE = True
except ImportError:
    GAIA_AVAILABLE = False
    print("⚠️  astroquery not installed. Install with: pip install astroquery")

OUTPUT_FILE = "gaia_candidates.csv"
MODEL_FILE = "anomaly_detector.pkl"

def query_red_supergiant_candidates(max_results=50):
    """
    Query Gaia DR3 for Red Supergiant candidates
    """
    print("\nQuerying Gaia DR3 for Red Supergiant candidates...")
    print("Criteria:")
    print("  - Temperature < 4000 K (red/cool)")
    print("  - Radius > 100 R☉ (very large)")
    print("  - Variable (shows brightness changes)")
    print("  - Valid parallax (distance measurement)")
    
    # ADQL query for Gaia
    query = f"""
    SELECT TOP {max_results}
        source_id,
        ra, dec,
        phot_g_mean_mag,
        phot_variable_flag,
        teff_gspphot AS temperature,
        radius_gspphot AS radius,
        lum_gspphot AS luminosity,
        distance_gspphot AS distance,
        radial_velocity
    FROM gaiadr3.gaia_source
    WHERE 
        teff_gspphot < 4000
        AND radius_gspphot > 100
        AND phot_variable_flag = 'VARIABLE'
        AND parallax > 0
        AND phot_g_mean_mag < 15
    ORDER BY radius_gspphot DESC
    """
    
    job = Gaia.launch_job(query)
    results = job.get_results()
    
    print(f"\n✓ Found {len(results)} Red Supergiant candidates")
    
    return results.to_pandas()

def query_light_curve(source_id):
    """
    Get time-series photometry for a specific Gaia source
    """
    query = f"""
    SELECT 
        time,
        mag,
        flux
    FROM gaiadr3.gaia_source_time_series
    WHERE source_id = {source_id}
    ORDER BY time
    """
    
    try:
        job = Gaia.launch_job(query)
        results = job.get_results()
        
        if len(results) < 5:
            return None
        
        return results.to_pandas()
    except:
        return None

def compute_features_from_gaia(times, mags):
    """
    Compute same 16 features from Gaia light curve
    (Compatible with your trained model)
    """
    if len(times) < 5:
        return None
    
    # Sort by time
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

def main():
    print("=" * 70)
    print("GAIA RED SUPERGIANT PRECURSOR SEARCH")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Query Gaia DR3 for Red Supergiant candidates")
    print("  2. Download their light curves")
    print("  3. Score them with your trained model")
    print("  4. Identify potential pre-supernova stars")
    
    # Check dependencies
    if not GAIA_AVAILABLE:
        print("\n" + "=" * 70)
        print("SETUP REQUIRED")
        print("=" * 70)
        print("\nTo use this script, install astroquery:")
        print("  pip install astroquery")
        print("\nThen run this script again.")
        return
    
    # Check for trained model
    try:
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
        print(f"\n✓ Loaded trained model ({len(feature_cols)} features)")
    except FileNotFoundError:
        print(f"\n✗ Model not found: {MODEL_FILE}")
        print("  Run 'python3 train_model.py' first to train a model")
        return
    
    # Query Gaia for candidates
    try:
        candidates = query_red_supergiant_candidates(max_results=20)
    except Exception as e:
        print(f"\n✗ Error querying Gaia: {e}")
        print("\nTroubleshooting:")
        print("  - Check internet connection")
        print("  - Gaia archive might be down (try later)")
        print("  - Update astroquery: pip install --upgrade astroquery")
        return
    
    print("\n" + "=" * 70)
    print("TOP RED SUPERGIANT CANDIDATES")
    print("=" * 70)
    print(candidates[['source_id', 'temperature', 'radius', 'luminosity', 'distance']].head(10).to_string(index=False))
    
    # Download light curves and score
    print("\n" + "=" * 70)
    print("DOWNLOADING LIGHT CURVES AND SCORING")
    print("=" * 70)
    
    scored_candidates = []
    
    for idx, row in candidates.iterrows():
        source_id = row['source_id']
        print(f"\n[{idx+1}/{len(candidates)}] Gaia Source {source_id}")
        print(f"  Temperature: {row['temperature']:.0f} K")
        print(f"  Radius: {row['radius']:.1f} R☉")
        
        # Get light curve
        print("  Downloading light curve...", end=" ")
        lc = query_light_curve(source_id)
        
        if lc is None or len(lc) < 5:
            print("✗ Insufficient data")
            continue
        
        print(f"✓ {len(lc)} observations")
        
        # Extract features
        times = lc['time'].values
        mags = lc['mag'].values
        
        features = compute_features_from_gaia(times, mags)
        
        if features is None:
            print("  ✗ Could not compute features")
            continue
        
        # Score with model
        # Ensure feature order matches training
        feature_vector = np.array([features.get(col, 0) for col in feature_cols]).reshape(1, -1)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale and predict
        X_scaled = scaler.transform(feature_vector)
        prediction = model.predict(X_scaled)[0]
        anomaly_score = -model.score_samples(X_scaled)[0]
        
        is_anomaly = (prediction == -1)
        
        print(f"  Anomaly Score: {anomaly_score:.3f} {'🚨 FLAGGED!' if is_anomaly else '✓ Normal'}")
        
        # Save result
        result = {
            'source_id': source_id,
            'ra': row['ra'],
            'dec': row['dec'],
            'temperature': row['temperature'],
            'radius': row['radius'],
            'luminosity': row['luminosity'],
            'distance': row['distance'],
            'n_observations': len(lc),
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly
        }
        
        scored_candidates.append(result)
        
        # Limit to avoid timeout
        if idx >= 9:  # Score first 10 for now
            print("\n  (Limiting to first 10 candidates for demo)")
            break
    
    if not scored_candidates:
        print("\n✗ No candidates could be scored")
        return
    
    # Create results DataFrame
    df_results = pd.DataFrame(scored_candidates)
    df_results['rank'] = df_results['anomaly_score'].rank(ascending=False)
    df_results = df_results.sort_values('anomaly_score', ascending=False)
    
    # Save results
    df_results.to_csv(OUTPUT_FILE, index=False)
    
    # Display results
    print("\n" + "=" * 70)
    print("🎯 TOP PRECURSOR CANDIDATES FROM GAIA")
    print("=" * 70)
    
    display_cols = ['rank', 'source_id', 'temperature', 'radius', 'anomaly_score', 'is_anomaly']
    print(df_results[display_cols].head(10).to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total candidates analyzed: {len(df_results)}")
    print(f"Flagged as anomalous: {df_results['is_anomaly'].sum()}")
    print(f"Anomaly rate: {df_results['is_anomaly'].sum()/len(df_results)*100:.1f}%")
    print(f"\nHighest anomaly score: {df_results['anomaly_score'].max():.3f}")
    print(f"Mean anomaly score: {df_results['anomaly_score'].mean():.3f}")
    
    # Comparison to training data
    print("\n" + "=" * 70)
    print("COMPARISON TO SUPERNOVA TRAINING DATA")
    print("=" * 70)
    
    try:
        df_sn = pd.read_csv("anomaly_scores.csv")
        sn_threshold = df_sn['anomaly_score'].quantile(0.80)
        
        print(f"Your supernova training set threshold (top 20%): {sn_threshold:.3f}")
        print(f"\nGaia stars exceeding this threshold:")
        
        high_priority = df_results[df_results['anomaly_score'] > sn_threshold]
        if len(high_priority) > 0:
            print(f"  🚨 {len(high_priority)} GAIA STARS SHOW PRECURSOR-LIKE BEHAVIOR!")
            print("\nHigh-priority watchlist:")
            for _, row in high_priority.iterrows():
                print(f"  • Gaia {row['source_id']}: score={row['anomaly_score']:.3f}")
        else:
            print("  None - all Gaia candidates show more normal behavior")
            
    except FileNotFoundError:
        pass
    
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review high-scoring candidates in detail")
    print("2. Cross-reference with SIMBAD/VizieR for known classifications")
    print("3. Request spectroscopic follow-up for top candidates")
    print("4. Monitor these stars with ZTF or other surveys")
    print("5. PUBLISH if any explode after being on your watchlist!")
    print("\nYou're now doing REAL supernova prediction research! 🌟")

if __name__ == "__main__":
    main()
