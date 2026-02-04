#!/usr/bin/env python3
"""
train_model.py - Train anomaly detection model for supernova precursors

This implements the "Watchlist Approach" from your research document:
- Train unsupervised anomaly detector on "normal" supernova behavior
- Flag unusual patterns that could indicate pre-explosion activity
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = "supernova_features.csv"
MODEL_FILE = "anomaly_detector.pkl"
RESULTS_FILE = "anomaly_scores.csv"

def prepare_data(df):
    """
    Prepare features for ML model
    """
    # Separate metadata from features
    metadata_cols = ['name', 'sn_type']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Get feature matrix
    X = df[feature_cols].values
    
    # Handle any NaN or Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, feature_cols, df[metadata_cols]

def train_anomaly_detector(X, contamination=0.15):
    """
    Train Isolation Forest anomaly detector
    
    contamination: Expected proportion of anomalies (15% = conservative)
    """
    print("\nTraining Isolation Forest anomaly detector...")
    print(f"  - Training samples: {X.shape[0]}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Expected anomaly rate: {contamination*100:.1f}%")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    # This finds objects that are "isolated" in feature space
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled)
    
    # Get anomaly scores (-1 = anomaly, 1 = normal)
    predictions = model.predict(X_scaled)
    
    # Get anomaly scores (lower = more anomalous)
    scores = model.score_samples(X_scaled)
    
    # Invert so higher = more anomalous
    anomaly_scores = -scores
    
    return model, scaler, predictions, anomaly_scores

def analyze_type_specific(df, predictions):
    """
    Analyze anomalies by supernova type
    """
    df_analysis = df.copy()
    df_analysis['is_anomaly'] = (predictions == -1)
    
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION BY TYPE")
    print("=" * 60)
    
    type_summary = df_analysis.groupby('sn_type')['is_anomaly'].agg(['sum', 'count'])
    type_summary['anomaly_rate'] = (type_summary['sum'] / type_summary['count'] * 100).round(1)
    type_summary.columns = ['Anomalies', 'Total', 'Anomaly Rate (%)']
    
    print(type_summary.to_string())
    
    # Highlight Type IIn (should have high anomaly rate due to precursors)
    if 'IIn' in df_analysis['sn_type'].values or 'II-P' in df_analysis['sn_type'].values:
        print("\n📌 KEY INSIGHT:")
        print("   Type IIn SNe show circumstellar interaction (precursor activity)")
        print("   High anomaly rate for Type IIn = model is working!")

def main():
    print("=" * 60)
    print("TRAINING ANOMALY DETECTION MODEL")
    print("=" * 60)
    
    # Load features
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"\n✗ File not found: {INPUT_FILE}")
        print("  Run: python build_features.py first")
        return
    
    print(f"\nLoaded {len(df)} supernovae with {len(df.columns)-2} features")
    
    # Save original dataframe for scoring all objects
    df_original = df.copy()
    
    # OPTIONAL: Filter to only Type II for training
    # This makes Type Ia automatically anomalous
    USE_TYPE_II_FILTER = True  # Set to True to enable filtering
    
    if USE_TYPE_II_FILTER:
        print("\n" + "=" * 60)
        print("FILTERING TRAINING SET TO TYPE II ONLY")
        print("=" * 60)
        print(f"Original sample: {len(df)} supernovae")
        print("\nType distribution (before filtering):")
        print(df['sn_type'].value_counts().to_string())
        
        # Keep only Type II variants for "normal" training set
        type_ii_mask = df['sn_type'].str.contains('II', case=False, na=False)
        df_train = df[type_ii_mask].copy()
        
        print(f"\nTraining sample: {len(df_train)} supernovae (Type II only)")
        print("\nType distribution (training set):")
        print(df_train['sn_type'].value_counts().to_string())
        print("\n→ Will train on Type II, then score ALL supernovae")
        print("=" * 60)
    else:
        df_train = df.copy()
    
    # Prepare training data
    X_train, feature_cols, metadata_train = prepare_data(df_train)
    
    # Prepare ALL data for scoring (including filtered objects)
    X_all, _, metadata_all = prepare_data(df_original)
    
    # Train model on training set (possibly filtered)
    model, scaler, predictions_train, anomaly_scores_train = train_anomaly_detector(X_train, contamination=0.2)
    
    # Score ALL supernovae (including any that were filtered out)
    print("\nScoring all supernovae with trained model...")
    X_all_scaled = scaler.transform(X_all)
    predictions_all = model.predict(X_all_scaled)
    anomaly_scores_all = -model.score_samples(X_all_scaled)
    
    # Create results dataframe with ALL supernovae
    results = metadata_all.copy()
    results['anomaly_score'] = anomaly_scores_all
    results['is_anomaly'] = (predictions_all == -1)
    results['rank'] = results['anomaly_score'].rank(ascending=False)
    
    # Sort by anomaly score
    results = results.sort_values('anomaly_score', ascending=False)
    
    # Save results
    results.to_csv(RESULTS_FILE, index=False)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
    
    # Print results
    print("\n" + "=" * 60)
    print("TOP 10 MOST ANOMALOUS SUPERNOVAE")
    print("=" * 60)
    print(results[['rank', 'name', 'sn_type', 'anomaly_score', 'is_anomaly']].head(10).to_string(index=False))
    
    # Type-specific analysis
    analyze_type_specific(metadata_all, predictions_all)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Anomalies detected: {sum(predictions_all == -1)}/{len(predictions_all)}")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Model saved to: {MODEL_FILE}")
    
    # Check if SN2009ip is flagged (famous precursor case)
    if 'SN2009ip' in results['name'].values:
        sn2009ip_row = results[results['name'] == 'SN2009ip'].iloc[0]
        print("\n" + "=" * 60)
        print("VALIDATION CHECK: SN 2009ip (Known Precursor Case)")
        print("=" * 60)
        print(f"  Anomaly Score: {sn2009ip_row['anomaly_score']:.3f}")
        print(f"  Flagged as Anomaly: {sn2009ip_row['is_anomaly']}")
        print(f"  Rank: #{int(sn2009ip_row['rank'])} / {len(results)}")
        
        if sn2009ip_row['is_anomaly']:
            print("  ✓ SUCCESS! Model correctly identified known precursor!")
        else:
            print("  ⚠ Model did not flag this case (may need tuning)")
    
    print("\n✓ Ready to visualize: python visualize.py")

if __name__ == "__main__":
    main()
