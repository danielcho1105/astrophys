#!/usr/bin/env python3
"""
visualize.py - Visualize anomaly detection results

Creates plots showing:
1. Anomaly score distribution
2. Feature space visualization (PCA)
3. Type-specific anomaly rates
4. Example light curves of anomalous cases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

INPUT_FILE = "supernova_features.csv"
RESULTS_FILE = "anomaly_scores.csv"
DATA_DIR = "data"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_anomaly_distribution(results):
    """
    Plot distribution of anomaly scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of anomaly scores
    ax1.hist(results['anomaly_score'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(results[results['is_anomaly']]['anomaly_score'].min(), 
                color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
    ax1.set_xlabel('Anomaly Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot by anomaly status
    sns.boxplot(data=results, x='is_anomaly', y='anomaly_score', ax=ax2)
    ax2.set_xlabel('Anomaly Flag', fontsize=12)
    ax2.set_ylabel('Anomaly Score', fontsize=12)
    ax2.set_title('Anomaly Scores: Normal vs Anomalous', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['Normal', 'Anomaly'])
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/anomaly_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/anomaly_distribution.png")
    plt.close()

def plot_pca_visualization(df, results):
    """
    Visualize anomalies in 2D PCA space
    """
    # Get features
    metadata_cols = ['name', 'sn_type']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize and reduce to 2D
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot normal points
    normal_mask = ~results['is_anomaly'].values
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
              c='blue', alpha=0.5, s=100, label='Normal', edgecolors='black')
    
    # Plot anomalies
    anomaly_mask = results['is_anomaly'].values
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
              c='red', alpha=0.8, s=200, marker='*', label='Anomaly', edgecolors='black')
    
    # Label top anomalies
    top_anomalies = results.nlargest(5, 'anomaly_score')
    for idx, row in top_anomalies.iterrows():
        pos = X_pca[idx]
        ax.annotate(row['name'], (pos[0], pos[1]), 
                   fontsize=9, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Supernova Feature Space (PCA Projection)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pca_visualization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/pca_visualization.png")
    plt.close()

def plot_type_distribution(results):
    """
    Plot anomaly rates by supernova type
    """
    # Count by type
    type_counts = results.groupby('sn_type').agg({
        'is_anomaly': ['sum', 'count']
    })
    type_counts.columns = ['anomalies', 'total']
    type_counts['anomaly_rate'] = (type_counts['anomalies'] / type_counts['total'] * 100)
    type_counts = type_counts.sort_values('anomaly_rate', ascending=False)
    
    # Only show types with at least 2 objects
    type_counts = type_counts[type_counts['total'] >= 2]
    
    if len(type_counts) == 0:
        print("  ⚠ Not enough data for type distribution plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red' if 'IIn' in idx or 'II-P' in idx else 'steelblue' 
              for idx in type_counts.index]
    
    bars = ax.bar(range(len(type_counts)), type_counts['anomaly_rate'], color=colors, edgecolor='black')
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax.set_title('Anomaly Detection Rate by Supernova Type', fontsize=14, fontweight='bold')
    ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Expected Rate (20%)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add count labels
    for i, (idx, row) in enumerate(type_counts.iterrows()):
        ax.text(i, row['anomaly_rate'] + 2, f"n={int(row['total'])}", 
               ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/type_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/type_distribution.png")
    plt.close()

def plot_example_light_curve(sn_name):
    """
    Plot light curve for a specific supernova
    """
    filepath = os.path.join(DATA_DIR, f"{sn_name}.json")
    
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r') as f:
            sn_data = json.load(f)
        
        if sn_name not in sn_data:
            return False
        
        event = sn_data[sn_name]
        
        if 'photometry' not in event:
            return False
        
        # Extract photometry
        times = []
        mags = []
        bands = []
        
        for obs in event['photometry']:
            if 'time' in obs and 'magnitude' in obs:
                try:
                    times.append(float(obs['time']))
                    mags.append(float(obs['magnitude']))
                    bands.append(obs.get('band', 'V'))
                except (ValueError, TypeError):
                    continue
        
        if len(times) < 5:
            return False
        
        # Convert to arrays
        times = np.array(times)
        mags = np.array(mags)
        bands = np.array(bands)
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        mags = mags[sort_idx]
        bands = bands[sort_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Different colors for different bands
        unique_bands = np.unique(bands)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
        
        for band, color in zip(unique_bands, colors):
            mask = bands == band
            if np.sum(mask) > 0:
                ax.scatter(times[mask], mags[mask], label=f'{band} band', 
                          alpha=0.6, s=50, c=[color], edgecolors='black')
                ax.plot(times[mask], mags[mask], c=color, alpha=0.3, linestyle='-')
        
        # Invert y-axis (brighter = lower magnitude in astronomy)
        ax.invert_yaxis()
        
        # Add secondary y-axis showing flux (more physical than magnitude)
        ax3 = ax.twinx()
        
        # Convert magnitude to relative flux (arbitrary zero-point)
        # F/F0 = 10^(-mag/2.5), normalized to brightest observation
        mag_min = np.min(mags)
        flux_relative = 10**((mag_min - mags) / 2.5)
        
        # Set flux axis limits based on magnitude limits
        mag_lims = ax.get_ylim()
        flux_lims = (10**((mag_min - mag_lims[0]) / 2.5), 
                     10**((mag_min - mag_lims[1]) / 2.5))
        ax3.set_ylim(flux_lims)
        ax3.set_ylabel('Relative Flux [F/F₀]\n(normalized to peak)', 
                      fontsize=11, color='darkgreen')
        ax3.tick_params(axis='y', labelcolor='darkgreen')
        
        # Add secondary x-axis showing days from first observation
        ax2 = ax.twiny()
        first_time = np.min(times)
        ax2.set_xlim(ax.get_xlim())
        days_span = np.max(times) - first_time
        ax2.set_xlabel('Days Since First Observation', fontsize=12, color='darkblue')
        ax2.tick_params(axis='x', labelcolor='darkblue')
        
        # Format the ticks on secondary axis
        original_ticks = ax.get_xticks()
        days_ticks = original_ticks - first_time
        ax2.set_xticks(original_ticks)
        ax2.set_xticklabels([f'{int(d)}' for d in days_ticks])
        
        # Primary axis labels with scientific context
        ax.set_xlabel('Modified Julian Date (MJD)\n[Standard astronomical time]', fontsize=11)
        ax.set_ylabel('Apparent Magnitude [mag]\n(fainter → higher values)', fontsize=11)
        
        # Enhanced title with context
        ax.set_title(f'Photometric Light Curve: {sn_name}\n' + 
                    f'Temporal Evolution of Brightness ({len(times)} observations)', 
                    fontsize=13, fontweight='bold')
        
        # Add grid and legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Add text box with scientific info
        time_span_days = np.max(times) - np.min(times)
        mag_range = np.max(mags) - np.min(mags)
        flux_ratio = 10**(mag_range / 2.5)
        info_text = (f'Monitoring: {time_span_days:.0f} days\n'
                    f'Δmag: {mag_range:.2f}\n'
                    f'Flux variation: {flux_ratio:.1f}×')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/lightcurve_{sn_name}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/lightcurve_{sn_name}.png")
        plt.close()
        
        return True
        
    except Exception as e:
        return False

def main():
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Load data
    try:
        df = pd.read_csv(INPUT_FILE)
        results = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}")
        print("  Run: python train_model.py first")
        return
    
    print(f"\nCreating plots in {OUTPUT_DIR}/...\n")
    
    # 1. Anomaly distribution
    plot_anomaly_distribution(results)
    
    # 2. PCA visualization
    plot_pca_visualization(df, results)
    
    # 3. Type distribution
    plot_type_distribution(results)
    
    # 4. Example light curves for top anomalies
    print("\nGenerating light curve plots for top anomalies...")
    top_anomalies = results.nlargest(5, 'anomaly_score')
    
    lc_count = 0
    for _, row in top_anomalies.iterrows():
        if plot_example_light_curve(row['name']):
            lc_count += 1
    
    print(f"✓ Generated {lc_count} light curve plots")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated plots:")
    print("  1. anomaly_distribution.png - Score distributions")
    print("  2. pca_visualization.png - 2D feature space")
    print("  3. type_distribution.png - Anomaly rates by type")
    print("  4. lightcurve_*.png - Individual light curves")
    
    print("\n" + "=" * 60)
    print("PROTOTYPE COMPLETE!")
    print("=" * 60)
    print("\nYour ML pipeline has:")
    print("  ✓ Downloaded real supernova data")
    print("  ✓ Extracted time-series features")
    print("  ✓ Trained anomaly detection model")
    print("  ✓ Identified unusual supernova behavior")
    print("  ✓ Generated publication-quality plots")
    print("\nNext steps:")
    print("  - Review plots to understand results")
    print("  - Check anomaly_scores.csv for full results")
    print("  - Tune model parameters in train_model.py")
    print("  - Scale to ZTF/Gaia live data (Option B)")

if __name__ == "__main__":
    main()
