#!/usr/bin/env python3
"""
get_data.py - Download supernova light curve data from Open Supernova Catalog

This script downloads a sample of well-documented supernovae with good photometry
from the OSC GitHub repositories. We focus on Type IIn SNe (like SN 2009ip) which
show precursor activity.
"""

import os
import json
import requests
from tqdm import tqdm
import time

# Create data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# List of interesting supernovae with known precursor activity
# These are from the 2005-2024 repos and include Type IIn SNe
SAMPLE_SUPERNOVAE = [
    # Famous precursor cases
    "SN2009ip",
    "SN2010mc", 
    "SN2015bh",
    "SN2006jc",
    "SN2010jl",
    
    # Type IIn with circumstellar interaction
    "SN2005ip",
    "SN2006bo",
    "SN2006gy",
    "SN2008fq",
    "SN2010al",
    "SN2011ht",
    "SN2013L",
    "SN2014G",
    "SN2015da",
    
    # Type II for comparison
    "SN2004et",
    "SN2012aw",
    "SN2013ej",
    "SN2017eaw",
    
    # Type Ia for "normal" comparison
    "SN2011fe",
    "SN2014J",
    
    # Recent well-studied cases
    "SN2016bdu",
    "SN2019zrk",
    "SN2020jfo",
]

def download_supernova(sn_name):
    """
    Download a single supernova JSON file from OSC repos
    """
    # Try different year ranges
    year_ranges = [
        "sne-2020-2024",
        "sne-2015-2019", 
        "sne-2010-2014",
        "sne-2005-2009",
        "sne-2000-2004"
    ]
    
    for year_range in year_ranges:
        url = f"https://raw.githubusercontent.com/astrocatalogs/{year_range}/master/{sn_name}.json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Save to file
                filepath = os.path.join(DATA_DIR, f"{sn_name}.json")
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✓ Downloaded {sn_name}")
                return True
                
        except Exception as e:
            continue
    
    print(f"✗ Could not find {sn_name}")
    return False

def main():
    print("=" * 60)
    print("DOWNLOADING SUPERNOVA DATA FROM OPEN SUPERNOVA CATALOG")
    print("=" * 60)
    print(f"\nTarget: {len(SAMPLE_SUPERNOVAE)} supernovae")
    print(f"Destination: ./{DATA_DIR}/\n")
    
    success_count = 0
    
    for sn_name in tqdm(SAMPLE_SUPERNOVAE, desc="Downloading"):
        if download_supernova(sn_name):
            success_count += 1
        time.sleep(0.5)  # Be nice to GitHub
    
    print(f"\n{'=' * 60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successfully downloaded: {success_count}/{len(SAMPLE_SUPERNOVAE)} supernovae")
    print(f"Data saved to: ./{DATA_DIR}/")
    
    if success_count > 0:
        print("\n✓ Ready to run: python build_features.py")
    else:
        print("\n✗ No data downloaded. Check your internet connection.")

if __name__ == "__main__":
    main()
