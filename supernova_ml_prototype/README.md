# Supernova Precursor Detection - ML Prototype

An anomaly detection pipeline for identifying pre-supernova behavior in astronomical time-series data.

## 🎯 What This Does

This prototype implements the **"Watchlist Approach"** from your research document:
- Downloads real supernova light curves from the Open Supernova Catalog
- Extracts statistical features from time-series photometry
- Trains an unsupervised anomaly detector (Isolation Forest)
- Flags unusual stellar behavior that could indicate pre-explosion activity
- Validates against known precursor cases (e.g., SN 2009ip)

## 🚀 Quick Start (4 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Data (~2 minutes)
```bash
python get_data.py
```
Downloads ~25 well-documented supernovae from OSC GitHub repos.

### Step 3: Extract Features (~30 seconds)
```bash
python build_features.py
```
Parses JSON files, extracts light curves, computes 16 time-series features.

### Step 4: Train Model & Visualize (~1 minute)
```bash
python train_model.py
python visualize.py
```
Trains anomaly detector and generates plots.

## 📊 What You Get

After running all scripts, you'll have:

### Data Files:
- `data/` - Supernova JSON files from OSC
- `supernova_features.csv` - Extracted features for ML
- `anomaly_scores.csv` - Model predictions & rankings

### Outputs:
- `anomaly_detector.pkl` - Trained model (can reuse)
- `plots/anomaly_distribution.png` - Score distributions
- `plots/pca_visualization.png` - 2D feature space
- `plots/type_distribution.png` - Anomaly rates by SN type
- `plots/lightcurve_*.png` - Individual light curves

## 🔬 The Science

### Features Extracted (16 total):
Based on LAISS paper methodology:
- **Statistical**: mean, median, std, range, amplitude
- **Shape**: skewness, kurtosis, autocorrelation
- **Time-based**: observation count, time span, cadence
- **Variability**: magnitude differences, linear trend
- **Behavior**: trend strength, erratic changes

### Model: Isolation Forest
- Unsupervised anomaly detection
- Identifies objects "isolated" in 16D feature space
- No labeled training data needed
- Flags top 20% as anomalous

### Key Validation:
- Type IIn SNe (circumstellar interaction) should have high anomaly rates
- SN 2009ip (famous precursor case) should be flagged

## 📈 Understanding Results

### Check These Files:

**1. anomaly_scores.csv**
- Sorted by anomaly score (higher = more unusual)
- Top objects are prime precursor candidates
- `is_anomaly=True` means model flagged it

**2. plots/pca_visualization.png**
- Red stars = anomalies (isolated from main cluster)
- Blue dots = normal SNe
- Top 5 anomalies are labeled

**3. plots/type_distribution.png**
- Type IIn should show HIGH anomaly rate (good!)
- Type Ia should show LOW anomaly rate (normal)

## 🎓 Next Steps

### To Improve This Prototype:
1. **More data**: Download 100s of SNe instead of 25
2. **More features**: Add color indices (B-V), spectroscopic data
3. **Time windows**: Look specifically at 1-2 years before explosion
4. **MESA simulations**: Generate synthetic precursor data

### To Scale to Real Research (Option B):
1. **Query ZTF**: Use ZTF API to get live Red Supergiant monitoring
2. **Add Gaia**: Cross-match with Gaia DR3 for stellar parameters
3. **Real-time pipeline**: Deploy model to flag new precursors
4. **Coordinate observations**: Share watchlist with observatories

## 📚 Files Explained

- `get_data.py` - Downloads supernova JSON from OSC repos
- `build_features.py` - Parses photometry, computes features
- `train_model.py` - Trains Isolation Forest anomaly detector
- `visualize.py` - Generates all plots and summaries
- `requirements.txt` - Python package dependencies

## 🐛 Troubleshooting

**"No data downloaded"**
- Check internet connection
- GitHub might be rate-limiting (wait 1 hour)

**"No valid light curves"**
- Some SNe lack good photometry
- Script will skip and continue

**"Import errors"**
- Run: `pip install -r requirements.txt`
- May need Python 3.8+

## 🏆 Success Criteria

Your prototype works if:
- ✅ Downloads data successfully
- ✅ Extracts >10 light curves with features
- ✅ Model flags ~20% as anomalies
- ✅ Type IIn SNe show elevated anomaly rates
- ✅ SN 2009ip is in top anomalies (if downloaded)

## 📖 Research Context

This implements concepts from:
- LAISS pipeline (Villar et al.) - Feature extraction
- SN 2009ip study (Mauerhan et al.) - Precursor validation
- Your research document - "Watchlist Approach" methodology

## 💡 Demo Tips

When presenting:
1. Show `plots/pca_visualization.png` - visual impact!
2. Explain: "Red stars are anomalies isolated in feature space"
3. Point to SN 2009ip if flagged: "Model caught the famous precursor!"
4. Show `anomaly_scores.csv` top 10: "These are your watchlist candidates"
5. Discuss: "Next step = deploy on live ZTF data"

---

**Built with:** Python, scikit-learn, pandas, matplotlib
**Data source:** Open Supernova Catalog (OSC)
**Method:** Unsupervised anomaly detection via Isolation Forest
