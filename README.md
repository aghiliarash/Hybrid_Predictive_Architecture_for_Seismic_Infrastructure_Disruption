# Hybrid Predictive Architecture for Seismic Infrastructure Disruption

**Operationalizing Governance as a Predictive Variable in Seismic Infrastructure Disruption**

---

## 📋 Overview

This repository provides a complete **simulation-calibrated ML pipeline** for studying seismic infrastructure disruption under uncertainty. The study operationalizes governance variables as predictive factors alongside physical hazard features to forecast infrastructure downtime.

### The Challenge
Live megaproject governance data during seismic events is structurally unavailable. This work demonstrates a **transparent computational proof-of-concept** to evaluate how governance mechanisms interact with physical phenomena to determine infrastructure disruption outcomes.

### Key Innovation
A **hybrid predictive architecture** that combines:
- Physical hazard variables (magnitude, distance, ground motion)
- Governance variables (approval backlogs, contractor mobilization, coordination complexity)
- Contextual factors (season, soil conditions, maintenance status)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/aghiliarash/Hybrid_Predictive_Architecture_for_Seismic_Infrastructure_Disruption.git
cd Hybrid_Predictive_Architecture_for_Seismic_Infrastructure_Disruption
```

### Dependencies

```
numpy
pandas
scikit-learn
scipy
xgboost (optional)
shap (optional)
matplotlib
```

Install via pip:
```bash
pip install numpy pandas scikit-learn scipy xgboost shap matplotlib
```

### Run the Pipeline

```bash
python dcfo_simulation_pipeline.py
```

All outputs will be saved to the `model_outputs/` directory.

---

## 📁 Repository Structure

| File | Purpose |
|------|---------|
| `dcfo_simulation_pipeline.py` | **Master script** — Complete reproducible pipeline |
| `README.md` | This file |

### Output Files Generated

The pipeline generates:
- **Data**: Synthetic disruption scenarios, correlation matrices
- **Models**: Classification and regression benchmarks
- **Analysis**: Permutation importance, SHAP values, feature ablation
- **Figures**: Correlation heatmaps, SHAP beeswarm plots
- **Tables**: Model performance, coefficients, feature importance

---

## 🔧 Methodological Approach

### 1. **Data-Generating Process (DGP)**

The continuous downtime outcome is generated as a transparent linear structural model:

```
downtime_days = β₀ + Σ(βᵢ × zᵢ) + interactions + ε

where:
  β₀ = 9.0 (operational baseline)
  zᵢ = standardized feature i
  ε ~ N(0, 1.25²) (measurement noise)
```

**Physical Effects:**
- PGA (1.65), PGV (0.85), fault distance (-0.70), soil softness (0.30)

**Governance Effects:**
- Approval backlog (1.35), contractor mobilization (-1.10), historical downtime (0.75), active authorities (0.45)

**Interactions:**
- PGA × soil softness (0.55) — amplification in soft soil
- Approval backlog × active authorities (0.45) — coordination burden
- Contractor mobilization × approval backlog (-0.40) — readiness mitigates delays

### 2. **Feature Generation via Gaussian Copulas**

Physical and governance variables are generated with specified correlation structures to preserve realistic dependencies:

```
Target Physical Correlation:
                Mw    Distance    PGA    PGV
  Mw          1.00     -0.55     0.65   0.58
  Distance   -0.55      1.00    -0.60  -0.52
  PGA         0.65     -0.60     1.00   0.82
  PGV         0.58     -0.52     0.82   1.00

Target Governance Correlation:
                Backlog   Mobilization   History   Authorities
  Backlog        1.00       -0.35        0.40       0.45
  Mobilization  -0.35        1.00       -0.30      -0.25
  History        0.40       -0.30        1.00       0.30
  Authorities    0.45       -0.25        0.30       1.00
```

### 3. **Model Pipeline**

The pipeline benchmarks multiple architectures:

| Model | Type | Role |
|-------|------|------|
| **GBM-Fortify** | Classification | Primary model — regularized for stability |
| GBM | Classification | Baseline gradient boosting |
| Random Forest | Classification | Ensemble alternative |
| XGBoost | Classification | Accelerated boosting (if available) |

**Primary Model Selection:** GBM-Fortify is regularized for manuscript reproducibility:
- Learning rate: 0.03 (conservative)
- Max depth: 3 (shallow trees)
- Min samples per leaf: 5 (prevent overfitting)
- 300 estimators with 0.8 subsampling

### 4. **Interpretability Analysis**

Three complementary approaches to understand feature importance:

1. **Permutation Importance** — Accuracy drop when feature is shuffled
2. **SHAP Values** — Shapley additive explanations per prediction
3. **Feature Ablation** — Hybrid vs. physical-only model comparison

---

## 📊 Key Features

### Data Diversity
- **500 synthetic scenarios** with realistic correlation structures
- **10 feature groups** spanning hazard, governance, and context
- **Binary target** (prolonged disruption) and continuous outcome (downtime days)

### Interpretability-First Design
- ✅ Transparent DGP — all coefficients specified and documented
- ✅ No black-box tuning — deterministic correlation matrices and distributions
- ✅ Multiple validation approaches — SHAP, permutation, ablation
- ✅ Reproducible — fixed random seed, deterministic outputs

### Institutional Validity
- Governance variables calibrated to megaproject literature
- Physical parameters matched to seismic catalogs (İzmit 1999 baseline)
- Interaction terms grounded in infrastructure disruption theory

---

## ⚠️ Important Notes

### What This Is
✅ A **simulation-calibrated proof-of-concept** for governance-inclusive ML  
✅ A **computational transparency exercise** demonstrating method integration  
✅ A **reproducible benchmark** for hypothesis testing at the simulation level  

### What This Is NOT
❌ Empirical causal inference on real disruption events  
❌ Operational deployment model without external validation  
❌ Substitute for field data and domain expertise  

### Proper Use
All inferences should be **bounded to the calibrated simulation architecture**. The model coefficients represent computational relationships in the DGP, not empirical estimates of real-world effects.

---

## 📈 Expected Outputs

After running the pipeline, examine:

1. **Classification_Model_Benchmark.csv** — Model performance comparison
2. **GBM-Fortify_Permutation_Importance_All_Features.csv** — Feature rankings
3. **GBM-Fortify_SHAP_MeanAbs_Importance.csv** — SHAP-based importance
4. **GBM-Fortify_Feature_Ablation_Test.csv** — Governance uplift over physical-only
5. **GBM-Fortify_SHAP_Category_Shares.csv** — Governance vs. physical vs. contextual split
6. **Synthetic_Disruption_Scenarios.csv** — Full dataset for external analysis
7. **Figure_*.png** — Correlation heatmaps and SHAP visualizations

---

## 🔬 Calibration Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_SCENARIOS | 500 | Sample size for simulation |
| RANDOM_SEED | 42 | Reproducibility |
| Test split | 20% | Holdout evaluation |
| CV folds | 10 | Stratified K-fold |
| Primary model learning rate | 0.03 | Conservative for stability |
| Primary model max_depth | 3 | Prevent overfitting |
| Min samples leaf | 5 | Regularization |
| Noise scale (σ) | 1.25 | DGP stochasticity |

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@software{seismic_governance_2024,
  author = {Author Name},
  title = {Hybrid Predictive Architecture for Seismic Infrastructure Disruption},
  year = {2024},
  url = {https://github.com/aghiliarash/Hybrid_Predictive_Architecture_for_Seismic_Infrastructure_Disruption}
}
```

---

## 📜 License

MIT License. Free for academic and research use with attribution.

---

## ✉️ Contact & Contributing

Questions or feedback? Please open a GitHub issue.

**Last Updated:** 2024  
**Maintainer:** aghiliarash
