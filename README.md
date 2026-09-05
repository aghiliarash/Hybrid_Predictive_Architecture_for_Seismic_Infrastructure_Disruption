# Governance-Aware Machine Learning for Post-Disaster Infrastructure Recovery

**Operationalizing Governance as a Predictive Variable in Seismic Infrastructure Disruption**
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-optional-lightgrey)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-optional-lightgrey)](https://shap.readthedocs.io/)
[![Reproducible](https://img.shields.io/badge/Reproducibility-Seed%2042-green)]()

</div>
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

The continuous downtime outcome is generated as a transparent, non-linear structural model that compounds administrative delays conditionally based on physical damage:

```
D_i = max(0.5, 9.0 + 1.65×z(PGA) + 0.85×z(PGV) - 0.70×z(FD) + Governance_Penalty + Interactions + ε)

where:
  Governance_Penalty establishes a hard non-linear discontinuity:
    If z(PGA) > 0.50 (Severe Damage): 
       Penalty = 1.80×z(AB) - 1.20×z(CMI) + 0.50×[z(AB) × z(AA)]
    Else (Minor Damage): 
       Penalty = 0.40×z(AB) - 0.30×z(CMI) + 0.10×z(AA)

  ε ~ N(0, 1.25²) (unobserved random disturbance)
```

### 2. **Feature Generation via Gaussian Copulas**

Physical and governance variables are generated with specified correlation structures to preserve realistic dependencies across the simulation space:

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

The pipeline benchmarks multiple architectures, maintaining a strict validation split to prevent data leakage:

| Model | Type | Role |
|-------|------|------|
| **Tuned-GBM** | Classification | Primary model — regularized for stability |
| GBM | Classification | Baseline unconstrained gradient boosting |
| Random Forest | Classification | Ensemble alternative |
| XGBoost | Classification | Accelerated boosting (if available) |

**Primary Model Selection:** The *Tuned-GBM* is highly regularized for manuscript reproducibility and to prevent overfitting to the synthetic noise:
- Learning rate: 0.03
- Max depth: 3 (shallow trees)
- Min samples per leaf: 5
- 300 estimators with 0.8 subsampling

### 4. **Interpretability Analysis**

Three complementary approaches are utilized to robustly understand feature importance:

1. **Permutation Importance** — Accuracy drop when individual features are shuffled
2. **SHAP Values** —  Exact Shapley additive explanations (TreeExplainer) per prediction
3. **Feature Ablation** — Hybrid vs. Physical-only baseline model comparison

---

## 📊 Key Features

### Data Diversity
- **1000 synthetic scenarios** with realistic, multi-variate correlation structures
- **10 feature groups** spanning physical hazards, administrative governance, and infrastructure context
- **Binary target** (prolonged disruption > 9 days) and continuous evaluation outcome (downtime days)

### Interpretability-First Design
- ✅ **Transparent DGP** — all coefficients and conditional step-functions strictly documented
- ✅ **No black-box tuning** — deterministic correlation matrices and distributions
- ✅ **Robust validation** — strict 85/15 split, 10-fold CV on the development set, plus SHAP, permutation, and ablation on the independent holdout
- ✅ **Reproducible** — fixed random seeds and deterministic outputs

### Institutional Validity
- Governance parameters heuristically calibrated to megaproject literature and field reports (e.g., AFAD 2023)
- Physical parameters calibrated to regional seismic catalogs (EAFZ / 2023 Kahramanmaraş context)

---

## ⚠️ Important Notes

### What This Is
- ✅ A **simulation-calibrated proof-of-concept** for governance-inclusive machine learning  
- ✅ A **computational transparency exercise** demonstrating method integration and signal recovery through non-linear noise
- ✅ A **reproducible benchmark** for hypothesis testing at the simulation level

### What This Is NOT
- ❌ Empirical causal inference on real historical disruption events
- ❌ An operational deployment tool ready for immediate field use without PMIS data linkage
- ❌ A substitute for physical field data and engineering domain expertise

### Proper Use
All inferences should be **strictly bounded to the calibrated simulation architecture.** The model coefficients and exact SHAP percentages represent computational relationships and model attributions within the specified DGP, not empirical estimates of real-world phenomena.

---

## 📈 Expected Outputs

After running the pipeline, examine the model_outputs/ directory for:
1. **Classification_Model_Benchmark.csv** — Model performance comparison across architectures
2. **Tuned-GBM_Permutation_Importance_All_Features.csv** — Feature rankings by holdout accuracy drop
3. **Tuned-GBM_SHAP_MeanAbs_Importance.csv** — SHAP-based absolute importance rankings
4. **Tuned-GBM_Feature_Ablation_Test.csv** — Governance accuracy uplift over the physical-only baseline
5. **Tuned-GBM_SHAP_Category_Shares.csv** — Normalized categorical split (Governance vs. Physical vs. Contextual)
6. **Synthetic_Disruption_Scenarios.csv** — Full 10,000 scenario dataset for external analysis
7. **Figure_*.png** — High-resolution correlation heatmaps and SHAP beeswarm plots

---

## 🔬 Calibration Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_SCENARIOS | 1000 | Scaled sample size for robust simulation |
| RANDOM_SEED | 42 | Guaranteed reproducibility |
| Test split | 15% | Independent Holdout evaluation |
| CV folds | 10 | Stratified K-fold performed exclusively on the 85% development set |
| Primary model learning rate | 0.03 | Conservative learning rate for the Tuned-GBM |

---

## 📚 Citation

If you utilize this pipeline or framework, please cite the associated manuscript:

```bibtex
@article{aghili_ugural_2026,
  author = {Aghili, Seyedarash and Uğural, Mehmet Nurettin},
  title = {Governance-Aware Machine Learning for Post-Disaster Infrastructure Recovery: A Synthetic Proof-of-Concept},
  year = {2026},
  journal = {Pending Publication},
  url = {https://github.com/aghiliarash/Hybrid_Predictive_Architecture_for_Seismic_Infrastructure_Disruption}
}
```

---

## 📜 License

MIT License. free for academic and research use with attribution. See [LICENSE](LICENSE) for full text.


---

## ✉️ Contact & Contributing

Questions or feedback? Please open a GitHub issue.

**Last Updated:** 2026  
**Maintainer:** aghiliarash
