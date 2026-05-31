# Hybrid Predictive Architecture for Seismic Infrastructure Disruption

## Overview
This repository contains the data-generating process (DGP) and machine learning pipeline for the study: *"Operationalizing Governance as a Predictive Variable in Seismic Infrastructure Disruption Models."* 

To overcome the structural unavailability of live megaproject governance data under seismic conditions, this study utilizes a simulation-calibrated proof-of-concept design. This codebase transparently generates 500 infrastructure disruption scenarios anchored to real-world datasets (EAFZ, AFAD, ALOS PALSAR-2, UNDRR/IRP) and trains a Gradient Boosting Machine (GBM) to forecast prolonged downtime.

## Repository Structure
*   `dcfo_simulation_pipeline.py`: The master script. Contains the Gaussian Copula sampling for physical hazard dependency, the governance variable generation, the continuous downtime mathematical function, and the complete ML validation pipeline (GBM training, Feature Ablation, and Permutation Importance).

## Dependencies
*   `numpy`, `pandas`, `scikit-learn`, `xgboost`, `scipy`

## Methodological Note
The outcome variable (`downtime_days`) is generated as a joint function of physical, governance, and contextual features. The ML pipeline is designed to test the *computational separability and non-redundancy* of these features under adversarial attribution conditions, not to "discover" causal laws from simulated data.

## License
MIT License. Free for academic and research use with attribution.

"""
Data-Generating Process and ML Pipeline for Hybrid Seismic Disruption Forecasting.
This script executes the Gaussian copula feature generation, downtime DGP, 
and Gradient Boosting Machine validation (including Permutation and Ablation).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm, truncnorm, nbinom, gamma, poisson, bernoulli, uniform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

# =============================================================================
# 1. UTILITY FUNCTIONS & COPULA SETUP
# =============================================================================

def nearest_positive_definite(corr):
    """Ensures the correlation matrix is positive definite to prevent LinAlg errors."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[eigvals < 1e-8] = 1e-8
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_pd))
    return corr_pd / np.outer(d, d)

def gaussian_copula_samples(n, corr, random_state=42):
    """Generates correlated uniform samples using a Gaussian copula."""
    rng = np.random.default_rng(random_state)
    corr = nearest_positive_definite(np.array(corr))
    z = rng.multivariate_normal(mean=np.zeros(corr.shape[0]), cov=corr, size=n)
    return norm.cdf(z)

def zscore(x):
    """Standardizes variables to zero mean and unit variance."""
    return (x - np.mean(x)) / np.std(x, ddof=1)

# =============================================================================
# 2. SCENARIO GENERATION (PHYSICAL & GOVERNANCE FEATURES)
# =============================================================================

n_scenarios = 500
seed = 42
rng = np.random.default_rng(seed)

# A. Physical Hazard Dependency Structure (Mw, Fault Distance, PGA, PGV)
physical_corr = np.array([
    [1.00, -0.55,  0.65,  0.58],
    [-0.55, 1.00, -0.60, -0.52],
    [0.65, -0.60,  1.00,  0.82],
    [0.58, -0.52,  0.82,  1.00]
])
u_phys = gaussian_copula_samples(n_scenarios, physical_corr, random_state=seed)

# Marginal Transformations (Physical)
mw_mean, mw_sd, mw_low, mw_high = 6.3, 0.55, 5.0, 7.5
Mw = truncnorm.ppf(u_phys[:, 0], a=(mw_low-mw_mean)/mw_sd, b=(mw_high-mw_mean)/mw_sd, loc=mw_mean, scale=mw_sd)
fault_distance = lognorm.ppf(u_phys[:, 1], s=0.45, scale=np.exp(np.log(35)))
PGA = lognorm.ppf(u_phys[:, 2], s=0.35, scale=np.exp(np.log(0.22)))
PGV = lognorm.ppf(u_phys[:, 3], s=0.40, scale=np.exp(np.log(22)))

# B. Governance Dependency Structure (Backlog, CMI, Hist. Downtime, Authorities)
gov_corr = np.array([
    [1.00, -0.35,  0.40,  0.45],
    [-0.35, 1.00, -0.30, -0.25],
    [0.40, -0.30,  1.00,  0.30],
    [0.45, -0.25,  0.30,  1.00]
])
u_gov = gaussian_copula_samples(n_scenarios, gov_corr, random_state=seed+1)

# Marginal Transformations (Governance & Contextual)
approval_backlog = nbinom.ppf(u_gov[:, 0], n=3, p=0.4).astype(int)
CMI = uniform.ppf(u_gov[:, 1], loc=0.4, scale=0.5)
historical_downtime = gamma.ppf(u_gov[:, 2], a=2.1, scale=1.8)
active_authorities = poisson.ppf(u_gov[:, 3], mu=4.2).astype(int)

event_season = rng.integers(1, 5, size=n_scenarios)
winter_season = np.where(event_season == 1, 1, 0)
maintenance_late = bernoulli.rvs(0.35, size=n_scenarios, random_state=seed+2)
soil_class = rng.choice([0, 1, 2], size=n_scenarios, p=[0.35, 0.45, 0.20])
soil_soft = np.where(soil_class == 2, 1, 0)

# =============================================================================
# 3. CONTINUOUS DOWNTIME DATA-GENERATING PROCESS (DGP)
# =============================================================================

# Standardize inputs for DGP algebraic function
z_PGA, z_PGV, z_FD = zscore(PGA), zscore(PGV), zscore(fault_distance)
z_AB, z_CMI, z_HD = zscore(approval_backlog), zscore(CMI), zscore(historical_downtime)
z_AA = zscore(active_authorities)

epsilon = rng.normal(loc=0.0, scale=1.25, size=n_scenarios)

# Core Downtime Equation (Matches Manuscript Section 3.5)
downtime_days = (
    9.0
    + 1.65 * z_PGA
    + 0.85 * z_PGV
    - 0.70 * z_FD
    + 1.35 * z_AB
    - 1.10 * z_CMI
    + 0.75 * z_HD
    + 0.45 * z_AA
    + 0.35 * maintenance_late
    + 0.30 * soil_soft
    + 0.25 * winter_season
    + 0.55 * (z_PGA * soil_soft)
    + 0.45 * (z_AB * z_AA)
    - 0.40 * (z_CMI * z_AB)
    + epsilon
)

# Enforce practical bounds and binarize
downtime_days = np.maximum(downtime_days, 0.5)
prolonged_disruption = (downtime_days > 9.0).astype(int)

df = pd.DataFrame({
    "PGA": PGA, "PGV": PGV, "fault_distance_km": fault_distance,
    "approval_backlog": approval_backlog, "contractor_mobilization_index": CMI,
    "historical_downtime": historical_downtime, "active_approval_authorities": active_authorities,
    "winter_season": winter_season, "maintenance_late": maintenance_late,
    "soil_soft": soil_soft, "prolonged_disruption": prolonged_disruption
})

# =============================================================================
# 4. ML PIPELINE & PERMUTATION IMPORTANCE
# =============================================================================

print("\n--- Training Primary GBM Model ---")
X = df.drop(columns=["prolonged_disruption"])
y = df["prolonged_disruption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

gbm = GradientBoostingClassifier(learning_rate=0.05, max_depth=6, n_estimators=500, subsample=0.8, random_state=42)
gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)
print(f"GBM Holdout Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"GBM Holdout AUC-ROC:  {roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]):.3f}")

print("\n--- Running Permutation Importance (Copy these results to Appendix C) ---")
perm = permutation_importance(gbm, X_test, y_test, n_repeats=30, random_state=42, scoring="accuracy")
perm_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Accuracy_Drop": perm.importances_mean,
    "Std_Deviation": perm.importances_std
}).sort_values("Mean_Accuracy_Drop", ascending=False)
print(perm_df.head(5).to_string(index=False))

# =============================================================================
# 5. FEATURE ABLATION TEST (PHYSICAL ONLY VS. HYBRID)
# =============================================================================

print("\n--- Running Feature Ablation Test ---")
phys_features = ["PGA", "PGV", "fault_distance_km", "winter_season", "soil_soft"]

X_train_phys, X_test_phys = X_train[phys_features], X_test[phys_features]
gbm_phys = GradientBoostingClassifier(learning_rate=0.05, max_depth=6, n_estimators=500, subsample=0.8, random_state=42)
gbm_phys.fit(X_train_phys, y_train)

print(f"Physical-Only Model Accuracy: {accuracy_score(y_test, gbm_phys.predict(X_test_phys)):.3f}")
print(f"Hybrid Model Accuracy:        {accuracy_score(y_test, y_pred):.3f}")
