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
Final Reproducibility Pipeline
Hybrid Seismic Infrastructure Disruption Forecasting

Primary model:
    Regularized GBM-Fortify

Purpose:
    1. Generate correlated physical and governance features using Gaussian copulas.
    2. Generate continuous downtime using a transparent DGP.
    3. Create binary prolonged-disruption target.
    4. Benchmark GBM, GBM-Fortify, Random Forest, and XGBoost if available.
    5. Select GBM-Fortify as the primary model.
    6. Run SHAP, permutation importance, and ablation on GBM-Fortify.
    7. Export all tables and figures for manuscript reproducibility.

Important:
    - The simulation is a transparent computational experiment.
    - Coefficients are not empirical causal estimates.
    - All claims should be bounded to the calibrated simulation architecture.
"""

# =============================================================================
# 0. IMPORTS AND GLOBAL SETTINGS
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    norm,
    lognorm,
    truncnorm,
    nbinom,
    gamma,
    poisson,
    bernoulli,
    uniform
)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.inspection import permutation_importance

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


RANDOM_SEED = 42
N_SCENARIOS = 500
OUTPUT_DIR = "model_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================

def nearest_positive_definite(corr):
    """
    Ensures that a correlation matrix is positive definite.
    """
    corr = np.asarray(corr, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[eigvals < 1e-8] = 1e-8

    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / np.outer(d, d)

    return corr_pd


def gaussian_copula_samples(n, corr, random_state=42):
    """
    Generates correlated uniform samples using a Gaussian copula.
    """
    local_rng = np.random.default_rng(random_state)
    corr_pd = nearest_positive_definite(corr)

    z = local_rng.multivariate_normal(
        mean=np.zeros(corr_pd.shape[0]),
        cov=corr_pd,
        size=n
    )

    u = norm.cdf(z)
    return u


def zscore(x):
    """
    Standardizes an array to zero mean and unit variance.
    """
    x = np.asarray(x)
    return (x - np.mean(x)) / np.std(x, ddof=1)


def rmse(y_true, y_pred):
    """
    Root Mean Square Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_to_csv(df, filename, **kwargs):
    path = os.path.join(OUTPUT_DIR, filename)

    if "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8-sig"

    try:
        df.to_csv(path, **kwargs)
        return path

    except PermissionError:
        base, ext = os.path.splitext(filename)
        fallback = os.path.join(OUTPUT_DIR, f"{base}_fallback{ext}")

        warnings.warn(
            f"Permission denied for {path}. Saving fallback to {fallback}.",
            UserWarning
        )

        df.to_csv(fallback, **kwargs)
        return fallback


def save_correlation_heatmap(corr_df, title, filename):
    """
    Saves a correlation heatmap using matplotlib.
    """
    labels = corr_df.columns.tolist()
    values = corr_df.values

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)


def safe_shap_values(model, X):
    """
    Computes SHAP values robustly across SHAP versions.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed.")

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[1]

    return values


# =============================================================================
# 2. COPULA CORRELATION MATRICES
# =============================================================================

# Physical variable order: Mw, fault distance, PGA, PGV
physical_corr_target = np.array([
    [1.00, -0.55,  0.65,  0.58],
    [-0.55, 1.00, -0.60, -0.52],
    [0.65, -0.60,  1.00,  0.82],
    [0.58, -0.52,  0.82,  1.00]
])

physical_corr_labels = [
    "Mw",
    "fault_distance_km",
    "PGA",
    "PGV"
]

# Governance variable order:
# approval backlog, contractor mobilization, historical downtime, active authorities
governance_corr_target = np.array([
    [1.00, -0.35,  0.40,  0.45],
    [-0.35, 1.00, -0.30, -0.25],
    [0.40, -0.30,  1.00,  0.30],
    [0.45, -0.25,  0.30,  1.00]
])

governance_corr_labels = [
    "approval_backlog",
    "contractor_mobilization_index",
    "historical_downtime",
    "active_approval_authorities"
]

safe_to_csv(
    pd.DataFrame(
        physical_corr_target,
        index=physical_corr_labels,
        columns=physical_corr_labels
    ),
    "Target_Physical_Copula_Correlation.csv"
)

safe_to_csv(
    pd.DataFrame(
        governance_corr_target,
        index=governance_corr_labels,
        columns=governance_corr_labels
    ),
    "Target_Governance_Copula_Correlation.csv"
)


# =============================================================================
# 3. SCENARIO GENERATION
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1 Physical hazard variables
# -----------------------------------------------------------------------------

u_phys = gaussian_copula_samples(
    n=N_SCENARIOS,
    corr=physical_corr_target,
    random_state=RANDOM_SEED
)

mw_mean = 6.3
mw_sd = 0.55
mw_low = 5.0
mw_high = 7.5

a = (mw_low - mw_mean) / mw_sd
b = (mw_high - mw_mean) / mw_sd

Mw = truncnorm.ppf(
    u_phys[:, 0],
    a=a,
    b=b,
    loc=mw_mean,
    scale=mw_sd
)

fault_distance_km = lognorm.ppf(
    u_phys[:, 1],
    s=0.45,
    scale=np.exp(np.log(35))
)

PGA = lognorm.ppf(
    u_phys[:, 2],
    s=0.35,
    scale=np.exp(np.log(0.22))
)

PGV = lognorm.ppf(
    u_phys[:, 3],
    s=0.40,
    scale=np.exp(np.log(22))
)


# -----------------------------------------------------------------------------
# 3.2 Governance variables
# -----------------------------------------------------------------------------

u_gov = gaussian_copula_samples(
    n=N_SCENARIOS,
    corr=governance_corr_target,
    random_state=RANDOM_SEED + 1
)

approval_backlog = nbinom.ppf(
    u_gov[:, 0],
    n=3,
    p=0.4
).astype(int)

contractor_mobilization_index = uniform.ppf(
    u_gov[:, 1],
    loc=0.4,
    scale=0.5
)

historical_downtime = gamma.ppf(
    u_gov[:, 2],
    a=2.1,
    scale=1.8
)

active_approval_authorities = poisson.ppf(
    u_gov[:, 3],
    mu=4.2
).astype(int)


# -----------------------------------------------------------------------------
# 3.3 Contextual variables
# -----------------------------------------------------------------------------

event_season = rng.integers(1, 5, size=N_SCENARIOS)

winter_season = np.where(
    event_season == 1,
    1,
    0
)

maintenance_late = bernoulli.rvs(
    0.35,
    size=N_SCENARIOS,
    random_state=RANDOM_SEED + 2
)

soil_class = rng.choice(
    [0, 1, 2],
    size=N_SCENARIOS,
    p=[0.35, 0.45, 0.20]
)

soil_soft = np.where(
    soil_class == 2,
    1,
    0
)


# =============================================================================
# 4. CONTINUOUS DOWNTIME DATA-GENERATING PROCESS
# =============================================================================

z_PGA = zscore(PGA)
z_PGV = zscore(PGV)
z_FD = zscore(fault_distance_km)

z_AB = zscore(approval_backlog)
z_CMI = zscore(contractor_mobilization_index)
z_HD = zscore(historical_downtime)
z_AA = zscore(active_approval_authorities)

epsilon = rng.normal(
    loc=0.0,
    scale=1.25,
    size=N_SCENARIOS
)

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

downtime_days = np.maximum(
    downtime_days,
    0.5
)

prolonged_disruption = (
    downtime_days > 9.0
).astype(int)


# =============================================================================
# 5. BUILD DATAFRAME AND EXPORT SCENARIOS
# =============================================================================

df = pd.DataFrame({
    "Mw": Mw,
    "PGA": PGA,
    "PGV": PGV,
    "fault_distance_km": fault_distance_km,
    "soil_soft": soil_soft,
    "approval_backlog": approval_backlog,
    "contractor_mobilization_index": contractor_mobilization_index,
    "historical_downtime": historical_downtime,
    "active_approval_authorities": active_approval_authorities,
    "maintenance_late": maintenance_late,
    "winter_season": winter_season,
    "downtime_days": downtime_days,
    "prolonged_disruption": prolonged_disruption
})

safe_to_csv(
    df,
    "Synthetic_Disruption_Scenarios.csv",
    index=False
)


# =============================================================================
# 6. CORRELATION VALIDATION AND FIGURES
# =============================================================================

realized_physical_corr = df[
    ["Mw", "fault_distance_km", "PGA", "PGV"]
].corr(method="spearman")

realized_governance_corr = df[
    [
        "approval_backlog",
        "contractor_mobilization_index",
        "historical_downtime",
        "active_approval_authorities"
    ]
].corr(method="spearman")

safe_to_csv(
    realized_physical_corr,
    "Realized_Physical_Spearman_Correlation.csv"
)

safe_to_csv(
    realized_governance_corr,
    "Realized_Governance_Spearman_Correlation.csv"
)

save_correlation_heatmap(
    realized_physical_corr,
    "Realized Physical Hazard Spearman Correlation",
    "Figure_Physical_Copula_Realized_Correlation.png"
)

save_correlation_heatmap(
    realized_governance_corr,
    "Realized Governance Spearman Correlation",
    "Figure_Governance_Copula_Realized_Correlation.png"
)


# =============================================================================
# 7. FEATURE SETS AND TRAIN / TEST SPLIT
# =============================================================================

model_features = [
    "PGA",
    "PGV",
    "fault_distance_km",
    "soil_soft",
    "approval_backlog",
    "contractor_mobilization_index",
    "historical_downtime",
    "active_approval_authorities",
    "maintenance_late",
    "winter_season"
]

physical_only_features = [
    "PGA",
    "PGV",
    "fault_distance_km",
    "soil_soft",
    "winter_season"
]

governance_features = [
    "approval_backlog",
    "contractor_mobilization_index",
    "historical_downtime",
    "active_approval_authorities",
    "maintenance_late"
]

physical_features = [
    "PGA",
    "PGV",
    "fault_distance_km",
    "soil_soft"
]

contextual_features = [
    "winter_season"
]

X = df[model_features]
y_class = df["prolonged_disruption"]
y_reg = df["downtime_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.20,
    stratify=y_class,
    random_state=RANDOM_SEED
)

train_idx = X_train.index
test_idx = X_test.index

X_train_reg = X.loc[train_idx]
X_test_reg = X.loc[test_idx]

y_train_reg = y_reg.loc[train_idx]
y_test_reg = y_reg.loc[test_idx]


# =============================================================================
# 8. CLASSIFICATION MODEL BENCHMARK
# =============================================================================

classification_models = {
    "GBM": GradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        random_state=RANDOM_SEED
    ),

    "GBM-Fortify": GradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=3,
        n_estimators=300,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=RANDOM_SEED
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        bootstrap=True,
        random_state=RANDOM_SEED
    )
}

if XGBOOST_AVAILABLE:
    classification_models["XGBoost"] = XGBClassifier(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED
    )

classification_results = []

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_SEED
)

fitted_classifiers = {}

print("\n================ CLASSIFICATION BENCHMARK ================")

for model_name, model in classification_models.items():

    model.fit(
        X_train,
        y_train
    )

    fitted_classifiers[model_name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    holdout_acc = accuracy_score(
        y_test,
        y_pred
    )

    holdout_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cv_acc = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    cv_auc = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    train_acc = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    train_cv_gap = train_acc - cv_acc.mean()

    classification_results.append({
        "Model": model_name,
        "Holdout_Accuracy": holdout_acc,
        "Holdout_AUC_ROC": holdout_auc,
        "CV_Accuracy_Mean": cv_acc.mean(),
        "CV_Accuracy_SD": cv_acc.std(),
        "CV_AUC_Mean": cv_auc.mean(),
        "CV_AUC_SD": cv_auc.std(),
        "Training_Accuracy": train_acc,
        "Train_CV_Gap": train_cv_gap
    })

    print(f"\n{model_name}")
    print(f"  Holdout Accuracy: {holdout_acc:.3f}")
    print(f"  Holdout AUC-ROC:  {holdout_auc:.3f}")
    print(f"  CV Accuracy Mean: {cv_acc.mean():.3f}")
    print(f"  CV Accuracy SD:   {cv_acc.std():.3f}")
    print(f"  Training Acc:     {train_acc:.3f}")
    print(f"  Train-CV Gap:     {train_cv_gap:.3f}")

classification_results_df = pd.DataFrame(
    classification_results
)

safe_to_csv(
    classification_results_df,
    "Classification_Model_Benchmark.csv",
    index=False
)


# =============================================================================
# 9. PRIMARY MODEL SELECTION
# =============================================================================

PRIMARY_MODEL_NAME = "GBM-Fortify"

primary_model = fitted_classifiers[PRIMARY_MODEL_NAME]

primary_y_pred = primary_model.predict(X_test)
primary_y_prob = primary_model.predict_proba(X_test)[:, 1]

primary_accuracy = accuracy_score(
    y_test,
    primary_y_pred
)

primary_auc = roc_auc_score(
    y_test,
    primary_y_prob
)

primary_train_accuracy = accuracy_score(
    y_train,
    primary_model.predict(X_train)
)

primary_cv_accuracy = classification_results_df.loc[
    classification_results_df["Model"] == PRIMARY_MODEL_NAME,
    "CV_Accuracy_Mean"
].iloc[0]

primary_cv_accuracy_sd = classification_results_df.loc[
    classification_results_df["Model"] == PRIMARY_MODEL_NAME,
    "CV_Accuracy_SD"
].iloc[0]

primary_train_cv_gap = classification_results_df.loc[
    classification_results_df["Model"] == PRIMARY_MODEL_NAME,
    "Train_CV_Gap"
].iloc[0]


# =============================================================================
# 10. REGRESSION MODEL BENCHMARK FOR CONTINUOUS DOWNTIME
# =============================================================================

regression_models = {
    "GBM Regressor": GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        random_state=RANDOM_SEED
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        bootstrap=True,
        random_state=RANDOM_SEED
    )
}

if XGBOOST_AVAILABLE:
    regression_models["XGBoost Regressor"] = XGBRegressor(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED
    )

regression_results = []

print("\n================ REGRESSION BENCHMARK ================")

for model_name, model in regression_models.items():

    model.fit(
        X_train_reg,
        y_train_reg
    )

    y_reg_pred = model.predict(X_test_reg)

    model_rmse = rmse(
        y_test_reg,
        y_reg_pred
    )

    model_mae = mean_absolute_error(
        y_test_reg,
        y_reg_pred
    )

    model_r2 = r2_score(
        y_test_reg,
        y_reg_pred
    )

    regression_results.append({
        "Model": model_name,
        "RMSE_days": model_rmse,
        "MAE_days": model_mae,
        "R2": model_r2
    })

    print(f"\n{model_name}")
    print(f"  RMSE: {model_rmse:.3f} days")
    print(f"  MAE:  {model_mae:.3f} days")
    print(f"  R2:   {model_r2:.3f}")

regression_results_df = pd.DataFrame(
    regression_results
)

safe_to_csv(
    regression_results_df,
    "Regression_Model_Benchmark.csv",
    index=False
)


# =============================================================================
# 11. PERMUTATION IMPORTANCE — PRIMARY MODEL
# =============================================================================

print("\n================ PERMUTATION IMPORTANCE ================")
print(f"Primary model for permutation importance: {PRIMARY_MODEL_NAME}")

perm = permutation_importance(
    primary_model,
    X_test,
    y_test,
    n_repeats=30,
    random_state=RANDOM_SEED,
    scoring="accuracy"
)

perm_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Accuracy_Drop": perm.importances_mean,
    "Std_Deviation": perm.importances_std
}).sort_values(
    "Mean_Accuracy_Drop",
    ascending=False
)

perm_df["Rank"] = np.arange(
    1,
    len(perm_df) + 1
)

perm_df = perm_df[
    [
        "Rank",
        "Feature",
        "Mean_Accuracy_Drop",
        "Std_Deviation"
    ]
]

print(perm_df.to_string(index=False))

safe_to_csv(
    perm_df,
    "GBM-Fortify_Permutation_Importance_All_Features.csv",
    index=False
)


# =============================================================================
# 12. FEATURE ABLATION — PRIMARY MODEL
# =============================================================================

print("\n================ FEATURE ABLATION TEST ================")
print(f"Primary model for ablation: {PRIMARY_MODEL_NAME}")

primary_physical_model = GradientBoostingClassifier(
    learning_rate=0.03,
    max_depth=3,
    n_estimators=300,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=RANDOM_SEED
)

primary_physical_model.fit(
    X_train[physical_only_features],
    y_train
)

physical_pred = primary_physical_model.predict(
    X_test[physical_only_features]
)

physical_prob = primary_physical_model.predict_proba(
    X_test[physical_only_features]
)[:, 1]

physical_accuracy = accuracy_score(
    y_test,
    physical_pred
)

physical_auc = roc_auc_score(
    y_test,
    physical_prob
)

hybrid_accuracy = primary_accuracy
hybrid_auc = primary_auc

ablation_uplift = hybrid_accuracy - physical_accuracy

ablation_df = pd.DataFrame([
    {
        "Model": "Physical-only GBM-Fortify",
        "Features": ", ".join(physical_only_features),
        "Holdout_Accuracy": physical_accuracy,
        "AUC_ROC": physical_auc
    },
    {
        "Model": "Hybrid physical-governance GBM-Fortify",
        "Features": ", ".join(model_features),
        "Holdout_Accuracy": hybrid_accuracy,
        "AUC_ROC": hybrid_auc
    }
])

ablation_df["Accuracy_Improvement_vs_Physical_Only"] = [
    0.0,
    ablation_uplift
]

print(ablation_df.to_string(index=False))

safe_to_csv(
    ablation_df,
    "GBM-Fortify_Feature_Ablation_Test.csv",
    index=False
)


# =============================================================================
# 13. SHAP ANALYSIS — PRIMARY MODEL
# =============================================================================

if SHAP_AVAILABLE:

    print("\n================ SHAP ANALYSIS ================")
    print(f"Primary model for SHAP: {PRIMARY_MODEL_NAME}")

    shap_values = safe_shap_values(
        primary_model,
        X_test
    )

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_abs_SHAP": mean_abs_shap
    }).sort_values(
        "Mean_abs_SHAP",
        ascending=False
    )

    shap_df["Rank"] = np.arange(
        1,
        len(shap_df) + 1
    )

    shap_df = shap_df[
        [
            "Rank",
            "Feature",
            "Mean_abs_SHAP"
        ]
    ]

    print("\nMean absolute SHAP values:")
    print(shap_df.to_string(index=False))

    safe_to_csv(
        shap_df,
        "GBM-Fortify_SHAP_MeanAbs_Importance.csv",
        index=False
    )

    gov_sum = shap_df.loc[
        shap_df["Feature"].isin(governance_features),
        "Mean_abs_SHAP"
    ].sum()

    phys_sum = shap_df.loc[
        shap_df["Feature"].isin(physical_features),
        "Mean_abs_SHAP"
    ].sum()

    context_sum = shap_df.loc[
        shap_df["Feature"].isin(contextual_features),
        "Mean_abs_SHAP"
    ].sum()

    total_sum = shap_df["Mean_abs_SHAP"].sum()
    gov_phys_total = gov_sum + phys_sum

    shap_category_df = pd.DataFrame([
        {
            "Category": "Governance",
            "Mean_abs_SHAP_sum": gov_sum,
            "Share_of_All_Features": gov_sum / total_sum,
            "Share_of_Gov_Physical_Only": gov_sum / gov_phys_total
        },
        {
            "Category": "Physical",
            "Mean_abs_SHAP_sum": phys_sum,
            "Share_of_All_Features": phys_sum / total_sum,
            "Share_of_Gov_Physical_Only": phys_sum / gov_phys_total
        },
        {
            "Category": "Contextual",
            "Mean_abs_SHAP_sum": context_sum,
            "Share_of_All_Features": context_sum / total_sum,
            "Share_of_Gov_Physical_Only": np.nan
        }
    ])

    print("\nSHAP category shares:")
    print(shap_category_df.to_string(index=False))

    safe_to_csv(
        shap_category_df,
        "GBM-Fortify_SHAP_Category_Shares.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="dot",
        show=False
    )

    plt.title(
        "SHAP Beeswarm: Feature Impact on Prolonged Disruption Prediction",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlabel(
        "SHAP value: impact on GBM-Fortify model output"
    )

    plt.tight_layout()

    for ext in ["png", "pdf", "tiff"]:
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"GBM-Fortify_Figure_SHAP_Beeswarm.{ext}"
            ),
            dpi=600,
            bbox_inches="tight"
        )

    plt.close()

else:
    print("\nSHAP is not installed. Skipping SHAP analysis.")


# =============================================================================
# 14. DGP COEFFICIENT TABLE EXPORT
# =============================================================================

dgp_coefficients = pd.DataFrame([
    {
        "Term": "Intercept",
        "Symbol": "β0",
        "Coefficient": 9.00,
        "Interpretation": "Operational benchmark corresponding to 50% reduction from 18-day İzmit baseline"
    },
    {
        "Term": "Peak Ground Acceleration",
        "Symbol": "z(PGA)",
        "Coefficient": 1.65,
        "Interpretation": "Higher shaking increases downtime"
    },
    {
        "Term": "Peak Ground Velocity",
        "Symbol": "z(PGV)",
        "Coefficient": 0.85,
        "Interpretation": "Higher velocity increases track-bed and structural disruption risk"
    },
    {
        "Term": "Fault distance",
        "Symbol": "z(FD)",
        "Coefficient": -0.70,
        "Interpretation": "Greater distance from fault reduces expected disruption"
    },
    {
        "Term": "Approval-cycle backlog",
        "Symbol": "z(AB)",
        "Coefficient": 1.35,
        "Interpretation": "Larger approval backlog increases administrative delay"
    },
    {
        "Term": "Contractor mobilization index",
        "Symbol": "z(CMI)",
        "Coefficient": -1.10,
        "Interpretation": "Higher contractor readiness reduces downtime"
    },
    {
        "Term": "Historical corridor downtime",
        "Symbol": "z(HD)",
        "Coefficient": 0.75,
        "Interpretation": "Historical downtime proxies latent fragility"
    },
    {
        "Term": "Active approval authorities",
        "Symbol": "z(AA)",
        "Coefficient": 0.45,
        "Interpretation": "More authorities increase coordination burden"
    },
    {
        "Term": "Maintenance late-cycle status",
        "Symbol": "MS",
        "Coefficient": 0.35,
        "Interpretation": "Late maintenance cycle increases vulnerability"
    },
    {
        "Term": "Soft soil indicator",
        "Symbol": "SS",
        "Coefficient": 0.30,
        "Interpretation": "Soft soil increases disruption severity"
    },
    {
        "Term": "Winter season indicator",
        "Symbol": "WS",
        "Coefficient": 0.25,
        "Interpretation": "Winter conditions increase recovery difficulty"
    },
    {
        "Term": "PGA × soft soil",
        "Symbol": "z(PGA) × SS",
        "Coefficient": 0.55,
        "Interpretation": "Soft soil amplifies shaking effects"
    },
    {
        "Term": "Approval backlog × active authorities",
        "Symbol": "z(AB) × z(AA)",
        "Coefficient": 0.45,
        "Interpretation": "Administrative congestion worsens when more authorities are involved"
    },
    {
        "Term": "Contractor mobilization × approval backlog",
        "Symbol": "z(CMI) × z(AB)",
        "Coefficient": -0.40,
        "Interpretation": "Contractor readiness partially offsets approval backlog effects"
    },
    {
        "Term": "Noise term",
        "Symbol": "ε",
        "Coefficient": np.nan,
        "Interpretation": "ε ~ N(0, 1.25²), representing unobserved disruption factors"
    }
])

safe_to_csv(
    dgp_coefficients,
    "DGP_Coefficient_Table.csv",
    index=False
)


# =============================================================================
# 15. FINAL SUMMARY EXPORT
# =============================================================================

final_summary = pd.DataFrame([
    {
        "Primary_Model": PRIMARY_MODEL_NAME,
        "Holdout_Accuracy": primary_accuracy,
        "Holdout_AUC_ROC": primary_auc,
        "CV_Accuracy_Mean": primary_cv_accuracy,
        "CV_Accuracy_SD": primary_cv_accuracy_sd,
        "Training_Accuracy": primary_train_accuracy,
        "Train_CV_Gap": primary_train_cv_gap,
        "Physical_Only_Accuracy": physical_accuracy,
        "Hybrid_Accuracy": hybrid_accuracy,
        "Accuracy_Uplift": ablation_uplift
    }
])

safe_to_csv(
    final_summary,
    "GBM-Fortify_Final_Model_Summary.csv",
    index=False
)

print("\n================ FINAL SUMMARY ================")

print(f"Number of scenarios: {N_SCENARIOS}")
print(f"Class balance prolonged_disruption=1: {y_class.mean():.3f}")

print(f"\nPrimary model: {PRIMARY_MODEL_NAME}")
print(f"  Holdout accuracy: {primary_accuracy:.3f}")
print(f"  Holdout AUC-ROC:  {primary_auc:.3f}")
print(f"  CV accuracy mean: {primary_cv_accuracy:.3f}")
print(f"  CV accuracy SD:   {primary_cv_accuracy_sd:.3f}")
print(f"  Training accuracy:{primary_train_accuracy:.3f}")
print(f"  Train-CV gap:     {primary_train_cv_gap:.3f}")

print("\nAblation:")
print(f"  Physical-only accuracy: {physical_accuracy:.3f}")
print(f"  Hybrid accuracy:        {hybrid_accuracy:.3f}")
print(f"  Accuracy uplift:        {ablation_uplift:.3f}")

if SHAP_AVAILABLE:
    print("\nSHAP category shares saved to:")
    print(f"  {os.path.join(OUTPUT_DIR, 'GBM-Fortify_SHAP_Category_Shares.csv')}")

print("\nAll outputs saved in:")
print(f"  {OUTPUT_DIR}")

print("\nImportant manuscript note:")
print("  Use GBM-Fortify outputs as the final primary model values.")
print("  Do not mix full GBM SHAP/permutation/ablation values with GBM-Fortify performance.")
