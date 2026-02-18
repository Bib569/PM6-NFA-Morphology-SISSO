import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DESCRIPTOR SYMBOL MAPPING (reused from Jsc script for consistency)
# =============================================================================
SYMBOL_MAP = {
    'Mean_Distance':        r'$r_{\mathrm{DA}}$',
    'Plane_Angle':          r'$\theta_{\mathrm{plane}}$',
    'Backbone_Angle':       r'$\theta_{\mathrm{bb}}$',
    'Repulsion':            r'$\Delta E_{\mathrm{xrep}}$',
    'Exchange_Repulsion':   r'$\Delta E_{\mathrm{xrep}}$',
    'Inertia':              r'$I$',
    'vdW_Surface_Area':     r'$A_{\mathrm{vdW}}$',
    'vdW_Area':             r'$A_{\mathrm{vdW}}$',
    'Sphericity':           r'$\Phi$',
    'Density':              r'$\rho$',
    'Free_Volume':          r'$V_{\mathrm{free}}$',
    'Silhouette_Score':     r'$S_{\mathrm{sil}}$',
}

def symbol_of(name):
    """Return symbol for descriptor name."""
    return SYMBOL_MAP.get(name, name)

# =============================================================================
# JSC SHAP ANALYSIS
# =============================================================================
def compute_jsc_shap():
    data_path = r"../data/Extended_descriptors_Jsc_dataset.csv"
    dataset = pd.read_csv(data_path)

    selected_descriptors = ['Inertia', 'Free_Volume', 'Silhouette_Score', 'Sphericity', 'Plane_Angle', 'Backbone_Angle', 'Density']
    predictors = dataset[selected_descriptors].values

    # SISSO coefficients for Jsc
    a0 = 3.744042794645748e-09
    a1 = 2.643314493830391e-02
    a2 = 2.140499021222112e+04
    c0 = 1.897449756111861e+01

    feature_names = ['Inertia', 'Free_Volume', 'Silhouette_Score', 'Sphericity', 'Plane_Angle', 'Backbone_Angle', 'Density']

    n_samples = len(predictors)
    n_features = 7
    shap_values = np.zeros((n_samples, n_features))

    # Term 1 contributions
    shap_values[:, 0] = a0 * (predictors[:, 1] / (np.log(predictors[:, 2] + 1e-10) + 1e-10))
    shap_values[:, 1] = a0 * (predictors[:, 0] / (np.log(predictors[:, 2] + 1e-10) + 1e-10))
    shap_values[:, 2] = -a0 * (predictors[:, 0] * predictors[:, 1]) / (predictors[:, 2] * (np.log(predictors[:, 2] + 1e-10) ** 2) + 1e-10)

    # Term 2 contributions
    term2_arg = (predictors[:, 0] * predictors[:, 3]) - (predictors[:, 4] + predictors[:, 5])
    term2_sign = np.sign(term2_arg)
    shap_values[:, 0] += a1 * term2_sign * predictors[:, 3]
    shap_values[:, 3] += a1 * term2_sign * predictors[:, 0]
    shap_values[:, 4] -= a1 * term2_sign
    shap_values[:, 5] -= a1 * term2_sign

    # Term 3 contributions
    shap_values[:, 6] += a2 / (predictors[:, 1] + 1e-10) / ((predictors[:, 3] ** 6) + 1e-10)
    shap_values[:, 1] -= a2 * predictors[:, 6] / ((predictors[:, 1] ** 2) * ((predictors[:, 3] ** 6) + 1e-10))
    shap_values[:, 3] -= 6 * a2 * (predictors[:, 6] / (predictors[:, 1] + 1e-10)) / ((predictors[:, 3] ** 7) + 1e-10)

    return predictors, shap_values, feature_names

# =============================================================================
# VOC SHAP ANALYSIS
# =============================================================================
def compute_voc_shap():
    data_path = r"../data/Extended_descriptors_Voc_dataset.csv"
    dataset = pd.read_csv(data_path)

    selected_descriptors = ['Inertia', 'Sphericity', 'Plane_Angle', 'Backbone_Angle', 'Density', 'Free_Volume']
    predictors = dataset[selected_descriptors].values

    # SISSO coefficients for Voc
    a0 = -8.225202784056721e-04
    a1 = 3.978379967806665e-02
    a2 = -8.490265045362197e+02
    c0 = 1.312815899149556e+00

    feature_names = ['Inertia', 'Sphericity', 'Plane_Angle', 'Backbone_Angle', 'Density', 'Free_Volume']

    n_samples = len(predictors)
    n_features = 6
    shap_values = np.zeros((n_samples, n_features))

    # Term 1: a0 * |(Inertia * Sphericity) - (Plane_Angle + Backbone_Angle)|
    term1_arg = (predictors[:, 0] * predictors[:, 1]) - (predictors[:, 2] + predictors[:, 3])
    term1_sign = np.sign(term1_arg)

    shap_values[:, 0] = a0 * term1_sign * predictors[:, 1]
    shap_values[:, 1] = (a0 * term1_sign * predictors[:, 0] +
                         -6 * a2 * (predictors[:, 4] / (predictors[:, 5] + 1e-10)) / ((predictors[:, 1] ** 7) + 1e-10))

    term2_ratio = predictors[:, 2] / (predictors[:, 3] + 1e-10)
    term2_arg = term2_ratio - predictors[:, 4]
    shap_values[:, 2] = (-a0 * term1_sign +
                         a1 * np.sign(term2_arg) * (1.0 / (predictors[:, 3] + 1e-10)))
    shap_values[:, 3] = (-a0 * term1_sign +
                         -a1 * np.sign(term2_arg) * (predictors[:, 2] / ((predictors[:, 3] + 1e-10) ** 2)))
    shap_values[:, 4] = (-a1 * np.sign(term2_arg) +
                         a2 * (1.0 / (predictors[:, 5] + 1e-10)) / ((predictors[:, 1] ** 6) + 1e-10))
    shap_values[:, 5] = -a2 * (predictors[:, 4] / ((predictors[:, 5] + 1e-10) ** 2)) / ((predictors[:, 1] ** 6) + 1e-10)

    return predictors, shap_values, feature_names

# =============================================================================
# FF SHAP ANALYSIS
# =============================================================================
def compute_ff_shap():
    """
    FF SISSO Equation:
    FF = c0 + a0 * |(r_DA / theta) - ln(r_DA)| 
           + a1 * (theta / r_DA) / |r_DA - theta| 
           + a2 * (r_DA / V_free) / Phi^3
    
    Features:
        Index 0: Mean_Distance (r_DA)
        Index 1: Plane_Angle (theta)
        Index 2: Free_Volume (V_free)
        Index 3: Sphericity (Phi)
    """
    data_path = r"../data/Extended_descriptors_FF_dataset.csv"
    dataset = pd.read_csv(data_path)

    selected_descriptors = ['Mean_Distance', 'Plane_Angle', 'Free_Volume', 'Sphericity']
    predictors = dataset[selected_descriptors].values

    # SISSO coefficients for FF
    a0 = -4.416711687016814e+02
    a1 = -3.112633014087646e+03
    a2 = -2.082710749980925e+05
    c0 = 1.414282553823901e+03

    feature_names = ['Mean_Distance', 'Plane_Angle', 'Free_Volume', 'Sphericity']

    n_samples = len(predictors)
    n_features = 4
    shap_values = np.zeros((n_samples, n_features))

    # Extract features for clarity
    r_DA = predictors[:, 0]    # Mean_Distance
    theta = predictors[:, 1]   # Plane_Angle
    V_free = predictors[:, 2]  # Free_Volume
    Phi = predictors[:, 3]     # Sphericity

    # Small epsilon for numerical stability
    eps = 1e-10

    # =========================================================================
    # TERM 1: a0 * |(r_DA / theta) - ln(r_DA)|
    # =========================================================================
    term1_arg = (r_DA / (theta + eps)) - np.log(r_DA + eps)
    term1_sign = np.sign(term1_arg)

    # d/d(r_DA) of |(r_DA/theta) - ln(r_DA)| = sign * (1/theta - 1/r_DA)
    term1_contrib_r_DA = a0 * term1_sign * (1.0 / (theta + eps) - 1.0 / (r_DA + eps))

    # d/d(theta) of |(r_DA/theta) - ln(r_DA)| = sign * (-r_DA / theta^2)
    term1_contrib_theta = a0 * term1_sign * (-r_DA / ((theta + eps) ** 2))

    # =========================================================================
    # TERM 2: a1 * (theta / r_DA) / |r_DA - theta|
    #       = a1 * theta / (r_DA * |r_DA - theta|)
    # 
    # EXACT DERIVATIVES using quotient rule:
    # Let f = theta / (r_DA * |r_DA - theta|)
    # =========================================================================
    abs_diff = np.abs(r_DA - theta) + eps  # |r_DA - theta|
    sign_diff = np.sign(r_DA - theta)       # sign(r_DA - theta)
    denom = r_DA * abs_diff                 # r_DA * |r_DA - theta|
    denom_sq = denom ** 2 + eps             # (r_DA * |r_DA - theta|)^2

    # d/d(r_DA) of [theta / (r_DA * |r_DA - theta|)]
    # Using quotient rule: d/dx [N/D] = -N * (dD/dx) / D^2  (when N is constant)
    # D = r_DA * |r_DA - theta|
    # dD/d(r_DA) = |r_DA - theta| + r_DA * sign(r_DA - theta)
    dD_dr_DA = abs_diff + r_DA * sign_diff
    term2_contrib_r_DA = a1 * (-theta * dD_dr_DA) / denom_sq

    # d/d(theta) of [theta / (r_DA * |r_DA - theta|)]
    # Using quotient rule: d/dx [N/D] = (D * dN/dx - N * dD/dx) / D^2
    # N = theta, dN/d(theta) = 1
    # D = r_DA * |r_DA - theta|
    # dD/d(theta) = r_DA * d|r_DA - theta|/d(theta) = r_DA * (-sign(r_DA - theta))
    dD_dtheta = -r_DA * sign_diff
    term2_contrib_theta = a1 * (denom - theta * dD_dtheta) / denom_sq

    # =========================================================================
    # TERM 3: a2 * (r_DA / V_free) / Phi^3 = a2 * r_DA / (V_free * Phi^3)
    # =========================================================================
    # d/d(r_DA) of [r_DA / (V_free * Phi^3)] = 1 / (V_free * Phi^3)
    term3_contrib_r_DA = a2 / ((V_free + eps) * (Phi ** 3 + eps))

    # d/d(V_free) of [r_DA / (V_free * Phi^3)] = -r_DA / (V_free^2 * Phi^3)
    term3_contrib_V_free = -a2 * r_DA / (((V_free + eps) ** 2) * (Phi ** 3 + eps))

    # d/d(Phi) of [r_DA / (V_free * Phi^3)] = -3 * r_DA / (V_free * Phi^4)
    term3_contrib_Phi = -3 * a2 * r_DA / ((V_free + eps) * (Phi ** 4 + eps))

    # =========================================================================
    # AGGREGATE SHAP VALUES
    # =========================================================================
    shap_values[:, 0] = term1_contrib_r_DA + term2_contrib_r_DA + term3_contrib_r_DA  # Mean_Distance
    shap_values[:, 1] = term1_contrib_theta + term2_contrib_theta                      # Plane_Angle
    shap_values[:, 2] = term3_contrib_V_free                                           # Free_Volume
    shap_values[:, 3] = term3_contrib_Phi                                              # Sphericity

    return predictors, shap_values, feature_names

# =============================================================================
# PCE SHAP ANALYSIS
# =============================================================================
def compute_pce_shap():
    data_path = r"../data/Extended_descriptors_PCE_dataset.csv"
    dataset = pd.read_csv(data_path)

    selected_descriptors = ['Silhouette_Score', 'Density', 'Inertia', 'Mean_Distance', 'Plane_Angle']
    predictors = dataset[selected_descriptors].values

    # SISSO coefficients for PCE
    a0 = -6.953607399457429e+00
    a1 = 4.048810152677110e+01
    a2 = -4.847787651855820e+06
    c0 = -2.044838763921736e+00

    feature_names = ['Silhouette_Score', 'Density', 'Inertia', 'Mean_Distance', 'Plane_Angle']

    n_samples = len(predictors)
    n_features = 5
    shap_values = np.zeros((n_samples, n_features))

    # Extract features for clarity
    S_sil = predictors[:, 0]      # Silhouette_Score
    rho = predictors[:, 1]        # Density
    I = predictors[:, 2]          # Inertia
    r_DA = predictors[:, 3]       # Mean_Distance
    theta = predictors[:, 4]      # Plane_Angle

    # Term 1: a0 * |sqrt(S_sil) - (rho - S_sil)|
    term1_arg = np.sqrt(S_sil + 1e-10) - (rho - S_sil)
    term1_sign = np.sign(term1_arg)

    # Silhouette_Score (index 0): d/dS_sil of |sqrt(S_sil) - rho + S_sil|
    # = sign * (0.5/sqrt(S_sil) + 1)
    shap_values[:, 0] = a0 * term1_sign * (0.5 / np.sqrt(S_sil + 1e-10) + 1.0)

    # Density (index 1): 
    # Term 1: d/drho of |...| = sign * (-1) = -sign
    # Term 2: d/drho of [a1 * ln(I) / (rho * r_DA)] = -a1 * ln(I) / (rho^2 * r_DA)
    shap_values[:, 1] = (-a0 * term1_sign +
                         -a1 * np.log(I + 1e-10) / ((rho ** 2) * r_DA + 1e-10))

    # Inertia (index 2):
    # Term 2: d/dI of [a1 * ln(I) / (rho * r_DA)] = a1 / (I * rho * r_DA)
    shap_values[:, 2] = a1 / ((I + 1e-10) * rho * r_DA + 1e-10)

    # Mean_Distance (index 3):
    # Term 2: d/dr_DA of [a1 * ln(I) / (rho * r_DA)] = -a1 * ln(I) / (rho * r_DA^2)
    # Term 3: d/dr_DA of [a2 * exp(r_DA) / theta^6] = a2 * exp(r_DA) / theta^6
    shap_values[:, 3] = (-a1 * np.log(I + 1e-10) / (rho * (r_DA ** 2) + 1e-10) +
                         a2 * np.exp(r_DA) / ((theta ** 6) + 1e-10))

    # Plane_Angle (index 4):
    # Term 3: d/dtheta of [a2 * exp(r_DA) / theta^6] = -6 * a2 * exp(r_DA) / theta^7
    shap_values[:, 4] = -6 * a2 * np.exp(r_DA) / ((theta ** 7) + 1e-10)

    return predictors, shap_values, feature_names

# =============================================================================
# GENERIC PLOTTER FOR A SINGLE FIGURE
# =============================================================================
def plot_shap_single(predictors, shap_values, feature_names, filename):
    n_samples, n_features = shap_values.shape
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(range(n_features), key=lambda i: mean_abs_shap[i], reverse=True)
    ordered_features = [feature_names[i] for i in feature_importance]
    ordered_feature_labels = [symbol_of(n) for n in ordered_features]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = None

    for idx, feature_idx in enumerate(feature_importance):
        feature_vals = predictors[:, feature_idx]
        min_val = feature_vals.min()
        max_val = feature_vals.max()
        normalized_vals = (feature_vals - min_val) / (max_val - min_val + 1e-10)

        shap_vals = shap_values[:, feature_idx]
        y_pos = np.full(n_samples, n_features - 1 - idx)
        jitter = np.random.normal(0, 0.01, n_samples)

        scatter = ax.scatter(shap_vals, y_pos + jitter, c=normalized_vals, cmap='coolwarm',
                             alpha=0.8, s=120, edgecolors='none', vmin=0, vmax=1)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(ordered_feature_labels[::-1], fontsize=12)
    ax.set_xlabel('SHAP value', fontsize=13, fontweight='bold')
    ax.set_ylabel('Features', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')
    ax.set_facecolor('#E8E8F0')
    #ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Feature value', fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    #print(f"Saved {title} plot to '{filename}'")

# =============================================================================
# MAIN: GENERATE FOUR SEPARATE PLOTS
# =============================================================================
if __name__ == "__main__":
    print("Computing SHAP values for Jsc...")
    jsc_pred, jsc_shap, jsc_features = compute_jsc_shap()
    plot_shap_single(jsc_pred, jsc_shap, jsc_features, 'shap_sisso_Jsc_enhanced.png')

    print("Computing SHAP values for Voc...")
    voc_pred, voc_shap, voc_features = compute_voc_shap()
    plot_shap_single(voc_pred, voc_shap, voc_features, 'shap_sisso_Voc_enhanced.png')

    print("Computing SHAP values for FF...")
    ff_pred, ff_shap, ff_features = compute_ff_shap()
    plot_shap_single(ff_pred, ff_shap, ff_features, 'shap_sisso_FF_enhanced.png')

    print("Computing SHAP values for PCE...")
    pce_pred, pce_shap, pce_features = compute_pce_shap()
    plot_shap_single(pce_pred, pce_shap, pce_features, 'shap_sisso_PCE_enhanced.png')

    print("\nAll separate SHAP plots generated.")
