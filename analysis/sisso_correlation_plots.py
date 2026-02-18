import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# JSC COMPUTATION
# =============================================================================
def compute_jsc():
    file_path = '../data/Extended_descriptors_Jsc_dataset.csv'
    df = pd.read_csv(file_path)
    
    # SISSO coefficients for Jsc
    a0 = 3.744042794645748e-09
    a1 = 2.643314493830391e-02
    a2 = 2.140499021222112e+04
    c0 = 1.897449756111861e+01
    
    Inertia = df['Inertia']
    Free_Volume = df['Free_Volume']
    Silhouette_Score = df['Silhouette_Score']
    Sphericity = df['Sphericity']
    Plane_Angle = df['Plane_Angle']
    Backbone_Angle = df['Backbone_Angle']
    Density = df['Density']
    
    term1 = a0 * ((Inertia * Free_Volume) / np.log(Silhouette_Score))
    term2 = a1 * np.abs((Inertia * Sphericity) - (Plane_Angle + Backbone_Angle))
    term3 = a2 * ((Density / Free_Volume) / (Sphericity ** 6))
    
    pred_values = (c0 + term1 + term2 + term3).values
    exp_values = df['Jsc'].values
    
    return exp_values, pred_values

# =============================================================================
# VOC COMPUTATION
# =============================================================================
def compute_voc():
    file_path = '../data/Extended_descriptors_Voc_dataset.csv'
    df = pd.read_csv(file_path)
    
    # SISSO coefficients for Voc
    a0 = -8.225202784056721e-04
    a1 = 3.978379967806665e-02
    a2 = -8.490265045362197e+02
    c0 = 1.312815899149556e+00
    
    Inertia = df['Inertia']
    Sphericity = df['Sphericity']
    Plane_Angle = df['Plane_Angle']
    Backbone_Angle = df['Backbone_Angle']
    Density = df['Density']
    Free_Volume = df['Free_Volume']
    
    term1 = a0 * np.abs((Inertia * Sphericity) - (Plane_Angle + Backbone_Angle))
    term2 = a1 * np.abs((Plane_Angle / Backbone_Angle) - Density)
    term3 = a2 * ((Density / Free_Volume) / (Sphericity ** 6))
    
    pred_values = (c0 + term1 + term2 + term3).values
    exp_values = df['Voc'].values
    
    return exp_values, pred_values

# =============================================================================
# FF COMPUTATION
# =============================================================================
def compute_ff():
    file_path = '../data/Extended_descriptors_FF_dataset.csv'
    df = pd.read_csv(file_path)
    
    # SISSO coefficients for FF
    a0 = -4.416711687016814e+02
    a1 = -3.112633014087646e+03
    a2 = -2.082710749980925e+05
    c0 = 1.414282553823901e+03
    
    Mean_Distance = df['Mean_Distance']
    Plane_Angle = df['Plane_Angle']
    Free_Volume = df['Free_Volume']
    Sphericity = df['Sphericity']
    
    term1 = a0 * np.abs((Mean_Distance / Plane_Angle) - np.log(Mean_Distance))
    term2 = a1 * ((Plane_Angle / Mean_Distance) / np.abs(Mean_Distance - Plane_Angle))
    term3 = a2 * ((Mean_Distance / Free_Volume) / (Sphericity ** 3))
    
    pred_values = (c0 + term1 + term2 + term3).values
    exp_values = df['FF'].values
    
    return exp_values, pred_values

# =============================================================================
# PCE COMPUTATION
# =============================================================================
def compute_pce():
    file_path = '../data/Extended_descriptors_PCE_dataset.csv'
    df = pd.read_csv(file_path)
    
    # SISSO coefficients for PCE
    a0 = -6.953607399457429e+00
    a1 = 4.048810152677110e+01
    a2 = -4.847787651855820e+06
    c0 = -2.044838763921736e+00
    
    Silhouette_Score = df['Silhouette_Score']
    Density = df['Density']
    Inertia = df['Inertia']
    Mean_Distance = df['Mean_Distance']
    Plane_Angle = df['Plane_Angle']
    
    term1 = a0 * np.abs(np.sqrt(Silhouette_Score) - (Density - Silhouette_Score))
    term2 = a1 * (np.log(Inertia + 1e-10) / (Density * Mean_Distance + 1e-10))
    term3 = a2 * (np.exp(Mean_Distance) / ((Plane_Angle ** 6) + 1e-10))
    
    pred_values = (c0 + term1 + term2 + term3).values
    exp_values = df['PCE'].values
    
    return exp_values, pred_values

# =============================================================================
# PLOTTING FUNCTION FOR SINGLE SUBPLOT
# =============================================================================
def plot_correlation(ax, exp_values, pred_values, xlabel, ylabel):
    """Plot correlation on a given axis."""
    # Calculate metrics
    pearson_r = np.corrcoef(exp_values, pred_values)[0, 1]
    r2 = r2_score(exp_values, pred_values)
    rmse = np.sqrt(mean_squared_error(exp_values, pred_values))
    max_ae = np.max(np.abs(exp_values - pred_values))
    
    # Linear regression
    X = exp_values.reshape(-1, 1)
    y = pred_values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    # Scatter plot
    ax.scatter(exp_values, pred_values, alpha=0.7, color='steelblue', s=50, 
               edgecolors='white', linewidth=0.5)
    ax.plot(exp_values, y_pred.flatten(), color='darkorange', linewidth=2, 
            label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')
    ax.plot([exp_values.min(), exp_values.max()], [exp_values.min(), exp_values.max()], 
            'k--', alpha=0.7, linewidth=1, label='Ideal (y=x)')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    
    # Metrics text box
    textstr = f'r = {pearson_r:.4f}\nR$^2$ = {r2:.4f}\nRMSE = {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    
    return pearson_r, r2, rmse, max_ae

# =============================================================================
# MAIN: CREATE 2x2 COMBINED PLOT
# =============================================================================
if __name__ == "__main__":
    print("Computing SISSO predictions for all properties...")
    
    # Compute all predictions
    jsc_exp, jsc_pred = compute_jsc()
    voc_exp, voc_pred = compute_voc()
    ff_exp, ff_pred = compute_ff()
    pce_exp, pce_pred = compute_pce()
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Jsc plot (top-left)
    jsc_xlabel = r'$J_{\mathrm{SC}}^{\mathrm{exp}}$ [mA cm$^{-2}$]'
    jsc_ylabel = r'$J_{\mathrm{SC}}^{\mathrm{SISSO}}$ [mA cm$^{-2}$]'
    r_jsc, r2_jsc, rmse_jsc, maxae_jsc = plot_correlation(axes[0, 0], jsc_exp, jsc_pred, jsc_xlabel, jsc_ylabel)
    
    # Voc plot (top-right)
    voc_xlabel = r'$V_{\mathrm{OC}}^{\mathrm{exp}}$ [V]'
    voc_ylabel = r'$V_{\mathrm{OC}}^{\mathrm{SISSO}}$ [V]'
    r_voc, r2_voc, rmse_voc, maxae_voc = plot_correlation(axes[0, 1], voc_exp, voc_pred, voc_xlabel, voc_ylabel)
    
    # FF plot (bottom-left)
    ff_xlabel = r'$FF^{\mathrm{exp}}$'
    ff_ylabel = r'$FF^{\mathrm{SISSO}}$'
    r_ff, r2_ff, rmse_ff, maxae_ff = plot_correlation(axes[1, 0], ff_exp, ff_pred, ff_xlabel, ff_ylabel)
    
    # PCE plot (bottom-right)
    pce_xlabel = r'$PCE_{\mathrm{max}}^{\mathrm{exp}}$ [%]'
    pce_ylabel = r'$PCE_{\mathrm{max}}^{\mathrm{SISSO}}$ [%]'
    r_pce, r2_pce, rmse_pce, maxae_pce = plot_correlation(axes[1, 1], pce_exp, pce_pred, pce_xlabel, pce_ylabel)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('SISSO_correlation_combined_2x2.png', dpi=600, bbox_inches='tight', facecolor='white')
    print("\nCombined plot saved as 'SISSO_correlation_combined_2x2.png'")
    
    # Print summary
    print("\n" + "="*70)
    print("SISSO MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\nJsc:  r = {r_jsc:.4f}, R² = {r2_jsc:.4f}, RMSE = {rmse_jsc:.4f}")
    print(f"Voc:  r = {r_voc:.4f}, R² = {r2_voc:.4f}, RMSE = {rmse_voc:.4f}")
    print(f"FF:   r = {r_ff:.4f}, R² = {r2_ff:.4f}, RMSE = {rmse_ff:.4f}")
    print(f"PCE:  r = {r_pce:.4f}, R² = {r2_pce:.4f}, RMSE = {rmse_pce:.4f}")
    print("="*70)
    
    plt.show()
