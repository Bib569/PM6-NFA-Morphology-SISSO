#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Mutual Information + KDE Grid Plots for all 4 target properties
- Jsc (Extended_descriptors_Jsc_dataset.csv)
- Voc (Extended_descriptors_Voc_dataset.csv)
- FF  (Extended_descriptors_FF_dataset.csv)
- PCE (Extended_descriptors_PCE_dataset.csv)
No plot titles are rendered (as requested).
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

# =============================================================================
# DESCRIPTOR / TARGET SYMBOLS
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
    'Jsc':                  r'$J_{\mathrm{SC}}$',
    'Voc':                  r'$V_{\mathrm{OC}}$',
    'FF':                   r'$FF$',
    'PCE':                  r'$PCE_{\mathrm{max}}$',
}

def symbol_of(name):
    return SYMBOL_MAP.get(name, name)

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    'dpi': 600,
    'figsize': (18, 14),
    'n_neighbors': 3,
    'random_state': 42,
}

# (csv_file, target_column, output_png)
PROPERTIES = [
    ('../data/Extended_descriptors_Jsc_dataset.csv', 'Jsc', 'MI_kde_grid_Jsc.png'),
    ('../data/Extended_descriptors_Voc_dataset.csv', 'Voc', 'MI_kde_grid_Voc.png'),
    ('../data/Extended_descriptors_FF_dataset.csv',  'FF',  'MI_kde_grid_FF.png'),
    ('../data/Extended_descriptors_PCE_dataset.csv', 'PCE', 'MI_kde_grid_PCE.png'),
]

# =============================================================================
# HELPERS
# =============================================================================
def calculate_mi_for_target(df, target_column, descriptors):
    """Compute mutual information for each descriptor vs target."""
    print(f"  Calculating MI for {target_column} ...")
    mi_values = {}
    for i, descriptor in enumerate(descriptors, 1):
        X = df[descriptor].values.reshape(-1, 1)
        y = df[target_column].values
        mi = mutual_info_regression(
            X, y,
            n_neighbors=CONFIG['n_neighbors'],
            random_state=CONFIG['random_state']
        )[0]
        mi_values[descriptor] = mi
        if i % 3 == 0:
            print(f"    Progress: {i}/{len(descriptors)}")
    return mi_values

def create_mi_kde_plot(df, target_column, descriptors, mi_values, output_file):
    """Create MI+KDE grid for one target (no suptitle)."""
    n_descriptors = len(descriptors)
    n_cols = 3
    n_rows = int(np.ceil(n_descriptors / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=CONFIG['figsize'])
    axes = axes.flatten()

    for i, descriptor in enumerate(descriptors):
        ax = axes[i]
        mi_val = mi_values[descriptor]
        x_data = df[target_column]
        y_data = df[descriptor]

        # KDE plot
        try:
            sns.kdeplot(
                x=x_data,
                y=y_data,
                ax=ax,
                fill=True,
                cmap="Blues",
                bw_adjust=0.8,
                thresh=0.05,
                levels=10,
                alpha=0.8
            )
        except Exception as e:
            print(f"   Warning: KDE failed for {descriptor}: {e}; falling back to scatter.")
            ax.scatter(x_data, y_data, alpha=0.5, s=20, c='blue')

        # MI annotation
        ax.text(
            0.05, 0.90,
            f'MI={mi_val:.2f}',
            transform=ax.transAxes,
            fontstyle='italic',
            fontsize=14,
            color='brown',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        # Labels
        ax.set_xlabel(symbol_of(target_column), fontsize=12)
        ax.set_ylabel(symbol_of(descriptor), fontsize=12)

        # Style
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.5)
        ax.set_facecolor('#f8f9fa')

    # Remove unused axes
    for j in range(n_descriptors, len(axes)):
        fig.delaxes(axes[j])

    # No suptitle (per request)
    plt.tight_layout()
    plt.savefig(output_file, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_file}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("COMBINED MI + KDE GRID PLOTS (NO TITLES)")
    print("="*80)

    for csv_file, target_column, output_file in PROPERTIES:
        print(f"\n{'-'*60}\nProcessing {target_column}\n{'-'*60}")

        # Load data
        df = pd.read_csv(csv_file)
        print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns from {csv_file}")

        # All numeric cols except the target are descriptors
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        descriptors = [col for col in numerical_cols if col != target_column]
        print(f"  Using {len(descriptors)} descriptors")

        # MI calc & plot
        mi_vals = calculate_mi_for_target(df, target_column, descriptors)
        create_mi_kde_plot(df, target_column, descriptors, mi_vals, output_file)

    print("\nAll MI+KDE plots generated successfully.")
