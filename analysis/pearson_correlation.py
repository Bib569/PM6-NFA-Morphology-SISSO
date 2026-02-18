#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Pearson Correlation Grid Plots for All 4 Target Properties
====================================================================
Creates 4 separate Pearson correlation grid plots (no titles) for:
- Jsc (from Extended_descriptors_Jsc_dataset.csv)
- Voc (from Extended_descriptors_Voc_dataset.csv)
- FF (from Extended_descriptors_FF_dataset.csv)
- PCE (from Extended_descriptors_PCE_dataset.csv)
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# =============================================================================
# DESCRIPTOR SYMBOL MAPPING
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
# CONFIGURATION
# =============================================================================
CONFIG = {
    'dpi': 600,
    'figsize': (18, 14)
}

# Property configurations: (csv_file, target_column, output_file)
PROPERTIES = [
    ('../data/Extended_descriptors_Jsc_dataset.csv', 'Jsc', 'Pearson_corr_grid_Jsc.png'),
    ('../data/Extended_descriptors_Voc_dataset.csv', 'Voc', 'Pearson_corr_grid_Voc.png'),
    ('../data/Extended_descriptors_FF_dataset.csv', 'FF', 'Pearson_corr_grid_FF.png'),
    ('../data/Extended_descriptors_PCE_dataset.csv', 'PCE', 'Pearson_corr_grid_PCE.png'),
]

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================
def create_pearson_plot(df, target_column, descriptors, output_file):
    """Create Pearson correlation grid plot with KDE for a specific target (no title)."""
    
    n_descriptors = len(descriptors)
    n_cols = 3
    n_rows = int(np.ceil(n_descriptors / n_cols))
    
    print(f"\n  Creating Pearson correlation grid for {target_column} ({n_rows}x{n_cols})...")
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=CONFIG['figsize'])
    axes = axes.flatten()
    
    pearson_values = {}
    
    for i, descriptor in enumerate(descriptors):
        ax = axes[i]
        
        # Calculate Pearson correlation
        pearson_corr = df[target_column].corr(df[descriptor])
        pearson_values[descriptor] = pearson_corr
        
        x_data = df[target_column]
        y_data = df[descriptor]
        
        # Scatter plot
        sns.scatterplot(
            x=x_data, 
            y=y_data, 
            ax=ax, 
            color='blue', 
            s=10, 
            alpha=0.7, 
            edgecolor='w', 
            linewidth=0.5
        )
        
        # KDE plot
        try:
            sns.kdeplot(
                x=x_data, 
                y=y_data, 
                ax=ax, 
                fill=True, 
                cmap="Blues", 
                bw_method='scott', 
                common_norm=False,
                alpha=0.5
            )
        except Exception as e:
            print(f"   Warning: Could not create KDE for {descriptor}: {e}")
        
        # Pearson correlation annotation
        ax.text(
            0.05, 0.90, 
            f'r={pearson_corr:.2f}', 
            transform=ax.transAxes, 
            fontstyle='oblique', 
            fontsize=16,
            color='brown',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Labels
        ax.set_xlabel(symbol_of(target_column), fontsize=12)
        ax.set_ylabel(symbol_of(descriptor), fontsize=12)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
    
    # Remove empty subplots
    for j in range(n_descriptors, len(axes)):
        fig.delaxes(axes[j])
    
    # NO TITLE - as requested
    # Adjust layout
    plt.tight_layout()
    
    # Save
    print(f"  Saving plot to: {output_file}")
    plt.savefig(output_file, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    return pearson_values

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*80)
    print("COMBINED PEARSON CORRELATION PLOTS FOR ALL 4 PROPERTIES")
    print("="*80)
    
    all_results = {}
    
    for csv_file, target_column, output_file in PROPERTIES:
        print(f"\n{'='*60}")
        print(f"Processing: {target_column}")
        print(f"{'='*60}")
        
        # Load data
        print(f"  Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"  Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Get descriptors (all numerical columns except target)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        descriptors = [col for col in numerical_cols if col != target_column]
        
        print(f"  Found {len(descriptors)} descriptors")
        
        # Create plot
        pearson_values = create_pearson_plot(df, target_column, descriptors, output_file)
        
        # Store results
        all_results[target_column] = pearson_values
        
        # Print top 5 correlations
        sorted_desc = sorted(descriptors, key=lambda x: abs(pearson_values[x]), reverse=True)
        print(f"\n  Top 5 descriptors for {target_column} (by |r|):")
        for i, desc in enumerate(sorted_desc[:5], 1):
            print(f"    {i}. {symbol_of(desc)} ({desc}): r = {pearson_values[desc]:.4f}")
    
    # Final summary
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nOutput files:")
    for csv_file, target_column, output_file in PROPERTIES:
        print(f"  - {output_file}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
