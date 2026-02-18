#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptor VIF Analysis Plots
==============================
Creates two plot files:
1. Combined 1x2 subplot: Initial VIF and Final VIF comparison
2. Single plot: Iterative Elimination Progress

Uses LaTeX descriptor symbols for publication-quality figures.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'vif_threshold': 5.0,
    'dpi': 300,
}

# Deep, high-contrast color palette
COLOR_PALETTE = {
    'high': '#c41e3a',        # deep crimson red
    'mid': '#ff8c00',         # dark orange
    'low': '#1f77b4',         # deep navy blue
    'threshold': '#2ca02c',   # deep forest green
    'high_line': '#d62728',   # dark red for high-VIF line
}

# Descriptor symbol mapping (raw name -> LaTeX symbol)
DESCRIPTOR_SYMBOLS = {
    'Mean_Distance': r'$r_{DA}$',
    'Plane_Angle': r'$\theta_{\mathrm{plane}}$',
    'Backbone_Angle': r'$\theta_{\mathrm{bb}}$',
    'Repulsion': r'$\Delta E_{\mathrm{xrep}}$',
    'Inertia': r'$I$',
    'vdW_surf_area': r'$A_{\mathrm{vdW}}$',
    'Sphericity': r'$\Phi$',
    'Density': r'$\rho$',
    'Free_Volume': r'$V_{\mathrm{free}}$',
    'Silhouette_Score': r'$S_{\mathrm{sil}}$',
    # Additional descriptors (using similar notation style)
    'Total': r'$E_{\mathrm{total}}$',
    'Electrostatic': r'$E_{\mathrm{elec}}$',
    'Dispersion': r'$E_{\mathrm{disp}}$',
    'Dimer_Volume': r'$V_{\mathrm{dimer}}$',
}

def get_descriptor_symbol(descriptor_name):
    """Get LaTeX symbol for a descriptor, or create one if not defined."""
    if descriptor_name in DESCRIPTOR_SYMBOLS:
        return DESCRIPTOR_SYMBOLS[descriptor_name]
    else:
        # Create a symbol for undefined descriptors
        # Convert underscore-separated words to subscript notation
        parts = descriptor_name.split('_')
        if len(parts) > 1:
            return r'$' + parts[0][0].upper() + r'_{\mathrm{' + ''.join(parts[1:]).lower() + r'}}$'
        else:
            return r'$' + descriptor_name[0].upper() + r'$'


def format_log_tick(val):
    """Custom formatter to show values like 1×10^6 on the log x-axis."""
    if val == 0:
        return "0"
    exp = int(np.floor(np.log10(val)))
    if exp >= 4:
        mant = val / (10 ** exp)
        mant_str = f"{mant:g}"
        return rf"${mant_str}\times 10^{exp}$"
    if val >= 1000:
        return f"{val:.0f}"
    if val >= 1:
        return f"{val:g}"
    return f"{val:.2g}"


def create_vif_comparison_plot():
    """Create 1x2 subplot with Initial and Final VIF comparison."""
    
    print("\n" + "="*80)
    print("CREATING VIF COMPARISON PLOT (1x2 SUBPLOT)")
    print("="*80)
    
    # Load data
    initial_vif = pd.read_csv('../data/vif/Descriptor_VIF_initial_results.csv')
    final_vif = pd.read_csv('../data/vif/Descriptor_VIF_final_results.csv')
    
    # Convert descriptor names to LaTeX symbols
    initial_vif['Symbol'] = initial_vif['Descriptor'].apply(get_descriptor_symbol)
    final_vif['Symbol'] = final_vif['Descriptor'].apply(get_descriptor_symbol)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor('white')
    for ax in (ax1, ax2):
        ax.set_facecolor('white')
        # Professional axis styling for publication - closed box with bold borders
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_visible(True)
        ax.tick_params(width=2.5, length=7, labelsize=12)
    
    vif_threshold = CONFIG['vif_threshold']
    
    # -------------------------------------------------------------------------
    # Plot 1: Initial VIF - All Descriptors (log scale)
    # -------------------------------------------------------------------------
    initial_plot = initial_vif.copy()
    
    y_pos = np.arange(len(initial_plot))
    colors = [COLOR_PALETTE['high'] if v >= 10 else COLOR_PALETTE['mid'] if v >= vif_threshold else COLOR_PALETTE['low'] 
             for v in initial_plot['VIF']]
    
    bars1 = ax1.barh(y_pos, initial_plot['VIF'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(initial_plot['Symbol'], fontsize=12)
    ax1.set_ylim(-0.5, len(initial_plot) - 0.5)
    
    ax1.set_xscale('log')
    xmin = max(0.8, initial_plot['VIF'].min() * 0.8)
    xmax = initial_plot['VIF'].max() * 1.15
    ax1.set_xlim(xmin, xmax)
    
    # Log ticks tailored to range (to better show very large VIFs if present)
    log_min = int(np.floor(np.log10(xmin)))
    log_max = int(np.ceil(np.log10(xmax)))
    ticks = []
    for exp in range(log_min, log_max + 1):
        for m in (1, 2, 5):
            tick = m * (10 ** exp)
            if xmin <= tick <= xmax:
                ticks.append(tick)
    ax1.set_xticks(ticks)
    ax1.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda val, _: format_log_tick(val)))
    ax1.tick_params(axis='x', which='both', labelsize=11, width=2, length=6)
    ax1.tick_params(axis='y', which='both', labelsize=11, width=2, length=6)
    
    ax1.set_xlabel('VIF (log scale)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Descriptors', fontsize=14, fontweight='bold')
    ax1.axvline(x=vif_threshold, color=COLOR_PALETTE['threshold'], linestyle='--', linewidth=2.2, label=f'Threshold = {vif_threshold}')
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=False, shadow=False)
    ax1.grid(False)
   # ax1.set_title('Initial VIF (all descriptors)', fontsize=13, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Plot 2: Final VIF - Retained Descriptors
    # -------------------------------------------------------------------------
    final_plot = final_vif.copy()
    
    y_pos = np.arange(len(final_plot))
    colors = [COLOR_PALETTE['high'] if v >= 10 else COLOR_PALETTE['mid'] if v >= vif_threshold else COLOR_PALETTE['low'] 
             for v in final_plot['VIF']]
    
    bars2 = ax2.barh(y_pos, final_plot['VIF'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(final_plot['Symbol'], fontsize=12)
    ax2.set_ylim(-0.5, len(final_plot) - 0.5)
 #   ax2.set_title('Final VIF (retained descriptors)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('VIF (linear)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Descriptors', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', which='both', labelsize=11, width=2, length=6)
    ax2.tick_params(axis='y', which='both', labelsize=11, width=2, length=6)
    ax2.axvline(x=vif_threshold, color=COLOR_PALETTE['threshold'], linestyle='--', linewidth=2.2, label=f'Threshold = {vif_threshold}')
    ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.93, 1.0),
               framealpha=0.95, edgecolor='black', fancybox=False, shadow=False)
    ax2.grid(False)
    
    # Adjust layout
  #  fig.suptitle('Variance Inflation Factor Comparison', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    plt.savefig('VIF_comparison_plot.png', dpi=CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Saved: VIF_comparison_plot.png")
    
    # Save plot data to CSV
    initial_plot_data = initial_vif[['Descriptor', 'VIF']].copy()
    initial_plot_data['Symbol'] = initial_vif['Symbol']
    initial_plot_data.to_csv('VIF_comparison_initial_data.csv', index=False)
    
    final_plot_data = final_vif[['Descriptor', 'VIF']].copy()
    final_plot_data['Symbol'] = final_vif['Symbol']
    final_plot_data.to_csv('VIF_comparison_final_data.csv', index=False)
    
    print("✅ Saved: VIF_comparison_initial_data.csv")
    print("✅ Saved: VIF_comparison_final_data.csv")
    
    plt.close()


def create_iteration_progress_plot():
    """Create single plot showing iterative elimination progress."""
    
    print("\n" + "="*80)
    print("CREATING ITERATION PROGRESS PLOT")
    print("="*80)
    
    # Load iteration history
    history_df = pd.read_csv('../data/vif/Descriptor_VIF_iteration_history.csv')
    
    vif_threshold = CONFIG['vif_threshold']
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Create twin axis for number of descriptors
    ax2 = ax1.twinx()
    
    # Plot Max VIF on primary axis
    line1 = ax1.plot(history_df['Iteration'], history_df['Max_VIF'], 
                     'o-', color='#e74c3c', linewidth=2.5, markersize=10, 
                     markerfacecolor='white', markeredgewidth=2, label='Max VIF')
    
    # Add threshold line
    ax1.axhline(y=vif_threshold, color='#3498db', linestyle='--', 
                linewidth=2.5, label=f'Threshold = {vif_threshold}')
    
    # Plot number of descriptors on secondary axis
    line2 = ax2.plot(history_df['Iteration'], history_df['Num_Descriptors'], 
                     's-', color='#27ae60', linewidth=2.5, markersize=10,
                     markerfacecolor='white', markeredgewidth=2, label='Number of Descriptors')
    
    # Axis labels
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max VIF', fontsize=12, fontweight='bold', color='#e74c3c')
    ax2.set_ylabel('Number of Descriptors', fontsize=12, fontweight='bold', color='#27ae60')
    
    # Tick colors
    ax1.tick_params(axis='y', labelcolor='#e74c3c', labelsize=11)
    ax2.tick_params(axis='y', labelcolor='#27ae60', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10,
               framealpha=0.9, edgecolor='gray')
    
    # Grid
    ax1.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set x-axis to show integer iterations
    ax1.set_xticks(history_df['Iteration'])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('VIF_iteration_progress_plot.png', dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Saved: VIF_iteration_progress_plot.png")
    
    # Save plot data to CSV
    history_df.to_csv('VIF_iteration_progress_data.csv', index=False)
    print("✅ Saved: VIF_iteration_progress_data.csv")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("DESCRIPTOR VIF ANALYSIS - PLOT GENERATION")
    print("="*80)
    print("Creating:")
    print("  1. VIF Comparison Plot (1x2 subplot: Initial vs Final)")
    print("  2. Iteration Progress Plot (single plot)")
    print("="*80)
    
    # Create plots
    create_vif_comparison_plot()
    create_iteration_progress_plot()
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  • VIF_comparison_plot.png")
    print("  • VIF_comparison_initial_data.csv")
    print("  • VIF_comparison_final_data.csv")
    print("  • VIF_iteration_progress_plot.png")
    print("  • VIF_iteration_progress_data.csv")
    print("="*80)



