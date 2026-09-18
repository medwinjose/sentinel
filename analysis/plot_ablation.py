import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pareto(df, regime_name, ax):
    # Filter by regime
    regime_df = df[df['Regime'] == regime_name]
    
    # Calculate mean stats
    mean_base_pnl = regime_df['Base_PnL'].mean()
    mean_base_var = regime_df['Base_Inv_Var'].mean()
    mean_mod_pnl = regime_df['Mod_PnL'].mean()
    mean_mod_var = regime_df['Mod_Inv_Var'].mean()
    
    # Scatter all points for background
    ax.scatter(regime_df['Base_Inv_Var'], regime_df['Base_PnL'], alpha=0.1, color='blue', label='Base (Individual)')
    ax.scatter(regime_df['Mod_Inv_Var'], regime_df['Mod_PnL'], alpha=0.1, color='orange', label='Trust-Mod (Individual)')
    
    # Plot means
    ax.scatter(mean_base_var, mean_base_pnl, color='darkblue', marker='X', s=200, label='Base (Mean)')
    ax.scatter(mean_mod_var, mean_mod_pnl, color='darkred', marker='X', s=200, label='Trust-Mod (Mean)')
    
    # Arrow showing movement
    ax.annotate('', xy=(mean_mod_var, mean_mod_pnl), xytext=(mean_base_var, mean_base_pnl),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    
    ax.set_title(f'Regime: {regime_name}')
    ax.set_xlabel('Inventory Variance (Risk)')
    ax.set_ylabel('PnL (Reward)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

def main():
    results_path = '../engine/benches/results/sweep_results.csv'
    if not os.path.exists(results_path):
        print(f"Error: Could not find {results_path}")
        return
        
    df = pd.read_csv(results_path)
    regimes = df['Regime'].unique()
    
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 6))
    if len(regimes) == 1:
        axes = [axes]
        
    for i, regime in enumerate(regimes):
        plot_pareto(df, regime, axes[i])
        
    plt.tight_layout()
    
    plot_path = '../engine/benches/results/pareto_frontier.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()
