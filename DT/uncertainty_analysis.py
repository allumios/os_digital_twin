"""
Uncertainty analysis module.
Propagates geometric measurement tolerances through the forward model
using Monte Carlo sampling. This is separate from the BMU, which only
accounts for frequency measurement noise. The purpose is to quantify
how much of the total stiffness uncertainty comes from each source.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from forward_model import solve_eigen

FD = "figures"
os.makedirs(FD, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.grid': True,
    'grid.alpha': 0.25, 'savefig.dpi': 300,
})


def propagate_geometric(nom, tols, n_mc=10000, seed=42):
    """Monte Carlo propagation of measurement tolerances."""
    rng = np.random.default_rng(seed)
    f_mc = np.zeros((n_mc, 3))
    k_mc = np.zeros((n_mc, 3))
    for i in range(n_mc):
        L1 = rng.uniform(nom['L1'] - tols['L1'], nom['L1'] + tols['L1'])
        L2 = rng.uniform(nom['L2'] - tols['L2'], nom['L2'] + tols['L2'])
        L3 = rng.uniform(nom['L3'] - tols['L3'], nom['L3'] + tols['L3'])
        d  = rng.uniform(nom['d']  - tols['d'],  nom['d']  + tols['d'])
        b  = rng.uniform(nom['b']  - tols['b'],  nom['b']  + tols['b'])
        E  = rng.uniform(nom['E']  - tols['E'],  nom['E']  + tols['E'])
        m  = rng.uniform(nom['m']  - tols['m'],  nom['m']  + tols['m'])
        I = b * d**3 / 12.0
        k = [4 * 12 * E * I / Li**3 for Li in [L1, L2, L3]]
        k_mc[i, :] = k
        f_mc[i, :], _ = solve_eigen(k, [m, m, m])
    return f_mc, k_mc


def sensitivity(nom, tols):
    """One-at-a-time sensitivity for each parameter."""
    I_nom = nom['b'] * nom['d']**3 / 12.0
    k_nom = [4*12*nom['E']*I_nom/Li**3 for Li in [nom['L1'], nom['L2'], nom['L3']]]
    f_nom, _ = solve_eigen(k_nom, [nom['m']]*3)
    results = {}
    for pname in ['L1','L2','L3','d','b','E','m']:
        hi_args = dict(nom); lo_args = dict(nom)
        hi_args[pname] = nom[pname] + tols[pname]
        lo_args[pname] = nom[pname] - tols[pname]
        for label, args in [('hi', hi_args), ('lo', lo_args)]:
            I = args['b'] * args['d']**3 / 12.0
            k = [4*12*args['E']*I/Li**3 for Li in [args['L1'], args['L2'], args['L3']]]
            f, _ = solve_eigen(k, [args['m']]*3)
            if label == 'hi': f_hi, k_hi = f, k
            else: f_lo, k_lo = f, k
        df = np.abs(f_hi - f_lo) / 2
        dk_pct = max(abs(k_hi[i] - k_lo[i]) / (2*k_nom[i]) * 100 for i in range(3))
        results[pname] = {'df': df, 'dk_pct': dk_pct}
    return results, f_nom, k_nom


def plot_budget(k_post_std, k_mc_std, fname):
    """Uncertainty budget: BMU vs geometric vs combined."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(3); w = 0.22
    labels = ['k1 (bottom)', 'k2 (middle)', 'k3 (top)']
    combined = np.sqrt(k_post_std**2 + k_mc_std**2)
    b1 = ax.bar(x - w, k_post_std, w, label='BMU posterior (frequency noise)',
                color='C0', edgecolor='0.2')
    b2 = ax.bar(x, k_mc_std, w, label='Geometric MC (measurement tolerances)',
                color='C1', edgecolor='0.2')
    b3 = ax.bar(x + w, combined, w, label='Combined (RSS)',
                color='C3', edgecolor='0.2')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Uncertainty, std (N/m)')
    ax.set_title('Uncertainty budget by source')
    ax.legend(fontsize=8)
    for bars in [b1, b2, b3]:
        for b in bars:
            ax.annotate(f'{b.get_height():,.0f}',
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FD, fname), dpi=300)
    plt.close()
    print(f"  saved {fname}")


def plot_tornado(sens, sigma_freq, f_nom, fname):
    """Sensitivity tornado: which tolerance affects each frequency most."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    order = ['d','L1','L2','L3','E','b','m']
    descs = {'d': 'Column depth', 'L1': 'Storey 1 height', 'L2': 'Storey 2 height',
             'L3': 'Storey 3 height', 'E': "Young's modulus", 'b': 'Column width',
             'm': 'Floor mass'}
    for mode, ax in enumerate(axes):
        vals = [sens[p]['df'][mode] for p in order]
        colors = ['C3' if v > sigma_freq[mode] else 'C0' if v > 0.01 else '0.7'
                  for v in vals]
        y = np.arange(len(order))
        ax.barh(y, vals, color=colors, edgecolor='0.3', height=0.55)
        ax.axvline(sigma_freq[mode], color='green', ls=':', lw=1.2,
                   label=f'sigma_freq = {sigma_freq[mode]:.4f}')
        ax.set_yticks(y)
        if mode == 0:
            ax.set_yticklabels([descs[p] for p in order])
        ax.set_xlabel('Delta f (Hz)')
        ax.set_title(f'Mode {mode+1}, f = {f_nom[mode]:.1f} Hz', fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
    plt.suptitle('Sensitivity of frequencies to measurement tolerances', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FD, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {fname}")
