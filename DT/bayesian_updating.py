"""
Bayesian model updating using TMCMC.
Frequency-only Gaussian likelihood. MAC is computed separately as a diagnostic.
Based on the algorithm by Ching and Chen (2007), following Lye et al. (2021).
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from forward_model import solve_eigen

FD = "figures"
os.makedirs(FD, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.grid': True,
    'grid.alpha': 0.25, 'savefig.dpi': 300, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 9,
})


def log_lik(theta, masses, f_meas, sigma):
    """
    Gaussian log-likelihood on frequency residuals.
    MAC is not part of the likelihood. It is evaluated after the updating
    as a separate check on mode shape agreement.
    """
    f_pred, _ = solve_eigen(list(theta), masses)
    return -0.5 * np.sum(((f_meas - f_pred) / sigma)**2)


def tmcmc(logl_fn, lo, hi, ns=1000, beta=0.2, seed=42):
    """
    Transitional Markov Chain Monte Carlo.
    Samples from p_j(theta) ~ L(theta)^pj * prior(theta),
    with pj increasing from 0 to 1 across stages.
    """
    rng = np.random.default_rng(seed)
    nd = len(lo)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    theta = rng.uniform(lo, hi, size=(ns, nd))
    pj = 0.0
    stages = []
    while pj < 1.0:
        logL = np.array([logl_fn(t) for t in theta])
        # find next tempering parameter by bisection
        def covw(dp):
            lw = dp * logL - np.max(dp * logL)
            w = np.exp(lw)
            return np.std(w) / np.mean(w) - 1.0
        lp, hp = 0.0, 1.0 - pj
        if covw(hp) <= 0:
            pj1 = 1.0
        else:
            for _ in range(50):
                mid = (lp + hp) / 2
                if covw(mid) > 0: hp = mid
                else: lp = mid
            pj1 = pj + lp
        # importance weights
        logw = (pj1 - pj) * logL
        w = np.exp(logw - np.max(logw))
        wn = w / np.sum(w)
        # weighted covariance for the proposal
        mu = np.sum(wn[:, None] * theta, axis=0)
        diff = theta - mu
        cov = sum(wn[i] * np.outer(diff[i], diff[i]) for i in range(ns))
        pc = beta**2 * cov + 1e-10 * np.eye(nd)
        # resample and run MH chains
        idx = rng.choice(ns, size=ns, replace=True, p=wn)
        tn = np.empty((ns, nd))
        acc = 0
        for i in range(ns):
            cur = theta[idx[i]].copy()
            lc = pj1 * logl_fn(cur)
            for _ in range(3):
                cand = cur + rng.multivariate_normal(np.zeros(nd), pc)
                if np.all(cand >= lo) and np.all(cand <= hi):
                    la = pj1 * logl_fn(cand)
                    if np.log(rng.random()) < (la - lc):
                        cur = cand; lc = la; acc += 1
            tn[i] = cur
        ar = acc / (ns * 3)
        km = np.mean(tn, axis=0)
        ks = np.std(tn, axis=0)
        stages.append({'p': pj1, 'mean': km.copy(), 'std': ks.copy(), 'acc': ar})
        print(f"  stage {len(stages):>2d}: p = {pj1:.4f}, "
              f"k = [{km[0]:.0f}, {km[1]:.0f}, {km[2]:.0f}], "
              f"std = [{ks[0]:.0f}, {ks[1]:.0f}, {ks[2]:.0f}], "
              f"acc = {ar:.1%}")
        theta = tn
        pj = pj1
    return theta, stages


# plotting

def plot_prior_posterior(samples, k_lo, k_hi, k_base, fname):
    """
    Prior and posterior on the same axes for each parameter.
    Uniform prior shown as a flat PDF line, posterior as a histogram.
    This is the standard representation in BMU literature (Lye et al., 2021).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ['k1 (bottom storey)', 'k2 (middle storey)', 'k3 (top storey)']
    prior_height = 1.0 / (k_hi - k_lo)
    for i, ax in enumerate(axes):
        # posterior histogram
        ax.hist(samples[:, i], bins=40, density=True, alpha=0.6,
                color='C3', edgecolor='0.4', lw=0.4, label='Posterior', zorder=3)
        # uniform prior as a flat line
        xp = [k_lo, k_lo, k_hi, k_hi]
        yp = [0, prior_height, prior_height, 0]
        ax.plot(xp, yp, color='C0', lw=1.5, label='Prior (uniform)', zorder=2)
        # analytical reference
        ax.axvline(k_base[i], color='0.4', ls=':', lw=1,
                   label=f'Analytical k_ref = {k_base[i]:,.0f}', zorder=1)
        # posterior mean
        m = np.mean(samples[:, i])
        ax.axvline(m, color='C3', ls='-', lw=1.2,
                   label=f'Posterior mean = {m:,.0f}', zorder=4)
        ax.set_xlabel(f'{labels[i]} (N/m)')
        ax.set_ylabel('Probability density')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(labels[i], fontsize=11)
    plt.suptitle('Prior and posterior distributions', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FD, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {fname}")


def plot_marginals(samples, fname):
    """Marginal histograms with mean and 1-sigma, 2-sigma bands."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ['k1 (bottom)', 'k2 (middle)', 'k3 (top)']
    for i, ax in enumerate(axes):
        m, s = np.mean(samples[:, i]), np.std(samples[:, i])
        ax.hist(samples[:, i], bins=40, density=True, alpha=0.65,
                color='C0', edgecolor='0.4', lw=0.4)
        ax.axvline(m, color='C3', lw=1.2, label=f'mean = {m:,.0f}')
        ax.axvspan(m-s, m+s, alpha=0.12, color='C3', label=f'1 std = {s:,.0f}')
        ax.axvspan(m-2*s, m+2*s, alpha=0.06, color='C3', label=f'2 std = {2*s:,.0f}')
        ax.set_xlabel(f'{labels[i]} (N/m)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
    plt.suptitle('Posterior marginal distributions', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FD, fname), dpi=300)
    plt.close()
    print(f"  saved {fname}")


def plot_pairwise(samples, fname):
    """Pairwise scatter with correlation values."""
    pairs = [(0,1,'k1','k2'), (0,2,'k1','k3'), (1,2,'k2','k3')]
    corr = np.corrcoef(samples.T)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        ax.scatter(samples[:, i], samples[:, j], s=1.5, alpha=0.3, color='C3')
        ax.set_xlabel(f'{xl} (N/m)')
        ax.set_ylabel(f'{yl} (N/m)')
        ax.set_title(f'r = {corr[i,j]:.3f}', fontsize=10)
    plt.suptitle('Posterior pairwise scatter', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FD, fname), dpi=300)
    plt.close()
    print(f"  saved {fname}")


def plot_3d(samples, fname):
    """3D scatter of posterior samples."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[:,0], samples[:,1], samples[:,2],
               s=1, alpha=0.25, c='C3', edgecolors='none')
    ax.set_xlabel('\nk1 (N/m)', fontsize=9, labelpad=12)
    ax.set_ylabel('\nk2 (N/m)', fontsize=9, labelpad=12)
    ax.set_zlabel('\nk3 (N/m)', fontsize=9, labelpad=12)
    ax.tick_params(axis='x', labelsize=7, pad=4)
    ax.tick_params(axis='y', labelsize=7, pad=4)
    ax.tick_params(axis='z', labelsize=7, pad=4)
    ax.view_init(elev=25, azim=135)
    ax.set_title('3D posterior distribution', fontsize=11, pad=20)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)
    plt.savefig(os.path.join(FD, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {fname}")


def plot_freq_comp(fm, fp, fname):
    """Measured vs posterior frequencies."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3); w = 0.3
    ax.bar(x - w/2, fm, w, label='Measured', color='0.35', edgecolor='0.1')
    ax.bar(x + w/2, fp, w, label='Posterior model', color='C3', alpha=0.75, edgecolor='0.1')
    for i in range(3):
        ax.annotate(f'{fm[i]:.2f}', (x[i]-w/2, fm[i]), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=8)
        ax.annotate(f'{fp[i]:.2f}', (x[i]+w/2, fp[i]), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['f1', 'f2', 'f3'])
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Measured vs posterior predicted frequencies')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FD, fname), dpi=300)
    plt.close()
    print(f"  saved {fname}")


def plot_convergence(stages, fname):
    """TMCMC convergence: tempering parameter and stiffness evolution."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    it = range(1, len(stages) + 1)
    a1.plot(it, [s['p'] for s in stages], 'o-', color='C0', ms=4)
    a1.set(xlabel='Stage', ylabel='Tempering parameter p',
           title='TMCMC tempering', ylim=[-0.05, 1.1])
    for j, (c, l) in enumerate(zip(['C0','C3','C2'], ['k1','k2','k3'])):
        a2.errorbar(list(it), [s['mean'][j] for s in stages],
                    yerr=[s['std'][j] for s in stages],
                    fmt='o-', color=c, ms=3, capsize=3, lw=0.8, label=l)
    a2.set(xlabel='Stage', ylabel='Stiffness (N/m)',
           title='Parameter convergence')
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FD, fname), dpi=300)
    plt.close()
    print(f"  saved {fname}")
