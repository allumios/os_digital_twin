"""
Main pipeline for the open-source digital twin.
Modules:
  M1: system identification (FFT, damping, frequency extraction from free vib tails)
  M2: forward model (geometry-based stiffness, priors, mode shapes)
  M3: Bayesian model updating (TMCMC, frequency-only likelihood)
  M4: validation (Session 2 free vibration tails)
  M5: uncertainty (geometric tolerance propagation)
"""
import numpy as np, os, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
from forward_model import base_k, solve_eigen
from signal_processing import (
    load_sheet, fft_amp, psd, top_peaks, est_damping, mode_shape, mac,
    plot_fft_grid, plot_overlay, plot_sigma, plot_damping, plot_modes,
)
from bayesian_updating import (
    log_lik, tmcmc, plot_prior_posterior, plot_marginals,
    plot_pairwise, plot_3d, plot_freq_comp, plot_convergence,
)
from uncertainty_analysis import (
    propagate_geometric, sensitivity, plot_budget, plot_tornado,
)

for d in ['figures', 'output/accelerograms', 'output/free_vib_windows',
          'output/amplitude', 'output/psd']:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 11, 'savefig.dpi': 300, 'lines.linewidth': 0.6,
    'axes.grid': True, 'grid.alpha': 0.25, 'legend.fontsize': 8,
})

M = np.array(MASSES)
fs = SAMPLING_RATE
KB = [base_k(E_MATERIAL, COLUMN_WIDTH, COLUMN_DEPTH, L, N_COLUMNS) for L in STOREY_HEIGHTS]


def find_max_tail(ch, fs):
    """Find where excitation ends and return the start of free decay."""
    N = len(ch); dur = N / fs
    win = int(0.5 * fs); half = win // 2
    rms = np.array([np.std(ch[i:i+win]) for i in range(0, N-win, half)])
    times = np.arange(len(rms)) * (half / fs)
    peak = np.max(rms)
    if peak == 0: return 0, dur
    above = np.where(rms > 0.25 * peak)[0]
    if len(above) == 0: return 0, dur
    last = times[above[-1]]
    start = last + 0.5
    return start, dur - start


def m1():
    """System identification from all tests using personalised free vibration tails."""
    print("\n" + "="*60)
    print("  M1: system identification")
    print("="*60)

    # FFT grids, overlays, accelerograms and individual plots
    # are generated separately by generate_plots.py to keep the main pipeline fast

    # free vibration tails with personalised maximum lengths
    # Session 1 = calibration, Session 2 = held out for validation
    
    def extract_tails(sf, ss, label):
        rows = []
        for key, sn in ss.items():
            try:
                t_arr, c1, c2, c3 = load_sheet(sf, sn)
                N = len(c3); dur = N / fs
                dstart, tail = find_max_tail(c3, fs)
                tail_s = max(1, int(tail))
                nt = int(tail_s * fs)
                rms_tail = np.std(c3[-nt:])
                rms_mid = np.std(c3[N//4:3*N//4])
                ratio = rms_tail / rms_mid if rms_mid > 0 else 0
                
                # a tail is considered clean if:
                #   1. it is at least 5 seconds long (for adequate FFT resolution)
                #   2. RMS ratio is below 0.20 (tail amplitude is at most 20% of
                #      the mid-record level, confirming the excitation has stopped)
                clean = (tail_s >= 5) and (ratio < 0.20)
                note = ''
                if not clean:
                    if ratio >= 0.20: note = '  EXCLUDED (tail partially excited)'
                    elif tail_s < 5: note = '  EXCLUDED (tail shorter than 5 s)'
                
                ch_tail = c3[-nt:]
                f, a = fft_amp(ch_tail, fs)
                pf, _ = top_peaks(f, a, n=3)
                res = fs / len(ch_tail)
                print(f"  {label:<4} {sn.strip():<22} {dur:>4.0f}s {tail_s:>4}s {res:>5.3f} "
                      f"{pf[0]:>7.2f} {pf[1]:>7.2f} {pf[2]:>7.2f}{note}")
                
                if not clean: continue  # do not include in calibration or validation
                
                rows.append({
                    'session': label, 'test': sn.strip(), 'key': key,
                    'total': dur, 'tail': tail_s, 'res': res,
                    'f1': pf[0], 'f2': pf[1], 'f3': pf[2], 'ratio': ratio,
                })
                # plot free vib window
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
                tp = np.arange(N) / fs
                ax1.plot(tp, c3, color='0.3', lw=0.2)
                ax1.axvspan(dur - tail_s, dur, alpha=0.12, color='C3',
                            label=f'Free vib window ({tail_s} s)')
                ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Accel. (a.u.)')
                ax1.set_title(f'{label} {sn.strip()}, Floor 3', fontsize=10)
                ax1.legend(fontsize=8)
                ax2.plot(f, a, color='0.2', lw=0.6)
                ax2.set_xlim([3, 38]); ax2.set_ylim(bottom=0)
                ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Amplitude')
                ax2.set_title(f'FFT of tail ({tail_s} s, res = {res:.3f} Hz)', fontsize=10)
                for fp in pf:
                    if not np.isnan(fp):
                        idx_pk = np.argmin(np.abs(f - fp))
                        ax2.plot(fp, a[idx_pk], 'o', color='C3', ms=4)
                        ax2.annotate(f'{fp:.1f}', (fp, a[idx_pk]), xytext=(5, -10),
                                     textcoords='offset points', fontsize=8, color='C3')
                plt.tight_layout()
                plt.savefig(f'output/free_vib_windows/{label}_{key}.png', dpi=200)
                plt.close()
            except: pass
        return rows

    print("\n  Session 1 (calibration):")
    print(f"  {'Ses':<4} {'Test':<22} {'Total':>5} {'Tail':>5} {'Res':>6} "
          f"{'f1':>7} {'f2':>7} {'f3':>7}")
    s1_data = extract_tails(SESSION_1_FILE, S1_SHEETS, 'S1')

    print("\n  Session 2 (validation, not used for calibration):")
    print(f"  {'Ses':<4} {'Test':<22} {'Total':>5} {'Tail':>5} {'Res':>6} "
          f"{'f1':>7} {'f2':>7} {'f3':>7}")
    s2_data = extract_tails(SESSION_2_FILE, S2_SHEETS, 'S2')

    # calibration target: mean and sigma from Session 1 only
    f1 = [d['f1'] for d in s1_data]
    f2 = [d['f2'] for d in s1_data]
    f3 = [d['f3'] for d in s1_data]
    f_mean = np.array([np.mean(f1), np.mean(f2), np.mean(f3)])
    sigma = np.array([np.std(f1), np.std(f2), np.std(f3)])
    print(f"\n  calibration target (S1 mean, n={len(f1)}):")
    print(f"    f = [{f_mean[0]:.3f}, {f_mean[1]:.3f}, {f_mean[2]:.3f}] Hz")
    print(f"    sigma = [{sigma[0]:.4f}, {sigma[1]:.4f}, {sigma[2]:.4f}] Hz")

    # sigma figure (S1 measurements only)
    plot_sigma(sigma, f_mean,
               {'f1': f1, 'f2': f2, 'f3': f3}, 'fig_sigma.png')

    # damping from Session 1 free vibration
    print("\n  damping (Session 1 free vibration):")
    _, c1, c2, c3 = load_sheet(SESSION_1_FILE, S1_SHEETS['free_vib'])
    dd = []
    for ml, fn, fl, fh, sig in [('Mode 1', 7.2, 5, 9.5, c3),
                                  ('Mode 2', 21, 17.5, 24, c1),
                                  ('Mode 3', 30.5, 27, 34, c3)]:
        try:
            z, a, A0, td, es = est_damping(sig, fs, fn, fl, fh)
            print(f"    {ml}: zeta = {z:.4f} ({z*100:.2f}%)")
            dd.append((td, es, z, a, A0, fn, ml))
        except Exception as e:
            print(f"    {ml}: failed ({e})")
    if dd: plot_damping(dd, 'fig_damping.png')

    # mode shapes from S1 harmonic tests at the S1 mean frequencies
    phi_exp = []
    for i, (key, ft) in enumerate(zip(['mode1','mode2','mode3'], f_mean)):
        phi = mode_shape(SESSION_1_FILE, S1_SHEETS[key], fs, ft)
        phi_exp.append(phi)

    return sigma, f_mean, phi_exp, s1_data, s2_data


def m2(phi_exp, f_meas):
    """Forward model at geometry-based stiffness."""
    print("\n" + "="*60)
    print("  M2: forward model")
    print("="*60)
    for i, (L, k) in enumerate(zip(STOREY_HEIGHTS, KB)):
        print(f"  storey {i+1}: L = {L*1000:.1f} mm, k_ref = {k:,.0f} N/m")
    fg, mg = solve_eigen(KB, M)
    print(f"  prediction:  [{fg[0]:.2f}, {fg[1]:.2f}, {fg[2]:.2f}] Hz")
    print(f"  measured:    [{f_meas[0]:.3f}, {f_meas[1]:.3f}, {f_meas[2]:.3f}] Hz")
    for i in range(3):
        print(f"  mode {i+1}: {abs(fg[i]-f_meas[i])/f_meas[i]*100:.1f}% discrepancy")
    macs = []
    for i in range(3):
        if phi_exp[i] is not None:
            mc = mac(phi_exp[i], mg[:,i]); macs.append(mc)
    if all(p is not None for p in phi_exp):
        plot_modes(phi_exp, mg, fg, macs, 'fig_mode_shapes.png')
    return fg, mg


def m3(sigma, f_meas, phi_exp):
    """3-parameter BMU with frequency-only likelihood."""
    print("\n" + "="*60)
    print("  M3: Bayesian model updating")
    print("="*60)
    FM = np.array(f_meas)
    print(f"  target:  {FM}")
    print(f"  sigma:   {sigma}")
    def logl(th): return log_lik(th, M, FM, sigma)
    samp, stages = tmcmc(logl, [K_LO]*3, [K_HI]*3,
                          ns=NSAMPLES, beta=TMCMC_BETA, seed=SEED)
    km = np.mean(samp, axis=0); ks = np.std(samp, axis=0)
    fp, mp = solve_eigen(list(km), M)
    print(f"\n  posterior:")
    for i, l in enumerate(['bottom','middle','top']):
        print(f"    k{i+1} = {km[i]:,.0f} +/- {ks[i]:,.0f} N/m ({l})")
    print(f"  predicted: [{fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f}] Hz")
    if all(p is not None for p in phi_exp):
        print("  MAC (diagnostic):")
        for i in range(3):
            print(f"    mode {i+1}: {mac(phi_exp[i], mp[:,i]):.3f}")
    corr = np.corrcoef(samp.T)
    print(f"  correlations: {corr[0,1]:.3f}, {corr[0,2]:.3f}, {corr[1,2]:.3f}")
    plot_prior_posterior(samp, K_LO, K_HI, KB, 'fig_prior_posterior.png')
    plot_marginals(samp, 'fig_marginals.png')
    plot_pairwise(samp, 'fig_pairwise.png')
    plot_3d(samp, 'fig_3d.png')
    plot_freq_comp(FM, fp, 'fig_freq_comp.png')
    plot_convergence(stages, 'fig_convergence.png')
    return samp, km, ks, fp


def m4(samp, f_meas, s1_data, s2_data):
    """Validation against independent Session 2 free vibration tails."""
    print("\n" + "="*60)
    print("  M4: validation (S2 data, not used in calibration)")
    print("="*60)
    n = len(samp); fa = np.zeros((n, 3))
    for i in range(n): fa[i,:], _ = solve_eigen(list(samp[i]), M)
    fm = np.mean(fa, axis=0)
    flo = np.percentile(fa, 2.5, axis=0)
    fhi = np.percentile(fa, 97.5, axis=0)
    print(f"  95% CI: [{flo[0]:.2f},{fhi[0]:.2f}], [{flo[1]:.2f},{fhi[1]:.2f}], [{flo[2]:.2f},{fhi[2]:.2f}]")

    all_pass = True
    for d in s2_data:
        fv = [d['f1'], d['f2'], d['f3']]
        ok = all(flo[i] <= fv[i] <= fhi[i] for i in range(3))
        if not ok: all_pass = False
        fails = [f"f{i+1}" for i in range(3) if not (flo[i] <= fv[i] <= fhi[i])]
        note = ''
        if d.get('ratio', 0) > 0.25: note = ' (tail partially excited)'
        if d.get('tail', 99) < 5: note = ' (short tail)'
        tag = "pass" if ok else f"FAIL ({', '.join(fails)})"
        print(f"    {d['test']:<22} [{fv[0]:.2f}, {fv[1]:.2f}, {fv[2]:.2f}]  {tag}{note}")

    # validation figure
    FM = np.array(f_meas)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    mode_labels = ['Mode 1 (f1)', 'Mode 2 (f2)', 'Mode 3 (f3)']
    
    for mi, ax in enumerate(axes):
        # posterior CI as a shaded band
        ax.axhspan(flo[mi], fhi[mi], alpha=0.15, color='C3', label='95% CI')
        ax.axhline(fm[mi], color='C3', ls='-', lw=1, label=f'Posterior mean ({fm[mi]:.2f} Hz)')
        
        # S1 calibration points
        s1_vals = [d[f'f{mi+1}'] for d in s1_data]
        ax.plot(range(len(s1_vals)), s1_vals, 'o', color='C0', ms=5, zorder=5,
                label=f'S1 calibration (n={len(s1_vals)})')
        
        # S2 validation points
        s2_vals = [d[f'f{mi+1}'] for d in s2_data]
        x2 = range(len(s1_vals), len(s1_vals) + len(s2_vals))
        for j, (xp, fv) in enumerate(zip(x2, s2_vals)):
            inside = flo[mi] <= fv <= fhi[mi]
            ax.plot(xp, fv, 'x' if inside else 's', 
                    color='C2' if inside else 'C3',
                    ms=7, mew=1.5, zorder=5,
                    label='S2 validation' if j == 0 else None)
        
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(mode_labels[mi], fontsize=10)
        ax.set_xlabel('Test index')
        ax.legend(fontsize=7, loc='best')
        
        # vertical line separating S1 and S2
        ax.axvline(len(s1_vals) - 0.5, color='0.5', ls=':', lw=0.8)
        ax.text(len(s1_vals)/2, ax.get_ylim()[1], 'S1', ha='center', fontsize=8, color='0.5')
        ax.text(len(s1_vals) + len(s2_vals)/2, ax.get_ylim()[1], 'S2', ha='center', fontsize=8, color='0.5')
    
    plt.suptitle('Validation: S1 calibration vs S2 independent tests', fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/fig_validation.png', dpi=300)
    plt.close()
    
    print(f"  result: {'pass' if all_pass else 'partial'}")
    return all_pass


def m5(ks, sigma):
    """Geometric uncertainty propagation."""
    print("\n" + "="*60)
    print("  M5: uncertainty analysis")
    print("="*60)
    nom = {'L1': STOREY_HEIGHTS[0], 'L2': STOREY_HEIGHTS[1],
           'L3': STOREY_HEIGHTS[2], 'd': COLUMN_DEPTH,
           'b': COLUMN_WIDTH, 'E': E_MATERIAL, 'm': MASSES[0]}
    tols = {'L1': TOL_L, 'L2': TOL_L, 'L3': TOL_L,
            'd': TOL_D, 'b': TOL_B, 'E': 0, 'm': 0}
    sens, f_nom, _ = sensitivity(nom, tols)
    for p in ['d','L1','L2','L3','b']:
        print(f"    {p:<5} dk_max = {sens[p]['dk_pct']:.1f}%")
    f_mc, k_mc = propagate_geometric(nom, tols)
    k_mc_std = np.std(k_mc, axis=0)
    combined = np.sqrt(ks**2 + k_mc_std**2)
    print(f"  BMU std:      [{ks[0]:>7,.0f}, {ks[1]:>7,.0f}, {ks[2]:>7,.0f}]")
    print(f"  Geom std:     [{k_mc_std[0]:>7,.0f}, {k_mc_std[1]:>7,.0f}, {k_mc_std[2]:>7,.0f}]")
    print(f"  Combined:     [{combined[0]:>7,.0f}, {combined[1]:>7,.0f}, {combined[2]:>7,.0f}]")

    # tornado with only geometric params
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    order = ['L1','L2','L3','d','b']
    descs = {'d': 'Column depth', 'L1': 'Storey 1', 'L2': 'Storey 2',
             'L3': 'Storey 3', 'b': 'Column width'}
    for mode, ax in enumerate(axes):
        vals = [sens[p]['df'][mode] for p in order]
        colors = ['C3' if v > sigma[mode] else 'C0' if v > 0.01 else '0.7' for v in vals]
        y = np.arange(len(order))
        ax.barh(y, vals, color=colors, edgecolor='0.3', height=0.55)
        ax.axvline(sigma[mode], color='green', ls=':', lw=1.2,
                   label=f'sigma = {sigma[mode]:.3f}')
        ax.set_yticks(y)
        if mode == 0: ax.set_yticklabels([descs[p] for p in order])
        ax.set_xlabel('Delta f (Hz)')
        ax.set_title(f'Mode {mode+1}', fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
    plt.suptitle('Sensitivity to geometric tolerances (E and mass fixed)', fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/fig_sensitivity.png', dpi=300, bbox_inches='tight'); plt.close()
    plot_budget(ks, k_mc_std, 'fig_uncertainty_budget.png')


if __name__ == '__main__':
    print("="*60)
    print("  digital twin for structural dynamics")
    print("  Osman Mukuk, Strathclyde 2025-26")
    print("="*60)
    sigma, f_meas, phi_exp, s1_data, s2_data = m1()
    fg, mg = m2(phi_exp, f_meas)
    samp, kp, ks, fp = m3(sigma, f_meas, phi_exp)
    ok = m4(samp, f_meas, s1_data, s2_data)
    m5(ks, sigma)
    print("\n" + "="*60)
    print(f"  k = [{kp[0]:,.0f} +/- {ks[0]:,.0f}, "
          f"{kp[1]:,.0f} +/- {ks[1]:,.0f}, "
          f"{kp[2]:,.0f} +/- {ks[2]:,.0f}] N/m")
    print(f"  f = [{fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f}] Hz")
    print(f"  validation: {'pass' if ok else 'partial'}")
    print("="*60)
