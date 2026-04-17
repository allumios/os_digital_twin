"""
Generates the detailed amplitude and PSD plots per test per floor.
Run separately from the main pipeline because these take a few minutes.
Usage: python generate_plots.py
"""
import numpy as np, os, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
from signal_processing import load_sheet, fft_amp, psd, top_peaks

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'savefig.dpi': 300, 'lines.linewidth': 0.6, 'axes.grid': True,
    'grid.alpha': 0.25,
})
fs = SAMPLING_RATE

for sl, sf, ss in [('S1', SESSION_1_FILE, S1_SHEETS),
                    ('S2', SESSION_2_FILE, S2_SHEETS)]:
    for key, sn in ss.items():
        try:
            t, c1, c2, c3 = load_sheet(sf, sn)
            dur = len(c1) / fs
            for spec in ['amplitude', 'psd']:
                folder = f'output/{spec}/{sl}_{key}'
                os.makedirs(folder, exist_ok=True)
                for fi, (ch, fl) in enumerate(zip([c1,c2,c3],
                                                   ['Floor 1','Floor 2','Floor 3'])):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if spec == 'amplitude':
                        f, y = fft_amp(ch, fs)
                        ax.plot(f, y, color='0.2', lw=0.6)
                        ax.set_ylim(bottom=0)
                        ax.set_ylabel('Amplitude (a.u.)')
                    else:
                        f, y = psd(ch, fs)
                        ax.semilogy(f, y, color='0.2', lw=0.6)
                        ax.set_ylabel('PSD')
                    ax.set_xlim([3, 38])
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_title(f'{sl} {sn.strip()}, {fl} ({dur:.0f} s)', fontsize=10)
                    pf, _ = top_peaks(f, y if spec == 'amplitude' else np.sqrt(y), n=3)
                    for fp in pf:
                        if not np.isnan(fp):
                            idx = np.argmin(np.abs(f - fp))
                            ax.plot(fp, y[idx], 'o', color='C3', ms=4, zorder=5)
                            ax.annotate(f'{fp:.2f}', (fp, y[idx]), xytext=(5, -10),
                                        textcoords='offset points', fontsize=8, color='C3')
                    plt.tight_layout()
                    plt.savefig(f'{folder}/floor{fi+1}.png', dpi=300)
                    plt.close()
            print(f"  {sl} {sn.strip()}")
        except Exception as e:
            print(f"  {sl} {key}: skip ({e})")

print("done")
