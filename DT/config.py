"""
Configuration for the digital twin.
Edit this file to match your structure. Other scripts read from here.
"""

# structure (EN AW-6082-T6 aluminium, Cosmos cert 3.1)
MASSES = [5.36, 5.36, 5.36]               # floor masses, kg (weighed)
E_MATERIAL = 70e9                          # Pa, from material certificate
COLUMN_DEPTH = 3.45e-3                     # m, caliper measurement
COLUMN_WIDTH = 25.4e-3                     # m, 1" flat bar from mill cert
N_COLUMNS = 4
STOREY_HEIGHTS = [0.1845, 0.1525, 0.1520] # m, ruler measurement

# geometric tolerances for uncertainty propagation
# E and mass are treated as fixed, following Bonney et al. (2022)
TOL_L = 0.5e-3    # m, ruler
TOL_D = 0.01e-3   # m, digital caliper
TOL_B = 0.1e-3    # m, mill cert

# data files
SESSION_1_FILE = 'data/2026_01_12.xlsx'
SESSION_2_FILE = 'data/Input_signal_50.xlsx'
SAMPLING_RATE = 2048.0

S1_SHEETS = {
    'free_vib': 'Free Vibration',
    'impact':   'Impact Test',
    'mode1':    '1st Frequency',
    'mode2':    '2nd Frequency',
    'mode3':    '3rd Frequency',
    'eq':       'EQ',
}
S2_SHEETS = {
    'free_vib': 'Free_Vibration',
    'impact':   'Impact Test',
    'mode1':    '1st Frequency',
    'mode2':    '2nd Frequency',
    'mode3':    '3rd Frequency',
    'eq':       'EQ',
}

# TMCMC settings
K_LO = 20000.0
K_HI = 150000.0
NSAMPLES = 1000
TMCMC_BETA = 0.2
SEED = 42
