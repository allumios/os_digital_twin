# Digital twin for structural dynamics

3-storey aluminium shear frame, EN AW-6082-T6 (E = 70 GPa).
Updates storey stiffness [k1, k2, k3] using TMCMC with frequency-only likelihood.
Frequencies extracted from personalised free vibration tails after each test.
Measurement uncertainty estimated empirically from repeated independent measurements.

Author: Osman Mukuk
Supervisor: Dr Marco De Angelis
University of Strathclyde, 2025-26

## Run
    pip install -r requirements.txt
    python run_digital_twin.py

## Pipeline
  M1: system identification
  M2: forward model
  M3: Bayesian model updating (TMCMC, frequency-only)
  M4: validation against Session 2 data
  M5: geometric uncertainty propagation

## References
  Ching and Chen (2007) ASCE J. Eng. Mech. 133(7), 816-832
  Lye et al. (2021) MSSP 159, 107760
  Bonney et al. (2022) Data-Centric Engineering 3, e1
  Chopra (2012) Dynamics of Structures, 4th edn
