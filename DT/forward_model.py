"""
Forward model: eigenvalue problem for a 3-DOF shear frame.
Given storey stiffnesses [k1, k2, k3], predicts natural frequencies and mode shapes.
"""
import numpy as np
from scipy.linalg import eig

def base_k(E, b, d, L, n=4):
    """Analytical storey stiffness assuming fixed-fixed columns: k = n * 12EI / L^3"""
    return n * 12.0 * E * (b * d**3 / 12.0) / L**3

def solve_eigen(k_vec, masses):
    """Solve K*phi = w^2*M*phi. Returns frequencies in Hz and mode shapes (columns)."""
    m = np.asarray(masses, float)
    k = np.asarray(k_vec, float)
    n = len(m)
    M = np.diag(m)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k[i]
        if i + 1 < n:
            K[i, i] += k[i + 1]
            K[i, i + 1] = -k[i + 1]
            K[i + 1, i] = -k[i + 1]
    evals, evecs = eig(K, M)
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    for i in range(n):
        evecs[:, i] /= np.max(np.abs(evecs[:, i]))
    return np.sqrt(np.abs(evals)) / (2 * np.pi), evecs
