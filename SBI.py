from __future__ import annotations

import numpy as np
import camb
from camb import model, initialpower  # type: ignore
from tqdm import tqdm
from typing import Tuple

# -----------------------------------------------------------------------------  
# Helper functions

def compute_physical_densities(Omega_m: float, H0: float, omega_b: float = 0.049
                               ) -> Tuple[float, float]:
    """Convert density parameters to CAMB's physical densities (Ωh²)."""
    h       = H0 / 100.0
    ombh2   = omega_b * h ** 2
    omch2   = (Omega_m - omega_b) * h ** 2
    return ombh2, omch2


def camb_power_spectrum(
    H0: float,
    Omega_m: float,
    n_s: float,
    k: np.ndarray,
) -> np.ndarray:
    """Return the linear matter power spectrum P(k) at z=0 for one cosmology."""
    ombh2, omch2 = compute_physical_densities(Omega_m, H0)

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=n_s)
    pars.set_matter_power(redshifts=[0.0], kmax=k.max())
    results = camb.get_results(pars)

    # CAMB returns P(k) on its own grid; ask for interpolation onto ours
    Pk = results.get_matter_power_spectrum(minkh=k.min(), maxkh=k.max(), npoints=len(k))[0]
    return Pk


def add_noise(Pk: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add multiplicative Gaussian noise with fractional standard deviation *sigma*."""
    noise = rng.normal(scale=sigma * Pk)
    return Pk + noise

# -----------------------------------------------------------------------------  
# Dataset generation   
def sample_parameters(n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Draw *n* parameter triplets from broad uniform priors."""
    H0       = rng.uniform(60.0, 80.0, size=n)
    Omega_m  = rng.uniform(0.25, 0.35, size=n)
    n_s      = rng.uniform(0.92, 1.00, size=n)
    return H0, Omega_m, n_s


def generate_dataset(
    n_obs: int = 1_200,
    num_k: int = 1_200,
    k_min: float = 1e-4,
    k_max: float = 1.0,
    noise_sigma: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Generate *n_obs* noisy power spectra and return a dict with parameters."""
    rng  = np.random.default_rng(seed)
    k    = np.logspace(np.log10(k_min), np.log10(k_max), num_k)

    Pk_obs    = np.empty((n_obs, num_k), dtype=np.float32)

    H0, Omega_m, n_s = sample_parameters(n_obs, rng)

    for i in tqdm(range(n_obs), desc="Simulating", unit="spec"):
        Pk = camb_power_spectrum(H0[i], Omega_m[i], n_s[i], k)
        Pk_obs[i] = add_noise(Pk, noise_sigma, rng)

    return {
        "Pk_obs": Pk_obs,
        "H0": H0,
        "Omega_m": Omega_m,
        "n_s": n_s,
     # optionally include the k-modes used
    }

# -----------------------------------------------------------------------------  
data = generate_dataset(n_obs=100, seed=42)
for key in data:
    print(key)

print(data)    

