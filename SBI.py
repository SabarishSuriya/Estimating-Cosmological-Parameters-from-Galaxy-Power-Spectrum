"""
Generate a **dataset** of noisy matter power spectra \(P(k)\) for Simulation‑Based
Inference with BayesFlow.

This overwrites the previous single‑spectrum version and now supports an
arbitrary number of observations (default **1 200**).  Each observation is
created by

1. Randomly drawing cosmological parameters
   \(H_0,\;\Omega_m,\;n_s\) from simple uniform priors.
2. Computing the linear matter power spectrum with **CAMB**.
3. Adding Gaussian observational noise.

The script stores everything in a compressed **NumPy .npz** file:

* ``k``              → \(k\)-modes (shape ``(num_k,)``)
* ``Pk_theory``   → noiseless power spectra (shape ``(n_obs, num_k)``)
* ``Pk_obs``      → noisy observations     (shape ``(n_obs, num_k)``)
* ``H0``, ``Omega_m``, ``n_s`` → the true parameters for each observation

---
Installation
------------
```bash
pip install camb numpy tqdm
```

---
Quick start
-----------
Generate the default 1 200‑sample dataset:
```bash
python generate_power_spectra.py                # → data/pk_dataset.npz
```
Or customise:
```bash
python generate_power_spectra.py --n_obs 3000 --noise_sigma 0.03 \
                                 --out data/pk_custom.npz
```

You can then load the dataset in Python:
```python
import numpy as np
with np.load("data/pk_dataset.npz") as db:
    k          = db["k"]          # (1200,)
    Pk_obs     = db["Pk_obs"]     # (1200, 1200)
    true_theta = np.vstack([db["H0"], db["Omega_m"], db["n_s"]]).T  # (1200, 3)
```
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import camb
from camb import model, initialpower  # type: ignore
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

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
# -----------------------------------------------------------------------------

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
    """Generate *n_obs* noisy power spectra and return a dict ready to save."""
    rng  = np.random.default_rng(seed)
    k    = np.logspace(np.log10(k_min), np.log10(k_max), num_k)

    # Pre‑allocate output arrays
    Pk_theory = np.empty((n_obs, num_k), dtype=np.float32)
    Pk_obs    = np.empty_like(Pk_theory)

    H0, Omega_m, n_s = sample_parameters(n_obs, rng)

    for i in tqdm(range(n_obs), desc="Simulating", unit="spec"):
        Pk = camb_power_spectrum(H0[i], Omega_m[i], n_s[i], k)
        Pk_theory[i] = Pk
        Pk_obs[i]    = add_noise(Pk, noise_sigma, rng)

    return {
        "k": k,
        "Pk_theory": Pk_theory,
        "Pk_obs": Pk_obs,
        "H0": H0.astype(np.float32),
        "Omega_m": Omega_m.astype(np.float32),
        "n_s": n_s.astype(np.float32),
    }


def save_dataset(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **data)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser("Generate a dataset of noisy matter power spectra")
    p.add_argument("--n_obs", type=int, default=1_200, help="Number of observations")
    p.add_argument("--num_k", type=int, default=1_200, help="Number of k‑modes")
    p.add_argument("--k_min", type=float, default=1e-4)
    p.add_argument("--k_max", type=float, default=1.0)
    p.add_argument("--noise_sigma", type=float, default=0.05, help="Fractional noise σ_noise / P(k)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--out", type=str, default="data/pk_dataset.npz", help="Output .npz file")

    args = p.parse_args()

    data = generate_dataset(
        n_obs=args.n_obs,
        num_k=args.num_k,
        k_min=args.k_min,
        k_max=args.k_max,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )

    save_dataset(data, args.out)
    print(f"Saved {args.n_obs} spectra → {args.out}")


if __name__ == "__main__":
    main()
