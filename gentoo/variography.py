"""
variography.py

For performing variography-based hyperparameter initialization.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Literal, Callable, Any
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# Gentoo specific imports
from utilities import (_check_2d_coords)
from datasets import (Hyperparams,
                      GNSSData,
                      LiDARData,
                      InSARData,
                      VariogramResult)




def empirical_variogram(X: np.ndarray,
                        z: np.ndarray,
                        n_bins: int = 20,
                        max_range: Optional[float] = None) -> VariogramResult:
    
    """
    Simple isotropic semivariogram + exponential fit; robust to sparse GNSS datasets.
    """
    X = _check_2d_coords(X)
    z = np.asarray(z, float).reshape(-1)
    N = X.shape[0]
    
    if N < 5:
        return VariogramResult(
            h=np.linspace(0, 1, n_bins),
            gamma=np.zeros(n_bins),
            nugget=0.0,
            sill=float(np.var(z)),
            range_=float(np.ptp(X, axis=0).mean() / 3.0)
        )
    
    d  = cdist(X, X)
    iu = np.triu_indices(N, 1)
    h  = d[iu]
    g  = 0.5 * (z[:, None] - z[None, :])**2
    g  = g[iu]
    
    if max_range is None:
        max_range = float(np.percentile(h, 95))
        
    bins  = np.linspace(0.0, max_range, n_bins + 1)
    bc    = 0.5 * (bins[:-1] + bins[1:])
    gamma = np.zeros(n_bins)
    
    for b in range(n_bins):
        m        = (h >= bins[b]) & (h < bins[b+1])
        gamma[b] = np.mean(g[m]) if np.any(m) else np.nan
        
    if np.any(np.isnan(gamma)):
        mask         = ~np.isnan(gamma)
        gamma[~mask] = np.interp(bc[~mask], bc[mask], gamma[mask])
        
    # Fit gamma(h) = nug + (sill-nug) * (1 - exp(-h/r))
    def model(p):
        nug, sil, r = p
        return nug + (sil - nug) * (1.0 - np.exp(-bc / (r + 1e-12)))
    
    def loss(p):
        return float(np.mean((gamma - model(p))**2))
     
    nug0   = max(1e-8, 0.05 * np.var(z))
    sil0   = max(np.var(z), nug0 * 1.2)
    r0     = max_range / 3.0
    bounds = [(0.0, 10*np.var(z)+1e-6), (1e-10, 100*np.var(z)+1e-6), (1e-6, 10*max_range)]
    
    res    = minimize(loss,
                      x0=np.array([nug0, sil0, r0]),
                      method="L-BFGS-B",
                      bounds=bounds)
    
    nug, sil, r = res.x
    
    return VariogramResult(h=bc, gamma=gamma, nugget=float(nug), sill=float(sil), range_=float(r))


def hyperparams_from_variography(gnss: Optional[GNSSData],
                                 lidar: Optional[LiDARData] = None,
                                 default_ell: float = 10.0,
                                 shared_ell: bool = False,
                                 noise_mode: str = "fixed",           # {"fixed", "global_nugget"}
                                 log_noise0: Optional[float] = None) -> Hyperparams:  # if noise_mode == "global_nugget", stores log(sigma0)
    
    comps = []
    for data in [gnss, lidar]:
        if data is not None and data.X.size > 0:
            comps.append((data.X, data.U[:, 0], 0))
            comps.append((data.X, data.U[:, 1], 1))
            comps.append((data.X, data.U[:, 2], 2))
            
    if not comps:
        return Hyperparams(var_xyz=np.ones(3),
                           ell_xyz=np.full(3, default_ell),
                           shared_ell=shared_ell,
                           noise_mode=noise_mode,
                           log_noise0=log_noise0)
    
    var_out, ell_out = np.zeros(3), np.zeros(3)
    for j in range(3):
        
        Xcat, zcat = [], []
        for (X, z, idx) in comps:
            
            if idx == j:
                Xcat.append(X); zcat.append(z)
                
        Xc = np.vstack(Xcat); zc = np.concatenate(zcat)
        vg = empirical_variogram(Xc, zc, n_bins=20)
        
        var_out[j] = max(1e-8, vg.sill)
        ell_out[j] = max(1e-6, vg.range_)
        
    if shared_ell:
        ell_out[:] = np.mean(ell_out)
        
    return Hyperparams(var_xyz=var_out, ell_xyz=ell_out, shared_ell=shared_ell, noise_mode=noise_mode, log_noise0=log_noise0)

