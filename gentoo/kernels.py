"""
kernels.py

Kernels currently implemented for Gentoo:
 - RBF
 - Fault-aware (can accomodate any standard isotropic kernel)
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Literal, Callable, Any
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist

# Gentoo specific
from utilities import (_check_2d_coords,
                       signed_distance_to_polyline)

def rbf_kernel(X: np.ndarray,
               Z: np.ndarray,
               variance: float,
               lengthscale: float) -> np.ndarray:
    
    """
    RBF kernel: k(x,z) = variance * exp(-0.5 * ||x-z||^2 / ell^2).
    """
    
    X  = _check_2d_coords(X)
    Z  = _check_2d_coords(Z)
    d2 = cdist(X, Z, metric="sqeuclidean")
    
    return float(variance) * np.exp(-0.5 * d2 / (float(lengthscale)**2 + 1e-32))


@dataclass
class FaultGating:
    """
    Input-dependent gate for fault/discontinuity modeling.
    Computes signed distances to an oriented polyline and returns weights:
        w_plus(x) in [0,1], w_minus(x)=1-w_plus(x).
    mode='soft' : logistic transition with width (km);
    mode='hard' : binary step (exact barrier).
    """
    
    polyline: np.ndarray                # (M,2) rupture points (in same units as X)
    width: float = 0.05
    mode: Literal["soft", "hard"] = "soft"
    flip_sign: bool = False

    def weights(self,
                X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        sd = signed_distance_to_polyline(X,
                                         self.polyline,
                                         flip_sign=self.flip_sign)
        
        if self.mode == "hard":
            w_plus = (sd > 0).astype(float)
            
        else:
            z = np.clip(sd / (self.width + 1e-12), -50, 50)
            w_plus = 1.0 / (1.0 + np.exp(-z))
            
        w_minus = 1.0 - w_plus
        
        return w_plus, w_minus
    

def fault_gated_rbf(X: np.ndarray,
                    Z: np.ndarray,
                    var_plus: float,
                    ell_plus: float,
                    var_minus: Optional[float],
                    ell_minus: Optional[float],
                    gating: FaultGating) -> np.ndarray:
    """
    PSD fault-gated kernel:
      K = (w+ w+^T) ⊙ K_plus  +  (w- w-^T) ⊙ K_minus
    """
    
    wX_plus, wX_minus = gating.weights(X)
    wZ_plus, wZ_minus = gating.weights(Z)
    
    if var_minus is None:
        var_minus = var_plus
        
    if ell_minus is None:
        ell_minus = ell_plus
        
    Kp = rbf_kernel(X, Z, var_plus, ell_plus)
    Km = rbf_kernel(X, Z, var_minus, ell_minus)
    
    return np.outer(wX_plus, wZ_plus) * Kp + np.outer(wX_minus, wZ_minus) * Km


def fault_barrier_rbf(X: np.ndarray,
                      Z: np.ndarray,
                      var: float,
                      ell: float,
                      gating: FaultGating) -> np.ndarray:
    
    """
    Hard barrier: zero cross-fault correlation; same kernel both sides.
    """
    return fault_gated_rbf(X, Z, var, ell, var, ell, gating)