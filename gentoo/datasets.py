"""
datasets.py
"""
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal, Callable, Any
from dataclasses import dataclass, field

from utilities import (_check_2d_coords)

# ============================== Data Containers ============================== #

@dataclass
class GNSSData:
    X: np.ndarray       # (Ng,2) in km
    U: np.ndarray       # (Ng,3) [ux,uy,uz]
    var: np.ndarray     # (Ng,3) variances [sx^2, sy^2, sz^2]

    def __post_init__(self):
        
        self.X   = _check_2d_coords(self.X)
        self.U   = np.asarray(self.U, float)
        self.var = np.asarray(self.var, float)
        
        if self.U.shape != (self.X.shape[0], 3) or self.var.shape != (self.X.shape[0], 3):
            raise ValueError("U and var must be shaped (N,3).")


@dataclass
class LiDARData:
    X: np.ndarray
    U: np.ndarray
    var: np.ndarray

    def __post_init__(self):
        
        self.X   = _check_2d_coords(self.X)
        self.U   = np.asarray(self.U, float)
        self.var = np.asarray(self.var, float)
        
        if self.U.shape != (self.X.shape[0], 3) or self.var.shape != (self.X.shape[0], 3):
            raise ValueError("U and var must be shaped (N,3).")


@dataclass
class InSARData:
    X: np.ndarray        # (Ni,2)
    los: np.ndarray      # (Ni,)
    var: np.ndarray      # (Ni,)
    look_vec: np.ndarray # (Ni,3) [lx,ly,lz] per-pixel

    def __post_init__(self):
        
        self.X        = _check_2d_coords(self.X)
        self.los      = np.asarray(self.los, float).reshape(-1)
        self.var      = np.asarray(self.var, float).reshape(-1)
        self.look_vec = np.asarray(self.look_vec, float)
        
        if self.X.shape[0] != self.los.shape[0] or self.var.shape[0] != self.X.shape[0]:
            raise ValueError("los/var must be length N.")
            
        if self.look_vec.shape != (self.X.shape[0], 3):
            raise ValueError("look_vec must be shaped (N,3).")


@dataclass
class FusedDataset:
    """
    Unified scalar-observation view:
      y[i] = H[i,:] @ f(X[i]) + eps_i,   H[i] in R^3, f = [ux,uy,uz].
    'tags' helps track provenance (gnss_x, gnss_y, gnss_z, insar).
    """
    X: np.ndarray
    H: np.ndarray
    y: np.ndarray
    var: np.ndarray
    Xi: np.ndarray
    tags: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_sources(gnss: Optional[GNSSData],
                     insar_list: List[InSARData],
                     lidar: Optional[LiDARData] = None) -> "FusedDataset":
        
        Xs, Hs, ys, vs, tags = [], [], [], [], []
        meta: Dict[str, Any] = {}

        # GNSS 3 components -> 3 scalar obs
        if gnss is not None and gnss.X.size > 0:
            
            Ng = gnss.X.shape[0]
            for j, tag in enumerate(["gnss_x", "gnss_y", "gnss_z"]):
                
                Xs.append(gnss.X)
                e = np.zeros((Ng, 3)); e[:, j] = 1.0
                Hs.append(e)
                ys.append(gnss.U[:, j])
                vs.append(gnss.var[:, j])
                tags.extend([tag] * Ng)
                
            meta["idx_gnss"] = slice(0, 3 * Ng)

        # LiDAR (optional)
        if lidar is not None and lidar.X.size > 0:
            
            Nl = lidar.X.shape[0]
            for j, tag in enumerate(["lidar_x", "lidar_y", "lidar_z"]):
                
                Xs.append(lidar.X)
                e = np.zeros((Nl, 3)); e[:, j] = 1.0
                Hs.append(e)
                ys.append(lidar.U[:, j])
                vs.append(lidar.var[:, j])
                tags.extend([tag] * Nl)

        # InSAR tracks (ascending/descending stacked)
        idx_start = len(tags)
        for insar in insar_list:
            
            Xs.append(insar.X)
            Hs.append(insar.look_vec)
            ys.append(insar.los)
            vs.append(insar.var)
            tags.extend(["insar"] * insar.X.shape[0])
            
        if insar_list:
            meta["idx_insar"] = slice(idx_start, len(tags))
        
        xmin, xmax = -np.inf, np.inf
        ymin, ymax = -np.inf, np.inf
        for x in Xs:
            
            xx = x[:,0]
            yy = x[:,1]
            
            if xx.min() > xmin:
                xmin = xx.min()
            if xx.max() < xmax:
                xmax = xx.max()
            if yy.min() > ymin:
                ymin = yy.min()
            if yy.max() < ymax:
                ymax = yy.max()

        X   = np.vstack(Xs) if Xs else np.zeros((0, 2))
        Xi  = X[np.argwhere((X[:,0] > xmin) \
                            & (X[:,0] < xmax) \
                            & (X[:,1] > ymin) \
                            & (X[:,1] < ymax))].squeeze()
        H   = np.vstack(Hs) if Hs else np.zeros((0, 3))
        y   = np.concatenate(ys) if ys else np.zeros((0,))
        var = np.concatenate(vs) if vs else np.zeros((0,))
        
        return FusedDataset(X=X, H=H, y=y, var=var, Xi=Xi, tags=tags, meta=meta)
    

@dataclass
class Hyperparams:
    var_xyz: np.ndarray         # (3,)
    ell_xyz: np.ndarray         # (3,)
    shared_ell: bool = False
    noise_mode: str = "fixed"           # {"fixed", "global_nugget"}
    log_noise0: Optional[float] = None  # if noise_mode == "global_nugget", stores log(sigma0)
        
    def as_vector(self,
                  log_space: bool = True) -> np.ndarray:
        
        v = np.concatenate([self.var_xyz, (self.ell_xyz[:1] if self.shared_ell else self.ell_xyz)])
        
        return np.log(np.maximum(v, 1e-12)) if log_space else v
    
    def effective_sigma2(self, data) -> np.ndarray:
        """
        Return per-observation noise variances Sigma_ii (in natural space),
        combining reported variances with any learned nugget in this Hyperparams.
        """
        sigma2 = np.asarray(data.var, float).reshape(-1)    # reported per-point variances
        if self.noise_mode == "global_nugget" and self.log_noise0 is not None:
            sigma2 = sigma2 + float(np.exp(self.log_noise0))**2
            
        return sigma2

    @staticmethod
    def from_vector(v: np.ndarray,
                    shared_ell: bool,
                    log_space: bool = True) -> "Hyperparams":
        
        v = np.asarray(v, float)
        if log_space: 
            v = np.exp(v)
            
        var = v[:3]
        if shared_ell:
            ell = np.repeat(v[3], 3)
        else:
            ell = v[3:6]
            
        return Hyperparams(var_xyz=var, ell_xyz=ell, shared_ell=shared_ell)


@dataclass
class VariogramResult:
    h: np.ndarray
    gamma: np.ndarray
    nugget: float
    sill: float
    range_: float
        
        
@dataclass
class FitHistoryEntry:
    iter: int
    objective: float
    hyperparams_vec: np.ndarray
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelState:
    cache: Dict[str, Any] = field(default_factory=dict)