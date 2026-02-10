"""
utilities.py

Contains helper functions for other module components.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Dict, List, Literal, Callable, Any


def fps_select_inducing(X: np.ndarray,
                        M: int,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Farthest Point Sampling over X (N,2) returning M indices.
    Simple greedy variant; good enough for inducing point initialization.
    """
    
    X = _check_2d_coords(X)
    N = X.shape[0]
    
    if M >= N:
        return np.arange(N, dtype=int)
    
    rng   = np.random.default_rng(seed)
    start = int(rng.integers(0, N))
    inds  = [start]
    d2    = cdist(X, X[[start]], metric="sqeuclidean")[:, 0]
    
    for _ in range(1, M):
        
        i  = int(np.argmax(d2))
        inds.append(i)
        d2 = np.minimum(d2, cdist(X, X[[i]], metric="sqeuclidean")[:, 0])
        
    return np.array(inds, dtype=int)


def _check_2d_coords(X: np.ndarray) -> np.ndarray:
    
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Coordinates must be shaped (N, 2).")
        
    return X


def _safe_cholesky(K: np.ndarray,
                   jitter: float = 1e-6) -> np.ndarray:
    
    J = float(jitter)
    for _ in range(8):
        
        try:
            return cholesky(K + J * np.eye(K.shape[0]),
                            lower=True,
                            check_finite=False)
        
        except Exception:
            J *= 10.0
            
    raise np.linalg.LinAlgError("Cholesky failed; matrix not PD even with heavy jitter.")
    
    
def _seg_project_signed_dist(X: np.ndarray,
                             p0: np.ndarray,
                             p1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Signed distance to segment [p0,p1] using left-hand normal for the sign.
    """
    
    v      = p1 - p0
    L2     = float(v @ v) + 1e-32
    t      = ((X - p0) @ v) / L2
    t      = np.clip(t, 0.0, 1.0)
    proj   = p0 + np.outer(t, v)
    diff   = X - proj
    dist   = np.sqrt(np.sum(diff**2, axis=1))
    n_left = np.array([-v[1], v[0]]) / (np.sqrt(L2) + 1e-32)
    sign   = np.sign(diff @ n_left)
    sign[dist < 1e-12] = 0.0
    
    return sign * dist, proj


def signed_distance_to_polyline(X: np.ndarray, 
                                polyline: np.ndarray,
                                chunk: int = 5000,
                                flip_sign: bool = False) -> np.ndarray:
    """
    Compute signed distance from points X to an oriented open polyline (M,2).
    """
    
    X = _check_2d_coords(X)
    P = np.asarray(polyline, float)
    
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] < 2:
        raise ValueError("polyline must be shaped (M,2) with M>=2")
        
    N    = X.shape[0]
    best = np.full(N, np.inf)
    
    for s in range(0, N, chunk):
        
        e      = min(s + chunk, N)
        Xi     = X[s:e]
        sd_min = np.full(e - s, np.inf)
        
        for k in range(P.shape[0] - 1):
            sd, _ = _seg_project_signed_dist(Xi,
                                             P[k],
                                             P[k+1])
            upd         = np.abs(sd) < np.abs(sd_min)
            sd_min[upd] = sd[upd]
            
        best[s:e] = sd_min
        
    if flip_sign:
        best = -best
        
    return best


def _grid_from_points(X: np.ndarray,
                      values: dict):
    """
    Require a regular grid: unique X and Y form a full mesh.
    Returns xs, ys (ascending) and dict of 2D arrays (ny, nx).
    """
    X  = np.asarray(X, float).reshape(-1, 2)
    xs = np.unique(X[:, 0])
    ys = np.unique(X[:, 1])
    
    nx, ny = len(xs), len(ys)
    if nx * ny != X.shape[0]:
        raise ValueError("Sample points are not on a complete regular grid. Please regrid before raster export.")
        
    # Build index from (x,y) -> linear index in X
    # Assume X is not necessarily sorted by grid order
    idx_map = {}
    for i in range(X.shape[0]):
        idx_map[(X[i,0], X[i,1])] = i
        
    out = {}
    for name, vec in values.items():
        
        a = np.asarray(vec, float).reshape(-1)
        if a.shape[0] != X.shape[0]:
            raise ValueError(f"Band '{name}' has length {a.shape[0]} but X has length {X.shape[0]}.")
            
        M = np.full((ny, nx), np.nan, dtype=float)
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                M[iy, ix] = a[idx_map[(xv, yv)]]
        out[name] = M
        
    return xs, ys, out


def export_rasters(filename: str,
                   X: np.ndarray,
                   bands: dict,
                   driver: str = "GTiff",
                   crs_wkt: str = None) -> str:
    """
    Export bands (dict of name->1D array over X) to a multi-band raster.
    Default driver is 'GTiff'. If rasterio is unavailable, try NetCDF.
    """
    xs, ys, grids = _grid_from_points(X, bands)
    
    nx, ny = len(xs), len(ys)
    
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 1.0
    
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    if driver == "GTiff":
        
        try:
            import rasterio
            from rasterio.transform import from_origin
            transform  = from_origin(x_min - dx/2.0, y_max + dy/2.0, dx, dy)
            band_names = list(grids.keys())
            
            with rasterio.open(
                filename, 'w',
                driver='GTiff', height=ny, width=nx, count=len(band_names),
                dtype='float32', transform=transform,
                crs=crs_wkt if crs_wkt is not None else None,
                
            ) as dst:
                
                for i, name in enumerate(band_names, 1):
                    
                    dst.write(grids[name][::-1, :].astype('float32'), i)  # flip rows for north-up
                    try:
                        dst.set_band_description(i, name)
                    except Exception:
                        pass
                    
            return filename
        
        except Exception:
            driver = "NetCDF"

    if driver == "NetCDF":
        
        try:
            from netCDF4 import Dataset
            
            nc = Dataset(filename, 'w')
            
            nc.createDimension('y', ny)
            nc.createDimension('x', nx)
            
            vx    = nc.createVariable('x', 'f8', ('x',))
            vx[:] = xs
            vy    = nc.createVariable('y', 'f8', ('y',))
            vy[:] = ys
            
            for name, arr in grids.items():
                
                v = nc.createVariable(name,
                                      'f4', 
                                      ('y','x'),
                                      zlib=True,
                                      complevel=1)
                
                v[:, :] = arr.astype('float32')
                
            nc.title = "Fusion GP posterior bands (Sparta)"
            nc.close()
            
            return filename
        
        except Exception as e:
            raise RuntimeError("Raster export failed (need 'rasterio' for GeoTIFF or 'netCDF4' for NetCDF).") from e