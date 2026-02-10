"""
fusion_models.py
"""
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Literal, Callable, Any
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from numpy.linalg import norm

# Gentoo specific imports
from datasets import (FusedDataset,
                      Hyperparams,
                      FitHistoryEntry,
                      ModelState)
from kernels import (FaultGating,
                     rbf_kernel,
                     fault_gated_rbf)
from utilities import (fps_select_inducing,
                       _safe_cholesky,
                       cho_solve,
                       _check_2d_coords,
                       solve_triangular)

        
        
class BaseFusedModel:
    
    def __init__(self,
                 data: FusedDataset,
                 jitter: float = 1e-6,
                 random_state: Optional[int] = None):
        
        self.data         = data
        self.jitter       = float(jitter)
        self.random_state = random_state
        
        self.history: List[FitHistoryEntry] = []
        self.state = ModelState()

        
    def record(self,
               i: int,
               obj: float,
               hp_vec: np.ndarray,
               notes: Optional[Dict[str, Any]] = None):
        
        self.history.append(FitHistoryEntry(iter=i,
                                            objective=float(obj),
                                            hyperparams_vec=np.array(hp_vec),
                                            notes=notes or {}))

        
    def get_history(self) -> List[FitHistoryEntry]:
        return self.history

    
    def get_state_snapshot(self) -> Dict[str, Any]:
        return {"state": self.state.cache, "history": [e.__dict__ for e in self.history]}
    
    
    def plot_optimization_paths(self,
                                label: str = 'objective',
                                ax=None,
                                save=False,
                                save_name='optimization_path.png',
                                show=False):
        
        it  = [h.iter for h in self.history]
        obj = [h.objective for h in self.history]
        
        if ax is None:
            fig, ax = plt.subplots()
            
        ax.plot(
            it,
            obj,
            marker='o',
            linewidth=1.0
        )
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.set_title("Optimization Path")
        ax.grid(True, alpha=0.4)
        
        if show:
            plt.show()
        
        if save:
            fig.savefig(save_name, dpi=100)
            
        return fig, ax
    

    def plot_hyperparam_traces(self,
                               ax=None,
                               save=False,
                               save_name='hyperparam_history.png',
                               show=False):
        
        amps, ells = [], []
        for h in self.history:
            
            v = h.hyperparams_vec
            if len(v) in (4, 6, 7):
                hp = Hyperparams.from_vector(v,
                                             shared_ell=(len(v)==4),
                                             log_space=True)
                amps.append(hp.var_xyz)
                ells.append(hp.ell_xyz)
            else:
                continue
                
        if not amps:
            return None
        
        amps = np.array(amps)
        ells = np.array(ells)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
        axs[0].plot(
            amps[:, 0],
            label="var_x"
        )
        axs[0].plot(
            amps[:, 1],
            label="var_y"
        )
        axs[0].plot(
            amps[:, 2],
            label="var_z"
        )
        
        axs[0].set_title("Amplitude (variance)")
        axs[0].set_xlabel("Iter")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        axs[1].plot(
            ells[:, 0],
            label="ell_x"
        )
        axs[1].plot(
            ells[:, 1],
            label="ell_y"
        )
        axs[1].plot(
            ells[:, 2],
            label="ell_z"
        )
        
        axs[1].set_title("Lengthscale")
        axs[1].set_xlabel("Iteration")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        
        if save:
            fig.savefig(save_name, dpi=100) 
            
        if show:
            plt.show()
            
        return fig, (axs[0], axs[1])


# =================================== SVGP / VFE =================================== #


class SVGPModel(BaseFusedModel):
    """
    Variational inducing-point multi-output GP with independent latents (ux,uy,uz).
    ELBO: Titsias VFE; heteroscedastic Gaussian noise.
    """

    def __init__(self,
                 data: FusedDataset,
                 hp: Hyperparams,
                 M: int = 800,
                 inducing_selector: Literal["fps", "random"] = "fps",
                 shared_Z: bool = True,
                 fault: Optional[FaultGating] = None,
                 jitter: float = 1e-6, random_state: Optional[int] = None):
        
        super().__init__(data,
                         jitter=jitter,
                         random_state=random_state)
        
        self.hp       = hp
        self.M        = int(M)
        self.shared_Z = bool(shared_Z)
        self.selector = inducing_selector
        self.fault    = fault
        
        self.Zx: Optional[np.ndarray]   = None
        self.Zy: Optional[np.ndarray]   = None
        self.Zz: Optional[np.ndarray]   = None
        self._L_B: Optional[np.ndarray] = None
        self._w: Optional[np.ndarray]   = None

            
    def _choose_Z(self):
        
        X = self.data.Xi
        if self.selector == "fps":
            idx = fps_select_inducing(X,
                                      self.M,
                                      self.random_state)
            
        else:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0],
                             size=min(self.M, X.shape[0]),
                             replace=False)
        Z = X[idx, :]
        
        if self.shared_Z:
            self.Zx = self.Zy = self.Zz = Z.copy()
            
        else:
            self.Zx = Z.copy()
            self.Zy = X[fps_select_inducing(X, self.M, (None if self.random_state is None else self.random_state+1))]
            self.Zz = X[fps_select_inducing(X, self.M, (None if self.random_state is None else self.random_state+2))]
            
        self.state.cache["Z"] = {"Zx": self.Zx,
                                 "Zy": self.Zy,
                                 "Zz": self.Zz}

        
    def _k_latent(self,
                  which: int,
                  X: np.ndarray,
                  Z: np.ndarray,
                  hp: Hyperparams) -> np.ndarray:
        
        var = hp.var_xyz[which]
        ell = hp.ell_xyz[which]
        
        if self.fault is None:
            return rbf_kernel(X, Z, var, ell)
        
        else:
            return fault_gated_rbf(X, Z, var, ell, None, None, self.fault)

        
    def _build_Kuu_blocks(self,
                          hp: Hyperparams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        Kux = self._k_latent(0, self.Zx, self.Zx, hp)
        Kuy = self._k_latent(1, self.Zy, self.Zy, hp)
        Kuz = self._k_latent(2, self.Zz, self.Zz, hp)
        
        Lx = _safe_cholesky(Kux, self.jitter)
        Ly = _safe_cholesky(Kuy, self.jitter)
        Lz = _safe_cholesky(Kuz, self.jitter)
        
        self.state.cache["Kuu_chol"] = {"Lx": Lx, "Ly": Ly, "Lz": Lz}
        
        return Kux, Kuy, Kuz

    
    def _build_Kfu(self,
                   hp: Hyperparams) -> Tuple[np.ndarray, List[slice]]:
        
        H   = self.data.H
        Kx  = self._k_latent(0, self.data.X, self.Zx, hp) * H[:, [0]]
        Ky  = self._k_latent(1, self.data.X, self.Zy, hp) * H[:, [1]]
        Kz  = self._k_latent(2, self.data.X, self.Zz, hp) * H[:, [2]]
        Kfu = np.hstack([Kx, Ky, Kz])
        M   = self.M
        
        return Kfu, [slice(0, M), slice(M, 2*M), slice(2*M, 3*M)]

    
    def _elbo(self,
              theta_vec: np.ndarray) -> float:
        
        hp = Hyperparams.from_vector(theta_vec,
                                     shared_ell=self.hp.shared_ell,
                                     log_space=True)

        core_len = getattr(self, "_opt_core_len", len(self.hp.as_vector(log_space=True)))
        hp_core  = Hyperparams.from_vector(theta_vec[:core_len],
                                           shared_ell=self.hp.shared_ell,
                                           log_space=True)
        
        if getattr(self, "_learn_noise", False):
            hp_core.noise_mode = "global_nugget"
            hp_core.log_noise0 = theta_vec[core_len]
        else:
            # carry through any pre-set nugget
            hp_core.noise_mode = self.hp.noise_mode
            hp_core.log_noise0 = self.hp.log_noise0   
              
        hp     = hp_core
        sigma2 = hp_core.effective_sigma2(self.data)
        
        if self.Zx is None:
            self._choose_Z()

        Kux, Kuy, Kuz = self._build_Kuu_blocks(hp)
        
        Lx = self.state.cache["Kuu_chol"]["Lx"]
        Ly = self.state.cache["Kuu_chol"]["Ly"]
        Lz = self.state.cache["Kuu_chol"]["Lz"]
        
        Kux_inv = cho_solve((Lx, True), np.eye(Lx.shape[0]))
        Kuy_inv = cho_solve((Ly, True), np.eye(Ly.shape[0]))
        Kuz_inv = cho_solve((Lz, True), np.eye(Lz.shape[0]))
        
        Kuu = np.block([[Kux, np.zeros_like(Kux), np.zeros_like(Kux)],
                        [np.zeros_like(Kuy), Kuy, np.zeros_like(Kuy)],
                        [np.zeros_like(Kuz), np.zeros_like(Kuz), Kuz]])
        
        Kuu_inv = np.block([[Kux_inv, np.zeros_like(Kux_inv), np.zeros_like(Kux_inv)],
                            [np.zeros_like(Kuy_inv), Kuy_inv, np.zeros_like(Kuy_inv)],
                            [np.zeros_like(Kuz_inv), np.zeros_like(Kuz_inv), Kuz_inv]])

        Kfu, blocks = self._build_Kfu(hp)
        Kuf         = Kfu.T
        sigma2      = self.data.var.reshape(-1)
        Ainv        = 1.0 / (sigma2 + 1e-32)
        B           = Kuu + Kuf @ (Ainv[:, None] * Kfu)
        L_B         = _safe_cholesky(B,
                                     self.jitter)
        v           = Kuf @ (Ainv * self.data.y)
        w           = cho_solve((L_B, True),
                                v)

        # ELBO terms
        logdet_Kuu = 2.0 * (np.sum(np.log(np.diag(Lx))) \
                         + np.sum(np.log(np.diag(Ly))) \
                         + np.sum(np.log(np.diag(Lz))))
        
        logdet_B     = 2.0 * np.sum(np.log(np.diag(L_B)))
        logdet_Sigma = np.sum(np.log(sigma2 + 1e-32))
        quad         = float(self.data.y @ (Ainv * self.data.y) - v @ w)

        # Trace correction
        diag_Kff = (self.data.H[:, 0]**2) * hp.var_xyz[0] \
                 + (self.data.H[:, 1]**2) * hp.var_xyz[1] \
                 + (self.data.H[:, 2]**2) * hp.var_xyz[2]
        
        diag_Qff = np.zeros_like(sigma2)
        
        for ell, (Zl, var_l, ell_l, blk) in enumerate(
            [(self.Zx, hp.var_xyz[0], hp.ell_xyz[0], blocks[0]),
             (self.Zy, hp.var_xyz[1], hp.ell_xyz[1], blocks[1]),
             (self.Zz, hp.var_xyz[2], hp.ell_xyz[2], blocks[2])]
        ):
            
            Kl        = self._k_latent(ell,
                                       self.data.X,
                                       Zl,
                                       hp)  # (N,M)
            
            Kuu_inv_l = Kuu_inv[blk, blk]
            Arow      = Kl @ Kuu_inv_l
            diag_Qff += (self.data.H[:, ell]**2) * np.sum(Arow * Kl, axis=1)

        N    = self.data.X.shape[0]
        elbo = -0.5 * ((logdet_B - logdet_Kuu + logdet_Sigma) \
                    + quad \
                    + N * np.log(2.0 * np.pi)) \
                    - 0.5 * np.sum(Ainv * (diag_Kff - diag_Qff))

        self._L_B = L_B
        self._w   = w
        self.hp   = hp
        
        self.state.cache["B_chol"] = L_B
        self.state.cache["w"]      = w
        
        return float(elbo)
    
    
    def _build_svgp_hp_bounds(self,
                              theta0: np.ndarray,
                              hp_bounds: dict):
        """
        Map hp_bounds (natural space) onto the optimizer vector (log-space).
        Assumes theta0 packs: [log var_x, log var_y, log var_z, log ell_(x,y,z or 1), (optional) log noise]
        depending on self.hp.shared_ell.
        """
        
        eps = 1e-16

        # convenience to expand a scalar pair or a list of pairs to length n
        def _expand(bounds_item,
                    n,
                    name):
            
            if bounds_item is None:
                return [(None, None)] * n
            
            # scalar pair (lo, hi) -> replicate
            if isinstance(bounds_item, (tuple, list)) and len(bounds_item) == 2 and np.isscalar(bounds_item[0]):
                
                lo, hi = float(bounds_item[0]), float(bounds_item[1])
                return [(lo, hi)] * n
            
            # list of pairs -> verify length
            if isinstance(bounds_item, (list, tuple)) and len(bounds_item) == n and \
               all(isinstance(t, (list, tuple)) and len(t) == 2 for t in bounds_item):
                
                return [(float(t[0]), float(t[1])) for t in bounds_item]
            
            raise ValueError(f"hp_bounds['{name}'] must be (lo,hi) or list of {n} (lo,hi) pairs.")

        # unpack model layout
        p       = 3  # x,y,z latents
        shared = getattr(self.hp, "shared_ell", False)

        # natural-space bounds
        var_b   = hp_bounds.get("var", None)
        ell_b   = hp_bounds.get("ell", None)
        noise_b = hp_bounds.get("noise", None)  # only learning a scalar likelihood noise

        var_pairs = _expand(var_b, p, "var")
        ell_pairs = _expand(ell_b, 1 if shared else p, "ell")

        # assemble natural-space bounds in the order of theta0
        nat_bounds = []
        # variances (x,y,z)
        nat_bounds.extend(var_pairs)
        # lengthscales (1 if shared else 3)
        nat_bounds.extend(ell_pairs)
        
        # optional scalar noise at the end
        if noise_b is not None:
            if not (isinstance(noise_b, (tuple, list)) and len(noise_b) == 2):
                raise ValueError("hp_bounds['noise'] must be a (lo, hi) pair.")
                
            nat_bounds.append((float(noise_b[0]), float(noise_b[1])))

        # length match
        if len(nat_bounds) != len(theta0):
            raise ValueError(f"Bounds length {len(nat_bounds)} does not match parameter vector length {len(theta0)}. "
                             "Check shared_ell and whether noise is included.")

        # convert natural-space (positive) bounds to log-space
        log_bounds = []
        for (lo, hi), t0 in zip(nat_bounds, theta0):
            if lo is None and hi is None:
                log_bounds.append((None, None))
                continue
                
            # ensure strictly positive and lo < hi
            lo = max(eps, lo) if lo is not None else eps
            hi = max(lo * (1.0 + 1e-12), hi) if hi is not None else lo * 1e12
            
            log_bounds.append((np.log(lo), np.log(hi)))
            
        return log_bounds

    
    def fit(self,
            learn_hyperparams: bool = True,
            maxiter: int = 300,
            method: str = "L-BFGS-B",
            learn_noise: bool = False,
            noise_bounds: tuple[float, float] | None = None,
            noise_init: float | None = None,
            callback: Optional[Callable[[int, float, Hyperparams], None]] = None,
            hp_bounds: dict | None = None,
            **kwargs):
        
        theta0   = self.hp.as_vector(log_space=True)
        core_len = len(theta0)
        
        # teach optimizer about the nugget if requested
        self._opt_core_len = core_len  # let _elbo know split point
        self._learn_noise  = bool(learn_noise)
        
        it = {"k": 0}
        
        if self._learn_noise:
            
            self.hp.noise_mode = "global_nugget"
            if self.hp.log_noise0 is None:
                
                sig0 = 1e-4 if noise_init is None else max(float(noise_init), 1e-12)
                self.hp.log_noise0 = np.log(sig0)
                
            theta0 = np.concatenate([theta0, [self.hp.log_noise0]])
        
        bounds_vec         = None
        noise_bounds_in_hp = False
        if hp_bounds is not None:
            
            if self._learn_noise and ('noise' in hp_bounds):
                bounds_vec         = self._build_svgp_hp_bounds(theta0,
                                                                hp_bounds)
                noise_bounds_in_hp = True
            else:
                bounds_vec = self._build_svgp_hp_bounds(theta0[:core_len],
                                                        hp_bounds)

        if self._learn_noise and not noise_bounds_in_hp:
            
            lo, hi = (1e-10, 1e-2) if noise_bounds is None else noise_bounds
            lo     = max(lo, 1e-16)
            hi     = max(hi, lo*(1+1e-12))
            
            noise_log_bounds = (np.log(lo), np.log(hi))
            
            if bounds_vec is None:
                bounds_vec = [(None, None)] * core_len + [noise_log_bounds]
            else:
                bounds_vec = list(bounds_vec) + [noise_log_bounds]
        
        
        def obj(v):
            
            val = -self._elbo(v)
            self.record(it["k"], val, v, notes={"objective": "neg_elbo"})
            
            if callback: 
                callback(it["k"], -val, self.hp)
                
            it["k"] += 1
            return val
        
        if learn_hyperparams:
            res = minimize(
            fun=obj,
            x0=theta0,
            method=method,
            bounds=bounds_vec,
            options={"maxiter": maxiter, "disp": False}
        )

            # write back core hyperparameters
            self.hp = Hyperparams.from_vector(res.x[:core_len],
                                              shared_ell=self.hp.shared_ell,
                                              log_space=True)

            # persist learned nugget (if any)
            if self._learn_noise:
                self.hp.noise_mode     = "global_nugget"
                self.hp.log_noise0     = float(res.x[core_len])
                self.noise_nugget_var_ = float(np.exp(self.hp.log_noise0))**2
            else:
                self.noise_nugget_var_ = 0.0
            
            self.state.cache["opt_svgp"] = res
            
        else:
            _ = obj(theta0)

            
    def predict_latent(self,
                       Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        if self._L_B is None or self._w is None:
            raise RuntimeError("SVGP model not fitted.")
            
        Xstar = _check_2d_coords(Xstar)
        hp    = self.hp
        M     = self.M
        Kx    = self._k_latent(0, Xstar, self.Zx, hp)
        Ky    = self._k_latent(1, Xstar, self.Zy, hp)
        Kz    = self._k_latent(2, Xstar, self.Zz, hp)
        mean  = np.stack([Kx @ self._w[0:M],
                          Ky @ self._w[M:2*M],
                          Kz @ self._w[2*M:3*M]],
                         axis=1)

        # Diag variance: var - diag(K_*u (Kuu^{-1} - B^{-1}) K_u*)
        L_B = self._L_B
        
        Lx = self.state.cache["Kuu_chol"]["Lx"]
        Ly = self.state.cache["Kuu_chol"]["Ly"]
        Lz = self.state.cache["Kuu_chol"]["Lz"]
        
        Kux_inv = cho_solve((Lx, True),
                            np.eye(Lx.shape[0]))
        Kuy_inv = cho_solve((Ly, True),
                            np.eye(Ly.shape[0]))
        Kuz_inv = cho_solve((Lz, True),
                            np.eye(Lz.shape[0]))

        
        # helper
        def diag_term(Ks: np.ndarray,
                      blk_slice: slice,
                      Kuu_inv_blk: np.ndarray) -> np.ndarray:
            
            Ku_star = Ks.T
            
            # Solve for V = B^{-1} K_u* (use cho_solve)
            pad_top = blk_slice.start
            pad_bot = 3*M - blk_slice.stop
            Ku_pad  = np.pad(Ku_star,
                             ((pad_top, pad_bot),
                              (0, 0)),
                             mode="constant")
            V_full  = cho_solve((L_B, True),
                                Ku_pad)
            V_blk   = V_full[blk_slice, :]
            diag_A  = np.sum(Ks * (Kuu_inv_blk @ Ku_star).T, axis=1)
            diag_B  = np.sum(Ks * V_blk.T, axis=1)
            
            return diag_A - diag_B

        var_x = hp.var_xyz[0] - diag_term(Kx, slice(0, M), Kux_inv)
        var_y = hp.var_xyz[1] - diag_term(Ky, slice(M, 2*M), Kuy_inv)
        var_z = hp.var_xyz[2] - diag_term(Kz, slice(2*M, 3*M), Kuz_inv)
        
        var = np.stack([np.maximum(var_x, 0.0),
                        np.maximum(var_y, 0.0),
                        np.maximum(var_z, 0.0)],
                       axis=1)
        
        return mean, var

    
    def predict_output(self,
                       Xstar: np.ndarray,
                       Hstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        mean_f, var_f = self.predict_latent(Xstar)
        m             = np.sum(Hstar * mean_f, axis=1)
        v             = np.sum((Hstar**2) * var_f, axis=1)
        
        return m, v
    
    
# ------------------------------ SLFM + IVM (SoD) ------------------------------

class SLFMIVMModel(BaseFusedModel):
    """
    Independent latents for ux, uy, uz with IVM forward selection and posterior.
    Optional fault-gated kernel.
    """

    def __init__(self,
                 data: FusedDataset,
                 hp: Hyperparams,
                 active_size: int = 1500,
                 fault: Optional[FaultGating] = None,
                 jitter: float = 1e-6,
                 random_state: Optional[int] = None):
        
        super().__init__(data,
                         jitter=jitter,
                         random_state=random_state)
        
        self.hp                        = hp
        self.active_size               = int(active_size)
        self.fault                     = fault
        self.A_mask                    = np.zeros(self.data.X.shape[0], dtype=bool)
        self.A_idx: List[int]          = []
        self.L_A: Optional[np.ndarray] = None
        self.K_XA                      = np.zeros((self.data.X.shape[0], 0))
        
        # Cache gating weights
        if self.fault is not None:
            
            w_plus, w_minus = self.fault.weights(self.data.X)
            self._wX_plus   = w_plus
            self._wX_minus  = w_minus
            
            self.state.cache["fault_w"] = {"w_plus": w_plus, "w_minus": w_minus}

            
    def _k_latent(self,
                  ell: int,
                  X1: np.ndarray,
                  X2: np.ndarray,
                  hp: Hyperparams) -> np.ndarray:
        
        var   = hp.var_xyz[ell]
        ellsc = hp.ell_xyz[ell]
        
        if self.fault is None:
            return rbf_kernel(X1, X2, var, ellsc)
        
        else:
            return fault_gated_rbf(X1, X2, var, ellsc, None, None, self.fault)
        

    def _cov_y(self,
               X1: np.ndarray,
               H1: np.ndarray,
               X2: np.ndarray,
               H2: np.ndarray,
               hp: Hyperparams) -> np.ndarray:
        
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for ell in (0, 1, 2):
            K += (H1[:, [ell]] @ H2[:, [ell]].T) * self._k_latent(ell, X1, X2, hp)
            
        return K

    
    def _k_col(self,
               j: int) -> np.ndarray:
        
        Xj = self.data.X[j:j+1]
        Kj = np.zeros(self.data.X.shape[0])
        
        for ell in (0, 1, 2):
            
            kvec = self._k_latent(ell,
                                  self.data.X,
                                  Xj,
                                  self.hp).reshape(-1)
            
            Kj += (self.data.H[:, ell] * self.data.H[j, ell]) * kvec
            
        return Kj

    
    def _prior_diag(self) -> np.ndarray:
        
        H = self.data.H
        if self.fault is None:
            diag = (H[:, 0]**2) * self.hp.var_xyz[0] \
                 + (H[:, 1]**2) * self.hp.var_xyz[1] \
                 + (H[:, 2]**2) * self.hp.var_xyz[2]
            
        else:
            
            s2   = self._wX_plus**2 + self._wX_minus**2
            diag = (H[:, 0]**2) * (self.hp.var_xyz[0] * s2) \
                 + (H[:, 1]**2) * (self.hp.var_xyz[1] * s2) \
                 + (H[:, 2]**2) * (self.hp.var_xyz[2] * s2)
            
        return diag

    
    def _predictive_var_all(self,
                            A_mask: np.ndarray,
                            hp: Hyperparams) -> np.ndarray:
        
        X, H, var = self.data.X, self.data.H, self.data.var
        
        if np.sum(A_mask) == 0:
            return self._prior_diag() + var
        
        A_idx        = np.where(A_mask)[0]
        XA, HA, varA = X[A_idx], H[A_idx], var[A_idx]
        
        KA = self._cov_y(XA, HA, XA, HA, hp) + np.diag(varA + self.jitter)
        L  = _safe_cholesky(KA,
                            self.jitter)
        
        k_ii = self._prior_diag()
        K_iA = self._cov_y(X,
                           H,
                           XA,
                           HA,
                           hp)
        
        tmp = solve_triangular(L,
                               K_iA.T,
                               lower=True,
                               check_finite=False)
        
        Kinv_K = solve_triangular(L.T,
                                  tmp,
                                  lower=False,
                                  check_finite=False)
        
        return np.maximum(k_ii - np.sum(K_iA * Kinv_K.T, axis=1) + var, 0.0)

    
    def fit(self,
            learn_hyperparams: bool = True,
            maxiter: int = 200,
            method: str = 'L-BFGS-B',
            early_stop: bool = False,  # stop active set selection based on `var_tol` & `gain_tol`
            var_tol: float|None=None,  # threshold for variance
            gain_tol: float|None=None, # threshold for mutual information
            enforce_gnss_first: bool = False, # seed active set with all GNSS observations first, then add InSAR/LiDAR
            min_insar_after_gnss: int = 0, # set a minimum number of InSAR/LiDAR samples to be included post-GNSS
            var_bounds: list = None,
            ell_bounds: list = None): 
        
        
        #  --- diagnostics histories --- #
        
        # conditional variance at selected j
        self.s_history  = []           
        # Mutual-information style gain per iteration (nats)
        self.mi_history = []
        
        # Modality selection tracking
        self.count_insar_selected = 0
        self.count_gnss_selected  = 0
        self.insar_selected_xy    = []
        self.gnss_selected_xy     = []
        
        N      = self.data.X.shape[0]
        A_mask = self.A_mask.copy()

        # --- modality masks --- #
        tags   = np.asarray(self.data.tags, dtype=str)
        tags_l = np.char.lower(tags)

        is_gnss  = np.char.startswith(tags_l, 'gnss')
        is_insar = np.char.startswith(tags_l, 'insar')

        # Asc/Des detection:
        # prefer explicit prefixes, but fall back to substring hints if needed
        is_asc = np.char.startswith(tags_l, 'insar_asc') | (is_insar & (np.char.find(tags_l, 'asc') >= 0))
        is_des = np.char.startswith(tags_l, 'insar_des') | (is_insar & (np.char.find(tags_l, 'des') >= 0))

        num_gnss = int(np.sum(is_gnss))

        # If forcing GNSS-first, ensure cap can hold them all
        if enforce_gnss_first and (self.active_size < num_gnss):
            self.active_size = num_gnss
            
        # Greedy selection using Schur-exact conditional variance updates
        s = self._prior_diag() + self.data.var
        
        for k in range(min(self.active_size, N)):
            
            # pick j with largest s (excluding selected)
            s_masked = np.where(A_mask, -np.inf, s)

            # GNSS-first and asc/desc balancing
            if enforce_gnss_first:
                
                used_gnss = int(np.sum(A_mask & is_gnss))
                if used_gnss < num_gnss:
                    # restrict to GNSS until all included
                    s_masked = np.where(is_gnss, s_masked, -np.inf)
                    
                else:
                    
                    # after GNSS, prefer the less-represented InSAR geometry
                    used_asc = int(np.sum(A_mask & is_asc))
                    used_des = int(np.sum(A_mask & is_des))
                    
                    if used_asc <= used_des:
                        pref = is_asc
                    else:
                        pref = is_des
                        
                    # restrict to preferred modality if any eligible remain
                    eligible_pref = (~A_mask) & pref
                    if np.any(eligible_pref):
                        s_masked = np.where(pref, s_masked, -np.inf)
                    else:
                        # otherwise, any InSAR
                        s_masked = np.where(is_insar, s_masked, -np.inf)
                        
            j = int(np.argmax(s_masked))
            
            # Update chol
            A    = np.asarray(self.A_idx, dtype=int)
            K_Aj = (self._cov_y(self.data.X[A],
                                self.data.H[A],
                                self.data.X[j:j+1],
                                self.data.H[j:j+1],
                                self.hp).reshape(-1)
                    
                    if A.size > 0 else np.empty((0,), dtype=float))
            
            k_jj = float(self._cov_y(self.data.X[j:j+1],
                                     self.data.H[j:j+1],
                                     self.data.X[j:j+1],
                                     self.data.H[j:j+1],
                                     self.hp)
                           + max(self.data.var[j], 1e-12))

            # --- Information-theoretic early-stop (optional) --- #
            if early_stop:
                
                apply_stop = early_stop
                if enforce_gnss_first:
                    
                    used_gnss  = int(np.sum(A_mask & is_gnss))
                    used_insar = int(np.sum(A_mask & is_insar))
                    apply_stop = early_stop and (used_gnss >= num_gnss) and (used_insar >= int(min_insar_after_gnss))
                    
                # conditional variance at j from current s vector
                s_j = float(max(s[j], 1e-18))
                
                # predictive mean at j given current active set
                if len(self.A_idx) > 0 and (self.L_A is not None) and (self.L_A.size > 0):
                    
                    A     = np.asarray(self.A_idx, dtype=int)
                    yA    = self.data.y[A]
                    alpha = solve_triangular(self.L_A.T,
                                             solve_triangular(self.L_A,
                                                              yA,
                                                              lower=True,
                                                              check_finite=False),
                                             check_finite=False)
                    
                    K_Aj_ = self._cov_y(self.data.X[A],
                                        self.data.H[A],
                                        self.data.X[j:j+1],
                                        self.data.H[j:j+1],
                                        self.hp).reshape(-1)
                    
                    mu_j  = float(K_Aj_ @ alpha) if (alpha.size == K_Aj_.size and alpha.size > 0) else 0.0
                    
                else:
                    mu_j = 0.0
                    
                r_j      = float(self.data.y[j] - mu_j)
                sigma2_j = float(self.hp.effective_sigma2(self.data)[j]) \
                                 if hasattr(self.hp, 'effective_sigma2') \
                                 else float(self.data.var[j])
                
                tau2 = max(s_j - sigma2_j, 0.0)
                gain = 0.5 * np.log1p(tau2 / max(sigma2_j, 1e-18))
                
                # store histories
                self.s_history.append(s_j)
                self.mi_history.append(gain)
                
                # thresholds
                if var_tol is None:
                    
                    if hasattr(self.hp, 'effective_sigma2'):
                        sig2 = self.hp.effective_sigma2(self.data)
                    else:
                        sig2 = self.data.var
                        
                    var_tol_eff = float(np.median(sig2))
                    
                else:
                    var_tol_eff = float(var_tol)
                    
                # interpreted as MI tol (nats)
                gain_tol_eff = 1e-3 if gain_tol is None else float(gain_tol) 
                
                if apply_stop and (s_j <= var_tol_eff) and (gain <= gain_tol_eff):
                    # stop growing the active set
                    break
                    
            else:
                
                # still record s for plotting
                self.s_history.append(float(max(s[j], 1e-18)))
                self.gain_history.append(np.nan)
                
            if self.L_A is None:
                L_new    = np.array([[np.sqrt(max(k_jj + self.jitter, 1e-18))]], dtype=float)
                self.L_A = L_new
                self.A_idx.append(j)
                
            else:
                v = solve_triangular(self.L_A,
                                     K_Aj,
                                     lower=True,
                                     check_finite=False)

                v_max = float(np.max(np.abs(v))) if v.size else 0.0
                
                if not np.isfinite(v_max) or v_max > 1e150:
                    # force sj to lower bound
                    vv = float(k_jj + 1.0)  
                else:
                    vv = float(np.dot(v, v))
                    
                sj    = k_jj - vv
                L_new = np.zeros((self.L_A.shape[0] + 1, self.L_A.shape[1] + 1))
                
                L_new[:-1, :-1] = self.L_A
                L_new[-1, :-1]  = v
                L_new[-1, -1]   = np.sqrt(max(sj, 1e-18))
                
                self.L_A = L_new
            
                self.A_idx.append(j)
                
            # Full column and Schur update s <- s - p^2 / s_j
            K_col_j = self._k_col(j)

            if self.K_XA.shape[1] == 0 or self.L_A.shape[0] <= 1:
                KXA_alpha = np.zeros(N, dtype=float)
            else:
                alpha = cho_solve((self.L_A[:-1, :-1], True), K_Aj, check_finite=False)

                if not np.all(np.isfinite(alpha)):
                    alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
                    
                KXA_alpha = self.K_XA @ alpha
                
            # ----------------- #
            
            p            = K_col_j - KXA_alpha
            den          = max(float(self.L_A[-1, -1]**2), 1e-18)
            inv_sqrt_den = 1.0 / max(np.sqrt(den), 1e-18)
            q            = p * inv_sqrt_den
            s_pos        = np.maximum(s, 0.0)
            cap          = np.sqrt(s_pos)
            q            = np.clip(q, -cap, cap)
            s            = s - q*q
            s[A_mask]    = -np.inf
            
            # Update sets
            A_mask[j] = True

            # --- modality tracking --- #
            
            tag_j = str(self.data.tags[j]) if hasattr(self.data, 'tags') else ''
            xy_j  = self.data.X[j]
            
            if tag_j.startswith('gnss'):
                self.count_gnss_selected += 1
                self.gnss_selected_xy.append(xy_j)
            else:
                # default everything else to InSAR in this dataset
                self.count_insar_selected += 1
                self.insar_selected_xy.append(xy_j)
            
            # count asc/des specifically if possible
            if 'insar_asc' in tag_j or ('asc' in tag_j and 'insar' in tag_j):
                
                if not hasattr(self, 'count_insar_asc_selected'):
                    self.count_insar_asc_selected = 0
                    self.insar_asc_selected_xy    = []
                    
                self.count_insar_asc_selected += 1
                self.insar_asc_selected_xy.append(xy_j)
                
            elif 'insar_des' in tag_j or ('des' in tag_j and 'insar' in tag_j):
                
                if not hasattr(self, 'count_insar_des_selected'):
                    self.count_insar_des_selected = 0
                    self.insar_des_selected_xy    = []
                    
                self.count_insar_des_selected += 1
                self.insar_des_selected_xy.append(xy_j)
                
            self.K_XA = np.hstack([self.K_XA, K_col_j.reshape(-1, 1)])
            self.record(k,
                        float(np.mean(np.maximum(s, 0.0))),
                        self.hp.as_vector(),
                        notes={"event": "select", "index": j})
            
        self.A_mask = A_mask
        
        self.state.cache["A_idx"] = np.array(self.A_idx, dtype=int)
        self.state.cache["L_A"]   = self.L_A

        # Hyperparameter refinement
        if learn_hyperparams and len(self.A_idx) > 0:
            
#             A = np.array(self.A_idx, dtype=int)
#             XA, HA, yA, varA = self.data.X[A], self.data.H[A], self.data.y[A], self.data.var[A]
            
            A                = np.array(self.A_idx, dtype=int)
            XA, HA, yA, varA = self.data.X[A], self.data.H[A], self.data.y[A], self.data.var[A]
            theta0           = self.hp.as_vector(log_space=True)
            
            if ell_bounds is None and self.fault is not None:
                extent = float(np.max(np.ptp(XA, axis=0))) # km
                ell_lo = max(0.5, 0.20*extent)             # ~2% of extent, floor at 0.5 km
                ell_hi = max(1.0, 0.75*extent)             # up to ~75% of extent
                
            elif ell_bounds is not None:
                ell_lo = ell_bounds[0]
                ell_hi = ell_bounds[1]
                
                
            nvar   = 3
            bounds = []
            
            
            for j in range(nvar):
                if var_bounds is None:
                    vj = np.exp(theta0[j])
                    bounds.append((np.log(max(vj*1e-8, 1e-16)), np.log(vj*1e8)))
                    
                else:
                    bounds.append(var_bounds)
                    
            if self.hp.shared_ell:
                bounds.append((np.log(ell_lo), np.log(ell_hi)))
            else:
                for _ in range(nvar):
                    bounds.append((np.log(ell_lo), np.log(ell_hi)))
            
            
            def nll(theta_vec: np.ndarray) -> float:
                
                hp = Hyperparams.from_vector(theta_vec,
                                             shared_ell=self.hp.shared_ell,
                                             log_space=True)
                
                KA = self._cov_y(XA, HA, XA, HA, hp) + np.diag(varA + self.jitter)
                L  = _safe_cholesky(KA,
                                    self.jitter)
                
                alpha = cho_solve((L, True),
                                  yA,
                                  check_finite=False)
                
                nll = 0.5 * (yA @ alpha) \
                    + np.sum(np.log(np.diag(L))) \
                    + 0.5 * len(yA) * np.log(2.0 * np.pi)
                
                self.hp = hp
                
                self.record(len(self.history),
                            float(nll),
                            hp.as_vector(),
                            notes={"objective": "SoD_nll"})
                
                return float(nll)
            
            res = minimize(nll,
                           x0=self.hp.as_vector(log_space=True),
                           method=method,
                           bounds=bounds,
                           options={"maxiter": maxiter, "disp": False})
            
            self.state.cache["opt_sod"] = res

        # finalize modality arrays for user access
        self.gnss_selected_xy = np.asarray(self.gnss_selected_xy, dtype=float) \
                                if len(self.gnss_selected_xy)>0 \
                                else np.zeros((0,2))
        
        self.insar_selected_xy = np.asarray(self.insar_selected_xy, dtype=float) \
                                 if len(self.insar_selected_xy)>0 \
                                 else np.zeros((0,2))


    def predict_latent(self,
                       Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.L_A is None or len(self.A_idx) == 0:
            raise RuntimeError("Model not fitted or empty active set.")
            
        Xstar = _check_2d_coords(Xstar)
        A     = np.array(self.A_idx, dtype=int)
        
        XA, HA, yA = self.data.X[A], self.data.H[A], self.data.y[A]
        
        varA = self.data.var[A]
        L    = _safe_cholesky(self._cov_y(XA, HA, XA, HA, self.hp) + np.diag(varA + self.jitter),
                              self.jitter)
        
        alpha = cho_solve((L, True),
                          yA,
                          check_finite=False)

        def comp(ell: int):
            
            k_star_A = self._k_latent(ell,
                                      Xstar,
                                      XA,
                                      self.hp) * HA[:, [ell]].T
            mean     = k_star_A @ alpha
            v        = solve_triangular(L,
                                        k_star_A.T,
                                        lower=True,
                                        check_finite=False)
            var_diag = self.hp.var_xyz[ell] - np.sum(v**2, axis=0)
            
            return mean, np.maximum(var_diag, 0.0)

        mx, vx = comp(0)
        my, vy = comp(1)
        mz, vz = comp(2)
        
        mean = np.stack([mx, my, mz], axis=1)
        var  = np.stack([vx, vy, vz], axis=1)
        
        return mean, var

    
    def predict_output(self,
                       Xstar: np.ndarray,
                       Hstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict LOS for set of sample points.
        """
        
        m_f, v_f = self.predict_latent(Xstar)
        m        = np.sum(Hstar * m_f, axis=1)
        v        = np.sum((Hstar**2) * v_f, axis=1)
        
        return m, v

    
    def plot_ivm_info(self,
                      ax=None,
                      save=False,
                      save_name='active_set_diagnostics.png',
                      show=False):
        
        """
        Plot evolution of selected-point conditional variance (s_j) and MI (nats).
        """
            
        if not hasattr(self, 's_history') or not hasattr(self, 'mi_history') or len(self.s_history) == 0:
            raise RuntimeError("No IVM history stored. Run fit() first (with or without early_stop).")
            
        s_hist = np.asarray(self.s_history, float)[225:]
        g_hist = np.asarray(self.mi_history, float)[225:]
        it     = np.arange(1, len(s_hist)+1)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            own     = True
        else:
            own = False
            
        ax2 = ax.twinx()
        
        ax.plot(it,
                s_hist,
                lw=1,
                label='conditional variance, $s_j$')
        
        ax2.plot(it,
                 np.cumsum(g_hist),
                 lw=1,
                 ls=':',
                 label='mutual information, $I(v_A, v_j)$',
                 color='r')
        
        ax.set_xlabel('active-set iteration')
        ax.set_ylabel('$s_j$')
        ax2.set_ylabel('$I(v_A, v_j)$  [nats]')
        
        lines, labels   = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        ax.legend(lines+lines2, labels+labels2, loc='upper center')

        ax.set_ylim(0)
        ax2.set_ylim(0)
        
        if own:
            plt.tight_layout()
            return fig, (ax, ax2)
        
        if show:
            plt.show()
            
        if save:
            fig.savefig(save_name, dpi=100)
        
#         return ax, ax2


    def __post_init__(self):
        
        self.B = np.asarray(self.B, float)
        
        if self.B.shape != (3, self.Q):
            raise ValueError("B must be shaped (3, Q).")
            