
"""
Sparta, NC 2020 earthquake with a discontinuous (fault-gated) kernel.
Inputs:
  - alos_los.csv, sentinel1_los.csv (ascending LOS from two platforms)
  - lidar_ICP.csv (3-component LiDAR displacements with look vectors at those points)
  - sample_points.csv (or samples_points.csv) with look vectors at query locations
  - trace_points.csv (SE->NW ordered fault trace in km)

Outputs (same filenames as the other demo):
  - posterior_latents_{slfm,lmc,svgp}.csv
  - posterior_los_{slfm,lmc,svgp}.csv
  - posterior_latents_{slfm,lmc,svgp}.tif  (GeoTIFF by default; NetCDF fallback)
  - posterior_los_{slfm,lmc,svgp}.tif      (GeoTIFF by default; NetCDF fallback)
"""

import os, sys, time
sys.path.append('../../gentoo')
import numpy as np
import pandas as pd
from datetime import timedelta

from datasets import *
from utilities import *
from variography import *
from fusion_models import *
from kernels import *


# ============================= USER PARAMETERS ============================= #

DATA_DIR      = 'data'
OUT_DIR       = 'results'
EXPORT_FORMAT = 'NetCDF'   # or "GTiff"

ELL_BOUNDS    = (0.1, 5.0) # range for length scale hyperparameter
M_SIZE        = 700        # Inducing set size
MAXITER       = 300        # max number of iterations
Z_MODE        = "fps"      # Mode for selecting inducing set

SHARE_Z       = True       # Share inducing locations for all components
LEARN_HPS     = True       # Learn hyperparameters versus using Variography
SHARE_HPS     = False      # Share same hyperparameters for all components
   
GATE_MODE     = 'hard'     # Set mode for gated, fault-aware kernel
GATE_WIDTH    = 0.05       # set soft barrier width [m]. (will only take effect if mode='soft')
   
GEO_COORDS    = True       # project from cartesian to geocoordinates


# ============================== Load CSVs ============================== #

print('[status] Starting Meinong demo.')

alos_path     = "alos_los.csv"
s1_path       = "sentinel1_los.csv"
lidar_path    = "lidar_ICP.csv"
sample_path   = "sample_points.csv"
transect_path = 'transect.csv'
trace_path    = "trace_points.csv"

print(f'[I/O]  reading {alos_path}')
alos        = pd.read_csv(os.path.join(DATA_DIR, alos_path))
print(f'[I/O]  reading {s1_path}')
s1          = pd.read_csv(os.path.join(DATA_DIR, s1_path))
print(f'[I/O]  reading {lidar_path}')
lid         = pd.read_csv(os.path.join(DATA_DIR, lidar_path))
print(f'[I/O]  reading {sample_path}')
sample_grid = pd.read_csv(os.path.join(DATA_DIR, sample_path))
print(f'[I/O]  reading {transect_path}')
transect    = pd.read_csv(os.path.join(DATA_DIR, transect_path))
print(f'[I/O]  reading {trace_path}')
trace       = pd.read_csv(os.path.join(DATA_DIR, trace_path))

# =============================== Column checks =============================== #

def require(df,
            cols,
            name):
    
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

require(alos,        ["X","Y","los","sigma","lx","ly","lz"], "alos_los.csv")
require(s1,          ["X","Y","los","sigma","lx","ly","lz"], "sentinel1_los.csv")
require(lid,         ["X","Y","ux","uy","uz","sx","sy","sz","lxs","lys","lzs","lxa","lya","lza"], "lidar_ICP.csv")
require(sample_grid, ["X","Y","lxs","lys","lzs","lxa","lya","lza"], "lo_sample_points.csv")
require(trace,       ["X","Y"], "trace_points.csv")
require(transect,    ["X","Y","lxs","lys","lzs","lxa","lya","lza"], "transect.csv")

print('[status] Data load successful.')

# ============================ Build data containers ============================ #

print('[status] Builing data containers.')

# InSAR ALOS
Xa         = alos[["X","Y"]].to_numpy(float)
los_a      = alos["los"].to_numpy(float)
var_a      = (alos["sigma"].to_numpy(float))**2
look_a     = alos[["lx","ly","lz"]].to_numpy(float)
insar_alos = InSARData(Xa, los_a, var_a, look_a)

# InSAR Sentinel-1
Xs       = s1[["X","Y"]].to_numpy(float)
los_s    = s1["los"].to_numpy(float)
var_s    = (s1["sigma"].to_numpy(float))**2
look_s   = s1[["lx","ly","lz"]].to_numpy(float)
insar_s1 = InSARData(Xs, los_s, var_s, look_s)

# LiDAR 3-component
Xl    = lid[["X","Y"]].to_numpy(float)
Ul    = lid[["ux","uy","uz"]].to_numpy(float)
Vl    = (lid[["sx","sy","sz"]].to_numpy(float))**2
lidar = LiDARData(Xl, Ul, Vl)

# Fused dataset: LiDAR + both InSAR tracks
fused = FusedDataset.from_sources(gnss=None,
                                  insar_list=[insar_alos, insar_s1],
                                  lidar=lidar)

print(f"[info] Fused dataset (Sparta): \n\tN={fused.X.shape[0]} scalar obs "
      f"(LiDAR comps: {Xl.shape[0]*3}, InSAR: {Xa.shape[0] + Xs.shape[0]})")

# =============================== Fault-gated kernel =============================== #

poly  = trace[["X","Y"]].to_numpy(float)
fault = FaultGating(polyline=poly,
                    width=GATE_WIDTH,
                    mode=GATE_MODE)

# =========================  Hyperparameters (variography) ========================= #

hp0 = hyperparams_from_variography(gnss=None,
                                   lidar=lidar,
                                   default_ell=10.0,
                                   shared_ell=SHARE_HPS)
print("[info] Initial hyperparameters from variography (LiDAR):")
print("\tvar_xyz =", hp0.var_xyz, "\n\tell_xyz =", hp0.ell_xyz)

# store initial HPs for output
varx0 = hp0.var_xyz[0]
vary0 = hp0.var_xyz[1]
varz0 = hp0.var_xyz[2]

ellx0 = hp0.ell_xyz[0]
elly0 = hp0.ell_xyz[1]
ellz0 = hp0.ell_xyz[2]

# ========================== Query points & look vectors ========================== #

Xq      = sample_grid[["X","Y"]].to_numpy(float)
Hq_alos = sample_grid[["lxa","lya","lza"]].to_numpy(float)   # ALOS look vectors at query points
Hq_s1   = sample_grid[["lxs","lys","lzs"]].to_numpy(float)   # Sentinel-1 look vectors at query points

X_transect = transect[["X","Y"]].to_numpy(float)
Ht_alos    = transect[["lxa","lya","lza"]].to_numpy(float)   # ALOS look vectors at query points
Ht_s1      = transect[["lxs","lys","lzs"]].to_numpy(float)   # Sentinel-1 look vectors at query points


# ================================ SLFM + SVGP ==================================== #

print("[status] SVGP/VFE + fault barrier fitting...")

# initialize model
svgp = SVGPModel(
    fused,
    hp0,
    M=M_SIZE,
    inducing_selector=Z_MODE,
    shared_Z=SHARE_Z,
    jitter=1e-5,
    random_state=0,
    fault=fault
)

# timers
start_wall, start_cpu = time.perf_counter(), time.process_time()

# fit model
svgp.fit(
    learn_hyperparams=LEARN_HPS,
    maxiter=MAXITER,
    method="L-BFGS-B",
    hp_bounds={'ell': ELL_BOUNDS},
    learn_noise=False
)

wall_duration = time.perf_counter() - start_wall
cpu_duration = time.process_time() - start_cpu

print("[status] SVGP/VFE + fault barrier fit.")
print(f"\tWall Time: {timedelta(seconds=wall_duration)}")
print(f"\t CPU Time: {timedelta(seconds=cpu_duration)}")

print("[status] Sampling posterior...")

# timers
start_wall, start_cpu = time.perf_counter(), time.process_time()

# latent deformation field at sample grid
mf_svgp, vf_svgp = svgp.predict_latent(Xq)

# LOS projection of latent field at sample grid
mlosA_svgp, vlosA_svgp = svgp.predict_output(Xq, Hq_alos)
mlosS_svgp, vlosS_svgp = svgp.predict_output(Xq, Hq_s1)

# ...at LiDAR sample locations
mf_lidar, vf_lidar = svgp.predict_latent(Xl)

# LOS projection of latent field at ALOS/S-1 locations
mAlos_svgp, vAlos_svgp = svgp.predict_output(Xa, look_a)
mSlos_svgp, vSlos_svgp = svgp.predict_output(Xs, look_s)

# Transect
mf_svgp_transect, vf_svgp_transect       = svgp.predict_latent(X_transect)
mlosA_svgp_transect, vlosA_svgp_transect = svgp.predict_output(X_transect, Ht_alos)
mlosS_svgp_transect, vlosS_svgp_transect = svgp.predict_output(X_transect, Ht_s1)

wall_duration = time.perf_counter() - start_wall
cpu_duration = time.process_time() - start_cpu

print('[status] Sampling complete.')
print(f"\tWall Time: {timedelta(seconds=wall_duration)}")
print(f"\t CPU Time: {timedelta(seconds=cpu_duration)}")

# get and save inducing points
inducing_pts = svgp.Zx

# ================================== Save================================== #

os.makedirs(OUT_DIR,
            exist_ok=True)

pd.DataFrame(
    inducing_pts*1e3,
    columns=["X", "Y"]
).to_csv(
    os.path.join(
        OUT_DIR,
        "inducing_points.csv"
    ),
    index=False
)

pd.DataFrame(
    np.hstack((
        Xq*1e3,
        mf_svgp,
        vf_svgp)
    ),
    columns=[
        "X","Y",
        "ux_mean","uy_mean","uz_mean",
        "ux_var","uy_var","uz_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_latents_svgp.csv"
    ),
    index=False
)

pd.DataFrame(
    np.hstack((
        Xa*1e3,
        mAlos_svgp.reshape(-1,1),
        vAlos_svgp.reshape(-1,1))
    ),
    columns=[
        "X","Y",
        "losA_mean","losA_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_ALOS_svgp.csv"
    ),
    index=False
)

pd.DataFrame(
    np.hstack((
        Xs*1e3,
        mSlos_svgp.reshape(-1,1),
        vSlos_svgp.reshape(-1,1))
    ),
    columns=[
        "X","Y",
        "losS_mean","losS_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_S1_svgp.csv"
    ),
    index=False
)

pd.DataFrame(
    np.hstack((
        X_transect*1e3,
        mf_svgp_transect,
        vf_svgp_transect)
    ),
    columns=[
        "X","Y",
        "ux_mean","uy_mean","uz_mean",
        "ux_var","uy_var","uz_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_latents_svgp_transect.csv"
    ),
    index=False
)

pd.DataFrame(
    np.hstack((
        X_transect*1e3,
        mlosA_svgp_transect.reshape(-1,1),
        vlosA_svgp_transect.reshape(-1,1))
    ),
    columns=[
        "X","Y",
        "losA_mean","losA_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_ALOS_svgp_transect.csv"
    ),
    index=False
)

pd.DataFrame(
    np.hstack((
        X_transect*1e3,
        mlosS_svgp_transect.reshape(-1,1),
        vlosS_svgp_transect.reshape(-1,1))
    ),
    columns=[
        "X","Y",
        "losS_mean","losS_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_S1_svgp_transect.csv"
    ),
    index=False
)

csv_files = [
    "inducing_points.csv",
    "posterior_latents_svgp.csv",
    "posterior_ALOS_svgp.csv",
    "posterior_S1_svgp.csv",
    "posterior_latents_svgp_transect.csv",
    "posterior_ALOS_svgp_transect.csv",
    "posterior_S1_svgp_transect.csv"
]

print('[I/O]  Saving csv files...')
for csv_file in csv_files:
    print(f'\t{csv_file} saved.')
    
if GEO_COORDS:
    print(f'[info] Projecting csv files to geocoordinates...')
    for csv_file in csv_files:

        outfile = csv_file.split('.')[0] + '_ll.txt'
        
        inpath  = os.path.join(OUT_DIR,
                               csv_file)
        outpath = os.path.join(OUT_DIR,
                               outfile)
        
        gmt_cmd = f'gmt mapproject {inpath} -Jt-81.094/36.476/1:1 -R-81.1529166839/-81.0295833506/36.4619444444/36.5305555556 -C -Fe -I -q1: > {outpath}'
        os.system(gmt_cmd)
        print(f'\tProjected file: {outfile} saved.')
          
        
# generate optimization path figure and save
fig1, axs = svgp.plot_optimization_paths(label="negative ELBO",
                                         show=True)
fig1_n    = 'sparta_optimization_path.png'
fig1.savefig(os.path.join(OUT_DIR,
                          fig1_n),
             dpi=100)
print(f'[I/O]  {fig1_n} saved.')

# generate HP history figure and save
fig2, axs = svgp.plot_hyperparam_traces(show=True)
fig2_n    = 'sparta_HP_history.png'
fig2.savefig(os.path.join(OUT_DIR,
                          fig2_n),
             dpi=100)
print(f'[I/O]  {fig2_n} saved.')

# get optimal HP values for output                                    
opt_vars = svgp.hp.var_xyz
opt_varx = opt_vars[0]
opt_vary = opt_vars[1]
opt_varz = opt_vars[2]

opt_ells = svgp.hp.ell_xyz
opt_ellx = opt_ells[0]
opt_elly = opt_ells[1]
opt_ellz = opt_ells[2]

# save some relevent values not output with .csv and .tif/netcdf files
labels = [
     'var0x',
     'var0y',
     'var0z',
     'ell0x',
     'ell0y',
     'ell0z',
     'var_optx',
     'var_opty',
     'var_optz',
     'ell_optx',
     'ell_opty',
     'ell_optz',
]

values = [
    varx0,
    vary0,
    varz0,
    ellx0,
    elly0,
    ellz0,
    opt_vary,
    opt_varz,
    opt_ells,
    opt_ellx,
    opt_elly,
    opt_ellz,
]

print("[info] Final hyperparameters from optimization:")
print("\tvar_xyz =", svgp.hp.var_xyz, "\n\tell_xyz =", svgp.hp.ell_xyz)

fname = 'Sparta_svgp_results.txt'

with open(os.path.join(DATA_DIR, fname), 'w+') as outfile:  
    for label, value in zip(labels, values):
        outfile.writelines(f'{label}: {value}\n')
outfile.close()

# ========================== Raster export ========================== #

export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_x_svgp.tif" if EXPORT_FORMAT=="GTiff" else "posterior_x_svgp.nc"
    ),
    Xq*1e3,
    {"ux_mean": mf_svgp[:,0],
     "ux_var":  vf_svgp[:,0]},
    driver=EXPORT_FORMAT
    )

export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_y_svgp.tif" if EXPORT_FORMAT=="GTiff" else "posterior_y_svgp.nc"
    ),
    Xq*1e3,
    {"uy_mean": mf_svgp[:,1],
     "uy_var":  vf_svgp[:,1]},
    driver=EXPORT_FORMAT
    )

export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_z_svgp.tif" if EXPORT_FORMAT=="GTiff" else "posterior_z_svgp.nc"
    ),
    Xq*1e3,
    {"uz_mean": mf_svgp[:,2],
     "uz_var":  vf_svgp[:,2]},
    driver=EXPORT_FORMAT
    )

export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_ALOS_svgp.tif" if EXPORT_FORMAT=="GTiff" else "posterior_ALOS_svgp.nc"
    ),
    Xq*1e3,
    {"los_mean": mlosA_svgp,
     "los_var": vlosA_svgp},
    driver=EXPORT_FORMAT
    )
export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_S1_svgp.tif" if EXPORT_FORMAT=="GTiff" else "posterior_S1_svgp.nc"
    ),
    Xq*1e3,
    {"los_mean": mlosS_svgp,
     "los_var": vlosS_svgp},
    driver=EXPORT_FORMAT
    )

raster_files = [
    "posterior_x_svgp.{}",
    "posterior_y_svgp.{}",
    "posterior_z_svgp.{}",
    "posterior_ALOS_svgp.{}",
    "posterior_S1_svgp.{}",
]

print(f'[I/O]  Saving {EXPORT_FORMAT} files...')
for raster_file in raster_files:
    print(f'\t{raster_file.format('tif' if EXPORT_FORMAT=="GTIFF" else 'nc')} saved.')

    
if GEO_COORDS:
    print(f'[info] Projecting {EXPORT_FORMAT} files to geocoordinates...')
    for raster_file in raster_files:

        outfile = raster_file.split('.')[0] \
                + '_ll.{}'.format('tif' if EXPORT_FORMAT=="GTIFF" else 'nc')
        
        inpath  = os.path.join(OUT_DIR,
                               raster_file.format('tif' if EXPORT_FORMAT=="GTIFF" else 'nc'))
        outpath = os.path.join(OUT_DIR,
                               outfile)
        
        gmt_cmd = f'gmt grdproject {inpath} -Jt-81.094/36.476/1:1 -Fe -G{outpath} -C -I'
        os.system(gmt_cmd)
        print(f'\tProjected file: {outfile} saved')
              
print("[status] Done.")


