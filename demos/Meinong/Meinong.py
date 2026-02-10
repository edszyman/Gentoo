
"""
Fits SLFM+IVM for the Meinong 2016 dataset (ascending/descending InSAR + GNSS + sample grid).

Script:
  1) Loads CSVs.
  2) Builds FusedDataset.
  3) Initializes hyperparameters via variography (from GNSS).
  4) Fits and evaluates:
      - SLFM + IVM (active set + SoD)
  5) Predicts posterior at sample grid for:
      - 3D latents (ux,uy,uz)
      - LOS for ascending & descending (using look vectors in sample_grid)
  6) Saves results to CSV.
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
EXPORT_FORMAT = "NetCDF"  # or "GTiff"

ELL_BOUNDS      = (1, 35) # range for length scale hyperparameter
ACTIVE_SET_SIZE = 8000    # max size of active set
MIN_INSAR       = 800     # min number InSAR observations to include in active set
MAXITER         = 200     # max number of iterations
GAIN_TOL        = 5e-4    # MI tolerance (nats)
VAR_TOL         = 1.2e-3  # set variance tolerance (default is median per-point variance)

ENFORCE_GNSS    = True    # enforce inclusion of GNSS into active set first
LEARN_HPS       = True    # learn hyper-parameters versus just using variography
SHARE_HPS       = False   # Share same hyperparameters for all components
EARLY_STOP      = True    # allow stopping prior to max active size

GEO_COORDS      = True    # project from cartesian to geocoordinates


# ================================== Load CSVs ================================== #

print('[status] Starting Meinong demo.')

asc_path  = os.path.join(DATA_DIR, "ascending_data.csv")
des_path  = os.path.join(DATA_DIR, "descending_data.csv")
gnss_path = os.path.join(DATA_DIR, "gnss_data.csv")
grid_path = os.path.join(DATA_DIR, "sample_grid.csv")

print(f'[I/O]  reading {asc_path}')
asc  = pd.read_csv(asc_path)
print(f'[I/O]  reading {des_path}')
des  = pd.read_csv(des_path)
print(f'[I/O]  reading {gnss_path}')
gns  = pd.read_csv(gnss_path)
print(f'[I/O]  reading {grid_path}')
grid = pd.read_csv(grid_path)


# Sanity check columns
for df, name, req_cols in [
    (asc, "ascending_data.csv", ["X", "Y", "los", "sigma", "lx", "ly", "lz"]),
    (des, "descending_data.csv", ["X", "Y", "los", "sigma", "lx", "ly", "lz"]),
    (gns, "gnss_data.csv", ["X", "Y", "ux", "uy", "uz", "sx", "sy", "sz", "lxa", "lya", "lza", "lxd", "lyd", "lzd"]),
    (grid, "sample_grid.csv", ["X", "Y", "lxa", "lya", "lza", "lxd", "lyd", "lzd"]),
]:
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
        
print('[status] Data load successful.')

# ============================== Build data containers ============================== #

print('[status] Builing data containers.')

# GNSS
Xg   = gns[["X", "Y"]].to_numpy(float)
Ug   = gns[["ux", "uy", "uz"]].to_numpy(float)
Vg   = (gns[["sx", "sy", "sz"]].to_numpy(float))**2
gnss = GNSSData(Xg, Ug, Vg)

# InSAR ascending
Xa        = asc[["X", "Y"]].to_numpy(float)
los_a     = asc["los"].to_numpy(float)
var_a     = (asc["sigma"].to_numpy(float))**2
look_a    = asc[["lx", "ly", "lz"]].to_numpy(float)
insar_asc = InSARData(Xa,
                      los_a,
                      var_a,
                      look_a)

# InSAR descending
Xd        = des[["X", "Y"]].to_numpy(float)
los_d     = des["los"].to_numpy(float)
var_d     = (des["sigma"].to_numpy(float))**2
look_d    = des[["lx", "ly", "lz"]].to_numpy(float)
insar_des = InSARData(Xd,
                      los_d,
                      var_d,
                      look_d)

# Fused dataset (stack ascending & descending)
fused = FusedDataset.from_sources(gnss=gnss,
                                  insar_list=[insar_asc, insar_des])


balancing = False
if balancing:
    # --- add asc/desc tags so the balancer sees them ---
    idx_insar = fused.meta["idx_insar"]             # slice covering all InSAR rows
    n_asc     = insar_asc.X.shape[0]
    tags      = fused.tags
    tags[idx_insar.start : idx_insar.start + n_asc] = ["insar_asc"] * n_asc
    tags[idx_insar.start + n_asc : idx_insar.stop]  = ["insar_des"] * (idx_insar.stop - (idx_insar.start + n_asc))
    fused.tags = tags

print(f"[info] Fused dataset: \n\tN={fused.X.shape[0]} scalar obs "
      f"(GNSS comps: {np.sum([t.startswith('gnss') for t in fused.tags])}, "
      f"InSAR: {np.sum([t=='insar' for t in fused.tags])})")

# =================== Initial hyperparameters via variography ================== #
hp0 = hyperparams_from_variography(gnss=gnss,
                                   lidar=None,
                                   default_ell=10.0,
                                   shared_ell=SHARE_HPS)

print("[info] Initial hyperparameters from variography:")
print("\tvar_xyz =", hp0.var_xyz, "\n\tell_xyz =", hp0.ell_xyz)

## store initial HPs for output
# variance
varx0 = hp0.var_xyz[0]
vary0 = hp0.var_xyz[1]
varz0 = hp0.var_xyz[2]

# length scale
ellx0 = hp0.ell_xyz[0]
elly0 = hp0.ell_xyz[1]
ellz0 = hp0.ell_xyz[2]

# ---- Prepare sample grid and look vectors for predictions ----
Xq     = grid[["X", "Y"]].to_numpy(float)
Hq_asc = grid[["lxa", "lya", "lza"]].to_numpy(float)  # rows are look vectors
Hq_des = grid[["lxd", "lyd", "lzd"]].to_numpy(float)


# ============================= SLFM + IVM ===================================== #

print("[status] SLFM+IVM Fitting...")
# initialize model
slfm = SLFMIVMModel(
    fused,
    hp0,
    active_size=ACTIVE_SET_SIZE,
    jitter=1e-6,
    random_state=0
)

# timers
start_wall, start_cpu = time.perf_counter(), time.process_time()

# fit model
slfm.fit(
    learn_hyperparams=LEARN_HPS,
    maxiter=MAXITER,
    early_stop=EARLY_STOP,           
    enforce_gnss_first=ENFORCE_GNSS, 
    min_insar_after_gnss=MIN_INSAR,  
    gain_tol=GAIN_TOL,  
    var_tol=VAR_TOL,    
    ell_bounds=ELL_BOUNDS
)

wall_duration = time.perf_counter() - start_wall
cpu_duration = time.process_time() - start_cpu

print("[status] SLFM+IVM Fit.")
print(f"\tWall Time: {timedelta(seconds=wall_duration)}")
print(f"\t CPU Time: {timedelta(seconds=cpu_duration)}")

print("[status] Sampling posterior...")

# timers
start_wall, start_cpu = time.perf_counter(), time.process_time()

# Predict latent fields and LOS at grid points
mf_slfm, vf_slfm           = slfm.predict_latent(Xq)
mAlos_sample, vAlos_sample = slfm.predict_output(Xq, Hq_asc)
mDlos_sample, vDlos_sample = slfm.predict_output(Xq, Hq_des)

X = slfm.data.X


# get ascending/descending sample info
asc_active = []
des_active = []
asc_idx = []
des_idx = []
for idx in slfm.A_idx:
    
    if idx > int(8790+225):
        des_active.append([float(X[idx][0]), float(X[idx][1])])
        des_idx.append(int(idx - int(8790+225)))

    elif (idx < int(8790+225)) and (idx > 225):
        asc_active.append([float(X[idx][0]), float(X[idx][1])])
        asc_idx.append(int(idx-int(225)))


# get predictions at locations used in the active set
mGNSS_slfm, vGNSS_slfm = slfm.predict_latent(Xg)
mlosA_slfm, vlosA_slfm = slfm.predict_output(X[asc_idx,:], look_a[asc_idx,:])
mlosD_slfm, vlosD_slfm = slfm.predict_output(X[des_idx,:], look_d[des_idx,:])

wall_duration = time.perf_counter() - start_wall
cpu_duration = time.process_time() - start_cpu

print('[status] Sampling complete.')
print(f"\tWall Time: {timedelta(seconds=wall_duration)}")
print(f"\t CPU Time: {timedelta(seconds=cpu_duration)}")

# record number of ascending/descending samples in active set
Nasc = len(asc_idx)
Ndes = len(des_idx)

print(f'[info] Active set size(s):')
print(f'\tTotal = {len(slfm.A_idx)}\n\tGNSS = {Xg.shape[0]}\n\tAsc. = {Nasc}\n\tDes. = {Ndes}\n')

print("[info]Final hyperparameters from optimization:")
print("  var_xyz =", slfm.hp.var_xyz, "\n  ell_xyz =", slfm.hp.ell_xyz)

# ================================== Save ================================== #

os.makedirs(OUT_DIR,
            exist_ok=True)

# generate and save variance/mutual information plot
fig1, axs = slfm.plot_ivm_info()
fig1_n    = 'active_set_selection.png'
fig1.savefig(
    os.path.join(
        OUT_DIR,
        fig1_n
    ),
    dpi=100
)
print(f'[I/O]  {fig1_n} saved.')

## save to .csv files ##

# Ascending active set
pd.DataFrame(
    np.hstack((np.vstack(asc_active)*1e3,
               np.array(asc_idx).reshape(-1,1))
             ),
    columns=["X","Y","indx"]
).to_csv(
    os.path.join(
        OUT_DIR,
        "ascending_active_set.csv"
    ),
    index=False
)

# Descending active set
pd.DataFrame(
    np.hstack((
        np.vstack(des_active)*1e3,
        np.array(des_idx).reshape(-1,1)
        )
    ),
    columns=["X","Y","indx"]
).to_csv(
    os.path.join(
        OUT_DIR,
        "descending_active_set.csv"),
    index=False
)

# Posterior latents
pd.DataFrame(
    np.hstack((
        Xq*1e3,
        mf_slfm,
        vf_slfm)
    ),
    columns=[
        "X","Y",
        "ux_mean","uy_mean","uz_mean",
        "ux_var","uy_var","uz_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_latents_slfm.csv"
    ),
    index=False)

# Posterior ascending LOS
pd.DataFrame(
    np.hstack((
        Xa[asc_idx,:]*1e3,
        mlosA_slfm.reshape(-1,1),
        vlosA_slfm.reshape(-1,1))
    ),
    columns=[
        "X","Y",
        "losA_mean","losA_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_Alos_slfm.csv"
    ),
    index=False
)

# Posterior descending LOS
pd.DataFrame(
    np.hstack((
        Xd[des_idx,:]*1e3,
        mlosD_slfm.reshape(-1,1),
        vlosD_slfm.reshape(-1,1))
    ),
    columns=[
        "X","Y",
        "losD_mean","losD_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_Dlos_slfm.csv"
    ),
    index=False
)

# Posterior GNSS
pd.DataFrame(
    np.hstack((
        Xg*1e3,
        mGNSS_slfm,
        np.sqrt(vGNSS_slfm))
    ),
    columns=[
        "X","Y",
        "ux_mean","uy_mean","uz_mean",
        "ux_var","uy_var","uz_var"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "posterior_GNSS_slfm.csv"
    ),
    index=False
)

csv_files = [
    "ascending_active_set.csv",
    "descending_active_set.csv",
    "posterior_latents_slfm.csv",
    "posterior_Alos_slfm.csv",
    "posterior_Dlos_slfm.csv",
    "posterior_GNSS_slfm.csv"
]

print('\n[I/O]  Saving csv files...')
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
        
        gmt_cmd = f'gmt mapproject {inpath} -Jt120.59/22.94/1:1 -R120/121.1/22.45/23.53 -C -Fe -I -q1: > {outpath}'
        os.system(gmt_cmd)
        print(f'\tProjected file: {outfile} saved.')


# generate optimization path figure and save
fig2, axs = slfm.plot_optimization_paths(label="selection & SoD NLL")
fig2_n    = 'optimization_path.png'
fig2.savefig(
    os.path.join(
        OUT_DIR,
        fig2_n
    ),
    dpi=100
)
print(f'[I/O]  {fig2_n} saved.')

# generate HP history figure and save
fig3, axs = slfm.plot_hyperparam_traces()
fig3_n    = 'HP_history.png'
fig3.savefig(
    os.path.join(
        OUT_DIR,
        fig3_n
    ),
    dpi=100
)
print(f'[I/O]  {fig3_n} saved.')

# get optimal HP values for output                                    
opt_vars = slfm.hp.var_xyz
opt_varx = opt_vars[0]
opt_vary = opt_vars[1]
opt_varz = opt_vars[2]

opt_ells = slfm.hp.ell_xyz
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
     'Nasc',
     'Ndes'
]

values = [
    varx0,
    vary0,
    varz0,
    ellx0,
    elly0,
    ellz0,
    opt_varx,
    opt_vary,
    opt_varz,
    opt_ellx,
    opt_elly,
    opt_ellz,
    Nasc,
    Ndes
]

fname = 'Meinong_slfmivm_results.txt'

with open(os.path.join(DATA_DIR, fname), 'w+') as outfile:  
    for label, value in zip(labels, values):
        outfile.writelines(f'{label}: {value}\n')
outfile.close()
                     

# Raster export 

# Latent field rasters
export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_x_slfm.tif" if EXPORT_FORMAT=="GTiff" else "posterior_x_slfm.nc"
    ),
    Xq*1e3,
    {"ux_mean": mf_slfm[:,0],
     "ux_var":  vf_slfm[:,0]},
    driver=EXPORT_FORMAT
)
export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_y_slfm.tif" if EXPORT_FORMAT=="GTiff" else "posterior_y_slfm.nc"
    ),
    Xq*1e3,
    {"uy_mean": mf_slfm[:,1],
     "uy_var":  vf_slfm[:,1]},
    driver=EXPORT_FORMAT
)
export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_z_slfm.tif" if EXPORT_FORMAT=="GTiff" else "posterior_z_slfm.nc"
        ),
        Xq*1e3,
        {"uz_mean": mf_slfm[:,2],
         "uz_var":  vf_slfm[:,2]},
        driver=EXPORT_FORMAT
)

# LOS rasters: 'losA' -> ALOS; 'losD' -> Sentinel-1
export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_asc_slfm.tif" if EXPORT_FORMAT=="GTiff" else "posterior_asc_slfm.nc"
    ),
    Xq*1e3,
    {"los_mean": mAlos_sample, "los_var": vAlos_sample},
    driver=EXPORT_FORMAT
)
export_rasters(
    os.path.join(
        OUT_DIR,
        "posterior_des_slfm.tif" if EXPORT_FORMAT=="GTiff" else "posterior_des_slfm.nc"
    ),
    Xq*1e3,
    {"los_mean": mDlos_sample, "los_var": vDlos_sample},
    driver=EXPORT_FORMAT
)

raster_files = [
    "posterior_x_slfm.{}",
    "posterior_y_slfm.{}",
    "posterior_z_slfm.{}",
    "posterior_asc_slfm.{}",
    "posterior_des_slfm.{}",
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
        
        gmt_cmd = f'gmt grdproject {inpath} -Jt120.59/22.94/1:1 -Fe -G{outpath} -C -I'
        os.system(gmt_cmd)
        
        print(f'\tProjected file: {outfile} saved')
    
print("[status] Done.")
