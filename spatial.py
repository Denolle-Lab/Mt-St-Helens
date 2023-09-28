import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
from copy import deepcopy

from mpi4py import MPI
from obspy import UTCDateTime
from obspy.geodetics import degrees2kilometers, locations2degrees
import numpy as np
from matplotlib import pyplot as plt

from seismic.monitor.spatial import DVGrid
from seismic.monitor.dv import read_dv




# The maps should be like this
# old
lat = [46.09, 46.3]
lon = [-122.34, -122.1]
# New, slightly larger coverage
lat = [46.05, 46.36]
lon = [-122.45, -122.03]


# Y-extent
y = degrees2kilometers(lat[1]- lat[0])

# X-Extent
x = degrees2kilometers(locations2degrees(lat[0], lon[0], lat[0], lon[1]))

# Resolution
res = 1  # km

# Time-series
delta = 10*24*3600
start = UTCDateTime(year=1997, julday=175).timestamp
end = UTCDateTime(year=2023, julday=81).timestamp
times = np.arange(start, end, delta)

# inversion parameters
# geo-parameters
vel = 2  # km/s
# According to Gabrielli et al. (2020) Q_S^-1 = 0.0014, for 3 Hz, mfp about 38 km
#  Q_s = 2*pi*f*mf_path/v , mf_path = Q_s*v/(2*pi*f)
mf_path = vel/(2*np.pi*0.0014*3)
dt = .05 # s  # for the numerical integration

# needs to be thoroughly tested
corr_len = 2  # km; just a try
std_model = .064  # 3.2e-2



def generate_dvs(indir):
    dvs = glob.glob(os.path.join(indir, '*.npz'))
    for dv in dvs:
        yield read_dv(dv)


comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

for n in range(3):
    freq0 = 0.25*2**n

    indir = glob.glob(f'/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_QCpass_ddt/xstations_{freq0}-{freq0*2}*')

    # add auto and xcomp
    indir2 = glob.glob(f'/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_ddt/autoComponents_{freq0}-{freq0*2}*')
    indir3 = glob.glob(f'/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_ddt/betweenComponents_{freq0}-{freq0*2}*')
    if len(indir) > 1 or len(indir2) > 1 or len(indir3) > 1:
        raise ValueError('ambiguous directory')
    indir = indir[0]
    indir2 = indir2[0]
    indir3 = indir3[0]

    dvs_all = read_dv(os.path.join(indir, '*.npz'))
    dvs_all.extend(read_dv(os.path.join(indir2, '*.npz')))
    dvs_all.extend(read_dv(os.path.join(indir3, '*.npz')))

    # create grid
    dvgo = DVGrid(lat[0], lon[0], res, x, y, dt, vel, mf_path)

    grids = np.zeros(
        (len(dvgo.yaxis), len(dvgo.xaxis), len(times)), dtype=np.float64)



    pmap = (np.arange(len(times))*psize)/len(times)
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    ind = np.arange(len(times), dtype=int)[ind]


    outdir = os.path.join(
        f'/data/wsd01/st_helens_peter/spatial/ddt_crosssingle_cl{corr_len}_std{std_model}_largemap',
        f'{freq0}-{freq0*2}')
    os.makedirs(outdir, exist_ok=True)

    # Compute
    sing_counter = 0
    for ii, utc in zip(ind, times[ind]):
        dvg = deepcopy(dvgo)
        utc = UTCDateTime(utc)
        try:
            dvg.compute_dv_grid(dvs_all, utc, res, corr_len, std_model)
        except IndexError as e:
            print(e)
            grids[:, :, ii] += np.nan
            continue
        except np.linalg.LinAlgError as e:
            print(e)
            grids[:, :, ii] += np.nan
            sing_counter += 1
            continue

        # save raw data for joint plot and L-curve analysis
        np.savez(os.path.join(outdir, f'{utc.year}_{utc.julday}.npz'), dv=dvg.vel_change, xaxis=dvg.xaxis, yaxis=dvg.yaxis, statx=dvg.statx, staty=dvg.staty)
        grids[:, :, ii] = dvg.vel_change


    comm.Allreduce(MPI.IN_PLACE, [grids, MPI.DOUBLE], op=MPI.SUM)
    counter = comm.reduce(sing_counter, op=MPI.SUM, root=0)

    if rank == 0:
        np.savez(
            os.path.join(outdir, f'dvdt_3D.npz'), dv=grids, xaxis=dvgo.xaxis,
            yaxis=dvgo.yaxis, taxis=times, statx=dvg.statx, staty=dvg.staty)
        print(f' Number of Singular Matrices encountered {sing_counter}.')

    # del grids