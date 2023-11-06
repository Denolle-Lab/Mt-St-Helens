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
start = UTCDateTime(year=1997, julday=1).timestamp
end = UTCDateTime(year=2023, julday=240).timestamp
times = np.arange(start, end, delta)

# inversion parameters
# geo-parameters
vel = 2  # km/s
# According to Gabrielli et al. (2020) Q_S^-1 = 0.0014, for 3 Hz, mfp about 38 km
#  Q_s = 2*pi*f*mf_path/v , mf_path = Q_s*v/(2*pi*f)
mf_path = vel/(2*np.pi*0.0014*3)
dt = .05 # s  # for the numerical integration

# needs to be thoroughly tested
# from lcurve criterion
corr_len = 1  # km; just a try
std_model = .032  # 3.2e-2



def generate_dvs(indir):
    dvs = glob.glob(os.path.join(indir, '*.npz'))
    for dv in dvs:
        yield read_dv(dv)


comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

for n in range(3):
    freq0 = 0.25*2**n

    # long dvs
    infiles = f'/data/wsd01/st_helens_peter/dv/new_gap_handling_ddt/*_{freq0}-{freq0*2}_wl432000_*_srw/*.npz'
    if not len(glob.glob(infiles)):
        raise FileNotFoundError(f'{infiles} contains no files')

    # separately for clock shift
    infiles2 = f'/data/wsd01/st_helens_peter/dv/dv_separately_ddt/xstations_{freq0}-{freq0*2}_*/*.npz'
    if not len(glob.glob(infiles2)):
        raise FileNotFoundError(f'{infiles2} contains no files')

    dvs_all = read_dv(infiles)
    dvs_all += read_dv(infiles2)

    # create grid
    dvg = DVGrid(lat[0], lon[0], res, x, y, dt, vel, mf_path)

    grids = np.zeros(
        (len(dvg.yaxis), len(dvg.xaxis), len(times)), dtype=np.float64)



    pmap = (np.arange(len(times))*psize)/len(times)
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    ind = np.arange(len(times), dtype=int)[ind]


    outdir = os.path.join(
        f'/data/wsd01/st_helens_peter/spatial/new_gap_handling_ddt_crosssingle_cl{corr_len}_std{std_model}_largemap',
        f'{freq0}-{freq0*2}')
    os.makedirs(outdir, exist_ok=True)

    # Compute
    sing_counter = 0
    for ii, utc in zip(ind, times[ind]):
        print(f'working on {UTCDateTime(utc)}.')
        utc = UTCDateTime(utc)
        # Find available dvs at this time
        dvs = []
        ti = []
        for dv in dvs_all:
            if utc < dv.stats.corr_start[0] or utc > dv.stats.corr_end[-1]:
                # this dv is not inside of time series
                continue
            tii = np.argmin(abs(np.array(dv.stats.starttime) - utc))
            if dv.avail[tii]:
                ti.append(tii)
                dvs.append(dv)
        try:
            dvg.compute_dv_grid(dvs, utc, res, corr_len, std_model)
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
            os.path.join(outdir, f'dvdt_3D.npz'), dv=grids, xaxis=dvg.xaxis,
            yaxis=dvg.yaxis, taxis=times, statx=dvg.statx, staty=dvg.staty)
        print(f' Number of Singular Matrices encountered {sing_counter}.')

    # del grids