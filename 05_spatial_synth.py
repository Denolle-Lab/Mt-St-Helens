'''
Computes the resolution tests as in Makus et al. (2024) (SRL).

:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 26th February 2024 01:59:04 pm
Last Modified: Monday, 26th February 2024 02:04:11 pm
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob

from mpi4py import MPI
from obspy import UTCDateTime
from obspy.geodetics import degrees2kilometers, locations2degrees
import numpy as np

from seismic.monitor.spatial import DVGrid
from seismic.monitor.dv import read_dv


proj_dir = '/your/project/directory/'

# The maps should be like this
# New, slightly larger coverage
lat = [46.05, 46.36]
lon = [-122.45, -122.03]

# Y-extent
y = degrees2kilometers(lat[1]-lat[0])

# X-Extent
x = degrees2kilometers(locations2degrees(lat[0], lon[0], lat[0], lon[1]))

# Resolution / grid spacing of the model
res = 1  # km

# Time-series
delta = 182.5*24*3600
start = UTCDateTime(year=2007, julday=1).timestamp
end = UTCDateTime(year=2023, julday=240).timestamp
times = np.arange(start, end, delta)

# inversion parameters
# geo-parameters
vel = 2.5  # km/s, Ulberg et al. (2020)
mf_path = vel/(2*np.pi*0.0014*3)
dt = .05 # s  # for the numerical integration

corrupt = ['EDM', 'FL2', 'HSR', 'JUN', 'SHW', 'STD', 'SUG', 'YEL']

corrupt = [
    f'UW-UW.{c}-{c}.EHZ-EHZ' for c in corrupt] + ['CC.CC.SUG-SUG.EHZ-EHZ']



def generate_dvs(indir):
    dvs = glob.glob(os.path.join(indir, '*.npz'))
    for dv in dvs:
        yield read_dv(dv)


comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

std_models = [4e-3]
ns = np.arange(3)
std_models, ns = np.meshgrid(std_models, ns)
std_models = std_models.flatten()
ns = ns.flatten()

# for ii, (std_model, n) in enumerate(zip(std_models, ns)):
for n, std_model in zip(ns, std_models):
    corr_len = 2
    freq0 = 0.25*2**n

    # you have to enter the path containing one of the parameters
    # fine, coarse, etc (see below). It will choose a model based on that
    outdir = os.path.join(
        proj_dir,
        f'/synthetic_test_fine_dvfilt_allcc_cl{corr_len}_std{std_model}_largemap',
        f'{freq0}-{freq0*2}')
    os.makedirs(outdir, exist_ok=True)

    # long dvs
    infiles = os.path.join(
        proj_dir,
        f'/dv/*_{freq0}-{freq0*2}_wl432000_*_srw/*.npz')

    if not len(glob.glob(infiles)):
        raise FileNotFoundError(f'{infiles} contains no files')

    # separately for clock shift
    infiles2 = os.path.join(
        proj_dir,
        'dv/dv_separately_ddt/xstations_{freq0}-{freq0*2}_*/*.npz')
    if not len(glob.glob(infiles2)):
        raise FileNotFoundError(f'{infiles2} contains no files')

    dvs_all = read_dv(infiles)
    dvs_all += read_dv(infiles2)

    # create grid
    dvg = DVGrid(lat[0], lon[0], res, x, y, dt, vel, mf_path)

    # synthetic model
    chkb = np.zeros_like(dvg.xgrid)
    for ii, yy in enumerate(np.arange(y+1)):
        chkb[ii, :] = np.sin(
            4*np.pi*np.arange(x/1+1)/(x/1)+2.5) + np.cos(4*np.pi*yy/(y))
    if 'fine' in outdir:
        chkb = np.zeros_like(dvg.xgrid)
        for ii, yy in enumerate(np.arange(y+1)):
            chkb[ii, :] = np.sin(
                8*np.pi*np.arange(x/1+1)/(x/1)) + np.cos(8*np.pi*yy/(y)+1.5)
    elif 'coarse' in outdir:
        chkb = np.zeros_like(dvg.xgrid)
        for ii, yy in enumerate(np.arange(y+1)):
            chkb[ii, :] = np.sin(
                2*np.pi*np.arange(x/1+1)/(x/1)+4.5) + np.cos(2*np.pi*yy/(y))
    elif 'const' in outdir:
        chkb = np.ones_like(dvg.xgrid)
    elif 'vert' in outdir:
        chkb = np.ones_like(dvg.xgrid)
        chkb[:chkb.shape[0]//2] *= -1
    elif 'hor' in outdir:
        chkb = np.ones_like(dvg.xgrid)
        chkb[:, :chkb.shape[-1]//2] *= -1

    chkb /= 100

    grids = np.zeros(
        (len(dvg.yaxis), len(dvg.xaxis), len(times)), dtype=np.float64)
    grid_res = np.zeros_like(grids)

    pmap = (np.arange(len(times))*psize)/len(times)
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    ind = np.arange(len(times), dtype=int)[ind]

    # Compute
    sing_counter = 0
    for ii, utc in zip(ind, times[ind]):
        # print(f'working on {UTCDateTime(utc)}.')
        utc = UTCDateTime(utc)
        # Find available dvs at this time
        dvs = []
        ti = []
        removed = 0
        for dv in dvs_all:
            if utc < dv.stats.corr_start[0] or utc > dv.stats.corr_end[-1]:
                # this dv is not inside of time series
                continue
            if dv.stats.id in corrupt:
                removed += 1
                continue
            tii = np.argmin(abs(np.array(dv.stats.starttime) - utc))
            # Consider adding a corr threshold here? Discard everything below .5?
            if dv.avail[tii]:
                ti.append(tii)
                dvs.append(dv)
        # print(f'removed {removed} corrupt dvs')

        try:
            fwd_model = dvg.forward_model(
                chkb, dvs=dvs, utc=utc)
            # assign forward values
            for dv, fwd_val in zip(dvs, fwd_model):
                tii = np.argmin(abs(np.array(dv.stats.starttime) - utc))
                dv.value[tii] = fwd_val
            dvg.compute_dv_grid(dvs, utc, res, corr_len, std_model, compute_resolution=True)
        except IndexError as e:
            print(e)
            grids[:, :, ii] += np.nan
            grid_res[:, :, ii] += np.nan
            continue
        except np.linalg.LinAlgError as e:
            print(e)
            grids[:, :, ii] += np.nan
            grid_res[:, :, ii] += np.nan
            sing_counter += 1
            continue

        grids[:, :, ii] = dvg.vel_change
        grid_res[:, :, ii] = dvg.resolution

    statx = dvg.statx
    staty = dvg.staty

    comm.Allreduce(MPI.IN_PLACE, [grids, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, [grid_res, MPI.DOUBLE], op=MPI.SUM)

    array_size = statx.size
    array_size = comm.allreduce(array_size, op=MPI.MAX)
    Statx = np.zeros((psize, array_size))
    Statx[rank, :len(statx)] = statx
    Staty = np.zeros((psize, array_size))
    Staty[rank, :len(statx)] = staty

    comm.Allreduce(MPI.IN_PLACE, [Statx, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, [Staty, MPI.DOUBLE], op=MPI.SUM)
    Staty = np.concatenate(Staty)
    Statx = np.concatenate(Statx)
    jj = Statx == 0
    Staty = Staty[~jj]
    Statx = Statx[~jj]
    stats = np.vstack((Statx, Staty))
    stats = np.unique(stats, axis=1)
    Statx = stats[0]
    Staty = stats[1]

    counter = comm.reduce(sing_counter, op=MPI.SUM, root=0)
    if rank == 0:
        print('Saving files')
        np.savez(
            os.path.join(outdir, f'dvdt_3D.npz'), dv=grids, xaxis=dvg.xaxis,
            yaxis=dvg.yaxis, taxis=times, statx=Statx, staty=Staty,
            resolution=grid_res)
