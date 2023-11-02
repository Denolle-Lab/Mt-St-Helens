import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
from copy import deepcopy

from mpi4py import MPI
from obspy import UTCDateTime
from obspy.geodetics import degrees2kilometers, locations2degrees
import numpy as np
from matplotlib import pyplot as plt

from seismic.monitor.spatial import DVGrid, data_variance
from seismic.monitor.dv import read_dv

freq0 = 0.25


for freq0 in 0.25*2**np.arange(3):

    # long dvs
    infiles = glob.glob(f'/data/wsd01/st_helens_peter/dv/new_gap_handling_ddt/*_{freq0}-{freq0*2}_wl432000_*_srw/*.npz')

    # separately for clock shift
    infiles += glob.glob(f'/data/wsd01/st_helens_peter/dv/dv_separately_ddt/xstations_{freq0}-{freq0*2}_*/*.npz')

    dvs_all = read_dv(infiles)

    # The maps should be like this
    # old
    # New, slightly larger coverage
    lat = [46.05, 46.36]
    lon = [-122.45, -122.03]

    # Y-extent
    y = degrees2kilometers(lat[1]- lat[0])

    # X-Extent
    x = degrees2kilometers(locations2degrees(lat[0], lon[0], lat[0], lon[1]))

    # Resolution
    res = 1 # km

    # Time-series
    delta = (365.25/2)*24*3600
    start = UTCDateTime(year=1997, julday=1).timestamp
    end = UTCDateTime(year=2023, julday=280).timestamp
    times = np.arange(start, end, delta)

    # inversion parameters
    # geo-parameters
    vel = 2  # km/s
    # According to Gabrielli et al. (2020) Q_S^-1 = 0.0014, for 3 Hz, mfp about 38 km
    #  Q_s = 2*pi*f*mf_path/v , mf_path = Q_s*v/(2*pi*f)
    mf_path = vel/(2*np.pi*0.0014*3)
    dt = .05 # s  # for the numerical integration

    # create grid
    dvg = DVGrid(lat[0], lon[0], res, x, y, dt, vel, mf_path)

    comm = MPI.COMM_WORLD
    psize = comm.Get_size()
    rank = comm.Get_rank()

    pmap = (np.arange(len(times))*psize)/len(times)
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    ind = np.arange(len(times), dtype=int)[ind]


    # Computations for L-curve criterion, bi-yearly should be more than sufficient
    corr_lens = np.hstack((0.5, np.arange(1, 6, 1)))
    stds = [2e-3*(4**n) for n in range(4)]
    stds = np.hstack((stds, 0.064))


    residuals = np.zeros((len(stds)*len(corr_lens), len(times)))
    model_variances = np.zeros_like(residuals)

    ii = 0
    for std_model in stds:
        for corr_len in corr_lens:
            outdir = os.path.join(
                '/data/wsd01/st_helens_peter/spatial/new_gap_handling_tdependent_lcurve_singlecross/')
            os.makedirs(outdir, exist_ok=True)

            for utc in times[ind]:
                jj = np.where(times==utc)[0][0]
                utc = UTCDateTime(utc)
                ti = np.argmin(abs(np.array(dvs_all[0].stats.starttime) - utc))
                # Find available dvs at this time
                dvs = [dv for dv in dvs_all if dv.avail[ti]]
                try:
                    inv = dvg.compute_dv_grid(
                        dvs, utc, res, corr_len, std_model)
                except IndexError as e:
                    print(e)
                    continue
                except np.linalg.LinAlgError as e:
                    print(e)

                real = np.array([dv.value[ti] for dv in dvs])
                corr = np.array([dv.corr[ti] for dv in dvs])
                pred = dvg.forward_model(inv, dvs=dvs, utc=utc)

                pp = dvs[0].dv_processing
                sigma_d = data_variance(
                    corr, pp['freq_max'] - pp['freq_min'],
                    (pp['tw_start'], pp['tw_len'] + pp['tw_start']),
                    (pp['freq_max'] + pp['freq_min'])/2)
                residuals[ii, jj] = np.sqrt(np.mean((real - pred)**2/sigma_d**2))
                model_variances[ii, jj] = np.mean((inv-np.mean(inv))**2)
            ii += 1
    # gather results
    comm.Allreduce(MPI.IN_PLACE, [residuals, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, [model_variances, MPI.DOUBLE], op=MPI.SUM)


    if rank == 0:
        np.savez(
            os.path.join(outdir, f'Lcurve_{freq0}.npz'),
            model_variances=model_variances, residual=residuals,
            times=times, corr_len=np.tile(corr_lens, len(stds)),
            std_model=np.hstack([[std]*len(corr_lens) for std in stds]))
