"""Computes the Lcurves in Makus et al. (2024) (SRL)."""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from mpi4py import MPI
from obspy import UTCDateTime
from obspy.geodetics import degrees2kilometers, locations2degrees
import numpy as np

from seismic.monitor.spatial import DVGrid, data_variance
from seismic.monitor.dv import read_dv


corrupt = ['EDM', 'FL2', 'HSR', 'JUN', 'SHW', 'STD', 'SUG', 'YEL']

# Drop the auto correlations of anolog telemetered data
corrupt = [
    f'UW-UW.{c}-{c}.EHZ-EHZ' for c in corrupt] + ['CC.CC.SUG-SUG.EHZ-EHZ']


for freq0 in 0.25*2**np.arange(3):

    # long dvs
    infiles = f'dv/new_gap_handling/*_{freq0}-{freq0*2}_wl432000_*_srw/*.npz'

    # separately for clock shift
    infiles2 = f'/dv/dv_separately/xstations_{freq0}-{freq0*2}_*/*.npz'

    dvs_all = read_dv(infiles)
    dvs_all += read_dv(infiles2)
    dvs_all = [dv for dv in dvs_all if dv.stats.id not in corrupt]

    # The maps should be like this
    # old
    # New, slightly larger coverage
    lat = [46.05, 46.36]
    lon = [-122.45, -122.03]

    # Y-extent
    y = degrees2kilometers(lat[1] - lat[0])

    # X-Extent
    x = degrees2kilometers(locations2degrees(lat[0], lon[0], lat[0], lon[1]))

    # Resolution
    res = 1  # km

    # Time-series
    delta = (365.25/4)*24*3600
    start = UTCDateTime(year=2007, julday=1).timestamp
    end = UTCDateTime(year=2023, julday=240).timestamp
    times = np.arange(start, end, delta)

    # inversion parameters
    # geo-parameters
    vel = 2.5  # km/s, Wang et al (2019)
    # According to Gabrielli et al. (2020) Q_S^-1 = 0.0014, for 3 Hz
    mf_path = vel/(2*np.pi*0.0014*3)
    dt = .05  # s  # for the numerical integration

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
    corr_lens = np.arange(1, 6, 1)
    stds = [5e-4*(2**n) for n in range(9)]
    corrl_grid, stdg = np.meshgrid(corr_lens, stds)
    corrl_grid = corrl_grid.flatten()
    stdg = stdg.flatten()
    pmap = (np.arange(len(stdg))*psize)/len(stdg)
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    ind = np.arange(len(stdg), dtype=int)[ind]

    residuals = np.zeros((len(corrl_grid), len(times)))
    model_variances = np.zeros_like(residuals)
    vel_change = np.zeros((len(residuals), *dvg.xgrid.shape, len(times)))
    rms = np.zeros_like(residuals)

    for corr_len, std_model in zip(corrl_grid[ind], stdg[ind]):
        outdir = 'spatial/lcurve/'
        os.makedirs(outdir, exist_ok=True)
        ii = np.where((corrl_grid == corr_len) & (stdg == std_model))[0][0]
        for utc in times:
            print(f'working on {UTCDateTime(utc)}.')
            jj = np.where(times == utc)[0][0]
            utc = UTCDateTime(utc)
            # Find available dvs at this time
            dvg.align_dvs_to_grid(
                dvs_all, utc, 5, 0.4, outdir)
            # aligned in a previous step + newly aligned
            dvs = [
                dv for dv in dvs_all if dv.dv_processing.get('aligned', False)
                is not False]
            dvs_filt = []
            ti = []
            for dv in dvs:
                if dv.stats.corr_start[0] > utc or dv.stats.corr_end[-1] < utc:
                    continue
                tii = np.argmin(abs(np.array(dv.stats.corr_start)-utc))
                if np.isnan(dv.corr[tii]) or dv.corr[tii] <= 1e-3 or dv.corr[tii] >= 1:
                    continue
                ti.append(tii)
                dvs_filt.append(dv)
            try:
                inv = dvg.compute_dv_grid(
                    dvs_filt, utc, res, corr_len, std_model)
            except IndexError as e:
                print(e)
                continue
            except np.linalg.LinAlgError as e:
                print(e)
            except Exception as e:
                print(
                    f'Could not invert for grid time {utc}.\n',
                    f'Original error was {e}')
                continue
            vel_change[ii, ..., jj] = inv
            real = np.array([dv.value[tii] for dv, tii in zip(dvs_filt, ti)])
            corr = np.array([dv.corr[tii] for dv, tii in zip(dvs_filt, ti)])
            pred = dvg.forward_model(inv, dvs=dvs_filt, utc=utc)

            pp = dvs[0].dv_processing
            sigma_d = data_variance(
                corr, pp['freq_max'] - pp['freq_min'],
                (pp['tw_start'], pp['tw_len'] + pp['tw_start']),
                (pp['freq_max'] + pp['freq_min'])/2)
            residuals[ii, jj] = np.sqrt(np.mean((real - pred)**2/sigma_d**2))

            rms[ii, jj] = np.sqrt(np.nanmean(inv**2))

            print(f'{UTCDateTime(utc)} done.')
        print(f'Finished computation for std {std_model} and clen {corr_len}')
    # gather results
    comm.Allreduce(MPI.IN_PLACE, [residuals, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, [vel_change, MPI.DOUBLE], op=MPI.SUM)

    if rank == 0:
        np.savez(
            os.path.join(outdir, f'Lcurve_{freq0}.npz'),
            residual=residuals,
            vel_change=vel_change,
            times=times, corr_len=np.tile(corr_lens, len(stds)),
            std_model=np.hstack([[std]*len(corr_lens) for std in stds]),
            rms=rms)
