import glob
import os
import fnmatch

import numpy as np
from obspy import UTCDateTime
from scipy.stats import linregress
from scipy.interpolate import interp1d

from seismic.monitor.dv import read_dv
from seismic.monitor.monitor import average_components

infolder = glob.glob('/data/wsd01/st_helens_peter/dv/resp_removed/xstations_1.0-2.0_wl864000_*presmooth45d_srw')[0]
outfolder = '/data/wsd01/st_helens_peter/dv/shift_stack_2007_45dsmooth_trend/1.0-2.0'
zerotime = UTCDateTime(2007, 8, 15)
shifting_method = 'trend'  # trend or mean



def shift_zero_time(dv, t0):
    """
    This function shift the cruve so that dv/v(t0)=0
    depending on the average values
    around the requested time.
    """
    starts = np.array(dv.stats.starttime)
    jj = np.argmin(abs(starts[dv.avail]-t0))
    t = starts[dv.avail][jj]
    ii = np.where(starts == t)[0][0]
    if abs(t-t0) > (starts[1]-starts[0])*50:
        raise ValueError(f'station {dv.stats.id} not active during time')
    shift = np.nanmean(
            dv.value[ii-25:ii+25]*dv.corr[ii-25:ii+25])/np.nanmean(
                dv.corr[ii-25:ii+25])
    # shift = np.nanmedian(dv.value[ii-25:ii+25])
    roll = int(round(-shift/(dv.second_axis[1]-dv.second_axis[0])))
    dv.sim_mat = np.roll(dv.sim_mat, (roll, 0))
    dv.value = dv.second_axis[
        np.nanargmax(np.nan_to_num(dv.sim_mat), axis=1)]
    return dv

def shift_zero_time_trend(dv, t0):
    """
    This function shift the cruve so that dv/v(t0)=0
    by first computing a trend of the dv/v scatter values and shifting by the
    amount the trend is off at t0
    """
    ma = dv.corr[dv.avail] > 0.35  # mask
    t = np.array([t.datetime for t in dv.stats.starttime])[dv.avail][ma]
    if len(t) < 35:
        # less than a year:
        raise ValueError(
            f'Values are to unstable for station {dv.stats.id}.'
        )
    x = np.array([t.timestamp for t in dv.stats.starttime])[dv.avail][ma]
    r = linregress(
        x, dv.value[dv.avail][ma])
    y = r.intercept + r.slope*x
    xq = np.array([t.timestamp for t in dv.stats.starttime])
    # Interpolate onto the whole grid
    f = interp1d(x, y, fill_value='extrapolate')
    yq = f(xq)
    tq = np.array(dv.stats.starttime)

    # starts = np.array(dv.stats.starttime)
    # jj = np.argmin(abs(starts[dv.avail]-t0))
    # t = starts[dv.avail][jj]
    # ii = np.where(starts == t)[0][0]
    # if abs(t-t0) > (starts[1]-starts[0])*50:
    #     raise ValueError(f'station {dv.stats.id} not active during time')
    ii = np.argmin(abs(tq-t0))
    shift = yq[ii]

    roll = int(round(-shift/(dv.second_axis[1]-dv.second_axis[0])))
    dv.sim_mat = np.roll(dv.sim_mat, (roll, 0))
    dv.value = dv.second_axis[
        np.nanargmax(np.nan_to_num(dv.sim_mat), axis=1)]
    return dv

files = glob.glob(os.path.join(infolder, '*.npz'))
os.makedirs(outfolder, exist_ok=True)
off = 0
while len(files):
    dv = read_dv(files[0])
    # We name the files after coordinates
    outfile = os.path.join(
        outfolder,
        f'{dv.stats.evla}.{dv.stats.evlo}-{dv.stats.stla}.{dv.stats.stlo}')
    outfile2 = os.path.join(
        outfolder,
        f'{dv.stats.evla}.{dv.stats.evlo}-{dv.stats.stla}.{dv.stats.stlo}')
    if os.path.isfile(outfile) or os.path.isfile(outfile2):
        continue
    thisfiles = glob.glob(os.path.join(
        infolder, f'DV-{dv.stats.network}.{dv.stats.station}.*.npz'))
    if fnmatch.fnmatch(dv.stats.station, '*-SUG'):
        net = dv.stats.network.split('-')[0]
        stat = dv.stats.network.split('-')[0]
        if fnmatch.fnmatch(dv.stats.network, '*-CC'):
            # Look for similar from UW
            thisfiles.extend(glob.glob(f'DV-UW-{net}.SUG-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-UW.{stat}-SUG.*.npz'))
        else:
            thisfiles.extend(glob.glob(f'DV-CC-{net}.SUG-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-CC.{stat}-SUG.*.npz'))
    elif fnmatch.fnmatch(dv.stats.station, 'SUG-*'):
        net = dv.stats.network.split('-')[1]
        stat = dv.stats.network.split('-')[1]
        if fnmatch.fnmatch(dv.stats.network, 'CC-*'):
            # Look for similar from UW
            thisfiles.extend(glob.glob(f'DV-UW-{net}.SUG-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-UW.{stat}-SUG.*.npz'))
        else:
            thisfiles.extend(glob.glob(f'DV-CC-{net}.SUG-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-CC.{stat}-SUG.*.npz'))
    if fnmatch.fnmatch(dv.stats.station, '*-STD'):
        net = dv.stats.network.split('-')[0]
        stat = dv.stats.network.split('-')[0]
        if fnmatch.fnmatch(dv.stats.network, '*-CC'):
            # Look for similar from UW
            thisfiles.extend(glob.glob(f'DV-UW-{net}.STD-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-UW.{stat}-STD.*.npz'))
        else:
            thisfiles.extend(glob.glob(f'DV-CC-{net}.STD-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-CC.{stat}-STD.*.npz'))
    elif fnmatch.fnmatch(dv.stats.station, 'STD-*'):
        net = dv.stats.network.split('-')[1]
        stat = dv.stats.network.split('-')[1]
        if fnmatch.fnmatch(dv.stats.network, 'CC-*'):
            # Look for similar from UW
            thisfiles.extend(glob.glob(f'DV-UW-{net}.STD-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-UW.{stat}-STD.*.npz'))
        else:
            thisfiles.extend(glob.glob(f'DV-CC-{net}.STD-{stat}.*.npz'))
            thisfiles.extend(glob.glob(f'DV-{net}-CC.{stat}-STD.*.npz'))
    dvs = []
    for f in thisfiles:
        dvn = read_dv(f)
        if not np.any(dv.avail):
            print(f'{dv.id} does not have any data points...skipping')
            continu
        dvs.append(dvn)
        files.remove(f)

    if len(dvs) == 1:
        dv_stack = dvs[0]
    else:
        dv_stack = average_components(
            dvs, save_scatter=False, correct_shift=True,
            correct_shift_method='mean', correct_shift_overlap=6)
    try:
        if shifting_method == 'mean':
            dv_stack = shift_zero_time(dv_stack, zerotime)
        elif shifting_method == 'trend':
            dv_stack = shift_zero_time_trend(dv_stack, zerotime)
        else:
            raise ValueError(f'Shifting method {shifting_method} unknown.')
    except ValueError as e:
        print(e)
        off += 1
        continue
    dv_stack.save(outfile)
print(f'{off} stations could not be used for zero time {zerotime}.')