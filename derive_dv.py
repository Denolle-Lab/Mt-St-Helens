import os
import glob
import fnmatch

import numpy as np
from obspy import UTCDateTime

from seismic.monitor.dv import read_dv


# corrupt stations
skip = ['NED', 'SEP']
cut = ['EDM', 'FL2', 'HSR', 'JUN', 'SHW', 'SOS', 'STD']
cutend = UTCDateTime(2014, 1, 1)

files = glob.glob('/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final/*Components*60d*_srw/*.npz')

for infile in files:
    for skipfile in skip:
        if skipfile in infile:
            continue
    dv = read_dv(infile)
    for cutfile in cut:
        if cutfile in infile:
            ij = np.argmin(abs(np.array(dv.stats.starttime) - cutend))
            dv.corr[:ij] += np.nan
            dv.value[:ij] += np.nan
    dv.value = np.hstack((0, np.diff(dv.value)))
    dv.corr[1:] += dv.corr[:-1]
    dv.corr *= .5
    dv.corr[0] = 1
    outdir = os.path.split(os.path.dirname(infile))
    outdir = os.path.join(outdir[0]+'_ddt', outdir[1])
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, os.path.basename(infile))
    dv.save(outfile)