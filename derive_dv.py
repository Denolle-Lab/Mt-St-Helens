import os
import glob

import numpy as np

from seismic.monitor.dv import read_dv

files = glob.glob('/data/wsd01/st_helens_peter/dv/resp_removed/xstations*45d_srw/*.npz')

for infile in files:
    dv = read_dv(infile)
    dv.value = np.hstack((0, np.diff(dv.value)))
    dv.corr[1:] += dv.corr[:-1]
    dv.corr *= .5
    dv.corr[0] = 1
    outdir = os.path.split(os.path.dirname(infile))
    outdir = os.path.join(outdir[0]+'_ddt', outdir[1])
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, os.path.basename(infile))
    dv.save(outfile)