# average all dvs for pore pressure test
import os
import glob

import numpy as np

from seismic.monitor.dv import read_dv
from seismic.monitor.monitor import average_components

indir = '/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_QCpass'

freq = 0.25*2**(np.arange(3))

for f in freq:
    dir = glob.glob(os.path.join(indir, f'xstations_{f}-{f*2}*_srw'))[0]
    dvs = read_dv(os.path.join(dir, 'DV-*-*.*-*.*-*.npz'))
    dv_av = average_components(dvs, save_scatter=False)
    dv_av.save(os.path.join(dir, 'average.npz'))
