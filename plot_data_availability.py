'''
Script to  visualize the number of correlation functions that have passed
the quality control depnending on time.
'''

import os
import numpy as np
from matplotlib import pyplot as plt

from seismic.monitor.dv import read_dv
from seismic.plot.plot_utils import set_mpl_params

# Folder with the dvs
infolder = '/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_QCpass'

set_mpl_params()
for freq in 0.25*2**np.arange(3):
    indir = os.path.join(
        infolder, f'xstations_{freq}-{freq*2}*srw')
    dvs = read_dv(os.path.join(indir, '*.npz'))
    t = np.array([t.datetime for t in dvs[0].stats.corr_start])
    n_avail = np.sum([dv.avail.astype(int) for dv in dvs], axis=0)

    plt.figure(figsize=(12,9))
    plt.fill_between(t, n_avail, color='g')
    plt.ylabel(r'$N_{CF}$')
    plt.ylim((0, None))
    plt.xlim((min(t), max(t)))
    plt.title(f'available CF {freq}-{freq*2} Hz')
    plt.savefig(os.path.join(infolder, f'avail_{freq}.png'), dpi=300, facecolor='None')
    # save numpy array
    np.savez(os.path.join(infolder, f'avail_{freq}.npz'), t=t, n=n_avail)
    plt.close()
