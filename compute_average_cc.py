import glob
import os

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import numpy as np

from seismic.monitor.dv import read_dv


root = '/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_qc/'

infolder = glob.glob(os.path.join(root, 'autoComponents*wl3600*30d_srw'))[0]
infolder2 = glob.glob(os.path.join(root, 'autoComponents*_mrw182d'))[0]

outdir = '/data/wsd01/st_helens_peter/figures/mrw_srw_compare'

os.makedirs(outdir, exist_ok=True)
for infile in glob.glob(os.path.join(infolder, '*.npz')):
    infile_basename = os.path.basename(infile)

    dv_short = read_dv(infile)
    dv_stack = read_dv(os.path.join(infolder2, infile_basename))
    starttimes_short = np.array([cs.datetime for cs in dv_short.stats.corr_start])
    starttimes_stack = np.array([cs.datetime for cs in dv_stack.stats.corr_start])


    plt.figure(figsize=(16, 9))
    plt.scatter(starttimes_stack[dv_stack.avail], -100*dv_stack.value[dv_stack.avail], s=1, c='k', label='mrw', alpha=.6)
    plt.scatter(starttimes_short[dv_short.avail], -100*dv_short.value[dv_short.avail], s=1, c='r', label='srw', alpha=.6)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.title(dv_short.stats.id)
    plt.ylabel(r'\frac{dv}{v} [%]')
    plt.legend()
    plt.savefig(
        os.path.join(outdir, f'{dv_stack.stats.id}_dv.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(16, 9))
    plt.scatter(starttimes_stack, dv_stack.corr, s=1, c='k', label='mrw')
    plt.scatter(starttimes_short, dv_short.corr, s=1, c='r', label='srw')
    plt.hlines(
        np.nanmean(dv_stack.corr), min(np.array(starttimes_short)[dv_short.avail]), max(np.array(starttimes_short)[dv_short.avail]), colors='k')
    plt.hlines(
        np.nanmean(dv_short.corr), min(np.array(starttimes_short)[dv_short.avail]), max(np.array(starttimes_short)[dv_short.avail]), colors='r')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.title(dv_short.stats.id)
    plt.ylabel('correlation coefficient')
    
    plt.legend()
    plt.savefig(
        os.path.join(outdir, f'{dv_stack.stats.id}_cc.png'), dpi=300)
    plt.close()




