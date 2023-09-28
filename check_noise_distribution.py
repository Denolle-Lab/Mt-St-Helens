import os
import glob

from matplotlib import pyplot as plt
from obspy import UTCDateTime
import numpy as np

from seismic.monitor.dv import read_dv
from seismic.plot.plot_utils import set_mpl_params


infiles = '/data/wsd01/st_helens_peter/dv/resp_actually_removed_longtw_final_qc/autoComponents_1.0-2.0_wl3600_tw4.0-40.0__1b_mute_SW_pretmoothnosmoothd_srw/DV-*.npz'
outfolder = '/data/wsd01/st_helens_peter/figures/noise_distribution'


set_mpl_params()
os.makedirs(outfolder, exist_ok=True)

for infile in glob.glob(infiles):
    dv = read_dv(infile)
    corr_starts = np.array([UTCDateTime(cs) for cs in dv.stats.corr_start])

    strange_noise = corr_starts[dv.corr > 0.6]

    strange_noise_time = [sn.hour for sn in strange_noise]
    
    dv_corr_hour = np.zeros(24)
    dv_corr_hour_count = np.zeros(24)

    corr_start_hour = np.array([cs.hour for cs in corr_starts])

    for i in range(24):
        dv_corr_hour[i] = np.nanmean(dv.corr[np.where(corr_start_hour == i)])
        dv_corr_hour_count[i] = len(np.where(~np.isnan((dv.corr[corr_start_hour == i])))[0])
    
    plt.figure(figsize=(15, 12))
    plt.subplot(221)
    plt.hist(strange_noise_time, bins=24)
    plt.title('Number of dv/v estimates with corr > 0.6 per hour')
    plt.xlabel('Hour (UTC)')
    plt.ylabel('N')
    plt.subplot(222)
    plt.bar(np.arange(24), dv_corr_hour)
    plt.title('Average dv.corr per hour')
    plt.xlabel('Hour (UTC)')
    plt.ylabel('average correlation coefficient')

    plt.subplot(223)
    plt.bar(np.arange(24), dv_corr_hour_count)
    plt.title('Number of dv/v estimates per hour')
    plt.xlabel('Hour (UTC)')
    plt.ylabel('N')
    plt.savefig(os.path.join(outfolder, f'{dv.stats.network}.{dv.stats.station}.{dv.stats.channel}.png'), dpi=300, bbox_inches='tight')
    plt.close()

