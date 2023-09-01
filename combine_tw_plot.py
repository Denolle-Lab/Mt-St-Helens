from glob import glob
from copy import deepcopy
import os

import numpy as np
from matplotlib import pyplot as plt

from seismic.monitor.dv import read_dv
from seismic.monitor.monitor import average_components
from seismic.plot.plot_utils import set_mpl_params


dir = '/data/wsd01/st_helens_peter/dv/resp_removed_multitw'

for ii in range(3):
    f = 1/(2**ii)
    dv_files = glob(os.path.join(dir, f'xstations_{f}-{f*2}*srw/DV-*.npz'))
    for dvf in dv_files:
        net, stat, cha, _ = os.path.basename(dvf).split('.')
        net = net[3:]
        outfile = os.path.join(dir, f'figs/{f}-{f*2}/dv_compare_{net}.{stat}.{cha}.png')
        outfile2 = os.path.join(dir, f'figs/{f}-{f*2}/dv_compare_{net}.{stat}.{cha}_only_sums.png')
        if os.path.isfile(outfile) and os.path.isfile(outfile2):
            continue
        dvs = read_dv(os.path.join(dir, f'xstations_{f}-{f*2}*srw/DV-{net}.{stat}.{cha}.npz'))

        set_mpl_params()

        vals = []
        # Find the Fit towards the stack, mean, and median
        # Find average correlation coefficient

        tw_starts = []
        vals = []
        corrs = []

        for ii, dv in enumerate(dvs):
            if dv.dv_processing['tw_len'] > 30:
                dv_total_calc_i = ii
                continue
            tw_starts.append(dv.dv_processing['tw_start'])
            vals.append(dv.value)
            corrs.append(dv.corr)
        dv_total_calc = dvs.pop(dv_total_calc_i)
        mean_val = np.nanmean(vals, axis=0)
        dv_sim_mat_stack = average_components(dvs)
        

        twl = dvs[0].dv_processing['tw_len']
        scale = twl/7

        plt.figure(figsize=(16, 10))

        for val, twe, dv in zip(vals, tw_starts, dvs):
            plt.scatter([t.datetime for t in dv.stats.corr_start], -100*scale*np.ma.masked_invalid(val)+twe+twl, c=dv.corr, s=.5, cmap='inferno_r')
            #normalise sim_mat
            dv.sim_mat = dv.sim_mat.T/np.nanmax(dv.sim_mat, axis=1)
            dv.sim_mat = dv.sim_mat.T
        # Make another one but with normalised sim_mats
        dv_sim_mat_stack_norm = average_components(dvs)


        plt.scatter([t.datetime for t in dv_total_calc.stats.corr_start], -100*scale*dv_total_calc.value-twl, c=dv_total_calc.corr, s=1, cmap='inferno_r', marker='v', label='dv joint')
        plt.scatter([t.datetime for t in dv_sim_mat_stack.stats.corr_start], -100*scale*dv_sim_mat_stack.value-2*twl, c=dv_sim_mat_stack.corr, s=1, cmap='inferno_r', marker='*', label='sim mat stack')
        plt.scatter([t.datetime for t in dv_sim_mat_stack_norm.stats.corr_start], -100*dv_sim_mat_stack_norm.value-3*twl, c=dv_sim_mat_stack_norm.corr, s=40, cmap='inferno_r', marker='+', label='sim mat stack normed')
        plt.plot([t.datetime for t in dv_sim_mat_stack.stats.corr_start], -100*scale*mean_val-4*twl, label='mean dv')
        plt.colorbar()
        plt.legend()
        plt.ylabel('lag time window [s]')
        plt.xlabel('year')
        plt.suptitle(f'{net}.{stat}.{cha} {f}-{f*2}HZ ')
        plt.hlines(0, dv_sim_mat_stack.stats.corr_start[0].datetime, dv_sim_mat_stack.stats.corr_start[-1].datetime, colors='k', linestyles='dashed')
        t = np.array(dv_sim_mat_stack.stats.corr_start)
        tlim = (t[dv_sim_mat_stack.avail][0].datetime, t[dv_sim_mat_stack.avail][-1].datetime)
        plt.xlim(tlim)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=300)
        plt.close()

        # Plot only the totals
        plt.figure(figsize=(16, 9))
        plt.suptitle(f'{net}.{stat}.{cha} {f}-{f*2}HZ ')
        plt.scatter([t.datetime for t in dv_total_calc.stats.corr_start], -100*dv_total_calc.value, c=dv_total_calc.corr, s=40, cmap='inferno_r', marker='v', label='dv joint')
        plt.scatter([t.datetime for t in dv_sim_mat_stack.stats.corr_start], -100*dv_sim_mat_stack.value, c=dv_sim_mat_stack.corr, s=40, cmap='inferno_r', marker='*', label='sim mat stack')
        plt.scatter([t.datetime for t in dv_sim_mat_stack_norm.stats.corr_start], -100*dv_sim_mat_stack_norm.value, c=dv_sim_mat_stack_norm.corr, s=40, cmap='inferno_r', marker='+', label='sim mat stack normed')
        plt.plot([t.datetime for t in dv_sim_mat_stack.stats.corr_start], -100*mean_val, 'k--', label='mean dv', zorder=0)
        plt.colorbar()
        plt.legend()
        plt.ylabel(r'dv/v [%]')
        plt.xlabel('year')
        plt.xlim(tlim)
        ylim = 100*np.nanmax(abs(np.array(vals)))
        plt.ylim((-ylim, ylim))
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile2, dpi=300)
        plt.close()