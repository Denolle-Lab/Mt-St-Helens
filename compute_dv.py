import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '32'

import glob

from copy import deepcopy

import yaml
import numpy as np

from seismic.monitor.monitor import Monitor

yaml_f = '/home/pmakus/mt_st_helens/Mt-St-Helens/params.yaml'
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)

# General options
options['net'] = {'network': '*', 'station': '*'}

os.chdir('/data/wsd01/st_helens_peter')

meth = 'xstations'
# methods = ['xstations'] #, 'autoComponents', 'betweenComponents']


for ii in range(3):
    if ii != 2:
        continue

    f = (1/(2**ii), 2/(2**ii))

# f = (0.2, 4)

# tw_len = np.ceil(35/f[0])  # as in Hobiger, et al (2016)
    if 0.5 >= f[0] >= .25:
        tw_len = 60
    elif f[0] == 1:
        tw_len = 35
    elif f[0] == 2:
        tw_len = 30
    elif f[0] == 4:
        tw_len = 15
    tws = np.floor(7/f[0])
    smoothlen_d = None

    if f == (0.2, 4):
        tw_len = 80
        tws = 7

    if f[0] >= 2:
        smoothlen_d = 9
    elif f[0] >= 0.5:
        smoothlen_d = 18
    elif f[0] >= 0.2:
        smoothlen_d = 36
    elif f[0] >= 0.1:
        smoothlen_d = 63
    else:
        smoothlen_d = 90

    # new standard smoothing for new 0.25, 0.5, and 1.0 Hz
    smoothlen_d = 45

    # # Extra long smoothing
    # smoothlen_d = 365
    try:
        corrdir = glob.glob(f'corrs_response_removed_longtw/{meth}_*_{f[0]}-{f[1]}*')[0]
    except IndexError:
        if int(f[0])==f[0]:
            f[0] = int(f[0])
        if int(f[1])==f[1]:
            f[1] = int(f[1])
        corrdir = glob.glob(f'corrs_response_removed_longtw/{meth}_*_{f[0]}-{f[1]}*')[0]

    # One data point represents 10 days, should be good enough with such
    # a long smoothing
    date_inc = 864000
    win_len = 864000
    options['dv']['win_len'] = win_len
    options['dv']['date_inc'] = date_inc

    dvdir = f'dv/resp_removed/{meth}_{f[0]}-{f[1]}_wl{win_len}_tw{tws}-{tw_len}_1b_mute_SW_presmooth{smoothlen_d}d_srw'
    options['dv']['preprocessing'][0]['args']['wsize'] = int(smoothlen_d/(date_inc/86400))  # Less smoothing for more precise drop check
    options['co']['subdir'] = corrdir
    options['dv']['subdir'] = dvdir
    options['fig_subdir'] = dvdir + '_fig'
    options['dv']['tw_start'] = tws
    options['dv']['tw_len'] = tw_len
    options['dv']['freq_min'] = f[0]
    options['dv']['freq_max'] = f[1]
    options['dv']['dt_ref'] = {'win_inc' : 0, 'method': 'mean', 'percentile': 50}

    m = Monitor(deepcopy(options))
    m.compute_velocity_change_bulk()
        # m.compute_components_average(method='AutoComponents')
        # m.compute_components_average(method='CrossComponents')
    # m.compute_components_average(method='CrossStations')
