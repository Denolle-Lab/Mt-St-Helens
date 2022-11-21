import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '32'

import glob

from copy import deepcopy

import yaml
import numpy as np

from seismic.monitor.monitor import Monitor

os.chdir('/home/pmakus/st_helens/clusters')


yaml_f = '../params.yaml'
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)

# General options
options['net'] = {'network': '*', 'station': '*'}
options['proj_dir'] = '/home/pmakus/st_helens/clusters'


meth = 'xstations'
# methods = ['xstations', 'autoComponents', 'betweenComponents']


# for meth in methods:
for ii in range(4):
    if ii == 0 or ii>2:
        continue
    # f = [0.0625*(2**ii), 0.125*(2**ii)]
    f = [0.25*(2**ii), 0.5*(2**ii)]

    if 0.5 >= f[0] >= .25:
        tw_len = 52
    elif f[0] == 1:
        tw_len = 32
    elif f[0] == 2:
        tw_len = 30
    elif f[0] == 4:
        tw_len = 15
    tws = np.floor(7/f[0]) - 1
    smoothlen_d = None

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
    
    date_inc = 86400
    win_len = 86400
    options['dv']['win_len'] = win_len
    options['dv']['date_inc'] = date_inc

    options['dv']['tw_start'] = tws
    options['dv']['tw_len'] = tw_len
    options['dv']['freq_min'] = f[0]
    options['dv']['freq_max'] = f[1]
    options['dv']['dt_ref'] = {'win_inc' : 0, 'method': 'mean', 'percentile': 50}
    options['dv']['preprocessing'][0]['args']['wsize'] = int(smoothlen_d/(date_inc/86400))  # Less smoothing for more precise drop check

    corrdir = f'{meth}_no_response_removal_{f[0]}-{f[1]}_1b_SW'

    dvdir = f'vel_change/with_tt_early/{meth}_{f[0]}-{f[1]}_wl86400_tw{tws}-{tw_len}_1b_mute_SW_presmooth{smoothlen_d}d_srw'
    
    options['co']['subdir'] = corrdir
    options['dv']['subdir'] = dvdir
    options['fig_subdir'] = dvdir + '_fig'

    
    m = Monitor(deepcopy(options))
    m.compute_velocity_change_bulk()
        # m.compute_components_average(method='AutoComponents')
        # m.compute_components_average(method='CrossComponents')
    # m.compute_components_average(method='CrossStations')
