import os
from glob import glob

from mpi4py import MPI
import numpy as np
import yaml

from seismic.monitor.monitor import Monitor

yaml_f = 'params.yaml'
path2 = '/home/pmakus/st_helens/clusters/xstations_no_response_removal_{f}-*_1b_SW_cl{n}/{net}.{stat}.h5'

freqs = [0.25*2**n for n in range(3)]
freqs = [0.25]
stations = ['EDM-HSR', 'EDM-SHW', 'EDM-SOS', 'HSR-SOS', 'HSR-SHW', 'SHW-SOS']
net = 'UW-UW'


def monitor_cluster(
        n, infile: str, yaml_f: str, f: float, net: str, stat: str):

    with open(yaml_f) as file:
        options = yaml.load(file, Loader=yaml.FullLoader)
    options['proj_dir'] = '.'
    dvopt = options['dv']
    dvopt['win_len'] = 86400
    dvopt['date_inc'] = 86400

    if 0.5 >= f >= .25:
        tw_len = 60
    elif f == 1:
        tw_len = 35
    elif f == 2:
        tw_len = 30
    elif f == 4:
        tw_len = 15
    tws = np.floor(7.5/f)
    smoothlen_d = None

    if f >= 2:
        smoothlen_d = 9
    elif f >= 0.5:
        smoothlen_d = 18
    elif f >= 0.2:
        smoothlen_d = 72
    elif f >= 0.1:
        smoothlen_d = 63
    else:
        smoothlen_d = 90

    dvopt['freq_min'] = f
    dvopt['freq_max'] = f*2
    dvopt['tw_start'] = tws
    dvopt['tw_len'] = tw_len
    dvdir = os.path.split(os.path.dirname(infile))
    dvdir = os.path.join(
        dvdir[0], 'vel_change',
        f'{dvdir[1]}_{tws}-{tw_len}')
    dvopt['subdir'] = dvdir
    options['fig_subdir'] = dvopt['subdir'] + '_fig'

    dvopt['preprocessing'][0]['args']['wsize'] = int(
        smoothlen_d
        /(dvopt['date_inc']/86400))
    dvopt['dt_ref'] = {'win_inc' : 0, 'method': 'mean', 'percentile': 50}
    # This is required because of the multicore hack I'm using
    os.makedirs(dvopt['subdir'], exist_ok=True)
    os.makedirs(options['fig_subdir'], exist_ok=True)


    m = Monitor(options)
    m.compute_velocity_change(
        infile, 'subdivision', net, stat, 'EHZ-EHZ')


comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()


# Execute the script for this
# Very ugly for-loop stacking, but lazy

for stat in stations:
    for f in freqs:
        for n in range(4):
            if rank == n and psize >= 4:
                inf = glob(path2.format(f=f, net=net, stat=stat, n=n))[0]
                monitor_cluster(n, inf, yaml_f, f, net, stat)
            elif rank == 0 and psize < 4:
                print('use at least 4 cores..computing with single core now')
                inf = glob(path2.format(f=f, net=net, stat=stat, n=n))[0]
                monitor_cluster(n, inf, yaml_f, f, net, stat)

