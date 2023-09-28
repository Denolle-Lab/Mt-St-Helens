# Execute this script using (in bash) for multicore
# $mpirun python this_script.py

import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from copy import deepcopy

import yaml
from mpi4py import MPI
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


from seismic.correlate.correlate import Correlator
from seismic.trace_data.waveform import Store_Client


yaml_f = '/home/pmakus/mt_st_helens/Mt-St-Helens/params.yaml'
root = '/data/wsd01/st_helens_peter'


# Create mask

def create_utcl(loc: str):
    with open(loc, 'r') as f:
        # skip header
        f.readline()
        f.readline()
        lines = f.readlines()
    datetimes = [line.split('\t')[:2] for line in lines]
    utcl = [
        UTCDateTime('-'.join(dt[0].split('/')[::-1]) + f'T{dt[1]}')
        for dt in datetimes]
    return utcl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    client = Client('IRIS')
else:
    client = None
client = comm.bcast(client, root=0)


# Client is not needed if read_only
sc = Store_Client(client, root, read_only=True)

if rank == 0:
    with open(yaml_f) as file:
        options = yaml.load(file, Loader=yaml.FullLoader)
else:
    options = None
options = comm.bcast(options, root=0)

options['save_comps_separately'] = False
options['co']['combination_method'] = 'betweenStations'

options['co']['subdivision'] = {
    'corr_inc': 3600,
    'corr_len': 3600,
    'recombine_subdivision': False,
    'delete_subdivision': False}


# Fix EDM response removal problem
options['co']['xcombinations'] = [
    'UW-UW.EDM-YEL',
    'UW-UW.EDM-HSR',
    'UW-UW.EDM-JUN',
    'UW-UW.EDM-SHW',
    'UW-UW.EDM-SOS',
    'UW-UW.EDM-STD',
    'UW-UW.EDM-SUG',
    'CC-UW.SEP-EDM',
    'CC-UW.SUG-EDM',
    'CC-UW.SUG-EDM',
    'CC-UW.JRO-EDM',
    'CC-UW.NED-EDM',
    'CC-UW.STD-EDM',
    'CC-UW.SWF2-EDM',
    'CC-UW.SWFL-EDM',
    'CC-UW.VALT-EDM',
    'PB-UW.B202-EDM',
    'PB-UW.B203-EDM',
    'PB-UW.B204-EDM',
]

# next compute a very broad one with 0.25-4Hz
for ii in range(3):
    if ii != 2:
        continue
    # Set bp: frequency
    f = (1/(2**ii), 2/(2**ii))
    # Length to save in s
    lts = 50/f[0] + 30  # 10s extra because its xcorr
    # lts = 20*(1/f[0]) + 35
    options['co']['corr_args']['lengthToSave'] = lts
    # sample rate
    fs_theo = f[1] * 2
    if fs_theo <= .75:
        fs = 1
    elif fs_theo <= 1.5:
        fs = 2
    elif fs_theo <= 4:
        fs = 5
    elif fs_theo <= 10:
        fs = 12.5
    else:
        fs = 25
    options['co']['sampling_rate'] = fs
    if fs/2 >= f[1]*2:
        fupzero = f[1]*2
    else:
        fupzero = fs/2

    # Decide about preprocessing
    options['co']['preProcessing'] = [
        {'function': 'seismic.correlate.preprocessing_stream.detrend_st',
        'args':{'type':'linear'}},
        # {'function': 'seismic.correlate.preprocessing_stream.stream_mask_at_utc',
        #     'args': {'starts': utcl,
        #             'masklen': 300, # seconds
        #             'reverse': False}},
        {'function': 'seismic.correlate.preprocessing_stream.cos_taper_st',
            'args': {'taper_len': 30, # seconds
                    'lossless': True}},
        {'function': 'seismic.correlate.preprocessing_stream.stream_filter',
            'args': {'ftype':'bandpass',
                    'filter_option': {'freqmin': 0.01,
                                    'freqmax': fs/2}}}]


    options['co']['corr_args']['FDpreProcessing'] = [
        {'function':'seismic.correlate.preprocessing_fd.spectralWhitening',
            'args':{'joint_norm':False}},
        {'function':'seismic.correlate.preprocessing_fd.FDfilter',
            'args':{'flimit':[f[0]/2,f[0],f[1],fupzero]}}]

    options['co']['corr_args']['TDpreProcessing'] = [
        {'function':'seismic.correlate.preprocessing_td.detrend',
        'args':{'type':'linear'}},
        {'function':'seismic.correlate.preprocessing_td.TDfilter',
        'args':{'type':'bandpass','freqmin':f[0],'freqmax':f[1]}},
        {'function':'seismic.correlate.preprocessing_td.signBitNormalization',
        'args': {}}]


    options['co']['subdir'] = os.path.join(
        'corrs_response_removed_longtw',
        f'xstations_{fs}_{f[0]}-{f[1]}_wl{lts}_1b_SW'
    )
    # Do the actual computation
    c = Correlator(sc, deepcopy(options))
    c.pxcorr()
