"""
This script computes the auto and self-correlations used in
Makus et al, 2024 (SRL).
"""
# Execute this script using (in bash) for multicore
# $mpirun python this_script.py

import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from copy import deepcopy

import yaml
from obspy.clients.fdsn import Client
from mpi4py import MPI

from seismic.correlate.correlate import Correlator
from seismic.trace_data.waveform import Store_Client


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    client = Client('IRIS')
else:
    client = None
client = comm.bcast(client, root=0)


yaml_f = './params.yaml'

# Folder in which you have your seismic database
root = '/path/to/your/seismic/database/'

# Client is not needed if read_only
sc = Store_Client(client, root, read_only=True)
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)

for method in ['autoComponents', 'betweenComponents']:
    options['co']['combination_method'] = method
    for ii in range(3):
        # Set bandpass frequency
        f = (1/(2**ii), 2/(2**ii))

        # Length to save in s
        lts = 50/f[0] + 20
        options['co']['corr_args']['lengthToSave'] = lts

        if 2.5 >= f[1]*2:
            fupzero = f[1]*2
        else:
            fupzero = 2.5

        # No spectral whitening for auto
        if method == 'autoComponents':
            options['co']['corr_args']['FDpreProcessing'] = [
                {'function':'seismic.correlate.preprocessing_fd.FDfilter',
                    'args':{'flimit':[f[0]/2,f[0],f[1],fupzero]}}]
        else:
            options['co']['corr_args']['FDpreProcessing'] = [
                {'function':'seismic.correlate.preprocessing_fd.spectralWhitening',
                    'args':{'joint_norm':False}},
                {'function':'seismic.correlate.preprocessing_fd.FDfilter',
                    'args':{'flimit':[f[0]/2,f[0],f[1],fupzero]}}]

        options['co']['corr_args']['TDpreProcessing'] = [
            {'function':'seismic.correlate.preprocessing_td.detrend',
            'args':{'type':'linear'}},
            {'function':'seismic.correlate.preprocessing_td.taper',
            'args':{'type':'cosine_taper', 'p': 0.03}},
            {'function':'seismic.correlate.preprocessing_td.TDfilter',
            'args':{'type':'bandpass','freqmin':f[0],'freqmax':f[1]}},
            {'function':'seismic.correlate.preprocessing_td.signBitNormalization',
            'args': {}}]

        # Set folder name
        if method == 'autoComponents':
            options['co']['subdir'] = os.path.join(
                'corrs_response_removed',
                 f'{method}_{f[0]}-{f[1]}'
            )
        else:
            options['co']['subdir'] = os.path.join(
                'corrs_response_removed',
                f'{method}_{f[0]}-{f[1]}'
            )
        # Do the actual computation
        c = Correlator(sc, deepcopy(options))
        c.pxcorr()
