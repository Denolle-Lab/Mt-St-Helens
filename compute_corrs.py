# Execute this script using (in bash) for multicore
# $mpirun python this_script.py

import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '32'

from copy import deepcopy

import yaml
from obspy.clients.fdsn import Client

from seismic.correlate.correlate import Correlator
from seismic.trace_data.waveform import Store_Client


yaml_f = '/home/pmakus/mt_st_helens/Mt-St-Helens/params.yaml'
root = '/data/wsd01/st_helens_peter'

# Client is not needed if read_only
sc = Store_Client(Client('IRIS'), root, read_only=True)
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)

for method in ['betweenComponents', 'autoComponents']:
    options['co']['combination_method'] = method
    for ii in range(5):
        # Set bp: frequency
        f = (4/(2**ii), 8/(2**ii))
        # Length to save in s
        lts = 50/f[0]
        lts = 20*(1/f[0]) + 10
        options['co']['corr_args']['lengthToSave'] = lts
        # sample rate
        fs_theo = f[1] * 2
        if fs_theo <= .75:
            fs = 1
        elif fs_theo <= 1.5:
            fs = 2
        elif fs_theo <= 4:
            fs = 5
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
            {'function': 'seismic.correlate.preprocessing_stream.cos_taper_st',
                'args': {'taper_len': 10, # seconds
                        'lossless': True}},
            {'function': 'seismic.correlate.preprocessing_stream.stream_filter',
                'args': {'ftype':'bandpass',
                        'filter_option': {'freqmin': 0.01,
                                        'freqmax': fs/2}}}]

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
            {'function':'seismic.correlate.preprocessing_td.TDfilter',
            'args':{'type':'bandpass','freqmin':f[0],'freqmax':f[1]}},
            {'function':'seismic.correlate.preprocessing_td.signBitNormalization',
            'args': {}}]

        # Set folder name
        if method == 'autoComponents':
            options['co']['subdir'] = os.path.join(
                'corrs_response_removed',
                 f'{method}_{fs}_{f[0]}-{f[1]}_wl{lts}_1b'
            )
        else:
            options['co']['subdir'] = os.path.join(
                'corrs_response_removed',
                f'{method}_{fs}_{f[0]}-{f[1]}_wl{lts}_1b_SW'
            )
        # Do the actual computation
        c = Correlator(sc, deepcopy(options))
        c.pxcorr()
