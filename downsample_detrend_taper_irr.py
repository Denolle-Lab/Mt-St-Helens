'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 19th October 2023 12:10:39 pm
Last Modified: Thursday, 19th October 2023 02:47:14 pm
'''
import glob
from mpi4py import MPI
import os

from obspy import read, read_inventory
import numpy as np

from seismic.utils.miic_utils import resample_or_decimate, gap_handler, stream_require_dtype
from seismic.correlate.preprocessing_stream import detrend_st

infolder = '/data/wsd01/st_helens_peter/mseed'
outfolder = '/data/wsd01/st_helens_peter/mseed_preprocessed'
inventory = '/data/wsd01/st_helens_peter/inventory/*'

SDS_FMTSTR = os.path.join(
    "{year}", "{network}", "{station}", "{channel}.{sds_type}",
    "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    inv = read_inventory(inventory)
    for ii, infile in enumerate(
            glob.glob(os.path.join(infolder, '*', '*', '*', '*', '*'))):
        if ii % size != rank:
            continue
        print(infile)
        st = read(infile)
        try:
            st = resample_or_decimate(st, 10)
            st.split()
            st = detrend_st(st, 'linear')
            st = gap_handler(st, max_interpolation_length=1, retain_len=80, taper_len=20)
            if not st[0].stats.station == 'EDM' and st[0].stats.channel == 'EHZ':
                st.remove_response(inventory=inv, output='VEL', taper=False)
            stream_require_dtype(st, np.float32)
            outfile = os.path.join(
                outfolder,
                SDS_FMTSTR.format(
                    year=st[0].stats.starttime.year,
                    network=st[0].stats.network,
                    station=st[0].stats.station,
                    channel=st[0].stats.channel,
                    location=st[0].stats.location,
                    sds_type='D',
                    doy=st[0].stats.starttime.julday)
            )
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            st.write(outfile, format='MSEED')
        except Exception as e:
            print(f'Error for {infile}', e)

main()