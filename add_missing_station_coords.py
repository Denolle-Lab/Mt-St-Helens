'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 6th November 2023 10:40:29 am
Last Modified: Monday, 6th November 2023 10:55:03 am
'''
import os
import glob
import re

from obspy import read_inventory

from seismic.monitor.dv import read_dv


dv_files = glob.glob('/data/wsd01/st_helens_peter/dv/new_gap_handling_ddt/*/*.npz')
dv_files += glob.glob('/data/wsd01/st_helens_peter/dv/dv_separately_ddt/*/*/*.npz')

for dv_file in dv_files:
    dv = read_dv(dv_file)
    rewrite = False
    if 'stla' not in dv.stats or 'stlo' not in dv.stats:
        # extract station name
        stat = dv.stats.station.split('-')[0]
        net = dv.stats.network.split('-')[0]
        # read station coordinates from inventory
        inv = read_inventory(f'/data/wsd01/st_helens_peter/inventory/{net}.{stat}.xml')
        dv.stats.stla = inv[0][0].latitude
        dv.stats.stlo = inv[0][0].longitude
        rewrite = True
    if 'evla' not in dv.stats or 'evlo' not in dv.stats:
        # get coordinates of second station
        stat = dv.stats.station.split('-')[1]
        net = dv.stats.network.split('-')[1]
        # read station coordinates from inventory
        inv = read_inventory(f'/data/wsd01/st_helens_peter/inventory/{net}.{stat}.xml')
        dv.stats.evla = inv[0][0].latitude
        dv.stats.evlo = inv[0][0].longitude
        rewrite = True
    if rewrite:
        print(f'added coordinates to {dv.stats.id}')
        dv.save(dv_file)

