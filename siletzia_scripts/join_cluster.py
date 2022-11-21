import os

from seismic.db.corr_hdf5 import CorrelationDataBase
from seismic.correlate.stream import CorrStream

network = 'UW-UW'
station = 'EDM-HSR'
freqs = [0.5, 1.0]
stations = ['EDM-HSR', 'EDM-SHW', 'EDM-SOS', 'HSR-SHW', 'HSR-SOS', 'SHW-SOS']
net = 'UW-UW'
path = 'clusters/xstations_no_response_removal_{f}-{f2}_1b_SW_cl{n}/{net}.{stat}.h5'
nnn = [
    [[0, 1], [0, 1]],
    [[0, 1, 3], [1, 2, 3]],
    [[0, 1, 2, 3], [1, 2, 3]],
    [[0, 1, 2], [1, 2, 3]],
    [[3], [0, 1]],
    [[0,1,2,3], [1, 2,3]]]


def join_clusters(n: list, path_tmpl: str, freq0: float, net: str, stat: str):
    cst = CorrStream()
    p2 = path.format(f=freq0, f2=2*freq0, n=''.join([str(m) for m in n]), net=net, stat=stat)
    if os.path.isfile(p2):
        print(f'file {p2} exists.')
        return
    for m in n:
        p = path.format(f=freq0, f2=2*freq0, n=m, net=net, stat=stat)
        with CorrelationDataBase(p, mode='r') as cdb:
            co = cdb.get_corr_options()
            cst.extend(cdb.get_data(net, stat, 'EHZ-EHZ', 'subdivision'))
    # Write new file
    os.makedirs(os.path.dirname(p2), exist_ok=True)
    with CorrelationDataBase(p2, corr_options=co) as cdb:
        cdb.add_correlation(cst)


for station, nn in zip(stations, nnn):
    for freq, n in zip(freqs, nn):
        if len(n) == 1 or len(n) == 4:
            continue
        join_clusters(n, path, freq, network, station)