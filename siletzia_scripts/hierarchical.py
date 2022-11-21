import os
from datetime import datetime
import locale
import glob

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster import hierarchy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.dates as mdates
from mpi4py import MPI

from seismic.plot.plot_utils import set_mpl_params
from seismic.correlate.correlate import CorrStream
from seismic.plot.plot_utils import set_mpl_params
from seismic.db.corr_hdf5 import CorrelationDataBase



# freq = 1.0
# max_d = 10
k = 4  # Always create 4 clusters

freqs = [0.25*2**n for n in range(3)]


def main(path, cluster_file, network, station):
    with CorrelationDataBase(path, mode='r') as cdb:
        cst = cdb.get_data(network, station, '??Z-??Z', 'subdivision')
    # probably easier to handle in matrix form

    np_computed = True
    if not len(glob.glob(f'{cluster_file}_*.npy')):
        np_computed = False
        cb = cst.create_corr_bulk(channel='??Z-??Z', inplace=False)
        data = cb.data
        np.nan_to_num(data, copy=False)

        Z = linkage(data, 'ward')

        set_mpl_params()


        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            truncate_mode='lastp',  # show only the last p merged clusters
            p=32,  # show only the last p merged clusters

        )

        ax = plt.gca()
        # ax.set_yticklabels('')
        # ax.set_xticklabels('')
        # ax.set_axis_off()
        # plt.hlines(max_d, ax.get_xlim()[0], ax.get_xlim()[1], colors='k', linestyles='--')
        plt.savefig(f'clusters_response_removed/dendrogram_{cb.stats.network}_{cb.stats.station}_{freq}.png', dpi=300)
        plt.close()

        print('clustering')
        clusters = fcluster(Z, k, criterion='maxclust')
        print('clustered, creating histogram data')
        corr_starts = [t.datetime for t in cb.stats.corr_start]
        y = []
        for ii in range(max(clusters)):
            y.append([])
        for ii, jj in enumerate(clusters):
            y[jj-1].append(corr_starts[ii])

        print('plotting histogram')
        plt.figure(figsize=(10, 8))
        plt.hist(y, 35, histtype='bar')
        plt.legend(list(range(max(clusters))))
        ax = plt.gca()
        ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())

        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%h %y'))
        plt.title(f'Cluster distribution\n{cb.stats.network}.{cb.stats.station} Z-Z {freq}-{freq*2} Hz')
        plt.savefig(f'clusters_response_removed/distribution_hierach_{cb.stats.network}.{cb.stats.station}_{freq}.png', dpi=300)
        plt.close()

        print('Sorting CorrTraces')
        inds = []
        os.makedirs(os.path.dirname(cluster_file), exist_ok=True)
        for ii in range(max(clusters)):
            ind = np.where(clusters==ii+1)[0]
            inds.append(ind)
            with open(f'{cluster_file}_{ii}.npy', 'wb') as f:
                np.save(f, ind)
        print('saved')
        corrsts = []
        for ii, ind in enumerate(inds):
            corrsts.append(CorrStream())
            corrsts[ii].extend([cst[i] for i in ind])
            save_cl_n(str(ii), corrsts, path)
            print('saving cluster ', ii)
    else:
        print('Found file with clustered indices, loading')
        inds = []
        for fi in glob.glob(f'{cluster_file}_*.npy'):
            with open(fi, 'rb') as f:
                inds.append(np.load(f))
        corrsts = []
        for ii, ind in enumerate(inds):
            corrsts.append(CorrStream())
            corrsts[ii].extend([cst[i] for i in ind])
            save_cl_n(str(ii), corrsts, path)
            print('saving cluster ', ii)

    print('computing average corrs')

    if not np_computed:
        stacks = []
        for cst in corrsts:
            stacks.append(cst.stack())

        
        print('creating plot')
        set_mpl_params()
        fig = plt.figure(figsize=(14, 4))

        plt.subplots_adjust(wspace=0.1)

        ax1 = fig.add_subplot(131)




        color_theme = ['darkblue', 'orange', 'green' , 'red']#, 'k']

        hierarchy.set_link_color_palette(color_theme)

        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            truncate_mode='lastp',  # show only the last p merged clusters
            p=32,  # show only the last p merged clusters
            # color_threshold=max_d,
            above_threshold_color='grey'

        )

        ax1.set_yticklabels('')
        ax1.set_xticklabels('')
        ax1.set_axis_off()
        ax1.set_title('(a)')

        ax2 = fig.add_subplot(132)


        plt.hist(y, 26, histtype='bar', color=color_theme)

        ax2.xaxis.set_major_locator(mpl.dates.AutoDateLocator())

        ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%h %y'))
        ax2.set_title('(b)')
        ax2.set_ylabel('N')
        plt.xticks(rotation=45)

        ax3 = fig.add_subplot(133)


        for (ii, stack), color in zip(enumerate(stacks), color_theme):
            ax3.plot(stack[0].times(), stack[0].data+ii, linewidth=0.8, color=color)
        ax3.set_xlabel(r"$\tau$ [s]")
        ax3.set_yticklabels('');
        ax3.set_title('(c)')
        plt.yticks([])


        ax2.legend([0, 1, 2, 3, 4], loc="lower center", ncol=3, bbox_to_anchor=(0.22, -0.05), bbox_transform=fig.transFigure)

        plt.savefig(f'clusters_response_removed/hierarchical_merged_{network}.{station}_{freq}.png', dpi=300, bbox_inches='tight')
        plt.close()


def save_cl_n(n: str, corrsts, inpath:str):
    os.makedirs(f'clusters_response_removed/xstations_{freq}-{freq*2}_1b_SW_cl{n}', exist_ok=True)
    outpath = f'clusters_response_removed/xstations_{freq}-{freq*2}_1b_SW_cl{n}/{corrsts[0][0].stats.network}.{corrsts[0][0].stats.station}.h5'
    with CorrelationDataBase(inpath, mode='r') as cdb:
        co = cdb.get_corr_options()
    with CorrelationDataBase(outpath, corr_options=co) as cdb:
        for m in n:
            cdb.add_correlation(corrsts[int(m)])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
psize = comm.Get_size()
for freq in freqs:
    # for stat in stations:
    #     path = f'corr_stuff/xstations_no_response_removal_*_{freq}-{freq*2}_wl*_1b_SW/{net}.{stat}.h5'
    

    paths = glob.glob(
        f'corrs_response_removed_longtw/xstations_*_{freq}-{freq*2}_wl*_1b_SW/*.h5')
    pmap = (np.arange(len(paths))*psize)/len(paths)
    pmap = pmap.astype(np.int32)
    for path, p in zip(paths, pmap):
        if rank != p:
            continue
        net, stat, _ = os.path.basename(path).split('.')

        cluster_file = f'clusters_response_removed/{stat}/indf{freq}'
        main(path, cluster_file, net, stat)
