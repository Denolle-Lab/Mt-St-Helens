import glob
import os

from seismic.monitor.dv import read_dv

files = glob.glob('/data/wsd01/st_helens_peter/dv/shift_stack_2007_45dsmooth_trend/1.0-2.0/*.npz')


figures = os.path.join(os.path.dirname(files[0]), 'figures')
os.makedirs(figures, exist_ok=True)

for f in files:
    dv = read_dv(f)
    fn = '.'.join(os.path.basename(f).split('.')[:-1]) + '.png'
    dv.plot(
        figure_file_name=fn, save_dir=figures,
        normalize_simmat=True, sim_mat_Clim=[-1, 1])
