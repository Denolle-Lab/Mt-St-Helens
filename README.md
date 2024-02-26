# Electronic Supplement to "Analysing Volcanic, Tectonic, and Environmental Influences on the Seismic Velocity from 25 Years of Data at Mount St. Helens"

This repository contains a somewhat lose collection of scripts and jupyter notebooks to reproduce the results from Makus et al., 2024 (currently in review).

Packages required for the scripts are listed in ``requirements.txt``.

## Scripts and Notebooks
Not all (though most) scripts need to be run, to obtain results. In general, all jupyter notebooks (file ending ``.ipynb``), containing plotting and data interpretataion. Python scripts (file ending ``.py``) contain the computation scripts

### How to run scripts
Many of the scripts support mpi (i..e, multi-threading).
You can run them as follows:

```bash
mpirun -n $n_cores python python_script.py
```

### Order to run the scripts in
All files start with a two-digit number indicating the order that they have to be ran in.


To compute dv/v, SeisMIC is used. Head to the [GitHub repository](https://github.com/PeterMakus/SeisMIC) and the [documentation](https://petermakus.github.io/SeisMIC/) for information on installation and how to use. You can also ask me, whenever something comes up (or open an issue in said repository).

