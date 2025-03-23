Reproducible material for Geophysical Journal International - Wang N. and Ravasi M. "Imaging the Volve ocean-bottom field data with the Upside-down Rayleigh-Marchenko method"

install_conda:
	conda env create -f environment.yml && conda activate UD_RM 


## Project structure
This repository is organized as follows:

* :open_file_folder: **udrm**: python library containing routines for upside-down RM method;
* :open_file_folder: **dataset**: folder containing data 
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);


## Notebooks
The following notebooks are provided:

- :orange_book: ``1project_wavefield_processed.ipynb``: notebook performing basic data processing for Volve field data;
- :orange_book: ``2volve_sep_rec.ipynb``: notebook performing wavefield separation on the receiver side using PZ summation and source-deghosting;
- :orange_book: ``3volve_sep_src_deghosting.ipynb``: notebook performing wavefield separation on the source side using source-deghosting;
- :orange_book: ``4GF_one_point_modifyshift.ipynb``: notebook performing Green's function retrieval for Volve field data;
- :orange_book: ``5raymckimaging_upd_volve.ipynb``: notebook performing Green's function retrieval for Volve field data in an imaging area;

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate UD_RM
```

