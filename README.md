# Controlling the output of a generative model by finding latent feature representations

Repository to our master's thesis. Use conda with environment.yml to import DP environment.

To successfully reproduce our work, you will need the following software requirements:
- gcc 12.2.1 
- CUDA 11 or newer
- nvcc 11 or newer
- conda 23.1.0 or newer
- *nix operating system

To successfully run jupyter notebook, run the following commands in the following order:
- `git clone https://github.com/Wicwik/DP.git`
- `cd DP`
- `conda env create -n DP --file environment.yml`
- `conda activate DP`
- `jupyter notebook`


Data can be found [here](http://data.belanec.eu/mt_data.zip). 

The MLV dataset can found [here](https://data.belanec.eu/mlv.tar.gz).

Repo with some data can be found [here](https://data.belanec.eu/DP.zip)
