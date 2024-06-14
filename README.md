<p align="center">
  <img width="300" height="424" src="https://github.com/tgac-vumc/OncoBLADE/blob/main/logo.png">
</p>

# OncoBLADE: Malignant cell fraction-informed deconvolution
OncoBLADE is a Bayesian deconvolution method designed to estimate cell type-specific gene expression profiles and fractions from bulk RNA profiles of tumor specimens by integrating prior knowledge on cell fractions. You can find the [preprint of OncoBLADE at Research Square](https://www.researchsquare.com/article/rs-4252952/v1).

<p align="center">
  <img width="75%" height="75%" src="https://github.com/tgac-vumc/OncoBLADE/blob/main/VisualAbstract.png">
</p>



#### Demo notebook is available under `jupyter`. See below how to open it. 


## System Requirements

### Hardware Requirements

OncoBLADE can run on the minimal computer spec, such as Binder (1 CPU, 2GB RAM on Google Cloud), when data size is small. However, OncoBLADE can significantly benefit from the larger amount of CPUs and RAM.

### OS Requirements

The package development version is tested on Linux operating systems. (CentOS 7 and Ubuntu 16.04). 

## Installation

### Using pip

The python package of BLADE is available on pip.
You can simply (takes only <1min):

```
pip install OncoBLADE
```

We tested BLADE with `python => 3.6`.


### Using Conda

One can create a conda environment contains BLADE and also other dependencies to run [Demo](https://github.com/tgac-vumc/BLADE/blob/master/jupyter/BLADE%20-%20Demo%20script.ipynb).
The environment definition is in [environment.yml](https://github.com/tgac-vumc/BLADE/environment.yml).

### Step 1: Installing Miniconda 3
First, please open a terminal or make sure you are logged into your Linux VM. Assuming that you have a 64-bit system, on Linux, download and install Miniconda 3 with:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
On MacOS X, download and install with:

```
curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### Step 2: Create a conda environment

You can install all the necessary dependency using the following command (may takes few minutes; `mamba` is quicker in general).

```
conda env create --file environment.yml
```

Then, the `OncoBLADE` environment can be activate by:

```
conda activate OncoBLADE
```

### Step 3: Running a demo script

You can find a demo script under `jupyter` folder.
You can open the script using the command below after activating the `OncoBLADE` environment:

```
jupyter notebook jupyter/OncoBLADE\ -\ Demo script.ipynb
```


## Overview of OncoBLADE (In progress)
In the OncoBLADE package, you can load the following functions and modules.

- `BLADE`: A class object contains core algorithms of `OncoBLADE`, an extended version of `BLADE`. Users can reach internal variables (`Nu`, `Omega`, and `Beta`) and functions for calculating objective functions (ELBO function) and gradients with respect to the variational parameters. There also is an optimization function (`BLADE.Optimize()`) for performing L-BFGS optimization. OncoBLADE features an iterative update to optimize hyperparameter `Alpha` and integration of prior expectation of subset of cell types. In BLADE, `Alpha` is a user-defined hyperparameter.

To run classic BLADE and OncoBLADE, we provide main functinos `BLADE_framework` for BLADE and `Framework_Iterative` for OncoBLADE. 

See below to obtain the current estimate of cellualr fractions, gene expression profiles per cell type and per sample:
  - `ExpF(self.Beta)` : returns a `Nsample` by `Ngene` matrix contains estimated fraction of each cell type in each sample.
  - `self.Nu`: a `Nsample` by `Ngene` by `Ncell` multidimensional array contains estimated gene expression levels of each gene in each cell type for each sample.
  - `numpy.mean(self.Nu,0)`: To obtain a estimated gene expression profile per cell type, we can simply take an average across the samples.

- `Framework_Iterative`: OncoBLADE framework based on the `BLADE` class module above. Users need to provide the following input/output arguments.
  - Input arguments
    - `X`: a `Ngene` by `Ncell` matrix contains average gene expression profiles per cell type (a signature matrix) in log-scale.
    - `stdX`: a `Ngene` by `Ncell` matrix contains standard deviation per gene per cell type (a signature matrix of gene expression variability).
    - `Y`: a `Ngene` by `Nsample` matrix contains bulk gene expression data. This should be in linear-scale data without log-transformation.
    - `Expectation`: a `Nsample` by `Ncell` matrix contains the expected cell fraction used to inform OncoBLADE [Optional]
    - `Ind_Marker`: Index for marker genes. By default, `[True]*Ngene` (all genes used without filtering). For the genes with `False` they are excluded in the first phase (Empirical Bayes) for finidng the best hyperparameters.
    - `Ind_sample`: Index for the samples used in the first phase (Empirical Bayes). By default, `[True]*Nsample` (all samples used).
    - `Alpha`, `Alpha0`, `Kappa0` and `SY`: hyperparameters used in the model. `Alpha` is also optimzed, while others are fixed. By default, `Alpha=1`, `Alpha0=0.1`, 'Kappa0=1`, `sY=1`.
    - `IterMax`: Number of maximum iteration between variational parameter optimization by L-BFGS and updating hyperparameter `Alpha`. By default, `iterMax=100`.
    - `Nrep`: Number of random initial guess used to run OncoBLADE. The best in terms of ELBO function will be chosed among the local optimum. By default, `Nrep=3`.
    - `Njob`: Number of jobs executed in parallel. By default, `Njob=10`.

  - Output values
    - `final_obj`: A final `BLADE` object with optimized variational parameters and hyperparameters.
    - `conv`: The ELBO function value (i.e., local optimum) for the final `BLADE` object.
    - `outs`: `Nrep` BLADE objects optimized with all random initial guesses with their final ELBO values.


### TBD

Functions for purification: `Parallel_Purification`, `Purify_AllGenes`
