# Online Planning in POMDPs with Self-Improving Simulators

This is the codebase accompanying the IJCAI2022 paper "[Online Planning in POMDPs with Self-Improving Simulators](https://arxiv.org/abs/2201.11404)" by Jinke He, Miguel Suau, Hendrik Baier, Michael Kaisers, Frans A. Oliehoek.

## Dependencies
* [Singularity](https://sylabs.io/docs/)
* [libtorch](https://pytorch.org/cppdocs/installing.html) (C++ version of PyTorch cpu): `download and unzip libtorch (version 1.10) into third-party/libtorch`
* [yaml-cpp](https://github.com/jbeder/yaml-cpp): `git clone https://github.com/jbeder/yaml-cpp third-party/yaml-cpp`

## Reproducing results for online planning experiments

### Singularity Container

We implemented our online planning experiments in C++ and provided a Singularity definition file ([singularity/FADMEN.def](singularity/FADMEN.def)) to resolve the dependencies. 

To run the code, first build the singularity container with the command: `sudo singularity build singularity/FADMEN.sif singularity/FADMEN.def`.

To execute a command under the singularity container, use `./run` + command.

This codebase does not support the use of GPUs. All inference is done in CPUs.

### Compile
`./run bash scripts/build.sh`

### General
`./run ./scripts/run_benchmark` + path to config file   

### Grab A Chair - simulation controlled experiments: Figure 2(a), 2(b) and 2(c)

We repeat this experiment for $2500$ times. 

for lambda $\lambda \in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 3.0]$ `./run scripts/run_benchmark configs/GAC/simulation_controlled/mec_0.3_lam_{lambda}.yaml`

### Grab A Chair - time controlled experiments: Figure 3(a)

We repeat this experiment for $2500$ times. 

#### Baseline (planning with global simulator):

`./run scripts/run_benchmark configs/GTC/time_controlled/global.yaml`

#### Our method (planning with self-improving simulator)

for lambda $\lambda \in [0.7,1.0,2.0]$ `./run scripts/run_benchmark configs/GTC/time_controlled/0.015625sec_mec_0.3_lam_{lambda}.yaml`

### Grid Traffic Control - time controlled experiments: Figure 3(b)

We repeat this experiment for $1000$ times.

#### Baseline (planning with global simulator):

`./run scripts/run_benchmark configs/GTC/time_controlled/global.yaml`

#### Our method (planning with self-improving simulator)

for lambda $\lambda \in [0.3, 0.4, 0.5, 0.6, 0.7]$ `./run scripts/run_benchmark configs/GTC/time_controlled/ep1500_gru_8_mean_mec_0.1_lam_{lambda}_0.0625sec.yaml`

### Plotting results
Results are smoothed with `gaussian_filter1d` from `scipy.ndimage` where `sigma` is set to $1$. 

## Contact

Feel free to contact us if you are interested in this work! 

J.He-4@tudelft.nl / jinkehe1996@gmail.com

## Acknowledgment
This project had received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 758824 — INFLUENCE).
