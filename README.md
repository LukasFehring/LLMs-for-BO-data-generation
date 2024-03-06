# LLMs for BO - Data Generation

This repository contains the data generation process for [LLMs for BO](https://github.com/automl-private/LLMs-for-BO). The code is heavily based on the forked repository [Symbolic Explanations for Hyperparameter Optimization](https://github.com/automl/symbolic-explanations/). The main modifications are the additions of acquisition functions and transformation to be run with Hydra and schedule slurm jobs with [hydra submitit](https://hydra.cc/docs/plugins/submitit_launcher/). Additionally before every configuration in the evaluation process (after the initial samples), (20) datapoints from the search space and their values in the acquisition function are sampled. 

## Installation

You can install the required packages with `pip`. First, build an environment with Python 3.9,
```
conda create -n data-generation python=3.9
```
then install the required packages with the following commands:
```
conda install gxx_linux-64>=13.1.0
pip install -r requirements.txt
```

To install HPOBench, please run the following after activating the environment:
```
pip install git+https://github.com/automl/HPOBench.git
```

Additionally, you need to install both `hydra-core` and `hydra-colorlog` manualy with

```
pip install hydra-core
pip install hydra-colorlog
```

### Hydra
To run jobs with hydra without slurm you need to make sure that `hydra_config/base.yaml` starts with
```yaml
defaults:
  - _self_
  # - slurm
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog
```
Below you configure additional parameters.

You can then run jobs by calling 
```bash
python run_sampling_hpobench.py
```

### Hydra and Slurm
For simplicity, we use [hydra-submitit](https://hydra.cc/docs/plugins/submitit_launcher/) to schedule slurm jobs. By doing so one only has to specify the specs in `hydra_config/slurm.yaml`, and the bash files are built on the fly.

You can then schedule jobs by calling 
```bash
python run_sampling_hpobench.py --m
```
followed by additional parameters

E.g.
```bash
python run_sample_acquisition_functions.py -m "data-generation.job_id=range(365)"
```
