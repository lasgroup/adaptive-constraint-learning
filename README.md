
# Interactively Learning Preference Constraints in Linear Bandits


This repository contains source code to reproduce experiments in the paper ["Interactively Learning Preference Constraints in Linear Bandits"](TODO). The code is provided as is and will not be maintained. Here we provide instructions on how to set up and run the code to reproduce the experiments reported in the paper.


### Citation

David Lindner, Sebastian Tschiatschek, Katja Hofmann, and Andreas Krause . **Interactively Learning Preference Constraints in Linear Bandits**. In _International Conference on Machine Learning (ICML)_, 2022.

```
@inproceedings{lindner2022interactively,
    title={Interactively Learning Preference Constraints in Linear Bandits},
    author={Lindner, David and Tschiatschek, Sebastian and Hofmann, Katja and Krause, Andreas},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2022},
}
```


## Setup

We recommend using [Anaconda](https://www.anaconda.com/) to set up an environment with the dependencies of this repository. After installing Anaconda, the following commands set up the environment:
```
conda create -n cbai python=3.9
conda activate cbai
pip install -e .
```


## Reproducing the Experiments

The main code for running experiments is in `experiment.py`. We use [`sacred`](https://github.com/IDSIA/sacred) for tracking parameters and results, and we provide all necessary config files to reproduce the experiments presented in the paper. We provide a utility script called `run_sacred_experiments.py` to run experiments for a specific config file (the experiments can optionally be parallelized). Results can be plotted using `make_plots.py`. Here, we provide the exact commands that reproduce the results in a specific section, including necessary preparation.


### Synthetic Experiments (Section 4.1)

The following four commands reproduce the synthetic experiments:
```
python run_sacred_experiments.py --config experiment_configs/synthetic/irrelevant_dimensions_eps.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/synthetic/irrelevant_dimensions_dim.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/synthetic/unit_sphere_arms.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/synthetic/unit_sphere_dim.json --n_jobs 1
```


### Comparing ACOL to Regret Minimization (Section 4.2)

The following command compares G-ACOL to MaxRew-F and MaxRew-U in the 1-dimensional test case:
```
python run_sacred_experiments.py --config experiment_configs/synthetic/regret_comparison.json --n_jobs 1
```


### Preference Learning Experiments (Section 4.3)


For the driving experiments, it is necessary to first compute a set of candidate driving policies that correspond to the arms of a CBAI problem. This can be done using the `scripts/driver_compute_candidate_policies.py` script. However, for convenience we also provide a set of precomputed policies in `driver_candidate_policies_1000.pkl`; so, this step can be skipped.


Once the candidate policies are available, the constraint learning experiments can be run with the following commands:
```
python run_sacred_experiments.py --config experiment_configs/driver/driver_target_velocity_theory.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/driver/driver_target_velocity_tuned.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/driver/driver_target_location_theory.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/driver/driver_target_location_tuned.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/driver/driver_blocked_theory.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/driver/driver_blocked_tuned.json --n_jobs 1
```
