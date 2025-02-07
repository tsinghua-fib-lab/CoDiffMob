# CoDiffMob
Code for the paper "Noise Matters: Diffusion Model-based Urban Mobility Generation with Collaborative Noise Priors"

## Getting Started

**step 1**: clone the repository

**step 2**: create a virtual environment with conda or virtualenv

```bash
# with conda
conda create -n noise python=3.10
conda activate noise
# with virtualenv
virtualenv -p python3.10 noise
source noise/bin/activate
```

**step 3**: install the dependencies

```bash
pip install -r requirements.txt
```

## Training the diffusion model

Entrypoint to the diffusion model training is train.py. The script takes a configuration file as input, which specifies the dataset path, hyperparameters, and other settings. The configuration file is in the YAML format, an example is provided in the `configs` directory.

- Modify the configuration file `configs/{city}.yml` to specify the dataset path and other hyperparameters.

- Run the following command to train the diffusion model

```bash
python3 train.py --config configs/{city}.yml --save /path/to/log_dir
```

The trained model will be saved in the log directory and you can check the training process in tensorboard by running `tensorboard --logdir /path/to/log_dir`. We trained our model for 200 epochs on both ISP and MME datasets.

## Generating synthetic trajectories
We introduce a two-stage method to generate synthetic trajectories. The first stage is to sample location transitions based on EPR model and flow data. The second stage is to build noise prior and generate synthetic trajectories based on it by the trained diffusion model.

**step 1**: sample location transitions

- Obtain the transition matrix, population, and move probability as priot knowledge of the generation task. 
- Run the script `scripts/epr_sampling.py` to sample location transitions.

```bash
python3 scripts/epr_sampling.py --num-traj {num} --save /path/to/save_dir
```

**step 2**: generate synthetic trajectories

- Construct the noise prior from the sampled location transitions
- Run the genration process by the trained diffusion model based on the noise prior

The generation process is in the `generate.ipynb` notebook. You can run the notebook to see the synthetic trajectories.
