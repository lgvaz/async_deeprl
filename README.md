# Asynchronous Q-learning 

## About

Tensorflow implementation of the asynchronous Q-learning (and Double Q-learning) method described in Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning".

## Usage
### Training
To start training with default hyperparameters, run:  
`python main.py <env_name>`  
For Double Q-learning, run:  
`python main.py <env_name> --double_learning=Y`

### Loading
After running you will be prompted to give a name for the current experiment. A folder called _experiments_ will be created, where checkpoints, logs and videos will be stored. If that name already exists, training will start from last checkpoint.  

### Options
For a list of hyperparameters that can be specified, run:  
`python main.py -h`

### Visualize training
Some statistics are recorderd while training, for visualization run:  
`tensorboard --logdir=<summary_dir>`
