# PlaNet
This repo contains some naive implementations (only RSSM) of a purely model-based reinforcement learning algorithm that solves control tasks from images by efficient planning in a learned latent space. 

<p align="left">
  <img src="gifs/walker-walk.gif" alt="walker-walk" width="256" />
  <img src="gifs/cartpole-balance.gif" alt="cartpole-balance" width="256" />
</p>

## Installation

To install this project, simply run the following command after cloning the repo:

```shell
pip install poetry
make install
```


## Method

This project implements  Learning Latent Dynamics for Planning from Pixels (RSSM) ([Hafner et al., 2019](https://arxiv.org/abs/1811.04551))


## Run 

To run the training script
```shell
cd scripts
python trainer.py --config config/path/here
```


To run the evaluation script
```bash
cd scripts
python tester.py --config config/path/here
```
