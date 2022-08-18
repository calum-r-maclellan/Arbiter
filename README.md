# Arbiter
PyTorch implementation for 'Arbiter: Hyper-Learning for Gradient-Based Batch Size Adaptation'

In this project, we develop a new but simple parameterisation scheme for tuning the batch size with gradients. With this scheme, we let a neural network meta-learn schedules for the batch size. (add more details about loss landscape, importance of the batch size, etc)

## Implementation
provide steps for people to run the codes themselves on their own machines: also refer to notebooks which unify everything into a simple script, and run networks on Google Colab.

## Background 

## Architecture

## Experiments
Our experiments are structured as followed: 
- (1) demonstrate feasibility of gradient-based scheme: do the schedules align with our expectation from learning dynamics theory? we thus devise experiments to see if the meta-gradients enable the optimal batch size to respond correctly, given different dynamical scenarios (i.e. different initial learning rates, batch sizes, and momentums). 
- (2) illustrate performance of Arbiter's heuristics in comparison to baseline scheduling heuristics, where we compare with CABS (http://auai.org/uai2017/proceedings/papers/141.pdf) and a simple multi-step schedule. We transpose CABS from tensorflow to PyTorch manually, which is okay since CABS is a simple yet effective method. 
- (3) our scheme formulates batch size tuning as a hyper-parameter optimisation (HPO) problem driven by hyper-gradients. There are no other gradient-based methods to compare with, and so we instead compare with SOTA black-box HPO methods, including random search and Bayesian optimisation (BO) for different initialisation (i.e. learning rates, momentums, and architectures). We get RS from RayTune, but code up BO ourselves; we adhere to standard settings for GP modelling. We provide the script for this in the repo: see bayes_opt.py

## Results


## Conclusions


## References
