### INF581 - Reinforcement Learning Project

**Authors**: Jeremy Perez - Yazid Mouline  - Alexandre Abou Chahine  

Three approaches to solve the BipedalWalker-v3

This repository contains three different reinforcement learning algorithms implemented to  solve the BipedalWalker-v3 environment present in the Gym framework

#### Evolution approach
---


- The gif directory contains .gif which shows our best performing agent using this strategy
- The agents directory contains .dill file which is a saving of the best agents
- The agent is decribed in evolution_agent.py file while selection methods are described in evolution_functions.py
- The jupyer notebook can be run easily using Colab, or the same requirements as for the lab sessions. Note that you can run by uploading the saved parameters, or you can re run the training. With the parameters in place, it could take quite some time.



#### Deep Deterministic Policy Gradient (DDPG)
---
- This technique is mostly based on the [paper](https://arxiv.org/pdf/1509.02971.pdf) which describes the idea behind the training mecanism of the actor and the critic 
- It is part of the family of soft actor critic algorithms (A2C) which are model-free RL algorithm that try to solve the problem of continuous action space 
- The actor and critic classes which makes up the agent are both described separately but depend on each other for the training
- The environment class is simply a wrapper arround the gym environment to adapt certain functionalities to our needs


#### World Models
---
- This technique is mostly based on the [paper](https://arxiv.org/abs/1803.10122) which adds a representation learning algorithm in order for the agent to improve its training using a RL algorithm
- A Recurrent Neural Network is combined with the evolution techniques  seen previously to try to improve its performance


#### How to run the code
---
There is one repositoty for each approach and they all contain:

- Multiple python script implementing the necessary classes for the agent
- train.py: a script to perform the training for the agent
- run.py: a script to play the best agent defined by the training procedure
- A jupyter notebook where the algorithm was tested 

For example to launch the best agent developped using the evolutionary approach run the following commands in the terminal 

```Bash
$ cd path/to/project/Evolution
$ python -m venv evolution_env
$ source evolution_env/bin/activate
$ pip install -r requirements.txt
$ python train.py # To train the agent
$ python run.py # To run the agent
```

##### References

1. Continuous Control with Deep Reinforcement Learning (DDPG) ([pdf](https://arxiv.org/pdf/1509.02971.pdf)).
2. World Models ([pdf](https://arxiv.org/abs/1803.10122))