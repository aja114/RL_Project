### INF581 - Reinforcement Learning Project

Jeremy Perez - Yazid Mouline  - Alexandre Abou Chahine  

Three approaches to solve the BipedalWalker-v3

This repository contains three different reinforcement learning algorithms implemented to  solve the BipedalWalker-v3 environment present in the Gym framework

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

#### Evolution approach

---

- In the folder /evolution you will find a jupyer notebook, a .dill file and a .gif.

- The .gif shows our best performing agent using this strategy
- The .dill file contains a saving of this best agent
- The jupyer notebook can be run easily using Colab, or the same requirements as for the lab sessions. Note that you can run by uploading the saved parameters, or you can re run the training. With the parameters in place, it could take quite some time.



#### Deep Deterministic Policy Gradient (DDPG)

---



#### World Models

---



#### References

1. Continuous Control with Deep Reinforcement Learning (DDPG) ([pdf](https://arxiv.org/pdf/1509.02971.pdf)).