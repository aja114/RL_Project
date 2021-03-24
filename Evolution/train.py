from scipy.special import softmax
import math
import gym
import numpy as np
import time
import dill

from evolution_agent import Agent
from evolution_functions import select_next_gen, select_parents
from env_class import Environment

env = Environment()

#Algorithm parameter
n_input = env.observation_space.shape[0]
n_output = env.action_space.shape[0]
n_hidden = 128
n_hidden2 = 64

mu_prob = 0.10
generation = 100
population = 80

prop_selected = 0.3

save_best_agent = False

n_selected = int(prop_selected * population)
best_agents = []
agent_set = [Agent(n_input, n_hidden, n_hidden2, n_output, mu_prob) for i in range(population)]
best, selected, next_children = select_next_gen(agent_set, n_selected)

best_agents.append(best)
score_evolution = []
time_evolution = []


print("\n"+"*"*100)
print("TRAINING START\n")

for gen in range(1, generation+1):
    time_gen1 = time.time()
    for new_child in next_children:
        #time1 = time.time()
        parents = select_parents(selected)
        #time2 = time.time()
        #print("Time for select_parents:", time2-time1)
        new_child.reborn(parents)
        #time3 = time.time()
        #print("Time reborn:", time3-time2)
        new_child.mutate()
        #time4 = time.time()
        #print("Time for mutate:", time4 -time3)
        #new_child.evaluate(env)
        #time5 = time.time()
        #print("Time for evaluate:", time5-time4)
    for agent in agent_set:
        agent.evaluate(env)
    best, selected, next_children = select_next_gen(agent_set, n_selected)
    best_agents.append(best)
    time_gen2 = time.time()
    print("Generation:",gen,"Score:",best.score)
    score_evolution.append(best.score)
    time_evolution.append(time_gen2 - time_gen1)
    print("Time for generation:", time_gen2 - time_gen1)

print("\n"+"*"*100)
print("TRAINING ENDED\n")

best_agent = best_agents[-1]

# Render the best agent 
print("Rendering best agent\n")
observation = env.reset()
for t in range(1000):
    env.render_wrapper.render()
    observation, reward, done, info = env.step(best_agent.act(observation))
env.close()
env.render_wrapper.make_gif("gif/Evolution")

# Save the parameters of the best agent
if save_best_agent:
    with open('agents/best_agent.dill', 'wb') as fp:
        dill.dump(best_agent, fp)


