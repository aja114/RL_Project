import numpy as np
from scipy.special import softmax

def xavier_init(n_in, n_out, layer=1):
    #return np.random.normal(0, 1/(n_in**(layer - 1)), size = (n_in, n_out))
    glorot = 1.0*np.sqrt(6.0/(n_in+n_out))
    return np.random.uniform(-glorot,glorot,size=(n_in,n_out))    

def select_next_gen(agent_set, n_selected):
    n_best = int(n_selected*0.8)
    n_random = n_selected - n_best
    sorted_agents = sorted(agent_set, key = lambda agent : agent.score, reverse = True)
    #for agent in sorted_agents:
     #   print(agent.score)
    next_gen = sorted_agents[:n_best]
    next_random = np.random.choice(sorted_agents, size = n_random, replace = False)
    for rand in next_random:
        next_gen.append(rand)
    not_selected = []
    for agent in agent_set:
        #print(next_gen)
        #print("ee")
        if agent not in next_gen:
            not_selected.append(agent)
    return sorted_agents[0], next_gen, not_selected

def select_parents(agent_set):
    return np.random.choice(agent_set, size = 2, p = softmax([agent.score for agent in agent_set]))
