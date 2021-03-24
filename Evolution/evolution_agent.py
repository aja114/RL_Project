import math
from scipy.special import softmax
import numpy as np
import seaborn as sns
import time
from evolution_functions import xavier_init

class Agent():
    def __init__(self, n_input, n_hidden, n_hidden2, n_output, mu_prob):
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.mu_prob = mu_prob
        self.network = {'Layer 1' : xavier_init(n_input, n_hidden , layer = 1),
                        'Bias 1'  : np.zeros((1,n_hidden)),
                        'Layer 2' : xavier_init(n_hidden, n_hidden2 , layer = 2),
                        'Bias 2'  : np.zeros((1,n_hidden2)),
                        'Layer 3': xavier_init(n_hidden2, n_output , layer = 3),
                        'Bias 3': np.zeros((1, n_output))}
        self.score = 0

    def reborn(self, parents):
        #The agent reborns with a recombination of its parents parameters
        parent1, parent2 = parents
        for key in self.network:
            mask = np.random.choice([0,1],size=self.network[key].shape,p=[.5,.5])
            self.network[key] = np.where(mask==1,parent1.network[key],parent2.network[key])
            #if np.random.random() > 0.5:
            #    self.network[node] = parent1.network[node]
            #else:
            #    self.network[node] = parent2.network[node]
                
    def mutate(self):
        for key in self.network:
            mask = np.random.choice([0,1],size=self.network[key].shape,p=[1-self.mu_prob,self.mu_prob])
            random = xavier_init(mask.shape[0],mask.shape[1])
            self.network[key] = np.where(mask==1,self.network[key]+random,self.network[key])
            #if np.random.random() > self.mu_prob:
            #    self.network[node] += np.random.normal(0, 0.05 * np.abs(self.network[node]) )
                #if np.argwhere(np.isnan(self.network[node])):
                #    print("nanNode")
    
    def act(self, state):
        if(state.shape[0] != 1):
            state = state.reshape(1,-1)
        net = self.network
        layer_one = np.tanh(np.dot(state,net['Layer 1']) + net['Bias 1'])
        layer_two = np.tanh(np.dot(layer_one, net['Layer 2']) + net['Bias 2'])
        layer_three = np.tanh(np.dot(layer_two, net['Layer 3']) + net['Bias 3'])
        #print(layer_two, layer_two[0])
        return layer_three[0]

        
    def evaluate(self, env):
        scores = []
        for i in range(2):
            state = env.reset()
            score = 0
            done = False
            while not done:
                state, reward, done, _ = env.step(self.act(state))
                score += reward
            #print(score)
            scores.append(score)
        self.score = sum(scores)/2
        #print(self.score)


