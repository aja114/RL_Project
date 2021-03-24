## TO OPEN directly the best performing agent saved
import dill
from env_class import Environment
from evolution_agent import Agent

env = Environment()

print("The pretrained agent has been loaded")
with open('agents/best_agent.dill', 'rb') as fp:
	best_agent = dill.load(fp)

observation = env.reset()

# Render the best agent 
print("Rendering best agent\n")
observation = env.reset()
for t in range(1000):
    env.render_wrapper.render()
    observation, reward, done, info = env.step(best_agent.act(observation))
env.close()


env.close()
env.render_wrapper.make_gif("gif/Evolution_best")

