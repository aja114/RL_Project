import gym
import numpy as np
import imageio
import time
from IPython.display import Image

class RenderWrapper:
    def __init__(self, env, force_gif=False):
        self.env = env
        self.force_gif = force_gif
        self.reset()

    def reset(self):
        self.images = []

    def render(self):
        img = self.env.render(mode='rgb_array')
        
        if self.force_gif:
            self.images.append(img)

    def make_gif(self, filename="render"):
        if self.force_gif:
            imageio.mimsave(filename + '.gif', [np.array(img) for i, img in enumerate(self.images) if i%2 == 0], fps=29)
            return Image(open(filename + '.gif','rb').read())

    @classmethod
    def register(cls, env, force_gif=False):
        env.render_wrapper = cls(env, force_gif=True)

def Environment():
    env = gym.make('BipedalWalker-v3')
    RenderWrapper.register(env, force_gif=True)
    return env


