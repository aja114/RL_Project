import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import sys
from agent_class import Agent


os.system("rm -f tmp/*")
tf.disable_eager_execution()


# Using flags for usefulness as they can be set when running the code from the command line
tf.app.flags.DEFINE_float('learning_rate_Actor', 0.0001,
                          'Learning rate for the policy estimator')
tf.app.flags.DEFINE_float('learning_rate_Critic', 0.001,
                          'Learning rate for the state-value estimator')
tf.app.flags.DEFINE_float('gamma', 0.99, 'Future discount factor')
tf.app.flags.DEFINE_float(
    'tau', 0.01, 'Update rate for the target networks parameter')
tf.app.flags.DEFINE_float('noise_dev', 0.02, 'Standard deviation for the noise component')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size for the updates')


TF_FLAGS = tf.app.flags.FLAGS

agent = Agent(TF_FLAGS)

total_rewards = agent.train(num_episodes=2000, display_step=10, max_iterations=1000)

pd.DataFrame(np.array(total_rewards)).to_csv("tmp/results.csv")
plt.plot(np.array(total_rewards))

agent.play_one_episode()



