import numpy as np
import tensorflow.compat.v1 as tf
import random

from env_class import Environment
from actor_class import Actor
from critic_class import Critic
from noise_class import OUActionNoise


class Agent:

    def __init__(self, TF_FLAGS):
        ''' This class build the Agent that learns in the environment via the actor-critic algorithm. '''

        self.env = Environment()
        self.TF_FLAGS = TF_FLAGS

        # Define the actor network and a "stationary" target for the training
        self.actor_target = Actor(
            scope='target', target_network=None, env=self.env, flags=TF_FLAGS)
        self.actor = Actor(
            scope='actor', target_network=self.actor_target, env=self.env, flags=TF_FLAGS)

        # Define the critic network and a "stationary" target for the training
        self.critic_target = Critic(
            scope='target', target_network=None, env=self.env, flags=TF_FLAGS)
        self.critic = Critic(
            scope='critic', target_network=self.critic_target, env=self.env, flags=TF_FLAGS)

        # Start the TF sessions
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        # Pass it to the four networks in total
        self.critic.set_session(self.session)
        self.actor.set_session(self.session)
        self.actor_target.set_session(self.session)
        self.critic_target.set_session(self.session)

        # Initialise network weights and assert the correct initialisation
        self.critic.init_target_network()
        for lp, tp in zip(self.critic.param, self.critic_target.param):
            assert np.array_equal(self.session.run(lp), self.session.run(tp))

        self.actor.init_target_network()
        for lp, tp in zip(self.actor.param, self.actor_target.param):
            assert np.array_equal(self.session.run(lp), self.session.run(tp))

        # Create the noise object to add to the actor network
        self.actor_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.TF_FLAGS.noise_dev) * np.ones(1))

        # experience replay buffer
        self.memory = []
        self.memory_size = 15000

    def update_agent(self):
        if len(self.memory) > self.TF_FLAGS.batch_size:
            # Randomly chose a batch from the replay buffer
            indexes = random.sample(
                range(len(self.memory)-1), self.TF_FLAGS.batch_size)

            states = np.array([self.memory[i][0] for i in indexes])
            actions = np.array([self.memory[i][1] for i in indexes])
            rewards = np.array([self.memory[i][2] for i in indexes])
            next_states = np.array([self.memory[i][3] for i in indexes])
            dones = np.array([self.memory[i][4] for i in indexes])

            self.update_critic(states, actions, rewards, next_states, dones)
            self.update_actor(states, actions, rewards, next_states, dones)

    def update_critic(self, states, actions, rewards, next_states, dones):
        next_actions = np.array(self.actor_target.get_action(next_states))
        q_nexts = self.critic.target_network.calculate_Q(
            next_states, next_actions)
        q_targets = rewards.reshape(-1, 1) + self.TF_FLAGS.gamma * q_nexts
        self.critic.train(states, actions, q_targets)

    def update_actor(self, states, actions, rewards, next_states, dones):
        predicted_action = np.array(self.actor.get_action(states))
        q_gradients = self.critic.compute_gradients(states, predicted_action)[0]
        self.actor.train(states, q_gradients)

    def add2memory(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

        if len(self.memory) > self.memory_size:
            # Discard the first half of the memory
            self.memory = self.memory[int(self.memory_size)//2:]

    def train_one_episode(self, max_iterations=500, render=False):
        ''' Play an episode in the OpenAI Gym '''
        
        state = self.env.reset()

        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()

        done = False
        total_reward = 0
        iters = 0

        # Loop for the episode
        while not done and iters < max_iterations:

            # Sample an action from the actor distribution
            prev_state = state
            action = self.actor.get_action(state.reshape(1, -1), self.actor_noise).reshape(-1)

            # Obtain a <state, reward, done> tuple from the environment
            state, reward, done, _ = self.env.get_env().step(action)

            total_reward += reward

            if render:
                self.env.render()

            self.add2memory(prev_state, action, reward, state, done)

            self.update_agent()
            
            iters += 1

        self.critic.update_target_parameter()
        self.actor.update_target_parameter()

        return total_reward

    def train(self, num_episodes=100, display_step=10, max_iterations=500):
        print("\n"+"*"*100)
        print("TRAINING START\n")
        '''Run the environment for a particular number of episodes. '''
        total_rewards = []

        for n in range(0, num_episodes):

            if n % display_step == 0 and n >= display_step:
                avg_reward = sum(
                    total_rewards[n-display_step: n]) / display_step
                print("episodes: %i, avg_reward (last: %i episodes): %.2f" %
                      (n, display_step, avg_reward))
                total_reward = self.train_one_episode(render=False, max_iterations=max_iterations)
                # self.env.make_gif(f"tmp/episode number {n}")

            else:
                total_reward = self.train_one_episode(render=False, max_iterations=max_iterations)

            total_rewards.append(total_reward)

        print("\n"+"*"*100)
        print("TRAINING END\n")

        return total_rewards

    def play_one_episode(self, max_iterations=500):
        '''Runs and records one episode using the trained actor and critic'''
        # Get the initial state and reshape it
        state = self.env.reset()
        state = state.reshape(1, self.env.get_state_size())
        done = False
        iters = 0
        total_reward = 0

        # Loop for the episode
        while not done and iters < max_iterations:

            # Sample an action from the gauss distribution
            action = self.actor.get_action(state)

            # Obtain a <state, reward, done> tuple from the environment
            state, reward, done, _ = self.env.get_env().step(action.reshape(-1))
            state = state.reshape(1, self.env.get_state_size())
            total_reward += reward

            self.env.render()
            iters += 1

        return self.env.make_gif()
