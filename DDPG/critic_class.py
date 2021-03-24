import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.summary.writer.writer import FileWriter


class Critic:

    def __init__(self, scope, target_network, env, flags):
        """
        This class implements the Critic for the stochastic policy gradient model.
        The critic provides a state-value for the current state environment where 
        the agent operates.

        :param scope: within this scope the parameters will be defined
        :param target_network: instance of the Actor(target-network class)
        :param env: instance of the openAI environment
        :param FLAGS: TensorFlow flags which contain thevalues for hyperparameters

        """

        self.TF_FLAGS = flags
        self.env = env

        if scope == 'target':

            with tf.variable_scope(scope):

                self.states = tf.placeholder(tf.float32, shape=(
                    None, self.env.state_size), name='states')
                self.actions = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_action_size()), name='actions')
                self.q = self.create_network(scope='q_target_network')
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_target_network')

        else:

            with tf.variable_scope(scope):

                # Add the target network instance
                self.target_network = target_network

                # Create the placeholders for the inputs to the network
                self.states = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_state_size()), name='states')
                self.actions = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_action_size()), name='actions')

                # Create the network with the goal of predicting the action-value function
                self.q = self.create_network(scope='q_network')
                self.q_targets = tf.placeholder(
                    tf.float32, shape=(None, 1), name='q_targets')

                # The parameters of the network
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_network')

                with tf.name_scope('q_network_loss'):
                    # Difference between targets value and calculated ones by the model
                    self.loss = tf.losses.mean_squared_error(
                        self.q_targets, self.q)

                with tf.name_scope('train_q_network'):
                    # Optimiser for the training of the critic network
                    self.train_opt = tf.train.AdamOptimizer(
                        self.TF_FLAGS.learning_rate_Critic).minimize(self.loss)

                with tf.name_scope('q_network_gradient'):
                    # Compute the gradients to be used for the actor model training
                    self.actor_loss = -tf.math.reduce_mean(self.q)
                    self.gradients = tf.gradients(self.actor_loss, self.actions)

                with tf.name_scope('update_q_target'):
                    # Perform a soft update of the parameters: Critic network parameters = Local Parameters (LP) and Target network parameters (TP)
                    # TP = tau * LP + (1-tau) * TP
                    self.update_opt = [tp.assign(tf.multiply(self.TF_FLAGS.tau, lp)+tf.multiply(
                        1-self.TF_FLAGS.tau, tp)) for tp, lp in zip(self.target_network.param, self.param)]

                with tf.name_scope('initialize_q_target_network'):
                    # Set the parameters of the local network equal to the target one
                    # LP = TP
                    self.init_target_op = [tp.assign(lp) for tp, lp in zip(
                        self.target_network.param, self.param)]

                FileWriter('logs/train', graph=self.train_opt.graph).close()

    def create_network(self, scope):
        '''Build the neural network that estimates the action-values '''
        state_layer_size1 = 128
        state_layer_size2 = 64
        action_layer_size = 64
        state_action_layer_size1 = 128
        state_action_layer_size2 = 64

        with tf.variable_scope(scope):

            h1_state = tf.layers.dense(self.states, state_layer_size1, tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.initializers.glorot_normal(),
                                       bias_initializer=tf.zeros_initializer()
                                       )

            h1_state_n = tf.layers.batch_normalization(h1_state)

            h2_state = tf.layers.dense(h1_state_n, state_layer_size2, tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.initializers.glorot_normal(),
                                       bias_initializer=tf.zeros_initializer()
                                       )

            h2_state_n = tf.layers.batch_normalization(h2_state)

            h1_action = tf.layers.dense(self.actions, action_layer_size, tf.nn.relu, use_bias=True,
                                        kernel_initializer=tf.initializers.glorot_normal(),
                                        bias_initializer=tf.zeros_initializer()
                                        )

            h1_action_n = tf.layers.batch_normalization(h1_action)

            state_action = tf.concat([h2_state_n, h1_action_n], axis=1)

            h1_state_action = tf.layers.dense(state_action, state_action_layer_size1, tf.nn.relu,
                                              use_bias=True,
                                              kernel_initializer=tf.initializers.glorot_normal(),
                                              bias_initializer=tf.zeros_initializer()
                                              )

            h1_state_action_n = tf.layers.batch_normalization(h1_state_action)

            h2_state_action = tf.layers.dense(h1_state_action_n, state_action_layer_size2, tf.nn.relu,
                                              use_bias=True,
                                              kernel_initializer=tf.initializers.glorot_normal(),
                                              bias_initializer=tf.zeros_initializer()
                                              )

            h2_state_action_n = tf.layers.batch_normalization(h2_state_action)

            q = tf.layers.dense(h2_state_action_n, 1, None,
                                kernel_initializer=tf.initializers.glorot_normal())

        return q

    def compute_gradients(self, states, actions):
        '''Compute the gradients of the action_value estimator neural network '''

        feed_dict = {
            self.states: states,
            self.actions: actions
        }

        q_gradient = self.session.run(self.gradients, feed_dict)

        return q_gradient

    def calculate_Q(self, states, actions):
        '''Compute the action-value '''

        feed_dict = {self.states: states,
                     self.actions: actions}

        q_next = self.session.run(self.q, feed_dict)

        return q_next

    def train(self, states, action, targets):
        '''Train the critic network '''
        feed_dict = {
            self.states: states,
            self.actions: action,
            self.q_targets: targets
        }

        self.session.run(self.train_opt, feed_dict)

    def set_session(self, session):
        '''Set the session '''
        self.session = session

    def init_target_network(self):
        '''Initialize the parameters of the target-network '''
        self.session.run(self.init_target_op)

    def update_target_parameter(self):
        '''Update the parameters of the target-network '''
        self.session.run(self.update_opt)
