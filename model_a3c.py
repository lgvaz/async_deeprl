import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Approximator:
    '''
    Function approximator to estimates actions probabilities and states values.
    '''

    def __init__(self, env_name, experiment_name, num_actions, optimizer_name,
                 learning_rate, global_step, scope, clip_norm, create_summary=False):
        '''
        Creates the network

        Args:
            env_name: Name of the gym environment (Used on tensorflow summary)
            experiment_name: Name of current experiment (Used on tensorflow summary)
            num_features: Number of possible observations of environment
            num_actions: Number of possible actions
            optimizer_name: The name of the optimizer to be used
            learning_rate: Learning rate used when performing gradient descent
            global_step: A global step tensor shared by all threads
            scope: The scope used by the network
            clip_norm: The value used to clip the gradients by a l2-norm,
                       if 0, gradients will not be clipped
            create_summary: Whether the network should create summaries or not,
                            only the main network should create summaries
        '''
        self.env_name = env_name
        self.experiment_name = experiment_name
        # Placeholders
        self.states = tf.placeholder(name='states',
                                     shape=[None, 84, 84, 4],
                                     dtype=tf.float32)
        self.td_targets = tf.placeholder(name='td_targets',
                                      shape=[None],
                                      dtype=tf.float32)
        self.actions = tf.placeholder(name='chosen_actions',
                                       shape=[None],
                                       dtype=tf.int32)

        # Define network architecture
        with tf.variable_scope(scope):
            # Convolutional layers
            self.conv = slim.stack(self.states, slim.conv2d, [
                        (16, (8, 8), 4),
                        (32, (4, 4), 2)
            ])
            self.fc = slim.fully_connected(slim.flatten(self.conv), 256)
            self.action_probs = slim.fully_connected(self.fc, num_actions,
                                                     activation_fn=tf.nn.softmax)
            self.state_value = slim.fully_connected(self.fc, 1,
                                                    activation_fn=None)

        # Optimization process

        batch_size = tf.shape(self.states)[0]
        # Pick only the actions which were chosen
        # action_ids = (i_batch * NUM_ACTIONS) + action
        actions_ids = tf.range(batch_size) * num_actions + self.actions
        chosen_actions_probs = tf.gather(tf.reshape(self.action_probs, [-1]), actions_ids)

        # Use entropy to encourage exploration
        self.entropy = -tf.reduce_sum(self.action_probs * tf.log(self.action_probs), 1, name='entropy')
        # Calculate loss for policy and state-value function
        td_errors = self.td_targets - self.state_value
        self.loss = tf.reduce_sum(- tf.log(chosen_actions_probs) * td_errors
                                  - self.entropy
                                  + tf.squared_difference(self.td_targets, self.state_value))
        # Calculate learning rate
        self.learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, 1e5, 1e-2, staircase=True)

        if optimizer_name == 'rms':
            opt = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6)
        elif optimizer_name == 'adam':
            opt = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError('{} optmizer is not supported'.format(optimizer_name))
        # Get list of variables given by scope
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        # Calcuate gradients
#        grads_and_vars  = opt.compute_gradients(self.loss, local_vars)
        grads = tf.gradients(self.loss, local_vars)
        if clip_norm > 0:
#           grads_and_vars = [(tf.clip_by_value(grad, -1, 1), var)
#                             for grad, var in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)
            grads_and_vars = list(zip(clipped_grads, local_vars))
        else:
            grads_and_vars = list(zip(grads, local_vars))
        self.training_op = opt.apply_gradients(grads_and_vars)

        if create_summary:
            # Add summaries
            tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'loss']), self.loss)
            tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'learning_rate']), self.learning_rate)
            tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'max_action_prob']), tf.reduce_max(self.action_probs))
            tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'average_action_prob']), tf.reduce_mean(self.action_probs))
            tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'max_state_value']), tf.reduce_max(self.state_value))
            tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'average_state_value']), tf.reduce_mean(self.state_value))
#            tf.summary.histogram('/'.join([self.env_name, self.experiment_name, 'states_value']), self.state_value)
            tf.summary.histogram('/'.join([self.env_name, self.experiment_name, 'action_probs']), self.action_probs)

    def predict_actions_probs(self, sess, states):
        '''
        Compute probabilities for each action given the state

        Args:
            sess: Tensorflow session to be used
            states: Environment observations
        '''
        return sess.run(self.action_probs, feed_dict={self.states: states})

    def predict_state_value(self, sess, states):
        '''
        Compute the value of given state

        Args:
            sess: Tensorflow session to be used
            states: Environment observations
        '''
        return sess.run(self.state_value, feed_dict={self.states: states})

    def update(self, sess, states, actions, targets):
        '''
        Performs the optimization process

        Args:
            sess: Tensorflow session to be used
            states: Enviroment observations
            actions: Actions to be updated (Probally performed actions)
            targets: TD targets
        '''
        feed_dict = {self.states: states,
                     self.actions: actions,
                     self.td_targets: targets}
        sess.run(self.training_op, feed_dict=feed_dict)

    def create_summary_op(self, sess, logdir):
        '''
        Creates summary writer

        Args:
            sess: Tensorflow session to be used
            logdir: Directory for saving summary

        Returns:
            A summary writer operation
        '''
        # Check if path already exists
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Create placeholders to track some statistics
        avg_reward = tf.placeholder(name='average_reward',
                                    shape=(),
                                    dtype=tf.float32)
        avg_length = tf.placeholder(name='average_length',
                                    shape=(),
                                    dtype=tf.float32)
        eval_reward = tf.placeholder(name='evaluation_reward',
                                     shape=(),
                                     dtype=tf.float32)
        eval_length = tf.placeholder(name='evaluation_length',
                                     shape=(),
                                     dtype=tf.float32)
        episode_global_step = tf.placeholder(name='global_step',
                                     shape=(),
                                     dtype=tf.int32)
        # Create some summaries
        tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'average_reward']), avg_reward)
        tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'average_length']), avg_length)
        tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'evaluation_reward']), eval_reward)
        tf.summary.scalar('/'.join([self.env_name, self.experiment_name, 'evaluation_length']), eval_length)
        # Merge all summaries
        merged = tf.summary.merge_all()

        def summary_writer(states, actions, targets, average_reward, average_length,
                           evaluation_reward, evaluation_length, global_step):
            feed_dict = {
                self.states: states,
                self.actions: actions,
                self.td_targets: targets,
                avg_reward: average_reward,
                avg_length: average_length,
                eval_reward: evaluation_reward,
                eval_length: evaluation_length,
                episode_global_step: global_step
            }
            summary, step = sess.run([merged, episode_global_step],
                                     feed_dict=feed_dict)

            # Write summary
            writer.add_summary(summary, step)

        return summary_writer
