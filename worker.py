import gym
import itertools
import numpy as np
import tensorflow as tf
from threading import Thread, Lock
from atari_envs import AtariWrapper
from utils import copy_vars, egreedy_policy, get_epsilon_op

class Worker:
    def __init__(self, env_name, num_actions, num_workers, num_steps,
                 stop_exploration, final_epsilon_list, change_epsilon_step, discount_factor,
                 online_update_step, target_update_step, online_net, target_net, global_step,
                 double_learning, num_stacked_frames, sess, coord, saver, summary_writer, savepath, videodir):
        self.env_name = env_name
        self.num_actions = num_actions
        self.num_workers = num_workers
        self.num_steps = num_steps
        self.stop_exploration = stop_exploration
        self.final_epsilon_list = final_epsilon_list
        self.change_epsilon_step = change_epsilon_step
        self.discount_factor = discount_factor
        self.online_update_step = online_update_step
        self.target_update_step = target_update_step
        self.online_net = online_net
        self.target_net = target_net
        self.global_step = global_step
        self.double_learning = double_learning
        self.num_stacked_frames = num_stacked_frames
        self.sess = sess
        self.coord = coord
        self.saver = saver
        self.summary_writer = summary_writer
        self.savepath = savepath
        self.videodir = videodir
        # Target update operation
        self.target_update = copy_vars(sess=sess, from_scope='online', to_scope='target')
        self.target_update()
        # Shared global step
        self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)
        self.ep_rewards = []
        self.ep_lengths = []
        # Creates locks
        self.global_step_lock = Lock()
        self.create_env_lock = Lock()
        self.stats_lock = Lock()

    def run(self):
        # Create workers
        threads = []
        for i_worker in range(self.num_workers):
            t = Thread(target=self._run_worker, args=(i_worker,))
            threads.append(t)
            t.daemon = True
            t.start()
        self.coord.join(threads)

    def _run_worker(self, name):
        '''
        Creates a parallel thread that runs a gym environment
        and updates the networks
        '''
        # Creates the function that calculates epsilon based on current step
        get_epsilon, final_epsilon = get_epsilon_op(self.final_epsilon_list, self.stop_exploration)

        print('Starting worker {} with final epsilon {}'.format(name, final_epsilon))
        # Starting more than one env at once may break gym
        with self.create_env_lock:
            env = AtariWrapper(self.env_name, self.num_stacked_frames)

        # Repeat until maximum steps limit is reached
        while not self.coord.should_stop():
            state = env.reset()
            experience = []
            ep_reward = 0
            for local_step in itertools.count():
                # Increment global step
                with self.global_step_lock:
                    self.sess.run(self.increment_global_step)
                    global_step_value = self.sess.run(self.global_step)
                # Compute action values with online net
                action_values = self.online_net.predict(self.sess, state[np.newaxis])
                # Compute epsilon and choose an action based on a egreedy policy
                epsilon = get_epsilon(global_step_value)
                action_probs = egreedy_policy(action_values, epsilon)
                action = np.random.choice(np.arange(self.num_actions), p=action_probs)
                # Do the action
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                reward = np.clip(reward, -1, 1)
                # Build frames history

                # Calculate simple Q learning max action value
                if self.double_learning == 'N':
                    next_action_values = self.target_net.predict(self.sess, next_state[np.newaxis])
                    next_max_action_value = np.max(next_action_values)
                # Calculate double Q learning max action value
                if self.double_learning == 'Y':
                    next_action_values = self.online_net.predict(self.sess, next_state[np.newaxis])
                    next_action = np.argmax(next_action_values)
                    next_action_values_target = self.online_net.predict(self.sess, next_state[np.newaxis])
                    next_max_action_value = np.squeeze(next_action_values_target)[next_action]
                # Calculate TD target
                if not done:
                    td_target = reward + self.discount_factor * next_max_action_value
                else:
                    td_target = reward
#                td_target = reward + (1 - done) * self.discount_factor * next_max_action_value
                # Store experience
                experience.append((state, action, td_target))

                # Update online network
                if local_step % self.online_update_step == 0 or done:
                    # Unpack experience
                    states, actions, targets = zip(*experience)
                    experience = []
                    # Updating using Hogwild! (without locks)
                    self.online_net.update(self.sess, states, actions, targets)

                # Update target network
                if global_step_value % self.target_update_step == 0:
                    print('Updating target network...')
                    self.target_update()

                # Reroll epsilon
                if global_step_value % self.change_epsilon_step == 0:
                    get_epsilon, final_epsilon = get_epsilon_op(self.final_epsilon_list, self.stop_exploration)
                    print('Final epsilon for worker {} changed to {}'.format(name, final_epsilon))

                # Write logs and checkpoint
                if global_step_value % 200000 == 0:
                    with self.stats_lock:
                        average_reward = np.mean(self.ep_rewards)
                        average_length = np.mean(self.ep_lengths)
                        self.ep_rewards = []
                        self.ep_lengths = []
                    # Run evaluation
                    print('Running evaluation...')
                    evaluation_reward, evaluation_length = self._run_evaluation()
                    print('[Average reward: {}]'.format(average_reward), end='')
                    print('[Average length: {}]'.format(average_length))
                    print('[Evaluation reward: {}]'.format(evaluation_reward), end='')
                    print('[Evaluation length: {}]'.format(evaluation_reward))
                    print('Writing summary...')
                    self.summary_writer(states, actions, targets, average_reward, average_length,
                                        evaluation_reward, evaluation_length, global_step_value)
                    print('Saving model...')
                    self.saver.save(self.sess, self.savepath)
                    print('Done!')

                # If the maximum step is reached, request all threads to stop
                if global_step_value >= self.num_steps:
                    # with self.global_step_lock:
                        self.coord.request_stop()

                # Update state
                if done:
                    break
                state = next_state

            with self.stats_lock:
                self.ep_rewards.append(ep_reward)
                self.ep_lengths.append(local_step)
            print('[Worker: {}]'.format(name), end='')
            print('[Step: {}]'.format(global_step_value), end='')
            print('[Reward: {}]'.format(ep_reward), end='')
            print('[Length: {}]'.format(local_step))

    def _run_evaluation(self):
        # Create env with monitor
        env = AtariWrapper(self.env_name, self.num_stacked_frames, self.videodir)
        state = env.reset()
        ep_reward = 0
        # Repeat until episode finish
        for i_step in itertools.count():
            # Choose an action
            action_values = self.online_net.predict(self.sess, state[np.newaxis])
            action = np.argmax(action_values)
            # Execute action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            # Update state
            if done:
                break
            state = next_state
        env.close()

        return ep_reward, i_step
