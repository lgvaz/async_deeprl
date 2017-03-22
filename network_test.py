import numpy as np
import tensorflow as tf
from model_a3c import Approximator

'''
Test if the network optimization is working correctly
'''

# Create shared global step
global_step = tf.Variable(name='global_step', initial_value=0, trainable=False, dtype=tf.int32)

net = Approximator(
    env_name='Breakout-v0',
    experiment_name='Net_test',
    num_actions=3,
    optimizer_name='rms',
    learning_rate=3e-4,
    global_step=global_step,
    scope='test_net',
    clip_norm=5.0
)

# Create a artificial state
test_state = np.random.uniform(size=(1, 84, 84, 4))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Calculate predictions
    old_actions_probs = net.predict_actions_probs(sess, test_state)
    old_state_value = net.predict_state_value(sess, test_state)
    print('Old action probs: {}'.format(old_actions_probs))
    print('Old state value: {}'.format(old_state_value))
    # Optimize to a specific target
    target = [100]
    action = [0]
    for _ in range(200):
        net.update(sess, test_state, action, target)
    # Calculate new predictions
    new_actions_probs = net.predict_actions_probs(sess, test_state)
    new_state_value = net.predict_state_value(sess, test_state)
    print('New action probs: {}'.format(new_actions_probs))
    print('New state value: {}'.format(new_state_value))
