import numpy as np
import tensorflow as tf

def copy_vars(sess, from_scope, to_scope):
    '''
    Create operations to copy variables (weights) between two graphs

    Args:
        sess: The current tensorflow session
        from_scope: name of graph to copy varibles from
        to_scope: name of graph to copy varibles to
    '''
    # Get variables within defined scope
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    # Create operations that copy the variables
    op_holder = [to_var.assign(from_var) for from_var, to_var in zip(from_vars, to_vars)]

    def run_op():
        # Runs the operation
        sess.run(op_holder)

    return run_op

def egreedy_policy(q_value, epsilon):
    '''
    Returns actions probabilities based on an epsilon greedy policy
    '''
    # Sample an action
    num_actions = len(np.squeeze(q_value))
    actions = (np.ones(num_actions) * epsilon) / num_actions
    best_action = np.argmax(np.squeeze(q_value))
    actions[best_action] += 1 - epsilon
    return actions

def get_epsilon_op(final_epsilon_list, stop_exploration):
    final_epsilon = np.random.choice(final_epsilon_list, p=[0.4, 0.3, 0.3])
    epsilon_step = -np.log(final_epsilon) / stop_exploration

    def get_epsilon(step):
        # Exponentially decay epsilon until it reaches the minimum
        if step <= stop_exploration:
            new_epsilon = np.exp(-epsilon_step * step)
            return new_epsilon
        else:
            return final_epsilon
    return get_epsilon, final_epsilon
