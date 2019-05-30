from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import DataCollector,BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce

NUM_ENVS = 32

def compute_template(inputs):
    return outputs

def dense_layers(input,NUM_LAYERS,LAYSIZE):
    cur_out = input
    for _ in range(NUM_LAYERS):
        cur_out = tf.layers.dense(
            inputs=cur_out,
            units=LAYSIZE,
            activation=tf.nn.relu
        )
    return cur_out

def resnet_dense(input,num_layers,layersize):
    prev_out = tf.layers.dense(
        inputs=input,
        units=layersize,
        activation=None
    )
    for _ in range(num_layers):
        prev_out = dense_layers(input,2,layersize) + prev_out

    return prev_out

def prod(l):
    p = 1
    for v in l:
        p *= v
    return v

def combine_inputs(input1,input2):
    return np.concatenate([input1,input2],axis=-1)

def main():
    all_envs = MultiProcessEnvs(NUM_ENVS)

    LAYER_SIZE = 128
    RAND_SIZE = 8

    action_shape = all_envs.action_space()
    observation_shape = all_envs.observation_space()

    @listize
    def main_reduction(input):
        return resnet_dense(input,4,LAYER_SIZE)

    @listize
    def action_generator_fn(processed_input, rand_vec):
        tot_vec = tf.concat([processed_input, rand_vec],axis=-1)
        fin_vec = resnet_dense(tot_vec,3,LAYER_SIZE)
        action_vec = tf.layers.dense(
            inputs=fin_vec,
            units=prod(action_shape),
            activation=None
        )
        return action_vec

    @listize
    def critic_combine(processed_input,action):
        combined_vec = dense_layers(processed_input,3,LAYER_SIZE)

        eval_val = tf.layers.dense(
            inputs=combined_vec,
            units=1,
            activation=None
        )

        tot_vec = tf.concat([combined_vec, action],axis=-1)
        fin_vec = resnet_dense(tot_vec,3,LAYER_SIZE)
        advantage_val = tf.layers.dense(
            inputs=fin_vec,
            units=1,
            activation=None
        )
        return eval_val,advantage_val

    observation_size = observation_shape[0]
    input_shapes = [(None,observation_size)]
    input_processor = SimpleLearner("input_processor",input_shapes,main_reduction)
    proc_input_shapes = [(None,LAYER_SIZE*2),(None,RAND_SIZE)]
    action_generator = SimpleLearner("action_generator",proc_input_shapes,action_generator_fn)
    actor_critic_input_shapes = [(None,LAYER_SIZE*2),(None,prod(action_shape))]
    actor_critic = SimpleLearner("actor_critic",actor_critic_input_shapes,critic_combine)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prev_prev_input_vec = np.zeros(shape=[NUM_ENVS,LAYER_SIZE],dtype=np.float32)
        prev_input = np.zeros(shape=[NUM_ENVS,observation_size],dtype=np.float32)
        prev_rewards = np.zeros(shape=[NUM_ENVS],dtype=np.float32)
        for t in range(10):
            current_input = all_envs.get_observations()

            [current_input_vec] = input_processor.run_calc(sess,[current_input])
            combined = combine_inputs(prev_input_vec,current_input_vec)

            rand_vec = np.random.binomial(size=[NUM_ENVS,RAND_SIZE], n=1, p=0.5)

            [actions] = action_generator.run_calc(sess,[combined,rand_vec])

            eval, advantage = actor_critic.run_calc(sess,[combined,action])

            #update state
            all_envs.set_actions(actions)

            rewards = all_envs.get_rewards()

            # calculate previous gradients

            are_new_envs = all_envs.are_new()
            for i in range(NUM_ENVS):
                if are_new_envs[i]:
                    prev_input_vec[i] = np.zeros([LAYER_SIZE],dtype=np.float32)

            filtered_eval = np.copy(eval)
            for i in range(NUM_ENVS):
                if are_new_envs[i]:
                    filtered_eval[i] = 0

            DECAY_VALUE = 0.9
            true_eval = DECAY_VALUE*filtered_eval + rewards

            actor_gradient = actor_critic.calc_input_gradients(sess,[combined,action],advantage)
            input_coalecing_gradient = action_generator.calc_input_gradients(sess,[combined,rand_vec],actor_gradient)


    all_envs.close()

if __name__ == "__main__":
    main()
