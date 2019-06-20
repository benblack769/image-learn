from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce
from model import Runner
from state_storage import StateStorage,StateData

NUM_ENVS = 4

def new_randvec(RAND_SIZE):
    return np.random.binomial(size=[NUM_ENVS,RAND_SIZE], n=1, p=0.5).astype(np.float32)

def sanitize_actions(actions):
    return actions

def main():
    all_envs = Envs(NUM_ENVS)
    action_shape = list(all_envs.action_space())
    observation_shape = list(all_envs.observation_space())

    LAYER_SIZE = 128
    RAND_SIZE = 8
    model = Runner(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)
    env_storage = StateStorage()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)
        for t in range(1000000):
            current_input = all_envs.get_observations()

            randvec = new_randvec(RAND_SIZE)
            actions = model.calc_action_data(sess,current_input,prev_input,randvec)

            all_envs.set_actions(sanitize_actions(actions))

            rewards = all_envs.get_rewards()

            new_data = StateData(current_input,randvec,actions,rewards)
            env_storage.add_data(new_data)

            if env_storage.should_sample():
                train_data = env_storage.sample_train_data()
                model.run_gradient_update(sess,train_data)

            prev_input = current_input
            

    all_envs.close()

if __name__ == "__main__":
    main()
