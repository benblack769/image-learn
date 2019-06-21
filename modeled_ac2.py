from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce
from model import Runner
from state_storage import StateStorage,StateData
import time

NUM_ENVS = 8

BATCH_SIZE = 32

def new_randvec(RAND_SIZE):
    return np.random.binomial(size=[NUM_ENVS,RAND_SIZE], n=1, p=0.5).astype(np.float32)


def main():
    all_envs = MultiProcessEnvs(NUM_ENVS)
    action_shape = list(all_envs.action_space())
    observation_shape = list(all_envs.observation_space())

    def sanitize_actions(actions):
        print("actions")
        print(actions)
        actions = np.maximum(actions,all_envs.example_env.action_space.low+0.1)
        print(actions)
        actions = np.minimum(actions,all_envs.example_env.action_space.high-0.1)
        print(actions)
        return actions

    KEEP_SIZE = 2000
    LAYER_SIZE = 64
    RAND_SIZE = 8
    model = Runner(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)
    env_storage = StateStorage(NUM_ENVS,KEEP_SIZE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)

        for t in range(KEEP_SIZE+5):
            current_input = all_envs.get_observations()

            randvec = new_randvec(RAND_SIZE)

            actions = all_envs.random_actions()

            all_envs.set_actions(actions)

            rewards = all_envs.get_rewards()

            new_datas = [StateData(ci,rv,act,rw)
                            for ci,rv,act,rw in zip(current_input,randvec,actions,rewards)]
            env_storage.add_data(new_datas)

            prev_input = current_input

        '''for t in range(1000):
            train_data = env_storage.sample_train_data(BATCH_SIZE)
            true_reward,eval_cost,eval,adv,randc_est,action = model.run_critic_update(sess,train_data)
            if t % 64 >= 0:
                template = "{0:25}{1:25}{2:25}{3:25}{4:25}{5:100}"
                print(template.format(true_reward,eval_cost,eval,adv,randc_est,str(action)))
        print("finished critic calc")'''

        for t in range(1000000):
            current_input = all_envs.get_observations()

            randvec = new_randvec(RAND_SIZE)
            actions = model.calc_action_data(sess,current_input,prev_input,randvec)

            rand_actions = all_envs.random_actions()
            for i in range(NUM_ENVS):
                if random.random() < 0.1:
                    actions[i] = rand_actions[i]

            all_envs.set_actions(actions)

            rewards = all_envs.get_rewards()

            new_datas = [StateData(ci,rv,act,rw)
                            for ci,rv,act,rw in zip(current_input,randvec,actions,rewards)]
            env_storage.add_data(new_datas)

            if env_storage.should_sample():
                for i in range(10):
                    train_data = env_storage.sample_train_data(BATCH_SIZE)
                    true_reward,eval_cost,eval,adv,randc_est,action = model.run_gradient_update(sess,train_data)
                    if t % 64 >= 0:
                        template = "{0:25}{1:25}{2:25}{3:25}{4:25}{5:100}"
                        print(template.format(true_reward,eval_cost,eval,adv,randc_est,str(action)))

            prev_input = current_input


    all_envs.close()

if __name__ == "__main__":
    main()
