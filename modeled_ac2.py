from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce
from model import Runner
from state_storage import StateStorage,StateData
import time
import os

NUM_ENVS = 8

BATCH_SIZE = 32

def new_randvec(RAND_SIZE):
    return np.random.binomial(size=[NUM_ENVS,RAND_SIZE], n=1, p=0.5).astype(np.float32)

def main():
    all_envs = MultiProcessEnvs(NUM_ENVS)
    action_shape = list(all_envs.action_space())
    observation_shape = list(all_envs.observation_space())


    KEEP_SIZE = 1000
    LAYER_SIZE = 48
    RAND_SIZE = 8
    model = Runner(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)
    env_storage = StateStorage(NUM_ENVS,KEEP_SIZE)

    saver = tf.train.Saver()

    raw_action_sampler = all_envs.example_env.action_space.sample

    SAVE_NAME = "save_model/model.ckpt"
    with tf.Session() as sess:
        if os.path.exists(SAVE_NAME+".index"):
            saver.restore(sess, SAVE_NAME)
        else:
            sess.run(tf.global_variables_initializer())

        prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)

        randvec = None
        for t in range(1000000):
            current_input = all_envs.get_observations()

            #if t % 16 == 0:
            #    randvec = new_randvec(RAND_SIZE)

            actions = model.steady_sample_action(sess,current_input,prev_input,raw_action_sampler)

            rand_actions = all_envs.random_actions()
            for i in range(NUM_ENVS):
                if t < KEEP_SIZE//10 or random.random() < 0.01:
                    actions[i] = rand_actions[i]

            all_envs.set_actions(actions)

            rewards = all_envs.get_rewards()

            new_datas = [StateData(ci,act,rw)
                            for ci,act,rw in zip(current_input,actions,rewards)]
            env_storage.add_data(new_datas,[1000]*NUM_ENVS)

            prev_input = current_input

            if env_storage.should_sample():
                for i in range(1):
                    train_idxs = env_storage.sample_idxs(BATCH_SIZE)
                    train_data = env_storage.get_idxs(train_idxs)
                    #print(train_data)
                    better_cost,eval_cost,sampled_cost,randvec_pred_cost,action = model.run_update(sess,train_data,raw_action_sampler)

                    env_storage.update_idxs(train_idxs,[1000]*NUM_ENVS)

                    if t % 64 == 0:
                        template = "{0:25}{1:25}{2:25}{3:25}{4:100}"
                        print(template.format(better_cost,eval_cost,sampled_cost,randvec_pred_cost,str(action[0])))


            if t % 1024 == 0:
                saver.save(sess,SAVE_NAME)


    all_envs.close()

if __name__ == "__main__":
    main()
