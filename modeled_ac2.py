from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce
from model import MainModel
import time
import os
from tf_state_storage import StorageAccessor,StateData,StoredData,StateQueue,data_generator


NUM_ENVS = 8

BATCH_SIZE = 32

KEEP_SIZE = 10000
LAYER_SIZE = 48
RAND_SIZE = 8

def new_randvec(RAND_SIZE):
    return np.random.binomial(size=[NUM_ENVS,RAND_SIZE], n=1, p=0.5).astype(np.float32)

class StorageManager:
    def __init__(self,storage_accessor):
        self.accessor = storage_accessor
        self.queue = StateQueue()
        observation_shape = storage_accessor.input_shape
        action_shape = storage_accessor.action_shape
        self.stored_data_t = StoredData(
            prev_input=tf.placeholder(shape=[NUM_ENVS,]+observation_shape,dtype=tf.float32),
            cur_input=tf.placeholder(shape=[NUM_ENVS,]+observation_shape,dtype=tf.float32),
            next_input=tf.placeholder(shape=[NUM_ENVS,]+observation_shape,dtype=tf.float32),
            action=tf.placeholder(shape=[NUM_ENVS,]+action_shape,dtype=tf.float32),
            true_reward=tf.placeholder(shape=[NUM_ENVS,1],dtype=tf.float32),
        )
        self.add_data_op = self.accessor.add_data(NUM_ENVS,self.stored_data_t)

    def add(self,sess,state_data):
        self.queue.add(state_data)
        if self.queue.should_get():
            stored_data = self.queue.get()
            feed_dict = {place:val for place,val in zip(self.stored_data_t.listify(),stored_data.listify())}
            sess.run(self.add_data_op,feed_dict=feed_dict)

    def get_idxs(self,idxs):
        return self.accessor.get_idxs(idxs)

class Runner:
    def __init__(self,KEEP_SIZE,NUM_ENVS,BATCH_SIZE,observation_shape,action_shape,LAYER_SIZE,RAND_SIZE):

        self.model = MainModel(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)
        self.stor_accessor = StorageAccessor(KEEP_SIZE,observation_shape,action_shape)
        self.env_storage = StorageManager(self.stor_accessor)

        self.runner_true_input1 = tf.placeholder(shape=[NUM_ENVS,]+observation_shape,dtype=tf.float32)
        self.runner_true_input2 = tf.placeholder(shape=[NUM_ENVS,]+observation_shape,dtype=tf.float32)
        self.runner_current_randvec = tf.placeholder(shape=[NUM_ENVS,RAND_SIZE],dtype=tf.float32)

        self.runner_gen_action = self.model.steady_sample_action(self.runner_true_input1,self.runner_true_input2,NUM_ENVS)

        '''self.placeholder_storeddata = StoredData(
            prev_input=tf.placeholder(shape=[BATCH_SIZE,]+observation_shape,dtype=tf.float32),
            cur_input=tf.placeholder(shape=[BATCH_SIZE,]+observation_shape,dtype=tf.float32),
            next_input=tf.placeholder(shape=[BATCH_SIZE,]+observation_shape,dtype=tf.float32),
            action=tf.placeholder(shape=[BATCH_SIZE,]+action_shape,dtype=tf.float32),
            true_reward=tf.placeholder(shape=[BATCH_SIZE,1],dtype=tf.float32),
        )'''
        self.access_storeddata = data_generator(self.stor_accessor,BATCH_SIZE)

        self.train_update,self.better_cost,self.eval_cost,self.sampled_cost,self.randvec_pred_cost,self.actions = self.model.run_update(self.access_storeddata,BATCH_SIZE)

    def run_train_iter(self,sess):
        #feed_dict = {place:val for place,val in zip(self.placeholder_storeddata.listify(),self.access_storeddata.listify())}
        vals = sess.run([self.train_update,self.better_cost,self.eval_cost,self.sampled_cost,self.randvec_pred_cost,self.actions])
        cost_vals = vals[1:]
        return cost_vals

    def calc_action(self,sess,prev_input,input,randvec):
        actions = sess.run(self.runner_gen_action,feed_dict={
            self.runner_true_input1:prev_input,
            self.runner_true_input2:input,
            self.runner_current_randvec:randvec,
        })
        return actions

def main():
    all_envs = MultiProcessEnvs(NUM_ENVS)
    action_shape = list(all_envs.action_space())
    observation_shape = list(all_envs.observation_space())


    runner = Runner(KEEP_SIZE,NUM_ENVS,BATCH_SIZE,observation_shape,action_shape,LAYER_SIZE,RAND_SIZE)

    saver = tf.train.Saver()

    SAVE_NAME = "save_model/model.ckpt"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(SAVE_NAME+".index"):
            saver.restore(sess, SAVE_NAME)

        prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)

        randvec = None
        for t in range(1000000):
            current_input = all_envs.get_observations()

            if t % 1 == 0:
                randvec = new_randvec(RAND_SIZE)

            actions = runner.calc_action(sess,current_input,prev_input,randvec)

            rand_actions = all_envs.random_actions()
            for i in range(NUM_ENVS):
                if t < KEEP_SIZE//10:
                    actions[i] = rand_actions[i]

            all_envs.set_actions(actions)

            rewards = all_envs.get_rewards()

            new_datas = StateData(current_input,actions,rewards)
            runner.env_storage.add(sess,new_datas)

            prev_input = current_input

            if t > KEEP_SIZE//10:
                for i in range(1):
                    better_cost,eval_cost,sampled_cost,randvec_pred_cost,action = runner.run_train_iter(sess)

                    if t % 64 == 0:
                        template = "{0:25}{1:25}{2:25}{3:25}{4:100}"
                        print(template.format(better_cost,eval_cost,sampled_cost,randvec_pred_cost,str(action[0])))


            if (t+1) % 1024 == 0:
                saver.save(sess,SAVE_NAME)


    all_envs.close()

if __name__ == "__main__":
    main()
