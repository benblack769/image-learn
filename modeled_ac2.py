from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce
from model import MainModel
import time
import os
from state_storage import StateData,IterStorage,Sampler


NUM_ENVS = 32

BATCH_SIZE = 64

KEEP_SIZE = 100000
LAYER_SIZE = 48
RAND_SIZE = 8

NUM_EPOCS = 10
BATCHES_PER_EPOC = 5000


class Runner:
    def __init__(self,model,BATCH_SIZE):
        self.runner_true_input1 = tf.placeholder(shape=[NUM_ENVS,]+model.observation_shape,dtype=tf.float32)
        self.runner_true_input2 = tf.placeholder(shape=[NUM_ENVS,]+model.observation_shape,dtype=tf.float32)

        gen_randvec = model.gen_randoms(NUM_ENVS)
        self.gen_actions = model.calc_action(self.runner_true_input1,self.runner_true_input2,gen_randvec)

        self.true_prev_input = tf.placeholder(shape=[BATCH_SIZE,]+model.observation_shape,dtype=tf.float32)
        self.true_cur_input = tf.placeholder(shape=[BATCH_SIZE,]+model.observation_shape,dtype=tf.float32)
        self.true_next_input = tf.placeholder(shape=[BATCH_SIZE,]+model.observation_shape,dtype=tf.float32)
        self.true_action = tf.placeholder(shape=[BATCH_SIZE,]+model.action_shape,dtype=tf.float32)
        self.true_reward = tf.placeholder(shape=[BATCH_SIZE,1],dtype=tf.float32)

        NUM_SAMPLES = 8

        self.sample_batch_actions,self.sample_batch_values = model.calc_sample_batch(self.true_prev_input,self.true_cur_input,NUM_SAMPLES,BATCH_SIZE)
        self.sample_softmax = tf.nn.softmax(self.sample_batch_values,axis=1)

        self.critic_update,self.critic_cost = model.critic_update(self.true_prev_input,self.true_cur_input,self.true_next_input,self.true_action,self.true_reward)

        self.distinguisher_update,self.distinguisher_cost = model.destinguisher_update(self.true_prev_input,self.true_cur_input,self.true_action,BATCH_SIZE)

        self.actor_update,self.actor_cost = model.actor_update(self.true_prev_input,self.true_cur_input,BATCH_SIZE)


    def run_critic_update(self,sess,storage_data):
        _,cost = sess.run([self.critic_update,self.critic_cost],feed_dict={
            self.true_prev_input:storage_data.prev_input,
            self.true_cur_input:storage_data.cur_input,
            self.true_next_input:storage_data.next_input,
            self.true_action:storage_data.action,
            self.true_reward:storage_data.true_reward,
        })
        return cost

    def run_actor_update(self,sess,storage_data):
        _,cost = sess.run([self.actor_update,self.actor_cost],feed_dict={
            self.true_prev_input:storage_data.prev_input,
            self.true_cur_input:storage_data.cur_input,
        })
        return cost

    def run_distinguisher_update(self,sess,storage_data):
        _,cost = sess.run([self.distinguisher_update,self.distinguisher_cost],feed_dict={
            self.true_prev_input:storage_data.prev_input,
            self.true_cur_input:storage_data.cur_input,
            self.true_action:storage_data.action,
        })
        return cost

    def calc_action(self,sess,prev_input,cur_input):
        return sess.run(self.gen_actions,feed_dict={
            self.runner_true_input1:prev_input,
            self.runner_true_input2:cur_input,
        })

    def run_sampler(self,sess,storage_data):
        actions,probs = sess.run([self.sample_batch_actions,self.sample_softmax],feed_dict={
            self.true_prev_input:storage_data.prev_input,
            self.true_cur_input:storage_data.cur_input,
        })
        actions = []
        values = []
        for action_set,prob_set in zip(actions,probs):
            idx = np.random.choice(len(prob_set),p=prob_set)
            action = action_set[idx]
            actions.append(action)
            values.append(prob_set[idx])

        return actions,values

def main():
    all_envs = MultiProcessEnvs(NUM_ENVS)
    action_shape = list(all_envs.action_space())
    observation_shape = list(all_envs.observation_space())

    model = MainModel(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)
    runner = Runner(model,BATCH_SIZE)

    def unif_gen_actions():
        return np.random.uniform(low=-1.0,high=1.0,size=[NUM_ENVS,]+action_shape)

    gen_action_fn = unif_gen_actions

    saver = tf.train.Saver()
    os.makedirs("save_model",exist_ok=True)
    SAVE_NAME = "save_model/model.ckpt"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(SAVE_NAME+".index"):
            saver.restore(sess, SAVE_NAME)

        iter_store = IterStorage()

        prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)

        for t in range(1000000):
            iter_store = IterStorage()
            for x in range(BATCHES_PER_EPOC):
                current_input = all_envs.get_observations()
                #randvec = new_randvec(RAND_SIZE)

                if t == 0:
                    actions = all_envs.random_actions()
                else:
                    actions = runner.calc_action(sess,current_input,prev_input)

                all_envs.set_actions(actions)

                rewards = all_envs.get_rewards()

                new_datas = StateData(current_input,actions,rewards)
                iter_store.add(new_datas)

                prev_input = current_input

            sampler = Sampler(iter_store.consoladate())
            del iter_store

            COST_SIZE = 128
            # train
            tot_cost = 0
            for x in range(BATCHES_PER_EPOC*NUM_EPOCS):
                cost = runner.run_critic_update(sess,sampler.get_batch(BATCH_SIZE))
                tot_cost += cost
                if x % COST_SIZE == COST_SIZE-1:
                    print(tot_cost/COST_SIZE)
                    tot_cost = 0

            tot_cost = 0
            for x in range(BATCHES_PER_EPOC*NUM_EPOCS):
                cost = runner.run_distinguisher_update(sess,sampler.get_batch(BATCH_SIZE))
                tot_cost += cost
                if x % COST_SIZE == COST_SIZE-1:
                    print(tot_cost/COST_SIZE)
                    tot_cost = 0

            tot_cost = 0
            for x in range(BATCHES_PER_EPOC*NUM_EPOCS):
                cost = runner.run_distinguisher_update(sess,sampler.get_batch(BATCH_SIZE))
                tot_cost += cost
                if x % COST_SIZE == COST_SIZE-1:
                    print(tot_cost/COST_SIZE)
                    tot_cost = 0

            saver.save(sess,SAVE_NAME)


    all_envs.close()

if __name__ == "__main__":
    main()
