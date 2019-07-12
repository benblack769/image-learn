from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce
from model import MainModel
import time
import os
from state_storage import StateData,IterStorage,DataSampler,concat_stores


NUM_ENVS = 32

BATCH_SIZE = 64

KEEP_SIZE = 100000
LAYER_SIZE = 80
RAND_SIZE = 8

NUM_EPOCS = 5
BATCHES_PER_EPOC = 3000


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

        sample_batch_actions,sample_batch_logits = model.calc_sample_batch(self.true_prev_input,self.true_cur_input,NUM_SAMPLES,BATCH_SIZE)
        sample_idxs = tf.multinomial(sample_batch_logits,1)
        sample_idxs = tf.reshape(sample_idxs,[BATCH_SIZE,])
        #self.sample_probs = tf.nn.softmax(sample_batch_logits,axis=1)
        #self.probs = tf.gather_nd(self.sample_probs,sample_idxs)
        flat_actions = tf.reshape(sample_batch_actions,[BATCH_SIZE*NUM_SAMPLES*2,]+model.action_shape)
        flat_sample_idxs = tf.range(BATCH_SIZE,dtype=tf.int64)*NUM_SAMPLES*2 + sample_idxs

        flat_probs = tf.reshape(tf.nn.softmax(sample_batch_logits),[BATCH_SIZE*NUM_SAMPLES*2])
        self.sample_probs = tf.reduce_mean(tf.gather(flat_probs,flat_sample_idxs,axis=0))

        self.sample_actions = tf.gather(flat_actions,flat_sample_idxs,axis=0)

        self.critic_update,self.critic_cost = model.critic_update(self.true_prev_input,self.true_cur_input,self.true_next_input,self.true_action,self.true_reward)

        self.distinguisher_update,self.distinguisher_cost = model.destinguisher_update(self.true_prev_input,self.true_cur_input,self.sample_actions,BATCH_SIZE)

        self.actor_update,self.actor_cost = model.actor_update(self.true_prev_input,self.true_cur_input,BATCH_SIZE)

    def run_all_updates(self,sess,storage_data):
        _,critic_cost,_,dist_cost,_,actor_cost,sample_probs = sess.run([
            self.critic_update,self.critic_cost,
            self.distinguisher_update,self.distinguisher_cost,
            self.actor_update,self.actor_cost,self.sample_probs],

            feed_dict={
            self.true_prev_input:storage_data.prev_input,
            self.true_cur_input:storage_data.cur_input,
            self.true_next_input:storage_data.next_input,
            self.true_action:storage_data.action,
            self.true_reward:storage_data.true_reward,
        })
        return critic_cost,dist_cost,actor_cost,sample_probs

    def calc_action(self,sess,prev_input,cur_input):
        return sess.run(self.gen_actions,feed_dict={
            self.runner_true_input1:prev_input,
            self.runner_true_input2:cur_input,
        })

    def run_sampler(self,sess,storage_data):
        chosen_actions = []
        chosen_values = []
        for action_set,prob_set in zip(actions,probs):
            idx = np.random.choice(len(prob_set),p=prob_set)
            action = action_set[idx]
            chosen_actions.append(action)
            chosen_values.append(prob_set[idx])

        return np.stack(chosen_actions),chosen_values

def main():
    all_envs = MultiProcessEnvs(NUM_ENVS)
    action_shape = list(all_envs.action_space())
    observation_shape = list(all_envs.observation_space())

    model = MainModel(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)
    runner = Runner(model,BATCH_SIZE)

    saver = tf.train.Saver()
    os.makedirs("save_model",exist_ok=True)
    SAVE_NAME = "save_model/model.ckpt"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(SAVE_NAME+".index"):
            saver.restore(sess, SAVE_NAME)

        iter_store = IterStorage()

        prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)

        all_stores = None
        for t in range(1000000):
            print("running iteration started",flush=True)
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

            all_stores = concat_stores(all_stores,iter_store.consoladate()) if all_stores is not None else iter_store.consoladate()
            sampler = DataSampler(all_stores)

            print("training iteration started",flush=True)
            COST_SIZE = 1024
            # train
            tot_crit_cost = tot_dist_cost = tot_actor_cost = tot_prob = 0
            for x in range(int(BATCHES_PER_EPOC*NUM_EPOCS)):
                cur_batch = sampler.get_batch(BATCH_SIZE)
                critic_cost,dist_cost,actor_cost,probs = runner.run_all_updates(sess,cur_batch)
                tot_crit_cost += critic_cost
                tot_dist_cost += dist_cost
                tot_actor_cost += actor_cost
                tot_prob += probs
                if x % COST_SIZE == COST_SIZE-1:
                    res_str = "{0:25}{1:25}{2:25}{3:25}".format(tot_crit_cost/COST_SIZE,tot_dist_cost/COST_SIZE,tot_actor_cost/COST_SIZE,tot_prob/COST_SIZE)
                    print(res_str,flush=True)
                    tot_crit_cost = tot_dist_cost = tot_actor_cost = tot_prob = 0

            saver.save(sess,SAVE_NAME)


    all_envs.close()

if __name__ == "__main__":
    main()
