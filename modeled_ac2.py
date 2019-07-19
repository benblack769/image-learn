from gym_wrapper import MultiProcessEnvs,Envs
import tensorflow as tf
from helper_models import BaseLearner,SimpleLearner, listize
import numpy as np
import random
from functools import reduce
from model import MainModel
import time
import os
from state_storage import StateData,StoredData,IterStorage,DataSampler,concat_stores,sample_data_by_weights
from tensorflow.data import Dataset

NUM_ENVS = 128

BATCH_SIZE = 128

LAYER_SIZE = 128
RAND_SIZE = 8

NUM_EPOCS = 3
BATCHES_PER_EPOC = 300

BATCHES_TO_KEEP = 2500
KEEP_SIZE = NUM_ENVS*(BATCHES_TO_KEEP-2)

def change_shape(*data_tensors):
    tensors = [tf.reshape(tens,[BATCH_SIZE]+tens.get_shape().as_list()[1:]) for tens in data_tensors]
    return tuple(tensors)

def sqr(x):
    return x * x

class Runner:
    def __init__(self,model,KEEP_SIZE,BATCH_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.runner_true_input1 = tf.placeholder(shape=[NUM_ENVS,]+model.observation_shape,dtype=tf.float32)
        self.runner_true_input2 = tf.placeholder(shape=[NUM_ENVS,]+model.observation_shape,dtype=tf.float32)

        self.calc_plc_hold = StoredData(
            prev_input=tf.placeholder(shape=[BATCH_SIZE,]+model.observation_shape,dtype=tf.float32),
            cur_input=tf.placeholder(shape=[BATCH_SIZE,]+model.observation_shape,dtype=tf.float32),
            next_input=tf.placeholder(shape=[BATCH_SIZE,]+model.observation_shape,dtype=tf.float32),
            action=tf.placeholder(shape=[BATCH_SIZE,]+model.action_shape,dtype=tf.float32),
            true_reward=tf.placeholder(shape=[BATCH_SIZE,1],dtype=tf.float32)
        )

        next_eval = model.calc_eval(self.calc_plc_hold.cur_input,self.calc_plc_hold.next_input)
        cur_eval,cur_advantage = model.calc_advantage(self.calc_plc_hold.prev_input,self.calc_plc_hold.cur_input,self.calc_plc_hold.action)
        true_eval = next_eval * 0.9 + self.calc_plc_hold.true_reward
        advantage_comparitor = ((cur_eval - true_eval))
        self.adv_calc_advantage_cost = sqr(advantage_comparitor-cur_advantage)

        #gen_randvec = model.gen_randoms(NUM_ENVS)
        #self.gen_actions = model.calc_action(self.runner_true_input1,self.runner_true_input2,gen_randvec)
        RUN_NUM_SAMPLES = 8
        self.gen_actions,_ = model.calc_sample_batch(self.runner_true_input1,self.runner_true_input2,RUN_NUM_SAMPLES,NUM_ENVS)

        self.store_placeholds = StoredData(
            prev_input=tf.placeholder(shape=[KEEP_SIZE,]+model.observation_shape,dtype=tf.float32),
            cur_input=tf.placeholder(shape=[KEEP_SIZE,]+model.observation_shape,dtype=tf.float32),
            next_input=tf.placeholder(shape=[KEEP_SIZE,]+model.observation_shape,dtype=tf.float32),
            action=tf.placeholder(shape=[KEEP_SIZE,]+model.action_shape,dtype=tf.float32),
            true_reward=tf.placeholder(shape=[KEEP_SIZE,1],dtype=tf.float32)
        )
        self.dataset = (Dataset.zip(
                tuple([Dataset.from_tensor_slices(place_hold) for place_hold in self.store_placeholds.listify()])
            )
            .repeat(100)
            .shuffle(10000)
            .batch(BATCH_SIZE)
            .map(change_shape)
            .prefetch(4)
        )

        self.iterator = self.dataset.make_initializable_iterator()
        self.true_prev_input,self.true_cur_input,self.true_next_input,self.true_action,self.true_reward = self.iterator.get_next()

        TRAIN_NUM_SAMPLES = 8

        self.sample_actions,self.sample_probs = model.calc_sample_batch(self.true_prev_input,self.true_cur_input,TRAIN_NUM_SAMPLES,BATCH_SIZE)

        self.critic_update,self.critic_cost = model.critic_update(self.true_prev_input,self.true_cur_input,self.true_next_input,self.true_action,self.true_reward)

        self.distinguisher_update,self.distinguisher_cost = model.destinguisher_update(self.true_prev_input,self.true_cur_input,self.sample_actions,BATCH_SIZE)

        self.actor_update,self.actor_cost = model.actor_update(self.true_prev_input,self.true_cur_input,BATCH_SIZE)

    def run_all_updates(self,sess):
        _,critic_cost,_,dist_cost,_,actor_cost,sample_probs = sess.run([
            self.critic_update,self.critic_cost,
            self.distinguisher_update,self.distinguisher_cost,
            self.actor_update,self.actor_cost,self.sample_probs])
        return critic_cost,dist_cost,actor_cost,sample_probs

    def set_store_data(self,sess,storage_data):
        feed_dict = {place:data for place,data in zip(self.store_placeholds.listify(),storage_data.listify())}
        sess.run(self.iterator.initializer,feed_dict=feed_dict)

    def calc_action(self,sess,prev_input,cur_input):
        return sess.run(self.gen_actions,feed_dict={
            self.runner_true_input1:prev_input,
            self.runner_true_input2:cur_input,
        })

    def run_advantage_cost(self,sess,storage_data):
        feed_dict={place:val for place,val in zip(self.calc_plc_hold.listify(),storage_data.listify())}
        return sess.run(self.adv_calc_advantage_cost,feed_dict=feed_dict)

    def run_all_adv_costs(self,sess,storage_data):
        IN_SIZE = len(storage_data.cur_input)
        all_costs = []
        for x in range(0,IN_SIZE-self.BATCH_SIZE,self.BATCH_SIZE):
            store_batch_data = StoredData.from_list([d[x:x+self.BATCH_SIZE] for d in storage_data.listify()])
            all_costs.append(self.run_advantage_cost(sess,store_batch_data))
        return np.concatenate(all_costs,axis=0)

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
    all_envs =  MultiProcessEnvs(NUM_ENVS)
    action_shape = list(all_envs.action_space())
    observation_shape = list(all_envs.observation_space())

    model = MainModel(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)
    runner = Runner(model,KEEP_SIZE,BATCH_SIZE)

    saver = tf.train.Saver()
    os.makedirs("save_model",exist_ok=True)
    SAVE_NAME = "save_model/model.ckpt"

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(SAVE_NAME+".index"):
            saver.restore(sess, SAVE_NAME)

        iter_store = IterStorage()

        prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)

        all_stores = None
        for t in range(1000000):
            print("running iteration started",flush=True)
            iter_store = IterStorage()

            range_size = BATCHES_TO_KEEP if t == 0 else BATCHES_PER_EPOC
            for x in range(range_size):
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

            print("info gathering started",flush=True)

            if all_stores is None:
                all_stores = iter_store.consoladate()
            else:
                all_weights = runner.run_all_adv_costs(sess,all_stores)
                all_stores = sample_data_by_weights(all_stores,all_weights,KEEP_SIZE-NUM_ENVS*(BATCHES_PER_EPOC-2))
                all_stores = concat_stores(all_stores,iter_store.consoladate())

            runner.set_store_data(sess,all_stores)

            print("training iteration started",flush=True)
            COST_SIZE = 200
            # train
            tot_crit_cost = tot_dist_cost = tot_actor_cost = tot_prob = 0
            for x in range(int(BATCHES_PER_EPOC*NUM_EPOCS)):
                critic_cost,dist_cost,actor_cost,probs = runner.run_all_updates(sess)
                tot_crit_cost += critic_cost
                tot_dist_cost += dist_cost
                tot_actor_cost += actor_cost
                tot_prob += probs[0]
                if x % COST_SIZE == COST_SIZE-1:
                    res_str = "{0:25}{1:25}{2:25}{3:25}".format(tot_crit_cost/COST_SIZE,tot_dist_cost/COST_SIZE,tot_actor_cost/COST_SIZE,tot_prob/COST_SIZE)
                    print(res_str,flush=True)
                    tot_crit_cost = tot_dist_cost = tot_actor_cost = tot_prob = 0

            saver.save(sess,SAVE_NAME)


    all_envs.close()

if __name__ == "__main__":
    main()
