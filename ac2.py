import numpy as np
import random
from gym_wrapper import MultiProcessEnvs,Envs
from loss_ballancing import TensorLossBallancer

NUM_ENVS = 32

all_envs = MultiProcessEnvs(NUM_ENVS)

import tensorflow as tf

def init_envs():
    for _ in range(50):
        all_envs.set_actions(all_envs.random_actions())

def main_input_reduction(input):
    DEPTH = 6
    lay1size = 16
    CONV1_SIZE=[3,3]
    POOL_SIZE=[2,2]
    POOL_STRIDES=[2,2]
    orig_reduction = tf.layers.dense(
        inputs=input,
        units=lay1size,
        activation=tf.nn.relu
    )
    cur_out = orig_reduction
    for x in range(DEPTH):
        lay1_outs = tf.layers.conv2d(
            inputs=cur_out,
            filters=lay1size,
            kernel_size=CONV1_SIZE,
            padding="same",
            activation=tf.nn.relu)
        lay_1_pool = tf.layers.average_pooling2d(
            inputs=lay1_outs,
            pool_size=POOL_SIZE,
            strides=POOL_STRIDES,
            padding='same',
        )
        #basic_outs.append(lay1_outs)
        cur_out = lay_1_pool
        print(cur_out.shape)

    cur_out = tf.layers.flatten(cur_out)
    fc_layer_size = 128
    cur_out = tf.layers.dense(
        inputs=cur_out,
        units=fc_layer_size,
        activation=tf.nn.relu
    )
    cur_out = tf.layers.dense(
        inputs=cur_out,
        units=fc_layer_size,
        activation=tf.nn.relu
    )
    return cur_out

class AdvantageModel:
    def __init__(self):
        self.num_lays = 2
        fc_layer_size = 128
        self.lay1 = tf.layers.Dense(
            units=fc_layer_size,
            activation=tf.nn.relu,
            name="hithere_MY_NAME_IS_BENJAMIN_BLACK"
        )
        self.lay2 = tf.layers.Dense(
            units=fc_layer_size,
            activation=tf.nn.relu
        )
        self.lay3 = tf.layers.Dense(
            units=1,
            activation=None
        )

    def val(self,action_vec,input_vec):
        tot_vec = tf.concat([action_vec,input_vec],axis=1)
        cur_out = self.lay1(tot_vec)
        cur_out = self.lay2(cur_out)
        cur_out = self.lay3(cur_out)

        return cur_out

def sqr(x):
    return x * x

def value_cost(aprox_val, true_val):
    return tf.reduce_sum(sqr(aprox_val - true_val))


def main():
    input_shape = all_envs.observation_space()
    input_img = tf.placeholder(tf.float32,(None,)+input_shape)
    input_action = tf.placeholder(tf.float32,(None,1))
    value_comparitor = tf.placeholder(tf.float32,(None,1))
    #input_img = tf.ones(dtype=tf.float32,shape=(32,)+input_shape)*2

    with tf.variable_scope("main_opt"):
        input_vec = main_input_reduction(input_img)

        action_vec = tf.layers.dense(
            inputs=input_vec,
            units=1,
            activation=None,
        )
        action_vec = tf.nn.sigmoid(action_vec) * 5
        eval_val = tf.layers.dense(
            inputs=input_vec,
            units=1,
            activation=None
        )

    with tf.variable_scope("critic_opt"):
        advant_model = AdvantageModel()
        input_vec_critic = main_input_reduction(input_img)
        critic_val = advant_model.val(action_vec,input_vec_critic)

    #critic_val_no_grad = advant_model.val(tf.stop_gradient(action_vec),input_vec_critic)

    value_loss = value_cost(eval_val,value_comparitor)

    advantage_value = critic_val #- tf.stop_gradient(eval_val))
    advantage_comparitor = -(value_comparitor - tf.stop_gradient(eval_val))

    advantage_loss = tf.reduce_sum(advantage_value)

    critic_loss = value_cost(advantage_value,advantage_comparitor)

    main_collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_opt')
    critic_collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_opt')

    #print("main_collection")
    #print(main_collection)
    #print("critic_collection")
    #print(critic_collection)

    VALUE_COST_WEIGHT = 0.2
    MINIMIZE_COST_WEIGHT = 1.0-VALUE_COST_WEIGHT
    ballancer = TensorLossBallancer(VALUE_COST_WEIGHT,MINIMIZE_COST_WEIGHT,ADAPT_COEF=0.95,MAG_ADJUST_COEF=0.99)

    bal_val_loss,bal_min_loss,ballancer_ops = ballancer.adjust(value_loss,advantage_loss)

    ballanced_minimizer_cost = advantage_loss + value_loss#bal_val_loss + bal_min_loss

    ADAM_learning_rate = 0.001
    optimizer_critic = tf.train.RMSPropOptimizer(learning_rate=ADAM_learning_rate)
    optimizer_min = tf.train.RMSPropOptimizer(learning_rate=ADAM_learning_rate)
    critic_opt = optimizer_critic.minimize(critic_loss,var_list=critic_collection)

    #optimizer_min = tf.train.GradientDescentOptimizer(learning_rate=ADAM_learning_rate)
    minimizer_opt = optimizer_min.minimize(ballanced_minimizer_cost,var_list=main_collection)

    tf.summary.scalar('advantage_loss', advantage_loss)
    tf.summary.scalar('value_loss', value_loss)
    tf.summary.scalar('critic_loss', critic_loss)
    tf.summary.histogram('actions', action_vec)
    merged = tf.summary.merge_all()

    #assval = tf.zeros(dtype=tf.float32,shape=(128,))
    with tf.Session() as sess:
        #train_writer = tf.summary.FileWriter('./train',
        #                                      sess.graph)
        sess.run(tf.global_variables_initializer())

        sum_idx = 0
        for epoc in range(100000):
            all_inputs = []
            all_evals = []

            tot_val_loss = 0
            tot_advant_loss = 0
            tot_crit_loss = 0
            num_steps = 1
            for step in range(num_steps+1):
                # step enviornments
                new_input = all_envs.get_observations()

                #print("started action")
                # new action generator
                new_action_vec,new_eval_val = sess.run([action_vec,eval_val],feed_dict={
                    input_img: new_input,
                })
                new_actions = all_envs.manipulate_actions(new_action_vec)

                all_envs.set_actions(new_actions)
                action_rewards = all_envs.get_rewards()
                #print("executed action")

                if step != num_steps:
                    all_inputs.append(new_input)
                if step != 0:
                    DEGRADE_VAL = 0.9
                    true_rewards = action_rewards + DEGRADE_VAL*new_eval_val
                    for j in range(NUM_ENVS):
                        print(new_actions[j],"\t",new_action_vec[j][0],"\t",true_rewards[j][0])
                    all_evals.append(true_rewards)

            for step in range(num_steps):
                cur_input = all_inputs[step]
                true_reward = all_evals[step]
                # minimizer train step:
                _,_, val_loss_val,advantage_val,_,crit_loss_val = sess.run([minimizer_opt,ballancer_ops,value_loss,advantage_loss,critic_opt,critic_loss],feed_dict={
                    input_img: cur_input,
                    value_comparitor: true_reward,
                })

                print("min2")

                tot_val_loss += val_loss_val
                tot_advant_loss += advantage_val
                tot_crit_loss += crit_loss_val

            #train_writer.add_summary(summary, sum_idx)
            sum_idx += 1
            print("current loss totals:")
            print(tot_val_loss/num_steps)
            print(tot_advant_loss/num_steps)
            print(tot_crit_loss/num_steps)

        #sess.run(tf.assign(tf.get_variable(''))
        #for x in range(1000000000):
        pass




if __name__ == "__main__":
    main()