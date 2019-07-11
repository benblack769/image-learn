import tensorflow as tf
import numpy as np
import random

from helper_models import calc_apply_grads, prod

def tf_sum(x):
    return tf.reduce_mean(x)

def tf_sumd1(x):
    return tf.reduce_mean(x,axis=1)

def sqr(x):
    return x * x

class Dense:
    def __init__(self,input_dim,out_dim,activation):
        stddev = 1.0/input_dim**(0.5**0.5)
        self.weights = tf.Variable(tf.random_normal([input_dim,out_dim], stddev=stddev),
                              name="weights")
        self.biases = tf.Variable(tf.ones(out_dim)*0.01,name="biases")
        self.activation = activation

    def calc(self,input_vec):
        linval = tf.matmul(input_vec,self.weights) + self.biases
        return (linval if self.activation is None else
                    self.activation(linval))

    def vars(self):
        return [self.weights,self.biases]

def batch_norm(input):
    means = tf.reduce_mean(input,axis=1,keepdims=True)
    input -= (means)
    mags = tf.sqrt(tf.reduce_mean(sqr(input),axis=1,keepdims=True))
    input /= (mags+0.01)
    return input

class BatchNorm:
    def __init__(self,layer_size):
        self.degrade_val = tf.float32(0.9)
        #self.layer_means = tf.Variable(tf.zeros(layer_size,dtype=tf.float32),dtype=tf.float32,name="batchnorm_means")
        #self.layer_mags = tf.Variable(tf.ones(layer_size,dtype=tf.float32),dtype=tf.float32,name="batchnorm_mags")

    def calc_no_update(self,input):
        subbed = input - self.layer_means
        divved = input / self.layer_mags
        return divved

    def update(self,input):
        new_means = self.degrade_val * self.layer_means + (1.0-self.degrade_val) * input
        self.layer_means = tf.reduce_mean(new_means,axis=1)
        new_mags_sqrd = self.degrade_val * sqr(self.layer_mags) + (1.0-self.degrade_val) * sqr(input)
        self.layer_mags = tf.sqrt(tf.reduce_mean(new_mags_sqrd,axis=1))

    def calc(self,input):
        means = tf.reduce_mean(input,axis=1)
        input -= means
        mags = tf.sqrt(tf.reduce_mean(sqr(input),axis=1))
        input /= mags
        return input

def repeat_items(tensor,num_repeats):
    input_shape = tensor.get_shape().as_list()
    reshape_d0 = tf.reshape(tensor,[input_shape[0]]+[1]+input_shape[1:])
    tiled = tf.tile(reshape_d0,[1]+[num_repeats]+[1]*(len(input_shape)-1))
    #flat_tiled = tf.reshape(tiled,[input_shape[0]*num_repeats]+input_shape[1:])
    return tiled

def batch_norm_activ(input):
    val = input
    val = batch_norm(val)
    val = tf.nn.relu(val)
    return val

class StackedDense:
    def __init__(self,num_layers,layer_size):
        self.layers = [Dense(
            input_dim=layer_size,
            out_dim=layer_size,
            activation=batch_norm_activ
        ) for _ in range(num_layers)]

    def calc(self,input):
        cur_out = input
        for dense_lay in self.layers:
            cur_out = dense_lay.calc(cur_out)
        return cur_out

    def vars(self):
        return sum((l.vars() for l in self.layers),[])

class ResetDense:
    def __init__(self,num_folds,layers_per,layer_size):
        assert num_folds > 0
        self.dense_stack = [StackedDense(layers_per,layer_size) for _ in range(num_folds)]

    def calc(self,input):
        cur_out = input
        for dense in self.dense_stack:
            cur_out = cur_out + dense.calc(cur_out)
        return cur_out

    def vars(self):
        return sum((l.vars() for l in self.dense_stack),[])


class InputProcessor:
    def __init__(self,INPUT_DIM,OUTPUT_DIM,LAYER_SIZE):
        self.input_spreader = Dense(INPUT_DIM*2,LAYER_SIZE,None)
        self.process_resnet = ResetDense(4,2,LAYER_SIZE)

    def calc(self,input1,input2):
        concat = tf.concat([input1,input2],axis=1)
        in_spread = self.input_spreader.calc(concat)
        return self.process_resnet.calc(in_spread)

    def vars(self):
        return self.input_spreader.vars() + self.process_resnet.vars()

class CriticCalculator:
    def __init__(self,ACTION_DIM,RAND_SIZE,LAYER_SIZE):
        self.combined_calc = StackedDense(2,LAYER_SIZE)
        self.eval_calc = StackedDense(1,LAYER_SIZE)
        self.eval_condenser = Dense(LAYER_SIZE,1,None)

        self.action_spreader = Dense(ACTION_DIM,LAYER_SIZE,None)
        self.action_processor = StackedDense(3,LAYER_SIZE)
        self.full_combiner = Dense(LAYER_SIZE*2,LAYER_SIZE,None)
        self.actor_critic_calc = StackedDense(3,LAYER_SIZE)
        self.advantage_condensor = Dense(LAYER_SIZE,1,None)


    def vars(self):
        return (
            self.combined_calc.vars() +
            self.eval_calc.vars() +
            self.eval_condenser.vars() +
            self.action_spreader.vars() +
            self.action_processor.vars() +
            self.full_combiner.vars() +
            self.actor_critic_calc.vars() +
            self.advantage_condensor.vars()
        )

    def calc_eval(self,input_vec):
        combined_vec = self.combined_calc.calc(input_vec)
        eval_vec = self.eval_calc.calc(combined_vec)

        eval_val = self.eval_condenser.calc(eval_vec)
        return eval_val

    def calc(self,input_vec,action):
        combined_vec = self.combined_calc.calc(input_vec)
        eval_vec = self.eval_calc.calc(combined_vec)

        eval_val = self.eval_condenser.calc(eval_vec)

        spread_action = self.action_spreader.calc(action)
        action_vec = self.action_processor.calc(spread_action)
        tot_vec = tf.concat([input_vec, action_vec],axis=1)
        combined_vec = self.full_combiner.calc(tot_vec)
        fin_vec = self.actor_critic_calc.calc(combined_vec)

        advantage_val = self.advantage_condensor.calc(fin_vec)

        return [eval_val,advantage_val]


class Actor:
    def __init__(self,RAND_SIZE,ACTION_DIM,LAYER_SIZE):
        self.input_spreader = Dense(RAND_SIZE+LAYER_SIZE,LAYER_SIZE,None)
        self.process_resnet = ResetDense(4,2,LAYER_SIZE)
        self.condenser = Dense(LAYER_SIZE,ACTION_DIM,tf.tanh)

    def calc(self,input_vec,rand_vec):
        concat = tf.concat([input_vec,rand_vec],axis=1)
        in_spread = self.input_spreader.calc(concat)
        calced_val =  self.process_resnet.calc(in_spread)
        fin_action = self.condenser.calc(calced_val)
        return fin_action

    def vars(self):
        return (self.input_spreader.vars() +
            self.process_resnet.vars() +
            self.condenser.vars())

class Distinguisher:
    def __init__(self,RAND_SIZE,ACTION_DIM,LAYER_SIZE):
        self.action_spreader = Dense(ACTION_DIM,LAYER_SIZE,None)
        self.action_process_net = StackedDense(3,LAYER_SIZE)
        self.combiner = Dense(LAYER_SIZE*2,LAYER_SIZE,None)
        self.combined_processor = StackedDense(3,LAYER_SIZE)
        self.condenser = Dense(LAYER_SIZE,1,None)
        self.randvec_pred_condenser = Dense(LAYER_SIZE,RAND_SIZE,None)

    def vars(self):
        return (self.action_spreader.vars() +
            self.action_process_net.vars() +
            self.combiner.vars() +
            self.combined_processor.vars() +
            self.condenser.vars() +
            self.randvec_pred_condenser.vars())

    def calc(self,input_vec,action):
        in_spread = self.action_spreader.calc(action)
        calced_action_vec = self.action_process_net.calc(in_spread)
        concatted = tf.concat([input_vec,calced_action_vec],axis=1)
        combined = self.combiner.calc(concatted)
        processed = self.combined_processor.calc(combined)
        fin_action = self.condenser.calc(processed)
        randvec_pred = self.randvec_pred_condenser.calc(processed)
        return fin_action,randvec_pred


class MainModel:
    def __init__(self,action_shape,observation_shape,
        LAYER_SIZE,
        RAND_SIZE):
        action_size = prod(action_shape)
        observation_size = prod(observation_shape)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.RAND_SIZE = RAND_SIZE

        self.actor_input_processor = InputProcessor(observation_size,LAYER_SIZE,LAYER_SIZE)
        self.critic_input_processor = InputProcessor(observation_size,LAYER_SIZE,LAYER_SIZE)
        self.distinguisher_input_processor = InputProcessor(observation_size,LAYER_SIZE,LAYER_SIZE)
        self.critic_calculator = CriticCalculator(action_size,RAND_SIZE,LAYER_SIZE)
        self.actor = Actor(RAND_SIZE,action_size,LAYER_SIZE)
        self.distinguisher = Distinguisher(RAND_SIZE,action_size,LAYER_SIZE)

    def calc_action(self,input1,input2,randvec):
        input_vec = self.actor_input_processor.calc(input1,input2)
        actions = self.actor.calc(input_vec,randvec)
        return actions

    def calc_eval(self,input1,input2):
        input_vec = self.critic_input_processor.calc(input1,input2)
        eval = self.critic_calculator.calc_eval(input_vec)
        return eval

    def calc_advantage(self,input1,input2,actions):
        input_vec = self.critic_input_processor.calc(input1,input2)
        eval,adv = self.critic_calculator.calc(input_vec,actions)
        return adv

    def calc_distinguish(self,input1,input2,actions,randvals):
        distinguisher_vec = self.distinguisher_input_processor.calc(input1,input2)
        is_true_logits,randvec_pred = self.distinguisher.calc(actions,randvals)
        return is_true_logits,randvec_pred

    def random_actions(self,shape):
        return tf.random_uniform(shape=shape+self.action_shape,minval=-1,maxval=1,dtype=tf.float32)

    def gen_randoms(self,size):
        rand_int = tf.random_uniform(shape=[size,self.RAND_SIZE],minval=0,maxval=1+1,dtype=tf.int32)
        rand_float = tf.cast(rand_int,tf.float32)
        return rand_float

    def calc_sample_batch(self,input1,input2,num_samples,IN_LEN):
        def spread(vals,factor=1):
            return tf.reshape(vals,[IN_LEN,num_samples*factor]+vals.get_shape().as_list()[1:])
        def flatten(vals,factor=1):
            return tf.reshape(vals,[IN_LEN*num_samples*factor]+vals.get_shape().as_list()[2:])

        gen_actions = self.calc_action(
            flatten(repeat_items(input1,num_samples)),
            flatten(repeat_items(input2,num_samples)),
            self.gen_randoms(num_samples*IN_LEN)
        )
        random_actions = self.random_actions([IN_LEN,num_samples])
        all_actions = tf.concat([spread(gen_actions),random_actions],axis=1)
        action_vals = self.calc_advantage(
            flatten(repeat_items(input1,num_samples*2),2),
            flatten(repeat_items(input2,num_samples*2),2),
            flatten(all_actions,2)
        )
        return all_actions,spread(action_vals,2)

    def critic_update(self,prev_input,cur_input,next_input,action,true_reward):
        next_eval = tf.stop_gradient(self.calc_eval(cur_input,next_input))
        true_eval = next_eval * 0.9 + true_reward

        critic_input_vector = self.critic_input_processor.calc(prev_input,cur_input)
        eval,advantage = self.critic_calculator.calc(critic_input_vector,action)

        advantage_comparitor = tf.stop_gradient((eval - true_eval))#tf.stop_gradient(tf.cast(tf.math.greater(true_eval,eval),tf.float32))
        advantage_cost = tf_sum(sqr(advantage_comparitor - advantage))
        eval_cost = tf_sum(sqr(eval - true_eval))

        tot_cost = advantage_cost + eval_cost

        critic_learning_rate = 0.001
        critic_optimzer = tf.train.RMSPropOptimizer(learning_rate=critic_learning_rate)

        all_vars = (
            self.critic_input_processor.vars() +
            self.critic_calculator.vars()
        )
        update = critic_optimzer.minimize(tot_cost,var_list=all_vars)
        return update,tot_cost

    def destinguisher_update(self,input1,input2,actions_true,IN_LEN):
        gen_randvals = self.gen_randoms(IN_LEN)
        actions_gen = self.calc_action(input1,input2,gen_randvals)

        actions = tf.concat([actions_true,actions_gen],axis=0)
        comparitors = tf.concat([tf.ones([IN_LEN,1]),tf.zeros([IN_LEN,1])],axis=0)
        info_mask = tf.concat([tf.zeros([IN_LEN,1]),tf.ones([IN_LEN,1])],axis=0)
        randvals = tf.concat([tf.zeros_like(gen_randvals),gen_randvals],axis=0)

        def tile(inpt):
            return tf.tile(inpt,[2]+[1]*(len(inpt.get_shape().as_list())-1))

        distinguisher_vec = self.distinguisher_input_processor.calc(tile(input1),tile(input2))
        is_true_logits,randvec_pred = self.distinguisher.calc(distinguisher_vec,actions)

        distinguish_cost = tf_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=is_true_logits,labels=comparitors))
        info_cost = tf_sum(info_mask*sqr(randvec_pred-randvals))
        tot_cost = distinguish_cost + info_cost

        distinguisher_learning_rate = 0.001
        distinguisher_optimzer = tf.train.RMSPropOptimizer(learning_rate=distinguisher_learning_rate)

        all_vars = (
            self.distinguisher.vars() +
            self.distinguisher_input_processor.vars()
        )
        update = distinguisher_optimzer.minimize(tot_cost,var_list=all_vars)

        return update,tot_cost

    def actor_update(self,input1,input2,IN_LEN):
        randvals = self.gen_randoms(IN_LEN)
        actions = self.calc_action(input1,input2,randvals)

        distinguisher_vec = self.distinguisher_input_processor.calc(input1,input2)
        is_true_logits,randvec_pred = self.distinguisher.calc(distinguisher_vec,actions)
        is_true_logits,randvec_pred = is_true_logits, randvec_pred

        #is_true_probs = tf.sigmoid(is_true_logits)

        is_true_cost = tf_sum(-is_true_logits) #maximize probs
        randvec_cost = tf_sum(sqr(randvec_pred - randvals))

        tot_cost = is_true_cost + randvec_cost

        actor_learning_rate = 0.0901
        actor_optimzer = tf.train.GradientDescentOptimizer(learning_rate=actor_learning_rate)

        all_vars = (
            self.actor.vars() +
            self.actor_input_processor.vars()
        )
        update = actor_optimzer.minimize(tot_cost,var_list=all_vars)
        return update,tot_cost
