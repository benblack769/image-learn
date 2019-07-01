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
        self.biases = tf.Variable(tf.zeros(out_dim),name="biases")
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
    input /= (mags)
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
        self.sampled_prob_cond = Dense(LAYER_SIZE,1,None)
        self.better_cond = Dense(LAYER_SIZE,1,None)
        self.randvec_pred_condenser = Dense(LAYER_SIZE,RAND_SIZE,None)


    def vars(self):
        return (
            self.combined_calc.vars() +
            self.eval_calc.vars() +
            self.eval_condenser.vars() +
            self.action_spreader.vars() +
            self.action_processor.vars() +
            self.full_combiner.vars() +
            self.actor_critic_calc.vars() +
            self.sampled_prob_cond.vars() +
            self.randvec_pred_condenser.vars() +
            self.better_cond.vars()
        )

    def calc(self,input_vec,action):
        combined_vec = self.combined_calc.calc(input_vec)
        eval_vec = self.eval_calc.calc(combined_vec)

        eval_val = self.eval_condenser.calc(eval_vec)

        spread_action = self.action_spreader.calc(action)
        action_vec = self.action_processor.calc(spread_action)
        tot_vec = tf.concat([input_vec, action_vec],axis=1)
        combined_vec = self.full_combiner.calc(tot_vec)
        fin_vec = self.actor_critic_calc.calc(combined_vec)

        sampled_prob = self.sampled_prob_cond.calc(fin_vec)
        radvec_prediction = self.randvec_pred_condenser.calc(fin_vec)
        better_conf = self.better_cond.calc(fin_vec)

        return [eval_val,sampled_prob,better_conf,radvec_prediction]

'''
class RandVecCalc:
    def __init__(self,ACTION_DIM,RAND_SIZE,LAYER_SIZE):
        self.full_combiner = Dense(LAYER_SIZE+RAND_SIZE+ACTION_DIM,LAYER_SIZE,None)
        self.main_calc = StackedDense(3,LAYER_SIZE)
        self.condenser = Dense(LAYER_SIZE,RAND_SIZE,None)


    def vars(self):
        return (
            self.full_combiner.vars() +
            self.main_calc.vars() +
            self.condenser.vars()
        )

    def calc(self,input_vec,action,randvec):
        tot_vec = tf.concat([input_vec, action, randvec],axis=1)
        combined_vec = self.full_combiner.calc(tot_vec)
        fin_vec = self.main_calc.calc(combined_vec)

        radvec_prediction = self.condenser.calc(fin_vec)

        return [eval_val,advantage_val,better_conf,prev_radvec_prediction]
'''

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



class Runner:
    def __init__(self,action_shape,observation_shape,
        LAYER_SIZE,
        RAND_SIZE):
        action_size = prod(action_shape)
        observation_size = prod(observation_shape)
        self.action_shape = action_shape
        self.RAND_SIZE = RAND_SIZE

        self.actor_input_processor = InputProcessor(observation_size,LAYER_SIZE,LAYER_SIZE)
        self.critic_input_processor = InputProcessor(observation_size,LAYER_SIZE,LAYER_SIZE)
        self.critic_calculator = CriticCalculator(action_size,RAND_SIZE,LAYER_SIZE)
        self.actor = Actor(RAND_SIZE,action_size,LAYER_SIZE)

        self.true_input1 = tf.placeholder(shape=[None,]+observation_shape,dtype=tf.float32)
        self.true_input2 = tf.placeholder(shape=[None,]+observation_shape,dtype=tf.float32)
        self.true_action = tf.placeholder(shape=[None,action_size],dtype=tf.float32)
        self.current_randvec = tf.placeholder(shape=[None,RAND_SIZE],dtype=tf.float32)
        self.true_eval = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.was_true_action = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.was_good_sample = tf.placeholder(shape=[None,1],dtype=tf.float32)

        self.actor_input_vector = self.actor_input_processor.calc(self.true_input1,self.true_input2)
        self.critic_input_vector = self.critic_input_processor.calc(self.true_input1,self.true_input2)
        self.eval,self.sampled_logits,self.better_logits,self.randvec_pred = self.critic_calculator.calc(self.critic_input_vector,self.true_action)
        self.chosen_action = self.actor.calc(self.actor_input_vector,self.current_randvec)

        self.better_probs = tf.sigmoid(self.better_logits)

        better_comparitor = tf.stop_gradient(tf.cast(tf.math.greater(self.true_eval,self.eval),tf.float32))
        self.better_cost = tf_sum(self.was_true_action * tf.nn.sigmoid_cross_entropy_with_logits(labels=better_comparitor,logits=self.better_logits))
        self.eval_cost = tf_sum(self.was_true_action * sqr(self.eval-self.true_eval))
        self.sampled_cost = tf_sum((1.0-self.was_true_action) * tf.nn.sigmoid_cross_entropy_with_logits(labels=self.was_good_sample,logits=self.sampled_logits))
        #advantage_comparitor = tf.stop_gradient(-(self.true_eval - self.eval))
        self.was_randvec_sampled = (1.0-self.was_true_action) * (1.0-self.was_good_sample)
        self.randvec_pred_cost = tf_sum(self.was_randvec_sampled * sqr(self.current_randvec - self.randvec_pred))


        tot_cost = (
            self.better_cost +
            self.sampled_cost +
            self.eval_cost +
            self.randvec_pred_cost
        )

        critic_learning_rate = 0.001
        actor_learning_rate = 0.00001
        self.critic_optimzer = tf.train.RMSPropOptimizer(learning_rate=critic_learning_rate)
        self.actor_optimzer = tf.train.GradientDescentOptimizer(learning_rate=actor_learning_rate)

        _,self.critic_update_op,self.critic_grad_mag = calc_apply_grads(
            inputs=[self.true_input1,self.true_input2],
            outputs=[tot_cost],
            outputs_costs=[1.0],
            variables=self.critic_calculator.vars() + self.critic_input_processor.vars(),
            optimizer=self.critic_optimzer
        )
        _,self.actor_sampled_logits,self.actor_better_logits,self.actor_randvec_pred = self.critic_calculator.calc(self.critic_input_vector,self.chosen_action)
        self.actor_better_probs = tf.sigmoid(self.actor_better_logits)
        self.actor_randvec_pred_costs = 0.01*tf_sumd1(sqr(self.current_randvec - self.actor_randvec_pred))
        self.actor_sampled_costs = tf_sumd1(-self.actor_sampled_logits)
        self.actor_better_costs = tf_sumd1(-self.actor_better_logits)
        self.total_actor_cost = self.actor_sampled_costs + self.actor_better_costs + self.actor_randvec_pred_costs

        _,self.actor_update_op,self.actor_grad_mag = calc_apply_grads(
            inputs=[self.critic_input_vector],
            outputs=[self.total_actor_cost],
            outputs_costs=[1.0],
            variables=self.actor.vars() + self.actor_input_processor.vars(),
            optimizer=self.actor_optimzer
        )

        self.combined_update = tf.group(
            self.actor_update_op,
            self.critic_update_op
        )

    def calc_action_data(self,sess,input1,input2,randvec):
        [chosen_action] = sess.run([self.chosen_action],feed_dict={
            self.true_input1:input1,
            self.true_input2:input2,
            self.current_randvec:randvec,
        })
        return chosen_action

    def gen_randoms(self,size):
        return np.random.uniform(size=[size,self.RAND_SIZE]).astype(np.float32)

    def evaluate_better_prob(self,sess,input1,input2,actions):
        [better_probs] = sess.run([self.better_probs],feed_dict={
            self.true_input1:input1,
            self.true_input2:input2,
            self.true_action:actions,
        })
        return better_probs

    def steady_sample_action(self,sess,input1,input2,raw_action_sampler):
        num_inputs = len(input1)
        NUM_CALCS = 8
        calced_samples = self.calc_action_data(sess=sess,
            input1=np.repeat(input1, NUM_CALCS, axis=0),
            input2=np.repeat(input2, NUM_CALCS, axis=0),
            randvec=self.gen_randoms(NUM_CALCS*num_inputs)
        )
        NUM_RAW = 8
        NUM_SAMP = NUM_RAW + NUM_CALCS
        raw_samples = np.stack([raw_action_sampler() for _ in range(NUM_RAW*num_inputs)])

        action_size = prod(self.action_shape)
        reshaped_calcs = calced_samples.reshape([num_inputs,NUM_CALCS,action_size])
        reshaped_raws = raw_samples.reshape([num_inputs,NUM_RAW,action_size])
        all_samples = np.concatenate([reshaped_calcs,reshaped_raws],axis=1)
        flat_samples = np.reshape(all_samples,[NUM_SAMP*num_inputs,action_size])

        sample_probs = self.evaluate_better_prob(sess=sess,
            input1=np.repeat(input1, NUM_SAMP, axis=0),
            input2=np.repeat(input2, NUM_SAMP, axis=0),
            actions=flat_samples
        )
        target_probs = sqr(np.random.uniform(size=[num_inputs,NUM_SAMP]))

        reshaped_probs = sample_probs.reshape([num_inputs,NUM_SAMP])
        distance_target = sqr(reshaped_probs - target_probs)
        best_choices = np.argmax(-distance_target,axis=1)
        #print(reshaped_probs[0])
        #print(reshaped_probs[1])
        #print(np.asarray([reshaped_probs[samp_idx][best_idx] for (samp_idx, best_idx) in enumerate(best_choices)],dtype=np.float32))
        best_actions = [all_samples[samp_idx][best_idx] for (samp_idx, best_idx) in enumerate(best_choices)]
        return best_actions

    def run_update(self,sess,input_dict,raw_action_sampler):
        [next_eval] = sess.run([self.eval],feed_dict={
            self.true_input1:input_dict['input'],
            self.true_input2:input_dict['next_input'],
        })
        true_eval = input_dict['reward'] + next_eval * 0.9

        IN_LEN = len(input_dict['input'])
        true_actions = input_dict['action']
        sampled_actions = self.steady_sample_action(sess,
            input_dict['prev_input'],
            input_dict['input'],
            raw_action_sampler
        )
        gen_random_vals = self.gen_randoms(IN_LEN)
        gen_actions = self.calc_action_data(sess,
            input_dict['prev_input'],
            input_dict['input'],
            gen_random_vals
        )
        all_actions = np.concatenate([true_actions,sampled_actions,gen_actions],axis=0)
        all_randvecs = np.concatenate([np.zeros([IN_LEN,self.RAND_SIZE]),np.zeros([IN_LEN,self.RAND_SIZE]),gen_random_vals],axis=0)
        all_true_evals = np.concatenate([true_eval,np.zeros([IN_LEN,1]),np.zeros([IN_LEN,1])],axis=0)
        was_good_sample = np.concatenate([np.zeros([IN_LEN,1]),np.ones([IN_LEN,1]),np.zeros([IN_LEN,1])],axis=0)
        was_true_action = np.concatenate([np.ones([IN_LEN,1]),np.zeros([IN_LEN,1]),np.zeros([IN_LEN,1])],axis=0)

        NUM_TILES = 3
        prev_inputs = np.concatenate([input_dict['prev_input']]*NUM_TILES,axis=0)
        inputs = np.concatenate([input_dict['input']]*NUM_TILES,axis=0)
        [_,better_cost,eval_cost,sampled_cost,randvec_pred_cost,actions] = sess.run([self.combined_update,self.better_cost,self.eval_cost,self.sampled_cost,self.randvec_pred_cost,self.chosen_action],feed_dict={
            self.true_input1:prev_inputs,
            self.true_input2:inputs,
            self.true_action:all_actions,
            self.current_randvec:all_randvecs,
            self.true_eval:all_true_evals,
            self.was_true_action:was_true_action,
            self.was_good_sample:was_good_sample,
        })
        return better_cost,eval_cost,sampled_cost,randvec_pred_cost,actions
