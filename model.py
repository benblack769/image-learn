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

def repeat_axis0(tensor,num_repeats):
    return tf.concat([tensor]*num_repeats,axis=0)

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

        sampled_prob = self.sampled_prob_cond.calc(fin_vec)
        radvec_prediction = self.randvec_pred_condenser.calc(fin_vec)
        better_conf = self.better_cond.calc(fin_vec)

        return [eval_val,sampled_prob,better_conf,radvec_prediction]


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



class MainModel:
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

    def calc_action(self,input1,input2,randvec):
        input_vec = self.actor_input_processor.calc(input1,input2)
        actions = self.actor.calc(input_vec,randvec)
        return actions

    def calc_eval(self,input1,input2):
        input_vec = self.critic_input_processor.calc(input1,input2)
        eval = self.critic_calculator.calc_eval(input_vec)
        return eval

    def random_actions(self,shape):
        return tf.random_uniform(shape=shape+self.action_shape,minval=-1,maxval=1,dtype=tf.float32)

    def gen_randoms(self,size):
        rand_int = tf.random_uniform(shape=[size,self.RAND_SIZE],minval=0,maxval=1+1,dtype=tf.int32)
        rand_float = tf.cast(rand_int,tf.float32)
        return rand_float

    def calc_better_prob(self,input1,input2,actions):
        input_vec = self.critic_input_processor.calc(input1,input2)
        eval,_,_,_ = self.critic_calculator.calc_eval(input_vec)
        return eval

    def evaluate_better_prob(self,input1,input2,actions):
        input_vec = self.critic_input_processor.calc(input1,input2)
        _,_,better_logits,_ = self.critic_calculator.calc(input_vec,actions)
        better_probs = tf.sigmoid(better_logits)
        return better_probs

    def evaluate_update(self,
        true_input1,
        true_input2,
        true_action,
        true_eval,
        current_randvec,
        was_true_action,
        was_good_sample):

        actor_input_vector = self.actor_input_processor.calc(true_input1,true_input2)
        critic_input_vector = self.critic_input_processor.calc(true_input1,true_input2)
        eval,sampled_logits,better_logits,randvec_pred = self.critic_calculator.calc(critic_input_vector,true_action)
        chosen_action = self.actor.calc(actor_input_vector,current_randvec)

        better_comparitor = tf.stop_gradient(tf.cast(tf.math.greater(true_eval,eval),tf.float32))
        better_cost = tf_sumd1(was_true_action * tf.nn.sigmoid_cross_entropy_with_logits(labels=better_comparitor,logits=better_logits))
        eval_cost = tf_sumd1(was_true_action * sqr(eval-true_eval))
        sampled_cost = tf_sumd1((1.0-was_true_action) * tf.nn.sigmoid_cross_entropy_with_logits(labels=was_good_sample,logits=sampled_logits))
        #advantage_comparitor = tf.stop_gradient(-(true_eval - eval))
        was_randvec_sampled = (1.0-was_true_action) * (1.0-was_good_sample)
        randvec_pred_cost = tf_sumd1(was_randvec_sampled * sqr(current_randvec - randvec_pred))

        tot_cost = (
            better_cost +
            sampled_cost +
            eval_cost +
            randvec_pred_cost
        )

        critic_learning_rate = 0.0001
        actor_learning_rate = 0.00001
        critic_optimzer = tf.train.RMSPropOptimizer(learning_rate=critic_learning_rate)
        actor_optimzer = tf.train.GradientDescentOptimizer(learning_rate=actor_learning_rate)

        _,critic_update_op,critic_grad_mag = calc_apply_grads(
            inputs=[true_input1,true_input2],
            outputs=[tot_cost],
            outputs_costs=[1.0],
            variables=self.critic_calculator.vars() + self.critic_input_processor.vars(),
            optimizer=critic_optimzer
        )
        _,actor_sampled_logits,actor_better_logits,actor_randvec_pred = self.critic_calculator.calc(critic_input_vector,chosen_action)
        actor_better_probs = tf.sigmoid(actor_better_logits)
        actor_randvec_pred_costs = 0.01*tf_sumd1(sqr(current_randvec - actor_randvec_pred))
        actor_sampled_costs = tf_sumd1(-actor_sampled_logits)
        actor_better_costs = 0#tf_sumd1(-actor_better_logits)
        total_actor_cost = actor_sampled_costs + actor_better_costs + actor_randvec_pred_costs

        _,actor_update_op,actor_grad_mag = calc_apply_grads(
            inputs=[critic_input_vector],
            outputs=[total_actor_cost],
            outputs_costs=[1.0],
            variables=self.actor.vars() + self.actor_input_processor.vars(),
            optimizer=actor_optimzer
        )

        combined_update = tf.group(
            actor_update_op,
            critic_update_op
        )
        return [combined_update,better_cost,tf_sum(better_cost),tf_sum(eval_cost),tf_sum(sampled_cost),tf_sum(randvec_pred_cost),chosen_action[0]]

    def steady_sample_action(self,input1,input2,IN_LEN):
        num_inputs = IN_LEN
        NUM_CALCS = 8
        calced_samples = self.calc_action(
            input1=repeat_axis0(input1, NUM_CALCS),
            input2=repeat_axis0(input2, NUM_CALCS),
            randvec=self.gen_randoms(num_inputs*NUM_CALCS)
        )
        NUM_RAW = 8
        NUM_SAMP = NUM_RAW + NUM_CALCS
        raw_samples = self.random_actions([num_inputs*NUM_RAW])
        action_size = prod(self.action_shape)
        reshaped_calcs = tf.reshape(calced_samples,[num_inputs,NUM_CALCS,action_size])
        reshaped_raws = tf.reshape(raw_samples,[num_inputs,NUM_RAW,action_size])
        all_samples = tf.concat([reshaped_calcs,reshaped_raws],axis=1)
        flat_samples = tf.reshape(all_samples,[NUM_SAMP*num_inputs,action_size])

        sample_probs = self.evaluate_better_prob(
            input1=repeat_axis0(input1, NUM_SAMP),
            input2=repeat_axis0(input2, NUM_SAMP),
            actions=flat_samples
        )
        target_probs = sqr(tf.random_uniform(shape=[num_inputs,NUM_SAMP],maxval=1.0))

        reshaped_probs = tf.reshape(sample_probs,[num_inputs,NUM_SAMP])
        distance_target = sqr(reshaped_probs - target_probs)
        best_choices = tf.argmax(-distance_target,axis=1,output_type=tf.int32)
        #print(reshaped_probs[0])
        #print(reshaped_probs[1])
        #print(np.asarray([reshaped_probs[samp_idx][best_idx] for (samp_idx, best_idx) in enumerate(best_choices)],dtype=np.float32))
        choice_idxs = tf.range(num_inputs,dtype=tf.int32)*NUM_SAMP+best_choices
        best_actions = tf.gather(flat_samples,choice_idxs,axis=0)
        return best_actions

    def run_update(self,stored_data,IN_LEN):
        next_eval = self.calc_eval(stored_data.cur_input,stored_data.next_input)
        true_eval = next_eval * 0.9 + stored_data.true_reward

        true_actions = stored_data.action
        sampled_actions = self.steady_sample_action(
            stored_data.prev_input,
            stored_data.cur_input,
            IN_LEN
        )
        gen_random_vals = self.gen_randoms(IN_LEN)
        gen_actions = self.calc_action(
            stored_data.prev_input,
            stored_data.cur_input,
            gen_random_vals
        )
        all_actions = tf.concat([true_actions,sampled_actions,gen_actions],axis=0)
        all_randvecs = tf.concat([self.gen_randoms(IN_LEN*2),gen_random_vals],axis=0)
        all_true_evals = tf.concat([true_eval,tf.zeros([IN_LEN,1]),tf.zeros([IN_LEN,1])],axis=0)
        was_good_sample = tf.concat([tf.zeros([IN_LEN,1]),tf.ones([IN_LEN,1]),tf.zeros([IN_LEN,1])],axis=0)
        was_true_action = tf.concat([tf.ones([IN_LEN,1]),tf.zeros([IN_LEN,1]),tf.zeros([IN_LEN,1])],axis=0)

        NUM_TILES = 3
        prev_inputs = repeat_axis0(stored_data.prev_input,NUM_TILES)
        inputs = repeat_axis0(stored_data.cur_input,NUM_TILES)

        res_list = self.evaluate_update(
            true_input1=prev_inputs,
            true_input2=inputs,
            true_action=all_actions,
            current_randvec=all_randvecs,
            true_eval=all_true_evals,
            was_true_action=was_true_action,
            was_good_sample=was_good_sample,
        )
        item_costs = res_list[1]
        input_costs = tf.reduce_mean(tf.reshape(item_costs,[NUM_TILES,IN_LEN]),axis=0)
        res_list[1] = input_costs
        return res_list
