import tensorflow as tf

from helper_models import calc_apply_grads, prod

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

class StackedDense:
    def __init__(self,num_layers,layer_size):
        self.layers = [Dense(
            input_dim=layer_size,
            out_dim=layer_size,
            activation=tf.nn.relu
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
        self.input_spreader = Dense(INPUT_DIM*2,LAYER_SIZE,tf.nn.relu)
        self.process_resnet = ResetDense(4,2,LAYER_SIZE)

    def calc(self,input1,input2):
        concat = tf.concat([input1,input2],axis=1)
        in_spread = self.input_spreader.calc(concat)
        return self.process_resnet.calc(in_spread)

    def gradient_update_calc(self,input1,input2,out_costs,optimizer):
        _, var_update = calc_apply_grads(
            inputs=[input1,input2],
            outputs=[self.calc(input1,input2)],
            outputs_costs=[out_costs],
            variables=self.vars(),
            optimizer=optimizer
        )
        return var_update

    def vars(self):
        return self.input_spreader.vars() + self.process_resnet.vars()

class CriticCalculator:
    def __init__(self,ACTION_DIM,RAND_SIZE,LAYER_SIZE):
        self.name = "critic_calc"
        with tf.variable_scope(self.name):
            self.combined_calc = StackedDense(2,LAYER_SIZE)
            self.eval_calc = StackedDense(1,LAYER_SIZE)
            self.eval_condenser = Dense(LAYER_SIZE,1,None)
            self.randvec_pred_condenser = Dense(LAYER_SIZE,RAND_SIZE,tf.nn.sigmoid)

            self.full_combiner = Dense(LAYER_SIZE+RAND_SIZE+ACTION_DIM,LAYER_SIZE,None)
            self.actor_critic_calc = StackedDense(3,LAYER_SIZE)
            self.advantage_cond = Dense(LAYER_SIZE,1,None)
            self.rand_cost_cond = Dense(LAYER_SIZE,RAND_SIZE,None)

        self.myvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def vars(self):
        return self.myvars

    def calc(self,input_vec,action,randvec):
        combined_vec = self.combined_calc.calc(input_vec)
        eval_vec = self.eval_calc.calc(combined_vec)

        eval_val = self.eval_condenser.calc(eval_vec)
        prev_radvec_prediction = self.randvec_pred_condenser.calc(eval_vec)

        tot_vec = tf.concat([input_vec, action, randvec],axis=1)
        combined_vec = self.full_combiner.calc(tot_vec)
        fin_vec = self.actor_critic_calc.calc(combined_vec)

        advantage_val = self.advantage_cond.calc(fin_vec)
        current_randcosts = self.rand_cost_cond.calc(fin_vec)

        return [eval_val,advantage_val,current_randcosts,prev_radvec_prediction]

    def gradient_update_calc(self,input,action,randvec,true_eval,true_prev_randvec,true_randvec_costs):
        calc_vals = self.calc(input,action,randvec)

        [input_vec_grad], var_update = calc_apply_grads(
            inputs=[input1,input2],
            outputs=calc_vals,
            outputs_costs=[out_costs],
            variables=self.vars(),
            optimizer=optimizer
        )
        return input_vec_grad,var_update


class Actor:
    def __init__(self,RAND_SIZE,ACTION_DIM,LAYER_SIZE):
        self.input_spreader = Dense(RAND_SIZE+LAYER_SIZE,LAYER_SIZE,tf.nn.relu)
        self.process_resnet = ResetDense(4,2,LAYER_SIZE)
        self.condenser = Dense(LAYER_SIZE,ACTION_DIM,None)

    def calc(self,input_vec,rand_vec):
        concat = tf.concat([input_vec,rand_vec],axis=1)
        in_spread = self.input_spreader.calc(concat)
        calced_val =  self.process_resnet.calc(in_spread)
        fin_action = self.condenser.calc(calced_val)
        return fin_action

    def gradient_update_calc(self,input_vec,rand_vec,action_grad,optimizer):
        _, var_update = calc_apply_grads(
            inputs=[input1,input2],
            outputs=[self.calc(input_vec,rand_vec)],
            outputs_costs=[action_grad],
            variables=self.vars(),
            optimizer=optimizer
        )
        return var_update

    def vars(self):
        return (self.input_spreader.vars() +
            self.process_resnet.vars() +
            self.condenser.vars())

def tf_sum(x):
    return tf.reduce_sum(x,axis=1)

class Runner:
    def __init__(self,action_shape,observation_shape,
        LAYER_SIZE,
        RAND_SIZE):
        action_size = prod(action_shape)
        observation_size = prod(observation_shape)
        self.action_shape = action_shape
        self.RAND_SIZE = RAND_SIZE

        LAYER_SIZE = 128

        self.actor_input_processor = InputProcessor(observation_size,LAYER_SIZE,LAYER_SIZE)
        self.critic_input_processor = InputProcessor(observation_size,LAYER_SIZE,LAYER_SIZE)
        self.critic_calculator = CriticCalculator(action_size,RAND_SIZE,LAYER_SIZE)
        self.actor = Actor(RAND_SIZE,action_size,LAYER_SIZE)

        self.true_input1 = tf.placeholder(shape=[None,]+observation_shape,dtype=tf.float32)
        self.true_input2 = tf.placeholder(shape=[None,]+observation_shape,dtype=tf.float32)
        self.true_action = tf.placeholder(shape=[None,action_size],dtype=tf.float32)
        self.current_randvec = tf.placeholder(shape=[None,RAND_SIZE],dtype=tf.float32)
        self.prev_randvec = tf.placeholder(shape=[None,RAND_SIZE],dtype=tf.float32)
        self.true_randvec_costs = tf.placeholder(shape=[None,RAND_SIZE],dtype=tf.float32)
        self.true_eval = tf.placeholder(shape=[None,1],dtype=tf.float32)

        self.actor_input_vector = self.actor_input_processor.calc(self.true_input1,self.true_input2)
        self.critic_input_vector = self.critic_input_processor.calc(self.true_input1,self.true_input2)
        self.eval,self.advantage,self.randcosts,self.randvec_pred = self.critic_calculator.calc(self.critic_input_vector,self.true_action,self.current_randvec)
        self.chosen_action = self.actor.calc(self.actor_input_vector,self.current_randvec)

        eval_cost = tf_sum(sqr(self.eval-self.true_eval))
        advantage_comparitor = tf.stop_gradient(-(self.true_eval - self.eval))
        advantage_cost = tf_sum(sqr(self.advantage-advantage_comparitor))
        randvec_pred_cost = tf_sum(sqr(self.prev_randvec - self.randvec_pred))
        randcosts_pred_cost = tf_sum(sqr(self.true_randvec_costs - self.randcosts))

        learning_rate = 0.001
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        _,critic_update_op = calc_apply_grads(
            inputs=[self.true_input1,self.true_input2],
            outputs=[eval_cost,advantage_cost,randvec_pred_cost,randcosts_pred_cost],
            outputs_costs=[1.0,1.0,1.0,1.0],
            variables=self.critic_calculator.vars(),
            optimizer=self.optimizer
        )
        _,_,self.actor_advantage_val,_ = self.critic_calculator.calc(self.critic_input_vector,self.chosen_action,self.current_randvec)
        _,actor_update_op = calc_apply_grads(
            inputs=[self.true_input1,self.true_input2],
            outputs=[self.actor_advantage_val],
            outputs_costs=[1.0],
            variables=self.critic_calculator.vars(),
            optimizer=self.optimizer
        )

        self.combined_update = tf.group(
            actor_update_op,
            critic_update_op
        )


    def calc_action_data(self,sess,input1,input2,randvec):
        [chosen_action] = sess.run([self.chosen_action],feed_dict={
            self.true_input1:input1,
            self.true_input2:input2,
            self.current_randvec:randvec,
        })
        return chosen_action

    def run_gradient_update(self,sess,input_dict):
        next_eval,randvec_reconstr = sess.run([self.eval,self.randvec_pred],feed_dict={
            self.true_input1:input_dict['input'],
            self.true_input2:input_dict['next_input'],
        })
        randvec_reconstr_costs = sqr(randvec_reconstr-input_dict['cur_randvec'])

        [critic_in_grad] = sess.run([self.combined_update],feed_dict={
            self.true_input1:input_dict['prev_input'],
            self.true_input2:input_dict['input'],
            self.true_action:input_dict['action'],
            self.current_randvec:input_dict['cur_randvec'],
            self.prev_randvec:input_dict['prev_randvec'],
            self.true_eval:input_dict['reward'],
            self.true_randvec_costs:randvec_reconstr_costs,
        })