import tensorflow as tf
import numpy as np

def calc_apply_grads(inputs,outputs,outputs_costs,variables,optimizer):
    assert len(outputs_costs) == len(outputs)
    '''
    we wish to minimize sum(outputs_costs) wrt. inputs

    i.e.
    dout/din = sum(outputs_costs)

    solve the inverse derivative and we must put in
    y = sum(outputs_costs) * outputs

    into the tf.gradients function
    '''
    total_outputs = [out*outcost for out,outcost in zip(outputs_costs,outputs)]

    input_tensors = inputs + variables

    tensors_grads = tf.gradients(
        ys=total_outputs,
        xs=input_tensors
    )
    inputs_grads = tensors_grads[:len(inputs)]
    variables_grads = tensors_grads[len(inputs):]

    var_grad_pairs = zip(variables_grads,variables)

    var_update_op = optimizer.apply_gradients(var_grad_pairs)
    return inputs_grads,var_update_op


class BaseLearner:
    def __init__(self,inputs,outputs,var_collection):
        assert isinstance(inputs,list)
        assert isinstance(outputs,list)
        assert isinstance(var_collection,list)

        self.variables = var_collection
        self.inputs = inputs
        self.outputs = outputs
        learning_rate = 0.001
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        self.output_costs = [tf.placeholder(shape=out.shape,dtype=tf.float32) for out in outputs]
        self.in_grads,self.var_update = self._calc_and_apply_gradients(self.output_costs)

    def num_inputs(self):
        return len(self.inputs)

    def _calc_gradients(self,outputs_costs):
        assert len(outputs_costs) == len(self.outputs)
        '''
        we wish to minimize sum(outputs_costs) wrt. inputs

        i.e.
        dout/din = sum(outputs_costs)

        solve the inverse derivative and we must put in
        y = sum(outputs_costs) * outputs

        into the tf.gradients function
        '''
        total_outputs = [out*outcost for out,outcost in zip(outputs_costs,self.outputs)]

        input_tensors = self.inputs + self.variables

        tensors_grads = tf.gradients(
            ys=total_outputs,
            xs=input_tensors
        )
        inputs_grads = tensors_grads[:len(self.inputs)]
        variables_grads = tensors_grads[len(self.inputs):]
        return inputs_grads,variables_grads

    def _apply_var_gradients(self,var_grads):
        assert len(var_grads) == len(self.variables)
        var_grad_pairs = zip(var_grads,self.variables)
        return self.optimizer.apply_gradients(var_grad_pairs)

    def _calc_and_apply_gradients(self,output_costs):
        in_grads,var_grads = self._calc_gradients(output_costs)
        return in_grads,self._apply_var_gradients(var_grads)

    def run_calc(self,sess,input_values):
        feed_dict = {inp:val for inp,val in zip(self.inputs, input_values)}
        return sess.run(self.outputs,feed_dict=feed_dict)

    def get_feed_dict(self,inputs_values,output_costs):
        input_feed_dict = {inp:val for inp,val in zip(self.inputs, input_values)}
        out_cost_feed_dict = {oc:val for oc,val in zip(self.output_costs, output_costs)}
        total_dict = {**input_feed_dict, **out_cost_feed_dict}
        return total_dict

    def run_calc_apply_gradients(self,sess,inputs_values,output_costs):
        all = sess.run([self.var_update]+self.in_grads,feed_dict=self.get_feed_dict(inputs_values,output_costs))
        in_grads = all[:len(input_feed_dict)]
        return in_grads

    def calc_input_gradients(self,sess,inputs_values,output_costs):
        return sess.run(self.in_grads,feed_dict=self.get_feed_dict(inputs_values,output_costs))


def prod(l):
    p = 1
    for v in l:
        p *= v
    return v

def listize(scalarfn):
    '''
    Turn a function that has multiple arguments and returns a scalar or tuple to
    a function that accepts a list and returns a list
    '''
    def listized(inputs):
        outs = scalarfn(*inputs)
        if isinstance(outs,tuple):
            outs = list(outs)
        elif not isinstance(outs,list):
            outs = [outs]
        return outs
    return listized

class SimpleLearner(BaseLearner):
    def __init__(self,name,input_shapes,process_fn):
        inputs = [tf.placeholder(shape=s,dtype=tf.float32) for s in input_shapes]

        with tf.variable_scope(name):
            outputs = process_fn(inputs)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        BaseLearner.__init__(self,inputs,outputs,variables)
