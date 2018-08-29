from ggcnn.helper import *
import tensorflow as tf
import numpy as np
import math
         
def make_variable(name, shape, initializer=tf.truncated_normal_initializer(), regularizer=None):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=dtype)
    return var
    
def make_bias_variable(name, shape):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), dtype=dtype)
    return var

def make_variable_with_weight_decay(name, shape, stddev=0.01, wd=0.0005, initializerType = "normal"):
    dtype = tf.float32
    regularizer = None
    if wd is not None and wd > 1e-7:
        def regularizer(var):
            return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    if initializerType == "normal":
        var = make_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), regularizer=regularizer)
    elif initializerType == "uniform":
        var = make_variable(name, shape, initializer=tf.random_uniform_initializer(0, maxval=stddev), regularizer=regularizer)
    elif initializerType == "identity":
        var = make_variable(name, shape, initializer=tf.initializers.identity(), regularizer=regularizer)
    return var
    
def make_bn(input, phase, axis=-1, epsilon=0.001, mask=None, num_updates=None, name=None):
    default_decay = GraphCNNGlobal.BN_DECAY
    with tf.variable_scope(name, default_name='BatchNorm') as scope:
        input_size = input.get_shape()[axis].value
        if axis == -1:
            axis = len(input.get_shape())-1
        axis_arr = [i for i in range(len(input.get_shape())) if i != axis]
        if mask == None:
            batch_mean, batch_var = tf.nn.moments(input, axis_arr)
        else:
            batch_mean, batch_var = tf.nn.weighted_moments(input, axis_arr, mask)
        gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
        beta = make_bias_variable('bias', input_size)
        ema = tf.train.ExponentialMovingAverage(decay=default_decay, num_updates=num_updates)
        
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
      
# def batch_mat_mult(A, B):
#     A_shape = tf.shape(A)
#     A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    
#     # So the Tensor has known dimensions
#     if B.get_shape()[1] == None:
#         axis_2 = -1
#     else:
#         axis_2 = B.get_shape()[1]
#     result = tf.matmul(A_reshape, B)
#     result = tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
#     return result
    
# def make_softmax_layer(V, axis=1, name=None):
#     with tf.variable_scope(name, default_name='Softmax') as scope:
#         max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
#         exp = tf.exp(tf.subtract(V, max_value))
#         prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True))
#         return prob

# def update_adjacency_weighting(V, A, global_step, distance_mat):
#     with tf.variable_scope('AdjacencyAdjustment') as scope:

#         no_features = V.get_shape()[1].value

#         W = make_variable_with_weight_decay('M_W', [no_features, 1], stddev = 0.001, wd=0.0005, initializerType = 'normal')
#         M = tf.matmul(W, tf.transpose(W))


#         d1 = tf.reduce_sum(tf.multiply(V, tf.matmul(V, M)), axis = 1, keepdims = True)
#         d2 = tf.matmul(V, tf.matmul(M, tf.transpose(V)))
#         D = tf.nn.relu( d1 + tf.transpose(d1) - tf.scalar_mul(2, d2)) + 1E-7  # Set negative values to small epsilon (sqrt gradient undefined at 0)
#         D = tf.sqrt(D)
        
#         dist_beta = make_variable_with_weight_decay('dist_beta', [1], stddev = 0.001, wd=0.0005, initializerType = 'normal')
#         D = D + dist_beta * distance_mat

#         G = tf.exp(tf.negative(D) / 10)
        
# #         gs = tf.floor(global_step / 20)
# #         decay = tf.maximum(tf.pow(0.99, gs), 0.5)
# #         A_comb = decay * A + (1 - decay) * tf.ones_like(A)
#         result = tf.multiply(A, tf.expand_dims(G,1))
#         result = tf.multiply( tf.divide(result, tf.reduce_sum(result)) , tf.reduce_sum(A))

#         return result, M, dist_beta

def make_graphcnn_layer(V, A, no_filters, name = None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        # A.shape = (N, 1, N)
        # V.shape = (N, C)
        # Shape indices decreased by 1 compared to default Graph-CNN
        # no_A = A.get_shape()[1].value
        no_A = 1
        no_features = V.get_shape()[1].value
        W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])

        A_shape = tf.shape(A)
        A_reshape = tf.reshape(A, tf.stack([A_shape[0]*no_A, A_shape[0]]))
        n = tf.matmul(A_reshape, V)
        n = tf.reshape(n, [A_shape[0], no_A*no_features])
        # result = batch_mat_mult(n, W) + batch_mat_mult(V, W_I) + b
        # n = tf.matmul(A, V)
        result = tf.matmul(n, W) + tf.matmul(V, W_I) + b
        return result
    
def make_embedding_layer(V, no_filters, name=None):
    with tf.variable_scope(name, default_name='Embed') as scope:
        no_features = V.get_shape()[-1].value
        W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=1.0/math.sqrt(no_features))
        b = make_bias_variable('bias', [no_filters])
        # V_reshape = tf.reshape(V, (-1, no_features))
        # s = tf.slice(tf.shape(V), [0], [len(V.get_shape())-1])
        # s = tf.concat([s, tf.stack([no_filters])], 0)
        # result = tf.reshape(tf.matmul(V_reshape, W) + b, s)
        result = tf.matmul(V, W) + b
        return result

def make_linkage_layer(V, V_aux, A_linkage, no_filters, name = None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        # A_linkage.shape = (N, N_aux)
        # V.shape = (N, C)
        # V_aux.shape = (N_aux, C')

        no_A = 1
        no_features = V.get_shape()[1].value
        no_features_aux = V_aux.get_shape()[1].value
        no_filters = no_features
        W = make_variable_with_weight_decay('weights', [no_features_aux, no_filters], stddev=math.sqrt(1.0/(no_features_aux*(2)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
#         W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
#         b = make_bias_variable('bias', [no_filters])
        
        n = tf.matmul(A_linkage, V_aux)

#         result = tf.matmul(n, W) + tf.matmul(V, W_I) + b
        result = tf.matmul(n, W)# + V
        return result

def make_reverse_linkage_layer(V, V_aux, A_linkage, no_filters, name = None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        # A_linkage.shape = (N, N_aux)
        # V.shape = (N, C)
        # V_aux.shape = (N_aux, C')

        no_A = 1
        no_features = V.get_shape()[1].value
        no_features_aux = V_aux.get_shape()[1].value
        no_filters = no_features_aux
        W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=math.sqrt(1.0/(no_features*(2)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
#         W_I = make_variable_with_weight_decay('weights_I', [no_features_aux, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features_aux*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
#         b = make_bias_variable('bias', [no_filters])
        
        n = tf.matmul(A_linkage, V)

#         result = tf.matmul(n, W) + tf.matmul(V_aux, W_I) + b
#         result = tf.matmul(n, W) + V_aux
        result = tf.matmul(n,W)
        return result