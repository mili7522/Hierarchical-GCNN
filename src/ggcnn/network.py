from ggcnn.layers import *

class GraphCNNNetwork(object):
    def __init__(self):
        self.current_V = None
        self.current_A = None
        self.current_mask = None
        self.labels = None
        self.network_debug = False
        self.current_V_auxilary = None
        self.current_A_auxilary = None
        self.current_linkage = None
        
    def create_network(self, input):
        self.current_V = input[0]
        self.current_A = input[1]
        self.labels = input[2]
        self.current_mask = input[3]
        self.current_V_auxilary = input[4]
        self.current_A_auxilary = input[5]
        self.current_forward_linkage = input[6]
        self.current_reverse_linkage = input[7]
        self.current_mask_auxilary = input[8]
        
        if self.network_debug:
            size = tf.reduce_sum(self.current_mask, axis=1)
            self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), tf.reduce_max(size), tf.reduce_mean(size)], message='Input V Shape, Max size, Avg. Size:')
        
        return input
        
        
    def make_batchnorm_layer(self, input = None, input_type = None, mask = None):
        if input is None:
            input = getattr(self, input_type)
        if mask is not None:
            mask = getattr(self, mask)
        input = make_bn(input, self.is_training, mask = mask, num_updates = self.global_step)
        if input_type is not None:
            setattr(self, input_type, input)
        return input
    
        
    # Equivalent to 0-hop filter
    def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Embed') as scope:
            self.current_V = make_embedding_layer(self.current_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer(input_type = "current_V", mask = "current_mask")
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V, self.current_A, self.current_mask
    
    def make_auxilary_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Auxilary_Embed') as scope:
            self.current_V_auxilary = make_embedding_layer(self.current_V_auxilary, no_filters)
            if with_bn:
                self.make_batchnorm_layer(input_type = "current_V_auxilary")
            if with_act_func:
                self.current_V_auxilary = tf.nn.relu(self.current_V_auxilary)
        return self.current_V_auxilary, self.current_A_auxilary
        
    def make_dropout_layer(self, input = None, input_type = None, keep_prob=0.5):
        if input is None:
            if input_type is None:
                input_type = "current_V"
            input = getattr(self, input_type)
        input = tf.cond(self.is_training, lambda:tf.nn.dropout(input, keep_prob=keep_prob), lambda:(input))
        if input_type is not None:
            setattr(self, input_type, input)
        return input
    
        
    def make_graphcnn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            self.current_V = make_graphcnn_layer(self.current_V, self.current_A, no_filters)
            if with_bn:
                self.make_batchnorm_layer(input_type = "current_V", mask = "current_mask")
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape())-1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var], message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V

    # def make_adjacency_adjustment_layer(self, name = None):
    #     with tf.variable_scope(name, default_name='AdjacencyAdjustment') as scope: 
    #         self.current_A, self.M, self.dist_beta = update_adjacency_weighting(self.M_features, self.current_A, self.global_step, self.distance_mat)
    #     # self.M = make_variable('M', [no_features, no_features], initializer=tf.random_uniform_initializer(0, maxval=0.001))


    def make_auxilary_graphcnn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Auxilary-Graph-CNN') as scope:
            self.current_V_auxilary = make_graphcnn_layer(self.current_V_auxilary, self.current_A_auxilary, no_filters)
            if with_bn:
                self.make_batchnorm_layer(input_type = "current_V_auxilary")
            if with_act_func:
                self.current_V_auxilary = tf.nn.relu(self.current_V_auxilary)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V_auxilary, np.arange(len(self.current_V_auxilary.get_shape())-1))
                self.current_V_auxilary = tf.Print(self.current_V_auxilary, [tf.shape(self.current_V_auxilary), batch_mean, batch_var], message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V_auxilary


    def make_auxilary_linkage_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Auxilary-Linkage-Graph-CNN') as scope:
            V_linkage = make_linkage_layer(self.current_V, self.current_V_auxilary, self.current_forward_linkage, no_filters)
            
#             V_linkage = tf.cond(self.is_training, lambda:tf.nn.dropout(V_linkage, keep_prob=0.5), lambda:(V_linkage))

            if with_bn:
                V_linkage = self.make_batchnorm_layer(V_linkage, mask = "current_mask")

            self.current_V = V_linkage + self.current_V

            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V
    
    def make_reverse_auxilary_linkage_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Reverse-Auxilary-Linkage-Graph-CNN') as scope:
            V_linkage = make_reverse_linkage_layer(self.current_V, self.current_V_auxilary, self.current_reverse_linkage, no_filters)
#             V_linkage = tf.cond(self.is_training, lambda:tf.nn.dropout(V_linkage, keep_prob=0.5), lambda:(V_linkage))
            
            if with_bn:
                V_linkage = self.make_batchnorm_layer(V_linkage)
            
            self.current_V_auxilary = V_linkage + self.current_V_auxilary

            if with_act_func:
                self.current_V_auxilary = tf.nn.relu(self.current_V_auxilary)
        return self.current_V_auxilary