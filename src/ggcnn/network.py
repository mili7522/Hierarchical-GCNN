from ggcnn.layers import *

class GraphCNNNetwork(object):
    def __init__(self):
        self.inputs = None
        self.network_debug = False

        
    def create_network(self, input):
        self.current_mask = input['mask']
        self.labels = input['labels']
        self.current_values = input
        
        # if self.network_debug:
        #     size = tf.reduce_sum(self.current_mask, axis=1)
        #     self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), tf.reduce_max(size), tf.reduce_mean(size)], message='Input V Shape, Max size, Avg. Size:')
        
        return input
        
        
    def make_batchnorm_layer(self, input_type = 'V', l = 0):
        if (l == 0) and (input_type == 'V'):
            mask = self.current_mask
        else:
            mask = None
        input = self.current_values['level_{}'.format(l)][input_type]
        output = make_bn(input, self.is_training, mask = mask, num_updates = self.global_step)
        self.current_values['level_{}'.format(l)][input_type] = output
        return output
    
    # Equivalent to 0-hop filter
    def make_graph_embedding_layer(self, no_filters, l = 0, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Embed_level_{}'.format(l)) as scope:
            self.current_values['level_{}'.format(l)]['V'] = make_embedding_layer(self.current_values['level_{}'.format(l)]['V'], no_filters)
            if with_bn:
                self.make_batchnorm_layer(input_type = 'V', l = l)
            if with_act_func:
                self.current_values['level_{}'.format(l)]['V'] = tf.nn.relu(self.current_values['level_{}'.format(l)]['V'])
        return self.current_values['level_{}'.format(l)]['V'], self.current_values['level_{}'.format(l)]['A'], self.current_mask if l == 0 else None
        
    def make_dropout_layer(self, input_type = 'V', l = 0, keep_prob=0.5):
        input = self.current_values['level_{}'.format(l)][input_type]
        self.current_values['level_{}'.format(l)][input_type] = tf.cond(self.is_training, lambda:tf.nn.dropout(input, keep_prob=keep_prob), lambda:(input))
        return self.current_values['level_{}'.format(l)][input_type]
    
        
    def make_graphcnn_layer(self, no_filters, l = 0, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN_level_{}'.format(l)) as scope:
            self.current_values['level_{}'.format(l)]['V'] = make_graphcnn_layer(self.current_values['level_{}'.format(l)]['V'], self.current_values['level_{}'.format(l)]['A'], no_filters)
            if with_bn:
                self.make_batchnorm_layer(input_type = 'V', l = l)
            if with_act_func:
                self.current_values['level_{}'.format(l)]['V'] = tf.nn.relu(self.current_values['level_{}'.format(l)]['V'])
            # if self.network_debug:
            #     batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape())-1))
            #     self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var], message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_values['level_{}'.format(l)]['V']


    def make_embedding_operation_layer(self, no_filters, l = 0, name=None, with_bn=True, with_act_func=False):
        with tf.variable_scope(name, default_name='Embedding_operation_level_{}'.format(l)) as scope:
            self.current_values['level_{}'.format(l)]['V_linkage'] = make_reverse_linkage_layer(self.current_values['level_{}'.format(l)]['V'], self.current_values['level_{}'.format(l+1)]['V'], self.current_values['level_{}'.format(l)]['E'], no_filters)

            if with_bn:
                self.make_batchnorm_layer(input_type = 'V_linkage', l = l)

            self.current_values['level_{}'.format(l+1)]['V'] = self.current_values['level_{}'.format(l)]['V_linkage'] + self.current_values['level_{}'.format(l+1)]['V']

            if with_act_func:
                self.current_values['level_{}'.format(l+1)]['V'] = tf.nn.relu(self.current_values['level_{}'.format(l+1)]['V'])

        return self.current_values['level_{}'.format(l+1)]['V']
    
    def make_projection_operation_layer(self, no_filters, l = 1, name=None, with_bn=True, with_act_func=False):
        with tf.variable_scope(name, default_name='Projection_operation_level_{}'.format(l)) as scope:
            self.current_values['level_{}'.format(l)]['V_linkage'] = make_linkage_layer(self.current_values['level_{}'.format(l-1)]['V'], self.current_values['level_{}'.format(l)]['V'], self.current_values['level_{}'.format(l)]['P'], no_filters)
            
            if with_bn:
                self.make_batchnorm_layer(input_type = 'V_linkage', l = l)
            
            self.current_values['level_{}'.format(l-1)]['V'] = self.current_values['level_{}'.format(l)]['V_linkage'] + self.current_values['level_{}'.format(l-1)]['V']

            if with_act_func:
                self.current_values['level_{}'.format(l-1)]['V'] = tf.nn.relu(self.current_values['level_{}'.format(l-1)]['V'])
        return self.current_values['level_{}'.format(l-1)]['V']

    
    # def make_linkage_adjustment_layer(self, name = None, twoD_W = False):
    #     with tf.variable_scope(name, default_name='LinkageAdjustment') as scope:
    #         if self.M is None:
    #             no_features = self.initial_V.get_shape()[1].value
    #             if twoD_W:
    #                 W = make_variable_with_weight_decay('M_W', [no_features, no_features], stddev = math.sqrt(1.0/(2 * no_features)), initializerType = 'normal')
    #             else:
    #                 W = make_variable_with_weight_decay('M_W', [no_features, 1], stddev = math.sqrt(1.0/(2 * no_features)), initializerType = 'normal')
    #             self.M = tf.matmul(W, tf.transpose(W))
    #         self.current_forward_linkage = update_linkage_weighting(self.initial_V, self.initial_V_auxilary, self.current_forward_linkage, self.M)
    
    # def make_reverse_linkage_adjustment_layer(self, name = None, twoD_W = False):
    #     with tf.variable_scope(name, default_name='ReverseLinkageAdjustment') as scope:
    #         if self.M is None:
    #             no_features = self.initial_V.get_shape()[1].value
    #             if twoD_W:
    #                 W = make_variable_with_weight_decay('M_W', [no_features, no_features], stddev = math.sqrt(1.0/(2 * no_features)), initializerType = 'normal')
    #             else:
    #                 W = make_variable_with_weight_decay('M_W', [no_features, 1], stddev = math.sqrt(1.0/(2 * no_features)), initializerType = 'normal')
    #             self.M = tf.matmul(W, tf.transpose(W))
    #         self.current_reverse_linkage = update_linkage_weighting(self.initial_V_auxilary, self.initial_V, self.current_reverse_linkage, tf.transpose(self.M))