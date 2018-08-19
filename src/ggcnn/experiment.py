from ggcnn.helper import print_ext
from ggcnn.network import GraphCNNNetwork
import numpy as np
import tensorflow as tf
import time

class GGCNNExperiment():
    def __init__(self, dataset_name, model_name, net_constructor):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_iterations = 200
        self.iterations_per_test = 5
        self.display_iter = 5
        self.train_batch_size = 0
        self.test_batch_size = 0
        self.reports = {}
        self.silent = False
        
        self.net_constructor = net_constructor
        self.net = GraphCNNNetwork()
        tf.reset_default_graph()

        ###
        self.loss_type = "cross_entropy"

    # def __init__(self, name, dataset, net_constructor):
    #     '''
    #     '''
    #     self.name = name
    #     self.net_constructor = net_constructor

    #     # self.preprocess_data(dataset)

    def print_ext(self, *args):
        if self.silent == False:
            print_ext(*args)


    def preprocess_data(self, dataset):
        self.largest_graph = dataset[0].shape[0]
        self.graph_size = [self.largest_graph]
        
        # self.graph_vertices = np.expand_dims(dataset[0].astype(np.float32), axis=0)
        # self.graph_adjacency = np.expand_dims(dataset[1].astype(np.float32), axis=0)
        # self.graph_labels = np.expand_dims(dataset[2].astype(np.int64), axis=0)
        self.graph_vertices = dataset[0].astype(np.float32)
        self.graph_adjacency = dataset[1].astype(np.float32)
        self.graph_labels = dataset[2].astype(np.int64)
        
        self.no_samples = self.graph_labels.shape[0]  # Decreased from 1
        
        # single_sample = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]

    
    def create_data(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                self.print_ext('Creating training Tensorflow Tensors')
                
                
                vertices = self.graph_vertices
                adjacency = self.graph_adjacency
                labels = self.graph_labels

                # train_input_mask = np.zeros([1, self.largest_graph, 1]).astype(np.float32)
                # train_input_mask[:, self.train_idx, :] = 1
                train_input_mask = np.zeros([self.largest_graph, 1]).astype(np.float32)
                train_input_mask[self.train_idx, :] = 1

                self.train_input = [vertices, adjacency, labels, train_input_mask]
                
                # test_input_mask = np.zeros([1, self.largest_graph, 1]).astype(np.float32)
                # test_input_mask[:, self.test_idx, :] = 1
                test_input_mask = np.zeros([self.largest_graph, 1]).astype(np.float32)
                test_input_mask[self.test_idx, :] = 1
                self.test_input = [vertices, adjacency, labels, test_input_mask]
                
                

    def create_loss_function(self):
        self.print_ext('Creating loss function and summaries')
        
        with tf.variable_scope('loss') as scope:
            # self.net.current_V = tf.Print(self.net.current_V, [tf.reduce_mean(self.net.current_mask)])
            # self.net.current_V = tf.Print(self.net.current_V, [tf.shape(self.net.current_V), tf.shape(self.net.labels)])
            # self.net.current_V = tf.stop_gradient(tf.abs(self.net.current_mask-1) * self.net.current_V) + self.net.current_mask * self.net.current_V

            inv_sum = (1./tf.reduce_sum(self.net.current_mask))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net.current_V, labels=self.net.labels)
            cross_entropy = tf.multiply(tf.squeeze(self.net.current_mask), tf.squeeze(cross_entropy))
            cross_entropy = tf.reduce_sum(cross_entropy)*inv_sum

            correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.current_V, 1), self.net.labels), tf.float32)
            correct_prediction = tf.multiply(tf.squeeze(self.net.current_mask), tf.squeeze(correct_prediction))
            accuracy = tf.reduce_sum(correct_prediction)*inv_sum
            
            tf.add_to_collection('losses', cross_entropy)
            tf.summary.scalar('loss', cross_entropy)
            
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
            
            max_acc = tf.cond(self.net.is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))
            
            tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('accuracy', accuracy)
            
            self.reports['accuracy'] = accuracy
            self.reports['max acc.'] = max_acc
            self.reports['cross_entropy'] = cross_entropy
            
            #### Added to compute prediction
#            y_pred = tf.nn.softmax(self.net.current_V)
            self.y_pred_cls = tf.argmax(self.net.current_V, 1)
            ####

    def create_input_variable(self, input):
        output = []
        for i in range(len(input)):
            placeholder = tf.placeholder(tf.as_dtype(input[i].dtype), shape=input[i].shape)
            var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = input[i]
            output.append(var)
        return output

    def build_network(self):

        tf.reset_default_graph()

        self.variable_initialization = {}

        self.print_ext('Creating training network')
        self.net.is_training = tf.placeholder(tf.bool, shape=(), name = 'is_training')
        self.net.global_step = tf.Variable(0, name='global_step', trainable=False)

        train_input = self.create_input_variable(self.train_input)
        test_input = self.create_input_variable(self.test_input)

        input = tf.cond(self.net.is_training, lambda: train_input, lambda: test_input)
        self.net_constructor.create_network(self.net, input)
        self.create_loss_function()


    def run(self):
        
        self.print_ext('Training model "%s"!' % self.model_name)

        self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name)
        self.test_summary_path = './summary/%s/test/%s' %(self.dataset_name, self.model_name)
        self.train_summary_path = './summary/%s/train/%s' %(self.dataset_name, self.model_name)


        i = 0

        self.print_ext('Preparing training')
        loss = tf.add_n(tf.get_collection('losses'))
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        
        
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer().minimize(loss, global_step = self.net.global_step)
        
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer(), feed_dict = self.variable_initialization)
        
        summary_merged = tf.summary.merge_all()
    
        self.print_ext('Starting threads')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        self.print_ext('Starting training. train_batch_size:', self.train_batch_size, 'test_batch_size:', self.test_batch_size)
        wasKeyboardInterrupt = False

        writer = tf.summary.FileWriter('./Graphs', tf.get_default_graph())

        try:
            total_training = 0.0
            total_testing = 0.0
            start_at = time.time()
            last_summary = time.time()
            while i < self.num_iterations:
                if i % self.iterations_per_test == 0:
                    start_temp = time.time()
                    summary, reports = sess.run([summary_merged, self.reports], feed_dict={self.net.is_training:0})
                    total_testing += time.time() - start_temp
                    self.print_ext('Test Step %d Finished' % i)
                    for key, value in reports.items():
                        self.print_ext('Test Step %d "%s" = ' % (i, key), value)
                    
                start_temp = time.time()
                summary, _, reports = sess.run([summary_merged, train_step, self.reports], feed_dict={self.net.is_training:1})
                total_training += time.time() - start_temp
                i += 1
                if ((i-1) % self.display_iter) == 0:
                    total = time.time() - start_at
                    self.print_ext('Training Step %d Finished Timing (Training: %g, Test: %g) after %g seconds' % (i-1, total_training/total, total_testing/total, time.time()-last_summary)) 
                    for key, value in reports.items():
                        self.print_ext('Training Step %d "%s" = ' % (i-1, key), value)
                    last_summary = time.time()            
                if (i-1) % 100 == 0:
                    total_training = 0.0
                    total_testing = 0.0
                    start_at = time.time()
            if i % self.iterations_per_test == 0:
                summary = sess.run(summary_merged, feed_dict={self.net.is_training:0})
                self.print_ext('Test Step %d Finished' % i)
        except KeyboardInterrupt as err:
            self.print_ext('Training interrupted at %d' % i)
            wasKeyboardInterrupt = True
            raisedEx = err
        finally:
            self.print_ext('Training completed, starting cleanup!')
            coord.request_stop()
            coord.join(threads)
            self.print_ext('Cleanup completed!')
            writer.close()
            if wasKeyboardInterrupt:
                raise raisedEx
            
            
            return sess.run([self.max_acc_test, self.net.global_step, self.y_pred_cls], feed_dict={self.net.is_training:0})