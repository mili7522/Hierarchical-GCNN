from ggcnn.helper import print_ext
from ggcnn.network import GraphCNNNetwork
import numpy as np
import tensorflow as tf
import time
from sklearn.decomposition import PCA

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
        l = 0
        dataset = dataset.copy()
        while True:
            level_dataset = dataset.get('level_{}'.format(l))
            if (level_dataset is None) or (len(level_dataset) == 0):
                break
            level_dataset['graph_size'] = level_dataset['features'].shape[0]

            for key in ['features', 'adj_mat']:
                level_dataset[key] = level_dataset[key].astype(np.float32)

            embedding = level_dataset.get('embedding')
            if embedding is None:
                level_dataset['has_embedding'] = False
            else:
                level_dataset['has_embedding'] = True
                level_dataset['embedding'] = embedding.astype(np.float32)

            projection = level_dataset.get('projection')
            if projection is None:
                level_dataset['has_projection'] = False
            else:
                level_dataset['has_projection'] = True
                level_dataset['projection'] = projection.astype(np.float32)
            
            setattr(self, 'level_{}_dataset'.format(l), level_dataset)
            l += 1
        self.number_of_layers = l

        if self.loss_type == "cross_entropy":
            self.level_0_dataset['labels'] = self.level_0_dataset['labels'].astype(np.float32)
        elif self.loss_type == "linear":
            self.level_0_dataset['labels'] = self.level_0_dataset['labels'].astype(np.int32)

    
    def create_data(self, train_idx, test_idx, n_components = 10):
        self.train_idx = train_idx
        self.test_idx = test_idx
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                self.print_ext('Creating training Tensorflow Tensors')
                
                # ### PCA
                # if n_components is not None:
                #     pca = PCA(n_components=n_components, svd_solver='full')
                #     reduced_features = pca.fit_transform(vertices.copy())
                #     print("PCA variance ratio: ", pca.explained_variance_ratio_)
                #     reduced_features_auxilary = pca.transform(auxilary_vertices.copy())
                # ###
                # else:
                #     reduced_features = vertices
                #     reduced_features_auxilary = auxilary_vertices
                
                train_input_mask = np.zeros([self.level_0_dataset['graph_size'], 1]).astype(np.float32)
                train_input_mask[self.train_idx, :] = 1
                self.level_0_dataset['train_input_mask'] = train_input_mask

                test_input_mask = np.zeros([self.level_0_dataset['graph_size'], 1]).astype(np.float32)
                test_input_mask[self.test_idx, :] = 1
                self.level_0_dataset['test_input_mask'] = test_input_mask

    def create_loss_function(self):
        self.print_ext('Creating loss function and summaries')
        
        with tf.variable_scope('loss') as scope:
            # self.net.current_V = tf.Print(self.net.current_V, [tf.reduce_mean(self.net.current_mask)])
            # self.net.current_V = tf.Print(self.net.current_V, [tf.shape(self.net.current_V), tf.shape(self.net.labels)])
            # self.net.current_V = tf.stop_gradient(tf.abs(self.net.current_mask-1) * self.net.current_V) + self.net.current_mask * self.net.current_V

            inv_sum = (1./tf.reduce_sum(self.net.current_mask))
            
            if self.loss_type == "cross_entropy":
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net.current_values['level_0']['V'], labels=self.net.labels)
                cross_entropy = tf.multiply(tf.squeeze(self.net.current_mask), tf.squeeze(cross_entropy))
                cross_entropy = tf.reduce_sum(cross_entropy)*inv_sum


                correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.current_values['level_0']['V'], 1), self.net.labels), tf.float32)
                correct_prediction = tf.multiply(tf.squeeze(self.net.current_mask), tf.squeeze(correct_prediction))
                accuracy = tf.reduce_sum(correct_prediction)*inv_sum
                
                tf.add_to_collection('losses', cross_entropy)
                tf.summary.scalar('loss', cross_entropy)
            
                self.max_acc_train = tf.Variable(tf.zeros([]), name = "max_acc_train")
                self.max_acc_test = tf.Variable(tf.zeros([]), name = "max_acc_test")
            
                max_acc = tf.cond(self.net.is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))
                
                tf.summary.scalar('max_accuracy', max_acc)
                tf.summary.scalar('accuracy', accuracy)

                self.reports['accuracy'] = accuracy
                self.reports['max acc.'] = max_acc
                self.reports['cross_entropy'] = cross_entropy

            elif self.loss_type == "linear":
                linear_loss = tf.losses.mean_squared_error(labels = tf.squeeze(self.net.labels), predictions = tf.squeeze(self.net.current_values['level_0']['V']), weights = tf.squeeze(self.net.current_mask))

                tf.add_to_collection('losses', linear_loss)
                tf.summary.scalar('loss', linear_loss)
                
                self.min_loss_train = tf.Variable(tf.fill([], np.inf), name = "min_loss_train")
                self.min_loss_test = tf.Variable(tf.fill([], np.inf), name = "min_loss_test")

                min_loss = tf.cond(self.net.is_training, lambda: tf.assign(self.min_loss_train, tf.minimum(self.min_loss_train, linear_loss)), lambda: tf.assign(self.min_loss_test, tf.minimum(self.min_loss_test, linear_loss)))

                tf.summary.scalar('min_loss', min_loss)
                self.reports['min loss'] = min_loss
                self.reports['loss'] = linear_loss


    def create_input_variable(self):

        def initialize_with_placeholder(item):
            placeholder = tf.placeholder(tf.as_dtype(item.dtype), shape=item.shape)
            var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = item
            return var
        
        train_output = {}
        test_output = {}
        for l in range(self.number_of_layers):
            train_output['level_{}'.format(l)] = {}
            test_output['level_{}'.format(l)] = {}

            level_dataset = getattr(self, 'level_{}_dataset'.format(l))

            for new_key, old_key in zip(['V', 'A'], ['features', 'adj_mat']):
                var = initialize_with_placeholder(level_dataset[old_key])
                train_output['level_{}'.format(l)][new_key] = var
                test_output['level_{}'.format(l)][new_key] = var
            if level_dataset['has_embedding']:
                var = initialize_with_placeholder(level_dataset['embedding'])
                train_output['level_{}'.format(l)]['E'] = var
                test_output['level_{}'.format(l)]['E'] = var
            if level_dataset['has_projection']:
                var = initialize_with_placeholder(level_dataset['projection'])
                train_output['level_{}'.format(l)]['P'] = var
                test_output['level_{}'.format(l)]['P'] = var
            if l == 0:
                train_output['mask'] = initialize_with_placeholder(level_dataset['train_input_mask'])
                test_output['mask'] = initialize_with_placeholder(level_dataset['test_input_mask'])
                var = initialize_with_placeholder(level_dataset['labels'])
                train_output['labels'] = var
                test_output['labels'] = var

        return train_output, test_output

    def build_network(self):

        tf.reset_default_graph()

        self.variable_initialization = {}

        self.print_ext('Creating training network')
        self.net.is_training = tf.placeholder(tf.bool, shape=(), name = 'is_training')
        self.net.global_step = tf.Variable(0, name='global_step', trainable=False, dtype = tf.float32)

        train_input, test_input = self.create_input_variable()

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
        
        
        # sess = tf.Session()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer(), feed_dict = self.variable_initialization)
            
            summary_merged = tf.summary.merge_all()
        
            self.print_ext('Starting threads')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.print_ext('Starting training. train_batch_size:', self.train_batch_size, 'test_batch_size:', self.test_batch_size)
            wasKeyboardInterrupt = False

            # writer = tf.summary.FileWriter('./Graphs', tf.get_default_graph())

            all_training_reports = []
            all_testing_reports = []
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
                        all_testing_reports.append(reports)
                        
                    start_temp = time.time()
                    summary, _, reports = sess.run([summary_merged, train_step, self.reports], feed_dict={self.net.is_training:1})
                    all_training_reports.append(reports)
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
                # writer.close()

#                 current_A, dist_beta = sess.run([self.net.current_A, self.net.dist_beta], feed_dict={self.net.is_training:0})
#                 print(current_A)
#                 print(dist_beta)

                if wasKeyboardInterrupt:
                    raise raisedEx
                
                
#                 return sess.run([self.max_acc_test, self.net.global_step], feed_dict={self.net.is_training:0}), current_A
#                 return current_A
                return all_training_reports, all_testing_reports, sess.run(self.net.current_values['level_0']['V'], feed_dict={self.net.is_training:0})
