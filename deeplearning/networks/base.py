import os, socket, sys
from time import time
import numpy as np
from scipy import stats
import tensorflow as tf
tf_float_prec = tf.float64
from pdb import set_trace as st

from abc import ABC, abstractmethod
from collections import OrderedDict
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.datasets.outliers import generate_outliers
from dovebirdia.deeplearning.layers.base import Dense
from dovebirdia.deeplearning.regularizers.base import orthonormal_regularizer

machine = socket.gethostname()
if machine == 'pengy':

    import matplotlib.pyplot as plt

else:

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

class AbstractNetwork(ABC):

    """
    Abstract base class for network
    """

    def __init__(self, params=None):

        assert isinstance(params,dict)

        dictToAttributes(self,params)

        # hold, etc.
        self._history = {
            'train_loss':list(),
            'val_loss':list(),
            'test_loss':list(),
            'train_mse':list(),
            'val_mse':list(),
            'test_mse':list(),
            }

    ##################
    # Public Methods #
    ##################

    def compile(self):

        """
        Build Network
        """

       ############################
        # Build Network
        ############################
        self._buildNetwork()

        ############################
        # Set Optimizer
        ############################
        self._setLoss()

        ############################
        # Set Optimizer
        ############################
        self._setOptimizer()

    @abstractmethod
    def fit(self, dataset=None, save_model=False):

        pass

    @abstractmethod
    def predict(self, x=None):

        pass

    @abstractmethod
    def evaluate(self, x=None, y=None, t=None, save_results=True):

        pass

    ###################
    # Private Methods #
    ###################

    @abstractmethod
    def _buildNetwork(self):

        pass

    @abstractmethod
    def _setLoss(self):

        pass

    @abstractmethod
    def _setOptimizer(self):

        pass

    @abstractmethod
    def _saveModel(self):

        pass

class FeedForwardNetwork(AbstractNetwork):

    """
    Feed Forward Network Class
    """

    def __init__(self, params=None):

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    def fit(self, dataset=None, dr_params=None, save_model=False):

        if dataset is not None:

            return self._fit(dataset, save_model)

        elif dr_params is not None:

            return self._fitDomainRandomization(dr_params, save_model)

    def predict(self, dataset=None):

        pass

    def evaluate(self, x=None, y=None, labels=None,
                 eval_ops = None,
                 attributes = None,
                 save_results=None):

        assert x is not None
        assert y is not None
        
        # default evaluation ops list
        eval_ops_dict = OrderedDict()
        eval_ops_dict = {
            'loss_op':self._loss_op,
            'mse_op':self._mse_op,
            'y_hat':self._y_hat
        }

        # add custom ops to list
        if eval_ops is not None:

            for eval_op in eval_ops:

                eval_op_key = '_' + eval_op

                if eval_op_key in self.__dict__.keys():

                    eval_ops_dict.update({eval_op:self.__dict__[eval_op_key]})

        # dictionary of results lists
        for eval_op in eval_ops_dict.keys():

            self._history[eval_op] = list()
        
        with tf.Session() as sess:

            # backwards compatibility
            try:

                model_results_path = './results/trained_model.ckpt'
                tf.train.Saver().restore(sess, model_results_path)

            except:

                model_results_path = './results/tensorflow_model.ckpt'
                tf.train.Saver().restore(sess, model_results_path)

            for trial, (X,Y) in enumerate(zip(x,y)):

                # if X.ndim > 2:

                #     X = np.squeeze(X,axis=-1)

                # if Y.ndim > 3:

                #     Y = np.squeeze(Y,axis=-1)

                test_feed_dict = {self._X:X, self._y:Y,self._mask:np.ones(shape=Y.shape)}

                # run ops
                eval_ops_results = sess.run(list(eval_ops_dict.values()),feed_dict=test_feed_dict)

                # append ops results to history
                for eval_ops_key,eval_ops_result in zip(eval_ops_dict.keys(),eval_ops_results):

                    self._history[eval_ops_key].append(eval_ops_result)
                    
        # append x and y to history
        self._history['x'] = np.asarray(x)
        self._history['y'] = np.asarray(y)
        self._history.update({'y_hat':np.asarray(self._history['y_hat'])})
        
        # add additionaly class attributes to history
        if attributes is not None:

            for attr in attributes:

                attr_key = '_' + attr

                if attr_key in self.__dict__.keys():

                    self._history[attr] = self.__dict__[attr_key]

        # save predictions
        if save_results is not None:
            
            saveDict(save_dict=self._history, save_path='./results/' + save_results + '.pkl')

        return self._history

    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):
        
        self._y_hat = Dense(name='layers',
                            weight_initializer=self._weight_initializer,
                            weight_regularizer=None,
                            bias_initializer=self._bias_initializer,
                            activation=self._activation).build(self._X, self._hidden_dims)

        self._y_hat = Dense(name='output',
                            weight_initializer=self._weight_initializer,
                            weight_regularizer=None,
                            bias_initializer=self._bias_initializer,
                            activation=self._output_activation).build(self._y_hat, [self._output_dim])

    def _setPlaceholders(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf_float_prec, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf_float_prec, shape=(None,self._input_dim), name='y')
        self._t = tf.placeholder(dtype=tf_float_prec, shape=(None,1), name='t')
        self._mask = tf.placeholder(dtype=tf_float_prec, shape=(None,self._input_dim), name='mask')
        
    def _setLoss(self):

        self._mse_op = tf.cast(self._loss(self._y,self._y_hat,weights=self._mask), tf_float_prec)
        self._loss_op = self._mse_op + tf.cast(tf.losses.get_regularization_loss(), tf_float_prec)

    def _setOptimizer(self):

        if self._optimizer.__name__ == 'AdamOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)

        elif self._optimizer.__name__ == 'MomentumOptimizer':

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.compat.v1.train.exponential_decay(self._learning_rate, global_step, self._decay_steps, self._decay_rate, staircase=self._staircase)
            self._optimizer_op = self._optimizer(learning_rate=learning_rate, momentum=self._momentum, use_nesterov=self._use_nesterov).minimize(self._loss_op, global_step=global_step)

        elif self._optimizer.__name__ == 'GradientDescentOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)

        elif self._optimizer.__name__ == 'RMSPropOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate, momentum=self._momentum).minimize(self._loss_op)

        else:

            print('Define a valid optimizer')
            sys.exit(1)

    def _fit(self, dataset=None, save_model=False):

        dictToAttributes(self,dataset)

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)

            for k, v in zip(variables_names, values):

                print(v.shape, k)
                
            # start time
            start_time = time()

            for epoch in range(1, self._epochs+1):          

                # lists to hold epoch training and validation losses
                epoch_train_loss = list()
                epoch_train_mse = list()

                # shuffle training set
                np.random.shuffle(self._x_train)

                # loop over training examples
                for x_train_trial in self._x_train:
                    
                    if np.ndim(x_train_trial) == 1:

                        x_train_trial = np.expand_dims(x_train_trial,axis=-1)

                    # add outliers to training data
                    if self._outliers:

                        train_outliers = generate_outliers(shape=x_train_trial.shape,
                                                           p_outlier=self._p_outlier,
                                                           outlier_range=self._outlier_range)

                        # truth and truth + outliers
                        x_train_trial = x_train_trial + train_outliers
                        y_train_trial = x_train_trial
                        
                    else:

                        x_train_trial, y_train_trial = x_train_trial, x_train_trial

                    # plt.figure(figsize=(6,6))
                    # plt.plot(x_train_trial,label='x',marker=None)
                    # plt.plot(y_train_trial,label='y')
                    # plt.grid()
                    # plt.legend()
                    # plt.show()
                    # plt.close()

                    #############
                    # train model
                    #############
                    
                    # generate minibatches
                    x_train_mbs, y_train_mbs = self._generateMinibatches(x_train_trial,y_train_trial)

                    # loop over mini batches
                    for x_train_mb,y_train_mb in zip(x_train_mbs,y_train_mbs):

                        # training op
                        _, train_loss, train_mse = sess.run([self._optimizer_op,self._loss_op,self._mse_op], feed_dict={self._X:x_train_mb, self._y:y_train_mb})
                        epoch_train_loss.append(train_loss)
                        epoch_train_mse.append(train_mse)
                        
                # validation loss
                try:
                    
                    epoch_val_loss = list()
                    epoch_val_mse = list()
                    
                    for x_val_trial in self._x_val:

                        if np.ndim(x_val_trial) == 1:

                            x_val_trial = np.expand_dims(x_val_trial,axis=-1)

                        # add noise to training data
                        if self._outliers:

                            val_outliers = generate_outliers(shape=x_val_trial.shape,
                                                             p_outlier=self._p_outlier,
                                                             outlier_range=self._outlier_range)

                            # truth and truth + outliers
                            x_val_trial = x_val_trial + val_outliers
                            y_val_trial = x_val_trial

                        else:

                            x_val_trial, y_val_trial = x_val_trial, x_val_trial 

                        val_loss, val_mse = sess.run([self._loss_op,self._mse_op], feed_dict={self._X:x_val_trial, self._y:y_val_trial})
                        epoch_val_loss.append(val_loss)
                        epoch_val_mse.append(val_mse)

                        # y_val_hat = sess.run(self._y_hat, feed_dict={self._X:x_val_trial, self._y:y_val_trial})
                        # plt.figure(figsize=(6,6))
                        # plt.plot(y_val_trial,label='True')
                        # plt.plot(x_val_trial,label='Input')
                        # plt.plot(y_val_hat,label='Pred')
                        # plt.grid()
                        # plt.legend()
                        # plt.show()
                        # plt.close()

                except:

                    pass
                
                self._history['train_loss'].append(np.asarray(epoch_train_loss).mean())
                self._history['val_loss'].append(np.asarray(epoch_val_loss).mean())
                self._history['train_mse'].append(np.asarray(epoch_train_mse).mean())
                self._history['val_mse'].append(np.asarray(epoch_val_mse).mean())
                
                if epoch % 1 == 0:

                    print('Epoch {epoch}, Training Loss/MSE {train_loss:0.4}/{train_mse:0.4}, Val Loss/MSE {val_loss:0.4}/{val_mse:0.4}'.format(epoch=epoch,
                                                                                                                                                train_loss=self._history['train_loss'][-1],
                                                                                                                                                train_mse=self._history['train_mse'][-1],        
                                                                                                                                                val_loss=self._history['val_loss'][-1],
                                                                                                                                                val_mse=self._history['val_mse'][-1]))

            self._history['runtime'] = (time() - start_time) / 60.0

            # if test set was passed compute test loss
            try:

                epoch_test_loss = list()
                epoch_test_mse = list()
                
                for x_test_trial in self._x_test:

                    if np.ndim(x_test_trial) == 1:

                        x_test_trial = np.expand_dims(x_test_trial,axis=-1)

                    x_test_trial, y_test_trial = x_test_trial, x_test_trial 

                    test_loss, test_mse = sess.run([self._loss_op,self._mse_op], feed_dict={self._X:x_test_trial, self._y:y_test_trial})
                    epoch_test_loss.append(test_loss)
                    epoch_test_mse.append(test_mse)

                    # y_val_hat = sess.run(self._y_hat, feed_dict={self._X:x_val_trial, self._y:y_val_trial})
                    # plt.figure(figsize=(6,6))
                    # plt.plot(y_val_trial,label='True')
                    # plt.plot(x_val_trial,label='Input')
                    # plt.plot(y_val_hat,label='Pred')
                    # plt.grid()
                    # plt.legend()
                    # plt.show()
                    # plt.close()

                self._history['test_loss'].append(np.asarray(epoch_test_loss).mean())
                self._history['test_mse'].append(np.asarray(epoch_test_mse).mean())

            except:

                pass

            # save model
            if save_model:

                self._saveModel(sess)

        plt.figure(figsize=(6,6))
        plt.plot(self._history['train_loss'],label='Train Loss')
        plt.plot(self._history['val_loss'],label='Val Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.savefig('loss_plot')

        return self._history

    def _fitDomainRandomization(self, dr_params=None, save_model=False):

        # create domainRandomizationDataset object
        self._dr_dataset = DomainRandomizationDataset(dr_params)
        
        # dictionaries to hold training and validation data
        train_feed_dict = dict()
        val_feed_dict = dict()

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)

            for k, v in zip(variables_names, values):

                print(k, v.shape)

            # start time
            start_time = time()

            for epoch in range(1, self._epochs+1):

                # generate training and validation datasets
                train_data = self._dr_dataset.generateDataset()
                val_data = self._dr_dataset.generateDataset()

                # train and val loss lists
                train_loss_list = list()
                val_loss_list = list()
                train_mse_list = list()
                val_mse_list = list()

                # train on all trials
                for x_train, y_train, mask_train, x_val, y_val, mask_val in zip(train_data['x_test'], train_data['y_test'], train_data['mask'],
                                                                                val_data['x_test'], val_data['y_test'], val_data['mask']):

                    
                    # plt.figure(figsize=(18,12))

                    # plt.subplot(231)
                    # plt.plot(x_train[:,0],label='x0',marker=None)
                    # plt.plot(y_train[:,0],label='y0',marker=None)
                    # plt.title(np.array_equal(x_train[:,0],y_train[:,0]))
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(232)
                    # plt.plot(x_train[:,1],label='x1',marker=None)
                    # plt.plot(y_train[:,1],label='y1',marker=None)
                    # plt.title(np.array_equal(x_train[:,1],y_train[:,1]))
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(233)
                    # plt.scatter(x_train[:,0],x_train[:,1],label='x',marker=None)
                    # plt.scatter(y_train[:,0],y_train[:,1],label='x',marker=None)
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(234)
                    # plt.plot(x_val[:,0],label='x0',marker=None)
                    # plt.plot(y_val[:,0],label='y0',marker=None)
                    # plt.title(np.array_equal(x_val[:,0],y_val[:,0]))
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(235)
                    # plt.plot(x_val[:,1],label='x1',marker=None)
                    # plt.plot(y_val[:,1],label='y1',marker=None)
                    # plt.title(np.array_equal(x_val[:,1],y_val[:,1]))
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(236)
                    # plt.scatter(x_val[:,0],x_val[:,1],label='x',marker=None)
                    # plt.scatter(y_val[:,0],y_val[:,1],label='y',marker=None)
                    # plt.grid()
                    # plt.legend()
                    
                    # plt.show()
                    # plt.close()
                                    
                    if not self._train_ground:

                        y_train = x_train
                        y_val = x_val

                    # train on minibatches
                    x_train_mb, y_train_mb, mask_train_mb = self._generateMinibatches(x_train,y_train,mask_train)

                    for x_mb, y_mb, mask_mb in zip(x_train_mb,y_train_mb,mask_train_mb):

                        x_mb, y_mb, mask_mb = x_mb, y_mb, mask_mb

                        train_feed_dict.update({self._X:x_train,self._y:y_train,self._mask:mask_train,self._t:train_data['t']})
                        sess.run(self._optimizer_op, feed_dict=train_feed_dict)
                        
                    # loss op
                    train_loss, train_mse = sess.run([self._loss_op,self._mse_op],feed_dict=train_feed_dict)
                    val_feed_dict.update({self._X:x_val,self._y:y_val,self._mask:mask_val,self._t:val_data['t']})
                    val_loss, val_mse = sess.run([self._loss_op,self._mse_op],feed_dict=val_feed_dict)
                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)
                    train_mse_list.append(train_mse)
                    val_mse_list.append(val_mse)

                    self._history['train_loss'].append(np.asarray(train_loss).mean())
                    self._history['val_loss'].append(np.asarray(val_loss).mean())
                    self._history['train_mse'].append(np.asarray(train_mse).mean())
                    self._history['val_mse'].append(np.asarray(val_mse).mean())

                    # if epoch % 1 == 0:

                    #     plt.figure(figsize=(6,6))

                    #     plt.subplot(111)
                    #     plt.plot(x_train[:,0])
                    #     plt.plot(y_train[:,0])
                    #     plt.grid()
                    #     plt.title('X0')

                        # plt.subplot(132)
                        # plt.plot(x_train[:,1])
                        # plt.plot(y_train[:,1])
                        # plt.grid()
                        # plt.title('X1')

                        # plt.subplot(133)
                        # plt.plot(x_train[:,2])
                        # plt.plot(y_train[:,2])
                        # plt.grid()
                        # plt.title('X2')
                        
                        #plt.show() 
                        #plt.savefig(os.getcwd() + self._results_dir + str(epoch))
                        #plt.close()
                                        
                    print('Epoch {epoch}, Training Loss/MSE {train_loss:0.4}/{train_mse:0.4}, Val Loss/MSE {val_loss:0.4}/{val_mse:0.4}'.format(epoch=epoch,
                                                                                                                                                train_loss=self._history['train_loss'][-1],
                                                                                                                                                train_mse=self._history['train_mse'][-1],        
                                                                                                                                                val_loss=self._history['val_loss'][-1],
                                                                                                                                                val_mse=self._history['val_mse'][-1]))



            self._history['runtime'] = (time() - start_time) / 60.0
            
            if save_model:

                self._saveModel(sess,'trained_model.ckpt')

        plt.figure(figsize=(6,6))
        plt.plot(self._history['train_loss'],label='Train Loss')
        plt.plot(self._history['val_loss'],label='Val Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.savefig('loss_plot')
                
        return self._history

    def _generateMinibatches(self, X, y=None,mask=None):

        X_mb, y_mb, mask_mb = None, None, None
        
        X_mb = [X[i * self._mbsize:(i + 1) * self._mbsize,:] for i in range((X.shape[0] + self._mbsize - 1) // self._mbsize )]

        if y is not None:

            y_mb = [y[i * self._mbsize:(i + 1) * self._mbsize] for i in range((y.shape[0] + self._mbsize - 1) // self._mbsize )]

        if mask is not None:

            mask_mb = [mask[i * self._mbsize:(i + 1) * self._mbsize] for i in range((mask.shape[0] + self._mbsize - 1) // self._mbsize )]

        return X_mb, y_mb, mask_mb

    def _saveModel(self, tf_session=None, model_name=None):

        assert tf_session is not None
        assert model_name is not None

        # save Tensorflow variables

        # name of file weights are saved to
        #self._trained_model_file = os.getcwd() + self._results_dir + 'trained_model.ckpt'
        self._trained_model_file = os.getcwd() + self._results_dir + model_name
        
        # save everything
        tf.train.Saver().save(tf_session, self._trained_model_file)
