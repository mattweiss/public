import os, socket, sys
from time import time
import numpy as np
from scipy import stats
import tensorflow as tf
from pdb import set_trace as st

from abc import ABC, abstractmethod
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
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

    def evaluate(self, x=None, y=None, t=None, save_results=True):

        pass

    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None), name='y')

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

    def _setLoss(self):

        self._loss_op = tf.cast(self._loss(self._y, self._y_hat), tf.float64) + tf.cast(tf.losses.get_regularization_loss(), tf.float64)

    def _setOptimizer(self):

        if self._optimizer.__name__ == 'AdamOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)

        elif self._optimizer.__name__ == 'MomentumOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate, momentum=self._momentum,use_nesterov=self._use_nesterov).minimize(self._loss_op)

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
                epoch_val_loss = list()

                # shuffle training set
                training_samples = self._x_train[self._trials]

                for x_train_trial in training_samples:

                    if np.ndim(x_train_trial) == 1:

                        x_train_trial = np.expand_dims(x_train_trial,axis=-1)

                    # training op
                    _, train_loss = sess.run([self._optimizer_op,self._loss_op], feed_dict={self._X:x_train_trial, self._y:x_train_trial})
                    epoch_train_loss.append(train_loss)

                # validation loss
                try:

                    for x_val_trial in self._x_val[self._trials]:

                        if np.ndim(x_val_trial) == 1:

                            x_val_trial = np.expand_dims(x_val_trial,axis=-1)

                        epoch_val_loss.append(sess.run(self._loss_op, feed_dict={self._X:x_val_trial, self._y:x_val_trial}))

                except:

                    pass

                self._history['train_loss'].append(np.asarray(epoch_train_loss).mean())
                self._history['val_loss'].append(np.asarray(epoch_val_loss).mean())

                if epoch % 1 == 0:

                    print('Epoch {epoch}, Training Loss {train_loss:0.4}, Val Loss {val_loss:0.4}'.format(epoch=epoch,
                                                                                                          train_loss=self._history['train_loss'][-1],
                                                                                                          val_loss=self._history['val_loss'][-1]))

            self._history['runtime'] = (time() - start_time) / 60.0

            # test mse
            test_loss_list = list()
            test_pred_list = list()
            train_loss_list = list()
            train_pred_list = list()

            for x_train_trial in self._x_train[self._trials]:

                if np.ndim(x_train_trial) == 1:

                    x_train_trial = np.expand_dims(x_train_trial,axis=-1)

                train_loss, train_pred = sess.run([self._loss_op,self._y_hat], feed_dict={self._X:x_train_trial, self._y:x_train_trial})
                train_loss_list.append(train_loss)
                train_pred_list.append(train_pred)

            for x_test_trial in self._x_test[self._trials]:

                if np.ndim(x_test_trial) == 1:

                    x_test_trial = np.expand_dims(x_test_trial,axis=-1)

                test_loss, test_pred = sess.run([self._loss_op,self._y_hat], feed_dict={self._X:x_test_trial, self._y:x_test_trial})
                test_loss_list.append(test_loss)
                test_pred_list.append(test_pred)

            if save_model:

                self._saveModel(sess)

        self._history['test_loss'] = np.asarray(test_loss_list).mean()
        self._history['test_pred'] = np.asarray(test_pred_list)
        self._history['test_true'] = np.asarray(self._x_test[self._trials])

        self._history['train_pred'] = np.asarray(train_pred_list)
        self._history['train_true'] = np.asarray(self._x_train[self._trials])

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

            self._history['p_value'] = list()

            for epoch in range(1, self._epochs+1):

                # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
                dr_data = self._dr_dataset.generateDataset()

                # train and val loss lists
                train_loss = list()
                val_loss = list()

                # train on all trials
                #for x_train, y_train, x_val, y_val in zip(dr_data['x_train'],dr_data['y_train'],dr_data['x_val'],dr_data['y_val']):

                x_train, y_train, x_val, y_val = dr_data['x_train'], dr_data['y_train'], dr_data['x_val'], dr_data['y_val']

                x_train = np.squeeze(x_train, axis=0)
                y_train = np.squeeze(y_train, axis=0)
                x_val = np.squeeze(x_val, axis=0)
                y_val = np.squeeze(y_val, axis=0)

                train_feed_dict.update({self._X:x_train, self._y:y_train} if self._train_ground else {self._X:x_train, self._y:x_train})
                val_feed_dict.update({self._X:x_val, self._y:y_val} if self._train_ground else {self._X:x_val, self._y:x_val})

                # training op
                x_train_mb, y_train_mb = self._generateMinibatches(x_train,y_train)

                for x_mb, y_mb in zip(x_train_mb,y_train_mb):

                    train_feed_dict[self._X] = x_mb
                    train_feed_dict[self._y] = y_mb

                    #_, z,z_hat,decoder,y_hat = sess.run([self._optimizer_op, self._z, self._z_hat_pri,self._decoder,self._y_hat], feed_dict=train_feed_dict)
                    sess.run(self._optimizer_op, feed_dict=train_feed_dict)

                    # for idx, (x,y) in enumerate(zip(y_hat[:5],y_mb[:5])):

                    # plt.figure(figsize=(6,6))
                    # plt.scatter(range(x_mb.shape[0]),x_mb,label='x_mb')
                    # plt.plot(y_mb,label='y_mb')
                    # plt.grid()
                    # plt.legend()
                    # #plt.savefig('./{idx}'.format(idx=idx))
                    # plt.show()
                    # plt.close()

                # loss op
                train_loss.append(sess.run(self._loss_op, feed_dict=train_feed_dict))
                val_loss.append(sess.run(self._loss_op, feed_dict=val_feed_dict))

                self._history['train_loss'].append(np.asarray(train_loss).mean())
                self._history['val_loss'].append(np.asarray(val_loss).mean())

                if len(self._history['train_loss']) > self._history_size:

                    self._history['train_loss'].pop(0)
                    self._history['val_loss'].pop(0)

                print('Epoch {epoch} training loss {train_loss:.4f} Val Loss {val_loss:.4f}'.format(epoch=epoch,
                                                                                                    train_loss=self._history['train_loss'][-1],
                                                                                                    val_loss=self._history['val_loss'][-1]))

            self._history['runtime'] = (time() - start_time) / 60.0

            if save_model:

                self._saveModel(sess)

        return self._history

    def _generateMinibatches(self, X, y):

        X_mb = [X[i * self._mbsize:(i + 1) * self._mbsize,:] for i in range((X.shape[0] + self._mbsize - 1) // self._mbsize )]
        y_mb = [y[i * self._mbsize:(i + 1) * self._mbsize] for i in range((y.shape[0] + self._mbsize - 1) // self._mbsize )]

        return X_mb, y_mb

    def _saveModel(self, tf_session=None):

        assert tf_session is not None

        # save Tensorflow variables

        # name of file weights are saved to
        self._trained_model_file = os.getcwd() + self._results_dir + 'trained_model.ckpt'

        # save everything
        tf.train.Saver().save(tf_session, self._trained_model_file)
