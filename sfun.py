# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# SFUN.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter('ignore', UserWarning)
import sys
import numpy as np
import util
from datetime import datetime
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda, Conv1D, UpSampling1D
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.metrics import mean_absolute_error

class SFUN(object):

    def __init__(self, config, train_bn=True):
        """Set SFUN parameters.

            Args:
            config: dict
            train_bn: boolean

        """

        self._config = config
        self.train_bn = train_bn
        self.checkpoints_path = ''
        self.history_filename = ''
        self.num_freq = len(util.get_frequencies())

        self.model = self.setup_model()

    def setup_model(self):
        """Creates a SFUN object.

            Returns: keras model

        """

        self.checkpoints_path = os.path.join(self._config['training']['session_dir'], 'checkpoints')
        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)
        self.history_filename = 'history_' + self._config['training']['session_dir'][self._config['training']['session_dir'].rindex('/') + 1:] + '.csv'

        self.model, inputs_mask = self.build_model(train_bn=self.train_bn)
        self.compile_sfun(self.model, inputs_mask, self._config['training']['lr'])


        self._config['dataset']['num_freq'] = self.num_freq
        config_path = os.path.join(self._config['training']['session_dir'], 'config.json')


        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            checkpoints = os.listdir(self.checkpoints_path)
            checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
            last_checkpoint = checkpoints[-1]
            last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
            self.epoch_num = int(last_checkpoint[11:16])

            print('Loading Sound Field Network model from epoch: %d' % self.epoch_num)
            self.model.load_weights(last_checkpoint_path)

        else:

            print('Building new Sound Field Network model...')
            self.epoch_num = 0
            self.model.summary()

        if not os.path.exists(config_path):
            util.save_config(config_path, self._config)

        return self.model


    def build_model(self, train_bn=True):
        """Creates a SFUN model.

            Args:
            train_bn: boolean (optional)

            Returns: keras model, K.tensor

        """

        inputs_sf = Input((self._config['dataset']['xSamples'], self._config['dataset']['ySamples'], self.num_freq), name='inputs_sf')
        inputs_mask = Input((self._config['dataset']['xSamples'], self._config['dataset']['ySamples'], self.num_freq), name='inputs_mask')

        def encoder_layer(sf_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = util.PConv2D(filters, kernel_size, strides=2, padding='same', name='encoder_partialconv_'+str(encoder_layer.counter))([sf_in, mask_in])
            if bn:
                conv = BatchNormalization(name='encoder_bn_'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask

        encoder_layer.counter = 0
        e_conv1, e_mask1 = encoder_layer(inputs_sf, inputs_mask, 64, 5, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 3)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 3)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)

        def decoder_layer(sf_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_sf = UpSampling2D(size=(2,2), name='upsampling_sf_'+str(decoder_layer.counter))(sf_in)
            up_mask = UpSampling2D(size=(2,2), name='upsampling_mk_'+str(decoder_layer.counter))(mask_in)
            concat_sf = Concatenate(axis=3)([e_conv,up_sf])
            concat_mask = Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = util.PConv2D(filters, kernel_size, padding='same', name='decoder_partialconv_'+str(decoder_layer.counter))([concat_sf, concat_mask])
            if bn:
                conv = BatchNormalization(name='encoder_bn_'+str(decoder_layer.counter))(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            decoder_layer.counter += 1
            return conv, mask

        decoder_layer.counter = encoder_layer.counter
        d_conv5, d_mask5 = decoder_layer(e_conv4, e_mask4, e_conv3, e_mask3, 256, 3)
        d_conv6, d_mask6 = decoder_layer(d_conv5, d_mask5, e_conv2, e_mask2, 128, 3)
        d_conv7, d_mask7 = decoder_layer(d_conv6, d_mask6, e_conv1, e_mask1, 64, 3)
        d_conv8, d_mask8 = decoder_layer(d_conv7, d_mask7, inputs_sf, inputs_mask, self.num_freq, 3, bn=False)
        outputs = Conv2D(self.num_freq, 1, activation = 'sigmoid', name='outputs_sf')(d_conv8)

        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_sf, inputs_mask], outputs=outputs)

        return model, inputs_mask

    def compile_sfun(self, model, inputs_mask, lr):
        """Configures the model.

            Args:
            model: keras model
            inputs_mask: K.tensor
            lr: float

        """
        model.compile(
            optimizer = Adam(lr=lr),
            loss=self.loss_total(inputs_mask),
            metrics=[self.PSNR]
        )

    def loss_total(self, mask):
        """ Creates a loss function.

        Args:
        mask: K.tensor

        Returns: function

        """
        def loss(y_true, y_pred):

            valid_loss = self.loss_valid(mask, y_true, y_pred)
            hole_loss = self.loss_hole(mask, y_true, y_pred)

            return self._config['training']['loss']['valid_weight']*valid_loss + self._config['training']['loss']['hole_weight']*hole_loss


        return loss


    def loss_hole(self, mask, y_true, y_pred):
        """ Computes L1 loss within the mask.

        Args:
            mask: K.tensor
            y_true: K.tensor
            y_pred: K.tensor

        Returns: K.tensor

        """
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """ Computes L1 loss outside the mask

        Args:
            mask: K.tensor
            y_true: K.tensor
            y_pred: K.tensor

        Returns: K.tensor

        """
        return self.l1(mask * y_true, mask * y_pred)


    def fit_model(self, train_generator, num_steps_train, val_generator, num_steps_val, epochs):
        """Fit SFUN

        Args:
            train_generator: generator
            num_steps_train: int
            val_generator: generator
            num_steps_val: int
            epochs: int

        """
        print('Training starts!')

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=num_steps_train,
            validation_data=val_generator,
            validation_steps=num_steps_val,
            epochs=epochs,
            verbose=1,
            initial_epoch=self.epoch_num,
            callbacks=[
                CSVLogger(os.path.join(self._config['training']['session_dir'], self.history_filename), append=True),
                ModelCheckpoint(os.path.join(self.checkpoints_path, 'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5'))
            ]
        )

    def summary(self):
        """Print summary of the SFUN model"""
        print(self.model.summary())

    @staticmethod
    def PSNR(y_true, y_pred):
        """ Defines PSNR metric

        Args:
            y_true: K.tensor
            y_pred: K.tensor

        Returns: K.tensor

        """

        return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


    @staticmethod
    def l1(y_true, y_pred):
        """Calculates the L1.

        Args:
            y_true: K.tensor
            y_pred: K.tensor

        Returns: K.tensor
        """
        return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])

    def predict(self, sample):
        """ Generates output predictions for the input samples.

        Args:
        sample: list=[np.ndarray, np.ndarray]

        Returns: np.ndarray
        """

        return self.model.predict(sample)

