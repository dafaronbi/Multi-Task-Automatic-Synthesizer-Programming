#import needed libraries
from tensorflow.keras import layers, losses, Model, initializers
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

#shortcuts for tensorflow stuff
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras
import tensorflow.keras.backend as K

#define shapes
l_dim = 64
i_dim = (25000, 128, 431, 1)
o_dim = (25000, 315)

#get optimizer
optimizer = tf.keras.optimizers.Adam() 

#batch_size
batch_size = 32

#number of batches in one epoch
batches_epoch = i_dim[0] // batch_size

#warmup amount
warmup_it = 100*batches_epoch

#parameter input for dynamic filters
v_dims = 4

#class for sampling in vae
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    @tf.function
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#class for regularizarion with warmup
class W_KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, iters: tf.Variable, warm_up_iters: int, latent_size: int):
        self._iters = np.array([iters])
        self._warm_up_iters = np.array([warm_up_iters])
        self.latent_size = latent_size

    @tf.function
    def __call__(self, activation):
        # note: activity regularizers automatically divide by batch size
        mu =  activation[:self.latent_size]
        log_var = activation[self.latent_size:]
        k = np.min([self._iters / self._warm_up_iters, 1])
        return -0.5 * k * K.sum(1 + log_var - K.square(mu) - K.exp(log_var))



def autoencoder(latent_dim,input_dim, output_dim):
    """
    autoencoder: Create autoencoder model
    :param latent_dim (numpy.shape): size of latent dimensions
    :param input_dim (numpy.shape): size of input dimensions
    :param output_dim (numpy.shape): size of output dimensions
    """

    #set input size
    inp = layers.Input((input_dim[-3],input_dim[-2],1))

    #convolutional layers and pooling
    encoder = layers.Conv2D(4,3,1,"same", activation='relu')(inp)
    pool = layers.MaxPool2D(2,1,"same")(encoder)
    conv = layers.Conv2D(4,3,1,"same", activation='relu')(pool)
    pool2 = layers.MaxPool2D(2, 1, "same")(conv)

    #fully connected flat layers
    encoder_flat = layers.Flatten()(pool2)

    #latent dimension
    latent_layer = layers.Dense(latent_dim, activation='relu')(encoder_flat)

    #decoder layers to spectrogram
    decoder = layers.Dense(input_dim[-3]*input_dim[-2], activation='relu')(latent_layer)
    decoder_reshaped = layers.Reshape((input_dim[-3],input_dim[-2]),name='spectrogram')(decoder)

    #decoder layers to synth parameters
    decoder_b = layers.Dense(output_dim[-1],name='synth_params', activation='sigmoid')(latent_layer)

    #generate model
    return Model(inputs=inp, outputs=[decoder_reshaped, decoder_b])

def autoencoder2(latent_dim,input_dim, output_dim):
    """
    autoencoder: Create autoencoder model
    :param latent_dim (numpy.shape): size of latent dimensions
    :param input_dim (numpy.shape): size of input dimensions
    :param output_dim (numpy.shape): size of output dimensions
    """

    #set input size
    inp = layers.Input((input_dim[-3],input_dim[-2],1))

    #convolutional layers and pooling
    encoder = layers.Conv2D(8,3,1,"same", activation='relu')(inp)
    pool = layers.MaxPool2D(2,2,"same")(encoder)
    conv = layers.Conv2D(8,3,1,"same", activation='relu')(pool)
    pool2 = layers.MaxPool2D(2, 2, "same")(conv)

    #fully connected flat layers
    encoder_flat = layers.Flatten()(pool2)

    #latent dimension
    latent_layer = layers.Dense(latent_dim, activation='relu')(encoder_flat)

    #decoder layers to spectrogram
    decoder = layers.Conv2DTranspose(8, 3, 2, "same", activation='relu',output_padding=(1,0))(pool2)
    decoder_2 = layers.Conv2DTranspose(1, 3, 2, "same", activation='sigmoid',name='spectrogram',output_padding=(1,1))(decoder)

    #decoder layers to synth parameters
    decoder_b = layers.Dense(output_dim[-1],name='synth_params', activation='sigmoid')(latent_layer)

    #generate model
    return Model(inputs=inp, outputs=[decoder_2, decoder_b])

def autoencoder3(latent_dim,input_dim, output_dim):
    """
    autoencoder: Create autoencoder model
    :param latent_dim (numpy.shape): size of latent dimensions
    :param input_dim (numpy.shape): size of input dimensions
    :param output_dim (numpy.shape): size of output dimensions
    """

    #set input size
    inp = layers.Input((input_dim[-3],input_dim[-2],1))

    #convolutional layers and pooling
    encoder = layers.Conv2D(8,3,1,"same", activation='relu')(inp)
    pool = layers.MaxPool2D(2,2,"same")(encoder)
    conv = layers.Conv2D(8,3,1,"same", activation='relu')(pool)
    pool2 = layers.MaxPool2D(2, 2, "same")(conv)

    #fully connected flat layers
    encoder_flat = layers.Flatten()(pool2)

    #latent dimension
    latent_layer = layers.Dense(latent_dim, activation='relu')(encoder_flat)

    #decoder layers to spectrogram
    decoder = layers.Conv2DTranspose(8, 3, 2, "same", activation='relu',output_padding=(1,0))(pool2)
    decoder_2 = layers.Conv2DTranspose(1, 3, 2, "same", activation='relu',name='spectrogram',output_padding=(1,1))(decoder)

    #decoder layers to synth parameters
    decoder_conv = layers.Conv2DTranspose(8, 3, 2, "same", activation='relu')(pool2)
    decoder_conv_drop = layers.Dropout(.2)(decoder_conv)
    decoder_flat = layers.Flatten()(decoder_conv_drop)
    decoder_b_inner = layers.Dense(256, activation='relu')(decoder_flat)
    decoder_b_inner_drop = layers.Dropout(.2)(decoder_b_inner)
    decoder_b = layers.Dense(output_dim[-1],name='synth_params', activation='relu')(decoder_b_inner_drop)

    #generate model
    return Model(inputs=inp, outputs=[decoder_2, decoder_b])

def vae(latent_dim,input_dim, output_dim,optimizer,warmup_it):
    """
    autoencoder: Create autoencoder model
    :param latent_dim (numpy.shape): size of latent dimensions
    :param input_dim (numpy.shape): size of input dimensions
    :param output_dim (numpy.shape): size of output dimensions
    """

    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    #set input size
    inp = layers.Input((input_dim[-3],input_dim[-2],1))

    #convolutional layers and pooling
    encoder = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(8,3,1,"same")(inp)))
    encoder_pool = layers.MaxPool2D(2,2,"same")(encoder)
    encoder_conv = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(8,3,1,"same")(encoder_pool)))
    encoder_pool2 = layers.MaxPool2D(2, 2, "same")(encoder_conv)

    #latent dimentions
    z_flat = layers.Flatten()(encoder_pool2)
    z_mean = layers.Dense(latent_dim, name="z_mean")(z_flat)
    z_log_var = layers.Dense(latent_dim, name="z_log_var",)(z_flat)
    z_regular = tf.keras.layers.Concatenate(activity_regularizer=W_KLDivergenceRegularizer(optimizer.iterations,warmup_it,latent_dim))([z_mean,z_log_var])
    z = Sampling()([z_mean, z_log_var])
    # z_dense = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),activation=None)(z_flat)
    # z = tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=W_KLDivergenceRegularizer(optimizer.iterations,warmup_it))(z_dense)


    #decoder layers to spectrogram
    decoder_a = layers.Activation('relu')(layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1])(z))
    decoder_a_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_a)
    decoder_a_deconv= layers.Activation('relu')(layers.Conv2DTranspose(8, 3, 2, "same",output_padding=(1,1))(decoder_a_reverse_flat))
    decoder_a_deconv_2 = layers.Conv2DTranspose(1, 3, 2, "same",name='spectrogram',activation= "tanh",output_padding=(1,0))(decoder_a_deconv)

    #decoder layers to synth parameters
    decoder_b = layers.Activation('relu')(layers.Dense(1024)(z))
    decoder_b_h1 = layers.Activation('relu')(layers.Dense(1024)(decoder_b))
    decoder_b_h2 = layers.Activation('relu')(layers.Dense(1024)(decoder_b_h1))
    decoder_b_out = layers.Dense(output_dim[-1],name='synth_params', activation="tanh")(decoder_b_h2)

    #generate model
    return Model(inputs=inp, outputs=[decoder_a_deconv_2, decoder_b_out])

import dynfilt_layers

def dynamic_vae(latent_dim,input_dim, output_dim,optimizer,warmup_it,param_dims):
    """
    autoencoder: Create autoencoder model
    :param latent_dim (numpy.shape): size of latent dimensions
    :param input_dim (numpy.shape): size of input dimensions
    :param output_dim (numpy.shape): size of output dimensions
    """

    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    #set input size
    inp = layers.Input((input_dim[-3],input_dim[-2],1))
    params_inp = layers.Input((None,param_dims))      

    # parameters network
    # TODO: add more layers
    dyn_filts = layers.Activation('relu')(layers.Dense(512,name="dyn_filt_dense")(params_inp))
    dyn_filts = layers.Dense(1024,name="dyn_filt_dense_2")(dyn_filts)
    dyn_filts = layers.Reshape((1,1024,1,-1),name="dyn_filt_reshape")(dyn_filts)

    #convolutional layers and pooling
    encoder = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(8,3,1,"same")(inp)))
    encoder_pool = layers.MaxPool2D(2,2,"same")(encoder)
    encoder_conv = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(8,3,1,"same")(encoder_pool)))
    encoder_pool2 = layers.MaxPool2D(2, 2, "same")(encoder_conv)

    #latent dimentions
    z_flat = layers.Flatten()(encoder_pool2)
    z_mean = layers.Dense(latent_dim, name="z_mean")(z_flat)
    z_log_var = layers.Dense(latent_dim, name="z_log_var",)(z_flat)
    z_regular = tf.keras.layers.Concatenate(activity_regularizer=W_KLDivergenceRegularizer(optimizer.iterations,warmup_it,latent_dim))([z_mean,z_log_var])
    z = Sampling()([z_mean, z_log_var])
    # z_dense = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),activation=None)(z_flat)
    # z = tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=W_KLDivergenceRegularizer(optimizer.iterations,warmup_it))(z_dense)


    #decoder layers to spectrogram
    decoder_a = layers.Activation('relu')(layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1])(z))
    decoder_a_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_a)
    decoder_a_deconv= layers.Activation('relu')(layers.Conv2DTranspose(8, 3, 2, "same",output_padding=(1,1))(decoder_a_reverse_flat))
    decoder_a_deconv_2 = layers.Conv2DTranspose(1, 3, 2, "same",name='spectrogram',activation= "tanh",output_padding=(1,0))(decoder_a_deconv)

    #decoder layers to synth parameters
    decoder_b = layers.Activation('relu')(layers.Dense(1024)(z))
    decoder_b_h1 = layers.Activation('relu')(layers.Dense(1024)(decoder_b))
    decoder_b_h2 = layers.Activation('relu')(layers.Dense(1024)(decoder_b_h1))
    decoder_b_h2 = layers.Reshape((1,decoder_b_h2.shape[1],1))(decoder_b_h2)
    decoder_b_out = dynfilt_layers.Conv2D(padding="VALID")(decoder_b_h2, dyn_filts)
    #decoder_b_out = layers.Dense(output_dim[-1],name='synth_params', activation="tanh")(decoder_b_h2)

    #generate model
    return Model(inputs=(inp, params_inp), outputs=[decoder_a_deconv_2, decoder_b_out])



def vae_flow(latent_dim,input_dim, output_dim):
    """
    autoencoder: Create autoencoder model
    :param latent_dim (numpy.shape): size of latent dimensions
    :param input_dim (numpy.shape): size of input dimensions
    :param output_dim (numpy.shape): size of output dimensions
    """

    # prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    # #set input size
    # inp = layers.Input((input_dim[-3],input_dim[-2],1))

    # #convolutional layers and pooling
    # encoder = layers.Conv2D(8,3,1,"same", activation='relu')(inp)
    # encoder_pool = layers.MaxPool2D(2,2,"same")(encoder)
    # encoder_conv = layers.Conv2D(8,3,1,"same", activation='relu')(encoder_pool)
    # encoder_pool2 = layers.MaxPool2D(2, 2, "same")(encoder_conv)

    # #latent dimentions
    # z_flat = layers.Flatten()(encoder_pool2)
    # # z_mean = layers.Dense(latent_dim, name="z_mean")(z_flat)
    # # z_log_var = layers.Dense(latent_dim, name="z_log_var")(z_flat)
    # # z = Sampling(activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0))([z_mean, z_log_var])
    # z_dense = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),activation=None)(z_flat)
    # z = tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0))(z_dense)

    # # Autoregresive transformation for posterior distribution
    # zt = tfpl.AutoregressiveTransform(tfb.AutoregressiveNetwork(params=2, hidden_units=[16], activation='relu'))(z)

    # #decoder layers to spectrogram
    # decoder_a = layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1],activation='relu')(zt)
    # decoder_a_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_a)
    # decoder_a_deconv= layers.Conv2DTranspose(8, 3, 2, "same", activation='relu',output_padding=(1,0))(decoder_a_reverse_flat)
    # decoder_a_deconv_2 = layers.Conv2DTranspose(1, 3, 2, "same", activation='relu',name='spectrogram',output_padding=(1,1))(decoder_a_deconv)

    # #decoder layers to synth parameters
    # decoder_b = layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1], activation='relu')(zt)
    # decoder_b_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_b)
    # decoder_b_conv = layers.Conv2DTranspose(8, 3, 2, "same", activation='relu')(decoder_b_reverse_flat)
    # decoder_b_conv_drop = layers.Dropout(.2)(decoder_b_conv)
    # decoder_b_flat = layers.Flatten()(decoder_b_conv_drop)
    # decoder_b_inner = layers.Dense(256, activation='relu')(decoder_b_flat)
    # decoder_b_inner_drop = layers.Dropout(.2)(decoder_b_inner)
    # decoder_b_out = layers.Dense(output_dim[-1],name='synth_params', activation='relu')(decoder_b_inner_drop)

    # #generate model
    # return Model(inputs=inp, outputs=[decoder_a_deconv_2, decoder_b_out])

    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    #set input size
    inp = layers.Input((input_dim[-3],input_dim[-2],1))

    #convolutional layers and pooling
    encoder = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(8,3,1,"same")(inp)))
    encoder_pool = layers.MaxPool2D(2,2,"same")(encoder)
    encoder_conv = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(8,3,1,"same")(encoder_pool)))
    encoder_pool2 = layers.MaxPool2D(2, 2, "same")(encoder_conv)

    #latent dimentions
    z_flat = layers.Flatten()(encoder_pool2)
    z_mean = layers.Dense(2, name="z_mean")(z_flat)
    z_log_var = layers.Dense(2, name="z_log_var",)(z_flat)
    z_sample = tfpl.DistributionLambda(lambda t: tfd.Sample(tfd.Normal(loc=t[..., 0], scale=t[..., 1]), sample_shape=[2]))([z_mean,z_log_var])
    zt = tfpl.AutoregressiveTransform(tfb.AutoregressiveNetwork(params=2, hidden_units=[10], activation='relu'))(zy_sample)


    #decoder layers to spectrogram
    decoder_a = layers.Activation('relu')(layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1])(zt))
    decoder_a_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_a)
    decoder_a_deconv= layers.Activation('relu')(layers.Conv2DTranspose(8, 3, 2, "same",output_padding=(1,1))(decoder_a_reverse_flat))
    decoder_a_deconv_2 = layers.Conv2DTranspose(1, 3, 2, "same",name='spectrogram',activation= "tanh",output_padding=(1,0))(decoder_a_deconv)

    #decoder layers to synth parameters
    decoder_b = layers.Activation('relu')(layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1])(zt))
    decoder_b_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_b)
    decoder_b_conv = layers.Activation('relu')(layers.Conv2DTranspose(8, 3, 2, "same")(decoder_b_reverse_flat))
    decoder_b_conv_drop = layers.Dropout(.2)(decoder_b_conv)
    decoder_b_flat = layers.Flatten()(decoder_b_conv_drop)
    decoder_b_inner = layers.Activation('relu')(layers.Dense(256)(decoder_b_flat))
    decoder_b_inner_drop = layers.Dropout(.2)(decoder_b_inner)
    decoder_b_out = layers.Dense(output_dim[-1],name='synth_params', activation="tanh")(decoder_b_inner_drop)

    #generate model
    return Model(inputs=inp, outputs=[decoder_a_deconv_2, decoder_b_out])


#dictionary to store models for each cli input
get_model = {"ae":autoencoder(l_dim,i_dim,o_dim),"ae2": autoencoder2(l_dim,i_dim,o_dim), "ae3": autoencoder3(l_dim,i_dim,o_dim), "vae": vae(l_dim,i_dim,o_dim,optimizer,warmup_it), "dynamic_vae":dynamic_vae(l_dim,i_dim,o_dim,optimizer,warmup_it,v_dims)}#, "vae_flow": vae_flow(l_dim,i_dim,o_dim)}

