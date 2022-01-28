#import needed libraries
from tensorflow.keras import layers, losses, Model
import tensorflow_probability as tfp
import tensorflow as tf

#class for sampling in vae
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



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

def vae(latent_dim,input_dim, output_dim):
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
    encoder = layers.Conv2D(8,3,1,"same", activation='relu')(inp)
    encoder_pool = layers.MaxPool2D(2,2,"same")(encoder)
    encoder_conv = layers.Conv2D(8,3,1,"same", activation='relu')(encoder_pool)
    encoder_pool2 = layers.MaxPool2D(2, 2, "same")(encoder_conv)

    #latent dimentions
    z_flat = layers.Flatten()(encoder_pool2)
    # z_mean = layers.Dense(latent_dim, name="z_mean")(z_flat)
    # z_log_var = layers.Dense(latent_dim, name="z_log_var")(z_flat)
    # z = Sampling(activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0))([z_mean, z_log_var])
    z_dense = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),activation=None)(z_flat)
    z = tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0))(z_dense)


    #decoder layers to spectrogram
    decoder_a = layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1],activation='relu')(z)
    decoder_a_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_a)
    decoder_a_deconv= layers.Conv2DTranspose(8, 3, 2, "same", activation='relu',output_padding=(1,0))(decoder_a_reverse_flat)
    decoder_a_deconv_2 = layers.Conv2DTranspose(1, 3, 2, "same", activation='relu',name='spectrogram',output_padding=(1,1))(decoder_a_deconv)

    #decoder layers to synth parameters
    decoder_b = layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1], activation='relu')(z)
    decoder_b_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_b)
    decoder_b_conv = layers.Conv2DTranspose(8, 3, 2, "same", activation='relu')(decoder_b_reverse_flat)
    decoder_b_conv_drop = layers.Dropout(.2)(decoder_b_conv)
    decoder_b_flat = layers.Flatten()(decoder_b_conv_drop)
    decoder_b_inner = layers.Dense(256, activation='relu')(decoder_b_flat)
    decoder_b_inner_drop = layers.Dropout(.2)(decoder_b_inner)
    decoder_b_out = layers.Dense(output_dim[-1],name='synth_params', activation='relu')(decoder_b_inner_drop)

    #generate model
    return Model(inputs=inp, outputs=[decoder_a_deconv_2, decoder_b_out])

def vae_flow(latent_dim,input_dim, output_dim):
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
    encoder = layers.Conv2D(8,3,1,"same", activation='relu')(inp)
    encoder_pool = layers.MaxPool2D(2,2,"same")(encoder)
    encoder_conv = layers.Conv2D(8,3,1,"same", activation='relu')(encoder_pool)
    encoder_pool2 = layers.MaxPool2D(2, 2, "same")(encoder_conv)

    #latent dimentions
    z_flat = layers.Flatten()(encoder_pool2)
    # z_mean = layers.Dense(latent_dim, name="z_mean")(z_flat)
    # z_log_var = layers.Dense(latent_dim, name="z_log_var")(z_flat)
    # z = Sampling(activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0))([z_mean, z_log_var])
    z_dense = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),activation=None)(z_flat)
    z = tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0))(z_dense)


    #decoder layers to spectrogram
    decoder_a = layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1],activation='relu')(z)
    decoder_a_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_a)
    decoder_a_deconv= layers.Conv2DTranspose(8, 3, 2, "same", activation='relu',output_padding=(1,0))(decoder_a_reverse_flat)
    decoder_a_deconv_2 = layers.Conv2DTranspose(1, 3, 2, "same", activation='relu',name='spectrogram',output_padding=(1,1))(decoder_a_deconv)

    #decoder layers to synth parameters
    decoder_b = layers.Dense(encoder_pool2.shape[-3] * encoder_pool2.shape[-2] * encoder_pool2.shape[-1], activation='relu')(z)
    decoder_b_reverse_flat = layers.Reshape(encoder_pool2.shape[1:])(decoder_b)
    decoder_b_conv = layers.Conv2DTranspose(8, 3, 2, "same", activation='relu')(decoder_b_reverse_flat)
    decoder_b_conv_drop = layers.Dropout(.2)(decoder_b_conv)
    decoder_b_flat = layers.Flatten()(decoder_b_conv_drop)
    decoder_b_inner = layers.Dense(256, activation='relu')(decoder_b_flat)
    decoder_b_inner_drop = layers.Dropout(.2)(decoder_b_inner)
    decoder_b_out = layers.Dense(output_dim[-1],name='synth_params', activation='relu')(decoder_b_inner_drop)

    #generate model
    return Model(inputs=inp, outputs=[decoder_a_deconv_2, decoder_b_out])