#import needed libraries
from tensorflow.keras import layers, losses, Model

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