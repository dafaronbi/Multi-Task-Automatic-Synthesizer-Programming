#import needed modules
import numpy as np
import ds
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import sys


def main():
    #list GPUs that tensor flow can use
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))


    #load data
    train_data = ds.melParamData("train","data")
    test_data = ds.melParamData("test","data")
    validation_data = ds.melParamData("validation","data")

    #define shapes
    l_dim = 64
    i_dim = train_data.get_mels()[...,np.newaxis].shape
    o_dim = train_data.get_params().shape

    #dictionary to store models for each cli input
    get_model = {"ae":model.autoencoder(l_dim,i_dim,o_dim),"ae2": model.autoencoder2(l_dim,i_dim,o_dim), "ae3": model.autoencoder3(l_dim,i_dim,o_dim), "vae": model.vae(l_dim,i_dim,o_dim), "vae": model.vae(l_dim,i_dim,o_dim), "vae_flow": model.vae_flow(l_dim,i_dim,o_dim)
    }

    #dictionary of losses
    get_loss = {"ae": losses.MeanSquaredError(),"ae2": losses.MeanSquaredError(),"ae3" : losses.MeanSquaredError(),"vae": losses.MeanSquaredError(),"vae_flow": losses.MeanSquaredError()}

    #make directory to save model if not already made
    if not os.path.isdir("saved_models/"+ sys.argv[1]):
         os.makedirs("saved_models/"+ sys.argv[1])

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "saved_models/"+ sys.argv[1] +"/cp-{epoch:04d}.ckpt"

    #epoch size
    epochs=500

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = ds.melParamData("train","data").get_mels().shape[0] // batch_size

    #save freq is every 100 epochs
    save_freq = batches_epoch*100

    # Create a callback that saves the model's weights every 50 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=save_freq)

    #create model
    m = get_model[sys.argv[1]]

    #view summary of model
    m.summary()

    #compile model
    m.compile(optimizer='adam', loss=get_loss[sys.argv[1]])

    #update learning rate
    m.optimizer.lr.assign(1e-3)

    #train model
    m.fit(train_data.get_mels()[...,np.newaxis],[train_data.get_mels(),train_data.get_params()], epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    #print evaluation on test set
    loss, loss1,loss2 = m.evaluate(test_data.get_mels(),[test_data.get_mels(),test_data.get_params()],2)
    print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))

if __name__ == "__main__":
    main()