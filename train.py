#import needed modules
import numpy as np
import ds
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import sys


def main():

    m_size = model.data_size[sys.argv[3]]
    print(m_size)
    #parameter input for dynamic filters
    v_dims = 4

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    print(batches_epoch)

    #warmup amount
    warmup_it = 100*batches_epoch

    #list GPUs that tensor flow can use
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))


    #load data
    train_data = ds.melParamData("train", sys.argv[3])
    test_data = ds.melParamData("test", sys.argv[3])
    validation_data = ds.melParamData("validation", sys.argv[3])

    param_dim = train_data.get_params().shape[-1]

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)
    o_dim = (None, param_dim)

    get_model = {"ae" : model.autoencoder(l_dim,i_dim,o_dim),"ae2": model.autoencoder2(l_dim,i_dim,o_dim), "ae3": model.autoencoder3(l_dim,i_dim,o_dim), "vae": model.vae(l_dim,i_dim,o_dim,model.optimizer,warmup_it), "dynamic_vae": model.dynamic_vae(l_dim,i_dim,o_dim,model.optimizer,warmup_it,v_dims)}#, "vae_flow": vae_flow(l_dim,i_dim,o_dim)}

    #make vst hot encoding
    vst_hot_train = np.array([np.random.rand(param_dim,4)]*len(train_data))
    vst_hot_test = np.array([np.random.rand(param_dim,4)]*len(test_data))
    vst_hot_valid = np.array([np.random.rand(param_dim,4)]*len(validation_data))

    #make directory to save model if not already made
    if not os.path.isdir("saved_models/"+ sys.argv[1] + "_" + sys.argv[3]):
        os.makedirs("saved_models/"+ sys.argv[1] + "_" + sys.argv[3])

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "saved_models/"+ sys.argv[1] + "_" + sys.argv[3] + "/cp-{epoch:04d}.ckpt"

    #epoch size
    epochs= int(sys.argv[2])

    #save freq is every 100 epochs
    save_freq = batches_epoch*100

    # Create a callback that saves the model's weights every 50 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=save_freq)

    #dictionary of losses
    get_loss = {"ae": losses.MeanSquaredError(),"ae2": losses.MeanSquaredError(),"ae3" : losses.MeanSquaredError(),"vae": losses.MeanSquaredError(),"dynamic_vae": losses.MeanSquaredError(),"vae_flow": losses.MeanSquaredError()}

    #create model
    m = get_model[sys.argv[1]]

    #view summary of model
    m.summary()

    #compile model
    m.compile(optimizer=model.optimizer, loss=get_loss[sys.argv[1]])

    #update learning rate
    m.optimizer.lr.assign(1e-4)

    #train model
    if sys.argv[1] == "dynamic_vae":
        m.fit([train_data.get_mels()[...,np.newaxis], vst_hot_train],[train_data.get_mels(),train_data.get_params()], epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    else:
        m.fit([train_data.get_mels()[...,np.newaxis]],[train_data.get_mels(),train_data.get_params()], epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    #print evaluation on test set
    loss, loss1,loss2 = m.evaluate([test_data.get_mels(),vst_hot_test],[test_data.get_mels(),test_data.get_params()],2)
    print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))

if __name__ == "__main__":
    main()