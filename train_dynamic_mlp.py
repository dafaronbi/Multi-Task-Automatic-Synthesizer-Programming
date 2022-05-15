#import needed modules
import numpy as np
import ds
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import sys
from data import all_data
import random
import pickle


def main():

    #load in data
    with open("/vast/df2322/asp_data/dynamic_mels.pkl", 'rb') as handle:
        mels = pickle.load(handle)
    with open("/vast/df2322/asp_data/dynamic_params.pkl", 'rb') as handle:
        params = pickle.load(handle)
    with open("/vast/df2322/asp_data/dynamic_kernels.pkl", 'rb') as handle:
        kernels = pickle.load(handle)

    m_size = len(mels)

    r_mels,r_params,r_kernels = zip(*random.sample(list(zip(mels,params,kernels)), m_size))

    train_mels = r_mels[:m_size - m_size//5]
    train_params = r_params[:m_size - m_size//5]
    train_kernels = r_kernels[:m_size - m_size//5]

    valid_mels = r_mels[m_size - m_size//5: m_size - m_size//10]
    valid_params = r_params[m_size - m_size//5: m_size - m_size//10]
    valid_kernels = r_kernels[m_size - m_size//5: m_size - m_size//10]

    test_mels = r_mels[m_size - m_size//10:]
    test_params = r_params[m_size - m_size//10:]
    test_kernels = r_kernels[m_size - m_size//10:]

    #use synth dataloader
    train_data_load = all_data.SynthDataGenerator(len(train_mels),train_mels,train_params,train_kernels)
    valid_data_load = all_data.SynthDataGenerator(len(valid_mels),valid_mels,valid_params,valid_kernels)
    test_data_load = all_data.SynthDataGenerator(len(test_mels),test_mels,test_params,test_kernels)

    with open("/vast/df2322/asp_data/dynamic_mlp/test_dynamic_mels.pkl", 'wb') as handle:
        pickle.dump(test_mels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("/vast/df2322/asp_data/dynamic_mlp/test_dynamic_params.pkl", 'wb') as handle:
        pickle.dump(test_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("/vast/df2322/asp_data/dynamic_mlp/test_dynamic_kernels.pkl", 'wb') as handle:
        pickle.dump(test_kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print(m_size)

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

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #make directory to save model if not already made
    if not os.path.isdir("saved_models/dynamic_mlp"):
        os.makedirs("saved_models/dynamic_mlp")

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "saved_models/dynamic_mlp/cp-{epoch:04d}.ckpt"

    #epoch size
    epochs= 500

    #save freq is every 100 epochs
    save_freq = batches_epoch*100

    # Create a callback that saves the model's weights every 50 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=save_freq)

    #create model
    m = model.dynamic_mlp_vae(64, i_dim, 0, model.optimizer, warmup_it,0)

    #view summary of model
    m.summary()

    #compile model
    m.compile(optimizer=model.optimizer, loss=losses.MeanSquaredError())

    #update learning rate
    m.optimizer.lr.assign(1e-4)

    #train model
    m.fit(train_data_load, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    #print evaluation on test set
    loss, loss1,loss2,loss3,loss4 = m.evaluate(test_data_load,2)
    print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))

if __name__ == "__main__":
    main()